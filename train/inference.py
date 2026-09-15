"""Гибридный инференс: категория пользователя -> модель -> рекомендации.

Использование:
    python -m train.inference --user-id 12345 --k 10

Категория определяется размером истории пользователя в ТРЕНИРОВОЧНОЙ части
(все действия pos+neg после единой разметки):
    cold: размер <= 30  -> fpmc
    warm: 31..70        -> sasrec
    hot:  >= 71         -> light_gcn
Fallback: если размер истории в диапазоне [25, 35] (близко к границе cold/warm) —
используется mostPopular.

Соответствие категория->модель задано в train/config.py (MODEL_BY_GROUP) и будет
уточнено оператором вручную после получения точных метрик.

Требует артефакты моделей из train/artifacts/ (создаются train/run_training.py).
"""

import argparse
import os

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from common.data import build_sequences_with_maps, read_files
from common.groups import categorize_by_history_size
from common.labels import assign_labels
from common.utils import get_device, seed_everything

from .config import load_runtime_config

# Конфиг выбирается переменной окружения TRAIN_CONFIG
# (по умолчанию train.config; docker — train.config_docker).
CFG = load_runtime_config()
ARTIFACTS = CFG.ARTIFACTS
GROUPS = CFG.GROUPS
MODELS = CFG.MODELS
MODEL_BY_GROUP = CFG.MODEL_BY_GROUP
TRAIN = CFG.TRAIN

_BATCH = 4096


def _load_train_data():
    """Загружает train-недели и строит (user_to_idx, item_to_idx, pos_history, seen, hist_size)."""
    train_df = read_files(
        [f"{TRAIN.data_dir}/train/week_{i:02}.parquet"
         for i in range(TRAIN.start_week, TRAIN.end_week + 1)])
    labeled = assign_labels(train_df, TRAIN.min_timespent_pos)
    pos_train = labeled.filter(pl.col("label") == 1)
    neg_train = labeled.filter(pl.col("label") == 0)

    user_to_idx = {u: i for i, u in enumerate(train_df["user_id"].unique().sort().to_list())}
    item_to_idx = {i: j for j, i in enumerate(train_df["item_id"].unique().sort().to_list())}

    pos_history = build_sequences_with_maps(pos_train, user_to_idx, item_to_idx)
    seen: dict = {u: set() for u in user_to_idx.values()}
    for user_id, items in neg_train.group_by("user_id").agg("item_id").iter_rows():
        u = user_to_idx.get(user_id)
        if u is None:
            continue
        seen[u].update(item_to_idx[i] for i in items if i in item_to_idx)
    for u, seq in pos_history.items():
        seen[u].update(seq)

    hist_size = {int(uid): int(n) for uid, n in
                 labeled.group_by("user_id").agg(pl.len().alias("n")).iter_rows()}
    return user_to_idx, item_to_idx, pos_history, seen, hist_size


# ---------------------------------------------------------------------------
# Рекомендации каждой моделью
# ---------------------------------------------------------------------------

def _rec_most_popular(uid: int, k: int, device: str):
    from most_popular.model import MostPopular

    art = torch.load(ARTIFACTS["most_popular"], weights_only=False)
    u = art["user_to_idx"].get(uid)
    if u is None:
        return []
    model = MostPopular(art["popularity"], art["item_ranking"])
    idxs = model.recommend(art["seen"].get(u, set()), top_n=k)
    idx2id = {v: i for i, v in art["item_to_idx"].items()}
    return [idx2id[it] for it in idxs]


def _rec_fpmc(uid: int, k: int, device: str, pos_history: dict, seen: dict):
    from fpmc.model import FPMC

    art = torch.load(ARTIFACTS["fpmc"], weights_only=False)
    u = art["user_to_idx"].get(uid)
    if u is None or u not in art["last_item"]:
        return []
    model = FPMC(art["n_users"], art["n_items"], art["dim"]).to(device).eval()
    model.load_state_dict(art["model_state_dict"])

    l = art["last_item"][u]
    cand = np.array([i for i in range(art["n_items"]) if i not in seen[u]], dtype=np.int64)
    if len(cand) == 0:
        return []
    u_t = torch.full((1,), u, device=device, dtype=torch.long)
    l_t = torch.full((1,), l, device=device, dtype=torch.long)
    scores = torch.empty(len(cand), device=device)
    with torch.no_grad():
        for s in range(0, len(cand), _BATCH):
            chunk = torch.as_tensor(cand[s:s + _BATCH], device=device)
            scores[s:s + len(chunk)] = model.score(
                u_t.expand(len(chunk)), l_t.expand(len(chunk)), chunk)
    top = torch.topk(scores, k=min(k, len(cand))).indices.cpu().tolist()
    idx2id = {v: i for i, v in art["item_to_idx"].items()}
    return [idx2id[int(cand[i])] for i in top]


def _rec_light_gcn(uid: int, k: int, device: str, seen: dict):
    from light_gcn.data import build_normalized_adjacency
    from light_gcn.model import LightGCN

    art = torch.load(ARTIFACTS["light_gcn"], weights_only=False)
    u = art["user_to_idx"].get(uid)
    if u is None:
        return []
    model = LightGCN(art["n_users"], art["n_items"], art["dim"], art["n_layers"]).to(device).eval()
    model.load_state_dict(art["model_state_dict"])
    model.norm_adj = build_normalized_adjacency(
        art["n_users"], art["n_items"], art["train_users"], art["train_items"], device)

    with torch.no_grad():
        user_emb, item_emb = model()
    cand = np.array([i for i in range(art["n_items"]) if i not in seen[u]], dtype=np.int64)
    if len(cand) == 0:
        return []
    u_t = torch.full((1,), u, device=device, dtype=torch.long)
    scores = torch.empty(len(cand), device=device)
    with torch.no_grad():
        for s in range(0, len(cand), _BATCH):
            chunk = torch.as_tensor(cand[s:s + _BATCH], device=device)
            scores[s:s + len(chunk)] = model.score(
                user_emb, item_emb, u_t.expand(len(chunk)), chunk)
    top = torch.topk(scores, k=min(k, len(cand))).indices.cpu().tolist()
    idx2id = {v: i for i, v in art["item_to_idx"].items()}
    return [idx2id[int(cand[i])] for i in top]


def _rec_sasrec(uid: int, k: int, device: str, user_to_idx: dict, pos_history: dict, seen: dict):
    from sasrec.model import SASRec

    art = torch.load(ARTIFACTS["sasrec"], weights_only=False)
    model = SASRec(len(art["id_to_idx"]) + 1, art["emb_width"], art["max_len"],
                   MODELS.sasrec["num_heads"], MODELS.sasrec["num_layers"],
                   MODELS.sasrec["dropout"]).to(device).eval()
    model.load_state_dict(art["model_state_dict"])

    u = user_to_idx.get(uid)
    if u is None:
        return []
    seq = pos_history.get(u, [])[-art["max_len"]:]
    if not seq:
        return []
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    positions = torch.arange(len(seq), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        uv = F.normalize(model(input_ids, positions), p=2, dim=1)
        iw = F.normalize(model.item_embedding.weight, p=2, dim=1)
        scores = torch.matmul(uv, iw.T)[0]
    scores[0] = -float("inf")                       # padding
    scores[list(seen[u])] = -float("inf")           # всё увиденное (pos+neg)
    n_avail = min(k, int((scores > -1e30).sum().item()))
    if n_avail == 0:
        return []
    top = torch.topk(scores, k=n_avail).indices.cpu().tolist()
    idx2id = {v: i for i, v in art["id_to_idx"].items()}
    return [idx2id[int(i)] for i in top if int(i) in idx2id]


RECOMMENDERS = {
    "most_popular": lambda uid, k, device, *ctx: _rec_most_popular(uid, k, device),
    "fpmc": _rec_fpmc,
    "light_gcn": _rec_light_gcn,
    "sasrec": _rec_sasrec,
}


def recommend(model_name: str, uid: int, k: int, device: str,
              user_to_idx: dict, pos_history: dict, seen: dict) -> list:
    return RECOMMENDERS[model_name](uid, k, device, user_to_idx, pos_history, seen)


def parse_args():
    p = argparse.ArgumentParser(description="Гибридный инференс VK-LSVD")
    p.add_argument("--user-id", type=int, required=True, help="Исходный user_id")
    p.add_argument("--k", type=int, default=None, help="Количество рекомендаций (default: из конфига)")
    return p.parse_args()


def main():
    args = parse_args()
    k = args.k or TRAIN.k
    device = get_device()
    seed_everything(TRAIN.seed)

    print("Загрузка train-данных для определения истории пользователя...")
    user_to_idx, _, pos_history, seen, hist_size = _load_train_data()

    uid = int(args.user_id)
    if uid not in hist_size:
        print(f"Пользователь {uid} отсутствует в тренировочной части датасета.")
        return

    size = hist_size[uid]
    if GROUPS.fallback_min <= size <= GROUPS.fallback_max:
        group, model_name = "fallback", "most_popular"
    else:
        group = categorize_by_history_size(size, GROUPS.cold_max, GROUPS.warm_max)
        model_name = MODEL_BY_GROUP[group]

    if not os.path.exists(ARTIFACTS[model_name]):
        print(f"[error] нет артефакта {ARTIFACTS[model_name]} — запустите "
              f"python -m train.run_training")
        return

    rec = recommend(model_name, uid, k, device, user_to_idx, pos_history, seen)
    print(f"user_id={uid} | размер истории (pos+neg)={size} | группа={group} | модель={model_name}")
    print(f"рекомендации (k={k}): {rec}")


if __name__ == "__main__":
    main()