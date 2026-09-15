"""Метрики Recall/NDCG/MRR@k по группам пользователей для всех моделей.

Использование:
    python -m train.evaluate_groups

Читает артефакты из train/artifacts/, строит группы cold/warm/hot
(common.groups) и сохраняет таблицу метрик в train/results/metrics_by_group.csv
(читается pandas/polars). Модели без артефакта пропускаются.
"""

import os
from typing import cast

import polars as pl
import torch
from torch.utils.data import DataLoader

from common.data import read_files, read_week
from common.groups import GROUP_NAMES, split_users_by_history
from common.metrics import eval_metrics
from common.utils import get_device, seed_everything

from .config import load_runtime_config

# Конфиг выбирается переменной окружения TRAIN_CONFIG
# (по умолчанию train.config; docker — train.config_docker).
CFG = load_runtime_config()
ARTIFACTS = CFG.ARTIFACTS
GROUPS = CFG.GROUPS
MODEL_NAMES = CFG.MODEL_NAMES
PATHS = CFG.PATHS
RESULTS_FILE = CFG.RESULTS_FILE
TRAIN = CFG.TRAIN


# ---------------------------------------------------------------------------
# Сбор per-user рангов для каждой модели
# ---------------------------------------------------------------------------

def _collect_most_popular(device) -> dict:
    from most_popular.config import MostPopularConfig
    from most_popular.data import load_and_build
    from most_popular.evaluate import evaluate
    from most_popular.model import MostPopular

    cfg = MostPopularConfig(
        data_dir=TRAIN.data_dir, start_week=TRAIN.start_week, end_week=TRAIN.end_week,
        k=TRAIN.k, min_timespent_pos=TRAIN.min_timespent_pos, seed=TRAIN.seed,
    )
    bundle = load_and_build(cfg)
    art = torch.load(ARTIFACTS["most_popular"], weights_only=False)
    model = MostPopular(art["popularity"], art["item_ranking"])
    user_idxs, gt_positions = evaluate(model, bundle, k=cfg.k, return_per_user=True)
    idx_to_id = {v: k for k, v in bundle.user_to_idx.items()}
    return {int(idx_to_id[int(u)]): p for u, p in zip(user_idxs, gt_positions)}


def _collect_fpmc(device) -> dict:
    from fpmc.data import DataConfig, load_and_build
    from fpmc.model import FPMC
    from fpmc.train import evaluate

    cfg = DataConfig(
        data_dir=TRAIN.data_dir, start_week=TRAIN.start_week, end_week=TRAIN.end_week,
        k=TRAIN.k, min_history=TRAIN.min_history,
        min_timespent_pos=TRAIN.min_timespent_pos, seed=TRAIN.seed,
    )
    bundle = load_and_build(cfg)
    art = torch.load(ARTIFACTS["fpmc"], weights_only=False)
    model = FPMC(bundle.n_users, bundle.n_items, art["dim"]).to(device)
    model.load_state_dict(art["model_state_dict"])
    user_idxs, gt_positions = evaluate(model, bundle, device=device, k=cfg.k,
                                       return_per_user=True)
    idx_to_id = {v: k for k, v in bundle.user_to_idx.items()}
    return {int(idx_to_id[int(u)]): p for u, p in zip(user_idxs, gt_positions)}


def _collect_light_gcn(device) -> dict:
    from light_gcn.data import DataConfig, load_and_build
    from light_gcn.model import LightGCN
    from light_gcn.train import evaluate

    cfg = DataConfig(
        data_dir=TRAIN.data_dir, start_week=TRAIN.start_week, end_week=TRAIN.end_week,
        k=TRAIN.k, min_history=TRAIN.min_history,
        min_timespent_pos=TRAIN.min_timespent_pos, seed=TRAIN.seed,
    )
    bundle = load_and_build(cfg)
    art = torch.load(ARTIFACTS["light_gcn"], weights_only=False)
    model = LightGCN(bundle.n_users, bundle.n_items, art["dim"], art["n_layers"]).to(device)
    model.load_state_dict(art["model_state_dict"])
    user_idxs, gt_positions = evaluate(model, bundle, device=device, k=cfg.k,
                                       return_per_user=True)
    idx_to_id = {v: k for k, v in bundle.user_to_idx.items()}
    return {int(idx_to_id[int(u)]): p for u, p in zip(user_idxs, gt_positions)}


def _collect_sasrec(device) -> dict:
    from sasrec.config import SASRecConfig
    from sasrec.data import (
        SequentialDataset,
        TrainIterableDataset,
        collate_fn,
    )
    from sasrec.model import SASRec
    from sasrec.train import evaluate

    cfg = SASRecConfig(
        data_dir=TRAIN.data_dir, start_week=TRAIN.start_week, end_week=TRAIN.end_week,
        k=TRAIN.k, max_history=TRAIN.max_history,
        min_timespent_pos=TRAIN.min_timespent_pos, seed=TRAIN.seed,
    )
    art = torch.load(ARTIFACTS["sasrec"], weights_only=False)
    id_to_idx = art["id_to_idx"]
    model = SASRec(len(id_to_idx) + 1, art["emb_width"], art["max_len"],
                   cfg.num_heads, cfg.num_layers, cfg.dropout).to(device)
    model.load_state_dict(art["model_state_dict"])
    val_iter = TrainIterableDataset(cfg, id_to_idx, [cfg.val_week])
    val_samples = list(val_iter)
    loader = DataLoader(
        SequentialDataset(val_samples), batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=lambda batch: collate_fn(batch, cfg.max_history),
    )
    user_ids, gt_positions = cast(
        tuple, evaluate(model, loader, device=device, top_k=cfg.k, return_per_user=True))
    return {int(u): p for u, p in zip(user_ids, gt_positions)}


COLLECTORS = {
    "most_popular": _collect_most_popular,
    "fpmc": _collect_fpmc,
    "light_gcn": _collect_light_gcn,
    "sasrec": _collect_sasrec,
}


# ---------------------------------------------------------------------------
# Агрегация по группам
# ---------------------------------------------------------------------------

def _metrics_for_users(uid_positions: dict, users: list, k: int) -> dict:
    """Метрики по подмножеству пользователей (только те, кто есть в оценке)."""
    selected = {u: uid_positions[u] for u in users if u in uid_positions}
    user_list = list(selected.keys())
    if not user_list:
        return {"n_users": 0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0}
    metrics = eval_metrics([selected[u] for u in user_list], k)
    return {"n_users": len(user_list), **metrics}


def main():
    device = get_device()
    seed_everything(TRAIN.seed)
    print(f"device: {device} | k={TRAIN.k} | группа до {GROUPS.group_limit} юзеров")

    train_df = read_files(
        [f"{TRAIN.data_dir}/train/week_{i:02}.parquet"
         for i in range(TRAIN.start_week, TRAIN.end_week + 1)])
    val_df = read_week(TRAIN.data_dir, TRAIN.val_week)

    groups = split_users_by_history(
        val_df, train_df, TRAIN.min_timespent_pos,
        cold_max=GROUPS.cold_max, warm_max=GROUPS.warm_max,
        group_limit=GROUPS.group_limit, seed=TRAIN.seed,
    )
    print("Размеры групп:", {g: len(users) for g, users in groups.items()})

    uid_positions: dict = {}
    for name in MODEL_NAMES:
        path = ARTIFACTS[name]
        if not os.path.exists(path):
            print(f"[skip] {name}: нет артефакта {path}")
            continue
        print(f"[eval] {name} ...")
        uid_positions[name] = COLLECTORS[name](device)

    rows = []
    for group in GROUP_NAMES:
        for name in MODEL_NAMES:
            if name not in uid_positions:
                continue
            row = _metrics_for_users(uid_positions[name], groups[group], TRAIN.k)
            rows.append({"group": group, "model": name, **row})

    os.makedirs(PATHS.results_dir, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS_FILE)
    print(f"\nСохранено: {RESULTS_FILE}")
    print(df)


if __name__ == "__main__":
    main()