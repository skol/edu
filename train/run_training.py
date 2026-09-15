"""Последовательное обучение всех моделей VK-LSVD с сохранением артефактов.

Использование:
    python -m train.run_training                        # спрашивает при наличии артефактов
    python -m train.run_training --force-overwrite on   # автоматическая перезапись
    python -m train.run_training --models fpmc,sasrec   # только выбранные модели

Логика перезаписи: если артефакт уже существует и --force-overwrite off
(по умолчанию) — оператору предлагается использовать существующее состояние
(y) или переобучить (n). Значение on включает автоматическую перезапись.
"""

import argparse
import os
from typing import cast

import torch
from torch.utils.data import DataLoader

from common.utils import get_device, seed_everything

from .config import load_runtime_config

# Конфиг выбирается переменной окружения TRAIN_CONFIG
# (по умолчанию train.config; docker — train.config_docker).
CFG = load_runtime_config()
ARTIFACTS = CFG.ARTIFACTS
MODELS = CFG.MODELS
MODEL_NAMES = CFG.MODEL_NAMES
PATHS = CFG.PATHS
TRAIN = CFG.TRAIN


def parse_args():
    p = argparse.ArgumentParser(description="Обучение всех моделей VK-LSVD")
    p.add_argument("--force-overwrite", type=str, choices=["on", "off"], default="off",
                   help="on: автоматическая перезапись существующих артефактов; "
                        "off (default): спрашивать оператора")
    p.add_argument("--models", type=str, default=",".join(MODEL_NAMES),
                   help="Через запятую список моделей для обучения")
    return p.parse_args()


def _should_train(model_name: str, force_overwrite: str) -> bool:
    """Возвращает True, если модель нужно (пере)обучить."""
    path = ARTIFACTS[model_name]
    if not os.path.exists(path):
        return True
    if force_overwrite == "on":
        print(f"[{model_name}] артефакт {path} существует — перезаписываем "
              f"(--force-overwrite on)")
        return True
    answer = input(
        f"[{model_name}] найден существующий артефакт {path}. "
        f"Использовать его? [y/N]: "
    ).strip().lower()
    return answer not in ("y", "yes")


def _save(model_name: str, payload: dict):
    os.makedirs(PATHS.artifacts_dir, exist_ok=True)
    torch.save(payload, ARTIFACTS[model_name])
    print(f"[{model_name}] сохранено: {ARTIFACTS[model_name]}")


# ---------------------------------------------------------------------------
# Обучение каждой модели
# ---------------------------------------------------------------------------

def train_most_popular(device) -> dict:
    from most_popular.config import MostPopularConfig
    from most_popular.data import load_and_build
    from most_popular.evaluate import evaluate
    from most_popular.model import MostPopular

    cfg = MostPopularConfig(
        data_dir=TRAIN.data_dir, start_week=TRAIN.start_week, end_week=TRAIN.end_week,
        k=TRAIN.k, min_timespent_pos=TRAIN.min_timespent_pos, seed=TRAIN.seed,
    )
    bundle = load_and_build(cfg)
    model = MostPopular(bundle.popularity, bundle.item_ranking)
    print(f"metrics: {evaluate(model, bundle, k=cfg.k)}")

    return {
        "popularity": bundle.popularity,
        "item_ranking": bundle.item_ranking,
        "user_to_idx": bundle.user_to_idx,
        "item_to_idx": bundle.item_to_idx,
        "seen": bundle.seen,
        "k": cfg.k,
    }


def train_fpmc(device) -> dict:
    from fpmc.data import DataConfig, load_and_build
    from fpmc.model import FPMC
    from fpmc.train import evaluate, train_model

    cfg = DataConfig(
        data_dir=TRAIN.data_dir, start_week=TRAIN.start_week, end_week=TRAIN.end_week,
        epochs=TRAIN.epochs, lr=TRAIN.lr, k=TRAIN.k,
        min_history=TRAIN.min_history, min_timespent_pos=TRAIN.min_timespent_pos,
        seed=TRAIN.seed, **MODELS.fpmc,
    )
    bundle = load_and_build(cfg)
    model = FPMC(n_users=bundle.n_users, n_items=bundle.n_items, dim=cfg.dim).to(device)
    train_model(model, bundle, device=device,
                epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed)
    print(f"metrics: {evaluate(model, bundle, device=device, k=cfg.k)}")

    return {
        "model_state_dict": model.state_dict(),
        "user_to_idx": bundle.user_to_idx,
        "item_to_idx": bundle.item_to_idx,
        "last_item": bundle.last_item,
        "n_users": bundle.n_users,
        "n_items": bundle.n_items,
        "dim": cfg.dim,
        "min_history": cfg.min_history,
        "k": cfg.k,
    }


def train_light_gcn(device) -> dict:
    from light_gcn.data import DataConfig, load_and_build
    from light_gcn.model import LightGCN
    from light_gcn.train import evaluate, train_model

    cfg = DataConfig(
        data_dir=TRAIN.data_dir, start_week=TRAIN.start_week, end_week=TRAIN.end_week,
        epochs=TRAIN.epochs, lr=TRAIN.lr, k=TRAIN.k,
        min_history=TRAIN.min_history, min_timespent_pos=TRAIN.min_timespent_pos,
        seed=TRAIN.seed, **MODELS.light_gcn,
    )
    bundle = load_and_build(cfg)
    model = LightGCN(n_users=bundle.n_users, n_items=bundle.n_items,
                     dim=cfg.dim, n_layers=cfg.n_layers).to(device)
    train_model(model, bundle, device=device,
                epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed)
    print(f"metrics: {evaluate(model, bundle, device=device, k=cfg.k)}")

    return {
        "model_state_dict": model.state_dict(),
        "user_to_idx": bundle.user_to_idx,
        "item_to_idx": bundle.item_to_idx,
        "train_users": bundle.train_users,   # рёбра для пересборки матрицы смежности
        "train_items": bundle.train_items,
        "n_users": bundle.n_users,
        "n_items": bundle.n_items,
        "dim": cfg.dim,
        "n_layers": cfg.n_layers,
        "k": cfg.k,
    }


def train_sasrec(device) -> dict:
    import torch.optim as optim

    from common.embeddings import load_embeddings_only

    from sasrec.config import SASRecConfig
    from sasrec.data import (
        SequentialDataset,
        TrainIterableDataset,
        collate_fn,
        read_week_item_ids,
    )
    from sasrec.embeddings import build_weight_matrix
    from sasrec.model import SASRec
    from sasrec.train import evaluate, train_epoch

    cfg = SASRecConfig(
        data_dir=TRAIN.data_dir, start_week=TRAIN.start_week, end_week=TRAIN.end_week,
        epochs=TRAIN.epochs, lr=TRAIN.lr, k=TRAIN.k,
        max_history=TRAIN.max_history, min_timespent_pos=TRAIN.min_timespent_pos,
        seed=TRAIN.seed, **MODELS.sasrec,
    )

    # словарь — предварительный проход по колонке item_id
    all_ids = set()
    for week in range(cfg.start_week, cfg.end_week + 1):
        all_ids.update(read_week_item_ids(cfg, week))
    all_ids.update(read_week_item_ids(cfg, cfg.val_week))
    id_to_idx = {old_id: idx + 1 for idx, old_id in enumerate(sorted(all_ids))}
    num_items = len(id_to_idx) + 1

    vecs_np, id_to_pos = load_embeddings_only(cfg.meta_data_root, cfg.emb_file, cfg.emb_width)
    weight_matrix = build_weight_matrix(id_to_idx, vecs_np, id_to_pos, cfg.emb_width)

    train_dataset = TrainIterableDataset(
        cfg, id_to_idx,
        range(cfg.start_week + cfg.history_weeks, cfg.end_week + 1),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        collate_fn=lambda batch: collate_fn(batch, cfg.max_history),
    )
    val_iter = TrainIterableDataset(cfg, id_to_idx, [cfg.val_week])
    val_samples = list(val_iter)
    val_loader = DataLoader(
        SequentialDataset(val_samples), batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        collate_fn=lambda batch: collate_fn(batch, cfg.max_history),
    )

    model = SASRec(num_items, cfg.emb_width, cfg.max_history,
                   cfg.num_heads, cfg.num_layers, cfg.dropout).to(device)
    with torch.no_grad():
        model.item_embedding.weight.copy_(weight_matrix.to(device))
        model.item_embedding.weight[0].zero_()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    for epoch in range(cfg.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        metrics = cast(dict, evaluate(model, val_loader, device=device, top_k=cfg.k))
        print(f"epoch {epoch + 1}/{cfg.epochs}: loss={loss:.5f} | "
              + " | ".join(f"{m}={v:.5f}" for m, v in metrics.items()))

    return {
        "model_state_dict": model.state_dict(),
        "id_to_idx": id_to_idx,
        "emb_width": cfg.emb_width,
        "max_len": cfg.max_history,
        "history_weeks": cfg.history_weeks,
        "num_items": num_items,
        "k": cfg.k,
    }


TRAINERS = {
    "most_popular": train_most_popular,
    "fpmc": train_fpmc,
    "light_gcn": train_light_gcn,
    "sasrec": train_sasrec,
}


def main():
    args = parse_args()
    device = get_device()
    seed_everything(TRAIN.seed)
    print(f"device: {device} | тренировочные недели {TRAIN.start_week}..{TRAIN.end_week}, "
          f"валидация {TRAIN.val_week}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in models:
        if name not in TRAINERS:
            print(f"[skip] неизвестная модель: {name}")
            continue
        if not _should_train(name, args.force_overwrite):
            print(f"[{name}] пропускаем обучение (используем существующее состояние)")
            continue
        print(f"\n=== Обучение {name} ===")
        payload = TRAINERS[name](device)
        _save(name, payload)

    print("\nГотово.")


if __name__ == "__main__":
    main()