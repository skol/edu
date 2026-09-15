"""Точка входа: словарь -> ленивое обучение SASRec -> метрики.

Пример:
    python -m sasrec.main --start-week 0 --end-week 24 --epochs 5
"""

import argparse
from typing import cast

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from common.cli import add_common_args, config_from_args
from common.embeddings import load_embeddings_only
from common.utils import get_device, seed_everything

from .config import SASRecConfig
from .data import (
    SequentialDataset,
    TrainIterableDataset,
    collate_fn,
    read_week_item_ids,
)
from .embeddings import build_weight_matrix
from .model import SASRec
from .train import evaluate, train_epoch


def parse_args():
    p = argparse.ArgumentParser(description="SASRec для VK-LSVD")
    add_common_args(p)
    p.add_argument("--history-weeks", type=int, default=None,
                   help="Сколько недель истории приходится на один target-сэмпл")
    p.add_argument("--batch-size", type=int, default=None, help="Размер батча")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Число воркеров DataLoader")
    p.add_argument("--emb-width", type=int, default=None, help="Ширина эмбеддингов")
    p.add_argument("--num-heads", type=int, default=None, help="Число голов внимания")
    p.add_argument("--num-layers", type=int, default=None, help="Число слоёв Transformer")
    p.add_argument("--dropout", type=float, default=None, help="Dropout")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config_from_args(args, SASRecConfig)
    seed_everything(cfg.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Temporal setup: [{cfg.history_weeks} weeks] -> 1 week")

    if cfg.end_week < cfg.start_week + cfg.history_weeks:
        raise RuntimeError(
            "Нет ни одного тренировочного окна: end_week должен быть "
            ">= start_week + history_weeks"
        )

    # --------------------------------------------------------
    # Словарь строится предварительным проходом только по колонке item_id
    # (все train-недели + валидационная неделя), без материализации сэмплов.
    # --------------------------------------------------------
    print("Building vocabulary (item_id pre-pass)...")
    all_ids = set()
    for week in range(cfg.start_week, cfg.end_week + 1):
        all_ids.update(read_week_item_ids(cfg, week))
    all_ids.update(read_week_item_ids(cfg, cfg.val_week))  # неделя 25 (validation/)
    id_to_idx = {old_id: idx + 1 for idx, old_id in enumerate(sorted(all_ids))}
    num_items = len(id_to_idx) + 1
    print(f"\nVocabulary: {len(id_to_idx):,} items")

    # --------------------------------------------------------
    # Train: ленивый IterableDataset (недели читаются на лету)
    # --------------------------------------------------------
    train_targets = range(cfg.start_week + cfg.history_weeks, cfg.end_week + 1)
    train_dataset = TrainIterableDataset(cfg, id_to_idx, train_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: collate_fn(batch, cfg.max_history),
    )

    # --------------------------------------------------------
    # Validation: один слайдинг-сэмпл на позитив валидационной недели.
    # Материализуем один раз (валидация = 1 неделя, объём мал) — иначе
    # каждая эпоха перечитывала бы валидационные недели заново.
    # --------------------------------------------------------
    val_iter = TrainIterableDataset(cfg, id_to_idx, [cfg.val_week])
    val_samples = list(val_iter)
    print(f"\nVal samples: {len(val_samples):,}")
    if not val_samples:
        raise RuntimeError("No validation samples.")
    val_dataset = SequentialDataset(val_samples)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: collate_fn(batch, cfg.max_history),
    )

    # --------------------------------------------------------
    # Pretrained embeddings -> весовая матрица
    # --------------------------------------------------------
    vecs_np, id_to_pos = load_embeddings_only(cfg.meta_data_root, cfg.emb_file, cfg.emb_width)
    weight_matrix = build_weight_matrix(
        id_to_idx=id_to_idx, vecs_np=vecs_np, id_to_pos=id_to_pos, emb_width=cfg.emb_width
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = SASRec(
        num_items=num_items,
        emb_dim=cfg.emb_width,
        max_len=cfg.max_history,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    with torch.no_grad():
        model.item_embedding.weight.copy_(weight_matrix.to(device))
    # padding embedding должен оставаться нулевым.
    with torch.no_grad():
        model.item_embedding.weight[0].zero_()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_ndcg = -1.0
    for epoch in range(cfg.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        metrics = cast(dict, evaluate(model, val_loader, device=device, top_k=cfg.k))
        print(f"Epoch {epoch + 1:02d}: Loss={loss:.5f} | "
              + " | ".join(f"{m}={v:.5f}" for m, v in metrics.items()))
        ndcg = metrics.get(f"ndcg@{cfg.k}", -1.0)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "id_to_idx": id_to_idx,
                    "config": {
                        "emb_width": cfg.emb_width,
                        "max_len": cfg.max_history,
                        "history_weeks": cfg.history_weeks,
                        "top_k": cfg.k,
                    },
                    "best_ndcg": best_ndcg,
                },
                "sasrec_best.pth",
            )
            print("  -> saved best model")
    print("\nTraining finished.")


if __name__ == "__main__":
    main()