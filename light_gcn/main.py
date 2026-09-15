"""Точка входа: загрузка данных -> тренировка LightGCN -> метрики.

Пример:
    python -m light_gcn.main --epochs 10 --dim 64 --k 10
"""

import argparse
import time

import torch

from common.cli import add_common_args, config_from_args
from common.utils import get_device, seed_everything

from .data import DataConfig, load_and_build
from .model import LightGCN
from .train import evaluate, train_model


def parse_args():
    p = argparse.ArgumentParser(description="LightGCN для VK-LSVD")
    add_common_args(p)
    p.add_argument("--dim", type=int, default=64, help="Размерность эмбеддингов")
    p.add_argument("--n-layers", type=int, default=3, help="Число слоёв графа")
    p.add_argument("--batch-size", type=int, default=1024, help="Размер батча")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config_from_args(args, DataConfig)
    device = get_device()
    seed_everything(cfg.seed)
    print(f"device: {device}")

    t0 = time.time()
    bundle = load_and_build(cfg)
    print(f"users={bundle.n_users} items={bundle.n_items} "
          f"edges={len(bundle.train_users)} val_users={len(bundle.val_user_idxs)} "
          f"({time.time()-t0:.1f}s)")

    model = LightGCN(n_users=bundle.n_users, n_items=bundle.n_items,
                     dim=cfg.dim, n_layers=cfg.n_layers)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    train_model(model, bundle, device=device,
                epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
                seed=cfg.seed)

    t1 = time.time()
    metrics = evaluate(model, bundle, device=device, k=cfg.k)
    print(f"metrics ({time.time()-t1:.1f}s): {metrics}")

    # сохраняем состояние модели на диск
    torch.save(model.state_dict(), "light_gcn.pt")
    print("saved to light_gcn.pt")


if __name__ == "__main__":
    main()