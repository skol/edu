"""Точка входа: загрузка данных -> MostPopular -> метрики на валидации.

Пример:
    python -m most_popular.main --start-week 0 --end-week 24 --k 10
"""

import argparse
import time

from common.cli import add_common_args, config_from_args
from common.utils import seed_everything

from .config import MostPopularConfig
from .data import load_and_build
from .evaluate import evaluate
from .model import MostPopular


def parse_args():
    p = argparse.ArgumentParser(description="MostPopular для VK-LSVD")
    add_common_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config_from_args(args, MostPopularConfig)
    seed_everything(cfg.seed)

    print(f"Тренировочные недели: {cfg.start_week}..{cfg.end_week}, контрольная: {cfg.val_week}")

    t0 = time.time()
    bundle = load_and_build(cfg)
    print(f"users={bundle.n_users} items={bundle.n_items} val_users={len(bundle.val_user_idxs)} (загрузка {time.time()-t0:.1f}s)")

    model = MostPopular(bundle.popularity, bundle.item_ranking)

    t1 = time.time()
    metrics = evaluate(model, bundle, k=cfg.k)
    print(f"метрики ({time.time()-t1:.1f}s): {metrics}")

    # пример рекомендаций для первого валидируемого юзера
    u = bundle.val_user_idxs[0]
    rec = model.recommend(bundle.seen[u], top_n=cfg.k)
    print(f"пример рекомендаций для user_idx={u}: {rec}")


if __name__ == "__main__":
    main()