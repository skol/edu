"""Метрики next-item рекомендаций: Recall@k, NDCG@k, MRR@k."""

import numpy as np


def _ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """ranks: 1-based ранги верных айтемов (0 = не попал в топ-k)."""
    dcg = 0.0
    for r in ranks:
        if 0 < r <= k:
            dcg += 1.0 / np.log2(r + 1)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def eval_metrics(gt_positions: list[np.ndarray], k: int) -> dict:
    """Подсчёт метрик по рангам верных айтемов.

    Args:
        gt_positions: для каждого юзера массив 1-based рангов верных айтемов
                      среди отсортированных кандидатов; 0 = не попал в топ.
        k: глубина топ-k.

    Returns:
        dict с recall@k, ndcg@k, mrr@k.
    """
    n_users = len(gt_positions)
    total_hits = 0
    total_gt = 0
    ndcg_sum = 0.0
    mrr_sum = 0.0

    for positions in gt_positions:
        hits = int(np.sum((positions > 0) & (positions <= k)))
        total_hits += hits
        total_gt += len(positions)
        ndcg_sum += _ndcg_at_k(positions, k)
        valid = positions[positions > 0]
        mrr_sum += (1.0 / valid[0]) if valid.size else 0.0

    return {
        f"recall@{k}": float(total_hits / total_gt if total_gt else 0.0),
        f"ndcg@{k}": float(ndcg_sum / n_users if n_users else 0.0),
        f"mrr@{k}": float(mrr_sum / n_users if n_users else 0.0),
    }