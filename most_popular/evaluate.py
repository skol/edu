"""Метрики next-item рекомендаций: Recall@k, NDCG@k, MRR@k."""

from common.metrics import eval_metrics

from .data import DataBundle


def evaluate(model, bundle: DataBundle, k: int = 10, return_per_user: bool = False):
    """Recall@k / NDCG@k / MRR@k на юзерах из валидации.

    Из кандидатов исключаются все взаимодействия пользователя (pos+neg).

    Returns:
        dict метрик; при return_per_user=True — (user_idxs, gt_positions).
    """
    gt_positions = []
    for pos, u in enumerate(bundle.val_user_idxs):
        gt = bundle.val_gt[pos]
        ranks = model.rank_positions(bundle.seen[u], gt)
        gt_positions.append(ranks)
    metrics = eval_metrics(gt_positions, k)
    if return_per_user:
        return bundle.val_user_idxs, gt_positions
    return metrics