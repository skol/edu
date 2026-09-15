"""Тренировка SASRec с BPR-подобным loss и оценка Hit@k / NDCG@k."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.metrics import eval_metrics
from common.negatives import sample_negative
from common.utils import get_device, seed_everything

from .model import SASRec


def train_epoch(model: SASRec, loader: DataLoader, optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for _, input_batch, pos_batch, target_batch, skipped_batch in loader:
        optimizer.zero_grad()
        input_batch = input_batch.to(device)
        pos_batch = pos_batch.to(device)
        target_batch = target_batch.to(device)
        user_vecs = model(input_batch, pos_batch)
        user_vecs_norm = F.normalize(user_vecs, p=2, dim=1)
        pos_embs = model.item_embedding(target_batch)
        pos_embs_norm = F.normalize(pos_embs, p=2, dim=1)

        # ---- Negative sampling (единый helper из common.negatives) ----
        neg_indices = []
        num_items = model.item_embedding.num_embeddings
        for tgt, skipped in zip(target_batch.detach().cpu().tolist(), skipped_batch):
            # явные негативы юзера (без padding-индекса 0); target тоже исключается
            neg_idx = sample_negative(
                [s for s in skipped if s != 0],
                num_items,
                exclude=(tgt, 0),
            )
            neg_indices.append(neg_idx)
        neg_indices = torch.tensor(neg_indices, dtype=torch.long, device=device)
        neg_embs = model.item_embedding(neg_indices)
        neg_embs_norm = F.normalize(neg_embs, p=2, dim=1)

        pos_scores = (user_vecs_norm * pos_embs_norm).sum(dim=1)
        neg_scores = (user_vecs_norm * neg_embs_norm).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Loss became NaN/Inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    # len(loader) недоступен для IterableDataset, поэтому считаем батчи сами
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: SASRec, loader: DataLoader, device: str, top_k: int = 10,
             return_per_user: bool = False):
    """Recall@k / NDCG@k / MRR@k (+ Hit@k) на валидационных сэмплах.

    Из кандидатов исключаются позитивная история (input) и явные негативы
    (skipped), кроме самого target. Ранги агрегируются по user_id — это
    позволяет считать метрики по группам пользователей.

    Returns:
        dict метрик; при return_per_user=True — (user_ids, gt_positions).
    """
    model.eval()

    item_weights = model.item_embedding.weight
    item_weights_norm = F.normalize(item_weights, p=2, dim=1)
    if not torch.isfinite(item_weights_norm).all():
        raise RuntimeError("NaN/Inf in item embeddings during evaluation")

    user_ranks: dict = {}   # user_id -> список 1-based рангов (0 = не попал в топ-k)
    for user_ids, input_batch, pos_batch, target_batch, skipped_batch in loader:
        input_batch = input_batch.to(device)
        pos_batch = pos_batch.to(device)
        target_batch = target_batch.to(device)
        user_vecs = model(input_batch, pos_batch)
        user_vecs_norm = F.normalize(user_vecs, p=2, dim=1)
        if not torch.isfinite(user_vecs_norm).all():
            raise RuntimeError("NaN/Inf in user vectors during evaluation")
        scores = torch.matmul(user_vecs_norm, item_weights_norm.T)

        # Не рекомендуем padding.
        scores[:, 0] = -float("inf")

        # Не рекомендуем уже просмотренные items (позитивная история).
        seen = input_batch.ne(0)
        scores.scatter_(
            1, input_batch,
            torch.where(
                seen,
                torch.full_like(input_batch, -float("inf"), dtype=scores.dtype),
                scores.gather(1, input_batch),
            ),
        )
        # Не рекомендуем явно негативные items (skip/dislike), кроме target.
        for i, skipped in enumerate(skipped_batch):
            tgt = target_batch[i].item()
            for s in skipped:
                if s != 0 and s != tgt:
                    scores[i, s] = -float("inf")
        _, topk_indices = torch.topk(scores, k=top_k, dim=1)
        for i, (uid, tgt) in enumerate(zip(user_ids, target_batch.tolist())):
            matches = (topk_indices[i] == tgt)
            rank = int(torch.where(matches)[0][0].item()) + 1 if matches.any() else 0
            user_ranks.setdefault(int(uid), []).append(rank)

    user_ids_arr = sorted(user_ranks.keys())
    gt_positions = [np.asarray(user_ranks[u], dtype=int) for u in user_ids_arr]
    metrics = eval_metrics(gt_positions, top_k)
    n_users = len(gt_positions)
    metrics[f"hit@{top_k}"] = float(
        sum(int((p > 0).any()) for p in gt_positions) / n_users if n_users else 0.0
    )

    if return_per_user:
        return user_ids_arr, gt_positions
    return metrics