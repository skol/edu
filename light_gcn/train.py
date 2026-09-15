"""Тренировка LightGCN с BPR-loss и оценка на валидации."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.metrics import eval_metrics

from .data import DataBundle, LightGCNDataset, build_normalized_adjacency
from .model import LightGCN


def _prepare_adjacency(model: LightGCN, bundle: DataBundle, device: str):
    """Строит нормализованную матрицу смежности и закрепляет её в модели."""
    norm_adj = build_normalized_adjacency(
        bundle.n_users, bundle.n_items, bundle.train_users, bundle.train_items, device)
    model.norm_adj = norm_adj


def train_model(
    model: LightGCN,
    bundle: DataBundle,
    device: str,
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-3,
    log_every: int = 200,
    seed: int = 42,
):
    """Обучает LightGCN через BPR-потерю с негативным семплированием."""
    _prepare_adjacency(model, bundle, device)

    dataset = LightGCNDataset(
        bundle.train_users, bundle.train_items,
        n_items=bundle.n_items,
        explicit_negatives=bundle.explicit_negatives,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device).train()
    step = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for u, pos_i, neg_i in loader:
            u, pos_i, neg_i = (u.to(device), pos_i.to(device), neg_i.to(device))
            # полный проход по графу для текущих эмбеддингов
            user_emb, item_emb = model()
            loss, _, _ = model.bpr_loss(user_emb, item_emb, u, pos_i, neg_i)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        print(f"--- epoch {epoch}/{epochs} avg_loss {total_loss / n_batches:.4f}")


def evaluate(
    model: LightGCN,
    bundle: DataBundle,
    device: str,
    k: int = 10,
    batch_cand: int = 4096,
    return_per_user: bool = False,
):
    """Recall@k / NDCG@k / MRR@k на юзерах из валидации.

    Из кандидатов исключаются все взаимодействия пользователя (pos+neg).

    Returns:
        dict метрик; при return_per_user=True — (user_idxs, gt_positions).
    """
    _prepare_adjacency(model, bundle, device)
    model.to(device).eval()

    all_scores = []   # np-массивы скоров для валидных юзеров
    kept_pos = []     # позиция юзера в bundle.val_user_idxs
    last_ids = []     # списки кандидатов для каждого юзера

    with torch.no_grad():
        user_emb, item_emb = model()

        for pos, u in enumerate(bundle.val_user_idxs):
            # исключаем все взаимодействия пользователя (pos+neg)
            seen = bundle.seen[u]
            cand = np.array([i for i in range(bundle.n_items) if i not in seen],
                            dtype=np.int64)
            if len(cand) == 0:
                continue

            u_t = torch.full((1,), u, device=device, dtype=torch.long)
            scores = torch.empty(len(cand), device=device)
            for s in range(0, len(cand), batch_cand):
                chunk = torch.as_tensor(cand[s:s + batch_cand], device=device)
                # скор = <эмбеддинг юзера, эмбеддинг ролика>
                scores[s:s + len(chunk)] = model.score(
                    user_emb, item_emb, u_t.expand(len(chunk)), chunk)
            all_scores.append(scores.cpu().numpy())
            kept_pos.append(pos)
            last_ids.append(cand)

    # приведём к единой размерности по максимуму кандидатов
    # (юзеры могут иметь разное число кандидатов — дополним -inf в хвосте)
    max_cand = max(len(s) for s in all_scores)
    mat = np.full((len(all_scores), max_cand), -np.inf)
    for i, s in enumerate(all_scores):
        mat[i, :len(s)] = s
    # какой item_idx стоит в каждом столбце матрицы для каждого юзера
    col_to_item = [np.concatenate([last_ids[i],
                                   np.zeros(max_cand - len(last_ids[i]), dtype=np.int64)])
                   for i in range(len(all_scores))]

    # строим gt_positions напрямую по матрице
    pred_order = np.argsort(-mat, axis=1)
    gt_positions = []
    for i in range(mat.shape[0]):
        # ранг каждого столбца у юзера i
        rank = np.empty(max_cand, dtype=int)
        rank[pred_order[i]] = np.arange(1, max_cand + 1)
        gt_set = set(bundle.val_gt[kept_pos[i]])
        col_of_gt = [c for c in range(max_cand) if col_to_item[i][c] in gt_set]
        positions = np.array([rank[c] for c in col_of_gt], dtype=int)
        gt_positions.append(positions)

    metrics = eval_metrics(gt_positions, k)
    if return_per_user:
        return bundle.val_user_idxs[kept_pos], gt_positions
    return metrics