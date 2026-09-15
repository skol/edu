"""Загрузка и сборка матрицы item-эмбеддингов."""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from common.embeddings import load_embeddings_only


def build_weight_matrix(
    id_to_idx: Dict[int, int],
    vecs_np: np.ndarray,
    id_to_pos: Dict[int, int],
    emb_width: int,
) -> torch.Tensor:
    """Собирает весовую матрицу эмбеддингов; для отсутствующих — случайная инициализация."""
    num_items = len(id_to_idx) + 1
    weight_matrix = torch.zeros(num_items, emb_width, dtype=torch.float32)
    missing_count = 0
    stdv = 1.0 / np.sqrt(emb_width)
    for old_id, new_idx in id_to_idx.items():
        pos = id_to_pos.get(old_id)
        if pos is not None:
            vec = torch.from_numpy(vecs_np[pos])
            vec = F.normalize(vec, p=2, dim=0)
            if not torch.isfinite(vec).all():
                raise RuntimeError(f"Bad embedding for item {old_id}")
            weight_matrix[new_idx].copy_(vec)
        else:
            weight_matrix[new_idx].uniform_(-stdv, stdv)
            missing_count += 1

    print("Weight matrix NaN:", torch.isnan(weight_matrix).sum().item())
    print("Weight matrix Inf:", torch.isinf(weight_matrix).sum().item())

    if not torch.isfinite(weight_matrix).all():
        raise RuntimeError("weight_matrix contains NaN/Inf")
    print(f"Missing pretrained embeddings: {missing_count:,}")

    return weight_matrix