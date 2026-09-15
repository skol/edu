"""Подготовка данных VK-LSVD для LightGCN.

Файлы VK-LSVD не содержат колонки времени — порядок взаимодействий задаёт
порядок строк внутри parquet (недели идут train: 00..24, validation: 25).
Поэтому последовательность пользователя = его строки в глобальном порядке файлов.

LightGCN работает с двудольным графом "пользователь -- ролик". Разметка действий
на позитив/негатив — единая для всех моделей (common.labels.assign_labels):
положительные рёбра графа строятся только по позитивным действиям, а явные
негативы (skip/dislike) используются в negative sampling.
"""

import os
import random
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from common.config import CommonConfig
from common.data import (
    build_id_maps,
    build_sequences_with_maps,
    build_validation,
    read_files,
)
from common.labels import assign_labels
from common.negatives import sample_negative

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


@dataclass
class DataConfig(CommonConfig):
    """Конфигурация данных LightGCN: общие настройки + модель-специфичные поля.

    min_history в графовой модели не применяется (в граф попадают все
    пользователи, как и раньше), чтобы не менять качество модели.
    """

    dim: int = 64
    n_layers: int = 3
    batch_size: int = 1024

    train_files: list = field(default_factory=list)
    val_files: list = field(default_factory=list)

    def __post_init__(self):
        if not self.train_files:
            self.train_files = [f"{self.data_dir}/train/week_{i:02}.parquet"
                                for i in self.train_weeks]
        if not self.val_files:
            dir_name = 'validation' if self.val_week == 25 else 'train'
            self.val_files = [f"{self.data_dir}/{dir_name}/week_{self.val_week:02}.parquet"]


@dataclass
class DataBundle:
    """Готовые индексы и рёбра графа для обучения LightGCN."""
    user_to_idx: dict
    item_to_idx: dict
    n_users: int
    n_items: int

    # положительные рёбра графа "пользователь -> ролик" (только позитивные действия)
    train_users: np.ndarray
    train_items: np.ndarray

    # позитивная история (user_idx -> list[item_idx])
    history: dict
    # ВСЕ взаимодействия юзера (pos+neg) для исключения из кандидатов при оценке
    seen: dict
    # явные негативы (skip/dislike) юзера для negative sampling
    explicit_negatives: dict

    # валидация: user_idx -> список верных (позитивных) item_idx
    val_user_idxs: np.ndarray
    val_gt: list


def load_and_build(cfg: DataConfig) -> DataBundle:
    """Читает train/validation parquet и строит граф взаимодействий.

    Каждый позитивный ролик в истории пользователя становится положительным
    ребром (u, i) — именно эти рёбра образуют матрицу смежности для LightGCN.
    """
    train_df = read_files(cfg.train_files)
    val_df = read_files(cfg.val_files)

    # ---------- единая разметка позитив/негатив ----------
    train_labeled = assign_labels(train_df, cfg.min_timespent_pos)
    val_labeled = assign_labels(val_df, cfg.min_timespent_pos)
    val_pos = val_labeled.filter(pl.col("label") == 1)

    # ---------- словари по полному train (pos и neg получают индексы) ----------
    user_to_idx, item_to_idx = build_id_maps(train_df)

    # ---------- позитивная история (хронологический порядок строк) ----------
    pos_train = train_labeled.filter(pl.col("label") == 1)
    history = build_sequences_with_maps(pos_train, user_to_idx, item_to_idx)

    # ---------- явные негативы и множество «всего увиденного» ----------
    neg_train = train_labeled.filter(pl.col("label") == 0)
    seen: dict = {u: set() for u in user_to_idx.values()}
    explicit_negatives: dict = {}
    for user_id, items in neg_train.group_by("user_id").agg("item_id").iter_rows():
        u = user_to_idx.get(user_id)
        if u is None:
            continue
        lst = sorted({item_to_idx[i] for i in items if i in item_to_idx})
        explicit_negatives[u] = lst
        seen[u].update(lst)
    for u, seq in history.items():
        seen[u].update(seq)

    # ---------- рёбра графа (каждый позитивный просмотр = коллаборативный сигнал) ----------
    edges_u: list[int] = []     # пользователи положительных рёбер
    edges_i: list[int] = []     # соответствующие ролики
    for u, seq in history.items():
        edges_u.extend([u] * len(seq))
        edges_i.extend(seq)

    # ---------- валидация (target = только позитивные действия недели валидации) ----------
    val_user_idxs, val_gt = build_validation(
        val_pos, train_df, user_to_idx, item_to_idx, seen)

    return DataBundle(
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        n_users=len(user_to_idx),
        n_items=len(item_to_idx),
        train_users=np.asarray(edges_u, dtype=np.int64),
        train_items=np.asarray(edges_i, dtype=np.int64),
        history=history,
        seen=seen,
        explicit_negatives=explicit_negatives,
        val_user_idxs=val_user_idxs,
        val_gt=val_gt,
    )


def build_normalized_adjacency(n_users: int, n_items: int,
                               users: np.ndarray, items: np.ndarray,
                               device=None) -> torch.Tensor:
    """Нормализованная матрица смежности A~ = D^{-1/2} A D^{-1/2} (разреженная).

    Граф двудольный: узлы 0..n_users-1 — пользователи,
    узлы n_users..n_users+n_items-1 — ролики. Вес ребра (a,b) = 1/sqrt(deg(a)*deg(b)),
    что соответствует симметричной нормализации LightGCN.

    Returns:
        COO-тензор формы [N, N], N = n_users + n_items.
    """
    N = n_users + n_items
    u_nodes = users
    i_nodes = items + n_users  # сдвиг индексов роликов за блок пользователей

    # оба направления (граф неориентированный)
    row = np.concatenate([u_nodes, i_nodes])
    col = np.concatenate([i_nodes, u_nodes])

    deg = np.bincount(row, minlength=N).astype(np.float64)
    # симметричная нормализация: 1/sqrt(deg(отправитель) * deg(получатель))
    val = 1.0 / np.sqrt(deg[row] * deg[col])

    indices = torch.as_tensor(np.stack([row, col]), dtype=torch.long)
    values = torch.as_tensor(val, dtype=torch.float32)
    # отключаем проверку инвариантов разреженности (у нас заведомо корректные рёбра)
    with torch.sparse.check_sparse_tensor_invariants(False):
        adj = torch.sparse_coo_tensor(indices, values, (N, N))
    if device is not None:
        adj = adj.to(device)
    return adj.coalesce()


class LightGCNDataset(Dataset):
    """Тренировочный датасет: по положительному ребру семплирует негативный ролик.

    Негатив берётся из явных негативов пользователя (skip/dislike), если они
    есть, иначе — случайный ролик (common.negatives.sample_negative).
    """

    def __init__(self, users: np.ndarray, items: np.ndarray, n_items: int,
                 explicit_negatives: dict | None = None, seed: int = 42):
        self.users = torch.as_tensor(users)
        self.items = torch.as_tensor(items)
        self.n_items = n_items
        self.explicit_negatives = explicit_negatives or {}
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = int(self.users[idx].item())
        pos_item = int(self.items[idx].item())
        neg = sample_negative(
            self.explicit_negatives.get(u, ()),
            self.n_items,
            exclude=(pos_item,),
            rng=self.rng,
        )
        return self.users[idx], self.items[idx], neg