"""Подготовка данных VK-LSVD для FPMC.

Файлы VK-LSVD не содержат колонки времени — порядок взаимодействий задаёт
порядок строк внутри parquet (недели идут train: 00..24, validation: 25).
Поэтому последовательность пользователя = его строки в глобальном порядке файлов.

Разметка действий на позитив/негатив — единая для всех моделей
(common.labels.assign_labels): триплеты строятся только по позитивным действиям,
а явные негативы (skip/dislike) используются в negative sampling.
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
    """Конфигурация данных FPMC: общие настройки + модель-специфичные поля."""

    dim: int = 64
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
    """Готовые индексы для обучения и валидации."""
    user_to_idx: dict
    item_to_idx: dict
    n_users: int
    n_items: int

    # тренировочные триплеты (u, last_item, next_item) — только по позитивным действиям
    train_u: np.ndarray
    train_last: np.ndarray
    train_next: np.ndarray

    # позитивная история (index -> список item_idx в хронологическом порядке)
    history: dict
    # последний позитивный айтем истории каждого юзера (для MC-части)
    last_item: dict
    # ВСЕ взаимодействия юзера (pos+neg) для исключения из кандидатов при оценке
    seen: dict
    # явные негативы (skip/dislike) юзера для negative sampling
    explicit_negatives: dict

    # валидация: user_idx -> список верных (позитивных) item_idx
    val_user_idxs: np.ndarray
    val_gt: list


def load_and_build(cfg: DataConfig) -> DataBundle:
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

    # ---------- FPMC-специфичные триплеты (u, last, next) ----------
    last_item: dict = {}        # user_idx -> last item_idx
    triplets = []               # (u, last, next)
    for u, seq in history.items():
        if len(seq) >= cfg.min_history:
            last_item[u] = seq[-1]
            for t in range(1, len(seq)):
                triplets.append((u, seq[t - 1], seq[t]))

    if triplets:
        triplets = np.asarray(triplets, dtype=np.int64)
    else:
        triplets = np.empty((0, 3), dtype=np.int64)

    # ---------- валидация (target = только позитивные действия недели валидации) ----------
    val_user_idxs, val_gt = build_validation(
        val_pos, train_df, user_to_idx, item_to_idx, seen)

    return DataBundle(
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        n_users=len(user_to_idx),
        n_items=len(item_to_idx),
        train_u=triplets[:, 0],
        train_last=triplets[:, 1],
        train_next=triplets[:, 2],
        history=history,
        last_item=last_item,
        seen=seen,
        explicit_negatives=explicit_negatives,
        val_user_idxs=val_user_idxs,
        val_gt=val_gt,
    )


class FPMCDataset(Dataset):
    """Тренировочный датасет: по триплету семплирует негативный айтем.

    Негатив берётся из явных негативов пользователя (skip/dislike), если они
    есть, иначе — случайный айтем (common.negatives.sample_negative).
    """

    def __init__(self, bundle: DataBundle, n_items: int, seed: int = 42):
        self.u = torch.as_tensor(bundle.train_u)
        self.last = torch.as_tensor(bundle.train_last)
        self.next = torch.as_tensor(bundle.train_next)
        self.n_items = n_items
        self.explicit_negatives = bundle.explicit_negatives
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u = int(self.u[idx].item())
        pos_item = int(self.next[idx].item())
        neg = sample_negative(
            self.explicit_negatives.get(u, ()),
            self.n_items,
            exclude=(pos_item,),
            rng=self.rng,
        )
        return self.u[idx], self.last[idx], self.next[idx], neg