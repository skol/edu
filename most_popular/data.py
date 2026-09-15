"""Загрузка и подготовка данных VK-LSVD для MostPopular.

Правила разметки примеров (единые для всех моделей проекта, common.labels):
  - положительный: timespent >= min_timespent_pos ИЛИ одна из реакций
    like / share / bookmark / click_on_author / open_comments;
  - отрицательный: НЕ позитив И (timespent < min_timespent_pos ИЛИ dislike);
  - строки, не попавшие ни в одну категорию, отбрасываются.

Популярность айтема = число положительных взаимодействий за тренировочный период.
Айтем «ещё не видел» пользователь, если у него нет НИ ОДНОЙ записи (позитивной или
негативной) с этим айтемом за тренировочный период.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from common.data import build_id_maps, read_week, read_weeks
from common.labels import assign_labels


@dataclass
class DataBundle:
    """Подготовленные данные для обучения/валидации MostPopular."""

    user_to_idx: dict            # user_id -> внутренний индекс
    item_to_idx: dict            # item_id -> внутренний индекс
    n_users: int
    n_items: int

    # популярность айтемов по внутреннему индексу (число положительных примеров)
    popularity: np.ndarray
    # порядок айтемов по убыванию популярности (внутренние индексы)
    item_ranking: np.ndarray

    # seen[user_idx] -> set(item_idx): что юзер уже видел за тренировочный период
    seen: dict

    # валидация: user_idx -> список верных (положительных) item_idx на валид. неделе
    val_user_idxs: np.ndarray
    val_gt: list


def load_and_build(cfg) -> DataBundle:
    train_df = read_weeks(cfg.data_dir, cfg.train_weeks)
    val_df = read_week(cfg.data_dir, cfg.val_week)

    train_df = assign_labels(train_df, cfg.min_timespent_pos)
    val_df = assign_labels(val_df, cfg.min_timespent_pos)

    # ---------- словари по тренировочному периоду ----------
    user_to_idx, item_to_idx = build_id_maps(train_df)

    # ---------- популярность айтемов (число положительных примеров) ----------
    pos = train_df.filter(pl.col("label") == 1)
    pop_counts = (
        pos.group_by("item_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    popularity = np.zeros(len(item_to_idx), dtype=np.int64)
    for item_id, cnt in pop_counts.iter_rows():
        popularity[item_to_idx[item_id]] = cnt
    # порядок айтемов по убыванию популярности (стабильный по индексу для детерминизма)
    item_ranking = np.argsort(-popularity, kind="stable")

    # ---------- что видел каждый юзер (все взаимодействия, pos и neg) ----------
    seen: dict = {}
    for user_id, item_ids in train_df.group_by("user_id").agg("item_id").iter_rows():
        u = user_to_idx[user_id]
        seen[u] = {item_to_idx[i] for i in item_ids}

    # ---------- валидация ----------
    # целевые айтемы — положительные на валидационной неделе, которых юзер НЕ видел
    val_pos = val_df.filter(pl.col("label") == 1)
    val_g = val_pos.group_by("user_id").agg("item_id")
    val_user_idxs = []
    val_gt = []
    for user_id, item_ids in val_g.iter_rows():
        u = user_to_idx.get(user_id)
        if u is None:
            continue
        gt = [item_to_idx[i] for i in item_ids
              if i in item_to_idx and item_to_idx[i] not in seen[u]]
        if gt:
            val_user_idxs.append(u)
            val_gt.append(gt)

    return DataBundle(
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        n_users=len(user_to_idx),
        n_items=len(item_to_idx),
        popularity=popularity,
        item_ranking=item_ranking,
        seen=seen,
        val_user_idxs=np.asarray(val_user_idxs, dtype=np.int64),
        val_gt=val_gt,
    )