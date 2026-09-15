"""Разбиение пользователей валидации на группы холодный/тёплый/горячий.

Размер истории пользователя = число ВСЕХ его действий (позитивных и негативных)
в тренировочном периоде, размеченных единой функцией common.labels.assign_labels.

Группы (пороги — параметры):
  - cold: размер истории <= cold_max (по умолчанию 30);
  - warm: cold_max < размер <= warm_max (по умолчанию 31..70);
  - hot:  размер > warm_max (>= 71).

Каждая группа ограничена по размеру (group_limit, по умолчанию 100 — быстрые
тестовые прогоны; для продакшена лимит задаётся явно). Если подходящих
пользователей больше лимита — выбирается случайное (воспроизводимое) подмножество.
В группы попадают только пользователи, присутствующие в тренировочной части.
"""

import random

import polars as pl

from .labels import assign_labels

# Порядок групп для вывода.
GROUP_NAMES = ["cold", "warm", "hot"]


def user_history_sizes(train_df: pl.DataFrame, min_timespent_pos: int) -> dict:
    """Размер истории каждого пользователя: число всех размеченных действий."""
    labeled = assign_labels(train_df, min_timespent_pos)
    counts = labeled.group_by("user_id").agg(pl.len().alias("n"))
    return {int(user_id): int(n) for user_id, n in counts.iter_rows()}


def split_users_by_history(
    val_df: pl.DataFrame,
    train_df: pl.DataFrame,
    min_timespent_pos: int,
    cold_max: int = 30,
    warm_max: int = 70,
    group_limit: int = 100,
    seed: int = 42,
) -> dict:
    """Возвращает {group: list[user_id]} для пользователей валидации.

    Args:
        val_df: валидационный/тестовый датасет (неделя 25).
        train_df: тренировочный датасет (недели [start_week, end_week]).
        min_timespent_pos: порог времени просмотра для разметки позитив/негатив.
        cold_max: верхняя граница «холодной» истории (<=).
        warm_max: верхняя граница «тёплой» истории (cold_max < size <= warm_max).
        group_limit: максимальный размер каждой группы; при превышении —
            случайный выбор подмножества.
        seed: seed случайного выбора (воспроизводимость).

    Returns:
        dict с ключами cold/warm/hot и списками исходных user_id.
    """
    history_sizes = user_history_sizes(train_df, min_timespent_pos)
    val_users = set(val_df["user_id"].unique().to_list())

    candidates: dict = {"cold": [], "warm": [], "hot": []}
    for user_id, size in history_sizes.items():
        if user_id not in val_users:
            continue
        if size <= cold_max:
            candidates["cold"].append(user_id)
        elif size <= warm_max:
            candidates["warm"].append(user_id)
        else:
            candidates["hot"].append(user_id)

    rng = random.Random(seed)
    groups: dict = {}
    for group in GROUP_NAMES:
        pool = sorted(candidates[group])
        if len(pool) > group_limit:
            pool = rng.sample(pool, group_limit)
        groups[group] = pool
    return groups


def categorize_by_history_size(size: int, cold_max: int = 30, warm_max: int = 70) -> str:
    """Категория пользователя по размеру истории: cold / warm / hot."""
    if size <= cold_max:
        return "cold"
    if size <= warm_max:
        return "warm"
    return "hot"