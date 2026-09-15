"""Общая подготовка данных VK-LSVD: чтение parquet и построение индексов.

Файлы VK-LSVD не содержат колонки времени — порядок взаимодействий задаёт
порядок строк внутри parquet. Поэтому последовательность пользователя = его
строки в глобальном порядке файлов.
"""

import numpy as np
import polars as pl


def read_files(files: list[str]) -> pl.DataFrame:
    """Читает несколько parquet-файлов и конкатенирует их."""
    return pl.concat([pl.read_parquet(f) for f in files])


def read_week(data_dir: str, week: int) -> pl.DataFrame:
    """Читает parquet конкретной недели, ища его и в train/, и в validation/."""
    for sub in ("train", "validation"):
        path = f"{data_dir}/{sub}/week_{week:02}.parquet"
        try:
            return pl.read_parquet(path)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Не найдена неделя {week} ни в {data_dir}/train, ни в {data_dir}/validation"
    )


def read_weeks(data_dir: str, weeks) -> pl.DataFrame:
    """Читает несколько недель подряд и конкатенирует их."""
    return pl.concat([read_week(data_dir, w) for w in weeks])


def build_id_maps(df: pl.DataFrame):
    """Строит словари user_id -> idx и item_id -> idx по отсортированным id."""
    users_sorted = df["user_id"].unique().sort().to_list()
    items_sorted = df["item_id"].unique().sort().to_list()
    user_to_idx = {u: i for i, u in enumerate(users_sorted)}
    item_to_idx = {i: j for j, i in enumerate(items_sorted)}
    return user_to_idx, item_to_idx


def build_sequences_with_maps(df: pl.DataFrame, user_to_idx: dict, item_to_idx: dict) -> dict:
    """Строит хронологические истории пользователей по готовым словарям.

    Args:
        df: подвыборка строк (например, только позитивные действия).
        user_to_idx, item_to_idx: словари исходных id -> компактные индексы.

    Returns:
        history: dict[user_idx -> list[item_idx]] в хронологическом порядке
        (по глобальному порядку строк).
    """
    df = df.with_row_index("__idx")
    grouped = (df.group_by("user_id", maintain_order=True)
               .agg([pl.col("item_id").alias("items"),
                     pl.col("__idx").alias("idx")]))

    history: dict = {}
    for user_id, items, idx in grouped.iter_rows():
        # сортируем по глобальному индексу строки -> хронологический порядок
        order = np.argsort(idx)
        seq = [item_to_idx[items[o]] for o in order]
        history[user_to_idx[user_id]] = seq

    return history


def build_user_sequences(df: pl.DataFrame):
    """Строит хронологические истории пользователей.

    Returns:
        (user_to_idx, item_to_idx, history), где history[user_idx] =
        list[item_idx] в хронологическом порядке (по глобальному порядку строк).
    """
    user_to_idx, item_to_idx = build_id_maps(df)
    history = build_sequences_with_maps(df, user_to_idx, item_to_idx)
    return user_to_idx, item_to_idx, history


def build_validation(val_df, train_df, user_to_idx, item_to_idx, history):
    """Готовит валидационные пары (user_idx -> верные item_idx).

    Берёт юзеров из трейна, у которых есть целевые айтемы в валидации.
    Верными считаются айтемы, которых НЕТ в истории юзера.

    Returns:
        (val_user_idxs: np.ndarray, val_gt: list).
    """
    val_mask = val_df["user_id"].is_in(train_df["user_id"]) \
        & val_df["item_id"].is_in(train_df["item_id"])
    val_df = val_df.filter(val_mask).with_row_index("__vidx")

    val_g = (val_df.group_by("user_id", maintain_order=True)
             .agg([pl.col("item_id").alias("items"), pl.col("__vidx").alias("idx")]))

    val_user_idxs = []
    val_gt = []
    for user_id, items, idx in val_g.iter_rows():
        u = user_to_idx[user_id]
        # верные айтемы, которых НЕТ в истории (иначе предсказывать нечего)
        gt = [item_to_idx[i] for i in items if item_to_idx[i] not in set(history[u])]
        if gt:
            val_user_idxs.append(u)
            val_gt.append(gt)

    return np.asarray(val_user_idxs, dtype=np.int64), val_gt