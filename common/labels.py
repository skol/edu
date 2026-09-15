"""Единая разметка действий VK-LSVD на позитив/негатив для всех моделей.

Позитив: timespent >= min_timespent_pos ИЛИ одна из реакций
         like / share / bookmark / click_on_author / open_comments.
Негатив: НЕ позитив И (timespent < min_timespent_pos ИЛИ dislike).

Правило «позитив побеждает»: при наличии любого позитивного сигнала действие
считается позитивным, даже если стоит dislike.

null-семантика: пустые флаги реакций трактуются как False, пустой timespent — как 0.
"""

import polars as pl

# Реакции, делающие действие позитивным.
POS_REACTIONS = ["like", "share", "bookmark", "click_on_author", "open_comments"]
# Реакция, делающая действие негативным (при отсутствии позитивных сигналов).
NEG_REACTION = "dislike"

_ALL_COLS = POS_REACTIONS + [NEG_REACTION]


def assign_labels(df: pl.DataFrame, min_timespent_pos: int) -> pl.DataFrame:
    """Добавляет колонку label: 1 (позитив), 0 (негатив). Неоднозначные строки удаляются.

    Args:
        df: DataFrame с колонками timespent и флагами реакций
            (like, dislike, share, bookmark, click_on_author, open_comments).
        min_timespent_pos: порог времени просмотра (>=) для позитивного действия.

    Returns:
        DataFrame с колонкой label (Int8), без неоднозначных строк.
    """
    df = df.with_columns(
        [pl.col(c).fill_null(False).alias(c) for c in _ALL_COLS]
    ).with_columns(pl.col("timespent").fill_null(0))

    has_pos_reaction = None
    for c in POS_REACTIONS:
        col = df[c]
        has_pos_reaction = col if has_pos_reaction is None else has_pos_reaction | col

    positive = (df["timespent"] >= min_timespent_pos) | has_pos_reaction
    negative = (~positive) & ((df["timespent"] < min_timespent_pos) | df[NEG_REACTION])

    return df.with_columns(
        pl.when(positive)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.when(negative).then(pl.lit(0, dtype=pl.Int8)).otherwise(None))
        .alias("label")
    ).filter(pl.col("label").is_not_null())