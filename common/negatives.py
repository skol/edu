"""Единый negative sampling для всех моделей VK-LSVD.

Семантика единая для fpmc, light_gcn и sasrec:
  - приоритет — случайный из ЯВНЫХ негативов пользователя (skip/dislike из его
    истории, полученных через common.labels.assign_labels);
  - если явных негативов нет (или все исключены) — случайный равномерный айтем.
"""

import random


def sample_negative(explicit_negatives, n_items: int, exclude=(), rng=None) -> int:
    """Возвращает индекс негативного айтема.

    Args:
        explicit_negatives: последовательность индексов явно негативных айтемов
            пользователя (может быть пустой).
        n_items: общее число айтемов словаря.
        exclude: индексы, которые нельзя возвращать (например, правильный ответ
            или padding-индекс 0).
        rng: optional random.Random; если None — используется глобальный random.

    Returns:
        int — индекс негативного айтема.
    """
    excluded = set(exclude)
    candidates = [i for i in explicit_negatives if i not in excluded]
    if candidates:
        return rng.choice(candidates) if rng is not None else random.choice(candidates)

    while True:
        idx = rng.randrange(n_items) if rng is not None else random.randrange(n_items)
        if idx not in excluded:
            return idx