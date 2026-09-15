"""Модель MostPopular: рекомендация самых популярных айтемов, которые юзер не видел."""

import numpy as np


class MostPopular:
    """Ранжирует все айтемы по популярности и рекомендует не увиденные юзером.

    Args:
        popularity: np.ndarray длины n_items — число положительных примеров на айтем.
        item_ranking: np.ndarray длины n_items — индексы айтемов по убыванию популярности.
    """

    def __init__(self, popularity: np.ndarray, item_ranking: np.ndarray):
        self.popularity = popularity
        self.item_ranking = item_ranking

    def recommend(self, seen, top_n: int) -> list[int]:
        """Возвращает топ-N айтемов (внутренние индексы), которые юзер ещё не видел."""
        out: list[int] = []
        for it in self.item_ranking:
            if it not in seen:
                out.append(int(it))
                if len(out) >= top_n:
                    break
        return out

    def rank_positions(self, seen, gt: set[int]) -> np.ndarray:
        """1-based ранги верных айтемов gt в глобальном ранжировании без учёта seen.

        Возвращает массив той же длины, что и gt. Ранг 0 означает, что айтем
        не попал в рекомендации.
        """
        rank = 0
        positions = np.zeros(len(gt), dtype=int)
        remaining = set(gt) - set(seen)
        for it in self.item_ranking:
            if it in seen:
                continue
            rank += 1
            if it in remaining:
                idx_in_gt = [i for i, g in enumerate(gt) if g == it]
                positions[idx_in_gt] = rank
                remaining.difference_update({it})
                if not remaining:
                    break
        return positions