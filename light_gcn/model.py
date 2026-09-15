"""LightGCN (Light Graph Convolution Network) — PyTorch-реализация.

Ссылка на статью: He et al., "LightGCN: Simplifying and Powering Graph
Convolution Network for Recommendation" (SIGIR 2020).

Ключевая идея: в отличие от классических GCN здесь убираются нелинейные
преобразования и фичи — остаётся только простая линейная агрегация соседей:

    E^{(k+1)} = A~ E^{(k)},  где A~ = D^{-1/2} A D^{-1/2}

Итоговый эмбеддинг — среднее по всем слоям (включая нулевой):

    e = 1/(K+1) * sum_k e^{(k)}

Это упрощение делает модель лёгкой, устойчивой к переобучению и эффективной
для рекомендаций на больших двудольных графах "пользователь -- ролик".
"""

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64,
                 n_layers: int = 3, norm_adj: torch.Tensor | None = None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        # разреженная нормализованная матрица смежности [N, N]
        self.norm_adj = norm_adj

        # базовые эмбеддинги без дополнительных параметров на слой
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self._init_weights()

    def _init_weights(self):
        for emb in (self.user_emb, self.item_emb):
            nn.init.normal_(emb.weight, std=0.1)

    def forward(self):
        """Полный проход по графу.

        Returns:
            user_final, item_final — агрегированные по слоям эмбеддинги [n_users, dim],
            [n_items, dim].
        """
        # объединяем эмбеддинги в одну матрицу [N, dim]
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        embs = [all_emb]
        for _ in range(self.n_layers):
            # простое распространение сигнала от соседей
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)

        # усредняем по всем слоям (включая нулевой)
        out = torch.stack(embs).mean(0)
        return out[:self.n_users], out[self.n_users:]

    def score(self, user_emb, item_emb, user_ids, item_ids):
        """Скор предсказания <эмбеддинг юзера, эмбеддинг ролика> для пар.

        Args:
            user_emb: итоговые эмбеддинги всех юзеров.
            item_emb: итоговые эмбеддинги всех роликов.
            user_ids: индексы юзеров [batch].
            item_ids: индексы роликов [batch].

        Returns:
            Тензор скоров [batch].
        """
        return (user_emb[user_ids] * item_emb[item_ids]).sum(-1)

    def bpr_loss(self, user_emb, item_emb, user_ids, pos_ids, neg_ids):
        """BPR-потеря: верный ролик должен иметь скор выше случайного негатива.

        Returns:
            loss, pos_score, neg_score
        """
        pos_score = self.score(user_emb, item_emb, user_ids, pos_ids)
        neg_score = self.score(user_emb, item_emb, user_ids, neg_ids)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-9).mean()
        return loss, pos_score, neg_score