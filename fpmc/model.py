"""FPMC (Factorized Personalized Markov Chain) — PyTorch-реализация.

Идея: скор предсказания следующего айтема i для пользователя u
с учётом последнего просмотренного айтема l раскладывается на две части:

    score(u, l, i) = <U_u, V_i>                 (MF-часть: вкус пользователя)
                   + sum_d BU_{u,d} * T_{l,d} * T_{i,d}   (MC-часть: персонализированный переход)

MC-часть — это CP-декомпозиция тензора переходов "пользователь x последний x следующий".
Вместо явного тензора (огромного) храним три низкоранговых фактора:
BU (пользователь), T (айтем, общий для "последнего" и "следующего").
"""

import torch
import torch.nn as nn


class FPMC(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        # --- MF-часть (вкус пользователя) ---
        self.user_pf = nn.Embedding(n_users, dim)   # U
        self.item_pf = nn.Embedding(n_items, dim)   # V

        # --- MC-часть (переходы) ---
        self.user_mc = nn.Embedding(n_users, dim)   # BU
        self.item_mc = nn.Embedding(n_items, dim)   # T (и "последний", и "следующий")

        self._init_weights()

    def _init_weights(self):
        for emb in (self.user_pf, self.item_pf, self.user_mc, self.item_mc):
            nn.init.normal_(emb.weight, std=0.1)

    def score(self, user_ids, last_ids, item_ids):
        """Скор для троек (u, l, i). Возвращает тензор [batch]. """
        # MF-часть
        mf = (self.user_pf(user_ids) * self.item_pf(item_ids)).sum(-1)

        # MC-часть: sum_d BU[u,d] * T[l,d] * T[i,d]
        mc = (self.user_mc(user_ids) * self.item_mc(last_ids) * self.item_mc(item_ids)).sum(-1)

        return mf + mc

    def forward(self, user_ids, last_ids, pos_item_ids, neg_item_ids):
        """BPR-целевые логиты для обучающей пары (pos > neg).

        Returns:
            loss, pos_score, neg_score
        """
        pos_score = self.score(user_ids, last_ids, pos_item_ids)
        neg_score = self.score(user_ids, last_ids, neg_item_ids)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-9).mean()
        return loss, pos_score, neg_score