"""Настройки рекомендательной системы MostPopular для VK-LSVD.

Все общие настройки (тренировочные недели, валидационная неделя, топ-k, порог
timespent и т.д.) наследуются из CommonConfig, чтобы одинаковые настройки
одинаково назывались и работали во всех моделях проекта.

У most_popular нет градиентного обучения, поэтому epochs/lr/min_history/
max_history не используются.
"""

from dataclasses import dataclass

from common.config import CommonConfig


@dataclass
class MostPopularConfig(CommonConfig):
    pass