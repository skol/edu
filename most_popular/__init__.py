"""MostPopular — рекомендательная система для VK-LSVD.

Рекомендует следующий видео ролик исходя из самых популярных на платформе,
которые пользователь ещё не видел.
"""

from .config import MostPopularConfig
from .data import DataBundle, load_and_build
from .evaluate import evaluate
from .model import MostPopular

__all__ = [
    "MostPopularConfig",
    "DataBundle",
    "load_and_build",
    "MostPopular",
    "evaluate",
]