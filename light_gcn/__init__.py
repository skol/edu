"""LightGCN (Light Graph Convolution Network) на PyTorch для VK-LSVD.

Модель предсказывает следующий ролик для пользователя через двудольный граф
"пользователь -- ролик": обучает эмбеддинги без нелинейных слоёв (только
нормализованная агрегация соседей) и ранжирует кандидатов по скалярному
произведению эмбеддингов юзера и ролика.
"""

from .data import DataBundle, DataConfig, LightGCNDataset, load_and_build
from .model import LightGCN
from .train import evaluate, train_model

__all__ = [
    "DataBundle",
    "DataConfig",
    "LightGCNDataset",
    "load_and_build",
    "LightGCN",
    "train_model",
    "evaluate",
]