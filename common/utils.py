"""Общие утилиты: выбор устройства и воспроизводимость."""

import random

import numpy as np
import torch


def get_device() -> str:
    """Возвращает доступное устройство: CUDA, если есть, иначе CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 42):
    """Фиксирует seed для воспроизводимости во всех библиотеках."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Для воспроизводимости.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False