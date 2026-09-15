"""Настройки финального пайплайна VK-LSVD.

Дефолты настроены «быстрыми»: лимит группы пользователей для метрик = 100
(чтобы тестовые прогоны не занимали много времени). Для продакшн-прогона все
значения задаются явно.
"""

import os
from dataclasses import dataclass, field


@dataclass
class TrainSettings:
    """Общие настройки обучения (одинаковые для всех моделей)."""

    data_dir: str = "data/raw/VK-LSVD/subsamples/up0.001_ip0.001"
    start_week: int = 0
    end_week: int = 24          # включительно; валидация всегда на неделе end+1
    epochs: int = 10
    lr: float = 1e-3
    k: int = 10                 # количество роликов в прогнозе
    min_history: int = 2
    max_history: int = 32
    min_timespent_pos: int = 5
    seed: int = 42

    @property
    def val_week(self) -> int:
        """Валидационная/тестовая неделя (25 при end_week=24)."""
        return self.end_week + 1


@dataclass
class GroupSettings:
    """Пороги групп пользователей и лимиты выборки."""

    cold_max: int = 30          # холодный: размер истории <= cold_max
    warm_max: int = 70          # тёплый: cold_max < size <= warm_max; горячий: > warm_max
    group_limit: int = 100      # макс. размер группы для метрик (быстрые дефолты)
    fallback_min: int = 25      # зона fallback'а mostPopular: [fallback_min, fallback_max]
    fallback_max: int = 35


@dataclass
class ModelSettings:
    """Модель-специфичные параметры обучения."""

    fpmc: dict = field(default_factory=lambda: {"dim": 64, "batch_size": 1024})
    light_gcn: dict = field(default_factory=lambda: {
        "dim": 64, "n_layers": 3, "batch_size": 1024,
    })
    sasrec: dict = field(default_factory=lambda: {
        "history_weeks": 2, "batch_size": 64, "num_workers": 0,
        "emb_width": 64, "num_heads": 4, "num_layers": 2, "dropout": 0.1,
    })
    most_popular: dict = field(default_factory=dict)


@dataclass
class PathSettings:
    """Пути к артефактам и результатам."""

    artifacts_dir: str = "train/artifacts"
    results_dir: str = "train/results"


# ---- единая точка конфигурации ----
TRAIN = TrainSettings()
GROUPS = GroupSettings()
MODELS = ModelSettings()
PATHS = PathSettings()

MODEL_NAMES = ["most_popular", "fpmc", "light_gcn", "sasrec"]

ARTIFACTS = {
    "most_popular": os.path.join(PATHS.artifacts_dir, "most_popular.pt"),
    "fpmc": os.path.join(PATHS.artifacts_dir, "fpmc.pt"),
    "light_gcn": os.path.join(PATHS.artifacts_dir, "light_gcn.pt"),
    "sasrec": os.path.join(PATHS.artifacts_dir, "sasrec.pt"),
}

RESULTS_FILE = os.path.join(PATHS.results_dir, "metrics_by_group.csv")

# Соответствие категория -> модель (хардкод до получения точных метрик;
# итоговое соответствие оператор укажет вручную).
MODEL_BY_GROUP = {
    "cold": "fpmc",
    "warm": "sasrec",
    "hot": "light_gcn",
}


def load_runtime_config():
    """Загружает модуль конфига из переменной окружения TRAIN_CONFIG.

    По умолчанию — train.config (локальный запуск); для docker-запуска на
    GPU-инфраструктуре — train.config_docker (устанавливается в Dockerfile и
    docker-compose.yml). Модуль должен экспортировать TRAIN, GROUPS, MODELS,
    PATHS, MODEL_NAMES, ARTIFACTS, RESULTS_FILE, MODEL_BY_GROUP.
    """
    import importlib
    import os

    name = os.environ.get("TRAIN_CONFIG", "train.config")
    return importlib.import_module(name)