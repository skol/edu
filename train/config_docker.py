"""Конфигурация обучения в Docker (Yandex Cloud).

Активируется переменной окружения TRAIN_CONFIG=train.config_docker
(выставляется в docker-compose.yml и Dockerfile).

Пути рассчитаны на WORKDIR /app:
  - датасет монтируется в /app/data (read-only);
  - результаты обучения пишутся в /app/train/artifacts;
  - метрики — в /app/train/results.
Параметры заданы явно для полного (прод) прогона: все недели 0..24,
валидация 25, epochs=10, k=10, группы пользователей до 1000 человек.
"""

import os

from .config import (
    GroupSettings,
    MODEL_BY_GROUP as _MODEL_BY_GROUP,
    MODEL_NAMES as _MODEL_NAMES,
    ModelSettings,
    PathSettings,
    TrainSettings,
)

TRAIN = TrainSettings(
    data_dir="/app/data/raw/VK-LSVD/subsamples/up0.001_ip0.001",
    start_week=0,
    end_week=24,          # включительно; валидация на неделе 25
    epochs=10,
    lr=1e-3,
    k=10,
    min_history=2,
    max_history=32,
    min_timespent_pos=5,
    seed=42,
)

GROUPS = GroupSettings(
    cold_max=30,
    warm_max=70,
    group_limit=1000,     # продакшн: группы до 1000 юзеров; для быстрого теста уменьшить
    fallback_min=25,
    fallback_max=35,
)

MODELS = ModelSettings()

PATHS = PathSettings(
    artifacts_dir="/app/train/artifacts",
    results_dir="/app/train/results",
)

MODEL_NAMES = _MODEL_NAMES

ARTIFACTS = {
    name: os.path.join(PATHS.artifacts_dir, f"{name}.pt") for name in MODEL_NAMES
}

RESULTS_FILE = os.path.join(PATHS.results_dir, "metrics_by_group.csv")

MODEL_BY_GROUP = _MODEL_BY_GROUP