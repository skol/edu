# Образ для обучения моделей VK-LSVD на GPU-инфраструктуре (Yandex Cloud).
# Датасет в образ НЕ включается: каталог data/ исключён .dockerignore и
# монтируется при старте через docker-compose volumes.

FROM python:3.11-slim

WORKDIR /app

# Минимальный набор системных пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch с CUDA. Индекс можно поменять под конкретную GPU/драйвер
# (cu121/cu124/cu126 — см. https://pytorch.org/get-started/locally/).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch

# Остальные зависимости
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Код проекта (данные исключены .dockerignore)
COPY . .

# Конфиг для docker-запуска (переопределяет пути под /app и параметры обучения)
ENV TRAIN_CONFIG=train.config_docker
ENV PYTHONUNBUFFERED=1

# Обучение всех моделей; артефакты пишутся в смонтированные каталоги
# /app/train/artifacts и /app/train/results.
CMD ["python", "-m", "train.run_training"]