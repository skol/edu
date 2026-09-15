# Обучение моделей VK-LSVD в Docker на Yandex Cloud

Прототип запуска обучения четырёх моделей (most_popular, fpmc, light_gcn,
sasrec) в контейнере на GPU-виртуальной машине Yandex Cloud.

## Что в комплекте

| Файл | Назначение |
|---|---|
| `Dockerfile` | Образ: Python 3.11 + PyTorch CUDA + зависимости проекта. **Датасет в образ не входит.** |
| `.dockerignore` | Исключает `data/`, `.git`, `*.pt` и пр. из контекста сборки. |
| `docker-compose.yml` | Сервис `train`: монтирует датасет `./data -> /app/data` (read-only), результаты `./artifacts -> /app/train/artifacts` и `./results -> /app/train/results`, включает NVIDIA GPU. |
| `requirements.txt` | numpy / polars / duckdb (torch ставится отдельно с CUDA). |
| `train/config_docker.py` | Конфиг docker-запуска (`TRAIN_CONFIG=train.config_docker`): полный период недель `0..24` → валидация `25`, `epochs=10`, `k=10`, группы до 1000 юзеров, пути `/app/...`. |
| `train/run_training.py` | Последовательное обучение моделей с сохранением артефактов. |
| `train/evaluate_groups.py` | Метрики по группам cold/warm/hot → `train/results/metrics_by_group.csv`. |

Выбор конфига: локально используется `train.config`, в Docker —
`train.config_docker` (переменная окружения `TRAIN_CONFIG`).

---

## Шаг 1. Подготовка (локально, опционально)

```bash
git clone -b final <URL_репозитория> vk-lsvd
cd vk-lsvd
```

Проверить, что код работает (маленькое окно недель, CPU):

```bash
python -m train.run_training --models most_popular   # быстро, без обучения остальных
```

---

## Шаг 2. Создание GPU-виртуальной машины в Yandex Cloud

Вариант A — консоль: Compute Cloud → Создать ВМ → GPU-платформа
(например `gpu-standard-v3`, 1×V100/A100), Ubuntu 22.04 LTS, диск SSD
от 300 ГБ, публичный IP (для доступа), SSH-ключ. **Обязательно отметьте
«Прерываемая ВМ»** — это дешевле на ~30–50%, а при остановке артефакты
сохраняются на диске.

Вариант B — CLI (`yc`):

```bash
yc compute instance create \
  --name vk-lsvd-train \
  --platform gpu-standard-v3 \
  --cores 4 --memory 48GB --gpus 1 \
  --create-boot-disk size=300GB,image-folder-id=standard-images,image-family=ubuntu-2204-lts \
  --ssh-key ~/.ssh/id_ed25519.pub \
  --preemptible \
  --zone ru-central1-a
```

Получить адрес:

```bash
yc compute instance get vk-lsvd-train --format json | grep -i address
```

## Шаг 3. Установка Docker и NVIDIA runtime на ВМ

```bash
ssh ubuntu@<VM_IP>

# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# выйти и зайти заново (или newgrp docker)

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверка GPU в docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Шаг 4. Клонирование проекта и загрузка датасета

```bash
cd ~
git clone -b final <URL_репозитория> vk-lsvd
cd vk-lsvd

# Структура данных на ВМ (монтируется в контейнер как /app/data):
#   vk-lsvd/data/raw/VK-LSVD/subsamples/up0.001_ip0.001/{train,validation}/week_*.parquet
mkdir -p data
```

Загрузка датасета (любой способ):

```bash
# вариант 1: scp с локальной машины (из каталога, где лежит data/)
scp -r data ubuntu@<VM_IP>:~/vk-lsvd/

# вариант 2: из Object Storage Yandex
#   yc storage s3api create-bucket ...
#   aws s3 sync s3://bucket/vk-lsvd-data ./data --endpoint-url https://storage.yandexcloud.net
```

## Шаг 5. Запуск обучения

```bash
cd ~/vk-lsvd
docker compose up --build      # соберёт образ и запустит обучение
```

Что происходит:

1. Контейнер обучает модели по очереди: most_popular → fpmc → light_gcn → sasrec.
2. Артефакты каждой модели сохраняются сразу после её обучения в
   `./artifacts/` на хосте (`/app/train/artifacts` в контейнере).
3. После обучения прогоните метрики по группам:

```bash
docker compose run --rm train python -m train.evaluate_groups
```

Результат — `./results/metrics_by_group.csv`.

Для перезапуска одной модели (если артефакт уже есть, скрипт спросит):

```bash
docker compose run --rm train python -m train.run_training --models fpmc --force-overwrite on
```

## Шаг 6. Перенос результатов в git (данные должны быть доступны всегда)

**До остановки ВМ** скопируйте результаты на свою машину:

```bash
# с локальной машины
scp -r ubuntu@<VM_IP>:~/vk-lsvd/artifacts ./
scp -r ubuntu@<VM_IP>:~/vk-lsvd/results ./
```

Затем закоммитьте в репозиторий:

```bash
# *.pt/*.pth в .gitignore — поэтому используем force-add
git add -f artifacts/*.pt results/*.csv
git commit -m "artifacts: trained models + metrics by groups"
git push
```

Альтернатива: Git LFS (`git lfs track "artifacts/*.pt"`) для больших файлов.

Рекомендация: сам датасет храните в Object Storage (bucket) — он дешёвый,
доступен с любой ВМ и не потеряется при удалении инстанса.

---

## Шаг 7. ЧЕК-ЛИСТ: ничего не забыть, все платные сервисы выключены

Выполняйте строго по порядку:

1. **Результаты скопированы локально / в git?** (`artifacts/`, `results/`)
   → Шаг 6 выше. Без этого не останавливайте ВМ.
2. **Остановить контейнеры:** `docker compose down`
3. **Остановить ВМ (перестаёт платить за GPU/CPU, платит только диск):**
   ```bash
   yc compute instance stop vk-lsvd-train
   ```
   Или удалить совсем (после копирования данных):
   ```bash
   yc compute instance delete vk-lsvd-train
   ```
4. **Проверить, что ВМ остановлена:**
   ```bash
   yc compute instance list    # статус STOPPED или запись отсутствует
   ```
5. **Освободить статические IP (иначе платите за них):**
   ```bash
   yc vpc address list
   yc vpc address delete <address-id>
   ```
6. **Удалить ненужные снапшоты дисков:**
   ```bash
   yc compute snapshot list
   yc compute snapshot delete <snapshot-id>
   ```
7. **Object Storage (если использовался):** хранение почти бесплатно, но если
   бакет больше не нужен — удалите:
   ```bash
   yc storage bucket list
   yc storage bucket delete <bucket-name>
   ```
8. **Managed Service for Databases / Managed Kubernetes / Балансировщики**
   (если создавались для этой задачи) — удалить через консоль или `yc delete`.
9. **Финальная проверка в консоли:** Биллинг → «Сводка/Детализация» за сегодня
   должна показывать ~0 ₽ (кроме возможного хранения бакета/дисков).
10. **Страховка на будущее:** настройте бюджет-алерт
    (Billing → Бюджеты, лимит, например, 1000 ₽) — уведомление, если расходы
    превысят порог. Повторный запуск — только когда действительно нужен.

---

## Советы по стоимости («разумные деньги»)

- **Сначала CPU-smoke:** прогнать обучение на малом окне недель на дешёвой
  CPU-ВМ (`--start-week 20 --end-week 22 --epochs 1`) или локально, чтобы
  убедиться, что пайплайн работает, прежде чем тратить GPU-часы.
- **Прерываемая ВМ** (preemptible): дешевле на ~30–50%; артефакты пишутся
  после каждой модели, поэтому прерывание не теряет прогресс — просто
  перезапустите `docker compose up` и ответьте «n» (использовать существующее).
- **Время на GPU:** полное обучение 4 моделей на субвыборке `up0.001_ip0.001`
  занимает порядка нескольких часов на одной V100/A100; точные тарифы
  смотрите в консоли Yandex Cloud перед запуском.
- **Диск и данные:** после переноса артефактов в git ВМ можно удалять —
  при необходимости датасет восстанавливается из Object Storage.