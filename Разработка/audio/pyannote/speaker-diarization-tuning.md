**Тонкая настройка (Fine-tuning)** модели `pyannote/speaker-diarization-community-1` (как и всей линейки pyannote 3.x/4.0) технически заключается в дообучении её внутреннего компонента — **модели сегментации (Segmentation)**. Остальные части пайплайна (извлечение эмбеддингов спикеров и кластеризация) остаются предобученными, а их поведение корректируется с помощью **параметров инференса**. [1](https://github.com/pyannote/pyannote-audio/issues/1748), [2](https://huggingface.co/paris-iea/speaker-diarization/resolve/main/technical_report_2.1.pdf)

# Тонкая настройка (Fine-tuning) сегментации

Пайплайны `pyannote` не дообучаются целиком как единая сеть. Вместо этого вы дообучаете модель сегментации (которая отвечает за поиск пауз, наложений речи и предварительных границ) на своих данных. [1](https://colab.research.google.com/github/pyannote/pyannote-audio/blob/develop/tutorials/training_a_model.ipynb), [2](https://github.com/pyannote/pyannote-audio/issues/1748)

## Шаг 1: Подготовка датасета

Вам нужно зарегистрировать свой датасет (в формате AMI/согласно гайдам библиотеки) в конфигурационном файле `database.yaml`:
```yaml
Databases:
  MyDataset: /path/to/my/audio/folder/{uri}.wav

Protocols:
  MyDataset:
    SpeakerDiarization:
      CustomProtocol:
        train:
          annotation: /path/to/train.rttm
          annotated: /path/to/train.uem
        development:
          annotation: /path/to/dev.rttm
          annotated: /path/to/dev.uem
```
## Шаг 2: Код дообучения (через PyTorch Lightning)

Используйте класс `SpeakerDiarization` из `pyannote.audio.tasks`: [1](https://github.com/pyannote/pyannote-audio/discussions/1971)
```python
import pytorch_lightning as pl
from pyannote.database import FileFinder, get_protocol
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization

# 1. Загружаем протокол данных
pl.seed_everything(42)
protocol = get_protocol("MyDataset.SpeakerDiarization.CustomProtocol", preprocessors={"audio": FileFinder()})

# 2. Инициализируем задачу
seg_task = SpeakerDiarization(
    protocol, 
    duration=10.0,              # Длина чанка аудио (рекомендуется 10.0)
    max_speakers_per_chunk=3,   # Макс. спикеров в одном чанке
    max_speakers_per_frame=2,   # Макс. одновременных спикеров (наложения)
    batch_size=32,
    num_workers=2
)

# 3. Загружаем базовую модель сегментации из community-1
# В community-1 используется модель сегментации (обычно pyannote/segmentation-3.0 или новая архитектура 4.0)
model = Model.from_pretrained("pyannote/segmentation-3.0", token="YOUR_HF_TOKEN")
model.task = seg_task

# 4. Запуск обучения
trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1)
trainer.fit(model)

# Сохраняем веса
model.save_pretrained("my_finetuned_segmentation")
```
## Шаг 3: Сборка кастомного пайплайна

После обучения вы подменяете модель сегментации внутри стандартного пайплайна `community-1`: [1](https://huggingface.co/diarizers-community/speaker-segmentation-fine-tuned-callhome-zho)
```python
from pyannote.audio import Pipeline, Model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем базовый пайплайн
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token="YOUR_HF_TOKEN")
pipeline.to(device)

# Загружаем вашу дообученную модель сегментации и подменяем её
finetuned_seg = Model.from_pretrained("my_finetuned_segmentation").to(device)
pipeline._segmentation.model = finetuned_seg
```
# Параметры при инференсе (Inference Parameters)

При вызове собранного пайплайна на целевом аудио вы можете жестко задавать ограничения, что критически меняет качество работы алгоритма кластеризации. [1](https://huggingface.co/pyannote/speaker-diarization-community-1), [2](https://huggingface.co/paris-iea/speaker-diarization/resolve/main/technical_report_2.1.pdf)
```python
# Базовый инференс
diarization = pipeline("audio.wav")
```
## Ограничение количества спикеров (Самый мощный инструмент тюнинга)

Если вы заранее знаете контекст записи, передайте эти параметры. Они отсекают ложные срабатывания кластеризации:

- **Точное количество:** Если известно точное число участников (например, интервью тет-а-тет или подкаст на 3 человек):
```python
diarization = pipeline("audio.wav", num_speakers=2)
````
- **Диапазон спикеров:** Если точное число неизвестно, но есть рамки (например, на созвоне точно не менее 2 и не более 5 человек): [1](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/pyannote-speaker-diarization-production), [2](https://github.com/pyannote/hf-speaker-diarization-3.1), [3](https://huggingface.co/pyannote/speaker-diarization/blob/main/README.md)
```python
diarization = pipeline("audio.wav", min_speakers=2, max_speakers=5)
```
## Оптимизация по скорости и памяти (Особенно для длинных аудио)

По умолчанию передача пути к файлу строкой заставляет модель повторно перерасчитывать метаданные. Для стабильности и утилизации GPU загружайте аудио заранее через `torchaudio`: [1](https://github.com/pyannote/pyannote-audio/issues/1557)
```python
import torchaudio

# Предобработка: загрузка тензора на нужный девайс
waveform, sample_rate = torchaudio.load("audio.wav")

# Передача словаря напрямую в пайплайн (ускоряет инференс в разы и снижает нагрузку на CPU)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```
## Извлечение эмбеддингов (Если нужен downstream-анализ)

Если помимо разметки по времени вам нужны векторы голосов спикеров (например, чтобы сравнивать их с базой данных известных клиентов/сотрудников), используйте флаг `return_embeddings`: [1](https://github.com/pyannote/pyannote-audio/issues/1750)
```python
diarization, embeddings = pipeline("audio.wav", return_embeddings=True)

for i, speaker in enumerate(diarization.labels()):
    # embeddings[i] — это усредненный вектор (embedding) данного спикера
    print(f"Спикер {speaker}, размер эмбеддинга: {embeddings[i].shape}")
```
## Тюнинг внутренних порогов пайплайна (Без переобучения)

Вы можете поменять порог чувствительности кластеризации голосов вручную, если модель слишком часто дробит одного и того же человека на разных спикеров (или наоборот — объединяет разных в одного). [1](https://huggingface.co/paris-iea/speaker-diarization/resolve/main/technical_report_2.1.pdf)

- `clustering.threshold`: порог агломеративной кластеризации. _Меньше значение — модель склонна создавать больше спикеров; выше значение — модель будет сильнее объединять похожие голоса._ [1](https://huggingface.co/paris-iea/speaker-diarization/resolve/main/technical_report_2.1.pdf)
```python
# Посмотреть текущие гиперпараметры инференса:
print(pipeline.parameters())

# Изменить порог кластеризации на лету перед инференсом:
new_params = {
    "clustering": {
        "threshold": 0.55  # подберите под ваш домен (по умолчанию обычно ~0.7)
    }
}
pipeline.instantiate(new_params)

# Запуск с новыми порогами
diarization = pipeline("audio.wav")
```
