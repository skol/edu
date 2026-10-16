Чтобы получить список спикеров, временные интервалы их реплик и соответствующие им векторные эмбеддинги (вектора голоса), в современных версиях библиотеки `pyannote.audio` (начиная с версии 3.0+) достаточно передать аргумент **`return_embeddings=True`** при вызове пайплайна. [1](https://github.com/pyannote/pyannote-audio/issues/1750)

```python
import torch
from pyannote.audio import Pipeline

# 1. Инициализируем пайплайн (требуется токен Hugging Face)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", 
    use_auth_token="YOUR_HUGGINGFACE_TOKEN"
)

# Отправляем на GPU, если доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

# 2. Запускаем диаризацию с флагом возврата эмбеддингов
diarization, embeddings = pipeline("audio.wav", return_embeddings=True)

# 3. Получаем временные интервалы и спикеров
print("--- Хронология реплик ---")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Спикер {speaker}: {turn.start:.2f}s -> {turn.end:.2f}s")

# 4. Получаем уникальные эмбеддинги для каждого спикера
print("\n--- Эмбеддинги спикеров ---")
for i, speaker in enumerate(diarization.labels()):
    # embeddings — это матрица, где индекс строки соответствует индексу в diarization.labels()
    speaker_embedding = embeddings[i]
    
    print(f"Спикер: {speaker}")
    print(f"Размерность эмбеддинга: {speaker_embedding.shape}") # Обычно (256,) для версий 3.x
    print(f"Пример вектора (первые 5 значений): {speaker_embedding[:5]}\n")
```
# Как это устроено внутри:

- **`diarization`** (объект `Annotation`) — содержит всю временную сетку аудиозаписи: кто, в какой момент и сколько говорил. [1](https://vexascribe.com/pyannote-audio)

- **`embeddings`** (массив `numpy.ndarray` формата `N x D`) — содержит усредненные **центроидные эмбеддинги** для каждого из `N` найденных спикеров. Порядок строк в этом массиве строго соответствует списку уникальных ID спикеров, возвращаемому методом `diarization.labels()`. [1](https://soniqo.audio/ru/guides/diarize), [2](https://github.com/pyannote/pyannote-audio/issues/1750)

- **Размерность:** В моделях `pyannote 3.0` и `3.1` по умолчанию используется модель `wespeaker-voxceleb-resnet34-LM`, поэтому эмбеддинги имеют **размерность 256**. В старых версиях (2.1) использовался ECAPA-TDNN с размерностью 512. [1](https://github.com/pyannote/pyannote-audio/discussions/1512), [2](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/pyannote-speaker-diarization-production), [3](https://soniqo.audio/ru/guides/diarize)

Если вам потребуется **сопоставить спикеров между разными аудиофайлами**, вы можете сохранить эти векторы (`speaker_embedding`) и сравнивать их между собой с помощью косинусного расстояния (`scipy.spatial.distance.cosine`). [1](https://github.com/pyannote/pyannote-audio/discussions/1667), [2](https://medium.com/@sapkotabinit2002/speaker-identification-and-clustering-using-pyannote-dbscan-and-cosine-similarity-dfa08b5b2a24)
