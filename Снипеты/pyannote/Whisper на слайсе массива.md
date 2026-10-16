Библиотека **`faster_whisper`** принимает на вход numpy-массивы напрямую, но со строгим ограничением: аудио должно быть **одномерным (1D) массивом** типа **`float32`** с частотой дискретизации **16 000 Гц**.

Если вы делаете слайс (срез) аудио-массива, вам нужно убедиться, что срез сохраняет правильную форму (shape) и тип данных.

```python
import numpy as np
from faster_whisper import WhisperModel

# 1. Загружаем модель
model = WhisperModel("tiny", device="cpu", compute_type="float32")

# 2. Допустим, у вас есть исходный массив аудио (16kHz, float32)
# shape: (количество_сэмплов,)
audio_data = np.random.randn(160000).astype(np.float32)  # Пример: 10 секунд тишины

# 3. Делаем слайс (например, с 2-й по 5-ю секунду)
start_sample = 2 * 16000
end_sample = 5 * 16000
audio_slice = audio_data[start_sample:end_sample]

# 4. Передаем слайс в модель
segments, info = model.transcribe(audio_slice, beam_size=5)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

# На что важно обратить внимание (Чек-лист):

- **Размерность:** Убедитесь, что массив после слайса одномерный (`audio_slice.ndim == 1`). Если вы взяли срез из многоканального аудио (например, стерео матрица), выберите один канал: `stereo_audio[start:end, 0]`.
- **Тип данных:** Обязательно `np.float32`. Значения должны быть нормализованы (обычно в диапазоне от `-1.0` до `1.0`). Если тип изменился, примените `.astype(np.float32)`.
- **Непрерывность в памяти (C-contiguous):** Иногда после сложных срезов numpy создает _view_ (представление), которое не лежит в памяти подряд. `faster_whisper` (а точнее `ctranslate2` под капотом) может выдать ошибку. Чтобы этого избежать, безопасно делать копию среза:
```python
audio_slice = np.ascontiguousarray(audio_data[start:end], dtype=np.float32)
```
