В модели **`pyannote/speaker-diarization-community-1`** (как и во всем фреймворке `pyannote.audio` версии 4.0) «пресеты» не реализованы в виде готовых текстовых названий вроде `preset="podcast"`. Вместо этого управление поведением пайплайна под разные типы роликов и аудиозаписей происходит через **динамические параметры инициализации** (`num_speakers`, `min_speakers`, `max_speakers`), а также тонкую настройку гиперпараметров в `config.yaml` (чувствительность VAD и кластеризации). [1](https://github.com/pyannote/pyannote-audio), [2](https://www.pyannote.ai/changelog/community-1-now-available), [3](https://huggingface.co/pyannote/speaker-diarization-community-1), [4](https://huggingface.co/pyannote-community/speaker-diarization-community-1/blob/main/config.yaml)

Ниже приведены готовые конфигурационные наборы параметров для кода на Python под основные типы входных роликов. [1](https://github.com/pyannote/hf-speaker-diarization-3.1), [2](https://habr.com/ru/articles/1062068/)

# Пресет: Подкаст или Интервью (2-3 спикера, студийный/чистый звук)

- **Особенности:** Голоса четкие, наложений (перебиваний) мало, количество участников известно заранее.
- **Цель:** Исключить ложные появления «лишних» спикеров из-за эха или шума.
```python
# Задаем строгое количество спикеров
diarization = pipeline("podcast.wav", num_speakers=2) 
```
# Пресет: Созвон в Zoom / Google Meet (Групповой митинг, 3–8 спикеров)

- **Особенности:** Спикеры часто перебивают друг друга, кто-то шуршит микрофоном, точное число участников плавает.
- **Цель:** Дать модели свободу в определении спикеров, но ограничить рамками, чтобы кашель или фоновый шум не создали SPEAKER_09. [1](https://habr.com/ru/articles/1062068/)
```python
# Ограничиваем разумный диапазон участников
diarization = pipeline("zoom_meeting.wav", min_speakers=2, max_speakers=6)
```
# Пресет: Лекция, Видеоурок или Вебинар (1 главный спикер + редкие вопросы)

- **Особенности:** 95% времени говорит один человек. Попытка запустить в режиме `auto` часто приводит к тому, что модель дробит голос лектора на разные кластеры (например, если он отошел от микрофона или изменил интонацию).
- **Цель:** Удержать фокус на главном спикере.
```python
# Если вопросов из зала не планируется вообще:
diarization = pipeline("lecture.wav", num_speakers=1)

# Если вопросы возможны, но спикеров точно мало:
diarization = pipeline("webinar.wav", min_speakers=1, max_speakers=3)
```
# Пресет: Шумное видео / Уличный репортаж / Стрим

- **Особенности:** Много посторонних звуков (машины, музыка, толпа). Стандартные настройки начнут принимать шумы за человеческую речь.
- **Решение:** Здесь параметров метода `pipeline()` недостаточно — нужно временно переопределить внутренние пороги чувствительности обнаружения речи (**VAD**) и кластеризации (**VBx**). [1](https://huggingface.co/pyannote-community/speaker-diarization-community-1/blob/main/config.yaml)
```python
# Загружаем пайплайн
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")

# Корректируем внутренние пороги перед обработкой:
# Повышаем порог threshold, чтобы отсечь слабые фоновые шумы
pipeline.parameters["clustering"]["threshold"] = 0.65  # По умолчанию обычно ~0.60

# Запускаем обработку
diarization = pipeline("street_video.wav", min_speakers=1, max_speakers=4)
```
# 💡 Важная фишка Community-1 для всех типов роликов

Модель `Community-1` возвращает новый тип разметки — **`exclusive_speaker_diarization`**. Если вы используете диаризацию в связке с Whisper для создания субтитров или текстовой расшифровки, используйте этот вывод, так как он принудительно убирает перекрытия голосов и идеально ложится на таймкоды фраз. [1](https://huggingface.co/pyannote-community/speaker-diarization-community-1/blob/main/README.md), [2](https://www.pyannote.ai/changelog/community-1-now-available)
```python
output = pipeline("video.wav")

# Стандартная разметка (с наложениями)
standard_diar = output.speaker_diarization 

# Эксклюзивная разметка (без наложений, идеальна для текста/STT)
exclusive_diar = output.exclusive_speaker_diarization 
```
