# Коллаторы для обучения GigaOmni

Эта директория содержит **модальные коллаторы**, которые используются в `llmfoundry/data/omni_data.py` для сборки батча из примеров датасета `StreamingOmniDataset`.

**Ключевая идея:** каждый модальный коллатор отвечает **только за свою модальность**, а общий `OmniCollator` собирает итоговый батч в формате, который ожидает модель.

На текущий момент реализованы:

| Класс | Файл | Модальность (`data_modality`) | Описание |
|---|---|---|---|
| `TextCollator` | `text_collator.py` | `DataModality.TEXT` (`"text"`) | Делает паддинг `input_id` и `labels`, учитывает выравнивание длины для корректной работы TP / SP через  (`divider`). `split_batch` обрезает каждый микробатч до его самой длинной строки (кратно `divider`), чтобы короткий сэмпл не тащил в модель паддинг всего device batch. Паддинг — это `pad_token_id` с меткой `-100`; микробатчи с левым паддингом и целиком из паддинга не обрезаются; `TextFlashAttnCollator` обрезает `position_ids` до той же длины. `attention_mask` по умолчанию (`return_attention_mask: null`) отдаётся только при левом паддинге: при паддинге в конце causal attention не видит хвостовые паддинги, а без маски LM не уходит в unpad (синхронизация с хостом на каждом слое внимания); явные `true` / `false` переопределяют это, оба текстовых коллатора ведут себя одинаково |
| `ImageCollator` | `image_collator.py` | `DataModality.IMAGE` (`"image"`) | Cклеивает `pixel_values` по crop’ам и сохраняет `n_crops_per_sample` |
| `VideoCollator` | `video_collator.py` | `DataModality.VIDEO` (`"video"`) | Аналогично `ImageCollator`, но для `video_only` модальности (кадры); дополнительно несёт `num_crops_per_frame` (crop'ов на кадр у всех видео семпла, 0 без видео или если они различаются — нужен только для token reduction). |
| `VideoAudioCollator` | `video_audio_collator.py` | `DataModality.VIDEO_AUDIO` (`"video_audio"`) | `AudioCollator` для звуковых дорожек `video_audio`-файлов: та же тройка `spectrograms` / `spectrogram_lengths` / `num_spectrograms_per_sample`, но в отдельном бандле `video_audio` рядом с бандлом `video` (кадры того же файла). |
| `QwenImageCollator` | `qwen_image_collator.py` | `DataModality.IMAGE` (`"image"`) | Аналогично `ImageCollator`, но дополнительно возвращает `grid_thw` и `n_images_per_sample` для Qwen VE |
| `QwenVideoCollator` | `qwen_video_collator.py` | `DataModality.VIDEO` (`"video"`) | Аналогично `VideoCollator` (без `num_crops_per_frame`), но дополнительно возвращает `grid_thw` и `n_frames_per_sample` для Qwen VE |


В `llmfoundry/omni_data.py` создаётся общий `OmniCollator`, который:

1. для каждого модального коллатора берёт строковое значение `data_modality.value` и вызывает коллатор на списке примеров этой модальности длины `batch_size`, где элемент — это `dict` (если модальность есть) или `None` (если модальности нет)
2. складывает результат текстового коллатора в корневые ключи батча (`input_ids`, `labels`)
3. все остальные модальности помещает в `batch["modality_inputs"]`

---

## Как добавить новый коллатор

Каждый коллатор должен наследоваться от `BaseCollator` из `base_collator.py`. Интерфейс через `abc` требует определить:

- Переменную класса **`data_modality: DataModality`** — enum-значение модальности данных из `llmfoundry.data.processors.base_processor`. Его строковое значение (`data_modality.value`) соответствует типу файла в элементе датасета и ключу в `modality_inputs`;
- Метод **`__call__(examples: List[Optional[Dict[str, torch.Tensor]]])`** — список размера `batch_size` с элементами данной модальности, полученных из соответствующего процессора;
- Метод **`split_batch(batch, microbatch_size: int)`** — метод, который нарезает уже собранный батч данной модальности на микробатчи размера `microbatch_size`. Он используется композером для микробатчинга и DynamicSP. На выход метод должен предоставить список из содержимого `modality_inputs` данной модальности для каждого микробатча, для примера см. `image_collator.py`;

Коллатор должен учитывать, что батч может **вообще не содержать** примеров данной модальности (на вход - список из всех None).

Чтобы коллатор можно было использовать из конфига, он должен быть доступен как атрибут модуля `llmfoundry.data.collators` (см. `__init__.py`).

---

## Как задавать коллаторы в конфиге обучения

Коллаторы объявляются в `train_loader.dataset.collators` (и аналогично в eval/icl-конфигах), как список словарей. 

Каждый элемент содержит поля:

- **`name`**: имя класса коллатора (должно совпадать с экспортом в `llmfoundry.data.collators`)
- остальные поля — обязательные и опциональные агрументы конструктора коллатора

При этом `TextCollator` добавляется в коде **автоматически** первым, в yaml конфига обучения необходимо описать **только дополнительные модальные коллаторы**.

Пример из конфигурации коллатора для default/anyres:

```yaml
image_collator: &image_collator
  name: ImageCollator

train_loader:
  name: omni
  dataset:
    collators:
      - <<: *image_collator
```

Пример для qwen:

```yaml
image_collator: &image_collator
  name: QwenImageCollator

video_collator: &video_collator
  name: QwenVideoCollator

train_loader:
  name: omni
  dataset:
    collators:
      - <<: *image_collator
      - <<: *video_collator
```

Важно: если процессор возвращает `grid_thw` (например, `image_aspect_ratio: "qwen"`), в конфиге необходимо использовать qwen-коллаторы, иначе поле `grid_thw` не попадёт в `modality_inputs` для энкодера.

Аналогичный формат используется в `icl_tasks[*].sft_args.collators`.
