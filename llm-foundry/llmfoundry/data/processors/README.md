# Процессоры данных для обучения омни-модели

## Общее описание

Процессоры отвечают за предобработку мультимодальных файлов (изображений, видео, аудио) и встраивание модальных токенов в текстовую последовательность сэмпла. Каждый процессор обрабатывает ровно одну модальность.

Все процессоры наследуются от базового класса `BaseProcessor` (`base_processor.py`), который, в свою очередь, наследуется от `ProcessorMixin` из библиотеки `transformers`. `BaseProcessor` является `abc.ABC`-интерфейсом и требует реализовать `__init__`, `__call__` и `get_num_tokens`. Также `BaseProcessor` хранит общий текстовый токенизатор `tokenizer`, если он передан в конструктор.

### Жизненный цикл процессора в пайплайне данных

При загрузке сэмпла класс `StreamingOmniDataset.__getitem__` последовательно вызывает все зарегистрированные процессоры:

```python
for processor in self.processors:
    dataset_sample.update(processor(files=files, dataset_sample=dataset_sample))
```

Каждый вызов получает:
- `files` — список файлов сэмпла, каждый из которых является словарём с полями:
  - `type` (str) — модальность: `"image"`, `"audio"`, `"video"`, `"video_audio"`.
  - `embed` (bool) — нужно ли встраивать файл в последовательность.
  - `path` (str) — путь к файлу.
  - `content` (Optional) — содержимое файла, подготовленное датасетом. Для image это может быть декодированный `PIL.Image` или сырые `bytes`. Если задан, процессор использует его вместо чтения файла с диска по `path`.
- `dataset_sample` — текущее состояние сэмпла вида:

```python
{
    "text": {
        "input_ids": torch.LongTensor,
        "labels": torch.LongTensor,
    }
}
```

Процессор возвращает обновлённый `dataset_sample`, добавляя в него ключ с данными своей модальности:

```python
{
    "text": {
        "input_ids": torch.LongTensor,   # расширенный токенами модальности
        "labels": torch.LongTensor,       # расширенный токенами модальности
    },
    "<modality>": {                       # например, "image", "audio", "video"
        ...                               # тензоры данных модальности
    }
}
```

### NFS-mode и S3-mode стримов

Датасет поддерживает два режима подгрузки файлов:

| Режим | Признак в конфиге | Где лежат данные файла | Что готовит датасет |
|---|---|---|---|
| NFS | задан `files_folder_path` | на FS кластера | резолвит `file["path"]` до абсолютного |
| **S3** | задан `remote: s3://...`, **без** `files_folder_path` | в `payloads` (tar-архив) внутри MDS-семпла | распаковывает tar и кладёт содержимое файла в `file["content"]` (для image это могут быть сырые `bytes` или декодированный `PIL.Image`) |

### Существующие процессоры

| Класс | Файл | Модальность (`data_modality`) | Описание |
|---|---|---|---|
| `GigaOmniVisionProcessor` | `gigaomni_vision_processor.py` | `DataModality.IMAGE` | Предобработка изображений; поддерживает режимы `anyres`, `anyres_onevision`, `qwen`, `pad`. Выполняет тайлинг изображений и встраивает `image_token_id` токены. |
| `GigaOmniSpeechProcessor` | `gigaomni_speech_processor.py` | `DataModality.AUDIO` | Читает аудиофайлы, вычисляет спектрограммы (фильтербанки) и встраивает `audio_token_id` токены. |
| `GigaOmniVideoProcessor` | `gigaomni_video_processor.py` | `DataModality.VIDEO` | Составной процессор: внутри использует `GigaOmniVisionProcessor` и `GigaOmniSpeechProcessor` для кадров и аудиодорожки видео. Поддерживает чередованный (`interleaved`) и последовательный (`sequential`) порядок токенов, а также добавление временны́х меток кадров через текстовый токенизатор. |

---

## Как добавить новый процессор

### 1. Создать файл процессора

Создайте файл с именем, содержащим подстроку `gigaomni`, например `gigaomni_depth_processor.py`, в директории `llmfoundry/data/processors/`.

### 2. Унаследоваться от `BaseProcessor` и задать `data_modality`

Добавьте новую модальность в enum `base_processor.py::DataModality` и укажите его значение в аттрибуте класса процессора `data_modality`. Затем, релализуйте процессор в соответствии с интерфесом, описанным в `base_processor.py::BaseProcessor`.

```python
from .base_processor import BaseProcessor, DataModality

class GigaOmniDepthProcessor(BaseProcessor):
    data_modality = DataModality.DEPTH  # enum-значение модальности
    attributes = []
    valid_kwargs = [...]       # список параметров __init__ для сериализации

    def __init__(self, ..., **kwargs):
        super().__init__(**kwargs)
        ...
    
    def __call__(self, ...):
        ...
    
    def get_num_tokens(self, ...):
        ...
```

Значение `data_modality` должно быть членом enum `DataModality` из `base_processor.py`. Для новой модальности сначала добавьте новое значение в `DataModality`, а затем используйте его в процессоре. Строковое значение `data_modality.value` должно быть уникальным среди процессоров и соответствовать значению поля `type` в записях `files` датасета.

Если процессору нужен текстовый токенизатор, передайте `tokenizer` в `super().__init__(tokenizer=tokenizer, **kwargs)` или оставьте его в `**kwargs`. Базовый класс принимает путь/имя токенизатора или готовый `PreTrainedTokenizerBase`, сохраняет его в подпапку `tokenizer/` и восстанавливает в `from_pretrained`.

### 3. Реализовать метод `__call__`

Метод выполняет предобработку файлов и встраивает токены модальности в текстовую последовательность.

**Сигнатура:**

```python
def __call__(
    self,
    files: Optional[List[Dict[str, Any]]] = None,
    dataset_sample: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    input_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, Any]]:
```

**Входные данные.**
- `files` — список словарей с метаданными (как в датасете); процессор сам фильтрует файлы по `type`;
- `dataset_sample` — уже подготовленный сэмпл;
- `input_ids`, `labels`, `attention_mask` — альтернативный способ передать текстовую часть сэмпла без `dataset_sample`.

Для текстовой части сэмпла передаётся либо `dataset_sample`, либо пара `input_ids` / `labels` (не оба варианта одновременно). Вспомогательный метод `self._prepare_dataset_sample(...)` нормализует оба варианта в единый формат `dataset_sample`.

**Возвращаемое значение.** Словарь, который будет объединён с текущим `dataset_sample` через `.update()`:

```python
{
    "text": {
        "input_ids": ...,  # расширенный вставкой токенов модальности
        "labels": ...,
    },
    "depth": {             # ключ совпадает с data_modality.value
        "pixel_values": ...,
    }
}
```

Если файлов модальности в сэмпле нет, метод должен вернуть неизменённый `dataset_sample`.

### 4. Реализовать метод `get_num_tokens`

Метод возвращает суммарное количество токенов, которые будут добавлены в последовательность для переданных файлов. Используется при вычислении статистик и метриках длины сэмпла.

**Сигнатура:**

```python
def get_num_tokens(
    self,
    files: Optional[List[Dict[str, Any]]] = None,
) -> int:
```

### 5. Зарегистрировать процессор в `__init__.py`

Добавьте импорт нового класса в `llmfoundry/data/processors/__init__.py`.

---

## Как задать процессор в конфиге обучения

Процессоры задаются в секции `dataset.processors` загрузчика данных. Каждый элемент списка должен содержать поле `name` с именем класса процессора (как он зарегистрирован в `__init__.py`) и все необходимые параметры конструктора.

Чтобы не дублировать конфигурацию процессора между `train_loader` и секцией `icl_tasks`, удобно использовать YAML-якоря (`&`) и ссылки на них (`<<: *`):

```yaml
vision_processor: &vision_processor
  name: GigaOmniVisionProcessor
  image_aspect_ratio: anyres
  num_image_tokens_per_crop: 256
  image_token_id: ${image_token_id}
  image_processor: ${ve_path}
  image_grid_pinpoints: ${image_grid_pinpoints}

train_loader:
  name: omni
  dataset:
    processors:
      - <<: *vision_processor

icl_tasks:
  - label: mmmu
    ...
    sft_args:
      processors:
        - <<: *vision_processor
```

Процессоры применяются последовательно в том порядке, в котором они перечислены в конфиге. Каждый процессор получает `dataset_sample`, уже обновлённый предыдущими процессорами.
