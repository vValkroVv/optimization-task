# Мультимодальные компоненты GigaOmni


## Структура директории и основные сущности

```
modalities/
├── __init__.py                          # Реестр модальностей
├── base_component.py                    # Базовый интерфейс компонентов BaseOmniComponent
├── base_modality.py                     # Базовые классы BaseModality / BaseModalityConfig
├── data_flow.py                         # DataFlow — конфигурация потока данных между компонентами
├── vision_modality.py                   # Реализация VisionModality (изображения)
├── video_only_modality.py               # Реализация VideoOnlyModality (видео без звука)
├── encoders/
│   ├── __init__.py                      # Реестр энкодеров
│   ├── intern_vit/
│   │   ├── configuration_intern_vit.py
│   │   └── modeling_intern_vit.py       # InternViTForVisualLanguageModeling
│   └── qwen2_5_vit/
│       ├── configuration_qwen2_5_vit.py # Qwen2_5VisionConfig
│       └── modeling_qwen2_5_vit.py      # Qwen2_5ViTForVisualLanguageModeling
└── adapters/
    ├── __init__.py                      # Реестр адаптеров
    ├── vision_pixel_shuffle_adapter.py  # VisionPixelShuffleAdapter
    ├── vision_qwen_2_5_adapter.py       # VisionQwen2_5Adapter (Qwen2.5 ViT → LLM)
    └── video_token_reduction.py         # VideoTokenReduction (сабмодуль адаптера)
```

### Основные сущности

| Сущность | Описание |
|---|---|
| **Модальность** | Определяет поток и правила промежуточных преобразований данных для решения задачи конкретной модальности. Задаёт выход модальности — последовательность эмбеддингов, которые будут встроены во вход LLM. |
| **Энкодер** | Преобразует специфичные для модальности входы в последовательность эмбеддингов и (опционально) любую дополнительную информацию для дальнейшего использования в адаптерах, головах или LLM. |
| **Адаптер** | Преобразует последовательность эмбеддингов из одного или нескольких энкодеров в единую последовательность эмбеддингов токенов, `hidden_size` которых совпадает с `hidden_size` LLM. |
| **Голова** | Содержит модули для генерации нетекстовых модальностей (напр., диффузия или VAE decoder для image output), а также преобразует выходы LLM и (опционально) энкодеров/адаптеров в лосс. |
| **Компонент** | Любой энкодер, адаптер или голова, вызываемый через `DataFlow`. Компоненты наследуются от `BaseOmniComponent` и обязаны реализовать `unwrap_tuple_output`. |

### Жизненный цикл данных в `forward`

Все компоненты работают через единый объект `data_state` — датакласс `GigaOmniDataState`, хранящий промежуточные результаты всех стадий обработки:

```
modality_inputs  →  encoders  →  adapters  → merge_modality_embeddings → LLM  →  heads -> loss
```

На стадиях энкодеров, адаптеров и голов `GigaOmniForCausalLM._forward_components(...)` сначала вызывает у каждой модальности `process_inputs_for_component(...)`, который делегирует в `process_inputs_for_encoders`, `process_inputs_for_adapters` или `process_inputs_for_heads`. Эти методы могут читать из `data_state`, валидировать его и добавлять `custom_inputs`.

После прохода всех адаптеров отдельно вызывается `process_inputs_for_language_model`, где модальность может дополнительно изменить текстовые `input_ids` / `labels` / `attention_mask` или подготовить другие данные перед LLM. Затем `merge_modality_embeddings` вставляет эмбеддинги модальности на позиции placeholder-токенов во входе LLM.

---

## 2. Модальности

### 2.1 Интерфейс `BaseModality`

Каждая модальность наследуется от `BaseModality` (`base_modality.py`) и должна реализовать следующие методы:

```python
class MyModality(BaseModality):
    config_class = MyModalityConfig

    def get_custom_input_contract(self) -> Dict[str, List[str]]: ...

    def process_inputs_for_encoders(
        self, data_state: Dict[str, Dict[str, Any]]
    ) -> None: ...

    def process_inputs_for_adapters(
        self, data_state: Dict[str, Dict[str, Any]]
    ) -> None: ...

    def process_inputs_for_language_model(
        self, data_state: Dict[str, Dict[str, Any]]
    ) -> None: ...

    def process_inputs_for_heads(
        self, data_state: Dict[str, Dict[str, Any]]
    ) -> None: ...

    def update_modality_inputs_with_dummy_values(
        self, modality_inputs: Dict[str, Dict[str, Any]]
    ) -> None: ...
```

#### Описание методов

| Метод | Когда вызывается | Назначение |
|---|---|---|
| `process_inputs_for_encoders` | До прохода всех энкодеров | Валидация/подготовка входных данных для энкодеров. |
| `process_inputs_for_adapters` | До прохода всех адаптеров | Валидация/подготовка входных данных для адаптеров. |
| `process_inputs_for_language_model` | После всех адаптеров, до LLM | Валидация/подготовка входных данных для LLM. |
| `process_inputs_for_heads` | До прохода всех голов | Валидация/подготовка входных данных для голов. |
| `update_modality_inputs_with_dummy_values` | В начале `forward` и `generate` | Гарантирует наличие всех ключей в `modality_inputs` (заполняет нулевыми тензорами), чтобы граф вычислений оставался корректным при отсутствии данных модальности в батче. |
| `get_custom_input_contract` | При инициализации модели, во время `data_flow.validate_graph(...)` | Объявляет допустимые bundle-ы и ключи в `custom_inputs`, которые модальность может сформировать в `process_inputs_for_*`. |

### 2.2 Конфигурация `BaseModalityConfig`

Конфиг модальности наследуется от `BaseModalityConfig` (который в свою очередь наследуется от `transformers.PretrainedConfig`) и содержит три ключевых поля:

```python
class MyModalityConfig(BaseModalityConfig):
    def __init__(
        self,
        data_flow_preset: Optional[str] = None,
        data_flow: Optional[DataFlow] = None,
        modality_output: Optional[Dict[str, Union[str, int]]] = None,
        **kwargs,
    ): ...
```

---

### 2.3 Конфигурация `data_flow`

`data_flow` описывает, какие компоненты (энкодеры, адаптеры, головы) нужны данной модальности и как данные передаются между ними. Это позволяет одному физическому компоненту (например, `InternViTForVisualLanguageModeling`) обрабатывать входы сразу нескольких модальностей.

Начиная с текущей схемы конфигурирования, основным источником и для `data_flow`, и для `modality_output` является `data_flow_preset`. То есть в типовом случае достаточно указать только:

```yaml
modalities:
  - name: VisionModality
    data_flow_preset: vision_intern_vit
```

Опциональные блоки `data_flow` и `modality_output` остаются доступными и работают как override поверх выбранного пресета (shallow merge по верхнему уровню словаря).

Источник preset-описаний: `llmfoundry/models/gigaomni/modalities/data_flow_presets.py`.
В этом файле хранятся `data_flow_presets` и `modality_output_presets`;
Именно эти registries используются по ключу `data_flow_preset` для выбора дефолтных `data_flow` и `modality_output`.

Текущие preset-ключи:

- `vision_intern_vit`
- `vision_qwen_vit`
- `video_only_intern_vit`
- `video_only_qwen_vit`
- `video_only_token_reduction_intern_vit`

#### Как работает override

В `BaseModalityConfig.merge_from_preset(...)` используется **shallow merge** на верхнем уровне:

```python
resolved = {**preset_value, **override_value}
```

Это значит:

- можно переопределить верхнеуровневый ключ целиком (например, весь `data_flow.adapters` или весь `modality_output`);
- нельзя точечно изменить вложенное поле внутри списка/словаря без передачи полного верхнеуровневого блока.

Пример точечного override для `modality_output` (работает, т.к. `modality_output` плоский):

```yaml
modalities:
  - name: VisionModality
    data_flow_preset: vision_intern_vit
    modality_output:
      placeholder_token_id: ${image_token_id}
```

Пример для `data_flow`: чтобы изменить `adapters`, нужно передать **весь** `adapters` блок целиком:

```yaml
modalities:
  - name: VisionModality
    data_flow_preset: vision_intern_vit
    data_flow:
      adapters:
        - name: VisionPixelShuffleAdapter
          inputs:
            - type: encoder_outputs
              name: intern_vit.image
              keys: ["pixel_embeds"]
          outputs:
            - name: pixel_shuffle_mlp.image
              keys: ["pixel_embeds"]
```

Структура `data_flow` в YAML-конфиге:

```yaml
data_flow:
  encoders:
    - name: <имя_компонента>          # должен совпадать с ключом в model.encoders
      inputs:
        - type: modality_inputs       # конкретно для стадии encoders допустимы только modality_inputs | custom_inputs
          name: <ключ_в_источнике>    # например, "image" или "InternViTForVisualLanguageModeling.image"
          keys: ["pixel_values"]      # список ключей тензоров, которые передаются в компонент
      outputs:
        - name: <ключ_в_encoder_outputs>
          keys: ["pixel_embeds"]
  adapters:
    - name: <имя_компонента>
      inputs:
        - type: encoder_outputs
          name: <ключ_из_encoder_outputs>
          keys: ["pixel_embeds"]
      outputs:
        - name: <ключ_в_adapter_outputs>
          keys: ["pixel_embeds"]
  heads: []
```

Допустимые типы для `inputs[].type` в целом:

| Тип | Описание |
|---|---|
| `modality_inputs` | Сырые входные тензоры из батча (pixel values, spectrogramms и т.д.) |
| `encoder_outputs` | Выходы ранее выполненных энкодеров |
| `adapter_outputs` | Выходы ранее выполненных адаптеров |
| `custom_inputs` | Произвольные тензоры, собранные модальностью в `process_inputs_for_*` |
| `language_model_outputs` | Выходы LLM, доступные только на стадии голов |

Но на каждой стадии разрешён только подмножество этих типов:

| Стадия | Разрешённые `inputs[].type` |
|---|---|
| `encoders` | `modality_inputs`, `custom_inputs` |
| `adapters` | `modality_inputs`, `custom_inputs`, `encoder_outputs` |
| `heads` | `modality_inputs`, `custom_inputs`, `encoder_outputs`, `adapter_outputs`, `language_model_outputs` |


`DataFlow` также жёстко задаёт контракт между `_forward_components(...)` и модульными компонентами:

- у каждого `DataFlowComponent` число `inputs` и `outputs` должно совпадать;
- внутри одной стадии имя output bundle (`outputs[].name`) должно быть уникальным;
- при вызове компонента `forward` получает `List[Dict[str, Tensor]]`, а не один словарь и не один элемент батча;
- `unwrap_tuple_output(inputs, outputs)` должен вернуть `List[Dict[str, Tensor]]` той же длины, что и `data_flow.outputs` для данного компонента;
- каждый словарь из `unwrap_tuple_output` сопоставляется своему `outputs[i]` и записывается в `data_state`.

`DataFlow` валидируется в два этапа:

1. `validate_schema()` при построении `DataFlow` проверяет форму самого графа: типы полей, равенство числа `inputs` и `outputs`, уникальность output bundle внутри стадии.
2. `validate_graph()` при инициализации модели проверяет связность с реальными компонентами: наличие нужных `encoders` / `adapters` / `heads`, существование upstream bundle-ов, наличие требуемых ключей, корректность `modality_output` и `custom_input_contract`.

Во время `forward` дополнительно вызываются:

- `validate_runtime_inputs()` перед запуском стадии;
- `validate_runtime_outputs()` после `unwrap_tuple_output()`.

---

### 2.4 Конфигурация `modality_output`

`modality_output` сообщает модели, откуда брать финальные эмбеддинги токенов модальности для вставки во вход LLM и по какому `placeholder_token_id` их идентифицировать в `input_ids`.

Важно: при наличии `data_flow_preset` значение `modality_output` по умолчанию также берется из пресета. Явно задавать `modality_output` в YAML нужно только если вы хотите переопределить preset-значение.

```yaml
modality_output:
  type: adapter_outputs           # откуда брать
  name: pixel_shuffle_mlp.image   # ключ внутри выбранного словаря
  key: pixel_embeds               # ключ конкретного тензора
  placeholder_token_id: 128004    # id токена-заглушки в input_ids, на место которого вставляются эмбеддинги
```

Механизм вставки реализован в `GigaOmniForCausalLM.merge_modality_embeddings`: для каждой модальности все позиции `placeholder_token_id` в плоском тензоре `input_ids` заменяются соответствующими эмбеддингами через `torch.masked_scatter`.

---

### 2.5 Реализованные модальности


| Модальность | Описание |
|---|---|
| **VisionModality** | Модальность для обработки изображений. |
| **VideoOnlyModality** | Модальность для обработки видео без звука. |


---

## 3. Компоненты: энкодеры, адаптеры, головы

### 3.1 Как добавить новый компонент

1. Создайте файл в соответствующей поддиректории: `encoders/`, `adapters/` или `heads/`.
2. Реализуйте класс, наследующийся от `BaseOmniComponent` и `torch.nn.Module` (или `transformers.PreTrainedModel`).
3. Реализуйте обязательный метод `unwrap_tuple_output` (см. [раздел 3.3](#33-обязательный-метод-unwrap_tuple_output)).
4. Зарегистрируйте класс в `__init__.py` соответствующей директории.
5. Опционально реализуйте методы для GigaFSDP (см. [раздел 3.4](#34-опциональные-методы-для-fsdp)).

---

### 3.2 Конфигурирование компонентов

Конфигурация компонента в YAML определяется набором ключей:

| Ключ | Обязателен | Описание |
|---|---|---|
| `name` | да | Имя класса компонента (должно совпадать с именем в `__init__.py`) |
| `alias` | нет | Альтернативное имя, под которым компонент регистрируется в `model.encoders` / `model.adapters` / `model.heads`. По умолчанию совпадает с `name`. Полезно, когда один класс используется для нескольких модальностей. |
| `init_device` | нет | Устройство инициализации (`"cpu"`, `"meta"`). По умолчанию `"cpu"`. |
| `init_seed` | нет | Seed для воспроизводимой инициализации весов. По умолчанию `42`. |
| `pretrained_model_name_or_path` | нет | Путь к чекпоинту для загрузки весов. |
| *прочие ключи* | нет | Передаются в конструктор класса или в `config_class` |

---

### 3.3 Обязательный метод `unwrap_tuple_output`

Все компоненты **обязаны** наследоваться от `BaseOmniComponent` (`base_component.py`) и реализовывать метод `unwrap_tuple_output`. Интерфейс наследуется от `abc.ABC`, поэтому компонент без реализации этого метода не должен инстанцироваться.

Метод преобразует кортежный вывод `forward` в список словарей, выровненный с `inputs` / `data_flow.outputs` для данного компонента из всех использующих его модальностей.

```python
def unwrap_tuple_output(
    self,
    inputs: List[Dict[str, torch.Tensor]],
    outputs: Tuple[torch.Tensor, ...],
) -> List[Dict[str, torch.Tensor]]:
    ...
```

---

### 3.4 Опциональные методы для GigaFSDP

При использовании GigaFSDP-обёртки (`gigafsdp_wrappers_gigaomni.py`) каждый компонент по умолчанию оборачивается целиком как один GigaFSDP-юнит. Для более гранулярного контроля компонент может реализовать два опциональных метода.

#### `get_giga_fsdp_modules_to_wrap_with_names`

Возвращает список `(модуль, имя)` субмодулей, каждый из которых должен быть обёрнут в отдельный GigaFSDP-юнит. 
```python
def get_giga_fsdp_modules_to_wrap_with_names(
    self, prefix: str
) -> List[Tuple[torch.nn.Module, str]]:
    return [
        (self.layer_1, f"{prefix}.layer_1"),
        (self.layer_2, f"{prefix}.layer_2"),
        ...
    ]
```

Если метод не определён, `_get_components_modules_to_wrap_with_names` оборачивает весь компонент как `(component, "<name>")`.

#### `get_giga_fsdp_modules_for_activation_checkpointing`

Возвращает список субмодулей, для которых будет включён activation checkpointing.

```python
def get_giga_fsdp_modules_for_activation_checkpointing(
    self,
) -> List[torch.nn.Module]:
    return list(self.encoder.layers)
```

Если метод не определён, `_get_components_modules_for_activation_checkpointing` добавляет компонент целиком в список checkpointed-модулей.

---

### 3.5 Реализованные компоненты

#### Энкодеры

| Класс | Файл | Описание |
|---|---|---|
| `InternViTForVisualLanguageModeling` | `encoders/intern_vit/modeling_intern_vit.py` | Vision encoder на основе `OpenGVLab/InternViT-300M-448px-V2_5` |
| `Qwen2_5ViTForVisualLanguageModeling` | `encoders/qwen2_5_vit/modeling_qwen2_5_vit.py` | Vision encoder Qwen2.5 ViT (flash-attn varlen, RoPE по патч-сетке); конфиг `Qwen2_5VisionConfig` в `configuration_qwen2_5_vit.py` |

#### Адаптеры

| Класс | Файл | Описание |
|---|---|---|
| `VisionPixelShuffleAdapter` | `adapters/vision_pixel_shuffle_adapter.py` | Pixel-shuffle адаптер, поддерживает редукцию для видео токенов |
| `VisionQwen2_5Adapter` | `adapters/vision_qwen_2_5_adapter.py` | Адаптер для выхода Qwen2.5 ViT (`pixel_embeds`, `window_index`): patch merger и выравнивание размерности под LLM |

#### Головы

На текущий момент реализованных голов нет. Директория `heads/` не создана — добавьте её при реализации первой головы.
