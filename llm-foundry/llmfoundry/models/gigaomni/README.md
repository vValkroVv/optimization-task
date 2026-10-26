# GigaOmni

## 1. Общее описание

**GigaOmni** — мультимодальная языковая модель, объединяющая языковую модель (LLM) с произвольным набором модально-специфичных компонентов (энкодеры, адаптеры, головы). Модель предназначена для решения задач, в которых вход или выход содержит нетекстовые данные: изображения, видео и другие модальности в будущем.

На текущий момент поддерживаются следующие модальности:

| Модальность | Класс | Описание |
|---|---|---|
| Изображения | `VisionModality` | Обработка одного или нескольких статических изображений |
| Видео (без звука) | `VideoOnlyModality` | Обработка видео без аудиодорожки |

Модель реализована в виде `PreTrainedModel` (HuggingFace Transformers) и регистрируется под типом `gigaomni`. Поддерживает стандартные методы `from_pretrained` / `save_pretrained`, совместима с GigaFSDP для распределённого обучения.

Подробная документация по мультимодальным компонентам (модальностям, энкодерам, адаптерам, головам) находится в [modalities/README.md](modalities/README.md).

---

## 2. Шаги форварда модели

Ниже описан полный пайплайн одного прохода `forward` (`GigaOmniForCausalLM.forward`).

### Шаг 1. Инициализация `data_state`

Для каждой зарегистрированной модальности вызывается `update_modality_inputs_with_dummy_values` — метод гарантирует, что все ожидаемые ключи присутствуют в `modality_inputs` (отсутствующие заполняются dummy-тензорами). Это необходимо для сохранения корректности вычислительного графа в батчах, где данные некоторой модальности отсутствуют.

После этого собирается единый объект `data_state`:

```
data_state = {
    "modality_inputs":        { "text": {...}, "image": {...}, ... },
    "custom_inputs":          {},
    "encoder_outputs":        {},
    "adapter_outputs":        {},
    "language_model_outputs": {},
    "head_outputs":           {},
    "modality_outputs":       {},
    "real_modality_input_keys": {...},
    "pad_to_multiple_of":     2 * tp_sp,  # в generate() 1
    "pad_token_id":           ...,        # pad id конфига LLM или 0
}
```

### Шаг 2. Форвард энкодеров

Для каждой модальности вызывается `process_inputs_for_encoders` — подготовка и валидация входных данных для энкодеров.

Затем каждый зарегистрированный энкодер вызывается ровно один раз. Для него собирается список входных bundle-ов из `data_flow` всех модальностей, которые на него ссылаются. Результаты после `unwrap_tuple_output(...)` записываются в `data_state["encoder_outputs"]`.

### Шаг 3. Получение текстовых эмбеддингов

`input_ids` передаются через слой эмбеддингов LLM (`get_input_embeddings`). Полученный тензор `inputs_embeds` сохраняется в `data_state["encoder_outputs"]["text"]`.

### Шаг 4. Форвард адаптеров

Аналогично энкодерам, `_forward_components(..., component_type="adapter")` сначала вызывает `process_inputs_for_adapters` для всех модальностей, после чего каждый адаптер запускается один раз со списком входов, собранным из `data_flow`.

Адаптеры приводят выходы энкодеров к `hidden_size` LLM и при необходимости могут возвращать дополнительные тензоры. Например, `VisionPixelShuffleAdapter` в режиме video token reduction возвращает не только `pixel_embeds`, но и обновлённые `input_ids` / `labels` / `attention_mask` / `text_mask`. Результаты записываются в `data_state["adapter_outputs"]`.

### Шаг 5. Подготовка ко входу в LLM

Для каждой модальности вызывается `process_inputs_for_language_model` — финальная подготовка перед вставкой эмбеддингов. На этом шаге модальность может изменить текстовые поля в `data_state["modality_inputs"]["text"]`.

Это важно, потому что некоторые адаптеры меняют длину текстовой последовательности. В текущем коде такое делает `VideoOnlyModality`: она забирает из `adapter_outputs` обновлённые `input_ids` / `labels` / `attention_mask`, а при наличии `text_mask` ещё и фильтрует `encoder_outputs["text"]["inputs_embeds"]`. После фильтрации она допаддивает последовательность до `data_state.pad_to_multiple_of` (`2 * tp_sp` в `forward`, 1 в `generate()`; задаёт модель): token reduction выкидывает произвольное число токенов, а LLM шардирует последовательность по SP только при кратной длине.

### Шаг 6. Слияние мультимодальных эмбеддингов

`merge_modality_embeddings` заменяет позиции `placeholder_token_id` в плоском тензоре `inputs_embeds` соответствующими эмбеддингами каждой модальности через `torch.masked_scatter`. Количество placeholder-токенов должно строго совпадать с числом токенов на выходе модальности.

### Шаг 7. Форвард LLM

Собранный `inputs_embeds` подаётся в языковую модель. Для вызова используются текстовые `labels`, `attention_mask` и `position_ids` уже из `data_state`, а не напрямую из аргументов `forward`.

### Шаг 8. Форвард голов (TODO)

Форвард голов (`self.heads`) пока не реализован. Головы предназначены для генерации нетекстовых выходов и вычисления дополнительных лоссов.

---

## 3. Ограничения, учитывавшиеся при реализации

- **Произвольное число компонентов на модальность.** Каждая модальность может использовать любое количество энкодеров, адаптеров и голов, описывая поток данных между ними через конфигурацию `data_flow`.

- **Общие компоненты между модальностями.** Разные модальности могут ссылаться на один и тот же энкодер, адаптер или голову (например, `InternViTForVisualLanguageModeling` используется одновременно для `VisionModality` и `VideoOnlyModality`). Физически компонент в `nn.ModuleDict` хранится один раз.

- **Произвольные комбинации модальностей при обучении.** Модель корректно работает при включении любого подмножества реализованных модальностей, или вообще без них (т.е. исключительно на текстах). `update_modality_inputs_with_dummy_values` гарантирует, что граф вычислений остаётся неизменным вне зависимости от наличия мультимодальных данных.

- **Ровно один форвард через каждый компонент (требование GigaFSDP).** `_forward_components` итерируется по `nn.ModuleDict` и вызывает каждый энкодер/адаптер/голову ровно один раз, собирая все необходимые входные bundle-ы через `data_flow` перед вызовом. Это исключает повторные форварды через один и тот же модуль.

- **Контракт компонента определяется `DataFlow`.** После рефакторинга `DataFlow` проверяет не только наличие компонент, но и форму графа: число `inputs` и `outputs` у каждого `DataFlowComponent` должно совпадать, имена output bundle внутри стадии должны быть уникальны, а допустимые источники данных зависят от стадии (`encoders` / `adapters` / `heads`).

- **Компоненты должны возвращать тензор или кортеж тензоров (требование PyTorch / GigaFSDP).** Каждый компонент обязан реализовывать метод `unwrap_tuple_output`, который конвертирует кортежный вывод `forward` в список словарей для записи в `data_state`. Этот список выравнивается с входными bundle-ами, собранными через `data_flow`, а не с элементами батча. Это позволяет `forward` всегда возвращать тензоры (а не произвольные Python-объекты), что необходимо для корректной работы GigaFSDP.

---

## 4. Примеры конфигурации модели


### 4.1. Конфигурация для обучения vision модальности

Ниже приведён сокращённый пример конфигурации для GigaOmni c LLM типа GigaMix для vision модальностью.

```yaml
model:
  name: gigaomni_causal_lm
  init_device: meta
  language_model:
    init_device: ${model.init_device}
    name: giga_mix_causal_lm
    pretrained: false                    # True, если необходимо подгрузить веса из huggingface
    pretrained_model_name_or_path: null  # Путь до чекпоинта для подгрузки модели 

    # Остальные параметры LLM, как в соответствующем конфиге
    vocab_size: 128256
    max_position_embeddings: 262144
    attention_dropout: 0.0
    # ....

  encoders:
    - name: InternViTForVisualLanguageModeling
      pretrained_model_name_or_path: ...
      init_device: ${model.init_device}

  adapters:
    - name: VisionPixelShuffleAdapter
      init_device: ${model.init_device}
      llm_hidden_size: ${model.language_model.hidden_size}  
      ve_hidden_size: 1024                                  # Стандартное значение для InternViT энкодера
      downsample_ratio: 0.5

  modalities:
    - name: VisionModality
      data_flow_preset: vision_intern_vit
      modality_output:
        type: adapter_outputs
        name: pixel_shuffle_mlp.image
        key: pixel_embeds
        placeholder_token_id: ${image_token_id}
```

### 4.2. Конфигурация для обучения video+audio модальности (переиспользование компонентов)

Ниже приведен пример приведён сокращённый пример конфигурации для GigaOmni c LLM типа GigaMix для video модальности, с поддержкой audio.
Фактически поддержки такого обучения на данный момент нет, но будет реализована в будущем. Пример необходим для демонстрации конфигурации в случаях, когда необходимо переиспользовать один компонент в нескольких модальностях.

```yaml
model:
  name: gigaomni_causal_lm
  init_device: meta
  language_model:
    init_device: ${model.init_device}
    name: giga_mix_causal_lm
    pretrained: false                    # True, если необходимо подгрузить веса из huggingface
    pretrained_model_name_or_path: null  # Путь до чекпоинта для подгрузки модели 

    # Остальные параметры LLM, как в соответствующем конфиге
    vocab_size: 128256
    max_position_embeddings: 262144
    attention_dropout: 0.0
    # ....

  encoders:
    - name: InternViTForVisualLanguageModeling
      pretrained_model_name_or_path: ...
      init_device: ${model.init_device}
    
    - name: Conformer
      pretrained_model_name_or_path: ...
      init_device: ${model.init_device}
      # Остальные параметры Conformer

  adapters:
    - name: VisionPixelShuffleAdapter
      init_device: ${model.init_device}
      llm_hidden_size: ${model.language_model.hidden_size}  
      ve_hidden_size: 1024                                  # Стандартное значение для InternViT энкодера
      downsample_ratio: 0.5
    
    - name: AudioMLPAdaper
      init_device: ${model.init_device}
      # Остальные параметры AudioMLPAdaper

  modalities:
    - name: VisionModality
      data_flow_preset: vision_intern_vit
      modality_output:
        type: adapter_outputs
        name: pixel_shuffle_mlp.image
        key: pixel_embeds
        placeholder_token_id: ${image_token_id}

    - name: AudioModality
      data_flow:
        encoders:
          - name: Conformer
            inputs:
              - type: modality_inputs
                modality_inputs: audio
                keys: ["spectrograms"]
            outputs:
              - type: conformer.audio
                keys: ["audio_embeds"]

        adapers:
          - name: AudioMLPAdaper
            inputs:
              - type: encoder_outputs
                name: conformer.audio
                keys: ["audio_embeds"]
            outputs:
              - name: mlp.audio
                keys: ["audio_embeds"]

      modality_output:
        type: adapter_outputs
        name: mlp.audio
        key: audio_embeds
        placeholder_token_id: ${audio_token_id}

    - name: VideoModality
      data_flow:
        encoders:
          - name: InternViTForVisualLanguageModeling
            inputs:
              - type: modality_inputs
                name: video
                keys: ["pixel_values"]
            outputs:
              - name: intern_vit.video
                keys: ["pixel_embeds"]

          - name: Conformer
            inputs:
              - type: modality_inputs
                modality_inputs: video
                keys: ["spectrograms"]
            outputs:
              - name: conformer.video
                keys: ["audio_embeds"]

        adapters:
          - name: VisionPixelShuffleAdapter
            inputs:
              - type: encoder_outputs
                name: intern_vit.video
                keys: ["pixel_embeds"]
            outputs:
              - name: pixel_shuffle_mlp.video
                keys: ["pixel_embeds"]
            
          - name: AudioMLPAdaper
            inputs:
              - type: encoder_outputs
                name: conformer.video
                keys: ["audio_embeds"]
            outputs:
              - name: audio_mlp.video
                keys: ["audio_embeds"]

      modality_output:
        type: custom_inputs
        name: video_interleaved_emebds
        key: video_embeds
        placeholder_token_id: ${video_image_token_id}
```
