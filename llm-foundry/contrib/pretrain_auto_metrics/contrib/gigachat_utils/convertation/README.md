# Скрипт конвертации чекпоинтов

## Актуальный пайплайн gigafsdp → HF checkpoint

Для training-чекпоинтов текущий рекомендуемый сценарий такой:

```bash
CKPT_PREFIX=/path/to/checkpoint/ep_ba${STEPS}
cd ~/llm-foundry-conv && export PYTHONPATH=$(pwd)/:$(pwd)/contrib/composer:$(pwd)/contrib/streaming:$(pwd)/contrib/gigafsdp:$(pwd)/contrib/flash-linear-attention
cd ~/llm-foundry-conv/contrib/gigachat-utils

python convertation/map-fsdp-to-fsdp.py \
  --input-config-path configs/default.yaml

python convertation/scripts/merge_safetensors_ranks.py \
  --base-dir ${CKPT_PREFIX}/artifacts/hf_checkpoint \
```

На втором шаге `merge_safetensors_ranks.py`:
- склеивает rank-шарды в единый `model.safetensors.index.json`
- пересобирает `config.json` для `gigachat3_5`
- копирует `configuration_gigachat3_5.py` и `modeling_gigachat3_5.py`
- чинит tokenizer post-processor перед финальным `save_pretrained`
- удаляет промежуточные артефакты, если указан `--cleanup-intermediate-artifacts`

### Полезные флаги `merge_safetensors_ranks.py`

| Параметр | Описание |
|----------|----------|
| `--model` | Целевой тип модели: `gigachat3_5`, `gigachat3_5_dense` или `deepseek_v3`. Если не указан, берётся `model_family` из `hf_assembly_plan.json` (его пишет стриминговый маппер), иначе определяется по конфигу (`model_type`, затем `attention_type`/`linear_attention_type`). Для `gigachat3_5` конфиг другого типа приводится к нему принудительно; для `gigachat3_5_dense` несовместимый конфиг (MLA, MoE) — ошибка |
| `--no-cleanup-intermediate-artifacts` | Не удалять `composer_state.pt`, manifest и rank metadata после сборки (по умолчанию они удаляются) |
| `--llmfoundry-config` | Явный путь к `llm_foundry_config.yaml`, если автообнаружение не сработало |

### Откуда берутся файлы в HF чекпоинте

| Файл | Источник |
|------|----------|
| `config.json` | Embedded HF config из `composer_state.pt` + поля из `llm_foundry_config.yaml` + приведение к целевому типу модели (`--model`) |
| `modeling_gigachat3_5.py` | Уже подготовленный файл из `convertation/models/gigachat3_5/` |
| `configuration_gigachat3_5.py` | Уже подготовленный файл из `convertation/models/gigachat3_5/` |
| `modeling_gigachat3_5_dense.py`, `configuration_gigachat3_5_dense.py` | Для dense GQA моделей (`--model gigachat3_5_dense`): файлы из `convertation/models/gigachat3_5_dense/` |
| `tokenizer.*` | `tokenizer_path` из manifest / yaml / embedded config |

`merge_safetensors_ranks.py` не запускает `modular_model_converter.py`. Регенерация `modeling_gigachat3_5.py` — это отдельная dev-операция; она обновляет файл в `convertation/models/gigachat3_5/` и никогда не запускается автоматически во время сборки чекпоинта.

### Ручная/dev-постобработка уже собранного HF чекпоинта

```bash
python convertation/scripts/generate_hf_model_files.py \
  --checkpoint-dir /path/to/hf_checkpoint
```

При необходимости можно добавить:
- `--llmfoundry-config /path/to/llm_foundry_config.yaml`
- `--cleanup-intermediate-artifacts`

### Регенерация modeling_*.py из modular-источника (dev)

Обновить pre-generated `modeling_gigachat3_5.py` в `convertation/models/gigachat3_5/`:

```bash
# проверить, что окружение codegen настроено корректно:
python convertation/utils/modeling_codegen.py --dry-run

# регенерировать файл для нужной модели (по умолчанию gigachat3_5):
python convertation/utils/modeling_codegen.py --model gigachat3_5

# dense GQA вариант; его modular наследует классы из modeling_gigachat3_5.py,
# поэтому в staging копируются все шаблоны из convertation/models/:
python convertation/utils/modeling_codegen.py --model gigachat3_5_dense
```

Список моделей и их файлов (шаблон, modular, имена классов, `auto_map`) описан в `convertation/utils/hf_model_specs.py` (`MODEL_SPECS`).

Требует `libcst`, `ruff` в venv:
```bash
python -m pip install libcst ruff
export PATH="$HOME/.local/bin:$PATH" # чтобы заработал `ruff ...`
```
Скрипт конвертации вендорен в `convertation/codegen/`. Файлы модели временно копируются в директорию моделей установленного `transformers` и удаляются после конвертации.

---


Конвертация чекпоинтов производится путем запуска скрипта `map-fsdp-to-fsdp.py`.

Основные параметры:

* `--input-checkpoint-path` - путь к входному fsdp / gigafsdp чекпоинту (для input_tp_size > 1  - путь до __rank0__ чекпоинта)
* `--output-checkpoint-path` - путь к выходному чекпоинту (для output_tp_size > 1  - путь до __rank0__ чекпоинта)
* `--mappers` - набор преобразований которые будут применяться к чекпоинтам (доступные преобразования можно посмотреть через --help)
* `--input-tp-size` - tp_size входного чекпоинта
* `--output-tp-size` - tp_size выходного чекпоинта
* `--merge-lora` - параметр для слияния lora слоев, когда нужно сохранить модель целиком.
* `--print-final-model-structure` - принт слоев и их размерностей (работает для --output-tp-size 1)
* `--skip-optimizer` - удалять ли оптимизатор из финального чекпоинта. Если true, то все мапперы будут игнорировать оптимизатор, тем самым сэкономив память и время.
* `--input-config-path` - путь к yaml конфигу с параметрами конвертации
* `--torch-dtype` - параметр для map_deepseek_to_hf маппера
* `--tokenizer-path` - параметр для to_hf мапперов
* `--input-tp-embedding-type` - тип шардированного эмбеддинга при конвертации из tp чекпойнта
* `--output-tp-embedding-type` - тип шардированного эмбеддинга при конвертации в tp чекпойнт
* `--hf-llm-onnx` - при конвертации visual модели в HF сохранять также HF_llm_only + ONNX_visual_encoder_projector для последующей конвертации в TRT
* `--model-type` - задаёт `model_type` для конфига HF модели (`model_type` в `config.json`). Используется для конвертации модели (dense) в один из заданных классов (Llama, Qwen2, Qwen3)
* `--use_ema` - параметр, который включает загрузку и использование EMA-параметров модели. Важно: перед включением убедитесь, что ваш чекпойнт действительно содержит корректные EMA-параметры (см. раздел конфига callbacks/ema_weights») -- итерация конвертации чекпойнта должна совпадать с итерацией, начиная с которой были активированы EMA-параметры (по умолчанию: False)
* `--chunk_layers` - число слоёв, обрабатываемых за один проход сохранения в safetensors для контроля пиковой памяти и скорости (по умолчанию: 2)
* `--skip_ema_mapper` — управляет маппингом/удалением EMA-ключей. По умолчанию (`False`) EMA-ключи переименуются/могут удалиться. При `True` EMA-ключи остаются как есть. **Отличие от `--use_ema`:** тот выбирает, какие веса взять в итоговую модель (EMA или обычные) и всегда удаляет EMA-ключи. Если нужно **сохранить EMA-ключи** в чекпоинте — ставьте `--skip_ema_mapper=True`; иначе по умолчанию EMA-ключи могут пропасть/переименоваться и модель, ожидающая EMA-префиксы, не загрузится.


Пример запуска через yaml конфиг (альтернатива запуска полностью через командную строку):

```bash
python map-fsdp-to-fsdp.py --input-config-path configs/default.yaml
```

Также можно выполнить конвертацию через сабмит задачи на cloud. Для запуска необходимо прописать параметры конвертации в `config/default.yaml` и сабмитнуть задачу:

Также можно выполнить конвертацию через сабмит задачи на cloud. Для запуска необходимо сформировать YAML конфиг конвертации (пример в `config/default.yaml`) и заполнить забмит скрипт нужными значениями.

```bash
python scripts/submit_convertation_job.py
```

или

```bash
python scripts/submit_multinode_convertation_job.py
```

Параметры YAML-конфига идентичны аргументам основного скрипта.

```scripts/submit_convertation_job.py```:
В случае конвертации больших чекпоинтов необходимо запускать задачу на >1 GPU для увеличения доступной RAM (каждая карта добавляет 250 ГБ RAM на ноде). По умолчанию конвертация выполняется на 1 GPU.

Для чекпоинта размера N-B нужно порядка 8 * N Gb для конвертации без оптимизатора и 3 * 8 * N Gb с оптимизатором. Например, для конвертации 3B модели без оптимизатора нужно ~25Gb (1 карта), а для 90b без оптимизатора ~720Gb (3-4 карты).

```scripts/submit_multinode_convertation_job.py```:
Скрипт отправляет несколько одиночных джобов (по одной ноде и одному процессу на джоб), чтобы параллельно выполнить конвертацию частей чекпойнта. Каждый джоб получает свой `--global-rank`, а общее число рангов задаётся через `--world-size`. Это удобно, когда конвертация большого чекпойнта разбивается на ранги/шарды и требует суммарно больше RAM, чем доступно на одной ноде.

На данный момент скрипт предполагает, что пользователь понимает последовательность мапперов, которые используются.

### (!!!) Гайд по конвертации моделей после обновления
1. Если у вас шардированная dense модель обученная на старом коде - сконвертируйте ее в монолит и стартуйте с него на новом коде. Мапперы остаются прежними.
2. Если у вас dense модель обученная на новом / старом коде в монолитном формате - используйте стандартные мапперы как и раньше
3. Если вы учите новую линейку MOE моделей используйте только новые мапперы:
    * `map_mixed_gigafsdp_to_fsdp` - только для ep=1 и на выходе старый fsdp формат чекпоинтов.
    * `map_gigafsdp_to_fsdp_safetensor` - целевое решение (для ep>1) и формат с которого можно стартовать обучение llm-foundry (подробнее см. ниже)
    * `map_gigafsdp_to_hf_safetensor` - для конвертации в hf формат (подробнее см. ниже)

## Обновление логики работы мапперов
Конвертация новой линейки модели производится с помощью маппера `map_mixed_gigafsdp_to_fsdp`, который имеет ряд особенностей:
- он принимает на вход полную папку чекпоинта (`*/*/ep*_ba*`), вместо пути до ранга 0. Те родительскую папку относительно маппера `map_gigafsdp_to_fsdp`
- он умеет обрабатывать смешанные тп/еп размеры (по сути любые MOE модели). Прошлые мапперы так не могут.
- он не учитывает параметры `input_tp_size` (тк по структуре папки мы можем сами вычислить эти размеры) и `output_tp_size` (тк больше не будет необходимости делать выходной чекпоинт в тп формате). Оставляйте их значением = 1, по умолчанию.
- В данный момент маппер не поддерживает конвертацию оптимизатора. Выставляйте `skip_optimizer=true`

## Описание пайплайна

Скрипт конвертации последовательно применяет мапперы, определённые в переменной mappers. Мапперы принимают на вход state_dict и возвращают изменённый state_dict, что позволяет гибко комбинировать их. Мапперы делятся на два типа:

1) **Простые** — выполняют одно преобразование.

2) **Комбинированные (fused mappers)** — объединяют несколько простых мапперов в правильной последовательности.\
Для сложных конвертаций рекомендуется использовать готовые комбинированные мапперы во избежание ошибок.

Если для вашего сценария нет подходящего маппера, соберите последовательность самостоятельно. Учтите следующие моменты:

* При изменении tp_size и работе с HF-чекпоинтами требуется конвертация в классический монолитный формат (monolith), который не поддерживает packed attention. В этом случае сначала примените map_packed_to_base_attn для перевода внимания в базовый формат.

В текущей реализации:

* Если output-tp-size не задан, используется input-tp-size.

* Большинство мапперов не изменяют tp-size чекпоинта. Для изменения используйте маппер map_tp_to_tp.

## Доступные мапперы

| Маппер  | Описание  |
|:-------------:|:---------------:|
| remove_fsdp_optimizer | Удаляет state оптимизатора из чекпоинта fsdp |
| remove_gigafsdp_optimizer | Удаляет state оптимизатора из чекпоинта gigafsdp |
| map_base_to_packed_attn | Конвертирует fsdp чекпоинт из формата с классическим attention в формат с packed attention |
| map_packed_to_base_attn | Обратная операция для предыдущего маппера |
| map_monolith_fsdp_to_hf | Конвертирует и сохраняет монолитный *классический* fsdp чекпоинт в hf формат |
| map_hf_to_monolith_fsdp | Конвертирует hf чекпоинт в *классический* fsdp чекпоинт |
| map_tp_to_monolith | Конвертирует *классический* fsdp чекпоинт c tp_size > 1 в монолитный |
| map_fsdp_lora_atapter_to_hf | Конвертирует fsdp чекпоинт после LoRA дообучения и сохраняет только адаптер может сохранить full_model (обязательно укажите --tokenizer-path)|
| map_monolith_to_tp | Конвертирует *классический* монолитный fsdp чекпоинт в чекпоинт с tp_size > 1 |
| map_gigafsdp_to_fsdp | Конвертирует shard gigafsdp в monolith fsdp (работает с tp_size > 1 при этом tp_size сохраняется). Позволяет использовать все fsdp мапперы. По умолчанию если применяется только этот маппер и input-tp-size != output-tp-size, то после конвертации чекпоинта применяется map_tp_to_tp |
| map_ema_to_weights | В зависимости от `--use_ema` либо заменяет исходные веса на EMA и удаляет EMA-ключи (`--use_ema=True`), либо оставляет исходные веса и удаляет только EMA-ключи (`--use_ema=False`). В обоих случаях EMA-ключи удаляются. Если в ходе конвертации не хотим выполнять преобразование EMA-весов (их переименование и удаление), то можно выставить в аргументах `--skip_ema_mapper=True` (по умолчанию: False) |
| map_gigafsdp_to_fsdp_safetensor | Конвертирует mixed gigafsdp чекпоинт в монолитный FSDP и сохраняет веса поштучно в .safetensors (работает с packed attention и ep_size > 1, с **tp_size > 1 не тестировалось**). Учитывает --use_ema и --chunk_layers, поддерживает мультиранговый запуск (--global-rank/--world-size). **После запуска этого маппера для получения итогового чекпоинта обязательно запустить скрипт (ниже информация).** |
| map_gigafsdp_to_omni_safetensor | Конвертирует gigafsdp чекпоинт в safetensors формат. Может принимать чекпоинт текстовой модели - тогла слои LLM будут переименованы под архитектуру GigaOmni. |
| map_gigafsdp_to_hf_safetensor | Конвертирует mixed gigafsdp чекпоинт в HF формат и сохраняет веса в .safetensors (работает только с DeepSeekV3, то есть поддерживает только MLA (Multi Latent Attention), поддерживается ep_size > 1, с **tp_size > 1 не тестировалось**). Чтобы добавить поддержку своей модели, реализуйте собственный маппер в `base_mappers.py` и передавайте/выбирайте соответствующий маппер в зависимости от целевой архитектуры. На rank=0 дополнительно сохраняет конфиг/токенайзер; учитывает `--use_ema` и `--chunk_layers`, поддерживает мультиранговый запуск. **После запуска этого маппера для получения итогового чекпоинта обязательно запустить скрипт (ниже информация).**|
| map_gigafsdp_dense_gqa_to_hf_safetensor | То же, что `map_gigafsdp_to_hf_safetensor`, но для dense моделей с GQA (`attention_type: LlamaAttention`, без MoE слоёв, dense MTP), в том числе гибридных с linear attention. Маппер `Gigachat35DenseMapper` наследует `DeepseekV3Mapper`: fused `qkv_proj` режется на `q/k/v_proj`, остальное (нормы, linear attention, MTP) общее с родителем. Результат — `model_type: gigachat3_5_dense` (`convertation/models/gigachat3_5_dense/`), `merge_safetensors_ranks.py` подхватывает семейство из манифеста. **После запуска обязательно `merge_safetensors_ranks.py`.** |
| map_gigafsdp_slm_to_hf_safetensor | Конвертация dense SLM-чекпоинтов (`attention_type: LlamaAttention`: GQA с qk-norm) в собственный HF-формат `model_type: gigachat_slm` (`convertation/models/gigachat_slm/`). Маппер `GigachatSLMMapper` наследует `DeepseekV3Mapper`. **После запуска обязательно `merge_safetensors_ranks.py`.** |
| map_hf_safetensor_to_local | **Обратная** конвертация: HF safetensors → **local-monolith** safetensors. Поддерживает DeepSeekV3 / GigaChat 3.0 и GigaChat 3.5. Семейство модели и тип внимания каждого слоя определяются автоматически из `config.json`; layout MoE (`ggemm`/`sonic`) и `mtp_block_type` также выводятся из конфига (`moe_kernel`/`moe_impl`, `nextn_is_sparse`). Поддерживает FP8-деквантизацию входа и мультиранговый запуск (`--global-rank`/`--world-size`, `--chunk-layers`). Терминальный маппер — `save_checkpoint` не вызывается. **После запуска всех рангов обязательно запустить `merge_safetensors_ranks.py` (см. ниже).** |
| map_local_safetensor_to_hf | **Прямая** конвертация (обратная к `map_hf_safetensor_to_local`): local-monolith safetensors → HF DeepSeekV3 safetensors. Мультиранговый запуск. Терминальный маппер. **После запуска всех рангов обязательно запустить `merge_safetensors_ranks.py`.** |
| map_mixed_gigafsdp_to_mtp_safetensors | gigafsdp distcp → **local-monolith** safetensors с расширением MTP-голов: загружает distcp один раз, применяет EMA, реплицирует `mtp_block.mtp_layers.0` / `mtp_norms.0` в головы 1...N (аргумент `num_new_mtp_heads`, по умолчанию 2), сохраняет шарды по рангам в `rank{N:02d}/`. Поддерживает мультиранговый запуск. Терминальный маппер — `save_checkpoint` не вызывается. **После запуска всех рангов обязательно запустить `merge_safetensors_ranks.py` (см. ниже).** Детали MTP-голов и naming-конвенций — см. ниже раздел «MTP-мапперы». |
| map_safetensors_to_mtp_safetensors | HF safetensors → **HF safetensors** с расширением MTP-голов: читает HF-чекп (должен уже иметь ≥1 готовую MTP голову), дублирует head 0 в `num_new_mtp_heads` дополнительных голов (по умолчанию 2) под `model.layers.{num_hidden_layers + num_nextn_predict_layers + j}.*`. На rank 0 пишет обновлённый `config.json` + `composer_state.pt` (с увеличенным `num_nextn_predict_layers`). Поддерживает мультиранговый запуск, шарды лежат в `rank{N:02d}/`. **После всех рангов — `merge_safetensors_ranks.py`.** Детали — см. раздел «MTP-мапперы». |
| map_safetensors_graft_mtp_safetensors | HF safetensors + HF safetensors → **HF safetensors** (graft MTP-голов из другого чекпа): принимает base через `--input-checkpoint-path` и donor через `mtp_donor_path`. По умолчанию пересаживает все donor-головы, которых нет у base; режим append (дописать после) или replace (перезаписать существующие) контролируется `replace_base_heads`. На rank 0 пишет обновлённый `config.json`, `composer_state.pt` и `hf_assembly_plan.json` (триггер HF-постобработки в `merge_safetensors_ranks.py`). Поддерживает мультиранговый запуск. **После всех рангов — `merge_safetensors_ranks.py`** — он автоматически дособерёт HF-чекп (tokenizer, modeling_*.py, finalize config). Детали и параметры — см. раздел «MTP-мапперы». |
| __map_tp_to_tp__ | Это фьюз нескольких мапперов для удобной смены tp_size у fsdp чекпоинта с packed attention. |
| __map_fsdp_to_hf__ | Конвертирует Gigar / Gigaseek **fsdp** чекпоинт в hf формат (работает с packed attention и tp_size > 1) |
| __map_hf_to_fsdp__ | Конвертирует HF model модель в Gigar **fsdp** чекпоинт (модель с packed attention и опционально работает с output_tp_size > 1) |
| __map_gigafsdp_to_hf__ | Конвертирует Gigar / Gigaseek **gigafsdp** чекпоинт в hf формат (работает с packed attention и tp_size > 1) |
| __map_fsdp_to_omni_fspd__ | Изменяет название слоев в *классическом* fsdp чекпоинте Gigar / Gigaseek на подходяещие для GigaOmni (работает с packed attention и tp_size > 1) |
| __map_vision_fsdp_to_omni_fsdp__ | Изменяет название слоев в fsdp чекпоинте GigaVision на подходящие для GigaOmni. Работает только для стандартной архитектуры intern_vit -> pixel_shuffle -> LLM |
| __map_expand_embeddings_and_lm_head_fsdp__ | Расширяет слой эмбеддингов и lm_head *классического* монолитного чекпоинта на заданное количество токенов c нормальной инициализацией новых (работает с packed attention) |
| 

## MTP-мапперы

Три маппера для работы с MTP (multi-token prediction) головами. Отличаются источником входа и naming-конвенцией выхода.

### Safetensors naming: HF vs local-monolith

| Конвенция | Ключ MTP head `i` |
|-----------|-------------------|
| **HF DeepSeek-style** | `model.layers.{num_hidden_layers + i}.*` (MTP голова идёт как «лишний» layer-блок) |
| **Local-monolith** | `mtp_block.mtp_layers.{i}.*` + `mtp_block.mtp_norms.{i}.*` (отдельный блок монолита) |

| Маппер | Вход | Выход |
|--------|------|-------|
| `map_mixed_gigafsdp_to_mtp_safetensors` | gigafsdp distcp | **local-monolith** safetensors |
| `map_safetensors_to_mtp_safetensors` | HF safetensors | **HF** safetensors |
| `map_safetensors_graft_mtp_safetensors` | 2× HF safetensors (base + donor) | **HF** safetensors |

### Параметры расширения голов (replicate)

`map_mixed_gigafsdp_to_mtp_safetensors` и `map_safetensors_to_mtp_safetensors` **копируют head 0** в дополнительные слоты:

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `num_new_mtp_heads` | `2` | Сколько новых голов создать клонированием head 0. У входного чекпа должна быть ≥1 готовая MTP голова. Финальное число голов = `1 + num_new_mtp_heads` для gigafsdp-варианта; для safetensors-варианта = `num_nextn_predict_layers + num_new_mtp_heads`. |

### Параметры graft (`map_safetensors_graft_mtp_safetensors`)

Пересаживает реальные обученные головы из donor-чекпа, а не клонирует head 0 base'а. Полезно когда у тебя есть отдельный mtp-train run и хочется присоединить его головы к sft-чекпу.

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `mtp_donor_path` | — (обязательный) | Путь к donor HF-чекпу (должен иметь больше MTP голов чем base). |
| `donor_mtp_heads` | `range(base_M, donor_M)` (все, которых нет у base) | YAML-список 0-индексированных индексов donor-голов для пересадки. Например `[1, 2]`. |
| `replace_base_heads` | `false` | `false` — donor-головы **дописываются ПОСЛЕ** base-голов (`num_nextn_predict_layers` бампается на `len(donor_mtp_heads)`). `true` — donor-голова `d` **перезаписывает** base-голову `d` in-place (config не меняется, каждый `d` должен быть `< base_M`). |
| `shard_gb` | `4.0` | Cap размера одного выходного safetensors-шарда в GiB. |

**Пример (append).** Base имеет 1 голову (`num_nextn_predict_layers=1`), donor имеет 3 (`num_nextn_predict_layers=3`). С `donor_mtp_heads=[1, 2]`, `replace_base_heads=false`:
- В выходе 3 головы: head 0 — от base, heads 1–2 — от donor.
- `num_nextn_predict_layers` в финальном config.json становится 3.

**Пример (replace).** То же самое base и donor, но `donor_mtp_heads=[0]`, `replace_base_heads=true`:
- В выходе 1 голова: head 0 — от donor (base'овская стёрта).
- `num_nextn_predict_layers` остаётся 1.

---

## Основные компоненты самых распространенных мапперов:

* **map_tp_to_tp** - Комбинированный маппер, предназначенный для смены tp_size fsdp модели с packed attention. Маппер представляет собой последовательность простых:
    1) *packed_to_base_attn* конвертируем packed attention в классический attention
    2) *map_tp_to_monolith* конвертируем fsdp чекпоинт в классический монолитный
    3) *map_monolith_to_tp* конвертируем классический монолитный чекпоинт в tp fsdp
    4) *map_base_to_packed_attn* возвращаем packed attention
* **map_fsdp_to_hf** - Комбинированный маппер, конвертирующий ТП fsdp модель с packed attention в HF модель
    1) *packed_to_base_attn* конвертируем packed attention в классический attention
    2) *map_tp_to_monolith* конвертируем fsdp чекпоинт в классический монолитный (для tp_size > 1)
    3) *map_monolith_fsdp_to_hf* конвертируем классический монолитный чекпоинт в HF
* **map_gigafsdp_to_hf** - Комбинированный маппер, конвертирующий gigafsdp ТП модель с packed attention в hf модель
    1) *map_gigafsdp_to_fsdp* конвертируем gigafsdp чекпоинт в fsdp
    2) *map_fsdp_to_hf* конвертируем fsdp чекпоинит в hf

## Примеры конвертаций

Привести чекпоинт в формат .safetensors (указать корректный конфиг, обратите внимание на параметры `--use-ema` и `--chunk-layers`, а также на параметры `WORLD_SIZE` (общее число ранков/нод) и `N_GPUS_PER_JOB` (число GPU на один ранк/ноду) в `scripts/submit_multinode_convertation_job.py` ):

```bash
python scripts/submit_multinode_convertation_job.py
```

Обязательный шаг после завершения всех рангов:
```bash
python merge_safetensors_ranks.py --base-dir /path/to/output_dir
```

Где `/path/to/output_dir` — общий каталог, куда каждый ранг записал свои части (*.safetensors и метаданные).

Привести чекпоинт из __классического формата__ в новый (packed attention):
```bash
python convertation/map-fsdp-to-fsdp.py  \
    --input-checkpoint-path .../ep0-ba3000-rank0.pt \
    --output-checkpoint-path ..._packed/ep0-ba3000-rank0.pt \
    --mappers map_base_to_packed_attn  \
    --print-final-model-structure \
    --input-tp-size 1
```

Привести монолитную модель в __новом формате__ в hf чекпоинт:

```bash
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path .../ep0-ba3000-rank0.pt \
    --output-checkpoint-path .../hf_3k \ #folder
    --mappers map_fsdp_to_hf
```

Привести ТП=4 gigafsdp модель в __в новом формате__ в ТП=1 fsdp чекпоинт:

```bash
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path .../ep0-ba3000-rank0 \ #folder
    --output-checkpoint-path .../ep0-ba3000-rank0.pt \
    --mappers map_gigafsdp_to_fsdp \
    --input-tp-size 4 \
    --output-tp-size 1 \
    --input-tp-embedding-type EmbeddingParallelEmbedding
```

Привести TP модель __в новом формате__ в hf чекпоинт:

```bash
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path .../ep0-ba3000-rank0.pt \
    --output-checkpoint-path .../hf_3k \ #folder
    --mappers map_fsdp_to_hf \
    --input-tp-size 4 \
    --input-tp-embedding-type EmbeddingParallelEmbedding
```

Перевести монолитную модель Lora __в новом формате__ в hf чекпоинт:

```bash
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path .../ep0-ba3000-rank0.pt \
    --output-checkpoint-path .../hf_3k \ #folder
    --mappers map_fsdp_lora_to_hf \
    --input-tp-size 1
```


Привести TP=4 модель __в новом формате__ в TP=8 чекпоинт:

```bash
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path .../ep0-ba3000-rank0.pt \
    --output-checkpoint-path ..._tp8/ep0-ba3000-rank0.pt \
    --mappers map_tp_to_tp \
    --input-tp-size 4 \
    --output-tp-size 8 \
    --input-tp-embedding-type EmbeddingParallelEmbedding
    --output-tp-embedding-type EmbeddingParallelEmbedding
```

Привести TP=4 модель __в классическом формате__ в TP=8 чекпоинт:

```bash
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path .../ep0-ba3000-rank0.pt \
    --output-checkpoint-path ..._tp8/ep0-ba3000-rank0.pt \
    --mappers map_tp_to_monolith  map_monolith_to_tp \
    --input-tp-size 4 \
    --output-tp-size 8 \
    --input-tp-embedding-type EmbeddingParallelEmbedding
```

Привести монолитную модель в __новом формате__ в hf чекпоинт + ONNX_visual_encoder_projector:

```bash
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path .../ep0-ba3000-rank0.pt \
    --output-checkpoint-path .../hf_3k \ #folder
    --mappers map_fsdp_to_hf \
    --hf-llm-onnx
```

## Добавление новых этапов конвертаций

Для добавления нового преобразования нужно вписать в `mappers.py` функцию вида:

```python
@sequence_apply #опциональный декоратор
@time_memory_stat #опциональный декоратор, который выводит время работы маппера и используемую RAM
def my_custom_mapper(fsdp_state_dict : Dict,
                     args : Optional[argparse.Namespace] = None) -> Dict:
    #some ckpt editing
    return new_state_dict
```

`@sequence_apply` - декоратор, который позволяет применять ваш маппер к списку state dict'ов. Нужно если преобразование должно применяться к отдельным tp чекпоинтам.

После написания маппера его нужно добавить в словарь `MAP_CHECKPOINT_FUNCTIONS` по аналогии с другими и дописать в таблицу в этом README зачем он нужен.

С этого момента ваш маппер доступен для вызова из скрипта.


## Подготовка больших чекпоинтов для конвертации на кластере

### Вариант A: итоговый размер подготовленной к конвертации папки `< 2 TB`

1. Запускаем подготовку:

```python
# utils_script/submit_preparation_job.py
SRC = "<ПУТЬ_К_ИСХОДНОМУ_ЧЕКПОИНТУ>"    # корень исходного FSDP чекпоинта
DST = "<ПУТЬ_К_ПАПКЕ_НАЗНАЧЕНИЯ>"       # куда сложить подготовленный чекпоинт
PAIR_ACTION = "keep-nonema"             # оставить non-EMA веса (EMA удалить), возможные варианты: "keep-ema", "keep-nonema", "drop-both", "none"
```

2. На выходе в `<DST>` появится **готовая папка** с обрезанным чекпоинтом.

3. Запускаем конвертацию для полученной папки (ваш стандартный конвертационный скрипт), указав:

* `--input-checkpoint-path <DST>`
* остальные параметры под вашу модель/кластер.

---

### Вариант B: итоговый размер подготовленной к конвертации папки `≥ 2 TB`

1. Запускаем подготовку тем же способом (важно убедиться, что в скрипте указан параметр `--split-into-two`):

```python
# utils_script/submit_preparation_job.py
SRC = "<ПУТЬ_К_ИСХОДНОМУ_ЧЕКПОИНТУ>"
DST = "<ПУТЬ_К_ПАПКЕ_НАЗНАЧЕНИЯ>"
PAIR_ACTION = "keep-nonema"
```

2. На выходе в `<DST>` будут **две папки**: `ema_1`, `ema_2`:

```
<DST>/
  ├─ ema_1/
  └─ ema_2/
```

3. Для **каждой** папки (`ema_1`, `ema_2`) запустите **конвертацию** (как делали ранее), указывая путь к соответствующей поддиректории в качестве входа, а в качестве выхода, например, `<HF_DST>`.

4. После конвертации для **каждой** из папок вызовите `merge_safetensors_ranks.py` (как раньше) — чтобы склеить шарды ранков в единые файлы.

5. Затем выполните финальный merge из `utils_script`:

```bash
python utils_script/merge_hf_safetensors.py \
  --src1 <HF_DST>/ema_1 \
  --src2 <HF_DST>/ema_2 \
  --dst  <HF_DST>/merged
```

Готовый чекпоинт для автоматических метрик/дальнейшей работы будет лежать в:

```
<HF_DST>/merged
```

---
