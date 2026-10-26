### Запуск обучения (эксперименты)

См. [документацию](https://gitlab.ai.cloud.ru/gigallms/llm-foundry/-/tree/main/sber_scripts) по `submit_template.py`

Необходимые значения для параметров:

- `SUBMISSION_TYPE = "cloud"`
- `REGION = "SR008"`
- `USER_LOGIN, EXP_NAME` - на свое усмотрение
- `CONFIG_NAME` - название `editable`-конфига
- `DEFAULT_CONFIG_NAME` - название `default`-конфига
- `N_GPUS, N_WORKERS` - число карт (на ноде) и число нод, соответственно
- `BASE_IMAGE = "cr.ai.cloud.ru/34c0435a-f6ea-4d20-a4dd-1649d623b5a5/llm-foundry-train:2025_02_03_commit277c3722_audio_v0.2"`
- `SCRIPT_TYPE = "pytorch2"`

### Запуск обучений в бесконечной джобе (для дебагинга, разработки)

Старт бесконечной джобы описан в [документации](https://gitlab.ai.cloud.ru/gigallms/llm-foundry/-/tree/main/sber_scripts) по `submit_template.py`.

Необходимые параметры не отличаются от вышеописанных, за исключением `SCRIPT_TYPE = "infinity_job"`.

В скриптах `sber_scripts/infinite_jobs/run.sh` и `sber_scripts/infinite_jobs/clean.sh` следует установить корректный параметр `NNODES`.

Внутри бесконечной джобы (из корня `llm-foundry`) запуск обучения осуществляется с помощью скрипта `sber_scripts/infinite_jobs/run.sh`:

```
bash sber_scripts/infinite_jobs/run.sh tts {DEFAULT_CONFIG} {EDITABLE_CONFIG}
```

Пути следует указывать относительно `tts/default_configs` и `tts/editable_configs`, соответственно.


### Конвертация gigafsdp-чекпоинта LLM в монолитный

Перейти в директорию `contrib/gigachat-utils` (см. необходимый commit в репозитории)

```
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path {SHARDED_PATH} \
    --output-checkpoint-path {OUT_PATH} \
    --mappers map_gigafsdp_to_fsdp map_fsdp_to_vision_fspd \
    --input-tp-size {TP_SIZE} \
    --skip-optimizer
```

Более детально см. [README](https://gitlab.ai.cloud.ru/gigallms/gigafsdp/-/tree/master#работа-с-чекпоинтами)

Для того, чтобы начать подгружать LLM нужно указать опцию `load_path: {OUT_PATH}` в конфиге.

При каждом изменении топологии сети (добавление LoRA, изменение ранга, разморозка каких-то слоев и т.д.) нужен новый `{OUT_PATH}_sharded`, который сохраняет "закешированный" чекпоинт в формате gigafsdp. Поэтому следует удалить (почитайте README).

Важно заметить, что при конвертации через маппер `map_fsdp_to_vision_fsdp` предполагается, что LLM находится в Composer модели с полем `_language_model`.



### Конвертация чекпоинта обучения в монолитный

Аналогично предыдущему

```
python convertation/map-fsdp-to-fsdp.py \ 
    --input-checkpoint-path {SHARDED_PATH} \
    --output-checkpoint-path {OUT_PATH} \
    --mappers map_gigafsdp_to_fsdp \
    --input-tp-size {TP_SIZE} \
    --input-config-path {INPUT_CONFIG_PATH} \
    --tokenizer {TOKENIZER_PATH} \
    --skip-optimizer
```

Артефакты обучения в `llm_foundry` сохраняют входящий конфиг в `artifacts/llm_foundry_config.yaml` и токенизатор в `artifacts/tokenizer`


### Запуск LoRA PEFT-finetuning

Аналогично обычному обучению, но с измененным конфигом. В секцию `model` следует добавить `lora` с примерными опциями:

```
    lora:
      args:
        lora_alpha: {ALPHA}
        peft_type: LORA
        r: {RANK}
        target_modules: '^.*(qkv_proj|o_proj|gate_proj|up_proj|down_proj).*$'
        use_dora: false
        use_rslora: false
        lora_dropout: {DROPOUT}

```


### Конвертация чекпоинта обучения в HF-формат

Для запуска инференса сначала нужно сконвертировать чекпоинт GigaFSDP в HuggingFace формат. Для этого зайдите в модуль `gigachat_utils`

```
cd contrib/gigachat-utils
export CODE_PATH='../..'
export PYTHONPATH=$CODE_PATH:$CODE_PATH/contrib/composer:$CODE_PATH/contrib/streaming:$CODE_PATH/contrib/gigafsdp
```

В случае модели без LoRA-адаптера:
```
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path {SHARDED_PATH} \
    --output-checkpoint-path {OUT_PATH} \
    --mappers map_gigafsdp_to_fsdp map_fsdp_to_hf \
    --input-tp-size 1 \
    --input-config-path {INPUT_CONFIG_PATH} \
    --tokenizer {TOKENIZER_PATH} \
    --skip-optimizer
```

В случае модели с LoRA-адаптером:
```
python convertation/map-fsdp-to-fsdp.py \
    --input-checkpoint-path {SHARDED_PATH} \
    --output-checkpoint-path {OUT_PATH} \
    --mappers map_gigafsdp_to_fsdp map_fsdp_lora_to_hf \
    --input-tp-size 1 \
    --input-config-path {INPUT_CONFIG_PATH} \
    --tokenizer {TOKENIZER_PATH} \
    --skip-optimizer
```


### Запуск инференса модели

Основной конфиг для скрипта инференса находится в `sber_configs/tts/inference/config.yaml`

- Параметры кодека для озвучки и процедуры сэмплирования находятся в данном конфиге
- Важной опцией является `use_force_text`, которую необходимо выставить в `True` для TTS-сетапа (текст генерации "форсится" из датасета) и в `False` для QA (параллельная генерация текста и аудио)

В скрипте `sber_scripts/tts/run_inference.sh` необходимо указать

- `MODEL_CFG` - путь до конфига модели (есть в `artifacts/llm_foundry_config.yaml` обученной модели)
- `MODEL_HF_CKPT_PATH` - путь до HF-чекпоинта
- `INFERENCE_CFG` - путь до конфига инференса
- `SAVE_PATH` - путь для сохранения `codes` и `wavs`

Запуск (из корня `llm-foundry`):

```
bash sber_scripts/tts/run_inference.sh 
```

### Подсчет OBM-метрик


```
Not Implemented Yet
```
