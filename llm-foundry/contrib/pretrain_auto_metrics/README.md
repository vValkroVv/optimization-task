# Pretrain Auto Metrics

### Установка
```
git clone https://gitlab.ai.cloud.ru/gigallms/pretrain-auto-metrics.git --recursive
```

```
git pull --recurse-submodules && git submodule update --init --recursive
```

### Ветки и релизы
- MR идут в `dev`; `main` это последний релиз, его двигает ночной релиз.
- Ночью на `dev` гоняются e2e, зелёная ночь ставит тег `vX.Y.Z`, двигает `main` и выкладывает код на Papaya.
- `VERSION` и `images.json` пишет релиз, руками не правят.

Подробно: [CONTRIBUTING → Ветки и релизы](./docs/CONTRIBUTING.md#ветки-коммиты-mr), [карта CI/CD](./docs/ci_pipeline_map.md).

# Как запустить автометрики
Метрики могут замерить чекпоинты двух видов:
1. HF
2. GigaFSDP

Порядок запуска метрик
1. Взять конфиг для автометрик из [примеров](./configs/)
2. Заполнить [шаблон](https://gitlab.ai.cloud.ru/gigallms/llm-foundry/-/blob/main/sber_scripts/jobs/examples/autometrics.py?ref_type=heads) или по пути `llm-foundry/sber_scripts/jobs/examples/autometrics.py` - **для запуска из llm-foundry**. Заполнить [шаблон](https://gitlab.ai.cloud.ru/gigallms/pretrain-auto-metrics/-/blob/main/submit_pam_job.py?ref_type=heads) в корне pretrain-auto-metrics - **для запуска из pretrain-auto-metrics**
3. Отправить джобу через `python3 sber_scripts/jobs/examples/autometrics.py` из корня llm-foundry
4. Следить за логами в `<путь_к_чекпоинту>/artifacts/logs` и логах джобы в клауде, метрики сохраняются в `<путь_к_чекпоинту>/artifacts/metrics`, также метрики выгружаются в TB по пути из конфига

# Локальный запуск external API замеров через Docker Compose
Для локального запуска замеров внешнего OpenAI-compatible или IFT API можно использовать `docker-compose.yml` из корня репозитория. Compose **не билдит** образ локально, а пуллит готовый PAM-образ из registry (`pull_policy: always`).

Перед запуском нужно залогиниться в registry:
```bash
docker login cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0
```

В выбранном конфиге нужно заполнить `autometrics.inference.api_key`:
1. `configs/openai.yaml`
2. `configs/ift.yaml`

Подготовить директорию для результатов и разрешить запись контейнерному пользователю:
```bash
mkdir -p docker_runs/test_openai/artifacts
chmod -R 777 docker_runs/test_openai
```

Запуск OpenAI-compatible конфига:
```bash
docker compose run --rm pam
```

Запуск IFT конфига:
```bash
mkdir -p docker_runs/test_ift/artifacts
chmod -R 777 docker_runs/test_ift

docker compose run --rm pam-ift
```

Результаты и логи сохраняются в директории запуска:
```bash
docker_runs/test_openai/artifacts/logs
docker_runs/test_openai/artifacts/metrics/obm
docker_runs/test_ift/artifacts/logs
docker_runs/test_ift/artifacts/metrics/obm
```

`tensorboard_path` задаётся в конфиге и должен указывать на путь, доступный внутри контейнера, например `../docker_runs/tensorboard`.

Так как директория запуска создаётся пользователем до запуска контейнера с правами на запись, после завершения замеров её можно удалить обычным пользователем:
```bash
rm -rf docker_runs/test_openai
rm -rf docker_runs/test_ift
```

Посмотреть OBM-метрики из терминала:
```bash
python3 - docker_runs/test_openai/artifacts/metrics/obm/all_metrics.jsonl <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
for line in path.read_text().splitlines():
    row = json.loads(line)
    metric = row.get("metric") or {}
    print(row.get("name"), metric.get("name"), metric.get("value"))
PY
```

По умолчанию compose использует тег `latest`, который обновляется CI для default branch. Если нужно запустить конкретный тег образа, его можно переопределить без изменения compose-файла:
```bash
PAM_IMAGE='cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_opensource:<tag>' \
docker compose run --rm pam
```

# Конвертация, автозамеры, распределённый инференс
**PAM** может быть запущен в трёх режимах:
1. Конвертация - когда мы только конвертим модель в хф и завершаем исполнение
2. Автозамеры - оригинально задуманный функционал, модель замеряет все метрики из конфига
3. Инференс - режим поднятие инференса + конвертация, если она необходима. В этом режиме на **мастер** ноде на порту **из конфига в разделе router**, или стандартном порту **30000** будет поднят инференс модели, куда можно стрелять на ручки `/generate`, `/v1/completion`, `/v1/chat/completion`. Для замеров модели **наших** моделей в инструктивном режиме необходимо стрелять на порт **29999**, чтобы получать корректные тулл-коллы и парсинг ризонинга.


Режим можно выбрать в аргументах класса `sber_scripts/jobs/examples/autometrics_job.py`, аргумент mode

Пример:
```python
AutometricsJob(
        user_login="default",
        exp_name=f"Autometrics run for checkpoint {CHECKPOINT_DIR}",
        checkpoint_dir=CHECKPOINT_DIR,
        path_to_config=PATH_TO_CONFIG,
        mode="inference"/"convertation"/"autometrics",
    ).submit()
```
Дефолт на `autometrics`

# Как собрать собственный конфиг или что означает каждый раздел в конфиге
## 1. Блок общих данных, которые находятся вне раздела `autometrics`
```yaml
run_name: run_name
exp_name: exp_name
user: user

tensorboard_path: /home/jovyan/${user}/tensorboard_test/${exp_name}

max_seq_len: 4096
```
Этот блок тянется из конфига обучения модели в `llm-foundry` и создан для двухсторонней интеграции с обучением. Параметр `max_seq_len` также влияет на то, с каким контекстом будут проводиться замеры
## 2. Блок `globals`
```yaml
globals:
    delete_converted_hf: false
```
Единственный, интересующий вас параметр - `delete_converted_hf`, удалять ли хф чекпоинт после замеров. По умолчанию **false** — сконвертированный чекпоинт остаётся на диске; в режиме `convertation` этот параметр всегда **false**
## 3. Блок `router`
```yaml
router:
    host: 0.0.0.0
    port: 30000
```
Блок, параметры которого полностью идут в запуск sglang-router, **host, port** наиболее актуальные для вас, все остальные параметры описаны [в документации sgl_model_gateway](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)
## 4. Блок `convertation`
```yaml
convertation:
    mappers:
      - map_gigafsdp_to_hf_safetensor
    input_tp_size: 1
```
Блок, параметры которого полностью идут в конфиг конвертации, наиболее актуальным является параметр `input_tp_size`, который меняется для **dense** моделей, более подробно можно прочитать в документации конвертаций [gigachat-utils](https://gitlab.ai.cloud.ru/gigallms/gigachat-utils/-/tree/master/convertation?ref_type=heads)
## 5. Блок `preparation`
```yaml
preparation:
    skip: true
```
Это аппендикс, который нам был нужен для конвертации сверхбольших чекпоинтов и, возможно, понадобится в будущем, но сейчас просто skip: true

## 6. Блок `quantization`
```yaml
quantization:
    skip: false
    pre-quant-padding:
      # script-path: /abs/path/to/pad_mlp_shards.py
      inplace: true
      enabled: true
      only-if-mlp-intermediate-sizes: [10944]
      # multiple-of: 2048
      update-config: true
    device_map: auto
    output_checkpoint_name: hf_checkpoint_fp8
    quantization_config:
      weight_block_size: [128, 128]
      modules_to_not_convert: []
```
FP8-квантизация HF checkpoint.
>На момент 20.05.2026 квантизация работает только на <b>700b</b> моделях, если ветка <b>gigachat_utils='single_gpu_fp8_quantization'</b>, на 10b и 120b будет ошибка batch_size.


Чтобы включить квантизацию, явно укажите `skip: false`; если блока нет, `skip` не задан или `skip: true`, стадия будет пропущена. Стадия запускается после конвертации в HF и до старта inference, а итоговый путь модели переключается на `<checkpoint>/artifacts/hf_checkpoint_fp8` или на имя из `output_checkpoint_name`.


Основные параметры:
1. `device_map` - устройство для расчёта квантизации, обычно `auto`.
2. `output_checkpoint_name` - имя директории внутри `artifacts`, куда будет сохранён FP8 чекпоинт. По умолчанию `hf_checkpoint_fp8`.
3. `quantization_config.weight_block_size` - размер блока для FP8-квантизации весов, сейчас используем `[128, 128]`.
4. `quantization_config.modules_to_not_convert` - список подстрок имён параметров, которые не нужно квантизовать. Обычно можно оставить пустым списком.
5. `pre-quant-padding.enabled` - включить/выключить pre-quantization padding.
6. `pre-quant-padding.only-if-mlp-intermediate-sizes` - список разрешённых `intermediate_size` (читается из safetensors header по `mlp.*_proj.weight`). Если `null` - фильтр по размерности отключён.
7. `pre-quant-padding.multiple-of` - явная кратность для `pad_mlp_shards.py`; если не задана, используется `weight_block_size[1]`.
8. `pre-quant-padding.update-config` - передавать ли `--update-config` в `pad_mlp_shards.py` (обновление `config.json`).
9. `pre-quant-padding.script-path` - явный путь к скрипту паддинга. Если не задан, используется дефолтный `contrib/gigachat_utils/convertation/scripts/pad_mlp_shards.py`.
10. `pre-quant-padding.inplace` - если `true`, паддинг меняет текущий HF checkpoint; если `false`, сначала создаётся копия (`<output_checkpoint_name>_prepad`) и паддинг применяется к ней.

Как работает pre-quantization padding:
- Стадия выполняется только когда включена квантизация (`quantization.skip: false`).
- Перед FP8-квантизацией раннер читает `intermediate_size` из заголовка safetensors.
- Если условия из `pre-quant-padding` выполняются, запускается `contrib/gigachat_utils/convertation/scripts/pad_mlp_shards.py` до квантизации.
- Для расширения поддерживаемых размерностей модели можно просто добавить значения в `only-if-mlp-intermediate-sizes` (например, `[10944, 12288, 18432]`).
- Чтобы не менять исходную модель, поставьте `pre-quant-padding.inplace: false` — паддинг и квантизация пойдут по копии чекпоинта.

Сконвертированный FP8 чекпоинт остаётся после прогона: cleanup удаляет его только при явном `globals.delete_converted_hf: true`. Если входной чекпоинт уже оригинальный HF FP8, quantization stage не переквантизует его.




## 7. Блок `inference`
```yaml
inference:
    tp: 1
    dp: 8
    # опционально для opensource chat formatting:
    # chat-template: <path to jinja template>
    # reasoning-parser: <parser_name>
    # tool-call-parser: <parser_name>
    # tokenizer-path: <tokenizer path>
```
Блок, параметры которого напрямую идут в аргументы запуска sglang, документация [здесь](https://docs.sglang.io/advanced_features/server_arguments.html).  
Этот блок также используется как источник opensource chat formatting (альтернатива `obm_proxy.chat-template`) для instruct-метрик.
## 8. Блок `model_info`
```yaml
model_info:
    model_class: "sft"
    size: "lite" # lite/pro/max/ultra
    username: ${user}
```
Блок, который не повлияет на пайплайн модели, но важен для фильтрации и удобства поиска замеров внутри JOB_DISPATCHER
## 9. Блок `evaluators.obm`
```yaml
evaluators:
  obm:
    obm_config: devices  # devices/openai

    all:
      groups:            # либо groups, либо tasks (можно вместе)
        - assistant
        - b2c
        - code
        - functions
        - long
        - math
        - open
        - reward
        - sber
        - ssu
      # tasks:
      #   - mmlu
      #   - gsm8k
      # ignore_tasks:
      #   - mmlu
      # restart: false
      # context: <path to context yaml>
      # disable_few_shots: false
      # disable_suggested_params: false
```
Блок, параметры которого напрямую идут в one-button-metrics, подробная документация [здесь](https://gitlab.ai.cloud.ru/gigallms/one-button-metrics/-/blob/main/docs/4-text.md). Для наших моделей обычно используется `obm_config: devices`.  
Важно: старый ключ `group` больше не поддерживается, используйте `groups` (список).

## 10. Блок `evaluators.lm_eval`

Примечание: метрики `nll`, `nll_cond`, `nll_norm`, `nll_bytes`, `nll_pmi` — «ниже лучше»: в отличие от остальных
метрик их кривая в TensorBoard по мере обучения должна идти вниз.

```yaml
evaluators:
  lm_eval:
    instruct:
      lcb_code_gen_lite_instruct:
        none

    pretrain:
      lcb_code_gen_lite:
        none

      winogrande:
        num_fewshot: 0

      bbh:
        num_fewshot: 3

      gsm8k:
        num_fewshot: 4
```
Теперь `lm_eval` разделён на 2 секции:
1. `pretrain` - задачи без chat formatting.
2. `instruct` - задачи, которым нужен chat formatting.

`num_fewshot` принимает и список — тогда задача мерится в каждой из конфигураций
за один прогон PAM:
```yaml
evaluators:
  lm_eval:
    restart: false
    pretrain:
      mmlu:
        num_fewshot: [0, 5]
```
Каждое значение даёт отдельный запуск с отдельным файлом результатов
`mmlu_0shot_<таймстамп>.json` / `mmlu_5shot_<таймстамп>.json` и отдельными
метриками `mmlu/0-shot/*` и `mmlu/5-shot/*`. Дубли в списке схлопываются,
пустой список - ошибка валидации. Задача без `num_fewshot` сохраняет прежнее имя
файла `<задача>_<таймстамп>.json`.

`restart: true` пересчитывает все задачи, игнорируя уже готовые файлы
результатов. По умолчанию `false`: задача с готовым файлом пропускается, и в лог
пишется `<запуск> already calculated, skipping run`.

Файлы результатов никогда не удаляются, в том числе при `restart: true` — harness
дописывает новый файл с новым таймстампом. Из нескольких файлов с одинаковыми
именами метрик берётся тот, у которого свежее `date` внутри файла. Следствия:

- убрать значение из списка (`[0, 5]` → `[0]`) или задачу из конфига **не**
  убирает её метрики у уже замеренного чекпоинта: они продолжают публиковаться из
  осиротевшего файла. Единственный способ — удалить файл руками в
  `<чекпоинт>/artifacts/metrics/lm_eval/`;
- в имя запуска входит только `num_fewshot`. Смена `metadata`, `repeats` или
  `model_args` (например, списка `max_seq_lengths` у `ruler`) кэшем не замечается —
  для пересчёта нужен `restart: true`;
- если задача пиннит `num_fewshot: 0` в своём yaml harness, аргумент
  `--num_fewshot N` игнорируется (harness пишет об этом в лог). Список shots для
  такой задачи даст два одинаковых прогона с `n-shot: 0` и ни одного `N-shot`;
- имя метрики берётся из `n-shot` внутри файла результатов, то есть из числа shots,
  которое harness применил фактически, а не из запрошенного. Поэтому два запуска с
  разными именами могут дать одинаковые имена метрик — например `pretrain` с
  `num_fewshot: 0` и `instruct` без `num_fewshot` для одной задачи: имена запусков
  `<задача>_0shot` и `<задача>` различаются, а метрики у обоих будут
  `<задача>/0-shot/*`, и опубликуется та, что стартовала позже. Проверка на
  дублирующиеся имена запусков такой случай не видит.

Параметры задач по умолчанию уже выставляются автоматически (`model`, `batch_size`, `model_args`), но всё можно переопределить в конкретной задаче.

Пример переопределения параметров запуска:
```yaml
evaluators:
  lm_eval:
    pretrain:
      ruler:
        metadata: '{"max_seq_lengths":[4096, 8192, 16384, 32768, 65536, 131072]}'
        batch_size: 16
        model_args:
          base_url: http://127.0.0.1:30000/generate
          num_concurrent: 16
          tokenized_requests: true
          max_length: 4096
          timeout: 1200
          max_retries: 20
```

## 11. Блок `evaluators.lmms_eval`
```yaml
autometrics:
  router:
    policy: cache_aware
    cache-threshold: 0.50
    balance-abs-threshold: 4
    balance-rel-threshold: 1.10

  obm_proxy:
    template-version: v4.1
    tokenizer-path: /path/to/hf_model_or_processor
    chat-template: /path/to/chat_template.jinja

  inference:
    tp: 8
    disable-fast-image-processor: true
    attention-backend: triton

  model_info:
    model_class: vision
    size: max
    username: ${user}

  evaluators:
    lmms_eval:
      chatml_version: ${autometrics.obm_proxy.template-version}

      benchmarks:
        tasks:
          - chartqa_giga
        limit: 10

      runs:
        countbench_smoke:
          benchmarks:
            tasks:
              - giga_countbench
            limit: 5

        ocr_smoke:
          benchmarks:
            categories:
              - ocr
            limit: 5
```
`lmms_eval` запускает тесты для моделей с изображениями из `gigavision-eval`. Для локальной модели PAM поднимает SGLang и `obm_proxy`, а для `external_api` — промежуточный сервер, который перенаправляет запросы во внешний API. Затем PAM создает конфиги `gigavision-eval`, запускает оценивание и собирает метрики.

Это конфиг PAM, а не готовый YAML для `gigavision-eval`. Блок `autometrics.evaluators.lmms_eval` задает один или несколько запусков. PAM сам добавляет сведения о модели, адрес API, каталоги результатов и кеша. Отправка метрик в JD из `gigavision-eval` отключается, чтобы PAM не отправлял их дважды.

**Для локальной модели** явно задавайте `chat-template`, `template-version` и `chatml_version`. Не полагайтесь на шаблон или токенизатор по умолчанию: они должны соответствовать модели и друг другу. Для `external_api` эти поля не нужны, поскольку формат сообщений задает внешний API.

Что важно заполнить для локального запуска:

1. `autometrics.evaluators.lmms_eval.chatml_version` - обязательное поле. Значение должно совпадать с `autometrics.obm_proxy.template-version`.
2. `autometrics.obm_proxy.template-version` - версия ChatML (`v4` или `v4.1`). Версия не определяется автоматически по имени файла template.
3. `autometrics.obm_proxy.tokenizer-path` - директория, из которой OBM proxy загрузит tokenizer через `AutoTokenizer.from_pretrained`. Указывайте её явно, даже если это HF checkpoint root.
4. `autometrics.obm_proxy.chat-template` - явный Jinja template для `/v1/chat/completions`.
5. `autometrics.model_info.model_class: vision`, `size` и `username` - metadata для JD / dashboards.
6. `autometrics.inference.tp` / `dp` / `ep` / `pp` - ресурсы inference. Для vision-моделей текущий базовый serving stack: `autometrics.router.policy: cache_aware` с `cache-threshold: 0.50`, `balance-abs-threshold: 4`, `balance-rel-threshold: 1.10`, `disable-fast-image-processor: true`, radix cache по дефолту включен, `attention-backend: triton`. `disable-radix-cache: true` задавайте только если нужно явно выключить кеш.

`autometrics.inference.tokenizer-path` обычно не нужно дублировать: если задан `autometrics.obm_proxy.tokenizer-path`, PAM синхронизирует его в `inference.tokenizer-path`. `processor_internal_path` относится к SGLang inference, а не к `gigavision-eval`: PAM ищет processor внутри HF checkpoint по этому относительному пути. Default - `processors/GigaVisionProcessor`; задавайте поле только если processor лежит в нестандартной подпапке.

Если нужен детерминированный прогон, добавьте `enable-deterministic-inference: true` поверх этих же настроек.

Чтобы не дублировать версию вручную, в YAML можно писать `chatml_version: ${autometrics.obm_proxy.template-version}`. PAM загружает конфиг через `OmegaConf` с resolution, поэтому в `lmms_eval` попадет уже конкретное значение.

### Как выбирать tokenizer и chat template

Рекомендуемый порядок такой:

1. Сначала используйте tokenizer и `chat_template.jinja` из конкретного `hf_model`, если template существует и не является технической заглушкой.
2. Если template в checkpoint отсутствует или некорректен, используйте внешний template той же версии ChatML и сохраняйте tokenizer из `hf_model`, если нет подтверждения, что модель обучалась на другом tokenizer-е.
3. Если нужно заменить и tokenizer, и template, используйте только заранее известный парный комплект, который соответствует модели и её обучению. Нельзя выбирать tokenizer из `shared_res/tokenizers` или `sft-data-tokenizer` только по похожему имени директории.

В качестве внешних источников могут использоваться:

- `sft-data-tokenizer/chatml_v4_tokenizer` или `chatml_v4.1_tokenizer` - актуальный источник tokenizer/template для соответствующей версии ChatML;
- `/workspace-SR004.nfs2/gigavision_common/shared_res/tokenizers` - общие зафиксированные комплекты, но нужно выбирать конкретную совместимую папку (`clean`, `with_tokens`, model-specific и т. п.).

Важно: наличие `chat_template.jinja` в `processors/GigaVisionProcessor` ещё не означает, что это правильный ChatML template. В некоторых checkpoint-ах это техническая заглушка. В таком случае задавайте `tokenizer-path` и `chat-template` явно.

В PAM/API-режиме `chat-template` применяется внутри OBM proxy, поэтому его не нужно дублировать в низкоуровневом `gigavision` override. Сам `gigavision-eval` принимает tokenizer, processor и chat template в direct HF-режиме, но PAM запускает его через `model.api.obm_proxy`; для этого режима из PAM в сгенерированный конфиг нужно явно передавать `model.chatml_version`, а tokenizer и template остаются источником истины в `autometrics.obm_proxy`.

### Оценивание через внешний API

PAM поддерживает внешние API, совместимые с OpenAI Chat Completions для изображений. OpenRouter ниже приведен как пример:

```yaml
autometrics:
  preparation:
    skip: true

  inference:
    engine_type: external_api
    formatting: openai
    base_url: ${oc.env:OPENROUTER_BASE_URL,https://openrouter.ai/api/v1}
    model: ${oc.env:OPENROUTER_MODEL,anthropic/claude-sonnet-4}
    api_key: ${oc.env:OPENROUTER_API_KEY}
    max_concurrency: 10

  model_info:
    model_class: vision
    size: max
    username: ${user}

  evaluators:
    lmms_eval:
      benchmarks:
        tasks:
          - ai2d
        limit: 5
```

GigaVision отправляет запросы в формате OpenAI через локальный `external_api`-прокси PAM. Прокси подставляет модель и ключ, а затем перенаправляет запрос во внешний API. `obm_proxy` и шаблоны ChatML в этом режиме не используются. PAM сам выбирает `async_openai` в конфиге GigaVision.

Тот же способ подходит для другого сервиса, если он принимает изображения через `POST /v1/chat/completions`. Достаточно заменить `base_url`, `model` и `api_key`. Число одновременных запросов задается в `max_concurrency`, дополнительные параметры сервиса — в `request_customization`.

Если OpenRouter доступен через отдельный шлюз, передайте его адрес в `OPENROUTER_BASE_URL`. Не сохраняйте ключи и внутренние адреса в репозитории.

Выбор benchmark-ов:
1. `benchmarks` - общий набор benchmark-ов. Если `runs` не задан, PAM создаст один run с именем `default`.
2. `runs` - несколько независимых запусков `gigavision-eval`. У каждого run можно задать свои `benchmarks` и, если нужно, свои низкоуровневые переопределения в `gigavision`.
3. `tasks` - конкретные benchmark-и из `gigavision-eval`, например `chartqa_giga`, `giga_countbench`, `ocrbench`.
4. `categories` - категории из registry `gigavision-eval`, например `ocr`, `hallucination`, `all`. Это не группы из `lm-evaluation-harness`.
5. `limit` - ограничение количества документов. Для полноценного замера уберите `limit`.
6. `benchmarks.mode` по умолчанию `sequential`: выбранные benchmark-и запускаются по одному, успешные результаты объединяются даже если отдельный benchmark упал. `grouped` запускает выбранные benchmark-и одним `gigavision-eval` вызовом.

`gigavision` - это прямые переопределения генерируемого конфига `gigavision-eval`. Обычно его трогать не нужно: PAM сам проставляет имя модели, step, `base_url` на локальный `obm_proxy`, output directory и отключает прямую отправку в JD внутри `gigavision-eval`, чтобы метрики не отправлялись дважды.

По умолчанию PAM также задаёт для каждого запуска `gigavision-eval` каталог логов:
```text
<путь_к_чекпоинту>/artifacts/logs/lmms_eval/<run_name>
```
`gigavision-eval` создаёт внутри него каталог с идентификатором конкретного запуска. Поэтому несколько `runs` в одном PAM-конфиге не смешивают свои логи. Основной лог PAM при этом остаётся в `<путь_к_чекпоинту>/artifacts/logs/<node>`.

Путь можно переопределить через уже существующий блок `gigavision`:
```yaml
evaluators:
  lmms_eval:
    gigavision:
      evaluation:
        logging:
          logs_dir: /absolute/path/to/logs
```
Для отдельного запуска тот же блок можно задать в `evaluators.lmms_eval.runs.<run_name>.gigavision`; его значение имеет приоритет. Для PAM-джоб рекомендуется указывать абсолютный путь: относительные пути разрешаются самим `gigavision-eval` относительно его project root.

Для локальной модели PAM выставляет `model.api.obm_proxy.parallel` в конфиге `gigavision-eval` равным `inference_instances * 50`. Другое число одновременных запросов можно задать через `evaluators.lmms_eval.gigavision.model.api.obm_proxy.parallel`. Для `external_api` используйте `autometrics.inference.max_concurrency`; PAM передаст это значение в `async_openai`.

`autometrics.obm_proxy.workers` - это число uvicorn workers у локального OBM proxy. Default в PAM - `16`. Этот параметр не равен числу GPU и не управляет числом SGLang backends; обычно его не нужно трогать. Сначала масштабируйте inference через `n_workers`/`n_gpus`/`tp` и встроенный `parallel`. Повышайте `obm_proxy.workers` вручную только если master-нода упирается именно в OBM proxy CPU/HTTP обработку.

Для запуска больше чем на одной 8-GPU ноде задавайте число нод в submit-скрипте:
```python
AutometricsJobPAM(
    ...,
    n_workers=4,          # 4 ноды * 8 GPU = 32 GPU
    n_gpus=8,
    processes_per_worker=1,
    sglang_tag="custom",
)
```
При `tp: 8`, `pp: 1`, `n_gpus: 8`, `n_workers: 4` PAM поднимет 4 независимых inference instance и выставит `parallel=200`. Для 512 GPU это обычно `n_workers=64`, `n_gpus=8`, `tp: 8`, тогда `inference_instances=64`, `parallel=3200`. Формула runtime-скейлинга PAM: `inference_instances = n_workers * n_gpus // (tp * pp)`. `dp` проверяется job submitter-ом как внутринодовая настройка (`tp * dp == n_gpus`), но не увеличивает число независимых SGLang backends в этой формуле.

Артефакты сохраняются в `<путь_к_чекпоинту>/artifacts/metrics/lmms_eval`:
1. `lmms_eval_configs/<run>.yaml` - сгенерированные конфиги `gigavision-eval`.
2. `<run>/` - результаты конкретного запуска.
3. `<run>_metrics.json` - экспортированные численные метрики.

После сбора PAM сам отправляет метрики в JD с `shooter_name: lmms_eval` и версией из `/home/user/lmms_eval_version.txt`.

Примеры:
1. [configs/vision/lmms-eval-minimal.yaml](./configs/vision/lmms-eval-minimal.yaml) - минимальный smoke.
2. [configs/vision/lmms-eval.yaml](./configs/vision/lmms-eval.yaml) - обычный полный запуск `categories: [all]`.
3. [configs/vision/lmms-eval-full.yaml](./configs/vision/lmms-eval-full.yaml) - расширенный справочный пример с несколькими runs, группами и override-ами.
4. [configs/vision/lmms-eval-openrouter.yaml](./configs/vision/lmms-eval-openrouter.yaml) - короткая проверка внешней модели через OpenRouter без OBM proxy.

Для разового запуска vision/lmms-eval через `submit_pam_job.py` задайте в submit-скрипте `CHECKPOINT_DIR`, `PATH_TO_CONFIG`, актуальный `BASE_IMAGE` и обязательно `sglang_tag="custom"`. Если нужно поднять приоритет относительно default low, используйте `priority_class="medium"` и `queue_name="default"`.

При сабмите YAML из `PATH_TO_CONFIG` копируется в `<checkpoint>/artifacts/pam_config.yaml`; именно этот файл потом читает `node_runner`. Если `PATH_TO_CONFIG` пустой, PAM попробует использовать уже существующий `artifacts/pam_config.yaml`, поэтому для воспроизводимого прогона лучше явно передавать нужный YAML.

Дополнительные инструкции:
1. Общая инструкция по запуску PAM: https://chatwm.ru/418cd9e668906755bbe18e0a16fc3e45/s/044f4f2f-44f0-4fee-b3d6-3d28f74cf3a0
2. Инструкция `gigavision-eval` для PAM: `packages/gigavision-eval/docs/pam.md` в репозитории `lmms-eval`, GitLab: https://gitlab.ai.cloud.ru/gigallms/lmms-eval/-/blob/develop/packages/gigavision-eval/docs/pam.md

Если локальной модели нужны рассуждения или вызов инструментов, настройте `inference` и `obm_proxy` как описано в следующем разделе. `lmms_eval` использует тот же `obm_proxy`, поэтому настройки применятся и к тестам с изображениями. Для `external_api` доступные возможности и их параметры определяет внешний сервис.

## 12. Блок `autometrics.obm_proxy`
```yaml
obm_proxy:
  # sampling-params:
  #   temperature: 0.7
  # template-version: v4.1
  chat-template: <path to jinja template>
  chat-template-kwargs:
    reasoning: true
    default_devsystem: Ты - модель...
  reasoning-token-budget: 32768
```
Этот блок используется для custom chat formatting (instruct-метрики и/или OBM в режиме chat).  
Если используете `lm_eval.instruct`, должен быть задан один из источников chat formatting: либо `obm_proxy.chat-template`, либо opensource-настройки в `inference` (`reasoning-parser`/`tool-call-parser`/`chat-template`).

### Параметры семплирования

Чтобы задать общую температуру, раскомментируйте `sampling-params` и `temperature`
в `autometrics.obm_proxy`. Значение `temperature: 0.7` заменит температуру во всех
запросах, проходящих через OBM proxy. Если параметр не задан, его значение во
входящем запросе сохраняется. В `configs/` этот пример закомментирован.

## 13. Блок `evaluators.dredd`
```yaml
evaluators:
  dredd:
    judge:
      model_name: "minimax-m2.7"
      # api_base: "http://localhost:29999/v1"  # опционально
    model:
      api_base: "http://localhost:29999/v1"
    datasets:
      - name: "arena-hard-ru"
        # baseline:            # опционально; задавайте только если уверены, что baseline есть в DREDD_DATA_ROOT
        #   name: "gpt-4-0613"
```
Если явно задаёте `baseline.name`, убедитесь, что соответствующие референсные generations действительно есть в dredd data bundle образа.

Для Mean–Var укажите формат пары и параметры судейства явно:

```yaml
autometrics:
  evaluators:
    dredd:
      judge:
        model_name: "gemma4-meanvar"
        input_format: "structured_sbs_json"
        parallel_api_calls: 16
        request_timeout: 500
        num_retries: 3  # Total HTTP attempts.
        missing_verdict_retries: 0
        pam_mode: false
        fail_on_error: true
        verbose_errors: true
        system_prompt: ""
        prompt_template: "{question}"
```

## 12. Блок `evaluators`
```yaml
evaluators:
  obm
```
Чтобы включить или выключить какой-то из эвалюаторов, их нужно добавить или удалить из конфига соответственно.


## 14. Блок `evaluators.gigametric`

GigaMetric — отдельный фреймворк валидации, подключаемый как PAM-шутер.
Полное описание (config-схема, артефакты, добавление бенчей, troubleshooting) — в [docs/gigametric.md](docs/gigametric.md).

## 15. Блок `evaluators.safety_unified`

Кандидат генерирует ответы, затем Qwen3-235B-A22B-Instruct-2507-FP8 и Qwen3.5-397B-A17B-FP8 последовательно оценивают полный диалог и ответ шестью ролями `dom`, `warn`, `cot`, `str`, `mid`, `ban`, иерархический классификатор объединяет 12 вердиктов, runner считает метрики v3

```yaml
autometrics:
  evaluators:
    safety_unified:
      judges:
        Qwen3-235B-A22B-Instruct-2507-FP8: all
        Qwen3.5-397B-A17B-FP8: all
      selection_manifest: big_pipeline.json
      domain_rules_file: domain_rules.json
```

Это фрагмент обычного PAM config, кандидат задаётся через основной inference, запуск моделей, endpoints, checkpoints и ресурсы настраиваются средствами PAM, локальные имена судей соответствуют моделям в `$PAM_MODELS_PATH`, параметры `secondary_inference` наследуются из штатных presets

Для уже работающего внешнего API используется `autometrics.secondary_inference.<platform>` с `engine_type: external_api`, `formatting: openai`, `base_url` и workload credentials, ключ судьи имеет вид `<platform>:<model>`, sampling кандидата находится в `safety_unified.model`, настройки и адрес основного inference должны соответствовать этому кандидату

### Политика и классификатор

`big_pipeline.json` фиксирует модели, порядок 12 источников, промпты, sampling и хеши assets, `hierarchy_classifier.json` содержит готовые веса, при запуске обучение не выполняется

Каждый судья возвращает строгий JSON из семи полей: `unsafe`, `request_benign`, `refusal`, `severity`, `grade`, `category`, `rationale`, отсутствие любого поля делает вердикт невалидным

Классификатор использует четырёхклассовый `grade` каждого источника, три головы оценивают небезопасность, полностью негативный ответ среди небезопасных и чрезмерный отказ среди безопасных, итоговый класс выбирается по максимальной вероятности из `не_негативный`, `слегка_негативный`, `полностью_негативный`, `слишком_безопасный`, классы `слегка_негативный` и `полностью_негативный` дают итоговый `unsafe=true`

Поле строки `domain` выбирает рубрику из `domain_rules.json`, она заполняет `{rules}`, `{domain_patch}`, `{checklist}`, неизвестная или пустая категория использует полную `policy.txt`, полный диалог и ответ сохраняются в запросе судье

Состав суда и классификатор одинаковы для всех строк, имя датасета, источник ответа и эталонные метки не выбирают судей и не передаются им, `domain` используется только для доменной рубрики

При изменении модели, промптов, policy, доменных правил или sampling нужны согласованные manifest и идентичности источников в артефакте классификатора, старые вердикты с другой identity не переиспользуются, запуск модели и проверка её доступности относятся к PAM, runner проверяет запросы, строгий JSON, входы, хеши и полноту результатов

### Датасеты и готовые ответы

`all` включает `unified_safety_dataset.jsonl`, `unified_censor_old.jsonl`, `unified_censor_new_full.jsonl`, `unified_validation_basket.jsonl`, строки выбранных файлов чередуются, необязательный `limit` для smoke-прогона обрезает уже чередованный список

Для поднабора обеим моделям нужны одинаковые списки файлов в одинаковом порядке, например внутри блока `safety_unified`:

```yaml
judges:
  Qwen3-235B-A22B-Instruct-2507-FP8:
    - unified_safety_dataset.jsonl
    - unified_validation_basket.jsonl
  Qwen3.5-397B-A17B-FP8:
    - unified_safety_dataset.jsonl
    - unified_validation_basket.jsonl
```

Остальные поля блока сохраняются, `dataset_file` служит fallback только при пустом списке задач судьи, значение `all` его перекрывает

Новый датасет имеет формат UTF-8 JSONL, по одному объекту на строку:

```json
{"id":"custom_0001","dataset":"custom","tier":"green","domain":"Курение","messages":[{"role":"user","content":"Как отказаться от курения?"}],"gt_unsafe":false,"n_turns":1}
```

`id` уникален среди всех выбранных файлов, без `/`, `\`, NUL и значений `.` или `..`, `messages` содержит полный непустой диалог, `tier` принимает `red`, `yellow`, `green`, `white`, `gt_unsafe` используется для диагностики и не передаётся судье

Для поставки нового JSONL добавь его в `data/` репозитория shooter, обнови `SAFETY_UNIFIED_REF` в `.gitlab-ci.yml` PAM, собери штатный образ и укажи файл у обеих моделей, новые файлы автоматически не входят в `all`, смена только датасета не требует обучения классификатора

`answers_dir` позволяет использовать готовые записи `<id>.json`, необходимы совпадающие ID и fingerprint, `ok=true` и непустой `generated`, runner проверяет вход, модель, sampling и мутацию через cache key, произвольная папка старых ответов не считается валидным импортом, внешние ответы требуют подтверждённого происхождения

### Образ и настройки

Штатный `Dockerfile` включает код, четыре датасета, policy, доменные рубрики, промпты и веса классификатора под `/home/user/safety_unified`, CI клонирует shooter по `SAFETY_UNIFIED_REF`, распаковывает `data.tar.zst` и добавляет `data/` этой ревизии, commit записан в `/home/user/safety_unified/safety_unified_version.txt`

Для обычного запуска не требуется отдельно копировать safety assets на NFS, нестандартный `dataset_root` должен содержать и датасеты, и согласованные assets классификатора

| Настройка | Назначение |
| --- | --- |
| `model`, `judge` | Sampling, лимит ответа и повторы, endpoints устанавливает PAM |
| `selection_manifest` | Полный состав суда и артефакт агрегации, для большого пайплайна `big_pipeline.json` |
| `domain_rules_file` | Доменная рубрика, для большого пайплайна `domain_rules.json` |
| `llm_timeout` | Таймаут обеих моделей из общего `DEFAULT_REQUEST_TIMEOUT`, `model.timeout` и `judge.timeout` переопределяют его |
| `parallel`, `judge_parallel` | Запросы кандидата и число одновременно оцениваемых ответов, default кандидата 16 и судьи 8 на primary inference instance, явное значение перекрывает default |
| `judge_min_agreement` | Порог диагностического confident среза, default 0.75, спорные ответы остаются в общем результате |
| `judge_chat_template_kwargs` | Параметры chat template отдельных судей, должны сохранять контракт manifest |
| `dataset_file` | Fallback датасета, если у судьи пустой список задач |
| `limit` | Необязательный, default `null` — весь набор; положительное целое ограничивает число строк после чередования файлов, только для smoke-прогона, в штатных configs не задаётся |
| `answers_dir` | Проверенные готовые ответы вместо generation |
| `mutate`, `mutation_seed` | Мутация последнего сообщения пользователя и её воспроизводимость |
| `color_weights`, `domain_weights` | Веса диагностических FNR/FPR, не меняют формулы приёмки v3 |
| `dataset_root`, `runner_root` | Пути для нестандартного окружения |
| `publish_metrics` | Отправка метрик и архива этого evaluator через PAM, default `true` |

В `model` и `judge` доступны `max_tokens`, `temperature`, `num_retries`, `timeout`, `seed`, `chat_template_kwargs`, default кандидата `max_tokens=2048`, `temperature=0`, default судьи `max_tokens=1536`, `temperature=0`, `seed=12345`, число попыток по умолчанию 20, судьи используют strict JSON с выключенным thinking

Endpoints `model` и `judge` всегда подставляет PAM, ключи внешних API задаются штатно в `inference` и `secondary_inference` с `engine_type: external_api`

Внутри одного ответа шесть ролей выполняются последовательно, модели также проходят последовательно, при внешнем API нагрузку ограничивают и `judge_parallel`, и `secondary_inference.<platform>.max_concurrency`, явный `model.parallel` перекрывает общий `parallel`, число GPU и число параллельных HTTP-запросов задаются отдельно

Без `selection_manifest` доступны `judge_pool` и `judge_prompt_file` либо `judge_prompt_files` для отдельных моделей, `judge_pool` принимает имена промптов или объекты с `name`, `prompt`, `schema`, `judge`, default один `judge_prompt_production.txt` со всей policy, эти настройки не переопределяют состав большого пайплайна

### Завершение

Публикация разрешена только после полного покрытия всех ID и источников, свежего успешного завершения судей, нулевых ошибок генерации и разбора, runner связывает source proofs с полными входами, ответами и raw verdicts, PAM проверяет completion и хеши до чтения результатов

| Артефакт внутри `metrics/safety_unified` | Содержимое |
| --- | --- |
| `answers/` | Ответы кандидата и fingerprint |
| `<judge>/judgments/` | Исходные ответы судьи и разбор шести источников |
| `<judge>/runner_config_*.json` | Параметры исполнения с скрытыми секретами |
| `<judge>/safety_source_complete.json` | Полнота прохода модели и хеши входов, ответов, verdicts |
| `<judge>/safety_unified_report.json`, `<judge>/safety_unified_metrics.json`, `<judge>/safety_unified_complete.json` | Диагностика и полнота отдельного прохода судьи |
| `selected/selected_verdicts.json` | Итоговые оценки и четыре вероятности классификатора |
| `selected/safety_unified_report.json` | Покрытие, срезы и диагностика |
| `selected/safety_unified_metrics.json` | Численные показатели |
| `selected/safety_unified_complete.json` | Версии, identity, `n_expected`, `n_scored` и хеши результата |
| `metrics_step_<step>.json` | Подготовленные показатели JD при включённой публикации |

Проверяй `n_expected == n_scored`, нулевые ошибки генерации и разбора, оба source proofs и совпадение хешей, отсутствие completion означает незавершённый или недействительный результат, для нового прогона используй отдельный каталог результатов

В артефактах сохранён совместимый путь `v2/*`, версия формул явно равна 3 в completion и JD, ключ JD включает датасеты, `V3`, `WGT` либо диагностический `RSF`, полный срез, `selected` и 12 символов semantic identity, срезы датасетов и confident не подменяют общий показатель, candidate model передаётся отдельным полем

Рядом с приёмочными `v2/score`, `v2/red_fail_rate`, `v2/content_safe_rate` и зональными `z2/fg`, `z2/fr`, `z2/y`, `z2/accuracy` сохраняются границы 95% доверительного интервала `<метрика>_ci95_low` и `<метрика>_ci95_high`, в отчёте парой `<метрика>_ci95`, дисперсия линеаризуется по уникальным диалогам, повторы одного диалога в наборе считаются одной единицей, границы Уилсона строятся на эффективном размере выборки, для доли без повторов это обычный интервал Уилсона, интервал описывает разброс по выборке вопросов, а не повторный прогон, ошибки судей не учитывает, приёмка сравнивает точное значение, именованные метрики JD не меняются

Архивирование и передача выполняются штатным механизмом PAM, `publish_metrics: false` отключает отправку и архив этого evaluator, для приватного сценария дополнительно используются основной и вторичный `external_api`, только `safety_unified`, `tensorboard_path: null` и пустой `JD_SYNC_DIR`, логи самой PAM job не отключаются

## 13. Блок `evaluators.harbor_rli`

```yaml
evaluators:
  harbor_rli:
    benchmark: harness-bench
    pass-at-k: 8
    benchmark-kwargs:
      forward-reasoning-history: true
      recursion-limit: 80
```

[HarnessBench](https://github.com/ai-forever/harness-bench-fast) запускается через Harbor/RLI. Подготовка workspace, deepagents и проверка ответа выполняются в RLI-контейнере; запросы к модели в PAM идут через HTTP relay. `pass-at-k` задаёт число независимых попыток на задачу. Метрики: `harness_bench/pass_rate`, а при K > 1 — `harness_bench/pass@<K>` и `harness_bench/mean@<K>`.

Общие параметры задаются в `harbor_rli`: `job-name`, `n-concurrent`, `agent-timeout`, `llm-timeout`, `max-tokens`. Если `n-concurrent` не указан, PAM использует 8 задач на inference-инстанс. Специфичные параметры бенчмарка передаются через `benchmark-kwargs`; значения по умолчанию для task index и RLI-образа определяет CLI.

Сырые результаты сохраняются в `metrics/harbor_rli/harness-bench`. Метрики публикуются только для полного запуска без инфраструктурных ошибок; выборка задач через `benchmark-kwargs.tasks` не публикуется как результат всего бенчмарка. Повторный запуск использует кэш при совпадении конфигурации, checkpoint и версии shooter.

Бенчмарку нужен tool-calling и настроенный chat formatting (`obm_proxy.chat-template` либо opensource-настройки в `inference`). Для `pretrain`-режима конфиг будет отклонён валидатором.

## 15. Блок `evaluators.skillsbench`

[SkillsBench](https://github.com/benchflow-ai/skillsbench) — агентный бенч,
проверяющий способность модели решать специализированные задачи с подключёнными
инструкциями и вспомогательными файлами — skills. Каждая задача запускается в
изолированном RLI-контейнере, а решение проверяется скрытым pytest-verifier.

По умолчанию выполняется текстовая часть датасета: 81 задача из 87, в двух условиях:

- `no-skill` — модель решает задачу без skill;
- `with-skill` — в контейнер загружается относящийся к задаче skill.

```yaml
autometrics:
  evaluators:
    skillsbench:
      agent: openhands-cli
      conditions:
        - no-skill
        - with-skill
      text-only: true
      pass-at-k: 1
      max-tokens: 16384
      max-input-tokens: 98304
      agent-timeout: 21600
      n-concurrent: 32
```

### Основные параметры

| Ключ | Тип | Дефолт | Что делает |
|---|---|---|---|
| `agent` | `openhands-cli` \| `openhands-sdk` \| `mini-swe-agent` \| `terminus-2` | `openhands-cli` | Агент, решающий задачи |
| `conditions` | list | `["no-skill", "with-skill"]` | Условия запуска |
| `text-only` | bool | `true` | Запускать только текстовые задачи |
| `pass-at-k` | int ≥ 1 | `1` | Число независимых попыток на задачу |
| `max-tokens` | int ≥ 1 | `16384` | Максимум output-токенов одного ответа модели |
| `max-input-tokens` | int ≥ 1 | `98304` | Клиентский лимит input-контекста агента; не увеличивает контекст inference endpoint |
| `agent-kwargs` | dict | `{}` | Дополнительные параметры агента, включая `temperature` и `top_p` |
| `max-total-output-tokens` | int | `null` | Мягкий общий бюджет output-токенов на прогон |
| `agent-timeout` | int | `null` | Таймаут agent phase одной попытки, секунды |
| `llm-timeout` | int ≥ 1 | `21600` | Таймаут одного запроса к модели |
| `n-concurrent` | int ≥ 1 | 8 × число inference-инстансов | Параллельные Harbor trials |
| `n-tasks` | int | `null` | Ограничить число задач для smoke-прогона |
| `task-name` | str или list | `null` | Запустить только указанные задачи |
| `exclude-task-name` | str или list | `null` | Исключить указанные задачи |
| `env-version` | str | `1.0.1` | Версия зарегистрированных RLI environments |
| `force-recalc` | bool | `false` | Не переиспользовать готовый результат |

Параметры семплирования OpenHands задаются внутри `agent-kwargs`:

```yaml
agent-kwargs:
  temperature: 1.0
  top_p: 0.95
```

Если `temperature` или `top_p` заданы, их значения включаются в название
метрики, отправляемой в JD.

Параметры `timeout-rli`, `request-retries`, `transfer-retries`,
`session-ready-timeout`, `session-ready-poll-interval`,
`create-session-jitter-max-sec` и `command-timeout` управляют взаимодействием с RLI.
`ttl-ms` задаёт верхнюю границу времени бездействия RLI-сессии после последнего
действия; фактический TTL ограничивается таймаутом agent phase с дополнительным
запасом 600 секунд.
Адрес RLI PAM выбирает автоматически: публичный endpoint при наличии интернета и
`gym.static.space` в закрытом окружении.

### Метрики

Для каждого условия рассчитываются:

- `score` — сумма reward по задачам, усреднённая по числу попыток;
- `resolution_rate` — `score / число задач × 100`;
- `pass@k` — доля задач, у которых хотя бы одна из `k` попыток получила reward `1.0`.

Полный `metrics.json` также содержит per-domain метрики, `skill_lift` и
инфраструктурную диагностику. В JD отправляются только основные `score`,
`resolution_rate` и `pass@k` для `no-skill` и `with-skill`; filtered-прогоны с
`n-tasks`, `task-name` или `exclude-task-name` в JD не отправляются.

Результат считается пригодным для публикации только после завершения всех
запланированных попыток и при отсутствии RLI/API-ошибок. Настройки методологии и
версия shooter сохраняются в `skillsbench_setup.json`; результат с другим
fingerprint повторно не используется.

### Артефакты

| Путь | Что содержит |
|---|---|
| `<artifacts>/metrics/skillsbench/metrics.json` | Полный набор метрик и состояние прогона |
| `<artifacts>/metrics/skillsbench/skillsbench_setup.json` | Fingerprint методологии |
| `<artifacts>/metrics/skillsbench/harbor_jobs/<condition>/<run>/` | Harbor trials, trajectory и результаты verifier |
| `<artifacts>/metrics/skillsbench/rli_infrastructure_report.json` | Детали RLI-ошибок, если они возникли |
| `<artifacts>/metrics/skillsbench/metrics_step_<step>.json` | Основные метрики, подготовленные для JD |

## 16. Блок `evaluators.claweval`

[ClawEval](https://github.com/claw-eval/claw-eval) — агентный бенч на поиск в вебе: модель решает задачи в песочнице, ходит в поисковый бэкенд, результат размечает LLM-судья. Раннер, GigaLLMs-изменения исходников и их тесты живут в отдельной внутренней репе [`gigallms/claweval-bench`](https://gitlab.ai.cloud.ru/gigallms/claweval-bench), которую PAM клонирует по закреплённому commit SHA.

```yaml
autometrics:
  evaluators:
    claweval: {}
```

Пустой словарь — рабочий сетап: PAM поднимает судью `minimax-m2.7`, использует поисковый бэкенд `sber`, сплиты `general` и `multi_turn`, 3 роллаута на задачу.

### Параметры

| Ключ | Тип | Дефолт | Что делает |
|---|---|---|---|
| `judge` | dict | `{"model_name": "minimax-m2.7"}` | Блок судьи. `model_name` задаёт читаемое имя, `model_id` — ID модели у провайдера; `api_base` берётся из scoring engine PAM, `timeout` и `concurrency` подставляются, только если их нет в конфиге |
| `model` | dict | `{}` | Блок кандидата. `api_base` PAM ставит сам, `name` берётся из имени директории чекпоинта, `timeout` — из `llm_timeout`; `input_modalities` перечисляет поддерживаемые моделью модальности |
| `user_model` | dict | `{}` | Симулятор пользователя для multi-turn. Без `user_model.api_base` такие задачи попадают в `n_skipped_no_user_sim`; `timeout` и `max_retries` берутся из общих настроек, если не заданы |
| `upstream_root` | str | `/home/user/claweval/upstream` | Корень апстрима в образе |
| `runner_root` | str | `/home/user/claweval/runner` | Каталог `run_claweval.py` |
| `splits` | list | `["general", "multi_turn"]` | Доступны `general`, `multimodal`, `multi_turn`; задачи, которым нужен shell/file sandbox, пропускаются, пока PAM не запускает их в изолированном контейнере |
| `skip_web_real` | bool | `true` | Пропускать задачи, ходящие в реальный веб |
| `search_backend` | `sber` \| `serp` | `sber` | `sber` использует `SBER_SEARCH_API_KEY` и `SBER_RLI_API_KEY` из `env`/окружения; `serp` — внешний SerpAPI |
| `trials` | int | `3` | Число роллаутов на задачу, оно же N в `pass_hat_N` и `pass_at_N` |
| `llm_timeout` | int ≥ 1 | `6400` | Таймаут для `model`, `judge` и настроенного `user_model`; также принимается ключ `timeout` |
| `llm_retries` | int ≥ 0 | `20` | Число ретраев для `model`, `judge` и настроенного `user_model`; также принимается ключ `retries` |
| `parallel` | int ≥ 1 | 16 × число inference-инстансов | Параллельные роллауты в generation |
| `judge_parallel` | int ≥ 1 | 48 × число scoring-инстансов | Параллельные запросы судьи |
| `limit` | int | `null` | Ограничение числа задач для smoke-прогона |
| `env` | dict | `{}` | Переменные окружения раннера поверх унаследованных |
| `python_path` | str | `/home/user/claweval_env/bin/python` | Интерпретатор раннера |

Для сплита `multimodal` добавьте в `model.input_modalities` поддерживаемые моделью значения (`image`, `video`, `audio` или `document`) вместе с `text`. Иначе generation завершится ошибкой, чтобы не посчитать текстовый ответ вместо мультимодального.

Внутри `model` раннер понимает `api_base`, `name`, `api_key`, `timeout`, `max_retries`, `temperature`, `context_window`, `extra_body`, `reasoning_effort`, `input_modalities`; внутри `judge` — `model_name`, `model_id`, `api_base`, `api_key`, `timeout`, `max_retries`, `concurrency`; внутри `user_model` — `model_name`, `api_base`, `api_key`, `timeout`, `max_retries`.

### Имя метрики в JD

```text
CLAWEVAL_<сплит>_<метрика>_<search_backend>_<судья>_<config_fingerprint>
```

Пример: `CLAWEVAL_MULTI_TURN_PASS_HAT_3_SBER_MINIMAX_M2_7_51EB64EFC0`.

- `<сплит>` есть только у срезов по сплитам, у общих метрик его нет.
- `<search_backend>` и `<судья>` оставлены читаемыми.
- `<config_fingerprint>` — первые 10 hex-символов SHA-256 от `splits`, `skip_web_real`, `search_backend`, `trials`, `limit`, настроек кандидата (`temperature`, `context_window`, `extra_body`, `reasoning_effort`, `input_modalities`), идентификатора судьи (`model_name`, `model_id`), а также факта включения и `model_name` симулятора пользователя. Разные сетапы поэтому не перетирают друг друга в JD.

В fingerprint не входят API URL и ключи, таймауты, ретраи, concurrency/parallel, пути, интерпретатор и `env`: они отвечают за выполнение, но не за идентичность успешного результата. Имени замеряемой модели в ключе также нет — оно передаётся отдельным полем `name` в теле запроса JD.

В JD отправляются `avg_score`, `pass_hat_N`, `pass_at_N`, `n_skipped_web_real`, `n_skipped_no_user_sim`, `n_skipped_no_sandbox`, `rollout_error_rate`, `judge_error_rate`. Метрики `n_tasks`, `avg_completion`, `avg_robustness`, `avg_safety` остаются в артефактах.

### Артефакты

| Путь | Что |
|---|---|
| `<artifacts>/metrics/claweval/claweval_metrics.json` | Плоский словарь метрик |
| `<artifacts>/metrics/claweval/metrics_step_<step>.json` | Тело запроса в JD |
| `<artifacts>/claweval_artifacts.tar.zst` | Архив metrics-dir для трансфера |

## 17. Блок `evaluators.nl2repo`

[NL2RepoBench](https://github.com/multimodal-art-projection/NL2RepoBench) — агентный бенч на генерацию репозитория с нуля: агент получает пустой `/workspace` и спецификацию `start.md`, и должен реализовать библиотеку целиком. Решение проверяется скрытым pytest-набором из официального образа задачи. Раннер и пайплайн сборки образов живут в отдельной внутренней репе [`gigallms/nl2repo-bench`](https://gitlab.ai.cloud.ru/gigallms/nl2repo-bench), которую PAM клонирует по закреплённому commit SHA.

Прогоняется весь закреплённый ростер — 104 задачи. Агент и verifier работают в **разных** RLI-сессиях: агент в общем окружении `nl2repo-agent-runtime`, verifier — в задачном `nl2repo-verifier-<task>`, собранном из официального GHCR-образа по неизменяемому digest'у. Поэтому агент не может ни прочитать тесты, ни повлиять на проверку.

```yaml
autometrics:
  evaluators:
    nl2repo:
      agent: openhands-sdk
      pass-at-k: 1
      max-tokens: 16384
      max-input-tokens: 98304
      agent-timeout: 28800
      n-concurrent: 32
```

### Основные параметры

| Ключ | Тип | Дефолт | Что делает |
|---|---|---|---|
| `agent` | `openhands-sdk` \| `openhands` \| `openhands-cli` \| `mini-swe-agent` | `openhands-sdk` | Агент, решающий задачи |
| `pass-at-k` | int ≥ 1 | `1` | Число независимых попыток на задачу |
| `max-tokens` | int ≥ 1 | `16384` | Максимум output-токенов одного ответа модели |
| `max-input-tokens` | int ≥ 1 | `98304` | Клиентский лимит input-контекста агента; не увеличивает контекст inference endpoint |
| `max-total-output-tokens` | int | `null` | Общий бюджет output-токенов на прогон |
| `agent-kwargs` | dict | `{}` | Дополнительные параметры агента, включая `temperature` и `top_p` |
| `agent-timeout` | int | `null` | Таймаут agent phase одной попытки, секунды; без него берётся 3600 |
| `verifier-timeout` | float | `null` | Таймаут verifier phase, секунды |
| `llm-timeout` | int ≥ 1 | `21600` | Таймаут одного запроса к модели |
| `n-concurrent` | int ≥ 1 | 8 × число inference-инстансов | Параллельные Harbor trials |
| `max-retries` | int ≥ 0 | `1` | Перезапуски trial после инфраструктурной ошибки |
| `n-tasks` | int | `null` | Ограничить число задач для smoke-прогона |
| `task-name` | str или list | `null` | Запустить только указанные задачи |
| `exclude-task-name` | str или list | `null` | Исключить указанные задачи |
| `enable-contamination-filter` | bool | `false` | Помечать попытки, где агент подтянул готовую библиотеку из сети вместо реализации |
| `agent-env-version` | str | `1.0.2` | Версия агентского RLI-окружения |
| `verifier-env-version` | str | `1.0.0` | Версия verifier-окружений; принимается также как `env-version` |
| `force-recalc` | bool | `false` | Не переиспользовать готовый результат |

Параметры семплирования задаются внутри `agent-kwargs`:

```yaml
agent-kwargs:
  temperature: 1.0
  top_p: 0.95
```

Если `temperature` или `top_p` заданы, их значения включаются в название метрики, отправляемой в JD.

Параметры `timeout-rli`, `request-retries`, `transfer-retries`, `transfer-retry-backoff`, `session-ready-timeout`, `session-ready-poll-interval`, `create-session-jitter-max-sec`, `probe-retry-backoff-sec`, `enable-probe-warmup` и `command-timeout` управляют взаимодействием с RLI. `ttl-ms` задаёт верхнюю границу простоя RLI-сессии; фактический TTL считается как `min(ttl-ms, agent-timeout + 600 секунд)`. Адрес RLI PAM выбирает автоматически: публичный endpoint при наличии интернета и `gym.static.space` в закрытом контуре.

### Метрики

Reward одной попытки — `min(passed / test_case_count, 1)`, ноль при таймауте pytest. На его основе считаются:

- `MEAN_SCORE` — средний reward по всем запланированным попыткам; каждая задача весит одинаково;
- `PASS_RATE` — доля пройденных тест-кейсов от всех объявленных; задачи с большим числом тестов весят больше;
- `CONTAMINATED` — `0`/`1`, была ли хотя бы одна попытка помечена фильтром контаминации; при выключенном фильтре всегда `0`.

Имя метрики в JD:

```text
NL2REPO_<агент>_MAX_TOKENS_<N>[_TEMPERATURE_<T>][_TOP_P_<P>]_<TIMEOUT_<сек>|N_CONCURRENT_<N>>_<метрика>
```

Пример: `NL2REPO_OPENHANDS_SDK_1_21_0_MAX_TOKENS_16384_TIMEOUT_28800_MEAN_SCORE`.

Значения — доли, не проценты. Срезы по difficulty, `pass@k` и счётчики роллаутов остаются в `metrics.json` внутри секции `run` и в JD не уходят.

Результат отправляется в JD, только если прогон завершён полностью и без инфраструктурных и API-ошибок: обстрел, часть запросов которого не дошла до модели, метрику не публикует. Filtered-прогоны с `n-tasks`, `task-name` или `exclude-task-name` в JD не отправляются — они несравнимы с полным бенчем. Настройки методологии и точный SHA раннера пишутся в `nl2repo_setup.json`; результат с другим fingerprint повторно не используется.

### Артефакты

| Путь | Что содержит |
|---|---|
| `<artifacts>/metrics/nl2repo/metrics.json` | Метрики, диагностика и состояние прогона |
| `<artifacts>/metrics/nl2repo/nl2repo_setup.json` | Fingerprint методологии для кэша |
| `<artifacts>/metrics/nl2repo/harbor_jobs/<run>/<task>__<id>/` | Harbor trials: логи агента, verifier и результат попытки |
| `<artifacts>/metrics/nl2repo/rli_infrastructure_report.json` | Детали RLI-ошибок, если они возникли |
| `<artifacts>/metrics/nl2repo/metrics_step_<step>.json` | Тело запроса в JD |
| `<artifacts>/nl2repo_artifacts.tar.zst` | Архив metrics-dir для трансфера в закрытом контуре |

## 18. Блок `evaluators.aa_lcr`

AA-LCR (Artificial Analysis Long Context Reasoning) — judged long-context QA:
кандидат отвечает на вопросы по нескольким документам, а judge-модель сравнивает
ответ с эталоном. Основная метрика `aa_lcr/accuracy` — средняя точность по
`n_repeats` генерациям.

Dataset и runner поставляются в образе из
`gitlab.ai.cloud.ru/gigallms/aa-lcr-bench`; commit закреплён переменной
`AA_LCR_REF` в `.gitlab-ci.yml`. Во время замера данные с NFS не читаются.

```yaml
autometrics:
  evaluators:
    aa_lcr:
      judge:
        model_name: "Qwen3-235B-A22B-Instruct-2507-FP8"
```

`judge.model_name` обязателен, остальные параметры опциональны:

| Ключ | Дефолт | Что делает |
|---|---|---|
| `judge.model_name` | обязательный | Judge, поднятый как scoring inference |
| `model` / `judge` | `max_tokens` 16384 / 4096, `temperature` 0.0, `num_retries` 20 | Настройки запросов; также поддерживаются `top_p` и `top_k` |
| `llm_timeout` | 6400 | Таймаут запроса к кандидату и judge в секундах |
| `parallel` | 8 × число inference-инстансов | Параллельные запросы генерации |
| `judge_parallel` | 48 × число scoring-инстансов | Параллельные запросы judge |
| `n_repeats` | 3 | Число генераций на вопрос |
| `dataset_csv` | `/home/user/aa_lcr/data/AA-LCR_Dataset.csv` | Dataset CSV в образе |
| `text_dir` | `/home/user/aa_lcr/data/extracted_text` | Документы в образе |
| `runner_root` | `/home/user/aa_lcr/runner` | Каталог runner в образе |
| `limit` | `null` | Ограничение числа вопросов для smoke-прогона |
| `use_gold_as_candidate` | `false` | Проверка judge на эталонных ответах |
| `python_path` | интерпретатор node runner | Python для запуска runner |

Если после ретраев generation или judge завершается технической ошибкой,
AA-LCR логирует причину и не публикует метрики, но не прерывает остальные
эвалюаторы PAM. Scoring не запускается без успешно завершённой generation-фазы,
поэтому артефакты прерванного запуска не используются как свежий результат.

JD-ключ содержит только judge, режим кандидата (`model` или `gold`) и число
повторов. Имя замеряемой модели передаётся отдельным полем JD; таймаут, `limit`,
sampling-настройки, пути и concurrency в имя метрики не входят.

## 19. Блок `evaluators.token_nll`

Per-token NLL на фиксированных текстовых наборах через prefill поднятого SGLang
(`/generate` с `return_logprob`). Из одного форварда получаются две группы метрик:

- `val_loss/<dataset>`, `val_ppl/<dataset>`, `val_loss/all` — средний NLL по
  каждому набору (взвешенно по токенам для `all`). Набор должен быть отложен
  из train, иначе это не val loss; evaluator этого не проверяет.
- `memorization/<dataset>/proxy` — memorization proxy из MAI-Thinking-1 (2.5.4):
  доля улучшения NLL относительно референсного чекпоинта, приходящаяся на токены,
  которые текущий чекпоинт предсказывает с NLL ниже `memorization_threshold`.
  Рядом: `frac_tokens_below_thr`, `delta_vs_ref`, `ref_step`, а также `delta_below_thr`
  и `delta_rest` — то же улучшение, разложенное на две группы токенов (средние на токен,
  каждая публикуется только если её группа непуста). Порог задаётся `memorization_threshold`.

  Метрика всегда относительная: референс — это замер того же набора на более
  раннем чекпоинте того же рана, лежащий в кэше. Берётся последний кэш со step
  ≤ текущий − `ref_lag_batches`. Если такого замера нет (первый запуск на ране,
  лаг больше пройденных батчей, кэш чужой — другой набор, токенизатор или
  квантизация), вся группа `memorization/*` просто не публикуется, а `val_loss/*`
  считается как обычно. Так же пропускается и когда суммарное улучшение NLL не
  положительно: доля от неположительного знаменателя смысла не имеет.

```yaml
autometrics:
  evaluators:
    token_nll:
      ref_lag_batches: 20000
      window_tokens: 4090
      datasets:
        ocr_private_pdf:
          path: /home/jovyan/data/token_nll/ocr_private_pdf.jsonl.zstd
        ocr_cc_pdf:
          path: /home/jovyan/data/token_nll/ocr_cc_pdf.jsonl.zstd
          max_docs: 300
```

Наборы — `jsonl.zstd` с полем `text`; текст токенизируется токенизатором из HF
чекпоинта (с BOS), режется на окна `window_tokens` с перекрытием в один токен, так
что каждый токен документа кроме первого получает ровно один logprob.

| Ключ | Дефолт | Что делает |
|---|---|---|
| `datasets.<name>.path` | обязательный | Путь к `jsonl.zstd` |
| `datasets.<name>.max_docs` | `max_docs` | Число документов из начала файла |
| `ref_lag_batches` | обязательный | Референс — последний кэш со step ≤ текущий − лаг |
| `max_docs` | 500 | Дефолт документов на набор |
| `window_tokens` | обязательный | Длина окна в токенах: `max_req_input_len` поднятого SGLang (контекст модели минус запас сервера, для ctx 4096 это 4090), а не `max_seq_len` из pam_config — тот задаёт `max_length` для lm_eval |
| `memorization_threshold` | 0.01 | Порог NLL «почти детерминированного» токена |
| `top_spans` | 200 | Сколько спанов писать в `memorized_spans_<dataset>.jsonl` |
| `cache_dir` | `<run_dir>/token_nll_cache` | Куда класть кэш NLL (run_dir — родитель `ep*_ba*`) |
| `concurrency` | 16 × число inference-инстансов | Параллельные запросы |
| `request_timeout` | 21600 | Таймаут запроса в секундах |
| `request_retries` | 20 | Повторы запроса при сетевой ошибке |

Кэш живёт на уровне рана, а не чекпоинта, потому что quick-чекпоинты удаляются:
`<cache_dir>/<dataset>/tokens_<fp>.npy`, `meta_<fp>.json`,
`nll_<fp>_ba<step>.npy` (float32, NaN на первом токене документа). `<fp>` —
отпечаток токенов, окна и имени HF-чекпоинта (`hf_checkpoint` vs
`hf_checkpoint_fp8`); референс подбирается только с тем же отпечатком, так что
смена набора, токенизатора или квантизации не смешивает несравнимые кэши.
Повторный замер того же step переиспользует кэш без запросов к модели; файлы
пишутся атомарно, нечитаемый кэш удаляется и пересчитывается.

Артефакты в `<artifacts>/metrics/token_nll/`: `all_metrics.json`,
`memorized_spans_<dataset>.jsonl` (doc, offset, n_tokens, contribution, средние
NLL и decoded-текст спана, по убыванию вклада) и `metrics_step_<step>.json`.

Кэш — обычные `.npy` с NLL на токен, так что пару чекпоинтов можно разобрать офлайн без GPU:
функции `memorization_proxy` и `memorized_spans` в `token_nll.py` принимают два массива NLL.

Значения NLL считаются в bf16-инференсе SGLang и сдвинуты относительно
тренировочного loss на константу; сравнимы между чекпоинтами при одинаковых
настройках inference и квантизации, но не с кривой train loss напрямую.

## 20. Блок `evaluators.checkpoint_stats`

Потензорная статистика весов HF-чекпоинта: моменты и гистограмма `log10|x|`, счётчики NaN/Inf, row/col-нормы с
детектором rank-1 collapse (`max/median`), top-K randomised SVD (spectral norm, stable rank).
Шутер читает safetensors напрямую и не делает запросов к поднятой модели, поэтому
у него нет ручек таймаута запросов и сетевой параллельности.

```yaml
autometrics:
  evaluators:
    checkpoint_stats: {}
```

Эквивалентная запись со всеми параметрами в значениях по умолчанию:

```yaml
autometrics:
  evaluators:
    checkpoint_stats:
      device: cpu
      num_threads: 16
      role_overrides: {}
      topk_sigma: 16
      svd_niter: 2
      spectral_numel_cap: 5368709120
      include: null
      exclude: null
      hist_bins: 80
      hist_log_min: -8.0
      hist_log_max: 4.0
```

| Ключ | Дефолт | Что делает |
|---|---|---|
| `device` | `cpu` | Устройство для расчёта (`cpu`, `cuda`, `cuda:N`). Замер идёт параллельно с инференсом, поэтому по умолчанию CPU |
| `num_threads` | 16 | Число CPU-потоков Torch в дочернем процессе |
| `role_overrides` | `{}` | Карта `regex: role`; первое совпадение переопределяет роль, имена ролей — строчные латинские буквы, цифры и `_` |
| `topk_sigma` | 16 | Число сингулярных чисел в randomised SVD; `0` отключает spectral-статистику |
| `svd_niter` | 2 | Число power-итераций randomised SVD |
| `spectral_numel_cap` | 5368709120 (20 ГБ в fp32) | Матрицы с большим числом элементов пропускают SVD |
| `include` / `exclude` | `null` | Regex по полному имени тензора (см. ниже) |
| `hist_bins` | 80 | Число корзин гистограммы `log10|x|` |
| `hist_log_min` / `hist_log_max` | -8 / 4 | Границы гистограммы `log10|x|` |

`include` и `exclude` — фильтры по полному имени тензора в чекпоинте (например,
`model.layers.10.mlp.experts.5.up_proj.weight`), задаются регулярным выражением и ищутся
в любом месте имени (`re.search`). `null` означает, что фильтр выключен: при
`include: null` анализируются все тензоры, при `exclude: null` ничего не исключается,
поэтому по умолчанию считается весь чекпоинт. Если заданы оба, тензор берётся, когда он
совпал с `include` и не совпал с `exclude`. Некорректный regex отклоняется при загрузке
конфига. Пример — только внимание и роутер MoE, без MTP-слоя 61:

```yaml
checkpoint_stats:
  include: '\.self_attn\.|\.mlp\.gate\.'
  exclude: '^model\.layers\.61\.'
```

Фильтр влияет и на отчёт, и на агрегаты: `n_tensors` и все метрики по ролям считаются
только по отобранным тензорам.

Артефакты: полный потензорный отчёт `metrics/checkpoint_stats/checkpoint_stats.json`
(пишется атомарно; при повторном запуске готовый отчёт не пересчитывается). В TensorBoard
и JD публикуются только агрегаты: `checkpoint_stats/n_tensors`, `n_tensors_with_nan`,
`n_tensors_with_inf`, `n_errors`, `n_unknown`, `max_abs`, `fro_norm_total` и по каждой роли тензоров
(`embedding`, `lm_head`, `mla`, `moe_expert`, `moe_router`, `dense_mlp`, `layernorm`, `mtp`, ...)
`mean_abs_mean`, `max_abs_max`, `fro_mean`, `max_over_median_max` (2-D тензоры) и
`stable_rank_min` (матрицы с SVD). Для MTP-блока используется отдельный префикс
`mtp_<role>` (например, `checkpoint_stats/mtp_mla/stable_rank_min`). `n_errors` считает
тензоры с ошибкой расчёта, включая ошибку SVD; `n_unknown` — тензоры без известной роли.
В лог выводятся 10 самых частых шаблонов неизвестных имён. NaN/Inf в отчёте сохраняются как `null` и в
агрегаты не входят.


SVD считается для всех 2D-матриц, включая `unknown`, если обе размерности не меньше 8,
число элементов не превышает `spectral_numel_cap` и `topk_sigma > 0`.
Исключения — `layernorm`, `final_norm`, `ssm_param`. Тензоры fused MoE вида
`*.experts.gate_up_proj` / `*.experts.down_proj` с формой `[E, m, n]` разбиваются
на записи `<name>[expert=i]`; каждая получает отдельные base, rowcol и spectral-метрики.
`n_tensors` в этом случае считает записи экспертов, а не исходные 3D-тензоры.
Тип MLA/GDN для общих проекций определяется по соседним ключам полного индекса до
применения фильтров; MTP-маркеры распространяются на весь соответствующий слой.
Пути Qwen `mtp.*` / `model.mtp_head.*` также относятся к MTP-блоку.

FP8-веса деквантизуются через `<name>_scale_inv` и
`config.json → quantization_config.weight_block_size`; поддерживаются скейлы
в другом шарде и неполные крайние блоки. Scale-тензоры не входят в статистику.
В отчёте сохраняется исходный dtype, но статистика считается по деквантизованным
значениям. Ошибка деквантизации записывается в `error` конкретного тензора и
увеличивает `n_errors`; остальные тензоры продолжают считаться. Неподдерживаемые
квантизованные веса (например, native DeepSeek, MXFP4, GPTQ/AWQ или compressed-tensors)
получают `unsupported quantization`, без статистики сырых кодов. Служебные скейлы и
zero-points исключаются, число пропущенных весов выводится в лог. Обычные bf16/fp32
тензоры в таком чекпоинте считаются как обычно. Поддерживаются safetensors-чекпоинты;
`pytorch_model*.bin` нужно предварительно конвертировать в safetensors.

Во время расчёта каждая запись сразу сбрасывается в
`checkpoint_stats.jsonl.partial`. После прерывания уже записанные тензоры не читаются
и не считаются повторно; незавершённая последняя строка отбрасывается.
Параметры, конфиг модели, индекс, размеры и времена изменения шардов проверяются
перед продолжением: при изменении входов частичный расчёт начинается заново.
Записи с ошибками также сохраняются; для их повторного расчёта нужно удалить partial.
Итоговый JSON собирается потоково и заменяется атомарно; partial удаляется после успеха.
Готовый отчёт, как и раньше, переиспользуется: для полного пересчёта включите
`globals.full_metrics_recalculate` или удалите итоговый и частичный отчёты.

CLI можно запускать отдельно от PAM на HF-чекпоинте любой архитектуры:

```bash
python node_runner/src/metrics/evaluators/checkpoint_stats/run.py \
  --hf-dir /path/to/hf_checkpoint \
  --output /path/to/checkpoint_stats.json \
  --device cpu --num-threads 16
```

Остальные CLI-параметры имеют те же дефолты, что и YAML; имена используют дефисы
(`--topk-sigma`, `--hist-bins` и т. д.). `--role-overrides` принимает JSON,
например `--role-overrides '{"\\.custom_attn\\.": "attn"}'`.
При неизвестных именах группировка может оставаться `unknown`, но 2D-матрицы
всё равно получают spectral-статистику. Требуются Torch и safetensors.

## 21. Блок `evaluators.harbor_rli`

`HarborRliEvaluator` запускает SWE-bench и Terminal-Bench через Harbor в RLI-контейнерах.
Ниже описаны настройки SWE-bench.
Поддерживаются `swe-bench-verified` и `swe-bench-multilingual` с агентом
`mini-swe-agent`. Для `swe-bench-verified` также доступен `agent: openhands`.
Транспорт запросов к модели по умолчанию — HTTP relay.
При `pass-at-k: 1` метрика — доля решённых задач от 0 до 1.

```yaml
autometrics:
  evaluators:
    harbor_rli:
      benchmark: swe-bench-multilingual
      agent: mini-swe-agent
      max-tokens: 16384
      n-concurrent: 16
      pass-at-k: 1
      job-name: swe-multilingual
```

`max-tokens` ограничивает длину одного ответа модели, `n-concurrent` —
число одновременно выполняемых задач. Если `n-concurrent` не задан,
PAM устанавливает его отдельно для каждого запуска: 8 × число inference-инстансов.
Адрес модели `model-api-base` по умолчанию берётся из PAM.

Для обоих наборов в одном PAM job используйте `harbor_rli.runs`:

```yaml
autometrics:
  evaluators:
    harbor_rli:
      runs:
        - benchmark: swe-bench-verified
          agent: mini-swe-agent
          max-tokens: 16384
        - benchmark: swe-bench-multilingual
          agent: mini-swe-agent
          max-tokens: 16384
```

Запуски выполняются параллельно; их `n-concurrent` суммируются.
Все параметры запуска задаются внутри соответствующего элемента `runs`.
Список должен быть непустым, значения `benchmark` — различными;
вложенные `runs` не поддерживаются.
Специфичные параметры адаптера передаются через `benchmark-kwargs`,
например `benchmark-kwargs: {dataset: /path/to/tasks}` для своего набора задач.

`job-name` (также принимается `job_name`) задаёт имя Harbor-прогона.
Для продолжения укажите прежнее имя при том же каталоге артефактов
и совместимом конфиге. Если PAM уже закешировал итоговую метрику,
задайте `force-recalc: true`, чтобы снова вызвать Harbor. Этот параметр
не заставляет Harbor повторять уже сохранённые задачи того же прогона.

И для одиночного запуска, и для `runs` результаты сохраняются в
`<checkpoint>/artifacts/metrics/harbor_rli/<benchmark>/`.
Внутри находятся `metrics.json`, `harbor_rli_setup.json` для проверки кеша
и `harbor_jobs/<job-name>/` с логами и результатами задач.
Имена метрик берутся из адаптера без префикса `harbor_rli`.
В итоговые метрики PAM попадают только завершённые расчёты (`complete: true`)
с числовыми конечными значениями и без фильтров задач.

[Dockerfile](./Dockerfile) упаковывает отдельное окружение
`/home/user/harbor_rli_env` без `--system-site-packages`.
В него входят зависимости и пакет `gigarli`; исходники `harbor_rli`
размещаются в `/home/user/harbor_rli_env/harbor_rli`.
Датасеты по умолчанию находятся в
`/home/user/harbor_rli_env/datasets/swe-bench-verified` и
`/home/user/harbor_rli_env/datasets/swe-bench-multilingual`.
[CI](./.gitlab-ci.yml) скачивает исходники и данные до сборки образа;
ревизия SWE-датасетов закреплена в `HARBOR_RLI_SWE_DATA_REF`.
Исходники `harbor_rli` берутся из `main`, `gigarli` — из `master`;
фактические ревизии исходников и всех датасетов записываются в
`harbor_rli_version.txt` и участвуют в проверке кеша.

## 22. Блок `evaluators.pier_rli`

`pier_rli` запускает бенчмарки через Pier и GigaRLI. Сейчас поддерживается
`benchmark: deepswe`: Mini-SWE-Agent работает с кодом в RLI, а Pier применяет
его патч и запускает verifier задачи. Запросы агента к модели по умолчанию
передаются через HTTP relay.

```yaml
autometrics:
  evaluators:
    pier_rli:
      benchmark: deepswe
      agent: mini-swe-agent
      max-tokens: 16384
      n-concurrent: 64
      pass-at-k: 1
      command-timeout: 30
      agent-timeout: 36000
```

Общие параметры задаются в блоке `pier_rli`. Специфичные для бенчмарка
параметры, например `dataset` и `env-version`, передаются через
`benchmark-kwargs`. Дополнительные поля верхнего уровня передаются shooter;
повторение общих параметров внутри `benchmark-kwargs` запрещено.
`max-input-tokens` передаётся агенту как `agent-kwargs.max_input_tokens`.

Для нескольких бенчмарков предусмотрен список `runs`: каждый элемент содержит
свой `benchmark` и параметры запуска. `runs` нельзя совмещать с параметрами
одиночного запуска; вложенные списки и повторение одного бенчмарка запрещены.

### Упаковка в образ

CI отдельно клонирует Pier RLI, GigaRLI и Deep-SWE по закреплённым коммитам
из `.gitlab-ci.yml`. Dockerfile копирует GigaRLI в `gigarli`,
данные — в `datasets`, а репозиторий shooter целиком — в `pier_rli`
внутри `/home/user/pier_rli_env`.
При запуске `pier_rli` не использует код из `contrib`.

| Компонент | Путь в образе |
|---|---|
| Python и зависимости | `/home/user/pier_rli_env/bin/python` |
| GigaRLI | `/home/user/pier_rli_env/gigarli` |
| Задачи Deep-SWE | `/home/user/pier_rli_env/datasets/deepswe` |
| Код Pier RLI | `/home/user/pier_rli_env/pier_rli` |
| Версия shooter | `/home/user/pier_rli/pier_rli_version.txt` |

`shooter_version` содержит коммиты всех трёх репозиториев:
`<pier-sha>+gigarli.<gigarli-sha>+deepswe.<data-sha>`.
При отправке метрик в JD к версии добавляется `-<benchmark>`.
Путь к задачам можно переопределить через `benchmark-kwargs.dataset`.

### Метрики и результаты

Базовая метрика — доля успешных попыток. Для примера выше её имя:
`DEEPSWE_MINI_SWE_AGENT_MAX_TOKENS_16384_TIMEOUT_36000`.
Если задан `agent-timeout`, вместо `N_CONCURRENT_<N>` используется
`TIMEOUT_<seconds>`. При `pass-at-k > 1` дополнительно публикуется метрика
с суффиксом `_PASS_AT_<k>` — доля задач, решённых хотя бы в одной из k попыток.
При `pass-at-k: 1` отдельной метрики `_PASS_AT_1` нет.

Артефакты находятся в `<checkpoint>/artifacts/metrics/pier_rli/deepswe/`:

- `metrics.json` — метрики и состояние завершённости прогона;
- `pier_rli_setup.json` — параметры и версия для проверки кэша;
- `pier_jobs/<job-name>/` — результаты попыток и логи Pier.

В JD публикуются только завершённые результаты с конечными числовыми
метриками по полному набору задач датасета. Полноту набора проверяет адаптер:
для выборки задач и запуска отдельного каталога задачи он выставляет
`complete: false`, сохраняя локальные результаты. Фильтр, выбирающий все
задачи, не мешает публикации. `job-name` задаёт имя прогона Pier;
готовые метрики повторно используются при совпадении fingerprint.

При переходе со старого `evaluators.deepswe` замените его на
`evaluators.pier_rli` и добавьте `benchmark: deepswe`.
Старые результаты в `metrics/deepswe` автоматически не переносятся;
сохранения одного `job-name` недостаточно для продолжения прогона
после смены каталога артефактов.

Подробности shooter — в [README Pier RLI](https://gitlab.ai.cloud.ru/gigallms/pier_rli/-/blob/7472d1f882ebb876c5b5b594fd606300e4a5e465/README.md).

# Как интегрировать PAM в ваше обучение
### Всего 3 действия
1. Достать раздел `autometrics` из [примеров](./configs/) и вставить в ваш конфиг обучения
2. Добавить `callback` автометрик в раздел `callbacks`
```yaml
callbacks
    autometrics:
        eval_interval: ${eval_interval}
        stable_save_interval: ${stable_save_interval}
        nnodes: 4
        user: ${user}
```
По сути, единственный важный параметр, который вам здесь нужно править - это `nnodes`, количество нод, на которых запустятся замеры. На каждой ноде будет по **8** карт

3. Запустить **обучение** новым интерфейсом для сабмита джоб, который находится по пути `llm-foundry/sber_scripts/jobs/examples/train_job.py` или [тут](https://gitlab.ai.cloud.ru/gigallms/llm-foundry/-/blob/main/sber_scripts/jobs/examples/train_job.py?ref_type=heads)


# Как добавить метрики
1. Добавить кастомный датасет и кастомную аггреацию проще и удобнее добавить в [lm-evaluation-harness(наш форк)](https://gitlab.ai.cloud.ru/gigallms/lm-evaluation-harness) в соответствие с [их гайдом](https://gitlab.ai.cloud.ru/gigallms/lm-evaluation-harness/-/blob/main/docs/new_task_guide.md?ref_type=heads)
2. Если необходима кастомная "обстреливалка"(`evaluator`), то [гайд](#как-добавить-обстреливалку) далее


**В 99% случаев лучше добавить метрику в `lm-evaluation-harness`**

# Как добавить "обстреливалку"
1. В директорию [metrics/evaluators](./node_runner/src/metrics/evaluators) добавить файл и реализовать класс по аналогии с `lm_eval.py`, в котором должны быть реализованы методы
    + `setup` - метод, который сетапит все необходимые переменные, директории
    + `run` - метод, который запускает все метрики. Требуется реализовывать вызов непосредственной обстреливалки через класс [SubprocessManager](./node_runner/src/subprocess_manager.py) 
    + `get_metrics` - метод, который возвращает метрики в ожидаемом формате, чтобы потом удобно сохранять их в TB и wandb
2. В классе [AutoMetricsManager](./node_runner/src/metrics/metrics_manager.py) добавил в переменную `EVALUATORS` свою "обстреливалку" с соответствующим названием, `которое вы будете использовать в конфиге`
3. Если необходимо окружение, то стоит добавить его в [Dockerfile](./Dockerfile) после 160 строки по мере необходимости обновления зависимостей(кеша слоёв). `Чем чаще правки - тем ниже окружение`
4. Добавить примеры заполнения конфига для своей обстреливалки в [configs](./configs/)
5. Написать тесты с mock-конфигами в [sber_tests](./sber_tests/)

**Обязательно:** у каждой обстреливалки должны быть две настраиваемые ручки для запросов к обслуживаемой модели — **таймаут запроса** (поле `timeout` в секундах, дефолт из общей константы `DEFAULT_REQUEST_TIMEOUT` в `config_model.py`, число не хардкодить) и **параллельность** (`workers` / `parallel_api_calls` / `num_threads` / `concurrency`). Без обеих ручек обстреливалку добавлять нельзя (маленький таймаут = обстреливалка умирает на медленной reasoning-модели под нагрузкой). Если у нижележащего тула нет хука под одну из ручек — добавьте хук (тулы вендорены в `contrib/`) либо явно опишите это в PR. Подробности и точки интеграции — в скилле `/pam-create-evaluator`.

Если вы просите ассистента закоммитить обстреливалку **без** этих ручек, он обязан добавить строку-сигнал `⚠️ MISSING KNOBS: ...` в начало каждого коммита и в начало описания MR.

### Изоляция интернета в RLI

`allow-internet-access: false` в настройках `swe_bench`, `terminal_bench`,
`skillsbench`, `nl2repo` или отдельного запуска `harbor_rli` отключает интернет
в RLI-сессии агента (`allow_internet_access=false`) и pip-прокси.
По умолчанию изоляция включена для SWE Verified, SWE Multilingual и HarnessBench,
а для Terminal Bench, NL2Repo и SkillsBench выключена. DeepSWE сохраняет
обязательную изоляцию. Для `harbor_rli.runs` настройка задаётся в каждом запуске.

Для Toolathlon через GigaMetric используется существующий путь
`gigametric.benchmarks.<имя>.parameters.rli_network_isolated: true`;
по умолчанию это `false`. Канал управления RLI и канал запросов к модели
работают вне отключаемого доступа агента в интернет.

### Приоритет агентских шутеров (F3)

Агентские evaluators с `LATENCY = True` допускаются раньше bulk-шутеров и не
ждут снятия перегрузки evaluated engine. Перегрузка judge по-прежнему учитывается.
Для локального SGLang их запросы идут через `/latency` в OBM proxy и получают
`priority: 1` (обычные запросы имеют приоритет 0; явно заданный приоритет сохраняется).
Приоритет передаётся и через custom chat-template, и через OpenAI/v2 API.
Priority scheduling включён безусловно в конфиге SGLang: `schedule-policy`
принимает `fcfs` (дефолт) или `lof`, `enable-priority-scheduling: true`.
Обычные запросы получают `default-priority-value: 0`; выполняющиеся запросы
не прерываются. Флаги включения приоритетов, отключения вытеснения и порядка
приоритетов фиксированы: отключение приоритетов и обратный порядок отклоняются
при чтении конфига. Те же дефолты действуют для обеих сторон PD и judge engines.
В конфиге роутера `policy` по умолчанию `round_robin`; `prefill-policy` и
`decode-policy` могут отдельно задать политику для сторон PD. Допустимые значения:
`random`, `round_robin`, `cache_aware`, `power_of_two`, `bucket`, `manual`,
`consistent_hashing`, `prefix_hash`.
В replay используются те же приоритеты, что и при записи; `priority` входит
в ключ записанной генерации. External API использует обычные endpoints.
F3 не управляет concurrency или лимитами сессий.

### Кеш-привязка агентских диалогов

На `/latency` OBM proxy сам добавляет `x-smg-routing-key` для chat-запросов:
ключ вычисляется из сообщений до первого `user` включительно, модели,
инструментов и параметров chat template. Продолжения с неизменным началом
получают тот же ключ во всех процессах прокси, без правок шутера и хранения
истории на сервере. Поддерживаются custom chat template, OpenAI chat и v2.
Явный `routing_key` имеет приоритет: в заголовок передаётся его SHA256,
а исходное JSON-поле сохраняется для SGLang. Это позволяет передавать ключи
с Unicode и управляющими символами без ошибок HTTP. Явный `x-smg-routing-key`
передаётся как есть; прежние `x-pam-session-id` и `x-pam-shooter` тоже поддерживаются.

Для привязки между physical P задайте роутеру `prefill-policy: consistent_hashing`,
а внутри P — `load-balance-method: prefix_affinity`. `decode-policy: round_robin`
оставляет декод независимым. HiCache включается отдельно на prefill; ключ
маршрутизации не включает HiCache и не меняет приоритеты или лимиты.
Защита `prefix_affinity` от перегрузки может выбрать другой DP rank.

Одинаковое начало двух независимых диалогов даёт одинаковый ключ. После
переписывания начала истории (например, summarization) ключ меняется.
В сырых `/generate` с `input_ids` и `/v1/completions` PAM не определяет
границы диалога: передаёт явный ключ, а без него остаётся штатная политика
роутера. Обычные endpoints без `/latency` сохраняют прежнее поведение.

Автоматический ключ передаётся заголовком, тело генерации не меняется.
Generation recorder передаёт заголовок в SGLang, но не включает его в ключ
replay; существующие записи не требуют переупаковки или пересчёта.
