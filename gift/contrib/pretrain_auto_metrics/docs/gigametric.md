# GigaMetric shooter в PAM

PAM-adapter, который запускает [GigaMetric](https://gitverse.ru/valid_sbeco/GigaMetric.git) как сабпроцесс на master-pod-е и читает результаты через `validation.pam_export`.

## TL;DR

```bash
# 1. Скопировать smoke-конфиг и поправить под свою модель
cp configs/gigametric/gm_smoke_minimal.yaml configs/gigametric/my_run.yaml

# 2. Указать checkpoint + config в submit_pam_job.py:
#    checkpoint_dir, path_to_config

# 3. Запустить
python submit_pam_job.py
```

## Архитектура

PAM на master-pod-e:

1. Поднимает SGLang на `localhost:30000` с конверченным HF-чекпоинтом + obm_proxy на `127.0.0.1:{api_port}` (proxy перед SGLang).
2. Регистрирует `GigaMetricEvaluator` (см. `node_runner/src/metrics/metrics_manager.py`).
3. `setup()` распаковывает `/home/user/papayametric.tar.zst` в `/home/user/papayametric` и материализует `validation_config` (inline или из файла) в `/home/user/gigametric_configs/<validation_id>/`.
4. `run()` стартует `/home/user/papayametric/run_validation.py` через `SubprocessManager`, cwd = распакованный GigaMetric tree, runtime python = baked `/home/user/papayametric_venv/bin/python`. Local-модель в `validation_config.models` адресуется по obm_proxy api_port.
5. `get_metrics()` читает плоский dict `gigametric/<model_id>/<benchmark_id>/<metric>` из `<base_dir>/<validation_id>/state.json` через `validation.pam_export.get_pam_metrics`. Если задан `local_model_id` — фильтрует+решейпит в плоский `{<bench>: value}` для JD.
6. `after_run()` отправляет метрики в JD, тарит артефакты, симлинкает их для трансфера.

## Config (PAM-side)

`evaluators.gigametric` в PAM YAML — поля `GigaMetricEvaluatorConfig` (`node_runner/src/config_model.py`):

| Field | Type | Default | Note |
|---|---|---|---|
| `base_dir` | Path | `<metrics_dir>/runs` | Куда GigaMetric пишет `state.json`, `metrics.jsonl`, summary. |
| `benchmarks` | dict[str, dict] | `{}` | Minimal mode: ключ — id запуска, значение — preset из `default.yaml` и плоские overrides. |
| `max_concurrent_requests` | int \| None | `16 * inference_instances` | Число конкурентных item pipelines каждого бенчмарка в minimal mode. Явное значение отключает динамический дефолт. |
| `validation_config` | dict | None | Inline-конфиг GigaMetric (см. `contrib/papayametric/validation/configs/example_full.yaml`). |
| `validation_config_path` | Path | None | Альтернатива inline: путь к готовому YAML. |
| `env` | dict[str, str] | `{}` | Доп. env-переменные для subprocess-а (`OPENAI_API_KEY`, и т.п.). |
| `runtime_python` | str | `/home/user/papayametric_venv/bin/python` | Интерпретатор с base-deps GigaMetric. |
| `local_model_id` | str | `None`; имя checkpoint-а в minimal mode | Id локальной модели, метрики которой уходят в JD. В minimal mode PAM использует его при генерации `models`. |
| `shooter_version` | str | `image-baked` | Fallback-версия, если per-run `gigametric_version.txt` не записан. |

Нужно задать ровно один источник: непустой `benchmarks`, `validation_config` или
`validation_config_path`.

### Minimal mode

Для обычного запуска достаточно перечислить benchmark id. PAM берёт `name`,
`primary_metric`, aggregation и остальные defaults из встроенного в образ
`validation/configs/default.yaml`, а локальную модель и URL генерирует сам.

```yaml
evaluators:
    gigametric:
        benchmarks:
            math500: {}
            gpqa:
                few_shot_n: 5
            rubq: {}
            mmlu_pro:
                max_concurrent_requests: 32
```

Ключ является benchmark id из `default.yaml`. Неизвестные поля benchmark-а
автоматически попадают в его `parameters`.
Поля `name`, `primary_metric`, `aggregation`, `credentials` и `parameters`
остаются на уровне benchmark entry. Локальный override имеет приоритет над
глобальным `max_concurrent_requests`, а глобальное значение PAM — над значением
из default-preset.

Если `max_concurrent_requests` не задан, PAM вычисляет бюджет
`Q = 128 * inference_instances`, выставляет каждому бенчмарку `Q / 8`, а в
`llm_config.semaphore` сгенерированной локальной модели записывает `Q`. При
стандартном `depth_first` / `max_concurrency` это допускает до восьми
GigaMetric tasks одновременно. Это scheduler budget по item pipelines, а не
строгий межпроцессный предел физических HTTP-запросов. Явный общий или
per-benchmark `max_concurrent_requests` переопределяет только параллельность
соответствующих бенчмарков; semaphore остаётся равным `Q`.

`gm_smoke_minimal.yaml` намеренно фиксирует `max_concurrent_requests: 64`,
чтобы smoke-нагрузка не масштабировалась с размером стенда.

По умолчанию PAM задаёт всем бенчам таймаут 21600 секунд, а LLM-запросам —
таймаут 21600 секунд через `llm_config.parameters.timeout`, 20 повторов и
паузу 10 секунд между повторами.
Явные значения в полном `validation_config` сохраняются.

### Маршрут через obm_proxy

GigaMetric ходит в `/v1/chat/completions` (через `openai.AsyncOpenAI`). PAM роутит запросы:

```
GigaMetric → http://127.0.0.1:{obm_proxy.api_port}/v1/chat/completions
          → obm_proxy (chat_handlers_mode = opensource | custom)
          → SGLang :30000
```

`GigaMetricEvaluatorConfig.apply_defaults` автоматически переписывает `validation_config.models[].llm_config.api_url`, если он указывает на `http://(localhost|127.0.0.1):30000/...` → `http://127.0.0.1:{obm_proxy.api_port}/...`. Внешние API-модели (GigaChat IFT и т.п.) остаются как есть.

**Обязательно** одно из:
* `autometrics.inference.reasoning-parser` / `tool-call-parser` / `chat-template` — opensource mode.
* `autometrics.obm_proxy.chat-template` — custom mode.

Иначе при загрузке конфига вылетит `gigametric targets local SGLang and requires chat formatting`. Если в `models` нет ни одной модели на localhost:30000 — chat formatting не требуется.

### JD-формат метрик

При `local_model_id` задан PAM **фильтрует** raw-output `pam_export.get_pam_metrics` по local-модели и **решейпит** ключи:

| До (pam_export raw) | После (JD payload) |
|---|---|
| `gigametric/qwen_4b_local/math500/exact_match_avg = 0.42` | `math500 = 0.42` |
| `gigametric/qwen_4b_local/ifbench/score = 0.9` | `ifbench = 0.9` |
| `gigametric/gigachat_2_max_ift/math500/...` | (отфильтровано) |

Совпадает с каноном `metrics_step_<step>.json` других шутеров (см. `metrics.zip`: `metrics/obm/metrics_step_0.json`).

## Где артефакты

| Путь | Что |
|---|---|
| `<base_dir>/<validation_id>/state.json` | Канонический статус прогона. Контракт между GigaMetric и PAM. |
| `<base_dir>/<validation_id>/gigametric_version.txt` | Per-run версия (если GigaMetric успел записать). |
| `/home/user/papayametric_scoring/<validation_id>/` | Локальные scoring records, включая сырые ответы модели. |
| `<artifacts>/metrics/gigametric/metrics_step_<step>.json` | JD-payload, который PAM отправляет в JDmetrics. |
| `<artifacts>/gigametric_artifacts.tar.zst` | Тарбол всего metrics-dir для transfer-а. |

## Полный `validation_config`

Полный формат нужен для нескольких моделей, кастомного scheduler или benchmark
entries, которых нет в `default.yaml`. PAM пробрасывает блок
`evaluators.gigametric.validation_config` в `run_validation.py --config` как-есть.
Контракт полей — на стороне GigaMetric
(`contrib/papayametric/validation/core/config_loader.py`).

### Скелет

```yaml
evaluators:
    gigametric:
        base_dir: /home/jovyan/${user}/gigametric_runs
        local_model_id: my_local         # обязательно если model.api_url=localhost:30000
        env:
            GIGACHAT_IFT_TOKEN: ${oc.env:GIGACHAT_IFT_TOKEN,''}
        validation_config:
            validation_name: my_run
            strategy: breadth_first
            scheduler:
                max_parallel_runs: auto
                fail_fast: false
            defaults:                     # уровень 1: defaults для всех benchmark/model
                llm_config:
                    max_retries: 20
                    retry_wait_sec: 10
                    parameters:
                        timeout: 21600
                benchmark_config:
                    timeout: 21600
                    fail_fast_consecutive_item_errors: 20
                    parameters:
                        max_concurrent_requests: 1
            benchmarks:                   # уровень 2: per-bench параметры
                math500:
                    name: math500-adapter
                    primary_metric: exact_match_avg
                    parameters:
                        num_items: 10     # override defaults для math500
            models:
                - id: my_local
                  llm_config:
                      model: openai
                      api_url: http://localhost:30000/v1  # перепишет apply_defaults
                      api_key: not-needed
                      semaphore: 4
                      parameters: {}
                  benchmarks:
                      - math500
```

### Где брать список доступных бенчмарков

```bash
cd contrib/papayametric
/home/user/papayametric_venv/bin/python -c "
from src.benchmarks.registry import BenchmarkAdapterRegistry
for n in sorted(BenchmarkAdapterRegistry().list_adapters()): print(n)
"
```

Исполняемые benchmark-preset с рекомендуемыми параметрами находятся в
`contrib/papayametric/validation/configs/default.yaml`. Комментированный
reference-конфиг находится в `validation/configs/example_full.yaml`. Полный
каталог бенчей с описаниями и поддерживаемыми параметрами — в
`contrib/papayametric/docs/CATALOG.md`.

Адаптерный код: `contrib/papayametric/src/benchmarks/<adapter_name>.py`. local-конфиг адаптера (default параметры): `contrib/papayametric/src/benchmarks/<adapter_dir>/local_config.py`.

### Приоритет параметров полного конфига

Параметры (`max_concurrent_requests`, `num_items`, `timeout`, `temperature`, `max_tokens`, и т.п.) могут жить на 3 уровнях. Приоритет от высокого к низкому:

| Приоритет | Источник | Пример |
|---|---|---|
| 1 (наивысший) | `validation_config.benchmarks.<bench>.parameters` | `benchmarks.math500.parameters.num_items: 10` |
| 2 | `validation_config.defaults.benchmark_config.parameters` | `defaults.benchmark_config.parameters.max_concurrent_requests: 1` |
| 3 (наименьший) | `local_config.py` адаптера | `BBACLocalConfig.max_concurrent_requests = 1` |

То же для llm_config:
| Приоритет | Источник |
|---|---|
| 1 | `validation_config.models[].llm_config` (и `.parameters`) |
| 2 | `validation_config.defaults.llm_config` |
| 3 | дефолты `LLMConfig` / handler-specific (`BaseLLMHandler`) |

**Подводный камень `parameters: {}`:** OmegaConf.merge с `null` НЕ обнуляет nested-поля. Если хочешь отказаться от наследования `parameters`, напиши явный пустой dict `parameters: {}`, иначе унаследуются `defaults.llm_config.parameters` (которые могут включать `max_tokens=4096` и т.п., меняющие поведение SGLang). См. `contrib/papayametric/CLAUDE.md` правило #4.

### Привязка бенча к модели

`validation_config.models[].benchmarks` — список id из секции `validation_config.benchmarks`. Только перечисленные бенчи прогонятся для этой модели. Можно гонять несколько моделей по разным наборам бенчей в одном run.

## gigametric_judged: бенчи с LLM-судьёй

Судейские бенчи GigaMetric замеряет отдельный эвалуатор `gigametric_judged`
(`GigaMetricJudgedEvaluator`), архитектура — копия `obm_judged`:

* `ParallelInferenceEvaluator`, зарегистрирован в `EVALUATORS` → на каждого судью из карты
  `judges` (`secondary_model_names`) `AutoMetricsManager` заводит задачу, которой
  нужны SUT и этот судья; судьи идут друг за другом, ноды между SUT и судьёй
  делит `DynamicScheduler`, задача зовёт `run_on_model_name(<judge_model_name>)`.
* `judges: {<judge_model_name>: [<bench_id>, ...]}` — судья → список бенчей
  (как в OBM). На судью — один подъём движка и один subprocess
  `run_validation.py` со своим `validation_id` (`pam_<ts>_<uuid8>_<judge_slug>`).
  `get_metrics()` мержит метрики по всем judge-ранам (бенчи между судьями не
  пересекаются).
* `<judge_model_name>` — директория чекпоинта под `$PAM_MODELS_PATH`, имя
  обязано быть в `InferenceManager.MODEL_NAME_PARAMS_MAPPER`.
* Адреса судей `AutoMetricsManager` кладёт в `config.judge_endpoints`, конфиг
  читает их как `judge_endpoints[судья]`, у каждого судьи свой роутер и свой адрес; PAM кладёт
  judge-LLMConfig (openai handler, `<url судьи>/v1`) один раз в общий слот
  `defaults.benchmark_config.judge_llm_config` per-judge конфига (аналог
  единого `llm_judge_shooter` в obm). GigaMetric применяет его ко всем бенчам
  среза; per-adapter инжекции и allowlist на стороне PAM нет — использует ли
  адаптер `judge_llm_config`, определяется на стороне GigaMetric.

Выбор модели-судьи: судья обязан отвечать в формате, который ожидает
judge-промпт адаптера. Специализированные судьи со своим протоколом вердикта
(pollux-judge-32b: `[FEEDBACK]…[RESULT] X [END]`) адаптеры bbh/hle не
распарсят — метрика обнулится; hle дополнительно требует от судьи function
calling. Безопасный выбор — универсальная instruct-модель
(например, `Qwen3-235B-A22B-Instruct-2507-FP8`).

Правила разделения (зеркально obm/obm_judged):

* В `gigametric` (BASE) судейские бенчи не запускать — без судьи они дают ~0.
* В `gigametric_judged` бенчи без записи в `judges` игнорируются (warning).
* Внешний судья (GigaChat IFT по API и т.п.) — это режим обычного
  `gigametric`: `judge_llm_config` с внешним `api_url`/`api_key` прямо в
  validation_config, PAM-механика подъёма судьи не нужна.

Пример: `configs/gigametric/qwen_4b_judged.yaml`. Метрики уходят в JD с
`shooter_name="gigametric"`.

## Ограничения / нюансы

* **Никаких токенов внешних API в репо.** Если адаптеру нужен API-токен (GigaChat IFT, OpenAI, Anthropic), пробрось его через `evaluators.gigametric.env`, а не хардкодь в YAML.
* **SGLang custom kernels** требуют пересборки Docker-образа. `sglang_tag` в `submit_pam_job.py` — `opensource` (default) или `custom`.
* **Chat formatting обязателен** для local-модели. PAM enforce'ит (см. `_validate_chat_formatting_strategy`); иначе obm_proxy 404'ит /v1/chat/completions.
* **Адаптеры с judge / LLM-user / docker / MCP** требуют отдельной настройки. См. `docs/gigametric_adapters_safety.md` (TODO) для классификации. Для smoke оставить только `math500-adapter`.
* **Per-adapter uv-envs.** Каждый адаптер ставит deps в `<papayametric_root>/.uv_envs/<adapter>/`. В Docker-образе GigaMetric source архивируется как `/home/user/papayametric.tar.zst` и в runtime распаковывается в `/home/user/papayametric`.

## Troubleshoot

| Симптом | Где смотреть |
|---|---|
| `ModuleNotFoundError: validation.pam_export` | Проверить `/home/user/papayametric.tar.zst` в образе и runtime-распаковку в `/home/user/papayametric`. Локально fallback — `contrib/papayametric/`. |
| `state.json missing` | GigaMetric упал до записи. Смотреть `autometrics_log_file` + per-bench логи в `<base_dir>/<validation_id>/`. |
| `JD POST failed` | `_send_metrics_to_jd` retry-ит 3 раза с экспонентой. После — молча возвращает None (тенасити `reraise=False`). Метрики остаются в `metrics_step_<step>.json` локально. |
| Adapter `.uv_envs/<adapter>` corrupt | Перезапустить pod: runtime tree в `/home/user/papayametric` будет заново распакован при следующем запуске. |

## Файлы

* `node_runner/src/metrics/evaluators/gigametric.py` — адаптеры (`GigaMetricEvaluator`, `GigaMetricJudgedEvaluator`).
* `node_runner/src/config_model.py` — `GigaMetricEvaluatorConfig`, `GigaMetricJudgedEvaluatorConfig`.
* `Dockerfile` — `papayametric_bundle` stage + source archive/runtime venv.
* `configs/gigametric/`:
  * `mock.yaml` — end-to-end PAM↔GigaMetric wiring без API/GPU (mock_model). Smoke для CI.
  * `gm_smoke_minimal.yaml` — минимальный плоский пример с пятью бенчмарками.
  * `gm_all_baskets_judged.yaml` — benchmark id из `default.yaml`, разделённые на обычный и judged-запуски.
  * `gigachat_700b.yaml` — пример полного `validation_config` для GigaChat-700b.
  * `qwen_4b.yaml` — пример полного `validation_config` для Qwen3-4B-Instruct.
  * `qwen_4b_judged.yaml` — Qwen3-4B-Instruct, судейские бенчи через `gigametric_judged`.
* `contrib/papayametric/` — submodule с upstream-фреймворком.


## Toolathlon / Среда через RLI

Пресеты `configs/alignment_prod/agent/*.yaml` используют `preset: toolathlon`
и запускают пять каталогов через `runner: rli`.

| Каталог | ID в PAM | Задач после исключений |
| --- | --- | ---: |
| `finalpool` | `toolathlon_finalpool` | 22 |
| `ru_local_extended` | `toolathlon_local_extended` | 8 |
| `ru_office_local` | `toolathlon` | 42 |
| `ru_online_stable` | `toolathlon_online_stable` | 9 |
| `ru_online_unstable` | `toolathlon_online_unstable` | 8 |

Полный профиль — 89 задач, одна попытка и 16 workers на каталог.
Метрики рассчитываются отдельно по каталогам.
`rli_tasks_from_mapping: true` выбирает задачи из ENV mapping GigaMetric;
`task_names` позволяет явно задать свой набор.
Общий список исключений с причинами хранится в `EXCLUDED_TASKS` в
`src/benchmarks/toolathlon/task_filter_utils.py` GigaMetric.

Для запуска нужны GigaMetric и SDK `gigarli` в PAM-образе.
Оба упакованы в `papayametric.tar.zst` и штатно распаковываются рядом:
GM — в `/home/user/papayametric`, SDK — в `/home/user/rli_sdk/gigarli`.
PAM добавляет `/home/user/rli_sdk` и прежний путь `/home/user/gigarli/src`
в начало `PYTHONPATH`, сохраняя совместимость с готовыми образами и остальные пути; окружение наследуется процессом GM и его адаптером Toolathlon.
Внешний путь `rli_sdk` не меняется при анонимизации PAM, а оригинальное имя
пакета `gigarli` сохраняется внутри архива, как и исходники GM.

Внутри каждой RLI-сессии рабочая папка агента — `/workspace/dumps/workspace`,
как в `containerized`. PAM использует общий RLI runner GigaMetric; разделение
результатов по каталогу и задаче сохраняется на хосте. Исправление прежнего пути
`/workspace/dumps/<catalog>/<task>/workspace` требует нового PAM-образа с
обновлённым GigaMetric. CI получает HEAD `zhmax/toolathlon-rli` при сборке;
RLI-образы задач пересобирать для этой правки не требуется.

Из PAM нужен доступ к API RLI (`RLI_API_BASE`, по умолчанию
`https://rli.sberdevices.ru`) и API модели. Session-клиент автоматически добавляет
RLI hostname в исключения прокси. Внешние сервисы инструментов доступны из RLI;
загрузка ассетов, сборка ENV и развёртывание сервисов на PAM-хосте в профиле отключены.

RLI-адаптер GigaMetric сохраняет дампы в `<storage.base_path>/toolathlon_dumps`,
а логи — в `<storage.logs_path>/toolathlon`. Они входят в штатный архив
`metrics/gigametric/runs/scoring/<validation_id>.tar.gz`.
Перед закрытием RLI-сессии GigaMetric выгружает весь `/workspace/dumps`,
включая workspace, полные ответы инструментов и `run.log`. Перед распаковкой
проверяется SHA-256 архива. При таймауте сначала останавливается процесс задачи
с дочерними процессами. Это копирование после выполнения, а не live mount:
при потере сессии ещё не выгруженные файлы могут быть утрачены.
Операция реализована в файловом транспорте GigaMetric поверх штатного GigaRLI;
отдельной версии SDK не требуется. Существующий PAM-образ нужно пересобрать
с обновлённой веткой GM, RLI-образы задач менять не нужно.

Прямой запуск и сборка ENV описаны в GigaMetric:
`src/benchmarks/toolathlon/README.md` и `src/benchmarks/toolathlon/rli_images/README.md`.
