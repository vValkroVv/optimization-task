# swe_bench_harbor_rli

`contrib/swe_bench_harbor_rli` - адаптер для запуска SWE-bench в инфраструктуре PAM через Harbor и RLI.

Код в этом каталоге помогает запускать SWE-bench через Harbor, но вместо локальных контейнеров использует RLI sessions. Здесь находятся CLI, RLI environment, wrappers для поддерживаемых агентов и retry policy для нестабильных create/step/transfer операций.

## Как устроен запуск

Типовой запуск выглядит так:

1. Вызывается CLI с собранным конфигом:

   ```bash
   python -m swe_bench_harbor_rli.cli run \
     --metrics-dir <artifacts>/swe_bench \
     --config-json '<json>'
   ```

2. `swe_bench_harbor_rli.cli` нормализует конфиг, строит команду `harbor run`, выбирает RLI-compatible agent wrapper и передаёт Harbor кастомную environment implementation:

   ```text
   --environment-import-path swe_bench_harbor_rli.src.environment:RLIEnvironment
   --environment-kwarg rli_config_json=<json>
   ```

3. Harbor читает dataset SWE-bench, планирует trials, запускает agent/verifier lifecycle и вызывает методы `BaseEnvironment`.

4. `RLIEnvironment` реализует эти методы поверх RLI:
   - `start()` создаёт RLI session через `create_session`, дожидается ready status и выполняет bootstrap session (apply_patch shim, `/logs` + `/tests` layout, установка `uv`, hardening apt);
   - `exec()` отправляет shell command через `step`;
   - `upload_file`/`upload_dir` и `download_file`/`download_dir` делают transfer через shell/base64/tar поверх `step`;
   - `stop()` закрывает session через `close_session`.

5. Harbor пишет свои job artifacts в `harbor_jobs`, а CLI агрегирует последний `result.json` в `metrics.json` с `pass_rate`, `passed`, `total`, `errors`.

## Harbor и RLI

SWE-bench запускается через Harbor: он читает dataset, выбирает tasks, управляет parallel trials, retries, verifier, artifacts и итоговым `result.json`.

Этот адаптер собирает `harbor run` и подключает к нему RLI-compatible компоненты:

- `cli.py` строит корректный `harbor run` с RLI environment и agent wrappers;
- `src/environment.py` подменяет Harbor environment lifecycle на RLI sessions API;
- `agents/*` адаптируют поддерживаемые агенты к RLI terminal/session behavior;
- `src/rli_policy.py` задаёт retry/hardening policy для нестабильных create/step/transfer операций.

## Запуск как root

RLI использует task-specific SWE-bench images. Версия `1.0.1` добавляет offline test runtime, а `1.0.2` — окружения поддерживаемых harness-агентов. Недостающую runtime-обвязку создаёт bootstrap при старте session. Поэтому:

- агент и verifier выполняются от `root` (как и в upstream SWE-bench — `test.sh` пишет `/parser.py` и использует `uv` из `/root/.local/bin`, то есть рассчитан только на root);
- недостающую обвязку (`/logs`, `/tests`, `uv`, apply_patch shim) создаёт bootstrap при старте session, а не Docker image.

## Имя RLI environment

Если `env-name` не задан явно, имя строится из SWE-bench instance id по схеме `<env-prefix>-<short_id>`, где `short_id` - часть instance id после `__`:

```text
astropy__astropy-12907  ->  swe-bench-astropy-12907
```

Версия environment берётся из `env-version` (по умолчанию `1.0.2`).

## GigaRLI

`contrib/gigarli` - минимальный клиент и Pydantic-схемы для RLI sessions. `swe_bench_harbor_rli` использует его как транспортный слой и добавляет поверх него логику, которая нужна Harbor и SWE-bench.

`gigarli` фиксирует только базовые операции:

- `create_session`
- `get_session`
- `step`
- `close_session`

В этой интеграции Harbor запускает benchmark, `swe_bench_harbor_rli.src.environment.RLIEnvironment` выполняет Harbor environment API через `gigarli`, а CLI стартует процесс Harbor и забирает метрики.

## Структура

```text
contrib/swe_bench_harbor_rli/
├── README.md
├── __init__.py
├── cli.py
├── agents/
│   ├── agentless/
│   ├── llm_tunnel/
│   ├── mini_swe_agent/
│   │   ├── __init__.py
│   │   ├── mini_swe_agent.py
│   │   └── rli_container_agent.py
│   ├── openhands_sdk/
│   ├── terminus/
│   └── prediction_patch.py
├── local_verifier/
├── src/
│   ├── __init__.py
│   ├── apply_patch_shim.py
│   ├── environment.py
│   ├── errors.py
│   ├── gigarli_bootstrap.py
│   ├── local_rli_test_setup.py
│   ├── rli_policy.py
│   └── verifier.py
├── swe_uv_image_poc/
├── utils/
│   ├── __init__.py
│   └── log_analysis.py
└── tests/
```

## CI pipeline обновления RLI-образов

Pipeline поддерживает task-specific SWE-bench ENV images версий `1.0.1` и
`1.0.2`. PAM по умолчанию использует harness-ready environments `1.0.2`.
Исходные images не изменяются: каждая версия создаётся отдельным производным
слоем. CI выполняет сборку и push в registry, а регистрация ENV в RLI
запускается отдельно после успешной validation.

CI подключён из корневого `.gitlab-ci.yml`:

```yaml
include:
  - local: swe_uv_image_poc/.gitlab-ci.yml
```

Pipeline запускается с переменными:

| Переменная | Значение | Назначение |
| --- | --- | --- |
| `SWE_IMAGE_VERSION` | `1.0.1` / `1.0.2` | Версия собираемых images; по умолчанию `1.0.1`. |
| `SWE_IMAGE_MODE` | `pilot` | Задачи без комментария в `pilot_tasks.txt`. |
| `SWE_IMAGE_MODE` | `batch` | Очередной batch; индекс задаётся через `SWE_BATCH_INDEX`, начиная с `0`. |
| `SWE_IMAGE_MODE` | `all` | Все выбранные manifest-задачи вне pilot-набора. |
| `SWE_MAX_PARALLEL_JOBS` | `5` | Максимальное число параллельных image builds. |

Для `1.0.1` текущий dataset snapshot собран полностью: `500/500` задач
отмечены как готовые в `pilot_tasks.txt`. Состояние `1.0.2` отслеживается
отдельными version-specific task lists и CI artifacts; завершённость `1.0.1`
не означает завершённость `1.0.2`.

Pipeline состоит из следующих этапов:

1. `prepare:swe-dataset` делает sparse checkout
   `harbor-framework/harbor-datasets` на зафиксированной ревизии
   `37db108843a49bb31a592e37a75e2c40dc3f9749` и сохраняет 500 задач как
   artifact.
2. `manifest:swe-images` проверяет наличие всех 500 задач, определяет source и
   target image и генерирует child pipeline.
3. Для `1.0.2` job `build:harness-runtime-1.0.2` собирает общий runtime и
   публикует ссылку, закреплённую SHA digest.
4. Child pipeline создаёт отдельный build job для каждой выбранной задачи и
   ограничивает параллелизм через `resource_group`.
5. `build_push_uv_image.sh` создаёт `1.0.1`, а
   `build_push_image_1_0_2.sh` создаёт `1.0.2`; оба варианта проходят offline
   oracle validation до push в registry.
6. `summary:batch` собирает `batch-summary.json` с результатом, digest и
   стадией ошибки по каждой задаче.

Image наследуется от официального task-specific SWE-bench image и получает:

- `uv`, `zstd` и служебные системные зависимости;
- архив parser runtime
  `/usr/local/share/.dev_logs/test_runtime.tar.zst`;
- wheel cache для offline `pip install`;
- task-specific compatibility constraints;
- локальный HTTP/HTTPS HTTPBin для legacy Requests tasks.

Версия `1.0.2` наследуется от task-specific `1.0.1` и дополнительно получает:

- `tmux` для `terminus-2`;
- Mini-SWE-Agent runtime в `/root/.local`;
- OpenHands SDK venv в `/opt/openhands-sdk-venv`;
- SWE-agent runtime, repository и configs в `/opt/sweagent-*`.

Validation выполняется в два прохода. Install-проход запускается с
`--network none`; oracle test-проход — во внутренней Docker-сети без
внешнего egress, но с рабочим сетевым стеком. Оба прохода используют
`PIP_NO_INDEX=1`, `UV_OFFLINE=1` и запускаются без proxy variables.

Offline validation ограничивает сеть только проверочному контейнеру и не
добавляет firewall/egress-запрет в опубликованный image.

При регистрации `1.0.2` обязателен финальный `sha256` image digest. Скрипт
регистрации задаёт `cpu.requests=1000`, `cpu.limits=2000`,
`memory.requests=4096` и `memory.limits=8192`. Перед обновлением PAM pointer
необходимо сначала влить child-изменения в `swe_bench_harbor_rli/master` и
сверить pointer-кандидат с актуальным SHA remote `master`.

Для доступа к registry CI использует защищённые GitLab variables
`RLI_USER` и `RLI_PASSWORD`. Опциональные `DOCKERHUB_USER` и
`DOCKERHUB_TOKEN` снимают anonymous pull limits для source images.

Task artifacts сохраняются в `swe_uv_image_poc/outputs/validation/<task_id>/`
для `1.0.1` и `outputs/validation-1.0.2/<task_id>/` для `1.0.2`.

### Основные файлы

- `cli.py` - entrypoint `python -m swe_bench_harbor_rli.cli run`. Собирает команду Harbor, прокидывает `rli_config_json`, сохраняет `harbor_command.json`, `harbor_stdout.log` и итоговый `metrics.json`.

- `src/environment.py` - реализация Harbor `BaseEnvironment` поверх RLI sessions. Здесь находятся `create_session`, ожидание ready status, probe warmup, upload/download helpers, apply_patch shim, подготовка `/logs` + `/tests` layout, установка `uv`, bootstrap apt и `step` execution.

- `src/rli_policy.py` - retry policy для create-session, probe/bootstrap steps и transfer operations. Отделяет нормальные ошибки команд от инфраструктурной нестабильности RLI/exec-proxy.

- `src/gigarli_bootstrap.py` - добавляет repo-local `contrib/gigarli/src` в import path, чтобы `gigarli` можно было использовать без отдельной установки пакета.

- `src/apply_patch_shim.py` - shim, который устанавливается в RLI session как `/usr/local/bin/apply_patch`, потому что некоторые агенты ожидают наличие такого инструмента в контейнере.

- `agents/terminus/` - wrapper для `terminus-2`, который проксирует terminal
  commands в RLI session и использует установленный в image `tmux`.

- `agents/mini_swe_agent/` - два варианта Mini-SWE-Agent:
  `mini-swe-agent` работает внутри RLI container через LLM tunnel, а
  `mini-swe-agent-proxy` работает в Harbor process и проксирует команды в RLI.

- `agents/openhands_sdk/` - агент `openhands`: OpenHands SDK работает внутри
  RLI container, а LLM-запросы передаются host-side endpoint через tunnel.

- `agents/agentless/` - агент `agentless`, выполняющий model-side логику в
  Harbor process и команды редактирования через RLI environment.

- `agents/llm_tunnel/` - общий idempotent LLM tunnel для агентов, работающих
  внутри RLI container.

- `tests/` - unit tests для CLI, environment, policy, agents и bootstrap логики.

## Запуск тестов

`gigarli` требует Python 3.11+ (использует `datetime.UTC`). Тесты используют `pytest` + `pytest-asyncio` и импортируют `swe_bench_harbor_rli.*` и `gigarli` без package install, поэтому в `PYTHONPATH` нужны корень `contrib` и `contrib/gigarli/src`:

```bash
cd contrib
PYTHONPATH=$PWD:$PWD/gigarli/src \
  python -m pytest swe_bench_harbor_rli/tests --asyncio-mode=auto
```

## Параметры `DEFAULT_CONFIG`

CLI имеет свой `DEFAULT_CONFIG`; вызывающая сторона передаёт частичный payload, который накладывается поверх дефолтов.

| Параметр | Default | Описание |
| --- | --- | --- |
| `dataset` | `/home/user/swe-bench-verified/datasets/swebench-verified` | Dataset для Harbor (в образ кладётся sparse-checkout'ом из `harbor-framework/harbor-datasets`). Если значение выглядит как локальный путь или absolute path, CLI передаёт `--path`; иначе передаёт `--dataset`. |
| `agent` | `mini-swe-agent` | Агент. CLI поддерживает `mini-swe-agent`, `mini-swe-agent-proxy`, `openhands`, `agentless` и `terminus-2`; PAM config не включает `terminus-2`. Пользовательский `agent-import-path` запрещён. |
| `agent-kwargs` | `{}` | Дополнительные kwargs выбранного агента. |
| `model-name` | `openai/pam-model` | Имя модели для Harbor `--model`. Сам endpoint задаётся через `model-api-base`. |
| `n-concurrent` | `4` | Параллелизм Harbor trials, передаётся как `--n-concurrent`. |
| `agent-timeout` | `None` | Абсолютный timeout выполнения агента в секундах. Если задан, заменяет task-specific agent timeout Harbor; при `None` используется timeout задачи. |
| `agent-timeout-multiplier` | `None` | Множитель task timeout только для выполнения агента; передаётся Harbor как `--agent-timeout-multiplier`. |
| `environment-build-timeout-multiplier` | `None` | Множитель Harbor timeout для `environment.start()`; например, `4.0` увеличивает стандартные 600 секунд до 2400 секунд. |
| `pass-at-k` | `1` | Число независимых Harbor attempts для каждой задачи. Задача считается решённой, если хотя бы один из `k` attempts получил reward `1`. Все `k` attempts выполняются полностью. |
| `api_key` | `dummy` | API key для agent runtime. CLI выставляет его в `OPENAI_API_KEY` и `MSWEA_API_KEY`, если переменные ещё не заданы. |
| `max-tokens` | `32768` | Лимит генерации для агента, передаётся как agent kwarg `max_tokens`. |
| `max-total-output-tokens` | `None` | Мягкий общий лимит output-токенов для всего Harbor job. После достижения агент выбрасывает `OutputTokenBudgetExceededError`, и Harbor отменяет остальные trials. При `n-concurrent > 1` возможно превышение лимита уже выполняющимися trials. |
| `llm-timeout` | `6400` | Timeout одного LLM-запроса в секундах. |
| `env-api-base` | `None` | Явный Base URL RLI API. Если не задан, endpoint выбирается `gigarli`. |
| `env-prefix` | `swe-bench` | Префикс имени RLI environment. Если `env-name` не задан, итоговое имя строится как `<env-prefix>-<short_id>`. |
| `env-version` | `1.0.2` | Версия RLI environment, передаётся в `SessionEnv.version`. |
| `ttl-ms` | `64800000` | Верхняя граница TTL RLI session в миллисекундах (18 часов). Фактическое значение: `min(ttl-ms, (agent_timeout + 600) * 1000)`. `agent_timeout` берётся из `agent-timeout`, а при его отсутствии — из `[agent].timeout_sec` конкретной задачи в `task.toml`. Таймаут verifier и `agent-timeout-multiplier` в расчёте TTL не участвуют. |
| `post-reverify` | `false` | После primary run повторно проверяет нерешённые задачи выбранным grader. |
| `post-reverify-grader` | `rli` | Grader для post-reverify. |
| `verify-ssl` | `false` | Флаг SSL verification для HTTP-клиента RLI. Передаётся в `httpx` как `verify`. |
| `use-pip-proxy` | `false` | Передаётся в `CreateSessionRequest.use_pip_proxy`. Управляет включением pip proxy на стороне RLI session. |
| `timeout-rli` | `1620` | HTTP timeout в секундах для запросов `RLIClient` к RLI API. Это не timeout отдельной benchmark-команды. |
| `request-retries` | `0` | Retry budget для `create_session`; по умолчанию повторные попытки отключены. |
| `request-retry-backoff` | `2.0` | Базовый exponential backoff для `create_session` retries. В policy cap по умолчанию 30 секунд. |
| `transfer-retries` | `2` | Retry budget для upload/download helper commands. Значение `2` означает до 3 попыток всего. |
| `transfer-retry-backoff` | `2.0` | Backoff между transfer retries. Сейчас применяется как `backoff * attempt`. |
| `session-ready-timeout` | `600.0` | Максимальное ожидание, пока `get_session` начнёт возвращать ready/running status. |
| `session-ready-poll-interval` | `1.0` | Интервал polling для `get_session` во время ожидания ready status. |
| `enable-probe-warmup` | `true` | Выполняет probe step после ready status и перед bootstrap. |
| `create-session-jitter-max-sec` | `60.0` | Максимальная случайная задержка перед `create_session`. |
| `probe-retry-backoff-sec` | `2.0` | Базовый exponential backoff для probe step. |
| `command-timeout` | `180` | Default deadline для RLI `step` commands в секундах. CLI конвертирует значение в `RequestLimits.deadline_ms`. |
| `verifier-timeout` | `2400` | Timeout verifier в секундах. |
| `quiet` | `true` | Если `true`, CLI добавляет Harbor флаг `--quiet`. |
| `yes` | `true` | Если `true`, CLI добавляет Harbor флаг `--yes`, чтобы запуск не требовал интерактивного подтверждения. |
| `max-retries` | `1` | Retry count Harbor на уровне trials, передаётся как `--max-retries`. Это отдельно от RLI request/transfer retries. |

## Дополнительные поддерживаемые поля

Эти поля не входят в `DEFAULT_CONFIG`, но CLI и config model их поддерживают:

| Параметр | Описание |
| --- | --- |
| `model-api-base` | OpenAI-compatible endpoint модели. |
| `max-input-tokens` | Лимит входного контекста для агента. |
| `task-name` / `task-id` | Один task или список task names для `--include-task-name`. Также используется как `task_id` для RLI, если задан. |
| `exclude-task-name` | Один task или список task names для `--exclude-task-name`. |
| `n-tasks` | Ограничение числа задач для Harbor (`--n-tasks`). |
| `temperature`, `top-p`, `reasoning-effort` | Agent kwargs для модели. |
| `env-name` | Полное имя RLI environment. Если задано, переопределяет схему `<env-prefix>-<short_id>`. |
| `seed` | Seed, передаваемый в RLI config, если нужен воспроизводимый session setup. |
| `probe-warmup-command` | Команда для probe step. Должна быть дешёвой и безопасной. |
| `rli-trace-path` | Путь JSONL trace для requests `gigarli.RLIClient`. По умолчанию CLI пишет trace в `<metrics-dir>/rli_trace.jsonl`. |
| `disable-verification` | Если `true`, CLI добавляет Harbor `--disable-verification`. |
| `session-config` | Дополнительный session config, пробрасываемый в `rli_config_json` для environment-specific расширений. |

## Артефакты запуска

В `--metrics-dir` CLI пишет:

- `harbor_command.json` - фактическая команда Harbor;
- `harbor_stdout.log` - stdout/stderr Harbor процесса;
- `rli_trace.jsonl` - trace RLI requests, если не переопределён `rli-trace-path`;
- `rli_infrastructure_report.json` - корреляция infrastructure errors Harbor с
  событиями RLI trace;
- `harbor_jobs/` - job artifacts Harbor;
- `metrics.json` - агрегированные метрики (`pass_rate`, `passed`, `total`, `errors`, `job_result_path`).
