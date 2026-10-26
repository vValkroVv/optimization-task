# terminal_bench

`contrib/terminal_bench` - адаптер для запуска Terminal-Bench в инфраструктуре PAM через Harbor и RLI.

Код в этом каталоге помогает запускать upstream Terminal-Bench через Harbor, но вместо локальных контейнеров использует RLI sessions. Здесь находятся CLI, RLI environment, wrappers для поддерживаемых агентов и retry policy для нестабильных create/step/transfer операций.

## Как устроен запуск

Типовой запуск из `node_runner` выглядит так:

1. `TerminalBenchEvaluator` собирает конфигурацию запуска и вызывает:

   ```bash
   /home/user/terminal_bench_env/bin/python -m terminal_bench.cli run \
     --metrics-dir <artifacts>/terminal_bench \
     --config-json '<json>'
   ```

2. `terminal_bench.cli` нормализует конфиг, строит команду `harbor run`, выбирает RLI-compatible agent wrapper и передаёт Harbor кастомную environment implementation:

   ```text
   --environment-import-path terminal_bench.src.environment:RLIEnvironment
   --environment-kwarg rli_config_json=<json>
   ```

3. Harbor читает dataset Terminal-Bench, планирует trials, запускает agent/verifier lifecycle и вызывает методы `BaseEnvironment`.

4. `RLIEnvironment` реализует эти методы поверх RLI:
   - `start()` создаёт RLI session через `create_session`;
   - `exec()` отправляет shell command через `step`;
   - `upload_file`/`upload_dir` и `download_file`/`download_dir` делают transfer через shell/base64/tar поверх `step`;
   - `stop()` закрывает session через `close_session`.

5. Harbor пишет свои job artifacts в `harbor_jobs`, а CLI агрегирует последний `result.json` в `metrics.json` с `pass_rate`, `passed`, `total`, `errors`.

## Harbor и RLI

Terminal-Bench запускается через Harbor: он читает dataset, выбирает tasks, управляет parallel trials, retries, verifier, artifacts и итоговым `result.json`.

Этот адаптер собирает `harbor run` и подключает к нему RLI-compatible компоненты:

- `cli.py` строит корректный `harbor run` с RLI environment и agent wrappers;
- `src/environment.py` подменяет Harbor environment lifecycle на RLI sessions API;
- `agents/*` адаптируют поддерживаемые агенты к RLI terminal/session behavior;
- `src/rli_policy.py` задаёт retry/hardening policy для нестабильных create/step/transfer операций.

## GigaRLI

`contrib/gigarli` - минимальный клиент и Pydantic-схемы для RLI sessions. `terminal_bench` использует его как транспортный слой и добавляет поверх него логику, которая нужна Harbor и Terminal-Bench.

`gigarli` фиксирует только базовые операции:

- `create_session`
- `get_session`
- `step`
- `close_session`

В этой интеграции Harbor запускает benchmark, `terminal_bench.src.environment.RLIEnvironment` выполняет Harbor environment API через `gigarli`, а `node_runner` стартует CLI и забирает метрики.

## Структура

```text
contrib/terminal_bench/
├── README.md
├── __init__.py
├── cli.py
├── agents/
│   ├── mini_swe_agent/
│   │   ├── __init__.py
│   │   └── mini_swe_agent.py
│   └── terminus/
│       ├── __init__.py
│       ├── terminus_agent.py
│       └── tmux_session.py
├── src/
│   ├── __init__.py
│   ├── apply_patch_shim.py
│   ├── environment.py
│   ├── gigarli_bootstrap.py
│   └── rli_policy.py
├── utils/
│   ├── __init__.py
│   └── log_analysis.py
└── tests/
```

### Основные файлы

- `cli.py` - entrypoint `python -m terminal_bench.cli run`. Собирает команду Harbor, прокидывает `rli_config_json`, сохраняет `harbor_command.json`, `harbor_stdout.log` и итоговый `metrics.json`.

- `src/environment.py` - реализация Harbor `BaseEnvironment` поверх RLI sessions. Здесь находятся `create_session`, ожидание ready status, probe warmup, upload/download helpers, path rewrites для `/tests` и `/logs`, `apply_patch` shim, bootstrap apt и `step` execution.

- `src/rli_policy.py` - retry policy для create-session, probe/bootstrap steps и transfer operations. Отделяет нормальные ошибки команд от инфраструктурной нестабильности RLI/exec-proxy.

- `src/gigarli_bootstrap.py` - добавляет repo-local `contrib/gigarli/src` в import path, чтобы `gigarli` можно было использовать без отдельной установки пакета.

- `src/apply_patch_shim.py` - shim, который устанавливается в RLI session как `/usr/local/bin/apply_patch`, потому что некоторые агенты ожидают наличие такого инструмента в контейнере.

- `agents/terminus/` - RLI wrapper для `terminus-2`, включая tmux/session handling.

- `agents/mini_swe_agent/` - RLI wrapper для `mini-swe-agent`.

- `tests/` - unit tests для CLI, environment, policy, agents и bootstrap логики.

## Runtime layout в Docker image

В Dockerfile runtime собирается так:

- Harbor установлен внутри контейнера в `/home/user/terminal_bench_env` вместе с зависимостями;
- upstream dataset Terminal-Bench checkout лежит в `/home/user/terminal-bench-2`;
- этот адаптер копируется в `/home/user/terminal_bench`;
- `contrib/gigarli` копируется в `/home/user/contrib/gigarli`;
- `PYTHONPATH` содержит `/home/user/contrib/gigarli/src`, `/home/user/contrib` и `/home/user`, чтобы imports `gigarli` и `terminal_bench.*` работали без package install.

## Конфигурация запуска

Для production-запуска через `node_runner` источником дефолтов является `TerminalBenchEvaluator._DEFAULT_RUN_CONFIG` из `node_runner/src/metrics/evaluators/terminal_bench.py`. CLI также имеет свой `DEFAULT_CONFIG` для прямых запусков, но `node_runner` передаёт в CLI уже собранный payload.

`TerminalBenchEvaluator._STARTUP_HARDENING_DEFAULTS` дополнительно применяет startup hardening, если соответствующие поля не были явно заданы пользователем в конфиге:

| Параметр | Default | Описание |
| --- | --- | --- |
| `request-retries` | `3` | Количество retries для `POST /v1/sessions` после первой попытки. Значение `3` означает до 4 попыток всего. |
| `enable-probe-warmup` | `true` | После ready status выполнить короткий probe step перед bootstrap. Помогает отловить раннюю нестабильность session/exec-proxy. |
| `probe-warmup-command` | `pwd` | Команда probe warmup, запускается внутри RLI session от `root`. |
| `create-session-jitter-max-sec` | `3.0` | Случайная задержка от `0` до этого значения перед первой попыткой `create_session`, чтобы не создавать burst при параллельном старте. |
| `probe-retry-backoff-sec` | `2.0` | Базовый backoff для retry probe step. Backoff экспоненциальный и ограничен 8 секундами. |

## Параметры `_DEFAULT_RUN_CONFIG`

| Параметр | Default | Описание |
| --- | --- | --- |
| `dataset` | `/home/user/terminal-bench-2` | Dataset для Harbor. Если значение выглядит как локальный путь или absolute path, CLI передаёт `--path`; иначе передаёт `--dataset`. |
| `agent` | `terminus-2` | Поддерживаемый агент. Разрешены `terminus-2` и `mini-swe-agent`; CLI сам выбирает нужный `--agent-import-path`. Пользовательский `agent-import-path` запрещён. |
| `model-name` | `openai/pam-model` | Имя модели для Harbor `--model`. Сам endpoint задаётся через `model-api-base`, который добавляет `TerminalBenchEvaluator`. |
| `n-concurrent` | `4` | Параллелизм Harbor trials, передаётся как `--n-concurrent`. |
| `agent-timeout` | `None` | Абсолютный timeout выполнения агента в секундах. Если задан, заменяет task-specific agent timeout Harbor; при `None` используется timeout задачи. |
| `pass-at-k` | `1` | Число независимых Harbor attempts для каждой задачи. Задача считается решённой, если хотя бы один из `k` attempts получил reward `1`. Все `k` attempts выполняются полностью. |
| `agent-timeout-multiplier` | `None` | Множитель task timeout только для выполнения агента; передаётся Harbor как `--agent-timeout-multiplier`. |
| `environment-build-timeout-multiplier` | `None` | Множитель Harbor timeout для `environment.start()`; например, `4.0` увеличивает стандартные 600 секунд до 2400 секунд. |
| `api_key` | `dummy` | API key для agent runtime. CLI выставляет его в `OPENAI_API_KEY` и `MSWEA_API_KEY`, если переменные ещё не заданы. |
| `max-tokens` | `32768` | Лимит генерации для агента, передаётся как agent kwarg `max_tokens`. |
| `max-total-output-tokens` | `None` | Мягкий общий лимит output-токенов для всего Harbor job. После достижения агент выбрасывает `OutputTokenBudgetExceededError`, и Harbor отменяет остальные trials. При `n-concurrent > 1` возможно превышение лимита уже выполняющимися trials. |
| `env-api-base` | `None` | Явный Base URL RLI API. Если не задан, endpoint выбирается `gigarli`. |
| `env-prefix` | `terminal-bench` | Префикс имени RLI environment. Если `env-name` не задан, итоговое имя строится как `<env-prefix>-<task_id>`. |
| `env-version` | `1.0.0` | Версия RLI environment, передаётся в `SessionEnv.version`. |
| `ttl-ms` | `64800000` | Верхняя граница TTL RLI session в миллисекундах (18 часов). Фактическое значение: `min(ttl-ms, (agent_timeout + 600) * 1000)`. `agent_timeout` берётся из `agent-timeout`, а при его отсутствии — из `[agent].timeout_sec` конкретной задачи в `task.toml`. Таймаут verifier и `agent-timeout-multiplier` в расчёте TTL не участвуют. |
| `verify-ssl` | `false` | Флаг SSL verification для HTTP-клиента RLI. Передаётся в `httpx` как `verify`. |
| `use-pip-proxy` | `false` | Передаётся в `CreateSessionRequest.use_pip_proxy`. Управляет включением pip proxy на стороне RLI session, если платформа это поддерживает. |
| `timeout-rli` | `180` | HTTP timeout в секундах для запросов `RLIClient` к RLI API. Это не timeout отдельной benchmark-команды. |
| `request-retries` | `3` | Retry budget для `create_session`. Повторяются transport errors, retryable RLI errors и HTTP `408/429/5xx`. |
| `request-retry-backoff` | `2.0` | Базовый exponential backoff для `create_session` retries. В policy cap по умолчанию 30 секунд. |
| `transfer-retries` | `2` | Retry budget для upload/download helper commands. Значение `2` означает до 3 попыток всего. |
| `transfer-retry-backoff` | `2.0` | Backoff между transfer retries. Сейчас применяется как `backoff * attempt`. |
| `session-ready-timeout` | `600.0` | Максимальное ожидание, пока `get_session` начнёт возвращать ready/running status. |
| `session-ready-poll-interval` | `1.0` | Интервал polling для `get_session` во время ожидания ready status. |
| `enable-probe-warmup` | `true` | Включает отдельный probe step после ready status и перед bootstrap-командами. |
| `probe-warmup-command` | `pwd` | Команда для probe step. Должна быть дешёвой и безопасной. |
| `create-session-jitter-max-sec` | `3.0` | Максимальная случайная задержка перед `create_session`. |
| `probe-retry-backoff-sec` | `2.0` | Базовый backoff для повторов probe step при инфраструктурной ошибке. |
| `command-timeout` | `600` | Default deadline для RLI `step` commands в секундах. CLI конвертирует значение в `RequestLimits.deadline_ms`. |
| `quiet` | `true` | Если `true`, CLI добавляет Harbor флаг `--quiet`. |
| `yes` | `true` | Если `true`, CLI добавляет Harbor флаг `--yes`, чтобы запуск не требовал интерактивного подтверждения. |
| `max-retries` | `1` | Retry count Harbor на уровне trials, передаётся как `--max-retries`. Это отдельно от RLI request/transfer retries. |

## Дополнительные поддерживаемые поля

Эти поля не входят в `_DEFAULT_RUN_CONFIG`, но CLI и config model их поддерживают:

| Параметр | Описание |
| --- | --- |
| `model-api-base` | OpenAI-compatible endpoint модели. `TerminalBenchEvaluator` задаёт его как `http://127.0.0.1:<api_port>/v1`. |
| `task-name` / `include-task-name` | Один task или список task names для `--include-task-name`. Также используется как `task_id` для RLI, если задан. |
| `exclude-task-name` | Один task или список task names для `--exclude-task-name`. |
| `n-tasks` / `limit` | Ограничение числа задач для Harbor. |
| `temperature`, `top-p`, `reasoning-effort` | Agent kwargs для модели. |
| `env-name` | Полное имя RLI environment. Если задано, переопределяет схему `<env-prefix>-<task_id>`. |
| `seed` | Seed, передаваемый в RLI config, если нужен воспроизводимый session setup. |
| `rli-trace-path` | Путь JSONL trace для requests `gigarli.RLIClient`. По умолчанию CLI пишет trace в `<metrics-dir>/rli_trace.jsonl`. |
| `disable-verification` | Если `true`, CLI добавляет Harbor `--disable-verification`. |
| `agent-kwargs` | Дополнительные kwargs для выбранного агента. Значения сериализуются в `--agent-kwarg key=value`. |
| `session-config` | Дополнительный session config, пробрасываемый в `rli_config_json` для environment-specific расширений. |

## Артефакты запуска

В `--metrics-dir` CLI пишет:

- `harbor_command.json` - фактическая команда Harbor;
- `harbor_stdout.log` - stdout/stderr Harbor процесса;
- `rli_trace.jsonl` - trace RLI requests, если не переопределён `rli-trace-path`;
- `harbor_jobs/` - job artifacts Harbor;
- `metrics.json` - агрегированные метрики для `node_runner`.

`node_runner` читает `metrics.json` и публикует метрику:

```text
terminal_bench/resolved_percent = pass_rate * 100
```
