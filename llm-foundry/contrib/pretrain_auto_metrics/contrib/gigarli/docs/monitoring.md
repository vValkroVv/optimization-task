# RLI Monitoring Helper

Этот документ описывает ручной скрипт мониторинга RLI:
`scripts/rli_monitoring.py`.

Скрипт нужен как diagnostic/ops helper для пользователя, который хочет быстро
посмотреть состояние RLI-сервиса. Он не должен запускаться при выполнении
benchmark-ов и не является частью runtime path benchmark adapters.

## Цель

Скрипт должен давать компактный снимок состояния RLI:
- доступен ли monitoring API;
- сколько read-only запросов успешно выполнилось;
- базовые dashboard-метрики;
- последние ошибки, если пользователь не отключил этот запрос;
- список failed monitoring requests, если какие-то endpoint-ы недоступны.

Ожидаемые основные источники:
- `GET /v1/system/dashboard?window=...`
- `GET /v1/errors/recent?limit=...`

Оба endpoint-а используются только для чтения.

## Граница ответственности

Скрипт относится к `scripts/`, а не к публичному Python API пакета.

Он не должен:
- экспортироваться через `gigarli.__init__`;
- расширять минимальный sessions-контракт;
- использоваться внутри benchmark execution path;
- влиять на `RLIClient`;
- требовать дополнительных зависимостей сверх стандартной библиотеки Python.

Он может:
- использоваться вручную из debug shell;
- помогать при разборе деградаций RLI;
- сохранять JSON snapshots по явному флагу;
- печатать человекочитаемый summary в stdout.

## Поведение по умолчанию

По умолчанию скрипт должен только печатать summary в stdout и не создавать
файлы.

JSON snapshot создаётся только если пользователь явно передал флаг сохранения,
например:

```bash
PYTHONPATH=src python3 scripts/rli_monitoring.py --save-json
```

Это важно, чтобы ручная диагностика не оставляла артефакты в репозитории без
явного намерения пользователя.

## CLI

Базовый запуск:

```bash
PYTHONPATH=src python3 scripts/rli_monitoring.py
```

Сохранить JSON snapshot:

```bash
PYTHONPATH=src python3 scripts/rli_monitoring.py --save-json
```

Сохранить JSON snapshot в отдельную директорию:

```bash
PYTHONPATH=src python3 scripts/rli_monitoring.py \
  --save-json \
  --output-dir /tmp/rli_monitoring_snapshots
```

Периодический мониторинг:

```bash
PYTHONPATH=src python3 scripts/rli_monitoring.py \
  --iterations 0 \
  --interval 60
```

Проверить другой RLI gateway:

```bash
PYTHONPATH=src python3 scripts/rli_monitoring.py \
  --base-url https://rli.sberdevices.ru
```

Ожидаемые параметры:
- `--base-url` — RLI gateway base URL.
- `--window` — окно dashboard-метрик, например `5m`, `15m`, `1h`.
- `--timeout` — HTTP timeout в секундах.
- `--insecure` / `--verify-tls` — управление TLS verification.
- `--use-proxy` — использовать `HTTP_PROXY` / `HTTPS_PROXY` из окружения.
- `--iterations` — количество снимков, `0` означает бесконечно.
- `--interval` — пауза между снимками.
- `--errors-recent-limit` — лимит для `/v1/errors/recent`.
- `--no-recent-errors` — не вызывать `/v1/errors/recent`.
- `--save-json` — записывать JSON snapshots.
- `--output-dir` — директория для JSON snapshots.
- `--quiet` — не печатать summary; валиден только вместе с `--save-json`.

Если `--quiet` передан без `--save-json`, скрипт должен завершаться с ошибкой
валидации аргументов, иначе пользователь не увидит результата.

## Summary

Summary в stdout должен оставаться компактным и пригодным для быстрой оценки:

```text
[2026-06-04T10:00:00Z] RLI monitoring snapshot
base_url: https://rli.sberdevices.ru
elapsed_ms: 123
requests: 2/2 ok
dashboard:
  active_sessions: 42
  sessions_by_status: {"running": 30, "creating": 5, "failed": 2, "closed": 100}
  avg_step_latency_ms: 180
  error_rate_percent_1m: 0.05
recent_errors: 3
```

Если часть запросов не прошла, summary должен явно показывать failed requests:

```text
failed_requests:
  errors_recent: status=404 code=NOT_FOUND url=https://.../v1/errors/recent?limit=20
```

## JSON Snapshots

При `--save-json` скрипт должен писать полный snapshot в JSON:
- `collected_at`
- `base_url`
- `elapsed_ms`
- `responses`
- `summary`

Рекомендуемое имя файла:

```text
rli_monitoring_2026-06-04T10-00-00Z.json
```

Также можно обновлять `latest.json` в той же директории.

## Поля Summary

Для dashboard достаточно выбрать наиболее полезные поля:
- `active_sessions`
- `sessions_by_status`
- `avg_step_latency_ms`
- `error_rate_percent_1m`

Для recent errors достаточно компактного набора:
- `timestamp`
- `http_status`
- `code`
- `category`
- `endpoint`
- `request_id`
- `id`
- `session_id`
- `message`

Если RLI вернул дополнительные поля, скрипт может оставить их в полном JSON
snapshot, но не обязан печатать их в stdout.

## Ошибки

Monitoring script должен быть tolerant к частично недоступному API:
- HTTP error одного endpoint-а не должен скрывать результат остальных;
- JSON body ошибки надо сохранить в `responses`;
- compact summary должен показать `status_code`, `error_code`, `url`;
- неожиданные transport ошибки должны возвращаться как failed request, а не как
  traceback по умолчанию.

Исключение — ошибки CLI-аргументов. Они должны завершать процесс с кодом `2`.

## Связь с client-side trace

`RLIClient(trace_path=...)` и `scripts/rli_monitoring.py` решают разные задачи.

Client-side trace:
- пишет локальный JSONL request/response trace для конкретного consumer-а;
- помогает разбирать конкретные `create/get/step/close` операции;
- находится на transport level sessions SDK.

Monitoring helper:
- читает platform-level monitoring endpoint-ы;
- показывает состояние RLI сервиса в целом;
- запускается только вручную.

Поэтому monitoring helper не заменяет `trace_path` и не должен смешиваться с
ним в контракте.
