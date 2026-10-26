# Sessions Contract

Этот документ фиксирует минимальный sessions-контракт для GigaRLI.

Первая итерация контракта покрывает 4 lifecycle-операции:
- `create_session`
- `get_session`
- `step`
- `close_session`

Для `step` используется response envelope с итоговым результатом команды.

Этот контракт используется как основа для:
- Pydantic-моделей в `src/gigarli/schemas/*`;
- клиентского SDK в `src/gigarli/client.py`;
- benchmark adapters, которые строят свою логику поверх этих 4 primitive
  operations.

На текущем этапе этот контракт уже реализован в коде:
- request/response models в `gigarli.schemas.common` и `gigarli.schemas.env`;
- client-side SDK в `gigarli.client`;
- smoke validation в `scripts/smoke_sessions.py`.
- optional client-level JSONL trace via `RLIClient(trace_path=...)`.

Репозиторий использует `src` layout, поэтому importable package лежит в
`src/gigarli`, а локальный запуск должен идти через `PYTHONPATH=src`.

Контракт ниже описан по фактически наблюдаемому API из
[rli_sessions_smoke.trace.jsonl](/workspace-SR008.nfs2/zhmax/dev/terminal_bench_dev/pretrain-auto-metrics/contrib/gigarli/rli_sessions_smoke.trace.jsonl).

## Базовые принципы

- `gigarli` описывает клиентский контракт к raw RLI API, а не отдельный
  промежуточный HTTP-service layer.

- Базовый request/response trace не является частью wire protocol, но уже
  поддерживается клиентом как transport-level observability feature.

- Все успешные ответы используют единый envelope:
  - `status = "ok"`
  - `result` — тело ответа
  - `meta` — метаданные запроса

- Все ответы с ошибкой используют единый envelope:
  - `status = "error"`
  - `error` — описание ошибки
  - `meta` — метаданные запроса

- `session_id` — основной идентификатор session lifecycle.

- Исполнение команд происходит через `POST /v1/sessions/step`.

- В клиентском коде `session_id` всегда должен передаваться явно.
  Клиент не должен хранить "текущую сессию" как скрытое глобальное состояние.

- При передаче `trace_path` клиент пишет JSONL события для:
  - `create_session`
  - `get_session`
  - `step`
  - `close_session`

## Общий response envelope

### Success

```json
{
  "status": "ok",
  "result": {},
  "meta": {
    "version": "rli-api-gateway@2.1.1",
    "request_id": "uuid",
    "session_id": "uuid",
    "runtime_ms": 123,
    "schema_version": "1.0"
  }
}
```

Замечания:
- `meta.session_id` присутствует не всегда;
- например в `create_session` он может отсутствовать, а в `get/step/close` — присутствовать.

### Error

```json
{
  "status": "error",
  "error": {
    "id": "uuid",
    "code": "INVALID_ARGUMENT",
    "category": "client_error",
    "message": "Human-readable error message",
    "retryable": false
  },
  "meta": {
    "version": "rli-api-gateway@2.1.1",
    "request_id": "uuid",
    "runtime_ms": 0,
    "schema_version": "1.0"
  }
}
```

Кроме минимальных полей, текущий RLI также возвращает расширенные поля ошибки.
Они уже поддерживаются в `ErrorDetails`:
- `title_ru`
- `title_en`
- `message_ru`
- `message_en`
- `user_action_ru`
- `user_action_en`
- `retry_strategy`
- `docs_url`

## Client-side trace

Это не часть HTTP API, а локальное поведение `RLIClient`.

При передаче `trace_path` клиент пишет JSONL записи для transport-level
операций:
- `create_session`
- `get_session`
- `step`
- `close_session`

Для каждой операции записывается итоговый request/response.

Каждое событие содержит:
- `ts`
- `stage`
- `success`
- `request`
- `response`
- `error` (только для failed client path)

Это нужно для:
- reverse-engineering observed API;
- step-level debugging в benchmark adapters;
- разборов request/response без дублирования trace-логики в consumers.

## Documented error codes

Источник:
- `GigaRLI v1.3.0.pdf`, section `Error Codes Reference`

Замечания:
- этот раздел описывает общую платформенную taxonomy, а не только minimal
  sessions API;
- поэтому здесь есть коды, относящиеся не только к `sessions`, но и к
  `verify`/platform-level flows;
- в runtime `error.code` должен оставаться строкой, а не жёстким enum, чтобы не
  ломаться при появлении новых кодов.

### Error categories

| Category | When | Suggested action |
| --- | --- | --- |
| `infrastructure_error` | k8s / ARI / network failed | create a new session |
| `execution_error` | command did not complete successfully | change the request |
| `client_error` | invalid request | fix the request |
| `verifier_error` | verifier pool failed | wait or create a new session |
| `internal_error` | platform bug | contact support with `error_id` |

### infrastructure_error

| Code | HTTP | Retry strategy | Meaning |
| --- | --- | --- | --- |
| `POD_KILLED_BY_POLICY` | `503` | `create_new_session` | Pod was killed by k8s policy enforcement |
| `POD_OOM_KILLED` | `503` | `create_new_session` | Pod was OOM-killed after exceeding memory limit |
| `POD_NEVER_REACHED` | `503` | `create_new_session` | Pod never became reachable from the start |
| `POD_SHELL_BROKEN` | `503` | `create_new_session` | Shell did not return a prompt after interrupt |
| `POD_TERMINATED` | `503` | `create_new_session` | Generic pod terminated condition |
| `POD_UNREACHABLE` | `503` | `create_new_session` | Generic pod unreachable condition |
| `ARI_CONTROLLER_UNAVAILABLE` | `503` | `wait_and_retry_step` | ARI returned `5xx`, circuit breaker opened, or transport failed |
| `EXEC_PROXY_UNREACHABLE` | `503` | `wait_and_retry_step` | Backend could not reach exec-proxy |
| `NETWORK_PARTITION` | `503` | `wait_and_retry_step` | TCP reset, connection reset, or generic I/O failure |
| `UNAVAILABLE` | `503` | `wait_and_retry_step` | Generic temporary unavailability |
| `OOM_KILLED` | `503` | `create_new_session` | Legacy process OOM-killed condition |

### execution_error

| Code | HTTP | Retry strategy | Meaning |
| --- | --- | --- | --- |
| `EXEC_TIMEOUT` | `504` | `split_command_or_increase_timeout` | Exec timeout was exceeded |
| `TIMEOUT` | `504` | `wait_and_retry_step` | Legacy generic timeout |
| `OUTPUT_TRUNCATED` | `507` | `fix_request` | Output exceeded `MAX_OUTPUT_BYTES` |
| `COMMAND_REJECTED` | `400` | `fix_request` | Command was rejected during sanitization |
| `WORKING_DIR_INVALID` | `400` | `fix_request` | Invalid or unsafe working directory |
| `IMAGE_INVALID` | `400` | `fix_request` | Invalid or not whitelisted image |

### client_error

| Code | HTTP | Retry strategy | Meaning |
| --- | --- | --- | --- |
| `INVALID_ARGUMENT` | `400` | `fix_request` | Request field validation failed |
| `UNPROCESSABLE_ENTITY` | `422` | `fix_request` | Invalid data format |
| `NOT_FOUND` | `404` | `fix_request` | Resource not found |
| `ALREADY_EXISTS` | `409` | `fix_request` | Duplicate resource |
| `PERMISSION_DENIED` | `403` | `not_recommended` | Caller has no permission |
| `SESSION_NOT_FOUND` | `404` | `create_new_session` | `session_id` does not exist |
| `SESSION_CLOSED` | `410` | `create_new_session` | Session is already closed |
| `SESSION_EXPIRED` | `410` | `create_new_session` | Session TTL has expired |
| `SESSION_NOT_READY` | `409` | `wait_and_retry_step` | Session is still creating or already failed |
| `RATE_LIMITED` | `429` | `wait_and_retry_step` | RPS limit was exceeded |
| `BAD_REQUEST_FLOOD` | `429` | `wait_and_retry_step` | Too many bad requests in a short period |
| `CLIENT_DISCONNECT` | `499` | `increase_client_timeout` | Client timed out before server completed the request |
| `RESOURCE_EXHAUSTED` | `429` | `wait_and_retry_step` | Backend has no free resources |
| `QUOTA_EXCEEDED` | `429` | `wait_and_retry_step` | Client quota was exceeded |

### verifier_error

| Code | HTTP | Retry strategy | Meaning |
| --- | --- | --- | --- |
| `VERIFIER_NOT_FOUND` | `404` | `fix_request` | Verifier is not registered |
| `VERIFIER_POD_KILLED` | `503` | `wait_and_retry_step` | Verifier pod died during execution |
| `VERIFIER_TIMEOUT` | `504` | `wait_and_retry_step` | Verifier did not complete in time |

### internal_error

| Code | HTTP | Retry strategy | Meaning |
| --- | --- | --- | --- |
| `INTERNAL` | `500` | `wait_and_retry_step` | Unhandled internal exception |
| `DATABASE_ERROR` | `503` | `wait_and_retry_step` | Database is unavailable |
| `S3_STORAGE_ERROR` | `503` | `wait_and_retry_step` | S3 storage is unavailable |

## 1. Create Session

### Request

`POST /v1/sessions`

```json
{
  "env": {
    "name": "terminal-bench-modernize-scientific-stack",
    "version": "1.0.0",
    "task_id": "modernize-scientific-stack"
  },
  "ttl_ms": 3600000,
  "use_pip_proxy": true
}
```

### Request fields

- `env.name: string`
- `env.version: string`
- `env.task_id: string`
- `ttl_ms?: integer`
- `use_pip_proxy?: boolean`

### Success response

HTTP status:
- `201 Created` observed

```json
{
  "status": "ok",
  "result": {
    "session_id": "5715df92-8658-4fa0-92af-6e146789e679",
    "created_at": "2026-05-30T12:17:21.317239741Z",
    "backend_ref": "https://arirli-app-preprod.apps.advdarlpd.sberdevices.ru/5715df92-8658-4fa0-92af-6e146789e679",
    "backend_type": "SWE",
    "shell_mode": "current",
    "tool_schemas": [],
    "terminal_url": "https://arirli-app-preprod.apps.advdarlpd.sberdevices.ru/5715df92-8658-4fa0-92af-6e146789e679/terminal",
    "workspace_path": "/"
  },
  "meta": {
    "version": "rli-api-gateway@2.1.1",
    "request_id": "0d3f1cb5-45c2-4332-b0ea-270ba7eddc48",
    "runtime_ms": 12309,
    "schema_version": "1.0"
  }
}
```

### Result fields

- `session_id: string`
- `created_at: string`
- `backend_ref: string`
- `backend_type: string`
- `shell_mode: string`
- `tool_schemas: array`
- `terminal_url: string`
- `workspace_path: string`

## 2. Get Session

### Request

`GET /v1/sessions/{session_id}`

Пример:

```text
GET /v1/sessions/5715df92-8658-4fa0-92af-6e146789e679
```

### Success response

HTTP status:
- `200 OK` observed

```json
{
  "status": "ok",
  "result": {
    "status": "RUNNING",
    "session_id": "5715df92-8658-4fa0-92af-6e146789e679",
    "env_name": "terminal-bench-modernize-scientific-stack",
    "env_version": "1.0.0",
    "task_id": "modernize-scientific-stack",
    "ttl_ms": 3600000,
    "backend_type": "SWE",
    "backend_ref": "https://arirli-app-preprod.apps.advdarlpd.sberdevices.ru/5715df92-8658-4fa0-92af-6e146789e679",
    "shell_mode": "current",
    "created_at": "2026-05-30T12:17:21.317240Z",
    "last_seen": "2026-05-30T12:17:21.317240Z"
  },
  "meta": {
    "version": "rli-api-gateway@2.1.1",
    "request_id": "6a2cc2e5-c0a7-4d7c-97db-1509f25fc923",
    "session_id": "5715df92-8658-4fa0-92af-6e146789e679",
    "runtime_ms": 1,
    "schema_version": "1.0"
  }
}
```

### Result fields

- `status: string`
  - observed values:
    - `RUNNING`
    - `CLOSED`
- `session_id: string`
- `env_name: string`
- `env_version: string`
- `task_id: string`
- `ttl_ms: integer`
- `backend_type: string`
- `backend_ref: string`
- `shell_mode: string`
- `created_at: string`
- `last_seen: string`
- `closed_at?: string`

## 3. Step

### Request

`POST /v1/sessions/step`

```json
{
  "session_id": "5715df92-8658-4fa0-92af-6e146789e679",
  "mode": "single",
  "actions": [
    {
      "type": "env_action",
      "name": "bash",
      "args": {
        "cmd": "set -x; pwd; whoami; echo HOME=$HOME"
      }
    }
  ],
  "limits": {
    "deadline_ms": 60000
  }
}
```

### Request fields

- `session_id: string`
- `mode: string`
  - observed:
    - `single`
- `actions: array`
- `actions[].type: string`
  - observed working value:
    - `env_action`
- `actions[].name: string`
  - observed working value:
    - `bash`
- `actions[].args: object`
- `actions[].args.cmd: string`
- `limits?: object`
- `limits.deadline_ms?: integer`

### Success response

HTTP status:
- `200 OK` observed

```json
{
  "status": "ok",
  "result": {
    "observation": {
      "output": "stdout:\\n\\n+ pwd\\n/\\n+ whoami\\nroot\\n+ echo HOME=/root\\nHOME=/root\\n\\nstderr:\\nroot@rli-5715df92--e679-dep-5944ccc6fb-nhs4n:/app#\\n"
    },
    "events": [
      {
        "action_index": 0,
        "stdout_lines": ["", "+ pwd", "/", "+ whoami", "root", "+ echo HOME=/root", "HOME=/root"],
        "stderr_lines": ["root@rli-5715df92--e679-dep-5944ccc6fb-nhs4n:/app#"],
        "stdout_meta": {
          "total_lines": 7,
          "returned_lines": 7,
          "total_bytes": 52
        },
        "stderr_meta": {
          "total_lines": 1,
          "returned_lines": 1,
          "total_bytes": 50
        },
        "stdout_ref": "\\n+ pwd\\n/\\n+ whoami\\nroot\\n+ echo HOME=/root\\nHOME=/root\\n",
        "stderr_ref": "root@rli-5715df92--e679-dep-5944ccc6fb-nhs4n:/app#",
        "exit_code": 0,
        "duration_ms": 511
      }
    ],
    "terminated": false,
    "agent_id": "default",
    "applied_limits": {
      "deadline_ms": 60000
    }
  },
  "meta": {
    "version": "rli-api-gateway@2.1.1",
    "request_id": "5e909f70-4618-467c-9912-19184710a6ad",
    "session_id": "5715df92-8658-4fa0-92af-6e146789e679",
    "runtime_ms": 514,
    "schema_version": "1.0"
  }
}
```

### Result fields

- `observation.output: string`
- `events: array`
- `events[].action_index: integer`
- `events[].stdout_lines: string[]`
- `events[].stderr_lines: string[]`
- `events[].stdout_meta: object`
- `events[].stderr_meta: object`
- `events[].stdout_ref: string`
- `events[].stderr_ref: string`
- `events[].exit_code: integer`
- `events[].duration_ms: integer`
- `terminated: boolean`
- `agent_id?: string`
- `applied_limits?: object`

## 4. Close Session

### Request

`POST /v1/sessions/close`

```json
{
  "session_id": "5715df92-8658-4fa0-92af-6e146789e679"
}
```

### Request fields

- `session_id: string`

### Success response

HTTP status:
- `200 OK` observed

```json
{
  "status": "ok",
  "meta": {
    "version": "rli-api-gateway@2.1.1",
    "request_id": "842c97ed-a206-472f-b42e-a7b3cd75daae",
    "session_id": "5715df92-8658-4fa0-92af-6e146789e679",
    "runtime_ms": 1,
    "schema_version": "1.0"
  }
}
```

Замечание:
- в observed response у `close_session` нет `result`, только `status` и `meta`.

## Session state transitions

Наблюдаемый happy-path:

1. `POST /v1/sessions`
2. `GET /v1/sessions/{session_id}` -> `status = RUNNING`
3. `POST /v1/sessions/step`
4. `POST /v1/sessions/close`
5. `GET /v1/sessions/{session_id}` -> `status = CLOSED`

Этот happy-path уже проверяется автоматически в
`scripts/smoke_sessions.py`.

## Границы контракта

Этот контракт описывает только sessions API.

Он предназначен для:
- унификации request/response моделей;
- унификации transport-клиента;
- повторного использования между Python-based benchmark adapters.

Он не описывает:
- benchmark-specific filesystem layout;
- file transfer helpers;
- workdir policies;
- verifier lifecycle;
- interactive `tmux` workflow;
- high-level helper API вроде `exec`;
- platform-level monitoring endpoint-ы вроде `GET /v1/system/dashboard`.

Benchmark-specific вещи должны строиться в benchmark adapters поверх этого
контракта. Platform-level monitoring остаётся отдельной ручной диагностикой.

Ручной monitoring helper живёт в `scripts/rli_monitoring.py`, но он остаётся
diagnostic tool вне sessions-контракта. Его ожидаемое поведение
описано отдельно в `docs/monitoring.md`.
