# gigarli

Минимальный репозиторий для унифицированного контракта работы с GigaRLI.

Цель:
- зафиксировать общий sessions-контракт для RLI;
- сделать его переиспользуемым для benchmark-ов, требующих создания среды в Docker-контейнере.

В первой итерации контракт минимальный и покрывает 4 lifecycle-операции:
- `create_session`
- `get_session`
- `step`
- `close_session`

В этот репозиторий не должны попадать:
- `terminal_bench`-специфичные пути вроде `/tests`, `/logs`, `reward.txt`;
- `swe`-специфичные вещи вроде `/testbed` или submission marker;
- `mera`-специфичные env mappings;
- high-level helper API вроде `exec`, `upload_file`, `download_file`, `tmux_session`.

Все такие вещи должны строиться снаружи, поверх минимального RLI-контракта.

На текущем этапе `gigarli` — это не отдельный HTTP-сервис, а Python package с:
- документацией контракта;
- Pydantic-моделями;
- клиентом к RLI API;
- smoke/test-слоем для проверки de facto API.

Уже реализовано:
- `gigarli.client.RLIClient`
- `gigarli.client.RLIClientError`
- схемы в `gigarli.schemas.common` и `gigarli.schemas.env`
- optional JSONL request tracing через `trace_path`
- package exports через `gigarli.__init__`
- рабочий smoke script для `create/get/step/close`
- ручной monitoring helper для read-only проверки состояния RLI

Python layout:
- репозиторий использует `src` layout;
- сам importable package живёт в `src/gigarli`;
- пакет должен использоваться без установки через добавление repo-local `src` в `PYTHONPATH`.

## Переиспользование HTTP-соединений

`RLIClient` создаёт и закрывает HTTP-клиент на каждый запрос, как и раньше.
Для переиспользования соединений явно выберите `RLIClientSession`: он создаёт
HTTP-клиент при первом запросе и использует его для `create/get/step/close`.
Методы работы с RLI-сессиями и обработка ошибок наследуются от `RLIClient`.

Используйте экземпляр в одном event loop. После завершения всех его запросов
вызовите `await client.aclose()` или используйте `contextlib.aclosing`:

```python
from contextlib import aclosing

from gigarli import RLIClientSession

async with aclosing(RLIClientSession()) as client:
    session = await client.get_session(session_id)
```

`aclose()` закрывает HTTP-соединения. RLI-сессии закрываются отдельно через
`close_session()`. Автоматические повторы запросов не добавляются.

## Структура

```text
gigarli/
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── contract.md
│   ├── design.md
│   └── monitoring.md
├── src/
│   └── gigarli/
│       ├── __init__.py
│       ├── client.py
│       └── schemas/
│           ├── __init__.py
│           ├── common.py
│           └── env.py
├── scripts/
│   ├── smoke_sessions.py
│   └── rli_monitoring.py
└── tests/
    ├── test_schemas.py
    ├── test_client.py
    └── test_rli_monitoring.py
```

## Что где лежит

### Корень репозитория

- `README.md`
  - общее описание репозитория и его роли.

- `pyproject.toml`
  - packaging metadata и repo-level tooling.

- `.gitignore`
  - локальные артефакты разработки.

### docs/

- `docs/contract.md`
  - формальное описание RLI sessions-контракта:
    - request/response модели;
    - expected behavior для `create/get/step/close`;
    - observed примеры реального API.

- `docs/design.md`
  - объяснение архитектуры:
    - зачем есть `client`;
    - почему контракт остаётся минимальным;
    - где проходит граница между общим контрактом и benchmark-specific адаптацией.

- `docs/monitoring.md`
  - описание ручного monitoring helper:
    - какие read-only endpoint-ы он читает;
    - какие CLI-флаги поддерживает;
    - почему JSON snapshots пишутся только по явному `--save-json`.

### src/gigarli/

Это основной Python package.

- `src/gigarli/client.py`
  - клиент к RLI sessions API;
  - только transport-level операции:
    - `create_session`
    - `get_session`
    - `step`
    - `close_session`
  - optional JSONL trace для всех transport-level операций при передаче
    `trace_path`;
  - стандартные RLI error envelope оформлены через `RLIClientError`.

- `src/gigarli/__init__.py`
  - публичный API пакета;
  - реэкспорт клиента, exception class и основных schema classes.

### src/gigarli/schemas/

- `src/gigarli/schemas/common.py`
  - общие модели:
    - `meta`
    - `errors`
    - `success/error envelope`
    - `limits`

- `src/gigarli/schemas/env.py`
  - модели sessions API:
    - create session
    - get session
    - step
    - close session

### scripts/

- `scripts/smoke_sessions.py`
  - минимальный smoke test реального RLI API;
  - нужен для быстрой проверки de facto контракта;
  - валидирует `create/get/step/close` встроенными assert-проверками;
  - не является частью публичного runtime API пакета.
  - запускается из корня репозитория с `PYTHONPATH=src`.

- `scripts/rli_monitoring.py`
  - ручной diagnostic helper для platform-level monitoring RLI;
  - читает read-only endpoint-ы `GET /v1/system/dashboard` и, если не отключено,
    `GET /v1/errors/recent`;
  - по умолчанию печатает compact summary в stdout и не создаёт файлы;
  - JSON snapshots пишет только при явном `--save-json`;
  - не используется при выполнении benchmark-ов.

### tests/

- `tests/test_schemas.py`
  - тесты на Pydantic-модели и сериализацию;
  - покрывают как minimal envelope, так и sessions models.

- `tests/test_client.py`
  - unit tests для transport-клиента;
  - проверяют URL/path, payload shape, error handling и отсутствие hidden current session state.

- `tests/test_rli_monitoring.py`
  - unit tests для ручного monitoring script;
  - проверяют CLI defaults, `--save-json`, `--quiet`, summary и JSON snapshots
    без реальных сетевых запросов.

## Запуск локально

Из корня репозитория:

```bash
PYTHONPATH=src python3 -m pytest tests -q
PYTHONPATH=src python3 scripts/smoke_sessions.py --insecure
python3 scripts/rli_monitoring.py
python3 scripts/rli_monitoring.py --save-json --output-dir /tmp/rli_monitoring_snapshots
```

## Граница ответственности

Этот репозиторий отвечает только за:
- минимальный sessions-контракт;
- общие модели;
- клиент;
- request/response tracing на уровне клиента;
- ручные diagnostic scripts в `scripts/`, не входящие в benchmark runtime.

Этот репозиторий не отвечает за:
- Harbor integration;
- verifier lifecycle;
- file transfer helpers;
- workdir hacks;
- benchmark-specific bootstrap;
- task-specific filesystem conventions;
- platform-level monitoring API как часть sessions-контракта.

Benchmark-specific вещи должны жить в benchmark adapters поверх `gigarli`.
Platform-level monitoring остаётся отдельной ручной диагностикой.
