# Design

## Задача

`gigarli` — это минимальный репозиторий для унифицированного контракта работы с
GigaRLI.

Цель репозитория:
- зафиксировать общий sessions-контракт;
- сделать его переиспользуемым между разными benchmark-ами;
- не переносить в контракт benchmark-specific логику.

Контракт в первой итерации намеренно узкий и покрывает 4 lifecycle-операции:
- `create_session`
- `get_session`
- `step`
- `close_session`

## Почему нужен отдельный репозиторий

Сейчас интеграции с RLI живут внутри конкретных benchmark-репозиториев и со
временем начинают смешивать несколько уровней:
- transport к RLI API;
- session lifecycle;
- benchmark-specific filesystem policy;
- verifier/agent-specific поведение.

Это приводит к дублированию и сильной связанности.

`gigarli` должен вынести в одно место только общее:
- модели контракта;
- единый response envelope;
- client-side transport.

В текущем состоянии репозитория это уже реализовано как:
- `schemas/common.py`
- `schemas/env.py`
- `client.py`
- package exports в `gigarli/__init__.py`
- `scripts/smoke_sessions.py`
- `scripts/rli_monitoring.py` как ручной diagnostic helper вне sessions-контракта

Репозиторий использует `src` layout:
- importable package живёт в `src/gigarli`;
- это убирает путаницу между repo root и package root;
- sibling consumers должны добавлять в `PYTHONPATH` именно repo-local `src`, а не полагаться на установку пакета.

## Что является источником дизайна

За основу берётся подход `gigagym`:
- `common` и `env` схемы как разделение между общей оболочкой и конкретным API.

При этом `gigarli` не копирует `gigagym` целиком.

Главные отличия:
- `gigarli` описывает только RLI sessions API;
- `gigarli` не вводит tools/verifiers в первую итерацию;
- `gigarli` не поднимает собственный HTTP service layer в первой итерации;
- `gigarli` не пытается быть общим framework-ом для любых сред.

## Основные сущности

### RLIClient

`RLIClient` — это клиентский transport-layer.

Он нужен для того, чтобы:
- отправлять HTTP-запросы в RLI;
- сериализовать request models;
- разбирать response models;
- централизовать retry/timeout/trace логику.

В текущей реализации:
- success responses парсятся в schema models;
- error responses поднимаются как `RLIClientError`;
- optional request/response trace пишется в JSONL через `trace_path`;
- клиент остаётся stateless относительно session lifecycle и не хранит
  "текущую сессию" внутри себя.

Важно:
- `RLIClient` не должен содержать benchmark-specific логику;
- `RLIClient` не должен предоставлять high-level API вроде `exec`,
  `upload_file`, `download_file`, `tmux_session`.

Его задача — только lifecycle primitives:
- `create_session`
- `get_session`
- `step`
- `close_session`

### Client policy layer

Поверх `RLIClient` в дальнейшем должен появиться отдельный client policy layer.

Это не часть core contract и не часть схем. Это слой runtime-поведения поверх
одного и того же sessions API.

Примеры того, что должно жить в policy layer:
- readiness polling после `create_session`;
- optional warmup перед первой реальной работой;
- retry policy для временных ошибок;
- timeout defaults;
- ранняя стабилизация newly created sessions.

Не относится к policy layer:
- базовый request/response trace на уровне `RLIClient`.

Этот trace уже реализован в клиенте, потому что это transport concern, а не
benchmark-specific runtime policy.

Почему это не включается в core contract:
- такие вещи не требуют новых endpoint-ов;
- они реализуются поверх уже существующих `create_session`, `get_session`,
  `step`, `close_session`;
- разным benchmark adapters могут понадобиться разные policy profiles.

Итоговая роль слоёв:
- `schemas` фиксируют shape API;
- `RLIClient` даёт минимальный transport SDK;
- client policy layer добавляет higher-level runtime behavior.

### Schemas

Схемы в `gigarli` делятся на два слоя:

- `schemas/common.py`
  - общий envelope;
  - общие ошибки;
  - метаданные;
  - лимиты.

- `schemas/env.py`
  - request/response модели для:
    - `create_session`
    - `get_session`
    - `step`
    - `close_session`

Это минимальный перенос идеи `gigagym`: отдельно хранить общую оболочку и
отдельно хранить схемы предметного API.

### Smoke / validation layer

Помимо клиента в репозитории есть smoke-слой:
- минимальный executable script против реального RLI;
- быстрый способ проверить, что de facto API не разъехался с контрактом;
- вспомогательный слой для reverse-engineering и тестов.

В текущей реализации `scripts/smoke_sessions.py`:
- использует именно `RLIClient`, а не ad-hoc HTTP-код;
- проверяет `create_session`, `get_session`, `step`, `close_session`;
- дополнительно делает post-close `get_session`;
- содержит встроенные assert-проверки на основные инварианты контракта.
- может работать вместе с client-level trace, если `trace_path` передан в
  `RLIClient`.

Это не runtime API и не benchmark adapter, а служебная валидация контракта.

### Manual monitoring helper

В `scripts/` также живёт ручной diagnostic helper для platform-level
monitoring RLI: `scripts/rli_monitoring.py`.

Его назначение отличается от smoke-слоя:
- smoke проверяет минимальный sessions-контракт end-to-end;
- monitoring helper читает read-only monitoring endpoint-ы RLI и печатает
  компактный summary состояния сервиса.

Такой скрипт не должен запускаться при выполнении benchmark-ов. Он нужен только
для ручной диагностики, когда пользователь сам хочет посмотреть состояние RLI.

Граница:
- не экспортировать через `gigarli.__init__`;
- не добавлять в `RLIClient`;
- не расширять sessions-контракт;
- не требовать дополнительных runtime-зависимостей;
- не создавать JSON-файлы по умолчанию.

Ожидаемое поведение и CLI зафиксированы в `docs/monitoring.md`.

## Что входит в контракт

В контракт входят:
- session create;
- session lookup;
- step execution;
- session close;
- единый envelope `status/result/meta` и `status/error/meta`;
- общие лимиты и метаданные.

На уровне кода это соответствует:
- `gigarli.schemas.common`
- `gigarli.schemas.env`
- `gigarli.client`

На уровне файловой структуры это лежит в:
- `src/gigarli/schemas/common.py`
- `src/gigarli/schemas/env.py`
- `src/gigarli/client.py`

## Что не входит в контракт

В контракт не входят:
- `terminal_bench`-специфичные пути:
  - `/tests`
  - `/logs`
  - `reward.txt`
  - `ctrf.json`
- `swe`-специфичные соглашения:
  - `/testbed`
  - submission marker
- `mera`-специфичные env mappings;
- helper-абстракции вроде:
  - `exec`
  - `upload_file`
  - `download_file`
  - `prepare_session`
  - `tmux_session`
- client policy behaviors:
  - warmup
  - readiness polling
  - retry profiles

Все такие вещи должны реализовываться в benchmark adapters поверх
минимального RLI-контракта.

## Граница между `gigarli` и benchmark adapters

### Что делает `gigarli`

- определяет модели контракта;
- даёт client-side SDK;
- даёт smoke/test слой для проверки реального API.

### Что делают benchmark adapters

Benchmark adapters используют `gigarli` как низкоуровневый sessions layer и
сверху реализуют свои политики.

Примеры:

- `terminal_bench`
  - `WORKDIR /app`
  - `/tests` и `/logs`
  - verifier lifecycle

- `swe`
  - `/testbed`
  - submission flow

- `mera`
  - env mapping
  - repo/test conventions

## Почему контракт остаётся минимальным

Почти всё runtime-поведение в текущем RLI фактически строится поверх `step`.

Поэтому, если добавить в общий контракт high-level вещи вроде:
- `exec`
- file transfer
- interactive session
- bootstrap

то контракт быстро станет:
- слишком широким;
- слишком opinionated;
- слишком завязанным на частные кейсы отдельных benchmark-ов.

Поэтому в первой итерации выбран узкий дизайн:
- минимальный sessions API;
- всё остальное — снаружи.

Это упрощает:
- повторное использование;
- тестирование;
- внедрение в существующие benchmark-репозитории.

Текущее состояние это подтверждает:
- schema models уже реализованы и покрыты unit tests;
- client уже реализован и покрыт unit tests;
- smoke script уже успешно проходит end-to-end против реального RLI.

## Почему сейчас нет server-side слоя

В `gigagym` есть полноценный server-side каркас, но для `gigarli` он пока не
нужен.

Причины:
- все текущие потребители `gigarli` — Python-based;
- основная задача сейчас — унифицировать client-side contract package;
- retries, tracing, auth и policy можно централизовать на уровне Python SDK;
- отдельный HTTP layer добавил бы лишнюю инфраструктурную сложность.

Если позже появится необходимость:
- жёстко централизовать policy enforcement;
- обслуживать не-Python клиентов;
- вынести transport logic в отдельный runtime service,

тогда server-side слой можно будет добавить как следующий этап.

Но в первой итерации `gigarli` остаётся библиотекой, а не сервисом.
