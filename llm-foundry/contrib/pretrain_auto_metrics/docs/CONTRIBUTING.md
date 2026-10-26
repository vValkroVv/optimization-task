# Как контрибьютить в PAM

Короткий маршрут: ветка от `dev` → минимальный диф в стиле репозитория → зелёный MR-пайплайн → ревью по чеклисту → merge в `dev` → ночной e2e → релиз в `main`.
Ниже что означает каждый шаг и куда смотреть. Правила для ассистентов лежат в [`.agents/skills/`](../.agents/skills/) и являются
источником истины, этот документ их пересказывает и связывает.

- [Окружение](#окружение)
- [Структура репозитория](#структура-репозитория)
- [Ветки, коммиты, MR](#ветки-коммиты-mr)
- [Релизы](#релизы)
- [Стиль кода](#стиль-кода)
- [Как добавить метрику](#как-добавить-метрику)
- [Как добавить обстреливалку (evaluator)](#как-добавить-обстреливалку-evaluator)
- [Конфиги](#конфиги)
- [Тесты](#тесты)
- [Сабмодули и `contrib/`](#сабмодули-и-contrib)
- [Docker-образ](#docker-образ)
- [Ревью](#ревью)

## Окружение

Клонирование с сабмодулями и обновление описаны в [README → Установка](../README.md#установка).
Локальный запуск замера внешнего API через compose: [README → Локальный запуск external API](../README.md#локальный-запуск-external-api-замеров-через-docker-compose).
Запуск на кластере: [README → Как запустить автометрики](../README.md#как-запустить-автометрики) и шаблон [`submit_pam_job.py`](../submit_pam_job.py).

## Структура репозитория

| Путь | Что там |
|---|---|
| [`node_runner/src/`](../node_runner/src/) | код раннера: [`config_model.py`](../node_runner/src/config_model.py) (pydantic-модели конфига), [`config_manager.py`](../node_runner/src/config_manager.py), [`orkestration_manager.py`](../node_runner/src/orkestration_manager.py), [`subprocess_manager.py`](../node_runner/src/subprocess_manager.py), [`subprocess_pool_runner.py`](../node_runner/src/subprocess_pool_runner.py) |
| [`node_runner/src/inference/`](../node_runner/src/inference/) | подъём инференса (sglang, PD, external API) |
| [`node_runner/src/metrics/`](../node_runner/src/metrics/) | [`metrics_manager.py`](../node_runner/src/metrics/metrics_manager.py) (фазы и реестр обстреливалок), [`evaluators/`](../node_runner/src/metrics/evaluators/) (по одной обстреливалке на файл), [`tensorboard_saver.py`](../node_runner/src/metrics/tensorboard_saver.py) |
| [`node_runner/src/obm_proxy/`](../node_runner/src/obm_proxy/), [`pam_proxies/`](../node_runner/src/pam_proxies/) | прокси между бенчмарками и движком |
| [`configs/`](../configs/) | базовые конфиги по размерам моделей, пресеты [`alignment_prod/`](../configs/alignment_prod/) (`agent`, `code`, `full`, `general`, `stem`), тематические наборы (`mera`, `gigametric`, `planbench`, `vision`) |
| [`contrib/`](../contrib/) | сабмодули: `lm_evaluation_harness`, `sglang_pretrain`, `gigachat_utils`, `job_submitter`, `dredd`, `terminal_bench`, `swe_bench_harbor_rli`, `gigarli`, ... |
| [`sber_tests/`](../sber_tests/) | unit-тесты (зеркалят `node_runner/src`), e2e, [`test_support/`](../sber_tests/test_support/), [`test_data/`](../sber_tests/test_data/) |
| [`Dockerfile`](../Dockerfile), [`.gitlab-ci.yml`](../.gitlab-ci.yml), [`build_utils/`](../build_utils/) | образ с бандлами бенчмарков, CI, пины внешних репозиториев (`refs.env`) |
| [`VERSION`](../VERSION), [`images.json`](../images.json) | тег последнего релиза и образы по регионам для job_submitter; пишет релиз |
| [`.agents/skills/`](../.agents/skills/) | правила: [`pam-code-style`](../.agents/skills/pam-code-style/SKILL.md), [`pam-create-evaluator`](../.agents/skills/pam-create-evaluator/SKILL.md), [`pam-tests`](../.agents/skills/pam-tests/SKILL.md), [`pam-review`](../.agents/skills/pam-review/SKILL.md), [`pam-measure`](../.agents/skills/pam-measure/SKILL.md) |
| [`docs/`](./) | этот документ, [карта CI/CD](./ci_pipeline_map.md), [gigametric.md](./gigametric.md) |

## Ветки, коммиты, MR

1. Ветка от актуального `origin/dev`, MR в `dev`. `main` это последний релиз: в неё не мержат и не пушат, её двигает релиз-бот.
2. Один MR = одна задача. Рефакторинг, форматирование и правки соседних компонентов едут отдельными MR
   (см. [Blast radius](../.agents/skills/pam-code-style/SKILL.md#blast-radius)).
3. Не двигать указатели сабмодулей, если это не суть изменения.
4. Коммиты на русском, в повелительной форме, по сути изменения: `Добавить таймаут в forge`, `Починить пустые метрики claweval при падении судьи`.
5. Push в ветку создаёт пайплайн с кнопками `build:*_sglang` (образ из своей ветки) и `trigger:papaya` (выкладка кода ветки на Papaya кнопками `papaya:init` → `papaya:to_*` в дочернем пайплайне).

Что запускается на MR: [карта пайплайнов → MR](./ci_pipeline_map.md#1-пайплайн-на-mr). Кратко:

| Стейдж | Что проверяет | Как чинить локально |
|---|---|---|
| `lint` | `ruff check` / `ruff format --check`, `yamllint -d relaxed` (без `contrib/`); не на Draft | `ruff check . --exclude contrib && ruff format . --exclude contrib` |
| `validate` | `check:submodules`: каждый сабмодуль в `contrib/` на последнем коммите своего `main`/`master`; `check:mr_target`: MR открыт не в `main`; `check:refs`: каждый пин `build_utils/refs.env` стоит на HEAD ветки по умолчанию своего репозитория | `git submodule update --remote contrib/<name>` и закоммитить указатель; MR в `main` пересоздать в `dev`; пин подвинуть на HEAD ветки по умолчанию, новому пину `X_REF` добавить `X_REPO_URL` в `refs.env` |
| `unit` | `test:unit`: `pytest -m unit sber_tests`, покрытие `node_runner` 100% с ветками | команда из [`sber_tests/README.md`](../sber_tests/README.md#как-запускать) |
| `integration` | `test:integrational` (только MR в `dev`): ручной и блокирующий, после зелёного `test:unit`; e2e на CPU на реплее записанных генераций в образе `PAM_IMAGE`. После успешных CPU-тестов доступен ручной [`review:pam`](./ci_pipeline_map.md#ручное-ревью-mr), который не блокирует merge | `pytest -m e2e sber_tests/e2e/test_e2e.py` с `PAM_E2E_OVERLAY=sber_tests/test_data/configs/10b_test_replay.yaml` и `PAM_E2E_CHECKPOINT=<пустой каталог>` |

Линтеры не блокируют merge, но замечания по стилю всё равно придут на ревью. GPU-e2e на MR не запускаются, их гоняет ночь на `dev`.

## Релизы

Подробно: [карта пайплайнов → ночной пайплайн](./ci_pipeline_map.md#2-ночной-пайплайн-на-dev).

- Каждую ночь (23:00 МСК) пайплайн на `dev` гоняет e2e `text`, `vision`, `430b`. Образ пересобирается, только если с последнего релиза менялись его входы: `Dockerfile`, `build_utils/` (с пинами `build_utils/refs.env`), requirements, сабмодули, которые идут в образ.
- Зелёная ночь с новыми коммитами выпускает релиз: коммит `release: vX.Y.Z` с `VERSION` (и `images.json`, если образ собирался), тег вместе с fast-forward `main` одним атомарным push, GitLab Release, возврат коммита в `dev`. Ночь без новых коммитов релиз не выпускает. Push в `main` выкладывает код на Papaya.
- Красная ночь релиз не выпускает, `main` и Papaya остаются на прошлой версии. Причину ищут по логам e2e; если виноват MR, его чинят или откатывают в `dev`.
- Версия растёт в patch. Minor/major: мейнтейнер ставит тег `vX.Y.0` на `main` (Code → Tags), следующая ночь продолжит от него.
- `VERSION` и `images.json` руками не правят: их пишет релиз. `images.json` задаёт образы, на которых job_submitter запускает замеры из этого клона.
- Hotfix: ветка `hotfix/*` от `main`, MR в `main` (мержит мейнтейнер), сразу же MR `main` → `dev`. Пока он не влит, ночной релиз падает на не-fast-forward push, тег при этом не создаётся. Тег на hotfix не ставится.

## Стиль кода

Полный текст: [`pam-code-style`](../.agents/skills/pam-code-style/SKILL.md). Суть в одной таблице:

| Правило | Почему |
|---|---|
| Сначала [лесенка](../.agents/skills/pam-code-style/SKILL.md#the-ladder): нужно ли вообще → уже есть в репо → есть в установленной зависимости → окупается ли абстракция → минимальный код | каждая новая сущность это то, что кто-то читает в 3 ночи |
| Явные методы, типы и control flow. Без `getattr`-диспетчеризации, строковых имён методов, monkey-patching, magic-dict вместо доменных структур | см. [Avoid dynamic and clever code](../.agents/skills/pam-code-style/SKILL.md#avoid-dynamic-and-clever-code) |
| Нет `utils`-свалок, лишних базовых классов, фабрик, DI-контейнеров, `async` без нужды, широких `try/except`, тихой обработки ошибок, спекулятивных флагов | [Forbidden by default](../.agents/skills/pam-code-style/SKILL.md#forbidden-by-default) |
| Защитная проверка без именованного падающего кейса не добавляется | если нельзя назвать вход, который ломает код без гарда, гард не нужен |
| Никаких захардкоженных хостов, эндпоинтов и путей окружения | PAM решает, куда стрелять, один раз |
| Комментарии только для неочевидных бизнес-правил и внешних ограничений; без TODO | код объясняет себя именами и типами |
| Фикс бага = корневая причина, а не симптом | грепнуть все вызовы функции перед правкой |

## Как добавить метрику

**В 99% случаев метрика добавляется в наш форк [lm-evaluation-harness](https://gitlab.ai.cloud.ru/gigallms/lm-evaluation-harness)**
по [их гайду](https://gitlab.ai.cloud.ru/gigallms/lm-evaluation-harness/-/blob/main/docs/new_task_guide.md), а в PAM
только декларируется задача в блоке `evaluators.lm_eval` конфига ([README → Блок `evaluators.lm_eval`](../README.md#10-блок-evaluatorslm_eval)).
Обстреливалка нужна, только если бенчмарк не ложится в harness: свой CLI, агентная среда, судья, параллельный прогон против других моделей.

Исходный раздел: [README → Как добавить метрики](../README.md#как-добавить-метрики).

## Как добавить обстреливалку (evaluator)

Источник правил: [`pam-create-evaluator`](../.agents/skills/pam-create-evaluator/SKILL.md). Короткая версия в README:
[Как добавить «обстреливалку»](../README.md#как-добавить-обстреливалку). Ниже полный маршрут со ссылками на код.

Лучшая новая обстреливалка это ближайшая существующая, в которой поменяли только специфичное для бенчмарка.
Приоритеты: **корректность → консистентность с референсом → минимальный диф → сопровождаемость**.

### 1. Прочитать перед правкой

- [`evaluators/interface.py`](../node_runner/src/metrics/evaluators/interface.py) — базовые классы и абстрактные методы:
  [`MetricsEvaluator`](../node_runner/src/metrics/evaluators/interface.py) (`setup`, `run`, `get_metrics`, `after_run`, `stages`),
  [`ScoringMetricsEvaluator`](../node_runner/src/metrics/evaluators/interface.py) (`run_phase(phase)`: `"generation"`, затем `SCORING_PHASE`; судья — единственное имя из `secondary_model_names` конфига),
  [`ParallelInferenceEvaluator`](../node_runner/src/metrics/evaluators/interface.py) (`run_on_model_name`).
  `stages()` возвращает список `Stage`: каким движкам (evaluated и/или судья) нужна стадия; базовые классы его уже реализуют.
  Адрес и число инстансов судьи берутся из конфига как `config.judge_endpoints[имя]`, их заполняет `AutoMetricsManager`.
- [`config_model.py`](../node_runner/src/config_model.py) — [`BaseConfigModel`](../node_runner/src/config_model.py#L167), [`MetricsEvaluatorConfig.apply_defaults()`](../node_runner/src/config_model.py#L197), константа [`DEFAULT_REQUEST_TIMEOUT`](../node_runner/src/config_model.py#L164).
- [`metrics_manager.py`](../node_runner/src/metrics/metrics_manager.py) — реестр `EVALUATORS` и задачи из стадий.

### 2. Выбрать референс

| Нужно | База | Референс |
|---|---|---|
| обёртка над существующим eval-фреймворком | `MetricsEvaluator` | [`lm_eval.py`](../node_runner/src/metrics/evaluators/lm_eval.py) |
| standalone CLI бенчмарка: выставить env с эндпоинтом, запустить тул, распарсить артефакты | `MetricsEvaluator` | [`harbor_rli.py`](../node_runner/src/metrics/evaluators/harbor_rli.py) |
| RLI / terminal-среда | `MetricsEvaluator` | [`terminal_bench.py`](../node_runner/src/metrics/evaluators/terminal_bench.py) |
| судья оценивает сохранённые генерации | `ScoringMetricsEvaluator` | [`dredd.py`](../node_runner/src/metrics/evaluators/dredd.py) |
| модель гоняется параллельно против вторичных моделей | `ParallelInferenceEvaluator` | [`tau2.py`](../node_runner/src/metrics/evaluators/tau2.py) |

Форма фаз важнее, чем «похожесть» бенчмарка по названию.

### 3. Какие файлы трогать

- `node_runner/src/metrics/evaluators/<name>.py`
- [`evaluators/__init__.py`](../node_runner/src/metrics/evaluators/__init__.py) — экспорт
- [`config_model.py`](../node_runner/src/config_model.py) — класс конфига
- [`metrics_manager.py`](../node_runner/src/metrics/metrics_manager.py) — регистрация в [`EVALUATORS`](../node_runner/src/metrics/metrics_manager.py); какие движки нужны, решает `stages()` базового класса
- все базовые [`configs/*.yaml`](../configs/) и нужные пресеты [`configs/alignment_prod/`](../configs/alignment_prod/) — как объявлены `terminal_bench` / `tau3` / `swe_bench`.
  Обстреливалка, прописанная только в конфиге автора, невидима для всех, кто меряет от базового конфига.
- [`Dockerfile`](../Dockerfile) — отдельный `*_bundle` stage под окружение бенчмарка, клон репозитория в [`.gitlab-ci.yml`](../.gitlab-ci.yml) (см. [Docker-образ](#docker-образ))

Тесты только по явной просьбе, тогда по [`pam-tests`](../.agents/skills/pam-tests/SKILL.md).

### 4. Правило расширения

Зеркалить референс, не импровизировать структуру:

- поле или метод копируется из референса — копировать;
- метода в референсе нет — взять ближайший из другой обстреливалки и минимально поправить;
- лишние для этого бенчмарка поля и методы референса выкинуть;
- сохранять сигнатуры и порядок команд внутри методов, одинаковое поведение master/worker;
- внешние тулы запускать через [`SubprocessManager`](../node_runner/src/subprocess_manager.py) / [`SubprocessPoolRunner`](../node_runner/src/subprocess_pool_runner.py), как делает обстреливалка с той же формой запуска;
- хелперы `MetricsEvaluator` для директорий метрик, парсинга чекпоинта, JD-payload, tar-архивов и симлинков переиспользовать, не переписывать.

### 5. Внутри обстреливалки нет `raise`

Обстреливалка не рейзит в рантайме: ни на валидации, ни на «невозможных» состояниях, ни `try/except` с повторным `raise`.

| Случай | Куда |
|---|---|
| упал сабпроцесс | `SubprocessManager.log_error()`, не `raise_for_exit_code()` |
| артефакты отсутствуют или не парсятся | вернуть пустые метрики и `Logger.error` с путём |
| неверный конфиг | валидация в модели конфига, там `raise` на старте уместен |
| упала фаза, после которой сохранять нечего: генерация `ScoringMetricsEvaluator`, скоринг с негодными выходами | `raise`: планировщик не поднимет судью под пустой скоринг и не сохранит метрики |

Причина в том, что теряется при рейзе: `DynamicScheduler._run_task` ловит `Exception` на каждый шаг обстреливалки и пропускает
`save_evaluator_metrics`, так что рейз выбрасывает и то, что бенчмарк успел посчитать.

Обратная обязанность: не исчезать молча. `save_evaluator_metrics` тихо возвращается при пустых `get_metrics()`, поэтому проглоченная ошибка
пропадает из отчёта без следа. Отсутствие метрики логируется явной строкой ошибки с причиной.
**Инфраструктурный сбой убирает метрику, а не занижает её.** Бенчмарк, оценивший модель на запросах, которые до неё не дошли, публикует ничего и говорит почему.

### 6. Конфиг

- Одна обстреливалка → один класс конфига, наследник [`BaseConfigModel`](../node_runner/src/config_model.py#L167) / [`MetricsEvaluatorConfig`](../node_runner/src/config_model.py#L197).
- Сериализация конфига в аргументы бенчмарка это метод модели (`model_dump` с явным `exclude`), не код в обстреливалке и не `setdefault`-цепочки по dict.
- Обстреливалка не мутирует конфиг.
- Валидация здесь и может рейзить. Бессмысленные значения не проходят: ноль ретраев, нулевой таймаут, пустой список задач. `ge=0` на счётчике ретраев это баг.
- Дефолты, зависящие от числа инстансов инференса, `base_url` или пути чекпоинта, идут через `apply_defaults()`.
- Runtime-only поля не попадают в CLI-аргументы бенчмарка.

### 7. Обязательные ручки: таймаут и параллельность

Без этих двух полей обстреливалка неполная:

- **`timeout`** (секунды), дефолт из [`DEFAULT_REQUEST_TIMEOUT`](../node_runner/src/config_model.py#L164), число не хардкодить; протащить в то, что читает тул: CLI-арг, env, ключ конфига.
  Примеры: forge / dredd в `config_model.py`, [`tau2.py`](../node_runner/src/metrics/evaluators/tau2.py) (`Tau2SamplingConfig.to_llm_args`), [`bfcl.py`](../node_runner/src/metrics/evaluators/bfcl.py) (`OPENAI_TIMEOUT`), OBM / lmms_eval (`timeout` в блоке `api`).
  Маленький клиентский таймаут это то, что убило forge на медленной reasoning-модели под нагрузкой.
- **параллельность** (`workers` / `parallel_api_calls` / `num_threads` / `concurrency`), обычно масштабируется от `inference_instances` в `apply_defaults()`.

Если у тула нет хука, хук добавляется в тул (они вендорены в `contrib/`). Если это вне скоупа, об этом пишется явно в MR,
а в начало каждого коммита и описания MR добавляется строка `⚠️ MISSING KNOBS: ...` (см. [Committing](../.agents/skills/pam-create-evaluator/SKILL.md#committing)).

### 8. Эндпоинты, пропуск работы, метрики

- Обстреливалка не знает, куда стреляет: не принимает эндпоинт параметром, не ветвится по SLURM-переменным, не проверяет интернет. PAM решает RLI vs proxy один раз через существующую проверку в интерфейсе и отдаёт одну известную env-переменную.
- Дорогая фаза пропускается по самим метрикам или по маркеру, записанному последним шагом фазы. Директория с выводом не доказательство: прерванный ран оставляет её, и следующий запуск отчитается обрезанным результатом как настоящим.
- `get_metrics()` возвращает `dict[str, float]`, значения как доли, не 0–100. `after_run()` шлёт в JD с явными стабильными ключами. `is_master`-проверок внутри обстреливалки нет, это делают фазы в `metrics_manager.py`.
- Сырые артефакты бенчмарка лежат в `metrics/<evaluator_name>`; сырые ответы модели сохраняются, если бенчмарк это умеет.

### 9. Чеклист перед MR

- [ ] класс, экспорт в `__init__.py`, конфиг-модель, ровно один реестр в `metrics_manager.py`;
- [ ] объявлена во всех базовых `configs/*.yaml` и нужных `alignment_prod/`;
- [ ] есть `timeout` от `DEFAULT_REQUEST_TIMEOUT` и ручка параллельности;
- [ ] зеркалит референс, новой формы нет, если бенчмарк не заставляет;
- [ ] пропускает прогон, если метрики уже есть;
- [ ] после технического падения не оставляет публикуемых метрик;
- [ ] раздел про блок `evaluators.<name>` добавлен в [README](../README.md#как-собрать-собственный-конфиг-или-что-означает-каждый-раздел-в-конфиге) по образцу соседних.

## Конфиги

Устройство конфига и смысл каждого раздела: [README → Как собрать собственный конфиг](../README.md#как-собрать-собственный-конфиг-или-что-означает-каждый-раздел-в-конфиге).
Модели всех блоков лежат в [`config_model.py`](../node_runner/src/config_model.py); там же валидация, алиасы и дефолты.
Новое поле сначала появляется в модели, потом в YAML. Для `SglangInferenceConfig` действует `extra=forbid`: опечатка в ключе роняет старт, что и требуется.

Базовые конфиги [`configs/*.yaml`](../configs/) именуются по размеру и точности модели (`37b.yaml`, `37b_fp8.yaml`, `430b_fp8_pd.yaml`), пресеты [`alignment_prod/<домен>/<размер>.yaml`](../configs/alignment_prod/) собирают наборы для продовых замеров.
Новый размер модели добавляется во все домены `alignment_prod`, а не только в тот, что нужен автору.

## Тесты

Правила: [`pam-tests`](../.agents/skills/pam-tests/SKILL.md). Структура и команды запуска: [`sber_tests/README.md`](../sber_tests/README.md).

- Unit-тест зеркалит путь модуля из `node_runner/src` под `sber_tests/unit/`, помечен `pytestmark = pytest.mark.unit`, использует фикстуры [`conftest.py`](../sber_tests/conftest.py) и хелперы [`test_support/`](../sber_tests/test_support/).
- Тестируется внешне наблюдаемое поведение и контракты, не приватная декомпозиция. Границы с sglang, OBM, токенизаторами, сабпроцессами, ФС и сетью мокаются.
- CI требует 100% покрытия `node_runner` с ветками, поэтому новый код без тестов пайплайн не пройдёт.
- Тесты добавляются по явной просьбе, кроме одного случая: фикс кэша, resume, локов или рестарта обязан идти с регрессионным тестом, воспроизводящим поломку. Такие баги невидимы на зелёном прогоне.
- E2E (`sber_tests/e2e/`) гоняются только ночью на `dev` джобой на кластере, локально их не запускать. Сьюты и их аргументы в [`submit_e2e_tests.py`](../sber_tests/submit_e2e_tests.py).

## Сабмодули и `contrib/`

- `contrib/` это сабмодули. Правки в тулы (harness, dredd, terminal_bench, ...) делаются MR в их репозитории, потом в PAM двигается указатель.
- `check:submodules` на MR в `dev` требует, чтобы каждый сабмодуль стоял на последнем коммите своего `main`/`master`. Джоба не блокирует merge, но если кто-то смержил в сабмодуль после создания вашей ветки, лучше обновить указатель.
- Указатель сабмодуля меняется только когда это цель MR.
- Бенчмарк-данные и вендорные тулы живут в образе, не на NFS и не в `contrib/`.

## Docker-образ

Правила из [Docker and data](../.agents/skills/pam-code-style/SKILL.md#docker-and-data):

- Репозитории клонируются в [`.gitlab-ci.yml`](../.gitlab-ci.yml) по пину на sha из [`build_utils/refs.env`](../build_utils/refs.env) и копируются в образ `COPY`; клонов внутри [`Dockerfile`](../Dockerfile) нет. Бамп внешнего репозитория = правка `refs.env`; новый репозиторий = пара `X_REPO_URL` + `X_REF` там же.
- Каждый бенчмарк это свой `*_bundle` stage; повторяющиеся слои выносятся в отдельный stage, а не копипастятся в финальный.
- Секреты только через `docker build --secret`, не build-args и не файлы в образе.
- Зависимости пинуются, источник должен быть достижим. Непинованный requirements это будущая поломка без владельца.
- Тесты в Dockerfile не запускаются.

Как собирается и куда пушится образ: [карта пайплайнов → ночной пайплайн](./ci_pipeline_map.md#2-ночной-пайплайн-на-dev) и [ручной пайплайн](./ci_pipeline_map.md#4-ручной-пайплайн-run-pipeline).

## Ревью

Ревьюер идёт по [`pam-review`](../.agents/skills/pam-review/SKILL.md). Чеклист там повторяет разделы выше:
`raise` внутри обстреливалки, конфиг в модели, покрытие конфигов, «обстреливалка не знает, куда стреляет», воспроизводимость сборки,
«от какого кейса это защищает», сигнал для пропуска фазы, идентичность кэша и resume, инфраструктурные сбои не есть качество модели,
таймаут и параллельность, комментарии, гигиена дифа, воспроизводимые вычислительные результаты, регрессионные тесты для кэша и рестарта,
границы follow-up'ов, lm-evaluation-harness.

Формат комментария: `path/file.py:LINE`, короткий заголовок, код из дифа, текст на русском в одно-три предложения с конкретным последствием.
Группы **Blocking / Worth fixing / Nit**. Пример формы в разделе [Output](../.agents/skills/pam-review/SKILL.md#output).
