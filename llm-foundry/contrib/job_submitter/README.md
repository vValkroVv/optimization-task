    # job-submitter

`job-submitter` - Python-сервис для унифицированного запуска ML/LLM джоб в двух типах окружений:

- Cloud-регионы (`SR004`, `SR006`, `SR008`) через `client_lib`;
- Slurm-кластеры ...

Сервис решает две основные задачи:

1. Сводит запуск разных сценариев (train, autometrics, ray cluster и т.д.) к единому Python API.
2. Валидирует параметры и конфиги до сабмита, чтобы отлавливать ошибки заранее.

## Быстрый старт

Требования:

- Python `>=3.10`;
- доступ к целевому окружению (Cloud/Slurm);
- корректные переменные окружения.

Установка:

```bash
uv sync
```

Пакет `sqlite-s3-query` устанавливается из локального PyPI.

Минимальный пример запуска:

```python
from job_submitter.jobs.any_script import AnyScriptJob

job = AnyScriptJob(
    user_login="your_login",
    exp_name="test-run",
    script="python -V",
)

result = job.submit()
print(result)
```

Минимальный CLI (пока поддерживается только kill):

```bash
python -m job_submitter kill <job-id>
# или после установки пакета:
js kill <job-id>
```

Команда сама определяет контекст запуска (`cloud`/`slurm`) и регион:
- для `cloud` использует `client_lib.kill(...)`;
- для `slurm` выполняет `scancel <job-id>`.

## Общие параметры для всех джоб (`BaseJob`)

Ниже перечислены параметры, которые доступны в любой джобе (если не переопределены в конкретном классе).

Важно: многие параметры резолвятся автоматически при `submit()`.  
Обычно достаточно задать только обязательные параметры конкретной джобы + бизнес-параметры (например, путь к конфигу/чекпоинту).

### Обязательные

- `user_login: str` - логин пользователя.
- `exp_name: str` - имя/описание эксперимента.

### Часто используемые

- `n_gpus: int = 8` - число GPU на ноду (валидные значения: `1`, `2`, `4`, `8`).
- `n_workers: int = 1` - число нод.
- `submission_type: str | None = None` - тип сабмита: `cloud` или `slurm` (обычно можно не указывать, берется из окружения/кластера).
- `region: str | None = None` - регион запуска (обычно можно не указывать, берется автоматически):
  - cloud: `SR004`, `SR006`, `SR008`;
  - slurm: ...
- `script_type: str = "binary"` - тип джобы для backend (`binary`, `pytorch2` и др.); в большинстве кейсов вручную не меняется.
- `processes_per_worker: int = 1` - число процессов на ноду (для slurm форсится в `1`).

### Ресурсы и приоритет

- `queue_name: str | None = None` - для cloud: очередь (`default`, `giga-prod`, `instruction-tuning`, `jupyter`, `online-rl`, `pretrain-data`, `gigachat-data`, `pretrain-ml`, `pretrain-ml-test`, `prod-ml-data`, `shared`); для slurm - команда (`pretrain`, `data`, `alignment`).
  - на `o1`/`o2` `queue_name` определяет партицию, QoS и `account`; если не задан, берётся команда пользователя из Slurm (`sacctmgr ... defaultaccount`);
  - на `o5` `queue_name` прокидывается как `qos` и валиден только с `partition=main` (то есть при `priority_class="high"`); в остальных сценариях должен быть `None`.
- `priority_class: str | None = "medium"` - класс приоритета (`low`, `medium`, `high`, `urgent`, `leads`).
  - На `o1`/`o2` (командные партиции и QoS) `priority_class` - это важность задачи внутри команды `<team>` = `queue_name`:

    | `priority_class` | sbatch | кого снимает | кто снимает его |
    |---|---|---|---|
    | `urgent` | `-p <team> --qos=<team>_high --account=<team>` | обычные и фоновые своей команды, `normal` | `leads` |
    | `high` | `-p <team> --qos=<team> --account=<team>` | фоновые своей команды, `normal` | `leads`, свои `urgent` |
    | `medium` | `-p <team> --qos=<team>_low --account=<team>` | `normal` | `leads`, свои `urgent` и `high` |
    | `low` | `-p main --qos=normal --account=<team>` | никого | все |
    | `leads` | `-p leads --qos=leads --account=leads` | всех | никто |

    `urgent` доступен только тем, у кого есть QoS `<team>_high` (лиды команды). `low`/`medium`/`high` могут быть сняты преемпшеном (`PreemptMode=REQUEUE`), поэтому задача должна уметь продолжаться с чекпоинта.
  - На `o5` (старая схема): `high` выбирает partition `main` и `account=<queue_name>`; `leads` выбирает partition `leads` и `account=leads`; остальные значения выбирают partition `middle` без QoS.
- `max_retry: int | None = None` - количество ретраев. Для slurm это включает requeue/restart-логику; значение должно быть `None` или больше `1`.
- `cpus_per_task: int | None = None` - CPU на задачу для slurm; если `None`, параметр не добавляется в sbatch-скрипт.
- `internet: bool = True` - доступ в интернет.
- `dry_run: bool = False` - "сухой" запуск: собрать/провалидировать конфиг без реального сабмита в backend.

### Образ, логи, окружение

- `base_image: str | None = None` - Docker/SQSH образ (если `None`, выбирается дефолт по региону).
- `workspace_name: str | None = None` - cloud workspace; если не указан, берется из `MLS_NTB_WORKSPACE_NAME`, а при отсутствии обоих значений submit завершается с ошибкой.
- `extra_env: dict[str, str] = {}` - дополнительные переменные окружения.
- `extra_mounts: str | None = None` - дополнительные монтирования.
- `job_logdir: str | None = None` - путь для job logs (slurm).
- `slurm_logdir: str | None = None` - путь для slurm logs (slurm).
- `slurm_submit_type: str = "sbatch"` - способ отправки в slurm (`sbatch` или `api`).
- `nodelist: str | None = None` - ограничение slurm-запуска конкретными нодами.
- `nice: int | None = None` - `--nice` для slurm: уступить очередь другим задачам того же класса (0 и больше; отрицательные значения только у администраторов).
- `exclude_nodes: str | None = None` - ноды, исключённые из slurm-запуска.
- `turn_off_first_srun: bool | None = None` - отключить первый `srun`.
- `debug: bool = False` - общий debug-флаг джобы; конкретное поведение зависит от класса джобы.

При локальном `sbatch` job-submitter задаёт `SLURM_UMASK=0002`, чтобы Slurm создавал stdout/stderr с правом записи для группы.

### Остановка процессов при завершении джобы (slurm)

При выходе приложения или внешнем сигнале GPU-шаблон останавливает процессы шага
в три этапа: TERM training group, TERM найденным процессам, KILL оставшимся. Ход
и результат пишутся в
`<job_logdir>/<job_id>/retry_<restart>/cleanup_<rank>_<hostname>.log`.

Бюджеты этапов задаются через `extra_env` — целые неотрицательные секунды:

- `PROCESS_SUPERVISOR_GRACE_S=10` — пауза перед сигналами отдельным процессам,
  чтобы свой супервизор джобы успел остановить детей сам. При выходе приложения
  этап пропускается. Джобе, чей супервизор на TERM делает долгую остановку
  (например, `ray stop`), это значение нужно поднять: иначе рассылка сигналов
  отдельным процессам придёт в середину его работы.
- `PROCESS_SHUTDOWN_GRACE_S=45` — ожидание после TERM. Здесь основной бюджет:
  процесс с GPU-контекстом разбирается внутри ядра непрерываемо и может занять
  десятки секунд. Этап завершается досрочно, как только цели исчезли, поэтому
  запас ничего не стоит, когда процессы выходят быстро.
- `PROCESS_SHUTDOWN_KILL_WAIT_S=5` — ожидание после KILL.

Сумма ограничена `KillWait`, а не `UnkillableStepTimeout`: по истечении
`KillWait` Slurm убивает шаг вместе с самим wrapper, поэтому всё, что
запланировано после, не выполнится и лог просто оборвётся. Три этапа дают в
сумме 60 секунд под типовой `KillWait=60`. Финальная проверка и дамп идут сверх
этого и выполняются по возможности: обычно они занимают меньше секунды, но
прогон, израсходовавший все этапы полностью, их потеряет.

Итог пишется в поле `phase`: `term-exit` или `kill-exit` — цели завершились,
`kill-timeout` — остались живые, `cleanup-unverified` — проверка неполна. Два
последних могут выставить `CLEANUP_FAILED`, и тогда успешное обучение завершится
ненулевым кодом.

### Что можно не задавать вручную

В типичном сценарии **необязательно** явно указывать:

- `region`
- `submission_type`
- `script_type`
- `base_image`
- `queue_name`
- `priority_class`
- `allocation_name` (подставляется для отдельных регионов, например `SR008`)

Они будут определены из окружения и/или дефолтной логики резолва в `BaseJob`.

## Параметры по каждой джобе

Ниже перечислены параметры конкретных классов джоб.  
Все они дополнительно принимают общие параметры из секции выше.

---

### `AnyScriptJob` (`job_submitter.jobs.any_script`)

Назначение: запуск произвольной команды/скрипта.

- `script: str` **(обяз.)** - команда, которая будет выполнена внутри джобы.

---

### `InfiniteJob` (`job_submitter.jobs.infinite`)

Назначение: служебная "бесконечная" джоба для удержания ресурсов или отладки окружения.

- `script: str = "sleep infinity"` - команда; по умолчанию бесконечный сон.

---

### `TrainJobLLMF` (`job_submitter.jobs.llmf.train`)

Назначение: запуск train-пайплайна LLMF (`scripts/train/train.py`) с валидацией train-конфига.

- `path_to_config: str | Path` **(обяз.)** - путь к train YAML-конфигу.
- `script_type: str = "pytorch2"` - тип запуска.
- `use_clearml: bool = False` - включение интеграции с ClearML.
- `script_args: str = ""` - дополнительные аргументы командной строки для train скрипта.
- `processes_per_worker: int = 8` - процессов на ноду (для cloud сценария).
- `metric_sampler: str | None = None` - включает metric sampler: имя пресета (`light`, `standard`, `full`) или путь к своему YAML того же вида (см. [Metric sampler](#metric-sampler)).
- `metric_sampler_overrides: dict = {}` - точечные правки конфига metric sampler, вложенные как в YAML, например `{"capture": {"every_n_steps": 4}}`.
- `metric_sampler_unsafe_reason: str | None = None` - причина, по которой запуск может превысить то, что принимает бэкенд metric sampler за шаг (короткий отладочный запуск и т.п.); с ней проверка ёмкости только предупреждает.

#### Metric sampler

Metric sampler записывает статистики обучения по каждому рангу и микробатчу; коллектор острова доставляет их в ClickHouse и Grafana. Работает на `o1` и `o5` для MoE-моделей.

```python
TrainJobLLMF(
    user_login="user",
    exp_name="my-exp",
    path_to_config="train.yaml",
    region="o1",
    n_workers=32,
    metric_sampler="standard",
    metric_sampler_overrides={"capture": {"every_n_steps": 4}, "families": {"attention": True}},
).submit()
```

Пресеты лежат в `job_submitter/jobs/llmf/metric_sampler_configs/`; каждое поле там прокомментировано, файл можно скопировать и передать путь к своей копии в `metric_sampler`.

| Пресет | Шаги | Семейства |
| --- | --- | --- |
| `light` | каждый 10-й | активации, лосс, эксперты; без мониторов и точных квантилей |
| `standard` | каждый | всё, кроме attention; точные квантили (как на продовом претрейне) |
| `full` | каждый | всё, включая attention |

Поля конфига:

- `capture.every_n_steps` - собирать каждый N-й шаг обучения; на остальных шагах сбор ничего не стоит.
- `capture.distributions_every_n_steps` - детальные распределения (гистограммы лосса экспертов, худшие сэмплы) каждый N-й шаг; кратно `every_n_steps`.
- `capture.window_steps` - собранных шагов в одном опубликованном окне: меньше - быстрее в Grafana, больше - меньше файлов.
- `families.activations | token_loss | experts | attention | monitors` - включение семейств метрик (`act/*`, `loss/*`, `router/* dispatcher/* expert_loss/*`, `attn/*`, `monitor/*`). Для `monitors` недостающие в train-конфиге мониторы добавляются с нужными настройками; мониторы, которые в конфиге уже есть, сохраняют свои настройки и проверяются.
- `exact_quantiles` - точные квантили активаций, лосса токенов и энтропии роутера.
- `activation_dump` - сырые активации ранга 0 каждые `every_n_steps` шагов (0 - выключено) для `points` и `layers`.

Настройки передаются в обучение аргументами `train.py` (`callbacks.metric_sampler.*`, `loggers.metric_sampler.*` и недостающие мониторы); сам train YAML не меняется. `output_dir` выводится автоматически: `<raw root острова>/<exp_name>/<run_name>` из train YAML.

Проверки при сабмите (действуют и для metric sampler, настроенного вручную в train YAML):

- остров с коллектором (`o1`, `o5`); metric sampler не задан дважды (пресет и `callbacks.metric_sampler` в YAML);
- известные ключи колбэка; положительные интервалы; `distributions_every_n_steps` кратен `every_n_steps`; корректный `activation_dump`;
- `output_dir` колбэка и логгера - там, где его читает коллектор; `loggers.tensorboard` с `rank_zero_only: true` перед логгером metric sampler;
- для семейства `monitors` - все нужные мониторы перед `metric_sampler` и их настройки (`stat_list: [amax, l2norm]`, `k_values: [1, 5, 20]`, без `log_pre_clip_gradients`, `memory_keys` по умолчанию);
- `tp_size` и `sp_size` равны 1, фиксированный `device_train_microbatch_size`, не `amp_fp16`, `gigafsdp_config.zero_stage: 1`, MoE-модель;
- ёмкость бэкенда: в среднем не больше 1.25e9 значений на шаг обучения; модель уровня 37b (34 слоя, 192 эксперта, 8 микробатчей) со всеми семействами пишет около 1e6 значений на ранг за шаг. Оценка берётся из train YAML (слои, эксперты, микробатчи, потоки данных) и числа GPU; при превышении сабмит отклоняется с минимальным `every_n_steps`. Запуск сверх ёмкости возможен только с `metric_sampler_unsafe_reason` - тогда выводится предупреждение с причиной.

---

### `TrainJobGIFT` (`job_submitter.jobs.gift.train`)

Назначение: запуск GIFT/TorchTitan через единый cloud/Slurm backend.

- `config_file: str | Path` **(обяз.)** - experiment YAML TorchTitan, например `configs/experiments/gift_moe_debug.yaml` (относительно корня checkout или абсолютный путь).
- `script_args: str = ""` - dot-path overrides поверх YAML, например `"--training.steps 100 --optimizer.lr 5e-5"`.
- `gift_root_path: str | Path | None = None` - корень checkout TorchTitan. Если не задан, используется `GIFT_ROOT_PATH`, затем расположение vendored `contrib/job_submitter`.
- `script_type: str = "pytorch2"` - launcher, который выставляет rank/world-size для cloud и Slurm.
- `processes_per_worker: int | None = None` - по умолчанию равно `n_gpus` в cloud; Slurm backend оставляет одну launcher-задачу на ноду и поднимает `n_gpus` worker-процессов.

Для запуска используется `gift_scripts/train.py` из checkout TorchTitan. Он выставляет рабочую директорию в корень checkout, поэтому относительные пути (сам YAML, его `imports`, датасеты) одинаково работают в cloud и Slurm. Перед отправкой job YAML вместе с `script_args` прогоняется через `scripts/config_check.py` того же checkout - тот же torch-free pre-submit, что и в lint-стадии CI TorchTitan; невалидный конфиг падает до запроса ресурсов кластера.

---

### `AutometricsJobGIFT` (`job_submitter.jobs.gift.autometrics`)

Назначение: запуск PAM-автометрик и inference-режима для экспортированного GIFT-чекпоинта.

- `checkpoint_dir: str` **(обяз.)** - полный Hugging Face checkpoint с `config.json` и `*.safetensors`.
- `path_to_config: str | None = None` - путь к PAM-конфигу; без него используется `<checkpoint_dir>/artifacts/pam_config.yaml`.
- `mode: str = "autometrics"` - `autometrics`, `convertation` или `inference`.
- `sglang_tag: str = "custom"` - PAM/SGLang-окружение (`opensource` или `custom`).
- `gift_root_path: str | Path | None = None` - корень TorchTitan; далее `GIFT_ROOT_PATH`, затем vendored layout.

Класс использует PAM-образы и запускает `contrib/pretrain_auto_metrics/node_runner/node_runner.py`. Нативный TorchTitan DCP-чекпоинт не принимается: перед замерами его нужно экспортировать в HF safetensors и положить рядом HF `config.json`/tokenizer assets.

### `AutometricsJobGIFTExternalAPI` (`job_submitter.jobs.gift.autometrics_external_api`)

Назначение: PAM-обстрел уже поднятого OpenAI-compatible/IFT API без локального чекпоинта. Поддерживается Cloud; конфиг валидируется общим `AutometricsExternalAPIAssertor`.

---

### `AutometricsJobLLMF` (`job_submitter.jobs.llmf.autometrics`)

Назначение: запуск `pretrain_auto_metrics` для LLMF.

Унаследованные обязательные параметры из `AutometricsJob`:

- `checkpoint_dir: str` **(обяз.)** - директория с чекпоинтом.
- `path_to_config: str | None = None` - путь к `pam_config.yaml` (если `None`, ожидается `<checkpoint_dir>/artifacts/pam_config.yaml`).
- `mode: str = "autometrics"` - режим (`autometrics`, `convertation`, `inference`).
- `sglang_tag: str = "custom"` - вариант sglang-окружения для выбора базового образа и внешних библиотек:
  - `opensource` - использовать sglang из базового образа;
  - `custom` - добавить локальный `sglang_pretrain/python` в `PYTHONPATH`.

---

### `AutometricsJobVERL` (`job_submitter.jobs.verl.autometrics`)

Назначение: запуск `pretrain_auto_metrics` в контуре VERL.

Параметры такие же, как у `AutometricsJobLLMF`:

- `checkpoint_dir: str` **(обяз.)**
- `path_to_config: str | None = None`
- `mode: str = "autometrics"`
- `sglang_tag: str = "custom"` - `opensource` использует sglang из образа, `custom` добавляет локальный `sglang_pretrain/python` в `PYTHONPATH`.

---

### `AutometricsJobPAM` (`job_submitter.jobs.pam.autometrics`)

Назначение: запуск `pretrain_auto_metrics` в контуре PAM.

Параметры такие же, как у `AutometricsJobLLMF`:

- `checkpoint_dir: str` **(обяз.)**
- `path_to_config: str | None = None`
- `mode: str = "autometrics"`
- `sglang_tag: str = "custom"` - `opensource` использует sglang из образа, `custom` добавляет локальный `sglang_pretrain/python` в `PYTHONPATH`.

---

### `RayClusterJobVERL` (`job_submitter.jobs.verl.cluster`)

Назначение: поднятие Ray-кластера для VERL.

Важно:
- сейчас поддерживается только `submission_type="cloud"`;
- для `submission_type="slurm"` джоба завершается с ошибкой (`Ray cluster is not supported for slurm yet`).

Дополнительные параметры от `RayBaseJobVERL`:

- `engine: str = "sglang"` - engine для выбора базового образа (`sglang` или `vllm`).
- `use_sharp: bool = False` - включить SHARP/NCCL related envs.

---

### `RayTrainJobVERL` (`job_submitter.jobs.verl.train`)

Назначение: запуск train-скрипта VERL с runtime env конфигом и, опционально, PAM autometrics-конфигом.

- `script_path: str` **(обяз.)** - путь к train shell-скрипту.
- `runtime_env_config_path: str` **(обяз.)** - путь к YAML runtime env для Ray.
- `pam_config_path: str | None = None` - путь к `pam_config.yaml` (если нужно запускать autometrics-часть).
- `pam_n_workers: int = 2` - число нод для PAM/autometrics валидации.
- `pam_queue_name: str | None = None` - очередь для PAM/autometrics submit; прокидывается в Ray `runtime_env` как `PAM_QUEUE_NAME`.
- `extra_mounts: str | None = "/dev/shm:/dev/shm"` - дефолтный mount для shared memory.
- `engine: str = "sglang"` - унаследовано от `RayBaseJobVERL`.
- `use_sharp: bool = False` - унаследовано от `RayBaseJobVERL`.

---

### `TestJobLLMF` (`job_submitter.jobs.llmf.test`)

Назначение: тестовый запуск произвольного LLMF-скрипта.

- `script: str` **(обяз.)** - команда запуска.
- `args: argparse.Namespace` **(обяз.)** - namespace аргументов теста (используется, например, для выбора `n_gpus` при `modality == "UNIT"`).

---

### `TestConvertationJobGigachatUtils` (`job_submitter.jobs.gigachat_utils.test_convertation`)

Назначение: тестовая джоба для сценариев convertation в gigachat-utils.

- `script: str` **(обяз.)** - команда запуска теста/скрипта.

## Важные проверки и ограничения

- Для `Autometrics*`:
  - `checkpoint_dir` должен существовать.
  - `mode` только `autometrics|convertation|inference`.
  - `sglang_tag` только `opensource|custom|custom_vision`.
  - конфиг autometrics проверяется детально (ключи, валидность `tp/dp/pp`, пути и т.д.).
- Для `TrainJobLLMF`:
  - `path_to_config` должен существовать;
  - если в train-конфиге включен autometrics, проверяется согласованность с autometrics-валидатором;
  - если включен metric sampler (полем `metric_sampler` или в train YAML), проверяются его настройки, мониторы, параллелизм и ёмкость бэкенда (см. [Metric sampler](#metric-sampler)).
- Для `TrainJobGIFT`:
  - задан `config_file`;
  - `config_file` существует, `gift_root_path` указывает на checkout с `gift_scripts/train.py`, `torchtitan/train.py` и `scripts/config_check.py`;
  - `scripts/config_check.py <config_file> <script_args>` завершается успешно (структурная валидация YAML, импортов, имён секций и компонентов; полный typed decode без зависимости от torch);
- Для `RayTrainJobVERL`:
  - `script_path` и `runtime_env_config_path` должны существовать;
  - при наличии `pam_config_path` он также валидируется.
- Базовые ограничения:
  - `n_gpus` только `1/2/4/8`;
  - для slurm partition `middle` на `o5` разрешено до `64` нод; для больших запусков нужен `priority_class="high"` или другой non-`middle` partition.

## Dry run режим

Если у джобы выставлен `dry_run=True`, то `submit()`:

- все равно собирает итоговый конфиг (`to_config()`);
- все равно запускает валидации (`assert_job(...)`);
- выполняет `_before_submit()` (например, для локальной подготовки артефактов/метаданных);
- **не вызывает** реальный сабмит в кластер;

Это удобно для безопасной проверки параметров и конфигов перед фактическим запуском.

## Пример: запуск VERL train

```python
from job_submitter.jobs.verl.train import RayTrainJobVERL

job = RayTrainJobVERL(
    user_login="your_login",
    exp_name="verl-train-exp",
    script_path="/path/to/train.sh",
    runtime_env_config_path="/path/to/runtime_env.yaml",
    pam_config_path="/path/to/pam_config.yaml",
    n_workers=8,
)

result = job.submit()
print(result)
```

## Сабмит через публичный API ML Space

По умолчанию `CloudBackend` ходит во внутренний gateway `GWAPI_ADDR`, как и раньше. С внешних
хостов, например с CI-раннера, gateway недоступен, и тот же бэкенд можно переключить на публичный
API `https://api.ai.cloud.ru/public/v2` переменной окружения. Переключение покрывает сабмит,
`monitor()` и `js kill`:

| Переменная | Значение |
|---|---|
| `JOB_SUBMITTER_CLOUD_API` | `gateway` (по умолчанию) или `public` |
| `MLS_CLIENT_ID`, `MLS_CLIENT_SECRET` | Key ID и Key Secret персонального ключа доступа (консоль cloud.ru → Мой профиль → Ключи доступа); ключи сервисных аккаунтов Distributed Train не принимает |
| `MLS_API_KEY`, `MLS_WORKSPACE_ID` | `x-api-key` и `x-workspace-id` воркспейса (Distributed Train → Воркспейсы → Параметры разработчика) |
| `CLUSTER_KEY`, `MLS_NTB_WORKSPACE_NAME` | регион и воркспейс, например `SR008` и `wm-train`; на ноутбуке они есть в окружении, на раннере задаются явно |
| `MLS_API_URL` | адрес API, по умолчанию `https://api.ai.cloud.ru/public/v2` |

Токен `service_auth` живёт час и обновляется автоматически. Статусы API приводятся к тем, что ждёт
`monitor()`, без учёта регистра: `inqueue`, `starting`, `pending` → `Pending`, `running`, `completing`
→ `Running`, `completed` → `Completed`, `failed` → `Failed`.

Код: выбор режима в `job_submitter/core/cloud_api.py`, клиент API в `job_submitter/infrastructure/mlspace/`
(`config`, `auth`, `client`, `job_handle`).

## Где смотреть код джоб

- `job_submitter/jobs/base.py` - общая база и резолв параметров;
- `job_submitter/jobs/autometrics.py` - общая логика autometrics;
- `job_submitter/jobs/llmf/*` - LLMF-джобы;
- `job_submitter/jobs/gift/*` - GIFT/TorchTitan train-джобы;
- `job_submitter/jobs/verl/*` - VERL-джобы;
- `job_submitter/jobs/pam/*` - PAM-джобы.

## Автотрек

При сабмите train-джоб (например `TrainJobLLMF`) эксперимент автоматически регистрируется в tracker-daemon через S3 inbox.

Автоматический публичный шеринг включается отдельно. Автором считается
`user_login`, а в `auto_track_users` передаются дополнительные логины пользователей
кластера. Для Papaya бот сам преобразует их в Mattermost-логины через
`configs/username_mapping.json`:

```python
TrainJobLLMF(
    user_login="author_login",
    exp_name="experiment-name",
    path_to_config="/path/to/train.yaml",
    enable_auto_track=True,
    enable_auto_share=True,
    auto_track_users=["infidelis", "goida"],
).submit()
```

В общем канале появится один тред эксперимента с автором в `Shared by` и
дополнительными наблюдателями в `Mentioned users`. Повторный сабмит или retry с тем
же путём не создаёт новый тред, а добавляет в существующий только новых
пользователей. Если `enable_auto_track=False`, параметр `enable_auto_share=True`
игнорируется с предупреждением в логах.
