---
name: pam-measure
description: >
  Запуск и мониторинг замеров автометрик (PAM) — замер чекпоинтов через
  submit_pam_job и замер внешних OpenAI-compatible/IFT API через
  submit_external_api_pam. Использовать, когда просят запустить замер,
  автометрики, PAM-джобу или проверить её результаты.
---

# Запуск замеров PAM

Замеры сабмитятся **с кластера SR008**, из клона этого репо на кластере (`<PAM_REPO>`). Submit-скрипт создаёт джобу в клауде через `contrib/job_submitter`.

## Сначала: где мы?

Проверить, выполняемся ли мы уже на кластере: `test -d /home/jovyan && echo on-cluster || echo local`.

- **На кластере** — команды выполнять напрямую; `<PAM_REPO>` — текущий клон репо.
- **Не на кластере** — **уточнить у пользователя** данные подключения: ssh-хост или алиас кластера и путь к клону репо (`<PAM_REPO>`). Не угадывать. Дальше каждую команду выполнять через `ssh -T -o BatchMode=yes <host> "..."`.

## Перед запуском

1. Если работаем с локали — привести клон на кластере к текущей локальной ветке (см. «Git на кластере» ниже) — всегда, force, без вопросов. Если мы уже на кластере, синк не нужен.
2. Уточнить у пользователя, если не сказано: чекпоинт/API, конфиг, очередь.

## Конфиги замеров

Рабочие конфиги (и внешних моделей, и локальных) лежат на кластере в директории `configs` рядом с клоном (`<PAM_REPO>/../configs`) — новые конфиги сохранять туда же и в `path_to_config` передавать путь из этой директории. Примеры-шаблоны — в `configs/` репо, но рабочую копию под конкретный замер класть в `<PAM_REPO>/../configs`.

## Git на кластере (синк с локалью)

Аутентификация в gitlab.ai.cloud.ru на кластере работает из любого шелла через `credential.helper=store` (`~/.git-credentials`) — clone/pull/fetch не требуют ни токена в env, ни `zsh -ic`.

**Правило: перед замером с локали клон на кластере всегда жёстко приводить к состоянию текущей локальной ветки, не спрашивая.** Правки tracked-файлов на кластере при этом затираются — это ожидаемо. Untracked-файлы не трогать (`git clean` не делать).

```bash
# локально: запушить ветку
git push origin <branch>
# на кластере: жёстко привести к origin/<branch>
ssh -T -o BatchMode=yes <host> "cd <PAM_REPO> && git fetch origin && git checkout -f <branch> && git reset --hard origin/<branch> && git submodule update --init --recursive --force"
```

## Важно: только через `zsh -ic`

Креды клауда (`GWAPI_ADDR`, `GWAPI_KEY`, `NB_PREFIX`, `NAMESPACE`) заданы **только в интерактивном zsh**. Голый `python3 ...` в неинтерактивном шелле упадёт с `KeyError: 'NB_PREFIX'`. Поэтому любой сабмит и мониторинг оборачивать в `zsh -ic '...'`:

```bash
# на кластере
zsh -ic 'cd <PAM_REPO> && python3 - 2>&1' <<'EOF'
<python-код>
EOF

# с локали — то же самое через ssh
ssh -T -o BatchMode=yes <host> "zsh -ic 'cd <PAM_REPO> && python3 - 2>&1'" <<'EOF'
<python-код>
EOF
```

(`2>&1` — иначе traceback теряется; приветствие из zshrc в выводе — норма.)

## Сценарий 1: замер чекпоинта (HF или GigaFSDP)

Шаблон — `submit_pam_job.py` в корне репо. Не редактировать его на кластере; вместо этого передать скрипт через stdin (паттерн `zsh -ic ... python3 -` выше):

```python
from contrib.job_submitter.job_submitter.jobs.pam.autometrics import AutometricsJobPAM

AutometricsJobPAM(
    user_login="default",
    exp_name="Autometrics run for checkpoint <CHECKPOINT_DIR>",
    checkpoint_dir="<CHECKPOINT_DIR>",
    path_to_config="<PATH_TO_CONFIG>",  # можно "" — тогда берётся <checkpoint>/artifacts/pam_config.yaml
    queue_name="shared",
).submit()
```

- Примеры конфигов: `configs/` (`700b.yaml`, `430b_fp8.yaml`, `deepseek-v3.1.yaml`, ...). Конфиг копируется в `<checkpoint>/artifacts/pam_config.yaml` (старый уходит в `.yaml.backup`).
- Полезные параметры `AutometricsJobPAM` (наследует BaseJob): `mode` (`autometrics` | `convertation` | `inference`), `region` (`SR004` | `SR008`), `n_gpus` (default 8), `n_workers`, `priority_class` (`low`|`medium`|`high`), `sglang_tag` (`opensource` | `custom`).

- **`obm_judged`, `forge`, `gigametric_judged`, `tau2` требуют чётное число нод ≥ 2** (`n_workers=2`): движок судьи поднимается одновременно с движком оцениваемой модели, и инстанс судьи берётся с хвоста нод, а оцениваемой — с головы (`_group_nnodes`/`inference_instances` в `node_runner/src/inference/sglang/sglang_engine.py`). Если они не помещаются рядом, планировщик пропускает задачу на первом тике: в логе `[scheduler] skip <evaluator>@<judge>: [...] can no longer be placed together`, остальные обстреливалки идут дальше. В мультиноде логи судьи лежат в `artifacts/logs/mpiworker-0/`, а лог самого OBM — не в `logs/`, а в `artifacts/metrics/<evaluator>/autometrics.log`.

## Отладка OBM без пересборки образа (`extra_env` + `PAM_OBM_PATH`)

`extra_env: dict[str, str]` — параметр любой джобы (`BaseJob`, `contrib/job_submitter/job_submitter/jobs/base.py`), значения уходят в окружение процесса замера. Через него подсовывается чекаут OBM с NFS вместо OBM из образа: если задан `PAM_OBM_PATH`, эвалюаторы `obm` и `obm_judged` не распаковывают `obm.tar.zst`, а запускают код прямо из указанной папки. Работает для обоих сценариев (чекпоинт и external API).

```python
AutometricsJobPAM(
    ...,
    extra_env={"PAM_OBM_PATH": "/home/jovyan/enpimashin/new_obm"},
).submit()
```

- Путь — корень чекаута OBM (внутри лежит пакет `obm/`), доступный с ноды замера. Папка не удаляется и не перезаписывается.
- Питон остаётся из образа (`/home/user/obm_env/bin/python`): берётся твой код, но зависимости образа. Новые пакеты в `requirements` OBM так не подхватятся.
- `shooter_version` в JD и префикс в S3 читаются из `/home/user/obm_version.txt`, то есть покажут коммит OBM **из образа**, а не из чекаута. Для отладки нормально, публиковать такой замер как эталонный — нет.
- `obm.log` пишется рядом с чекаутом (cwd джобы на NFS) — виден сразу, не дожидаясь копирования в артефакты.
- Если в клоне PAM на кластере нет поддержки `PAM_OBM_PATH` (`ObmEvaluator._get_obm_path`), переменная молча проигнорируется и OBM возьмётся из образа — при отладке сверять по логу строку `Using OBM from ...`.

## Сценарий 2: замер external API (OpenAI-compatible / IFT)

Шаблон — `submit_external_api_pam.py`. Тот же паттерн:

```python
from contrib.job_submitter.job_submitter.jobs.pam.autometrics_external_api import AutometricsJobPAMExternalAPI

AutometricsJobPAMExternalAPI(
    user_login="default",
    exp_name="External API PAM run with config <PATH_TO_CONFIG>",
    path_to_config="<PATH_TO_CONFIG>",  # configs/ift.yaml или configs/openai.yaml
    queue_name="shared",
    # output_path="./output",  # если нужно
).submit()
```

- В конфиге должен быть заполнен `autometrics.inference.api_key` — ключ спрашивать у пользователя, в git не коммитить.

## Мониторинг джобы через client_lib

Опрос состояния джобы — через `contrib/job_submitter/job_submitter/c_lib/client_lib.py`, тем же паттерном `zsh -ic ... python3 -`:

```python
from contrib.job_submitter.job_submitter.c_lib import client_lib

client_lib.jobs(region="SR008")                       # все джобы: created_at : job_name : status (список длинный — обрезать через | head)
print(client_lib.job_status("lm-mpi-job-<uuid>", region="SR008"))  # -> "Job status=Running"
client_lib.logs("lm-mpi-job-<uuid>", tail=100, region="SR008")     # логи джобы в stdout
client_lib.kill("lm-mpi-job-<uuid>", region="SR008")  # убить джобу — только по явному запросу пользователя
```

- Статусы: `Pending` → `Running` → `Completed` | `Failed` | `Terminated`.
- Submit печатает `Job "<job_name>" created.` — сохранить job_name и показать пользователю.
- После сабмита не считать джобу успешной: опросить `job_status`; если `Pending` — она в очереди, это нормально; если `Failed` — забрать логи через `client_lib.logs(...)` и показать причину.
- **Когда джоба `Running` — логи читать из файлов чекпоинта** (`tail`/`ls` в директории артефактов; с локали — по ssh, см. «Результаты»), а не через `client_lib.logs`. К `client_lib` возвращаться только если файловые логи пустые/зависли или состояние неопределённое — джоба могла уже уйти в `Completed`/`Failed`.
- **После сабмита заводить мониторинг через `/loop 10m`** (скилл loop, cron-режим) с промптом вида: «мониторь PAM-джобу <job_name> на SR008: следи за файловыми логами в <artifacts>/logs/mpimaster-0/; если логи зависли или ошибка — проверь статус через client_lib; на первых проверках Running сделай смоук-проверку содержимого кэшей (см. секцию ниже); по завершении покажи метрики и удали cron-джоб через CronDelete». Луп живёт только в текущей сессии.
- При мониторинге писать пользователю короткий апдейт с временной меткой после каждой проверки — ход мониторинга всегда заканчивать текстовым апдейтом, молчаливых ходов не оставлять.

## Смоук-проверка содержимого (обязательно, на первых проверках Running)

Счётчик строк в кэше и HTTP 200 в логах не доказывают, что замер осмысленный: джоба может часами жечь токены, записывая мусор (пример: obm_judged с судьёй, отвечающим не в том формате, — все запросы 200 OK, кэш растёт, но в каждой записи `"scores": []`, и нулевая метрика обнаруживается только по завершении). Как только появились первые записи результатов (первая-вторая проверка после перехода в Running), **прочитать 1–2 записи целиком** и проверить содержимое по типу эвалюатора:

- **obm_judged** (`metrics/obm_judged/cache/<task>.jsonl`): непустые `scores`, `n_parse_failures == 0`. Пустые `scores` при заполненных `judge_completions` = вердикты судьи не парсятся — сразу алерт пользователю с сырым примером ответа судьи, не ждать конца джобы.
- **obm** (`metrics/obm/cache/*.jsonl`): `completion.content` непустой, не обрывок и не сообщение об ошибке провайдера.
- **dredd**: считать в логе `Judgment failed` / `Missing verdict labels` — единичные таймауты норма, сплошные фейлы — алерт.
- **lm_eval / mera**: первые samples (`log_samples`) непустые и парсятся таск-экстрактором (не сплошные 0.0 по первым батчам).
- Общее правило для любого judged/нового бенча: проверить, что **скоринговый путь работает end-to-end на первых сэмплах**, а не только транспорт. Если скоры пустые/нулевые — показать сырой пример и спросить пользователя, продолжать ли джобу.

На последующих проверках достаточно дельты счётчиков + периодически (раз в ~час) повторять выборочную проверку одной записи.

## Результаты

- Логи замера: `<checkpoint>/artifacts/logs`.
- External API: если в джобу не передан `output_dir`, артефакты пишутся в `<PAM_REPO>/<autometrics.inference.model>` (например `<PAM_REPO>/GigaChat-3.5-432B-A28B`) — логи и метрики там, без промежуточного `artifacts/`.
- Метрики: `<checkpoint>/artifacts/metrics`, OBM — `.../metrics/obm/all_metrics.jsonl` (jsonl: `name`, `metric.name`, `metric.value`).
- TB — путь из конфига (`tensorboard_path`).
