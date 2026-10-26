# metric-sampler

Метрики обучения по каждому рангу и микробатчу: считаются на GPU во время шага,
собираются узлом в окна на NFS, коллектор переносит их в S3, ClickHouse
строит из них сводки для Grafana.

## Устройство пакета

Пакет разделён по стадиям конвейера; зависимости идут только сверху вниз, и
`sber_tests/unit/test_architecture.py` это проверяет.

| Пакет | Что делает | Зависит от |
| --- | --- | --- |
| `contract` | общий язык стадий: каталог метрик (`catalog.py`), их типы и раскладка значений (`metrics.py`), имена каталогов и файлов на NFS (`raw_layout.py`) | ничего |
| `capture` | сторона ранга: раскладка слотов (`layout.py`, `layout_plan.py`), кольцо в shared memory, рекордер, дампы активаций; `stats/` — статистики семейств, `kernels/` — Triton/CUDA | `contract` |
| `rank_capture.py` | конец захвата одного ранга для любого фреймворка: claim на попытку, кольцо, рекордер, дампы, процесс узла | `capture`, `node` |
| `node` | процесс узла: сливает кольца рангов в окна Parquet | `contract`, `capture` |
| `island` | коллектор: окна с NFS в S3 | `contract` |
| `cloud` | SQL ClickHouse: приём, сводки, представления для Grafana | — |

Интеграция с фреймворком обучения (сейчас колбэк llm-foundry) находит модули
модели, подключает наблюдатели и вызывает ядро пакета: `LayoutPlan` строит
раскладку, `DumpSettings` разбирает настройки дампов, `RankCapture` открывает и
закрывает ресурсы ранга, а статистики из `capture/stats` (в том числе
`MonitorActivationStats` для модулей, за которыми следит монитор активаций)
считают значения.

Сырые окна — источник правды: они неизменяемы и только дописываются. Сводки в
ClickHouse — проекции сырых данных; их можно пересчитать, сырые — нет.

## Router load by data stream

`router/stream_tokens_per_expert` counts each actual top-k router assignment in
the source token's data stream, before label/loss validity masking and before
dispatch or capacity handling. It includes input padding rows that reached the
router and does not include synthetic capacity padding. Its per-rank,
per-microbatch counts sum over streams to `router/tokens_per_expert` for the same
rank, step, layer and microbatch. The Experts dashboard shows expert share
within a stream, stream mix within an expert, per-stream assignment totals and
per-stream MaxVio. Those ratios use only the selected rank and microbatch
observations; MaxVio is `(largest expert count - mean expert count) / mean`
within one stream and is undefined when that stream has no assignments. Compare
these with the global MaxVio panel to distinguish overall expert imbalance from
concentration of particular streams.

## Раскладка на NFS

Пишется `metric_sampler.node.aggregator`:

```
<raw_root>/<exp>/<run>/attempt=<A>/metrics.json
<raw_root>/<exp>/<run>/attempt=<A>/node=<host>/w00000001-00000010/{scalars,expert_counts,router_vectors}.parquet
<raw_root>/<exp>/<run>/attempt=<A>/node=<host>/w00000001-00000010/manifest.json
<raw_root>/<exp>/<run>/attempt=<A>/node=<host>/_spool/
<raw_root>/<exp>/<run>/attempt=<A>/node=<host>/aggregator.log
```

Окно считается завершённым, когда в его директории появился `manifest.json`
(он пишется последним); файла семейства метрик может не быть, если в окне не
было строк этого семейства.

Callback и logger используют `window_steps: 10`. Штатная остановка публикует
остаток: после 75 шагов остаётся отдельное окно 71–75. Узловой процесс сначала
записывает полученные rank/step-данные в сжатый Zstd-журнал на NFS с `fsync`,
затем освобождает слот кольца. Журнал удаляется после публикации Parquet-окна.
Заполненное кольцо или очередь logger теперь вызывают явную ошибку вместо
продолжения обучения с молча пропущенными метриками. Заполнение кольца доступно
в `sampler/ring_used_slots` и `sampler/ring_capacity_slots`.

После аварийного завершения всех training-процессов остаток журнала можно
опубликовать командой от пользователя обучения:

```bash
metric-sampler-recover-node <raw_root>/EXP/RUN/attempt=ATTEMPT/node=HOST
```

Команда отказывается работать при живом writer или training-rank, удерживающем
lock. Требуются согласованные версии callback и пакета: callback удерживает
rank-lock до окончания записи. Повторное восстановление безопасно.
Если коллектор уже перенёс окно, восстановление сверяет его manifest в
`_collected`. Пока writer не подтвердил публикацию, retention сохраняет это окно.
Гарантия журнала начинается после записи на NFS: внезапная потеря узла не
сохраняет ещё находившиеся на GPU данные. Асинхронная очередь logger тоже
становится устойчивой только после записи своего окна.

## Коллектор (`metric_sampler.island`)

`Collector` работает на узле, где нет сервера ClickHouse, но есть `chdb`
(in-process движок). За один цикл он:

1. для каждой попытки (`attempt`) выбирает одну партию старейших готовых окон,
   до 512 MiB исходных Parquet или 128 окон; одно неделимое окно больше лимита
   обрабатывается отдельно. Список фиксируется на диске до загрузки, поэтому
   новые окна не меняют состав повторной попытки;
2. для каждого семейства метрик, встретившегося в батче, вставляет объединённые
   строки в S3 одним Parquet-файлом (`metric-sampler/v3/family=<family>/...`),
   явным списком путей окон (без glob), сжатие `zstd`;
3. сверяет число строк в S3 с суммой строк по исходным файлам; при расхождении
   бросает `CollectError` и не трогает и не загружает ничего дальше;
4. загружает манифесты узлов одним объектом (побайтовые копии `manifest.json`
   окон, по документу на окно), затем манифест батча (`.../batches/...json`, коммит-маркер батча,
   загружается последним);
5. переносит собранные директории окон в `<raw_root>/_collected/...`
   (`os.renames`) и удаляет из `_collected` директории старше
   `--collected-retention-hours`.

При сбое между загрузкой и перемещением директорий следующая попытка продолжает
ту же партию. Окна незавершённой партии и окна без подтверждения producer
защищены от удаления по retention.

Реестр метрик (`metrics.json`) загружается один раз на попытку и повторно —
только если его содержимое изменилось.

Учётные данные для S3 берутся из `~/.config/rclone/rclone.conf` (секция с
`access_key_id`/`secret_access_key`) и никогда не логируются.

## Запуск

```bash
python -m metric_sampler.island.cli \
    --raw-root /path/to/node \
    --remote-section <rclone-section> \
    --endpoint https://<s3-endpoint> \
    --bucket <bucket> \
    --prefix metric-sampler/v3 \
    --chdb-config /path/chdb-config.xml \
    --collected-retention-hours 24 \
    --loop-seconds 60
```

В Slurm это же запускается через `scripts/island_collector.sbatch`.
Постоянно работает только наблюдатель `scripts/island_collector_watch.sbatch` на одном CPU: он отправляет джобу коллектора,
когда на NFS ждут готовые окна, а коллектор завершается после 30 минут без окон. Адреса, пути и секции S3 скрипты берут из
окружения, с которым отправлены, в репозитории их нет:

| Переменная | Значение |
| --- | --- |
| `METRIC_SAMPLER_RAW_ROOT` | корень окон узлов на NFS |
| `METRIC_SAMPLER_RCLONE_CONF` | rclone.conf с ключами S3 |
| `METRIC_SAMPLER_S3_SECTION`, `METRIC_SAMPLER_S3_ENDPOINT`, `METRIC_SAMPLER_BUCKET` | секция, адрес и бакет S3 окон |
| `METRIC_SAMPLER_DUMP_S3_SECTION`, `METRIC_SAMPLER_DUMP_BUCKET`, `METRIC_SAMPLER_DUMP_PREFIX` | то же для дампов |
| `METRIC_SAMPLER_CHDB_CONFIG` | конфиг chdb |
| `METRIC_SAMPLER_COLLECTOR_SHARDS` | число шардов коллектора |
| `METRIC_SAMPLER_COLLECTOR_SBATCH` | путь к `island_collector.sbatch` для наблюдателя |

Ресурсы (`--cpus-per-task`, `--mem`) и файл лога (`--output`) задаются при `sbatch`.

## Тесты

```bash
python -m venv .venv
.venv/bin/pip install -e .[dev]
python -m pytest --cov=metric_sampler --cov-branch --cov-report=term-missing
ruff check .
mypy
```

Юнит-тесты не требуют кластера и `chdb`: chdb-сессия подменяется фейком в
`sber_tests/conftest.py`. `ruff` проверяет весь репозиторий, `mypy` в строгом режиме проверяет пакет
(настройки обоих в `pyproject.toml`).
