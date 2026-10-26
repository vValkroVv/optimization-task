# Stability — диагностика стабильности обучения MoE-моделей

Набор из четырёх независимых тулкитов для анализа стабильности pre-training
ранов: по TB-логам и по HF-чекпойнтам.

```
sber_scripts/analytics/stability/
├── instability/              # TB-логи → причинная диагностика одного/многих ранов
├── weights/                  # HF safetensors чекпойнты → per-tensor / cross-run диагностика
├── offline_monitoring_scripts/  # TB-логи → интерактивный Plotly per-layer дашборд (HTML)
├── small_model_proxy_utils/  # Поиск развала и критического lr → таблица "max stable lr / min broken lr"
└── SKILL.md                  # агентский плейбук: как из этих тулов собрать HTML-отчёт
```

Каждый модуль самодостаточен и имеет **свой подробный README** 

---

## Какой модуль когда

| Вход | Вопрос | Модуль |
|---|---|---|
| TB run dir (`events.out.tfevents.*`) | «что сломалось в этом ране и почему» | [`instability/`](instability/README.md) |
| Несколько TB run dirs | «чем раны отличаются, какие есть выбросы, когда разошлись» | [`instability/`](instability/README.md) (`compare`) |
| HF export (`<iter>/artifacts/hf_checkpoint/`) | «здоровы ли веса чекпойнта» | [`weights/`](weights/README.md) |
| Несколько HF export'ов | «какие тензоры разъехались между ранами/итерациями» | [`weights/`](weights/README.md) (`compare`) |
| 1–5 TB рана, нужно «покликать» | интерактивный per-layer дашборд для глазами | [`offline_monitoring_scripts/`](offline_monitoring_scripts/README.md) |
| LR-exp (десятки TB ранов `…-lr_X-…`) | «до какого LR сетап держится» | [`small_model_proxy_utils/`](small_model_proxy_utils/) |

Финальный артефакт ллм-диагностики — **один самодостаточный файл `story.html`** для каждого модуля (instability/weights)
(см. `SKILL.md`). 

---

## Общие правила запуска

* `instability` и `weights` — это **пакеты**, запускаются как `python -m` **из
  корня `sber_scripts/analytics/stability/`**:

  ```bash
  cd sber_scripts/analytics/stability
  python -m instability.report --tb-dir /path/to/RUN --out-dir output/exp/report
  python -m weights report output/exp/analyze.json
  ```

* `offline_monitoring_scripts` и `small_model_proxy_utils` — обычные скрипты,
  запускаются изнутри своей папки:

  ```bash
  cd offline_monitoring_scripts && python3 build_dashboard.py out/ run-a/ run-b/
  cd small_model_proxy_utils  && ./run_stability_comparison.sh /path/to/lr_sweep
  ```

* Зависимости — `requirements.txt` в каждой папке. 
* GPU нужен только для `weights analyze` / `weights joint` / `weights batch`.
  Всё остальное — CPU.


---

## Модули

### `instability/` — TB-логи → причинная диагностика

Парсит tensorboard event-файлы и дает аналитику по ранам.

**Команды** (`python -m instability.<cmd>`):

| Команда | Что | Вход |
|---|---|---|
| `report` | один ран → 14 кураторских PNG + `atlas/` (heatmap на КАЖДОЕ семейство мониторов) + CSV/MD + `summary.md` + `signals.json` | один TB dir |
| `compare` | N-ран diff: где раны расходятся, какие выбросы, когда открылся гэп; side-by-side панели | TB root + `--runs` |
| `temporal` | forward-lag prediction / cross-corr / Lasso AR-X + event-study — «что выросло раньше» | сэмпл ранов |
| `eda` | inventory десятков ранов + корреляции монитор↔max_vio | TB root |
| `projections` | то же, но сгруппировано по архитектурной части (router/up/down/attention/residual) | сэмпл ранов |
| `heatmap` | ad-hoc `layer × step` heatmap для любого `<tag>/layer_N` семейства | один TB dir |
| `tags` | **аудит тегов**: инвентарь всех мониторов рана + что не классифицировано/не грузится | один TB dir |
| `story` | собрать HTML из json-конфига + папки figures | config + figures |


Кэш парсинга — `~/.cache/instability_tb/{hash}.pkl.gz`, ключ `(path, mtime_ns,
size)`. Cold ≈ 100s на ~450 event-файлов, warm ≈ 0.1s. Сброс:
`rm -rf ~/.cache/instability_tb/`.

**Подробности:** [`instability/README.md`](instability/README.md).

---

### `weights/` — HF-чекпойнты → диагностика весов

Анализ safetensors-экспорта. Per-tensor моменты/спектр + MoE-joint + cross-run
diff.

**Команды** (`python -m weights <cmd>`):

| Команда | Что | GPU? |
|---|---|---|
| `analyze` | per-tensor `base` (моменты/гистограммы) + `rowcol` (rank-1-collapse) + `spectral` (top-K SVD) → `analyze.json` | да |
| `joint` | MoE joint Gram `Σ WᵀW` по `(layer, proj)` + gate↔up alignment → `joint.json` | да |
| `batch` | прогнать `analyze`/`joint` по всем `ep0_baN/artifacts/hf_checkpoint/` рана | да |
| `report` | компактный текстовый отчёт по JSON-ам (HEADLINES, per-group, outliers, multi-file deltas, trajectory) | нет (stdlib) |
| `compare` | cross-RUN diff: где N analyze.json расходятся per `(tensor, metric)` | нет (stdlib) |
| `dashboard` | PNG-дерево из analyze.json (+ joint.json) | нет |
| `story` | собрать shareable HTML из конфига + dashboard-папки | нет |

**Использование** — один чекпойнт:

```bash
# GPU-часть (analyze/joint)
python -m weights analyze .../hf_checkpoint --out output/exp/analyze.json --device cuda
python -m weights joint   .../hf_checkpoint --out output/exp/joint.json   --device cuda -k 4
# CPU-часть
python -m weights report  output/exp/analyze.json output/exp/joint.json
python -m weights dashboard output/exp/analyze.json output/exp/joint.json --out-dir output/exp/dashboard
python -m weights story    output/exp/story_config.json output/exp/dashboard --out output/exp/story.html
```

**Подробности:** [`weights/README.md`](weights/README.md).

---

### `offline_monitoring_scripts/` — интерактивный Plotly-дашборд

Один  HTML с per-layer метриками на Plotly: дропдауны по группам
метрик, layer-маркеры (обычный блок / нелинейный / MTP), сравнение 1–2 ранов.
В отличие от `instability.story` (статичные PNG + narrative) — это **живой
кликабельный** дашборд без текста-диагноза.

```bash
cd offline_monitoring_scripts
source venv/bin/activate
python3 build_dashboard.py out/ pretrain-v1/            # один ран
python3 build_dashboard.py out/ pretrain-v1/ pretrain-v2/   # сравнение двух
```

Структура: `dashboard/{patterns,collector,derivatives,builder}.py` +
`build_dashboard.py` (CLI). **Подробности:**
[`offline_monitoring_scripts/README.md`](offline_monitoring_scripts/README.md).

---

### `small_model_proxy_utils/` — классификатор развала

Берёт папку с раном-на-LR (имена вида `…-lr_0.01-…`, например `PapayaSeek-1B-lr0.0075-gs2048-ss2048-edges-router+experts-C0.7`), читает TB-логи параллельно
(ProcessPool), прогоняет collapse-критерии и выдаёт по каждому **setup** строку
`max_stable_lr / min_broken_lr`. Используется как дешёвый прокси «до какого LR
сетап не разваливается» на маленьких моделях.

```bash
cd small_model_proxy_utils
./run_stability_comparison.sh /path/to/lr_sweep_dir            # дефолтные пороги + 64 воркера
# или напрямую:
python run_stability_comparison.py /path/to/lr_sweep_dir artifacts/ stability_res \
    --min-full-steps 16000 --max-vio-threshold-share 0.2 --workers 64 --use_cache
```

Выход в `artifacts/`: `<name>.csv` (по сетапам), `<name>_runs.csv` (по ранам,
кэш для `--use_cache`), `<name>.md` (markdown-таблица, печатается в конце).

**Состав:**
* `run_stability_comparison.py` — CLI + параллельное чтение event-файлов.
* `src/io_utils.py` — парсинг имён ранов (`lr`, `setup_name`), чтение TB-скаляров
  в `RunTrace`, запись CSV/MD.
* `src/criteria.py` — collapse-критерии (`check_max_vio_layerwise_topk_average`
  активен; `activation_amax` / `loss_spike` закомментированы — пороги не
  откалиброваны).


---

## Распознавание тегов: задаёшь верхние семьи — остальное парсим сами

Принцип `instability`: **называешь только headline-семьи** (важные мониторы,
которым нужно своё человеческое имя группы), а все прочие теги тулкит
классифицирует **структурно сам** — по форме тега, без ручных правил. Когда в
composer добавляют новый монитор, дописывать паттерны обычно не нужно.

**Где задаются верхние семьи** — список `(подстрока → имя группы)`. Способы (по
приоритету, последующий перебивает предыдущий):

1. **В конфиге** — [`instability/config.py`](instability/config.py) →
   `MONITOR_GROUP_ALIASES` (встроенный набор: `l2_norm/grad → grad_norm`,
   `moe_seq_aux_loss → aux_loss`, …). Правь прямо тут для постоянных семей.
2. **Файлом-оверрайдом** — `$INSTABILITY_TAG_GROUPS` = путь к JSON
   `{"подстрока": "группа"}`; мёржится поверх встроенных при импорте.
3. **Флагом/в рантайме** — `python -m instability.tags --tb-dir RUN --tag-groups my.json`,
   или `features.register_tag_groups({...})` из кода. Семьи юзера ставятся
   **впереди** встроенных и выигрывают.

**Что происходит со всем остальным:**

* **Классификация** — [`features.py`](instability/features.py):
  `monitor_group(tag)` сначала пробует алиасы (твои верхние семьи), не нашёл —
  `_structural_group` выводит группу из префикса (`activations_kurtosis/… →
  activation_kurtosis`, `l2_norm/whatever → l2_whatever`, `*max_vio* → max_vio`).
  Глобальные скаляры (`loss/`, `lr-`, `time/`, …) остаются `other` и в per-tag
  анализы не лезут. `parse_tag` отдаёт `(group, projection, layer, family)`.
* **Загрузка — deny-list** — [`tb_helpers.py`](instability/tb_helpers.py) →
  `default_load_keep`: по умолчанию грузится **любой** монитор, кроме не-модельного
  шума (`data_profiler`, `timers_callback`, `metrics/`, `metrics_gauntlet`,
  `token_accuracy`, `optim_health`, `trainer/`). Любой тег с `residual` грузится
  всегда (включая slash-форму агрегата); прочая slash-форма
  `activations/…layers.{}…` без индекса слоя — отбрасывается (per-layer тулы её не
  ставят на ось, а per-layer близнецы `activations_<stat>/…layers.N…` и так грузятся).
  `DEFAULT_PREFIXES` остался как опциональный allow-list для явного
  `load_run_scalars(..., prefixes=DEFAULT_PREFIXES)`.
* **Графики** — [`atlas.py`](instability/atlas.py) рисует heatmap на каждое
  семейство; `compare`/`temporal`/`eda` берут всё с `group != "other"`.
* **Аудит** — `python -m instability.tags --tb-dir RUN` показывает весь набор
  мониторов рана: families по группам, `UNRECOGNIZED` (упали в `other` — кандидаты
  на новую headline-семью) и `LOGGED BUT NOT LOADED BY DEFAULT` (что deny-list
  отбросил — в норме только шум + non-residual slash-форма). `--csv` — выгрузить
  per-tag классификацию.

Итог: новый монитор подхватывается во все пайплайны **сам** (deny-list грузит всё
кроме шума → atlas + compare + temporal). Руками — только: (а) дать монитору
человеческое имя группы (одна строка в `MONITOR_GROUP_ALIASES` / `--tag-groups`),
(б) поднять его в одну из 14 кураторских фигур `report.py` или задать
direction/threshold, или (в) добавить семейство в `_NOISE_PREFIXES`, если новый
монитор — это шум, который НЕ нужен в анализе (`tags` подскажет).

---

## Как добавить новую фичу

| Хочу… | Где |
|---|---|
| **проверить, что новый монитор виден** | `python -m instability.tags --tb-dir RUN` — инвентарь + `NOT LOADED BY DEFAULT`. По умолчанию монитор грузится сам (deny-list); строку трогаешь только если это **шум** — тогда добавь префикс в `instability/tb_helpers.py:_NOISE_PREFIXES`. Классификацию новый тег получает сам (`features.py:parse_tag`/`_structural_group`). |
| **новую `instability`-команду** | добавь `instability/<cmd>.py` со своим `argparse`-`main()`; запускается как `python -m instability.<cmd>`. Общие хелперы — в `features.py` (классификация тегов, `growth_onset`) и `tb_helpers.py` (чтение TB, кэш). Дефолты ранов-селекции — Python-константы в `config.py` (`EDA_DEFAULTS`/`TEMPORAL_DEFAULTS`/`PROJECTIONS_DEFAULTS`) |
| **новую `weights`-команду** | добавь модуль `weights/<cmd>.py` с функцией `run(...)`, затем в `weights/__main__.py`: `_cmd_<name>`  + `sub.add_parser(...)` + `p.set_defaults(func=...)`. |
| **новую метрику в `weights analyze`** | добавь вычисление в `analyze.py`; чтобы она попала в отчёт — внеси в curated-набор `report.py:ANALYZE_CURATED` с направлением ранжирования (high/low/abs/alert). |
| **группу метрик в Plotly-дашборде** | `offline_monitoring_scripts/dashboard/builder.py` → `_GROUPS`. Derived-метрика → `derivatives.py:compute_derivatives()`. Новый источник сырых метрик → `_collect_*` в `collector.py` + вызов из `collect_run()`. Новый activation-паттерн → `patterns.py:ACTIVATION_PATTERN_TAILS` (или авто-дискаверится из TB-тегов). |
| **новую роль тензора** | если в `weights report` всплыл `role=unknown` — добавь regex в `analyze.classify()`. |

---

## Как поделиться отчётами по сети (без скачивания HTML-ок)

Все отчёты — self-contained HTML (картинки либо встроены, либо лежат рядом в
`figures/`). Чтобы не качать/перекидывать файлы, можно поднять **HTTP-сервер прямо из
нужной папки** и поделиться ссылкой через notebook-прокси.

```bash
cd output/<эксперимент>          # папка, где лежит story.html (+ figures/)
python3 -m http.server <port>    # например 9924
```

Затем откройте через прокси своего notebook-инстанса — берём базовый URL ноутбука
и дописываем `/proxy/<port>/`:

```
https://sr008-nb.ai.cloud.ru/sr008/notebook/<id>/proxy/<port>/
                                                      └────────┘
                                            тот же <port>, что и в http.server
```
---

## Агентский плейбук

[`SKILL.md`](SKILL.md) — как Claude/агент собирает из этих тулов единый
причинный HTML-отчёт: правила («каждое утверждение → график», «реальные данные,
не светофоры»), single-run / multi-run / checkpoint флоу, схема story-config,
anti-patterns. Если работаешь через агента — начни с него.
</content>
</invoke>
