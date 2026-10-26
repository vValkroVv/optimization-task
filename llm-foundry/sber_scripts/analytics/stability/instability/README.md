# instability/ — пайплайн анализа развала MoE-чекпойнтов по TB-логам

4 standalone-скрипта поверх tensorboard event-файлов. 

## Кэш TB-парсинга

Парсинг event-файлов — узкое место (на 458 файлов ≈ 100s). После
первого прогона результат лежит в `~/.cache/instability_tb/{hash}.pkl.gz`.
Хэш считается по `(путь, mtime_ns, size)` каждого event-файла:

* Если event-файлы не менялись — все 4 скрипта на этом TB запускаются
  **за 0.1–0.3s** (читают pickle вместо TB).
* Если хоть один файл изменился (новый чекпойнт дописан) — хэш меняется,
  парсим заново.

Команды:

```python
from instability.tb_helpers import clear_cache
clear_cache()              # удалить все cached blobs
```

или просто `rm -rf ~/.cache/instability_tb/`.

`load_run_scalars(..., cache=False)` обходит cache (диагностика).

---

## Что есть

| Скрипт | Что делает | Когда использовать |
|---|---|---|
| [`report.py`](report.py) | **Один ран → 14 кураторских PNG + `atlas/` + data tables + JSON + summary.md.** Все PNG в `OUT/figures/`, все таблицы и MD в `OUT/data/`. Глобальный обзор + max_vio heatmap + per-layer attribution + MoE deep-dive + activations/gradients/weights depth + system health + per-projection breakdown + router dynamics + loss decomposition + phase diagnosis + cross-layer correlation + spike clusters. **Плюс `atlas/<category>/<family>.png` — heatmap на КАЖДОЕ семейство мониторов рана** (см. ниже «Автоматическое распознавание тегов»). Для каждой кураторской PNG есть sibling CSV/MD → LLM может анализировать без парсинга пикселей. | Первое, что запускать на любом интересующем ране. Eyeball-режим без порогов. |
| [`tags.py`](tags.py) | **Аудит тегов**: грузит ВСЕ скаляры рана, классифицирует каждый (`parse_tag`), печатает инвентарь по группам, секцию `UNRECOGNIZED` и `NOT LOADED BY DEFAULT` (что deny-list отбросил). `--csv` дампит per-tag классификацию. | Дали новый TB — прогони первым: увидишь весь набор мониторов и не доезжает ли что-то до анализа. |
| [`atlas.py`](atlas.py) | **Plot-everything**: авто-дискаверит все per-layer семейства тегов и рисует `layer × step` heatmap на каждое — без ручных списков. Вызывается из `report.py` (можно и отдельно). | Когда нужно семейство, которого нет среди 14 кураторских фигур — оно уже в atlas. |
| [`compare.py`](compare.py) | **N-ран diff**: выравнивает каждый монитор по общему step-грид'у, ранжирует `(tag, layer)` по cross-run divergence, side-by-side панели + overlay. `--steps`/`--n-steps` — divergence trajectory (когда открылся гэп). | Несколько ранов одной модели: где и когда они разошлись. |
| [`eda.py`](eda.py) | **Inventory всех ранов в root**: features из имени, ключевые config-параметры, max_vio summary, корреляции монитор↔max_vio. CSV-выход. | Когда ранов десятки и нужна **таблица** «у кого что сломалось». |
| [`temporal.py`](temporal.py) | **Forward-lag prediction (на абсолютных уровнях) + cross-correlation + Lasso AR-X + event-study** вокруг `growth_onset`. Какие сигналы предсказывают взрыв `max_vio`. | Когда хочется **причинности**: «что выросло раньше — grad или max_vio?». |
| [`projections.py`](projections.py) | То же что `eda.py + temporal.py`, но по проекциям (`router / up / down / shared_expert / attention`) — отдельный CSV и PNG. | Когда хочется атрибутировать нестабильность к конкретной части архитектуры. |
| [`heatmap.py`](heatmap.py) | **Ad-hoc `layer × step` heatmap + CSV для произвольных `<tag>/layer_N` семейств.** Принимает один или несколько tag-префиксов (например, `moe_load_balance moe_maxvio`), строит layer×step тепловую карту + sidecar CSV. Reuses тот же кэш что и остальные скрипты. | Когда заметил per-layer сигнал в `report.py` и хочешь увидеть его эволюцию во времени по всему стеку. |
| [`story.py`](story.py) | **Shareable HTML-страница**: narrative-конфиг (JSON, который пишет агент/человек) + папка `figures/` (обычно из `report.py`) → один HTML с разделами Summary / Findings / Next steps + filterable gallery всех PNG. CSS/JS внутри скрипта; конфиг чисто содержательный. | Когда нужно поделиться анализом с командой одним файлом. |
| [`features.py`](features.py) | Helper-ы классификации тегов: `monitor_group`, `projection_group`, `family`, **`parse_tag` + `_structural_group`** (структурный разбор любого тега), `FEATURE_PATTERNS`, `growth_onset`. Используется остальными скриптами. | Не запускать напрямую. |
| [`tb_helpers.py`](tb_helpers.py) | Standalone TB-scalar reader + кэш + per-layer max_vio aggregator (`load_run_scalars`, `build_max_vio_series`, `parse_layer_from_tag`). **`default_load_keep`** — deny-list дефолтной загрузки (всё кроме `_NOISE_PREFIXES`; residual — всегда). `DEFAULT_PREFIXES` — опциональный allow-list. | Не запускать напрямую. |

---

## Как мы ищем нестабильность — 4 взаимодополняющих подхода

### 1. Eyeball-обзор одного ран-а — `report.py`

**Подход**: один PNG со всеми каноническими сигналами в едином step-масштабе:
- **Loss** (`loss/train/total`, `loss/train/loss_ce`) — есть ли спайки?
- **LR** (`lr-DecoupledAdamW/group0`) — warmup корректный, schedule не сломан?
- **Gradient L2** (`l2_norm/grad/global` + envelope по слоям) — взрыв?
- **max_vio** (`dispatcher_sum_metrics` + per-layer envelope) — router балансирует?
- **Activation amax** (топ-6 самых волатильных residual/output тегов) — есть прорыв?
- **Aux losses** (`moe_seq_aux_loss`, `hidden_z_loss`, `load_balance`) — регуляризация работает?

Плюс `max_vio_heatmap.png` — `[layer × step]` heatmap всех слоёв,
чтобы видеть **какие именно слои** дают пик `max_vio`, и в какие шаги
обучения. Второй panel — бинарная маска `max_vio > 4` для быстрого
визуального сканирования.

**Никаких порогов и verdict-а** — это чисто визуальный режим. Решение
«ран сломан или нет» оставляем человеку.

### 2. Корреляции монитор↔max_vio — `eda.py`

**Когда**: десятки/сотни ранов в root-папке TB, надо понять «какие
сигналы скоррелированы с глобальным max_vio».

**Подход**: для каждого ран-а считаем pearson(`max_vio`, `metric`) по
шагам. Затем агрегируем по `monitor_group` (см. `features.py`):
- `param_norm`, `grad_norm`, `update_norm` — L2-нормы;
- `aux_loss`, `router_z_loss`, `routing_load` — router-стороны;
- `activation_amax`, `activation_amean` — активации;
- `update_param_ratio`, `update_grad_gap`, `effective_lr` — derived metrics.

Дополнительно — **feature-conditioned correlations**: разница между
ранами **с фичей** vs **без фичи** (`norm_type-ZeroCenteredRMSNorm`,
`gated_attention-True`, `aux_loss_free_strategy-adaptive` и т.д. — 30+
паттернов из `features.py`). Это даёт hint «фича X увеличивает связь
grad_norm с max_vio» — кандидат на следующую проверку.

### 3. Временная причинность — `temporal.py`

**Когда**: нужно понять «что развалилось первым».

**Подход** — три независимых взгляда на lead/lag + event-study:
1. Detect `growth_onset` — первая точка, где `rolling_slope(log(max_vio))`
   удерживается выше `0.05 × baseline` минимум 5 окон.
2. **Forward-lag predictive F-test на абсолютных уровнях** (не на разностях —
   важны абсолютные значения): добавляет ли лаг сигнала X предсказательной силы к
   будущему `max_vio`. Плюс **cross-correlation** (знак пика = кто кого ведёт) и
   **Lasso AR-X** (интерпретируемые коэффициенты по лагам). Согласие трёх = вывод
   надёжен. С BH-FDR коррекцией.
3. Event-study: усреднённое поведение X вокруг `growth_onset`, нормированное на
   baseline. Видно «X рос за 20 шагов до взрыва».

### 4. Per-projection атрибуция — `projections.py`

**Когда**: уже знаем что развал есть и хотим узнать **в какой
архитектурной части**.

**Подход**: те же сигналы, но сгруппированные по
`projection_group(tag) → {router, up, down, shared_expert, attention,
moe_block}`. Отдельный CSV и PNG-сетка `[projection × metric]`. Сразу
видно «у router-а grad_norm растёт раньше всего» vs «у down_proj
spectral_norm выше нормы только на последних шагах».

---

## Автоматическое распознавание тегов

Новые мониторы появляются в composer регулярно. Чтобы **не дописывать руками**
паттерны на каждый — теги разбираются структурно. 4 уровня:

| Уровень | Где в коде | Что делает |
|---|---|---|
| **Загрузка — deny-list** | [`tb_helpers.py`](tb_helpers.py) → `default_load_keep` | По умолчанию грузится **любой** монитор, кроме не-модельного шума (`_NOISE_PREFIXES`: data_profiler, timers_callback, metrics/, metrics_gauntlet, token_accuracy, optim_health, trainer/). Исключение из `metrics/` — `_VAL_LOSS_PREFIXES` (`metrics/eval/LanguageCrossEntropy`, `metrics/eval/LanguagePerplexity`): это валидационный NLL и его перплексия, они есть только в ранах с `eval_loader`. Любой тег с `residual` — всегда; non-residual slash-форма `activations/…layers.{}…` без индекса слоя — нет. `DEFAULT_PREFIXES` оставлен как опциональный явный allow-list. |
| **Классификация** | [`features.py`](features.py) → `monitor_group` / `_structural_group` / `parse_tag` | Любой тег → группа: явное правило, иначе **структурный fallback** из префикса (никогда «other» для модельных мониторов). `parse_tag(tag)` → `TagInfo(group, projection, layer, family)`. |
| **Графики** | [`atlas.py`](atlas.py) → `build_families` | Рисует heatmap на каждое загруженное семейство. `compare`/`temporal`/`eda` берут любой тег с `group != "other"`. Новый монитор появляется сам. |
| **Аудит** | [`tags.py`](tags.py) | `python -m instability.tags --tb-dir RUN` — полный инвентарь + что не классифицировано + что не грузится. |

**Что автоматически:** загрузка (deny-list грузит всё кроме шума), распознавание и
классификация (0 правок), atlas-графики и участие в compare/temporal — для **любого**
нового монитора, без правки префиксов.

**Что НЕ автоматически:** 14 кураторских фигур `report.py` и `PLOT_SPECS` в
`projections.py` — это осознанный выбор «заголовочных» view (новый монитор и так
в atlas). И если новый монитор — это **шум**, которого не должно быть в анализе —
добавь его префикс в `_NOISE_PREFIXES` (аудит покажет, что отбрасывается).

**Сценарий «дали новый TB»:**

```bash
# 1) Аудит — весь набор мониторов + что не доезжает до анализа
python -m instability.tags --tb-dir /path/to/NEW_TB
#    UNRECOGNIZED          → должно быть 0
#    NOT LOADED BY DEFAULT → в норме только шум; если тут МОДЕЛЬНЫЙ монитор — баг фильтра
python -m instability.tags --tb-dir /path/to/NEW_TB --csv tags.csv --full   # дамп

# 2) Отчёт — atlas сам нарисует новые семейства
python -m instability.report --tb-dir /path/to/NEW_TB --out-dir ./outputs/report
ls ./outputs/report/figures/atlas/   # residual_contribution/ effective_lr/ moment_norm/ ...
```

---

## Запуск

Все команды — **из `sber_scripts/analytics/stability/`** (туда смотрит `python -m`).

```bash
cd sber_scripts/analytics/stability
```

### Один ран, eyeball-обзор (быстро, без порогов)

```bash
python -m instability.report \
    --tb-dir /path/to/RUN_NAME \
    --out-dir ./outputs/report_RUN_NAME
```

Выход: `figures/01_overview.png` + `figures/02_max_vio_heatmap.png` + `data/signals.json` + 14 других PNG/таблиц.

### Много ранов

```bash
# 1) Inventory + max_vio summary + correlations — for every run under tb_root
python -m instability.eda --tb-root /path/to/tb --out-dir ./outputs/run_X

# 2) Forward-lag prediction / CCF / Lasso AR-X + event-study — on a sampled subset
python -m instability.temporal --tb-root /path/to/tb --out-dir ./outputs/run_X

# 3) Per-projection analysis + PNG
python -m instability.projections --tb-root /path/to/tb --out-dir ./outputs/run_X
```

### Ad-hoc heatmap по интересующим tag-семействам

```bash
python -m instability.heatmap \
    --tb-dir /path/to/RUN_NAME \
    --out-dir ./outputs/heatmaps_RUN_NAME \
    --tags moe_load_balance moe_maxvio moe_expert_utilization \
    --threshold 4
```

Флаги: `--log-scale`, `--clip-pct LO HI`, `--threshold T` (бинарная
маска `value > T` — выбирай порог под scale метрики; `4` для `max_vio`,
`100` для `moe_load_balance`).

### Shareable HTML-история

После `report.py` — собрать одной командой narrative + filterable
gallery всех figures-PNG:

```bash
python -m instability.story \
    story_config.json \
    ./outputs/report_RUN_NAME/figures \
    --out ./outputs/story_RUN_NAME.html
```

Схема `story_config.json` описана в docstring [`story.py`](story.py).
HTML использует relative-пути к `figures/`, так что HTML и папка
должны лежать рядом для шейринга.

---

## CLI flags (shared by eda / temporal / projections)

| Flag | What |
|---|---|
| `--tb-root PATH` | **required** — directory with TB run subdirs. |
| `--out-dir PATH` | **required** — where CSV/PNG land. |
| `--runs name1 name2 …` | Explicit run names; skips `preferred`/`filters`. |
| `--runs-pattern 'glob*'` | fnmatch over subdirs of `tb_root`. |

Extras:
- `eda.py --feature-corr` — recompute `feature_monitor_correlation_summary.csv`
  from existing CSVs without re-walking raw TB events.
- `projections.py --skip-plots` / `--skip-inventory` — skip heavy steps.

`report.py` has its own flags — see `--help`.

---

## Stage defaults (preferred lists, filters, `n`)

There is **no YAML config** — everything lives in
[`config.py`](config.py) as Python constants:

* `EDA_DEFAULTS` — `min_event_files`, `max_runs`, `max_workers`.
* `TEMPORAL_DEFAULTS` — `n`, `filters={n_vio_steps_min, max_vio_max_min}`,
  `preferred=[...]`.
* `PROJECTIONS_DEFAULTS` — `n`, `filters={..., n_event_files_max}`,
  `preferred=[...]`.

Edit them in place if you need a different shortlist or threshold.

Run-selection algorithm (same as before, common to temporal/projections):
1. `--runs` → take exactly those (no filters applied).
2. `--runs-pattern` → fnmatch.
3. Otherwise: apply `filters` to `run_summary.csv`, then take
   `preferred` (whichever survived the filter), then top up by
   `(n_vio_steps, max_vio_max)` until `n` runs.

---

## `report.py` outputs — layout

```
OUT_DIR/
├── figures/                            # all PNGs
│   ├── 01_overview.png                 # 6 canonical signals + spike markers
│   ├── 02_max_vio_heatmap.png          # [layer × step] + threshold mask
│   ├── 03_layer_attribution.png        # 7-panel attribution heatmap stack
│   ├── 04_moe_health.png               # 8-panel MoE deep-dive
│   ├── 05_activations_depth.png        # 6 stacked activation heatmaps
│   ├── 06_gradients_depth.png          # 4 stacked gradient heatmaps
│   ├── 07_weights_depth.png            # 3 stacked weight heatmaps
│   ├── 08_system_health.png            # memory + throughput + timing
│   ├── 09_per_projection_depth.png     # Q/KV/O/Router/Gate-Up/Down/Shared
│   ├── 10_router_dynamics.png          # scores + utilization + balance + over
│   ├── 11_loss_decomposition.png       # components + ratio + slope + cumul
│   ├── 12_phase_diagnosis.png          # loss + LR + phase shading
│   ├── 13_cross_layer_corr.png         # [L × L] Pearson per signal
│   └── atlas/<category>/<family>.png   # plot-everything: heatmap на каждое семейство
└── data/                               # raw tables + narrative
    ├── 01_overview.csv
    ├── 02_max_vio_per_layer.csv
    ├── 03_layer_profile.csv
    ├── 04_moe_health.csv
    ├── 09_per_projection_depth.csv
    ├── 10_router_dynamics.csv
    ├── 11_loss_decomposition.csv
    ├── 12_phases.md
    ├── 13_cross_layer_corr.csv
    ├── 14_spike_clusters.md
    ├── spikes.csv                       # all MAD-z anomalies
    ├── summary.md                       # auto-narrative overview
    └── signals.json                     # raw aggregated series, all groups
```

### Depth correlations — как читать

`summary.md` содержит таблицу Pearson r между каждой per-layer метрикой
и индексом слоя:

| r | Что значит |
|---|---|
| `> +0.5` | **strong, grows with depth** — метрика растёт глубже |
| `+0.25..+0.5` | moderate, grows |
| `-0.25..+0.25` | weak / noise — depth-нейтрально |
| `-0.25..-0.5` | moderate, falls |
| `< -0.5` | **strong, falls with depth** — метрика падает глубже |

Типичные здоровые паттерны:
* `residual_grad_amax` **strong falls** (r ≈ -0.8) — нормально для backprop attenuation.
* `expert_utilization` **moderate** в обе стороны — глубоким слоям труднее использовать ВСЕ эксперты, это норм.
* `norm_weight_l2` **falls** — layernorm γ адаптируется вниз в поздних слоях (типичная RMSNorm-динамика).

Подозрительные:
* `max_vio` **strong grows** — настоящий depth-driven router collapse.
* `clip_share_*` **strong grows** + abs > 0.01 в каких-то слоях — clipping реально работает в этих слоях.
* `residual_amax` **strong grows БЕЗ хампа** (т.е. monotonic) — Sandwich-GatedNorm не справляется.

---

## Глоссарий метрик

| Сигнал | TB-тег | Что означает |
|---|---|---|
| **max_vio** | `dispatcher_layer_experts_max_vio_*` или `dispatcher_sum_metrics/experts_max_vio_*` | Router load imbalance. Нормирован на `num_experts`. Здоровое значение ~1.5–3. **> 4** = warning, **> 6** = серьёзно, **≥ 0.8 × num_experts** = collapsed. |
| **grad_norm/global** | `l2_norm/grad/global` | L2 норма всех градиентов. Резкий рост → backward взрывается. |
| **activations_amax** | `activations_amax/...` | Максимум активаций по модулям. Резкий рост → forward раздувает residual stream. |
| **moe_seq_aux_loss** | `moe_seq_aux_loss/layer_N` | Router auxiliary loss (заставляет балансировать). Стагнация/рост на высоком уровне → routing не справляется. |
| **hidden_z_loss** | `loss/train/hidden_z_loss*` | Z-loss на residual stream. Помогает с numerical stability. |
| **moe_z_loss_router** | `moe_z_loss_router/layer_N` | Router z-loss. Большое значение = пре-softmax логиты расходятся. |
| **update_param_ratio** | `derived/update_param_ratio` | `‖Δθ‖ / ‖θ‖`. Adam-overstepping indicator. |
| **growth_onset** | derived в `features.py` | Первая точка, где `rolling_slope(log(max_vio))` устойчиво положительный. Точка отсчёта для event-study. |

---

## Выходные файлы

| Скрипт | Файлы |
|---|---|
| `report.py` | `figures/01..14_*.png`, `figures/atlas/<category>/<family>.png`, `data/01..14_*.csv/md`, `data/spikes.csv`, `data/summary.md`, `data/signals.json` |
| `tags.py` | stdout-инвентарь + (опц.) `--csv` per-tag классификация |
| `eda.py` | `run_inventory.csv`, `run_summary.csv`, `metric_correlations.csv`, `feature_summary.csv`, `feature_monitor_correlation_summary.csv` |
| `temporal.py` | `granger_results.csv` (forward-lag predictive), `ccf_results.csv`, `lasso_arx_results.csv`, `event_pre_onset_results.csv`, `granger_group_summary.csv`, `feature_conditioned_temporal_summary.csv` |
| `projections.py` | `projection_event_summary.csv`, `absolute_threshold_summary.csv`, `event_plot_profiles.csv`, `projection_tag_inventory.csv`, `event_study_plots/*.png` |
