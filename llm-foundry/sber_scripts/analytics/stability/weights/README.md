# weights — анализ HF-чекпойнтов MoE-моделей

Скрипты диагностики чекпойнтов из HF-safetensors экспорта. **Только
анализ** — никакой ремедиации, записи в чекпойнт или модификаций.

## Содержание

1. [Установка](#1-установка)
2. [Состав модулей](#2-состав-модулей)
3. [Что ожидать на вход](#3-что-ожидать-на-вход)
4. [Команды](#4-команды)
5. [Типичные пайплайны](#5-типичные-пайплайны)
6. [Метрики и роли тензоров](#6-метрики-и-роли-тензоров)
7. [Производительность](#7-производительность)
8. [Дебаг](#8-дебаг)

---

## 1. Установка

```bash
pip install -r sber_scripts/analytics/stability/weights/requirements.txt
```

Зависимости: `numpy`, `torch`, `safetensors`, `tqdm`. 

Все команды запускаются **из `sber_scripts/analytics/stability/`**:

```bash
cd sber_scripts/analytics/stability
python3 -m weights <subcommand> [args]
```

`weights report` единственный полностью stdlib-only (не требует torch),
остальные команды — torch + safetensors.

---

## 2. Состав модулей

| Файл | Что внутри |
|---|---|
| `hf_reader.py` | `SafetensorsReader` — mmap-нутое чтение HF safetensors экспорта (`.names`, `.info`, `.load`, `.load_many`, `.iter_tensors`). |
| `analyze.py` | `classify()` + per-tensor stats: `base` (моменты, гистограммы), `rowcol` (rank-1-collapse), `spectral` (top-K SVD).|
| `joint.py` | Joint Gram (`Σ_e W_eᵀ W_e`) по `(layer, proj)` MoE-экспертам + gate↔up alignment score. Skip для слишком больших Gram-матриц. |
| `batch.py` | Прогон выбранных команд по всем `ep0_baN/artifacts/hf_checkpoint/` под run-папкой. Идемпотентный. |
| `report.py` | Компактный текстовый отчёт по analyze/joint JSON-ам. HEADLINES, per-group aggregates, top-N outliers, multi-file deltas, trajectory. |
| `_timing.py` | `StageTimer` — лёгкий профайлер. |
| `__main__.py` | Единый CLI. |

---

## 3. Что ожидать на вход

Каждая команда (кроме `report`) принимает либо **HF-папку** (для
одиночной итерации), либо **run-папку** (для `batch`).

### HF-папка (одна итерация)

Полный HF-export, обычно лежит в `<iter>/artifacts/hf_checkpoint/`:

```
ep0_ba25000/artifacts/hf_checkpoint/
├── config.json
├── model.safetensors.index.json
├── shard-r00-c00000.safetensors
├── shard-r00-c00001.safetensors
├── ...
├── tokenizer.json
└── tokenizer_config.json
```

Маленькие модели иногда в одном файле — single `*.safetensors` без
index.json тоже работает.

### Run-папка (для `batch`)

```
<RUN_NAME>/
├── ep0_ba2000/artifacts/hf_checkpoint/
├── ep0_ba4000/artifacts/hf_checkpoint/
├── ...
└── ep0_ba25000/artifacts/hf_checkpoint/
```

`batch` сам найдёт `ep0_baN/` совпадающие по паттерну (по умолчанию
`ep0_ba*`) у которых внутри есть `artifacts/hf_checkpoint/`.

---

## 4. Команды

### `analyze HF_DIR --out PATH`

Per-tensor `base` + `rowcol` (для 2-D) + `spectral` (top-K SVD, для
матриц подходящих ролей). Пишет `analyze.json`.

```bash
python -m weights analyze /path/to/ep0_ba25000/artifacts/hf_checkpoint \
    --out output/exp/weights_analyze_ba25000/analyze.json \
    --device cuda
```

Полезные флаги:
- `--include REGEX` / `--exclude REGEX` — отфильтровать тензоры по FQN.
- `--limit N` — debug, остановиться после N тензоров.
- `--topk-sigma K` — top-K для рандомизированной SVD (дефолт 16; 0 чтобы выключить spectral).
- `--svd-niter N` — итерации randomized SVD (дефолт 2).
- `--debug-timing` — печатать `[time]` лайны на каждый тензор.

### `joint HF_DIR --out PATH`

Joint Gram + alignment по MoE-экспертам. Для каждой `(layer, proj)`
группы (где proj ∈ {gate, up, down}) собирает все per-expert тензоры,
считает `G = Σ_e W_eᵀ W_e` streaming, eigh, alignment между gate и up.

```bash
python -m weights joint /path/to/ep0_ba25000/artifacts/hf_checkpoint \
    --out output/exp/weights_joint_ba25000/joint.json \
    --device cuda -k 4
```

Полезные флаги:
- `-k K` — top-K joint singular directions (дефолт 4).
- `--gpu-gram-cap-gb 30.0` — если Gram больше, группа скипается. С `--cpu-fallback` пробуется на CPU.
- `--max-gram-gb 200.0` — hard cap.

### `batch RUN_DIR --out OUT_DIR --commands analyze joint`

Прогоняет указанные команды по всем итерациям run-а. Идемпотентный —
по умолчанию skip-ает итерации с уже существующим JSON-ом
(`--no-skip-existing` чтобы переделать).

```bash
python -m weights batch /path/to/RUN_DIR \
    --out output/exp/batch \
    --commands analyze joint \
    --device cuda
```

Output: `<out>/<iter_name>/{analyze,joint}.json`.

### `report FILE [FILE ...]`

Stdlib-only текстовый отчёт. Auto-detects analyze/joint JSON.

```bash
# одиночный файл — HEADLINES + per-group aggregates + worst-N outliers
python -m weights report output/exp/weights_analyze_ba25000/analyze.json

# несколько analyze.json в порядке итераций — добавит DELTAS
python -m weights report output/exp/batch/ep0_ba*/analyze.json --top-n 5

# layer × iteration таблица для одной (role, proj) группы
python -m weights report output/exp/batch/ep0_ba*/analyze.json \
    --trajectory moe_expert/gate
```

Флаги: `--top-n`, `--all-metrics`, `--trajectory ROLE[/PROJ[/BLOCK]]`,
`--out PATH`.

---

## 5. Типичные пайплайны

### Один чекпойнт (CPU/GPU)

```bash
# 1. analyze + joint (нужен GPU).
python -m weights analyze /path/to/ep0_ba25000/artifacts/hf_checkpoint \
    --out output/exp/analyze_ba25000.json --device cuda
python -m weights joint /path/to/ep0_ba25000/artifacts/hf_checkpoint \
    --out output/exp/joint_ba25000.json --device cuda -k 4

# 2. Читаемый отчёт (CPU).
python -m weights report \
    output/exp/analyze_ba25000.json output/exp/joint_ba25000.json
```

### Эволюция через несколько чекпойнтов одного рана

```bash
# 1. Прогнать batch (часы).
python -m weights batch /path/to/RUN_DIR \
    --out output/exp/batch \
    --commands analyze joint \
    --device cuda

# 2. Cross-iteration delta + trajectory.
python -m weights report output/exp/batch/ep0_ba*/analyze.json --top-n 5
python -m weights report output/exp/batch/ep0_ba*/analyze.json \
    --trajectory moe_expert/down
```

### Сравнить два рана на одной итерации

```bash
python -m weights analyze /path/to/RUN_A/ep0_ba10000/artifacts/hf_checkpoint \
    --out output/exp/A.json --device cuda
python -m weights analyze /path/to/RUN_B/ep0_ba10000/artifacts/hf_checkpoint \
    --out output/exp/B.json --device cuda
python -m weights report output/exp/A.json output/exp/B.json
```

---

## 6. Метрики и роли тензоров

### Роли (поле `role` в JSON-записях)

| role | Что |
|---|---|
| `embedding`, `lm_head`, `final_norm` | топ-левел |
| `mla` | DeepSeek MLA-проекции: `q_a`/`q_b`/`kv_a`/`kv_b` |
| `linear_attn` | Mamba/DeltaNet-стиль: `qkv`/`qkv_conv1d`/`conv1d`/`gate`/`z`/`in_qkvz`/`in_ba`/`ba`/`o`/`out` |
| `ssm_param` | 1-D SSM/clip-style: `A_log`/`dt_bias`/`norm`/`o_norm` |
| `moe_router` | `mlp.gate.weight` + `mlp.gate.e_score_correction_bias` (proj: `weight`/`bias`) |
| `moe_expert` | `mlp.experts.<i>.<gate|up|down>_proj.weight` — у каждого эксперта своя запись с `expert=<i>` |
| `moe_shared_expert` | `mlp.shared_experts.<gate|up|down>_proj.weight` |
| `dense_mlp` | `mlp.<gate|up|down>_proj.weight` (когда слой не MoE) |
| `layernorm` | GatedRMSNorm и обычные layernorm-веса; proj несёт имя норма + суффикс (`gate_up`/`gate_down`/`weight`) |
| `mtp` | per-layer MTP-extras: `eh_proj`/`enorm`/`hnorm`/`embed_tokens`/`shared_head` |
| `unknown` | catch-all; если такие появляются в свежем отчёте — нужен новый regex в `classify` |

Поле `block` — `"main"` или `"mtp"`. Сравнивая per-layer метрики, держи
эти популяции отдельно.

### Метрики

См. в [`analyze.py`](analyze.py): `base` (n_nan/n_inf/моменты/гистограмма/
top-K хвост), `rowcol.row/col` (max_over_median — rank-1-collapse), `spectral`
(top-K randomized SVD: σ samples, spectral_norm, stable_rank,
effective_rank_entropy_topk). Curated set с направлениями ранжирования
описан в `report.py:ANALYZE_CURATED`.

---

## 7. Производительность

`analyze` на HF-export-е с ~3700 тензоров (9B MoE) на одной H100:
- CPU mmap I/O — миллисекунды на тензор.
- SVD доминирует (если включён): ~50-200 ms на матрицу 1280×1920 при
  topk_sigma=16, niter=2. ~5-10 минут на весь чекпойнт.
- Per-tensor RAM не растёт — каждый тензор обрабатывается и выкидывается.

`joint` на 64 экспертах × 39 слоёв × 3 проекций = 117 групп. Gram
1920×1920 = 14 MB, eigh быстрый. ~2-5 минут на чекпойнт.

`report` — секунды, читает JSON.

---
