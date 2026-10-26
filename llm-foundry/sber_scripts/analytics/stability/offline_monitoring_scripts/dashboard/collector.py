"""TensorBoard event loading and per-layer metric collection."""

from __future__ import annotations

import bisect
import gzip
import hashlib
import math
import os
import pickle
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from .patterns import (
    ACTIVATION_MTP_TAIL,
    ACTIVATION_PATTERN_TAILS,
    MAX_VIO_VARIANTS,
    MTP_PNAME,
    discover_pattern_tails,
    merge_static_with_discovered,
    split_activation_categories,
)

STATS = ("amax", "amean", "l2norm", "kurtosis", "actrel", "actabs")

Series = tuple[list[int], list[float]]
LayerSeries = dict[int, Series]
Variables = dict[str, LayerSeries]
RunDict = dict[str, Any]
RawData = dict[str, Series]  # tag -> (steps, values)

# Only load tags with these prefixes; everything else is skipped on the fly.
_RELEVANT_PREFIXES: tuple[str, ...] = (
    "activations_amax/",
    "activations_amean/",
    "activations_l2norm/",
    "activations_gradients_amax/",
    "activations_gradients_amean/",
    "activations_gradients_l2norm/",
    # massive-activation monitors (fraction of |x| satisfying a relative/absolute compare);
    # op + threshold are appended to the stat name, e.g. activations_actrel>100/...
    "activations_actrel",
    "activations_actabs",
    "l2_norm/",
    # native optimizer ratio monitors (replace dashboard-side derivatives where present)
    "effective_lr/",
    "update_weight_ratio/",
    "update_rate/",
    # amax of optimizer-state moments
    "amax/moment",
    "weights_l2norm/",
    "dispatcher_layer_experts_max_vio_batch_avg/",
    "dispatcher_layer_experts_max_vio_batch_no_pad_avg/",
    "dispatcher_layer_experts_max_vio_batch_no_pad_no_prefix_avg/",
    "dispatcher_layer_experts_max_vio_microbatch_avg/",
    "expert_activation/",
    "moe_",
)

_CACHE_DIR = Path.home() / ".cache" / "tb_dashboard"

# Workers: use all available cores capped at a reasonable limit.
_CPU_COUNT = os.cpu_count() or 4
_FILE_WORKERS = min(_CPU_COUNT, 16)  # processes per run
_RUN_WORKERS = 4  # concurrent runs (outer thread pool)


# ── Timing ─────────────────────────────────────────────────────────────────────


@contextmanager
def _timer(label: str):
    t0 = perf_counter()
    yield
    print(f"  ⏱  {label}: {perf_counter() - t0:.2f}s", flush=True)


# ── Low-level event reading (module-level so ProcessPoolExecutor can pickle) ───


def _extract_scalar(value: Any) -> float | None:
    if value.HasField("simple_value"):
        return float(value.simple_value)
    if value.HasField("tensor"):
        t = value.tensor
        if t.float_val:
            return float(t.float_val[0])
        if t.double_val:
            return float(t.double_val[0])
    return None


def _read_file_batch(files: list[Path]) -> RawData:
    """Read a batch of event files sequentially in one subprocess.

    Running batches (not individual files) in subprocesses reduces IPC overhead:
    each subprocess imports TensorBoard once and returns one merged dict.
    """
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader  # noqa: PLC0415

    out: RawData = {}
    for ef in files:
        try:
            for event in EventFileLoader(str(ef)).Load():
                if not event.HasField("summary"):
                    continue
                step = event.step
                for value in event.summary.value:
                    tag = value.tag
                    if not tag.startswith(_RELEVANT_PREFIXES):
                        continue
                    fval = _extract_scalar(value)
                    if fval is None:
                        continue
                    if tag not in out:
                        out[tag] = ([], [])
                    out[tag][0].append(step)
                    out[tag][1].append(fval)
        except Exception as exc:
            warnings.warn(f"error reading {ef.name}: {exc}")
    return out


def _merge_partials(partials: list[RawData]) -> RawData:
    """Merge ordered per-batch dicts; steps stay sorted because batches are ordered."""
    out: RawData = {}
    for partial in partials:
        for tag, (steps, values) in partial.items():
            if tag not in out:
                out[tag] = (list(steps), list(values))
            else:
                out[tag][0].extend(steps)
                out[tag][1].extend(values)
    return out


# ── Cache ──────────────────────────────────────────────────────────────────────


def _cache_key(path: Path) -> str:
    """Stable hash of directory path + mtime+size of every event file."""
    event_files = sorted(path.glob("events.out.tfevents.*"))
    if not event_files:
        event_files = sorted(path.rglob("events.out.tfevents.*"))
    parts = [str(path.resolve())]
    for ef in event_files:
        st = ef.stat()
        parts.append(f"{ef.name}:{st.st_mtime_ns}:{st.st_size}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


def _cache_load(path: Path) -> RawData | None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cf = _CACHE_DIR / f"{_cache_key(path)}.pkl.gz"
        if cf.exists():
            with gzip.open(cf, "rb") as f:
                return pickle.load(f)
    except Exception as exc:
        warnings.warn(f"cache read failed for {path.name}: {exc}")
    return None


def _cache_save(path: Path, raw: RawData) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cf = _CACHE_DIR / f"{_cache_key(path)}.pkl.gz"
        with gzip.open(cf, "wb", compresslevel=1) as f:
            pickle.dump(raw, f, protocol=5)
        size_mb = cf.stat().st_size / 1e6
        print(f"      → cached ({size_mb:.1f} MB) → {cf.name}", flush=True)
    except Exception as exc:
        warnings.warn(f"cache write failed for {path.name}: {exc}")


# ── Run loading ────────────────────────────────────────────────────────────────


def _load_scalars(path: Path, file_workers: int = _FILE_WORKERS) -> RawData:
    """Load one run: check cache first, then parallel-process event files."""
    cached = _cache_load(path)
    if cached is not None:
        print(f"  ✓ cache hit: {path.name}  ({len(cached)} tags)", flush=True)
        return cached

    event_files = sorted(path.glob("events.out.tfevents.*"))
    if not event_files:
        event_files = sorted(path.rglob("events.out.tfevents.*"))
    if not event_files:
        warnings.warn(f"no TensorBoard event files found in: {path}")
        return {}

    n = len(event_files)
    n_workers = min(n, file_workers)
    batch_size = max(1, math.ceil(n / n_workers))
    batches = [event_files[i : i + batch_size] for i in range(0, n, batch_size)]

    with _timer(f"{path.name}  ({n} files → {len(batches)} workers)"):
        try:
            with ProcessPoolExecutor(max_workers=len(batches)) as pool:
                results = list(pool.map(_read_file_batch, batches))
            out = _merge_partials(results)
        except Exception as exc:
            warnings.warn(
                f"ProcessPoolExecutor failed ({exc}), falling back to threads"
            )
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(lambda ef: _read_file_batch([ef]), event_files))
            out = _merge_partials(results)

    if out:
        print(f"      → {len(out)} relevant tags loaded", flush=True)
        _cache_save(path, out)
    else:
        warnings.warn(f"no relevant scalar tags found in: {path}")
    return out


# ── Helpers ────────────────────────────────────────────────────────────────────


def snap_value(steps: list[int], values: list[float], step: int) -> float | None:
    """Binary-search nearest-neighbor snap with no hard step-distance cap."""
    if not steps:
        return None
    j = bisect.bisect_left(steps, step)
    if j == len(steps):
        j -= 1
    elif j > 0 and (steps[j] - step) > (step - steps[j - 1]):
        j -= 1
    return values[j]


def _safe_name(text: str) -> str:
    text = text.strip(".").replace(".0", "")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def add_series(
    dst: Variables, name: str, layer: int, steps: list[int], values: list[float]
) -> None:
    if steps:
        dst.setdefault(name, {})[layer] = (steps, values)


# ── Collectors ─────────────────────────────────────────────────────────────────


def _collect_tensor_metric(
    raw: RawData,
    variables: Variables,
    metric_prefix: str,
    suffix: str,
    tails: dict[str, str],
    mtp_tail: str | None = None,
) -> None:
    if not tails and not mtp_tail:
        return
    patterns = [
        (base_name, stat, re.compile(rf"^{re.escape(metric_prefix)}_{stat}/{tail}$"))
        for base_name, tail in tails.items()
        for stat in STATS
    ]
    mtp_patterns = (
        [
            (
                MTP_PNAME,
                stat,
                re.compile(rf"^{re.escape(metric_prefix)}_{stat}/{mtp_tail}$"),
            )
            for stat in STATS
        ]
        if mtp_tail
        else []
    )

    mtp_raw: dict[str, Series] = {}
    for tag, series in raw.items():
        for base_name, stat, rx in mtp_patterns:
            if rx.match(tag):
                mtp_raw.setdefault(f"{base_name}__{suffix}_{stat}", series)
                break
        else:
            for base_name, stat, rx in patterns:
                m = rx.match(tag)
                if m:
                    add_series(
                        variables,
                        f"{base_name}__{suffix}_{stat}",
                        int(m.group("L")),
                        *series,
                    )
                    break

    real_layers = [
        layer
        for name, per_layer in variables.items()
        if f"__{suffix}_" in name
        for layer in per_layer
    ]
    if real_layers and mtp_raw:
        mtp_x = max(real_layers) + 1
        for name, series in mtp_raw.items():
            variables.setdefault(name, {})[mtp_x] = series


# Longest alternatives first so e.g. ``pre_clip_grad`` and ``moment_sq`` win over
# their prefixes (``grad`` / ``moment``). Per-projection fqn tags emitted by the new
# GigaFSDP-aware monitor share the same ``model...layers.N...`` shape and match here too.
_OPT_NORM_RX = re.compile(
    r"^l2_norm/(?P<kind>pre_clip_grad|grad|update|param|"
    r"moment_sq_bc|moment_sq|moment_bc|moment)/"
    r"(?:model\._module\.)?model\.(?:model\.)?layers\.(?P<L>\d+)\.(?P<rest>.+)$"
)
_OPT_NORM_STRIP = re.compile(r"\.(weight|flatten|bias)$")
_OPT_NORM_KIND_TO_METRIC = {
    "grad": "grad_norm",
    "update": "update_norm",
    "param": "weight_norm",
    "pre_clip_grad": "pre_clip_grad_norm",
    "moment": "moment_norm",
    "moment_sq": "moment_sq_norm",
    "moment_bc": "moment_bc_norm",
    "moment_sq_bc": "moment_sq_bc_norm",
}


def _collect_optimizer_norms(raw: RawData, variables: Variables) -> None:
    for tag, series in raw.items():
        m = _OPT_NORM_RX.match(tag)
        if m:
            module = _OPT_NORM_STRIP.sub("", m.group("rest"))
            add_series(
                variables,
                f"{_OPT_NORM_KIND_TO_METRIC[m.group('kind')]}__{_safe_name(module)}",
                int(m.group("L")),
                *series,
            )


# Native optimizer ratios (effective_lr / update_weight_ratio / update_rate). When the
# run logged these, the dashboard uses them directly instead of recomputing derivatives
# (see derivatives._add_module_ratios). They are written under the *same* variable names
# the derivative would have produced, so grouping/coloring downstream is unchanged.
_NATIVE_RATIO_RX = re.compile(
    r"^(?P<kind>effective_lr|update_weight_ratio|update_rate)/"
    r"(?:model\._module\.)?model\.(?:model\.)?layers\.(?P<L>\d+)\.(?P<rest>.+)$"
)
_NATIVE_RATIO_KIND_TO_METRIC = {
    "effective_lr": "eff_lr",
    "update_weight_ratio": "upd_over_weight",
    "update_rate": "grad_over_weight",
}


def _collect_native_ratios(raw: RawData, variables: Variables) -> None:
    for tag, series in raw.items():
        m = _NATIVE_RATIO_RX.match(tag)
        if m:
            module = _OPT_NORM_STRIP.sub("", m.group("rest"))
            add_series(
                variables,
                f"{_NATIVE_RATIO_KIND_TO_METRIC[m.group('kind')]}__{_safe_name(module)}",
                int(m.group("L")),
                *series,
            )


_AMAX_MOMENT_RX = re.compile(
    r"^amax/(?P<kind>moment_sq|moment)/"
    r"(?:model\._module\.)?model\.(?:model\.)?layers\.(?P<L>\d+)\.(?P<rest>.+)$"
)


def _collect_moment_amax(raw: RawData, variables: Variables) -> None:
    for tag, series in raw.items():
        m = _AMAX_MOMENT_RX.match(tag)
        if m:
            module = _OPT_NORM_STRIP.sub("", m.group("rest"))
            add_series(
                variables,
                f"{m.group('kind')}_amax__{_safe_name(module)}",
                int(m.group("L")),
                *series,
            )


# Massive/outlier-activation monitors: fraction of |x| satisfying a relative (actrel)
# or absolute (actabs) comparison, e.g. activations_actrel>100/... or activations_actabs<1000/...
_MASSIVE_ACT_RX = re.compile(
    r"^activations_(?P<stat>act(?:rel|abs)[<>][0-9.]+)/"
    r"model\.model\.layers\.(?P<L>\d+)\.(?P<rest>.+?)\.0$"
)
# `>`/`<` are unsafe in a variable name (raw `<` would be parsed as an HTML tag in the
# dashboard); encode them so the stat survives intact in the name (actrel>100 → actrel_gt_100).
_MASSIVE_OP_ENCODE = {">": "_gt_", "<": "_lt_"}


def _collect_massive_activations(raw: RawData, variables: Variables) -> None:
    """Per-module fraction of massive (outlier) activations → {actrel|actabs}_{op}_{thr}__{module}."""
    for tag, series in raw.items():
        m = _MASSIVE_ACT_RX.match(tag)
        if m:
            stat = m.group("stat")
            for op, enc in _MASSIVE_OP_ENCODE.items():
                stat = stat.replace(op, enc)
            add_series(
                variables,
                f"{stat}__{_safe_name(m.group('rest'))}",
                int(m.group("L")),
                *series,
            )


_WEIGHT_L2NORM_RX = re.compile(
    r"^weights_l2norm/model\.model\.layers\.(?P<L>\d+)\.(?P<rest>.+)$"
)
_WEIGHT_NAME_STRIP = re.compile(r"\.weight$")


def _collect_weight_norms(raw: RawData, variables: Variables) -> None:
    """Collect per-tensor weight L2 norms from weights_l2norm/ → weight_norm__{name}.

    Each weight tensor gets its own per-layer series. Derivatives then aggregate
    these into layer_amax__weight_norm via _layer_amax_of_group.
    """
    for tag, series in raw.items():
        m = _WEIGHT_L2NORM_RX.match(tag)
        if m:
            name = _WEIGHT_NAME_STRIP.sub("", m.group("rest"))
            add_series(
                variables,
                f"weight_norm__{_safe_name(name)}",
                int(m.group("L")),
                *series,
            )


def _collect_max_vio(raw: RawData, variables: Variables) -> None:
    for tag, series in raw.items():
        for rx, var_name in MAX_VIO_VARIANTS:
            m = rx.match(tag)
            if m:
                add_series(variables, var_name, int(m.group("L")), *series)
                break


def _collect_moe(raw: RawData, variables: Variables) -> None:
    """Collect moe_{metric}/layer_{L} tags into moe__{metric} variables."""
    rx = re.compile(r"^moe_(?P<metric>[^/]+)/layer_(?P<L>\d+)$")
    for tag, series in raw.items():
        m = rx.match(tag)
        if m:
            add_series(
                variables, f"moe__{m.group('metric')}", int(m.group("L")), *series
            )


def _collect_expert_activation(raw: RawData, variables: Variables) -> None:
    """Collect pre-aggregated expert_activation metrics (max_median_ratio, mean_norm_per_token)."""
    rx = re.compile(r"^expert_activation/(?P<metric>[^/]+)/layer_(?P<L>\d+)$")
    for tag, series in raw.items():
        m = rx.match(tag)
        if m:
            add_series(
                variables,
                f"expert_activation__{m.group('metric')}",
                int(m.group("L")),
                *series,
            )


def _collect_expert_act_stats(raw: RawData, variables: Variables) -> None:
    """Per-layer expert activation imbalance from block_sparse_moe expert indices.

    For each (component, layer), groups all per-expert amax series and computes:
      - amax  : max of expert amaxes  (scale of the hottest expert)
      - amean : mean of expert amaxes  (average scale across experts)
      - amax_to_amean : imbalance ratio — 1.0 means balanced, higher = more skewed
    """
    rx = re.compile(
        r"^activations_amax/model\.model\.layers\.(?P<L>\d+)"
        r"\.block_sparse_moe\.(?P<comp>[^.]+)\.(?P<E>\d+)$"
    )
    per_comp_layer: dict[tuple[str, int], list[Series]] = {}
    for tag, series in raw.items():
        m = rx.match(tag)
        if m:
            key = (m.group("comp"), int(m.group("L")))
            per_comp_layer.setdefault(key, []).append(series)

    for (comp, layer), expert_series in per_comp_layer.items():
        if len(expert_series) < 2:
            continue
        ref_steps = expert_series[0][0]
        amax_vals: list[float] = []
        amean_vals: list[float] = []
        ratio_vals: list[float] = []
        for step in ref_steps:
            vals = []
            for steps, values in expert_series:
                v = snap_value(steps, values, step)
                if v is not None and math.isfinite(v):
                    vals.append(v)
            if vals:
                e_amax = max(vals)
                e_amean = sum(vals) / len(vals)
                amax_vals.append(e_amax)
                amean_vals.append(e_amean)
                ratio_vals.append(e_amax / e_amean if e_amean > 0 else math.nan)
            else:
                amax_vals.append(math.nan)
                amean_vals.append(math.nan)
                ratio_vals.append(math.nan)
        var = f"expert_act__{comp}"
        add_series(variables, f"{var}__amax", layer, ref_steps, amax_vals)
        add_series(variables, f"{var}__amean", layer, ref_steps, amean_vals)
        add_series(variables, f"{var}__amax_to_amean", layer, ref_steps, ratio_vals)


# ── Pattern discovery ──────────────────────────────────────────────────────────


def _discover_merged_tails(
    all_tags: list[list[str]],
) -> tuple[dict[str, str], dict[str, str]]:
    act_merged = dict(ACTIVATION_PATTERN_TAILS)
    grad_merged: dict[str, str] = {}
    for tags in all_tags:
        if not tags:
            continue
        disc_act = discover_pattern_tails(tags, "activations", min_coverage=2)[0]
        act_merged = merge_static_with_discovered(act_merged, disc_act)
        disc_grad = discover_pattern_tails(
            tags, "activations_gradients", min_coverage=3
        )[0]
        for name, tail in disc_grad.items():
            grad_merged.setdefault(name, tail)

    activation_tails = split_activation_categories(act_merged).get("main", {})
    if not activation_tails:
        warnings.warn(
            "no main activation patterns found — activation metrics will be absent"
        )
    return activation_tails, grad_merged


# ── Run collection ─────────────────────────────────────────────────────────────


def collect_run(
    raw: RawData,
    label: str,
    activation_tails: dict[str, str],
    gradient_tails: dict[str, str],
) -> RunDict | None:
    from .derivatives import compute_derivatives  # noqa: PLC0415

    variables: Variables = {}
    with _timer(f"collect tensors  [{label}]"):
        _collect_tensor_metric(
            raw,
            variables,
            "activations",
            "act",
            activation_tails,
            mtp_tail=ACTIVATION_MTP_TAIL,
        )
        _collect_tensor_metric(
            raw, variables, "activations_gradients", "grad", gradient_tails
        )
        _collect_optimizer_norms(raw, variables)
        _collect_native_ratios(raw, variables)
        _collect_moment_amax(raw, variables)
        _collect_massive_activations(raw, variables)
        _collect_weight_norms(raw, variables)
        _collect_max_vio(raw, variables)
        _collect_moe(raw, variables)
        _collect_expert_activation(raw, variables)
        _collect_expert_act_stats(raw, variables)

    with _timer(f"derivatives      [{label}]"):
        compute_derivatives(variables)

    layers = sorted({layer for series in variables.values() for layer in series})
    if not layers:
        warnings.warn(
            f"no per-layer metrics found for run '{label}' "
            f"({len(raw)} tags loaded) — will be shown as empty in dashboard"
        )
        return {"label": label, "empty": True, "n_tags": len(raw)}

    mtp_layers = {
        layer
        for name, series in variables.items()
        if name.startswith(MTP_PNAME)
        for layer in series
    }
    max_real_layer = min(mtp_layers) - 1 if mtp_layers else max(layers)
    linear_layers = {
        layer
        for name, per_layer in variables.items()
        if "qkv_conv1d" in name
        for layer in per_layer
    }
    nonlinear = {
        layer
        for layer in layers
        if layer <= max_real_layer and layer not in linear_layers
    }

    return {
        "label": label,
        "variables": variables,
        "layers": layers,
        "max_real_layer": max_real_layer,
        "nonlinear": nonlinear,
    }


def load_runs(
    paths: list[Path],
    labels: list[str] | None = None,
    max_workers: int | None = None,
) -> list[RunDict]:
    """Load all runs (cache + parallel), then collect metrics."""
    if labels is None:
        labels = [p.name for p in paths]

    file_workers = min(_CPU_COUNT, max_workers) if max_workers else _FILE_WORKERS
    run_workers = min(_RUN_WORKERS, max_workers) if max_workers else _RUN_WORKERS

    print(
        f"\n── Loading {len(paths)} run(s)  [CPUs: {_CPU_COUNT}, file workers: {file_workers}] ──",
        flush=True,
    )
    t_total = perf_counter()

    print("Step 1/3  reading event files", flush=True)
    with ThreadPoolExecutor(max_workers=min(len(paths), run_workers)) as pool:
        futures = [pool.submit(_load_scalars, p, file_workers) for p in paths]
    raws: list[RawData] = []
    for fut, label in zip(futures, labels):
        try:
            raws.append(fut.result())
        except Exception as exc:
            warnings.warn(f"failed to load run '{label}': {exc}")
            raws.append({})

    print("\nStep 2/3  pattern discovery", flush=True)
    with _timer("discover tails"):
        all_tags = [list(raw.keys()) for raw in raws]
        activation_tails, gradient_tails = _discover_merged_tails(all_tags)
    print(
        f"      → {len(activation_tails)} activation tails, "
        f"{len(gradient_tails)} gradient tails",
        flush=True,
    )

    print("\nStep 3/3  collecting metrics + derivatives", flush=True)
    runs: list[RunDict] = []
    for raw, label in zip(raws, labels):
        run = collect_run(raw, label, activation_tails, gradient_tails)
        if run is not None:
            runs.append(run)

    elapsed = perf_counter() - t_total
    print(
        f"\n── Load complete in {elapsed:.2f}s  ({len(runs)}/{len(paths)} runs OK) ──\n",
        flush=True,
    )
    if not runs:
        warnings.warn("no runs could be processed — dashboard will be empty")
    return runs
