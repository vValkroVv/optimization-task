"""Standalone TB-scalar helpers used by eda / temporal / projections / report.

Replaces the cluster-only ``tb_moe_balance.py`` and provides a single
fast event-file reader shared across the instability toolkit.

Performance design (modeled on ``offline_monitoring_scripts/dashboard/collector.py``):

1. **File batching** — multiple event files per subprocess, not one
   per subprocess. Each worker imports TensorBoard once and returns
   one merged dict → IPC overhead × ``n_files`` collapses to × ``n_workers``.
2. **Prefix filtering inside the worker** — ``str.startswith(tuple)``
   is a built-in C call. Rejects ~95% of tags before any allocation.
3. **Disk cache** keyed by ``(path, mtime_ns, size)`` of every event
   file. Subsequent loads of an unchanged run are milliseconds.
4. **List-of-(steps, values)** instead of per-step dicts inside workers;
   merge in the master process is ``list.extend`` → O(1) per file.
5. **Workers default to ``min(cpu_count, 16)``**.

API kept compatible with the cluster helper so the existing
``eda.py`` / ``temporal.py`` / ``projections.py`` code does not need to
change beyond the import.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import os
import pickle
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_CPU_COUNT = os.cpu_count() or 4
DEFAULT_WORKERS = min(_CPU_COUNT, 16)

_CACHE_DIR = Path.home() / ".cache" / "instability_tb"
_CACHE_VERSION = "v2"  # bump if the cached blob layout changes

# Legacy curated allow-list. **No longer the default** — the default load now uses
# the ``default_load_keep`` deny-list below (keep every model monitor, drop only
# non-model noise). Kept for callers that want the old curated behaviour via an
# explicit ``load_run_scalars(..., prefixes=DEFAULT_PREFIXES)``.
DEFAULT_PREFIXES: tuple[str, ...] = (
    # losses + LR + system
    "loss/train/",
    "lr-",
    "throughput/",
    "memory/",
    "time/",
    "time_seconds/",
    "seq_length_warmup/",
    # optimizer norms / ratios / moments. Broadened on purpose: grad / update /
    # param / pre_clip_grad / moment / moment_sq all live under l2_norm/; the native
    # ratios and amax-of-moment sit alongside. A new l2_norm/* or *_ratio monitor is
    # then loaded automatically (run `python -m instability.tags` to see the set).
    "l2_norm/",
    "effective_lr/",
    "update_weight_ratio/",
    "update_rate/",
    "amax/",
    "derived/",
    # weights
    "weights_l2norm/",
    "weights_amax/",
    "weights_amean/",
    # activations + their gradients. Broad ``activations_`` catches every stat —
    # amax / amean / l2norm / kurtosis / massiverel<T> / massiveabs<T> — plus the
    # ``activations_gradients_*`` family, so new activation stats load automatically.
    # (NB: the slash form ``activations/<stat>/…layers.{}…`` is a cross-layer aggregate
    # with no per-layer index — deliberately not loaded; per-layer tools can't use it.)
    "activations_",
    # dispatcher / MoE families (both legacy + current naming)
    "dispatcher_layer_",
    "dispatcher_sum_metrics/",
    "moe_seq_aux_loss/",
    "moe_z_loss_router/",
    "moe_load_balance/",
    "moe_expert_utilization/",
    "moe_maxvio/",
    "moe_over_frac/",
    "moe_under_frac/",
    "moe_scale/",
    "moe_bias_",
    "moe_routing_",
    # expert activation distribution proxies
    "expert_activation/",
    # convenience for tests
    "old_moe_routing/",
)


# ---------------------------------------------------------------------------
# Default load filter (deny-list)
# ---------------------------------------------------------------------------
# Non-model families: eval harness, data profiler, per-callback timing, optimizer
# health probes, config bookkeeping. Never wanted in per-tag stability analysis, so
# the default load drops them and keeps everything else (every model monitor).
_NOISE_PREFIXES: tuple[str, ...] = (
    "data_profiler",
    "timers_callback",
    "metrics/",
    "metrics_gauntlet",
    "token_accuracy",
    "optim_health",
    "trainer/",
)

# Carved out of the "metrics/" deny-list: the validation NLL and its perplexity twin
# produced by `eval_loader` (see `metric_names` in the sber_configs templates).
_VAL_LOSS_PREFIXES: tuple[str, ...] = (
    "metrics/eval/LanguageCrossEntropy",
    "metrics/eval/LanguagePerplexity",
)


def default_load_keep(tag: str) -> bool:
    """Deny-list predicate for the default load: keep every model monitor, drop only
    the non-model noise in ``_NOISE_PREFIXES``. A brand-new monitor therefore flows
    into report / compare / temporal / eda / atlas automatically — audit a run's set
    with ``python -m instability.tags``.

    Carve-out for validation loss: ``metrics/`` as a whole is ICL-harness noise, but
    ``metrics/eval/LanguageCrossEntropy`` (per-token val NLL) and its perplexity twin
    are model-health signals on the same step axis as ``loss/train/*``, so they are
    kept. They only exist in runs configured with an ``eval_loader``.

    Two carve-outs around the slash-form ``activations/<stat>/…layers.{}…`` aggregates
    (cross-layer, no per-layer index — distinct from the per-layer ``activations_<stat>/
    …layers.N…`` form):

    * ``residual`` monitors are ALWAYS kept — including that slash-form aggregate.
    * other slash-form activation aggregates are dropped — per-layer tools can't place
      them on a layer axis, and their per-layer twins already load.
    """
    if tag.startswith(_VAL_LOSS_PREFIXES):
        return True
    if tag.startswith(_NOISE_PREFIXES):
        return False
    if "residual" in tag:
        return True
    if tag.startswith(("activations/", "activations_gradients/")):
        return False
    return True


# ---------------------------------------------------------------------------
# Tag classification helpers
# ---------------------------------------------------------------------------
_LAYER_RE = re.compile(r"/layer_(\d+)(?:$|/)|\.layers\.(\d+)\.")


def parse_layer_from_tag(tag: str) -> int | None:
    """Return the layer index encoded in ``tag``, or ``None``."""
    m = _LAYER_RE.search(tag)
    if not m:
        return None
    grp = next(filter(None, m.groups()), None)
    return int(grp) if grp is not None else None


_MAX_VIO_LAYER_RES = [
    re.compile(r"^dispatcher_layer_experts_max_vio_batch_avg/layer_\d+$"),
    re.compile(r"^moe_maxvio/layer_\d+$"),
]


def _is_max_vio_per_layer(tag: str) -> bool:
    return any(p.match(tag) for p in _MAX_VIO_LAYER_RES)


# ---------------------------------------------------------------------------
# Event-file reader (subprocess-safe, batched)
# ---------------------------------------------------------------------------
def _extract_scalar(value) -> float | None:
    if value.HasField("simple_value"):
        return float(value.simple_value)
    if value.HasField("tensor"):
        t = value.tensor
        if t.float_val:
            return float(t.float_val[0])
        if t.double_val:
            return float(t.double_val[0])
        if t.int_val:
            return float(t.int_val[0])
    return None


def _read_file_batch(
    args: tuple[list[str], tuple[str, ...] | None, tuple[str, ...] | None, bool],
) -> dict[str, tuple[list[int], list[float]]]:
    """Parse a batch of event files inside one subprocess.

    Filter at parse time: ``default_load_keep`` (deny-list) when ``deny_default``,
    else ``str.startswith(prefixes)`` (C-fast) or the legacy substring ``in`` test.
    Returns ``{tag: ([steps], [values])}`` with raw lists; deduplication and sort
    happen in the master.
    """
    files, prefixes, substrings, deny_default = args
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    use_prefixes = prefixes is not None
    out: dict[str, tuple[list[int], list[float]]] = {}
    for path in files:
        try:
            for ev in EventFileLoader(path).Load():
                if not ev.HasField("summary"):
                    continue
                step = int(ev.step)
                for v in ev.summary.value:
                    tag = v.tag
                    if deny_default:
                        if not default_load_keep(tag):
                            continue
                    elif use_prefixes:
                        if not tag.startswith(prefixes):
                            continue
                    elif substrings is not None:
                        if not any(s in tag for s in substrings):
                            continue
                    scalar = _extract_scalar(v)
                    if scalar is None:
                        continue
                    bucket = out.get(tag)
                    if bucket is None:
                        bucket = ([], [])
                        out[tag] = bucket
                    bucket[0].append(step)
                    bucket[1].append(scalar)
        except Exception as exc:  # noqa: BLE001 — surface but keep going
            warnings.warn(f"error reading {path}: {exc}")
    return out


def _merge_partials_to_arrays(
    partials: list[dict[str, tuple[list[int], list[float]]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Merge subprocess outputs into ``{tag: (steps, values)}`` arrays.

    Deduplicate duplicates at the same step by keeping max-abs (FSDP
    ranks log identical or close values; the extreme surfaces anomalies).
    """
    merged_lists: dict[str, tuple[list[int], list[float]]] = {}
    for part in partials:
        for tag, (steps, vals) in part.items():
            buf = merged_lists.get(tag)
            if buf is None:
                merged_lists[tag] = (list(steps), list(vals))
            else:
                buf[0].extend(steps)
                buf[1].extend(vals)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag, (steps, vals) in merged_lists.items():
        if not steps:
            continue
        s = np.fromiter(steps, dtype=np.int64, count=len(steps))
        v = np.fromiter(vals, dtype=np.float64, count=len(vals))
        # Sort by step (stable), then collapse duplicates with max-abs.
        order = np.argsort(s, kind="stable")
        s = s[order]
        v = v[order]
        # np.unique with return_inverse is fast in numpy.
        uniq, idx = np.unique(s, return_inverse=True)
        if uniq.size != s.size:
            absv = np.abs(v)
            best = np.full(uniq.size, -np.inf)
            best_val = np.empty(uniq.size, dtype=np.float64)
            # Vectorized "argmax by group via maximum.at on |v|".
            # We need the actual signed value, so do a two-pass:
            # 1. find max-abs per group, 2. pick a representative.
            np.maximum.at(best, idx, absv)
            # Mask out entries that are not the max-abs per group;
            # pick first match per group as the representative.
            keep = absv == best[idx]
            # If multiple rows share max-abs in the same group, np.unique
            # already keeps the first stable index.
            sel = np.zeros(uniq.size, dtype=np.int64)
            np.maximum.at(sel, idx, np.where(keep, np.arange(s.size), -1))
            sel = np.clip(sel, 0, s.size - 1)
            best_val = v[sel]
            s, v = uniq, best_val
        out[tag] = (s, v)
    return out


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def _cache_key(
    path: Path,
    prefixes: tuple[str, ...] | None,
    substrings: tuple[str, ...] | None,
    deny_default: bool = False,
) -> str:
    parts = [_CACHE_VERSION, str(path.resolve())]
    if deny_default:
        # Encode the deny-list itself so the cache invalidates if it changes.
        parts.append("DENY:" + "|".join(_NOISE_PREFIXES) + "|keep:residual")
    if prefixes is not None:
        parts.append("P:" + "|".join(prefixes))
    if substrings is not None:
        parts.append("S:" + "|".join(substrings))
    event_files = sorted(path.glob("events.out.tfevents.*"))
    if not event_files:
        event_files = sorted(path.rglob("events.out.tfevents.*"))
    for ef in event_files:
        st = ef.stat()
        parts.append(f"{ef.name}:{st.st_mtime_ns}:{st.st_size}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


def _cache_load(key: str) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    try:
        cf = _CACHE_DIR / f"{key}.pkl.gz"
        if cf.exists():
            with gzip.open(cf, "rb") as f:
                return pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"cache read failed: {exc}")
    return None


def _cache_save(key: str, out: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cf = _CACHE_DIR / f"{key}.pkl.gz"
        with gzip.open(cf, "wb", compresslevel=1) as f:
            pickle.dump(out, f, protocol=5)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"cache write failed: {exc}")


def clear_cache() -> int:
    """Remove all cached TB blobs. Returns count of files deleted."""
    if not _CACHE_DIR.exists():
        return 0
    n = 0
    for cf in _CACHE_DIR.glob("*.pkl.gz"):
        cf.unlink()
        n += 1
    return n


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------
def load_run_scalars(
    run_dir: str | Path,
    *,
    max_workers: int = DEFAULT_WORKERS,
    use_combined_filter: bool = True,
    tag_substrings: list[str] | None = None,
    prefixes: tuple[str, ...] | None = None,
    cache: bool = True,
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse all event files under ``run_dir`` into ``{tag: (steps, values)}``.

    Parameters
    ----------
    max_workers
        Subprocess pool size (default = ``min(cpu_count, 16)``).
    use_combined_filter
        When True and neither ``tag_substrings`` nor ``prefixes`` is
        passed, uses the ``default_load_keep`` deny-list (every model
        monitor minus non-model noise).
    tag_substrings
        Legacy filter: tag must contain at least one substring. Slower
        than ``prefixes`` but supported for back-compat.
    prefixes
        Tuple of prefixes; tag must ``startswith`` one. **Preferred**
        — uses the C-level ``str.startswith(tuple)`` fast path.
    cache
        If True (default), read/write to ``~/.cache/instability_tb/``.
        Invalidated automatically when any event file changes (mtime+size).
    verbose
        Print stage timings.
    """
    t0 = perf_counter()
    run_dir = Path(run_dir)

    # Resolve the filter. With no explicit prefixes/substrings, the default is the
    # ``default_load_keep`` deny-list (not DEFAULT_PREFIXES) — keep every model
    # monitor, drop only non-model noise, so new monitors load automatically.
    deny_default = prefixes is None and tag_substrings is None and use_combined_filter
    sub_tuple = tuple(tag_substrings) if tag_substrings else None

    cache_key = (
        _cache_key(run_dir, prefixes, sub_tuple, deny_default) if cache else None
    )
    if cache_key:
        cached = _cache_load(cache_key)
        if cached is not None:
            if verbose:
                print(
                    f"[tb_helpers] cache hit for {run_dir.name}  "
                    f"({len(cached)} tags, {perf_counter() - t0:.2f}s)",
                    flush=True,
                )
            return cached

    files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not files:
        files = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not files:
        if verbose:
            print(f"[tb_helpers] no event files under {run_dir}", flush=True)
        return {}

    n_files = len(files)
    n_workers = max(1, min(max_workers, n_files))
    batch_size = max(1, math.ceil(n_files / n_workers))
    batches: list[list[str]] = [
        [str(p) for p in files[i : i + batch_size]]
        for i in range(0, n_files, batch_size)
    ]
    if verbose:
        print(
            f"[tb_helpers] reading {n_files} event file(s) "
            f"in {len(batches)} batch(es) × {n_workers} worker(s)...",
            flush=True,
        )

    args = [(b, prefixes, sub_tuple, deny_default) for b in batches]
    partials: list[dict[str, tuple[list[int], list[float]]]] = []
    try:
        with ProcessPoolExecutor(max_workers=len(batches)) as pool:
            for part in pool.map(_read_file_batch, args):
                partials.append(part)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"ProcessPoolExecutor failed ({exc}); falling back to threads")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for part in pool.map(_read_file_batch, args):
                partials.append(part)

    if verbose:
        print(
            f"[tb_helpers]   parsed in {perf_counter() - t0:.1f}s, merging...",
            flush=True,
        )
    out = _merge_partials_to_arrays(partials)
    if verbose:
        print(
            f"[tb_helpers]   done: {len(out)} tag(s) in {perf_counter() - t0:.1f}s",
            flush=True,
        )

    if cache_key and out:
        _cache_save(cache_key, out)
    return out


# ---------------------------------------------------------------------------
# Series builders consumed by eda / temporal / projections
# ---------------------------------------------------------------------------
def build_max_vio_series(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Build the global per-step max_vio series from per-layer tags.

    Returns ``(steps, max_vio, per_layer)`` where ``max_vio[i]`` is the
    max across all matching per-layer tags at ``steps[i]``.
    Vectorized — no Python-level loop over (layers × steps).
    """
    per_layer: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tag, (s, v) in series.items():
        if not _is_max_vio_per_layer(tag):
            continue
        L = parse_layer_from_tag(tag)
        if L is None:
            continue
        per_layer[L] = (s, v)

    if not per_layer:
        return np.zeros(0, dtype=np.int64), np.zeros(0), {}

    # Union step set, sorted.
    all_steps_set: set[int] = set()
    for s, _ in per_layer.values():
        all_steps_set.update(int(x) for x in s)
    all_steps = np.array(sorted(all_steps_set), dtype=np.int64)
    n_steps = all_steps.size
    if n_steps == 0:
        return all_steps, np.zeros(0), per_layer

    # Build the layer×step grid vectorized (np.searchsorted + scatter).
    accum = np.full(n_steps, -np.inf)
    for L, (s, v) in per_layer.items():
        idx = np.searchsorted(all_steps, s)
        # Some s may not be in all_steps (only if floats vs ints). Guard.
        good = (idx < n_steps) & (all_steps[np.clip(idx, 0, n_steps - 1)] == s)
        if not np.any(good):
            continue
        sub_idx = idx[good]
        sub_v = v[good]
        np.maximum.at(accum, sub_idx, sub_v)
    accum[~np.isfinite(accum)] = np.nan
    return all_steps, accum, per_layer


def argmax_layer_histogram(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[int, int], dict[int, float]]:
    """Per-layer histogram of "was global argmax over per-layer max_vio
    at this step": ``(counts, share)`` keyed by layer."""
    steps, _, per_layer = build_max_vio_series(series)
    if steps.size == 0 or not per_layer:
        return {}, {}
    layers = sorted(per_layer.keys())
    n_steps = steps.size
    grid = np.full((len(layers), n_steps), -np.inf)
    for li, L in enumerate(layers):
        s, v = per_layer[L]
        idx = np.searchsorted(steps, s)
        good = (idx < n_steps) & (steps[np.clip(idx, 0, n_steps - 1)] == s)
        grid[li, idx[good]] = v[good]
    finite_mask = np.any(np.isfinite(grid), axis=0)
    if not finite_mask.any():
        return {L: 0 for L in layers}, {L: 0.0 for L in layers}
    argmax_layers = np.argmax(grid[:, finite_mask], axis=0)
    counts = {L: 0 for L in layers}
    for li in argmax_layers:
        counts[layers[int(li)]] += 1
    total = int(finite_mask.sum())
    share = {L: counts[L] / total for L in layers}
    return counts, share
