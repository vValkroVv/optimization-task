"""Compact textual reports over analyze / stats / joint JSONs.

Designed to be called by an LLM agent and read directly: aggregates
thousands of tensor records into per-(role, projection) tables, top-N
outliers per discovered metric, per-expert hot spots, joint-spectrum
alignment summaries, and (for multiple files) per-tensor deltas.

No plots, no PNGs — stdout only, fixed-width tables. Output is bounded:
each section caps at `--top-n` rows.

Usage:
    python -m weights report FILE [FILE ...] [--top-n N] [--out PATH]

Auto-detects file kind from top-level keys:
    {"tensors": ..., "meta": ...}       → analyze.json
    {"joint": ..., "alignment": ...}    → joint.json
    flat {fqn: {mean, std, ...}, ...}   → stats.json

Multiple files of the same kind → per-file report + a delta report
joined on tensor FQN (sorted by tensor name, useful for iteration
sequences like ep0_ba2000 / ep0_ba4000 / ...).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# File-kind detection
# ---------------------------------------------------------------------------
def detect_kind(blob: Any) -> str:
    if isinstance(blob, dict):
        if "tensors" in blob and isinstance(blob["tensors"], dict):
            return "analyze"
        if "joint" in blob or "alignment" in blob:
            return "joint"
        # stats.json is flat {fqn: {mean, std, quantiles, ...}}.
        # Check a sample value.
        for v in blob.values():
            if isinstance(v, dict) and ("std" in v or "quantiles" in v):
                return "stats"
    return "unknown"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _flatten_numeric(d: dict, prefix: str = "") -> dict[str, float]:
    """Walk a nested dict, return {dotted_path: float} for scalar leaves."""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_numeric(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            if math.isfinite(v):
                out[key] = float(v)
    return out


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "      -"
    ax = abs(x)
    if ax == 0:
        return "      0"
    if ax < 1e-3 or ax >= 1e5:
        return f"{x:>10.3e}"
    return f"{x:>10.4g}"


def _hdr(title: str, *, char: str = "=") -> None:
    bar = char * 78
    print(f"\n{bar}\n  {title}\n{bar}")


def _table(rows: list[list[str]], headers: list[str]) -> None:
    if not rows:
        print("  (empty)")
        return
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(c))
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        padded = r + [""] * (len(widths) - len(r))
        print(fmt.format(*padded))


# ---------------------------------------------------------------------------
# analyze.json report
# ---------------------------------------------------------------------------
def _analyze_record_metrics(rec: dict) -> dict[str, float]:
    """Flatten the numeric metrics across base / rowcol / spectral blocks."""
    out: dict[str, float] = {}
    for block in ("base", "rowcol", "spectral"):
        sub = rec.get(block)
        if isinstance(sub, dict):
            out.update(_flatten_numeric(sub, block))
    return out


# Curated metric set: which metrics carry diagnostic signal, which direction
# is "worse", and a short label. Keeps default output scannable; pass
# `--all-metrics` for the firehose.
#   direction:
#     "high"  — bigger is worse  (tail/scale/collapse-via-amplitude)
#     "low"   — smaller is worse (rank collapse)
#     "abs"   — extreme in either direction  (signed asymmetry)
#     "alert" — any nonzero is bad  (NaN / Inf counters)
ANALYZE_CURATED: list[tuple[str, str, str]] = [
    ("base.n_nan", "alert", "NaN count"),
    ("base.n_inf", "alert", "Inf count"),
    ("base.max_abs", "high", "outlier amplitude"),
    ("base.std", "high", "scale (drift = exploding training)"),
    ("base.mean_abs", "high", "average amplitude"),
    ("base.kurt_excess", "high", "tail heaviness (>5 = heavy)"),
    ("base.skew", "abs", "distribution asymmetry"),
    ("rowcol.row.max_over_median", "high", "row rank-1-collapse"),
    ("rowcol.col.max_over_median", "high", "col rank-1-collapse"),
    ("spectral.stable_rank", "low", "stable rank (low = collapsed)"),
    ("spectral.effective_rank_entropy_topk", "low", "effective rank"),
    ("spectral.spectral_norm", "high", "σ₁ (energy)"),
]


def _rank_key(direction: str):
    """Sort key producing 'worst-first' order for a given metric direction."""
    if direction == "low":
        return lambda v: v
    if direction == "abs":
        return lambda v: -abs(v)
    # "high" and "alert" both want descending by value.
    return lambda v: -v


def _report_analyze(blob: dict, label: str, top_n: int, all_metrics: bool) -> None:
    tensors: dict[str, dict] = blob.get("tensors", {})
    meta = blob.get("meta", {})

    _hdr(f"analyze.json — {label}")
    print(
        f"  tensors: {len(tensors)}   errors: {meta.get('n_errors', 0)}   "
        f"elapsed: {meta.get('elapsed_sec', '?')}s   "
        f"device: {meta.get('device', '?')}"
    )

    # Normalize records: handle both `projection` (older field name) and
    # `proj` (current). Block flag separates MTP from main stack.
    def _proj(rec: dict) -> str:
        return str(rec.get("projection") or rec.get("proj") or "-")

    def _block(rec: dict) -> str:
        return str(rec.get("block") or "main")

    def _grp_key(rec: dict) -> tuple[str, str, str]:
        # role | projection | block — three-way grouping keeps MTP and
        # main-stack tensors visually distinct.
        return (str(rec.get("role", "?")), _proj(rec), _block(rec))

    # ---- inventory ---------------------------------------------------------
    inv: dict[tuple[str, str, str], int] = {}
    for rec in tensors.values():
        k = _grp_key(rec)
        inv[k] = inv.get(k, 0) + 1
    rows = [
        [r, p, b, str(c)] for (r, p, b), c in sorted(inv.items(), key=lambda kv: -kv[1])
    ]
    print("\n  -- inventory (role × projection × block) --")
    _table(rows, ["role", "projection", "block", "count"])

    # Index: name → (flat metric dict, role, proj, layer, block).
    indexed: list[tuple[str, dict[str, float], str, str, Any, str]] = []
    for name, rec in tensors.items():
        flat = _analyze_record_metrics(rec)
        role, proj, block = _grp_key(rec)
        layer = rec.get("layer", "-")
        indexed.append((name, flat, role, proj, layer, block))

    # ---- HEADLINES: one worst tensor per curated metric --------------------
    _hdr("HEADLINES (worst tensor per curated metric)", char="-")
    headline_rows: list[list[str]] = []
    for metric, direction, desc in ANALYZE_CURATED:
        ents = [
            (d[metric], n, r, p, L, b) for n, d, r, p, L, b in indexed if metric in d
        ]
        if not ents:
            continue
        if direction == "alert":
            bad = [e for e in ents if e[0] > 0]
            if bad:
                tot = sum(int(e[0]) for e in bad)
                hits = len(bad)
                worst = max(bad, key=lambda e: e[0])
                headline_rows.append(
                    [
                        "!! ALERT",
                        metric,
                        _fmt(worst[0]),
                        f"{hits} tensor(s), Σ={tot}",
                        f"{worst[4]}/{worst[5]}",
                        worst[2],
                        worst[3],
                        worst[1],
                    ]
                )
            else:
                headline_rows.append(
                    ["ok", metric, "0", "(all zero)", "-", "-", "-", "-"]
                )
            continue
        ents.sort(key=lambda e: _rank_key(direction)(e[0]))
        v, n, r, p, L, b = ents[0]
        headline_rows.append(
            [
                direction,
                metric,
                _fmt(v),
                desc,
                f"{L}/{b}",
                r,
                p,
                n,
            ]
        )
    _table(
        headline_rows,
        [
            "dir",
            "metric",
            "value",
            "note / aggregate",
            "layer/block",
            "role",
            "proj",
            "name",
        ],
    )

    # ---- per-metric: aggregates + top-N -----------------------------------
    if all_metrics:
        # Firehose: every numeric metric found in any record.
        metric_specs: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for _, flat, *_ in indexed:
            for k in flat:
                if k not in seen:
                    seen.add(k)
                    # Heuristic direction: stable_rank-style "low" only for a
                    # tiny known set; otherwise default to "high".
                    if k in (
                        "spectral.stable_rank",
                        "spectral.effective_rank_entropy_topk",
                    ):
                        metric_specs.append((k, "low", ""))
                    elif k in (
                        "base.skew",
                        "base.mean",
                        "base.max_signed",
                        "base.min_signed",
                    ):
                        metric_specs.append((k, "abs", ""))
                    elif k.startswith("base.n_"):
                        metric_specs.append((k, "alert", ""))
                    else:
                        metric_specs.append((k, "high", ""))
        metric_specs.sort()
        hdr_label = "ALL metrics — aggregates + top-N (worst-first per direction)"
    else:
        metric_specs = ANALYZE_CURATED
        hdr_label = "curated metrics — aggregates + top-N (worst-first per direction)"

    _hdr(hdr_label, char="-")
    skipped: list[str] = []
    for metric, direction, _ in metric_specs:
        ents = [
            (d[metric], n, r, p, L, b) for n, d, r, p, L, b in indexed if metric in d
        ]
        if not ents:
            continue
        vals_all = [e[0] for e in ents]
        if max(vals_all) == min(vals_all):
            skipped.append(f"{metric}={_fmt(vals_all[0]).strip()}")
            continue
        grouped: dict[tuple[str, str, str], list[float]] = {}
        for v, _n, r, p, _l, b in ents:
            grouped.setdefault((r, p, b), []).append(v)
        rows = []
        for (r, p, b), vals in sorted(grouped.items()):
            rows.append(
                [
                    r,
                    p,
                    b,
                    str(len(vals)),
                    _fmt(min(vals)),
                    _fmt(_pct(vals, 0.5)),
                    _fmt(_pct(vals, 0.95)),
                    _fmt(max(vals)),
                ]
            )
        print(f"\n  metric: {metric}   (rank by: {direction})")
        _table(rows, ["role", "projection", "block", "n", "min", "p50", "p95", "max"])
        ents.sort(key=lambda e: _rank_key(direction)(e[0]))
        top = ents[:top_n]
        top_rows = [[_fmt(v), str(L), b, r, p, n] for v, n, r, p, L, b in top]
        print(f"  worst {top_n} by {direction}:")
        _table(top_rows, [metric, "layer", "block", "role", "proj", "name"])
    if skipped:
        print(f"\n  (skipped constant metrics: {', '.join(skipped)})")

    # ---- per-expert hot spots ---------------------------------------------
    pe_rows: list[tuple[float, str, int, int, str]] = []
    for name, rec in tensors.items():
        pe = rec.get("per_expert")
        if not isinstance(pe, dict):
            continue
        layer = rec.get("layer", "-")
        for pe_metric in ("abs_max", "std", "spectral_norm", "row_max_over_median"):
            arr = pe.get(pe_metric)
            if not isinstance(arr, list):
                continue
            for ei, v in enumerate(arr):
                if isinstance(v, (int, float)) and math.isfinite(v):
                    pe_rows.append((float(v), pe_metric, layer, ei, name))
    if pe_rows:
        _hdr("per-expert outliers (top by |value| per metric)", char="-")
        # Group by pe_metric, then top-N per metric.
        by_pe: dict[str, list] = {}
        for tup in pe_rows:
            by_pe.setdefault(tup[1], []).append(tup)
        for metric in sorted(by_pe):
            top = sorted(by_pe[metric], key=lambda e: -abs(e[0]))[:top_n]
            rows = [
                [_fmt(v), str(layer), f"e{ei}", name] for v, _, layer, ei, name in top
            ]
            print(f"\n  per_expert.{metric}:")
            _table(rows, [metric, "layer", "expert", "name"])


# ---------------------------------------------------------------------------
# stats.json report
# ---------------------------------------------------------------------------
def _report_stats(blob: dict, label: str, top_n: int) -> None:
    _hdr(f"stats.json — {label}")
    print(f"  params: {len(blob)}")

    by_metric: dict[str, list[tuple[float, str]]] = {}
    for name, rec in blob.items():
        if not isinstance(rec, dict):
            continue
        q = rec.get("quantiles") or {}
        std = rec.get("std")
        # Synthesize a few derived metrics that matter for tail behavior.
        derived: dict[str, float] = {}
        if (
            isinstance(std, (int, float))
            and std
            and isinstance(q.get("0.999"), (int, float))
        ):
            derived["upper_tail_ratio_q999_over_std"] = q["0.999"] / std
        qmax = max(
            (abs(q[k]) for k in ("0.0", "1.0") if isinstance(q.get(k), (int, float))),
            default=None,
        )
        if qmax is not None:
            derived["abs_max"] = qmax
        if isinstance(std, (int, float)) and std and qmax is not None:
            derived["abs_max_over_std"] = qmax / std

        flat = _flatten_numeric({k: v for k, v in rec.items() if k != "reservoir"})
        flat.update(derived)
        for k, v in flat.items():
            by_metric.setdefault(k, []).append((v, name))

    skipped: list[str] = []
    for metric in sorted(by_metric):
        entries = by_metric[metric]
        vals = [v for v, _ in entries]
        if max(vals) == min(vals):
            skipped.append(f"{metric}={_fmt(vals[0]).strip()}")
            continue
        print(
            f"\n  metric: {metric}   n={len(vals)}   "
            f"min={_fmt(min(vals))}  p50={_fmt(_pct(vals, 0.5))}  "
            f"p95={_fmt(_pct(vals, 0.95))}  max={_fmt(max(vals))}"
        )
        top = sorted(entries, key=lambda e: -abs(e[0]))[:top_n]
        _table(
            [[_fmt(v), name] for v, name in top],
            [metric, "name"],
        )
    if skipped:
        print(f"\n  (skipped constant metrics: {', '.join(skipped)})")


# ---------------------------------------------------------------------------
# joint.json report
# ---------------------------------------------------------------------------
def _report_joint(blob: dict, label: str, top_n: int) -> None:
    _hdr(f"joint.json — {label}")
    joint_block = blob.get("joint", {}) or {}
    align = blob.get("alignment", {}) or {}
    print(f"  joint groups: {len(joint_block)}   alignment layers: {len(align)}")

    # joint: rank by sigma[0] (top singular value) and joint_stable_rank
    by_sigma: list[tuple[float, str, float | None, float | None]] = []
    for key, rec in joint_block.items():
        if not isinstance(rec, dict):
            continue
        sig = rec.get("sigma")
        sigma1 = float(sig[0]) if isinstance(sig, list) and sig else None
        jsr = rec.get("joint_stable_rank")
        pr = rec.get("participation_ratio")
        pr1 = float(pr[0]) if isinstance(pr, list) and pr else None
        if sigma1 is not None:
            by_sigma.append(
                (sigma1, key, jsr if isinstance(jsr, (int, float)) else None, pr1)
            )

    if by_sigma:
        _hdr("joint: top groups by σ₁ (spectral norm)", char="-")
        top = sorted(by_sigma, key=lambda e: -e[0])[:top_n]
        _table(
            [
                [
                    _fmt(s),
                    key,
                    _fmt(jsr) if jsr is not None else "-",
                    _fmt(pr1) if pr1 is not None else "-",
                ]
                for s, key, jsr, pr1 in top
            ],
            ["sigma1", "group", "joint_sr", "participation_ratio[0]"],
        )
        _hdr("joint: lowest joint_stable_rank (= most rank-collapsed)", char="-")
        with_jsr = [t for t in by_sigma if t[2] is not None]
        bot = sorted(with_jsr, key=lambda e: e[2])[:top_n]
        _table(
            [
                [_fmt(jsr), key, _fmt(s), _fmt(pr1) if pr1 is not None else "-"]
                for s, key, jsr, pr1 in bot
            ],
            ["joint_sr", "group", "sigma1", "participation_ratio[0]"],
        )

    # alignment: full per-layer summary, sorted by explosion_score desc
    if align:
        _hdr("alignment per layer (sorted by explosion_score)", char="-")
        rows = []
        for layer_key, rec in align.items():
            if not isinstance(rec, dict):
                continue
            rows.append(
                (
                    rec.get("explosion_score")
                    if isinstance(rec.get("explosion_score"), (int, float))
                    else None,
                    layer_key,
                    rec.get("sigma_gate_1"),
                    rec.get("sigma_up_1"),
                    rec.get("cos_v_gate_up_top1"),
                    rec.get("cos_v_gate_up_top2"),
                    rec.get("joint_sr_gate"),
                    rec.get("joint_sr_up"),
                )
            )
        rows.sort(
            key=lambda r: -(r[0] if isinstance(r[0], (int, float)) else -math.inf)
        )
        _table(
            [
                [
                    layer,
                    _fmt(es) if es is not None else "-",
                    _fmt(sg),
                    _fmt(su),
                    _fmt(c1),
                    _fmt(c2),
                    _fmt(jsg) if jsg is not None else "-",
                    _fmt(jsu) if jsu is not None else "-",
                ]
                for es, layer, sg, su, c1, c2, jsg, jsu in rows
            ],
            [
                "layer",
                "expl_score",
                "sigma_g1",
                "sigma_u1",
                "cos_top1",
                "cos_top2",
                "jsr_gate",
                "jsr_up",
            ],
        )


# ---------------------------------------------------------------------------
# Multi-file delta report (analyze.json sequences)
# ---------------------------------------------------------------------------
def _delta_analyze(files: list[tuple[str, dict]], top_n: int) -> None:
    if len(files) < 2:
        return
    _hdr(f"DELTAS  ({files[0][0]}  →  {files[-1][0]})")
    first = files[0][1].get("tensors", {})
    last = files[-1][1].get("tensors", {})
    common = set(first) & set(last)
    if not common:
        print("  (no common tensor names)")
        return

    # Pick metrics present in both, in the first common record.
    sample = next(iter(common))
    metrics = sorted(
        set(_analyze_record_metrics(first[sample]))
        & set(_analyze_record_metrics(last[sample]))
    )
    if not metrics:
        print("  (no overlapping numeric metrics)")
        return

    for metric in metrics:
        deltas: list[tuple[float, float, float, str, Any, str, str]] = []
        for n in common:
            a = _analyze_record_metrics(first[n]).get(metric)
            b = _analyze_record_metrics(last[n]).get(metric)
            if a is None or b is None:
                continue
            rec = last[n]
            deltas.append(
                (
                    b - a,
                    a,
                    b,
                    str(rec.get("role", "?")),
                    rec.get("layer", "-"),
                    str(rec.get("projection") or "-"),
                    n,
                )
            )
        if not deltas:
            continue
        if all(d[0] == 0 for d in deltas):
            continue  # uninteresting metric for this pair
        print(f"\n  metric: {metric}   (n_common={len(deltas)})")
        # Biggest absolute change in either direction.
        top = sorted(deltas, key=lambda e: -abs(e[0]))[:top_n]
        rows = [
            [_fmt(d), _fmt(a), _fmt(b), str(layer), role, proj, name]
            for d, a, b, role, layer, proj, name in top
        ]
        _table(rows, ["Δ", "first", "last", "layer", "role", "proj", "name"])


# ---------------------------------------------------------------------------
# Trajectory mode — layer × iteration tables for one (role, projection)
# ---------------------------------------------------------------------------
_ITER_RE = re.compile(r"(ba\d+|ep\d+_ba\d+|iter[_-]?\d+|step[_-]?\d+)")


def _short_iter_label(label: str) -> str:
    """Pull a short iteration label from a file label like ``ep0_ba14000/analyze.json``.

    Falls back to the parent dir name or the full label if no pattern matches.
    """
    m = _ITER_RE.search(label)
    if m:
        return m.group(0)
    parent = label.rsplit("/", 1)[0] if "/" in label else label
    return parent or label


def _parse_trajectory_spec(spec: str) -> tuple[str, str | None, str | None]:
    """Parse ``ROLE``, ``ROLE/PROJ``, or ``ROLE/PROJ/BLOCK`` (case-sensitive)."""
    parts = spec.split("/")
    role = parts[0]
    proj = parts[1] if len(parts) > 1 and parts[1] else None
    block = parts[2] if len(parts) > 2 and parts[2] else None
    return role, proj, block


def _trajectory_analyze(
    files: list[tuple[str, dict]],
    spec: str,
    all_metrics: bool,
) -> None:
    """Print `metric → layer × iteration` tables for a (role, [proj], [block]).

    For each curated metric and each iteration file, aggregates over all
    tensors in the file matching the spec by `max` per layer (multiple
    matches per layer can happen when ``proj`` is omitted — that's the
    point: spot which (layer, projection) is drifting).

    Suppresses rows that are constant across iterations and metrics that
    have no matching tensors at all.
    """
    role, proj_filter, block_filter = _parse_trajectory_spec(spec)
    if not files:
        return

    _hdr(
        f"TRAJECTORY  role={role}"
        + (f"  proj={proj_filter}" if proj_filter else "")
        + (f"  block={block_filter}" if block_filter else "")
        + f"  ({len(files)} file(s))"
    )

    # Sort files by numeric iteration detected in the label (ba12000,
    # iter_500, step_42, etc.) so shell-glob alphabetic order
    # ('ba14000' before 'ba2000') doesn't scramble the trajectory.
    # Look for `(ba|iter|step)<N>` specifically; fall back to input
    # order for labels with no detectable iter-style number.
    _ITER_NUM_RE = re.compile(r"(?:ba|iter[_-]?|step[_-]?)(\d+)")

    def _iter_sort_key(entry):
        m = _ITER_NUM_RE.search(entry[0])
        return (0, int(m.group(1))) if m else (1, entry[0])

    files = sorted(files, key=_iter_sort_key)
    iter_labels = [_short_iter_label(lbl) for lbl, _ in files]

    # Collect: {metric: {layer: {iter_label: max_value}}}
    by_metric: dict[str, dict[Any, dict[str, float]]] = {}
    layers_seen: set[Any] = set()
    matched_total = 0
    for (lbl, blob), it in zip(files, iter_labels):
        tensors = blob.get("tensors", {})
        for rec in tensors.values():
            if str(rec.get("role", "")) != role:
                continue
            if proj_filter is not None:
                rec_proj = str(rec.get("proj") or rec.get("projection") or "")
                if rec_proj != proj_filter:
                    continue
            if block_filter is not None:
                if str(rec.get("block") or "main") != block_filter:
                    continue
            matched_total += 1
            layer = rec.get("layer", "-")
            layers_seen.add(layer)
            flat = _analyze_record_metrics(rec)
            for k, v in flat.items():
                slot = by_metric.setdefault(k, {}).setdefault(layer, {})
                # Reduce duplicates (same layer, multiple projs when filter
                # is broad) by max — surfaces the worst contributor.
                prev = slot.get(it)
                slot[it] = v if prev is None else max(prev, v)

    if not matched_total:
        print(
            f"  (no tensors match role={role!r}"
            + (f", proj={proj_filter!r}" if proj_filter else "")
            + (f", block={block_filter!r}" if block_filter else "")
            + ")"
        )
        return
    print(f"  matched {matched_total} tensor record(s) across files")

    def _layer_key(layer):
        return (0, layer) if isinstance(layer, (int, float)) else (1, str(layer))

    sorted_layers = sorted(layers_seen, key=_layer_key)

    # Pick metrics. Curated metrics first (worth-it diagnostics); with
    # --all-metrics, append the rest.
    curated_keys = [m for m, _, _ in ANALYZE_CURATED]
    metric_order = list(curated_keys)
    if all_metrics:
        extra = sorted(set(by_metric) - set(curated_keys))
        metric_order = metric_order + extra

    skipped: list[str] = []
    for metric in metric_order:
        if metric not in by_metric:
            continue
        per_layer = by_metric[metric]
        # Build a row per layer.
        rows: list[list[str]] = []
        all_vals: list[float] = []
        for layer in sorted_layers:
            cells = per_layer.get(layer, {})
            row = [str(layer)] + [
                _fmt(cells[it]) if it in cells else "      -" for it in iter_labels
            ]
            rows.append(row)
            all_vals.extend(cells.values())
        if not all_vals:
            continue
        if max(all_vals) == min(all_vals):
            skipped.append(f"{metric}={_fmt(all_vals[0]).strip()}")
            continue
        print(f"\n  metric: {metric}   (cell = max across matched tensors per layer)")
        _table(rows, ["layer"] + iter_labels)

    if skipped:
        print(f"\n  (skipped constant metrics: {', '.join(skipped)})")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(
    paths: list[Path],
    *,
    top_n: int = 8,
    out: Path | None = None,
    all_metrics: bool = False,
    trajectory: str | None = None,
) -> None:
    streams = []
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        f = out.open("w")
        streams = [sys.stdout, f]
    else:
        streams = [sys.stdout]

    class _Tee:
        def write(self, s: str) -> int:
            for s_ in streams:
                s_.write(s)
            return len(s)

        def flush(self) -> None:
            for s_ in streams:
                s_.flush()

    orig_stdout = sys.stdout
    sys.stdout = _Tee()  # type: ignore[assignment]
    try:
        loaded: list[tuple[str, dict, str]] = []  # (label, blob, kind)
        for p in paths:
            blob = json.loads(p.read_text())
            kind = detect_kind(blob)
            label = p.parent.name + "/" + p.name if p.parent.name else p.name
            loaded.append((label, blob, kind))

        analyzes = [(lbl, b) for lbl, b, k in loaded if k == "analyze"]

        # Trajectory mode short-circuits the normal per-file reports —
        # the whole point is a focused cross-iteration view of one group.
        if trajectory is not None:
            if not analyzes:
                print("(trajectory mode requires at least one analyze.json file)")
                return
            _trajectory_analyze(analyzes, trajectory, all_metrics)
            return

        for label, blob, kind in loaded:
            if kind == "analyze":
                _report_analyze(blob, label, top_n, all_metrics)
            elif kind == "stats":
                _report_stats(blob, label, top_n)
            elif kind == "joint":
                _report_joint(blob, label, top_n)
            else:
                _hdr(f"UNKNOWN file kind — {label}")
                print(f"  top-level keys: {list(blob)[:8]}")

        # Multi-file deltas only meaningful for matching kinds.
        if len(analyzes) >= 2:
            _delta_analyze(analyzes, top_n)
    finally:
        sys.stdout = orig_stdout
        if out:
            streams[1].close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="weights report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "files", type=Path, nargs="+", help="one or more analyze/stats/joint JSON files"
    )
    ap.add_argument(
        "--top-n", type=int, default=8, help="rows per ranking table (default 8)"
    )
    ap.add_argument(
        "--all-metrics",
        action="store_true",
        help="emit every numeric metric found in records "
        "instead of just the curated diagnostic set "
        "(default: curated). Direction is guessed for "
        "uncurated metrics.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="if set, also write the report to this file",
    )
    ap.add_argument(
        "--trajectory",
        default=None,
        metavar="ROLE[/PROJ[/BLOCK]]",
        help="Layer × iteration tables for ONE group. Pass multiple "
        "analyze.json files in iteration order. Examples: "
        "`moe_expert_packed/w1`, `linear_attn/ba`, "
        "`moe_shared_expert/down/main`. Cell = max across matched "
        "tensors per (layer, iteration). Skips constant rows. Use "
        "after a `weights report` first-pass identified a "
        "(role, proj) group worth tracking through time.",
    )
    args = ap.parse_args(argv)
    run(
        args.files,
        top_n=args.top_n,
        out=args.out,
        all_metrics=args.all_metrics,
        trajectory=args.trajectory,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
