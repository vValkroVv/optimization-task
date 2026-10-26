"""Render a folder of diagnostic plots from analyze.json (+ optional joint.json).

One command, one output directory, every (role, proj, block) group covered.

Output structure::

    <out_dir>/
    ├── 00_overview.png                       # role inventory + headlines
    ├── 01_metric_overview/<metric>.png       # per-group p50/p95/max per metric
    ├── 02_groups/<role>__<proj>__<block>/
    │   ├── <metric>.png                      # layer profile (per-layer dots/p95)
    │   └── experts_<metric>.png              # only for moe_expert: layer × expert heatmap
    ├── 03_joint_alignment.png                # only if joint.json supplied
    └── 04_abs_hist/<safe_name>.png           # log-|x| histograms for headline tensors

Usage::

    python -m weights dashboard analyze.json --out-dir dashboard/
    python -m weights dashboard analyze.json joint.json --out-dir dashboard/
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


# Same curated set as report.py, with direction.
CURATED: list[tuple[str, str, str]] = [
    ("base.n_nan", "alert", "NaN count"),
    ("base.n_inf", "alert", "Inf count"),
    ("base.max_abs", "high", "outlier amplitude"),
    ("base.std", "high", "scale"),
    ("base.mean_abs", "high", "average amplitude"),
    ("base.kurt_excess", "high", "tail heaviness"),
    ("base.skew", "abs", "distribution asymmetry"),
    ("rowcol.row.max_over_median", "high", "row rank-1-collapse"),
    ("rowcol.col.max_over_median", "high", "col rank-1-collapse"),
    ("spectral.stable_rank", "low", "stable rank"),
    ("spectral.effective_rank_entropy_topk", "low", "effective rank"),
    ("spectral.spectral_norm", "high", "σ₁ (energy)"),
]
CURATED_DIRECTION: dict[str, str] = {m: d for m, d, _ in CURATED}


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------
def _get(rec: dict, path: str) -> float | None:
    cur: Any = rec
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return (
        float(cur)
        if isinstance(cur, (int, float)) and math.isfinite(float(cur))
        else None
    )


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def _short(metric: str) -> str:
    return metric.rsplit(".", 1)[-1]


def _layer_sort(v: Any) -> tuple[int, Any]:
    return (0, v) if isinstance(v, (int, float)) else (1, str(v))


def _rank_key(direction: str):
    if direction == "low":
        return lambda v: v
    if direction == "abs":
        return lambda v: -abs(v)
    return lambda v: -v


# ---------------------------------------------------------------------------
# Page 1 — overview (role inventory + headlines)
# ---------------------------------------------------------------------------
def _plot_overview(indexed: list[tuple], meta: dict, out_path: Path) -> None:
    inv: dict[str, int] = defaultdict(int)
    for _, _, role, _, _, _ in indexed:
        inv[role] += 1
    items = sorted(inv.items(), key=lambda kv: -kv[1])

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])

    ax_meta = fig.add_subplot(gs[0, :])
    ax_meta.axis("off")
    ax_meta.text(
        0.01, 0.95, "analyze.json overview", fontsize=18, weight="bold", va="top"
    )
    ax_meta.text(
        0.01,
        0.55,
        f"hf_dir       : {meta.get('hf_dir', '?')}\n"
        f"tensors      : {meta.get('n_tensors', '?')}\n"
        f"errors       : {meta.get('n_errors', '?')}\n"
        f"device       : {meta.get('device', '?')}\n"
        f"elapsed      : {meta.get('elapsed_sec', '?')} s\n"
        f"topk_sigma   : {meta.get('topk_sigma', '?')}",
        fontsize=10,
        family="monospace",
        va="top",
    )

    ax_inv = fig.add_subplot(gs[1, 0])
    if items:
        labels = [k for k, _ in items]
        counts = [v for _, v in items]
        ax_inv.barh(labels, counts, color="#4C72B0")
        for i, c in enumerate(counts):
            ax_inv.text(c, i, f" {c}", va="center", fontsize=8)
        ax_inv.invert_yaxis()
    ax_inv.set_title("role inventory (tensor counts)", fontsize=11, weight="bold")
    ax_inv.set_xlabel("count")

    ax_h = fig.add_subplot(gs[1, 1])
    ax_h.axis("off")
    ax_h.set_title(
        "HEADLINES (worst tensor per curated metric)",
        fontsize=11,
        weight="bold",
        loc="left",
    )
    headlines = _collect_headlines(indexed)
    text_lines = []
    for metric, direction, val, (layer, role, proj, block, name) in headlines:
        if direction == "alert" and (val or 0) == 0:
            text_lines.append(f"  ok    {metric:<40} 0")
            continue
        tag = "!! " if direction == "alert" and (val or 0) > 0 else direction
        text_lines.append(
            f"  {tag:<5} {metric:<40} {val:>10.4g}  L{layer}/{block}  {role}/{proj}"
        )
    ax_h.text(0.0, 1.0, "\n".join(text_lines), fontsize=8, family="monospace", va="top")

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _collect_headlines(indexed: list[tuple]) -> list[tuple]:
    """For each curated metric: pick the single worst tensor (direction-aware)."""
    out = []
    for metric, direction, _desc in CURATED:
        candidates = [
            (_get(d, metric), n, r, p, L, b)
            for n, d, r, p, L, b in indexed
            if _get(d, metric) is not None
        ]
        if not candidates:
            continue
        if direction == "alert":
            bad = [c for c in candidates if c[0] > 0]
            if bad:
                w = max(bad, key=lambda e: e[0])
                out.append((metric, direction, w[0], (w[4], w[2], w[3], w[5], w[1])))
            else:
                out.append((metric, direction, 0.0, ("-", "-", "-", "-", "-")))
            continue
        candidates.sort(key=lambda e: _rank_key(direction)(e[0]))
        w = candidates[0]
        out.append((metric, direction, w[0], (w[4], w[2], w[3], w[5], w[1])))
    return out


# ---------------------------------------------------------------------------
# Per-metric overview — one PNG per curated metric.
# ---------------------------------------------------------------------------
def _plot_metric_overview(
    indexed: list[tuple],
    metric: str,
    direction: str,
    desc: str,
    out_path: Path,
) -> bool:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _n, d, r, p, _L, b in indexed:
        v = _get(d, metric)
        if v is None:
            continue
        grouped[(r, p, b)].append(v)
    if not grouped:
        return False
    all_vals = [v for vs in grouped.values() for v in vs]
    if max(all_vals) == min(all_vals):
        return False  # constant — skip

    items = []
    for k, vs in grouped.items():
        vs = sorted(vs)
        n = len(vs)
        p50 = vs[n // 2]
        p95 = vs[min(n - 1, int(0.95 * (n - 1)))]
        items.append((k, n, vs[0], p50, p95, vs[-1]))
    # Sort worst-first per direction so the eye lands on red.
    if direction == "low":
        items.sort(key=lambda r: r[3])  # by p50, smallest first = worst
    elif direction == "abs":
        items.sort(key=lambda r: -max(abs(r[2]), abs(r[5])))
    else:
        items.sort(key=lambda r: -r[5])  # by max, largest first

    fig, ax = plt.subplots(
        figsize=(11, max(4.0, 0.22 * len(items) + 1.5)),
        constrained_layout=True,
    )
    labels = [f"{r}/{p}/{b}" for (r, p, b), *_ in items]
    p50 = [r[3] for r in items]
    p95 = [r[4] for r in items]
    mx = [r[5] for r in items]
    mn = [r[2] for r in items]
    y = np.arange(len(items))
    ax.hlines(y, mn, mx, color="#cccccc", linewidth=2, label="min–max")
    ax.hlines(y, p50, p95, color="#666666", linewidth=4, label="p50–p95")
    ax.plot(p50, y, "o", color="#1f77b4", label="p50")
    ax.plot(mx, y, "s", color="#d62728", markersize=4, label="max")
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(metric)
    ax.set_title(
        f"{metric}   —   per-group distribution   (direction: {direction}; {desc})",
        fontsize=11,
        weight="bold",
    )
    ax.legend(loc="lower right", fontsize=7)
    if all(v > 0 for v in all_vals) and max(all_vals) / max(1e-30, min(all_vals)) > 100:
        ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Per-group layer profile — scatter per tensor at its layer.
# ---------------------------------------------------------------------------
def _plot_group_layer_profile(
    sub_idx: list[tuple],
    metric: str,
    direction: str,
    role: str,
    proj: str,
    block: str,
    out_path: Path,
) -> bool:
    pts: list[tuple[Any, float, Any]] = []  # (layer, value, expert)
    for n, d, _r, _p, L, _b in sub_idx:
        v = _get(d, metric)
        if v is None:
            continue
        m = re.search(r"\.experts\.(\d+)\.", n)
        expert = int(m.group(1)) if m else None
        pts.append((L, v, expert))
    if not pts:
        return False
    vals = [p[1] for p in pts]
    if max(vals) == min(vals):
        return False

    # Aggregate per layer.
    by_layer: dict[Any, list[float]] = defaultdict(list)
    for L, v, _ in pts:
        by_layer[L].append(v)
    layers = sorted(by_layer, key=_layer_sort)
    layer_max = [max(by_layer[L]) for L in layers]
    layer_p50 = [sorted(by_layer[L])[len(by_layer[L]) // 2] for L in layers]

    fig, ax = plt.subplots(
        figsize=(max(8, 0.45 * len(layers) + 3), 5), constrained_layout=True
    )
    x_labels = [str(L) for L in layers]
    x = np.arange(len(layers))
    # All individual points as scatter.
    if any(p[2] is not None for p in pts):
        # MoE — color by expert index, jitter on layer axis.
        rng = np.random.default_rng(0)
        for layer_i, L in enumerate(layers):
            sub = [(v, e) for ll, v, e in pts if ll == L]
            if not sub:
                continue
            ax.scatter(
                layer_i + rng.uniform(-0.25, 0.25, len(sub)),
                [s[0] for s in sub],
                c=[s[1] if s[1] is not None else 0 for s in sub],
                cmap="viridis",
                s=10,
                alpha=0.6,
                edgecolor="none",
            )
    else:
        for layer_i, L in enumerate(layers):
            for v in by_layer[L]:
                ax.scatter(layer_i, v, color="#4C72B0", s=20, alpha=0.7)
    ax.plot(
        x,
        layer_max,
        "-",
        color="#d62728",
        linewidth=1.5,
        label="per-layer max",
        alpha=0.8,
    )
    ax.plot(
        x,
        layer_p50,
        "-",
        color="#1f77b4",
        linewidth=1.5,
        label="per-layer p50",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45 if len(layers) > 20 else 0, fontsize=8)
    ax.set_xlabel("layer")
    ax.set_ylabel(metric)
    ax.set_title(
        f"{role} / {proj} / {block}   —   {metric}   "
        f"(n={len(pts)} tensors across {len(layers)} layer(s); direction: {direction})",
        fontsize=10,
        weight="bold",
    )
    if all(v > 0 for v in vals) and max(vals) / max(1e-30, min(vals)) > 100:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Per-expert × layer heatmap — only for moe_expert.
# ---------------------------------------------------------------------------
def _plot_expert_heatmap(
    sub_idx: list[tuple],
    metric: str,
    direction: str,
    role: str,
    proj: str,
    block: str,
    out_path: Path,
) -> bool:
    cells: list[tuple[int, int, float]] = []
    for n, d, _r, _p, L, _b in sub_idx:
        if L is None:
            continue
        v = _get(d, metric)
        if v is None:
            continue
        m = re.search(r"\.experts\.(\d+)\.", n)
        if not m:
            continue
        cells.append((int(L), int(m.group(1)), v))
    if not cells:
        return False
    layers = sorted({c[0] for c in cells})
    experts = sorted({c[1] for c in cells})
    if len(layers) < 2 or len(experts) < 2:
        return False

    layer_to_row = {L: i for i, L in enumerate(layers)}
    expert_to_col = {e: i for i, e in enumerate(experts)}
    grid = np.full((len(layers), len(experts)), np.nan, dtype=float)
    for L, e, v in cells:
        grid[layer_to_row[L], expert_to_col[e]] = v

    finite = grid[np.isfinite(grid)]
    if finite.size == 0 or finite.max() == finite.min():
        return False
    cmap = plt.get_cmap("viridis" if direction != "low" else "viridis_r").copy()
    cmap.set_bad(color="#cccccc")
    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
    use_log = vmin > 0 and vmax / vmin > 100
    norm = (
        LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)
        if use_log
        else Normalize(vmin=vmin, vmax=vmax)
    )

    fig, ax = plt.subplots(
        figsize=(max(8, 0.12 * len(experts) + 3), max(4, 0.25 * len(layers) + 2)),
        constrained_layout=True,
    )
    im = ax.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[experts[0] - 0.5, experts[-1] + 0.5, len(layers) - 0.5, -0.5],
    )
    yt = max(1, len(layers) // 24)
    ax.set_yticks(range(0, len(layers), yt))
    ax.set_yticklabels([f"L{layers[i]}" for i in range(0, len(layers), yt)], fontsize=8)
    xt = max(1, len(experts) // 24)
    ax.set_xticks(experts[::xt])
    ax.set_xticklabels([str(e) for e in experts[::xt]], fontsize=8)
    ax.set_xlabel("expert index")
    ax.set_ylabel("layer")
    direction_hint = "low = worse" if direction == "low" else "high = worse"
    ax.set_title(
        f"{role} / {proj} / {block}   —   {metric}   "
        f"(layer × expert; {direction_hint})",
        fontsize=10,
        weight="bold",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(metric)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Joint alignment scatter + per-layer panel.
# ---------------------------------------------------------------------------
def _plot_joint(joint_blob: dict, out_path: Path) -> bool:
    align = joint_blob.get("alignment", {}) or {}
    if not align:
        return False

    layers, sg, su, cos, expl, sr_g, sr_u = [], [], [], [], [], [], []
    for key, rec in align.items():
        m = re.match(r"L(\d+)$", key)
        if not m:
            continue
        layers.append(int(m.group(1)))
        sg.append(rec.get("sigma_gate_1") or 0)
        su.append(rec.get("sigma_up_1") or 0)
        cos.append(rec.get("cos_v_gate_up_top1") or 0)
        expl.append(rec.get("explosion_score") or 0)
        sr_g.append(rec.get("joint_sr_gate") or 0)
        sr_u.append(rec.get("joint_sr_up") or 0)
    if not layers:
        return False
    order = np.argsort(layers)
    layers = np.array(layers)[order]
    sg = np.array(sg)[order]
    su = np.array(su)[order]
    cos = np.array(cos)[order]
    expl = np.array(expl)[order]
    sr_g = np.array(sr_g)[order]
    sr_u = np.array(sr_u)[order]

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(
        np.clip(sg, 1e-3, None),
        np.clip(su, 1e-3, None),
        s=10 + 80 * (expl / max(expl.max(), 1e-30)),
        c=cos,
        cmap="magma",
        vmin=0,
        vmax=1,
        edgecolor="black",
        linewidth=0.5,
    )
    for L, x, y in zip(layers, sg, su):
        ax.annotate(f"L{L}", (x, y), fontsize=7, ha="left", va="bottom", color="#333")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("σ_gate_1 (log)")
    ax.set_ylabel("σ_up_1 (log)")
    ax.set_title(
        "joint alignment scatter\n(size = explosion_score, color = cos_top1)",
        fontsize=11,
        weight="bold",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("|cos_v_gate_up_top1|")
    ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(layers, expl, "-o", color="#d62728", label="explosion_score")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("explosion_score", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_yscale("log")
    ax2.set_title(
        "per-layer explosion_score & joint stable ranks", fontsize=11, weight="bold"
    )
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(layers, sr_g, "--", color="#1f77b4", label="joint_sr_gate")
    ax2b.plot(layers, sr_u, "--", color="#2ca02c", label="joint_sr_up")
    ax2b.set_ylabel("joint stable rank (lower = collapsed)")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Abs-hist for one tensor.
# ---------------------------------------------------------------------------
def _plot_abs_hist(
    tensor_rec: dict,
    name: str,
    log_min: float,
    log_max: float,
    out_path: Path,
) -> bool:
    base = tensor_rec.get("base") or {}
    counts = base.get("abs_hist_counts")
    if not counts:
        return False
    counts = np.asarray(counts, dtype=float)
    n_bins = len(counts)
    bin_edges = np.linspace(log_min, log_max, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(
        centers,
        counts,
        width=(log_max - log_min) / n_bins * 0.95,
        color="#4C72B0",
        alpha=0.85,
    )
    ax.set_yscale("log")
    ax.set_xlabel("log₁₀ |x|")
    ax.set_ylabel("count (log scale)")
    role = tensor_rec.get("role", "?")
    proj = tensor_rec.get("proj") or "-"
    block = tensor_rec.get("block") or "main"
    layer = tensor_rec.get("layer", "-")
    shape = tensor_rec.get("shape", "?")
    head = (
        f"max_abs={base.get('max_abs'):.4g}   "
        f"std={base.get('std'):.4g}   "
        f"kurt={base.get('kurt_excess'):.4g}   "
        f"skew={base.get('skew'):.4g}"
    )
    ax.set_title(
        f"{name}\n"
        f"shape={shape}  role={role}/{proj}  block={block}  layer={layer}\n"
        f"{head}",
        fontsize=9,
        weight="bold",
        loc="left",
    )
    ax.axvline(0, color="grey", lw=0.5)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(
    analyze_path: Path,
    joint_path: Path | None,
    out_dir: Path,
    *,
    max_hist: int = 20,
) -> None:
    blob = json.loads(analyze_path.read_text())
    tensors = blob.get("tensors", {})
    meta = blob.get("meta", {})

    # Index records: (name, flat-dict, role, proj, layer, block)
    indexed: list[tuple] = []
    for name, rec in tensors.items():
        role = str(rec.get("role", "?"))
        proj = str(rec.get("proj") or rec.get("projection") or "-")
        block = str(rec.get("block") or "main")
        layer = rec.get("layer")
        indexed.append((name, rec, role, proj, layer, block))

    joint_blob: dict | None = None
    if joint_path is not None:
        joint_blob = json.loads(joint_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    log_min = float(meta.get("hist", {}).get("log10_min", -8.0))
    log_max = float(meta.get("hist", {}).get("log10_max", 4.0))

    n_files = 0

    # ---- 00 overview ---------------------------------------------------
    _plot_overview(indexed, meta, out_dir / "00_overview.png")
    n_files += 1
    print("  wrote 00_overview.png", flush=True)

    # ---- 01 per-metric overview ----------------------------------------
    metric_dir = out_dir / "01_metric_overview"
    metric_dir.mkdir(parents=True, exist_ok=True)
    for metric, direction, desc in CURATED:
        ok = _plot_metric_overview(
            indexed, metric, direction, desc, metric_dir / f"{_safe(metric)}.png"
        )
        if ok:
            n_files += 1
    print(
        f"  wrote {sum(1 for _ in metric_dir.glob('*.png'))} metric-overview PNGs",
        flush=True,
    )

    # ---- 02 per-group layer profile + expert heatmap -------------------
    groups_dir = out_dir / "02_groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    by_group: dict[tuple[str, str, str], list[tuple]] = defaultdict(list)
    for tup in indexed:
        by_group[(tup[2], tup[3], tup[5])].append(tup)

    n_layer_plots = 0
    n_expert_plots = 0
    for (role, proj, block), sub in by_group.items():
        n_distinct_layers = len({t[4] for t in sub if t[4] is not None})
        if n_distinct_layers < 2:
            continue  # nothing useful to plot per layer
        gdir = groups_dir / f"{_safe(role)}__{_safe(proj)}__{_safe(block)}"
        gdir.mkdir(parents=True, exist_ok=True)
        for metric, direction, _desc in CURATED:
            ok = _plot_group_layer_profile(
                sub,
                metric,
                direction,
                role,
                proj,
                block,
                gdir / f"{_safe(metric)}.png",
            )
            if ok:
                n_layer_plots += 1
        if role == "moe_expert":
            for metric, direction, _desc in CURATED:
                ok = _plot_expert_heatmap(
                    sub,
                    metric,
                    direction,
                    role,
                    proj,
                    block,
                    gdir / f"experts_{_safe(metric)}.png",
                )
                if ok:
                    n_expert_plots += 1
    print(
        f"  wrote {n_layer_plots} layer-profile and {n_expert_plots} "
        f"per-expert heatmaps under 02_groups/",
        flush=True,
    )
    n_files += n_layer_plots + n_expert_plots

    # ---- 03 joint ------------------------------------------------------
    if joint_blob is not None:
        if _plot_joint(joint_blob, out_dir / "03_joint_alignment.png"):
            n_files += 1
            print("  wrote 03_joint_alignment.png", flush=True)
        else:
            print("  skipped joint alignment (no entries)", flush=True)

    # ---- 04 abs-hist for headline tensors ------------------------------
    headlines = _collect_headlines(indexed)
    hist_dir = out_dir / "04_abs_hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_hist = 0
    for _metric, _direction, _val, (_l, _r, _p, _b, name) in headlines:
        if name in seen or name == "-":
            continue
        seen.add(name)
        if name not in tensors:
            continue
        if _plot_abs_hist(
            tensors[name], name, log_min, log_max, hist_dir / f"{_safe(name)}.png"
        ):
            n_hist += 1
        if n_hist >= max_hist:
            break
    print(f"  wrote {n_hist} abs-hist PNGs", flush=True)
    n_files += n_hist

    print(f"\n[dashboard] total: {n_files} PNG(s) under {out_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="weights dashboard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="analyze.json (required), optionally joint.json too",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--max-hist",
        type=int,
        default=20,
        help="cap on abs-hist PNGs emitted for headline tensors",
    )
    args = ap.parse_args(argv)

    analyze_path: Path | None = None
    joint_path: Path | None = None
    for p in args.files:
        blob = json.loads(p.read_text())
        if "tensors" in blob:
            analyze_path = p
        elif "joint" in blob or "alignment" in blob:
            joint_path = p
        else:
            raise SystemExit(f"can't determine file kind: {p}")
    if analyze_path is None:
        raise SystemExit("need at least one analyze.json")

    run(analyze_path, joint_path, args.out_dir, max_hist=args.max_hist)


if __name__ == "__main__":
    main()
