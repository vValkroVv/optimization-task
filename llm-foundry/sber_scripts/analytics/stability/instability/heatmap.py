"""Ad-hoc layer × step heatmaps for arbitrary TB tag families.

``report.py`` already emits curated heatmaps for ``max_vio`` and a few
fixed signal families. This script handles **any** tag prefix that has
a ``<prefix>/layer_N`` shape — useful when you spot a suspect signal
in a run (e.g. ``moe_load_balance`` peaks on the last layer) and want
the full depth × time picture without modifying ``report.py``.

For each ``--tag`` prefix:

* Walks all ``<prefix>/layer_N`` scalars under the run.
* Builds a ``[layer × step]`` grid, NaN-padded where a layer is missing
  a step.
* Writes ``<out-dir>/<prefix-sanitised>.png`` (heatmap + threshold mask
  panel) and ``.csv`` (raw grid + step/layer headers).

If the tag family also has a ``<prefix>/mean`` companion scalar, it's
drawn as a small line panel above the heatmap so you can see the
global trajectory alongside the per-layer breakdown.

Usage
-----

.. code-block:: bash

    python -m instability.heatmap \\
        --tb-dir /path/to/RUN \\
        --out-dir output/<exp>/heatmap_<run>/ \\
        --tags moe_load_balance moe_maxvio moe_expert_utilization
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from instability.tb_helpers import load_run_scalars

_LAYER_RE = re.compile(r"^(.+)/layer_(\d+)$")
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(s: str) -> str:
    return _SAFE_RE.sub("_", s).strip("_")


def _gather_layer_series(
    scalars: dict[str, tuple[np.ndarray, np.ndarray]],
    prefix: str,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return ``[(layer_idx, steps, values), ...]`` sorted by layer_idx."""
    out = []
    for tag, (s, v) in scalars.items():
        m = _LAYER_RE.match(tag)
        if m and m.group(1) == prefix:
            out.append((int(m.group(2)), s, v))
    out.sort(key=lambda r: r[0])
    return out


def _build_grid(
    series: list[tuple[int, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Build the layer × step grid (NaN where missing)."""
    layer_ids = [li for li, _, _ in series]
    all_steps = sorted({int(st) for _, s, _ in series for st in s})
    col_of = {st: i for i, st in enumerate(all_steps)}
    grid = np.full((len(series), len(all_steps)), np.nan, dtype=np.float64)
    for row_i, (_, s, v) in enumerate(series):
        for st, val in zip(s, v):
            j = col_of.get(int(st))
            if j is not None:
                grid[row_i, j] = float(val)
    return grid, layer_ids, all_steps


def _plot_heatmap(
    grid: np.ndarray,
    layer_ids: list[int],
    steps: list[int],
    title: str,
    mean_series: tuple[np.ndarray, np.ndarray] | None,
    out_path: Path,
    *,
    log_scale: bool,
    clip_pct: tuple[float, float],
    threshold: float | None,
) -> None:
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        print(f"[heatmap] {title}: grid empty, skipping", flush=True)
        return
    if log_scale:
        pos = finite[finite > 0]
        if pos.size == 0:
            print(
                f"[heatmap] {title}: log-scale requested but all values ≤ 0", flush=True
            )
            return
        vmin = float(np.percentile(pos, clip_pct[0]))
        vmax = float(np.percentile(pos, clip_pct[1]))
        if vmax <= vmin:
            vmin, vmax = float(pos.min()), float(pos.max())
        norm = LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)
    else:
        vmin = float(np.percentile(finite, clip_pct[0]))
        vmax = float(np.percentile(finite, clip_pct[1]))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

    n_panels = 1 + (mean_series is not None) + (threshold is not None)
    height = max(5.0, 0.18 * len(layer_ids) + 2.5)
    if mean_series is not None:
        height += 1.4
    if threshold is not None:
        height += 0.18 * len(layer_ids) + 0.5

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(16, height),
        gridspec_kw={
            "height_ratios": (
                ([1.4] if mean_series is not None else [])
                + [0.18 * len(layer_ids) + 1.0]
                + ([0.18 * len(layer_ids) + 0.5] if threshold is not None else [])
            )
        },
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]
    axes = list(axes)

    if mean_series is not None:
        ax_mean = axes.pop(0)
        ax_mean.plot(mean_series[0], mean_series[1], lw=1.2, color="#222")
        ax_mean.set_ylabel(f"{title} / mean", fontsize=9)
        ax_mean.grid(alpha=0.3)
        ax_mean.set_title(
            f"{title}  —  layer × step heatmap"
            + (
                f"  ({'log' if log_scale else 'linear'} colour, "
                f"clip {clip_pct[0]:.0f}/{clip_pct[1]:.0f}% pct)"
            ),
            fontsize=12,
            weight="bold",
        )

    ax_raw = axes.pop(0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#cccccc")
    im = ax_raw.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[steps[0], steps[-1], len(layer_ids) - 0.5, -0.5],
    )
    yt_step = max(1, len(layer_ids) // 24)
    ax_raw.set_yticks(range(0, len(layer_ids), yt_step))
    ax_raw.set_yticklabels(
        [f"L{layer_ids[i]}" for i in range(0, len(layer_ids), yt_step)],
        fontsize=8,
    )
    ax_raw.set_ylabel("layer")
    if mean_series is None:
        ax_raw.set_title(
            f"{title}  —  layer × step heatmap"
            + (
                f"  ({'log' if log_scale else 'linear'} colour, "
                f"clip {clip_pct[0]:.0f}/{clip_pct[1]:.0f}% pct)"
            ),
            fontsize=12,
            weight="bold",
        )
    cbar = fig.colorbar(im, ax=ax_raw)
    cbar.set_label(title)

    if threshold is not None:
        ax_thr = axes.pop(0)
        thr_grid = np.where(np.isfinite(grid), (grid > threshold).astype(float), np.nan)
        cmap2 = plt.get_cmap("Reds").copy()
        cmap2.set_bad(color="#dddddd")
        ax_thr.imshow(
            np.ma.masked_invalid(thr_grid),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap2,
            vmin=0,
            vmax=1,
            extent=[steps[0], steps[-1], len(layer_ids) - 0.5, -0.5],
        )
        ax_thr.set_yticks(range(0, len(layer_ids), yt_step))
        ax_thr.set_yticklabels(
            [f"L{layer_ids[i]}" for i in range(0, len(layer_ids), yt_step)],
            fontsize=8,
        )
        ax_thr.set_ylabel("layer")
        ax_thr.set_title(
            f"mask: value > {threshold:g}",
            fontsize=10,
        )

    axes_used = [a for a in fig.axes if a not in (cbar.ax,)]
    axes_used[-1].set_xlabel("step")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_csv(
    grid: np.ndarray,
    layer_ids: list[int],
    steps: list[int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("layer," + ",".join(str(s) for s in steps) + "\n")
        for row_i, li in enumerate(layer_ids):
            cells = [
                ("" if not np.isfinite(grid[row_i, j]) else f"{grid[row_i, j]:.6g}")
                for j in range(len(steps))
            ]
            f.write(f"L{li}," + ",".join(cells) + "\n")


def run(
    tb_dir: Path,
    out_dir: Path,
    tags: list[str],
    *,
    log_scale: bool = False,
    clip_pct: tuple[float, float] = (1.0, 99.0),
    threshold: float | None = None,
    max_workers: int = 8,
) -> None:
    # Pull only the tags we asked for — startswith(tuple) is C-fast.
    prefixes = tuple(tags) + tuple(f"{t}/mean" for t in tags)
    scalars = load_run_scalars(
        tb_dir,
        prefixes=prefixes,
        max_workers=max_workers,
    )
    if not scalars:
        raise SystemExit(f"no scalars matching {tags!r} under {tb_dir}")

    for tag in tags:
        series = _gather_layer_series(scalars, tag)
        if not series:
            print(f"[heatmap] no '{tag}/layer_N' tags found — skipping", flush=True)
            continue
        grid, layer_ids, steps = _build_grid(series)
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            print(f"[heatmap] {tag}: grid empty — skipping", flush=True)
            continue

        mean_series = scalars.get(f"{tag}/mean")
        if mean_series is not None:
            ms_steps, ms_vals = mean_series
            mean_series = (np.asarray(ms_steps), np.asarray(ms_vals))

        safe = _safe_name(tag)
        png_path = out_dir / f"{safe}.png"
        csv_path = out_dir / f"{safe}.csv"
        _plot_heatmap(
            grid,
            layer_ids,
            steps,
            tag,
            mean_series,
            png_path,
            log_scale=log_scale,
            clip_pct=clip_pct,
            threshold=threshold,
        )
        _write_csv(grid, layer_ids, steps, csv_path)

        print(
            f"[heatmap] {tag}: {len(layer_ids)} layers × {len(steps)} steps, "
            f"finite={finite.size}, value range "
            f"[{float(finite.min()):.4g}, {float(finite.max()):.4g}]  →  "
            f"{png_path}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--tb-dir",
        required=True,
        type=Path,
        help="TB run directory (containing events.out.tfevents.*).",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Where to write <prefix>.png + <prefix>.csv files.",
    )
    ap.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="One or more tag prefixes. Each must have a <prefix>/layer_N "
        "shape in TB. Example: --tags moe_load_balance moe_maxvio "
        "moe_expert_utilization.",
    )
    ap.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log colour scale (useful for ratio-style or long-tailed metrics).",
    )
    ap.add_argument(
        "--clip-pct",
        nargs=2,
        type=float,
        default=(1.0, 99.0),
        metavar=("LO", "HI"),
        help="Percentiles for robust colour clipping. Default 1/99.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, add a second panel showing a binary mask of "
        "value > THRESHOLD per layer × step. Handy for spotting "
        "where a metric crosses a warning level.",
    )
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args(argv)

    run(
        args.tb_dir,
        args.out_dir,
        args.tags,
        log_scale=args.log_scale,
        clip_pct=tuple(args.clip_pct),
        threshold=args.threshold,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
