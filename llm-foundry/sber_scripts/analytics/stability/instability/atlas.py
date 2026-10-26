"""Comprehensive plot atlas — render *every* monitor family, not just curated ones.

``instability.report`` hand-picks ~15 figures. But a run logs hundreds of
per-layer monitor families (activations, gradients, per-projection grad/update/
param norms, effective-lr, residual contribution, moments, massive activations,
max_vio, router/aux losses, …) and the report drops most of them at load time.

This module takes the *opposite* stance — the same one ``weights dashboard``
takes: plot everything we can into a structured PNG tree and let the model pick
what's relevant from the gallery. For every monitor family it builds a
``layer × step`` heatmap (the most information-dense view: where in the stack and
when does the signal move). Per-expert / per-slot duplicates at a layer are
reduced by max. Scalar (non-per-layer) families are grouped into one line figure
per category.

Output::

    <out>/atlas/<category>/<family>.png        # layer × step heatmaps
    <out>/atlas/<category>/_scalars.png         # non-per-layer series

``<category>`` comes from ``features.monitor_group`` (activation_amax, grad_norm,
update_norm, param_norm, effective_lr, residual_contribution, moment_norm,
activation_massive, max_vio, aux_loss, …) with a prefix fallback.

CLI: ``python -m instability.atlas --tb-dir RUN --out-dir OUT [--max-figs N]``.
Also called by ``instability.report`` unless ``--no-atlas`` is passed.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np

from instability.features import monitor_group
from instability.tb_helpers import parse_layer_from_tag

_LAYERS_RE = re.compile(r"\.layers\.\d+")
_LAYER_SLASH_RE = re.compile(r"/layer_\d+")
_INT_SEG_RE = re.compile(r"\.\d+(?=\.|$)")  # expert / slot indices → {i}


def _category(tag: str) -> str:
    """Coarse bucket for a tag. monitor_group first, then a prefix fallback."""
    g = monitor_group(tag)
    if g != "other":
        return g
    head = tag.split("/", 1)[0]
    if "max_vio" in tag:
        return "max_vio"
    if head.startswith("activations_gradients"):
        return "act_gradients"
    if head.startswith("activations_"):
        return head
    if head.startswith("l2_norm"):
        parts = tag.split("/")
        return f"l2_{parts[1]}" if len(parts) > 1 else "l2_norm"
    if head.startswith("amax"):
        return "amax_moment"
    if head in ("effective_lr", "update_weight_ratio", "update_rate"):
        return head
    if head.startswith("weights_l2norm"):
        return "weight_norm"
    if head.startswith("expert_activation"):
        return "expert_activation"
    if head.startswith("moe_"):
        return "moe"
    if head.startswith("loss"):
        return "loss"
    if head.startswith("lr"):
        return "lr"
    return head or "misc"


def _family(tag: str) -> str:
    """Normalize layer + expert/slot indices out of the tag → one family key."""
    f = _LAYERS_RE.sub(".layers.{L}", tag)
    f = _LAYER_SLASH_RE.sub("/layer_{L}", f)
    f = _INT_SEG_RE.sub(".{i}", f)
    return f


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("_")[:120]


Series = tuple[np.ndarray, np.ndarray]


def build_families(
    raw: dict[str, Series],
) -> tuple[
    dict[str, dict[str, dict[int, Series]]],
    dict[str, dict[str, list[tuple[str, Series]]]],
]:
    """Group raw tags into per-layer families and scalar (non-layer) families.

    Returns ``(layered, scalars)`` where::

        layered[category][family][layer] = (steps, values)   # experts reduced by max
        scalars[category][family]        = [(tag, (steps, values)), ...]
    """
    layered: dict[str, dict[str, dict[int, list[Series]]]] = {}
    scalars: dict[str, dict[str, list[tuple[str, Series]]]] = {}
    for tag, series in raw.items():
        cat = _category(tag)
        fam = _family(tag)
        layer = parse_layer_from_tag(tag)
        if layer is None:
            scalars.setdefault(cat, {}).setdefault(fam, []).append((tag, series))
        else:
            layered.setdefault(cat, {}).setdefault(fam, {}).setdefault(
                layer, []
            ).append(series)

    # Reduce per-(family, layer) duplicates (experts/slots) by per-step max.
    reduced: dict[str, dict[str, dict[int, Series]]] = {}
    for cat, fams in layered.items():
        for fam, per_layer in fams.items():
            for layer, series_list in per_layer.items():
                reduced.setdefault(cat, {}).setdefault(fam, {})[layer] = (
                    series_list[0]
                    if len(series_list) == 1
                    else _max_reduce(series_list)
                )
    return reduced, scalars


def _max_reduce(series_list: list[Series]) -> Series:
    all_steps = np.array(
        sorted({int(s) for st, _ in series_list for s in st}), dtype=np.int64
    )
    if all_steps.size == 0:
        return all_steps, np.zeros(0)
    col = {int(s): j for j, s in enumerate(all_steps)}
    acc = np.full(all_steps.size, -np.inf)
    for st, vs in series_list:
        for s, v in zip(st, vs):
            j = col.get(int(s))
            if j is not None and np.isfinite(v):
                acc[j] = max(acc[j], float(v))
    acc[~np.isfinite(acc)] = np.nan
    return all_steps, acc


def _grid(per_layer: dict[int, Series]) -> tuple[list[int], np.ndarray, np.ndarray]:
    layers = sorted(per_layer)
    all_steps = sorted({int(s) for L in layers for s in per_layer[L][0]})
    col = {s: j for j, s in enumerate(all_steps)}
    grid = np.full((len(layers), len(all_steps)), np.nan)
    for i, L in enumerate(layers):
        st, vs = per_layer[L]
        for s, v in zip(st, vs):
            j = col.get(int(s))
            if j is not None:
                grid[i, j] = float(v)
    return layers, np.array(all_steps, dtype=np.int64), grid


def _plot_heatmap(layers, steps, grid, title, out_path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    finite = grid[np.isfinite(grid)]
    if finite.size == 0 or steps.size < 2:
        return False
    # Log color scale when strictly positive with a wide dynamic range.
    pos = finite[finite > 0]
    use_log = pos.size == finite.size and pos.size and (pos.max() / pos.min()) > 100
    if use_log:
        norm = LogNorm(vmin=float(pos.min()), vmax=float(pos.max()))
    else:
        vmin = float(np.percentile(finite, 1))
        vmax = float(np.percentile(finite, 99))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max())
        norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-12))

    fig, ax = plt.subplots(
        figsize=(13, max(2.6, 0.16 * len(layers) + 1.6)), constrained_layout=True
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[steps[0], steps[-1], len(layers) - 0.5, -0.5],
    )
    yt = max(1, len(layers) // 24)
    ax.set_yticks(range(0, len(layers), yt))
    ax.set_yticklabels([f"L{layers[i]}" for i in range(0, len(layers), yt)], fontsize=7)
    ax.set_xlabel("step")
    ax.set_ylabel("layer")
    ax.set_title(title + ("   [log]" if use_log else ""), fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _plot_scalars(cat, fams, out_path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [(tag, s) for fam in fams.values() for tag, s in fam]
    series = [(t, (st, v)) for t, (st, v) in series if st.size > 1]
    if not series:
        return False
    series = series[:40]  # keep the figure legible
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for tag, (st, v) in series:
        ax.plot(st, v, lw=1.0, label=tag.split("/")[-1][:32])
    ax.set_xlabel("step")
    ax.set_title(f"{cat} — scalar series ({len(series)})", fontsize=11)
    if len(series) <= 20:
        ax.legend(fontsize=6, ncol=2)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_atlas(
    raw: dict[str, Series],
    out_dir: Path,
    run_name: str,
    *,
    max_figs: int = 600,
    min_layers: int = 2,
) -> int:
    """Render the full atlas under ``out_dir/atlas/``. Returns #PNGs written."""
    atlas_dir = out_dir / "atlas"
    layered, scalars = build_families(raw)

    # Rank families: most layers first, then name — so if we hit the cap we keep
    # the broadest (most informative) coverage.
    jobs: list[tuple[str, str, dict[int, Series]]] = [
        (cat, fam, per_layer)
        for cat, fams in layered.items()
        for fam, per_layer in fams.items()
        if len(per_layer) >= min_layers
    ]
    jobs.sort(key=lambda j: (-len(j[2]), j[0], j[1]))

    written = 0
    truncated = 0
    for cat, fam, per_layer in jobs:
        if written >= max_figs:
            truncated += 1
            continue
        layers, steps, grid = _grid(per_layer)
        cat_dir = atlas_dir / _safe(cat)
        cat_dir.mkdir(parents=True, exist_ok=True)
        title = f"{fam}   ({len(layers)} layers)"
        if _plot_heatmap(layers, steps, grid, title, cat_dir / f"{_safe(fam)}.png"):
            written += 1

    for cat, fams in scalars.items():
        cat_dir = atlas_dir / _safe(cat)
        cat_dir.mkdir(parents=True, exist_ok=True)
        if _plot_scalars(cat, fams, cat_dir / "_scalars.png"):
            written += 1

    cats = sorted({_safe(c) for c in {**layered, **scalars}})
    print(
        f"[atlas] wrote {written} PNGs across {len(cats)} categories → {atlas_dir}",
        flush=True,
    )
    if truncated:
        warnings.warn(
            f"[atlas] hit --max-figs ({max_figs}); skipped {truncated} smaller "
            f"families. Raise --max-figs to plot them all."
        )
    return written


def run(
    tb_dir: Path,
    out_dir: Path,
    *,
    run_name: str | None = None,
    workers: int = 8,
    max_figs: int = 600,
) -> int:
    from instability.tb_helpers import load_run_scalars

    # Default (deny-list) load: every model monitor, so the atlas draws a heatmap
    # for every family — new monitors included — minus non-model noise.
    raw = load_run_scalars(tb_dir, max_workers=workers, verbose=True)
    if not raw:
        raise SystemExit(f"no relevant tags in {tb_dir}")
    return plot_atlas(raw, out_dir, run_name or tb_dir.name, max_figs=max_figs)


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tb-dir", required=True, type=Path)
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="atlas lands in <out-dir>/atlas/<category>/",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--max-figs",
        type=int,
        default=600,
        help="cap on heatmap PNGs (broadest families kept first)",
    )
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(
        args.tb_dir,
        args.out_dir,
        run_name=args.name,
        workers=args.workers,
        max_figs=args.max_figs,
    )


if __name__ == "__main__":
    _cli()
