"""Stability report for one TB run.

Output layout::

    OUT_DIR/
    ├── figures/               # all PNG plots
    │   ├── 01_overview.png
    │   ├── 02_max_vio_heatmap.png
    │   ├── 03_layer_attribution.png
    │   ├── 04_moe_health.png
    │   ├── 05_activations_depth.png
    │   ├── 06_gradients_depth.png
    │   ├── 07_weights_depth.png
    │   ├── 08_system_health.png
    │   ├── 09_per_projection_depth.png
    │   ├── 10_router_dynamics.png
    │   ├── 11_loss_decomposition.png
    │   ├── 12_phase_diagnosis.png
    │   ├── 13_cross_layer_corr.png
    │   ├── 14_spike_clusters.png
    │   └── atlas/             # comprehensive per-family layer×step heatmaps
    │       └── <category>/<family>.png   (plot-everything; pick what's relevant)
    └── data/                  # CSV / MD / JSON sibling for each PNG
        ├── 01_overview.csv
        ├── 02_max_vio_per_layer.csv
        ├── ...
        ├── spikes.csv
        ├── layer_profile.csv
        ├── summary.md         # auto-generated narrative summary
        └── signals.json       # raw series, all groups

Every PNG ships with a sibling table in ``data/`` containing the
underlying numbers — designed so an LLM can analyze results without
parsing pixels. CSV format for raw time-series; Markdown for narrative
or ranked tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow .use)
import numpy as np


# ---------------------------------------------------------------------------
# Tag patterns — what we read & plot. Calibrated for prod PapayaSeek TB.
# ---------------------------------------------------------------------------
# Each group: list of tag regexes, an aggregation rule per step, and
# whether to retain per-tag series (for envelopes / heatmaps) vs only
# the combined one.
SIGNAL_GROUPS: dict[str, dict[str, Any]] = {
    "loss": {
        "patterns": [r"^loss/train/total$", r"^loss/train/loss_ce$"],
        "reduction": "mean",  # FSDP ranks log the same value
        "keep_per_tag": True,  # plot total + CE separately
    },
    "lr": {
        "patterns": [r"^lr-.*/group0$"],
        "reduction": "mean",
    },
    "grad_norm_global": {
        "patterns": [r"^l2_norm/grad/global$"],
        "reduction": "max",
    },
    "grad_norm_layers": {
        # One value per layer's flat-param vector. Used for the envelope.
        "patterns": [r"^l2_norm/grad/.*layers\.\d+\..*\.flatten$"],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "max_vio_global": {
        # The dispatcher's summarized max_vio (already aggregated server-side).
        # Only present in the legacy / 120B-Sandwich TB convention; the
        # current / 2T-tokens convention doesn't log a dedicated global —
        # we synthesize it from the per-layer envelope in that case.
        "patterns": [r"^dispatcher_sum_metrics/experts_max_vio_.*_avg$"],
        "reduction": "max",
    },
    "max_vio_per_layer": {
        # One tag per MoE layer. Used for the envelope AND for the heatmap.
        # We *only* pull from the legacy dispatcher tag here — the
        # alternative ``moe_maxvio/layer_N`` is captured by the separate
        # ``moe_maxvio_per_layer`` group below. If a run only has the
        # newer naming, ``read_signals`` aliases that group into
        # ``max_vio_per_layer`` after parsing so downstream code keeps working.
        "patterns": [r"^dispatcher_layer_experts_max_vio_batch_avg/layer_\d+$"],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "activation_amax": {
        # Two naming flavors:
        # * 120B-Sandwich: ``...mlp_residual_output.0`` and similar named slots.
        # * 2T-tokens:     ``...layers.N_output.0`` (the whole layer's output).
        "patterns": [
            r"^activations_amax/model\..*\.(self_attn_output|attn_residual_output|"
            r"mlp_residual_output|attn_residual_contribution_output|"
            r"mlp_residual_contribution_output|lm_head_input)\.0$",
            r"^activations_amax/model\.model\.layers\.\d+_output\.0$",
            r"^activations_amax/model\.lm_head_input\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "aux_loss": {
        "patterns": [r"^moe_seq_aux_loss/layer_\d+$"],
        "reduction": "max",  # peak aux loss across layers
    },
    "hidden_z_loss": {
        "patterns": [r"^loss/train/hidden_z_loss.*$"],
        "reduction": "mean",
    },
    "load_balance": {
        "patterns": [r"^moe_load_balance/mean$"],
        "reduction": "mean",
    },
    # ---- per-layer attribution signals (used by plot_layer_attribution) ----
    # Each layer is one tag here; we keep them per-tag so we can build
    # the ``[layer × step]`` heatmaps in plot_layer_attribution. They're
    # NOT plotted on report.png to keep that view uncluttered.
    "layer_residual_amax": {
        # max activation magnitude in residual stream per layer.
        # 120B-Sandwich: ``mlp_residual_output.0`` slot.
        # 2T-tokens:     ``layers.N_output.0`` (whole-layer output).
        "patterns": [
            r"^activations_amax/model\.model\.layers\.\d+\.mlp_residual_output\.0$",
            r"^activations_amax/model\.model\.layers\.\d+_output\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "layer_clip_share_gate": {
        # what share of gate activations got clipped this step (per layer)
        "patterns": [
            r"^activations_amax/model\.model\.layers\.\d+\.mlp_gate_clip_share$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "layer_clip_share_up": {
        # what share of up-projection activations got clipped (per layer)
        "patterns": [
            r"^activations_amax/model\.model\.layers\.\d+\.mlp_up_clip_share$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "layer_attn_amax": {
        # attention output amax per layer — proxy for attention saturation
        "patterns": [
            r"^activations_amax/model\.model\.layers\.\d+\.attn_residual_output\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    # ---- residual-connection contributions (how much each sublayer writes into
    # the residual stream — the native residual-tracking monitors) ----
    "layer_attn_residual_contribution": {
        "patterns": [
            r"^activations_amax/model\.model\.layers\.\d+\.attn_residual_contribution_output\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "layer_mlp_residual_contribution": {
        "patterns": [
            r"^activations_amax/model\.model\.layers\.\d+\.mlp_residual_contribution_output\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    # ---- Expanded loss family ----
    "loss_components": {
        # All loss components — see also `loss` for headline total/CE only.
        "patterns": [
            r"^loss/train/loss_mtp$",
            r"^loss/train/hidden_z_loss$",
            r"^loss/train/hidden_z_loss_moe$",
            r"^loss/train/hidden_z_loss_attn$",
            r"^loss/train/hidden_z_loss_dense_mlp$",
        ],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    # ---- MoE router health (deep dive) ----
    "moe_maxvio_per_layer": {
        # Alternative (newer) per-layer max_vio metric.
        "patterns": [r"^moe_maxvio/layer_\d+$"],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "moe_load_balance_per_layer": {
        "patterns": [r"^moe_load_balance/layer_\d+$"],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "moe_expert_utilization_per_layer": {
        # Per-layer expert utilization. < 0.5 means router has dead experts.
        "patterns": [r"^moe_expert_utilization/layer_\d+$"],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "moe_expert_utilization_mean": {
        "patterns": [r"^moe_expert_utilization/mean$"],
        "reduction": "mean",
    },
    "moe_over_frac_per_layer": {
        # Fraction of "over-capacity" routing per layer.
        "patterns": [r"^moe_over_frac/layer_\d+$"],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "dispatcher_overloaded_per_layer": {
        "patterns": [
            r"^dispatcher_layer_overloaded_ratio/layer_\d+$",
            r"^dispatcher_layer_experts_overloaded_ratio/layer_\d+$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "dispatcher_globals": {
        # Global router health signals — line-plot category.
        "patterns": [
            r"^dispatcher_sum_metrics/empty_token_level$",
            r"^dispatcher_sum_metrics/overloaded_expert_ratio$",
            r"^dispatcher_sum_metrics/step_max_expert_load$",
            r"^dispatcher_sum_metrics/num_drop_steps$",
            r"^dispatcher_sum_metrics/token_drop_level$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    # ---- MoE router bias (aux-loss-free style) ----
    "moe_bias_stats_per_layer": {
        # Router bias statistics — useful when aux_loss_free strategy
        # is enabled (bias is the learned balancer). max_min = range.
        "patterns": [
            r"^moe_bias_(?:mean|std|max|min|max_min)/layer_\d+$",
            r"^moe_bias_updated_(?:mean|std|max|min|max_min)/layer_\d+$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    # ---- Expert-level activations ----
    "expert_activation_per_layer": {
        # Distribution-level signals computed across experts in each layer.
        "patterns": [
            r"^expert_activation/max_median_ratio/layer_\d+$",
            r"^expert_activation/mean_norm_per_token/layer_\d+$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    # ---- Activation gradients (residual / attention output backward) ----
    "layer_residual_grad_amax": {
        "patterns": [
            r"^activations_gradients_amax/model\.model\.layers\.\d+"
            r"\.self_attn\.post_attention_layernorm_grad_output\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "layer_residual_grad_l2norm": {
        "patterns": [
            r"^activations_gradients_l2norm/model\.model\.layers\.\d+"
            r"\.self_attn\.post_attention_layernorm_grad_output\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "layer_attn_o_grad_amax": {
        # Attention o-proj output gradient (proxy for attention gradient pressure).
        "patterns": [
            r"^activations_gradients_amax/model\.model\.layers\.\d+"
            r"\.self_attn\._attn\.o_proj_grad_output\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    # ---- Weights (per-layer L2 norm of the residual-stream norm weight) ----
    "layer_norm_weight_l2": {
        # L2 norm of post_attention_layernorm weights — drift signal.
        "patterns": [
            r"^weights_l2norm/model\.model\.layers\.\d+"
            r"\.self_attn\.post_attention_layernorm\.weight$",
        ],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "layer_o_proj_weight_l2": {
        # L2 norm of attention o-proj weight — drift over training.
        "patterns": [
            r"^weights_l2norm/model\.model\.layers\.\d+"
            r"\.self_attn\._attn\.o_proj\.weight$",
        ],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "layer_o_proj_weight_amax": {
        "patterns": [
            r"^weights_amax/model\.model\.layers\.\d+"
            r"\.self_attn\._attn\.o_proj\.weight$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    # ---- Activation mean (different slice than amax — average not peak) ----
    "layer_residual_amean": {
        "patterns": [
            r"^activations_amean/model\.model\.layers\.\d+\.mlp_residual_output\.0$",
        ],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "layer_residual_l2norm": {
        "patterns": [
            r"^activations_l2norm/model\.model\.layers\.\d+\.mlp_residual_output\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    # ---- System health ----
    "memory_alloc_retries": {
        # > 0 means GPU memory fragmented; precursor to OOM / hangs.
        "patterns": [r"^memory/alloc_retries$"],
        "reduction": "max",
    },
    "memory_peak_allocated": {
        "patterns": [r"^memory/peak_allocated_mem$"],
        "reduction": "max",
    },
    "memory_peak_reserved": {
        "patterns": [r"^memory/peak_reserved_mem$"],
        "reduction": "max",
    },
    "throughput_tokens": {
        "patterns": [r"^throughput/tokens_per_sec$"],
        "reduction": "mean",
    },
    "throughput_device_tokens": {
        "patterns": [r"^throughput/device/tokens_per_sec$"],
        "reduction": "mean",
    },
    "time_per_batch": {
        "patterns": [r"^time_seconds/batch/batch_total$"],
        "reduction": "mean",
    },
    "time_forward": {
        "patterns": [r"^time_seconds/batch/forward$"],
        "reduction": "mean",
    },
    "time_backward": {
        "patterns": [r"^time_seconds/batch/backward$"],
        "reduction": "mean",
    },
    "time_dataloader": {
        "patterns": [r"^time_seconds/batch/dataloader_fetch$"],
        "reduction": "mean",
    },
    # ---- Sequence length / batch shape (curriculum tracking) ----
    "seq_length": {
        "patterns": [r"^seq_length_warmup/curr_seq_len$"],
        "reduction": "mean",
    },
    "batch_size_curr": {
        "patterns": [r"^seq_length_warmup/curr_bs$"],
        "reduction": "mean",
    },
    # ---- Per-projection weight L2 norms (for projection breakdown) ----
    # Each family captures all 40 layers; we use these for the
    # per-projection depth plot, which buckets by projection type.
    "proj_weight_l2_q": {
        "patterns": [
            r"^weights_l2norm/.*self_attn\._attn\.q_proj\.weight$",
            r"^weights_l2norm/.*self_attn\._attn\.uq_proj\.weight$",
            r"^weights_l2norm/.*self_attn\._attn\.dq_proj\.weight$",
        ],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "proj_weight_l2_kv": {
        "patterns": [
            r"^weights_l2norm/.*self_attn\._attn\.kv_proj\.weight$",
            r"^weights_l2norm/.*self_attn\._attn\.dkv_proj\.weight$",
            r"^weights_l2norm/.*self_attn\._attn\.k_proj\.weight$",
            r"^weights_l2norm/.*self_attn\._attn\.v_proj\.weight$",
            r"^weights_l2norm/.*self_attn\._attn\.kr_proj\.weight$",
        ],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "proj_weight_l2_o": {
        "patterns": [r"^weights_l2norm/.*self_attn\._attn\.o_proj\.weight$"],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "proj_weight_l2_router": {
        "patterns": [r"^weights_l2norm/.*self_attn\._gate\.gate\.weight$"],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "proj_weight_l2_up_gate": {
        # Fused gate+up GEMM in PapayaSeek packed layout.
        "patterns": [
            r"^weights_l2norm/.*block_sparse_moe\.gmm_upgate\.weight$",
            r"^weights_l2norm/.*block_sparse_moe\.fused_swiglu\.weight$",
        ],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "proj_weight_l2_down": {
        "patterns": [r"^weights_l2norm/.*block_sparse_moe\.gmm_down\.weight$"],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    "proj_weight_l2_shared": {
        "patterns": [
            r"^weights_l2norm/.*shared_experts\.(?:up_gate|down|gate|up)_proj\.weight$",
        ],
        "reduction": "mean",
        "keep_per_tag": True,
    },
    # ---- Per-projection activation amax (proxy for projection saturation) ----
    "proj_act_amax_q": {
        "patterns": [
            r"^activations_amax/.*self_attn\._attn\.q_proj_input\.0$",
            r"^activations_amax/.*self_attn\._attn\.uq_proj_input\.0$",
            r"^activations_amax/.*self_attn\._attn\.dq_proj_input\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "proj_act_amax_o": {
        "patterns": [r"^activations_amax/.*self_attn\._attn\.o_proj_input\.0$"],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "proj_act_amax_router": {
        "patterns": [r"^activations_amax/.*self_attn\._gate\.gate_output\.0$"],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "proj_act_amax_down": {
        "patterns": [r"^activations_amax/.*block_sparse_moe\.gmm_down_output\.0$"],
        "reduction": "max",
        "keep_per_tag": True,
    },
    "proj_act_amax_up_gate": {
        "patterns": [
            r"^activations_amax/.*block_sparse_moe\.fused_swiglu_input\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
    # ---- Router score distribution / topk scores ----
    "router_scores_amax": {
        "patterns": [
            r"^activations_amax/.*self_attn\._gate\.scores_output\.0$",
            r"^activations_amax/.*self_attn\._gate\.topk_scores_output\.0$",
        ],
        "reduction": "max",
        "keep_per_tag": True,
    },
}


# ---------------------------------------------------------------------------
# Output directory helpers — figures/ + data/ split
# ---------------------------------------------------------------------------
def _ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    """Create ``OUT/figures`` and ``OUT/data`` subdirs and return them."""
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, data_dir


def _dump_time_series_table(
    items: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    *,
    step_col: str = "step",
) -> None:
    """Dump a list of ``(label, steps, values)`` series to a CSV.

    Columns: ``step`` + one column per label. Rows: union of every
    step seen, with values interpolated (last-known) at each step so
    the table is dense and immediately readable by an LLM.
    """
    if not items:
        return
    all_steps = sorted({int(st) for _, s, _ in items for st in s})
    if not all_steps:
        return
    headers = [step_col] + [label for label, _, _ in items]
    rows: list[list[Any]] = []
    # Build per-label step-indexed dicts (only points we actually have).
    by_label = [
        (label, {int(st): float(v) for st, v in zip(s, vs)}) for label, s, vs in items
    ]
    for st in all_steps:
        row: list[Any] = [st]
        for _, d in by_label:
            v = d.get(st)
            row.append("" if v is None or not math.isfinite(v) else f"{v:.6g}")
        rows.append(row)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _dump_layer_grid_table(
    layers: list[int],
    steps: np.ndarray,
    grid: np.ndarray,
    out_path: Path,
) -> None:
    """CSV: step + one column per layer. ``grid[i, j]`` = value at
    layer ``layers[i]``, step ``steps[j]``."""
    if grid.size == 0 or not layers:
        return
    headers = ["step"] + [f"L{L}" for L in layers]
    rows: list[list[Any]] = []
    for j, st in enumerate(steps):
        row: list[Any] = [int(st)]
        for i in range(len(layers)):
            v = grid[i, j]
            row.append("" if not np.isfinite(v) else f"{v:.6g}")
        rows.append(row)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Event-file reader — delegates to instability.tb_helpers for parsing,
# then classifies tags into the SIGNAL_GROUPS taxonomy.
# ---------------------------------------------------------------------------
def _combine_across_tags(
    per_tag: dict[str, tuple[np.ndarray, np.ndarray]],
    reduction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Single (steps, values) per group, aggregated across all tags in
    it. Used for headline lines in subplots (e.g. *peak* aux loss
    across all MoE layers).
    """
    all_steps = np.concatenate([s for s, _ in per_tag.values()])
    all_vals = np.concatenate([v for _, v in per_tag.values()])
    order = np.argsort(all_steps, kind="stable")
    all_steps, all_vals = all_steps[order], all_vals[order]
    uniq, idx = np.unique(all_steps, return_inverse=True)
    out = np.zeros(uniq.size, dtype=np.float64)
    if reduction == "max":
        np.maximum.at(out, idx, np.abs(all_vals))
    else:
        cnt = np.zeros(uniq.size, dtype=np.int64)
        np.add.at(out, idx, all_vals)
        np.add.at(cnt, idx, 1)
        out = out / np.maximum(cnt, 1)
    return uniq, out


def read_signals(
    tb_dir: Path,
    *,
    workers: int = 8,
    cache: bool = True,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Parse the TB run into ``{group: {tag_or_'_combined': (steps, values)}}``.

    Implementation notes
    --------------------
    Delegates to :func:`instability.tb_helpers.load_run_scalars` which:
      * batches event files per subprocess (fewer IPC roundtrips),
      * filters tags at parse time via ``str.startswith(tuple)``,
      * caches the parsed result keyed by ``(path, mtime_ns, size)`` of
        every event file under ``~/.cache/instability_tb/``.

    The cross-rank deduplication done there (max-of-abs per step) is
    the same semantic we want here for both ``max`` and ``mean`` group
    reductions — rank-duplicates carry identical values; the per-step
    extremum is the right representative.
    """
    from instability.tb_helpers import load_run_scalars

    # Default (deny-list) load: every model monitor minus non-model noise, so new
    # monitors are aggregated into the report automatically.
    raw_per_tag = load_run_scalars(
        tb_dir,
        max_workers=workers,
        cache=cache,
        verbose=True,
    )
    if not raw_per_tag:
        raise SystemExit(f"no relevant tags in {tb_dir}")

    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for gname, gcfg in SIGNAL_GROUPS.items():
        pats = [re.compile(p) for p in gcfg["patterns"]]
        matched = [t for t in raw_per_tag if any(p.search(t) for p in pats)]
        if not matched:
            continue
        red = gcfg["reduction"]
        per_tag: dict[str, tuple[np.ndarray, np.ndarray]] = {
            tag: raw_per_tag[tag] for tag in matched
        }
        per_tag["_combined"] = _combine_across_tags(per_tag, red)
        if gcfg.get("keep_per_tag"):
            out[gname] = per_tag
        else:
            out[gname] = {"_combined": per_tag["_combined"]}

    # ---- naming-convention fallback ----
    # If a run uses the newer ``moe_maxvio/layer_N`` convention without the
    # legacy ``dispatcher_layer_experts_max_vio_batch_avg/layer_N`` tags,
    # downstream code that looks up ``max_vio_per_layer`` would find nothing.
    # Alias the group so report.png / heatmap / depth-corr / spike-detection
    # all keep working without per-call conditionals.
    if not out.get("max_vio_per_layer") and out.get("moe_maxvio_per_layer"):
        out["max_vio_per_layer"] = out["moe_maxvio_per_layer"]
    # Same idea: if ``max_vio_global`` (legacy dispatcher_sum_metrics) is
    # absent, synthesize it from the per-layer envelope.
    if not out.get("max_vio_global", {}).get("_combined") and out.get(
        "max_vio_per_layer"
    ):
        from_layers = {
            t: v for t, v in out["max_vio_per_layer"].items() if t != "_combined"
        }
        if from_layers:
            steps, mx, _, _ = _layer_envelope(from_layers)
            if steps.size:
                out["max_vio_global"] = {"_combined": (steps, mx)}
    return out


# ---------------------------------------------------------------------------
# Layer-envelope helper (max & median per step across per-layer tags)
# ---------------------------------------------------------------------------
def _layer_envelope(
    per_tag: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(steps, layer_max, layer_median, layer_min)`` across all
    per-layer series, evaluated at every step seen anywhere.

    Vectorized: builds a [layer × step] grid (NaN where the layer has no
    sample) and reduces over the layer axis with nan-safe ops.
    """
    items = [(s, v) for tag, (s, v) in per_tag.items() if tag != "_combined"]
    if not items:
        z = np.zeros(0)
        return z, z, z, z
    all_steps = np.array(
        sorted({int(st) for s, _ in items for st in s}), dtype=np.int64
    )
    if all_steps.size == 0:
        z = np.zeros(0)
        return z, z, z, z
    step_to_col = {int(st): j for j, st in enumerate(all_steps)}
    grid = np.full((len(items), all_steps.size), np.nan, dtype=np.float64)
    for i, (s, vs) in enumerate(items):
        for st, val in zip(s, vs):
            j = step_to_col.get(int(st))
            if j is not None:
                grid[i, j] = float(val)
    mx = np.nanmax(grid, axis=0)
    md = np.nanmedian(grid, axis=0)
    mn = np.nanmin(grid, axis=0)
    return all_steps, mx, md, mn


# ---------------------------------------------------------------------------
# Transient-spike detector — per-step modified z-score
# ---------------------------------------------------------------------------
# Persistent badness is what :mod:`instability.score` catches via its
# sliding-window severity (peak × fraction). But that formula damps a
# *single-step* spike (fraction tiny → severity ~0). Training instability
# often manifests precisely as one-step blowups in grad norm, loss, or
# max_vio, so we run a dedicated detector on every step.
#
# Method — modified z-score (a.k.a. MAD-z, Iglewicz/Hoaglin 1993):
#     z_i = 0.6745 * (x_i - rolling_median) / rolling_MAD
# rolling stats computed over the previous ``baseline_window`` *samples*.
# We compare each new sample against the immediately-preceding window.
# More robust than mean+std (one prior spike doesn't poison the baseline).


@dataclass(frozen=True)
class SpikeEvent:
    signal: str  # which signal group ("loss", "grad_norm_global", ...)
    tag: str  # specific TB tag
    step: int  # step at which the spike occurred
    value: float  # value at that step
    baseline_median: float
    baseline_mad: float  # median absolute deviation (robust σ proxy)
    z: float  # modified z-score of the sample vs window

    def severity(self) -> float:
        """Rough severity ∈ [0, 1] for downstream consumers — saturates
        at |z| = 20 so the very largest spikes still differentiate."""
        return float(min(1.0, abs(self.z) / 20.0))


# Which groups to scan, with per-signal warmup. ``direction`` controls
# whether we care about upward, downward, or both kinds of anomaly:
# for stability signals like loss / grad / max_vio only *up* is bad.
SPIKE_GROUPS: dict[str, dict[str, Any]] = {
    "loss": {"warmup": 100, "z_threshold": 5.0, "per_tag": False, "direction": "up"},
    "grad_norm_global": {
        "warmup": 500,
        "z_threshold": 5.0,
        "per_tag": False,
        "direction": "up",
    },
    "max_vio_global": {
        "warmup": 200,
        "z_threshold": 5.0,
        "per_tag": False,
        "direction": "up",
    },
    "max_vio_per_layer": {
        "warmup": 200,
        "z_threshold": 6.0,
        "per_tag": True,
        "direction": "up",
    },
    "activation_amax": {
        "warmup": 1000,
        "z_threshold": 6.0,
        "per_tag": True,
        "direction": "up",
    },
    "aux_loss": {
        "warmup": 200,
        "z_threshold": 5.0,
        "per_tag": False,
        "direction": "up",
    },
}


def _detect_spikes_one_series(
    signal: str,
    tag: str,
    steps: np.ndarray,
    values: np.ndarray,
    *,
    warmup: int,
    baseline_window: int,
    z_threshold: float,
    direction: str = "up",
) -> list[SpikeEvent]:
    """Modified z-score spike detector applied to a single series.

    Skips data before ``warmup`` and requires at least
    ``baseline_window + 1`` post-warmup samples to fire (so the first
    candidate has a full window of priors).
    """
    if steps.size < baseline_window + 1:
        return []
    mask = (steps >= warmup) & np.isfinite(values)
    if mask.sum() < baseline_window + 1:
        return []
    s, v = steps[mask], values[mask]
    # Rolling median + MAD over the past ``baseline_window`` samples.
    # sliding_window_view returns shape (N-W+1, W) — windows[i] covers
    # samples [i, i+W). We compare v[i+W] (the next sample) to
    # windows[i] (the W priors).
    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except ImportError:  # pragma: no cover — numpy >= 1.20 always has it.
        return []
    if v.size <= baseline_window:
        return []
    windows = sliding_window_view(v, baseline_window)  # (N-W+1, W)
    medians = np.median(windows, axis=1)  # (N-W+1,)
    mads = np.median(np.abs(windows - medians[:, None]), axis=1)
    targets = v[baseline_window:]  # (N-W,)
    ref_med = medians[:-1]
    ref_mad = mads[:-1]
    z = 0.6745 * (targets - ref_med) / np.maximum(ref_mad, 1e-12)
    if direction == "up":
        bad = np.where(z > z_threshold)[0]
    elif direction == "down":
        bad = np.where(z < -z_threshold)[0]
    else:  # both
        bad = np.where(np.abs(z) > z_threshold)[0]
    return [
        SpikeEvent(
            signal=signal,
            tag=tag,
            step=int(s[baseline_window + i]),
            value=float(targets[i]),
            baseline_median=float(ref_med[i]),
            baseline_mad=float(ref_mad[i]),
            z=float(z[i]),
        )
        for i in bad
    ]


def find_spikes(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    baseline_window: int = 100,
    dedup_step_gap: int = 50,
) -> list[SpikeEvent]:
    """Scan all configured signal groups for per-step spikes.

    Returns a list of :class:`SpikeEvent`, sorted by ``(signal, step)``.
    Adjacent spikes in the *same* tag within ``dedup_step_gap`` steps
    are collapsed to the one with the largest ``|z|`` — otherwise a
    sustained plateau registers as dozens of redundant alerts.
    """
    out: list[SpikeEvent] = []
    for gname, gcfg in SPIKE_GROUPS.items():
        group = groups.get(gname)
        if not group:
            continue
        if gcfg["per_tag"]:
            items = [(t, s, v) for t, (s, v) in group.items() if t != "_combined"]
        else:
            cmb = group.get("_combined")
            if cmb is None:
                continue
            items = [("_combined", cmb[0], cmb[1])]
        for tag, s, v in items:
            out.extend(
                _detect_spikes_one_series(
                    gname,
                    tag,
                    s,
                    v,
                    warmup=gcfg["warmup"],
                    baseline_window=baseline_window,
                    z_threshold=gcfg["z_threshold"],
                    direction=gcfg.get("direction", "up"),
                )
            )
    # Dedup nearby alerts per (signal, tag).
    out.sort(key=lambda e: (e.signal, e.tag, e.step))
    deduped: list[SpikeEvent] = []
    for e in out:
        if (
            deduped
            and deduped[-1].signal == e.signal
            and deduped[-1].tag == e.tag
            and e.step - deduped[-1].step <= dedup_step_gap
        ):
            if abs(e.z) > abs(deduped[-1].z):
                deduped[-1] = e
            continue
        deduped.append(e)
    return deduped


def dump_spikes_csv(spikes: list[SpikeEvent], out_path: Path) -> None:
    rows = sorted(spikes, key=lambda e: -abs(e.z))
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["signal", "tag", "step", "value", "baseline_median", "baseline_mad", "z"]
        )
        for e in rows:
            w.writerow(
                [
                    e.signal,
                    e.tag,
                    e.step,
                    f"{e.value:.6g}",
                    f"{e.baseline_median:.6g}",
                    f"{e.baseline_mad:.6g}",
                    f"{e.z:.2f}",
                ]
            )


def print_spikes_summary(spikes: list[SpikeEvent], top_n: int = 12) -> None:
    if not spikes:
        print("\n[spikes] no transient spikes detected (z > thresholds).")
        return
    by_signal: dict[str, int] = defaultdict(int)
    for e in spikes:
        by_signal[e.signal] += 1
    print(f"\n[spikes] {len(spikes)} transient spike(s) detected:")
    for sig, n in sorted(by_signal.items(), key=lambda kv: -kv[1]):
        print(f"    {sig:<24} {n} event(s)")
    top = sorted(spikes, key=lambda e: -abs(e.z))[:top_n]
    print(f"\n  top {len(top)} by |z|:")
    print(
        f"    {'signal':<22} {'step':>7} {'|z|':>7}  {'value':>11}  "
        f"{'baseline':>11}  tag"
    )
    for e in top:
        print(
            f"    {e.signal:<22} {e.step:>7} {abs(e.z):>7.1f}  "
            f"{e.value:>11.4g}  {e.baseline_median:>11.4g}  {e.tag}"
        )


# ---------------------------------------------------------------------------
# Main panel plot (graphs only — no verdict/findings)
# ---------------------------------------------------------------------------
def _annotate_spikes(
    ax: Any,
    spikes: list[SpikeEvent],
    signals: tuple[str, ...],
) -> None:
    """Plot small red downward triangles along the bottom edge of an
    axis at every spike step belonging to one of ``signals``. We
    deduplicate by step (multiple per-tag spikes at the same step → one
    marker)."""
    if not spikes:
        return
    steps = sorted({e.step for e in spikes if e.signal in signals})
    if not steps:
        return
    ylo, yhi = ax.get_ylim()
    y_marker = ylo + (yhi - ylo) * 0.02
    ax.scatter(
        steps,
        [y_marker] * len(steps),
        marker="v",
        color="red",
        s=30,
        alpha=0.9,
        zorder=5,
        label=f"spikes ({len(steps)})",
    )
    # Refresh the legend so the new entry shows up.
    ax.legend(fontsize=8, loc="best")


def plot_report(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    out_path: Path,
    spikes: list[SpikeEvent] | None = None,
) -> None:
    spikes = spikes or []
    fig, axes = plt.subplots(3, 2, figsize=(18, 11), constrained_layout=True)
    (ax_loss, ax_lr), (ax_grad, ax_vio), (ax_act, ax_aux) = axes

    # ---- 1) Loss ----
    loss = groups.get("loss", {})
    plotted = False
    for tag, (s, v) in loss.items():
        if tag == "_combined":
            continue
        label = tag.split("/")[-1]
        ax_loss.plot(s, v, linewidth=1.4, label=label, alpha=0.9)
        plotted = True
    if plotted:
        ax_loss.legend(fontsize=9, loc="best")
    else:
        ax_loss.text(
            0.5,
            0.5,
            "no loss tags",
            ha="center",
            va="center",
            transform=ax_loss.transAxes,
        )
    ax_loss.set_title("Loss", fontsize=12, weight="bold")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(alpha=0.3)
    _annotate_spikes(ax_loss, spikes, ("loss",))

    # ---- 2) LR ----
    lr = groups.get("lr", {}).get("_combined")
    if lr is not None and lr[1].size:
        ax_lr.plot(lr[0], lr[1], linewidth=1.6, color="purple")
        ax_lr.set_yscale("log")
    else:
        ax_lr.text(
            0.5, 0.5, "no LR tags", ha="center", va="center", transform=ax_lr.transAxes
        )
    ax_lr.set_title("LR (group0)", fontsize=12, weight="bold")
    ax_lr.set_xlabel("step")
    ax_lr.set_ylabel("lr")
    ax_lr.grid(alpha=0.3)

    # ---- 3) gradient L2 norm — global + per-layer envelope ----
    gn = groups.get("grad_norm_global", {}).get("_combined")
    if gn is not None and gn[1].size:
        ax_grad.plot(gn[0], gn[1], linewidth=1.6, color="black", label="global")
    gl = groups.get("grad_norm_layers", {})
    if gl:
        s, mx, md, mn = _layer_envelope(gl)
        if s.size:
            ax_grad.plot(
                s, mx, color="red", linewidth=1.1, alpha=0.7, label="layer max"
            )
            ax_grad.plot(
                s, md, color="orange", linewidth=1.1, alpha=0.7, label="layer median"
            )
    ax_grad.set_title("Gradient L2 norm", fontsize=12, weight="bold")
    ax_grad.set_xlabel("step")
    ax_grad.set_ylabel("‖grad‖₂")
    ax_grad.set_yscale("log")
    ax_grad.grid(alpha=0.3, which="both")
    ax_grad.legend(fontsize=9, loc="best")
    _annotate_spikes(ax_grad, spikes, ("grad_norm_global",))

    # ---- 4) max_vio — router load imbalance ----
    mvg = groups.get("max_vio_global", {}).get("_combined")
    if mvg is not None and mvg[1].size:
        ax_vio.plot(
            mvg[0], mvg[1], color="black", linewidth=1.6, label="dispatcher global avg"
        )
    mvl = groups.get("max_vio_per_layer", {})
    if mvl:
        s, mx, md, mn = _layer_envelope(mvl)
        if s.size:
            ax_vio.plot(
                s, mx, color="red", linewidth=1.1, alpha=0.7, label="per-layer max"
            )
            ax_vio.plot(
                s,
                md,
                color="orange",
                linewidth=1.1,
                alpha=0.7,
                label="per-layer median",
            )
    ax_vio.axhline(4.0, color="red", linestyle="--", alpha=0.5, label="warn (>4)")
    ax_vio.set_title("max_vio  (router load imbalance)", fontsize=12, weight="bold")
    ax_vio.set_xlabel("step")
    ax_vio.set_ylabel("max_vio")
    ax_vio.grid(alpha=0.3)
    ax_vio.legend(fontsize=9, loc="best")
    _annotate_spikes(ax_vio, spikes, ("max_vio_global", "max_vio_per_layer"))

    # ---- 5) activation amax — top 6 most volatile residual/output tags ----
    aa = groups.get("activation_amax", {})
    non_combined = {t: v for t, v in aa.items() if t != "_combined"}
    if non_combined:
        ranked = sorted(
            non_combined.items(),
            key=lambda kv: -float(np.nanmax(np.abs(kv[1][1])) if kv[1][1].size else 0),
        )
        for tag, (s, v) in ranked[:6]:
            short = tag.split("model.")[-1].split(".0")[0][:54]
            ax_act.plot(s, np.abs(v), linewidth=1.1, alpha=0.85, label=short)
    ax_act.set_title(
        "Activation amax  (top-6 most volatile)", fontsize=12, weight="bold"
    )
    ax_act.set_xlabel("step")
    ax_act.set_ylabel("|activation|")
    ax_act.set_yscale("log")
    ax_act.grid(alpha=0.3, which="both")
    ax_act.legend(fontsize=7, loc="best", ncol=2)
    _annotate_spikes(ax_act, spikes, ("activation_amax",))

    # ---- 6) aux losses + load balance ----
    aux_lines: list[tuple[str, tuple[np.ndarray, np.ndarray] | None]] = [
        (
            "moe_seq_aux (peak across layers)",
            groups.get("aux_loss", {}).get("_combined"),
        ),
        ("hidden_z_loss (combined)", groups.get("hidden_z_loss", {}).get("_combined")),
        ("load_balance / mean", groups.get("load_balance", {}).get("_combined")),
    ]
    any_plot = False
    for label, item in aux_lines:
        if item is None or item[0].size == 0:
            continue
        ax_aux.plot(item[0], item[1], linewidth=1.3, label=label, alpha=0.85)
        any_plot = True
    if any_plot:
        ax_aux.legend(fontsize=8, loc="best")
    else:
        ax_aux.text(
            0.5,
            0.5,
            "no aux signals",
            ha="center",
            va="center",
            transform=ax_aux.transAxes,
        )
    ax_aux.set_title("Auxiliary losses / balance", fontsize=12, weight="bold")
    ax_aux.set_xlabel("step")
    ax_aux.set_ylabel("value")
    ax_aux.set_yscale("log")
    ax_aux.grid(alpha=0.3, which="both")
    _annotate_spikes(ax_aux, spikes, ("aux_loss",))

    fig.suptitle(f"stability report — {run_name}", fontsize=14, weight="bold")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# max_vio per-layer heatmap
# ---------------------------------------------------------------------------
_LAYER_INT_RE = re.compile(r"layer_(\d+)$")


def plot_max_vio_heatmap(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    out_path: Path,
) -> None:
    """``[layer × step]`` heatmap of dispatcher max_vio per layer.

    Lets you attribute a global peak to a specific layer (or set of
    layers) and a specific step range. The warning level (``> 4``) is
    overlaid as a contour for quick visual scanning.
    """
    from matplotlib.colors import Normalize

    mvl = groups.get("max_vio_per_layer", {})
    layered = [
        (int(m.group(1)), s, v)
        for tag, (s, v) in mvl.items()
        if tag != "_combined" and (m := _LAYER_INT_RE.search(tag))
    ]
    if not layered:
        print("[heatmap] no per-layer max_vio tags found — skipping", flush=True)
        return
    layered.sort(key=lambda t: t[0])
    layer_indices = [li for li, _, _ in layered]

    # Common step grid — union of every layer's steps.
    all_steps = sorted({int(st) for _, s, _ in layered for st in s})
    step_to_col = {st: i for i, st in enumerate(all_steps)}
    grid = np.full((len(layered), len(all_steps)), np.nan, dtype=np.float64)
    for row_i, (_, s, v) in enumerate(layered):
        for st, val in zip(s, v):
            j = step_to_col.get(int(st))
            if j is not None:
                grid[row_i, j] = float(val)

    # Plot. Two panels:
    #   (a) raw heatmap with vmin/vmax robust to outliers (1%..99%),
    #   (b) thresholded mask "value > 4" for instant scanning.
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        print("[heatmap] grid empty — skipping", flush=True)
        return
    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())

    fig, (ax_raw, ax_thr) = plt.subplots(
        2,
        1,
        figsize=(16, max(5.0, 0.18 * len(layered) * 2 + 2.5)),
        sharex=True,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#cccccc")

    im = ax_raw.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        extent=[all_steps[0], all_steps[-1], len(layered) - 0.5, -0.5],
    )
    yt_step = max(1, len(layered) // 24)
    ax_raw.set_yticks(range(0, len(layered), yt_step))
    ax_raw.set_yticklabels(
        [f"L{layer_indices[i]}" for i in range(0, len(layered), yt_step)], fontsize=8
    )
    ax_raw.set_ylabel("layer")
    ax_raw.set_title(
        f"max_vio per layer × step  —  raw  (vmin/vmax = 1%/99% pct: "
        f"{vmin:.2f}/{vmax:.2f})",
        fontsize=12,
        weight="bold",
    )
    fig.colorbar(im, ax=ax_raw, label="max_vio")

    # Thresholded view
    thr_grid = np.where(np.isfinite(grid), (grid > 4.0).astype(float), np.nan)
    cmap2 = plt.get_cmap("Reds").copy()
    cmap2.set_bad(color="#dddddd")
    ax_thr.imshow(
        np.ma.masked_invalid(thr_grid),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap2,
        vmin=0,
        vmax=1,
        extent=[all_steps[0], all_steps[-1], len(layered) - 0.5, -0.5],
    )
    ax_thr.set_yticks(range(0, len(layered), yt_step))
    ax_thr.set_yticklabels(
        [f"L{layer_indices[i]}" for i in range(0, len(layered), yt_step)], fontsize=8
    )
    ax_thr.set_ylabel("layer")
    ax_thr.set_xlabel("step")
    ax_thr.set_title("mask:  max_vio > 4   (red = bad)", fontsize=12, weight="bold")

    fig.suptitle(f"max_vio attribution — {run_name}", fontsize=14, weight="bold")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-layer attribution — depth-resolved heatmaps + ranking CSV
# ---------------------------------------------------------------------------
# When the global signals all look OK but ``max_vio_heatmap`` shows depth
# structure ("late layers degrade more"), this is what attributes the
# pattern to a specific failure mode. Stacks 4-6 ``[layer × step]``
# heatmaps and a ``layer_profile.csv`` ranking layers by each metric.

# Each layer-attribution row: (group_name, display_title, log_color_scale,
# alert_threshold_for_mask_or_None, cmap).
_LAYER_PANELS: list[tuple[str, str, bool, float | None, str]] = [
    ("max_vio_per_layer", "max_vio (router imbalance)", False, 4.0, "viridis"),
    ("layer_residual_amax", "residual stream amax (mlp output)", True, None, "magma"),
    ("layer_attn_amax", "attn residual amax", True, None, "magma"),
    (
        "layer_attn_residual_contribution",
        "attn → residual contribution amax",
        True,
        None,
        "magma",
    ),
    (
        "layer_mlp_residual_contribution",
        "mlp → residual contribution amax",
        True,
        None,
        "magma",
    ),
    ("layer_clip_share_gate", "clip share — mlp_gate", False, 0.01, "Reds"),
    ("layer_clip_share_up", "clip share — mlp_up", False, 0.01, "Reds"),
    ("aux_loss", "moe_seq_aux_loss", True, None, "plasma"),
    ("grad_norm_layers", "‖grad‖₂ per layer (FQN flatten)", True, None, "viridis"),
]


def _layer_grid(
    per_tag: dict[str, tuple[np.ndarray, np.ndarray]],
    layer_re: re.Pattern,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Build ``[layer × step]`` grid from per-tag series.

    Returns ``(layer_indices, step_array, grid)``. ``grid[i, j]`` is
    the metric value at ``layer_indices[i]`` on ``step_array[j]``;
    NaN where the tag had no value at that step.
    """
    layered: list[tuple[int, np.ndarray, np.ndarray]] = []
    for tag, (s, v) in per_tag.items():
        if tag == "_combined":
            continue
        m = layer_re.search(tag)
        if not m:
            continue
        layer = int(next(filter(None, m.groups())))
        layered.append((layer, s, v))
    if not layered:
        return [], np.zeros(0), np.zeros((0, 0))
    layered.sort(key=lambda t: t[0])
    layers = [li for li, _, _ in layered]
    all_steps = sorted({int(st) for _, s, _ in layered for st in s})
    step_to_col = {st: i for i, st in enumerate(all_steps)}
    grid = np.full((len(layered), len(all_steps)), np.nan, dtype=np.float64)
    for row, (_, s, v) in enumerate(layered):
        for st, val in zip(s, v):
            j = step_to_col.get(int(st))
            if j is not None:
                grid[row, j] = float(val)
    return layers, np.array(all_steps), grid


def plot_layer_attribution(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    out_path: Path,
) -> dict[str, Any]:
    """Multi-panel per-layer heatmap. Returns metadata about which
    panels were plotted (used by the profile CSV)."""
    from matplotlib.colors import Normalize, LogNorm

    layer_re = re.compile(r"layer_(\d+)(?:$|/)|\.layers\.(\d+)\.")

    # Pre-build grids for each panel we plan to show; drop empties.
    built: list[
        tuple[str, str, bool, float | None, str, list[int], np.ndarray, np.ndarray]
    ] = []
    for gname, title, log_c, thr, cmap in _LAYER_PANELS:
        g = groups.get(gname)
        if not g:
            continue
        layers, steps, grid = _layer_grid(g, layer_re)
        if grid.size == 0 or not np.isfinite(grid).any():
            continue
        built.append((gname, title, log_c, thr, cmap, layers, steps, grid))

    if not built:
        print("[layer-attribution] no per-layer signals to plot", flush=True)
        return {}

    n = len(built)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(16, max(2.4 * n, 3.5)),
        sharex=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    panel_summaries: dict[str, Any] = {}
    for ax, (gname, title, log_c, thr, cmap_name, layers, steps, grid) in zip(
        axes, built
    ):
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            ax.axis("off")
            continue
        lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="#cccccc")
        if log_c and lo > 0:
            norm = LogNorm(vmin=max(lo, 1e-30), vmax=max(hi, lo * 10))
        else:
            norm = Normalize(vmin=lo, vmax=hi)
        im = ax.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[int(steps[0]), int(steps[-1]), len(layers) - 0.5, -0.5],
        )
        yt_step = max(1, len(layers) // 16)
        ax.set_yticks(range(0, len(layers), yt_step))
        ax.set_yticklabels(
            [f"L{layers[i]}" for i in range(0, len(layers), yt_step)], fontsize=7
        )
        ax.set_ylabel("layer", fontsize=8)
        title_suffix = " (log color)" if (log_c and lo > 0) else ""
        ax.set_title(title + title_suffix, fontsize=10, weight="bold", loc="left")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

        # Per-layer summaries used by layer_profile.csv:
        mean_per_layer = np.nanmean(grid, axis=1)
        tail_n = max(5, grid.shape[1] // 20)
        tail = grid[:, -tail_n:]
        tail_mean_per_layer = np.nanmean(tail, axis=1)
        panel_summaries[gname] = {
            "layers": layers,
            "mean_per_layer": [
                None if not np.isfinite(x) else float(x) for x in mean_per_layer
            ],
            "tail_mean_per_layer": [
                None if not np.isfinite(x) else float(x) for x in tail_mean_per_layer
            ],
            "threshold": thr,
        }
    axes[-1].set_xlabel("step")
    fig.suptitle(f"per-layer attribution — {run_name}", fontsize=13, weight="bold")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return panel_summaries


def dump_layer_profile_csv(
    panel_summaries: dict[str, Any],
    out_path: Path,
) -> None:
    """Per-layer ranking across all attribution signals.

    Columns: layer + (mean_<signal>, tail_mean_<signal>) for every panel.
    Also adds ``depth_correlation_<signal>`` — Pearson r between the
    metric and layer index. Positive r → metric grows with depth.
    """
    if not panel_summaries:
        return
    # Union of layer indices across panels (might differ by signal availability).
    all_layers = sorted({L for s in panel_summaries.values() for L in s["layers"]})
    cols = ["layer"]
    for gname in panel_summaries:
        cols += [f"mean_{gname}", f"tail_mean_{gname}"]
    # Depth correlation (single scalar per signal, repeated on every row
    # for easy filtering; trivial overhead).
    depth_corr: dict[str, float] = {}
    for gname, s in panel_summaries.items():
        # Pair (layer, mean) before filtering — otherwise None gaps shift
        # the layer indices and the correlation runs on the wrong pairs.
        pairs = [
            (L, m) for L, m in zip(s["layers"], s["mean_per_layer"]) if m is not None
        ]
        if not pairs:
            depth_corr[gname] = float("nan")
            continue
        idx = np.array([L for L, _ in pairs], dtype=float)
        means = np.array([m for _, m in pairs], dtype=float)
        if means.size >= 3 and np.std(means) > 1e-12:
            depth_corr[gname] = float(np.corrcoef(idx.astype(float), means)[0, 1])
        else:
            depth_corr[gname] = float("nan")
    cols += [f"depth_corr_{g}" for g in panel_summaries]

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for L in all_layers:
            row: list[Any] = [L]
            for gname, s in panel_summaries.items():
                if L in s["layers"]:
                    i = s["layers"].index(L)
                    m = s["mean_per_layer"][i]
                    t = s["tail_mean_per_layer"][i]
                    row += [
                        ("" if m is None else f"{m:.6g}"),
                        ("" if t is None else f"{t:.6g}"),
                    ]
                else:
                    row += ["", ""]
            for gname in panel_summaries:
                v = depth_corr.get(gname, float("nan"))
                row.append("" if not np.isfinite(v) else f"{v:+.3f}")
            w.writerow(row)


# ---------------------------------------------------------------------------
# Multi-panel time-series helper (used by several focused PNGs)
# ---------------------------------------------------------------------------
def _line_panel(
    ax: Any,
    items: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
    ylabel: str,
    *,
    log_y: bool = False,
    alert_y: float | None = None,
    alert_label: str = "alert",
) -> None:
    """Plot one panel with ``len(items)`` lines and optional alert line."""
    any_data = False
    for label, s, v in items:
        if s.size == 0:
            continue
        ax.plot(s, v, linewidth=1.2, alpha=0.85, label=label)
        any_data = True
    ax.set_title(title, fontsize=10, weight="bold", loc="left")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3, which="both" if log_y else "major")
    if log_y:
        ax.set_yscale("log")
    if alert_y is not None and any_data:
        ax.axhline(alert_y, color="red", linestyle="--", alpha=0.5, label=alert_label)
    if any_data:
        ax.legend(fontsize=7, loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)


def _short_tag(t: str, prefix: str | None = None) -> str:
    """Compact label for legends."""
    if prefix and t.startswith(prefix):
        return t[len(prefix) :]
    if "/" in t:
        return t.split("/")[-1]
    return t[:30]


# ---------------------------------------------------------------------------
# MoE health — dedicated focus PNG
# ---------------------------------------------------------------------------
def plot_moe_health(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    out_path: Path,
) -> None:
    """8-panel dedicated MoE diagnostics. Combines global time series
    and per-layer aggregates for max_vio variants, expert utilization,
    overloaded ratios, drop / empty fractions, bias statistics."""
    fig, axes = plt.subplots(4, 2, figsize=(17, 13), constrained_layout=True)

    # Panel 1: max_vio variants (global)
    items: list[tuple[str, np.ndarray, np.ndarray]] = []
    for tag in (
        "dispatcher_sum_metrics/experts_max_vio_batch_avg",
        "dispatcher_sum_metrics/experts_max_vio_batch_no_pad_avg",
        "dispatcher_sum_metrics/experts_max_vio_microbatch_avg",
    ):
        g = groups.get("dispatcher_globals", {}).get(tag)
        if g is not None:
            items.append((tag.split("/")[-1], g[0], g[1]))
    # Also fall back to old "max_vio_global" if dispatcher_globals doesn't carry it.
    mvg = groups.get("max_vio_global", {}).get("_combined")
    if not items and mvg is not None:
        items.append(("global_avg", mvg[0], mvg[1]))
    _line_panel(
        axes[0, 0],
        items,
        "max_vio (global, all variants)",
        "max_vio",
        alert_y=4.0,
        alert_label="warn (>4)",
    )

    # Panel 2: dispatcher per-step health (drop / empty / overloaded / step_max_load)
    g = groups.get("dispatcher_globals", {})
    items = []
    for tag in (
        "dispatcher_sum_metrics/token_drop_level",
        "dispatcher_sum_metrics/overloaded_expert_ratio",
        "dispatcher_sum_metrics/empty_token_level",
        "dispatcher_sum_metrics/step_max_expert_load",
        "dispatcher_sum_metrics/num_drop_steps",
    ):
        d = g.get(tag)
        if d is not None:
            items.append((tag.split("/")[-1], d[0], d[1]))
    _line_panel(axes[0, 1], items, "dispatcher per-step counters", "value", log_y=False)

    # Panel 3: expert utilization mean (global) + per-layer envelope
    eu = groups.get("moe_expert_utilization_mean", {}).get("_combined")
    eu_pl = groups.get("moe_expert_utilization_per_layer", {})
    items = []
    if eu is not None:
        items.append(("global mean", eu[0], eu[1]))
    if eu_pl:
        s, mx, md, mn = _layer_envelope(eu_pl)
        if s.size:
            items.append(("per-layer max", s, mx))
            items.append(("per-layer median", s, md))
    _line_panel(axes[1, 0], items, "expert utilization", "utilization", log_y=False)

    # Panel 4: load balance global + per-layer envelope
    lb = groups.get("load_balance", {}).get("_combined")
    lb_pl = groups.get("moe_load_balance_per_layer", {})
    items = []
    if lb is not None:
        items.append(("global mean", lb[0], lb[1]))
    if lb_pl:
        s, mx, md, mn = _layer_envelope(lb_pl)
        if s.size:
            items.append(("per-layer max", s, mx))
            items.append(("per-layer median", s, md))
    _line_panel(axes[1, 1], items, "load balance", "balance", log_y=False)

    # Panel 5: aux_loss + z_loss per-layer
    aux_pl = groups.get("aux_loss", {})
    items = []
    if aux_pl:
        s, mx, md, mn = _layer_envelope(aux_pl)
        if s.size:
            items.append(("aux: layer max", s, mx))
            items.append(("aux: layer median", s, md))
    _line_panel(
        axes[2, 0],
        items,
        "moe_seq_aux_loss (per-layer envelope)",
        "aux loss",
        log_y=True,
    )

    # Panel 6: overloaded ratio per-layer envelope
    over_pl = groups.get("dispatcher_overloaded_per_layer", {})
    items = []
    if over_pl:
        s, mx, md, mn = _layer_envelope(over_pl)
        if s.size:
            items.append(("over: layer max", s, mx))
            items.append(("over: layer median", s, md))
    over_frac_pl = groups.get("moe_over_frac_per_layer", {})
    if over_frac_pl:
        s, mx, md, mn = _layer_envelope(over_frac_pl)
        if s.size:
            items.append(("over_frac max", s, mx))
    _line_panel(
        axes[2, 1],
        items,
        "overloaded experts (per-layer envelope)",
        "fraction",
        log_y=False,
    )

    # Panel 7: bias stats per-layer — max & std envelopes
    bias_pl = groups.get("moe_bias_stats_per_layer", {})
    items = []
    if bias_pl:
        # Split by stat type via tag suffix
        for stat in ("max", "min", "mean", "std"):
            stat_items = {
                tag: (s, v)
                for tag, (s, v) in bias_pl.items()
                if tag != "_combined" and tag.startswith(f"moe_bias_{stat}/")
            }
            if not stat_items:
                continue
            s, mx, md, mn = _layer_envelope(stat_items)
            if s.size:
                items.append((f"bias {stat}: layer max", s, mx))
    _line_panel(
        axes[3, 0],
        items,
        "router bias stats (per-layer envelope)",
        "bias value",
        log_y=False,
    )

    # Panel 8: expert activation max_median / norm_per_token
    ea_pl = groups.get("expert_activation_per_layer", {})
    items = []
    if ea_pl:
        ratio_items = {
            tag: (s, v)
            for tag, (s, v) in ea_pl.items()
            if tag != "_combined" and "max_median_ratio" in tag
        }
        if ratio_items:
            s, mx, md, mn = _layer_envelope(ratio_items)
            if s.size:
                items.append(("max_median_ratio: layer max", s, mx))
                items.append(("max_median_ratio: layer median", s, md))
        norm_items = {
            tag: (s, v)
            for tag, (s, v) in ea_pl.items()
            if tag != "_combined" and "mean_norm_per_token" in tag
        }
        if norm_items:
            s, mx, md, mn = _layer_envelope(norm_items)
            if s.size:
                items.append(("mean_norm_per_token: layer max", s, mx))
    _line_panel(
        axes[3, 1], items, "expert-activation distribution proxies", "value", log_y=True
    )

    fig.suptitle(f"MoE health — {run_name}", fontsize=14, weight="bold")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Activations / weights / gradients depth grids
# ---------------------------------------------------------------------------
def _stacked_heatmap(
    fig: Any,
    axes: list[Any],
    panels: list[tuple[str, str, bool, dict[str, tuple[np.ndarray, np.ndarray]]]],
    layer_re: re.Pattern,
) -> None:
    """Render ``[layer × step]`` heatmaps for each non-empty panel.
    panels = list of ``(group_name, display_title, log_color, per_tag_dict)``."""
    from matplotlib.colors import Normalize, LogNorm

    idx = 0
    for gname, title, log_c, per_tag in panels:
        if not per_tag or idx >= len(axes):
            continue
        layers, steps, grid = _layer_grid(per_tag, layer_re)
        if grid.size == 0 or not np.isfinite(grid).any():
            continue
        ax = axes[idx]
        finite = grid[np.isfinite(grid)]
        lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#cccccc")
        if log_c and lo > 0:
            norm = LogNorm(vmin=max(lo, 1e-30), vmax=max(hi, lo * 10))
            cmap = plt.get_cmap("magma").copy()
            cmap.set_bad(color="#cccccc")
        else:
            norm = Normalize(vmin=lo, vmax=hi)
        im = ax.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[int(steps[0]), int(steps[-1]), len(layers) - 0.5, -0.5],
        )
        yt_step = max(1, len(layers) // 16)
        ax.set_yticks(range(0, len(layers), yt_step))
        ax.set_yticklabels(
            [f"L{layers[i]}" for i in range(0, len(layers), yt_step)], fontsize=7
        )
        ax.set_ylabel("layer", fontsize=8)
        ax.set_title(
            title + (" (log)" if (log_c and lo > 0) else ""),
            fontsize=10,
            weight="bold",
            loc="left",
        )
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        idx += 1
    # Hide unused axes.
    for ax in axes[idx:]:
        ax.axis("off")
    if axes:
        # X-label only on the last visible.
        for ax in reversed(axes[:idx]):
            ax.set_xlabel("step")
            break


_DEPTH_PLOT_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _plot_depth_panels(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    candidates: list[tuple[str, str, bool]],
    run_name: str,
    suptitle: str,
    out_path: Path,
) -> None:
    """Stack per-layer heatmap panels and save to ``out_path``.

    ``candidates`` is a list of ``(group_name, panel_title, log_color)``;
    panels with no data in ``groups`` are skipped silently.
    """
    panels = [
        (name, title, log_c, groups.get(name, {})) for name, title, log_c in candidates
    ]
    panels = [(n, t, lc, d) for n, t, lc, d in panels if d]
    if not panels:
        return
    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, figsize=(16, max(2.4 * n, 4.0)), sharex=True, constrained_layout=True
    )
    axes_list = list(axes) if n > 1 else [axes]
    _stacked_heatmap(fig, axes_list, panels, _DEPTH_PLOT_LAYER_RE)
    fig.suptitle(f"{suptitle} — {run_name}", fontsize=13, weight="bold")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_activations_depth(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    out_path: Path,
) -> None:
    """Activation depth panels: residual stream amax/amean/l2norm,
    attention output amax, clip shares (gate/up)."""
    _plot_depth_panels(
        groups,
        [
            ("layer_residual_amax", "residual_amax (mlp output)", True),
            ("layer_residual_amean", "residual_amean", True),
            ("layer_residual_l2norm", "residual_l2norm", True),
            ("layer_attn_amax", "attn_amax", True),
            ("layer_clip_share_gate", "clip_share — mlp_gate", False),
            ("layer_clip_share_up", "clip_share — mlp_up", False),
        ],
        run_name,
        "Activation depth profile",
        out_path,
    )


def plot_gradients_depth(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    out_path: Path,
) -> None:
    """Gradient depth panels: FQN flat-param grad L2 + activation
    gradients into residual stream + attention o-proj backprop."""
    _plot_depth_panels(
        groups,
        [
            ("grad_norm_layers", "‖grad‖₂ (FQN flatten)", True),
            ("layer_residual_grad_amax", "activation grad amax (residual)", True),
            ("layer_residual_grad_l2norm", "activation grad l2norm (residual)", True),
            ("layer_attn_o_grad_amax", "activation grad amax (attn.o_proj_out)", True),
        ],
        run_name,
        "Gradient depth profile",
        out_path,
    )


def plot_weights_depth(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    out_path: Path,
) -> None:
    """Weights depth panels: post_attn_layernorm L2, o_proj L2 + amax."""
    _plot_depth_panels(
        groups,
        [
            ("layer_norm_weight_l2", "post_attention_layernorm.weight l2", True),
            ("layer_o_proj_weight_l2", "self_attn.o_proj.weight l2", True),
            ("layer_o_proj_weight_amax", "self_attn.o_proj.weight amax", True),
        ],
        run_name,
        "Weights depth profile",
        out_path,
    )


# ---------------------------------------------------------------------------
# System health (memory / throughput / time)
# ---------------------------------------------------------------------------
def plot_system_health(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 9.5), constrained_layout=True)

    # Memory peak / current (PapayaSeek reports already in GB).
    mp = groups.get("memory_peak_allocated", {}).get("_combined")
    mr = groups.get("memory_peak_reserved", {}).get("_combined")
    items = []
    if mp is not None:
        items.append(("peak_allocated", mp[0], mp[1]))
    if mr is not None:
        items.append(("peak_reserved", mr[0], mr[1]))
    _line_panel(axes[0, 0], items, "GPU memory", "GB")

    # Alloc retries (CRITICAL signal — > 0 means memory pressure / fragmentation)
    ar = groups.get("memory_alloc_retries", {}).get("_combined")
    items = []
    if ar is not None:
        items.append(("alloc_retries", ar[0], ar[1]))
    _line_panel(
        axes[0, 1],
        items,
        "memory alloc_retries (>0 = memory pressure)",
        "count",
        alert_y=1.0,
        alert_label="alert (>0)",
    )

    # Throughput
    th = groups.get("throughput_tokens", {}).get("_combined")
    th_dev = groups.get("throughput_device_tokens", {}).get("_combined")
    items = []
    if th is not None:
        items.append(("global tokens/sec", th[0], th[1]))
    if th_dev is not None:
        items.append(("per-device tokens/sec", th_dev[0], th_dev[1]))
    _line_panel(axes[1, 0], items, "throughput", "tokens / sec")

    # Time per batch
    tb = groups.get("time_per_batch", {}).get("_combined")
    items = []
    if tb is not None:
        items.append(("batch_total", tb[0], tb[1]))
    _line_panel(axes[1, 1], items, "time per batch", "seconds")

    # Time breakdown
    tf = groups.get("time_forward", {}).get("_combined")
    tbw = groups.get("time_backward", {}).get("_combined")
    td = groups.get("time_dataloader", {}).get("_combined")
    items = []
    if tf is not None:
        items.append(("forward", tf[0], tf[1]))
    if tbw is not None:
        items.append(("backward", tbw[0], tbw[1]))
    if td is not None:
        items.append(("dataloader_fetch", td[0], td[1]))
    _line_panel(axes[2, 0], items, "time breakdown (per batch)", "seconds")

    # Seq length / batch size
    sl = groups.get("seq_length", {}).get("_combined")
    bs = groups.get("batch_size_curr", {}).get("_combined")
    items = []
    if sl is not None:
        items.append(("seq_len", sl[0], sl[1]))
    if bs is not None:
        items.append(("batch_size", bs[0], bs[1]))
    _line_panel(
        axes[2, 1], items, "curriculum (seq / batch warmup)", "tokens / samples"
    )

    fig.suptitle(f"System health — {run_name}", fontsize=13, weight="bold")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary.md — depth correlations + findings
# ---------------------------------------------------------------------------
def write_summary_md(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    spikes: list[SpikeEvent],
    run_name: str,
    out_path: Path,
) -> None:
    """Markdown synthesis of the most informative numbers across all
    signals. This is what the user reads in 30 seconds to know the
    state of the run.
    """
    layer_re = re.compile(r"layer_(\d+)(?:$|/)|\.layers\.(\d+)\.")
    lines: list[str] = [
        f"# stability summary — {run_name}\n",
        "Auto-generated by `instability.report`. Numbers below are computed from "
        "the raw aggregated series in `signals.json`.\n",
    ]

    # ---- High-level snapshot ----
    last_step = 0
    nan_inf_total = 0
    for per_tag in groups.values():
        for s, v in per_tag.values():
            if s.size:
                last_step = max(last_step, int(s[-1]))
            nan_inf_total += int(np.sum(~np.isfinite(v)))
    lines += [
        "\n## Snapshot\n",
        f"* **Last step**: {last_step}",
        f"* **NaN/Inf total**: {nan_inf_total}",
        f"* **Transient spikes (z-score)**: {len(spikes)}",
    ]

    # ---- Loss summary ----
    losses = groups.get("loss", {})
    if losses:
        lines.append("\n## Loss\n")
        for tag in sorted(losses):
            if tag == "_combined":
                continue
            s, v = losses[tag]
            if v.size == 0:
                continue
            finite = v[np.isfinite(v)]
            if finite.size == 0:
                continue
            short = tag.split("/")[-1]
            tail = v[-min(200, v.size) :]
            tail_mean = float(np.nanmean(tail[np.isfinite(tail)]))
            lines.append(
                f"* `{short:<18}` start={float(v[0]):.4g}  "
                f"last={float(v[-1]):.4g}  "
                f"min={float(finite.min()):.4g}  "
                f"tail_mean={tail_mean:.4g}"
            )

    # ---- Loss components (mtp, hidden_z_loss breakdown) ----
    lc = groups.get("loss_components", {})
    if lc:
        printed_any = False
        for tag in sorted(lc):
            if tag == "_combined":
                continue
            s, v = lc[tag]
            if v.size == 0:
                continue
            if not printed_any:
                lines.append("\n## Loss components\n")
                printed_any = True
            short = tag.split("/")[-1]
            finite = v[np.isfinite(v)]
            lines.append(
                f"* `{short:<22}` start={float(v[0]):.4g}  "
                f"last={float(v[-1]):.4g}  "
                f"range=[{float(finite.min()):.3g}, {float(finite.max()):.3g}]"
            )

    # ---- MoE summary ----
    lines.append("\n## MoE router health\n")
    g = groups.get("dispatcher_globals", {})
    for tag in (
        "dispatcher_sum_metrics/token_drop_level",
        "dispatcher_sum_metrics/overloaded_expert_ratio",
        "dispatcher_sum_metrics/empty_token_level",
        "dispatcher_sum_metrics/step_max_expert_load",
        "dispatcher_sum_metrics/num_drop_steps",
    ):
        d = g.get(tag)
        if d is None:
            continue
        s, v = d
        if v.size == 0:
            continue
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            continue
        short = tag.split("/")[-1]
        lines.append(
            f"* `{short:<28}` max={float(finite.max()):.4g}  "
            f"mean={float(finite.mean()):.4g}  "
            f"last={float(v[-1]):.4g}"
        )
    # Expert utilization
    eu = groups.get("moe_expert_utilization_mean", {}).get("_combined")
    if eu is not None and eu[1].size:
        v = eu[1]
        lines.append(
            f"* `expert_utilization_mean    ` "
            f"min={float(np.nanmin(v)):.3f}  "
            f"last={float(v[-1]):.3f}"
        )

    # ---- Depth correlations (most informative cross-signal) ----
    def _depth_corr(per_tag: dict[str, tuple[np.ndarray, np.ndarray]]) -> float | None:
        # Build per-layer mean across the run, then Pearson r with layer index.
        per_layer: dict[int, float] = {}
        for tag, (s, v) in per_tag.items():
            if tag == "_combined":
                continue
            m = layer_re.search(tag)
            if not m:
                continue
            L = int(next(filter(None, m.groups())))
            finite = v[np.isfinite(v)]
            if finite.size == 0:
                continue
            per_layer[L] = float(np.mean(finite))
        if len(per_layer) < 3:
            return None
        idx = np.array(sorted(per_layer))
        vals = np.array([per_layer[L] for L in idx])
        if np.std(vals) < 1e-12:
            return None
        return float(np.corrcoef(idx.astype(float), vals)[0, 1])

    lines.append("\n## Depth correlations (Pearson r between metric and layer index)\n")
    lines.append("| Signal | r | Interpretation |")
    lines.append("|---|---:|---|")
    depth_groups = [
        ("max_vio_per_layer", "max_vio"),
        ("moe_maxvio_per_layer", "moe_maxvio (alt)"),
        ("moe_load_balance_per_layer", "load_balance"),
        ("moe_expert_utilization_per_layer", "expert_utilization"),
        ("dispatcher_overloaded_per_layer", "overloaded_ratio"),
        ("aux_loss", "aux_loss"),
        ("layer_residual_amax", "residual_amax"),
        ("layer_residual_amean", "residual_amean"),
        ("layer_attn_amax", "attn_amax"),
        ("layer_clip_share_gate", "clip_share_gate"),
        ("layer_clip_share_up", "clip_share_up"),
        ("grad_norm_layers", "grad_norm"),
        ("layer_residual_grad_amax", "residual_grad_amax"),
        ("layer_residual_grad_l2norm", "residual_grad_l2norm"),
        ("layer_o_proj_weight_l2", "o_proj_weight_l2"),
        ("layer_norm_weight_l2", "norm_weight_l2"),
        ("expert_activation_per_layer", "expert_activation_ratio"),
    ]
    for gname, label in depth_groups:
        per_tag = groups.get(gname)
        if not per_tag:
            continue
        r = _depth_corr(per_tag)
        if r is None:
            continue
        direction = "**grows** with depth" if r > 0 else "**falls** with depth"
        if abs(r) >= 0.5:
            interp = f"strong, {direction}"
        elif abs(r) >= 0.25:
            interp = f"moderate, {direction}"
        else:
            interp = "weak / noise"
        lines.append(f"| `{label}` | {r:+.3f} | {interp} |")

    # ---- Top spikes ----
    if spikes:
        lines.append("\n## Top transient spikes (by |z|)\n")
        lines.append("| step | signal | tag | value | baseline | |z| |")
        lines.append("|---:|---|---|---:|---:|---:|")
        for e in sorted(spikes, key=lambda e: -abs(e.z))[:10]:
            lines.append(
                f"| {e.step} | `{e.signal}` | `{e.tag.split('/')[-1]}` | "
                f"{e.value:.4g} | {e.baseline_median:.4g} | {abs(e.z):.1f} |"
            )

    # ---- Memory / throughput health ----
    lines.append("\n## System health\n")
    ar = groups.get("memory_alloc_retries", {}).get("_combined")
    if ar is not None and ar[1].size:
        v = ar[1]
        ar_max = int(np.nanmax(v))
        lines.append(
            f"* `memory.alloc_retries` max={ar_max} "
            f"{'**(warning: GPU memory pressure)**' if ar_max > 0 else '(no pressure)'}"
        )
    mp = groups.get("memory_peak_allocated", {}).get("_combined")
    if mp is not None and mp[1].size:
        v = mp[1]
        # Values are already in GB on PapayaSeek TBs (heuristic: < 1e6 means GB,
        # else bytes).
        scale = 1.0 if float(np.nanmax(v)) < 1e6 else 1e9
        unit = "GB" if scale == 1.0 else "GB (scaled from bytes)"
        lines.append(
            f"* `memory.peak_allocated` last={float(v[-1]) / scale:.1f} {unit}  "
            f"max={float(np.nanmax(v)) / scale:.1f} {unit}"
        )
    th = groups.get("throughput_tokens", {}).get("_combined")
    if th is not None and th[1].size > 10:
        v = th[1]
        mid = v[len(v) // 2]
        last = v[-1]
        drift = (last - mid) / max(mid, 1e-9)
        sign = "↑" if drift > 0 else "↓"
        lines.append(
            f"* `throughput.tokens_per_sec` mid={float(mid):.0f} last={float(last):.0f} "
            f"({sign}{abs(drift) * 100:.1f}% from mid-training)"
        )

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Per-projection depth profile (Q / KV / O / Router / Gate-Up / Down / Shared)
# ---------------------------------------------------------------------------
_PROJ_W_PANELS: list[tuple[str, str]] = [
    ("proj_weight_l2_q", "Q  (attention.q_proj.weight L2)"),
    ("proj_weight_l2_kv", "KV (attention.k/v_proj.weight L2)"),
    ("proj_weight_l2_o", "O  (attention.o_proj.weight L2)"),
    ("proj_weight_l2_router", "Router (gate.weight L2)"),
    ("proj_weight_l2_up_gate", "MoE  Gate+Up fused (gmm_upgate L2)"),
    ("proj_weight_l2_down", "MoE  Down (gmm_down L2)"),
    ("proj_weight_l2_shared", "Shared experts (L2)"),
]


def plot_per_projection_depth(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    png_path: Path,
    csv_path: Path,
) -> None:
    """``[layer × step]`` stacked heatmaps, one per projection family.

    Lets you see *which* projection is drifting and *when* — useful
    when global grad norm looks fine but a specific projection class
    might be developing pathology.
    """
    layer_re = re.compile(r"\.layers\.(\d+)\.")
    panels: list[tuple[str, str, bool, dict[str, tuple[np.ndarray, np.ndarray]]]] = []
    csv_rows: list[list[Any]] = []
    csv_rows.append(
        [
            "projection",
            "n_layers",
            "mean_first_layer",
            "mean_last_layer",
            "depth_corr_r",
            "tail_mean_first_layer",
            "tail_mean_last_layer",
            "drift_pct_overall",
        ]
    )

    for gname, title in _PROJ_W_PANELS:
        per_tag = groups.get(gname, {})
        non_combined = {t: v for t, v in per_tag.items() if t != "_combined"}
        if not non_combined:
            continue
        panels.append((gname, title, True, non_combined))
        # Per-projection summary row for csv_rows
        layers, steps, grid = _layer_grid(non_combined, layer_re)
        if grid.size == 0:
            continue
        mean_per_layer = np.nanmean(grid, axis=1)
        tail_n = max(5, grid.shape[1] // 20)
        tail_mean_per_layer = np.nanmean(grid[:, -tail_n:], axis=1)
        if mean_per_layer.size >= 3 and np.std(mean_per_layer) > 1e-12:
            r = float(
                np.corrcoef(np.arange(len(layers)).astype(float), mean_per_layer)[0, 1]
            )
        else:
            r = float("nan")
        # Drift = relative change from first time slice to last time slice.
        first_col = grid[:, : max(1, grid.shape[1] // 20)]
        last_col = grid[:, -max(1, grid.shape[1] // 20) :]
        first_mean = float(np.nanmean(first_col))
        last_mean = float(np.nanmean(last_col))
        drift = (
            (last_mean - first_mean) / max(abs(first_mean), 1e-12) * 100
            if math.isfinite(first_mean) and first_mean != 0
            else float("nan")
        )
        csv_rows.append(
            [
                title.split(" ", 1)[0],
                len(layers),
                f"{float(mean_per_layer[0]):.6g}",
                f"{float(mean_per_layer[-1]):.6g}",
                f"{r:+.3f}",
                f"{float(tail_mean_per_layer[0]):.6g}",
                f"{float(tail_mean_per_layer[-1]):.6g}",
                f"{drift:+.2f}",
            ]
        )

    if not panels:
        return
    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, figsize=(16, max(2.2 * n, 4.0)), sharex=True, constrained_layout=True
    )
    axes_list = list(axes) if n > 1 else [axes]
    _stacked_heatmap(fig, axes_list, panels, layer_re)
    fig.suptitle(
        f"per-projection weight L2 depth profile — {run_name}",
        fontsize=13,
        weight="bold",
    )
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerows(csv_rows)


# ---------------------------------------------------------------------------
# Router dynamics — entropy proxy + load distribution
# ---------------------------------------------------------------------------
def plot_router_dynamics(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    png_path: Path,
    csv_path: Path,
) -> None:
    """4 panels: routing-score amax envelope, expert utilization per layer,
    load_balance per layer, overloaded_ratio per layer."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), constrained_layout=True)

    # Panel 1: routing score amax envelope (proxy for routing confidence)
    rs = groups.get("router_scores_amax", {})
    items = []
    if rs:
        non_combined = {t: v for t, v in rs.items() if t != "_combined"}
        if non_combined:
            s, mx, md, mn = _layer_envelope(non_combined)
            if s.size:
                items.append(("scores amax: layer max", s, mx))
                items.append(("scores amax: layer median", s, md))
    _line_panel(
        axes[0, 0],
        items,
        "router output scores amax (per-layer envelope)",
        "score amax",
        log_y=False,
    )

    # Panel 2: expert utilization per-layer envelope + global mean
    eu = groups.get("moe_expert_utilization_mean", {}).get("_combined")
    eu_pl = groups.get("moe_expert_utilization_per_layer", {})
    items = []
    if eu is not None:
        items.append(("global mean", eu[0], eu[1]))
    if eu_pl:
        non_combined = {t: v for t, v in eu_pl.items() if t != "_combined"}
        s, mx, md, mn = _layer_envelope(non_combined)
        if s.size:
            items.append(("per-layer max", s, mx))
            items.append(("per-layer median", s, md))
            items.append(("per-layer min", s, mn))
    _line_panel(
        axes[0, 1],
        items,
        "expert utilization (global + per-layer)",
        "utilization",
        log_y=False,
    )

    # Panel 3: load balance per layer envelope + global mean
    lb = groups.get("load_balance", {}).get("_combined")
    lb_pl = groups.get("moe_load_balance_per_layer", {})
    items = []
    if lb is not None:
        items.append(("global mean", lb[0], lb[1]))
    if lb_pl:
        non_combined = {t: v for t, v in lb_pl.items() if t != "_combined"}
        s, mx, md, mn = _layer_envelope(non_combined)
        if s.size:
            items.append(("per-layer max", s, mx))
            items.append(("per-layer median", s, md))
    _line_panel(
        axes[1, 0], items, "load balance (global + per-layer)", "balance", log_y=False
    )

    # Panel 4: overloaded_ratio per layer envelope
    over_pl = groups.get("dispatcher_overloaded_per_layer", {})
    items = []
    if over_pl:
        non_combined = {t: v for t, v in over_pl.items() if t != "_combined"}
        s, mx, md, mn = _layer_envelope(non_combined)
        if s.size:
            items.append(("per-layer max", s, mx))
            items.append(("per-layer median", s, md))
    _line_panel(
        axes[1, 1],
        items,
        "overloaded experts ratio (per-layer envelope)",
        "fraction",
        log_y=False,
    )

    fig.suptitle(f"router dynamics — {run_name}", fontsize=13, weight="bold")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    # CSV: summarized router signals at last step + tail-mean across runs.
    rows: list[list[Any]] = []
    rows.append(
        [
            "signal",
            "global_mean_last",
            "global_mean_tail",
            "per_layer_max_last",
            "per_layer_median_last",
            "per_layer_min_last",
        ]
    )

    def _last(item) -> float | None:
        if item is None:
            return None
        s, v = item
        if v.size == 0:
            return None
        finite = v[np.isfinite(v)]
        return float(finite[-1]) if finite.size else None

    def _tail_mean(item, frac: float = 0.1) -> float | None:
        if item is None:
            return None
        s, v = item
        if v.size == 0:
            return None
        tail_n = max(10, int(v.size * frac))
        tail = v[-tail_n:]
        finite = tail[np.isfinite(tail)]
        return float(finite.mean()) if finite.size else None

    def _envelope_last(per_tag, agg) -> float | None:
        non_combined = {t: v for t, v in per_tag.items() if t != "_combined"}
        if not non_combined:
            return None
        s, mx, md, mn = _layer_envelope(non_combined)
        if s.size == 0:
            return None
        if agg == "max":
            return float(mx[-1])
        if agg == "median":
            return float(md[-1])
        if agg == "min":
            return float(mn[-1])
        return None

    for label, group_key in (
        ("expert_utilization", "moe_expert_utilization_per_layer"),
        ("load_balance", "moe_load_balance_per_layer"),
        ("overloaded_ratio", "dispatcher_overloaded_per_layer"),
    ):
        global_key = {
            "expert_utilization": "moe_expert_utilization_mean",
            "load_balance": "load_balance",
            "overloaded_ratio": None,
        }.get(label)
        g_combined = groups.get(global_key, {}).get("_combined") if global_key else None
        per_tag = groups.get(group_key, {})
        rows.append(
            [
                label,
                f"{_last(g_combined):.6g}" if _last(g_combined) is not None else "",
                f"{_tail_mean(g_combined):.6g}"
                if _tail_mean(g_combined) is not None
                else "",
                f"{_envelope_last(per_tag, 'max'):.6g}" if per_tag else "",
                f"{_envelope_last(per_tag, 'median'):.6g}" if per_tag else "",
                f"{_envelope_last(per_tag, 'min'):.6g}" if per_tag else "",
            ]
        )
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Loss decomposition + convergence rate
# ---------------------------------------------------------------------------
def plot_loss_decomposition(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    png_path: Path,
    csv_path: Path,
) -> None:
    """4 panels: (a) all loss components on log y, (b) ratio of each
    component to total over time, (c) smoothed relative slope of total
    loss vs step, (d) cumulative improvement curve.
    """
    loss = groups.get("loss", {})
    lc = groups.get("loss_components", {})
    if not loss:
        return
    total = loss.get("loss/train/total") or loss.get("_combined")
    ce = loss.get("loss/train/loss_ce")
    if total is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)

    # Panel (a): all components
    items = []
    items.append(("total", total[0], total[1]))
    if ce is not None:
        items.append(("ce", ce[0], ce[1]))
    if lc:
        for tag, (s, v) in lc.items():
            if tag == "_combined":
                continue
            short = tag.split("/")[-1]
            items.append((short, s, v))
    _line_panel(axes[0, 0], items, "loss components", "value", log_y=True)

    # Panel (b): component / total ratio
    s_total, v_total = total
    if s_total.size:
        step_to_total = {
            int(st): float(vt)
            for st, vt in zip(s_total, v_total)
            if math.isfinite(vt) and vt > 0
        }
        items = []
        if ce is not None:
            sx, vx = ce
            xs = []
            ys = []
            for st, vv in zip(sx, vx):
                t = step_to_total.get(int(st))
                if t and math.isfinite(vv):
                    xs.append(st)
                    ys.append(vv / t)
            if xs:
                items.append(("ce/total", np.array(xs), np.array(ys)))
        for tag, (sx, vx) in lc.items():
            if tag == "_combined":
                continue
            xs = []
            ys = []
            for st, vv in zip(sx, vx):
                t = step_to_total.get(int(st))
                if t and math.isfinite(vv):
                    xs.append(st)
                    ys.append(vv / t)
            if xs:
                short = tag.split("/")[-1]
                items.append((f"{short}/total", np.array(xs), np.array(ys)))
        _line_panel(
            axes[0, 1], items, "component / total ratio over time", "ratio", log_y=True
        )

    # Panel (c): smoothed relative slope
    s, v = total
    if v.size > 500:
        kernel = np.ones(200) / 200
        smoothed = np.convolve(v, kernel, mode="valid")
        sm_steps = s[199:]
        # Compute relative slope over rolling 500-sample window
        window_n = 500
        slopes = []
        slope_steps = []
        for i in range(0, smoothed.size - window_n, max(1, window_n // 5)):
            x = sm_steps[i : i + window_n].astype(float)
            y = smoothed[i : i + window_n]
            if y.size < 2 or np.std(y) < 1e-12:
                continue
            sl, _ = np.polyfit(x, y, 1)
            current = y.mean()
            if current <= 0:
                continue
            slopes.append(sl / current)
            slope_steps.append(int(x[-1]))
        if slopes:
            items = [("rel-slope per step", np.array(slope_steps), np.array(slopes))]
            _line_panel(
                axes[1, 0],
                items,
                "smoothed relative slope of total loss",
                "rel slope / step",
                log_y=False,
            )
            axes[1, 0].axhline(0, color="red", linestyle="--", alpha=0.5)
            axes[1, 0].axhline(-1e-5, color="orange", linestyle=":", alpha=0.5)

    # Panel (d): cumulative improvement
    s, v = total
    if v.size > 50:
        finite_mask = np.isfinite(v)
        s_f, v_f = s[finite_mask], v[finite_mask]
        if v_f.size:
            initial = float(v_f[0])
            current = v_f
            cum_improvement_pct = (initial - current) / initial * 100
            _line_panel(
                axes[1, 1],
                [("cum improvement % from start", s_f, cum_improvement_pct)],
                "cumulative loss improvement (% from start)",
                "%",
                log_y=False,
            )

    fig.suptitle(
        f"loss decomposition + convergence rate — {run_name}",
        fontsize=13,
        weight="bold",
    )
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    # CSV: aggregate per-component stats
    rows: list[list[Any]] = [
        [
            "component",
            "n_points",
            "start",
            "end",
            "min",
            "max",
            "tail_mean",
            "drop_pct_total",
        ]
    ]
    components = [("total", total)]
    if ce is not None:
        components.append(("loss_ce", ce))
    if lc:
        for tag, item in lc.items():
            if tag == "_combined":
                continue
            components.append((tag.split("/")[-1], item))
    for label, (s, v) in components:
        if v.size == 0:
            continue
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            continue
        tail_n = max(10, finite.size // 20)
        tail = finite[-tail_n:]
        drop = (float(finite[0]) - float(finite[-1])) / max(abs(finite[0]), 1e-12) * 100
        rows.append(
            [
                label,
                finite.size,
                f"{float(finite[0]):.6g}",
                f"{float(finite[-1]):.6g}",
                f"{float(finite.min()):.6g}",
                f"{float(finite.max()):.6g}",
                f"{float(tail.mean()):.6g}",
                f"{drop:+.2f}",
            ]
        )
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Phase detection + cross-layer correlation + spike clusters
# ---------------------------------------------------------------------------
def _detect_training_phases(
    s: np.ndarray,
    v: np.ndarray,
) -> list[tuple[str, int, int, str]]:
    """Heuristic phase detection on the loss curve.

    Returns a list of ``(phase, step_start, step_end, description)``:
    * ``warmup``     — first 5% of training where loss is still descending fast.
    * ``steady``     — middle region where loss is descending normally.
    * ``plateau``    — late region where descent rate ≤ 30% of mid-training rate.
    """
    out: list[tuple[str, int, int, str]] = []
    if v.size < 200:
        return out
    finite = np.isfinite(v)
    s_f, v_f = s[finite], v[finite]
    if v_f.size < 200:
        return out
    last = int(s_f[-1])
    warmup_end = int(last * 0.05)
    out.append(
        (
            "warmup",
            int(s_f[0]),
            warmup_end,
            "rapid initial descent (≤ 5% of total steps)",
        )
    )

    # Compute mid-training rate vs late-training rate
    mid_start = int(last * 0.30)
    mid_end = int(last * 0.60)
    late_start = int(last * 0.85)
    late_end = last

    def _slope_rel(s_seg: np.ndarray, v_seg: np.ndarray) -> float:
        if v_seg.size < 50:
            return float("nan")
        x = s_seg.astype(float)
        try:
            sl, _ = np.polyfit(x, v_seg, 1)
        except Exception:  # noqa: BLE001
            return float("nan")
        m = float(v_seg.mean())
        if m <= 0:
            return float("nan")
        return sl / m

    mid_mask = (s_f >= mid_start) & (s_f <= mid_end)
    late_mask = (s_f >= late_start) & (s_f <= late_end)
    mid_slope = _slope_rel(s_f[mid_mask], v_f[mid_mask])
    late_slope = _slope_rel(s_f[late_mask], v_f[late_mask])

    out.append(
        ("steady", warmup_end, mid_end, f"mid-training rel-slope {mid_slope:+.2e}/step")
    )
    if math.isfinite(mid_slope) and math.isfinite(late_slope):
        ratio = abs(late_slope) / max(abs(mid_slope), 1e-12)
        if ratio < 0.3:
            out.append(
                (
                    "plateau",
                    late_start,
                    late_end,
                    f"late slope {late_slope:+.2e}/step "
                    f"({ratio * 100:.0f}% of mid-training rate)",
                )
            )
        else:
            out.append(
                (
                    "converging",
                    late_start,
                    late_end,
                    f"late slope {late_slope:+.2e}/step "
                    f"({ratio * 100:.0f}% of mid-training rate)",
                )
            )
    return out


def plot_phase_diagnosis(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    png_path: Path,
    md_path: Path,
) -> None:
    """Annotated loss curve with detected phases + LR overlay."""
    total = groups.get("loss", {}).get("loss/train/total")
    lr = groups.get("lr", {}).get("_combined")
    if total is None:
        return
    s, v = total
    phases = _detect_training_phases(s, v)
    if not phases:
        return

    fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
    ax.plot(s, v, color="black", linewidth=1.5, label="loss/train/total")
    ax.set_yscale("log")
    ax.set_ylabel("loss (log)", color="black")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    palette = {
        "warmup": "#bce4f5",
        "steady": "#c2e8c4",
        "converging": "#fff2b3",
        "plateau": "#fbc6c8",
    }
    for phase, st, en, _ in phases:
        ax.axvspan(
            st, en, facecolor=palette.get(phase, "#dddddd"), alpha=0.4, edgecolor="none"
        )
        ax.text(
            (st + en) / 2,
            v.max() * 0.9,
            phase,
            ha="center",
            va="top",
            fontsize=11,
            weight="bold",
        )

    if lr is not None and lr[1].size:
        ax2 = ax.twinx()
        ax2.plot(lr[0], lr[1], color="purple", linewidth=1.2, alpha=0.8, label="LR")
        ax2.set_yscale("log")
        ax2.set_ylabel("LR (log)", color="purple")

    ax.set_title(f"phase diagnosis — {run_name}", fontsize=13, weight="bold")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    # MD report of phases
    lines = [
        f"# phase diagnosis — {run_name}\n",
        "Heuristic phase detection on the smoothed loss curve.\n",
        "| phase | step start | step end | description |",
        "|---|---:|---:|---|",
    ]
    for phase, st, en, desc in phases:
        lines.append(f"| {phase} | {st} | {en} | {desc} |")
    md_path.write_text("\n".join(lines) + "\n")


def plot_cross_layer_correlation(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    run_name: str,
    png_path: Path,
    csv_path: Path,
) -> None:
    """For each key per-layer signal, build a ``[layer × layer]``
    Pearson correlation matrix of layers' time-series. Layers whose
    metric moves together over training appear bright. Diagonal = 1.
    """
    layer_re = re.compile(r"layer_(\d+)(?:$|/)|\.layers\.(\d+)\.")
    panels_meta: list[tuple[str, str]] = [
        ("max_vio_per_layer", "max_vio"),
        ("layer_residual_amax", "residual_amax"),
        ("grad_norm_layers", "grad_norm"),
        ("aux_loss", "aux_loss"),
    ]

    grids: list[tuple[str, list[int], np.ndarray]] = []
    for gname, title in panels_meta:
        per_tag = groups.get(gname, {})
        non_combined = {t: v for t, v in per_tag.items() if t != "_combined"}
        if not non_combined:
            continue
        layers, steps, grid = _layer_grid(non_combined, layer_re)
        if grid.size == 0 or grid.shape[0] < 4 or grid.shape[1] < 10:
            continue
        # Pearson r between rows (layers).
        # Replace NaN with the column mean to allow correlation calc.
        col_mean = np.nanmean(grid, axis=0, keepdims=True)
        filled = np.where(np.isfinite(grid), grid, col_mean)
        # Drop layers with constant series
        stds = np.std(filled, axis=1)
        keep = stds > 1e-12
        if keep.sum() < 4:
            continue
        filled = filled[keep]
        L_kept = [layers[i] for i in range(len(layers)) if keep[i]]
        corr = np.corrcoef(filled)
        grids.append((title, L_kept, corr))

    if not grids:
        return

    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    csv_rows: list[list[Any]] = []
    for ax, (title, L_kept, corr) in zip(axes, grids):
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        yt_step = max(1, len(L_kept) // 10)
        ax.set_xticks(range(0, len(L_kept), yt_step))
        ax.set_xticklabels(
            [f"L{L_kept[i]}" for i in range(0, len(L_kept), yt_step)],
            fontsize=7,
            rotation=45,
        )
        ax.set_yticks(range(0, len(L_kept), yt_step))
        ax.set_yticklabels(
            [f"L{L_kept[i]}" for i in range(0, len(L_kept), yt_step)], fontsize=7
        )
        ax.set_title(title, fontsize=10, weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        # CSV: aggregate stats per signal
        # Off-diagonal mean correlation
        n_layers = len(L_kept)
        if n_layers > 1:
            mask = ~np.eye(n_layers, dtype=bool)
            mean_off = float(np.nanmean(corr[mask]))
            # Adjacent (layer i vs layer i+1)
            adj = [corr[i, i + 1] for i in range(n_layers - 1)]
            mean_adj = float(np.nanmean(adj)) if adj else float("nan")
            # Distant (first 25% vs last 25%)
            q1 = max(1, n_layers // 4)
            distant = corr[:q1, -q1:]
            mean_distant = float(np.nanmean(distant))
            csv_rows.append(
                [
                    title,
                    n_layers,
                    f"{mean_off:+.3f}",
                    f"{mean_adj:+.3f}",
                    f"{mean_distant:+.3f}",
                ]
            )

    fig.suptitle(f"cross-layer correlation — {run_name}", fontsize=13, weight="bold")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "signal",
                "n_layers",
                "mean_off_diag_r",
                "mean_adjacent_r",
                "mean_distant_r",
            ]
        )
        w.writerows(csv_rows)


def detect_spike_clusters(
    spikes: list[SpikeEvent],
    cluster_window: int = 500,
) -> list[dict[str, Any]]:
    """Group spikes that occur within ``cluster_window`` steps of each
    other. A cluster spanning ≥ 2 distinct *signals* is interesting:
    suggests a coincident event affecting multiple metrics."""
    if not spikes:
        return []
    sorted_spikes = sorted(spikes, key=lambda e: e.step)
    clusters: list[list[SpikeEvent]] = []
    current: list[SpikeEvent] = [sorted_spikes[0]]
    for e in sorted_spikes[1:]:
        if e.step - current[-1].step <= cluster_window:
            current.append(e)
        else:
            clusters.append(current)
            current = [e]
    clusters.append(current)
    out: list[dict[str, Any]] = []
    for cluster in clusters:
        signals_in_cluster = sorted({e.signal for e in cluster})
        peak = max(cluster, key=lambda e: abs(e.z))
        out.append(
            {
                "step_start": cluster[0].step,
                "step_end": cluster[-1].step,
                "n_events": len(cluster),
                "n_signals": len(signals_in_cluster),
                "signals": signals_in_cluster,
                "peak_step": peak.step,
                "peak_z": peak.z,
                "peak_signal": peak.signal,
                "peak_tag": peak.tag,
            }
        )
    return out


def write_spike_clusters_md(
    spikes: list[SpikeEvent],
    out_path: Path,
    *,
    cluster_window: int = 500,
) -> None:
    clusters = detect_spike_clusters(spikes, cluster_window=cluster_window)
    lines = [
        "# spike clusters\n",
        f"Grouped within {cluster_window}-step windows. Multi-signal "
        f"clusters indicate coincident events (worth investigating).\n",
        "| step range | n events | n signals | signals | peak z | peak signal/tag |",
        "|---|---:|---:|---|---:|---|",
    ]
    for c in clusters:
        sigs = ", ".join(c["signals"])
        lines.append(
            f"| {c['step_start']}..{c['step_end']} | {c['n_events']} | "
            f"{c['n_signals']} | {sigs} | {c['peak_z']:.1f} | "
            f"`{c['peak_signal']}`/`{c['peak_tag'].split('/')[-1]}` |"
        )
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# JSON dump — raw aggregated series, for downstream notebooks
# ---------------------------------------------------------------------------
def dump_json(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    out_path: Path,
) -> None:
    """Dump aggregated series to JSON.

    Streams to disk and uses the C-accelerated json encoder with
    ``separators=(",", ":")`` (no whitespace) — about 2× faster than
    the default ``json.dumps(obj)`` round-trip. Each numpy array is
    converted with ``.tolist()`` lazily as we iterate (avoids building
    one giant intermediate dict tree).
    """
    with out_path.open("w", encoding="utf-8") as f:
        f.write("{")
        first_group = True
        for gname, per_tag in groups.items():
            if not first_group:
                f.write(",")
            first_group = False
            f.write(json.dumps(gname, ensure_ascii=False))
            f.write(":{")
            first_tag = True
            for tag, (s, v) in per_tag.items():
                if not first_tag:
                    f.write(",")
                first_tag = False
                f.write(json.dumps(tag, ensure_ascii=False))
                f.write(':{"steps":')
                # int array — json encoder handles list of ints fast.
                f.write(json.dumps(s.tolist(), separators=(",", ":")))
                f.write(',"values":')
                f.write(json.dumps(v.tolist(), separators=(",", ":")))
                f.write("}")
            f.write("}")
        f.write("}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _emit_overview_table(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    out_path: Path,
) -> None:
    """CSV with the 6 canonical overview signals shown on 01_overview.png."""
    items: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, key, sub in (
        ("loss_total", "loss", "loss/train/total"),
        ("loss_ce", "loss", "loss/train/loss_ce"),
        ("lr", "lr", "_combined"),
        ("grad_norm_global", "grad_norm_global", "_combined"),
        ("max_vio_global", "max_vio_global", "_combined"),
        ("activation_amax_combined", "activation_amax", "_combined"),
        ("aux_loss_layer_peak", "aux_loss", "_combined"),
    ):
        per_tag = groups.get(key) or {}
        item = per_tag.get(sub)
        if item is None:
            continue
        s, v = item
        if v.size:
            items.append((label, s, v))
    _dump_time_series_table(items, out_path)


def _emit_moe_health_table(
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    out_path: Path,
) -> None:
    """CSV combining global MoE counters + per-layer envelopes."""
    items: list[tuple[str, np.ndarray, np.ndarray]] = []
    # Globals
    g = groups.get("dispatcher_globals", {})
    for tag in (
        "dispatcher_sum_metrics/token_drop_level",
        "dispatcher_sum_metrics/overloaded_expert_ratio",
        "dispatcher_sum_metrics/empty_token_level",
        "dispatcher_sum_metrics/step_max_expert_load",
        "dispatcher_sum_metrics/num_drop_steps",
    ):
        item = g.get(tag)
        if item is not None:
            items.append((tag.split("/")[-1], item[0], item[1]))
    eu = groups.get("moe_expert_utilization_mean", {}).get("_combined")
    if eu is not None:
        items.append(("expert_utilization_mean_global", eu[0], eu[1]))
    # Per-layer envelopes
    for label, key in (
        ("max_vio_per_layer_envelope_max", "max_vio_per_layer"),
        ("aux_loss_per_layer_envelope_max", "aux_loss"),
        (
            "expert_utilization_per_layer_envelope_max",
            "moe_expert_utilization_per_layer",
        ),
        ("load_balance_per_layer_envelope_max", "moe_load_balance_per_layer"),
    ):
        per_tag = groups.get(key, {})
        non_combined = {t: v for t, v in per_tag.items() if t != "_combined"}
        if not non_combined:
            continue
        s, mx, md, mn = _layer_envelope(non_combined)
        if s.size:
            items.append((label, s, mx))
    _dump_time_series_table(items, out_path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="instability.report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--tb-dir",
        required=True,
        type=Path,
        help="TB run directory with events.out.tfevents.*",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="output root. Plots land in OUT/figures/, tables and JSON in OUT/data/.",
    )
    ap.add_argument(
        "--workers", type=int, default=8, help="parallel event-file readers"
    )
    ap.add_argument(
        "--name", default=None, help="display name (default: tb-dir basename)"
    )
    ap.add_argument(
        "--no-heatmap", action="store_true", help="skip the max_vio heatmap"
    )
    ap.add_argument(
        "--no-layer-attribution",
        action="store_true",
        help="skip per-layer attribution + layer_profile.csv",
    )
    ap.add_argument(
        "--no-focused-pngs",
        action="store_true",
        help="skip moe_health / activations_depth / "
        "gradients_depth / weights_depth / system_health + summary.md",
    )
    ap.add_argument(
        "--no-advanced",
        action="store_true",
        help="skip per_projection_depth / router_dynamics / "
        "loss_decomposition / phase_diagnosis / "
        "cross_layer_corr / spike_clusters",
    )
    ap.add_argument(
        "--no-atlas",
        action="store_true",
        help="skip the comprehensive per-family layer×step "
        "heatmap atlas (figures/atlas/<category>/<family>.png)",
    )
    ap.add_argument("--no-json", action="store_true", help="skip the signals.json dump")
    ap.add_argument(
        "--no-spikes", action="store_true", help="skip transient-spike detection"
    )
    ap.add_argument("--spike-baseline", type=int, default=100)
    ap.add_argument("--spike-dedup-gap", type=int, default=50)
    args = ap.parse_args(argv)

    fig_dir, data_dir = _ensure_dirs(args.out_dir)
    run_name = args.name or args.tb_dir.name

    groups = read_signals(args.tb_dir, workers=args.workers)

    # ---- Spike detection ----
    spikes: list[SpikeEvent] = []
    if not args.no_spikes:
        spikes = find_spikes(
            groups,
            baseline_window=args.spike_baseline,
            dedup_step_gap=args.spike_dedup_gap,
        )
        print_spikes_summary(spikes)
        dump_spikes_csv(spikes, data_dir / "spikes.csv")
        print(f"\nWrote {data_dir / 'spikes.csv'}", flush=True)

    # ---- 01: Overview (was report.png) ----
    plot_report(groups, run_name, fig_dir / "01_overview.png", spikes=spikes)
    print(f"Wrote {fig_dir / '01_overview.png'}", flush=True)
    _emit_overview_table(groups, data_dir / "01_overview.csv")
    print(f"Wrote {data_dir / '01_overview.csv'}", flush=True)

    # ---- 02: max_vio heatmap (+ per-layer table) ----
    if not args.no_heatmap:
        plot_max_vio_heatmap(groups, run_name, fig_dir / "02_max_vio_heatmap.png")
        print(f"Wrote {fig_dir / '02_max_vio_heatmap.png'}", flush=True)
        mvl = groups.get("max_vio_per_layer", {})
        non_combined = {t: v for t, v in mvl.items() if t != "_combined"}
        if non_combined:
            layer_re = re.compile(r"layer_(\d+)$")
            layers, steps, grid = _layer_grid(non_combined, layer_re)
            _dump_layer_grid_table(
                layers,
                steps,
                grid,
                data_dir / "02_max_vio_per_layer.csv",
            )
            print(f"Wrote {data_dir / '02_max_vio_per_layer.csv'}", flush=True)

    # ---- 03: per-layer attribution + layer_profile.csv ----
    if not args.no_layer_attribution:
        summaries = plot_layer_attribution(
            groups,
            run_name,
            fig_dir / "03_layer_attribution.png",
        )
        if summaries:
            print(f"Wrote {fig_dir / '03_layer_attribution.png'}", flush=True)
            dump_layer_profile_csv(summaries, data_dir / "03_layer_profile.csv")
            print(f"Wrote {data_dir / '03_layer_profile.csv'}", flush=True)

    # ---- 04-08: focused PNGs ----
    if not args.no_focused_pngs:
        plot_moe_health(groups, run_name, fig_dir / "04_moe_health.png")
        _emit_moe_health_table(groups, data_dir / "04_moe_health.csv")
        print(f"Wrote {fig_dir / '04_moe_health.png'}", flush=True)
        print(f"Wrote {data_dir / '04_moe_health.csv'}", flush=True)

        plot_activations_depth(groups, run_name, fig_dir / "05_activations_depth.png")
        print(f"Wrote {fig_dir / '05_activations_depth.png'}", flush=True)

        plot_gradients_depth(groups, run_name, fig_dir / "06_gradients_depth.png")
        print(f"Wrote {fig_dir / '06_gradients_depth.png'}", flush=True)

        plot_weights_depth(groups, run_name, fig_dir / "07_weights_depth.png")
        print(f"Wrote {fig_dir / '07_weights_depth.png'}", flush=True)

        plot_system_health(groups, run_name, fig_dir / "08_system_health.png")
        print(f"Wrote {fig_dir / '08_system_health.png'}", flush=True)

        write_summary_md(groups, spikes, run_name, data_dir / "summary.md")
        print(f"Wrote {data_dir / 'summary.md'}", flush=True)

    # ---- 09-15: advanced analyses ----
    if not args.no_advanced:
        plot_per_projection_depth(
            groups,
            run_name,
            fig_dir / "09_per_projection_depth.png",
            data_dir / "09_per_projection_depth.csv",
        )
        if (fig_dir / "09_per_projection_depth.png").exists():
            print(f"Wrote {fig_dir / '09_per_projection_depth.png'}", flush=True)
        if (data_dir / "09_per_projection_depth.csv").exists():
            print(f"Wrote {data_dir / '09_per_projection_depth.csv'}", flush=True)

        plot_router_dynamics(
            groups,
            run_name,
            fig_dir / "10_router_dynamics.png",
            data_dir / "10_router_dynamics.csv",
        )
        if (fig_dir / "10_router_dynamics.png").exists():
            print(f"Wrote {fig_dir / '10_router_dynamics.png'}", flush=True)

        plot_loss_decomposition(
            groups,
            run_name,
            fig_dir / "11_loss_decomposition.png",
            data_dir / "11_loss_decomposition.csv",
        )
        if (fig_dir / "11_loss_decomposition.png").exists():
            print(f"Wrote {fig_dir / '11_loss_decomposition.png'}", flush=True)

        plot_phase_diagnosis(
            groups,
            run_name,
            fig_dir / "12_phase_diagnosis.png",
            data_dir / "12_phases.md",
        )
        if (fig_dir / "12_phase_diagnosis.png").exists():
            print(f"Wrote {fig_dir / '12_phase_diagnosis.png'}", flush=True)

        plot_cross_layer_correlation(
            groups,
            run_name,
            fig_dir / "13_cross_layer_corr.png",
            data_dir / "13_cross_layer_corr.csv",
        )
        if (fig_dir / "13_cross_layer_corr.png").exists():
            print(f"Wrote {fig_dir / '13_cross_layer_corr.png'}", flush=True)

        if spikes:
            write_spike_clusters_md(spikes, data_dir / "14_spike_clusters.md")
            print(f"Wrote {data_dir / '14_spike_clusters.md'}", flush=True)

    # ---- atlas: plot every monitor family, let the model pick ----
    if not args.no_atlas:
        from instability import atlas

        atlas.run(args.tb_dir, fig_dir, run_name=run_name, workers=args.workers)

    if not args.no_json:
        dump_json(groups, data_dir / "signals.json")
        print(f"Wrote {data_dir / 'signals.json'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
