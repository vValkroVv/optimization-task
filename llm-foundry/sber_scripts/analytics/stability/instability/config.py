"""CLI-driven config for instability/* (eda, temporal, projections).

Everything is plain Python — no YAML. Paths come from required CLI
flags (``--tb-root``, ``--out-dir``). Per-stage parameters (n,
filters, preferred-run shortlists) live as constants below; tweak
them in code if you need to.

Public API kept stable: callers (``eda.main``/``temporal.main``/
``projections.main``) still get ``ResolvedConfig`` from ``resolve(args)``
and pass per-stage records to ``select_runs``.

Run selection priority (unchanged):
    1. ``--runs A B C ...``    → take exactly these.
    2. ``--runs-pattern '*X*'`` → fnmatch over subdirs of ``tb_root``.
    3. preferred (per stage) + filters → original behaviour, with
       preferred shortlists hardcoded in this file.
"""

from __future__ import annotations

import argparse
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Stage defaults — edit these to retune what each pipeline picks up.
# ---------------------------------------------------------------------------
@dataclass
class StageConfig:
    n: int | None = None
    max_runs: int | None = None
    max_workers: int = 8
    min_event_files: int = 2
    filters: dict[str, float | int] = field(default_factory=dict)
    preferred: list[str] = field(default_factory=list)


EDA_DEFAULTS = StageConfig(
    min_event_files=2,
    max_runs=None,
    max_workers=8,
)

TEMPORAL_DEFAULTS = StageConfig(
    n=16,
    filters={"n_vio_steps_min": 300, "max_vio_max_min": 4},
    preferred=[
        "Papayaseek1T-cut-baseline",
        "Papayaseek1T-cut-baseline_NO_seq_len_warmup",
        "Papayaseek1T-cut-baseline_aux_loss_free_strategy-adaptive_correct",
        "Papayaseek1T-cut-baseline_moe_router_bias_update_rate-0_router_aux_loss_coef-0_correct",
        "Papayaseek1T-cut-baseline_moe_router_bias_update_rate-0.00025_router_aux_loss_coef-2.5e-05",
        "Papayaseek1T-cut-router_z_loss_eps-1e-05",
        "Papayaseek1T-cut-router_z_loss_eps-1e-06",
        "Papayaseek1T-cut-baseline_renorm_router_weights-False_new_init1",
        "Papayaseek1T-cut-baseline_renorm_router_weights-True_new_init1",
        "Papayaseek1T-cut-baseline_norm_type-ZeroCenteredRMSNorm_gated_attention-False",
        "Papayaseek1T-cut-baseline_norm_type-LlamaRMSNorm_gated_attention-True",
        "1Tcut_3.5_branch_postnorm_wrap",
    ],
)

PROJECTIONS_DEFAULTS = StageConfig(
    n=8,
    # n_event_files_max guards against monsters that take hours to read.
    filters={"n_vio_steps_min": 300, "max_vio_max_min": 4, "n_event_files_max": 450},
    preferred=[
        "Papayaseek1T-cut-baseline",
        "Papayaseek1T-cut-baseline_NO_seq_len_warmup",
        "Papayaseek1T-cut-baseline_aux_loss_free_strategy-adaptive_correct",
        "Papayaseek1T-cut-baseline_moe_router_bias_update_rate-0.00025_router_aux_loss_coef-2.5e-05",
        "Papayaseek1T-cut-router_z_loss_eps-1e-05",
        "Papayaseek1T-cut-router_z_loss_eps-1e-06",
        "Papayaseek1T-cut-baseline_renorm_router_weights-False_new_init1",
        "Papayaseek1T-cut-baseline_renorm_router_weights-True_new_init1",
        "1Tcut_3.5_branch_postnorm_wrap",
    ],
)


# ---------------------------------------------------------------------------
# Tag-group aliases — YOUR headline tag families.
# ---------------------------------------------------------------------------
# Map a tag SUBSTRING → a canonical group name. First match wins, so put more
# specific substrings before their prefixes (``moment_sq`` before ``moment``).
# These are the families you want named/curated; **every tag NOT matched here is
# auto-classified from its structure** (``features._structural_group``), so a new
# monitor only needs an entry if you want it under a specific name.
#
# Three ways to set your own:
#   1. Edit this list in place.
#   2. ``export INSTABILITY_TAG_GROUPS=/path/to/groups.json`` — a JSON object
#      ``{"substring": "group", ...}`` merged on top (yours win). No code edit.
#   3. ``features.register_tag_groups([("substring", "group"), ...])`` at runtime.
MONITOR_GROUP_ALIASES: tuple[tuple[str, str], ...] = (
    # derived signals (computed by add_derived_signals / projections)
    ("derived/layer_contrast", "layer_contrast"),
    ("derived/update_param_ratio", "update_param_ratio"),
    ("derived/grad_param_ratio", "grad_param_ratio"),
    ("derived/update_activation_ratio", "update_activation_ratio"),
    ("derived/grad_activation_ratio", "grad_activation_ratio"),
    ("derived/effective_lr", "effective_lr"),
    ("derived/update_minus_grad_log", "update_grad_gap"),
    ("derived/residual_contribution", "residual_contribution"),
    # native residual-connection monitors (own families, not lumped into activations)
    ("residual_contribution_output", "residual_contribution"),
    ("residual_output", "residual_stream"),
    # native optimizer ratios → aliased onto the derived-equivalent group names
    ("effective_lr/", "effective_lr"),
    ("update_weight_ratio/", "update_param_ratio"),
    ("update_rate/", "grad_param_ratio"),
    # optimizer norms / moments (specific before general)
    ("l2_norm/pre_clip_grad", "pre_clip_grad_norm"),
    ("l2_norm/moment_sq", "moment_sq_norm"),
    ("l2_norm/moment", "moment_norm"),
    ("amax/moment_sq", "moment_sq_amax"),
    ("amax/moment", "moment_amax"),
    ("activations_actrel", "activation_massive"),
    ("activations_actabs", "activation_massive"),
    ("l2_norm/param", "param_norm"),
    ("l2_norm/grad", "grad_norm"),
    ("l2_norm/update", "update_norm"),
    # MoE / router losses
    ("old_moe_routing", "routing_load"),
    ("moe_seq_aux_loss", "aux_loss"),
    ("moe_z_loss_router", "router_z_loss"),
    # NB: activations_amax / activations_amean intentionally absent —
    # _structural_group derives them (activations_<stat> → activation_<stat>).
)


# ---------------------------------------------------------------------------
# Resolved (per-invocation) config.
# ---------------------------------------------------------------------------
@dataclass
class ResolvedConfig:
    tb_root: Path
    output_dir: Path
    eda: StageConfig
    temporal: StageConfig
    projections: StageConfig
    runs_override: list[str] | None = None
    runs_pattern: str | None = None


def add_common_cli_args(ap: argparse.ArgumentParser) -> None:
    """Standard flag set for every instability/* script."""
    ap.add_argument(
        "--tb-root",
        required=True,
        help="Directory with TB run subdirs (each contains events.out.tfevents.*).",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Where to write CSV/PNG artefacts.",
    )
    ap.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Explicit run names (skips preferred/filters).",
    )
    ap.add_argument(
        "--runs-pattern",
        default=None,
        help="fnmatch glob over tb_root subdirs, e.g. 'Papayaseek*postnorm*'.",
    )


def resolve(args: argparse.Namespace) -> ResolvedConfig:
    tb_root = Path(args.tb_root).expanduser().resolve()
    if not tb_root.is_dir():
        raise SystemExit(f"--tb-root not a directory: {tb_root}")
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return ResolvedConfig(
        tb_root=tb_root,
        output_dir=output_dir,
        eda=EDA_DEFAULTS,
        temporal=TEMPORAL_DEFAULTS,
        projections=PROJECTIONS_DEFAULTS,
        runs_override=list(args.runs) if args.runs else None,
        runs_pattern=args.runs_pattern,
    )


# ---------------------------------------------------------------------------
# Run selection (unchanged semantics).
# ---------------------------------------------------------------------------
def select_runs(
    cfg: ResolvedConfig,
    stage: StageConfig,
    *,
    inv_df=None,
    summary_df=None,
) -> list[str]:
    if cfg.runs_override is not None:
        return [r for r in cfg.runs_override if (cfg.tb_root / r).is_dir()]

    if cfg.runs_pattern is not None:
        all_names = sorted(p.name for p in cfg.tb_root.iterdir() if p.is_dir())
        return [n for n in all_names if fnmatch.fnmatch(n, cfg.runs_pattern)]

    if summary_df is None:
        raise ValueError("select_runs: preferred-mode needs summary_df from eda.py.")

    import pandas as pd  # noqa: F401  (local: keep top-level import light)

    df = summary_df.copy()
    f = stage.filters
    if "n_vio_steps_min" in f:
        df = df[df["n_vio_steps"] >= f["n_vio_steps_min"]]
    if "max_vio_max_min" in f:
        df = df[df["max_vio_max"] >= f["max_vio_max_min"]]
    if "n_event_files_max" in f:
        if inv_df is None:
            raise ValueError("select_runs: filter n_event_files_max requires inv_df.")
        df = df.merge(inv_df[["run", "n_event_files"]], on="run", how="left")
        df = df[df["n_event_files"] <= f["n_event_files_max"]]

    valid = set(df["run"])
    out: list[str] = []
    for r in stage.preferred:
        if r in valid and r not in out:
            out.append(r)
        if stage.n is not None and len(out) >= stage.n:
            return out

    top = df.sort_values(["n_vio_steps", "max_vio_max"], ascending=[False, False])[
        "run"
    ].tolist()
    for r in top:
        if r in out:
            continue
        out.append(r)
        if stage.n is not None and len(out) >= stage.n:
            break
    return out
