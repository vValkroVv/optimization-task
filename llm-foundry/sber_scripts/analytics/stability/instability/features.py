"""Feature-flag detection and monitor/projection grouping.

Consolidates the constants and small parsers that the three instability
scripts (eda / temporal / projections) used to duplicate.

* ``FEATURE_PATTERNS`` — substring/regex flags inferred from run names.
* ``KEY_CONFIG_PATHS`` — points into ``config.yaml`` we want to surface.
* ``PROJECTION_PATTERNS`` — maps TB tags to a projection role (router / up / down / …).
* ``monitor_group`` / ``projection_group`` / ``family`` — bucket a TB tag into
  a coarse class used downstream for grouping/plotting.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from instability.config import MONITOR_GROUP_ALIASES


def rolling_slope(y: np.ndarray, win: int) -> np.ndarray:
    """Per-position slope of a linear fit over a window of length ``win``
    starting at that position. ``out[i]`` covers ``y[i : i + win]``; the
    last ``win - 1`` entries are NaN.
    """
    out = np.full(y.size, np.nan)
    if y.size < win or win < 3:
        return out
    x = np.arange(win, dtype=float)
    for i in range(y.size - win + 1):
        yy = y[i : i + win].astype(float)
        m = np.isfinite(yy)
        if m.sum() < max(3, win // 2):
            continue
        out[i] = np.polyfit(x[m], yy[m], 1)[0]
    return out


def growth_onset(
    steps: np.ndarray,
    y: np.ndarray,
    *,
    win: int = 8,
    slope_rel: float = 0.05,
    min_run: int = 5,
) -> float:
    """First step at which ``rolling_slope(y, win)`` stays positive and
    above ``slope_rel × baseline`` for at least ``min_run`` consecutive
    windows. ``baseline`` is the median of ``|y|`` over the first 10%
    (≥10 samples) of the series. Returns NaN if no onset is found.
    """
    if y.size < win + min_run:
        return math.nan
    base = np.nanmedian(np.abs(y[: max(10, y.size // 10)]))
    if not np.isfinite(base) or base <= 0:
        base = np.nanmedian(np.abs(y))
    if not np.isfinite(base) or base <= 0:
        return math.nan
    above = rolling_slope(y, win) > slope_rel * base
    streak = 0
    for i, ok in enumerate(above):
        streak = streak + 1 if ok else 0
        if streak >= min_run:
            return float(steps[i - min_run + 1])
    return math.nan


FEATURE_PATTERNS: list[tuple[str, str]] = [
    ("no_seq_len_warmup", r"NO_seq_len_warmup|no_warmup"),
    ("postnorm_wrap", r"postnorm[-_]?wrap|unwrapped_postnorm"),
    (
        "gated_attention",
        r"gated_attention-True|GatedAttention|gated_attention_gated_norm",
    ),
    ("zero_centered_gated_norm", r"ZeroCenteredGatedNorm"),
    ("zero_centered_rms_norm", r"ZeroCenteredRMSNorm|ZeroCenteredRmsNorm"),
    ("llama_rms_norm", r"LlamaRMSNorm"),
    ("aux_loss_free_adaptive", r"aux_loss_free_strategy-adaptive"),
    ("router_bias_update", r"moe_router_bias_update_rate"),
    ("router_aux_loss_zero", r"router_aux_loss_coef-0"),
    ("router_z_loss", r"router_z_loss"),
    ("router_sigmoid_input_scale", r"router_sigmoid_input_scale"),
    ("renorm_router_weights", r"renorm_router_weights-True"),
    ("new_init", r"new_init|correct-init|correct_init"),
    ("activation_clipping", r"activation_clipping-True|activation_clipping"),
    ("shared_expert_sigmoid", r"use_shared_expert_sigmoid-True|gated_shareds-True"),
    ("sandwichnorm", r"use_sandwichnorm-True"),
    ("mla_scaling_factor", r"use_mla_scaling_factor-True"),
    ("normalize_post_scale", r"normalize_post_scale-True"),
    ("optimizer_eps_1e20", r"optimizer_eps-1e-20|rms_norm_eps-1e-20"),
    ("liger", r"use_liger-True"),
    ("new_layout", r"use_new_layout-True"),
    ("float8_grouped_gemm", r"float8_grouped_gemm-True"),
    ("async_experts_quantization", r"async_experts_quantization-True"),
    ("deep_gemm_wgrad", r"deep_gemm_wgrad_True"),
    ("row2col_dg", r"row2col_dg-True"),
    ("prod_setup_no_activation_clipping", r"prod_setup_no_activation_clipping"),
    ("long_context_32k", r"32k|long_context"),
    ("varlen", r"varlen"),
    ("sp8", r"sp8"),
    ("theta2690745", r"theta2690745|thetta2690745"),
    ("no_yarn", r"no_yarn"),
    ("mtp_coef_0", r"mtp_coef_0"),
]


KEY_CONFIG_PATHS: list[tuple[str, tuple[str, ...]]] = [
    ("model.norm_type", ("model", "norm_type")),
    ("model.layernorm_type", ("model", "layernorm_type")),
    ("model.num_hidden_layers", ("model", "num_hidden_layers")),
    ("model.gated_attention", ("model", "gated_attention")),
    ("model.activation_clipping", ("model", "activation_clipping")),
    ("model.use_shared_expert_sigmoid", ("model", "use_shared_expert_sigmoid")),
    ("model.use_mla_scaling_factor", ("model", "use_mla_scaling_factor")),
    ("model.renorm_router_weights", ("model", "renorm_router_weights")),
    ("model.router_sigmoid_input_scale", ("model", "router_sigmoid_input_scale")),
    ("model.moe_router_bias_update_rate", ("model", "moe_router_bias_update_rate")),
    ("model.router_aux_loss_coef", ("model", "router_aux_loss_coef")),
    ("model.router_z_loss", ("model", "router_z_loss")),
    ("model.router_z_loss_eps", ("model", "router_z_loss_eps")),
    ("optimizer.eps", ("optimizer", "eps")),
    ("scheduler.t_warmup", ("scheduler", "t_warmup")),
    ("global_train_batch_size", ("global_train_batch_size",)),
    ("device_train_microbatch_size", ("device_train_microbatch_size",)),
    ("max_seq_len", ("max_seq_len",)),
    ("max_duration", ("max_duration",)),
]


PROJECTION_PATTERNS: list[tuple[str, str]] = [
    ("router", r"(router|gate)"),
    ("shared_expert", r"(shared_expert|shareds|shared)"),
    ("up", r"(upgate|up_proj|up_projection|w1|fc1|gmm_up|fused_swiglu)"),
    (
        "down",
        r"(down_proj|down_projection|w2|fc2|gmm_down|moe_output|block_sparse_moe_output)",
    ),
    (
        "attention",
        r"(self_attn|attention|q_proj|k_proj|v_proj|o_proj|qkv|kv_|q_a_proj|q_b_proj|kv_a_proj|kv_b_proj)",
    ),
]


def nested_get(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def flatten_keys(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_keys(v, key))
    else:
        out[prefix] = d
    return out


def infer_features_from_name(name: str) -> dict[str, bool]:
    return {
        feat: bool(re.search(pattern, name, flags=re.IGNORECASE))
        for feat, pattern in FEATURE_PATTERNS
    }


def read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


# Active alias table (substring → canonical group). Seeded from the user-editable
# ``config.MONITOR_GROUP_ALIASES``; extend it without touching code via
# ``$INSTABILITY_TAG_GROUPS`` (a JSON object) or ``register_tag_groups()`` at runtime.
# These are the *headline* families you want named; everything else is classified
# structurally (``_structural_group``).
_tag_aliases: list[tuple[str, str]] = list(MONITOR_GROUP_ALIASES)


def register_tag_groups(
    pairs: list[tuple[str, str]] | dict[str, str], *, prepend: bool = True
) -> None:
    """Add ``(substring, group)`` aliases at runtime. Prepended by default so the
    user's families win over the built-ins (and over earlier additions)."""
    global _tag_aliases
    items = list(pairs.items()) if isinstance(pairs, dict) else list(pairs)
    _tag_aliases = items + _tag_aliases if prepend else _tag_aliases + items


def _load_env_tag_groups() -> None:
    """Merge aliases from ``$INSTABILITY_TAG_GROUPS`` (path to a JSON object)."""
    path = os.environ.get("INSTABILITY_TAG_GROUPS")
    if not path or not Path(path).exists():
        return
    try:
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            register_tag_groups(data)
    except (OSError, ValueError):
        pass  # bad/unreadable override file → silently keep built-in aliases


_load_env_tag_groups()


def monitor_group(tag: str) -> str:
    """Canonical group for a tag: a user/built-in alias if one applies, else the
    structural group derived from the tag's prefix (never silently ``other`` for a
    model monitor). New monitors rarely need an alias — see ``config.MONITOR_GROUP_ALIASES``."""
    for needle, group in _tag_aliases:
        if needle in tag:
            return group
    return _structural_group(tag)


# Global training scalars (not per-layer model monitors): stay ``other`` so the
# per-tag cross-run / lead-lag analyses (which gate on group != "other") ignore them.
_SCALAR_PREFIXES = (
    "loss/",
    "loss_",
    "lr-",
    "lr/",
    "memory/",
    "time/",
    "time_seconds/",
    "throughput/",
    "seq_length_warmup/",
    "trainer/",
    "wall_clock/",
    "grad_accum",
)


def _structural_group(tag: str) -> str:
    """Coarse group derived purely from a tag's prefix structure.

    The fallback for tags no explicit ``monitor_group`` rule covers. A brand-new
    monitor (e.g. ``activations_kurtosis/...`` or ``l2_norm/whatever/...``) gets a
    stable, descriptive group from its prefix rather than ``other`` — so it flows
    into atlas/compare/temporal automatically. Run ``python -m instability.tags``
    to see what lands here.
    """
    if tag.startswith(_SCALAR_PREFIXES):
        return "other"
    head = tag.split("/", 1)[0]
    if head.startswith("activations_gradients_"):
        return "act_gradient_" + head[len("activations_gradients_") :]
    if head.startswith("activations_"):
        return "activation_" + head[len("activations_") :]
    if head.startswith("weights_"):
        return "weight_" + head[len("weights_") :]
    if head == "l2_norm":
        parts = tag.split("/")
        return "l2_" + (parts[1] if len(parts) > 1 else "norm")
    if head == "amax":
        parts = tag.split("/")
        return "amax_" + (parts[1] if len(parts) > 1 else "x")
    if "max_vio" in head:
        return "max_vio"
    if head.startswith("expert_activation"):
        return "expert_activation"
    if head.startswith("moe_"):
        return "moe"
    return head or "other"


def projection_group(tag: str) -> str:
    low = tag.lower()
    # Residual-connection monitors carry their sublayer in the slot name; map attn's
    # write to the "attention" projection and mlp/moe's write to "moe_block" so the
    # event-study can compare attn-vs-mlp residual contributions.
    if "attn_residual" in low:
        return "attention"
    if "mlp_residual" in low:
        return "moe_block"
    if "derived/update_param_ratio" in low:
        return "moe_block"
    if "derived/grad_param_ratio" in low:
        return "moe_block"
    if "derived/update_minus_grad_log" in low:
        return "moe_block"
    if "derived/effective_lr" in low:
        return "moe_block"
    if (
        "derived/update_activation_ratio" in low
        or "derived/grad_activation_ratio" in low
    ):
        if "gmm_upgate" in low or "fused_swiglu" in low:
            return "up"
        if "block_sparse_moe_output" in low:
            return "down"
        return "moe_block"
    for name, pat in PROJECTION_PATTERNS:
        if re.search(pat, low):
            return name
    if "block_sparse_moe.flatten" in low:
        return "moe_block"
    return "other"


def family(tag: str) -> str:
    if "derived/projection_ratio" in tag:
        return "projection_ratio"
    if "derived/update_param_ratio" in tag:
        return "update_param_ratio"
    if "derived/grad_param_ratio" in tag:
        return "grad_param_ratio"
    if "derived/update_minus_grad_log" in tag:
        return "update_grad_gap"
    if "derived/effective_lr" in tag:
        return "effective_lr"
    if "derived/update_activation_ratio" in tag:
        return "update_activation_ratio"
    if "derived/grad_activation_ratio" in tag:
        return "grad_activation_ratio"
    return monitor_group(tag)


# ---------------------------------------------------------------------------
# Structural tag parsing — one place that decomposes any TB tag.
# ---------------------------------------------------------------------------
_LAYER_OF = re.compile(r"/layer_(\d+)(?:$|/)|\.layers\.(\d+)(?:\.|$)")
_FAM_LAYERS = re.compile(r"\.layers\.\d+")
_FAM_LAYER_SLASH = re.compile(r"/layer_\d+")
_FAM_INT = re.compile(r"\.\d+(?=\.|$)")


def layer_of(tag: str) -> int | None:
    """Layer index from a tag (``…layers.N…`` or ``…/layer_N``), else None."""
    m = _LAYER_OF.search(tag)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def family_template(tag: str) -> str:
    """Tag with the layer index and any expert/slot indices replaced by
    placeholders — one template per distinct monitor *shape*. So every layer /
    expert of the same monitor collapses to a single family key."""
    f = _FAM_LAYERS.sub(".layers.{L}", tag)
    f = _FAM_LAYER_SLASH.sub("/layer_{L}", f)
    return _FAM_INT.sub(".{i}", f)


@dataclass(frozen=True)
class TagInfo:
    """Structural decomposition of one TB tag (see ``parse_tag``)."""

    tag: str
    group: str  # coarse semantic group (monitor_group, structural fallback)
    projection: (
        str  # up / down / router / shared_expert / attention / moe_block / other
    )
    layer: int | None
    family: str  # normalized template — one per monitor shape
    recognized: bool  # False ⇒ group is "other" (needs a human rule or is a scalar)


def parse_tag(tag: str) -> TagInfo:
    """Decompose a tag into (group, projection, layer, family). The single entry
    point for "what is this tag?" — used by ``instability.tags`` and anything that
    wants to inventory/classify tags without hand-maintained lists."""
    group = monitor_group(tag)
    return TagInfo(
        tag=tag,
        group=group,
        projection=projection_group(tag),
        layer=layer_of(tag),
        family=family_template(tag),
        recognized=group != "other",
    )
