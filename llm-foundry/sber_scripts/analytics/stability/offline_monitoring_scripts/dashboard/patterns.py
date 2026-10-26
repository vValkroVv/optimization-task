"""Activation / gradient pattern definitions and auto-discovery from TensorBoard tags."""

from __future__ import annotations

import re

_DYNAMIC_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]

MTP_PNAME = "mtp_mlp_down_input"

MAXVIO_COLOR = "#000000"
_MAX_VIO_PREFIXES = (
    "dispatcher_layer_experts_max_vio_batch_avg",
    "dispatcher_layer_experts_max_vio_batch_no_pad_avg",
    "dispatcher_layer_experts_max_vio_batch_no_pad_no_prefix_avg",
    "dispatcher_layer_experts_max_vio_microbatch_avg",
)
MAX_VIO_VARIANTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(rf"^{p}/layer_(?P<L>\d+)$"),
        "max_vio__" + p.split("max_vio_")[1],
    )
    for p in _MAX_VIO_PREFIXES
]

# Static hand-picked activation pattern tails.
# Regex tail must contain a named capture group ``(?P<L>\d+)`` for layer index.
ACTIVATION_PATTERN_TAILS: dict[str, str] = {
    # block_sparse_moe internals
    "fused_swiglu_input": r"model\.model\.layers\.(?P<L>\d+)\.block_sparse_moe\.fused_swiglu_input\.0",
    "gmm_down_input": r"model\.model\.layers\.(?P<L>\d+)\.block_sparse_moe\.gmm_down_input\.0",
    "gmm_down_output": r"model\.model\.layers\.(?P<L>\d+)\.block_sparse_moe\.gmm_down_output\.0",
    "gmm_upgate_input": r"model\.model\.layers\.(?P<L>\d+)\.block_sparse_moe\.gmm_upgate_input\.0",
    "block_sparse_moe_output": r"model\.model\.layers\.(?P<L>\d+)\.block_sparse_moe_output\.0",
    # top-level mlp (non-MoE layers)
    "mlp_up_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.mlp\.up_proj_input\.0",
    # layer-level
    "layer_output": r"model\.model\.layers\.(?P<L>\d+)_output\.0",
    # residual-connection tracking: how much each sublayer writes into the residual
    # (``*_residual_contribution``) and the residual stream after it (``*_residual``).
    "attn_residual_contribution": r"model\.model\.layers\.(?P<L>\d+)\.attn_residual_contribution_output\.0",
    "attn_residual": r"model\.model\.layers\.(?P<L>\d+)\.attn_residual_output\.0",
    "mlp_residual_contribution": r"model\.model\.layers\.(?P<L>\d+)\.mlp_residual_contribution_output\.0",
    "mlp_residual": r"model\.model\.layers\.(?P<L>\d+)\.mlp_residual_output\.0",
    # attention projections
    "attn_ba_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.ba_proj_input\.0",
    "attn_dkv_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.dkv_proj_input\.0",
    "attn_dq_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.dq_proj_input\.0",
    "attn_gate_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.gate_proj_input\.0",
    "attn_kr_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.kr_proj_input\.0",
    "attn_o_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.o_proj_input\.0",
    "attn_qkv_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.qkv_proj_input\.0",
    "attn_uk_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.uk_proj_input\.0",
    "attn_uq_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.uq_proj_input\.0",
    "attn_uv_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.uv_proj_input\.0",
    "attn_z_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.z_proj_input\.0",
    "qkv_conv1d_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn\.qkv_conv1d_input\.0",
    "attn_output_last": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._attn_output\.0",
    # gate
    "gate_gate_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._gate\.gate_input\.0",
    "gate_gate_output": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._gate\.gate_output\.0",
    "gate_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\._gate_input\.0",
    # layernorms
    "input_layernorm_input_last": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.input_layernorm_input\.0",
    "post_attention_layernorm": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.post_attention_layernorm_input\.0",
    # dense MLP inside self_attn
    "dense_mlp_gate_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.mlp\.gate_proj_input\.0",
    "dense_mlp_up_proj_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.mlp\.up_proj_input\.0",
    "dense_mlp_down_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.mlp\.down_proj_input\.0",
    "mlp_output": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.mlp_output\.0",
    # shared experts
    "shared_experts_gate_proj": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.shared_experts\.gate_proj_input\.0",
    "shared_experts_up_proj": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.shared_experts\.up_proj_input\.0",
    "shared_experts_down_input": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.shared_experts\.down_proj_input\.0",
    "shared_experts_output": r"model\.model\.layers\.(?P<L>\d+)\.self_attn\.shared_experts_output\.0",
}
ACTIVATION_MTP_TAIL = (
    r"model\.model\.mtp_block\.mtp_layers\.0\.decoder_layer\.mlp\.down_proj_input\.0"
)
ACTIVATION_PATTERN_COLORS: dict[str, str] = {
    "gmm_down_input": "#1f77b4",
    "gmm_down_output": "#ff7f0e",
    "gmm_upgate_input": "#aec7e8",
    "fused_swiglu_input": "#ffbb78",
    "block_sparse_moe_output": "#98df8a",
    "layer_output": "#2ca02c",
    "attn_residual_contribution": "#e377c2",
    "attn_residual": "#f7b6d2",
    "mlp_residual_contribution": "#ff7f0e",
    "mlp_residual": "#ffbb78",
    "mlp_up_proj_input": "#ff9896",
    "attn_ba_proj_input": "#c5b0d5",
    "attn_o_proj_input": "#9467bd",
    "attn_qkv_proj_input": "#c49c94",
    "attn_z_proj_input": "#f7b6d2",
    "attn_gate_proj_input": "#c7c7c7",
    "attn_kr_proj_input": "#dbdb8d",
    "attn_uk_proj_input": "#9edae5",
    "attn_uq_proj_input": "#17becf",
    "attn_uv_proj_input": "#bcbd22",
    "attn_dkv_proj_input": "#d62728",
    "attn_dq_proj_input": "#e377c2",
    "qkv_conv1d_input": "#17becf",
    "attn_output_last": "#e377c2",
    "gate_gate_input": "#7f7f7f",
    "gate_gate_output": "#8c564b",
    "gate_input": "#aec7e8",
    "input_layernorm_input_last": "#7f7f7f",
    "post_attention_layernorm": "#c7c7c7",
    "dense_mlp_gate_proj_input": "#dbdb8d",
    "dense_mlp_up_proj_input": "#9edae5",
    "dense_mlp_down_input": "#8c564b",
    "mlp_output": "#ff9896",
    "shared_experts_gate_proj": "#bcbd22",
    "shared_experts_up_proj": "#1f77b4",
    "shared_experts_down_input": "#9467bd",
    "shared_experts_output": "#d62728",
}


def color_map(names: list[str]) -> dict[str, str]:
    return {
        name: _DYNAMIC_PALETTE[i % len(_DYNAMIC_PALETTE)]
        for i, name in enumerate(names)
    }


def discover_pattern_tails(
    tags: list[str], metric_prefix: str, min_coverage: int = 3
) -> tuple[dict[str, str], None]:
    """Auto-discover per-layer suffix templates from TensorBoard tags.

    Returns (pattern_tails, None).  Templates seen in < min_coverage layers are dropped.
    """
    rx_main = re.compile(
        rf"^{re.escape(metric_prefix)}_amax/model\.model\.layers\.(\d+)(.+)$"
    )
    main_cov: dict[str, set[int]] = {}
    for t in tags:
        m = rx_main.match(t)
        if m:
            main_cov.setdefault(m.group(2), set()).add(int(m.group(1)))

    kept = sorted(
        [
            (suf, layers)
            for suf, layers in main_cov.items()
            if len(layers) >= min_coverage
        ],
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    pattern_tails: dict[str, str] = {}
    for suf, _ in kept:
        name = suf.lstrip(".").replace(".", "_").replace("/", "_")
        base = name
        i = 2
        while name in pattern_tails:
            name = f"{base}__{i}"
            i += 1
        pattern_tails[name] = rf"model\.model\.layers\.(?P<L>\d+){re.escape(suf)}"
    return pattern_tails, None


def merge_static_with_discovered(
    static: dict[str, str], discovered: dict[str, str]
) -> dict[str, str]:
    """Static keys win on name conflicts; tails already in static are skipped."""
    static_tails = set(static.values())
    out = dict(static)
    for name, tail in discovered.items():
        if tail not in static_tails and name not in out:
            out[name] = tail
    return out


def split_activation_categories(tails: dict[str, str]) -> dict[str, dict[str, str]]:
    """Split tails into ``main`` and ``clip``.

    Only ``main`` is plotted (see collector); the point of the split is to keep
    ``*_clip_share`` tails out of the activation-magnitude heatmaps — they are
    fractions in ``[0, 1]`` on a totally different scale. (There is no ``sigmoid``
    category: the model applies ``torch.sigmoid`` for gate scaling but never logs
    a ``sigmoid``-named activation slot, so the old bucket only ever stayed empty.)
    """
    cats: dict[str, dict[str, str]] = {"main": {}, "clip": {}}
    for name, tail in tails.items():
        if "clip" in name.lower():
            cats["clip"][name] = tail
        else:
            cats["main"][name] = tail
    return cats
