"""Sharding config for GIFT decoder compositions.

Per-layer dispatch: softmax layers reuse the ``_base`` plan; linear-attention
layers shard their norms + FFN/MoE but leave the attention with no TP plan (TP
deferred; FSDP shards its params regardless).
"""

from typing import TYPE_CHECKING


from torchtitan.config.schema.components.lora import unwrap_frozen
from torchtitan.config.schema.protocols.sharding import Placement, Replicate, Shard
from torchtitan.models.common.attention import GQAttention, MLA, SoftmaxAttention
from torchtitan.models.common.decoder_blocks.sharding import (
    dense_param_placement,
    gated_norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
    set_mla_attention_sharding,
)
from torchtitan.models.common.moe.sharding import (
    GROUPED_EXPERTS_PARAM_LAYOUT,
    set_moe_sharding_config,
)
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.common.decoder_blocks import ResidualTransformerBlock
    from torchtitan.models.gift_moe._base.model import GiftMoEModel


def set_gift_moe_sharding_config(
    config: "GiftMoEModel.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all GIFT sub-configs."""
    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    gated = gated_norm_config(enable_sp=enable_sp)
    config.norm.sharding_config = gated
    for layer_cfg in config.placed_layers():
        _set_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)

    if config.mtp is not None:
        _set_gift_mtp_sharding(config.mtp, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_gift_mtp_sharding(
    mtp_cfg,
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Shard the MTP heads.

    Each inner block is sharded by attention family, like a main layer: softmax
    through the shared module-based dispatch. All norms take the gated-norm
    plan (plain norm weights use identical placements); ``input_proj`` keeps a Replicate weight (contracts only the
    feature dim, so a seq-``Shard(1)`` input flows through to a ``Shard(1)`` output).
    """
    gated = gated_norm_config(enable_sp=enable_sp)
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Replicate())},
    )
    for module_cfg in mtp_cfg.placed_layers():
        _set_layer_sharding(module_cfg.block, enable_sp=enable_sp, enable_ep=enable_ep)
        module_cfg.token_norm.sharding_config = gated
        module_cfg.hidden_norm.sharding_config = gated
        module_cfg.final_norm.sharding_config = gated
        module_cfg.input_proj.sharding_config = replicate_weight


def _set_layer_sharding(layer_cfg, *, enable_sp: bool, enable_ep: bool) -> None:
    gated = gated_norm_config(enable_sp=enable_sp)
    for name in ("attention_norm", "ffn_norm", "post_attn_norm", "post_ffn_norm"):
        norm = getattr(layer_cfg, name)
        if norm is not None:
            norm.sharding_config = gated
    if isinstance(unwrap_frozen(layer_cfg.attention), SoftmaxAttention.Config):
        set_softmax_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)
    else:
        # Linear attention has no TP plan; FSDP still shards its parameters.
        _set_layer_ffn_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_layer_ffn_sharding(
    layer_cfg: "ResidualTransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """The FFN/MoE plan shared by softmax and linear-attention blocks."""
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )

    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout=GROUPED_EXPERTS_PARAM_LAYOUT,
        )


def set_softmax_layer_sharding(
    layer_cfg: "ResidualTransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Set sharding on one softmax-attention transformer layer, plus its FFN/MoE.

    The attention plan is per family -- MLA and GQA share the idea (low-rank
    projections and norms replicated, up-projections colwise, ``wo`` rowwise)
    but not the field names. Block norms are set by the shared GIFT caller.
    """
    attention = unwrap_frozen(layer_cfg.attention)
    if isinstance(attention, MLA.Config):
        set_mla_attention_sharding(attention, enable_sp=enable_sp)
    elif isinstance(attention, GQAttention.Config):
        set_gqa_attention_sharding(attention, enable_sp=enable_sp)
        set_gqa_inner_attention_local_map(attention.inner_attention)
    else:
        raise ValueError(
            f"{type(attention).__name__} is not a softmax attention; a linear "
            "family needs its own plan (see _set_layer_ffn_sharding)."
        )

    _set_layer_ffn_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)
