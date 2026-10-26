# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.config.schema.components.lora import unwrap_frozen
from torchtitan.config.schema.protocols.sharding import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torchtitan.models.common.attention import (
    FusedQKVLinear,
    GQAttention,
    MLA,
    QKVLinear,
)
from torchtitan.models.common.norms import GatedRMSNorm
from torchtitan.protocols.sharding import LocalMapConfig, NamedPlacement, ShardingConfig
from torchtitan.config.schema.protocols.types import MeshAxisName

DP_REPLICATE = MeshAxisName.DP_REPLICATE
DP_SHARD = MeshAxisName.DP_SHARD
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def dense_param_placement(*, tp: Placement) -> NamedPlacement:
    """Placement for dense-path params/buffers.

    DP/CP axes are ``Replicate`` at ``distribute_tensor`` time; FSDP reshards
    DP_SHARD post-parallelize. TP placement is caller-specified.

    TODO: Ideally placements would be defined on a computation mesh that
    has a single DP axis (no DP_REPLICATE vs DP_SHARD distinction). That
    requires a mesh switch between FSDP storage and computation — likely
    resolved by FlexShard. Revisit once FlexShard lands.
    """
    return {
        DP_REPLICATE: Replicate(),
        DP_SHARD: Replicate(),
        CP: Replicate(),
        TP: tp,
    }


def dense_activation_placement(
    *,
    tp: Placement,
    cp: Placement = Shard(1),
) -> NamedPlacement:
    """Placement for dense-path activations.

    DP axes are batch-sharded (``Shard(0)``). CP defaults to seq-sharded
    (``Shard(1)``); override to ``Replicate()`` for K/V after all-gather.
    TP placement is caller-specified.
    """
    return {
        DP_REPLICATE: Shard(0),
        DP_SHARD: Shard(0),
        CP: cp,
        TP: tp,
    }


def colwise_config() -> ShardingConfig:
    """ColwiseParallel: weight Shard(0), output Shard(-1)."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Shard(0)),
            "bias": dense_param_placement(tp=Shard(0)),
        },
        out_dst_shardings=dense_activation_placement(tp=Shard(-1)),
    )


def rowwise_config(*, output_sp: bool = False) -> ShardingConfig:
    """RowwiseParallel: weight Shard(1), bias Replicate (no-op if bias absent).

    ``output_sp=True``  -> output ``Shard(1)`` (reduce-scatter into SP region).
    ``output_sp=False`` -> output ``Replicate()`` (all-reduce).
    """
    out_tp: Placement = Shard(1) if output_sp else Replicate()
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Shard(1)),
            "bias": dense_param_placement(tp=Replicate()),
        },
        out_dst_shardings=dense_activation_placement(tp=out_tp),
    )


def norm_config(*, enable_sp: bool) -> ShardingConfig:
    """Norm sharding.

    ``enable_sp=True``: SequenceParallel — weight Replicate, activations
    Shard(1) across boundary.
    ``enable_sp=False``: Plain-TP — weight Replicate, activations pass
    through. Still distributes the norm's weight as a DTensor so it
    composes with DTensor activations (otherwise plain Tensor + DTensor
    would mix inside the op).
    """
    state = {"weight": dense_param_placement(tp=Replicate())}
    if not enable_sp:
        return ShardingConfig(state_shardings=state)
    return ShardingConfig(
        state_shardings=state,
        # Key is ``"x"``: RMSNorm subclasses ``nn.RMSNorm``, whose forward
        # parameter is ``x`` (``input`` belongs to nn.LayerNorm).
        in_src_shardings={"x": dense_activation_placement(tp=Shard(1))},
        in_dst_shardings={"x": dense_activation_placement(tp=Shard(1))},
        out_dst_shardings=dense_activation_placement(tp=Shard(1)),
    )


def gated_norm_config(*, enable_sp: bool) -> ShardingConfig:
    """Sharding for a self-gated norm (``GatedRMSNorm``).

    Same placements as :func:`norm_config` (norm weight + activations) but also
    declares the gate-MLP projections (``gate_up_weight`` / ``gate_down_weight``)
    as ``Replicate`` on TP — ``Module._shard_states`` raises if any owned param
    lacks a declared placement. The gate is a per-token hidden->r->hidden MLP, so
    its weights replicate and SP's ``Shard(1)`` activation flows through unchanged.

    The SP activation key is ``"x"`` to match ``GatedRMSNorm.forward(x)``.
    """
    replicate = dense_param_placement(tp=Replicate())
    state = {
        "weight": replicate,
        "gate_up_weight": replicate,
        "gate_down_weight": replicate,
    }
    if not enable_sp:
        return ShardingConfig(state_shardings=state)
    return ShardingConfig(
        state_shardings=state,
        in_src_shardings={"x": dense_activation_placement(tp=Shard(1))},
        in_dst_shardings={"x": dense_activation_placement(tp=Shard(1))},
        out_dst_shardings=dense_activation_placement(tp=Shard(1)),
    )


def set_qkv_linear_sharding(qkv_linear_cfg) -> None:
    """Colwise-shard each Q/K/V projection of a ``BaseQKVLinear``.

    Handles both ``QKVLinear`` (separate ``wq`` + ``wkv``) and
    ``FusedQKVLinear`` (single ``wqkv``).
    """
    qkv_linear_cfg = unwrap_frozen(qkv_linear_cfg)
    if isinstance(qkv_linear_cfg, FusedQKVLinear.Config):
        assert qkv_linear_cfg.wqkv is not None, "partial FusedQKVLinear.Config"
        qkv_linear_cfg.wqkv.sharding_config = colwise_config()
    elif isinstance(qkv_linear_cfg, QKVLinear.Config):
        assert qkv_linear_cfg.wq is not None and qkv_linear_cfg.wkv is not None, (
            "partial QKVLinear.Config"
        )
        qkv_linear_cfg.wq.sharding_config = colwise_config()
        qkv_linear_cfg.wkv.sharding_config = colwise_config()
    else:
        raise TypeError(
            f"set_qkv_linear_sharding requires QKVLinear.Config or "
            f"FusedQKVLinear.Config, got {type(qkv_linear_cfg).__name__}"
        )


def set_gqa_attention_sharding(attention_cfg, *, enable_sp: bool) -> None:
    """Standard GQA attention (``qkv_linear``/``wo``) TP sharding.

    Shared by llama3, qwen3, and llama4 -- all three have a GQA block whose
    ``forward(x, rope_cache, ...)`` takes ``x`` (per-SP layout, gathered to
    Replicate internally) and a plain ``rope_cache`` (annotated Replicate).

    Callers that have additional attention sub-state (e.g. ``qk_norm``,
    ``sinks``) set those after calling this helper.
    """
    attention_cfg = unwrap_frozen(attention_cfg)
    assert isinstance(attention_cfg, GQAttention.Config), (
        f"set_gqa_attention_sharding requires GQAttention.Config, "
        f"got {type(attention_cfg).__name__}"
    )
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": dense_activation_placement(tp=attn_x_placement),
            "rope_cache": dense_param_placement(tp=Replicate()),
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate()),
            "rope_cache": dense_param_placement(tp=Replicate()),
        },
    )
    qkv_linear, wo = attention_cfg._projections()  # resolved in __post_init__
    set_qkv_linear_sharding(qkv_linear)
    wo.sharding_config = rowwise_config(output_sp=enable_sp)


def _replicate_low_rank_norm(norm_cfg) -> ShardingConfig:
    """Replicate-weight sharding for an MLA low-rank norm (no activation plan).

    ``Module._shard_states`` requires every owned param have a declared TP
    placement. Plain ``RMSNorm`` owns only ``weight``; ``GatedRMSNorm`` (the
    gift_3_5 hybrid) also owns the gate-MLP projections. Both replicate on TP;
    ``enable_sp=False`` because these norms act on the low-rank latent, not the
    sequence-sharded activation.
    """
    if isinstance(unwrap_frozen(norm_cfg), GatedRMSNorm.Config):
        return gated_norm_config(enable_sp=False)
    return norm_config(enable_sp=False)


def set_mla_attention_sharding(attention_cfg, *, enable_sp: bool) -> None:
    """MLA attention TP sharding, shared by deepseek_v3 and the gift family.

    Low-rank projections and their norms (``wkv_a``, ``kv_norm``, ``wq_a``,
    ``q_norm``) keep Replicate weights -- still distributed, so DTensor
    activations flow through without mixing plain Tensor + DTensor in the
    matmul. Up-projections (``wkv_b``, ``wq_b``, ``wq``) are colwise and ``wo``
    is rowwise. ``x`` is gathered to Replicate on entry; ``freqs_cis`` is always
    Replicate.
    """
    attention_cfg = unwrap_frozen(attention_cfg)
    assert isinstance(attention_cfg, MLA.Config), (
        f"set_mla_attention_sharding requires MLA.Config, "
        f"got {type(attention_cfg).__name__}"
    )
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": dense_activation_placement(tp=attn_x_placement),
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate()),
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
    )
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Replicate())},
    )
    wkv_a, wkv_b, wo, kv_norm = (
        attention_cfg._projections()
    )  # resolved in __post_init__
    wkv_a.sharding_config = replicate_weight
    kv_norm.sharding_config = _replicate_low_rank_norm(kv_norm)

    wkv_b.sharding_config = colwise_config()
    wo.sharding_config = rowwise_config(output_sp=enable_sp)

    if attention_cfg.gate is not None:
        assert attention_cfg.gate.gate_proj is not None, "partial SigmoidGate.Config"
        attention_cfg.gate.gate_proj.sharding_config = colwise_config()

    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)

    if attention_cfg.q_lora_rank == 0:
        assert attention_cfg.wq is not None
        attention_cfg.wq.sharding_config = colwise_config()
    else:
        # Low-rank: wq_a + q_norm stay Replicate DTensors; wq_b is Colwise.
        assert attention_cfg.wq_a is not None
        assert attention_cfg.wq_b is not None
        assert attention_cfg.q_norm is not None
        attention_cfg.wq_a.sharding_config = replicate_weight
        attention_cfg.q_norm.sharding_config = _replicate_low_rank_norm(
            attention_cfg.q_norm
        )
        attention_cfg.wq_b.sharding_config = colwise_config()


def set_gqa_inner_attention_local_map(
    inner_attention_cfg, *, return_lse: bool = False
) -> None:
    """Install a ``LocalMapConfig`` on an inner-attention config.

    q/k/v arrive as ``(bs, seq, heads, head_dim)`` DTensors with heads
    TP-sharded (``Shard(2)``), regardless of SP. ``local_map`` converts them
    to local tensors before the kernel runs, then wraps outputs back.

    Declares placements over the full dense SPMD axis set (DP/CP/TP) so
    the LocalMap composes under ``full_dtensor`` (where the surrounding
    mesh is multi-axis); under non-full_dtensor, the (tp,)-only mesh only
    consumes the ``TP`` placement and the rest are ignored.

    Under ``full_dtensor`` + CP, q stays seq-sharded on the CP axis
    (``Shard(1)``) while k/v are ``Replicate`` on CP -- DTensor all-gathers
    k/v at the local_map boundary so the kernel sees full-length keys
    (matching the BlockMask's kv dimension). Q's local grad is naturally
    seq-sharded; k/v's local grads accumulate as ``Partial`` on CP and
    DTensor reduces them on the way out.

    ``return_lse=True`` is for kernels that return ``(output, lse)`` (e.g.,
    GPT-OSS's flash attention with ``return_lse=True``); both outputs share
    the same heads-sharded placement.
    """
    q_placements: NamedPlacement = dense_activation_placement(tp=Shard(2))
    kv_placements: NamedPlacement = dense_activation_placement(
        tp=Shard(2), cp=Replicate()
    )
    kv_grad_placements: NamedPlacement = dense_activation_placement(
        tp=Shard(2), cp=Partial()
    )
    out_src: NamedPlacement | tuple[NamedPlacement, ...] = (
        (q_placements, q_placements) if return_lse else q_placements
    )
    inner_attention_cfg.sharding_config = ShardingConfig(
        in_dst_shardings={
            "q": q_placements,
            "k": kv_placements,
            "v": kv_placements,
        },
        out_src_shardings=out_src,
        local_map=LocalMapConfig(
            in_grad_placements=(q_placements, kv_grad_placements, kv_grad_placements),
        ),
    )


def set_dense_ffn_sharding(
    feed_forward_cfg,
    *,
    attn_x_placement: Placement,
    enable_sp: bool,
) -> None:
    """Standard dense FFN (``w1``/``w2``/``w3``) TP sharding.

    Shared by llama3, qwen3, llama4, and deepseek_v3. ``attn_x_placement``
    should match the layout that the layer's attention block emits so the
    FFN's input wrap is a no-op redistribute when placements already agree.
    """
    feed_forward_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": dense_activation_placement(tp=attn_x_placement)},
        in_dst_shardings={"x": dense_activation_placement(tp=Replicate())},
    )
    w1, w2, w3 = feed_forward_cfg._projections()  # resolved in __post_init__
    w1.sharding_config = colwise_config()
    w3.sharding_config = colwise_config()
    w2.sharding_config = rowwise_config(output_sp=enable_sp)


def set_decoder_sharding_config(
    config, *, loss_parallel: bool, enable_sp: bool
) -> None:
    """Set sharding on root-level configs only: ``tok_embeddings``, ``norm``,
    ``output``, and the root ``freqs_cis`` buffer.

    Per-layer sharding (attention, feed_forward, per-layer norms) is the
    caller's responsibility — this helper does not walk ``config.layers``.

    ``enable_sp=True``  -> SequenceParallel: activations are ``Shard(1)`` between
    the embedding, norm, and output layers.
    ``enable_sp=False`` -> activations stay ``Replicate``; root norm is left
    unsharded (equivalent to the legacy ``NoParallel`` plan).
    """
    activation_tp: Placement = Shard(1) if enable_sp else Replicate()
    loss_tp: Placement = Shard(-1) if loss_parallel else Replicate()

    # freqs_cis buffer on the decoder root: Replicate on all axes.
    config.sharding_config = ShardingConfig(
        state_shardings={"freqs_cis": dense_param_placement(tp=Replicate())},
    )
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Shard(0))},
        in_src_shardings={"input": dense_activation_placement(tp=Replicate())},
        in_dst_shardings={"input": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=activation_tp),
    )
    config.norm.sharding_config = norm_config(enable_sp=enable_sp)

    config.lm_head.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Shard(0))},
        in_src_shardings={"input": dense_activation_placement(tp=activation_tp)},
        in_dst_shardings={"input": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=loss_tp),
    )
