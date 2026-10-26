# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization utilities for Qwen3-VL.

This module applies PT-D parallelisms and various training techniques
(activation checkpointing, compile, FSDP) to the Qwen3-VL model.
"""

import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.tools.utils import TORCH_DTYPE_MAP

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.models.qwen3_vl.model import Qwen3VLModel
from torchtitan.tools.logging import logger
from torchtitan.models.common.moe import reserve_hybridep_tokens


def _apply_tp_to_vision_encoder(
    vision_encoder: nn.Module,
    tp_mesh: DeviceMesh,
):
    """Apply tensor parallelism to the vision encoder.

    Hidden states flow as DTensor(Replicate) throughout — all ranks hold the
    full hidden_states. Only the linear layers (qkv, proj, fc1, fc2) are
    sharded via ColwiseParallel/RowwiseParallel to save memory. Norms operate
    on Replicate DTensors directly.
    """
    # NoParallel on patch_embed distributes its params as Replicate DTensors
    # on tp_mesh for FSDP mesh consistency. Its input hook wraps plain
    # pixel_values as DTensor(Replicate), and the output stays as DTensor
    # (Replicate) to flow through the rest of the vision encoder.
    parallelize_module(vision_encoder, tp_mesh, {"patch_embed": NoParallel()})

    # pos_embed is an nn.Parameter (not a submodule), so it can't be targeted
    # by parallelize_module's plan dict. We distribute it as Replicate DTensor
    # on tp_mesh for FSDP mesh consistency.
    vision_encoder.pos_embed = nn.Parameter(
        distribute_tensor(vision_encoder.pos_embed.data, tp_mesh, [Replicate()]),
        requires_grad=vision_encoder.pos_embed.requires_grad,
    )

    # TP plan for each vision transformer block.
    # hidden_states flows through as DTensor (Replicate) so residual adds work.
    # NoParallel on norms sets their params as Replicate DTensors on tp_mesh
    # (for consistent (fsdp, tp) mesh after FSDP).
    # RowwiseParallel uses use_local_output=False to return DTensor (Replicate)
    # so residual connections (hidden_states + attn/mlp output) stay in DTensor
    # space. ColwiseParallel uses use_local_output=True (default) to return
    # local shards for the internal attention/MLP computation.
    layer_plan = {
        "norm1": NoParallel(),
        "norm2": NoParallel(),
        "attn.qkv": ColwiseParallel(),  # needs plain tensor for reshape after qkv
        "attn.proj": RowwiseParallel(use_local_output=False),
        "mlp.linear_fc1": ColwiseParallel(use_local_output=False),
        "mlp.linear_fc2": RowwiseParallel(use_local_output=False),
    }

    for transformer_block in vision_encoder.layers.values():
        parallelize_module(transformer_block, tp_mesh, layer_plan)

    # TP plan for patch mergers (main + deepstack).
    # Mergers output DTensor(Replicate) — the model passes padded embeddings
    # directly to vision scatter and DeepStack.
    merger_plan = {
        "norm": NoParallel(),
        "linear_fc1": ColwiseParallel(use_local_output=False),
        "linear_fc2": RowwiseParallel(use_local_output=False),
    }

    parallelize_module(vision_encoder.merger, tp_mesh, merger_plan)
    for merger in vision_encoder.deepstack_merger_list:
        parallelize_module(merger, tp_mesh, merger_plan)

    logger.info("Applied Tensor Parallelism to the vision encoder")


def parallelize_qwen3_vl(
    model: Qwen3VLModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Qwen3-VL model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if parallelism.full_dtensor:
        raise NotImplementedError("full_dtensor is not supported yet.")

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallel_dims.cp_enabled:
        raise NotImplementedError("Context Parallel is not yet supported for Qwen3-VL.")

    if parallel_dims.tp_enabled:
        # TODO(@fegin): Apply TP to vision encoder (still uses parallelize_module path)
        if model.vision_encoder is not None:
            _apply_tp_to_vision_encoder(
                model.vision_encoder, parallel_dims.get_mesh("tp")
            )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)
        reserve_hybridep_tokens(
            model,
            training=training,
            parallelism=parallelism,
            parallel_dims=parallel_dims,
        )

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    # One segment walk covers both towers, and the vision blocks get their real
    # ``vision_encoder.layers.N`` FQNs rather than colliding with the decoder's.
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    # Apply torch.compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config)

    # Apply FSDP / HSDP unconditionally (fully_shard handles dp_shard=1)
    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        # Vision compute is small relative to the decoder's, so per-block
        # all-gathers cost more than one for the whole tower.
        #
        # DELTA vs the previous hand-written path, which wrapped the whole encoder
        # in a single group: the tower now spans ~7 groups (patch_embed,
        # rotary_pos_emb, the coalesced blocks, merger, and one per deepstack
        # merger). Every parameter's mesh and placements are unchanged; what
        # changes is how many all-gathers the tower costs. Also
        # ``vision_encoder.pos_embed`` is a bare parameter on the encoder, so it
        # belongs to no segment and lands in the root group.
        # TODO: measure, then either coalesce the tower's non-block parts too or
        # accept the split.
        coalesce_block_groups=("vision_encoder.layers",),
    )

    logger.info("Applied fully_shard to the Qwen3-VL model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the Qwen3-VL model")

    return model
