# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import apply_fsdp
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama4.model import Llama4Model
from torchtitan.tools.logging import logger
from torchtitan.models.common.moe import reserve_hybridep_tokens


def parallelize_llama(
    model: Llama4Model,
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
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallelism.full_dtensor:
        raise NotImplementedError("full_dtensor is not supported yet.")

    # CP: wrap inner attention forward BEFORE parallelize() so CP logic
    # runs inside the local_map boundary on local tensors.
    if parallel_dims.cp_enabled:
        apply_cp_to_forward(model, parallel_dims.get_mesh("cp"))

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

    # Set SP size/rank on EP dispatchers for sequence-parallel token splitting.
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

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
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    logger.info("Applied fully_shard to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model
