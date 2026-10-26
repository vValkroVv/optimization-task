# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

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
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
)
from torchtitan.tools.logging import logger


def parallelize_flux(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
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

    if parallel_dims.cp_enabled:
        apply_cp(model, parallel_dims.get_mesh("cp"))

    if model_compile_enabled:
        apply_compile(model, compile_config)

    dp_mesh = parallel_dims.get_activated_mesh(["dp_replicate", "fsdp"])
    assert dp_mesh is not None
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cpu_offload=training.enable_cpu_offload,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    logger.info("Applied fully_shard to the model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    enable_symm_mem: bool = False,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    TODO: fold into ``distributed.fsdp.apply_fsdp``, which derives its groups from
    ``named_segments()`` and would cover flux as-is. Two deltas need a
    numerics/perf check first, not a rename: this leaves ``cast_forward_inputs``
    at its default (the shared one sets it False), and it gives each of the four
    input projections its own group where the shared policy coalesces the leading
    run into one.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        cpu_offload (bool): Whether to offload model parameters to CPU. Defaults to False.
        enable_symm_mem (bool): Whether to enable symmetric-memory FSDP communication.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    linear_layers = [
        model.img_in,
        model.time_in,
        model.vector_in,
        model.txt_in,
    ]
    for layer in linear_layers:
        fully_shard(layer, **fsdp_config)

    for block in model.double_blocks:
        fully_shard(
            block,
            **fsdp_config,
        )

    for block in model.single_blocks:
        fully_shard(
            block,
            **fsdp_config,
        )

    # apply FSDP to last layer. Set reshard_after_forward=False for last layer to avoid gather right after reshard
    fully_shard(model.final_layer, **fsdp_config, reshard_after_forward=False)

    # Wrap all the rest of model
    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)


def apply_cp(model: nn.Module, cp_mesh: DeviceMesh) -> None:
    """
    Apply context parallelism to the Flux model.

    Args:
        model: The Flux model with double_blocks and single_blocks containing
            inner attention modules.
        cp_mesh: Device mesh for context parallel dimension

    Note:
        - Uses SDPA attention type
        - Applies to all inner_attention modules in double_blocks and single_blocks
    """
    # Found by kernel type, covering the double-stream blocks' three sites and
    # the single-stream blocks' one alike.
    apply_cp_to_forward(model, cp_mesh)

    logger.info("Applied Context Parallel to the Flux model")


def parallelize_encoders(
    t5_model: nn.Module,
    clip_model: nn.Module,
    parallel_dims: ParallelDims,
    *,
    training: TrainingConfig,
    enable_symm_mem: bool = False,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )

    dp_mesh = parallel_dims.get_activated_mesh(["dp_replicate", "fsdp"])
    assert dp_mesh is not None
    fsdp_config: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": mp_policy,
    }
    if training.enable_cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # NOTE: only apply FSDP to the T5 encoder, not the CLIP text encoder.
    # CLIP Text encoder has low computation / communication ratio, so it's not necessary to apply FSDP to it.
    for block in t5_model.hf_module.encoder.block:
        fully_shard(block, **fsdp_config)
    fully_shard(t5_model.hf_module, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(t5_model.hf_module)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(t5_model.hf_module)

    logger.info("Applied fully_shard to the T5 encoder model")

    return t5_model, clip_model
