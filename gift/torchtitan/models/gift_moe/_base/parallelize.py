# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

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
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.gfsdp import (
    fully_shard as g_fully_shard,
    MixedPrecisionPolicy as GMixedPrecisionPolicy,
    GFSDPModule,
)
from torchtitan.models.gift_moe import GiftMoEModel
from torchtitan.tools.logging import logger
from torchtitan.models.common.moe import reserve_hybridep_tokens


def parallelize_gift_moe(
    model: GiftMoEModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
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

    fsdp_kwargs = {
        "param_dtype": TORCH_DTYPE_MAP[training.mixed_precision_param],
        "reduce_dtype": TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        "pp_enabled": parallel_dims.pp_enabled,
        "cpu_offload": training.enable_cpu_offload,
        "reshard_after_forward_policy": parallelism.fsdp_reshard_after_forward,
        "ep_degree": parallel_dims.ep,
        "edp_mesh": edp_mesh,
        "enable_symm_mem": parallelism.enable_fsdp_symm_mem,
    }

    if parallelism.fsdp_backend == "gfsdp2":
        apply_gfsdp(model, dp_mesh, **fsdp_kwargs)
    elif parallelism.fsdp_backend == "fsdp2":
        apply_fsdp(model, dp_mesh, **fsdp_kwargs)
    else:
        raise ValueError(
            "wrong fsdp backend found, support only gfsdp2,"
            + f"fsdp2 but found {parallelism.fsdp_backend}"
        )

    logger.info("Applied fully_shard to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")
    return model


def apply_gfsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    enable_symm_mem: bool = False,
):
    """
    Apply data parallelism (via GFSDP2) to the model.
    """

    if ep_degree > 1 and edp_mesh is None:
        raise ValueError("requires edp_mesh when expert parallelism is enabled")
    if cpu_offload:
        raise ValueError("CPU offload is not realized in GFSDP yet")
    if enable_symm_mem:
        raise ValueError("GFSDP does not support FSDP symmetric memory yet")

    mp_policy = GMixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    def shard_block_unit(unit: nn.Module, block: nn.Module) -> None:
        if not block.moe_enabled:
            g_fully_shard(
                unit, **fsdp_config, reshard_after_forward=reshard_after_forward
            )
            return
        assert hasattr(block, "moe")

        param_meshes = None
        if ep_degree > 1:
            assert edp_mesh is not None
            param_meshes = [(edp_mesh, set(block.moe.experts.parameters()))]

        g_fully_shard(
            unit,
            **fsdp_config,
            param_meshes=param_meshes,
            reshard_after_forward=reshard_after_forward,
        )

    if getattr(model, "enable_weight_tying", False):
        raise ValueError("gfsdp2 does not support weight tying")

    if model.tok_embeddings is not None:
        g_fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for transformer_block in model.layers.values():
        shard_block_unit(transformer_block, transformer_block)

    norm, lm_head = model.norm, model.lm_head
    head: GFSDPModule | None = None
    if norm is not None and lm_head is not None:
        g_fully_shard(
            norm,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        head = g_fully_shard(
            lm_head,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        for mtp_layer in mtp.layers.values():
            shard_block_unit(mtp_layer, mtp_layer.block)

    # No ``reshard_after_forward`` on purpose: ``None`` turns on the root's
    # automatic mode, which governs every unit.
    root = g_fully_shard(model, **fsdp_config)

    # TODO (fedorovgv): still not quite sure about this solution - we should
    #   investigate whether two separate buffers are really needed here
    if head is not None and not pp_enabled:
        # The chunked loss holds an unreduced gradient across chunks, and the
        # shared buffer cannot carry it.
        head.set_own_grad_buffer()

    if pp_enabled:
        # The schedule suppresses sync for the whole stage, so every group needs
        # its own buffer. Costs the sum of them.
        root.set_allow_no_grad_reduce(True)

    # Disable FSDP's automatic gradient division for all FSDP modules
    for module in model.modules():
        if isinstance(module, GFSDPModule):
            module.set_gradient_divide_factor(1.0)

    chain = [
        module
        for module in (
            model.tok_embeddings,
            *model.layers.values(),
            norm,
            *(mtp.layers.values() if mtp is not None else ()),
            lm_head,
        )
        if isinstance(module, GFSDPModule)
    ]

    # Data buffers are dealt along this order; without it two units adjacent in
    # forward can share one and zero-3 dies on step 1.
    root.set_shard_order(chain)

    # D2H syncs in the expert path interfere with the implicit prefetch.
    if ep_degree == 1:
        return

    # An explicit list REPLACES the implicit prefetch, so a unit missing from the
    # chain is worse than un-prefetched.
    def _link(source: GFSDPModule, target: GFSDPModule, pass_type: str) -> None:
        getattr(source, f"set_modules_to_{pass_type}_prefetch")([target])

    for source, target in zip(chain, chain[1:]):
        _link(source, target, "forward")

    # Backward walks the same chain in reverse.
    reverse = list(reversed(chain))
    for source, target in zip(reverse, reverse[1:]):
        _link(source, target, "backward")
