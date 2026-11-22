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
from torchtitan.distributed.ffn_overlap import enable_ffn_overlap
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


def _has_tied_embedding_weights(model: nn.Module) -> bool:
    embedding = getattr(model, "tok_embeddings", None)
    head = getattr(model, "lm_head", None)
    return getattr(model, "enable_weight_tying", False) or (
        embedding is not None
        and head is not None
        and embedding.weight is head.weight
    )


def _validate_row_owner_request(
    model: GiftMoEModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
) -> None:
    if not parallelism.gfsdp_row_owner_embeddings:
        return
    if parallelism.fsdp_backend != "gfsdp2":
        raise ValueError(
            "gfsdp_row_owner_embeddings requires fsdp_backend='gfsdp2'"
        )
    if (parallel_dims.tp, parallel_dims.pp, parallel_dims.cp) != (1, 1, 1):
        raise ValueError("gfsdp_row_owner_embeddings currently requires TP/PP/CP1")
    if parallel_dims.dp_replicate != 1:
        raise ValueError("gfsdp_row_owner_embeddings does not support HSDP replicas")
    if training.enable_cpu_offload:
        raise ValueError("gfsdp_row_owner_embeddings does not support CPU offload")
    if _has_tied_embedding_weights(model):
        raise ValueError("gfsdp_row_owner_embeddings requires untied input weights")
    if model.tok_embeddings is None:
        raise ValueError("gfsdp_row_owner_embeddings requires tok_embeddings")


def _validate_tail_weight_retention_request(parallelism: ParallelismConfig) -> None:
    if parallelism.gfsdp_retain_tail_weights and parallelism.fsdp_backend != "gfsdp2":
        raise ValueError("gfsdp_retain_tail_weights requires fsdp_backend='gfsdp2'")


def _validate_mtp_transition_request(
    model: GiftMoEModel,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
) -> None:
    if not parallelism.gfsdp_mtp_transition_prefetch:
        return
    if parallelism.fsdp_backend != "gfsdp2":
        raise ValueError("gfsdp_mtp_transition_prefetch requires fsdp_backend='gfsdp2'")
    if not parallelism.gfsdp_split_mtp_ffn:
        raise ValueError("gfsdp_mtp_transition_prefetch requires gfsdp_split_mtp_ffn")
    if parallel_dims.pp != 1:
        raise ValueError("gfsdp_mtp_transition_prefetch currently requires PP1")
    mtp = getattr(model, "mtp", None)
    if mtp is None or "0" not in mtp.layers:
        raise ValueError("gfsdp_mtp_transition_prefetch requires mtp.layers.0")
    if len(mtp.layers) != 1:
        raise ValueError(
            "gfsdp_mtp_transition_prefetch currently requires exactly one MTP layer"
        )
    if not model.layers:
        raise ValueError("gfsdp_mtp_transition_prefetch requires a trunk layer")
    if model.norm is None or model.lm_head is None:
        raise ValueError(
            "gfsdp_mtp_transition_prefetch requires model.norm and model.lm_head"
        )


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

    _validate_row_owner_request(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
    )
    _validate_tail_weight_retention_request(parallelism)
    if parallelism.gfsdp_compact_expert_owner:
        from torchtitan.distributed.gfsdp._owner_transport import require_recipe

        require_recipe(parallel_dims, parallelism, training, compile_config, ac_config)
    if parallelism.gfsdp_fp8_expert_weights and parallelism.fsdp_backend != "gfsdp2":
        raise ValueError("gfsdp_fp8_expert_weights requires fsdp_backend='gfsdp2'")
    if parallelism.gfsdp_direct_expert_dw and (
        parallelism.fsdp_backend != "gfsdp2"
        or not parallel_dims.ep_enabled
        or (parallel_dims.tp, parallel_dims.pp, parallel_dims.cp) != (1, 1, 1)
    ):
        raise ValueError("gfsdp_direct_expert_dw requires GFSDP2 with EP and TP1/PP1/CP1")
    _validate_mtp_transition_request(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
    )

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

    split_ffn = parallelism.gfsdp_split_layer0_ffn or parallelism.gfsdp_split_mtp_ffn
    if split_ffn:
        if parallelism.fsdp_backend != "gfsdp2":
            raise ValueError("GFSDP FFN overlap requires fsdp_backend='gfsdp2'")
        if parallel_dims.tp > 1 or parallel_dims.pp > 1 or parallel_dims.cp > 1:
            raise ValueError("GFSDP FFN overlap currently requires TP/PP/CP1")
        if parallelism.gfsdp_split_layer0_ffn:
            if "0" not in model.layers:
                raise ValueError("gfsdp_split_layer0_ffn requires layers.0")
            enable_ffn_overlap(model.layers["0"])
        if parallelism.gfsdp_split_mtp_ffn:
            mtp = getattr(model, "mtp", None)
            if mtp is None or "0" not in mtp.layers:
                raise ValueError("gfsdp_split_mtp_ffn requires mtp.layers.0")
            enable_ffn_overlap(mtp.layers["0"])

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
        apply_gfsdp(
            model,
            dp_mesh,
            reuse_bf16_expert_weights=(parallelism.gfsdp_reuse_bf16_expert_weights),
            fp8_expert_weights=parallelism.gfsdp_fp8_expert_weights,
            direct_expert_dw=parallelism.gfsdp_direct_expert_dw,
            row_owner_embeddings=parallelism.gfsdp_row_owner_embeddings,
            retain_tail_weights=parallelism.gfsdp_retain_tail_weights,
            mtp_transition_prefetch=parallelism.gfsdp_mtp_transition_prefetch,
            **fsdp_kwargs,
        )
        model._get_fsdp_state()._owner_transport_requested = (
            parallelism.gfsdp_compact_expert_owner
        )
    elif parallelism.fsdp_backend == "fsdp2":
        if parallelism.gfsdp_reuse_bf16_expert_weights:
            logger.warning(
                "parallelism.gfsdp_reuse_bf16_expert_weights requires "
                "fsdp_backend='gfsdp2'; using the ordinary fsdp2 path"
            )
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
    reuse_bf16_expert_weights: bool = False,
    fp8_expert_weights: bool = False,
    direct_expert_dw: bool = False,
    row_owner_embeddings: bool = False,
    retain_tail_weights: bool = False,
    mtp_transition_prefetch: bool = False,
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

    mtp = getattr(model, "mtp", None)
    if mtp_transition_prefetch:
        if pp_enabled:
            raise ValueError("MTP transition prefetch currently requires PP1")
        if mtp is None or "0" not in mtp.layers:
            raise ValueError("MTP transition prefetch requires mtp.layers.0")
        if len(mtp.layers) != 1:
            raise ValueError(
                "MTP transition prefetch currently requires exactly one MTP layer"
            )
        if getattr(mtp.layers["0"], "_ffn_overlap_regions", None) is None:
            raise ValueError(
                "MTP transition prefetch requires a split mtp.layers.0 FFN"
            )
        if not model.layers:
            raise ValueError("MTP transition prefetch requires a trunk layer")
        if model.norm is None or model.lm_head is None:
            raise ValueError(
                "MTP transition prefetch requires model.norm and model.lm_head"
            )

    row_owner = None
    if row_owner_embeddings:
        if dp_mesh.ndim != 1:
            raise ValueError(
                "gfsdp_row_owner_embeddings requires one dense GFSDP shard axis "
                "without HSDP replicas"
            )
        if model.tok_embeddings is None:
            raise ValueError("gfsdp_row_owner_embeddings requires tok_embeddings")
        if _has_tied_embedding_weights(model):
            raise ValueError(
                "gfsdp_row_owner_embeddings requires untied input weights"
            )
        from torchtitan.distributed.gfsdp.row_owner_embedding import (
            RowOwnerEmbedding,
        )

        row_owner = RowOwnerEmbedding.from_embedding(model.tok_embeddings, dp_mesh)
        row_owner.enable_dedicated_nccl_process_group(4)
        model.tok_embeddings = row_owner

    mp_policy = GMixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if row_owner is not None:
        # The same final Parameter identity must be absent from every nested
        # unit and the root, not just from the former embedding wrapper.
        fsdp_config["ignored_params"] = {row_owner.weight}

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )
    cache_requested_groups = 0
    fp8_requested_groups = 0
    direct_dw_groups = 0

    def shard_block_unit(unit: nn.Module, block: nn.Module, fqn: str) -> None:
        nonlocal cache_requested_groups, fp8_requested_groups, direct_dw_groups
        if not block.moe_enabled:
            split = getattr(unit, "_ffn_overlap_regions", None) is not None
            if split:
                # Wrap the original w1/w2/w3 owner before excluding it from
                # the remainder parent. No parameter or FQN is replaced.
                g_fully_shard(
                    block.feed_forward,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
            parent = g_fully_shard(
                unit, **fsdp_config, reshard_after_forward=reshard_after_forward
            )
            if split:
                parent.set_buffer_region_child(block.feed_forward)
                ffn_fqn = f"{fqn}.{'block.' if unit is not block else ''}feed_forward"
                logger.info(
                    f"GFSDP FFN overlap active: {ffn_fqn} (w1/w2/w3 together); "
                    "FFN/remainder use disjoint regions of existing data/grad slots"
                )
            return
        assert hasattr(block, "moe")

        param_meshes = None
        expert_params: set[nn.Parameter] | None = None
        if ep_degree > 1:
            assert edp_mesh is not None
            expert_params = set(block.moe.experts.parameters())
            param_meshes = [(edp_mesh, expert_params)]

        cache_group_requested = (
            reuse_bf16_expert_weights
            and not pp_enabled
            and ep_degree > 1
            and param_dtype == torch.bfloat16
            and expert_params is not None
        )
        gfsdp_unit = g_fully_shard(
            unit,
            **fsdp_config,
            param_meshes=param_meshes,
            bf16_cache_params=(expert_params if cache_group_requested else None),
            reshard_after_forward=reshard_after_forward,
        )
        if cache_group_requested and any(
            param_group._bf16_cache_requested
            for param_group in gfsdp_unit._get_fsdp_state()._fsdp_param_groups
        ):
            cache_requested_groups += 1
        if (
            (fp8_expert_weights or direct_expert_dw)
            and not pp_enabled
            and expert_params is not None
        ):
            from torchtitan.models.common.moe.dispatch.hybridep import (
                HybridEPTokenDispatcher,
            )
            from torchtitan.models.common.moe.experts.gmm import GroupedExperts

            experts = block.moe.experts
            if (
                isinstance(experts, GroupedExperts)
                and experts._use_float8_grouped_gemm
                and isinstance(experts.token_dispatcher, HybridEPTokenDispatcher)
            ):
                for group in gfsdp_unit._get_fsdp_state()._fsdp_param_groups:
                    if all(
                        param._module_info.module is experts
                        for param in group.fsdp_params
                    ):
                        if fp8_expert_weights:
                            group.enable_fp8_weight_cache()
                            fp8_requested_groups += 1
                        if direct_expert_dw:
                            group._direct_expert_dw_requested = True
                            direct_dw_groups += 1

    if getattr(model, "enable_weight_tying", False):
        raise ValueError("gfsdp2 does not support weight tying")

    if model.tok_embeddings is not None and row_owner is None:
        g_fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for name, transformer_block in model.layers.items():
        shard_block_unit(transformer_block, transformer_block, f"layers.{name}")

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

    if mtp is not None:
        for name, mtp_layer in mtp.layers.items():
            shard_block_unit(mtp_layer, mtp_layer.block, f"mtp.layers.{name}")

    # No ``reshard_after_forward`` on purpose: ``None`` turns on the root's
    # automatic mode, which governs every unit.
    root = g_fully_shard(model, **fsdp_config)
    root.set_retain_tail_weights(retain_tail_weights)
    if fp8_expert_weights and fp8_requested_groups == 0:
        logger.warning(
            "No GIFT HybridEP FP8 expert group could request GFSDP FP8 weights"
        )
    if direct_expert_dw and direct_dw_groups == 0:
        raise ValueError("No GIFT HybridEP FP8 expert group could use direct dW")

    if reuse_bf16_expert_weights and cache_requested_groups == 0:
        logger.warning(
            "BF16 expert weight reuse was requested, but this GIFT MoE "
            "configuration has no supported PP1/EP>1/BF16 expert group; "
            "using the ordinary GFSDP path"
        )

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

    units = [
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

    def entered_units(unit: GFSDPModule) -> list[GFSDPModule]:
        if getattr(unit, "_ffn_overlap_regions", None) is None:
            return [unit]
        block = getattr(unit, "block", unit)
        return [unit, block.feed_forward]

    chain = [entered for unit in units for entered in entered_units(unit)]

    # Data buffers are dealt along this order; without it two units adjacent in
    # forward can share one and zero-3 dies on step 1.
    root.set_shard_order(chain)

    if mtp_transition_prefetch:
        assert mtp is not None
        last_trunk_layer = model.layers[list(model.layers.keys())[-1]]
        root.set_buffer_region_transition(
            source=mtp.layers["0"].block.feed_forward,
            data_target=last_trunk_layer,
            grad_target=norm,
        )

    if row_owner is not None:
        row_owner.attach_comm_context(root)

    # D2H syncs in the expert path interfere with the implicit prefetch.
    if ep_degree == 1 and not mtp_transition_prefetch:
        return

    # An explicit list REPLACES the implicit prefetch, so a unit missing from the
    # chain is worse than un-prefetched.
    def _link(source: GFSDPModule, target: GFSDPModule, pass_type: str) -> None:
        getattr(source, f"set_modules_to_{pass_type}_prefetch")([target])

    # EP1 uses implicit forward prefetch, including inference that skips MTP.
    if ep_degree > 1:
        for source, target in zip(chain, chain[1:]):
            _link(source, target, "forward")

    # A nested parent enters backward on the suffix gradient, before its FFN.
    # Reverse top-level units, retaining parent-before-child within each one.
    reverse = [
        entered for unit in reversed(units) for entered in entered_units(unit)
    ]
    for source, target in zip(reverse, reverse[1:]):
        _link(source, target, "backward")

    if mtp_transition_prefetch:
        assert head is not None and mtp is not None
        mtp_layer = mtp.layers["0"]
        head.set_modules_to_backward_prefetch([mtp_layer, mtp_layer.block.feed_forward])
