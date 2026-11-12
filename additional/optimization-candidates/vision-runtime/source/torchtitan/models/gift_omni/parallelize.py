"""Apply distributed execution policies to the native GIFT Omni model.

This module configures activation checkpointing, compilation, FSDP/HSDP, and
expert parallelism for the decoder and vision blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torchtitan.config import (
        ActivationCheckpointConfig,
        CompileConfig,
        ParallelismConfig,
        TrainingConfig,
    )
    from torchtitan.distributed import ParallelDims
    from torchtitan.models.gift_omni.model import GiftOmniModel


def apply_ac(*args: Any, **kwargs: Any) -> Any:
    """Apply activation checkpointing after its implementation is requested."""
    from torchtitan.distributed.activation_checkpoint import apply_ac as _apply_ac

    return _apply_ac(*args, **kwargs)


def apply_compile(model: GiftOmniModel, compile_config: CompileConfig) -> None:
    """Compile the text decoder while keeping modality components eager."""
    from torchtitan.distributed.compile import apply_compile as _apply_compile
    from torchtitan.protocols.module import ModuleDict
    from torchtitan.tools.logging import logger

    policy = (compile_config.scope, compile_config.backend, compile_config.dynamic)
    previous = getattr(model, "_omni_compile_policy", None)
    if previous is not None:
        if previous != policy:
            raise ValueError(
                f"Model already compiled with {previous}; cannot change to {policy}."
            )
        return

    # This container references existing decoder modules without changing the
    # model tree, parameter identities, or checkpoint names.
    text_modules = ModuleDict(
        {
            name: module
            for name, module in model.named_children()
            if name not in {"encoders", "adapters", "heads"}
        }
    )
    _apply_compile(text_modules, compile_config)
    logger.info("Omni compilation applies only to the text decoder")
    model._omni_compile_policy = policy  # pyrefly: ignore [bad-argument-type]


def apply_fsdp(*args: Any, **kwargs: Any) -> Any:
    """Apply FSDP after its implementation is requested."""
    from torchtitan.distributed.fsdp import apply_fsdp as _apply_fsdp

    return _apply_fsdp(*args, **kwargs)


def apply_gfsdp(
    model: GiftOmniModel,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool,
    reshard_after_forward_policy: str,
    forward_prefetch: bool,
    ep_degree: int,
    edp_mesh: DeviceMesh | None,
    enable_symm_mem: bool,
    coalesce_block_groups: tuple[str, ...],
) -> None:
    """Compose modality shard units with the existing GIFT GFSDP2 policy."""
    from torchtitan.distributed.fsdp import (
        _build_shard_units,
        get_fsdp_reshard_after_forward_policy,
    )
    from torchtitan.distributed.gfsdp import (
        fully_shard,
        GFSDPModule,
        MixedPrecisionPolicy,
    )
    from torchtitan.models.gift_moe._base.parallelize import (
        apply_gfsdp as _apply_gift_gfsdp,
    )

    # Validate before wrapping any modality; these are existing backend limits.
    if cpu_offload:
        raise ValueError("CPU offload is not implemented in GFSDP2.")
    if enable_symm_mem:
        raise ValueError("GFSDP2 does not support FSDP symmetric memory.")
    if getattr(model, "enable_weight_tying", False):
        raise ValueError("GFSDP2 does not support weight tying.")
    if ep_degree > 1 and edp_mesh is None:
        raise ValueError("GFSDP2 requires edp_mesh with expert parallelism.")
    if pp_enabled:
        raise ValueError("GIFT Omni does not support pipeline parallelism.")

    segments = list(model.named_segments())
    known_groups = {segment.group for segment in segments if segment.is_block}
    if unknown := set(coalesce_block_groups) - known_groups:
        raise ValueError(f"Unknown modality block groups: {sorted(unknown)}.")
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )
    units = _build_shard_units(
        segments,
        reshard_after_forward=reshard_after_forward,
        output_reshard_after_forward=reshard_after_forward,
        coalesce_groups=frozenset(coalesce_block_groups),
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    for unit in units:
        if unit.fqn.startswith(("encoders.", "adapters.", "heads.")):
            fully_shard(
                unit.modules,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=unit.reshard_after_forward,
            )

    # The Omni root adopts the GIFT decoder's attributes. Reuse its EP meshes,
    # separate norm/head units and the dedicated chunked-loss gradient buffer.
    _apply_gift_gfsdp(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=pp_enabled,
        cpu_offload=cpu_offload,
        reshard_after_forward_policy=reshard_after_forward_policy,
        ep_degree=ep_degree,
        edp_mesh=edp_mesh,
        enable_symm_mem=enable_symm_mem,
    )

    chain = []
    for unit in units:
        if unit.is_output:
            # A chunked loss re-enters the head, so GFSDP keeps output modules
            # separate rather than sharing a grouped forward hook with norm.
            chain.extend((module, module) for module in unit.modules)
        else:
            chain.append((unit.forward_handle, unit.backward_handle))

    # The loss applies lm_head after the root forward, including any MTP layers.
    head_units = [pair for pair in chain if pair[0] is model.lm_head]
    chain = [pair for pair in chain if pair[0] is not model.lm_head] + head_units
    if not isinstance(model, GFSDPModule) or any(
        not isinstance(module, GFSDPModule) for pair in chain for module in pair
    ):
        raise RuntimeError("GFSDP2 did not wrap every Omni shard unit.")
    model.set_shard_order([forward for forward, _ in chain])

    # Replace the text-only links before the first forward. Modality units must
    # participate in both chains and in data-buffer ordering.
    for index, (forward, _) in enumerate(chain):
        target = (
            [chain[index + 1][0]] if forward_prefetch and index + 1 < len(chain) else []
        )
        forward.set_modules_to_forward_prefetch(target)
    # Embeddings and modality towers are independent autograd branches. The
    # backward scheduler need not visit them in reverse first-use order.
    private = {model.tok_embeddings}
    for unit in units:
        if unit.fqn.startswith(("encoders.", "adapters.", "heads.")):
            private.add(unit.forward_handle)
    for forward, handle in chain:
        if forward in private:
            forward.set_own_data_buffer()
        # A self-target suppresses GFSDP's implicit reverse-forward prefetch.
        handle.set_modules_to_backward_prefetch([forward])

    def set_backward_chain(pairs):
        backward = list(reversed(pairs))
        for index, (forward, handle) in enumerate(backward):
            target = backward[index + 1][0] if index + 1 < len(backward) else forward
            handle.set_modules_to_backward_prefetch([target])

    text_chain = [
        pair
        for pair in chain
        if pair[0] not in private or pair[0] is model.tok_embeddings
    ]
    set_backward_chain(text_chain)

    from torchtitan.models.gift_omni.modalities.data_flow import ComponentStage

    branches = []
    for modality in model.modalities:
        prefixes = tuple(
            f"{prefix}.{route.component}"
            for stage, prefix in (
                (ComponentStage.ENCODER, "encoders"),
                (ComponentStage.ADAPTER, "adapters"),
            )
            for route in modality.config.data_flow.routes(stage)
        )
        branch_handles = {
            unit.forward_handle
            for unit in units
            if any(
                unit.fqn == prefix or unit.fqn.startswith(prefix + ".")
                for prefix in prefixes
            )
        }
        branches.append([pair for pair in chain if pair[0] in branch_handles])
    # A reused tower may have multiple backward entries. Do not speculate into
    # that branch; its private buffers still permit either execution order.
    counts = {}
    for branch in branches:
        for forward, _ in branch:
            counts[forward] = counts.get(forward, 0) + 1
    for branch in branches:
        if all(counts[forward] == 1 for forward, _ in branch):
            set_backward_chain(branch)


def _apply_audio_ac(
    model: GiftOmniModel, *, ac_config: ActivationCheckpointConfig
) -> None:
    """Checkpoint native Conformer layers while keeping the tower one FSDP unit."""
    if ac_config.mode != "full":
        return
    from torchtitan.distributed.activation_checkpoint import _apply_full_ac
    from torchtitan.models.gift_omni.modalities.encoders.conformer import (
        ConformerEncoder,
    )

    for encoder in model.encoders.values():
        if isinstance(encoder, ConformerEncoder):
            for index, layer in enumerate(encoder.encoder.layers):
                encoder.encoder.layers[index] = _apply_full_ac(layer, ac_config)


def _vision_block_groups(model: GiftOmniModel) -> tuple[str, ...]:
    """Return vision block groups in the Omni model."""
    return tuple(
        dict.fromkeys(
            segment.group
            for segment in model.named_segments()
            if segment.is_block and segment.fqn.startswith("encoders.")
        )
    )


def _configure_native_fla_autograd(
    model: GiftOmniModel,
    *,
    ac_config: ActivationCheckpointConfig,
    compile_config: CompileConfig,
) -> None:
    """Resolve GDN autograd from final policies, including CLI overrides.

    Full AC already recomputes each block. Native FLA autograd avoids the
    custom ops' additional forward inside backward. Selective AC and compiled
    attention still require those opaque ops, so retain them for those modes.
    """
    from torchtitan.models.common.attention.linear.gdn import GatedDeltaNet
    from torchtitan.tools.logging import logger

    requested = model.fla_native_autograd
    if requested is not None and not isinstance(requested, bool):
        raise ValueError("model.fla_native_autograd must be true, false, or null.")
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    compatible = ac_config.mode == "full" and (
        not model_compile_enabled or compile_config.scope == "submodules"
    )
    if requested is True and not compatible:
        raise ValueError(
            "model.fla_native_autograd=true requires activation_checkpoint.mode=full "
            "and either model compilation disabled or compile.scope=submodules."
        )
    enabled = compatible if requested is None else requested
    count = 0
    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            module.set_native_fla_autograd(enabled)
            count += 1
    if count:
        logger.info(
            "GDN native FLA autograd: %s (%d layers; QKV conv, scan, gated RMSNorm)",
            "enabled" if enabled else "disabled",
            count,
        )


def parallelize_gift_omni(
    model: GiftOmniModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
) -> GiftOmniModel:
    """Apply EP to GIFT and AC/FSDP/HSDP to the complete Omni model tree."""
    from torchtitan.config import TORCH_DTYPE_MAP
    from torchtitan.tools.logging import logger

    if parallel_dims.tp_enabled:
        raise ValueError("GIFT Omni does not support tensor parallelism.")
    if parallel_dims.cp_enabled:
        raise ValueError("GIFT Omni does not support context parallelism.")
    if parallelism.full_dtensor:
        raise NotImplementedError("GIFT Omni does not support full_dtensor yet.")

    _configure_native_fla_autograd(
        model, ac_config=ac_config, compile_config=compile_config
    )

    if parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

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
        _apply_audio_ac(model, ac_config=ac_config)
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

    match parallelism.fsdp_backend:
        case "fsdp2":
            shard = apply_fsdp
        case "gfsdp2":
            shard = apply_gfsdp
        case backend:
            raise ValueError(f"Unsupported Omni FSDP backend: {backend}.")

    shard(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        forward_prefetch=parallelism.fsdp_forward_prefetch,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
        coalesce_block_groups=_vision_block_groups(model),
    )

    logger.info("Applied %s fully_shard to GIFT Omni", parallelism.fsdp_backend)
    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to GIFT Omni")
    return model
