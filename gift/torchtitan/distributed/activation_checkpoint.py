# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import fnmatch
from functools import partial
import os

import torch
import torch._functorch.config
import torch.nn as nn
from torch._functorch.partitioners import get_default_op_list
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.config import ActivationCheckpointConfig as ACConfig
from torchtitan.config.configs import (
    SAC_CATEGORIES,
    parse_sac_save_spec as parse_save_spec,
)
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


# DeepGEMM's fp8 grouped GEMM; registered only with fp8 grouped experts.
_FP8_GROUPED_GEMM = "torchtitan.m_grouped_fp8_gemm_nt_contiguous.default"


# --- SAC save categories ------------------------------------------------------
#
# Every op selective AC may save belongs to one category; ``selective_save``
# (or a per-module ``save:<cat>+<cat>`` mode) names the saved ones, the rest
# are recomputed. Always-saved ops are saved regardless: recomputing them is
# either unsafe (topk: expert assignment must match between forward and
# recompute; D2H copies: a sync; fp8 amax: the scale must match) or pointless
# (an inductor-compiled region is the whole block).
def _resolve_ops(op_specs: list) -> set:
    """Op objects from specs: an op (always present) or (root, dotted_path)
    for conditionally registered custom ops, skipped when absent."""
    ops = set()
    for spec in op_specs:
        if isinstance(spec, tuple):
            obj, path = spec
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                ops.add(obj)
            except AttributeError:
                pass
        else:
            ops.add(spec)
    return ops


def _op_categories() -> tuple[dict, set]:
    """Returns ({op: category}, always_save_ops)."""
    aten = torch.ops.aten
    attn = [
        aten._scaled_dot_product_cudnn_attention.default,
        aten._scaled_dot_product_attention_math.default,
        aten._scaled_dot_product_fused_attention_overrideable.default,
        aten._scaled_dot_product_flash_attention.default,
        aten._scaled_dot_product_efficient_attention.default,
        torch._higher_order_ops.flex_attention,
        (torch.ops, "torch_attn._varlen_attn.default"),
        # flash-attn forward custom ops: FA2 dense + varlen, FA3 (one op).
        (torch.ops, "flash_attn._flash_attn_forward.default"),
        (torch.ops, "flash_attn._flash_attn_varlen_forward.default"),
        (torch.ops, "flash_attn_3._flash_attn_forward.default"),
        # Linear-attention fla kernels (opaque custom ops with their own backward).
        (torch.ops, "torchtitan.gdn_chunk.default"),
        (torch.ops, "torchtitan.kda_chunk.default"),
        (torch.ops, "torchtitan.qkv_conv1d_unpack.default"),
    ]
    comm = [
        torch.ops._c10d_functional.all_reduce.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_to_all_single.default,
        (torch.ops, "deepep.dispatch.default"),
        (torch.ops, "deepep.combine.default"),
        (torch.ops, "hybridep.dispatch.default"),
        (torch.ops, "hybridep.combine.default"),
    ]
    mm = [aten.mm.default, aten.linear.default]
    # Routed-expert GEMMs: torch._grouped_mm (bf16) and DeepGEMM's fp8 grouped
    # GEMM (registered with fp8 grouped experts). Not in the partitioner's
    # default compute-intensive list, so classify them explicitly.
    experts = [
        aten._grouped_mm.default,
        (torch.ops, _FP8_GROUPED_GEMM),
    ]
    always = [
        (torch.ops, "torchtitan.router_topk_indices.default"),
        aten.max.default,  # fp8 amax -> scaling factor
        (torch._higher_order_ops, "inductor_compiled_code"),
    ]
    # Registered when GroupedExperts is built with use_float8_grouped_gemm
    # (DeepEP path); inert otherwise.
    other = [(torch.ops, "torchtitan.fp8_blockwise_quantize.default")]

    cats: dict = {}
    for op in {o.default for o in get_default_op_list().compute_intensive_ops}:
        cats[op] = "other"
    for name, specs in (
        ("other", other),
        ("attn", attn),
        ("comm", comm),
        ("mm", mm),
        ("experts", experts),
    ):
        for op in _resolve_ops(specs):
            cats[op] = name
    cats.pop(aten.topk.default, None)
    always_ops = _resolve_ops(always)
    for op in always_ops:
        cats.pop(op, None)
    return cats, always_ops


def _get_save_ops() -> set:
    """Ops plain selective AC saves: every category but the routed experts,
    plus the always-saved ops (the graph_trainer memory policies use this)."""
    op_category, always_ops = _op_categories()
    return {op for op, cat in op_category.items() if cat != "experts"} | always_ops


def resolve_ac_mode(fqn: str, ac_config: ACConfig) -> str:
    """AC mode of the block at ``fqn``: first matching ``module_modes`` rule
    (``<fnmatch glob>=<mode>``), else the global ``mode``. Pure function, so
    anything (e.g. a stage balancer) can ask without touching the model."""
    for rule in ac_config.module_modes:
        pattern, sep, mode = rule.partition("=")
        if not sep:
            raise ValueError(f"module_modes entry {rule!r} is not '<glob>=<mode>'")
        if fnmatch.fnmatchcase(fqn, pattern):
            return mode
    return ac_config.mode


def _save_categories_for_mode(mode: str, ac_config: ACConfig) -> frozenset[str]:
    if mode == "selective":
        return parse_save_spec(ac_config.selective_save)
    if mode.startswith("save:"):
        return parse_save_spec(mode[len("save:") :])
    raise ValueError(f"not a selective AC mode: {mode!r}")


def _sac_context_fn(save, mm_recompute_shapes):
    """Bind policy values explicitly so Dynamo guards distinct SAC policies."""
    op_category, always_ops = _op_categories()
    fp8_grouped_gemm = next(iter(_resolve_ops([(torch.ops, _FP8_GROUPED_GEMM)])), None)
    # Some backends (e.g. PrivateUse1) register aten.linear as a leaf op
    # instead of decomposing it into aten.mm, so we must handle both.
    mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.linear.default)
    save_mm = "mm" in save
    save_mm_half = "mm_half" in save

    meta = {"forward_mm_count": 0, "recompute_mm_count": 0}

    def wrapped_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
        # Always save CUDA→CPU results to avoid recomputing them
        # (e.g. MoE D2H sync for all-to-all metadata).
        if (
            func == torch.ops.aten._to_copy.default
            and "cuda" in str(args[0].device)
            and "device" in kwargs
            and str(kwargs["device"]) == "cpu"
        ):
            return CheckpointPolicy.MUST_SAVE
        if func in always_ops:
            return CheckpointPolicy.MUST_SAVE

        category = op_category.get(func)
        if category is None:
            return CheckpointPolicy.PREFER_RECOMPUTE

        if func in mm_ops:
            weight_shape = args[1].shape
            # linear weight is (out, in); normalize to (in, out) to match mm
            if func == torch.ops.aten.linear.default:
                weight_shape = torch.Size((weight_shape[1], weight_shape[0]))
            if weight_shape in mm_recompute_shapes:
                return CheckpointPolicy.PREFER_RECOMPUTE
            mode = "recompute" if ctx.is_recompute else "forward"
            meta[f"{mode}_mm_count"] += 1
            # mm_half: save every second mm (the odd-numbered ones).
            keep = save_mm or (save_mm_half and meta[f"{mode}_mm_count"] % 2 == 1)
        elif category == "experts":
            # experts_up: only the up/gate projections (output narrower than
            # input: dim -> moe_hidden), the cheaper half to keep in memory.
            # _grouped_mm(x [R, in], w [E, in, out]) and the fp8 GEMM
            # (x [R, in], x_scale, w [E, out, in], ...): up/gate is in > out.
            if func is fp8_grouped_gemm:
                out_features = args[2].shape[1]
            else:
                out_features = args[1].shape[-1]
            up = args[0].shape[-1] > out_features
            keep = "experts" in save or ("experts_up" in save and up)
        else:
            keep = category in save
        if not keep:
            return CheckpointPolicy.PREFER_RECOMPUTE

        return CheckpointPolicy.MUST_SAVE

    return create_selective_checkpoint_contexts(wrapped_policy)


def _apply_op_sac(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    save: frozenset[str] | None = None,
    base_fqn: str | None = None,
    **checkpoint_kwargs,
) -> nn.Module:
    """Apply per-op selective activation checkpointing to the module,
    saving the outputs of the ops in categories ``save``."""
    if save is None:
        save = parse_save_spec(ac_config.selective_save)

    # Collect weight shapes to force-recompute, stored as mm RHS shape
    # (in_f, out_f). For aten.linear we transpose args[1].shape at lookup
    # time to match, since linear's weight is (out_f, in_f).
    mm_recompute_shapes = set()
    mm_recompute_fqns = ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns

    if mm_recompute_fqns:
        for module_fqn, submod in module.named_modules():
            fqn = f"{base_fqn}.{module_fqn}" if base_fqn else module_fqn
            if not any(f in fqn for f in mm_recompute_fqns):
                continue
            if not isinstance(submod, nn.Linear):
                raise ValueError(
                    "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                    f"a nn.Linear, but got: {submod}"
                )
            out_f, in_f = submod.weight.shape
            mm_recompute_shapes.add((in_f, out_f))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=partial(
            _sac_context_fn,
            tuple(sorted(save)),
            tuple(sorted(mm_recompute_shapes)),
        ),
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
        **checkpoint_kwargs,
    )


def _apply_full_ac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    """Apply full activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply full activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.

    Returns:
        nn.Module: The module with full activation checkpointing applied.
    """
    if not ac_config.preserve_rng_state and any(
        getattr(child, "_debug_force_load_balance", False) for child in module.modules()
    ):
        raise ValueError(
            "Forced random MoE routing requires preserve_rng_state=True with full AC"
        )
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def _apply_ac_to_block(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    mode: str,
    base_fqn: str | None = None,
    **checkpoint_kwargs,
) -> nn.Module:
    if mode == "full":
        return _apply_full_ac(module, ac_config)
    if mode == "selective" or mode.startswith("save:"):
        return _apply_op_sac(
            module,
            ac_config,
            save=_save_categories_for_mode(mode, ac_config),
            base_fqn=base_fqn,
            **checkpoint_kwargs,
        )
    raise ValueError(
        f"Invalid AC mode: {mode!r}. Valid: full, selective, none, save:<cat>[+<cat>]"
        f" with categories {SAC_CATEGORIES}"
    )


def apply_ac(
    model: Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to every repeated block in the model.

    Blocks come from ``model.named_segments()``, so every container the model
    holds is covered with no per-model wiring, and each ``base_fqn`` is the real
    FQN that ``per_op_sac_force_recompute_mm_shapes_by_fqns`` matches against.

    Args:
        model: The model to apply activation checkpointing to.
        ac_config: The activation checkpointing config.
        model_compile_enabled: Whether torch.compile is enabled for the model.

    Returns:
        None
    """
    # Disable dynamo LRU cache to workaround an interaction between SAC, PP, and Flex:
    #
    # When forward runs with a second PP microbatch, it triggers recompilation with dynamic
    # shapes enabled. Now there are two valid compiled graphs. By default, dynamo selects
    # the latest one (the dynamic shapes version), so the runtime wrapper expects an extra
    # symint output. When SAC caches the inductor HOP output from the static graph for
    # batch_idx=0, it would miss that symint and cause an assertion failure. The workaround
    # here is to disable the LRU cache, and select graphs in insertion order instead.
    #
    # Also see: https://github.com/pytorch/pytorch/issues/166926
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    if ac_config.mode == "memory_budget":
        assert model_compile_enabled, "Memory budget mode requires model to be compiled"
        if ac_config.visualize_memory_budget_pareto:
            pareto_dir = os.path.join(base_folder, "memory_budget_pareto")
            if not os.path.exists(pareto_dir):
                os.makedirs(pareto_dir, exist_ok=True)
            torch._functorch.config.memory_budget_pareto_dir = pareto_dir
            torch._functorch.config.visualize_memory_budget_pareto = True

        torch._functorch.config.activation_memory_budget = ac_config.memory_budget
        logger.info(f"Selected {ac_config.memory_budget} budget option")
    else:
        # Materialize first: ``replace`` re-registers into the containers the
        # walk reads.
        blocks = [s for s in model.named_segments() if s.is_block]
        if not blocks:
            logger.warning(
                f"no repeated block in {type(model).__name__}; skipping activation "
                "checkpointing. Expected for a pipeline stage holding only the stem "
                "or the tail; otherwise the model does not declare its blocks."
            )
            return
        modes: dict[str, str] = {}
        for segment in blocks:
            mode = resolve_ac_mode(segment.fqn, ac_config)
            modes[segment.fqn] = mode
            if mode == "none":
                continue
            segment.rebind(
                _apply_ac_to_block(
                    segment.module, ac_config, mode=mode, base_fqn=segment.fqn
                )
            )
        if ac_config.module_modes:
            logger.info(f"Per-module activation checkpointing modes: {modes}")

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
