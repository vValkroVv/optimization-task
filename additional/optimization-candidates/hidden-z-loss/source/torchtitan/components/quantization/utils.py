# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dense fp8 selection helpers driven by ``FP8Config``.

- Dense: ``apply_float8`` walks the model config tree and replaces any
  ``Linear.Config`` whose FQN matches ``dense_include_linear_layers`` with
  ``FP8Linear.Config`` (linears live in FFNs, attention, etc.), and enables the
  shared input quantizer on every ``FeedForward`` when
  ``dense_use_shared_quantizer`` is set; ``dense_fused_swiglu_quant=False``
  switches every ``FeedForward`` to a bf16 SwiGLU between fp8 projections.
- Sparse (MoE experts): the ``use_float8_grouped_gemm`` flag on every
  ``GroupedExperts.Config`` when ``FP8Config.sparse_use_grouped_gemm`` is set,
  with ``sparse_fused_swiglu_quant`` copied from ``FP8Config``.
"""

import re
from dataclasses import fields

import torch.nn as nn

from torchtitan.config.configs import FP8Config
from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.nn_modules import Linear
from torchtitan.tools.logging import logger


def enable_fp8_grouped_experts(
    experts: GroupedExperts.Config,
    *,
    deep_gemm_num_sms: int = 116,
    sparse_fused_swiglu_quant: bool = True,
) -> None:
    from torchtitan.config.schema.models.common.cross_section import (
        fp8_grouped_experts_issues,
    )

    issues = fp8_grouped_experts_issues(experts)
    if issues:
        raise ValueError("; ".join(str(issue) for issue in issues))
    experts.use_float8_grouped_gemm = True
    experts.deep_gemm_num_sms = deep_gemm_num_sms
    experts.sparse_fused_swiglu_quant = sparse_fused_swiglu_quant


def apply_float8(model_config, fp8: FP8Config, *, tensor_parallel_degree: int) -> None:
    """Select fp8 and assign derived flags from ``FP8Config``.

    Changing the dense projection selection requires a fresh materialized tree.
    """
    if fp8.dense_include_linear_layers and tensor_parallel_degree > 1:
        raise ValueError(
            "Dense fp8 does not support tensor parallelism: "
            "fp8.dense_include_linear_layers is set and "
            f"parallelism.tensor_parallel_degree is {tensor_parallel_degree}. "
        )
    if fp8.dense_include_linear_layers:
        from torchtitan.config.schema.models.common.cross_section import (
            dense_fp8_replacement_issues,
        )

        issues = dense_fp8_replacement_issues(
            model_config, fp8.dense_include_linear_layers
        )
        if issues:
            raise ValueError("; ".join(str(issue) for issue in issues))
        from torchtitan.ops.float8.cuda_gemm import FP8Linear

        linear_pattern = re.compile(fp8.dense_include_linear_layers)
        locations = [
            location
            for location in model_config.traverse(Linear.Config)
            if linear_pattern.search(location[0])
        ]
        if not locations:
            logger.warning(
                "Dense FP8 selector %r matched no Linear layers",
                fp8.dense_include_linear_layers,
            )
        for fqn, lin, parent, attr in locations:
            if isinstance(lin, FP8Linear.Config):
                continue
            new_cfg = FP8Linear.Config(
                **{f.name: getattr(lin, f.name) for f in fields(lin) if f.init}
            )
            for config_field in fields(lin):
                if not config_field.init:
                    setattr(new_cfg, config_field.name, getattr(lin, config_field.name))
            new_cfg.copy_metadata_from(lin)
            if isinstance(parent, (list, dict)):
                parent[attr] = new_cfg
            else:
                setattr(parent, attr, new_cfg)

    for _fqn, ffn, _parent, _attr in model_config.traverse(FeedForward.Config):
        # Inert on FFNs that did not end up fully fp8. Assign false as well:
        # runtime state must not override the current FP8Config policy.
        ffn.use_shared_quantizer = fp8.dense_use_shared_quantizer
        ffn.dense_fused_swiglu_quant = fp8.dense_fused_swiglu_quant
    for _fqn, experts, _parent, _attr in model_config.traverse(GroupedExperts.Config):
        if fp8.sparse_use_grouped_gemm:
            enable_fp8_grouped_experts(
                experts,
                deep_gemm_num_sms=fp8.sparse_deep_gemm_num_sms,
                sparse_fused_swiglu_quant=fp8.sparse_fused_swiglu_quant,
            )
        else:
            experts.use_float8_grouped_gemm = False
            experts.deep_gemm_num_sms = fp8.sparse_deep_gemm_num_sms
            experts.sparse_fused_swiglu_quant = fp8.sparse_fused_swiglu_quant


def fp8_fwd_flops_per_token(model_config) -> int:
    """Forward FLOPs per token running on FP8 tensor cores."""
    from torchtitan.models.common.attention.qkv import QKVLinear
    from torchtitan.models.common.moe import MoE
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    dense = sum(
        2
        * c.in_features
        * c.out_features
        # QKVLinear builds both wk and wv from the same wkv config.
        * (2 if isinstance(parent, QKVLinear.Config) and attr == "wkv" else 1)
        for _, c, parent, attr in model_config.traverse(FP8Linear.Config)
    )
    routed = sum(
        moe.router.top_k * 6 * moe.experts.dim * moe.experts.hidden_dim
        for _, moe, _, _ in model_config.traverse(MoE.Config)
        if moe.experts.use_float8_grouped_gemm
    )
    return dense + routed


def fp8_flops_fraction(model_config, num_flops_per_token: int) -> float:
    """Share of the per-token FLOPs running on FP8.

    ``num_flops_per_token`` is the forward count times 3 (fwd + 2x bwd).
    """
    if num_flops_per_token <= 0:
        return 0.0
    fraction = fp8_fwd_flops_per_token(model_config) / (num_flops_per_token / 3)
    if not 0.0 <= fraction <= 1.0:
        # The two counts are derived independently; a share outside [0, 1] means
        # they disagree, and past 2.0 the harmonic mean divides by zero.
        logger.warning(
            f"fp8 FLOPs share came out as {fraction:.4f}, outside [0, 1]; "
            "clamping for MFU. The fp8 and total FLOP counts disagree."
        )
        fraction = min(max(fraction, 0.0), 1.0)
    return fraction


def has_quantization(model_config) -> bool:
    """True if the model config selects any fp8 path (dense linear or MoE)."""
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    has_fp8_linear = any(
        True for _fqn, _c, _p, _a in model_config.traverse(FP8Linear.Config)
    )
    has_fp8_moe = any(
        config.use_float8_grouped_gemm
        for _fqn, config, _p, _a in model_config.traverse(GroupedExperts.Config)
    )
    return has_fp8_linear or has_fp8_moe


def log_float8_model_structure(model: nn.Module) -> None:
    """Log every fp8 projection in the built model without collapsing layers."""
    dense_projections: list[str] = []
    grouped_projections: list[str] = []

    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    for module_fqn, module in model.named_modules():
        if isinstance(module, FP8Linear):
            dense_projections.append(
                f"  {module_fqn}: FP8Linear({module.in_features} -> "
                f"{module.out_features}), weight={tuple(module.weight.shape)}"
            )
        elif isinstance(module, GroupedExperts) and module._use_float8_grouped_gemm:
            w13_shape = tuple(module.w13_E2FD.shape)
            w2_shape = tuple(module.w2_EDF.shape)
            grouped_projections.extend(
                (
                    f"  {module_fqn}.w13_E2FD: FP8GroupedLinear(fused w1+w3, "
                    f"{w13_shape[-1]} -> 2x{w13_shape[-2]}), weight={w13_shape}",
                    f"  {module_fqn}.w2_EDF: FP8GroupedLinear("
                    f"{w2_shape[-1]} -> {w2_shape[-2]}), weight={w2_shape}",
                )
            )

    if not dense_projections and not grouped_projections:
        return

    lines = [
        "FP8 projections in full model structure: "
        f"{len(dense_projections)} dense, {len(grouped_projections)} grouped"
    ]
    if dense_projections:
        lines.append("Dense FP8Linear projections:")
        lines.extend(dense_projections)
    if grouped_projections:
        lines.append("Grouped-expert FP8 projections:")
        lines.extend(grouped_projections)
    logger.info("\n".join(lines))


__all__ = [
    "FP8Config",
    "apply_float8",
    "fp8_flops_fraction",
    "fp8_fwd_flops_per_token",
    "enable_fp8_grouped_experts",
    "has_quantization",
    "log_float8_model_structure",
]
