# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import torch
import torch._inductor.config
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.config import CompileConfig
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


# TODO: Remove this monkeypatch once FakeTensorMode.__init__ is decorated with
# @torch.compiler.disable(recursive=True) upstream.
# See https://github.com/pytorch/pytorch/issues/178887
FakeTensorMode.__init__ = torch.compiler.disable(  # type: ignore[method-assign]
    FakeTensorMode.__init__, recursive=True
)


def apply_compile(model: Module, compile_config: CompileConfig) -> None:
    """
    Apply torch.compile to each repeated block, which makes compilation efficient
    due to repeated structure. Alternatively one can compile the whole model
    (after applying DP). Blocks come from ``model.named_segments()``, so every
    container is covered without per-model wiring.
    """
    excluded_fqns = compile_config.cudagraph_exclude_blocks
    if type(excluded_fqns) is not list:
        raise ValueError(
            "compile.cudagraph_exclude_blocks must be a list of block FQNs"
        )
    if excluded_fqns:
        if any(type(name) is not str or not name for name in excluded_fqns):
            raise ValueError(
                "compile.cudagraph_exclude_blocks must contain nonempty block FQNs"
            )
        if not compile_config.enable:
            raise ValueError(
                "compile.cudagraph_exclude_blocks requires compile.enable=true"
            )
        if "model" not in compile_config.components:
            raise ValueError(
                "compile.cudagraph_exclude_blocks requires 'model' in compile.components"
            )
        if compile_config.backend != "inductor":
            raise ValueError(
                "compile.cudagraph_exclude_blocks requires compile.backend='inductor'"
            )
        if len(set(excluded_fqns)) != len(excluded_fqns):
            raise ValueError(
                "compile.cudagraph_exclude_blocks contains duplicate block FQNs"
            )

    blocks = None
    if excluded_fqns:
        blocks = [segment for segment in model.named_segments() if segment.is_block]
        available_fqns = {segment.fqn for segment in blocks}
        unmatched_fqns = [
            name for name in excluded_fqns if name not in available_fqns
        ]
        if unmatched_fqns:
            raise ValueError(
                "compile.cudagraph_exclude_blocks contains unmatched block FQNs: "
                f"{unmatched_fqns}"
            )

    # Needed for torch.compile to handle data-dependent dynamic shapes in
    # token-choice MoE dispatch. Harmless for dense models.
    torch._dynamo.config.capture_scalar_outputs = True
    # Skip replaying forward side effects (e.g. RoPE cache updates) during
    # the AC recompute in backward. Eager AC replays the forward python
    # side-effects in backward, but torch.compile has no easy way to reapply
    # python mutations in the backward. Setting this flag accepts this eager
    # and compile divergence by skipping reapplication of side effects.
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    # Non-strict mode skips workload-size and aspect-ratio guards, avoiding
    # recompilation during dynamic seq-len warmup while preserving the
    # row/column reduction fusion used at full length.
    triton_config = torch._inductor.config.triton
    if hasattr(triton_config, "mix_order_reduction_non_strict_mode"):
        triton_config.mix_order_reduction_non_strict_mode = True
    else:
        logger.warning(
            "torch._inductor.config.triton has no mix_order_reduction_non_strict_mode; "
            "a seq-len warmup may recompile each block at the fusion thresholds"
        )

    compiled: dict[str, int] = {}
    segments = blocks if blocks is not None else model.named_segments()
    for segment in segments:
        if not segment.is_block:
            continue
        if segment.fqn in excluded_fqns:
            segment.module.compile(
                backend=compile_config.backend,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )
        else:
            segment.module.compile(backend=compile_config.backend, fullgraph=True)
        compiled[segment.group] = compiled.get(segment.group, 0) + 1

    if not compiled:
        logger.warning(
            f"no repeated block in {type(model).__name__}; skipping torch.compile. "
            "Expected for a pipeline stage holding only the stem or the tail; "
            "otherwise the model does not declare its blocks."
        )
        return

    counts = ", ".join(f"{n} in {group}" for group, n in compiled.items())
    logger.info(f"Compiling each block with torch.compile ({counts})")
    if excluded_fqns:
        assert blocks is not None
        matched = [segment.fqn for segment in blocks if segment.fqn in excluded_fqns]
        logger.info(
            "Disabled Inductor CUDA Graphs for "
            f"{len(matched)} block(s) with options={{'triton.cudagraphs': False}}: "
            f"{matched}"
        )


class EagerWindow:
    """Runs the blocks ``apply_compile`` compiled eagerly between open() and close().

    Two holders can want eager blocks over overlapping spans (a logged step,
    the pipeline metadata probe), so the slot is owned jointly: cleared for
    the first holder, restored when the last one lets go.
    """

    # Holders live on the module: the windows are separate objects, the slot is not.
    _HOLDERS = "_eager_window_holders"

    def __init__(self, model_parts: Iterable[nn.Module], *, owner: str) -> None:
        self._owner = owner
        # Taken once, while the slots are still filled: inside an open window
        # a compiled block is indistinguishable from a plain one.
        self._blocks = [
            (module, module._compiled_call_impl)
            for part in model_parts
            for module in part.modules()
            if getattr(module, "_compiled_call_impl", None) is not None
        ]

    def __len__(self) -> int:
        return len(self._blocks)

    def open(self) -> None:
        for module, _ in self._blocks:
            holders = module.__dict__.setdefault(self._HOLDERS, set())
            if not holders:
                module._compiled_call_impl = None
            # A set, not a count: a holder may re-open the window every step
            # while another one opens it once.
            holders.add(self._owner)

    def close(self) -> None:
        for module, compiled in self._blocks:
            holders = module.__dict__.get(self._HOLDERS)
            if not holders:
                continue
            holders.discard(self._owner)
            if not holders:
                module._compiled_call_impl = compiled
