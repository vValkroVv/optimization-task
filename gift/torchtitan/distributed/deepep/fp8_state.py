# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fp8 dispatch state: the contract between an EP backend and the experts.

``GroupedExperts._fp8_experts_forward`` reads exactly these fields, so the
backends fill in one type instead of each growing its own and the experts
probing for fields that may or may not be there. A backend that needs more --
DeepEP's padded-permutation bookkeeping, say -- subclasses and adds it; nothing
the experts read lives in a subclass.
"""

from dataclasses import dataclass

import torch

__all__ = ["FP8DispatchState"]


@dataclass(kw_only=True)
class FP8DispatchState:
    """What ``GroupedExperts._fp8_experts_forward`` reads off an fp8 dispatch.

    Attributes:
        permuted_probs: routing probs in the permuted layout. The fp8 experts
            apply them between their two grouped GEMMs, which is why this path
            requires ``score_before_experts=False``.
        padded_group_sizes_t: per-expert row counts of the padded layout, on the
            device. Always present -- it is the only form a traced region or a
            non-blocking dispatch can read without forcing a D2H sync.
        padded_group_sizes_list: the same counts on the host, which is the form
            DeepGEMM's k-grouped wgrad wants. ``None`` when they never came
            across (non-blocking dispatch) or under tracing.
        padded_group_sizes_cpu: the same counts again, as a CPU tensor. Not a
            duplicate of the list: the wgrad falls back to it exactly when the
            list is absent but the counts did reach the host -- a compiled
            blocking dispatch, where a Python list cannot travel through the
            graph to the backward but a tensor can. Reaching for the device
            tensor instead would cost that backward a sync. ``None`` when the
            counts never came across.
        padded_tokens: the fp8 payload, in the padded expert-major layout, for
            a backend that keeps it off the graph. ``None`` otherwise, because
            the payload then already reached the experts as the dispatcher's
            return value and a second copy of the reference would be one more
            thing to keep in step. Which case applies is stated by
            ``uses_gradient_edges``, never inferred from this being None.
        uses_gradient_edges: whether the payload's gradient travels on a
            separate edge instead of on the payload itself. A fact about how the
            backend wired its autograd, stated rather than inferred from which
            fields are populated. It decides what the experts feed the grouped
            GEMMs (``grad_edge``, ``fp8_grad_output``) and whether they owe the
            combine an ``output_fp8_edge``. Defaults to the ordinary case, where
            the payload is the dispatcher's return value and therefore its own
            edge; a backend that keeps it off the graph sets this and
            ``padded_tokens`` together.
        fp8_expert_weights: this rank's expert weights already quantized by the
            backend. ``None`` means the experts quantize them themselves.
        output_fp8_edge: set by the experts, not the dispatch -- the packed edge
            their bf16 output travels on, which the combine's backward fills
            with the fp8 gradient.
    """

    permuted_probs: torch.Tensor
    padded_group_sizes_t: torch.Tensor
    uses_gradient_edges: bool = False
    padded_tokens: torch.Tensor | None = None
    padded_group_sizes_list: list[int] | None = None
    padded_group_sizes_cpu: torch.Tensor | None = None
    fp8_expert_weights: list[torch.Tensor] | None = None
    output_fp8_edge: torch.Tensor | None = None
