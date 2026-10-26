# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The dispatch-observation contract between token dispatchers and monitoring.

Structure only — one value type and one slot for observing a call that no
``nn.Module`` hook can reach. No counters, no arming flags, no statistics:
those live in the monitoring callback, so a new MoE metric never needs a
change in the model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, kw_only=True)
class DispatchObservation:
    """One ``dispatch()`` call, as the monitoring sees it."""

    local_tokens_per_expert: torch.Tensor
    global_tokens_per_local_expert: torch.Tensor
    # Passed whole rather than as a row count, so the model does not have to
    # decide whether reading its shape is safe.
    routed_input: torch.Tensor
    # ``(T, K)`` per-token expert assignment, which the dispatcher already holds.
    # The masked histograms are built from it in the callback rather than in the
    # model, so a padding- or prefix-excluded variant costs the model nothing.
    topk_expert_ids_TK: torch.Tensor  # noqa: N815
    # Optional capacity-policy observation. Local expert counts remain router
    # demand; IDs and received counts describe retained execution (-1 = absent).
    requested_topk_scores: torch.Tensor | None = None
    capacity_keep_mask: torch.Tensor | None = None


DispatchObserver = Callable[[DispatchObservation], None]
"""Called once per ``dispatch()``, with everything the monitoring may read."""


class DispatchObservable(Protocol):
    """A token dispatcher, from the monitoring side.

    ``dispatch()`` is a plain method of a plain object — a dispatcher owns no
    parameters and is deliberately not an ``nn.Module`` — so there is no forward
    hook to register on it. Assigning this slot is the equivalent, and clearing
    it restores an uninstrumented dispatcher.
    """

    dispatch_monitor: DispatchObserver | None
