"""The token-dispatcher contract, and the shared state of the EP backends.

A dispatcher moves routed tokens to the ranks holding their experts and back:
``dispatch`` returns the metadata its own ``combine`` needs, so that type is a
parameter of the base rather than something each backend widens away.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar

import torch
from torch.distributed.tensor import DeviceMesh

from torchtitan.config.schema.models.common.moe.dispatch.base import (
    BaseTokenDispatcherConfig,
)
from torchtitan.config import Configurable

from torchtitan.protocols.moe_monitoring import DispatchObservation, DispatchObserver

from .capacity import select_capacity_routes

__all__ = ["BaseEPTokenDispatcher", "BaseTokenDispatcher"]

M = TypeVar("M")
"""Metadata a backend's ``dispatch`` hands to its own ``combine``."""


class BaseTokenDispatcher(Configurable, ABC, Generic[M]):
    """What every token dispatcher implements, for any expert-parallel degree.

    Not an nn.Module -- dispatchers have no learnable parameters or buffers.

    Binding ``M`` per backend keeps ``dispatch``/``combine`` overrides checkable:
    a backend states the metadata it produces and consumes, instead of every
    signature widening to a common supertype that none of them actually accept.
    """

    Config: TypeAlias = BaseTokenDispatcherConfig

    def __init__(self, config: Config):
        config.__post_init__()
        self.token_drop = config.token_drop
        self._capacity_num_tokens = config._capacity_num_tokens
        self.dispatch_monitor: DispatchObserver | None = None

    def _drop_routes(
        self,
        scores: torch.Tensor,
        expert_ids: torch.Tensor,
        *,
        num_experts: int,
        ep_group: torch.distributed.ProcessGroup | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.token_drop is None:
            return scores, expert_ids, None
        keep = select_capacity_routes(
            scores,
            expert_ids,
            num_experts=num_experts,
            policy=self.token_drop,
            ep_group=ep_group,
            max_num_tokens=self._capacity_num_tokens,
        )
        return scores * keep, expert_ids.masked_fill(~keep, -1), keep

    def wire_meshes(self, *, ep_mesh: DeviceMesh | None) -> None:
        """Receive the EP mesh from ``GroupedExperts.parallelize``.

        A no-op here: dispatching within one rank needs no mesh. The EP
        backends override it to keep ``ep_mesh``. Sequence sharding is expressed
        by placements, so no TP mesh is passed.
        """
        del ep_mesh

    def _dispatch(
        self,
        *,
        local_tokens_per_expert: torch.Tensor,
        global_tokens_per_local_expert: torch.Tensor,
        routed_input: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        requested_topk_scores: torch.Tensor | None = None,
        capacity_keep_mask: torch.Tensor | None = None,
    ) -> None:
        """Report one ``dispatch`` to the monitor, when one is installed.

        Every backend's ``dispatch`` calls this once before returning, with
        its own arguments and returns — the observation is assembled in one
        place instead of being copied per backend.
        """
        if self.dispatch_monitor is not None:
            self.dispatch_monitor(
                DispatchObservation(
                    local_tokens_per_expert=local_tokens_per_expert,
                    global_tokens_per_local_expert=global_tokens_per_local_expert,
                    routed_input=routed_input,
                    topk_expert_ids_TK=topk_expert_ids_TK,
                    requested_topk_scores=requested_topk_scores,
                    capacity_keep_mask=capacity_keep_mask,
                )
            )

    @abstractmethod
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, M]:
        """Route tokens to the experts that will process them.

        Args:
            x_TD: ``(T, D)`` this rank's tokens.
            topk_scores_TK: ``(T, K)`` routing scores.
            topk_expert_ids_TK: ``(T, K)`` selected expert indices.
            num_local_tokens_per_expert_E: ``(E,)`` token counts for this rank;
                its length is the expert count.
            score_before_experts: whether scores weigh the expert input; the
                owning MoE decides, so it is passed rather than stored.

        Returns:
            routed_input_RD: ``(R, D)`` tokens grouped by expert.
            num_tokens_per_local_expert_e: ``(e,)`` counts for the experts this
                rank owns, which the grouped matmul turns into offsets. At EP=1
                the local experts are all of them, so ``(e,) == (E,)`` and a
                backend may pass its input straight through.
            metadata: what ``combine`` needs to undo this.
        """

    @abstractmethod
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: M,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Bring expert outputs back to their tokens.

        Returns ``(T, D)`` shaped like ``x_TD`` -- this rank's own tokens, never
        the global sequence. Token indices are local, so a backend must not
        offset them: the caller declares the result sequence-sharded and DTensor
        owns the cross-rank view.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs.
            metadata: what ``dispatch`` returned.
            x_TD: ``(T, D)`` the original tokens, for shape and dtype.

        Returns:
            out_TD: ``(T, D)`` combined output.
        """

    def sync_combine(self) -> None:
        """Wait for an asynchronous combine. No-op unless a backend overrides."""


class BaseEPTokenDispatcher(BaseTokenDispatcher[M], ABC):
    """Shared state for dispatchers with EP > 1: the EP mesh.

    ``ep_mesh`` stays None until ``GroupedExperts.parallelize`` wires it, so a
    backend that is built but never parallelized (EP=1) can still say so.
    """

    def __init__(self, config: BaseTokenDispatcher.Config):
        super().__init__(config)
        self.ep_mesh: DeviceMesh | None = None

    def wire_meshes(self, *, ep_mesh: DeviceMesh | None) -> None:
        self.ep_mesh = ep_mesh

    def _mesh(self) -> DeviceMesh:
        """The wired EP mesh. Unset means this backend is dispatching before
        ``GroupedExperts.parallelize`` ran -- a wiring bug, not a config error."""
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "GroupedExperts.parallelize() sets it via wire_meshes()."
        )
        return self.ep_mesh
