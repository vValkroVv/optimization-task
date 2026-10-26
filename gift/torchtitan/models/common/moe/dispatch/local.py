"""Single-rank token dispatch, and the two steps every backend shares.

``local_reorder`` and ``score_and_scatter_add`` are plain functions: the
all-to-all backend runs them around its collectives and reuses them wholesale
when EP=1, which a base-class call could only express by inheriting from the
EP=1 dispatcher.
"""

from dataclasses import dataclass

import torch

from torch.distributed.device_mesh import DeviceMesh

from typing import TypeAlias

from torchtitan.config.schema.models.common.moe.dispatch.local import (
    LocalTokenDispatcherConfig,
)
from torchtitan.ops.scatter_add import deterministic_scatter_add

from .base import BaseTokenDispatcher

__all__ = [
    "local_reorder",
    "LocalDispatchMetadata",
    "LocalTokenDispatcher",
    "score_and_scatter_add",
]


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """Metadata returned by LocalTokenDispatcher.dispatch() for use in combine()."""

    token_indices_experts_sorted_N: torch.Tensor  # noqa: N815
    topk_scores_experts_sorted_N: torch.Tensor  # noqa: N815
    # Recorded on dispatch so combine applies the scores on the same side.
    score_before_experts: bool


def local_reorder(
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    *,
    top_k: int,
    score_before_experts: bool,
    keep_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder tokens by expert assignment for local expert computation.

    Groups tokens by expert index via argsort and optionally applies routing
    scores (when ``score_before_experts`` is True).

    Args:
        x_TD: ``(T, D)`` input tokens
        topk_scores_TK: ``(T, K)`` routing scores
        topk_expert_ids_TK: ``(T, K)`` expert indices
        top_k: ``K``, to map flattened positions back to tokens
        score_before_experts: apply scores here rather than after the experts
        keep_mask: optional retained assignments; compact before gathering tokens

    Returns:
        routed_input_ND: ``(N, D)`` where N is the retained route count. Tokens in expert-sorted
            order, score-weighted if ``score_before_experts``.
        token_indices_experts_sorted_N: ``(N,)`` token-to-original mapping
        topk_scores_experts_sorted_N: ``(N,)`` scores in expert-sorted order
    """
    # Reorder the token indices to match the order of the experts where N = T*K
    token_indices_experts_sorted_N = torch.argsort(
        topk_expert_ids_TK.view(-1), stable=True
    )
    if keep_mask is not None:
        token_indices_experts_sorted_N = token_indices_experts_sorted_N[
            keep_mask.reshape(-1)[token_indices_experts_sorted_N]
        ]
    topk_scores_experts_sorted_N = topk_scores_TK.reshape(-1)[
        token_indices_experts_sorted_N
    ]
    token_indices_experts_sorted_N = token_indices_experts_sorted_N // top_k
    routed_input_ND = x_TD[token_indices_experts_sorted_N]

    # Apply scores before expert computation if configured
    if score_before_experts:
        routed_input_ND = (
            routed_input_ND.to(torch.float32)
            * topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(x_TD.dtype)

    return (
        routed_input_ND,
        token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N,
    )


def score_and_scatter_add(
    routed_output_RD: torch.Tensor,
    *,
    x_TD: torch.Tensor,
    token_indices_experts_sorted_N: torch.Tensor,
    topk_scores_experts_sorted_N: torch.Tensor,
    score_before_experts: bool,
) -> torch.Tensor:
    """Weight expert outputs by their routing scores and add them back.

    Args:
        routed_output_RD: ``(R, D)`` expert outputs in token-sorted order
        x_TD: ``(T, D)`` original tokens; gives the output shape and dtype
        token_indices_experts_sorted_N: ``(N,)`` local token index per row
        topk_scores_experts_sorted_N: ``(N,)`` score per row
        score_before_experts: whether scores were already applied on dispatch

    Returns:
        out_TD: ``(T, D)`` combined output.
    """
    if not score_before_experts:
        routed_output_RD = (
            routed_output_RD.to(torch.float32)
            * topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_RD.dtype)

    dim = x_TD.shape[-1]
    return deterministic_scatter_add(
        torch.zeros_like(x_TD),
        token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, dim),
        routed_output_RD,
    )


class LocalTokenDispatcher(BaseTokenDispatcher[LocalDispatchMetadata]):
    """Token dispatcher for EP=1. Handles local token reordering only."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = LocalTokenDispatcherConfig

    def wire_meshes(self, *, ep_mesh: DeviceMesh | None) -> None:
        """Refuse an EP mesh: being the default, this is what an unresolved model
        arrives with, and tokens that reach no other rank fail nothing."""
        if ep_mesh is not None and ep_mesh.size() > 1:
            raise ValueError(
                f"LocalTokenDispatcher does not route across ranks, but expert "
                f"parallelism is {ep_mesh.size()}. Its tokens would stay on "
                "their own rank, so every rank would use the wrong experts."
            )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Args:
            x_TD: ``(T, D)`` all input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices per token
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert

        Returns:
            routed_input_RD: ``[R = sum(num_local_tokens_per_expert_E), input_dim(D)]``.
                Tokens sorted by expert index.
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert
            metadata: LocalDispatchMetadata for combine()
        """

        requested_scores = topk_scores_TK if self.token_drop is not None else None
        topk_scores_TK, topk_expert_ids_TK, keep = self._drop_routes(
            topk_scores_TK,
            topk_expert_ids_TK,
            num_experts=num_local_tokens_per_expert_E.numel(),
        )
        retained_counts = num_local_tokens_per_expert_E
        if keep is not None:
            retained_counts = torch.zeros_like(num_local_tokens_per_expert_E)
            retained_counts.scatter_add_(
                0,
                topk_expert_ids_TK.reshape(-1).clamp_min(0),
                keep.reshape(-1).to(retained_counts.dtype),
            )

        # No communication: compact the retained assignments and group by expert.
        (
            routed_input_RD,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = local_reorder(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            top_k=topk_scores_TK.size(-1),
            score_before_experts=score_before_experts,
            keep_mask=keep,
        )

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            score_before_experts=score_before_experts,
        )
        self._dispatch(
            local_tokens_per_expert=num_local_tokens_per_expert_E,
            global_tokens_per_local_expert=retained_counts,
            routed_input=routed_input_RD,
            topk_expert_ids_TK=topk_expert_ids_TK,
            requested_topk_scores=requested_scores,
            capacity_keep_mask=keep,
        )
        return routed_input_RD, retained_counts, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Score and scatter_add routed expert outputs. See the base contract."""
        return score_and_scatter_add(
            routed_output_RD,
            x_TD=x_TD,
            token_indices_experts_sorted_N=metadata.token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=metadata.topk_scores_experts_sorted_N,
            score_before_experts=metadata.score_before_experts,
        )
