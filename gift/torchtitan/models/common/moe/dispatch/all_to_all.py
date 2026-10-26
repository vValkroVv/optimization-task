from dataclasses import dataclass

import torch

from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)

from .base import BaseEPTokenDispatcher
from .local import local_reorder, LocalDispatchMetadata, score_and_scatter_add
from typing import TypeAlias

from torchtitan.config.schema.models.common.moe.dispatch.all_to_all import (
    AllToAllTokenDispatcherConfig,
)


@dataclass(frozen=True, kw_only=True)
class AllToAllDispatchMetadata(LocalDispatchMetadata):
    """Metadata returned by AllToAllTokenDispatcher.dispatch() for use in combine()."""

    input_shape: torch.Size
    permuted_indices: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]


def permute_to_expert_major(
    routed_input_RD: torch.Tensor,
    num_global_tokens_per_local_expert_E: torch.Tensor,
    *,
    ep_size: int,
) -> tuple[torch.Size, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder tokens from rank-major to expert-major layout.

    Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)

    Collapses token count matrix ``t_mat`` from ``(EP, e)`` to
    ``num_global_tokens_per_local_expert_e`` ``(e,)`` by summing across ranks.
    """
    e = num_global_tokens_per_local_expert_E.shape[0] // ep_size
    device = num_global_tokens_per_local_expert_E.device
    # (EP, e) matrix of token counts per (rank, local_expert)
    t_mat = num_global_tokens_per_local_expert_E.view(ep_size, e)

    # Where each (r, e) segment starts in the input (rank-major order)
    input_starts = (
        num_global_tokens_per_local_expert_E.cumsum(0)
        - num_global_tokens_per_local_expert_E
    ).view(ep_size, e)

    # Transpose to expert-major (e, EP) and flatten
    segment_lens = t_mat.t().reshape(-1)
    input_starts = input_starts.t().reshape(-1)

    # For each output position, find its input position:
    #   output[p] = input[input_starts[seg] + (p - output_starts[seg])]
    seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
        segment_lens
    )
    output_starts = segment_lens.cumsum(0) - segment_lens
    permuted_indices = (
        input_starts[seg_ids]
        # ``seg_ids`` already has one entry per output position.
        + torch.arange(seg_ids.numel(), device=device)
        - output_starts[seg_ids]
    )

    num_global_tokens_per_local_expert_e = t_mat.sum(0)
    return (
        routed_input_RD.shape,
        routed_input_RD[permuted_indices, :],
        permuted_indices,
        num_global_tokens_per_local_expert_e,
    )


def unpermute_from_expert_major(
    routed_output_RD: torch.Tensor,
    input_shape: torch.Size,
    permuted_indices: torch.Tensor,
) -> torch.Tensor:
    """Reverse :func:`permute_to_expert_major`."""
    out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
    out_unpermuted_RD[permuted_indices, :] = routed_output_RD
    return out_unpermuted_RD


class AllToAllTokenDispatcher(BaseEPTokenDispatcher[AllToAllDispatchMetadata]):
    """Token dispatcher for EP>1. Handles token reorder + all-to-all dispatch/combine.

    Handles the full token routing lifecycle:
    dispatch (reorder + EP all-to-all) and combine (reverse).

    ``ep_mesh`` is wired by the owning ``GroupedExperts.parallelize``
    override via ``wire_meshes``.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = AllToAllTokenDispatcherConfig

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, AllToAllDispatchMetadata]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        With SP, x_TD/topk_scores_TK/topk_expert_ids_TK are already
        the local SP shard (from DTensor Shard to_local via LocalMapConfig).

        Args:
            x_TD: ``(T, D)`` local token shard
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices
            num_local_tokens_per_expert_E: ``(E,)`` token counts for this local
                token shard

        Returns:
            routed_input_RD: ``[R = sum(num_tokens_per_local_expert_e), input_dim(D)]``.
                Tokens in expert-major order for local experts.
            num_tokens_per_local_expert_e: ``(num_local_experts,)`` token counts
            metadata: dispatch metadata for combine()
        """
        ep_mesh = self._mesh()
        ep_size = ep_mesh.size()
        requested_scores = topk_scores_TK if self.token_drop is not None else None
        topk_scores_TK, topk_expert_ids_TK, keep = self._drop_routes(
            topk_scores_TK,
            topk_expert_ids_TK,
            num_experts=num_local_tokens_per_expert_E.numel(),
            ep_group=ep_mesh.get_group(),
        )
        requested_counts = num_local_tokens_per_expert_E
        if keep is not None:
            num_local_tokens_per_expert_E = torch.zeros_like(requested_counts)
            num_local_tokens_per_expert_E.scatter_add_(
                0,
                topk_expert_ids_TK.reshape(-1).clamp_min(0),
                keep.reshape(-1).to(requested_counts.dtype),
            )
        (
            routed_input_ND,
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

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_global_tokens_per_local_expert_E = all_to_all_single(
                num_local_tokens_per_expert_E,
                None,
                None,
                group=ep_mesh,
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_global_tokens_per_local_expert_E = (
                torch.ops._c10d_functional.wait_tensor(
                    num_global_tokens_per_local_expert_E
                )
            )
            # non_blocking=True is safe in eager, but under torch.compile the
            # async D2H transfer can race with the subsequent .tolist()/.item()
            # calls, producing stale values and failing unbacked-symint guards.
            non_blocking = not torch.compiler.is_compiling()
            input_splits = (
                num_local_tokens_per_expert_E.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=non_blocking)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_global_tokens_per_local_expert_E.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            input_splits_list = input_splits.tolist()
            output_splits_list = output_splits.tolist()

        # All-to-all dispatch tokens to EP ranks
        routed_input_RD = all_to_all_single_autograd(
            routed_input_ND,
            output_splits_list,
            input_splits_list,
            ep_mesh,
        )

        # Reorder from rank-major to expert-major via _permute.
        #
        # num_global_tokens_per_local_expert_E layout after all-to-all
        # (e = local experts, EP = EP ranks):
        #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        # _permute reshuffles to:
        #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
        # TODO: Consider using num_global_tokens_per_local_expert_e as the
        # expert_bias_e update buffer, then all-gather on EP ranks. This
        # must preserve pre-drop demand for the balancing update.
        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        ) = permute_to_expert_major(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
            ep_size=ep_size,
        )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            score_before_experts=score_before_experts,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        self._dispatch(
            local_tokens_per_expert=requested_counts,
            global_tokens_per_local_expert=num_global_tokens_per_local_expert_e,
            routed_input=routed_input_RD,
            topk_expert_ids_TK=topk_expert_ids_TK,
            requested_topk_scores=requested_scores,
            capacity_keep_mask=keep,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + score + scatter_add.

        Token indices stay local and are never offset -- see the base
        ``combine`` for the contract.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens

        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        ep_mesh = self._mesh()
        # Reverse expert-major reordering
        routed_output_RD = unpermute_from_expert_major(
            routed_output_RD, metadata.input_shape, metadata.permuted_indices
        )

        # All-to-all combine: returns AsyncCollectiveTensor — the a2a runs
        # on the NCCL stream and won't block until the tensor is accessed.
        routed_output_RD = all_to_all_single_autograd(
            routed_output_RD,
            metadata.input_splits,
            metadata.output_splits,
            ep_mesh,
        )

        # Sized like the local shard: the reverse all-to-all returns this
        # rank's own tokens, and the caller declares the output sequence-sharded.
        return score_and_scatter_add(
            routed_output_RD,
            x_TD=x_TD,
            token_indices_experts_sorted_N=metadata.token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=metadata.topk_scores_experts_sorted_N,
            score_before_experts=metadata.score_before_experts,
        )
