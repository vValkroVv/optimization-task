from torchtitan.config.schema.models.common.moe.router import (
    TokenChoiceTopKRouterConfig,
)
from typing import TypeAlias


import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.config.schema.models.common.moe.const import ScoreFunc
from torchtitan.ops.router import router_topk_indices
from torchtitan.protocols.module import Module

__all__ = ["TokenChoiceTopKRouter"]


class TokenChoiceTopKRouter(Module):
    """Token-choice top-K routing: each token is routed to its top-K experts by
    router score.

    Optionally supports node-limited (group-limited) routing: experts are divided
    into groups (e.g. by node) and only ``num_limited_groups`` groups are considered
    before selecting top_k, reducing cross-node communication.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = TokenChoiceTopKRouterConfig

    def __init__(self, config: Config):
        super().__init__()
        assert config.gate is not None
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance
        self._debug_force_first_group = config._debug_force_first_group

    def _debug_force_load_balance_routing(
        self, scores_BLE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample experts uniformly in expectation, respecting group limits."""
        scores_for_choice_BLE = torch.rand_like(scores_BLE)
        if self.num_expert_groups is not None:
            scores_for_choice_BLE = self._get_node_limited_routing_scores(
                scores_for_choice_BLE
            )
        ids = router_topk_indices(scores_for_choice_BLE, self.top_k)
        return ids, scores_BLE.gather(dim=-1, index=ids)

    def _debug_force_first_group_routing(
        self, scores_BLE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route every token to experts ``0..top_k-1``: the first EP rank's experts
        (when ``top_k <= num_experts // ep_size``), so the other EP ranks receive
        nothing. Like Foundry's first_expert_group, return constant 1/top_k
        weights, independent of the gate and route_norm/route_scale. This debug
        routing does not propagate the expert-output gradient into the gate.
        """

        def _first_group(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            bs, slen, _ = scores.shape
            ids = (
                torch.arange(self.top_k, device=scores.device, dtype=torch.int64)
                .view(1, 1, self.top_k)
                .expand(bs, slen, self.top_k)
                .contiguous()
            )
            return ids, scores.new_full(ids.shape, 1.0 / self.top_k)

        if not isinstance(scores_BLE, DTensor):
            return _first_group(scores_BLE)
        return local_map(
            _first_group,
            in_placements=(scores_BLE.placements,),
            out_placements=(scores_BLE.placements, scores_BLE.placements),
            device_mesh=scores_BLE.device_mesh,
        )(scores_BLE)

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice_BLE: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
        and set expert scores in non-selected groups as -inf.

        Args:
            scores_for_choice_BLE: Router scores with expert_bias (if any), shape ``(B, L, E)``.

        Returns:
            scores_for_choice_BLE: shape ``(B, L, E)``.
        """
        assert self.num_expert_groups is not None
        assert self.num_limited_groups is not None
        experts_per_group = self.num_experts // self.num_expert_groups
        scores_grouped = scores_for_choice_BLE.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        top2_indices = router_topk_indices(scores_grouped, 2, sorted=True)
        top2_scores_in_group = scores_grouped.gather(-1, top2_indices)
        group_scores = top2_scores_in_group.sum(dim=-1)
        group_idx = router_topk_indices(group_scores, self.num_limited_groups)
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(-1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice_BLE = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).flatten(-2)

        return scores_for_choice_BLE

    def forward(
        self, x_BLD: torch.Tensor, expert_bias_E: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_BLD: Input ``(B, L, D)``.
            expert_bias_E: Optional load-balancing bias ``(E,)``.

        Returns:
            topk_scores_BLK: Routing scores ``(B, L, K)``.
            topk_expert_ids_BLK: Expert indices ``(B, L, K)``.
            scores_BLE: Full routing scores ``(B, L, E)`` (post sigmoid/softmax).
            logits_BLE: Raw pre-activation gate logits ``(B, L, E)`` (float32),
                used for the router z-loss.
        """
        if x_BLD.device.type in ("cpu", "meta"):
            logits_BLE = self.gate(x_BLD.float())
        else:
            with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
                logits_BLE = self.gate(x_BLD)

        # sigmoid / softmax run in float32 to avoid loss explosion; logits_BLE is float32 above.
        if self.score_func == ScoreFunc.SIGMOID:
            scores_BLE = torch.sigmoid(logits_BLE)
        elif self.score_func == ScoreFunc.SOFTMAX:
            scores_BLE = F.softmax(logits_BLE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self._debug_force_load_balance:
            (
                topk_expert_ids_BLK,
                topk_scores_BLK,
            ) = self._debug_force_load_balance_routing(scores_BLE)
        elif self._debug_force_first_group:
            (
                topk_expert_ids_BLK,
                topk_scores_BLK,
            ) = self._debug_force_first_group_routing(scores_BLE)
        else:
            scores_for_choice_BLE = (
                scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
            )
            # Apply node-limited routing if configured
            if self.num_expert_groups is not None:
                scores_for_choice_BLE = self._get_node_limited_routing_scores(
                    scores_for_choice_BLE
                )
            topk_expert_ids_BLK = router_topk_indices(scores_for_choice_BLE, self.top_k)
            # expert_bias steers the choice only; the gate value comes from the
            # unbiased scores.
            topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)

        if not self._debug_force_first_group:
            if self.route_norm:
                denominator = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
                topk_scores_BLK = topk_scores_BLK / denominator
            topk_scores_BLK = topk_scores_BLK * self.route_scale

        return (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
            logits_BLE,
        )
