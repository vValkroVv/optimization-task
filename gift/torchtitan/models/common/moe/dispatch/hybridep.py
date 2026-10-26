from dataclasses import dataclass
from torchtitan.config.schema.models.common.moe.dispatch.hybridep import (
    HybridEPTokenDispatcherConfig,
)
from typing import TYPE_CHECKING, TypeAlias

import torch


from .base import BaseEPTokenDispatcher

if TYPE_CHECKING:
    from torchtitan.distributed.deepep.hybridep import DispatchState


@dataclass(frozen=True, kw_only=True)
class HybridEPDispatchMetadata:
    """The opaque HybridEP handle ``dispatch`` produces and ``combine`` consumes.

    Its own type, not DeepEP's: the two share no field beyond the name.
    """

    state: "DispatchState"


class HybridEPTokenDispatcher(BaseEPTokenDispatcher[HybridEPDispatchMetadata]):
    """Token dispatcher using HybridEP for efficient token dispatch/combine.

    Uses HybridEP library kernels (GB200/NVLink72) instead of standard
    all-to-all collectives.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = HybridEPTokenDispatcherConfig

    def __init__(self, config: Config):
        super().__init__(config)
        self._capacity_config = config
        self.non_blocking_capacity_factor = config.non_blocking_capacity_factor
        self.pad_multiple = config.pad_multiple

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import hybridep

        for name, value in (
            ("num_sms_dispatch", config.num_sms_dispatch),
            ("num_sms_combine", config.num_sms_combine),
        ):
            if value is not None and value <= 0:
                raise ValueError(
                    f"HybridEP {name} must be a positive SM count, got {value}."
                )
        hybridep.set_num_sms(
            dispatch=config.num_sms_dispatch, combine=config.num_sms_combine
        )
        hybridep.set_kernel_options(
            comm_stream=config.comm_stream, fuse_permute=config.fuse_permute
        )

        self._dispatch_tokens = hybridep.dispatch_tokens
        self._combine_tokens = hybridep.combine_tokens
        self.reserve_tokens_per_rank = hybridep.reserve_tokens_per_rank

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        score_before_experts: bool,
        use_fp8_dispatch: bool = False,
        expert_weights: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, HybridEPDispatchMetadata]:
        """Dispatch through HybridEP.

        Args:
            use_fp8_dispatch: produce the padded expert-major fp8 layout. Not a
                knob of this backend: that layout exists for the fp8 grouped
                GEMMs, so ``GroupedExperts`` turns it on with them.
            expert_weights: this rank's expert weights, quantized alongside the
                communication instead of in the experts. fp8 dispatch only.
        """
        # HybridEP does not route by num_local_tokens_per_expert_E — it derives
        # the global counts from the other inputs; the histogram is only
        # reported to the monitor at the end.
        num_experts = num_local_tokens_per_expert_E.numel()
        ep_group = self._mesh().get_group()
        num_local_experts = num_experts // ep_group.size()
        if (
            self.token_drop is not None
            and self.non_blocking_capacity_factor is not None
        ):
            # The buffer and gate proof needs equal source shapes, guaranteed by
            # the configured training layout. Refuse dynamic/short local inputs;
            # do not read device counts and add a host sync to nonblocking mode.
            expected = self._capacity_num_tokens
            if expected is None or topk_scores_TK.shape[0] != expected:
                raise ValueError(
                    "HybridEP nonblocking token_drop requires equal source token "
                    "counts matching the resolved training layout; expected "
                    f"{expected}, got {topk_scores_TK.shape[0]}. Bind the dispatcher "
                    "with with_capacity_context or use blocking dispatch for "
                    "unequal/variable source lengths."
                )
            self._capacity_config.validate_capacity(
                num_tokens=topk_scores_TK.shape[0],
                top_k=topk_scores_TK.shape[1],
                num_experts=num_experts,
                ep_size=ep_group.size(),
            )
        requested_scores = topk_scores_TK if self.token_drop is not None else None
        topk_scores_TK, topk_expert_ids_TK, keep = self._drop_routes(
            topk_scores_TK,
            topk_expert_ids_TK,
            num_experts=num_experts,
            ep_group=ep_group,
        )

        hidden_states_RD, num_global_tokens_per_local_expert_e, state = (
            self._dispatch_tokens(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                num_local_experts,
                num_experts,
                ep_group,
                score_before_experts=score_before_experts,
                non_blocking_expert_capacity_factor=self.non_blocking_capacity_factor,
                pad_multiple=self.pad_multiple,
                use_fp8_dispatch=use_fp8_dispatch,
                expert_weights=expert_weights,
            )
        )

        metadata = HybridEPDispatchMetadata(state=state)
        # TODO (veveselov): under gradient edges this reports the edge, an expanded zero,
        # not the payload the experts compute on. Harmless while only the row
        # count is read -- it agrees either way -- but a monitor reading the
        # values would see zeros with nothing to say so. DeepEP reports the
        # payload here.
        self._dispatch(
            local_tokens_per_expert=num_local_tokens_per_expert_E,
            global_tokens_per_local_expert=num_global_tokens_per_local_expert_e,
            routed_input=hidden_states_RD,
            topk_expert_ids_TK=topk_expert_ids_TK,
            requested_topk_scores=requested_scores,
            capacity_keep_mask=keep,
        )
        return hidden_states_RD, num_global_tokens_per_local_expert_e, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: HybridEPDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Combine tokens via HybridEP."""
        return self._combine_tokens(
            routed_output_RD, metadata.state, pad_multiple=self.pad_multiple
        )
