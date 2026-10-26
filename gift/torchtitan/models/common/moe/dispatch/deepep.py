from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import torch


from .base import BaseEPTokenDispatcher
from torchtitan.config.schema.models.common.moe.dispatch.deepep import (
    DeepEPTokenDispatcherConfig,
)

if TYPE_CHECKING:
    from torchtitan.distributed.deepep.deepep import DispatchState


@dataclass(frozen=True, kw_only=True)
class DeepEPDispatchMetadata:
    """The opaque DeepEP handle ``dispatch`` produces and ``combine`` consumes."""

    state: "DispatchState"


class DeepEPTokenDispatcher(BaseEPTokenDispatcher[DeepEPDispatchMetadata]):
    """Token dispatcher using DeepEP for efficient token dispatch/combine.

    Uses DeepEP library kernels (H100/NVLink Switch) instead of standard
    all-to-all collectives. Combine is asynchronous — callers must call
    sync_combine() before using the result.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = DeepEPTokenDispatcherConfig

    def __init__(self, config: Config):
        super().__init__(config)

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac; the
        # entry points come from the same import rather than a second one.
        from torchtitan.distributed.deepep import deepep

        if config.num_sms is not None:
            deepep.set_num_sms(config.num_sms)

        self._dispatch_tokens = deepep.dispatch_tokens
        self._combine_tokens = deepep.combine_tokens
        self._sync_combine = deepep.sync_combine

    def sync_combine(self) -> None:
        """DeepEP's combine is asynchronous; wait before reading its output."""
        self._sync_combine()

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
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        """Dispatch through DeepEP.

        Args:
            use_fp8_dispatch: produce the padded expert-major fp8 layout. Not a
                knob of this backend: that layout exists for the fp8 grouped
                GEMMs, so ``GroupedExperts`` turns it on with them.
            expert_weights: this rank's expert weights, quantized alongside the
                communication instead of in the experts. fp8 dispatch only.
        """
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                "DeepEP dispatch does not support torch.compile: its ops are not "
                "traceable. Use the hybridep backend, or disable compile."
            )

        # DeepEP does not route by num_local_tokens_per_expert_E — it derives
        # the global counts from the other inputs; the histogram is only
        # reported to the monitor at the end.
        num_experts = num_local_tokens_per_expert_E.numel()
        ep_group = self._mesh().get_group()
        num_local_experts = num_experts // ep_group.size()
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
                use_fp8_dispatch=use_fp8_dispatch,
                expert_weights=expert_weights,
            )
        )

        metadata = DeepEPDispatchMetadata(state=state)
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
        metadata: DeepEPDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Combine tokens via DeepEP.

        Asynchronous: ``sync_combine()`` is deferred to ``MoE.forward`` so the
        wait overlaps with ``shared_experts``.
        """
        return self._combine_tokens(routed_output_RD, metadata.state)
