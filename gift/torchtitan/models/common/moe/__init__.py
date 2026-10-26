import math
from typing import TYPE_CHECKING, TypeAlias

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map


from torchtitan.config.schema.models.common.moe.moe import MoEConfig
from torchtitan.models.common.moe.const import ScoreFunc
from torchtitan.models.common.moe.dispatch import (
    AllToAllTokenDispatcher,
    BaseTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
    LocalTokenDispatcher,
)
from torchtitan.models.common.moe.experts import GroupedExperts
from torchtitan.models.common.moe.losses import router_z_loss, sequence_aux_loss
from torchtitan.models.common.moe.router import (
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.regularization import (
    AddAuxiliaryLoss,
    aux_loss_args,
    declare_aux_loss,
)
from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from torchtitan.config import ParallelismConfig, TrainingConfig
    from torchtitan.distributed import ParallelDims

__all__ = [
    "MoE",
    "GroupedExperts",
    "ScoreFunc",
    "TokenChoiceTopKRouter",
    "AllToAllTokenDispatcher",
    "BaseTokenDispatcher",
    "DeepEPTokenDispatcher",
    "HybridEPTokenDispatcher",
    "LocalTokenDispatcher",
    "reserve_hybridep_tokens",
]


class MoE(Module):
    """Mixture of Experts layer.

    The router stays on DTensor; ``GroupedExperts.forward`` converts to local for
    dispatch → expert compute → combine, then routed and shared-expert outputs are
    summed. Shared experts run on DTensor between the combine and its
    ``sync_combine()``, so they overlap whatever wait an asynchronous backend needs;
    a synchronous backend makes that sync a no-op.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = MoEConfig

    def __init__(self, config: Config):
        super().__init__()

        declare_aux_loss(self, "router_aux_loss", "router_z_loss")

        num_experts = config.experts.num_experts
        self.experts = config.experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert_E is accumulated in the model forward pass.
        #       expert_bias_E is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff:
            if not (
                math.isfinite(self.load_balance_coeff) and self.load_balance_coeff > 0
            ):
                raise ValueError(
                    "load_balance_coeff must be finite and >= 0, "
                    f"got {self.load_balance_coeff}"
                )
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self, x_BLD: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward over ``(B, L, D)`` input.

        Under TP, the MoE wrapper's ``sharding_config`` (set by
        ``set_moe_sharding_config``) redistributes input from sp_layout to the
        desired input layouts and the Partial output back to sp_layout. Operates on
        DTensors; the DTensor→local conversion happens at the GroupedExperts boundary.

        ``token_mask`` keeps padding out of the auxiliary losses and the bias counts;
        it does not change routing or dispatch.
        """
        B, L, D = x_BLD.shape

        # topk_scores_BLK and topk_expert_ids_BLK shape (B, L, K)
        # scores_BLE / logits_BLE shape (B, L, E) (post- / pre-activation)
        (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
            logits_BLE,
        ) = self.router(x_BLD, self.expert_bias_E)

        # Build routing map with scatter. scatter_ does not support mixed
        # local Tensor / DTensor arguments, so run the scatter on local tensors
        # under local_map when router outputs are DTensors.
        # TODO: Remove this local_map workaround once DTensor sharding
        # propagation supports scatter with mixed Tensor / DTensor arguments.
        def _generate_routing_map(
            scores_BLE: torch.Tensor,
            topk_expert_ids_BLK: torch.Tensor,
        ) -> torch.Tensor:
            return torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
                -1,
                topk_expert_ids_BLK,
                True,
            )

        if isinstance(topk_expert_ids_BLK, DTensor):
            assert isinstance(scores_BLE, DTensor), (
                "scores_BLE and topk_expert_ids_BLK should both be DTensors"
            )
            generate_routing_map = local_map(
                _generate_routing_map,
                in_placements=(
                    scores_BLE.placements,
                    topk_expert_ids_BLK.placements,
                ),
                out_placements=(scores_BLE.placements,),
                device_mesh=scores_BLE.device_mesh,
            )
        else:
            generate_routing_map = _generate_routing_map

        routing_map_BLE = generate_routing_map(
            scores_BLE,
            topk_expert_ids_BLK,  # pyrefly: ignore [bad-argument-count]
        )
        num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))

        # tokens_per_expert_E will be used to update the expert bias for load balancing,
        # and also to count the expert usage.
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert_E --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            counted_BLE = routing_map_BLE
            if token_mask is not None:
                counted_BLE = counted_BLE & token_mask.unsqueeze(-1)
            self.tokens_per_expert_E.add_(counted_BLE.sum(dim=(0, 1)))

        out_TD = self.experts(
            x_BLD,
            topk_scores_BLK,
            topk_expert_ids_BLK,
            num_local_tokens_per_expert_E,
        )

        # shared_experts runs in parallel with deepep combine communication.
        shared_out_BLD = (
            self.shared_experts(x_BLD) if self.shared_experts is not None else None
        )

        # shared_experts ran alongside a possibly asynchronous combine; the
        # experts know whether that needs waiting on.
        self.experts.sync_combine()

        out_BLD = out_TD
        if shared_out_BLD is not None:
            out_BLD = out_BLD + shared_out_BLD

        if self.training and self.router_aux_loss_coef > 0.0:
            aux = sequence_aux_loss(
                scores_BLE,
                token_mask=token_mask,
                top_k=self.router.top_k,
                score_func=self.router.score_func,
                coef=self.router_aux_loss_coef,
            )
            out_BLD = AddAuxiliaryLoss.apply(
                out_BLD, aux, *aux_loss_args(self, "router_aux_loss")
            )
        if self.training and self.router_z_loss_coef > 0.0:
            z = router_z_loss(
                logits_BLE, coef=self.router_z_loss_coef, token_mask=token_mask
            )
            out_BLD = AddAuxiliaryLoss.apply(
                out_BLD, z, *aux_loss_args(self, "router_z_loss")
            )
        return out_BLD

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert isinstance(buffer_device, torch.device)

        with torch.device(buffer_device):
            self.tokens_per_expert_E = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff:
                self.expert_bias_E = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )


def reserve_hybridep_tokens(
    model: torch.nn.Module,
    *,
    training: "TrainingConfig",
    parallelism: "ParallelismConfig",
    parallel_dims: "ParallelDims",
) -> None:
    """Size the HybridEP buffer for the most tokens a rank will dispatch.

    That is a full-length microbatch: sequence-length warmup only shortens it.
    HybridEP compiles its kernels for the buffer's token capacity and grows it,
    reallocating and recompiling, at each new maximum. The other backends size
    by bytes or allocate per call, so they are not touched.
    """
    dispatchers = [
        module.token_dispatcher
        for module in model.modules()
        if isinstance(
            getattr(module, "token_dispatcher", None), HybridEPTokenDispatcher
        )
    ]
    if not dispatchers:
        return
    microbatch = (
        parallelism.pipeline_parallel_microbatch_size
        if parallel_dims.pp_enabled
        else training.local_batch_size
    )
    # CP splits the sequence; under EP the MoE input is also Shard(1) over TP
    # (the router's placement), with or without SP. HybridEP implies EP.
    num_tokens = microbatch * training.seq_len // (parallel_dims.cp * parallel_dims.tp)
    for dispatcher in dispatchers:
        dispatcher.reserve_tokens_per_rank(num_tokens)
