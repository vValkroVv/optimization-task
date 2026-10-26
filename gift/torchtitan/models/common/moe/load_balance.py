"""Auxiliary-loss-free load balancing: the expert-bias update between steps."""

from typing import TYPE_CHECKING

import torch
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed.tensor import Replicate

from torchtitan.distributed import ParallelDims
from torchtitan.models.common.moe import MoE

if TYPE_CHECKING:
    # Only the annotation: the hook is registered through duck-typed
    # ``register_step_pre_hook``, so importing at runtime would tie the MoE
    # package to a component that no longer needs to know about MoE.
    from torchtitan.components.optimizer import OptimizersContainer

__all__ = ["register_moe_load_balancing_hook"]


def register_moe_load_balancing_hook(
    optimizers: "OptimizersContainer",
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register an optimizer step pre-hook for MoE auxiliary-loss-free load balancing.

    This function checks if MoE load balancing is enabled and, if so, registers
    a hook that updates expert biases before each optimizer step.

    Args:
        optimizers: The optimizers container to register the hook on.
        model_parts: List of model parts that may contain MoE layers.
        parallel_dims: Parallel dimensions for distributed communication.
    """

    # Resolved once: the set of MoEs is fixed by now
    # (``post_optimizer_build_fn`` runs after ``parallelize_fn``), and a walk of
    # every module is too expensive to repeat twice per step.
    moes = [
        module
        for model_part in model_parts
        for module in model_part.modules()
        if isinstance(module, MoE)
    ]

    def _update_expert_bias(parallel_dims: ParallelDims):
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.
        tokens_per_expert_E_list = []
        for moe in moes:
            if not moe.load_balance_coeff:
                return
            tokens_per_expert_E = moe.tokens_per_expert_E
            # The sign update is invariant to uniformly doubled counts. A
            # checkpoint wrapper does not guarantee recomputation (e.g. a frozen
            # prefix); rounding counts down would erase valid routing evidence.
            tokens_per_expert_E_list.append(tokens_per_expert_E)

        tokens_per_expert_E_by_layer = torch.vstack(tokens_per_expert_E_list)

        if parallel_dims.full_dtensor:
            # full_dtensor: DTensor mesh includes all axes (DP/CP/TP/EP).
            # redistribute Partial→Replicate covers everything.
            assert isinstance(
                tokens_per_expert_E_by_layer, torch.distributed.tensor.DTensor
            )
            dtensor_mesh = tokens_per_expert_E_by_layer.device_mesh
            # TODO: This incurs multiple sequential all-reduces, one per
            # SPMD mesh axis. We should provide a utility to do a single all-reduce
            # on the flattened global SPMD mesh.
            tokens_per_expert_E_by_layer = tokens_per_expert_E_by_layer.redistribute(
                placements=[Replicate()] * dtensor_mesh.ndim
            )
        else:
            # non-full_dtensor: DTensor mesh only has TP/EP (if enabled).
            # full_tensor() reduces on TP/EP, then all-reduce on loss_mesh
            # covers DP/CP separately.
            is_dtensor = isinstance(
                tokens_per_expert_E_by_layer, torch.distributed.tensor.DTensor
            )
            if is_dtensor:
                dtensor_mesh = tokens_per_expert_E_by_layer.device_mesh
                tokens_per_expert_E_by_layer = (
                    tokens_per_expert_E_by_layer.full_tensor()
                )
            if loss_mesh is not None:
                torch.distributed.all_reduce(
                    tokens_per_expert_E_by_layer,
                    group=loss_mesh.get_group(),
                    op=torch.distributed.ReduceOp.SUM,
                )
            if is_dtensor:
                tokens_per_expert_E_by_layer = (
                    torch.distributed.tensor.DTensor.from_local(
                        tokens_per_expert_E_by_layer,
                        # pyrefly: ignore [unbound-name]
                        device_mesh=dtensor_mesh,
                        placements=[Replicate()]
                        * dtensor_mesh.ndim,  # pyrefly: ignore [unbound-name]
                        run_check=False,
                    )
                )

        with torch.no_grad():
            for moe_layer_idx, moe in enumerate(moes):
                tokens_per_expert_E = tokens_per_expert_E_by_layer[
                    moe_layer_idx
                ].float()

                # update the expert bias
                # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                expert_bias_delta_E = moe.load_balance_coeff * torch.sign(
                    tokens_per_expert_E.mean() - tokens_per_expert_E
                )
                expert_bias_delta_E = expert_bias_delta_E - expert_bias_delta_E.mean()
                expert_bias = moe.expert_bias_E
                assert expert_bias is not None
                expert_bias.add_(expert_bias_delta_E)
                moe.tokens_per_expert_E.zero_()

    # Assumption: load_balance_coeff is set universally on all moe blocks.
    if moes and bool(moes[0].load_balance_coeff):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(parallel_dims)
        )
