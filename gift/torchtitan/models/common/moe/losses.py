"""MoE auxiliary losses.

torchtitan's default MoE balances load with the auxiliary-loss-free scheme
(https://arxiv.org/abs/2408.15664), driven by ``expert_bias_E`` updates rather than a
loss term. Some models additionally use a small DeepSeek-V3
sequence-wise auxiliary loss on top of that; :func:`sequence_aux_loss` implements it.
These scalars are injected into the graph via the grad-only
``AddAuxiliaryLoss`` splice in ``common/regularization.py``.
"""

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.experimental import local_map

from torchtitan.config.schema.models.common.moe.const import ScoreFunc
from torchtitan.models.common.regularization import masked_mean

__all__ = [
    "sequence_aux_loss",
    "router_z_loss",
]


def sequence_aux_loss(
    scores_BLE: torch.Tensor,
    *,
    top_k: int,
    score_func: ScoreFunc,
    coef: float,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """DeepSeek-V3 sequence-wise auxiliary load-balancing loss.

    Args:
        scores_BLE: Router scores ``(B, L, E)`` — the grad-connected, **pre-bias**
            softmax/sigmoid gate output.
        top_k: Experts selected per token.
        score_func: Router scoring function (sigmoid scores are re-normalized per token;
            softmax scores already sum to 1 over experts).
        coef: Loss weight (``router_aux_loss_coef``).
        token_mask: ``(B, L)``, true on trained-on tokens; padding would otherwise
            inflate both ``f_i`` and ``P_i``. ``None`` counts every position.

    Per sequence in the batch::

        f_i = E * (# times expert i is selected) / (L * top_k)   # greedy top-k, unbiased
        P_i = mean_over_L( normalized score of expert i )
        aux = coef * mean_over_B( sum_i f_i * P_i )

    Only ``P`` carries gradient (``f`` comes from discrete top-k indices), so the loss pushes
    the routed probability mass toward balance.
    """

    def _seq_aux_dots(
        scores_BLE: torch.Tensor, token_mask_BL: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-sequence ``sum_i f_i * P_i`` on a plain, full-``L`` tensor; returns ``(B,)``."""
        B, L, E = scores_BLE.shape
        scores_f = scores_BLE.float()

        _, aux_ids_BLK = torch.topk(scores_f, k=top_k, dim=-1, sorted=False)
        if score_func == ScoreFunc.SIGMOID:
            probs_BLE = scores_f / (scores_f.sum(dim=-1, keepdim=True) + 1e-20)
        else:  # softmax scores already sum to 1 over experts
            probs_BLE = scores_f

        if token_mask_BL is None:
            weights = torch.ones(
                B, L * top_k, dtype=torch.float32, device=scores_BLE.device
            )
        else:
            mask_BL1 = token_mask_BL.to(torch.float32).unsqueeze(-1)
            weights = mask_BL1.expand(B, L, top_k).reshape(B, L * top_k)
            probs_BLE = probs_BLE * mask_BL1

        ce_BE = torch.zeros(B, E, dtype=torch.float32, device=scores_BLE.device)
        ce_BE.scatter_add_(1, aux_ids_BLK.reshape(B, L * top_k), weights)
        ce_BE = ce_BE / (L * top_k / E)
        probs_mean_BE = probs_BLE.mean(dim=1)

        if token_mask_BL is not None:
            # Rescale both all-L averages to per-real-token. As a rescale rather
            # than a different divisor, an all-true mask scales by exactly 1.0.
            scale_B1 = L / mask_BL1.sum(dim=1).clamp(min=1.0)
            ce_BE = ce_BE * scale_B1
            probs_mean_BE = probs_mean_BE * scale_B1

        return (ce_BE * probs_mean_BE).sum(dim=1)

    if isinstance(scores_BLE, DTensor):
        placements = scores_BLE.placements
        # All-gather only the sequence axis L: any axis sharding tensor dim 1 (CP always; the
        # TP axis when EP is on) -> Replicate. Batch (Shard(0)) stays sharded; E is never sharded.
        in_pl = tuple(
            Replicate() if (isinstance(p, Shard) and p.dim == 1) else p
            for p in placements
        )
        # Output is per-sequence (B,): keep the batch Shard(0); gathered L axes become Replicate
        # (each rank redundantly holds the full-L result).
        out_pl = tuple(
            p if (isinstance(p, Shard) and p.dim == 0) else Replicate()
            for p in placements
        )
        # in_grad_placements == in_pl: L was all-gathered, so each rank's grad w.r.t. the full-L
        # scores is complete and identical (Replicate, sliced back to the L shard); the batch axis
        # is separable (Shard(0)). No cross-rank reduction is implied on any axis.
        args: tuple[torch.Tensor, ...] = (scores_BLE,)
        in_placements = (in_pl,)
        if token_mask is not None:
            # The (B, L) mask wants out_pl too: batch sharding kept, L replicated.
            args += (token_mask,)
            in_placements += (out_pl,)
        seq_dots_B = local_map(
            _seq_aux_dots,
            in_placements=in_placements,
            out_placements=(out_pl,),
            in_grad_placements=in_placements,
            device_mesh=scores_BLE.device_mesh,
            redistribute_inputs=True,  # all-gather the sequence axis L to in_pl
        )(*args)
    else:
        seq_dots_B = _seq_aux_dots(scores_BLE, token_mask)

    return coef * seq_dots_B.mean()


def router_z_loss(
    router_logits_BLE: torch.Tensor,
    *,
    coef: float,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Router z-loss: ``coef * logsumexp(router_logits, dim=-1)**2``, mean over tokens.

    Penalizes large router gate logits (the pre-activation gate output, not the
    softmax/sigmoid scores). Computed in float32 for stability (the gate already
    runs under a float32 autocast, so ``.float()`` is a no-op safeguard).
    """
    per_token = coef * torch.logsumexp(router_logits_BLE.float(), dim=-1).square()
    return masked_mean(per_token, token_mask)
