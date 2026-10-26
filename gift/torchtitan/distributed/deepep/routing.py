"""Dense HybridEP routing metadata with support for dropped (-1) assignments."""

import torch


def indices_to_map(
    expert_ids: torch.Tensor, weights: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build contiguous metadata with one trailing sink element for inactive slots."""
    num_tokens = expert_ids.shape[0]
    num_entries = num_tokens * num_experts
    valid = expert_ids >= 0
    offsets = torch.arange(0, num_entries, num_experts, device=expert_ids.device)[
        :, None
    ]
    indices = (
        (expert_ids.long() + offsets).masked_fill_(~valid, num_entries).reshape(-1)
    )
    routing = torch.zeros(num_entries + 1, dtype=torch.bool, device=expert_ids.device)
    routing.scatter_(0, indices, valid.reshape(-1))
    probs = torch.zeros(num_entries + 1, dtype=torch.float32, device=weights.device)
    probs.scatter_(0, indices, weights.float().masked_fill(~valid, 0).reshape(-1))
    return (
        routing[:num_entries].view(num_tokens, num_experts),
        probs[:num_entries].view(num_tokens, num_experts),
    )


def gather_route_gradients(
    dense_gradients: torch.Tensor, expert_ids: torch.Tensor
) -> torch.Tensor:
    """Gather gradients of retained assignments; dropped slots receive zero."""
    return dense_gradients.gather(1, expert_ids.long().clamp_min(0)).masked_fill(
        expert_ids < 0, 0
    )
