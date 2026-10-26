import torch
from torch.nn import functional as F
from grouped_gemm.ops import gmm

from llmfoundry.models.ops.float8.triton_kernels.fixed_transformer_engine_permutation import (
    moe_permute_with_probs as fused_permute_with_probs,
    moe_unpermute as fused_unpermute
)

def main():
    hidden_size = 8
    output_size = hidden_size
    num_local_experts = 3
    weight = torch.randn(hidden_size, output_size * num_local_experts, device='cuda', dtype=torch.bfloat16)
    weight = weight.view(num_local_experts, hidden_size, output_size)

    routing_ixes = torch.tensor([[1, 2], [0, 1], [0, 2], [1, 2]], dtype=torch.long, device='cuda')
    routing_map = F.one_hot(routing_ixes.ravel()).view(*routing_ixes.size(), -1).sum(-2).to(torch.int32)
    hidden_states = torch.randn(routing_map.size(0), hidden_size, device='cuda', dtype=torch.bfloat16)
    probs = torch.ones_like(routing_map, dtype=torch.float32) / num_local_experts
    tokens_per_expert = routing_map.sum(dim=0).long().cpu()
    hidden_shape_before_permute = hidden_states.size()

    (
        permuted_hidden_states,
        _,
        reversed_permutation_mapping,
    ) = fused_permute_with_probs(
        hidden_states.to(torch.bfloat16),
        probs.to(torch.bfloat16),
        routing_map.to(torch.int32),
        num_out_tokens=tokens_per_expert.sum(),
    )

    output = gmm(permuted_hidden_states, weight, tokens_per_expert)

    unpermute_outputs = fused_unpermute(
        output,
        reversed_permutation_mapping,
        merging_probs=None,
        restore_shape=hidden_shape_before_permute,
    )

    print(unpermute_outputs, unpermute_outputs.size())


if __name__ == "__main__":
    main()
