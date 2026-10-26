import unittest.mock as mock
from typing import Optional, Tuple

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig

TOLERANCE_BF16 = 1e-2
TOLERANCE_BF16_RELAXED = 2e-2
TOLERANCE_FP32 = 1e-4


def _gigamix_moe_block_test_config(
    num_routed_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    expert_intermediate_size: int,
    *,
    max_position_embeddings: int = 4096,
    moe_enable_deepep: bool = False,
    moe_deepep_num_sms: Optional[int] = None,
) -> GigaMixConfig:
    return GigaMixConfig(
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        intermediate_size=expert_intermediate_size,
        max_position_embeddings=max_position_embeddings,
        init_type="deepseek",
        init_device="cuda",
        tp_size=1,
        dispatcher_balancing_strategy="fixed_capacity",
        expert_capacity_tokens=2_000_000,
        drop_by_group_capacity=True,
        peak_capacity_microbatchsize=1,
        moe_deepep_num_sms=moe_deepep_num_sms,
        moe_enable_deepep=moe_enable_deepep,
        fused_mlp=False,
        use_cache=False,
    )


def _generate_routing_and_input(
    batch_size: int,
    hidden_dim: int,
    n_experts: int,
    top_k: int,
    drop_prob: float,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    input_std: float = 2e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m_tokens = batch_size
    probs = torch.rand(m_tokens, n_experts, dtype=dtype, device=device)
    probs = torch.softmax(probs, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=1)
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    X = torch.empty(m_tokens, hidden_dim, dtype=dtype, device=device).normal_(mean=0, std=input_std)
    token_indices = torch.arange(m_tokens, device=X.device, dtype=torch.int32).repeat_interleave(top_k)
    if drop_prob > 0:
        mask = torch.rand(token_indices.shape[0], device=device) > drop_prob
        top_k_indices = top_k_indices.view(-1)[mask]
        top_k_probs = top_k_probs.view(-1)[mask]
        token_indices = token_indices[mask]
    else:
        top_k_indices = top_k_indices.view(-1)
        top_k_probs = top_k_probs.view(-1)
    return X, top_k_indices, top_k_probs, token_indices


def _to_sparse_routing(
    length: int,
    E: int,
    k: int,
    k_idxs: torch.Tensor,
    k_weights: torch.Tensor,
    token_idxs: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_weights_sparse = torch.zeros((length, E), dtype=k_weights.dtype, device=device)
    k_idxs_sparse = torch.zeros((length, E), device=device).bool()
    if token_idxs is not None:
        k_weights_sparse[token_idxs.long(), k_idxs.long()] = k_weights
        k_idxs_sparse[token_idxs.long(), k_idxs.long()] = True
    else:
        row = torch.arange(length, device=device).repeat_interleave(k)
        k_weights_sparse[row, k_idxs.reshape(length, k).reshape(-1)] = k_weights.reshape(length, k).reshape(-1)
        k_idxs_sparse.scatter_(1, k_idxs.reshape(length, k), True)
    return k_idxs_sparse, k_weights_sparse


def _fused_swiglu_chunked(x: torch.Tensor) -> torch.Tensor:
    x = torch.chunk(x, 2, dim=-1)
    return F.silu(x[0]) * x[1]


def _naive_moe_forward(
    x: torch.Tensor,
    expert_p: torch.Tensor,
    expert_idxs: torch.Tensor,
    token_idxs: torch.Tensor,
    up_expert_weights: torch.Tensor,
    down_expert_weights: torch.Tensor,
) -> torch.Tensor:
    output_acc = torch.zeros(x.shape[0], x.shape[-1], dtype=x.dtype, device=x.device)
    for i in range(len(token_idxs)):
        e_i = expert_idxs[i].item()
        t_i = token_idxs[i].item()
        out = expert_p[i] * F.linear(
            _fused_swiglu_chunked(F.linear(x[t_i], up_expert_weights[e_i])),
            down_expert_weights[e_i],
        )
        output_acc[t_i].add_(out)
    return output_acc


def _naive_weights_to_curr(
    up_expert_weights: torch.Tensor,
    down_expert_weights: torch.Tensor,
    experts_module: nn.Module,
    config: GigaMixConfig,
) -> None:
    E, H, I = config.num_routed_experts, config.hidden_size, config.intermediate_size
    experts_module.weight2 = nn.Parameter(
        down_expert_weights.permute(0, 2, 1).contiguous().reshape(I * E, H)
    )
    experts_module.weight1 = nn.Parameter(
        up_expert_weights.permute(0, 2, 1).contiguous().reshape(H, 2 * I * E)
    )


def _gemm_to_sonic_w1(w1_gemm: torch.Tensor, E: int, H: int, I: int) -> torch.Tensor:
    w = w1_gemm.view(E, H, 2 * I)
    gate, up = w[:, :, :I], w[:, :, I:]
    w_interleaved = torch.stack([gate, up], dim=-1).reshape(E, H, 2 * I)
    return w_interleaved.permute(0, 2, 1).contiguous().view(H, 2 * I * E)


def _gemm_to_sonic_w2(w2_gemm: torch.Tensor, E: int, H: int, I: int) -> torch.Tensor:
    return w2_gemm.view(E, I, H).permute(0, 2, 1).contiguous().view(I * E, H)


def _gemm_weights_to_sonic(gemm_model: nn.Module, sonic_model: nn.Module, config: GigaMixConfig) -> None:
    E, H = config.num_routed_experts, config.hidden_size
    I = config.intermediate_size
    sonic_model.experts.weight1 = nn.Parameter(
        _gemm_to_sonic_w1(gemm_model.experts.weight1.data, E, H, I)
    )
    sonic_model.experts.weight2 = nn.Parameter(
        _gemm_to_sonic_w2(gemm_model.experts.weight2.data, E, H, I)
    )


def _grad_sonic_to_gemm_w1(grad: torch.Tensor, config: GigaMixConfig) -> torch.Tensor:
    E, H, I = config.num_routed_experts, config.hidden_size, config.intermediate_size
    w = grad.view(E, 2 * I, H).permute(0, 2, 1).contiguous()
    gate, up = w[:, :, 0::2], w[:, :, 1::2]
    return torch.cat([gate, up], dim=-1).contiguous().view(H, 2 * I * E)


def _grad_sonic_to_gemm_w2(grad: torch.Tensor, config: GigaMixConfig) -> torch.Tensor:
    E, H, I = config.num_routed_experts, config.hidden_size, config.intermediate_size
    return grad.view(E, H, I).permute(0, 2, 1).contiguous().view(I * E, H)


def _assert_close(name: str, ref: torch.Tensor, other: torch.Tensor, tolerance: float = 1e-2) -> None:
    diff = torch.abs(ref - other)
    max_diff = diff.max().item()
    assert max_diff < tolerance, f"{name} max_diff={max_diff} >= {tolerance}"


class TestNaiveVsDeepseekGMMMoe:

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("length", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("hid_dim, inter_dim, k", [(512, 512, 4)])
    def test_naive_vs_deepseek_forward(self, length, hid_dim, inter_dim, E, k, dtype, moe_modules):
        device = torch.device("cuda:0")
        X, k_idxs, k_weights, token_idxs = _generate_routing_and_input(
            length, hid_dim, E, k, drop_prob=0.0, device=device, dtype=dtype, input_std=0.02
        )
        X = X.requires_grad_()
        k_weights = k_weights.requires_grad_()
        k_idxs_sparse, k_weights_sparse = _to_sparse_routing(
            length, E, k, k_idxs, k_weights, token_idxs, device
        )
        config = _gigamix_moe_block_test_config(
            num_routed_experts=E,
            num_experts_per_tok=k,
            hidden_size=hid_dim,
            expert_intermediate_size=inter_dim,
        )
        with mock.patch("composer.utils.dist.get_ep_group_size", return_value=1):
            model = moe_modules.DeepseekGMMMoeBlock(config).to(device, dtype=dtype)
        for m in model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        up_naive = torch.empty(E, 2 * inter_dim, hid_dim, dtype=dtype, device=device)
        down_naive = torch.empty(E, hid_dim, inter_dim, dtype=dtype, device=device)
        nn.init.xavier_uniform_(up_naive)
        nn.init.xavier_uniform_(down_naive)
        _naive_weights_to_curr(up_naive, down_naive, model.experts, config)
        Y_deepseek = model(X, k_idxs_sparse, k_weights_sparse)
        Y_naive = _naive_moe_forward(
            X, k_weights.view(-1), k_idxs.view(-1), token_idxs.view(-1),
            up_naive, down_naive,
        )
        tol = TOLERANCE_BF16 if dtype == torch.bfloat16 else TOLERANCE_FP32
        _assert_close("Y", Y_naive, Y_deepseek, tolerance=tol)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("length", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("hid_dim, inter_dim, k", [(512, 512, 4)])
    def test_naive_vs_deepseek_backward(self, length, hid_dim, inter_dim, E, k, dtype, moe_modules):
        device = torch.device("cuda:0")
        X, k_idxs, k_weights, token_idxs = _generate_routing_and_input(
            length, hid_dim, E, k, drop_prob=0.0, device=device, dtype=dtype, input_std=0.02
        )
        X = X.requires_grad_()
        DY = torch.randn(length, hid_dim, dtype=dtype, device=device)
        k_weights = k_weights.requires_grad_()
        k_idxs_sparse, k_weights_sparse = _to_sparse_routing(
            length, E, k, k_idxs, k_weights, token_idxs, device
        )
        config = _gigamix_moe_block_test_config(
            num_routed_experts=E,
            num_experts_per_tok=k,
            hidden_size=hid_dim,
            expert_intermediate_size=inter_dim,
        )
        with mock.patch("composer.utils.dist.get_ep_group_size", return_value=1):
            model = moe_modules.DeepseekGMMMoeBlock(config).to(device, dtype=dtype)
        for m in model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        up_naive = torch.empty(E, 2 * inter_dim, hid_dim, dtype=dtype, device=device, requires_grad=True)
        down_naive = torch.empty(E, hid_dim, inter_dim, dtype=dtype, device=device, requires_grad=True)
        nn.init.xavier_uniform_(up_naive)
        nn.init.xavier_uniform_(down_naive)
        _naive_weights_to_curr(up_naive, down_naive, model.experts, config)
        Y_deepseek = model(X, k_idxs_sparse, k_weights_sparse)
        grads_deepseek = torch.autograd.grad(
            outputs=(Y_deepseek,),
            inputs=(X, k_weights, model.experts.weight1, model.experts.weight2),
            grad_outputs=(DY,),
        )
        Y_naive = _naive_moe_forward(
            X, k_weights.view(-1), k_idxs.view(-1), token_idxs.view(-1),
            up_naive, down_naive,
        )
        grads_naive = torch.autograd.grad(
            outputs=(Y_naive,),
            inputs=(X, k_weights, up_naive, down_naive),
            grad_outputs=(DY,),
        )
        dX_naive, dg_naive, dW1_naive, dW2_naive = grads_naive
        dW1_curr = dW1_naive.permute(0, 2, 1).contiguous().reshape(hid_dim, 2 * inter_dim * E)
        dW2_curr = dW2_naive.permute(0, 2, 1).contiguous().reshape(inter_dim * E, hid_dim)
        tol = TOLERANCE_BF16
        _assert_close("dX", grads_deepseek[0], dX_naive, tolerance=tol)
        _assert_close("dg", grads_deepseek[1], dg_naive, tolerance=tol)
        _assert_close("dW1", grads_deepseek[2], dW1_curr, tolerance=tol)
        _assert_close("dW2", grads_deepseek[3], dW2_curr, tolerance=tol)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("drop_prob", [0.3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("length", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("hid_dim, inter_dim, k", [(512, 512, 4)])
    def test_naive_vs_deepseek_varlen_k(self, length, hid_dim, inter_dim, E, k, dtype, drop_prob, moe_modules):
        device = torch.device("cuda:0")
        X, k_idxs, k_weights, token_idxs = _generate_routing_and_input(
            length, hid_dim, E, k, drop_prob=drop_prob, device=device, dtype=dtype, input_std=0.02
        )
        X = X.requires_grad_()
        k_weights = k_weights.requires_grad_()
        k_idxs_sparse, k_weights_sparse = _to_sparse_routing(
            length, E, k, k_idxs, k_weights, token_idxs, device
        )
        config = _gigamix_moe_block_test_config(
            num_routed_experts=E,
            num_experts_per_tok=k,
            hidden_size=hid_dim,
            expert_intermediate_size=inter_dim,
        )
        with mock.patch("composer.utils.dist.get_ep_group_size", return_value=1):
            model = moe_modules.DeepseekGMMMoeBlock(config).to(device, dtype=dtype)
        for m in model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        up_naive = torch.empty(E, 2 * inter_dim, hid_dim, dtype=dtype, device=device)
        down_naive = torch.empty(E, hid_dim, inter_dim, dtype=dtype, device=device)
        nn.init.xavier_uniform_(up_naive)
        nn.init.xavier_uniform_(down_naive)
        _naive_weights_to_curr(up_naive, down_naive, model.experts, config)
        Y_deepseek = model(X, k_idxs_sparse, k_weights_sparse)
        Y_naive = _naive_moe_forward(
            X, k_weights.view(-1), k_idxs.view(-1), token_idxs.view(-1),
            up_naive, down_naive,
        )
        _assert_close("Y varlen_k", Y_naive, Y_deepseek, tolerance=TOLERANCE_BF16_RELAXED)


class TestDeepseekGMMMoeVsSonicGMMMoe:

    @pytest.mark.skip(
        "Skip SonicMoE test until tests on h100 (sr008) would be available"
    )
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("length", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("hid_dim, inter_dim, k", [(512, 512, 4)])
    def test_deepseek_vs_sonic_forward(self, length, hid_dim, inter_dim, E, k, dtype, moe_modules):
        device = torch.device("cuda:0")
        X, k_idxs, k_weights, token_idxs = _generate_routing_and_input(
            length, hid_dim, E, k, drop_prob=0.0, device=device, dtype=dtype, input_std=0.02
        )
        X = X.requires_grad_()
        k_weights = k_weights.requires_grad_()
        k_idxs_sparse, k_weights_sparse = _to_sparse_routing(
            length, E, k, k_idxs, k_weights, token_idxs, device
        )
        config = _gigamix_moe_block_test_config(
            num_routed_experts=E,
            num_experts_per_tok=k,
            hidden_size=hid_dim,
            expert_intermediate_size=inter_dim,
        )
        with mock.patch("composer.utils.dist.get_ep_group_size", return_value=1):
            deepseek_model = moe_modules.DeepseekGMMMoeBlock(config).to(device, dtype=dtype)
            sonic_model = moe_modules.SonicGMMMoeBlock(config).to(device, dtype=dtype)
        deepseek_model.experts.reset_parameters_deepseek()
        _gemm_weights_to_sonic(deepseek_model, sonic_model, config)
        Y_deepseek = deepseek_model(X, None, k_idxs_sparse, k_weights_sparse)
        Y_sonic = sonic_model(X, None, k_idxs_sparse, k_weights_sparse)
        _assert_close("Y deepseek vs sonic", Y_deepseek, Y_sonic, tolerance=TOLERANCE_BF16_RELAXED)

    @pytest.mark.skip(
        "Skip SonicMoE test until tests on h100 (sr008) would be available"
    )
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("length", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("hid_dim, inter_dim, k", [(512, 512, 4)])
    def test_deepseek_vs_sonic_backward(self, length, hid_dim, inter_dim, E, k, dtype, moe_modules):
        device = torch.device("cuda:0")
        X, k_idxs, k_weights, token_idxs = _generate_routing_and_input(
            length, hid_dim, E, k, drop_prob=0.0, device=device, dtype=dtype, input_std=0.02
        )
        X = X.requires_grad_()
        DY = torch.randn(length, hid_dim, dtype=dtype, device=device)
        k_weights = k_weights.requires_grad_()
        k_idxs_sparse, k_weights_sparse = _to_sparse_routing(
            length, E, k, k_idxs, k_weights, token_idxs, device
        )
        config = _gigamix_moe_block_test_config(
            num_routed_experts=E,
            num_experts_per_tok=k,
            hidden_size=hid_dim,
            expert_intermediate_size=inter_dim,
        )
        with mock.patch("composer.utils.dist.get_ep_group_size", return_value=1):
            deepseek_model = moe_modules.DeepseekGMMMoeBlock(config).to(device, dtype=dtype)
            sonic_model = moe_modules.SonicGMMMoeBlock(config).to(device, dtype=dtype)
        deepseek_model.experts.reset_parameters_deepseek()
        _gemm_weights_to_sonic(deepseek_model, sonic_model, config)
        Y_deepseek = deepseek_model(X, None, k_idxs_sparse, k_weights_sparse)
        grads_deepseek = torch.autograd.grad(
            outputs=(Y_deepseek,),
            inputs=(X, k_weights, deepseek_model.experts.weight1, deepseek_model.experts.weight2),
            grad_outputs=(DY,),
            retain_graph=True,
        )
        Y_sonic = sonic_model(X, None, k_idxs_sparse, k_weights_sparse)
        grads_sonic = torch.autograd.grad(
            outputs=(Y_sonic,),
            inputs=(X, k_weights, sonic_model.experts.weight1, sonic_model.experts.weight2),
            grad_outputs=(DY,),
        )
        dW1_sonic_gemm = _grad_sonic_to_gemm_w1(grads_sonic[2], config)
        dW2_sonic_gemm = _grad_sonic_to_gemm_w2(grads_sonic[3], config)
        tol = TOLERANCE_BF16_RELAXED
        _assert_close("dX", grads_deepseek[0], grads_sonic[0], tolerance=tol)
        _assert_close("dg", grads_deepseek[1], grads_sonic[1], tolerance=tol)
        _assert_close("dW1", grads_deepseek[2], dW1_sonic_gemm, tolerance=tol)
        _assert_close("dW2", grads_deepseek[3], dW2_sonic_gemm, tolerance=tol)

    @pytest.mark.skip(
        "Skip SonicMoE test until tests on h100 (sr008) would be available"
    )
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("drop_prob", [0.3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("length", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("hid_dim, inter_dim, k", [(512, 512, 4)])
    def test_deepseek_vs_sonic_varlen_k(self, length, hid_dim, inter_dim, E, k, dtype, drop_prob, moe_modules):
        device = torch.device("cuda:0")
        X, k_idxs, k_weights, token_idxs = _generate_routing_and_input(
            length, hid_dim, E, k, drop_prob=drop_prob, device=device, dtype=dtype, input_std=0.02
        )
        X = X.requires_grad_()
        k_weights = k_weights.requires_grad_()
        k_idxs_sparse, k_weights_sparse = _to_sparse_routing(
            length, E, k, k_idxs, k_weights, token_idxs, device
        )
        config = _gigamix_moe_block_test_config(
            num_routed_experts=E,
            num_experts_per_tok=k,
            hidden_size=hid_dim,
            expert_intermediate_size=inter_dim,
        )
        with mock.patch("composer.utils.dist.get_ep_group_size", return_value=1):
            deepseek_model = moe_modules.DeepseekGMMMoeBlock(config).to(device, dtype=dtype)
            sonic_model = moe_modules.SonicGMMMoeBlock(config).to(device, dtype=dtype)
        deepseek_model.experts.reset_parameters_deepseek()
        _gemm_weights_to_sonic(deepseek_model, sonic_model, config)
        Y_deepseek = deepseek_model(X, None, k_idxs_sparse, k_weights_sparse)
        Y_sonic = sonic_model(X, None, k_idxs_sparse, k_weights_sparse)
        _assert_close("Y varlen_k deepseek vs sonic", Y_deepseek, Y_sonic, tolerance=TOLERANCE_BF16_RELAXED)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit and requires_gpu"])
