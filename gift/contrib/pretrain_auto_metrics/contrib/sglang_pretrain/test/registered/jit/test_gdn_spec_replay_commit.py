from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")


def ref_verify(h0, q, k, v, a, b, A_log, dt_bias):
    HV, H, K = v.shape[1], k.shape[1], k.shape[2]
    h = h0.float().clone()
    states, outs = [], []
    for t in range(k.shape[0]):
        x = a[t].float() + dt_bias.float()
        softplus_x = torch.where(x <= 20.0, torch.log(1.0 + torch.exp(x)), x)
        g = -torch.exp(A_log.float()) * softplus_x
        beta = 1.0 / (1.0 + torch.exp(-b[t].float()))
        qt = q[t].float().repeat_interleave(HV // H, 0)
        kt = k[t].float().repeat_interleave(HV // H, 0)
        qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6) * K**-0.5
        kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
        h = h * torch.exp(g)[:, None, None]
        vt = (v[t].float() - (h * kt[:, None, :]).sum(-1)) * beta[:, None]
        h = h + vt[:, :, None] * kt[:, None, :]
        outs.append((h * qt[:, None, :]).sum(-1))
        states.append(h.clone())
    return torch.stack(states), torch.stack(outs)


def test_reference_delta_rule_cpu():
    torch.manual_seed(0)
    HV, H, K, V = 4, 2, 16, 8
    h0 = torch.randn(HV, V, K)
    k = torch.randn(1, H, K)
    v = torch.randn(1, HV, V)
    a = torch.full((1, HV), -30.0)
    b = torch.full((1, HV), 30.0)
    states, _ = ref_verify(
        h0, torch.randn(1, H, K), k, v, a, b, torch.zeros(HV), torch.zeros(HV)
    )
    kn = k[0].repeat_interleave(HV // H, 0)
    kn = kn / torch.sqrt((kn * kn).sum(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close((states[0] * kn[:, None, :]).sum(-1), v[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("extra_steps", [0, 1])
@pytest.mark.parametrize("track", [False, True])
@pytest.mark.parametrize(
    "HV,H,L,S",
    [(8, 4, 3, 17), (64, 32, 3, 17), (64, 32, 48, 33)],
)
def test_replay_commit_matches_scatter(HV, H, L, S, track, extra_steps):
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import (
        GDNAttnBackend,
        _spec_replay_views,
    )
    from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_with_mask,
    )

    torch.manual_seed(2025)
    dev, dt = "cuda", torch.bfloat16
    P, D, N, K, V = 40, 3, 13, 128, 128
    kv_dim = H * K + HV * V

    A_log = [torch.randn(HV, device=dev) for _ in range(L)]
    dt_bias = [torch.randn(HV, device=dev, dtype=dt) for _ in range(L)]
    q = torch.randn(L, 1, N * D, H, K, device=dev, dtype=dt)
    k = torch.randn(L, 1, N * D, H, K, device=dev, dtype=dt)
    v = torch.randn(L, 1, N * D, HV, V, device=dev, dtype=dt)
    a = torch.randn(L, N * D, HV, device=dev, dtype=dt)
    b = torch.randn(L, N * D, HV, device=dev, dtype=dt)
    ssm0 = torch.randn(L, P, HV, V, K, device=dev)
    perm = torch.randperm(P, device=dev)
    cache_indices = perm[:N].to(torch.int32)
    cache_indices[3] = -1
    lcs = torch.randint(-1, D, (N,), device=dev)
    lcs[0], lcs[1], lcs[2] = -1, D - 1, 0
    track_indices = track_steps = None
    if track:
        track_indices = perm[N : 2 * N].clone()
        track_indices[4] = -1
        track_steps = torch.randint(-1, D, (N,), device=dev)
        track_steps[2], track_steps[3], track_steps[5] = D - 1, 1, -1
    cu = torch.arange(0, N * D + 1, D, dtype=torch.int32, device=dev)
    seats = torch.arange(S, dtype=torch.int32, device=dev)

    def verify(l, ssm, **kw):
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log[l],
            dt_bias=dt_bias[l],
            q=q[l],
            k=k[l],
            v=v[l],
            a=a[l],
            b=b[l],
            initial_state_source=ssm[l],
            initial_state_indices=cache_indices,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=True,
            intermediate_state_indices=seats,
            cache_steps=D,
            **kw,
        )

    ssm_old = ssm0.clone()
    inter = torch.zeros(L, S, D, HV, V, K, device=dev)
    o_old = [verify(l, ssm_old, intermediate_states_buffer=inter[l]) for l in range(L)]
    fused_mamba_state_scatter_with_mask(ssm_old, inter, cache_indices, lcs)
    if track:
        fused_mamba_state_scatter_with_mask(ssm_old, inter, track_indices, track_steps)

    ssm_new = ssm0.clone()
    buf = torch.zeros(L, S, D + extra_steps, kv_dim * 2 // 4 + 2 * HV, device=dev)
    o_new = []
    for l in range(L):
        kv, gb = _spec_replay_views(buf[l], dt, HV)
        o_new.append(verify(l, ssm_new, replay_kv=kv, replay_gb=gb))
        assert torch.equal(ssm_new, ssm0)
    cache = SimpleNamespace(
        temporal=ssm_new,
        intermediate_ssm=buf,
        conv=[torch.empty(0, dtype=dt)],
    )
    backend = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            get_speculative_mamba2_params_all_layers=lambda: cache
        )
    )
    GDNAttnBackend.replay_spec_commit(
        backend, cache_indices, lcs, track_indices, track_steps
    )

    for l in range(L):
        assert torch.equal(o_old[l], o_new[l])
    assert torch.equal(ssm_new, ssm_old)

    for n in range(N):
        if cache_indices[n] < 0:
            continue
        rows = slice(n * D, (n + 1) * D)
        states, outs = ref_verify(
            ssm0[0, cache_indices[n]],
            q[0, 0, rows],
            k[0, 0, rows],
            v[0, 0, rows],
            a[0, rows],
            b[0, rows],
            A_log[0],
            dt_bias[0],
        )
        torch.testing.assert_close(inter[0, n], states, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(
            o_old[0][0, rows].float(), outs, atol=2e-2, rtol=2e-2
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
