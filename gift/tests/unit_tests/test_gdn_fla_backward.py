import sys
from contextlib import nullcontext
from functools import partial, wraps
from inspect import unwrap
from types import ModuleType

import pytest
import torch

from torchtitan.models.common.attention.linear.fla_custom_ops import gdn_chunk_op
from torchtitan.models.common.attention.linear.gdn_fla_backward import gdn_fla_backward


def _stub_module(monkeypatch, name, **exports):
    module = ModuleType(name)
    vars(module).update(exports)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("direct", (None, False, True))
def test_gdn_chunk_uses_same_backend_for_forward_and_fallback(monkeypatch, direct):
    calls = []

    class NativeScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *leaves):
            return sum(leaves)

        @staticmethod
        def backward(ctx, grad):
            return tuple(grad.clone() for _ in range(7))

    def native(**kwargs):
        calls.append("native")
        leaves = (
            kwargs[name] for name in ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
        )
        return NativeScan.apply(*leaves), None

    @wraps(native)
    def dispatched(**kwargs):
        calls.append("dispatch")
        return native(**kwargs)

    _stub_module(
        monkeypatch, "fla.ops.gated_delta_rule", chunk_gated_delta_rule=dispatched
    )
    leaves = tuple(torch.ones(2, requires_grad=True) for _ in range(7))
    flag = () if direct is None else (direct,)

    output = gdn_chunk_op(*leaves, None, *flag)
    grads = torch.autograd.grad(output.sum(), leaves)

    assert torch.equal(output, torch.full_like(output, 7))
    assert all(torch.equal(g, torch.ones_like(g)) for g in grads)
    assert calls == (["native"] if direct else ["dispatch", "native"]) * 2


@pytest.mark.parametrize("packed", (False, True))
def test_direct_backward_rebuilds_only_native_backward_inputs(monkeypatch, packed):
    calls = []
    q, k, v = (torch.ones(2, 4, 2, 128, dtype=torch.bfloat16) for _ in range(3))
    raw_gate, beta = (torch.ones(2, 4, 2, dtype=torch.bfloat16) for _ in range(2))
    A_log, dt_bias = torch.ones(2), torch.ones(2, dtype=torch.bfloat16)
    grad_output = v.clone()
    cumulative_gate = raw_gate.float().cumsum(dim=1)
    cu_seqlens = torch.tensor([0, 1, 4], dtype=torch.int64) if packed else None
    chunk_indices = torch.tensor([[0, 0], [1, 0]]) if packed else None
    scale = 1.4426950408889634

    def prepare(offsets, chunk_size, *, cu_seqlens_cpu):
        calls.append("indices")
        assert offsets is cu_seqlens and chunk_size == 64
        assert cu_seqlens_cpu is None
        return chunk_indices

    def gate(**kwargs):
        calls.append("gate")
        assert kwargs["g"] is raw_gate
        assert kwargs["scale"] == scale
        assert kwargs["chunk_size"] == 64
        assert kwargs["cu_seqlens"] is cu_seqlens
        assert kwargs["chunk_indices"] is chunk_indices
        return cumulative_gate

    class Solver:
        def __getitem__(self, grid):
            assert grid == ((2 if packed else 1), 4)
            return self.run

        def run(self, **kwargs):
            calls.append("solve")
            assert kwargs["g"] is cumulative_gate
            assert kwargs["BT"] == 64 and kwargs["BC"] == 16
            assert kwargs["cu_seqlens"] is cu_seqlens
            assert kwargs["chunk_indices"] is chunk_indices
            assert kwargs["A"].shape == (2, 4, 2, 64)
            kwargs["A"].fill_(3)

    def backward(**kwargs):
        calls.append("backward")
        assert kwargs["g"] is cumulative_gate
        assert kwargs["g_input"] is raw_gate
        assert torch.equal(kwargs["A"], torch.full_like(kwargs["A"], 3))
        assert kwargs["do"] is grad_output
        assert kwargs["A_log"] is A_log and kwargs["dt_bias"] is dt_bias
        assert kwargs["chunk_size"] == 64
        assert kwargs["scale"] == 128**-0.5
        assert kwargs["initial_state"] is None and kwargs["dht"] is None
        assert kwargs["use_gate_in_kernel"] is True
        assert kwargs["cu_seqlens"] is cu_seqlens
        assert kwargs["chunk_indices"] is chunk_indices
        return (
            q * 1,
            k * 2,
            v * 3,
            beta * 5,
            raw_gate.float() * 4,
            None,
            A_log * 6,
            dt_bias * 7,
        )

    _stub_module(
        monkeypatch, "fla.ops.gated_delta_rule.gate", gdn_gate_chunk_cumsum=gate
    )
    _stub_module(
        monkeypatch,
        "fla.ops.gated_delta_rule.chunk_fwd",
        chunk_gated_delta_rule_fwd_kkt_solve_kernel=Solver(),
    )
    _stub_module(
        monkeypatch,
        "fla.ops.gated_delta_rule.chunk",
        chunk_gated_delta_rule_bwd=backward,
    )
    _stub_module(monkeypatch, "fla.ops.utils.constant", RCP_LN2=scale)
    _stub_module(monkeypatch, "fla.ops.utils.index", prepare_chunk_indices=prepare)
    _stub_module(monkeypatch, "triton", cdiv=lambda x, y: (x + y - 1) // y)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    leaves = (q, k, v, raw_gate, beta, A_log, dt_bias)
    grads = gdn_fla_backward(*leaves, grad_output, cu_seqlens)
    assert calls == (["indices"] if packed else []) + ["gate", "solve", "backward"]
    for i, (leaf, grad) in enumerate(zip(leaves, grads, strict=True), start=1):
        assert grad.dtype == leaf.dtype
        assert torch.equal(grad, torch.full_like(leaf, i))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native FLA on CUDA")
@pytest.mark.parametrize(
    ("compiled", "packed", "dtype"),
    (
        (False, False, torch.bfloat16),
        (True, False, torch.bfloat16),
        (True, True, torch.bfloat16),
        (False, False, torch.float16),
        (True, True, torch.float16),
        (False, False, torch.float32),
        (True, True, torch.float32),
    ),
)
def test_native_fla_direct_backward_matches_replay(
    compiled, packed, dtype, monkeypatch
):
    from fla.ops import gated_delta_rule
    from torch.utils.checkpoint import checkpoint

    from torchtitan.distributed.activation_checkpoint import _sac_context_fn

    monkeypatch.setattr(
        gated_delta_rule,
        "chunk_gated_delta_rule",
        unwrap(gated_delta_rule.chunk_gated_delta_rule),
    )
    rng = torch.Generator(device="cuda").manual_seed(2048)

    def sample(shape, sample_dtype=dtype):
        return torch.randn(shape, device="cuda", dtype=sample_dtype, generator=rng)

    if packed:
        b, s, hk, hv, dk, dv = 1, 197, 3, 6, 64, 96
        cu_seqlens = torch.tensor([0, 4, 71, 97, s], device="cuda", dtype=torch.int32)
    elif dtype == torch.bfloat16:
        b, s, hk, hv, dk, dv = 2, 4096, 16, 32, 128, 128
        cu_seqlens = None
    else:
        b, s, hk, hv, dk, dv = 2, 91, 3, 6, 64, 96
        cu_seqlens = None

    inputs = (
        torch.nn.functional.normalize(sample((b, s, hk, dk)), dim=-1),
        torch.nn.functional.normalize(sample((b, s, hk, dk)), dim=-1),
        sample((b, s, hv, dv)),
        sample((b, s, hv)),
        sample((b, s, hv)).sigmoid(),
        sample((hv,), torch.float32),
        sample((hv,), torch.float32),
    )
    grad_output = sample(inputs[2].shape)

    def run(direct):
        def scan(*leaves):
            return gdn_chunk_op(*leaves, cu_seqlens, direct)

        def checkpointed(*leaves):
            return checkpoint(
                scan,
                *leaves,
                use_reentrant=False,
                context_fn=partial(_sac_context_fn, ("experts_up",), ()),
            )

        fn = torch.compile(checkpointed, fullgraph=True) if compiled else scan
        leaves = tuple(t.detach().requires_grad_() for t in inputs)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            output = fn(*leaves)
            grads = torch.autograd.grad(output, leaves, grad_output)
        assert (
            any(
                event.key == "torchtitan::gdn_fla_backward"
                for event in profile.key_averages()
            )
            == direct
        )
        return (output, *grads)

    for expected, actual in zip(run(False), run(True), strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
