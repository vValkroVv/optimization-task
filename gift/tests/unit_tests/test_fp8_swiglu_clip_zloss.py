"""SwiGLU clipping, its clamp counts and the gate/up z-loss on the fp8 paths.

Run:
    pytest tests/unit_tests/test_fp8_swiglu_clip_zloss.py

Skipped without an SM90 GPU and the fp8 kernels, like ``test_fp8_dense.py``.

Three things, each against the reference that isolates it:

- *The clamp* is checked against a clipped bf16 SwiGLU on the module's own
  projection outputs, relative to the norm (fp8 is lossy).
- *The counts* are checked exactly: the fused SwiGLU+quant kernel and the bf16
  SwiGLU both count on the same bf16 GEMM outputs, so they must agree with the
  rule spelled out here, and with each other, bit for bit.
- *The z-loss* is grad-only: the forward is bitwise unchanged, the projections
  it does not touch get bitwise the same gradient, and the one it does gets the
  z-loss gradient on top.
"""

import os

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("DG_CACHE_FORCE_SKIP", "1")

pytest.importorskip("cudacherry")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the fp8 kernels are SM90-only",
)

from torchtitan.models.common.ffn import FeedForward  # noqa: E402
from torchtitan.models.common.regularization import (  # noqa: E402
    hidden_z_loss,
    HiddenZLoss,
)
from torchtitan.ops.float8.cuda_gemm import FP8Linear  # noqa: E402

DIM, HIDDEN, LIMIT = 2048, 1024, 0.5
DEV = "cuda"


def _rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm()).item()


def _expected_counts(gate: torch.Tensor, up: torch.Tensor, limit: float) -> list[int]:
    return [
        int(gate.ge(limit).sum()),
        int((up.le(-limit) | up.ge(limit)).sum()),
        gate.numel(),
    ]


def _clipped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float):
    return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


# --- dense ---


def _feed_forward(
    *, fused: bool, limit: float = 0.0, w1_z_loss: float = 0.0
) -> FeedForward:
    torch.manual_seed(0)
    z_loss = HiddenZLoss(coef=w1_z_loss) if w1_z_loss else None
    config = FeedForward.Config(
        dim=DIM,
        hidden_dim=HIDDEN,
        w1=FP8Linear.Config(in_features=DIM, out_features=HIDDEN, hidden_z_loss=z_loss),
        w2=FP8Linear.Config(in_features=HIDDEN, out_features=DIM),
        w3=FP8Linear.Config(in_features=DIM, out_features=HIDDEN),
        swiglu_limit=limit,
    )
    config.dense_fused_swiglu_quant = fused
    ffn = FeedForward(config)
    assert ffn.fp8_enabled
    return ffn.to(device=DEV, dtype=torch.bfloat16)


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "bf16_swiglu"])
def test_an_fp8_feed_forward_clips_and_counts_exactly(fused):
    ffn = _feed_forward(fused=fused, limit=LIMIT)
    ffn.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64, device=DEV)
    x = torch.randn(2, 128, DIM, dtype=torch.bfloat16, device=DEV)

    with torch.no_grad():
        out = ffn(x)
        # The same fp8 GEMMs the module runs, so the values match bit for bit.
        gate, up = ffn.w1(x), ffn.w3(x)

    expected_counts = _expected_counts(gate, up, LIMIT)
    # The limit bites, but not everywhere.
    assert 0 < expected_counts[0] < expected_counts[2]
    assert 0 < expected_counts[1] < expected_counts[2]
    assert ffn.swiglu_clip_counts.tolist() == expected_counts
    expected = (
        _clipped_swiglu(gate.float(), up.float(), LIMIT) @ ffn.w2.weight.float().t()
    )
    assert _rel_error(out, expected) < 0.066


def test_the_fused_kernel_counts_only_the_real_rows():
    """The fused kernel scans the 128-padded GEMM output: the padding rows must
    not be counted, and the scanned count covers only the real rows."""
    ffn = _feed_forward(fused=True, limit=LIMIT)
    ffn.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64, device=DEV)
    # 100 rows: the GEMM output is padded to 128.
    x = torch.randn(1, 100, DIM, dtype=torch.bfloat16, device=DEV)

    with torch.no_grad():
        ffn(x)
        gate, up = ffn.w1(x), ffn.w3(x)

    assert ffn.swiglu_clip_counts.tolist() == _expected_counts(gate, up, LIMIT)


def test_no_slot_means_no_counting_and_the_same_output():
    """An unmonitored step clamps exactly as a monitored one."""
    x = torch.randn(2, 128, DIM, dtype=torch.bfloat16, device=DEV)
    ffn = _feed_forward(fused=True, limit=LIMIT)
    with torch.no_grad():
        unmonitored = ffn(x)
        ffn.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64, device=DEV)
        monitored = ffn(x)
    assert torch.equal(unmonitored, monitored)


def test_a_gate_z_loss_is_refused_on_the_fused_swiglu():
    with pytest.raises(ValueError, match="dense_fused_swiglu_quant=False"):
        _feed_forward(fused=True, w1_z_loss=1e-2)


def _forward_backward(ffn: FeedForward, x: torch.Tensor, cotangent: torch.Tensor):
    ffn.train()
    for p in ffn.parameters():
        p.grad = None
    out = ffn(x)
    (out.float() * cotangent).sum().backward()
    return out.detach(), {name: p.grad.clone() for name, p in ffn.named_parameters()}


def test_an_fp8_gate_z_loss_is_grad_only():
    coef = 1.0
    x = torch.randn(2, 128, DIM, dtype=torch.bfloat16, device=DEV)
    # Small, so the z-loss gradient dominates the w1 gradient it lands on and
    # the fp8 quantization of the sum does not hide it.
    cotangent = 1e-5 * torch.randn(2, 128, DIM, device=DEV)

    out_off, grads_off = _forward_backward(_feed_forward(fused=False), x, cotangent)
    ffn = _feed_forward(fused=False, w1_z_loss=coef)
    out_on, grads_on = _forward_backward(ffn, x, cotangent)

    assert torch.equal(out_on, out_off)
    # The z-loss sits on w1's output: w2 and w3 see the same gradient.
    assert torch.equal(grads_on["w2.weight"], grads_off["w2.weight"])
    assert torch.equal(grads_on["w3.weight"], grads_off["w3.weight"])

    # ... and w1 gets d(z-loss)/d(gate) on top, through the fp8 wgrad.
    with torch.no_grad():
        gate = ffn.w1(x)
    gate = gate.detach().requires_grad_(True)
    hidden_z_loss(gate, coef=coef).backward()
    expected_delta = gate.grad.float().reshape(-1, HIDDEN).t() @ x.float().reshape(
        -1, DIM
    )
    delta = grads_on["w1.weight"].float() - grads_off["w1.weight"].float()
    # fp8 wgrad against fp32: the tolerance of test_fp8_dense's projections.
    assert _rel_error(delta, expected_delta) < 0.05


# --- sparse ---


def _fp8_experts(*, fused: bool, limit: float = 0.0, z_loss: float = 0.0):
    pytest.importorskip("deep_gemm")
    pytest.importorskip("deep_ep")
    from torchtitan.models.common.moe import GroupedExperts
    from torchtitan.models.common.moe.dispatch import DeepEPTokenDispatcher

    config = GroupedExperts.Config(
        dim=EXPERT_DIM,
        hidden_dim=EXPERT_HIDDEN,
        num_experts=NUM_EXPERTS,
        token_dispatcher=DeepEPTokenDispatcher.Config(),
        swiglu_limit=limit,
        w1_hidden_z_loss_coef=z_loss,
    )
    config.use_float8_grouped_gemm = True
    config.sparse_fused_swiglu_quant = fused
    experts = GroupedExperts(config)
    torch.manual_seed(0)
    for p in experts.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return experts.to(device=DEV, dtype=torch.bfloat16)


NUM_EXPERTS, EXPERT_DIM, EXPERT_HIDDEN, GROUP = 2, 256, 256, 128
# Routed rows per 128-row group; the rest of each group is padding.
ROUTED = [100, 128]


def _expert_inputs():
    """A padded expert-major payload, as an fp8 dispatch hands it over."""
    from torchtitan.distributed.deepep.fp8_state import FP8DispatchState

    torch.manual_seed(1)
    rows = NUM_EXPERTS * GROUP
    x = torch.randn(rows, EXPERT_DIM, dtype=torch.bfloat16, device=DEV)
    for e, routed in enumerate(ROUTED):
        # Padding rows are zero, so their GEMM output is zero and never clamps.
        x[e * GROUP + routed : (e + 1) * GROUP] = 0
    sizes = [GROUP] * NUM_EXPERTS
    state = FP8DispatchState(
        permuted_probs=torch.rand(rows, dtype=torch.float32, device=DEV),
        padded_group_sizes_t=torch.tensor(sizes, dtype=torch.int32, device=DEV),
        padded_group_sizes_list=sizes,
        padded_group_sizes_cpu=torch.tensor(sizes, dtype=torch.int32),
    )
    num_tokens_E = torch.tensor(ROUTED, dtype=torch.int32, device=DEV)
    return x, num_tokens_E, state


def test_fp8_experts_count_the_same_with_and_without_the_fused_swiglu():
    """Both SwiGLUs count on the same bf16 fc1 output, over the routed rows."""
    x, num_tokens_E, state = _expert_inputs()
    counts = {}
    for fused in (True, False):
        experts = _fp8_experts(fused=fused, limit=LIMIT)
        experts.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64, device=DEV)
        with torch.no_grad():
            experts._fp8_experts_forward(x, num_tokens_E, state)
        counts[fused] = experts.swiglu_clip_counts.tolist()

    assert counts[True] == counts[False]
    gate_clamped, up_clamped, scanned = counts[True]
    assert scanned == sum(ROUTED) * EXPERT_HIDDEN
    assert 0 < gate_clamped < scanned and 0 < up_clamped < scanned


def test_an_fp8_expert_gate_z_loss_is_grad_only():
    from torchtitan.models.common.moe.experts.gmm import _routed_row_mask

    coef = 1.0
    x, num_tokens_E, state = _expert_inputs()
    # Small, so the z-loss gradient dominates the w13 gradient it lands on and
    # the fp8 quantization of the sum does not hide it.
    cotangent = 1e-6 * torch.randn(x.shape[0], EXPERT_DIM, device=DEV)

    def run(z_loss: float):
        experts = _fp8_experts(fused=False, z_loss=z_loss)
        experts.train()
        seen = {}
        apply_z_loss = experts._apply_hidden_z_loss

        def spy(h, **kwargs):
            seen["fc1"] = h.detach()
            return apply_z_loss(h, **kwargs)

        experts._apply_hidden_z_loss = spy
        out = experts._fp8_experts_forward(x, num_tokens_E, state)
        (out.float() * cotangent).sum().backward()
        return out.detach(), experts.w13_E2FD.grad, experts.w2_EDF.grad, seen["fc1"]

    out_off, w13_off, w2_off, _ = run(0.0)
    out_on, w13_on, w2_on, fc1 = run(coef)

    assert torch.equal(out_on, out_off)
    assert torch.equal(w2_on, w2_off)
    # Only the gate half of w13 carries the z-loss.
    assert torch.equal(w13_on[:, 1], w13_off[:, 1])

    # ... and it gets d(z-loss)/d(gate) on top, over the routed rows only,
    # through the fp8 wgrad.
    mask = _routed_row_mask(
        x.shape[0], state.padded_group_sizes_t, num_tokens_E
    ).float()
    gate = fc1[:, :EXPERT_HIDDEN].float().requires_grad_(True)
    lse = torch.logsumexp(gate.abs(), dim=-1)
    (coef * (lse.square() * mask).sum() / mask.sum()).backward()
    expected_delta = torch.stack(
        [
            gate.grad[e * GROUP : (e + 1) * GROUP].t()
            @ x[e * GROUP : (e + 1) * GROUP].float()
            for e in range(NUM_EXPERTS)
        ]
    )
    delta = w13_on[:, 0].float() - w13_off[:, 0].float()
    assert _rel_error(delta, expected_delta) < 0.05
