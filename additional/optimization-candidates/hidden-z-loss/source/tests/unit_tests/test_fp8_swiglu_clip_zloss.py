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

The fused fp8 SwiGLU kernels of cudacherry (``swiglu_quantize_zloss`` dense,
``fused_swiglu_zloss`` sparse) also apply the z-loss. Two more references
isolate them:

- *Without a z-loss* they have the bits of the kernels with no z-loss: the
  CUDA ``swiglu_quantize`` and the torchtitan Triton ``fused_swiglu_quantized_fn``.
- *With a z-loss* they obey the cudacherry contract against the bf16 path with
  its z-loss splices: the gradient of the SwiGLU input at 1.05 times the fp8
  floor or less, the totals in 1e-4, and equal clip counts. The fp8 floor is
  the error of the ideal 1 x 128 fp8 rounding of the bf16 gradient.
"""

import contextlib
import dataclasses
import os
from functools import partial

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("DG_CACHE_FORCE_SKIP", "1")

pytest.importorskip("cudacherry")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the fp8 kernels are SM90-only",
)

from torchtitan.models.common.ffn import (  # noqa: E402
    _clipped_swiglu as ffn_swiglu,
    FeedForward,
)
from torchtitan.models.common.regularization import (  # noqa: E402
    attach_hidden_z_loss,
    hidden_z_loss,
    HiddenZLoss,
    suppress_aux_recording,
    wire_aux_loss,
)
from torchtitan.ops.float8 import _cudacherry  # noqa: E402
from torchtitan.ops.float8.cuda_gemm import FP8Linear  # noqa: E402

DIM, HIDDEN, LIMIT = 2048, 1024, 0.5
DEV = "cuda"
FP8_MAX, BLOCK = 448.0, 128
# The cudacherry contract: the gradient error against the bf16 path.
FLOOR_FACTOR, TOTALS_TOLERANCE = 1.05, 1e-4
# A power of two, so that the weighted totals of the two paths stay comparable.
SPLICE_SCALE = 0.5


def _rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm()).item()


def _ideal_1x128(x: torch.Tensor) -> torch.Tensor:
    """Give the fp32 value of the ideal 1 x 128 fp8 rounding of a 2-D tensor.

    One power-of-two scale for each block of 128 columns, thus the only rounding
    is the e4m3 round to nearest. O(numel).
    """
    rows, cols = x.shape
    blocks = x.float().reshape(rows, cols // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2((amax / FP8_MAX).clamp_min(1e-30))))
    rounded = (blocks / scale).to(torch.float8_e4m3fn).float() * scale
    return rounded.reshape(rows, cols)


def _dequantized(packed: torch.Tensor) -> torch.Tensor:
    """Give the fp32 value of a packed 1 x 128 operand as a 2-D tensor.

    The scales are (rows, blocks), or (blocks, rows) and padded when they are
    block-first. O(numel).
    """
    cols = packed._data.shape[-1]
    rows = packed._data.numel() // cols
    blocks = packed._data.reshape(rows, cols // BLOCK, BLOCK).float()
    scale = packed._scale
    if packed._scale_block_first:
        scale = scale[: cols // BLOCK, :rows].t()
    return (blocks * scale[..., None]).reshape(rows, cols)


def _assert_at_the_floor(actual: torch.Tensor, bf16_grad: torch.Tensor) -> None:
    """The dequantized fp8 gradient is at FLOOR_FACTOR times the fp8 floor or less.

    ``actual`` and ``bf16_grad`` are 2-D, with the same rows.
    """
    expected = bf16_grad.float()
    assert expected.norm() > 0
    floor = _rel_error(_ideal_1x128(expected), expected)
    error = _rel_error(actual, expected)
    assert error <= FLOOR_FACTOR * floor, f"error {error:.4e}, floor {floor:.4e}"


def _hook_modules(coefs: tuple[float, float]) -> torch.nn.ModuleDict:
    """Two identities with the ``attach_hidden_z_loss`` hooks of w1 and w3, wired.

    They are the splices of the bf16 path: the module FQNs give the totals the
    names ``hidden_z_loss/w1`` and ``hidden_z_loss/w3``, as in the FFN.
    """
    hooks = torch.nn.ModuleDict({"w1": torch.nn.Identity(), "w3": torch.nn.Identity()})
    for name, coef in zip(("w1", "w3"), coefs):
        attach_hidden_z_loss(HiddenZLoss(coef=coef), hooks[name])
    return hooks


def _totals(state) -> dict[str, float]:
    """Give the aux-loss totals of a wired model part by name."""
    return dict(zip(state.names, state.totals.tolist()))


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
    *, fused: bool, limit: float = 0.0, w1_z_loss: float = 0.0, w3_z_loss: float = 0.0
) -> FeedForward:
    torch.manual_seed(0)
    w1_spec, w3_spec = (
        HiddenZLoss(coef=coef) if coef else None for coef in (w1_z_loss, w3_z_loss)
    )
    config = FeedForward.Config(
        dim=DIM,
        hidden_dim=HIDDEN,
        w1=FP8Linear.Config(
            in_features=DIM, out_features=HIDDEN, hidden_z_loss=w1_spec
        ),
        w2=FP8Linear.Config(in_features=HIDDEN, out_features=DIM),
        w3=FP8Linear.Config(
            in_features=DIM, out_features=HIDDEN, hidden_z_loss=w3_spec
        ),
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


def _forward_backward(
    ffn: FeedForward, x: torch.Tensor, cotangent: torch.Tensor, *, train: bool = True
):
    """Give the output and the gradients (each parameter, and ``x``)."""
    ffn.train(train)
    for p in ffn.parameters():
        p.grad = None
    x = x.detach().clone().requires_grad_(True)
    out = ffn(x)
    (out.float() * cotangent).sum().backward()
    grads = {name: p.grad.clone() for name, p in ffn.named_parameters()}
    grads["x"] = x.grad
    return out.detach(), grads


def _route_to_swiglu_quantize(monkeypatch) -> None:
    """Make ``FeedForward`` call the CUDA ``swiglu_quantize``.

    The CUDA kernel has no z-loss, thus the z-loss arguments must be off.
    """

    def swiglu_quantize(*args, coefs, zloss_state, **kwargs):
        assert coefs == (0.0, 0.0)
        return _cudacherry.swiglu_quantize(*args, **kwargs)

    monkeypatch.setattr(_cudacherry, "swiglu_quantize_zloss", swiglu_quantize)


@pytest.mark.parametrize("limit", [0.0, LIMIT], ids=["no_clip", "clip"])
@pytest.mark.parametrize("rows", [128, 100], ids=["aligned", "padded"])
def test_the_fused_swiglu_without_a_z_loss_has_the_bits_of_swiglu_quantize(
    monkeypatch, limit, rows
):
    """The test compares the output, the clip counts and each gradient. 100 rows
    make the GEMMs pad their outputs."""
    x = torch.randn(2, rows, DIM, dtype=torch.bfloat16, device=DEV)
    cotangent = torch.randn(2, rows, DIM, device=DEV)
    runs = []
    for reference in (False, True):
        if reference:
            _route_to_swiglu_quantize(monkeypatch)
        ffn = _feed_forward(fused=True, limit=limit)
        if limit:
            ffn.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64, device=DEV)
        out, grads = _forward_backward(ffn, x, cotangent)
        counts = ffn.swiglu_clip_counts
        runs.append((out, grads, None if counts is None else counts.tolist()))

    (out, grads, counts), (expected_out, expected_grads, expected_counts) = runs
    assert torch.equal(out, expected_out)
    assert grads.keys() == expected_grads.keys()
    for name, grad in grads.items():
        assert torch.equal(grad, expected_grads[name]), name
    assert counts == expected_counts


def test_a_gate_z_loss_on_the_fused_swiglu_is_grad_only():
    """The fused fp8 SwiGLU applies the w1 output z-loss itself, thus w1 has no
    hook. The forward and w2 do not change, w1 gets the z-loss gradient, and
    the declared splice gets the total. In eval the z-loss is off."""
    coef = 1.0
    x = torch.randn(2, 128, DIM, dtype=torch.bfloat16, device=DEV)
    # Small, so the z-loss gradient dominates the w1 gradient it lands on and
    # the fp8 quantization of the sum does not hide it.
    cotangent = 1e-5 * torch.randn(2, 128, DIM, device=DEV)

    out_off, grads_off = _forward_backward(_feed_forward(fused=True), x, cotangent)
    ffn = _feed_forward(fused=True, w1_z_loss=coef)
    assert not ffn.w1._forward_hooks
    assert ffn._hidden_z_loss_coefs == (coef, 0.0)
    state = wire_aux_loss(ffn, device=torch.device(DEV))
    out_on, grads_on = _forward_backward(ffn, x, cotangent)

    assert torch.equal(out_on, out_off)
    assert torch.equal(grads_on["w2.weight"], grads_off["w2.weight"])
    # With a z-loss half on, the backward reads the bf16 inputs, not their fp8
    # copies: the up gradient changes by the fp8 rounding of the inputs only.
    assert _rel_error(grads_on["w3.weight"], grads_off["w3.weight"]) < 0.066

    with torch.no_grad():
        gate = ffn.w1(x)
    gate = gate.detach().requires_grad_(True)
    loss = hidden_z_loss(gate, coef=coef)
    loss.backward()
    expected_delta = gate.grad.float().reshape(-1, HIDDEN).t() @ x.float().reshape(
        -1, DIM
    )
    delta = grads_on["w1.weight"].float() - grads_off["w1.weight"].float()
    # fp8 wgrad against fp32: the tolerance of test_fp8_dense's projections.
    assert _rel_error(delta, expected_delta) < 0.05
    expected_totals = {"hidden_z_loss/w1": loss.item()}
    assert _totals(state) == pytest.approx(expected_totals, rel=TOTALS_TOLERANCE)

    state.reset()
    out_eval, grads_eval = _forward_backward(ffn, x, cotangent, train=False)
    assert torch.equal(out_eval, out_off)
    for name, grad in grads_eval.items():
        assert torch.equal(grad, grads_off[name]), name
    assert _totals(state) == {"hidden_z_loss/w1": 0.0}


def test_a_suppressed_forward_records_no_fused_z_loss():
    """The pipeline probe suppresses the recording around its forward: the
    kernel still applies the z-loss gradient, but the totals stay zero."""
    x = torch.randn(2, 128, DIM, dtype=torch.bfloat16, device=DEV)
    cotangent = 1e-5 * torch.randn(2, 128, DIM, device=DEV)
    ffn = _feed_forward(fused=True, w1_z_loss=1.0, w3_z_loss=0.5)
    state = wire_aux_loss(ffn, device=torch.device(DEV))
    _, expected_grads = _forward_backward(ffn, x, cotangent)
    assert all(total > 0 for total in _totals(state).values())

    state.reset()
    for p in ffn.parameters():
        p.grad = None
    with suppress_aux_recording():
        out = ffn(x)
    (out.float() * cotangent).sum().backward()
    assert state.totals.eq(0).all()
    for name, p in ffn.named_parameters():
        assert torch.equal(p.grad, expected_grads[name]), name


def _gemm_outputs(rows: int, padded_rows: int, scale: float = 1.0):
    """Give bf16 (1, padded_rows, HIDDEN) tensors with zero padding rows, as the
    fp8 GEMMs give them: gate, up and the gradient of the SwiGLU."""
    torch.manual_seed(2)
    tensors = []
    for std in (3.0, 3.0, scale):
        tensor = std * torch.randn(1, padded_rows, HIDDEN, device=DEV)
        tensor[:, rows:] = 0
        tensors.append(tensor.bfloat16())
    return tuple(tensors)


@pytest.mark.parametrize("grad_scale", [1.0, 0.0], ids=["mixed", "z_loss_only"])
@pytest.mark.parametrize(
    "coefs", [(1e-2, 0.0), (0.0, 5e-3), (1e-2, 5e-3)], ids=["w1", "w3", "both"]
)
def test_the_fused_swiglu_z_loss_meets_the_contract_on_the_bf16_hooks(
    coefs, grad_scale
):
    """``FeedForward._fused_swiglu`` against ``_clipped_swiglu`` with the hooks
    of w1 and w3, on the same GEMM outputs. With a zero output gradient, the
    gradient is the z-loss term alone."""
    rows, padded_rows = 100, 128
    gate, up, grad_y = _gemm_outputs(rows, padded_rows, grad_scale)

    ffn = _feed_forward(
        fused=True, limit=LIMIT, w1_z_loss=coefs[0], w3_z_loss=coefs[1]
    ).train()
    ffn.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64, device=DEV)
    state = wire_aux_loss(ffn, device=torch.device(DEV))
    state.scale.fill_(SPLICE_SCALE)
    fused_gate, fused_up = (t.clone().requires_grad_(True) for t in (gate, up))
    out = ffn._fused_swiglu(fused_gate, fused_up, (1, rows, DIM))
    out.backward(grad_y)

    hooks = _hook_modules(coefs)
    expected_state = wire_aux_loss(hooks, device=torch.device(DEV))
    expected_state.scale.fill_(SPLICE_SCALE)
    expected_counts = torch.zeros(3, dtype=torch.int64, device=DEV)
    ref_gate, ref_up = (t[:, :rows].clone().requires_grad_(True) for t in (gate, up))
    y = ffn_swiglu(hooks["w1"](ref_gate), hooks["w3"](ref_up), LIMIT, expected_counts)
    y.backward(grad_y[:, :rows])
    assert ffn.swiglu_clip_counts.tolist() == expected_counts.tolist()

    with torch.no_grad():
        ffn.eval()
        out_off = ffn._fused_swiglu(gate, up, (1, rows, DIM))
    assert torch.equal(out._data.view(torch.uint8), out_off._data.view(torch.uint8))
    assert torch.equal(out._scale, out_off._scale)
    for packed, expected in (
        (fused_gate.grad, ref_gate.grad),
        (fused_up.grad, ref_up.grad),
    ):
        actual = _dequantized(packed)[:rows]
        expected = expected.reshape(rows, HIDDEN)
        if expected.count_nonzero() == 0:
            assert actual.count_nonzero() == 0
        else:
            _assert_at_the_floor(actual, expected)
    totals, expected_totals = _totals(state), _totals(expected_state)
    assert totals == pytest.approx(expected_totals, rel=TOTALS_TOLERANCE)
    assert all(total > 0 for total in totals.values())


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


def test_an_fp8_expert_gate_z_loss_on_the_fused_swiglu_is_grad_only():
    """The fused SwiGLU of the experts applies the gate z-loss itself: the
    forward and w2 do not change, and the gate half of w13 gets the z-loss
    gradient on top, over the routed rows only."""
    coef = 1.0
    x, num_tokens_E, state = _expert_inputs()
    cotangent = 1e-6 * torch.randn(x.shape[0], EXPERT_DIM, device=DEV)

    def run(fused: bool, z_loss: float):
        experts = _fp8_experts(fused=fused, z_loss=z_loss)
        experts.train()
        out = experts._fp8_experts_forward(x, num_tokens_E, state)
        (out.float() * cotangent).sum().backward()
        return out.detach(), experts.w13_E2FD.grad, experts.w2_EDF.grad

    out_off, w13_off, w2_off = run(True, 0.0)
    out_on, w13_on, w2_on = run(True, coef)
    _, w13_bf16, _ = run(False, coef)

    assert torch.equal(out_on, out_off)
    assert torch.equal(w2_on, w2_off)
    assert torch.equal(w13_on[:, 1], w13_off[:, 1])
    # The bf16 SwiGLU applies the same z-loss: the same delta, through fp8 wgrad.
    delta = w13_on[:, 0].float() - w13_off[:, 0].float()
    expected_delta = w13_bf16[:, 0].float() - w13_off[:, 0].float()
    assert _rel_error(delta, expected_delta) < 0.05


# --- sparse: the fused SwiGLU step between the fp8 grouped GEMMs ---

# F is a tuned size of the cudacherry tables. The routed rows of each 128-row
# group: a padded group, an empty group, a full group and a short group.
SWIGLU_HIDDEN, SWIGLU_ROUTED, SWIGLU_LIMIT = 1024, (100, 0, 128, 37), 7.0


def _swiglu_experts(
    *, fused: bool, limit: float = 0.0, coefs: tuple[float, float] = (0.0, 0.0)
):
    """GroupedExperts for its ``_fp8_swiglu`` step alone, in training.

    The fp8 grouped GEMMs around that step need DeepGEMM and an EP dispatcher,
    and the step needs neither. Thus these experts keep the bf16 GEMMs, and only
    the flag of the fused SwiGLU is set.
    """
    from torchtitan.models.common.moe import GroupedExperts

    config = GroupedExperts.Config(
        dim=EXPERT_DIM,
        hidden_dim=SWIGLU_HIDDEN,
        num_experts=len(SWIGLU_ROUTED),
        swiglu_limit=limit,
        w1_hidden_z_loss_coef=coefs[0],
        w3_hidden_z_loss_coef=coefs[1],
    )
    config.sparse_fused_swiglu_quant = fused
    experts = GroupedExperts(config).to(DEV)
    if fused:
        # The fp8 experts declare this site at their build. These experts keep
        # the bf16 GEMMs, thus the test declares it.
        experts._declare_fused_swiglu_site()
    if limit:
        experts.swiglu_clip_counts = torch.zeros(3, dtype=torch.int64, device=DEV)
    return experts.train()


def _fc1_case(
    grad_scale: float = 1.0,
    *,
    routed_E: tuple[int, ...] = SWIGLU_ROUTED,
    group_rows: int = GROUP,
    tail_rows: int = 0,
):
    """Give (fc1, grad_y, the dispatch state, num_tokens) of a padded layout.

    fc1 is the bf16 output of the first grouped GEMM, and grad_y the gradient of
    the SwiGLU output. Their padding rows are zero, as the GEMMs give them. The
    tail rows are past the last group, as in a dispatch buffer of fixed capacity.
    They have values in fc1, but no expert owns them.
    """
    from torchtitan.distributed.deepep.fp8_state import FP8DispatchState

    torch.manual_seed(3)
    sizes = [group_rows] * len(routed_E)
    rows = sum(sizes) + tail_rows
    fc1 = 3.0 * torch.randn(rows, 2 * SWIGLU_HIDDEN, device=DEV)
    grad_y = grad_scale * torch.randn(rows, SWIGLU_HIDDEN, device=DEV)
    for e, routed in enumerate(routed_E):
        fc1[e * group_rows + routed : (e + 1) * group_rows] = 0
        grad_y[e * group_rows + routed : (e + 1) * group_rows] = 0
    grad_y[sum(sizes) :] = 0
    state = FP8DispatchState(
        permuted_probs=torch.rand(rows, dtype=torch.float32, device=DEV),
        padded_group_sizes_t=torch.tensor(sizes, dtype=torch.int32, device=DEV),
        padded_group_sizes_list=sizes,
        padded_group_sizes_cpu=torch.tensor(sizes, dtype=torch.int32),
    )
    num_tokens_E = torch.tensor(routed_E, dtype=torch.int32, device=DEV)
    return fc1.bfloat16(), grad_y.bfloat16(), state, num_tokens_E


def _run_fp8_swiglu(experts, case, *, use_edge: bool = False, suppress: bool = False):
    """Run ``_fp8_swiglu`` and its backward on leaf copies of the case.

    Give (the input of w2, the gradient of fc1, the gradient of the probs). With
    ``use_edge``, the gradient of fc1 comes back through an fp8 edge, as the
    first fp8 GEMM of a backend with gradient edges asks for it.
    """
    from torchtitan.ops.float8.fp8_tensor import fp8_grad_edge

    fc1, grad_y, state, num_tokens_E = case
    fc1 = fc1.clone().requires_grad_(True)
    probs = state.permuted_probs.clone().requires_grad_(True)
    state = dataclasses.replace(state, permuted_probs=probs)
    fc1_edge = None
    if use_edge:
        fc1_edge = fp8_grad_edge(*fc1.shape, device=fc1.device).requires_grad_(True)
    with suppress_aux_recording() if suppress else contextlib.nullcontext():
        edge, intermediate = experts._fp8_swiglu(fc1, fc1_edge, state, num_tokens_E)
    (intermediate if edge is None else edge).backward(grad_y)
    return intermediate, fc1.grad if fc1_edge is None else fc1_edge.grad, probs.grad


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Give the bytes of a tensor, for a bit comparison of fp8 values."""
    return tensor.contiguous().view(torch.uint8)


def _assert_same_packed_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """The fp8 data and the scales of two packed operands are bit for bit equal."""
    assert torch.equal(_bits(actual._data), _bits(expected._data))
    assert torch.equal(_bits(actual._scale), _bits(expected._scale))
    assert actual._scale_block_first == expected._scale_block_first


def _route_to_torchtitan_triton(monkeypatch) -> None:
    """Make ``GroupedExperts`` call the torchtitan Triton kernel.

    That kernel has no z-loss, thus the z-loss arguments must be off.
    """
    from torchtitan.ops.float8.triton_kernels.fused_swiglu_quantization import (
        fused_swiglu_quantized_fn,
    )

    def fused_swiglu(x, probs, limit, counts, *, coefs, out_fp8, fp8_edge, **_):
        assert coefs == (0.0, 0.0) and out_fp8
        return fused_swiglu_quantized_fn(fp8_edge, x, probs, limit, True, counts)

    monkeypatch.setattr(_cudacherry, "fused_swiglu_zloss", fused_swiglu)


@pytest.mark.parametrize("use_edge", [False, True], ids=["no_edge", "edge"])
@pytest.mark.parametrize("limit", [0.0, SWIGLU_LIMIT], ids=["no_clip", "clip"])
def test_the_fused_expert_swiglu_without_a_z_loss_has_the_torchtitan_bits(
    monkeypatch, limit, use_edge
):
    """The test compares the fp8 input of w2, the fp8 gradient of fc1 and the clip
    counts bit for bit."""
    case = _fc1_case()
    runs = []
    for reference in (False, True):
        if reference:
            _route_to_torchtitan_triton(monkeypatch)
        experts = _swiglu_experts(fused=True, limit=limit)
        payload, grad, grad_probs = _run_fp8_swiglu(experts, case, use_edge=use_edge)
        counts = experts.swiglu_clip_counts
        runs.append(
            (payload, grad, grad_probs, None if counts is None else counts.tolist())
        )

    (payload, grad, grad_probs, counts), expected = runs
    _assert_same_packed_bits(payload, expected[0])
    _assert_same_packed_bits(grad, expected[1])
    # grad_probs can be different in the last bits, because the two kernels add the
    # F terms of a row in a different sequence. The sum has cancellation, thus the
    # difference is near the fp32 precision of the terms (1e-4 or less), not of the sum.
    torch.testing.assert_close(grad_probs, expected[2], rtol=1e-4, atol=1e-4)
    assert counts == expected[3]


@pytest.mark.parametrize("grad_scale", [1.0, 0.0], ids=["mixed", "z_loss_only"])
@pytest.mark.parametrize(
    "coefs", [(1e-2, 0.0), (0.0, 5e-3), (1e-2, 5e-3)], ids=["w1", "w3", "both"]
)
def test_the_fused_expert_swiglu_z_loss_meets_the_contract_on_the_bf16_branch(
    coefs, grad_scale
):
    """The fused step against the bf16 step of the same experts (the z-loss
    splices of ``_apply_hidden_z_loss`` and ``fused_swiglu``). The forward is
    that of the z-loss off. With a zero output gradient, the gradient of fc1 is
    the z-loss term alone."""
    case = _fc1_case(grad_scale)
    results = {}
    for fused in (True, False):
        experts = _swiglu_experts(fused=fused, limit=SWIGLU_LIMIT, coefs=coefs)
        state = wire_aux_loss(experts, device=torch.device(DEV))
        state.scale.fill_(SPLICE_SCALE)
        payload, grad, grad_probs = _run_fp8_swiglu(experts, case)
        counts = experts.swiglu_clip_counts.tolist()
        results[fused] = (payload, grad, grad_probs, _totals(state), counts)
    payload_off, *_ = _run_fp8_swiglu(
        _swiglu_experts(fused=True, limit=SWIGLU_LIMIT), case
    )

    (payload, grad, grad_probs, totals, counts) = results[True]
    (_, bf16_grad, bf16_grad_probs, bf16_totals, bf16_counts) = results[False]
    _assert_same_packed_bits(payload, payload_off)
    _assert_at_the_floor(_dequantized(grad), bf16_grad)
    if grad_scale:
        # The bf16 branch rounds silu * up to bf16 before the product.
        assert _rel_error(grad_probs, bf16_grad_probs) < 1e-2
    else:
        assert grad_probs.count_nonzero() == bf16_grad_probs.count_nonzero() == 0
    assert totals == pytest.approx(bf16_totals, rel=TOTALS_TOLERANCE)
    assert all(total > 0 for total in totals.values())
    assert counts == bf16_counts


def test_the_fused_expert_z_loss_is_off_in_eval_and_unrecorded_in_the_probe():
    """Eval runs the z-loss off. Under ``suppress_aux_recording`` the kernel
    applies the z-loss gradient but records no total. An fp8 edge carries the
    same gradient as fc1."""
    case = _fc1_case(grad_scale=0.0)
    coefs = (1e-2, 5e-3)
    _, grad_off, _ = _run_fp8_swiglu(_swiglu_experts(fused=True), case)
    experts = _swiglu_experts(fused=True, coefs=coefs)
    state = wire_aux_loss(experts, device=torch.device(DEV))
    _, grad_on, _ = _run_fp8_swiglu(experts, case)
    assert all(total > 0 for total in _totals(state).values())

    state.reset()
    _, grad_edge, _ = _run_fp8_swiglu(experts, case, use_edge=True, suppress=True)
    _assert_same_packed_bits(grad_edge, grad_on)
    assert state.totals.eq(0).all()

    experts.eval()
    _, grad_eval, _ = _run_fp8_swiglu(experts, case)
    _assert_same_packed_bits(grad_eval, grad_off)
    assert state.totals.eq(0).all()


@pytest.mark.parametrize("partition", [True, False], ids=["partition", "plain_stream"])
def test_each_launch_of_the_sparse_kernels_gives_the_same_step(monkeypatch, partition):
    """torchtitan uses the presets, and each other launch must give the bits of the
    same step. Each launch has its own totals: one total stays on one stream."""
    case = _fc1_case()
    coefs = (1e-2, 5e-3)

    def run():
        experts = _swiglu_experts(fused=True, limit=SWIGLU_LIMIT, coefs=coefs)
        state = wire_aux_loss(experts, device=torch.device(DEV))
        result = _run_fp8_swiglu(experts, case, use_edge=True)
        return (*result, _totals(state), experts.swiglu_clip_counts.tolist())

    expected = run()
    monkeypatch.setattr(
        _cudacherry,
        "fused_swiglu_zloss",
        partial(_cudacherry.fused_swiglu_zloss, partition=partition),
    )
    payload, grad, grad_probs, totals, counts = run()

    _assert_same_packed_bits(payload, expected[0])
    _assert_same_packed_bits(grad, expected[1])
    assert torch.equal(grad_probs, expected[2])
    assert totals == expected[3]
    assert counts == expected[4]


@pytest.mark.parametrize(
    "layout",
    [
        dict(routed_E=(0, 0, 0, 0), group_rows=0),
        dict(routed_E=(0, 0, 0, 0)),
        dict(tail_rows=2 * GROUP),
    ],
    ids=["a_rank_with_no_rows", "groups_with_no_routed_rows", "tail_past_the_groups"],
)
def test_the_fused_expert_z_loss_takes_each_layout_of_a_dispatch(layout):
    """A rank can get no tokens, and a dispatch buffer can have rows that no expert
    owns. The fused step must agree with the bf16 step: the rows that no expert
    routes have no z-loss, and a rank with no routed rows has a total of zero."""
    case = _fc1_case(**layout)
    coefs = (1e-2, 5e-3)
    results = {}
    for fused in (True, False):
        experts = _swiglu_experts(fused=fused, limit=SWIGLU_LIMIT, coefs=coefs)
        state = wire_aux_loss(experts, device=torch.device(DEV))
        _, grad, _ = _run_fp8_swiglu(experts, case)
        results[fused] = (grad, _totals(state), experts.swiglu_clip_counts.tolist())

    (grad, totals, counts), (bf16_grad, bf16_totals, bf16_counts) = (
        results[True],
        results[False],
    )
    assert totals == pytest.approx(bf16_totals, rel=TOTALS_TOLERANCE, abs=1e-12)
    # The counts scan all rows of the layout, and the tail rows also clamp.
    assert counts == bf16_counts
    if bf16_grad.count_nonzero() == 0:
        assert _dequantized(grad).count_nonzero() == 0
    else:
        _assert_at_the_floor(_dequantized(grad), bf16_grad)


@pytest.mark.parametrize(
    "grad_mode", [torch.no_grad, torch.inference_mode], ids=["no_grad", "inference"]
)
def test_a_fused_expert_forward_without_a_graph_has_the_bits_of_the_z_loss_off(
    grad_mode,
):
    """Validation has no graph. A z-loss in the config must not change the w2 input."""
    fc1, _, state, num_tokens_E = _fc1_case()
    with torch.no_grad():
        _, expected = _swiglu_experts(fused=True)._fp8_swiglu(
            fc1, None, state, num_tokens_E
        )
    experts = _swiglu_experts(fused=True, coefs=(1e-2, 5e-3))
    aux_state = wire_aux_loss(experts, device=torch.device(DEV))
    for training in (True, False):
        experts.train(training)
        with grad_mode():
            _, payload = experts._fp8_swiglu(fc1, None, state, num_tokens_E)
        _assert_same_packed_bits(payload, expected)
    assert aux_state.totals.eq(0).all()
