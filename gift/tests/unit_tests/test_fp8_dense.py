# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The dense blockwise-fp8 path on the real cudacherry kernels.

Run:
    pytest tests/unit_tests/test_fp8_dense.py

Skipped without a suitable GPU, so the CPU CI job passes over it -- the same
convention as the other GPU-gated tests in this directory (``test_compile_moe``,
``test_flash_attention``, ``test_fsdp_moe_sharding``).

Two questions, and they need different references.

*Does it compute the right thing* is answered against bf16, relative to the
norm: blockwise fp8 is a lossy representation of these operands, and a
per-element tolerance would either hide a regression or flake.

*Does compiling it change anything* is answered against eager, bitwise. Every
kernel in this path is a custom op or an ``allow_in_graph`` autograd Function,
so a compiled block runs the same arithmetic in the same order -- a difference
here means Dynamo reordered or dropped something, which is exactly the failure
this path has had. Both flags ``apply_compile`` sets are set here, including the
one that decides what activation-checkpoint recompute replays.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)

pytest.importorskip("cudacherry")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the cudacherry fp8 kernels are SM90-only",
)

from torchtitan.models.common.ffn import FeedForward  # noqa: E402
from torchtitan.ops.float8.cuda_gemm import FP8Linear  # noqa: E402

DIM, HIDDEN = 512, 1024
DEV = "cuda"


@pytest.fixture(autouse=True)
def production_compile_flags(monkeypatch):
    """The Dynamo contract ``apply_compile`` establishes for the whole model."""
    monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", True)
    monkeypatch.setattr(
        torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint", True
    )
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class _Block(torch.nn.Module):
    """A parent for the FFN, like the transformer block ``apply_compile`` takes.

    The fp8 wrapper has to stay an intermediate of one graph: as a graph output
    AOTAutograd would assume its gradient is the same subclass, and the gradient
    of a quantized activation is a plain bf16 tensor.
    """

    def __init__(self, ffn: torch.nn.Module) -> None:
        super().__init__()
        self.ffn = ffn

    def forward(self, x):
        return self.ffn(x)


def _linear() -> FP8Linear:
    torch.manual_seed(0)
    linear = FP8Linear(
        FP8Linear.Config(in_features=DIM, out_features=HIDDEN, bias=False)
    )
    return linear.to(device=DEV, dtype=torch.bfloat16)


def _feed_forward(*, shared_quantizer: bool) -> FeedForward:
    torch.manual_seed(0)
    config = FeedForward.Config(
        dim=DIM,
        hidden_dim=HIDDEN,
        w1=FP8Linear.Config(in_features=DIM, out_features=HIDDEN),
        w2=FP8Linear.Config(in_features=HIDDEN, out_features=DIM),
        w3=FP8Linear.Config(in_features=DIM, out_features=HIDDEN),
    )
    config.use_shared_quantizer = shared_quantizer
    ffn = FeedForward(config)
    assert ffn.fp8_enabled
    return ffn.to(device=DEV, dtype=torch.bfloat16)


def _rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm()).item()


def _forward_backward(module, x, cotangent):
    x = x.detach().clone().requires_grad_(True)
    for parameter in module.parameters():
        parameter.grad = None
    out = module(x)
    (out.float() * cotangent).sum().backward()
    return (
        out.detach(),
        x.grad,
        [p.grad.clone() for p in module.parameters()],
    )


# --- does it compute the right thing ---


@pytest.mark.parametrize("rows", [128, 100], ids=["aligned", "padded"])
def test_an_fp8_projection_matches_bf16(rows):
    """100 rows exercise the padding path, where ``orig_shape`` has to survive
    the quantization and come back out of the GEMM."""
    linear = _linear()
    x = torch.randn(2, rows, DIM, dtype=torch.bfloat16, device=DEV)
    cotangent = torch.randn(2, rows, HIDDEN, device=DEV)

    out, x_grad, (w_grad,) = _forward_backward(linear, x, cotangent)

    expected = x.float() @ linear.weight.float().t()
    assert out.shape == (2, rows, HIDDEN)
    assert _rel_error(out, expected) < 0.039
    assert _rel_error(x_grad, cotangent @ linear.weight.float()) < 0.039
    expected_w = cotangent.reshape(-1, HIDDEN).t() @ x.float().reshape(-1, DIM)
    assert _rel_error(w_grad, expected_w) < 0.039


@pytest.mark.parametrize("shared_quantizer", [False, True])
def test_an_fp8_feed_forward_matches_bf16(shared_quantizer):
    """``use_shared_quantizer`` quantizes the input once for w1 and w3; it must
    not change what they compute."""
    ffn = _feed_forward(shared_quantizer=shared_quantizer)
    x = torch.randn(2, 128, DIM, dtype=torch.bfloat16, device=DEV)
    cotangent = torch.randn(2, 128, DIM, device=DEV)

    out, x_grad, _ = _forward_backward(ffn, x, cotangent)

    gate = x.float() @ ffn.w1.weight.float().t()
    up = x.float() @ ffn.w3.weight.float().t()
    expected = (F.silu(gate) * up) @ ffn.w2.weight.float().t()
    assert _rel_error(out, expected) < 0.066
    assert x_grad.shape == x.shape and x_grad.isfinite().all()


# --- does compiling it change anything ---


def _assert_compiled_matches_eager(module, x, cotangent):
    eager = _forward_backward(module, x, cotangent)
    compiled = torch.compile(module, fullgraph=True)
    traced = _forward_backward(compiled, x, cotangent)

    torch.testing.assert_close(traced[0], eager[0], rtol=0, atol=0)
    torch.testing.assert_close(traced[1], eager[1], rtol=0, atol=0)
    for compiled_grad, eager_grad in zip(traced[2], eager[2]):
        torch.testing.assert_close(compiled_grad, eager_grad, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [128, 100], ids=["aligned", "padded"])
def test_a_compiled_projection_matches_eager(rows):
    linear = _linear()
    x = torch.randn(2, rows, DIM, dtype=torch.bfloat16, device=DEV)
    cotangent = torch.randn(2, rows, HIDDEN, device=DEV)

    _assert_compiled_matches_eager(linear, x, cotangent)


@pytest.mark.parametrize("rows", [128, 100], ids=["aligned", "padded"])
def test_a_compiled_feed_forward_matches_eager(rows):
    """The whole block in one graph: two quantizations, the fused SwiGLU and
    three GEMMs, with the fp8 wrapper never crossing the graph boundary."""
    module = _Block(_feed_forward(shared_quantizer=True))
    x = torch.randn(2, rows, DIM, dtype=torch.bfloat16, device=DEV)
    cotangent = torch.randn(2, rows, DIM, device=DEV)

    _assert_compiled_matches_eager(module, x, cotangent)


def test_a_compiled_feed_forward_matches_eager_under_activation_checkpointing():
    """Recompute runs the forward a second time inside the backward, and the
    quantization it replays must land on the same bytes."""
    module = _Block(
        checkpoint_wrapper(
            _feed_forward(shared_quantizer=True),
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
    )
    x = torch.randn(2, 128, DIM, dtype=torch.bfloat16, device=DEV)
    cotangent = torch.randn(2, 128, DIM, device=DEV)

    _assert_compiled_matches_eager(module, x, cotangent)


def test_activation_checkpointing_does_not_change_the_result():
    plain = _feed_forward(shared_quantizer=True)
    checkpointed = checkpoint_wrapper(
        _feed_forward(shared_quantizer=True),
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    x = torch.randn(2, 128, DIM, dtype=torch.bfloat16, device=DEV)
    cotangent = torch.randn(2, 128, DIM, device=DEV)

    expected = _forward_backward(plain, x, cotangent)
    actual = _forward_backward(checkpointed, x, cotangent)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("fused", [True, False], ids=["fused_swiglu", "bf16_swiglu"])
def test_a_compiled_shared_quantizer_feed_forward_takes_dynamic_lengths(fused):
    """A changing sequence length (seq warmup) makes the compiled length a SymInt:
    the shared quantizer's ``orig_shape`` must still reach the fp8 GEMMs."""
    torch.manual_seed(0)
    config = FeedForward.Config(
        dim=DIM,
        hidden_dim=HIDDEN,
        w1=FP8Linear.Config(in_features=DIM, out_features=HIDDEN),
        w2=FP8Linear.Config(in_features=HIDDEN, out_features=DIM),
        w3=FP8Linear.Config(in_features=DIM, out_features=HIDDEN),
    )
    config.use_shared_quantizer = True
    config.dense_fused_swiglu_quant = fused
    ffn = FeedForward(config).to(device=DEV, dtype=torch.bfloat16)
    block = _Block(ffn)
    compiled = torch.compile(block, fullgraph=True, dynamic=True)
    for rows in (128, 192):
        x = torch.randn(2, rows, DIM, dtype=torch.bfloat16, device=DEV)
        cotangent = torch.randn(2, rows, DIM, device=DEV)
        eager = _forward_backward(block, x, cotangent)
        traced = _forward_backward(compiled, x, cotangent)
        torch.testing.assert_close(traced[0], eager[0], rtol=0, atol=0)
        torch.testing.assert_close(traced[1], eager[1], rtol=0, atol=0)
