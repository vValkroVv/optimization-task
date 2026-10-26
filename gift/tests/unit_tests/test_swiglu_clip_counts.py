"""SwiGLU clamp counts (``swiglu_clip_counts``) on the bf16 paths.

A module counts only while a caller has handed it an int64[3] slot -- ModellingSource
does so on a log step -- as [gate clamped, up clamped, gate elements scanned].
The counts are checked against the clamp rule spelled out here: gate is clamped
at ``>= limit``, up at ``<= -limit`` or ``>= limit``. The fp8 kernels are covered
in ``test_fp8_swiglu_clip_zloss.py``, the bf16 experts in ``test_swiglu_limit.py``.
"""

import torch

from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.moe.experts.swiglu import fused_swiglu
from torchtitan.models.common.nn_modules import Linear

DIM, HIDDEN, LIMIT = 16, 32, 0.5


def _expected_counts(gate: torch.Tensor, up: torch.Tensor, limit: float) -> list[int]:
    gate_clamped = int(gate.ge(limit).sum())
    up_clamped = int((up.le(-limit) | up.ge(limit)).sum())
    return [gate_clamped, up_clamped, gate.numel()]


def _slot() -> torch.Tensor:
    return torch.zeros(3, dtype=torch.int64)


def _feed_forward(limit: float) -> FeedForward:
    ffn = FeedForward.Config(
        dim=DIM,
        hidden_dim=HIDDEN,
        w1=Linear.Config(in_features=DIM, out_features=HIDDEN, bias=False),
        w2=Linear.Config(in_features=HIDDEN, out_features=DIM, bias=False),
        w3=Linear.Config(in_features=DIM, out_features=HIDDEN, bias=False),
        swiglu_limit=limit,
    ).build()
    torch.manual_seed(0)
    for p in ffn.parameters():
        torch.nn.init.normal_(p)
    return ffn


def test_feed_forward_counts_what_it_clamps():
    ffn = _feed_forward(LIMIT)
    ffn.swiglu_clip_counts = _slot()
    x = 3 * torch.randn(2, 4, DIM)

    ffn(x)

    # The module's own projections: a separate matmul could round a value
    # at the limit to the other side.
    gate, up = ffn.w1(x), ffn.w3(x)
    expected = _expected_counts(gate, up, LIMIT)
    # The input is scaled so both halves clamp, but not everywhere.
    assert 0 < expected[0] < expected[2] and 0 < expected[1] < expected[2]
    assert ffn.swiglu_clip_counts.tolist() == expected


def test_feed_forward_counts_accumulate_across_calls():
    """The module only adds to its slot; clearing it is the caller's job."""
    ffn = _feed_forward(LIMIT)
    ffn.swiglu_clip_counts = _slot()
    x = 3 * torch.randn(4, DIM)

    ffn(x)
    once = ffn.swiglu_clip_counts.clone()
    ffn(x)

    assert torch.equal(ffn.swiglu_clip_counts, 2 * once)


def test_feed_forward_without_a_slot_or_a_limit_counts_nothing():
    x = 3 * torch.randn(4, DIM)
    # No slot handed out: an unmonitored step still clamps, and pays nothing.
    ffn = _feed_forward(LIMIT)
    ffn(x)
    assert ffn.swiglu_clip_counts is None
    # A slot but no limit: nothing is clamped, so nothing is counted.
    ffn = _feed_forward(0.0)
    ffn.swiglu_clip_counts = _slot()
    ffn(x)
    assert ffn.swiglu_clip_counts.tolist() == [0, 0, 0]


def test_a_value_at_the_limit_counts_as_clamped():
    """The count follows the clamp: ``clamp(max=limit)`` at the limit is a clamp."""
    gate = torch.tensor([[LIMIT, LIMIT - 1e-3, -10.0]])
    up = torch.tensor([[LIMIT, -LIMIT, LIMIT - 1e-3]])
    counts = _slot()

    fused_swiglu(torch.cat([gate, up], dim=-1), swiglu_limit=LIMIT, clip_counts=counts)

    # fused_swiglu leaves the scanned count to its caller.
    assert counts.tolist() == [1, 2, 0]


def test_fused_swiglu_counts_are_not_changed_by_the_routing_probs():
    """Probs weigh the output after the clamp, so they cannot move the counts."""
    torch.manual_seed(0)
    h = 3 * torch.randn(8, 2 * HIDDEN)
    probs = torch.rand(8, 1)
    with_probs, without_probs = _slot(), _slot()

    out = fused_swiglu(h, probs, swiglu_limit=LIMIT, clip_counts=with_probs)
    fused_swiglu(h, swiglu_limit=LIMIT, clip_counts=without_probs)

    gate, up = h.chunk(2, dim=-1)
    assert torch.equal(with_probs, without_probs)
    assert with_probs.tolist()[:2] == _expected_counts(gate, up, LIMIT)[:2]
    expected = (
        torch.nn.functional.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
    ) * probs
    torch.testing.assert_close(out, expected)
