"""NaN target logits in EAGLE verify.

A single NaN (or +inf) in a target logits row turns the whole softmax row into
NaN. Both verify kernels then find no token with probability > 0 and fall back
to the last vocab id (VOCAB_SIZE - 1): tree_speculative_sampling_target_only
(speculative_sampling.cuh, `sampled_id = d - 1`) and
chain_speculative_sampling_triton (reject_sampling.py, `final_token =
VOCAB_SIZE - 1`). With SGLANG_SANITIZE_NAN_LOGITS=1 verify replaces NaN/inf
before the softmax, and the row samples a real token again.
"""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.utils.async_probe import sanitize_nan_logits

VOCAB = 12
FALLBACK = VOCAB - 1
ROWS = 4096
# Healthy target rows never give probability to the last id, so a FALLBACK
# emission can only come from the NaN fallback.
TARGET_LOGITS = [
    2.0,
    1.5,
    1.0,
    0.5,
    0.2,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    float("-inf"),
]

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _target_probs(nan_rows, sanitize, device):
    """(ROWS, 2, VOCAB) target probs; rows in nan_rows get one NaN logit at position 0."""
    logits = (
        torch.tensor(TARGET_LOGITS, dtype=torch.float32, device=device)
        .view(1, 1, VOCAB)
        .repeat(ROWS, 2, 1)
    )
    logits[nan_rows, 0, 3] = float("nan")
    flat = logits.view(ROWS * 2, VOCAB)
    with envs.SGLANG_SANITIZE_NAN_LOGITS.override(sanitize):
        sanitize_nan_logits(flat, "test")
    return F.softmax(flat, dim=-1).view(ROWS, 2, VOCAB).contiguous()


def _chain_inputs(device):
    candidates = torch.zeros((ROWS, 2), dtype=torch.int64, device=device)
    candidates[:, 1] = 2  # draft proposes a token the target likes
    retrive_index = torch.arange(ROWS * 2, dtype=torch.int64, device=device).view(
        ROWS, 2
    )
    return dict(
        predicts=torch.full((ROWS * 2,), -1, dtype=torch.int32, device=device),
        accept_index=torch.full((ROWS, 2), -1, dtype=torch.int32, device=device),
        accept_token_num=torch.zeros((ROWS,), dtype=torch.int32, device=device),
        candidates=candidates,
        retrive_index=retrive_index,
        uniform_samples=torch.rand((ROWS, 2), dtype=torch.float32, device=device),
        uniform_samples_for_final_sampling=torch.rand(
            (ROWS,), dtype=torch.float32, device=device
        ),
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )


def _verify(kernel, target_probs):
    device = target_probs.device
    kw = _chain_inputs(device)
    if kernel == "target_only":
        from sgl_kernel import tree_speculative_sampling_target_only

        tree_speculative_sampling_target_only(
            retrive_next_token=torch.tensor(
                [[1, -1]], dtype=torch.int64, device=device
            ).repeat(ROWS, 1),
            retrive_next_sibling=torch.full(
                (ROWS, 2), -1, dtype=torch.int64, device=device
            ),
            target_probs=target_probs,
            draft_probs=torch.zeros_like(target_probs),
            **kw,
        )
    else:
        from sglang.srt.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        draft_q = torch.zeros(VOCAB, dtype=torch.float32, device=device)
        draft_q[2] = 1.0  # the draft token was sampled from this q
        chain_speculative_sampling_triton(
            retrive_next_token=None,
            retrive_next_sibling=None,
            target_probs=target_probs,
            draft_probs=draft_q.view(1, 1, VOCAB).repeat(ROWS, 1, 1).contiguous(),
            **kw,
        )
    first = kw["predicts"][kw["accept_index"][:, 0].long()]
    return first, kw["accept_token_num"]


def test_sanitize_off_keeps_nan():
    logits = torch.tensor([[1.0, float("nan"), 0.0]])
    with envs.SGLANG_SANITIZE_NAN_LOGITS.override(False):
        sanitize_nan_logits(logits, "test")
    assert torch.isnan(logits[0, 1])
    assert torch.isnan(F.softmax(logits, dim=-1)).all()


def test_sanitize_on_replaces_nan_and_inf():
    logits = torch.tensor([[1.0, float("nan"), float("inf"), float("-inf"), 0.0]])
    with envs.SGLANG_SANITIZE_NAN_LOGITS.override(True):
        sanitize_nan_logits(logits, "test")
    assert torch.isfinite(logits).all()
    big = torch.tensor(1e30, dtype=torch.float32)
    assert logits[0, 1] == -big and logits[0, 3] == -big
    assert logits[0, 2] == big
    # Dividing by a temperature must not overflow back to inf -> NaN softmax.
    probs = F.softmax(logits / 0.7, dim=-1)
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


@needs_cuda
@pytest.mark.parametrize("kernel", ["target_only", "rejection_sampling"])
def test_nan_row_falls_back_to_last_vocab_id(kernel):
    torch.manual_seed(0)
    nan_rows = torch.arange(0, ROWS, 2, device="cuda")
    healthy = torch.arange(1, ROWS, 2, device="cuda")
    first, accepted = _verify(
        kernel, _target_probs(nan_rows, sanitize=False, device="cuda")
    )
    # Every NaN row rejects the draft and emits the fallback id.
    assert (first[nan_rows] == FALLBACK).all()
    assert (accepted[nan_rows] == 0).all()
    # Healthy rows never emit it (their target gives it zero probability).
    assert (first[healthy] >= 0).all() and (first[healthy] != FALLBACK).all()


@needs_cuda
@pytest.mark.parametrize("kernel", ["target_only", "rejection_sampling"])
def test_sanitized_nan_row_samples_real_tokens(kernel):
    torch.manual_seed(0)
    nan_rows = torch.arange(0, ROWS, 2, device="cuda")
    first, _ = _verify(kernel, _target_probs(nan_rows, sanitize=True, device="cuda"))
    assert (first >= 0).all()
    assert (first != FALLBACK).all()
