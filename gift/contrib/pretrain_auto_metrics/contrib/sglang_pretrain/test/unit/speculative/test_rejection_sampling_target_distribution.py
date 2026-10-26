"""Rejection-sampling verification must emit the target distribution.

With --speculative-use-rejection-sampling the draft samples t ~ q and the Triton
kernel accepts t with probability min(1, p(t) / q(t)), otherwise it samples from
norm(max(p - q, 0)). For any q the emitted token must follow p, and the bonus
token after an accepted draft must follow the next-position target probabilities.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_ROWS = 1 << 18
VOCAB = 12
TARGET_P0 = [0.30, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01, 0.0, 0.0]
TARGET_P1 = [0.05, 0.05, 0.40, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]
DRAFTS = {
    "equal_to_target": TARGET_P0,
    "peaked": [
        0.70,
        0.10,
        0.05,
        0.05,
        0.02,
        0.02,
        0.02,
        0.01,
        0.01,
        0.01,
        0.005,
        0.005,
    ],
    "uniform": [1.0 / VOCAB] * VOCAB,
    "covers_zero_target_tokens": [
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.25,
        0.25,
    ],
}


def _verify_chain(draft_q, seed=0):
    from sglang.srt.speculative.reject_sampling import (
        chain_speculative_sampling_triton,
    )

    torch.manual_seed(seed)
    device = "cuda"
    rows = NUM_ROWS
    q = torch.tensor(draft_q, dtype=torch.float32, device=device)
    q = q / q.sum()
    candidates = torch.zeros((rows, 2), dtype=torch.int64, device=device)
    # The draft token is sampled from the same q that the kernel is given.
    candidates[:, 1] = torch.multinomial(q, rows, replacement=True)
    retrive_index = torch.arange(rows * 2, dtype=torch.int64, device=device).view(
        rows, 2
    )
    target_probs = (
        torch.tensor([TARGET_P0, TARGET_P1], dtype=torch.float32, device=device)
        .unsqueeze(0)
        .repeat(rows, 1, 1)
        .contiguous()
    )
    # One draft step: draft_probs has num_steps rows, target_probs num_steps + 1.
    draft_probs = q.view(1, 1, VOCAB).repeat(rows, 1, 1).contiguous()
    predicts = torch.full((rows * 2,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((rows, 2), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((rows,), dtype=torch.int32, device=device)
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.rand((rows, 2), dtype=torch.float32, device=device),
        uniform_samples_for_final_sampling=torch.rand(
            (rows,), dtype=torch.float32, device=device
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    first = predicts[accept_index[:, 0].long()]
    accepted = accept_token_num == 1
    bonus = predicts[accept_index[accepted, 1].long()]
    return q.cpu().double(), first, accepted, bonus


def _max_sigma_gap(samples, probs):
    """Largest |empirical - expected| over tokens, in binomial standard deviations."""
    n = samples.numel()
    counts = torch.bincount(samples.long(), minlength=VOCAB).double().cpu()
    p = torch.tensor(probs, dtype=torch.float64)
    sigma = torch.sqrt(n * p * (1 - p)).clamp_min(1.0)
    return ((counts - n * p).abs() / sigma).max().item()


@pytest.mark.parametrize("draft", list(DRAFTS), ids=list(DRAFTS))
def test_emitted_token_follows_target(draft):
    q, first, accepted, bonus = _verify_chain(DRAFTS[draft])
    assert (first >= 0).all()
    assert _max_sigma_gap(first, TARGET_P0) < 5.0
    # Acceptance equals the overlap sum(min(p, q)).
    overlap = torch.minimum(torch.tensor(TARGET_P0, dtype=torch.float64), q).sum()
    rate = accepted.double().mean().item()
    assert abs(rate - overlap.item()) < 5 * (0.25 / NUM_ROWS) ** 0.5
    assert _max_sigma_gap(bonus, TARGET_P1) < 5.0


def test_mismatched_draft_probs_break_target_distribution():
    # Sensitivity check: q passed to the kernel differs from the q the draft sampled.
    from sglang.srt.speculative import reject_sampling

    original = reject_sampling.chain_speculative_sampling_triton

    def wrong_q(**kwargs):
        kwargs["draft_probs"] = torch.full_like(kwargs["draft_probs"], 1.0 / VOCAB)
        return original(**kwargs)

    reject_sampling.chain_speculative_sampling_triton = wrong_q
    try:
        _, first, _, _ = _verify_chain(DRAFTS["peaked"])
    finally:
        reject_sampling.chain_speculative_sampling_triton = original
    assert _max_sigma_gap(first, TARGET_P0) > 50.0
