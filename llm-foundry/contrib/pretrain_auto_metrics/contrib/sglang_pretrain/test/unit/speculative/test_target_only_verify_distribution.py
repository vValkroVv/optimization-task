"""Default EAGLE verification must emit the target distribution for any draft token.

Without --speculative-use-rejection-sampling the draft proposes its argmax (top-k)
and tree_speculative_sampling_target_only accepts token t with probability p(t),
otherwise it samples from p with t removed. Whatever token the draft proposes
(the target's argmax, a rare token, or one the target gives zero probability),
the emitted token must follow the target probabilities exactly, and the bonus
token after an accepted draft must follow the next-position target probabilities.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_ROWS = 1 << 18
VOCAB = 12
TARGET_P0 = [0.30, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01, 0.0, 0.0]
TARGET_P1 = [0.05, 0.05, 0.40, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]


def _verify_chain(draft_token, threshold_single=1.0, threshold_acc=1.0, seed=0):
    from sgl_kernel import tree_speculative_sampling_target_only

    torch.manual_seed(seed)
    device = "cuda"
    rows = NUM_ROWS
    # Chain of one draft: position 0 is the last emitted token, position 1 the draft.
    candidates = torch.zeros((rows, 2), dtype=torch.int64, device=device)
    candidates[:, 1] = draft_token
    retrive_index = torch.arange(rows * 2, dtype=torch.int64, device=device).view(
        rows, 2
    )
    retrive_next_token = torch.tensor(
        [[1, -1]], dtype=torch.int64, device=device
    ).repeat(rows, 1)
    retrive_next_sibling = torch.full((rows, 2), -1, dtype=torch.int64, device=device)
    target_probs = (
        torch.tensor([TARGET_P0, TARGET_P1], dtype=torch.float32, device=device)
        .unsqueeze(0)
        .repeat(rows, 1, 1)
        .contiguous()
    )
    predicts = torch.full((rows * 2,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((rows, 2), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((rows,), dtype=torch.int32, device=device)
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=torch.rand((rows, 2), dtype=torch.float32, device=device),
        uniform_samples_for_final_sampling=torch.rand(
            (rows,), dtype=torch.float32, device=device
        ),
        target_probs=target_probs,
        draft_probs=torch.zeros_like(target_probs),
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )
    first = predicts[accept_index[:, 0].long()]
    accepted = accept_token_num == 1
    bonus = predicts[accept_index[accepted, 1].long()]
    return first, accepted, bonus


def _max_sigma_gap(samples, probs):
    """Largest |empirical - expected| over tokens, in binomial standard deviations."""
    n = samples.numel()
    counts = torch.bincount(samples.long(), minlength=VOCAB).double().cpu()
    p = torch.tensor(probs, dtype=torch.float64)
    sigma = torch.sqrt(n * p * (1 - p)).clamp_min(1.0)
    return ((counts - n * p).abs() / sigma).max().item()


@pytest.mark.parametrize(
    "draft_token", [0, 4, 9, 10], ids=["argmax", "mid", "rare", "zero_prob"]
)
def test_emitted_token_follows_target(draft_token):
    first, accepted, bonus = _verify_chain(draft_token)
    assert (first >= 0).all()
    assert _max_sigma_gap(first, TARGET_P0) < 5.0
    # The draft is accepted with exactly its target probability.
    rate = accepted.double().mean().item()
    assert abs(rate - TARGET_P0[draft_token]) < 5 * (0.25 / NUM_ROWS) ** 0.5 + 1e-9
    if bonus.numel() > 1000:
        assert _max_sigma_gap(bonus, TARGET_P1) < 5.0


def test_relaxed_threshold_breaks_target_distribution():
    # Sensitivity check: lossy acceptance (threshold_acc < 1) must be detected.
    first, _, _ = _verify_chain(9, threshold_acc=0.05)
    assert _max_sigma_gap(first, TARGET_P0) > 50.0
