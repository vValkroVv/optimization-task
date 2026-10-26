"""top_p_renorm_prob at top_p = 1, the call EAGLE verify always makes.

Verify runs top_p_renorm_prob on every target row, including rows with
top_p = 1.0, where it should be a pure renormalization. For vocabularies of at
least air_top_p::NUM_BUCKETS (2048) FlashInfer uses the AIR radix top-p kernel,
which has no top_p >= 1 fast path: the threshold bucket is found by comparing a
sequential prefix sum against the tree-reduced row total, so fp32 rounding can
end the scan before the last non-empty bucket and zero the smallest
probabilities. These tests pin what that may cost: only the far tail can be
dropped, the dropped mass stays at fp32-rounding level, and the result is a
valid distribution proportional to the input on the kept tokens.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

ROWS = 64
VOCAB = 128256  # model vocab: AIR radix path
TOKENIZER_VOCAB = 128016  # ids >= this are reserved tokens the model never saw
MAX_LOST_MASS = 1e-5  # fp32 rounding level of a row sum


def _renorm_top_p_one(probs):
    from sgl_kernel import top_p_renorm_prob

    top_p = torch.ones(probs.shape[0], dtype=torch.float32, device=probs.device)
    return top_p_renorm_prob(probs.contiguous(), top_p)


def _check_rows(probs, out):
    probs, out = probs.double(), out.double()
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    torch.testing.assert_close(
        out.sum(-1),
        torch.ones(out.shape[0], dtype=out.dtype, device=out.device),
        atol=1e-5,
        rtol=0,
    )
    lost = []
    for r in range(probs.shape[0]):
        p, o = probs[r], out[r]
        dropped = (o == 0) & (p > 0)
        kept = o > 0
        lost.append(p[dropped].sum().item())
        if dropped.any():
            # Only the far tail may go: every dropped token is no more likely
            # than every kept token.
            assert p[dropped].max() <= p[kept].min()
        # Kept tokens keep their relative probabilities.
        torch.testing.assert_close(o[kept], p[kept] / p[kept].sum(), rtol=1e-4, atol=0)
    return torch.tensor(lost)


@pytest.mark.parametrize("scale", [1.0, 3.0, 8.0])
@pytest.mark.parametrize("seed", range(4))
def test_top_p_one_drops_at_most_rounding_mass(scale, seed):
    torch.manual_seed(seed)
    logits = torch.randn(ROWS, VOCAB, device="cuda") * scale
    # Reserved tokens: the model gives them almost no probability.
    logits[:, TOKENIZER_VOCAB:] = -30.0
    probs = torch.softmax(logits, dim=-1)
    lost = _check_rows(probs, _renorm_top_p_one(probs))
    assert lost.max().item() <= MAX_LOST_MASS, (
        f"top_p=1 dropped up to {lost.max().item():.3e} of a row's mass "
        f"(rows with any drop: {(lost > 0).sum().item()}/{ROWS})"
    )


def test_top_p_one_small_vocab_is_exact_renorm():
    # Below 2048 buckets FlashInfer uses TopPRenormProb, whose p >= 1 fast path
    # only divides by the row sum.
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(ROWS, 1024, device="cuda") * 3.0, dim=-1)
    out = _renorm_top_p_one(probs)
    assert ((out == 0) == (probs == 0)).all()
    torch.testing.assert_close(
        out, probs / probs.sum(-1, keepdim=True), rtol=1e-5, atol=1e-9
    )


@pytest.mark.xfail(
    strict=False,
    reason="Known AIR top-p hazard: with only a few finite logits a rounding "
    "overshoot can drop a whole bucket that carries real probability mass "
    "(observed ~7e-3 on a 32-finite-logit row at T=1.33).",
)
def test_top_p_one_sparse_rows_keep_all_mass():
    torch.manual_seed(0)
    logits = torch.full((128, VOCAB), float("-inf"), device="cuda")
    logits[:, :32] = torch.randn(128, 32, device="cuda") * 3.0
    temps = torch.empty(128, 1, device="cuda").uniform_(0.5, 1.5)
    probs = torch.softmax(logits / temps, dim=-1)
    lost = _check_rows(probs, _renorm_top_p_one(probs))
    assert lost.max().item() <= MAX_LOST_MASS
