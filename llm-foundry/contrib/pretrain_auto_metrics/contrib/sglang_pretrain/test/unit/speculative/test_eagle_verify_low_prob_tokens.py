"""Low-probability and reserved tokens in EAGLE verify sampling.

After a rejected draft both verify kernels sample the final token from the
residual relu(p - q) with u = coin * sum(residual), where the sum comes from a
block reduction and the scan that looks for u adds the same values in another
order. When the coin is close to 1 the scan can finish below u:

* tree_speculative_sampling_target_only (speculative_sampling.cuh) then
  returns the last id with non-zero residual;
* chain_speculative_sampling_triton (reject_sampling.py) returns
  VOCAB_SIZE - 1, whatever its probability.

The reserved ids 128016..128255 sit at the end of the vocab and keep a tiny
non-zero probability, so in that case verify emits 128255 (or another reserved
id) with a logprob around -30, as seen in RL rollouts. top_p_renorm_prob at
top_p = 1 does not reliably remove that tail.

These tests pin the sampled distribution on realistic rows: low-probability
tokens come out at their probability, and reserved ids do not come out with
random coins. The coin-near-1 case is an xfail that documents the kernel
fallback.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.sampling.sampling_params import TOP_K_ALL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 128256
TOKENIZER_VOCAB = 128016  # ids >= this are reserved tokens the model never saw
RESERVED_LOGIT = -30.0
ROWS = 1024
COIN_NEAR_ONE = 1.0 - 2.0**-24  # largest fp32 below 1
KERNELS = ["target_only", "rejection_sampling"]


def _logits(scale, seed):
    """(ROWS, 2, VOCAB) target logits: real ids ~ N(0, scale), reserved ids at -30."""
    torch.manual_seed(seed)
    logits = torch.randn(ROWS, 2, VOCAB, device="cuda") * scale
    logits[..., TOKENIZER_VOCAB:] = RESERVED_LOGIT
    return logits


def _verify_probs(logits, temperature):
    """Target probs the way EAGLE verify builds them: softmax(logits / T), then
    the top-k and top-p renorms with no-op parameters (top_k = all, top_p = 1)."""
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

    flat = logits.view(-1, VOCAB)
    n = flat.shape[0]
    probs = F.softmax(flat / temperature, dim=-1)
    probs = top_k_renorm_prob(
        probs, torch.full((n,), TOP_K_ALL, dtype=torch.int32, device="cuda")
    )
    probs = top_p_renorm_prob(probs, torch.ones(n, device="cuda"))
    return probs.view(logits.shape).contiguous()


def _verify(kernel, target_probs, draft, final_coins):
    """One verify step with a single draft token per row. The acceptance coin is
    COIN_NEAR_ONE, so every draft with p < 1 is rejected and the emitted token
    is the final sample from the residual. Returns the emitted ids."""
    rows = target_probs.shape[0]
    candidates = torch.zeros((rows, 2), dtype=torch.int64, device="cuda")
    candidates[:, 1] = draft
    kw = dict(
        predicts=torch.full((rows * 2,), -1, dtype=torch.int32, device="cuda"),
        accept_index=torch.full((rows, 2), -1, dtype=torch.int32, device="cuda"),
        accept_token_num=torch.zeros((rows,), dtype=torch.int32, device="cuda"),
        candidates=candidates,
        retrive_index=torch.arange(rows * 2, device="cuda").view(rows, 2),
        uniform_samples=torch.full((rows, 2), COIN_NEAR_ONE, device="cuda"),
        uniform_samples_for_final_sampling=final_coins,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    if kernel == "target_only":
        from sgl_kernel import tree_speculative_sampling_target_only

        tree_speculative_sampling_target_only(
            retrive_next_token=torch.tensor([[1, -1]], device="cuda").repeat(rows, 1),
            retrive_next_sibling=torch.full((rows, 2), -1, device="cuda"),
            target_probs=target_probs,
            draft_probs=torch.zeros_like(target_probs),
            **kw,
        )
    else:
        from sglang.srt.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        draft_probs = torch.zeros((rows, 1, VOCAB), device="cuda")
        draft_probs[torch.arange(rows, device="cuda"), 0, draft] = 1.0
        chain_speculative_sampling_triton(
            retrive_next_token=None,
            retrive_next_sibling=None,
            target_probs=target_probs,
            draft_probs=draft_probs,
            **kw,
        )
    assert (kw["accept_token_num"] == 0).all(), "every draft must be rejected"
    return kw["predicts"][kw["accept_index"][:, 0].long()].long()


@pytest.mark.parametrize("kernel", KERNELS)
def test_low_probability_tokens_come_out_at_their_rate(kernel):
    """With random final coins every bucket of true logprob is emitted as often
    as its residual probability says (5 sigma), down to the -20 tail, and
    reserved ids never come out."""
    temperature, draws = 1.0, 1000
    logits = _logits(scale=3.0, seed=0)
    probs = _verify_probs(logits, temperature)
    draft = logits[:, 0].argmax(-1)

    # Residual after rejecting the draft: p without the draft token, renormalized.
    true_lp = torch.log_softmax(logits[:, 0].double() / temperature, dim=-1)
    true_lp[torch.arange(ROWS, device="cuda"), draft] = -math.inf
    resid_lp = true_lp - torch.logsumexp(true_lp, dim=-1, keepdim=True)

    edges = [-math.inf, -20.0, -15.0, -10.0, -5.0, 0.0]
    bucket = torch.bucketize(resid_lp, torch.tensor(edges[1:-1], device="cuda"))
    resid_p = resid_lp.exp()
    expected = torch.zeros(len(edges) - 1, dtype=torch.float64, device="cuda")
    expected.scatter_add_(0, bucket.flatten(), resid_p.flatten())
    expected *= draws

    observed = torch.zeros_like(expected)
    reserved = 0
    rows = torch.arange(ROWS, device="cuda")
    for _ in range(draws):
        ids = _verify(kernel, probs, draft, torch.rand(ROWS, device="cuda"))
        reserved += (ids >= TOKENIZER_VOCAB).sum().item()
        observed.scatter_add_(
            0, bucket[rows, ids], torch.ones(ROWS, dtype=torch.float64, device="cuda")
        )

    assert reserved == 0
    n = ROWS * draws
    for b in range(len(edges) - 1):
        exp_b, obs_b = expected[b].item(), observed[b].item()
        sigma = math.sqrt(exp_b * (1 - exp_b / n))
        assert abs(obs_b - exp_b) <= 5 * sigma + 3, (
            f"logprob bucket [{edges[b]}, {edges[b + 1]}): "
            f"observed {obs_b:.0f}, expected {exp_b:.2f}"
        )


@pytest.mark.xfail(
    strict=False,
    reason="Kernel fallback: when the scan ends below coin * sum, target-only "
    "returns the last id with non-zero residual and the Triton RS kernel returns "
    "VOCAB_SIZE - 1. Reserved ids keep a tiny probability at the end of the "
    "vocab, so a coin close to 1 can emit 128255.",
)
@pytest.mark.parametrize("kernel", KERNELS)
def test_coin_near_one_stays_in_tokenizer_vocab(kernel):
    emitted = []
    for seed, (scale, temperature) in enumerate(
        [(2.0, 1.0), (4.0, 1.0), (2.0, 0.7), (6.0, 1.3)]
    ):
        logits = _logits(scale, seed)
        probs = _verify_probs(logits, temperature)
        ids = _verify(
            kernel,
            probs,
            logits[:, 0].argmax(-1),
            torch.full((ROWS,), COIN_NEAR_ONE, device="cuda"),
        )
        emitted.append(ids)
    emitted = torch.cat(emitted)
    bad = emitted[emitted >= TOKENIZER_VOCAB]
    assert bad.numel() == 0, (
        f"{bad.numel()}/{emitted.numel()} rows emitted reserved ids "
        f"{sorted(set(bad.tolist()))[:5]}"
    )
