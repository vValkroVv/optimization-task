"""Score and denominator checks for the shared-memory weights/LSE loaders.

Run this file under Compute Sanitizer racecheck as well as ordinary pytest.
Numerical checks alone do not reliably expose a late zero-fill of loaded weights.
"""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("the DSA kernels are Hopper-only (sm90)", allow_module_level=True)

from llmfoundry.models.layers.cudnn_dsa.score_recompute._interface_sm90 import (  # noqa: E402
    dense_attn_score_recompute,
    dense_indexer_score_recompute,
    sparse_attn_score_recompute,
    sparse_indexer_score_recompute,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_gpu]


def _inputs(heads, dim):
    generator = torch.Generator(device="cuda").manual_seed(41)
    q = torch.randn(1, 77, heads, dim, device="cuda", generator=generator).bfloat16() * 0.25
    k = torch.randn(1, 257, dim, device="cuda", generator=generator).bfloat16() * 0.25
    ids = torch.arange(128, device="cuda", dtype=torch.int32)[None, None].expand(1, 77, -1).contiguous()
    mask = torch.arange(257, device="cuda")[None] <= torch.arange(180, 257, device="cuda")[:, None]
    offset = torch.tensor([180], device="cuda", dtype=torch.int32)
    return q, k, ids, mask, offset


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("heads", [32, 64, 128])
def test_attention_scores_and_denominator(mode, heads):
    q, k, ids, mask, offset = _inputs(heads, 576)
    scale = 576 ** -0.5
    for factor in (0.5, 1.0, -0.5, 1.5) * 2:
        query = q * factor
        logits = torch.einsum("qhd,kd->qhk", query[0].float(), k[0].float()) * scale
        if mode == "dense":
            logits = logits.masked_fill(~mask[:, None], float("-inf"))
        lse = logits.logsumexp(-1)[None].contiguous()
        expected = logits.softmax(-1).sum(1)
        if mode == "dense":
            scores, denom = dense_attn_score_recompute(
                query, k.unsqueeze(2), lse, scale, q_causal_offsets=offset,
            )
            torch.testing.assert_close(scores[0][mask], expected[mask], rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(denom[0], expected.sum(-1), rtol=2e-4, atol=2e-5)
        else:
            expected = expected[:, :128]
            expected = expected / expected.sum(-1, keepdim=True)
            scores = sparse_attn_score_recompute(query, k, lse, ids, scale)
            torch.testing.assert_close(scores[0], expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_indexer_scores_and_denominator(mode):
    q, k, ids, mask, offset = _inputs(64, 128)
    generator = torch.Generator(device="cuda").manual_seed(42)
    weights = torch.randn(1, 77, 64, device="cuda", generator=generator).bfloat16() * 0.125
    dots = torch.einsum("qhd,kd->qhk", q[0].float(), k[0].float()).relu()
    scale = 128 ** -0.5
    for factor in (0.5, 1.0, -0.5, 1.5) * 2:
        current_weights = weights * factor
        logits = (dots * current_weights[0].float()[..., None]).sum(1) * scale
        if mode == "dense":
            logits = logits.masked_fill(~mask, float("-inf"))
            scores, denom = dense_indexer_score_recompute(
                q, k.unsqueeze(2), current_weights, sm_scale=scale, q_causal_offsets=offset,
            )
            torch.testing.assert_close(scores[0][mask], logits[mask], rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(denom[0], logits.logsumexp(-1), rtol=2e-4, atol=2e-5)
            # Dense backward applies the causal mask to these raw scores.
            prob = torch.where(mask, (scores[0] - denom[0, :, None]).exp(), 0.0)
            torch.testing.assert_close(prob, logits.softmax(-1), rtol=2e-4, atol=2e-5)
        else:
            scores = sparse_indexer_score_recompute(q, k, current_weights, ids, sm_scale=scale)
            torch.testing.assert_close(scores[0], logits[:, :128].softmax(-1), rtol=2e-4, atol=2e-5)
