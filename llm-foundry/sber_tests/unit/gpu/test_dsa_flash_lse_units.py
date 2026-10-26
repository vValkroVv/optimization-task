"""LSE-units tripwire: flash_mla_sparse_fwd must return NATURAL-log LSE.

The DSA operator feeds FlashMLA's lse straight into the cuDNN score-recompute/backward
wrappers, which consume natural log. FlashMLA's sparse-prefill epilogue converts its
internal log2-domain running max to natural log (logf(rL) + rM*CUDART_LN2_F), so no
LN2 conversion is needed in the operator. A FlashMLA version bump that changes the convention
would silently mis-normalize every stage-2 KL target and attention gradient; this test
turns that into a loud failure.
"""
import math

import pytest

import torch
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("the DSA kernels are Hopper-only (sm90)", allow_module_level=True)
pytest.importorskip(
    "llmfoundry.models.layers.dsa_operator",
    reason="the DSA kernel packages are not installed in this image",
)
import flash_mla
from llmfoundry.models.layers import cudnn_dsa

DEV = "cuda"


@pytest.fixture(scope="module")
def sparse_fwd_result():
    torch.manual_seed(0)
    sq, sk, hq, dqk, d_v, k = 64, 1024, 64, 576, 512, 128
    s_attn = dqk ** -0.5
    q = torch.randn(1, sq, hq, dqk, device=DEV, dtype=torch.bfloat16)
    kv = torch.randn(1, sk, dqk, device=DEV, dtype=torch.bfloat16)
    pos = torch.arange(sk - sq, sk, device=DEV)                     # every query has >= k prior keys
    ids = (pos[:, None] - torch.arange(k, device=DEV)[None, :]).to(torch.int32)
    ln = torch.full((sq,), k, device=DEV, dtype=torch.int32)
    out, _mx, lse = flash_mla.flash_mla_sparse_fwd(
        q[0], kv[0].unsqueeze(1), ids.unsqueeze(1), sm_scale=s_attn, d_v=d_v, topk_length=ln)
    # reference logsumexp over the SAME selected keys, fp32
    k_sel = kv[0][ids.long()]
    scores = torch.einsum("qhd,qkd->qhk", q[0].float(), k_sel.float()) * s_attn
    return dict(q=q, kv=kv, ids=ids, ln=ln, s_attn=s_attn, hq=hq,
                lse=lse if lse.shape == (sq, hq) else lse.T,
                ref_nat=torch.logsumexp(scores, dim=-1))


def _median_rel(a, b):
    return ((a - b).abs() / b.abs().clamp_min(1e-6)).median().item()


class TestFlashLseUnits:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_lse_is_natural_log(self, sparse_fwd_result):
        r = sparse_fwd_result
        assert _median_rel(r["lse"], r["ref_nat"]) < 1e-4, "lse != natural logsumexp"

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_lse_is_not_log2(self, sparse_fwd_result):
        r = sparse_fwd_result
        # discriminating check: the log2 hypothesis must be FAR off (~31% at these scales)
        assert _median_rel(r["lse"], r["ref_nat"] / math.log(2)) > 0.05

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_cudnn_targets_sum_to_one(self, sparse_fwd_result):
        """End-to-end units check: the cuDNN target recompute fed the UNCONVERTED lse
        must produce softmax rows summing to 1."""
        r = sparse_fwd_result
        tl = r["ln"].unsqueeze(0).contiguous()
        target = cudnn_dsa.sparse_attn_score_recompute_wrapper(
            r["q"], r["kv"], r["lse"].unsqueeze(0), r["ids"].unsqueeze(0),
            softmax_scale=r["s_attn"], qhead_per_kv_head=r["hq"], topk_length=tl)["target"]
        sums = target.float().sum(-1)
        assert (sums - 1).abs().max().item() < 1e-2, "targets do not sum to 1 (lse units broken)"
