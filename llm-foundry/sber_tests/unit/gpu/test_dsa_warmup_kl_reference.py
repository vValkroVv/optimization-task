"""Stage-1 dense warm-up KL and its indexer gradients against a pure-torch fp32 reference.

This is the only gradient the indexer gets during the whole warm-up, and it had no reference
test. Reference, all fp32 with autograd: target = row-normalised sum over heads of
softmax(q_attn . kv * s_attn) over causal keys; logits = s_idx * sum_h w_h * relu(iq_h . ik);
KL = sum t * (log t - log_softmax(logits)). Checked for a full sequence, an SP-style offset
chunk, packed documents, compute_value=False and linearity in d_kl.
"""
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
from llmfoundry.models.layers import dsa_operator as OP  # noqa: E402

import torch.nn.functional as F  # noqa: E402

DEV = "cuda"
HI, DI, HQ, DQK = 64, 128, 64, 576
REL_TOL, COS_TOL = 3e-2, 0.995


def _causal_doc_mask(positions_q, sk, cu):
    """(Sq, Sk) bool: key k is visible to query q iff k <= pos(q) and same document."""
    keys = torch.arange(sk, device=DEV)
    m = keys[None, :] <= positions_q[:, None]
    if cu is not None:
        bounds = cu[1:-1].to(DEV)
        dq = torch.bucketize(positions_q, bounds, right=True)
        dk = torch.bucketize(keys, bounds, right=True)
        m &= dq[:, None] == dk[None, :]
    return m


def reference(index_q, index_k, weights, q_attn, kv, positions_q, cu, s_attn, s_idx, coeff):
    """Returns (lse (1,Sq,Hq) fp32, kl value, d_index_q, d_index_k, d_weights) for loss = coeff * KL."""
    sk = kv.shape[1]
    mask = _causal_doc_mask(positions_q, sk, cu)
    neg = torch.tensor(float("-inf"), device=DEV)
    with torch.no_grad():
        logit = torch.einsum("shd,kd->shk", q_attn[0].float(), kv[0].float()) * s_attn
        logit = torch.where(mask[:, None, :], logit, neg)
        lse = torch.logsumexp(logit, dim=-1)                         # (Sq, Hq)
        p = torch.softmax(logit, dim=-1).sum(1)                      # (Sq, Sk) summed over heads
        t = p / p.sum(-1, keepdim=True)
    iq = index_q.detach().float().clone().requires_grad_(True)
    ik = index_k.squeeze(2).detach().float().clone().requires_grad_(True)
    w = weights.detach().float().clone().requires_grad_(True)
    li = (F.relu(torch.einsum("shd,kd->shk", iq[0], ik[0])) * w[0].unsqueeze(-1)).sum(1) * s_idx
    li = torch.where(mask, li, neg)
    logp = F.log_softmax(li, dim=-1)
    kl = (torch.where(mask, t, torch.zeros_like(t)) * (t.clamp_min(1e-9).log() - logp)).nan_to_num(0.0).sum()
    (coeff * kl).backward()
    return lse[None], kl.detach(), iq.grad, ik.grad.unsqueeze(2), w.grad


def assert_close(name, test, ref):
    test, ref = test.detach().float(), ref.detach().float()
    rel = ((test - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    cos = F.cosine_similarity(test.flatten(), ref.flatten(), dim=0).item()
    assert rel <= REL_TOL and cos >= COS_TOL, f"{name}: rel={rel:.4g} cos={cos:.5f}"


def run_case(sq=256, off=0, docs=None, coeff=0.37, compute_value=True, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *sh: (torch.randn(*sh, device=DEV, generator=g) * 0.5).bfloat16()
    sk = sq + off
    iq = r(1, sq, HI, DI).requires_grad_(True)
    ik = r(1, sk, 1, DI).requires_grad_(True)
    w = r(1, sq, HI).requires_grad_(True)
    qa, kv = r(1, sq, HQ, DQK), r(1, sk, DQK)
    s_attn, s_idx = DQK ** -0.5, DI ** -0.5
    positions = torch.arange(off, off + sq, device=DEV)
    cu = (torch.tensor([0, *torch.tensor(docs).cumsum(0).tolist()], device=DEV, dtype=torch.int32)
          if docs else None)

    lse, kl_ref, dq_ref, dk_ref, dw_ref = reference(iq, ik, w, qa, kv, positions, cu, s_attn, s_idx, coeff)
    kl = OP.dsa_warmup_kl(iq, ik, w, qa, kv, lse.contiguous(), s_attn, s_idx, offset=off,
                          cu_seqlens=cu, compute_value=compute_value)
    (coeff * kl).backward()
    if compute_value:
        rel = ((kl.float() - kl_ref) / kl_ref.clamp_min(1e-6)).abs().item()
        assert rel < 2e-2, f"KL value rel err {rel:.4g} (got {kl.item():.5g}, ref {kl_ref.item():.5g})"
    else:
        assert kl.item() == 0.0, "compute_value=False must return a zero placeholder value"
    assert_close("d_index_q", iq.grad, dq_ref)
    assert_close("d_index_k", ik.grad, dk_ref)
    assert_close("d_weights", w.grad, dw_ref)


class TestWarmupKLReference:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_full_sequence(self):
        run_case()

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_offset_sp_chunk(self):
        """Queries at global positions [off, off+Sq) over off+Sq keys (one SP rank's block)."""
        run_case(off=192)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_varlen_packed_documents(self):
        run_case(docs=[100, 60, 96])

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_varlen_offset_chunk_crossing_a_document_boundary(self):
        run_case(off=64, docs=[100, 60, 160])

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_gradients_do_not_depend_on_compute_value(self):
        """The logging-only KL scalar can be skipped; the analytic indexer gradient must not change."""
        run_case(compute_value=False)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_upstream_coefficient_scales_gradients(self):
        """d_kl (aux_kl_loss_koeff / S_loc in the layer) must scale d_index_* linearly."""
        g = torch.Generator(device=DEV).manual_seed(3)
        r = lambda *sh: (torch.randn(*sh, device=DEV, generator=g) * 0.5).bfloat16()
        sq = 128
        iq, ik, w = r(1, sq, HI, DI), r(1, sq, 1, DI), r(1, sq, HI)
        qa, kv = r(1, sq, HQ, DQK), r(1, sq, DQK)
        pos = torch.arange(sq, device=DEV)
        lse, *_ = reference(iq, ik, w, qa, kv, pos, None, DQK ** -0.5, DI ** -0.5, 1.0)
        grads = []
        for c in (1.0, 4.0):
            iq_c = iq.clone().requires_grad_(True)
            kl = OP.dsa_warmup_kl(iq_c, ik, w, qa, kv, lse.contiguous(), DQK ** -0.5, DI ** -0.5)
            (c * kl).backward()
            grads.append(iq_c.grad.float())
        assert_close("d_index_q scales with d_kl", grads[1], 4.0 * grads[0])
