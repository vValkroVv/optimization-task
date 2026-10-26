"""_DSAFunction (dsa_sparse_attention) forward+backward vs a pure-torch fp32 reference.

Absolute correctness of the fused operator: attention output, dq_attn/dkv (LM gradient)
and d_index_q/d_index_k/d_weights (KL gradient) against reference math computed with
torch autograd on the operator's OWN top-k selection. This is the test the SP gates
cannot replace: they compare SP vs single-GPU, but both sides share this operator, so
a units/math bug passes them -- only an independent reference catches it.

Covers: 64-head FlashMLA path, 32-head padded path, varlen (block-diagonal), an
offset SP-style chunk, and the -1-padding leak case (K > Sk with the last query).
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
from llmfoundry.models.layers import dsa_operator as OP

import torch.nn.functional as F  # noqa: E402

DEV = "cuda"
REL_TOL, COS_TOL = 3e-2, 0.995   # bf16 kernels vs fp32 reference


def _valid_mask(topk_length, k):
    return torch.arange(k, device=DEV)[None, None, :] < topk_length[None, :, None]


def _index_logits_topk(iq, ik, w, topk_safe, scale):
    out = []
    for b in range(iq.shape[0]):
        kg = ik[b][topk_safe[b].long()]
        ph = F.relu(torch.einsum("shd,skd->shk", iq[b], kg))
        out.append((ph * w[b].unsqueeze(-1)).sum(1) * scale)
    return torch.stack(out)


def ref_grads(index_q, index_k, weights, q_attn, kv, topk, topk_length,
              s_attn, s_idx, d_v, loss_coeff, dout):
    """fp32 autograd reference sharing the operator's selection (see dev harness dsa_op.py)."""
    b_sz, s, _, _ = q_attn.shape
    k = topk.shape[-1]
    topk_safe = topk.clamp_min(0)
    valid = _valid_mask(topk_length, k)
    neg = torch.full((1,), float("-inf"), device=DEV)

    qa = q_attn.detach().float().clone().requires_grad_(True)
    kvv = kv.detach().float().clone().requires_grad_(True)

    def sparse_attn(q, kvt):
        outs = []
        for b in range(b_sz):
            kg = kvt[b][topk_safe[b].long()]
            logit = torch.einsum("shd,skd->shk", q[b], kg) * s_attn
            logit = torch.where(valid[0].unsqueeze(1), logit, neg)
            p = torch.softmax(logit, dim=-1)
            outs.append(torch.einsum("shk,skd->shd", p, kg[..., :d_v]))
        return torch.stack(outs)

    out_ref = sparse_attn(qa, kvv)
    out_ref.backward(dout.float())

    iq = index_q.detach().float().clone().requires_grad_(True)
    ik = index_k.squeeze(2).detach().float().clone().requires_grad_(True)
    wr = weights.detach().float().clone().requires_grad_(True)
    with torch.no_grad():
        outs = []
        for b in range(b_sz):
            kg = kv[b].float()[topk_safe[b].long()]
            logit = torch.where(valid[0].unsqueeze(1),
                                torch.einsum("shd,skd->shk", q_attn[b].float(), kg) * s_attn, neg)
            p = torch.softmax(logit, dim=-1).sum(1)
            outs.append(p / p.sum(-1, keepdim=True))
        target = torch.stack(outs)
    li = _index_logits_topk(iq, ik, wr, topk_safe, s_idx)
    li = torch.where(valid, li, torch.full_like(li, float("-inf")))
    logp = F.log_softmax(li, dim=-1)
    kl = (torch.where(valid, target, torch.zeros_like(target)) *
          (target.clamp_min(1e-9).log() - logp).nan_to_num(0.0)).sum() * (loss_coeff / (b_sz * s))
    kl.backward()
    return dict(out=out_ref.detach(), dq_attn=qa.grad, dkv=kvv.grad,
                d_index_q=iq.grad, d_index_k=ik.grad.unsqueeze(2), d_weights=wr.grad)


def assert_close(name, test, ref):
    test, ref = test.detach().float(), ref.detach().float()
    rel = ((test - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    cos = F.cosine_similarity(test.flatten(), ref.flatten(), dim=0).item()
    assert rel <= REL_TOL and cos >= COS_TOL, f"{name}: rel={rel:.4g} cos={cos:.5f}"


def run_operator_case(s=512, hq=64, dqk=576, d_v=512, k=128, docs=None, off=0, seed=0,
                      length_storage_offset=0):
    hi, di = 64, 128
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *sh: (torch.randn(*sh, device=DEV, generator=g) * 0.5).bfloat16()
    skv = s + off
    iq = r(1, s, hi, di).requires_grad_(True)
    ik = r(1, skv, 1, di).requires_grad_(True)
    w = r(1, s, hi).requires_grad_(True)
    qa = r(1, s, hq, dqk).requires_grad_(True)
    kv = r(1, skv, dqk).requires_grad_(True)
    dout = r(1, s, hq, d_v)
    s_attn, s_idx = dqk ** -0.5, di ** -0.5
    cu = (torch.tensor([0, *torch.tensor(docs).cumsum(0).tolist()], device=DEV, dtype=torch.int32)
          if docs else None)
    positions = torch.arange(off, off + s, device=DEV)

    topk, tl = OP.select_topk(iq.detach(), ik.detach(), w.detach(), s_idx, k,
                             positions_q=positions, cu_seqlens=cu)
    if length_storage_offset:
        storage = torch.empty(s + length_storage_offset, device=DEV, dtype=torch.int32)
        view = storage[length_storage_offset:]
        view.copy_(tl)
        tl = view
        assert tl.is_contiguous() and tl.data_ptr() % 16 != 0
    out, kl_loss = OP.dsa_sparse_attention(iq, ik, w, qa, kv, positions, cu,
                                           s_attn, s_idx, d_v, k, offset=off,
                                           selection=(topk, tl))
    ((out.float() * dout.float()).sum() + kl_loss).backward()

    ref = ref_grads(iq, ik, w, qa, kv, topk, tl, s_attn, s_idx, d_v, float(s), dout)
    for name, got, want in [("out", out, ref["out"]), ("dq_attn", qa.grad, ref["dq_attn"]),
                            ("dkv", kv.grad, ref["dkv"]), ("d_index_q", iq.grad, ref["d_index_q"]),
                            ("d_weights", w.grad, ref["d_weights"]),
                            ("d_index_k", ik.grad, ref["d_index_k"])]:
        assert_close(name, got, want)


class TestDSAOperator:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("k", [2048, 4096])
    def test_large_topk_forward_backward(self, k):
        run_operator_case(s=17, k=k, hq=32, off=8192)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("heads", [32, 64, 128])
    def test_long_context_topk_forward_backward(self, heads):
        run_operator_case(s=5, k=4096, hq=heads, off=131067)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("heads", [32, 64])
    @pytest.mark.parametrize("storage_offset", [1, 2, 3])
    def test_unaligned_length_view(self, heads, storage_offset):
        """Odd SP block sizes produce contiguous length slices without CuTe alignment."""
        run_operator_case(s=129, hq=heads, docs=[1, 17, 111],
                          length_storage_offset=storage_offset)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_causal_64_heads_flashmla(self):
        run_operator_case(hq=64, docs=None)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("heads", [32, 64, 128])
    def test_varlen_heads(self, heads):
        run_operator_case(hq=heads, docs=[150, 120, 242])

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_causal_32_heads_padded(self):
        run_operator_case(hq=32, docs=None)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_offset_sp_chunk(self):
        """S queries at global positions [off, off+S) over off+S keys (an SP rank's chunk)."""
        run_operator_case(hq=64, off=256)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_padding_stays_masked_when_k_exceeds_keys(self):
        """K > Sk pads every row; the LAST query (pos Sk-1) is where -1 -> Sk-1 padding
        would NOT be causally masked, so it would double-attend the last key. Check that
        query's output against a full-attention reference row."""
        s, k, hq, dqk, d_v = 128, 256, 32, 576, 512
        g = torch.Generator(device=DEV).manual_seed(0)
        r = lambda *sh: (torch.randn(*sh, device=DEV, generator=g) * 0.5).bfloat16()
        iq, ik, w = r(1, s, 64, 128), r(1, s, 1, 128), r(1, s, 64)
        qa, kv = r(1, s, hq, dqk), r(1, s, dqk)
        out, _ = OP.dsa_sparse_attention(iq, ik, w, qa, kv, torch.arange(s, device=DEV), None,
                                         dqk ** -0.5, 128 ** -0.5, d_v, k)
        logit = torch.einsum("hd,kd->hk", qa[0, -1].float(), kv[0].float()) * dqk ** -0.5
        ref_last = torch.einsum("hk,kd->hd", torch.softmax(logit, -1), kv[0, :, :d_v].float())
        cos = F.cosine_similarity(out[0, -1].float().flatten(), ref_last.flatten(), 0).item()
        assert cos > 0.999, f"padding leaked into the last query (cos={cos:.5f})"
