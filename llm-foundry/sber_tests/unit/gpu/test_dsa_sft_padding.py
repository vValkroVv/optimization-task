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

DEV = "cuda"
HI, DI = 64, 128
HQ_FLASH, HQ_PADDED = 64, 32
KV_LORA, ROPE = 512, 64
DQK, DV = KV_LORA + ROPE, KV_LORA
S_ATTN = (128 + ROPE) ** -0.5
S_IDX = DI ** -0.5
K = 2048


def _rand(gen, *shape):
    return (torch.randn(*shape, device=DEV, dtype=torch.float32, generator=gen) * 0.5).to(torch.bfloat16)


def _inputs(S, seed, hq):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return (_rand(gen, 1, S, HI, DI), _rand(gen, 1, S, 1, DI), _rand(gen, 1, S, HI),
            _rand(gen, 1, S, hq, DQK), _rand(gen, 1, S, DQK), gen)


def _cu(doc_lens, total=None):
    b = [0]
    for d in doc_lens:
        b.append(b[-1] + d)
    if total is not None:
        b[-1] = total
    return torch.tensor(b, device=DEV, dtype=torch.int32)


def _run(iq, ik, iw, q_attn, kv, cu, need_grad=False, blk=None):
    """One selection+sparse pass; blk=(off,len) restricts to a rebalance block."""
    S = iq.shape[1]
    off, ln = blk if blk is not None else (0, S)
    sl = slice(off, off + ln)
    pos = torch.arange(off, off + ln, device=DEV)
    if need_grad:
        q_attn = q_attn.clone().requires_grad_(True)
        kv = kv.clone().requires_grad_(True)
    sel = OP.select_topk(iq[:, sl], ik, iw[:, sl], S_IDX, K, positions_q=pos,
                                   cu_seqlens=cu, q_max=off + ln - 1, q_min=off)
    out, kl = OP.dsa_sparse_attention(iq[:, sl], ik, iw[:, sl], q_attn[:, sl] if blk else q_attn,
                                      kv, positions_q=None, cu_seqlens=cu,
                                      softmax_scale=S_ATTN, index_scale=S_IDX, d_v=DV,
                                      topk=K, q_max=off + ln - 1, offset=off,
                                      selection=sel)
    return sel, out, kl, q_attn, kv


def _sets_equal(ids_a, ids_b, n_rows):
    a = ids_a[0, :n_rows].long().sort(-1).values
    b = ids_b[0, :n_rows].long().sort(-1).values
    return int((a != b).any(-1).sum())


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("hq", [HQ_FLASH, HQ_PADDED])
def test_short_docs_select_causal_prefix_and_match_dense_reference(hq):
    docs = [200, 100, 300, 424]
    S = sum(docs)
    iq, ik, iw, q_attn, kv, _ = _inputs(S, 1234, hq)
    cu = _cu(docs)
    sel, out, _, _, _ = _run(iq, ik, iw, q_attn, kv, cu)

    pos = torch.arange(S, device=DEV)
    doc = torch.bucketize(pos, cu[1:-1].to(pos.dtype), right=True)
    expect_len = (pos - cu[:-1].to(pos.dtype)[doc] + 1).to(torch.int32)
    assert torch.equal(sel[1], expect_len)

    q = q_attn[0].float()
    k = kv[0].float()
    scores = torch.einsum("shd,td->sht", q, k) * S_ATTN
    invalid = (pos[None, :] > pos[:, None]) | (doc[None, :] != doc[:, None])
    scores.masked_fill_(invalid[:, None, :], float("-inf"))
    ref = torch.einsum("sht,tv->shv", scores.softmax(-1), k[:, :DV]).unsqueeze(0)
    diff = (out.float() - ref).abs()
    assert diff.max() < 5e-2 and diff.mean() < 2e-3


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("hq", [HQ_FLASH, HQ_PADDED])
def test_pad_tail_merged_into_last_doc_leaves_real_rows_intact(hq):
    docs, pads = [2048, 1536, 2560], 1024
    S_r = sum(docs)
    S_p = S_r + pads
    iq, ik, iw, q_attn, kv, gen = _inputs(S_p, 4321, hq)
    cu_r, cu_p = _cu(docs), _cu(docs, total=S_p)

    sel_r, out_r, kl_r, qg_r, kvg_r = _run(
        iq[:, :S_r], ik[:, :S_r], iw[:, :S_r], q_attn[:, :S_r], kv[:, :S_r], cu_r, need_grad=True)
    sel_p, out_p, kl_p, qg_p, kvg_p = _run(iq, ik, iw, q_attn, kv, cu_p, need_grad=True)

    assert _sets_equal(sel_r[0], sel_p[0], S_r) == 0
    assert torch.equal(sel_r[1], sel_p[1][:S_r])
    out_diff = (out_p[:, :S_r].float() - out_r.float()).abs()
    # fused selection returns the top-k as an unsorted set, so the attention gather
    # order (and its bf16 summation) differs between the two runs
    max_tol, mean_tol = (5e-3, 1e-5)
    assert out_diff.max() < max_tol and out_diff.mean() < mean_tol

    dout = _rand(gen, 1, S_r, hq, DV)
    out_r.backward(dout)
    dout_p = torch.zeros(1, S_p, hq, DV, device=DEV, dtype=torch.bfloat16)
    dout_p[:, :S_r] = dout
    out_p.backward(dout_p)
    # order-noise baseline: same unpadded run with per-row shuffled top-k order
    ids, ln = sel_r
    r = torch.rand(ids.shape[1], ids.shape[2], device=DEV, generator=gen)
    r[ids[0] < 0] = 2.0
    ids_shuf = torch.gather(ids[0], 1, r.argsort(-1)).unsqueeze(0).contiguous()
    q_sh = q_attn[:, :S_r].clone().requires_grad_(True)
    kv_sh = kv[:, :S_r].clone().requires_grad_(True)
    out_sh, _ = OP.dsa_sparse_attention(
        iq[:, :S_r], ik[:, :S_r], iw[:, :S_r], q_sh, kv_sh, positions_q=None, cu_seqlens=cu_r,
        softmax_scale=S_ATTN, index_scale=S_IDX, d_v=DV, topk=K,
        q_max=S_r - 1, offset=0, selection=(ids_shuf, ln))
    out_sh.backward(dout)
    tol_q = max(4 * (q_sh.grad.float() - qg_r.grad.float()).abs().max().item(), 1e-6)
    tol_kv = max(4 * (kv_sh.grad.float() - kvg_r.grad.float()).abs().max().item(), 1e-6)

    assert (qg_p.grad[:, :S_r].float() - qg_r.grad.float()).abs().max() <= tol_q
    assert (kvg_p.grad[:, :S_r].float() - kvg_r.grad.float()).abs().max() <= tol_kv
    assert qg_p.grad[:, S_r:].abs().max() == 0, "pad rows must get no LM gradient"
    # pads DO contribute to the KL scalar (known, proportional to pad share)
    assert kl_p.detach() > kl_r.detach()


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("docs,pads", [
    ([2048, 1536, 2560], 1024),                                            # long docs
    ([300, 90, 47, 500, 61, 200, 830, 1024, 130, 70, 1900, 512, 240, 121], 1143),  # dialogs < one block
])
def test_rebalance_blocks_over_multidoc_cu_match_full_run(docs, pads):
    SP = 8
    S = sum(docs) + pads
    assert S % (SP * SP) == 0, "test geometry must satisfy the rebalance rule"
    S_loc, blk = S // SP, S // (SP * SP)
    iq, ik, iw, q_attn0, kv0, gen = _inputs(S, 4242, HQ_FLASH)
    cu = _cu(docs, total=S)
    dout = _rand(gen, 1, S, HQ_FLASH, DV)
    dout[:, sum(docs):] = 0

    sel_f, out_f, kl_f, qg_f, kvg_f = _run(iq, ik, iw, q_attn0, kv0, cu, need_grad=True)
    (out_f.float() * dout.float()).sum().backward()

    q_attn = q_attn0.clone().requires_grad_(True)
    kv = kv0.clone().requires_grad_(True)
    out_g = torch.zeros_like(out_f)
    ids_g = torch.full_like(sel_f[0], -2)
    len_g = torch.zeros_like(sel_f[1])
    kl_total, loss_terms = None, []
    for r in range(SP):
        for s in range(SP):
            off = s * S_loc + r * blk
            sl = slice(off, off + blk)
            pos = torch.arange(off, off + blk, device=DEV)
            sel = OP.select_topk(iq[:, sl], ik, iw[:, sl], S_IDX, K, positions_q=pos,
                                           cu_seqlens=cu, q_max=off + blk - 1, q_min=off)
            out_b, kl_b = OP.dsa_sparse_attention(
                iq[:, sl], ik, iw[:, sl], q_attn[:, sl], kv, positions_q=None, cu_seqlens=cu,
                softmax_scale=S_ATTN, index_scale=S_IDX, d_v=DV, topk=K,
                q_max=off + blk - 1, offset=off, selection=sel)
            out_g[:, sl] = out_b.detach()
            ids_g[:, sl] = sel[0]
            len_g[sl] = sel[1]
            kl_total = kl_b.detach() if kl_total is None else kl_total + kl_b.detach()
            loss_terms.append((out_b.float() * dout[:, sl].float()).sum())
    torch.stack(loss_terms).sum().backward()

    assert _sets_equal(sel_f[0], ids_g, S) == 0
    assert torch.equal(sel_f[1], len_g)
    out_tol = 5e-3   # unsorted fused ids: gather-order noise
    assert (out_f.float() - out_g.float()).abs().max() < out_tol
    assert abs(kl_f.detach().item() - kl_total.item()) < 0.01 * abs(kl_f.detach().item()) + 1e-3
    assert (qg_f.grad.float() - q_attn.grad.float()).abs().max() < 5e-3
    dkv_scale = kvg_f.grad.float().abs().max().clamp_min(1.0)
    assert (kvg_f.grad.float() - kv.grad.float()).abs().max() < 0.02 * dkv_scale


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("hq", [HQ_FLASH, HQ_PADDED])
def test_kl_excludes_pad_queries_when_real_len_given(hq):
    """q_real_max must (a) make the padded KL equal the unpadded KL, (b) zero the
    indexer gradients of pad query AND pad key rows, (c) keep real-row indexer
    grads equal to the unpadded run, and (d) leave the default path (None)
    polluting exactly as before."""
    docs, pads = [2048, 1536, 2560], 1024
    S_r = sum(docs)
    S_p = S_r + pads
    iq, ik, iw, q_attn, kv, _ = _inputs(S_p, 777, hq)
    cu_r, cu_p = _cu(docs), _cu(docs, total=S_p)

    def run(S, cu, q_real_max):
        iq_ = iq[:, :S].clone().requires_grad_(True)
        ik_ = ik[:, :S].clone().requires_grad_(True)
        iw_ = iw[:, :S].clone().requires_grad_(True)
        _, kl = OP.dsa_sparse_attention(
            iq_, ik_, iw_, q_attn[:, :S], kv[:, :S],
            positions_q=torch.arange(S, device=DEV), cu_seqlens=cu,
            softmax_scale=S_ATTN, index_scale=S_IDX, d_v=DV, topk=K,
            q_max=S - 1, q_real_max=q_real_max)
        kl.backward()
        return kl.detach(), iq_.grad, ik_.grad, iw_.grad

    kl_r, giq_r, gik_r, giw_r = run(S_r, cu_r, None)
    kl_m, giq_m, gik_m, giw_m = run(S_p, cu_p, torch.tensor(S_r, device=DEV))
    kl_u, _, _, _ = run(S_p, cu_p, None)

    # (a) masked padded KL == unpadded KL (same rows, same selected sets;
    # tolerance = kernel reduction-order noise on the wider tensors)
    assert abs(kl_m.item() - kl_r.item()) <= max(2e-3 * abs(kl_r.item()), 0.5)
    # (d) default path unchanged: pads still pollute without q_real_max
    assert kl_u.item() > kl_r.item()

    # (b) pad rows: zero indexer gradient for queries, weights and KEYS
    assert giq_m[:, S_r:].abs().max() == 0
    assert giw_m[:, S_r:].abs().max() == 0
    assert gik_m[:, S_r:].abs().max() == 0

    # (c) real rows match the unpadded run within kernel order-noise
    for gm, gr in ((giq_m, giq_r), (giw_m, giw_r), (gik_m, gik_r)):
        d = (gm[:, :S_r].float() - gr.float()).abs().max()
        assert d <= max(2e-2 * gr.float().abs().max(), 1e-6), f"indexer grad diff {d}"
