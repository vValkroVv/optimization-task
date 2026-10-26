"""KL value/scale and separation of the LM and indexer gradient paths."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("DSA kernels require Hopper", allow_module_level=True)
from llmfoundry.models.layers import dsa_operator as OP

pytestmark = [pytest.mark.unit, pytest.mark.requires_gpu]


@pytest.mark.parametrize("heads", [32, 64, 128])
def test_kl_value_scale_and_gradient_separation(heads):
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _check_contract(heads)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def _check_contract(heads):
    s, k = 64, 256  # every row has unused selection slots
    g = torch.Generator("cuda").manual_seed(27)

    def r(*shape, scale=0.3):
        return (
            (torch.randn(shape, device="cuda", generator=g) * scale)
            .bfloat16()
            .requires_grad_()
        )

    iq, ik, w = r(1, s, 64, 128), r(1, s, 1, 128), r(1, s, 64, scale=0.1)
    q, kv = r(1, s, heads, 576), r(1, s, 576)
    pos = torch.arange(s, device="cuda")
    ids, lengths = OP.select_topk(iq, ik, w, 128**-0.5, k, pos)
    safe = ids[0].clamp_min(0).long()
    valid = torch.arange(k, device="cuda")[None] < lengths[:, None]
    with torch.no_grad():
        keys = kv[0].float()[safe]
        logits = torch.einsum("shd,skd->shk", q[0].float(), keys) * 576**-0.5
        probs = logits.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        target = probs.sum(1)
        target = target / target.sum(-1, keepdim=True)
    riq = iq[0].detach().float().requires_grad_()
    rik = ik[0, :, 0].detach().float().requires_grad_()
    rw = w[0].detach().float().requires_grad_()
    index_logits = (
        torch.einsum("shd,skd->shk", riq, rik[safe]).relu() * rw.unsqueeze(-1)
    ).sum(1) * 128**-0.5
    logp = index_logits.masked_fill(~valid, -torch.inf).log_softmax(-1)
    predict = logp.exp()
    # This case does not activate the API's probability clamp. Pin that assumption.
    assert (predict[valid] > 1e-9).all() and (target[valid] > 1e-9).all()
    ref_kl = (target * (target.clamp_min(1e-9).log() - logp).nan_to_num(0.0)).sum()
    ref_grads = torch.autograd.grad(ref_kl * 0.37, (riq, rik, rw))

    def run():
        return OP.dsa_sparse_attention(
            iq,
            ik,
            w,
            q,
            kv,
            None,
            None,
            576**-0.5,
            128**-0.5,
            512,
            k,
            selection=(ids, lengths),
        )

    _, kl = run()
    torch.testing.assert_close(kl, ref_kl, rtol=0.01, atol=2e-5)
    (kl * 0.37).backward()
    assert q.grad is None and kv.grad is None, "KL must not train the attention target"
    for got, ref in zip((iq.grad[0], ik.grad[0, :, 0], w.grad[0]), ref_grads):
        err = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
        assert err < 0.01, f"non-unit KL upstream gradient changed: rel={float(err)}"
    for x in (iq, ik, w, q, kv):
        x.grad = None
    out, _ = run()
    out.float().square().mean().backward()
    assert iq.grad is None and ik.grad is None and w.grad is None
    assert q.grad is not None and kv.grad is not None
