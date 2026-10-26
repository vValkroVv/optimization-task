"""Fused selection recall against the intended unrounded fp32 indexer formula.

The stricter cutoff check against bf16-scaled queries is in test_dsa_topk_selection.
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
HI, DI = 64, 128


def torch_scores(iq, ik, w, scale):
    """(Sq, Sk) fp32 index scores from bf16 inputs, fp32 math."""
    ph = F.relu(torch.einsum("shd,kd->shk", iq.float(), ik.float()))
    return (ph * w.float().unsqueeze(-1)).sum(1) * scale


def _inputs(sq, sk, seed=0, w_scale=0.5):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *sh, s=0.5: (torch.randn(*sh, device=DEV, generator=g) * s).bfloat16()
    return r(1, sq, HI, DI), r(1, sk, 1, DI), r(1, sq, HI, s=w_scale)


def _recall(ids, lengths, ref_ids, ref_valid):
    rec = []
    for r in range(ids.shape[0]):
        got = ids[r, : int(lengths[r])].long()
        exp = ref_ids[r][ref_valid[r]]
        rec.append(torch.isin(got, exp).sum().item() / max(1, exp.numel()))
        assert int(lengths[r]) == exp.numel(), f"row {r}: {int(lengths[r])} selected vs {exp.numel()} valid keys"
    return torch.tensor(rec)


class TestFusedSelectionVsTorch:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("docs", [None, [1000, 600, 2496]])
    def test_recall_vs_exact_topk(self, docs):
        sq, sk, k = 64, 4096, 256
        iq, ik, w = _inputs(sq, sk)
        scale = DI ** -0.5
        pos = torch.arange(sk - sq, sk, device=DEV)
        cu = (torch.tensor([0, *torch.tensor(docs).cumsum(0).tolist()], device=DEV, dtype=torch.int32)
              if docs else None)
        ids, ln = OP.select_topk(iq, ik, w, scale, k, positions_q=pos, cu_seqlens=cu)
        sc = torch_scores(iq[0], ik[0, :, 0], w[0], scale)
        keys = torch.arange(sk, device=DEV)
        valid = keys[None, :] <= pos[:, None]
        if cu is not None:
            b = cu[1:-1]
            valid &= torch.bucketize(pos, b, right=True)[:, None] == torch.bucketize(keys, b, right=True)[None, :]
        sc = sc.masked_fill(~valid, float("-inf"))
        ref_v, ref_i = torch.topk(sc, k, dim=-1)
        rec = _recall(ids[0], ln, ref_i, torch.isfinite(ref_v))
        assert rec.mean() > 0.99 and rec.min() > 0.95, f"recall mean={rec.mean():.4f} min={rec.min():.4f}"
