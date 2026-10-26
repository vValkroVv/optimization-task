"""Vendored long-sequence rotary kernel: VALUES past the upstream grid limit.

Upstream puts sequence blocks on grid axis y (max 65535), so above 65535*BLOCK_M = 524 280
tokens it cannot launch; the vendored kernel moves them to axis x. The existing test only
checks that it runs at 512k/1M and matches torch at 4k, i.e. never a value from a block index
>= 65535. Here every variant is compared with torch at S = 600 000.
"""
import pytest

import torch
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
from llmfoundry.models.layers.flash_rotary_longseq import apply_rotary  # noqa: E402

DEV = "cuda"
S = 600_000          # > 524280
H, D, RD = 1, 64, 64


def torch_rotary(x, cos, sin, pos, interleaved, conjugate):
    """x (S,H,D) fp32, cos/sin (seqlen_ro, RD/2), pos (S,) absolute positions into cos/sin."""
    c, s = cos[pos].unsqueeze(1), sin[pos].unsqueeze(1)          # (S,1,RD/2)
    if conjugate:
        s = -s
    xr, xp = x[..., :RD], x[..., RD:]
    if interleaved:
        x0, x1 = xr[..., 0::2], xr[..., 1::2]
        o = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)
    else:
        x0, x1 = xr[..., : RD // 2], xr[..., RD // 2:]
        o = torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)
    return torch.cat([o, xp], dim=-1)


def _cos_sin(seqlen_ro, seed=0):
    inv = 1.0 / (10000.0 ** (torch.arange(0, RD, 2, device=DEV).float() / RD))
    ang = torch.arange(seqlen_ro, device=DEV).float()[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("conjugate", [False, True])
def test_batched_values_past_65535_blocks(interleaved, conjugate):
    x = torch.randn(1, S, H, D, device=DEV)
    cos, sin = _cos_sin(S)
    out = apply_rotary(x, cos, sin, interleaved=interleaved, conjugate=conjugate)
    ref = torch_rotary(x[0], cos, sin, torch.arange(S, device=DEV), interleaved, conjugate)
    torch.testing.assert_close(out[0], ref, rtol=1e-5, atol=1e-5)
    # the tail of the sequence is exactly what an axis-y grid could not address
    torch.testing.assert_close(out[0, -4096:], ref[-4096:], rtol=1e-5, atol=1e-5)


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_varlen_two_documents_with_tensor_offsets():
    docs = (0, 350_000, S)
    cu = torch.tensor(docs, device=DEV, dtype=torch.int32)
    offsets = torch.tensor([0, 1000], device=DEV, dtype=torch.int32)   # per-document position offset
    max_len = max(b - a for a, b in zip(docs[:-1], docs[1:]))
    cos, sin = _cos_sin(max_len + 1000)
    x = torch.randn(S, H, D, device=DEV)
    out = apply_rotary(x, cos, sin, seqlen_offsets=offsets, cu_seqlens=cu, max_seqlen=max_len, interleaved=False)
    pos = torch.cat([torch.arange(b - a, device=DEV) + int(offsets[i]) for i, (a, b) in enumerate(zip(docs[:-1], docs[1:]))])
    ref = torch_rotary(x, cos, sin, pos, interleaved=False, conjugate=False)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_partial_rotary_dim_keeps_pass_through_half_and_inplace():
    """rotary_dim < headdim: the untouched half must be copied (out-of-place) or left as is (inplace)."""
    x = torch.randn(1, S, H, 2 * RD, device=DEV)
    cos, sin = _cos_sin(S)
    out = apply_rotary(x, cos, sin, interleaved=False)
    assert torch.equal(out[..., RD:], x[..., RD:])
    ref = torch_rotary(x[0], cos, sin, torch.arange(S, device=DEV), False, False)
    torch.testing.assert_close(out[0], ref, rtol=1e-5, atol=1e-5)
    x2 = x.clone()
    out2 = apply_rotary(x2, cos, sin, interleaved=False, inplace=True)
    assert out2.data_ptr() == x2.data_ptr()
    torch.testing.assert_close(x2[0], ref, rtol=1e-5, atol=1e-5)
