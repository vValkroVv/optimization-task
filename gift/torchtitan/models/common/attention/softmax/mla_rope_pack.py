"""Fused partial RoPE and Q/K packing for the compiled MLA crop."""

import torch
import triton
import triton.language as tl


_SHAPE = (2, 4096, 32, 192)
_ROWS = 16
_TOKENS = _SHAPE[0] * _SHAPE[1]
_QUERY_TILES = _TOKENS * _SHAPE[2] // _ROWS


@triton.jit
def _rotate(re, im, cos, sin):
    return tl.fma(re, cos, -im * sin), tl.fma(re, sin, im * cos)


@triton.jit
def _coefficients(cache, positions, token, pair, CACHE_LENGTH: tl.constexpr):
    position = tl.load(positions + token)
    valid = (position >= 0) & (position < CACHE_LENGTH)
    tl.device_assert(valid, "MLA RoPE position outside the cache")
    offset = position[:, None] * 64 + 2 * pair[None, :]
    cos = tl.load(cache + offset, mask=valid[:, None], other=float("nan"))
    sin = tl.load(cache + offset + 1, mask=valid[:, None], other=float("nan"))
    return cos, sin


@triton.jit
def _mla_rope_pack(
    query, key, key_rope, cache, positions, out_q, out_k, ROWS: tl.constexpr
):
    # The fixed input views span at most 2 * 4096 * 32 * 256 elements.
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    token = row // 32
    head = row % 32
    plain = tl.arange(0, 128)
    pair = tl.arange(0, 32)
    cos, sin = _coefficients(cache, positions, token, pair, 4096)
    if tl.program_id(1) == 0:
        val = tl.load(query + row[:, None] * 192 + plain[None, :])
        rotary = query + row[:, None] * 192 + 128 + 2 * pair[None, :]
        out = out_q
    else:
        val = tl.load(key + (token * 8192 + head * 256)[:, None] + plain[None, :])
        rotary = key_rope + token[:, None] * 576 + 2 * pair[None, :]
        out = out_k
    re = tl.load(rotary).to(tl.float32)
    im = tl.load(rotary + 1).to(tl.float32)
    yr, yi = _rotate(re, im, cos, sin)
    tl.store(out + row[:, None] * 192 + plain[None, :], val)
    tl.store(
        out + row[:, None] * 192 + 128 + tl.arange(0, 64)[None, :],
        tl.interleave(yr, yi),
    )


@triton.jit
def _mla_rope_unpack(
    grad_q,
    grad_k,
    cache,
    positions,
    dq,
    dk_rope,
    ROWS: tl.constexpr,
    QUERY_TILES: tl.constexpr,
):
    if tl.program_id(0) < QUERY_TILES:
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        plain = tl.arange(0, 128)
        pair = tl.arange(0, 32)
        cos, sin = _coefficients(cache, positions, row // 32, pair, 4096)
        val = tl.load(grad_q + row[:, None] * 192 + plain[None, :])
        rotary = grad_q + row[:, None] * 192 + 128 + 2 * pair[None, :]
        re = tl.load(rotary).to(tl.float32)
        im = tl.load(rotary + 1).to(tl.float32)
        yr, yi = _rotate(re, im, cos, -sin)
        tl.store(dq + row[:, None] * 192 + plain[None, :], val)
        tl.store(
            dq + row[:, None] * 192 + 128 + tl.arange(0, 64)[None, :],
            tl.interleave(yr, yi),
        )
    else:
        token = tl.program_id(0) - QUERY_TILES
        head = tl.arange(0, 32)
        dim = tl.arange(0, 64)
        key_rotary = grad_k + token * 6144 + head[:, None] * 192 + 128 + dim[None, :]
        # The compiled expand backward keeps the shared-K head sum in FP32
        # through the inverse rotation, rounding only on the final store.
        grad = tl.sum(tl.load(key_rotary).to(tl.float32), axis=0)
        key_re, key_im = tl.split(tl.reshape(grad, (32, 2)))
        key_cos, key_sin = _coefficients(
            cache, positions, token + tl.arange(0, 1), tl.arange(0, 32), 4096
        )
        key_real, key_imag = _rotate(
            key_re,
            key_im,
            tl.reshape(key_cos, (32,)),
            -tl.reshape(key_sin, (32,)),
        )
        tl.store(dk_rope + token * 64 + dim, tl.interleave(key_real, key_imag))


class MLARoPEPack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, key_rope, cache, positions):
        out_q, out_k = torch.empty_like(query), torch.empty_like(query)
        cache = torch.view_as_real(cache)
        _mla_rope_pack[(_QUERY_TILES, 2)](
            query,
            key,
            key_rope,
            cache,
            positions,
            out_q,
            out_k,
            ROWS=_ROWS,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
        ctx.save_for_backward(cache, positions)
        return out_q, out_k

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cache, positions = ctx.saved_tensors
        grad_q = grad_q.contiguous()
        grad_k = grad_k.contiguous()
        dq = torch.empty_like(grad_q)
        dk_rope = grad_q.new_empty((*_SHAPE[:2], 64))
        _mla_rope_unpack[(_QUERY_TILES + _TOKENS,)](
            grad_q,
            grad_k,
            cache,
            positions,
            dq,
            dk_rope,
            ROWS=_ROWS,
            QUERY_TILES=_QUERY_TILES,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
        return dq, grad_k[..., :128], dk_rope, None, None
