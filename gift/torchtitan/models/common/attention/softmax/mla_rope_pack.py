"""Pack MLA queries and keys while applying partial complex RoPE."""

import torch
import triton
import triton.language as tl


_ROWS = 16


@triton.jit
def _rotate(re, im, cos, sin):
    return tl.fma(re, cos, -im * sin), tl.fma(re, sin, im * cos)


@triton.jit
def _position_indices(
    positions,
    batch,
    token,
    active,
    cache_length,
    psb,
    pss,
    HAS_POSITIONS: tl.constexpr,
    ALLOW_NEGATIVE: tl.constexpr,
):
    if HAS_POSITIONS:
        position = tl.load(positions + batch * psb + token * pss, active, 0).to(
            tl.int64
        )
        if ALLOW_NEGATIVE:
            # A shared position row uses advanced indexing in the PyTorch path.
            position = tl.where(position < 0, position + cache_length, position)
    else:
        position = token
    valid = (position >= 0) & (position < cache_length)
    tl.device_assert(~active | valid, "MLA RoPE position outside the cache")
    return position, active & valid


@triton.jit
def _coefficients(cache, position, pair, active, cs0, cs1, ROPE: tl.constexpr):
    offset = position[:, None] * cs0 + pair[None, :] * cs1
    mask = active[:, None] & (pair[None, :] < ROPE // 2)
    cos = tl.load(cache + offset, mask, other=float("nan"))
    sin = tl.load(cache + offset + 1, mask, other=float("nan"))
    return cos, sin


@triton.jit
def _pack_rows(
    plain,
    rotary,
    out,
    cache,
    position,
    active,
    valid_positions,
    cs0,
    cs1,
    plain_stride,
    rotary_stride,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    PLAIN_TILE: tl.constexpr,
    PAIR_TILE: tl.constexpr,
    INVERSE: tl.constexpr,
):
    for tile in range(tl.cdiv(NOPE, PLAIN_TILE)):
        dim = tile * PLAIN_TILE + tl.arange(0, PLAIN_TILE)
        mask = active[:, None] & (dim[None, :] < NOPE)
        value = tl.load(plain[:, None] + dim[None, :] * plain_stride, mask, 0)
        tl.store(out[:, None] + dim[None, :], value, mask)

    for tile in range(tl.cdiv(ROPE // 2, PAIR_TILE)):
        pair = tile * PAIR_TILE + tl.arange(0, PAIR_TILE)
        cos, sin = _coefficients(cache, position, pair, valid_positions, cs0, cs1, ROPE)
        mask = active[:, None] & (pair[None, :] < ROPE // 2)
        base = rotary[:, None] + 2 * pair[None, :] * rotary_stride
        re = tl.load(base, mask, 0).to(tl.float32)
        im = tl.load(base + rotary_stride, mask, 0).to(tl.float32)
        if INVERSE:
            sin = -sin
        yr, yi = _rotate(re, im, cos, sin)
        dim = tile * (2 * PAIR_TILE) + tl.arange(0, 2 * PAIR_TILE)
        tl.store(
            out[:, None] + NOPE + dim[None, :],
            tl.interleave(yr, yi),
            active[:, None] & (dim[None, :] < ROPE),
        )


@triton.jit
def _mla_rope_pack(
    query,
    key,
    key_rope,
    cache,
    positions,
    out_q,
    out_k,
    seqlen,
    query_tiles,
    cache_length,
    psb,
    pss,
    cs0,
    cs1,
    qsb,
    qss,
    qsh,
    qsd,
    ksb,
    kss,
    ksh,
    ksd,
    rsb,
    rss,
    rsd,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROWS: tl.constexpr,
    PLAIN_TILE: tl.constexpr,
    PAIR_TILE: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    ALLOW_NEGATIVE: tl.constexpr,
):
    batch = (tl.program_id(0) // query_tiles).to(tl.int64)
    row = (tl.program_id(0) % query_tiles).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    token = row // HEADS
    head = row % HEADS
    active = token < seqlen
    position, valid_positions = _position_indices(
        positions,
        batch,
        token,
        active,
        cache_length,
        psb,
        pss,
        HAS_POSITIONS,
        ALLOW_NEGATIVE,
    )
    width: tl.constexpr = NOPE + ROPE
    dst = (batch * seqlen * HEADS + row) * width
    if tl.program_id(1) == 0:
        q = query + batch * qsb + token * qss + head * qsh
        _pack_rows(
            q,
            q + NOPE * qsd,
            out_q + dst,
            cache,
            position,
            active,
            valid_positions,
            cs0,
            cs1,
            qsd,
            qsd,
            NOPE,
            ROPE,
            PLAIN_TILE,
            PAIR_TILE,
            False,
        )
    else:
        k = key + batch * ksb + token * kss + head * ksh
        rope = key_rope + batch * rsb + token * rss
        _pack_rows(
            k,
            rope,
            out_k + dst,
            cache,
            position,
            active,
            valid_positions,
            cs0,
            cs1,
            ksd,
            rsd,
            NOPE,
            ROPE,
            PLAIN_TILE,
            PAIR_TILE,
            False,
        )


@triton.jit
def _mla_rope_unpack(
    grad_q,
    grad_k,
    cache,
    positions,
    dq,
    dk_rope,
    seqlen,
    query_tiles,
    cache_length,
    psb,
    pss,
    cs0,
    cs1,
    qsb,
    qss,
    qsh,
    qsd,
    ksb,
    kss,
    ksh,
    ksd,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROWS: tl.constexpr,
    PLAIN_TILE: tl.constexpr,
    PAIR_TILE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    ALLOW_NEGATIVE: tl.constexpr,
):
    tiles = query_tiles + seqlen
    batch = (tl.program_id(0) // tiles).to(tl.int64)
    tile = tl.program_id(0) % tiles
    width: tl.constexpr = NOPE + ROPE
    if tile < query_tiles:
        row = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        token = row // HEADS
        active = token < seqlen
        position, valid_positions = _position_indices(
            positions,
            batch,
            token,
            active,
            cache_length,
            psb,
            pss,
            HAS_POSITIONS,
            ALLOW_NEGATIVE,
        )
        offset = (batch * seqlen * HEADS + row) * width
        src = grad_q + batch * qsb + token * qss + (row % HEADS) * qsh
        _pack_rows(
            src,
            src + NOPE * qsd,
            dq + offset,
            cache,
            position,
            active,
            valid_positions,
            cs0,
            cs1,
            qsd,
            qsd,
            NOPE,
            ROPE,
            PLAIN_TILE,
            PAIR_TILE,
            True,
        )
    else:
        key_token = (tile - query_tiles).to(tl.int64)
        key_position, key_valid_positions = _position_indices(
            positions,
            batch,
            key_token + tl.arange(0, 1),
            key_token < seqlen,
            cache_length,
            psb,
            pss,
            HAS_POSITIONS,
            ALLOW_NEGATIVE,
        )
        flat_token = batch * seqlen + key_token
        for pair_tile in range(tl.cdiv(ROPE // 2, PAIR_TILE)):
            pair = pair_tile * PAIR_TILE + tl.arange(0, PAIR_TILE)
            dim = pair_tile * (2 * PAIR_TILE) + tl.arange(0, 2 * PAIR_TILE)
            grad = tl.full((HEAD_TILE, 2 * PAIR_TILE), 0, tl.float32)
            for head_tile in range(tl.cdiv(HEADS, HEAD_TILE)):
                head = head_tile * HEAD_TILE + tl.arange(0, HEAD_TILE)
                key_offset = batch * ksb + key_token * kss + head[:, None] * ksh
                key_offset += (NOPE + dim[None, :]) * ksd
                mask = (head[:, None] < HEADS) & (dim[None, :] < ROPE)
                grad += tl.load(grad_k + key_offset, mask, 0).to(tl.float32)
            # Keep the shared-key sum in FP32 through the inverse rotation.
            re, im = tl.split(tl.reshape(tl.sum(grad, 0), (PAIR_TILE, 2)))
            cos, sin = _coefficients(
                cache, key_position, pair, key_valid_positions, cs0, cs1, ROPE
            )
            yr, yi = _rotate(
                re, im, tl.reshape(cos, (PAIR_TILE,)), -tl.reshape(sin, (PAIR_TILE,))
            )
            tl.store(
                dk_rope + flat_token * ROPE + dim, tl.interleave(yr, yi), dim < ROPE
            )


def _position_strides(positions: torch.Tensor | None) -> tuple[int, int, bool]:
    if positions is None:
        return 0, 0, False
    shared = positions.shape[0] == 1
    return 0 if shared else positions.stride(0), positions.stride(1), shared


class MLARoPEPack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, key_rope, cache, positions):
        # Triton sees storage pointers, so resolve lazy tensor views first.
        query, key, key_rope = (
            query.resolve_neg(),
            key.resolve_neg(),
            key_rope.resolve_neg(),
        )
        cache = torch.view_as_real(cache.resolve_conj().resolve_neg())
        if positions is not None:
            positions = positions.resolve_neg()
        batch, seqlen, heads, width = query.shape
        nope = key.shape[-1]
        rope = width - nope
        query_tiles = triton.cdiv(seqlen * heads, _ROWS)
        psb, pss, shared = _position_strides(positions)
        out_q, out_k = query.new_empty(query.shape), query.new_empty(query.shape)
        _mla_rope_pack[(batch * query_tiles, 2)](
            query,
            key,
            key_rope,
            cache,
            positions,
            out_q,
            out_k,
            seqlen,
            query_tiles,
            cache.shape[0],
            psb,
            pss,
            *cache.stride()[:2],
            *query.stride(),
            *key.stride(),
            *key_rope.stride(),
            HEADS=heads,
            NOPE=nope,
            ROPE=rope,
            ROWS=_ROWS,
            PLAIN_TILE=min(128, triton.next_power_of_2(max(1, nope))),
            PAIR_TILE=min(32, triton.next_power_of_2(rope // 2)),
            HAS_POSITIONS=positions is not None,
            ALLOW_NEGATIVE=shared,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
        ctx.save_for_backward(cache, positions)
        ctx.nope = nope
        return out_q, out_k

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cache, positions = ctx.saved_tensors
        grad_q, grad_k = grad_q.resolve_neg(), grad_k.resolve_neg()
        batch, seqlen, heads, width = grad_q.shape
        nope = ctx.nope
        rope = width - nope
        query_tiles = triton.cdiv(seqlen * heads, _ROWS)
        psb, pss, shared = _position_strides(positions)
        dq = grad_q.new_empty(grad_q.shape)
        dk_rope = grad_q.new_empty((batch, seqlen, rope))
        _mla_rope_unpack[(batch * (query_tiles + seqlen),)](
            grad_q,
            grad_k,
            cache,
            positions,
            dq,
            dk_rope,
            seqlen,
            query_tiles,
            cache.shape[0],
            psb,
            pss,
            *cache.stride()[:2],
            *grad_q.stride(),
            *grad_k.stride(),
            HEADS=heads,
            NOPE=nope,
            ROPE=rope,
            ROWS=_ROWS,
            PLAIN_TILE=min(128, triton.next_power_of_2(max(1, nope))),
            PAIR_TILE=min(32, triton.next_power_of_2(rope // 2)),
            HEAD_TILE=min(128, triton.next_power_of_2(heads)),
            HAS_POSITIONS=positions is not None,
            ALLOW_NEGATIVE=shared,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
        return dq, grad_k[..., :nope], dk_rope, None, None
