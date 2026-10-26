"""Adversarial tests for the fp8 blockwise all-gather quantization path.

The property under test is the whole point of the feature:

    all-gathering fp8 shards that were each quantized against a GLOBAL 128x128
    block scale must be bit-identical to quantizing the full bf16 weight once on
    a single rank.

Everything here runs in one process and simulates ``world_size`` ranks by
slicing a flat buffer, because that is exactly what GigaFSDP does: shard r owns
``flat[r * shard_size : (r + 1) * shard_size]``.

Run:
    cd contrib/gigafsdp
    python -m pytest tests/test_fp8_blockwise_ag.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from giga_fsdp.float8_utils import (  # noqa: E402
    FP8_TILE_SIZE,
    compute_local_tile_amaxes,
    fold_and_quantize_param,
    fp8_param_shape_is_supported,
    is_expert_param,
)

TILE = FP8_TILE_SIZE
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch, deliberately independent of Triton)
# ---------------------------------------------------------------------------

def ref_block_scales(w: torch.Tensor) -> torch.Tensor:
    """128x128 block scales of a [E, N, K] (or [N, K]) tensor.

    scale = 2 ** ceil(log2(amax / fp8_max)), computed in float64 so the
    reference never inherits the GPU's approximate log2.
    """
    if w.dim() == 2:
        w = w.unsqueeze(0)
    E, N, K = w.shape
    blocks = w.float().reshape(E, N // TILE, TILE, K // TILE, TILE)
    amax = blocks.abs().amax(dim=(2, 4)).double()          # [E, N/128, K/128]
    # Mirrors the kernel exactly: tl.maximum(block_amax, 1e-12) then pow2-ceil.
    # amax == 0 falls out of the same expression, no special case needed.
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp(min=1e-12) / FP8_MAX)))
    return scale.float()


def ref_quantize(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize [E, N, K] with a [E, N/128, K/128] block scale."""
    squeeze = w.dim() == 2
    if squeeze:
        w = w.unsqueeze(0)
    E, N, K = w.shape
    s = scale.reshape(E, N // TILE, 1, K // TILE, 1)
    q = (w.float().reshape(E, N // TILE, TILE, K // TILE, TILE) / s)
    q = q.reshape(E, N, K).to(torch.float8_e4m3fn)
    return q.squeeze(0) if squeeze else q


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    squeeze = q.dim() == 2
    if squeeze:
        q = q.unsqueeze(0)
        scale = scale.unsqueeze(0) if scale.dim() == 2 else scale
    E, N, K = q.shape
    s = scale.reshape(E, N // TILE, 1, K // TILE, 1)
    out = (q.float().reshape(E, N // TILE, TILE, K // TILE, TILE) * s).reshape(E, N, K)
    return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Rank simulation — mirrors GigaFSDP's flat-shard layout exactly
# ---------------------------------------------------------------------------

class ParamSpec:
    """One fp8 parameter inside a SuperTensor."""

    def __init__(self, shape, elem_offset):
        self.shape = tuple(shape)
        self.elem_offset = elem_offset
        self.n_experts = shape[0] if len(shape) == 3 else 1
        self.n_row_elem = shape[-2]
        self.n_cols = shape[-1]

    @property
    def numel(self):
        return self.n_experts * self.n_row_elem * self.n_cols

    @property
    def n_blocks(self):
        return self.n_experts * (self.n_row_elem // TILE) * (self.n_cols // TILE)


def simulate(flat, params, world_size, packed_scale_stride=None, scale_block_offset=0):
    """Run the full amax -> all-gather -> fold+quantize -> all-gather pipeline.

    ``flat`` is the padded flat buffer (numel % (world_size * 128) == 0).
    Returns (gathered_fp8, [per-param block scales from rank 0], per_rank_scales).
    """
    assert flat.numel() % (world_size * TILE) == 0
    shard_size = flat.numel() // world_size
    local_blocks = shard_size // TILE
    if packed_scale_stride is None:
        packed_scale_stride = local_blocks

    shards = [flat[r * shard_size:(r + 1) * shard_size].contiguous() for r in range(world_size)]

    # amax -> all-gather (rank-major packed layout)
    gathered = torch.zeros(world_size * packed_scale_stride, dtype=torch.float32,
                           device=flat.device)
    for r, sh in enumerate(shards):
        base = r * packed_scale_stride + scale_block_offset
        gathered[base:base + local_blocks] = compute_local_tile_amaxes(sh)

    # fold + quantize, independently on every rank
    per_rank_fp8, per_rank_scales = [], []
    for r, sh in enumerate(shards):
        out_fp8 = torch.zeros(shard_size, dtype=torch.float8_e4m3fn, device=flat.device)
        scales = []
        for p in params:
            bs = torch.empty(p.n_blocks, dtype=torch.float32, device=flat.device)
            fold_and_quantize_param(
                gathered_scales=gathered,
                scale_block_offset=scale_block_offset,
                packed_scale_stride=packed_scale_stride,
                local_scale_blocks=local_blocks,
                shard=sh,
                out_fp8=out_fp8,
                out_block_scale=bs,
                n_experts=p.n_experts,
                n_row_elem=p.n_row_elem,
                n_cols=p.n_cols,
                elem_offset=p.elem_offset,
                tile_offset=p.elem_offset // TILE,
                rank=r,
                world_size=world_size,
            )
            scales.append(bs)
        per_rank_fp8.append(out_fp8)
        per_rank_scales.append(scales)

    gathered_fp8 = torch.cat(per_rank_fp8)
    return gathered_fp8, per_rank_scales


def build_flat(params, world_size, device, fill=None):
    """Allocate a padded flat buffer holding ``params`` back to back."""
    total = max(p.elem_offset + p.numel for p in params)
    align = world_size * TILE
    padded = ((total + align - 1) // align) * align
    flat = torch.zeros(padded, dtype=torch.bfloat16, device=device)
    for p in params:
        w = fill(p) if fill is not None else torch.randn(p.numel, device=device)
        flat[p.elem_offset:p.elem_offset + p.numel] = w.reshape(-1).to(torch.bfloat16)
    return flat, padded, total


# ---------------------------------------------------------------------------
# 1. THE core invariant: no double quantization
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("shape", [
    (2, 256, 128),      # 3-D stacked experts
    (4, 256, 256),
    (1, 128, 128),      # single 128x128 block
    (512, 256),         # plain 2-D
])
def test_matches_single_rank_quantization_bitwise(world_size, shape):
    """Sharded+gathered fp8 must be BIT-IDENTICAL to one-shot quantization.

    This is the property the whole feature exists to provide.  Any deviation
    means an element was quantized against a scale derived from something other
    than its true global 128x128 block.
    """
    dev = torch.device("cuda")
    torch.manual_seed(0)
    p = ParamSpec(shape, elem_offset=0)
    flat, padded, total = build_flat([p], world_size, dev)
    if padded % (world_size * TILE) or (padded // world_size) % TILE:
        pytest.skip("shape/world_size combination not shard-aligned")

    got_fp8, per_rank_scales = simulate(flat, [p], world_size)

    w = flat[:total].reshape(shape)
    ref_s = ref_block_scales(w)
    ref_q = ref_quantize(w, ref_s)

    got_q = got_fp8[:total].reshape(shape)
    assert torch.equal(got_q.view(torch.int8), ref_q.view(torch.int8)), (
        f"fp8 payload differs from single-rank quantization: "
        f"{(got_q.view(torch.int8) != ref_q.view(torch.int8)).sum().item()} elements"
    )
    assert torch.equal(per_rank_scales[0][0].reshape(ref_s.shape), ref_s), \
        "block scales differ from the single-rank reference"


# ---------------------------------------------------------------------------
# 2. Cross-rank scale agreement — a mismatch here diverges the model silently
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_all_ranks_compute_identical_scales(world_size):
    """Every rank redundantly computes the FULL scale grid; all must agree bitwise.

    The scale grid is never all-gathered, so if ranks disagree the GEMMs on
    different ranks dequantize with different factors and the run diverges with
    no error anywhere.
    """
    dev = torch.device("cuda")
    torch.manual_seed(1)
    p = ParamSpec((4, 256, 256), elem_offset=0)
    flat, _, _ = build_flat([p], world_size, dev)

    _, per_rank_scales = simulate(flat, [p], world_size)
    ref = per_rank_scales[0][0]
    for r in range(1, world_size):
        assert torch.equal(per_rank_scales[r][0].view(torch.int32), ref.view(torch.int32)), \
            f"rank {r} scale grid differs from rank 0"


# ---------------------------------------------------------------------------
# 3. THE money test: no fp8 overflow. Targets approximate log2/exp2.
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("k", [-8, -4, -1, 0, 1, 4, 8])
@pytest.mark.parametrize("nudge", [0.0, 1e-6, -1e-6, 1e-3, -1e-3])
def test_no_overflow_when_amax_straddles_a_power_of_two(k, nudge):
    """scale = 2**ceil(log2(amax / 448)) must guarantee |w / scale| <= 448.

    tl.log2 / tl.exp2 lower to the approximate SFU instructions.  If log2
    returns k - eps for an amax marginally above 448 * 2**k, ceil yields k
    instead of k+1, scale is 2x too small, and w/scale exceeds fp8 max.  There
    is no clamp in the kernel (float8_utils.py:171), so the result depends on
    whether the fp8 downcast saturates.

    A scale that is 2x too small shows up as ~50% relative error against amax;
    correct rounding keeps it under the e4m3 quantization floor (~4%).
    """
    dev = torch.device("cuda")
    amax = FP8_MAX * (2.0 ** k) * (1.0 + nudge)
    shape = (1, 128, 128)
    p = ParamSpec(shape, elem_offset=0)

    def fill(spec):
        w = torch.zeros(spec.numel, device=dev, dtype=torch.float32)
        w[0] = amax          # exactly one element at the block amax
        w[1] = -amax / 3.0
        return w

    flat, _, total = build_flat([p], 1, dev, fill=fill)
    got_fp8, per_rank_scales = simulate(flat, [p], world_size=1)

    q = got_fp8[:total].reshape(shape)
    scale = per_rank_scales[0][0].reshape(1, 1, 1)

    qf = q.float()
    assert torch.isfinite(qf).all(), "fp8 payload contains Inf/NaN — downcast did not saturate"
    assert qf.abs().max().item() <= FP8_MAX, (
        f"|q| = {qf.abs().max().item()} exceeds fp8 max {FP8_MAX}: scale too small"
    )

    deq = dequantize(q, scale)
    w = flat[:total].reshape(shape).float()
    block_amax = w.abs().max().item()
    err = (deq - w).abs().max().item() / max(block_amax, 1e-30)
    assert err < 0.05, (
        f"relative error {err:.4f} vs block amax — scale is wrong by a factor of two "
        f"(amax={amax:.6g}, scale={scale.item():.6g}, expected>={amax / FP8_MAX:.6g})"
    )
    assert scale.item() * FP8_MAX >= block_amax * (1 - 1e-6), (
        f"scale {scale.item():.6g} cannot represent block amax {block_amax:.6g}"
    )


@cuda_only
def test_scale_is_an_exact_power_of_two():
    """A pow2 scale keeps dequantization exact and makes cross-rank agreement
    insensitive to the last mantissa bit.  Verify no scale has a mantissa."""
    dev = torch.device("cuda")
    torch.manual_seed(2)
    p = ParamSpec((4, 256, 256), elem_offset=0)
    flat, _, _ = build_flat([p], 4, dev)
    _, per_rank_scales = simulate(flat, [p], world_size=4)

    s = per_rank_scales[0][0]
    mant, _ = torch.frexp(s)
    assert torch.all((mant == 0.5) | (mant == 0.0)), \
        f"non-power-of-two scale present: {s[(mant != 0.5) & (mant != 0.0)][:8]}"


# ---------------------------------------------------------------------------
# 4. Alignment padding — the case that used to crash _fp8_all_gather
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("world_size", [2, 4])
def test_padded_buffer_is_untouched_and_weights_still_correct(world_size):
    """padded_numel > total_numel: the tail belongs to no parameter.

    The kernel only writes elements inside a parameter's range, so the padding
    must stay exactly as allocated (zeros), and the real weights must still
    match the single-rank reference.
    """
    dev = torch.device("cuda")
    torch.manual_seed(3)
    # 3 * 128 * 128 = 49152 elements; with world_size=4 the align is 512, and
    # we offset the param so total is deliberately NOT a multiple of ws * 128.
    shape = (3, 128, 128)
    p = ParamSpec(shape, elem_offset=0)
    align = world_size * TILE
    total = p.numel
    padded = ((total + align - 1) // align) * align
    if padded == total:                       # force a real padding tail
        padded += align
    flat = torch.zeros(padded, dtype=torch.bfloat16, device=dev)
    flat[:total] = torch.randn(total, device=dev).to(torch.bfloat16)

    got_fp8, per_rank_scales = simulate(flat, [p], world_size)

    assert got_fp8.numel() == padded
    tail = got_fp8[total:].view(torch.int8)
    assert torch.all(tail == 0), \
        f"{(tail != 0).sum().item()} padding bytes were written by the kernel"

    w = flat[:total].reshape(shape)
    ref_q = ref_quantize(w, ref_block_scales(w))
    assert torch.equal(got_fp8[:total].reshape(shape).view(torch.int8), ref_q.view(torch.int8))


# ---------------------------------------------------------------------------
# 5. Multi-parameter packing: elem_offset, tile_offset, scale_block_offset
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("world_size", [2, 4])
def test_two_params_in_one_supertensor(world_size):
    """w1 and w2 live back to back in one flat buffer with different offsets.

    Exercises elem_offset / tile_offset: an off-by-one there silently quantizes
    w2 against w1's scales.
    """
    dev = torch.device("cuda")
    torch.manual_seed(4)
    w1 = ParamSpec((2, 256, 128), elem_offset=0)
    w2 = ParamSpec((2, 128, 128), elem_offset=w1.numel)
    params = [w1, w2]
    flat, padded, total = build_flat(params, world_size, dev)

    got_fp8, per_rank_scales = simulate(flat, params, world_size)

    for p, bs in zip(params, per_rank_scales[0]):
        w = flat[p.elem_offset:p.elem_offset + p.numel].reshape(p.shape)
        ref_s = ref_block_scales(w)
        ref_q = ref_quantize(w, ref_s)
        got = got_fp8[p.elem_offset:p.elem_offset + p.numel].reshape(p.shape)
        assert torch.equal(got.view(torch.int8), ref_q.view(torch.int8)), \
            f"payload mismatch for param at offset {p.elem_offset}"
        assert torch.equal(bs.reshape(ref_s.shape), ref_s), \
            f"scale mismatch for param at offset {p.elem_offset}"


@cuda_only
def test_packed_amax_layout_with_scale_block_offset():
    """Several SuperTensors share one gathered amax buffer.

    ST #1 sits at scale_block_offset > 0 with packed_scale_stride spanning both.
    A wrong offset reads another SuperTensor's amaxes — which produces plausible
    but wrong scales, never an error.
    """
    dev = torch.device("cuda")
    torch.manual_seed(5)
    world_size = 2
    p = ParamSpec((2, 128, 256), elem_offset=0)
    flat, padded, total = build_flat([p], world_size, dev)

    local_blocks = (padded // world_size) // TILE
    foreign_blocks = 37                        # some other ST's region
    stride = foreign_blocks + local_blocks

    got_a, scales_a = simulate(flat, [p], world_size,
                               packed_scale_stride=local_blocks, scale_block_offset=0)
    got_b, scales_b = simulate(flat, [p], world_size,
                               packed_scale_stride=stride, scale_block_offset=foreign_blocks)

    assert torch.equal(got_a.view(torch.int8), got_b.view(torch.int8)), \
        "result depends on where the ST sits inside the packed amax buffer"
    assert torch.equal(scales_a[0][0], scales_b[0][0])


# ---------------------------------------------------------------------------
# 6. Degenerate blocks
# ---------------------------------------------------------------------------

@cuda_only
def test_all_zero_block_produces_finite_scale_and_zero_payload():
    dev = torch.device("cuda")
    p = ParamSpec((1, 128, 128), elem_offset=0)
    flat, _, total = build_flat([p], 1, dev, fill=lambda s: torch.zeros(s.numel, device=dev))

    got_fp8, per_rank_scales = simulate(flat, [p], world_size=1)
    s = per_rank_scales[0][0]
    assert torch.isfinite(s).all() and (s > 0).all(), f"degenerate scale: {s}"
    assert torch.all(got_fp8[:total].view(torch.int8) == 0)
    assert torch.isfinite(dequantize(got_fp8[:total].reshape(p.shape), s.reshape(1, 1, 1))).all()


@cuda_only
def test_block_amax_is_taken_over_the_full_128x128_block():
    """One large element must raise the scale for its ENTIRE block, and only it.

    Catches a fold that reduces over the wrong axis (e.g. per-row instead of
    per-block), which would still look sane on random data.
    """
    dev = torch.device("cuda")
    shape = (1, 256, 256)
    p = ParamSpec(shape, elem_offset=0)

    def fill(spec):
        w = torch.full((spec.n_row_elem, spec.n_cols), 1e-3, device=dev)
        w[200, 200] = 100.0          # block (1, 1)
        return w

    flat, _, total = build_flat([p], 1, dev, fill=fill)
    _, per_rank_scales = simulate(flat, [p], world_size=1)
    s = per_rank_scales[0][0].reshape(1, 2, 2)

    assert s[0, 1, 1] > s[0, 0, 0], "the spike did not raise its own block scale"
    assert s[0, 0, 0] == s[0, 0, 1] == s[0, 1, 0], "the spike leaked into other blocks"
    assert s[0, 1, 1].item() * FP8_MAX >= 100.0 * (1 - 1e-6)


# ---------------------------------------------------------------------------
# 7. Backward transpose: dgrad consumes W^T with a transposed block-scale grid
# ---------------------------------------------------------------------------

@cuda_only
def test_transposed_views_match_transposed_dequantization():
    """grouped_gemm.py:166 does (fp8_data.mT.contiguous(), fp8_scale.mT.contiguous()).

    That is only valid because blocks are square and 128-aligned, so block (i,j)
    of W^T is the transpose of block (j,i) of W and shares its amax.  Verify the
    identity that makes it true: dequant(W)^T == dequant(W^T).
    """
    dev = torch.device("cuda")
    torch.manual_seed(6)
    shape = (2, 256, 384)
    p = ParamSpec(shape, elem_offset=0)
    flat, _, total = build_flat([p], 2, dev)
    got_fp8, per_rank_scales = simulate(flat, [p], world_size=2)

    q = got_fp8[:total].reshape(shape)
    s = per_rank_scales[0][0].reshape(shape[0], shape[1] // TILE, shape[2] // TILE)

    deq_then_t = dequantize(q, s).mT
    deq_of_t = dequantize(q.mT.contiguous(), s.mT.contiguous())

    assert deq_of_t.shape == deq_then_t.shape
    assert torch.equal(deq_of_t, deq_then_t), \
        "transposing data and scale grids is not equivalent to transposing the dequantized weight"


# ---------------------------------------------------------------------------
# 8. compute_local_tile_amaxes
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("n_tiles", [1, 31, 32, 33, 1024, 4099])
def test_tile_amaxes_match_reference(n_tiles):
    """Includes n_tiles not divisible by BLOCK_M=32 — masks must not drop tiles."""
    dev = torch.device("cuda")
    torch.manual_seed(7)
    x = torch.randn(n_tiles * TILE, dtype=torch.bfloat16, device=dev)
    got = compute_local_tile_amaxes(x)
    ref = x.float().reshape(n_tiles, TILE).abs().amax(dim=1)
    assert torch.equal(got, ref), f"max diff {(got - ref).abs().max().item()}"


@cuda_only
def test_tile_amaxes_out_param_overwrites():
    dev = torch.device("cuda")
    x1 = torch.randn(64 * TILE, dtype=torch.bfloat16, device=dev)
    x2 = torch.randn(64 * TILE, dtype=torch.bfloat16, device=dev)
    out = torch.full((64,), -1.0, dtype=torch.float32, device=dev)
    compute_local_tile_amaxes(x1, out=out)
    compute_local_tile_amaxes(x2, out=out)
    assert torch.equal(out, x2.float().reshape(64, TILE).abs().amax(dim=1))


@cuda_only
def test_tile_amaxes_index_math_is_64bit():
    """float8_utils.py:66 promotes to int64 before the multiply.

    Allocating a >2**31-element shard is not feasible in a unit test, so assert
    the generated Triton IR actually uses 64-bit arithmetic rather than trusting
    the source.  Skips if the IR is unavailable in this Triton version.
    """
    dev = torch.device("cuda")
    x = torch.randn(64 * TILE, dtype=torch.bfloat16, device=dev)
    compute_local_tile_amaxes(x)

    from giga_fsdp.float8_utils import _compute_local_tile_amaxes_triton_kernel as k
    cache = getattr(k, "cache", None)
    if not cache:
        pytest.skip("Triton kernel cache not exposed in this version")
    kernels = [v for d in cache.values() for v in d.values()]
    if not kernels:
        pytest.skip("no compiled kernel found in cache")
    ttir = None
    for kern in kernels:
        asm = getattr(kern, "asm", None)
        if asm and "ttir" in asm:
            ttir = asm["ttir"]
            break
    if ttir is None:
        pytest.skip("TTIR not available")
    assert "i64" in ttir, "kernel index arithmetic is not 64-bit — overflows above 2**31 elements"


# ---------------------------------------------------------------------------
# 9. Determinism
# ---------------------------------------------------------------------------

@cuda_only
def test_repeated_runs_are_bitwise_identical():
    """Autotuning / scheduling must not change the result between calls."""
    dev = torch.device("cuda")
    torch.manual_seed(8)
    p = ParamSpec((2, 256, 256), elem_offset=0)
    flat, _, _ = build_flat([p], 4, dev)

    a_fp8, a_scales = simulate(flat, [p], 4)
    b_fp8, b_scales = simulate(flat, [p], 4)
    assert torch.equal(a_fp8.view(torch.int8), b_fp8.view(torch.int8))
    assert torch.equal(a_scales[0][0], b_scales[0][0])


# ---------------------------------------------------------------------------
# 10. Kernel edge cases
# ---------------------------------------------------------------------------

@cuda_only
def test_nan_in_shard_is_contained_to_its_own_block():
    """A NaN weight (post-divergence) must not poison neighbouring blocks.

    amax of the affected tile is NaN -> block_amax NaN -> scale NaN -> that
    whole 128x128 block quantizes to NaN.  That is acceptable and debuggable.
    What is NOT acceptable is the NaN spreading: every other block reads its own
    tile amaxes, so it must stay clean.  If this test fails, a single bad weight
    silently destroys the entire layer instead of one block.
    """
    dev = torch.device("cuda")
    shape = (1, 256, 256)                      # 2x2 = 4 blocks
    p = ParamSpec(shape, elem_offset=0)

    def fill(spec):
        w = torch.randn(spec.n_row_elem, spec.n_cols, device=dev)
        w[10, 10] = float("nan")               # block (0, 0)
        return w

    flat, _, total = build_flat([p], 1, dev, fill=fill)
    got_fp8, per_rank_scales = simulate(flat, [p], world_size=1)

    s = per_rank_scales[0][0].reshape(1, 2, 2)
    q = got_fp8[:total].reshape(shape).float().reshape(1, 2, TILE, 2, TILE)

    assert torch.isnan(s[0, 0, 0]) or torch.isnan(q[0, 0, :, 0, :]).any(), \
        "the NaN weight left no trace in its own block — amax fold is not reading it"

    clean = [(0, 1), (1, 0), (1, 1)]
    for br, bc in clean:
        assert torch.isfinite(s[0, br, bc]), \
            f"NaN leaked into the scale of clean block ({br},{bc}): {s[0, br, bc]}"
        assert torch.isfinite(q[0, br, :, bc, :]).all(), \
            f"NaN leaked into the payload of clean block ({br},{bc})"


@cuda_only
@pytest.mark.parametrize("amax", [
    3.0e38,      # near bf16 max (~3.39e38): scale ~2**120, still a normal float32
    1.0e-30,
    1.0e-36,
    1.0e-38,     # scale ~2**-136 -> SUBNORMAL float32
    6.0e-39,     # bf16 subnormal input, scale deep in float32 subnormal range
])
def test_extreme_magnitudes_do_not_produce_zero_or_inf_scale(amax):
    """scale = exp2(ceil(log2(amax / 448))) can land in the float32 subnormal range.

    For amax ~1e-38 the scale is around 2**-136, below the smallest normal
    float32 (2**-126).  If the kernel is compiled with flush-to-zero — which is
    common for fast-math paths — exp2 returns 0, scale_inv becomes inf, and
    every weight in the block turns into inf/NaN.  Nothing downstream checks it.

    A model with tiny weights in some block (dead expert, heavily decayed layer)
    would hit this in production and not in any random-data test.
    """
    dev = torch.device("cuda")
    shape = (1, 128, 128)
    p = ParamSpec(shape, elem_offset=0)

    def fill(spec):
        w = torch.zeros(spec.n_row_elem, spec.n_cols, device=dev)
        w[0, 0] = amax
        w[1, 1] = -amax / 4.0
        return w

    flat, _, total = build_flat([p], 1, dev, fill=fill)
    got_fp8, per_rank_scales = simulate(flat, [p], world_size=1)

    s = per_rank_scales[0][0]
    q = got_fp8[:total].reshape(shape)

    assert torch.isfinite(s).all(), f"non-finite scale for amax={amax:g}: {s}"
    assert (s > 0).all(), (
        f"scale flushed to zero for amax={amax:g} — exp2 underflowed into the "
        f"float32 subnormal range and got FTZ'd; scale_inv is now inf"
    )
    assert torch.isfinite(q.float()).all(), \
        f"payload is not finite for amax={amax:g} — division by a zero/subnormal scale"

    # the block must still be representable: scale * 448 >= amax
    w = flat[:total].reshape(shape).float()
    block_amax = w.abs().max().item()
    assert s.item() * FP8_MAX >= block_amax * (1 - 1e-6), (
        f"scale {s.item():g} cannot represent amax {block_amax:g}"
    )


@cuda_only
def test_realistic_layer_geometry():
    """Production-shaped parameter, not a toy.

    Every other test runs 256x256-ish shapes (~30 programs).  A real MoE layer
    is 2I=4096, H=7168 -> n_col_tiles=56, n_block_rows=32, thousands of programs
    and a different autotune pick.  Toy shapes exercise none of that.
    """
    dev = torch.device("cuda")
    free, _ = torch.cuda.mem_get_info()
    E, two_I, H = 4, 4096, 7168
    need = E * two_I * H * (2 + 1 + 1 + 4)      # bf16 + fp8 + gathered + fp32 ref
    if free < need:
        pytest.skip(f"needs ~{need / 2**30:.1f} GiB free, have {free / 2**30:.1f} GiB")

    torch.manual_seed(9)
    world_size = 8
    p = ParamSpec((E, two_I, H), elem_offset=0)
    flat, padded, total = build_flat([p], world_size, dev)

    got_fp8, per_rank_scales = simulate(flat, [p], world_size)

    w = flat[:total].reshape(E, two_I, H)
    ref_s = ref_block_scales(w)
    ref_q = ref_quantize(w, ref_s)

    assert ref_s.shape == (E, two_I // TILE, H // TILE)
    assert torch.equal(per_rank_scales[0][0].reshape(ref_s.shape), ref_s)
    assert torch.equal(got_fp8[:total].reshape(E, two_I, H).view(torch.int8),
                       ref_q.view(torch.int8))

    for r in range(1, world_size):
        assert torch.equal(per_rank_scales[r][0].view(torch.int32),
                           per_rank_scales[0][0].view(torch.int32)), \
            f"rank {r} disagrees on scales at production geometry"


# ---------------------------------------------------------------------------
# 11. Parameter selection predicates (no CUDA needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fqn,expected", [
    ("model.layers.0.block_sparse_moe.experts.weight1", True),
    ("model.layers.0.block_sparse_moe.experts.weight2", True),
    ("gemm.experts.weight1", True),
    # the trap: substring matching would accept these
    ("model.layers.0.mlp.shared_experts.weight1", False),
    ("model.layers.0.mlp.shared_experts.weight2", False),
    ("experts.weight3", False),
    ("experts_extra.weight1", False),
    ("weight1", False),
    ("", False),
])
def test_is_expert_param_is_segment_anchored(fqn, expected):
    assert is_expert_param(fqn) is expected


@pytest.mark.parametrize("shape,expected", [
    ((256, 128), True),
    ((4, 256, 128), True),
    ((256, 127), False),
    ((255, 128), False),
    ((128,), False),
    ((2, 3, 256, 128), False),
])
def test_fp8_param_shape_is_supported(shape, expected):
    assert fp8_param_shape_is_supported(shape) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
