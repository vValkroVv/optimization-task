"""Benchmark-only parallel vocabulary reductions for Stable CE forward."""

import torch
import triton
import triton.language as tl


@triton.jit
def tile_statistics(X, Y, PART_MAX, PART_SUM,
                    V: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    target = tl.load(Y + row)
    # The merge kernel handles ignored rows without reading partial statistics.
    if target != -100:
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + row.to(tl.int64) * V + cols, cols < V,
                         other=float("-inf")).to(tl.float32)
        maximum = tl.max(values, 0)
        denominator = tl.sum(tl.exp(values - maximum), 0)
        tl.store(PART_MAX + row * TILES + tile, maximum)
        tl.store(PART_SUM + row * TILES + tile, denominator)


@triton.jit
def merge_statistics(X, Y, PART_MAX, PART_SUM, LOSS, MAXIMUM, LOG_DEN,
                     V: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    target = tl.load(Y + row)
    if target == -100:
        tl.store(LOSS + row, 0.0)
        tl.store(MAXIMUM + row, 0.0)
        tl.store(LOG_DEN + row, 0.0)
    else:
        tl.device_assert((target >= 0) & (target < V), "CE target outside vocabulary")
        tiles = tl.arange(0, BLOCK)
        maxima = tl.load(PART_MAX + row * TILES + tiles, tiles < TILES, other=float("-inf"))
        partial = tl.load(PART_SUM + row * TILES + tiles, tiles < TILES, other=0.0)
        maximum = tl.max(maxima, 0)
        log_denominator = tl.log(tl.sum(partial * tl.exp(maxima - maximum), 0))
        chosen = tl.load(X + row.to(tl.int64) * V + target, (target >= 0) & (target < V),
                         other=0.0).to(tl.float32)
        tl.store(LOSS + row, -((chosen - maximum) - log_denominator))
        tl.store(MAXIMUM + row, maximum)
        tl.store(LOG_DEN + row, log_denominator)


@triton.jit
def cluster_forward(X, Y, LOSS, MAXIMUM, LOG_DEN, V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    target = tl.load(Y + row)
    if target == -100:
        tl.store(LOSS + row, 0.0)
        tl.store(MAXIMUM + row, 0.0)
        tl.store(LOG_DEN + row, 0.0)
    else:
        tl.device_assert((target >= 0) & (target < V), "CE target outside vocabulary")
        cols = tl.arange(0, BLOCK)
        values = tl.load(X + row.to(tl.int64) * V + cols, cols < V,
                         other=float("-inf")).to(tl.float32)
        maximum = tl.max(values, 0)
        log_denominator = tl.log(tl.sum(tl.exp(values - maximum), 0))
        chosen = tl.load(X + row.to(tl.int64) * V + target, (target >= 0) & (target < V),
                         other=0.0).to(tl.float32)
        tl.store(LOSS + row, -((chosen - maximum) - log_denominator))
        tl.store(MAXIMUM + row, maximum)
        tl.store(LOG_DEN + row, log_denominator)


def alloc_outputs(logits):
    rows = logits.shape[0]
    return tuple(torch.empty(rows, device=logits.device, dtype=torch.float32) for _ in range(3))


def launch_split(logits, target, partials, outputs, block, warps):
    rows, vocab = logits.shape
    tiles = triton.cdiv(vocab, block)
    tile_statistics[(rows, tiles)](
        logits, target, *partials, vocab, tiles, block,
        num_warps=warps, enable_fp_fusion=False,
    )
    merge_statistics[(rows,)](
        logits, target, *partials, *outputs, vocab, tiles, triton.next_power_of_2(tiles),
        num_warps=1, enable_fp_fusion=False, debug=True,
    )


def launch_cluster(logits, target, outputs, ctas, warps):
    rows, vocab = logits.shape
    cluster_forward[(rows,)](
        logits, target, *outputs, vocab, triton.next_power_of_2(vocab),
        num_ctas=ctas, num_warps=warps, enable_fp_fusion=False, debug=True,
    )
