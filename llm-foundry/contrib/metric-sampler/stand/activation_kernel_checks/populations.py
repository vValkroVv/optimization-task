from __future__ import annotations

import math

import torch

from stand.activation_kernel_checks.key_reference import (
    ONE_BLOCK_SELECTION,
    RAW_VALUES,
    SELECTION_DTYPES,
    SIGN_BIT_POSITION,
    from_raw_bits,
)

TENSOR_SEED = 0
TENSOR_ROWS = 4096
TENSOR_WIDTHS = (896, 1792, 7168)
TENSOR_DTYPES = (torch.bfloat16, torch.float32)
ACTIVATION_SCALE = 3.0
ZERO_ROW_STRIDE = 7
OUTLIER_VALUE = 1.0e6
NONCONTIGUOUS_WIDTH = 896
UNALIGNED_WIDTH = 300
WIDE_ROWS = 64
WIDE_ROW_WIDTHS = (4095, 4096)
OVERFLOWING_VALUE = 1e30

POPULATION_SEED = 7
SPECIAL_POPULATION_SIZE = 3 * 65536
SPECIAL_ROW_WIDTH = 256
SPECIAL_ELEMENTS = 200
SPECIAL_VALUE_SCALE = 1e-3
BFLOAT16_SUBNORMAL_BITS = 127
FLOAT16_SUBNORMAL_BITS = 1023
STRIDED_ROW_WIDTH = 896
ROW_STRIDE = 897


def special_population(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Small normal values, the first 200 replaced by the largest finite value of either sign
    (whose key shares its first digit with infinities and NaN), infinities and NaN of either sign."""
    values = (torch.randn(size, device=device) * SPECIAL_VALUE_SCALE).to(dtype)
    values[0:SPECIAL_ELEMENTS:4] = torch.finfo(dtype).max
    values[1:SPECIAL_ELEMENTS:4] = -torch.finfo(dtype).max
    values[2:SPECIAL_ELEMENTS:8] = float("inf")
    values[6:SPECIAL_ELEMENTS:8] = -float("inf")
    values[3:SPECIAL_ELEMENTS:4] = float("nan")
    bits = values.view(torch.int32 if dtype == torch.float32 else torch.int16)
    bits[7:SPECIAL_ELEMENTS:8] |= torch.iinfo(bits.dtype).min
    return values


def selection_populations(device: torch.device) -> list[tuple[str, torch.Tensor]]:
    """bf16/fp16 populations a selection from the keys' histogram could get wrong: every bit pattern
    (NaN payloads, subnormals, both zeros and infinities), ties and all-equal values, signed zeros,
    mostly or only non-finite values, the largest finite values, rows of any width, odd sizes and
    sizes around the one-block selection, a base pointer and rows that are not 16-byte aligned."""
    generator = torch.Generator().manual_seed(POPULATION_SEED)
    populations = []
    for dtype in SELECTION_DTYPES:
        populations += _populations_of_dtype(dtype, generator, device)
    return [(name, tensor.to(device)) for name, tensor in populations]


def _populations_of_dtype(
    dtype: torch.dtype, generator: torch.Generator, device: torch.device
) -> list[tuple[str, torch.Tensor]]:
    tag = str(dtype).removeprefix("torch.")
    finfo = torch.finfo(dtype)
    subnormal_bits = BFLOAT16_SUBNORMAL_BITS if dtype == torch.bfloat16 else FLOAT16_SUBNORMAL_BITS
    # Subnormals of either sign and zeros, then a few non-finite elements among a thousand finite.
    subnormals = from_raw_bits(
        torch.randint(0, subnormal_bits + 1, (50000,), generator=generator)
        | (torch.randint(0, 2, (50000,), generator=generator) << SIGN_BIT_POSITION), dtype)
    nonfinite = torch.tensor([math.nan, math.inf, -math.inf, -math.nan]).to(dtype)
    mostly_nonfinite = nonfinite[torch.randint(0, 4, (30000,), generator=generator)]
    mostly_nonfinite[::1000] = torch.randn(30, generator=generator).to(dtype)
    few_values = torch.tensor([-2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0]).to(dtype)
    base = torch.randn(300001, generator=generator).to(dtype)
    populations = [
        (f"every bit pattern {tag}", from_raw_bits(torch.arange(RAW_VALUES).repeat(4), dtype).reshape(-1, 512)),
        (f"random bits {tag}",
         from_raw_bits(torch.randint(0, RAW_VALUES, (300000,), generator=generator), dtype).reshape(-1, 500)),
        (f"all equal {tag}", torch.full((2000, 33), 1.5, dtype=dtype)),
        (f"all negative zero {tag}", torch.full((100, 100), -0.0, dtype=dtype)),
        (f"ties of eight values with signed zeros {tag}",
         few_values[torch.randint(0, 8, (100000,), generator=generator)].reshape(-1, 100)),
        (f"subnormals and zeros of either sign {tag}", subnormals.reshape(-1, 500)),
        (f"only non-finite {tag}", nonfinite[torch.randint(0, 4, (20000,), generator=generator)].reshape(-1, 200)),
        (f"mostly non-finite {tag}", mostly_nonfinite.reshape(-1, 300)),
        (f"one finite value among NaN {tag}",
         torch.cat((torch.full((9999,), math.nan, dtype=dtype),
                    torch.tensor([3.0], dtype=dtype))).reshape(-1, 100)),
        (f"largest finite beside non-finite {tag}",
         special_population(SPECIAL_POPULATION_SIZE, dtype, torch.device("cpu")).reshape(-1, SPECIAL_ROW_WIDTH)),
        (f"just past the one-block selection {tag}", base[:ONE_BLOCK_SELECTION + 1].reshape(1, -1)),
        (f"one row of a million {tag}", torch.randn(1, 1000003, generator=generator).to(dtype)),
        (f"wide rows {tag}", (torch.randn(40, 70000, generator=generator) * 100).to(dtype)),
        (f"odd shape {tag}", torch.randn(10001, 3, generator=generator).to(dtype)),
        (f"huge magnitudes {tag}", (torch.randn(2000, 70, generator=generator) * finfo.max / 8).to(dtype)),
    ]
    populations += [
        (f"misaligned base pointer {tag}", base.to(device)[1:].reshape(-1, 1000)),
        (f"rows of {STRIDED_ROW_WIDTH} in a stride of {ROW_STRIDE} {tag}",
         torch.randn(TENSOR_ROWS, ROW_STRIDE, generator=generator).to(dtype).to(device)[:, :STRIDED_ROW_WIDTH]),
    ]
    return populations


def make_tensors(device: torch.device) -> list[tuple[str, torch.Tensor]]:
    torch.manual_seed(TENSOR_SEED)
    tensors = []
    for width in TENSOR_WIDTHS:
        for dtype in TENSOR_DTYPES:
            base = (torch.randn(TENSOR_ROWS, width, device=device) * ACTIVATION_SCALE).to(dtype)
            tag = f"h={width} dtype={dtype}"
            tensors.append((f"normal {tag}", base))

            with_zeros = base.clone()
            with_zeros[::ZERO_ROW_STRIDE] = 0
            tensors.append((f"zeros {tag}", with_zeros))

            with_outliers = base.clone()
            with_outliers[0, 0] = OUTLIER_VALUE
            with_outliers[-1, -1] = -OUTLIER_VALUE
            tensors.append((f"outliers {tag}", with_outliers))

            with_nonfinite = base.clone()
            with_nonfinite[1, :3] = float("nan")
            with_nonfinite[2, :3] = float("inf")
            with_nonfinite[3, :3] = float("-inf")
            tensors.append((f"nonfinite {tag}", with_nonfinite))

    noncontiguous = torch.randn(TENSOR_ROWS, 2 * NONCONTIGUOUS_WIDTH, device=device)[:, ::2]
    tensors.append((f"noncontiguous h={NONCONTIGUOUS_WIDTH}", noncontiguous))
    for dtype in TENSOR_DTYPES:
        tensors.append((f"unaligned rows h={UNALIGNED_WIDTH} dtype={dtype}",
                        (torch.randn(TENSOR_ROWS, UNALIGNED_WIDTH + 1, device=device) * ACTIVATION_SCALE)
                        .to(dtype)[:, 1:]))
        for width in WIDE_ROW_WIDTHS:
            tensors.append(
                (f"wide rows h={width} dtype={dtype}",
                 (torch.randn(WIDE_ROWS, width, device=device) * ACTIVATION_SCALE).to(dtype))
            )
    tensors.append(
        ("finite FP32 square overflow", torch.tensor([[OVERFLOWING_VALUE, -OVERFLOWING_VALUE]], device=device))
    )
    tensors.append(("mixed NaN/Inf and square overflow", torch.tensor(
        [[OVERFLOWING_VALUE, -OVERFLOWING_VALUE, float("nan"), float("inf")], [1., -2., 0., -float("inf")]],
        device=device)))
    return tensors
