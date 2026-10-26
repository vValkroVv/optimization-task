from __future__ import annotations

import math
from collections.abc import Sequence
from types import ModuleType

import torch

from metric_sampler.capture.recorder import CaptureCapacityError
from metric_sampler.capture.stats.helpers.norms import MATRIX_DIMENSIONS
from metric_sampler.contract.metrics import INT32_MAX, QUANTILE_COUNTS_WIDTH, QUANTILE_LEVELS, WORDS_PER_FLOAT64

_PERCENT = 100
# (x + _CEILING_OFFSET) // _PERCENT is ceil(x / _PERCENT) for integer x.
_CEILING_OFFSET = _PERCENT - 1
QUANTILE_PERCENTILES = tuple(round(level * _PERCENT) for level in QUANTILE_LEVELS)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
# Limits of the CUDA selection: ranks one call takes, and columns one warp reduces per row.
_MAX_CUDA_RANKS = 7
_MAX_WARP_ROW_COLUMNS = 1024


def _quantiles_kernel() -> ModuleType:  # pragma: no cover
    from metric_sampler.capture.kernels import quantiles_kernel

    return quantiles_kernel


def quantile_slots(name: str, counts_name: str, *, float64: bool = False) -> tuple[tuple[str, int], ...]:
    """The slots of one quantile population: its quantiles (a Float64 one as two Int32 words each) and their
    population counts, as (metric name, width)."""
    words_per_quantile = WORDS_PER_FLOAT64 if float64 else 1
    return ((name, len(QUANTILE_LEVELS) * words_per_quantile), (counts_name, QUANTILE_COUNTS_WIDTH))


def validate_population(values: torch.Tensor, valid: torch.Tensor | None) -> None:
    if values.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"quantile dtype {values.dtype} is not supported")
    if valid is not None and (
        valid.dtype != torch.bool or valid.shape != values.shape or valid.device != values.device
    ):
        raise ValueError("quantile validity mask must be boolean with the same shape and device")
    if values.numel() > INT32_MAX:
        raise CaptureCapacityError("quantile population exceeds int32 counts")


def exact_quantiles(values: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact finite-population inverse empirical CDF; CUDA selects by radix histograms, not a sort."""
    if values.device.type == "cuda" and values.numel():  # pragma: no cover
        validate_population(values, valid)
        return _quantiles_kernel().launch_exact_quantiles(values, valid)
    return exact_quantiles_aten(values, valid)


def exact_quantiles_into(
    values: torch.Tensor,
    valid: torch.Tensor | None,
    quantiles: torch.Tensor,
    counts: torch.Tensor,
    first_pass: torch.Tensor | None = None,
) -> None:
    """`exact_quantiles` of a 16- or 32-bit population written into float32 `quantiles` and int32 `counts`.

    On CUDA the extension writes them without intermediate tensors, and skips its first pass given `first_pass`,
    the scratch its activation statistics counted for the same population: the histogram of the first digit, or
    for bf16 and fp16 the counts of every raw value, which leave nothing to refine.
    """
    if values.device.type == "cuda" and values.numel():  # pragma: no cover
        validate_population(values, valid)
        if _quantiles_kernel().launch_exact_quantiles_into(values, valid, quantiles, counts, first_pass):
            return
    selected, population = exact_quantiles(values, valid)
    quantiles.copy_(selected)
    counts.copy_(population)


def order_statistics(values: torch.Tensor, ranks: Sequence[int]) -> torch.Tensor:
    """Values at `ranks` of all of `values` in torch.sort order (NaN after +inf), float64 for a float64 source and
    float32 otherwise; CUDA selects a few ranks by radix histograms."""
    flat = values.detach().reshape(-1)
    if not 0 <= min(ranks) <= max(ranks) < flat.numel():
        raise ValueError(f"ranks {list(ranks)} are outside a population of {flat.numel()}")
    if (flat.is_cuda and flat.dtype != torch.float64 and len(ranks) <= _MAX_CUDA_RANKS
            and flat.numel() < INT32_MAX):  # pragma: no cover
        selected = _quantiles_kernel().launch_order_statistics(flat.contiguous(), list(ranks))
        if selected is not None:
            return selected
    ordered = torch.sort(flat).values
    return torch.stack([ordered[rank] for rank in ranks]).to(
        torch.float64 if flat.dtype == torch.float64 else torch.float32)


def _is_cuda_float32_rows(values: torch.Tensor) -> bool:
    """Whether the CUDA row kernels read `values`: a float32 matrix of at least one row, contiguous along its rows."""
    return (values.is_cuda and values.dtype == torch.float32 and values.ndim == MATRIX_DIMENSIONS
            and values.shape[0] > 0 and values.stride(-1) == 1)


def row_top_values(values: torch.Tensor, k: int) -> torch.Tensor:
    """`values.topk(k, dim=-1).values` of a 2D tensor; on CUDA, float32 rows within the kernel's column limit are
    selected by one warp per row."""
    if _is_cuda_float32_rows(values) and 1 <= k <= values.shape[1] <= _MAX_WARP_ROW_COLUMNS:  # pragma: no cover
        selected = _quantiles_kernel().launch_row_top_values(values, k)
        if selected is not None:
            return selected
    return values.topk(k, dim=-1).values


def row_entropy(values: torch.Tensor) -> torch.Tensor:
    """Float64 entropy of each row of a 2D tensor normalized to sum to one, divided by the log of the row length;
    on CUDA, float32 rows are reduced by one warp per row."""
    if _is_cuda_float32_rows(values) and values.shape[1] > 0:  # pragma: no cover
        entropy = _quantiles_kernel().launch_row_entropy(values)
        if entropy is not None:
            return entropy
    probabilities = values.to(torch.float64)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1) / math.log(values.shape[-1])


def row_margins(values: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    """`boundary[:, None] - values` of a 2D tensor and one boundary per row, a NaN value counting as +inf; on CUDA,
    float32 rows are subtracted by one warp per row in one pass."""
    if (_is_cuda_float32_rows(values) and values.shape[1] > 0 and boundary.dtype == torch.float32
            and boundary.shape == values.shape[:1]):  # pragma: no cover
        margins = _quantiles_kernel().launch_row_margins(values, boundary)
        if margins is not None:
            return margins
    return boundary[:, None] - torch.where(values.isnan(), math.inf, values)


def exact_quantiles_aten(values: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-population inverse empirical CDF in torch, with exact integer population counts.

    Flattens all axes and selects max(ceil(q*N)-1, 0) from the N sorted finite valid values, without interpolation
    or sampling. Float64 populations keep float64 values; other supported dtypes fit exactly in float32. Counts are
    [finite, NaN, +Inf, -Inf, masked]. An empty finite population gives nine NaNs; masked values enter only the
    masked count.
    """
    validate_population(values, valid)
    flat = values.detach().reshape(-1)
    output_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    if not flat.numel():
        return (
            torch.full((len(QUANTILE_PERCENTILES),), float("nan"), dtype=output_dtype, device=values.device),
            torch.zeros(QUANTILE_COUNTS_WIDTH, dtype=torch.int32, device=values.device),
        )
    mask = torch.ones_like(flat, dtype=torch.bool) if valid is None else valid.reshape(-1)
    finite = torch.isfinite(flat) & mask
    counts = torch.stack((
        finite.sum(),
        (torch.isnan(flat) & mask).sum(),
        ((flat == float("inf")) & mask).sum(),
        ((flat == -float("inf")) & mask).sum(),
        (~mask).sum(),
    ))
    finite_count = counts[0]
    ordered = torch.sort(torch.where(finite, flat, torch.full_like(flat, float("inf")))).values
    percentiles = torch.tensor(QUANTILE_PERCENTILES, dtype=torch.int64, device=flat.device)
    indices = ((finite_count * percentiles + _CEILING_OFFSET) // _PERCENT - 1).clamp_min_(0)
    selected = ordered[indices].to(output_dtype)
    quantiles = torch.where(finite_count > 0, selected, torch.full_like(selected, float("nan")))
    return quantiles, counts.to(torch.int32)
