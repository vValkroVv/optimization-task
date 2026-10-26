import math
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

try:  # torch >= 2.13: StridedShard is public and no longer subclasses Shard
    from torch.distributed.tensor.placement_types import StridedShard
except ImportError:  # older torch: _StridedShard is a Shard subclass
    from torch.distributed.tensor.placement_types import _StridedShard as StridedShard

from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.metrics.sample import MetricRecord, MetricValue
from torchtitan.config.schema.callbacks.metrics.sample import (
    DTensorMode,
    MetricStepKind,
    ReduceMode,
    ReduceOp,
    AxisSpec,
    MetricConfig,
    MetricStepConfig,
    ReduceStageConfig,
    StatOpConfig,
    StatOpName,
    ThresholdCmp,
)
from torchtitan.protocols.sharding_backend import ShardStats
from torchtitan.tools.logging import logger

from torchtitan.ops.float8.fp8_tensor import (
    Float8BlockwiseQTensor,
    InputLayoutType,
)

# Every placement type that shards a tensor dimension. isinstance(p, Shard)
# alone is NOT enough: FSDP2 x TP produces strided shards, and their class
# stopped subclassing Shard in torch 2.13.
SHARD_PLACEMENTS = (Shard, StridedShard)


def _is_shard_placement(placement: object, stats: ShardStats | None = None) -> bool:
    """Whether ``placement`` shards a tensor, ragged placements included."""
    if isinstance(placement, SHARD_PLACEMENTS):
        return True
    return stats is not None and isinstance(placement, stats.placement_types)


# Resolved (positional) reduction axes, as produced by AxisSpec.resolve.
ResolvedDims = int | tuple[int, ...]

_EPS = 1e-12
_ACC_DTYPE = torch.float64


def _replicate_partials(value: DTensor | torch.Tensor) -> DTensor | torch.Tensor:
    """Reduce Partial placements to Replicate, leaving other axes untouched.

    A ragged shard tensor never carries Partial placements and passes through.
    """
    if not any(isinstance(placement, Partial) for placement in value.placements):
        return value
    return value.redistribute(
        placements=[
            Replicate() if isinstance(placement, Partial) else placement
            for placement in value.placements
        ]
    )


def _reconstruct_full_tensor(value: DTensor) -> torch.Tensor:
    """Exact logical tensor for layouts DTensor.full_tensor() reorders wrong.

    full_tensor() of _StridedShard placements (FSDP2 x TP dim-0 nesting)
    reassembles through a strided all-gather that misplaces padding on uneven
    shards. But each strided local shard still covers a contiguous logical
    slice — only the gather order is interleaved — so the tensor can be
    rebuilt from first principles: every rank writes its local shard at the
    global offset (the same computation DCP checkpoints rely on), then the
    shard axes are all-reduce summed. Shards are disjoint, so the sum is
    exact and padding never enters.
    """
    value = _replicate_partials(value)
    local = value.to_local()
    local_shape, global_offset = compute_local_shape_and_global_offset(
        value.shape, value.device_mesh, value.placements
    )
    full = torch.zeros(tuple(value.shape), dtype=local.dtype, device=local.device)
    if all(size > 0 for size in local_shape):
        region = tuple(
            slice(offset, offset + size)
            for offset, size in zip(global_offset, local_shape)
        )
        full[region] = local
    for axis, placement in enumerate(value.placements):
        if isinstance(placement, SHARD_PLACEMENTS):
            dist.all_reduce(full, group=value.device_mesh.get_group(axis))
    return full


def dequantize_fp8(value: Float8BlockwiseQTensor) -> torch.Tensor:
    if value._input_layout_type != InputLayoutType.ROWWISE:
        raise NotImplementedError(
            "monitoring dequantizes ROWWISE fp8 only; "
            f"{value._input_layout_type} stores its data transposed relative to "
            "orig_shape and would come out as plausible, wrong magnitudes"
        )
    if value._scale.dtype != torch.float32:
        raise ValueError(f"expected fp32 scales, got {value._scale.dtype}")
    from cudacherry import dequantize_blk128x128, dequantize_vec128

    if value._block_scaling_dim == 1:
        dequantize = dequantize_vec128
    elif value._block_scaling_dim == 2:
        dequantize = dequantize_blk128x128
    else:
        raise ValueError(f"Unsupported block_scaling_dim: {value._block_scaling_dim}")
    return dequantize(
        value._data,
        value._scale,
        scale_block_first=value._scale_block_first,
        out_dtype=value._orig_dtype,
        out_shape=tuple(value.orig_shape),
    )


def _local_tensor(
    value: MetricValue,
    *,
    full_tensor: bool = False,
    dtype: torch.dtype | None = None,
    shard_stats: ShardStats | None = None,
) -> torch.Tensor:
    """A detached tensor: the local shard by default, or the full logical one.

    Returns ``value`` itself when there is nothing to do, which is the common
    case: every metric of a capture, and every step of every metric, re-enters
    here on a tensor the first call already normalized.
    """
    if not isinstance(value, torch.Tensor):
        return torch.tensor(
            float(value), dtype=torch.float32 if dtype is None else dtype
        )
    if isinstance(value, Float8BlockwiseQTensor):
        value = dequantize_fp8(value)
    elif shard_stats is not None and isinstance(value, shard_stats.tensor_types):
        value = (
            shard_stats.to_full_tensor(value)
            if full_tensor
            else shard_stats.to_local(value)
        )
    elif isinstance(value, DTensor):
        if not full_tensor:
            value = value.to_local()
        elif any(isinstance(p, StridedShard) for p in value.placements):
            value = _reconstruct_full_tensor(value)
        else:
            value = value.full_tensor()
    if value.requires_grad:
        value = value.detach()
    if value.is_complex():
        value = value.abs()
    if dtype is not None and value.dtype != dtype:
        value = value.to(dtype)
    return value


def plain_float_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """The full logical tensor as detached fp32, passing None through."""
    if tensor is None:
        return None
    return _local_tensor(tensor, full_tensor=True, dtype=torch.float32)


def _dtensor_reduce_groups(
    value: DTensor | torch.Tensor, shard_stats: ShardStats | None = None
) -> tuple[str, ...] | None:
    """Mesh axis names a partial stat must reduce over; None when it cannot."""
    mesh_names = tuple(value.device_mesh.mesh_dim_names or ())
    groups: list[str] = []
    for idx, placement in enumerate(value.placements):
        if not _is_shard_placement(placement, shard_stats):
            continue
        if idx >= len(mesh_names):
            return None
        groups.append(mesh_names[idx])
    return tuple(groups)


def _normalized_dims(dim: ResolvedDims, ndim: int) -> tuple[int, ...]:
    """Axes as non-negative indices."""
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    return tuple(d if d >= 0 else ndim + d for d in dims)


def _remaining_axes(axis: AxisSpec, dim: ResolvedDims | None) -> AxisSpec:
    if dim is None:
        return AxisSpec()
    normalized = set(_normalized_dims(dim, len(axis.names)))
    return AxisSpec(
        names=tuple(name for i, name in enumerate(axis.names) if i not in normalized)
    )


def _dim_numel(value: torch.Tensor, dim: ResolvedDims | None) -> int:
    """Number of elements a reduction over ``dim`` collapses per output element."""
    if dim is None:
        return value.numel()
    dims = (dim,) if isinstance(dim, int) else dim
    return math.prod(value.shape[d] for d in dims)


def _amin(value: torch.Tensor, dim: ResolvedDims | None) -> torch.Tensor:
    return value.min() if dim is None else value.amin(dim=dim)


def _amax(value: torch.Tensor, dim: ResolvedDims | None) -> torch.Tensor:
    return value.max() if dim is None else value.amax(dim=dim)


def histogram(values: torch.Tensor, nbins: int) -> torch.Tensor:
    """Count occurrences of each index, without a device sync: ``(nbins + 1,)``."""
    flat = values.detach().to(torch.long).flatten()
    flat = flat.masked_fill((flat < 0) | (flat >= nbins), nbins)
    out = torch.zeros(nbins + 1, dtype=torch.float32, device=flat.device)
    return out.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))


def _mask_fraction(mask: torch.Tensor, dim: ResolvedDims | None) -> torch.Tensor:
    """Fraction of True in ``mask``, reduced over ``dim``, without a copy."""
    # fp64: an fp32 counter stops being exact past 2**24, and one activation of
    # a 3B model is ~67M elements.
    total = mask.sum(dim=dim, dtype=_ACC_DTYPE)
    return total / max(1, _dim_numel(mask, dim))


def _assert_order_stat_size(op: str, numel: int, metric_name: str) -> None:
    if op in ("quantile", "min_median_ratio"):
        assert numel <= 2**24, (
            f"Metric '{metric_name}' has {numel} elements, exceeding "
            "torch.quantile's 2**24-element limit. Quantile and median cannot "
            "process this input, and gathering sharded weights first may waste memory."
        )
    if op == "gini":
        assert numel <= 2**24, (
            f"Metric '{metric_name}' has {numel} elements (> 2**24). "
            "This Gini implementation sorts the full input and is not optimized "
            "for large tensors, so it may cause CUDA OOM."
        )


def _quantile(value: torch.Tensor, q: float, dim: ResolvedDims | None) -> torch.Tensor:
    _assert_order_stat_size("quantile", value.numel(), "quantile")
    if value.dtype not in (torch.float32, torch.float64):
        value = value.float()
    return torch.quantile(value, q, dim=dim)


def _centered_residual(value: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """``value - mean`` as a fresh fp32 tensor, without mutating ``value``."""
    mean32 = mean.to(torch.float32)
    if value.dtype == torch.float32:
        return value - mean32
    centered = value.to(torch.float32)
    centered.sub_(mean32)
    return centered


def _central_moments(
    value: torch.Tensor, dim: ResolvedDims | None, orders: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """``E[(x - mean) ** k]`` for every k in ``orders``, from ONE residual."""
    mean = value.mean(dim=dim, keepdim=dim is not None, dtype=_ACC_DTYPE)
    centered = _centered_residual(value, mean)
    count = max(1, _dim_numel(value, dim))
    return tuple(
        torch.linalg.vector_norm(centered, ord=order, dim=dim, dtype=_ACC_DTYPE)
        .pow(order)
        .div(count)
        for order in orders
    )


def _safe_std(tensor: torch.Tensor, dim: ResolvedDims | None) -> torch.Tensor:
    if tensor.numel() <= 1:
        return torch.zeros((), device=tensor.device, dtype=_ACC_DTYPE)
    return _central_moments(tensor, dim, (2,))[0].sqrt()


def _coefficient_of_variation(
    tensor: torch.Tensor, dim: ResolvedDims | None
) -> torch.Tensor:
    mean = tensor.mean(dim=dim)
    return _safe_std(tensor, dim) / mean.abs().clamp_min(_EPS)


def _gini(tensor: torch.Tensor, dim: ResolvedDims | None) -> torch.Tensor:
    """Gini coefficient over one axis, or over the flattened tensor for None."""
    _assert_order_stat_size("gini", tensor.numel(), "gini")
    if dim is None:
        moved = tensor.flatten()
    elif isinstance(dim, int):
        # transpose(dim, -1) also moves the last axis to `dim`, so reducing
        # over -1 afterwards leaves the two swapped relative to any torch
        # reduction over the same axis. movedim only takes the reduced axis
        # out.
        moved = tensor.movedim(dim, -1)
    else:
        raise ValueError(f"gini supports a single axis, got {dim}")

    n = moved.shape[-1]
    if n == 0:
        return torch.zeros(moved.shape[:-1], device=tensor.device, dtype=torch.float32)

    sorted_values = moved.sort(dim=-1).values
    weights = torch.arange(1, n + 1, device=tensor.device, dtype=torch.float32)
    denom = n * sorted_values.sum(dim=-1).abs().clamp_min(_EPS)
    return (2 * (sorted_values * weights).sum(dim=-1) / denom) - (n + 1) / n


def _threshold_hits(
    abs_values: torch.Tensor, bar: torch.Tensor, cmp: ThresholdCmp
) -> torch.Tensor:
    """Count of |x| beyond ``bar`` in fp64; 0 for an empty local shard."""
    if abs_values.numel() == 0:
        return torch.zeros((), device=abs_values.device, dtype=torch.float64)
    mask = abs_values.gt(bar) if cmp == ThresholdCmp.gt else abs_values.lt(bar)
    return mask.sum(dtype=torch.float64)


def apply_stat_op(
    value: torch.Tensor, axis: AxisSpec, op: StatOpConfig
) -> torch.Tensor:
    """Evaluate one statistic on a plain (rank-local or full) tensor."""
    named = isinstance(op.dim, str) or (
        isinstance(op.dim, (list, tuple))
        and any(isinstance(entry, str) for entry in op.dim)
    )
    if named and len(axis.names) != value.ndim:
        raise ValueError(
            f"Stat '{op.op}' selects dim {op.dim!r} by axis name, but the "
            f"tensor is {value.ndim}-D while the configured axes name "
            f"{len(axis.names)} dims ({axis.names}) — a name would land on "
            "the wrong dimension. Fix the source's `axes`, capture fewer "
            "tensor shapes, or use an integer dim."
        )
    dim = axis.resolve(op.dim)
    name = op.op

    if name == StatOpName.ABS:
        return value.abs()
    if name == StatOpName.MEAN:
        return value.mean(dim=dim, dtype=_ACC_DTYPE)
    if name == StatOpName.SUM:
        return value.sum(dim=dim, dtype=_ACC_DTYPE)
    if name == StatOpName.MIN:
        return _amin(value, dim)
    if name == StatOpName.MAX:
        return _amax(value, dim)
    if name == StatOpName.STD:
        return _safe_std(value, dim)
    if name == StatOpName.VAR:
        return _central_moments(value, dim, (2,))[0]
    if name == StatOpName.ABS_MAX:
        return _amax(value.abs(), dim)
    if name == StatOpName.RMS:
        total = torch.linalg.vector_norm(value, ord=2, dim=dim, dtype=_ACC_DTYPE)
        return (total.square() / max(1, _dim_numel(value, dim))).sqrt()
    if name == StatOpName.NORM:
        p = 2.0 if op.p is None else op.p
        return torch.linalg.vector_norm(value, ord=p, dim=dim, dtype=_ACC_DTYPE)
    if name == StatOpName.COUNT:
        count = _dim_numel(value, dim)
        if dim is None:
            return torch.tensor(count, device=value.device, dtype=_ACC_DTYPE)
        dims = _normalized_dims(dim, value.ndim)
        shape = [s for i, s in enumerate(value.shape) if i not in dims]
        return torch.full(shape, count, device=value.device, dtype=_ACC_DTYPE)
    if name == StatOpName.NONZERO_COUNT:
        return (value != 0).sum(dim=dim, dtype=_ACC_DTYPE)
    if name == StatOpName.ZERO_RATIO:
        return _mask_fraction(value == 0, dim)
    if name == StatOpName.FINITE_RATIO:
        return _mask_fraction(torch.isfinite(value), dim)
    if name == StatOpName.NAN_COUNT:
        return torch.isnan(value).sum(dim=dim, dtype=_ACC_DTYPE)
    if name == StatOpName.INF_COUNT:
        return torch.isinf(value).sum(dim=dim, dtype=_ACC_DTYPE)
    if name == StatOpName.QUANTILE:
        return _quantile(value, 0.5 if op.q is None else op.q, dim)
    if name == StatOpName.KURTOSIS:
        var, m4 = _central_moments(value, dim, (2, 4))
        return m4 / var.square().clamp_min(_EPS**2)
    if name == StatOpName.ACTABS:
        # Fraction of |x| beyond an absolute threshold (massive activations).
        if op.threshold is None:
            raise ValueError("actabs requires a threshold")
        abs_value = value.abs()
        mask = (
            abs_value.gt(op.threshold)
            if op.cmp == ThresholdCmp.gt
            else abs_value.lt(op.threshold)
        )
        return _mask_fraction(mask, dim)
    if name == StatOpName.ACTREL:
        # Fraction of |x| beyond threshold * mean(|x|): outliers relative to
        # the tensor's own scale. One abs serves both the bar and the
        # comparison.
        if op.threshold is None:
            raise ValueError("actrel requires a threshold")
        abs_value = value.abs()
        bar = op.threshold * abs_value.mean(dtype=_ACC_DTYPE)
        mask = abs_value.gt(bar) if op.cmp == ThresholdCmp.gt else abs_value.lt(bar)
        return _mask_fraction(mask, dim)
    if name == StatOpName.ENTROPY:
        probs = value / value.sum(dim=dim, keepdim=True).clamp_min(_EPS)
        return -(probs * probs.clamp_min(_EPS).log()).sum(dim=dim)
    if name == StatOpName.RAW_ENTROPY_NORM:
        count = _dim_numel(value, dim)
        entropy = -(value * value.clamp_min(_EPS).log()).sum(dim=dim)
        return entropy / math.log(count) if count > 1 else entropy
    if name == StatOpName.MAX_MIN:
        return _amax(value, dim) - _amin(value, dim)
    if name == StatOpName.MAX_VIO:
        mean = value.mean(dim=dim)
        return (_amax(value, dim) - mean) / mean.clamp_min(1.0)
    if name == StatOpName.MAX_MIN_RATIO:
        low = _amin(value, dim)
        high = _amax(value, dim)
        return torch.where(low > 0, high / low, torch.full_like(high, float("inf")))
    if name == StatOpName.MIN_MEDIAN_RATIO:
        low = _amin(value, dim)
        median = _quantile(value, 0.5, dim)
        return torch.where(median > 0, low / median, torch.full_like(low, float("inf")))
    if name == StatOpName.POSITIVE_RATIO:
        return _mask_fraction(value > 0, dim)
    if name == StatOpName.CV:
        return _coefficient_of_variation(value, dim)
    if name == StatOpName.GINI:
        return _gini(value, dim)
    if name == StatOpName.BINCOUNT:
        if op.minlength:
            return histogram(value, op.minlength)[: op.minlength]
        return torch.bincount(value.to(torch.long).flatten()).to(torch.float32)

    raise ValueError(f"Unknown monitoring stat op: {name}")


def _partial_stat_op(
    value: DTensor | torch.Tensor,
    axis: AxisSpec,
    op: StatOpConfig,
    reducer: ParallelReducer,
) -> torch.Tensor | None:
    """Compute simple sharding-aware stats without materializing the full tensor.

    This handles reductions over the whole logical tensor. Axis-local stats
    return None and the caller falls back to the full tensor, because logical
    tensor axes are not mapped to mesh placements here. The accumulator math
    runs on the local shard and reduces fixed-size scalars, so it is exact for
    registered ragged shard tensors too, whose flat local shards differ in
    size per rank.
    """
    if op.dim is not None:
        dim = axis.resolve(op.dim)
        if (
            op.op == StatOpName.NORM
            and (op.p is None or float(op.p) == 2.0)
            and isinstance(value, reducer.ragged_tensor_types)
            and isinstance(dim, int)
            and value.ndim >= 1
            and -value.ndim <= dim < value.ndim
            and (dim % value.ndim) in (0, value.ndim - 1)
        ):
            groups = _dtensor_reduce_groups(value, reducer.shard_stats)
            if groups is None:
                return None
            assert reducer.shard_stats is not None
            squared = reducer.shard_stats.local_squared_norms(value, dim % value.ndim)
            if groups:
                squared = reducer.reduce_over_groups(squared, groups, ReduceOp.sum)
            return squared.sqrt_()
        return None

    value = _replicate_partials(value)

    if op.op == StatOpName.ABS:
        # Elementwise: the result stays a DTensor with the same placements.
        return value.abs()

    groups = _dtensor_reduce_groups(value, reducer.shard_stats)
    if groups is None:
        return None

    local = _local_tensor(value, shard_stats=reducer.shard_stats)
    if not groups:
        return apply_stat_op(local, axis, op)

    name = op.op
    if name == StatOpName.ENTROPY and isinstance(value, reducer.ragged_tensor_types):
        total = reducer.reduce_over_groups(
            local.sum(dtype=torch.float32), groups, ReduceOp.sum
        )
        probs = local.to(dtype=torch.float32, copy=True).div_(total.clamp_min(_EPS))
        log_probs = probs.clamp_min(_EPS).log_()
        local_entropy = -log_probs.mul_(probs).sum()
        return reducer.reduce_over_groups(local_entropy, groups, ReduceOp.sum)
    if name == StatOpName.RAW_ENTROPY_NORM and isinstance(
        value, reducer.ragged_tensor_types
    ):
        terms = local.to(dtype=torch.float32, copy=True)
        terms.clamp_min_(_EPS).log_().mul_(local)
        entropy = reducer.reduce_over_groups(-terms.sum(), groups, ReduceOp.sum)
        count = value.numel()
        return entropy / math.log(count) if count > 1 else entropy

    count = torch.tensor(local.numel(), device=local.device, dtype=_ACC_DTYPE)

    if name == StatOpName.COUNT:
        return reducer.reduce_over_groups(count, groups, ReduceOp.sum)
    if name in (StatOpName.SUM, StatOpName.MEAN, StatOpName.RMS):
        total = (
            torch.linalg.vector_norm(local, ord=2, dtype=_ACC_DTYPE).square()
            if name == StatOpName.RMS
            else local.sum(dtype=_ACC_DTYPE)
        )
        packed = reducer.reduce_over_groups(
            torch.stack((total, count)), groups, ReduceOp.sum
        )
        global_total, global_count = packed[0], packed[1].clamp_min(1.0)
        if name == StatOpName.SUM:
            return global_total.float()
        if name == StatOpName.MEAN:
            return (global_total / global_count).float()
        return (global_total / global_count).sqrt().float()
    if name in (StatOpName.STD, StatOpName.VAR, StatOpName.KURTOSIS, StatOpName.CV):
        # Two collectives, not one, and deliberately so. The raw-moment form
        # (var = Sx2/n - mean**2, and the four-term expansion for the fourth
        # moment) has every term O(mean**k) while the answer is O(std**k), so a
        # mean much larger than the spread cancels the result away: measured,
        # kurtosis flips sign at mean/std ~ 1e4 and std loses 2% at ~1e7, even
        # accumulating in fp64. Centering fixes it, and centering needs the
        # GLOBAL mean before the residuals can be summed.
        packed = reducer.reduce_over_groups(
            torch.stack((local.sum(dtype=_ACC_DTYPE), count)), groups, ReduceOp.sum
        )
        global_count = packed[1].clamp_min(1.0)
        mean = packed[0] / global_count
        centered = _centered_residual(local, mean)
        orders = (2, 4) if name == StatOpName.KURTOSIS else (2,)
        moments = reducer.reduce_over_groups(
            torch.stack(
                [
                    torch.linalg.vector_norm(centered, ord=k, dtype=_ACC_DTYPE).pow(k)
                    for k in orders
                ]
            ),
            groups,
            ReduceOp.sum,
        )
        var = (moments[0] / global_count).clamp_min(0.0)
        if name == StatOpName.VAR:
            return var.float()
        if name == StatOpName.STD:
            return var.sqrt().float()
        if name == StatOpName.CV:
            return (var.sqrt() / mean.abs().clamp_min(_EPS)).float()
        m4 = moments[1] / global_count
        return (m4 / var.square().clamp_min(_EPS**2)).float()
    if name == StatOpName.NORM:
        p = 2.0 if op.p is None else float(op.p)
        if p == 0:
            local_count = torch.count_nonzero(local).to(_ACC_DTYPE)
            return reducer.reduce_over_groups(local_count, groups, ReduceOp.sum)
        if p == float("inf"):
            # Uneven sharding can leave a rank with an empty local shard.
            local_value = (
                local.abs().amax().to(_ACC_DTYPE)
                if local.numel()
                else torch.zeros((), device=local.device, dtype=_ACC_DTYPE)
            )
            return reducer.reduce_over_groups(local_value, groups, ReduceOp.max)
        if p == float("-inf"):
            # min |x|; an empty shard contributes the identity of min.
            local_value = (
                local.abs().amin().to(_ACC_DTYPE)
                if local.numel()
                else torch.full((), float("inf"), device=local.device, dtype=_ACC_DTYPE)
            )
            return reducer.reduce_over_groups(local_value, groups, ReduceOp.min)
        if p < 0:
            # vector_norm rejects an empty tensor for a negative order; the
            # power sum has an identity (0), and |x| == 0 gives inf, as it should.
            local_power = local.abs().to(_ACC_DTYPE).pow(p).sum()
        else:
            local_power = torch.linalg.vector_norm(local, ord=p, dtype=_ACC_DTYPE).pow(
                p
            )
        global_power = reducer.reduce_over_groups(local_power, groups, ReduceOp.sum)
        return global_power.clamp_min(0.0).pow(1.0 / p)
    if name in (StatOpName.MIN, StatOpName.MAX, StatOpName.ABS_MAX):
        if local.numel() == 0:
            fill = {"min": float("inf"), "max": float("-inf"), "abs_max": 0.0}[name]
            local_value = torch.full((), fill, device=local.device, dtype=_ACC_DTYPE)
        else:
            local_value = apply_stat_op(local, axis, op).to(_ACC_DTYPE)
        reduce_op = (
            ReduceOp.max
            if name in (StatOpName.MAX, StatOpName.ABS_MAX)
            else ReduceOp.min
        )
        return reducer.reduce_over_groups(local_value, groups, reduce_op)
    if name in (StatOpName.MAX_MIN, StatOpName.MAX_MIN_RATIO, StatOpName.MAX_VIO):
        local_max = (
            local.max().to(_ACC_DTYPE)
            if local.numel()
            else torch.full((), float("-inf"), device=local.device, dtype=_ACC_DTYPE)
        )
        global_max = reducer.reduce_over_groups(local_max, groups, ReduceOp.max)
        if name == StatOpName.MAX_VIO:
            packed = reducer.reduce_over_groups(
                torch.stack((local.sum(dtype=_ACC_DTYPE), count)),
                groups,
                ReduceOp.sum,
            )
            mean = packed[0] / packed[1].clamp_min(1.0)
            return ((global_max - mean) / mean.clamp_min(1.0)).float()
        local_min = (
            local.min().to(_ACC_DTYPE)
            if local.numel()
            else torch.full((), float("inf"), device=local.device, dtype=_ACC_DTYPE)
        )
        global_min = reducer.reduce_over_groups(local_min, groups, ReduceOp.min)
        if name == StatOpName.MAX_MIN:
            return (global_max - global_min).float()
        return torch.where(
            global_min > 0,
            global_max / global_min,
            torch.full_like(global_max, float("inf")),
        ).float()
    if name in (StatOpName.NAN_COUNT, StatOpName.INF_COUNT, StatOpName.NONZERO_COUNT):
        return reducer.reduce_over_groups(
            apply_stat_op(local, axis, op), groups, ReduceOp.sum
        )
    if name == StatOpName.POSITIVE_RATIO:
        packed = reducer.reduce_over_groups(
            torch.stack(((local > 0).sum(dtype=_ACC_DTYPE), count)),
            groups,
            ReduceOp.sum,
        )
        return packed[0] / packed[1].clamp_min(1.0)
    if name in (StatOpName.ACTABS, StatOpName.ACTREL):
        if op.threshold is None:
            raise ValueError(f"{name} requires a threshold")
        abs_local = local.abs()
        if name == StatOpName.ACTREL:
            packed = reducer.reduce_over_groups(
                torch.stack((count, abs_local.sum(dtype=_ACC_DTYPE))),
                groups,
                ReduceOp.sum,
            )
            global_count = packed[0].clamp_min(1.0)
            bar = float(op.threshold) * packed[1] / global_count
            hits = _threshold_hits(abs_local, bar, op.cmp)
            global_hits = reducer.reduce_over_groups(hits, groups, ReduceOp.sum)
            return (global_hits / global_count).float()
        bar = torch.tensor(float(op.threshold), device=local.device, dtype=_ACC_DTYPE)
        hits = _threshold_hits(abs_local, bar, op.cmp)
        packed = reducer.reduce_over_groups(
            torch.stack((count, hits)), groups, ReduceOp.sum
        )
        return (packed[1] / packed[0].clamp_min(1.0)).float()
    if name in (StatOpName.FINITE_RATIO, StatOpName.ZERO_RATIO):
        # An empty local shard has ratio NaN (0/0); its contribution is 0.
        local_num = (
            apply_stat_op(local, axis, StatOpConfig(op=name)) * count
            if local.numel()
            else torch.zeros((), device=local.device, dtype=_ACC_DTYPE)
        )
        packed = reducer.reduce_over_groups(
            torch.stack((local_num, count)), groups, ReduceOp.sum
        )
        return packed[0] / packed[1].clamp_min(1.0)

    return None


def apply_post_stats(records: list[MetricRecord]) -> list[MetricRecord]:
    """Finish ``[reduce..., stat...]`` records once their reductions resolved.

    Runs right after ``finalize_pending_reductions`` (the caller pairs them),
    so the values here are plain reduced tensors — the stats are rank-local
    arithmetic, no collectives. Purely positional work: every rank computes
    the same thing, so nothing here can deadlock.
    """
    for index, record in enumerate(records):
        if not record.post_stats or record.pending_stages:
            continue
        value = _local_tensor(record.value)
        axes = record.post_axes
        for step in record.post_stats:
            if step.stat is None:
                raise ValueError("Metric step kind='stat' requires a stat config")
            value = apply_stat_op(value, axes, step.stat)
            if step.stat.dim is not None:
                axes = _remaining_axes(axes, axes.resolve(step.stat.dim))
        records[index] = replace(
            record, value=value.to(torch.float32), post_stats=(), post_axes=AxisSpec()
        )
    return records


# (metric, stat, dim) already warned about; process-wide so the warning comes
# once per run, not once per capture (pipelines are rebuilt for every sample).
_WARNED_FULL_TENSOR_FALLBACKS: set[tuple[str, str, str]] = set()


def _warn_full_tensor_fallback(
    metric: MetricConfig, op: StatOpConfig, value: torch.Tensor
) -> None:
    """Say once that a ``partial_stats`` metric silently gathers the full tensor.

    The fallback keeps the value exact, but costs a full-tensor gather per
    capture: under sequence/tensor parallelism that is a whole activation per
    module, microbatch and logged step (e.g. ``norm`` with a ``dim``).
    """
    key = (metric.name, op.op, str(op.dim))
    if key in _WARNED_FULL_TENSOR_FALLBACKS:
        return
    _WARNED_FULL_TENSOR_FALLBACKS.add(key)
    placements = tuple(str(p) for p in getattr(value, "placements", ()))
    logger.warning(
        "Monitoring metric '%s': stat '%s' (dim=%s) has no shard-local path for "
        "a %s with placements %s, so every capture gathers the full tensor "
        "(%d elements) on each rank. Drop the metric for sharded tensors, or set "
        "dtensor_mode='full_tensor' on it to accept the gather explicitly.",
        metric.name,
        op.op,
        op.dim,
        type(value).__name__,
        placements,
        value.numel(),
    )


def _validate_reduce_mode(metric: MetricConfig) -> None:
    """Reject the unimplemented offline reduction pipeline."""
    if metric.reduce.mode != ReduceMode.online:
        raise ValueError(
            f"Metric '{metric.name}' requests reduce mode "
            f"'{metric.reduce.mode}', but only ReduceMode.online is "
            "supported: the offline reduction pipeline is untested."
        )


@dataclass(slots=True)
class MetricPipeline:
    """Runs one metric's stat steps and distributed reductions over a value."""

    metric: MetricConfig
    reducer: ParallelReducer

    def collect_leading_reductions(self) -> tuple[ReduceStageConfig, ...]:
        """Return the reduction stages before the first statistic, if any."""
        stages = []
        for step in self.metric.effective_steps():
            if step.kind != MetricStepKind.reduce:
                break
            if step.reduce is None:
                raise ValueError("Metric step kind='reduce' requires a reduce config")
            stages.append(step.reduce)
        return tuple(stages)

    def compute_after_reductions(
        self,
        value: torch.Tensor,
        axes: AxisSpec,
        *,
        copies: frozenset[int] | None = None,
    ) -> torch.Tensor:
        """Finish a reduce-first metric after its shared leading stages."""
        remaining = self.metric.effective_steps()[
            len(self.collect_leading_reductions()) :
        ]
        result = self._compute(
            value, axes, defer_full_tensor=False, steps=remaining, copies=copies
        )
        assert result is not None
        return result

    def compute(self, value: MetricValue, axes: AxisSpec) -> torch.Tensor:
        """Run every step inline, collectives included."""
        result = self._compute(value, axes, defer_full_tensor=False)
        assert result is not None
        return result

    def _compute(
        self,
        value: MetricValue,
        axes: AxisSpec,
        *,
        defer_full_tensor: bool,
        steps: tuple[MetricStepConfig, ...] | None = None,
        copies: frozenset[int] | None = None,
    ) -> torch.Tensor | None:
        steps = self.metric.effective_steps() if steps is None else steps
        self._guard_double_reduction(
            value, tuple(step.reduce for step in steps if step.reduce is not None)
        )
        current: MetricValue = value
        current_axes = axes
        for step in steps:
            if step.kind == MetricStepKind.stat:
                stat_result = self._apply_stat_step(
                    current, current_axes, step, defer_full_tensor=defer_full_tensor
                )
                if stat_result is None:
                    return None
                current, current_axes = stat_result
            elif step.kind == MetricStepKind.reduce:
                if step.reduce is None:
                    raise ValueError(
                        "Metric step kind='reduce' requires a reduce config"
                    )
                if defer_full_tensor and isinstance(
                    current, self.reducer.ragged_tensor_types
                ):
                    return None
                local = _local_tensor(
                    current,
                    dtype=torch.float32,
                    shard_stats=self.reducer.shard_stats,
                )
                if copies is None:
                    current = self.reducer.reduce_stage(local, step.reduce)
                else:
                    current, copies = self.reducer.reduce_counting_copies(
                        local, (step.reduce,), copies
                    )
            else:
                raise ValueError(f"Unknown metric pipeline step kind: {step.kind}")
        # full_tensor when a DTensor survived the pipeline: nothing reduced it,
        # so to_local() would publish one shard under a whole-tensor tag. A
        # surviving ragged shard tensor is gathered for the same reason.
        if defer_full_tensor and isinstance(current, self.reducer.ragged_tensor_types):
            return None
        return _local_tensor(
            current,
            full_tensor=isinstance(current, self.reducer.sharded_tensor_types),
            dtype=torch.float32,
            shard_stats=self.reducer.shard_stats,
        )

    def compute_stats(
        self,
        value: MetricValue,
        axes: AxisSpec,
        *,
        full_value: torch.Tensor | None = None,
        defer_full_tensor: bool = False,
    ) -> (
        tuple[
            torch.Tensor,
            tuple[ReduceStageConfig, ...],
            tuple[MetricStepConfig, ...],
        ]
        | None
    ):
        """Run what can run now; return the deferred remainder of the chain.

        With ``defer_full_tensor``, return None if a ragged input needs the
        full tensor. The caller can finish shard-only metrics first, then
        rerun this metric with a shared ``full_value``.

        Three shapes, in the order they are tried:

        - ``[stat..., reduce...]`` (the shorthand form): stats run eagerly,
          the trailing reductions come back as pending stages. They act on
          the computed scalar, so ``finalize_pending_reductions`` fuses them
          across records — one collective per (group, op) chain.
        - ``[reduce..., stat...]``: ordinary tensors defer the reductions
          and post-stats until batch_end, allowing reductions from different
          records to fuse. Sharded tensors return None.
        - anything else (a reduce between stats): the fully inline
          ``compute``, nothing deferred.
        """
        steps = self.metric.effective_steps()
        numel = value.numel() if isinstance(value, torch.Tensor) else 1
        for step in steps:
            if step.stat is None or step.stat.op not in (
                StatOpName.ABS,
                StatOpName.QUANTILE,
                StatOpName.MIN_MEDIAN_RATIO,
                StatOpName.GINI,
            ):
                break
            if step.stat.op != StatOpName.ABS:
                _assert_order_stat_size(step.stat.op, numel, self.metric.name)
                break
        self._guard_double_reduction(
            value, tuple(step.reduce for step in steps if step.reduce is not None)
        )
        if (
            len(steps) == 2
            and isinstance(value, self.reducer.ragged_tensor_types)
            and self.metric.dtensor_mode == DTensorMode.partial_stats
            and steps[0].stat is not None
            and steps[0].stat.op == StatOpName.NORM
            and steps[0].stat.dim is not None
            and steps[1].stat is not None
            and steps[1].stat.op == StatOpName.MEAN
            and steps[1].stat.dim is None
        ):
            norms = _partial_stat_op(value, axes, steps[0].stat, self.reducer)
            if norms is not None:
                return norms.mean(dtype=torch.float32), (), ()

        if (
            steps
            and steps[0].kind == MetricStepKind.reduce
            and isinstance(value, self.reducer.sharded_tensor_types)
        ):
            if self.metric.reduce.mode != ReduceMode.online:
                raise ValueError(
                    f"Metric '{self.metric.name}' requests reduce mode "
                    f"'{self.metric.reduce.mode}', but only ReduceMode.online "
                    "is supported for reduce-first metrics of sharded tensors."
                )
            assert full_value is None
            return None

        current_value = full_value if full_value is not None else value
        pending_from = next(
            (
                index
                for index, step in enumerate(steps)
                if step.kind == MetricStepKind.reduce
            ),
            len(steps),
        )
        if any(step.kind != MetricStepKind.reduce for step in steps[pending_from:]):
            if self.metric.reduce.mode != ReduceMode.online:
                raise ValueError(
                    f"Metric '{self.metric.name}' interleaves a stat after a "
                    "reduce; that requires the online collective and cannot "
                    f"run with reduce mode '{self.metric.reduce.mode}'."
                )
            prefix = next(
                (
                    i
                    for i, step in enumerate(steps)
                    if step.kind != MetricStepKind.reduce
                ),
                len(steps),
            )
            if all(step.kind == MetricStepKind.stat for step in steps[prefix:]):
                # Sharded tensors are routed away above: the copies of their
                # gathered value must be counted once, which these fused
                # deferred collectives cannot do.
                return (
                    _local_tensor(
                        value,
                        full_tensor=isinstance(
                            value, self.reducer.sharded_tensor_types
                        ),
                        dtype=torch.float32,
                        shard_stats=self.reducer.shard_stats,
                    ),
                    tuple(
                        step.reduce
                        for step in steps[:prefix]
                        if step.reduce is not None
                    ),
                    tuple(steps[prefix:]),
                )
            result = self._compute(
                current_value, axes, defer_full_tensor=defer_full_tensor
            )
            return (result, (), ()) if result is not None else None

        current: MetricValue = current_value
        current_axes = axes
        for step in steps[:pending_from]:
            stat_result = self._apply_stat_step(
                current, current_axes, step, defer_full_tensor=defer_full_tensor
            )
            if stat_result is None:
                return None
            current, current_axes = stat_result

        _validate_reduce_mode(self.metric)

        pending = tuple(
            step.reduce for step in steps[pending_from:] if step.reduce is not None
        )
        if defer_full_tensor and isinstance(current, self.reducer.ragged_tensor_types):
            return None
        return (
            _local_tensor(
                current,
                full_tensor=isinstance(current, self.reducer.sharded_tensor_types),
                dtype=torch.float32,
                shard_stats=self.reducer.shard_stats,
            ),
            pending,
            (),
        )

    # Aliased reduce groups expanded to the mesh axes they span; ``None``
    # means "every axis" (world). Single mesh axes expand to themselves.
    # This fork's world mesh names its shard axes "fsdp"/"efsdp" (dp_shard,
    # cp and tp/ep folded in), so groups spanning data-parallel ranks list
    # them alongside the upstream names.
    _GROUP_SHARD_AXES = {
        "batch": frozenset({"dp_replicate", "dp_shard", "fsdp", "efsdp"}),
        "dp": frozenset({"dp_replicate", "dp_shard", "fsdp", "efsdp"}),
        "data": frozenset({"dp_replicate", "dp_shard", "fsdp", "efsdp"}),
        "loss": frozenset({"dp_replicate", "dp_shard", "cp", "fsdp", "efsdp"}),
        "fsdp": frozenset({"dp_shard", "cp", "fsdp", "efsdp"}),
        "sp": frozenset({"tp"}),
        "sequence": frozenset({"tp"}),
        "world": None,
        "global": None,
    }

    def _guard_double_reduction(
        self, value: MetricValue, pending: tuple[ReduceStageConfig, ...]
    ) -> None:
        """Refuse a sum stage over axes ``partial_stats`` already summed.

        ``partial_stats`` reduces over every shard axis of the DTensor's mesh;
        a configured ``sum`` stage over an overlapping group would multiply
        the result by that group's size. mean/max/min over the already-global
        value are numerically no-ops and stay allowed.
        """
        if not pending or not isinstance(value, self.reducer.sharded_tensor_types):
            return
        if self.metric.dtensor_mode != DTensorMode.partial_stats:
            return
        steps = self.metric.effective_steps()
        if bool(steps) and steps[0].kind == MetricStepKind.reduce:
            return
        shard_axes = set(_dtensor_reduce_groups(value, self.reducer.shard_stats) or ())
        if not shard_axes:
            return
        for stage in pending:
            if stage.op != ReduceOp.sum:
                continue
            groups = stage.group if isinstance(stage.group, tuple) else (stage.group,)
            for group in groups:
                axes = self._GROUP_SHARD_AXES.get(
                    group, None if group is None else frozenset({group})
                )
                if axes is None or axes & shard_axes:
                    raise ValueError(
                        f"Metric '{self.metric.name}': dtensor_mode="
                        f"partial_stats already summed over the shard axes "
                        f"{sorted(shard_axes)}, and the reduce stage over "
                        f"group {stage.group!r} would sum them again, "
                        "multiplying the result by the group size. Drop the "
                        "stage or use dtensor_mode=full_tensor."
                    )

    def _apply_stat_step(
        self,
        value: MetricValue,
        axes: AxisSpec,
        step: MetricStepConfig,
        *,
        defer_full_tensor: bool = False,
    ) -> tuple[torch.Tensor, AxisSpec] | None:
        """Apply one stat step and drop the reduced axes from the spec."""
        if step.stat is None:
            raise ValueError("Metric step kind='stat' requires a stat config")
        result = self._apply_stat(
            value, axes, step.stat, defer_full_tensor=defer_full_tensor
        )
        if result is None:
            return None
        if step.stat.dim is None:
            return result, axes
        return result, _remaining_axes(axes, axes.resolve(step.stat.dim))

    def _apply_stat(
        self,
        value: MetricValue,
        axes: AxisSpec,
        op: StatOpConfig,
        *,
        defer_full_tensor: bool = False,
    ) -> torch.Tensor | None:
        if not isinstance(value, self.reducer.sharded_tensor_types):
            return apply_stat_op(_local_tensor(value), axes, op)
        mode = self.metric.dtensor_mode
        if (
            defer_full_tensor
            and mode == DTensorMode.full_tensor
            and isinstance(value, self.reducer.ragged_tensor_types)
        ):
            return None
        if mode == DTensorMode.partial_stats:
            result = _partial_stat_op(value, axes, op, self.reducer)
            if result is not None:
                return result
            _warn_full_tensor_fallback(self.metric, op, value)
            if defer_full_tensor and isinstance(
                value, self.reducer.ragged_tensor_types
            ):
                return None
            # Ops the partial path cannot handle (e.g. axis-wise norms) fall
            # back to the exact full tensor instead of silently reporting
            # shard-local values under a global tag.
            mode = DTensorMode.full_tensor
        return apply_stat_op(
            _local_tensor(
                value,
                full_tensor=mode == DTensorMode.full_tensor,
                shard_stats=self.reducer.shard_stats,
            ),
            axes,
            op,
        )
