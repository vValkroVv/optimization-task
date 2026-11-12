import logging
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed.tensor._random as random
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._ops._math_ops import _NormPartial
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.optim.optimizer import (
    _foreach_supported_types as _optim_foreach_supported_types,
)
from torch.utils._foreach_utils import (
    _foreach_supported_types as _util_foreach_supported_types,
)

logger = logging.getLogger("gfsdp")


@dataclass(frozen=True)
class RaggedShard:
    local_numel: int
    global_offset: int
    shard_numels: tuple[int, ...]

    def __repr__(self) -> str:
        return f"RaggedShard(local_numel={self.local_numel}, global_offset={self.global_offset})"

    def __str__(self) -> str:
        return f"RS({self.local_numel}, {self.global_offset})"


def get_ragged_shard_axis(spec: DTensorSpec) -> int:
    """
    Search RaggedShard axis in the DTensorSpec
    """
    ragged_shard_axis = None
    for axis, placement in enumerate(spec.placements):
        if not isinstance(placement, RaggedShard):
            continue
        if ragged_shard_axis is not None:
            raise AssertionError(
                "Expected only one RaggedShard axis in the DTensorSpec, but got multiple"
            )
        ragged_shard_axis = axis
    if ragged_shard_axis is None:
        raise AssertionError("Expected one RaggedShard placement, but got none")
    return ragged_shard_axis


def get_ragged_shard_placement(spec: DTensorSpec) -> RaggedShard:
    return cast("RaggedShard", spec.placements[get_ragged_shard_axis(spec)])


def replace_ragged_shard(
    spec: DTensorSpec, new_placement: Placement
) -> tuple[Placement, ...]:
    """
    Replace the RaggedShard placement in the DTensorSpec with a new placement.

    After All-Gather, when all shards are gathered, the params are replicated across the mesh,
    so the RaggedShard placement should be replaced with Replicate placement.
    """
    placements = list(spec.placements)
    placements[get_ragged_shard_axis(spec)] = new_placement
    return tuple(placements)


def _ragged_flat_start(tensor: "RaggedShardDTensor") -> int:
    """Flat position of this rank's first element in the global tensor.

    ``RaggedShard.global_offset`` is relative to this rank's TP/EP partition, so
    the partition's own offset is added on top.
    """
    spec = tensor._spec
    replicated = replace_ragged_shard(spec, Replicate())
    shard_placements = [p for p in replicated if p.__class__ is Shard]
    partition_offset = 0
    if shard_placements:
        assert len(shard_placements) == 1
        _, offsets = compute_local_shape_and_global_offset(
            tensor.shape, spec.mesh, replicated
        )
        shard_dim = shard_placements[0].dim
        if shard_dim < 0:
            shard_dim += tensor.ndim
        stride = 1
        for d, size in enumerate(tensor.shape):
            if d != shard_dim:
                stride *= size
        partition_offset = offsets[shard_dim] * stride
    return partition_offset + get_ragged_shard_placement(spec).global_offset


def _wrap_partial(
    spec: DTensorSpec, local_value: torch.Tensor, partial: Placement
) -> DTensor:
    return DTensor(
        local_value,
        DTensorSpec(
            spec.mesh,
            tuple(
                partial if isinstance(placement, RaggedShard | Shard) else placement
                for placement in spec.placements
            ),
            TensorMeta(local_value.shape, local_value.stride(), local_value.dtype),
        ),
        requires_grad=False,
    )


def _get_norm_placement(norm_type: float) -> Placement:
    if norm_type == 0:
        return Partial("sum")
    if norm_type == float("inf"):
        return Partial("max")
    if norm_type == float("-inf"):
        return Partial("min")
    return _NormPartial(norm_type)


def _compute_local_vector_norm(
    local: torch.Tensor, norm_type: float, dtype: torch.dtype | None = None
) -> torch.Tensor:
    if local.numel() == 0:
        identity = float("inf") if norm_type < 0 else 0.0
        return local.new_full((), identity, dtype=dtype or local.dtype)
    return torch.ops.aten.linalg_vector_norm.default(local, norm_type, dtype=dtype)


def reduce_flat_segments(
    value: "RaggedShardDTensor",
    num_segments: int,
    *,
    reduce_op: str,
    identity: float,
    transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    sum_dtype: torch.dtype = torch.float64,
) -> DTensor:
    """Reduce flat expert segments locally and defer the cross-rank step.

    ``transform`` sees read-only rows and their global segment indices.
    Boundary rows may contain fewer elements than a whole segment.
    """
    spec = value._spec
    numel = spec.tensor_meta.shape.numel()
    if num_segments <= 0 or numel % num_segments:
        raise ValueError(
            f"{num_segments} segments do not evenly split {numel} elements"
        )
    for placement in spec.placements:
        if isinstance(placement, RaggedShard) or placement.__class__ is Replicate:
            continue
        if placement.__class__ is Shard and placement.dim == 0:
            continue
        raise ValueError(
            "flat-segment reductions need every shard to cover one contiguous "
            f"range of the flattening (Replicate or Shard(0)), got {placement}"
        )
    assert not value.requires_grad
    if transform is not None and reduce_op != "sum":
        raise ValueError("flat-segment transforms require a sum reduction")

    segment_numel = numel // num_segments
    local = value._local_tensor.reshape(-1)
    out = local.new_full((num_segments,), identity, dtype=torch.float64)
    if local.numel():
        start = _ragged_flat_start(value)
        first = start // segment_numel

        def reduce_rows(rows: torch.Tensor, index: int) -> torch.Tensor:
            if transform is not None:
                segments = torch.arange(
                    index, index + rows.shape[0], device=local.device
                )
                rows = transform(rows, segments)
            if reduce_op == "sum":
                return rows.sum(dim=1, dtype=sum_dtype)
            if reduce_op == "min":
                return rows.amin(dim=1).to(torch.float64)
            return rows.amax(dim=1).to(torch.float64)

        position = 0
        lead = start % segment_numel
        if lead:
            position = min(segment_numel - lead, local.numel())
            out[first] = reduce_rows(local[:position].view(1, -1), first)[0]
            first += 1

        whole = (local.numel() - position) // segment_numel
        if whole:
            end = position + whole * segment_numel
            out[first : first + whole] = reduce_rows(
                local[position:end].view(whole, segment_numel), first
            )
            position = end
            first += whole

        if position < local.numel():
            out[first] = reduce_rows(local[position:].view(1, -1), first)[0]
    return _wrap_partial(spec, out, Partial(reduce_op))


def local_squared_norms(value: "RaggedShardDTensor", dim: int) -> torch.Tensor:
    """Local squared contributions to each row or column norm."""
    if value.ndim < 1 or dim not in (0, value.ndim - 1):
        raise ValueError(f"Unsupported shape {tuple(value.shape)} or norm dim {dim}")
    if any(
        placement.__class__ is Shard and placement.dim != 0
        for placement in value.placements
    ):
        raise ValueError("Flat norm reductions require contiguous shard ranges")

    local = value._local_tensor.reshape(-1)
    if value.ndim == 1:
        return torch.linalg.vector_norm(local, ord=2, dtype=torch.float32).square()

    by_row = dim == value.ndim - 1
    nrows = math.prod(value.shape[:-1]) if by_row else value.shape[0]
    ncols = value.shape[-1] if by_row else math.prod(value.shape[1:])
    output_shape = value.shape[:-1] if by_row else value.shape[1:]
    out = local.new_zeros((nrows if by_row else ncols,), dtype=torch.float32)
    if not local.numel() or not ncols:
        return out.view(output_shape)

    start = _ragged_flat_start(value)
    row, lead = divmod(start, ncols)
    position = min(ncols - lead, local.numel()) if lead else 0
    if lead:
        head = local[:position]
        if by_row:
            out[row] = torch.linalg.vector_norm(
                head, ord=2, dtype=torch.float32
            ).square()
        else:
            out[lead : lead + position] += head.float().square()
        row += 1

    whole = (local.numel() - position) // ncols
    if whole:
        end = position + whole * ncols
        rows = local[position:end].view(whole, ncols)
        if by_row:
            out[row : row + whole] = torch.linalg.vector_norm(
                rows, ord=2, dim=1, dtype=torch.float32
            ).square()
        else:
            for chunk in rows.split(4096):
                out += torch.linalg.vector_norm(
                    chunk, ord=2, dim=0, dtype=torch.float32
                ).square()
        position = end
        row += whole

    if position < local.numel():
        tail = local[position:]
        if by_row:
            out[row] = torch.linalg.vector_norm(
                tail, ord=2, dtype=torch.float32
            ).square()
        else:
            out[: tail.numel()] += tail.float().square()
    return out.view(output_shape)


class RaggedShardDTensor(torch.Tensor):
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: DTensorSpec,
        *,
        requires_grad: bool,
    ) -> "RaggedShardDTensor":
        if local_tensor.requires_grad and not requires_grad:
            warnings.warn(
                "To construct DTensor from torch.Tensor, it's recommended to "
                "use local_tensor.detach() and make requires_grad consistent.",
                stacklevel=1,
            )

        # new method instruct wrapper tensor from local_tensor and add
        # placement spec, it does not do actual distribution
        assert spec.tensor_meta is not None, "TensorMeta should not be None!"
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            spec.tensor_meta.shape,
            strides=spec.tensor_meta.stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )

        r._spec = spec
        r._local_tensor = local_tensor
        return r

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"RaggedShardDTensor(local_tensor={self._local_tensor}, "
            f"device_mesh={self._spec.mesh}, placements={self._spec.placements})"
        )

    def __tensor_flatten__(self) -> Any:
        return ["_local_tensor"], (self._spec, self.requires_grad)

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Any, flatten_spec: Any, outer_size: Any, outer_stride: Any
    ) -> Any:
        assert flatten_spec is not None, (
            "Expecting spec to be not None from `__tensor_flatten__` return value!"
        )
        local_tensor = inner_tensors["_local_tensor"]
        spec, requires_grad = flatten_spec
        unflatten_tensor_meta = TensorMeta(
            shape=outer_size,
            stride=outer_stride,
            dtype=spec.tensor_meta.dtype,
        )
        unflatten_spec = DTensorSpec(
            spec.mesh,
            spec.placements,
            tensor_meta=unflatten_tensor_meta,
        )
        return RaggedShardDTensor(
            local_tensor,
            unflatten_spec,
            requires_grad=requires_grad,
        )

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]  # noqa: PLR0911, PLR0912, PLR0915
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        if func == torch.ops.aten.detach.default:
            (arg,) = args
            assert len(kwargs) == 0
            return RaggedShardDTensor(
                arg._local_tensor,
                arg._spec,
                requires_grad=False,
            )
        elif func == torch.ops.aten.clone.default:
            (arg,) = args
            assert len(kwargs) == 0
            return RaggedShardDTensor(
                arg._local_tensor.clone(),
                arg._spec,
                requires_grad=False,
            )
        elif func == torch.ops.aten._to_copy.default:
            (arg,) = args
            local_tensor = func(arg._local_tensor, **kwargs)
            tensor_meta = TensorMeta(
                shape=arg._spec.tensor_meta.shape,
                stride=arg._spec.tensor_meta.stride,
                dtype=local_tensor.dtype,
            )
            spec = DTensorSpec(
                mesh=arg._spec.mesh,
                placements=arg._spec.placements,
                tensor_meta=tensor_meta,
            )
            return RaggedShardDTensor(
                local_tensor,
                spec,
                requires_grad=arg.requires_grad,
            )
        elif func == torch.ops.aten.copy_.default:
            arg1, arg2 = args
            func(arg1._local_tensor, arg2._local_tensor, **kwargs)
            return arg1
        elif func in (
            torch.ops.aten.empty_like.default,
            torch.ops.aten.zeros_like.default,
            torch.ops.aten.new_zeros.default,
            torch.ops.aten.new_empty.default,
        ):
            arg, *args = args
            return RaggedShardDTensor(
                func(arg._local_tensor, *args, **kwargs),
                arg._spec,
                requires_grad=arg.requires_grad,
            )
        elif func in (
            torch.ops.aten.log_.default,
            torch.ops.aten.zero_.default,
            torch.ops.aten.mul_.Scalar,
            torch.ops.aten.mul_.Tensor,
            torch.ops.aten.fill_.Scalar,
            torch.ops.aten.exp_.default,
            torch.ops.aten.erfinv_.default,
            torch.ops.aten.clamp_.default,
            torch.ops.aten.add_.Scalar,
            torch.ops.aten.add_.Tensor,
        ):
            arg, *args = args
            func(
                arg._local_tensor,
                *(
                    arg._local_tensor if isinstance(arg, RaggedShardDTensor) else arg
                    for arg in args
                ),
                **kwargs,
            )
            return arg
        elif func in (
            torch.ops.aten.lt.Scalar,
            torch.ops.aten.gt.Scalar,
            torch.ops.aten.__or__.Tensor,
        ):
            arg, *rest = args
            return func(
                arg._local_tensor if isinstance(arg, RaggedShardDTensor) else arg,
                *(
                    x._local_tensor if isinstance(x, RaggedShardDTensor) else x
                    for x in rest
                ),
                **(kwargs or {}),
            )
        elif func in (torch.ops.aten.where.self,):
            condition, on_true, on_false = args
            arg = on_true if isinstance(on_true, RaggedShardDTensor) else on_false
            return RaggedShardDTensor(
                func(
                    condition,
                    on_true._local_tensor
                    if isinstance(on_true, RaggedShardDTensor)
                    else on_true,
                    on_false._local_tensor
                    if isinstance(on_false, RaggedShardDTensor)
                    else on_false,
                    **(kwargs or {}),
                ),
                arg._spec,
                requires_grad=arg.requires_grad,
            )
        elif func in (
            torch.ops.aten.exp.default,
            torch.ops.aten.expm1.default,
            torch.ops.aten.log.default,
            torch.ops.aten.neg.default,
            torch.ops.aten.abs.default,
            torch.ops.aten.sqrt.default,
            torch.ops.aten.pow.Tensor_Scalar,
            torch.ops.aten.clamp.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add.Scalar,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.div.Tensor,
            torch.ops.aten.div.Scalar,
        ):
            arg = next(x for x in args if isinstance(x, RaggedShardDTensor))
            for other in args:
                if isinstance(other, RaggedShardDTensor) and (
                    other._spec.mesh != arg._spec.mesh
                    or other._spec.placements != arg._spec.placements
                    or other.shape != arg.shape
                ):
                    raise NotImplementedError(
                        f"RaggedShardDTensor dispatch for {func} requires matching "
                        "meshes, placements (including shard offsets), and global shapes."
                    )
            return RaggedShardDTensor(
                func(
                    *(
                        x._local_tensor if isinstance(x, RaggedShardDTensor) else x
                        for x in args
                    ),
                    **kwargs,
                ),
                arg._spec,
                requires_grad=arg.requires_grad,
            )
        elif func in (torch.ops.aten.mean.default, torch.ops.aten.sum.default):
            (arg,) = args
            assert not arg.requires_grad
            local_value = arg._local_tensor.sum(**kwargs)
            if func == torch.ops.aten.mean.default:
                # Divide by the global numel: the partial sums then add up
                # to the exact mean, whatever the shard sizes.
                local_value = local_value / arg._spec.tensor_meta.shape.numel()
            return _wrap_partial(arg._spec, local_value, Partial("sum"))
        elif func in (
            torch.ops.aten.max.default,
            torch.ops.aten.amax.default,
            torch.ops.aten.min.default,
            torch.ops.aten.amin.default,
        ):
            arg, *rest = args
            dim = rest[0] if rest else kwargs.get("dim", [])
            keepdim = rest[1] if len(rest) > 1 else kwargs.get("keepdim", False)
            if dim or keepdim:
                raise NotImplementedError(
                    f"RaggedShardDTensor dispatch for {func} supports only "
                    "full reductions (no dim/keepdim)."
                )
            assert not arg.requires_grad
            is_max = func in (
                torch.ops.aten.max.default,
                torch.ops.aten.amax.default,
            )
            local = arg._local_tensor
            if local.numel() == 0:
                # A rank can own an empty ragged slice; max/min error on empty
                # inputs, so contribute the reduction identity instead.
                local_value = local.new_full(
                    (), float("-inf") if is_max else float("inf")
                )
            else:
                local_value = func(local)
            return _wrap_partial(
                arg._spec, local_value, Partial("max" if is_max else "min")
            )
        elif func == torch.ops.aten.linalg_vector_norm.default:
            arg = args[0]
            norm_type = float(args[1] if len(args) > 1 else kwargs.get("ord", 2))
            dim = args[2] if len(args) > 2 else kwargs.get("dim")
            keepdim = args[3] if len(args) > 3 else kwargs.get("keepdim", False)
            if (dim is not None and len(dim) > 0) or keepdim:
                raise NotImplementedError(
                    f"RaggedShardDTensor dispatch for {func} supports only "
                    "full reductions (no dim/keepdim)."
                )
            assert not arg.requires_grad
            local_value = _compute_local_vector_norm(
                arg._local_tensor, norm_type, dtype=kwargs.get("dtype")
            )
            return _wrap_partial(arg._spec, local_value, _get_norm_placement(norm_type))
        elif func in (torch.ops.aten.normal_.default, torch.ops.aten.uniform_.default):
            from ._random import _fill_random

            arg, *args = args
            _fill_random(arg, func, args, kwargs)
            return arg
        elif func == torch.ops.aten.rand_like.default:
            arg, *args = args
            assert random._rng_tracker is not None
            rng_context = random._rng_tracker._distribute_region(
                DTensorSpec(
                    arg._spec.mesh,
                    replace_ragged_shard(arg._spec, Shard(0)),
                    arg._spec.tensor_meta,
                )
            )
            with rng_context:
                return RaggedShardDTensor(
                    func(arg._local_tensor, *args, **kwargs),
                    arg._spec,
                    requires_grad=arg.requires_grad,
                )
        elif func == torch.ops.aten._foreach_sqrt.default:
            (args,) = args
            local_args = func([arg._local_tensor for arg in args], **kwargs)
            assert all(not arg.requires_grad for arg in args)
            return [
                RaggedShardDTensor(local_arg, arg._spec, requires_grad=False)
                for arg, local_arg in zip(args, local_args, strict=True)
            ]
        elif func == torch.ops.aten._foreach_norm.Scalar:
            args, norm_type = args
            norm_type = float(norm_type)
            local_tensors = [arg._local_tensor for arg in args]
            nonempty = [i for i, local in enumerate(local_tensors) if local.numel()]
            local_args: list[torch.Tensor | None] = [None] * len(local_tensors)
            if nonempty:
                fused = func([local_tensors[i] for i in nonempty], norm_type, **kwargs)
                for i, norm in zip(nonempty, fused, strict=True):
                    local_args[i] = norm
            for i, local in enumerate(local_tensors):
                if local_args[i] is None:
                    local_args[i] = _compute_local_vector_norm(
                        local, norm_type, dtype=kwargs.get("dtype")
                    )
            assert all(not arg.requires_grad for arg in args)
            return [
                _wrap_partial(arg._spec, local_arg, _get_norm_placement(norm_type))
                for arg, local_arg in zip(args, local_args, strict=True)
            ]
        elif func == torch.ops.aten._foreach_zero_.default:
            (args,) = args
            func([arg._local_tensor for arg in args], **kwargs)
        elif func in (
            torch.ops.aten._foreach_add_.Scalar,
            torch.ops.aten._foreach_mul_.Scalar,
            torch.ops.aten._foreach_div_.Scalar,
        ):
            args, scale = args
            func([arg._local_tensor for arg in args], scale, **kwargs)
        elif func == torch.ops.aten._foreach_div_.ScalarList:
            args, scalar_list = args
            func([arg._local_tensor for arg in args], scalar_list, **kwargs)
        elif func == torch.ops.aten._foreach_mul_.Tensor:
            args, tensor = args
            if isinstance(tensor, DTensor):
                assert tensor._spec.placements == (Replicate(),)
                tensor = tensor._local_tensor
            func([arg._local_tensor for arg in args], tensor, **kwargs)
        elif func == torch.ops.aten._foreach_add_.List:
            args1, args2 = args
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                **kwargs,
            )
        elif func == torch.ops.aten._foreach_lerp_.Scalar:
            args1, args2, scalar = args
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                scalar,
                **kwargs,
            )
        elif func in (
            torch.ops.aten._foreach_addcmul_.Scalar,
            torch.ops.aten._foreach_addcdiv_.Scalar,
        ):
            args1, args2, args3, scalar = args
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                [arg._local_tensor for arg in args3],
                scalar,
                **kwargs,
            )
        elif func == torch.ops.aten._foreach_addcdiv_.ScalarList:
            args1, args2, args3, scalar_list = args
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                [arg._local_tensor for arg in args3],
                scalar_list,
                **kwargs,
            )
        elif func in (
            torch.ops.aten._fused_adam_.default,
            torch.ops.aten._fused_adamw_.default,
        ):
            args1, args2, args3, args4, args5, args6 = args
            assert len(args5) == 0
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                [arg._local_tensor for arg in args3],
                [arg._local_tensor for arg in args4],
                args5,
                args6,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"RaggedShardDTensor dispatch for {func} is not implemented yet."
            )

    def to_local(self) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise NotImplementedError
        return self._local_tensor

    @property
    def device_mesh(self) -> DeviceMesh:
        return self._spec.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        return self._spec.placements

    def full_tensor(self) -> torch.Tensor:
        ragged_shard_axis = get_ragged_shard_axis(self._spec)
        ragged_shard_placement = get_ragged_shard_placement(self._spec)
        shard_numels = ragged_shard_placement.shard_numels

        # Uneven all_gather is NCCL-only; pad to equal chunks for every backend.
        max_numel = max(shard_numels)
        padded = self._local_tensor.new_zeros(max_numel)
        padded[: self._local_tensor.numel()] = self._local_tensor
        rows = padded.new_empty(len(shard_numels), max_numel).unbind(0)
        torch.distributed.all_gather(
            list(rows),
            padded,
            group=self._spec.mesh.get_group(ragged_shard_axis),
        )
        full_tensor = torch.cat(
            [row[:numel] for row, numel in zip(rows, shard_numels, strict=True)]
        )
        replicated_placements = replace_ragged_shard(self._spec, Replicate())
        if len(self._spec.placements) > 1:
            local_shape, _ = compute_local_shape_and_global_offset(
                self.shape, self._spec.mesh, replicated_placements
            )
            return DTensor(
                full_tensor.view(local_shape),
                DTensorSpec(
                    self._spec.mesh,
                    replicated_placements,
                    self._spec.tensor_meta,
                ),
                requires_grad=self.requires_grad,
            ).full_tensor()
        else:
            return full_tensor.view(self.size())


def install() -> None:
    """Teach torch about ``RaggedShardDTensor``. Idempotent; ``fully_shard`` calls it.

    A run on another backend imports this module -- the metrics and EMA
    callbacks do -- without sharding anything, and should not have torch's
    globals rewritten under it.
    """
    torch.serialization.add_safe_globals([RaggedShardDTensor, RaggedShard])
    # foreach over the for-loop implementation in the optimizer and in
    # clip_grad_norm_.
    if RaggedShardDTensor not in _optim_foreach_supported_types:
        _optim_foreach_supported_types.append(RaggedShardDTensor)
    if RaggedShardDTensor not in _util_foreach_supported_types:  # type: ignore[comparison-overlap]
        _util_foreach_supported_types.append(RaggedShardDTensor)  # type: ignore[arg-type]
