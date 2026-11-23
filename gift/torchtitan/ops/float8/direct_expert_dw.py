from __future__ import annotations

from collections.abc import Callable
from math import prod

import torch

_INT32_MAX = torch.iinfo(torch.int32).max
_Shape = tuple[int, int, int]

_Operand = tuple[torch.Tensor, torch.Tensor]
_NativeWriter = Callable[
    [_Operand, _Operand, torch.Tensor, list[int], torch.Tensor], None
]
_native_writer: _NativeWriter | None = None
_leases: dict[tuple[int, int], DirectExpertDwLease] = {}


def configure_direct_expert_dw(native_writer: _NativeWriter) -> None:
    """Install the native Hopper writer after selecting the rank's device."""
    global _native_writer
    if _native_writer is not None and _native_writer is not native_writer:
        raise RuntimeError("The native direct expert dW writer is already configured")
    _native_writer = native_writer


def is_supported_direct_expert_dw_shape(shape: _Shape) -> bool:
    """Check the pinned SM90 launcher's shape and integer bounds."""
    groups, rows, columns = shape
    # The flattened TMA output needs complete M tiles at expert boundaries.
    # DeepGEMM's host heuristics count tiles and round up using signed int.
    return (
        groups > 0
        and 0 < rows <= _INT32_MAX - 255
        and 0 < columns <= _INT32_MAX - 255
        and rows % 128 == 0
        and columns % 128 == 0
        and groups * rows <= _INT32_MAX
        and groups * (rows // 64) * (columns // 16) <= _INT32_MAX // 2
    )


def _slot_key(slot: torch.Tensor) -> tuple[int, int]:
    if (
        not slot.is_cuda
        or slot.dtype != torch.float32
        or slot.ndim != 1
        or not slot.is_contiguous()
        or slot.requires_grad
    ):
        raise ValueError("Direct expert dW requires a flat contiguous FP32 CUDA slot")
    device_index = slot.get_device()
    return device_index, slot.data_ptr()


class DirectExpertDwLease:
    """Host readiness for one backward. GFSDP keeps event and storage ownership."""

    def __init__(
        self, key: tuple[int, int], shapes: dict[int, _Shape], stream: torch.cuda.Stream
    ) -> None:
        self._key = key
        self._shapes = shapes
        self._stream = stream
        self._written: set[int] = set()

    def _check_stream(self) -> None:
        if torch.cuda.current_stream(self._key[0]) != self._stream:
            raise RuntimeError("Direct expert dW used a different CUDA stream")

    def _destination(self, slot: torch.Tensor, offset: int) -> torch.Tensor:
        self._check_stream()
        if offset in self._written:
            raise RuntimeError(
                f"Direct expert dW projection at {offset} was written twice"
            )
        if (shape := self._shapes.get(offset)) is None:
            raise ValueError(f"Direct expert dW offset {offset} is outside the lease")
        return slot.narrow(0, offset, prod(shape)).view(shape)

    def finish(self) -> None:
        """Check both launches were enqueued on the stream used for reduction."""
        if _leases.get(self._key) is not self:
            raise RuntimeError("The direct expert dW lease is stale")
        self._check_stream()
        if len(self._written) != 2:
            raise RuntimeError(
                "Direct expert dW requires one write for each projection"
            )
        del _leases[self._key]

    def abort(self) -> None:
        """Clear host readiness. The GFSDP owner must fence any queued writes."""
        if _leases.get(self._key) is self:
            del _leases[self._key]


def acquire_direct_expert_dw(
    slot: torch.Tensor, offsets: tuple[int, int], shapes: tuple[_Shape, _Shape]
) -> DirectExpertDwLease:
    """Bind the actual collective input and the w13/w2 offsets for this backward."""
    if _native_writer is None:
        raise RuntimeError("Configure the native direct expert dW writer before use")
    if any(not is_supported_direct_expert_dw_shape(shape) for shape in shapes):
        raise ValueError("Direct expert dW requires supported 128-aligned SM90 shapes")
    if shapes[0][0] != shapes[1][0]:
        raise ValueError("Direct expert dW projections must have the same expert count")
    w13_numel, w2_numel = map(prod, shapes)
    if offsets not in ((0, w13_numel), (w2_numel, 0)) or (
        slot.numel() != w13_numel + w2_numel
    ):
        raise ValueError(
            "Direct expert dW projections must exactly cover the flat slot"
        )
    key = _slot_key(slot)
    if key[1] % 16:
        raise ValueError("Direct expert dW slot address must be 16-byte aligned")
    if key in _leases:
        raise RuntimeError("The direct expert dW slot already has an active lease")
    lease = DirectExpertDwLease(
        key,
        dict(zip(offsets, shapes, strict=True)),
        torch.cuda.current_stream(slot.device),
    )
    _leases[key] = lease
    return lease


def _get_lease(slot: torch.Tensor) -> DirectExpertDwLease:
    lease = _leases.get(_slot_key(slot))
    if lease is None:
        raise RuntimeError("Direct expert dW did not receive the leased FP32 slot")
    return lease


def _write_direct_expert_dw(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    slot: torch.Tensor,
    offset: int,
    ks: list[int] | None,
    ks_cpu: torch.Tensor | None,
    ks_tensor: torch.Tensor,
) -> None:
    lease = _get_lease(slot)
    destination = lease._destination(slot, offset)
    if ks is None:
        if ks_cpu is None or ks_cpu.device.type != "cpu" or ks_cpu.numel() == 0:
            raise ValueError("Direct expert dW requires HybridEP's host K counts")
        ks = ks_cpu.tolist()
    num_groups, m, n = destination.shape
    if len(ks) != num_groups or any(k < 0 or k % 128 for k in ks):
        raise ValueError(
            f"Direct expert dW requires {num_groups} nonnegative, 128-aligned counts"
        )

    # Use the actual rows even if the requantizers produced a padded capacity tail.
    rows = sum(ks)
    blocks = rows // 128
    if rows * m > a_data.numel() or rows * n > b_data.numel():
        raise ValueError("Direct expert dW K counts exceed the operand capacity")
    writer = _native_writer
    assert writer is not None  # Acquiring a lease requires the native writer.
    writer(
        (a_data[: rows * m], a_scale[:, :blocks]),
        (b_data[: rows * n], b_scale[:, :blocks]),
        destination,
        ks,
        ks_tensor,
    )
    lease._written.add(offset)


write_direct_expert_dw = torch.library.custom_op(
    "torchtitan::write_direct_expert_dw",
    _write_direct_expert_dw,
    mutates_args={"slot"},
    device_types="cuda",
    schema="(Tensor a_data, Tensor a_scale, Tensor b_data, Tensor b_scale, Tensor(a!) slot, "
    "SymInt offset, SymInt[]? ks, Tensor? ks_cpu, Tensor ks_tensor) -> ()",
)


@write_direct_expert_dw.register_fake
def _fake_write_direct_expert_dw(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    slot: torch.Tensor,
    offset: int,
    ks: list[int] | None,
    ks_cpu: torch.Tensor | None,
    ks_tensor: torch.Tensor,
) -> None:
    return None


def _zero_direct_expert_dw(
    slot: torch.Tensor, offset: int, grad_output: torch.Tensor
) -> None:
    # The cotangent keeps this write in backward under AOT partitioning.
    lease = _get_lease(slot)
    lease._destination(slot, offset).zero_()
    lease._written.add(offset)


zero_direct_expert_dw = torch.library.custom_op(
    "torchtitan::zero_direct_expert_dw",
    _zero_direct_expert_dw,
    mutates_args={"slot"},
    device_types="cuda",
    schema="(Tensor(a!) slot, SymInt offset, Tensor grad_output) -> ()",
)


@zero_direct_expert_dw.register_fake
def _fake_zero_direct_expert_dw(
    slot: torch.Tensor, offset: int, grad_output: torch.Tensor
) -> None:
    return None
