# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import inspect
import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any, TYPE_CHECKING, TypeAlias, cast

import torch
import torch.distributed as dist

from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.config import Configurable
from torchtitan.config.schema.components.dataloader import (
    BaseDataLoaderConfig,
    ParallelAwareDataloaderConfig,
)
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.distributed import ParallelDims


# NOTE: This class deliberately inherits from `Exception` and not `StopIteration`.
# According to PEP 479, raising a `StopIteration` or its subclass from within a
# generator will wrap it in a `RuntimeError`. Since this exception is designed
# to be raised from a generator-based dataloader and caught by the training loop,
# inheriting from `StopIteration` would make it uncatchable and would crash the
# program.
# See: https://peps.python.org/pep-0479/
class DataloaderExhaustedError(Exception):
    """An exception that indicates dataloader exhaustion."""

    pass


class BaseDataLoader(Stateful, ABC, Configurable):
    """Base class for all dataloaders.

    This is used to enforce that all dataloaders have the methods defined in ``Stateful``,
    ``state_dict()`` and ``load_state_dict()``.
    """

    Config: TypeAlias = BaseDataLoaderConfig

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]: ...


_STATUS_BATCH = 0
_STATUS_EOF = 1
_STATUS_ERROR = 2

# per-tensor header slot: dtype code, ndim, then MAX_NDIM dims
_HEADER_MAX_NDIM = 8
_HEADER_SLOT = 2 + _HEADER_MAX_NDIM

# append only: the code order is part of the protocol
_WIRE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
)
_DTYPE_TO_CODE = {dtype: code for code, dtype in enumerate(_WIRE_DTYPES)}


class ReplicatedBatchDataLoader(BaseDataLoader):
    """Rank 0 of ``mesh`` reads batches; the other ranks receive them

    For axes whose ranks need identical batches but are not contiguous in
    global rank order (``pp``). ``owner_only_keys`` are never broadcast;
    ``None`` values survive; received tensors alias one shared buffer per
    batch — do not mutate them in place
    """

    _device: torch.device

    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        *,
        mesh: DeviceMesh,
        device: torch.device,
        owner_only_keys: Iterable[str] = (),
    ) -> None:
        if mesh.ndim != 1:
            raise ValueError(f"mesh must be one-dimensional, got {mesh.shape=}")

        self._group = mesh.get_group()
        self._mesh_rank = mesh.get_local_rank()
        self._source_rank = int(mesh.mesh.flatten()[0].item())
        self._device = device
        self._owner_only_keys = frozenset(owner_only_keys)
        self._dataloader = dataloader
        self._iterator: (
            Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]] | None
        ) = None

        # schema state, reset per iteration
        self._keys: list[str] | None = None
        self._broadcast_keys: list[str] = []
        self._header: torch.Tensor | None = None
        self._header_cpu: torch.Tensor | None = None
        self._pending: tuple[dict[str, Any], torch.Tensor] | None = None

        if self._mesh_rank == 0 and dataloader is None:
            raise ValueError("Mesh axis rank 0 requires a real dataloader")
        if self._mesh_rank != 0 and dataloader is not None:
            raise ValueError("Only mesh axis rank 0 may own the real dataloader")

        axis = mesh.mesh_dim_names[0] if mesh.mesh_dim_names else "?"
        logger.info(
            f"Replicated dataloader {axis!r} axis rank {self._mesh_rank}/{mesh.size()} "
            f"uses global rank {self._source_rank} as batch owner"
        )

    @property
    def is_owner(self) -> bool:
        return self._mesh_rank == 0

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        if self.is_owner:
            assert self._dataloader is not None
            self._iterator = iter(self._dataloader)
        else:
            self._iterator = None
        self._keys = None
        self._broadcast_keys = []
        self._header = None
        self._header_cpu = None
        self._pending = None
        return self

    def __next__(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self._keys is None:
            self._negotiate_schema()

        if self.is_owner:
            return self._owner_step()
        return self._receiver_step()

    def _owner_next_batch(self) -> tuple[dict[str, Any], torch.Tensor] | None:
        assert self._iterator is not None
        try:
            batch = next(self._iterator)
        except StopIteration:
            return None
        return self._prepare_batch(batch)

    def _negotiate_schema(self) -> None:
        """Agree on batch keys with iteration's single object broadcast
        (transport's only pickle); owner's first batch stays pending"""
        schema: dict[str, Any] | None = None
        owner_error: Exception | None = None
        if self.is_owner:
            try:
                self._pending = self._owner_next_batch()
            except Exception as error:
                owner_error = error
                schema = _error_status(error)
            else:
                if self._pending is None:
                    schema = {"status": "eof"}
                else:
                    schema = {"status": "schema", "keys": list(self._pending[0])}

        schema = self._broadcast_object(schema)
        status = schema["status"] if isinstance(schema, dict) else None
        if status == "eof":
            raise StopIteration
        if status == "error":
            _raise_owner_error(schema, owner_error)
        if status != "schema":
            raise RuntimeError(
                f"Invalid replicated dataloader schema from rank {self._source_rank}"
            )

        self._keys = schema["keys"]
        self._broadcast_keys = [
            key for key in self._keys if key not in self._owner_only_keys
        ]
        header_size = 1 + (len(self._broadcast_keys) + 1) * _HEADER_SLOT
        self._header = torch.empty(header_size, dtype=torch.int64, device=self._device)
        if self.is_owner:
            self._header_cpu = torch.empty(header_size, dtype=torch.int64)

    def _owner_step(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        header = self._header
        assert header is not None
        batch, self._pending = self._pending, None
        error: Exception | None = None
        tensors: list[torch.Tensor | None] = []

        if batch is None:
            try:
                batch = self._owner_next_batch()
            except Exception as caught:
                error = caught
        if error is None and batch is not None:
            input_dict, labels = batch
            if list(input_dict) != self._keys:
                error = ValueError(
                    f"Replicated dataloader batch keys changed within one "
                    f"iteration: expected {self._keys}, got {list(input_dict)}"
                )
            else:
                tensors = [input_dict[key] for key in self._broadcast_keys] + [labels]

        if error is not None:
            self._fill_header(_STATUS_ERROR, [])
        elif batch is None:
            self._fill_header(_STATUS_EOF, [])
        else:
            self._fill_header(_STATUS_BATCH, tensors)
        dist.broadcast(header, src=self._source_rank, group=self._group)

        if error is not None:
            _raise_owner_error(self._broadcast_object(_error_status(error)), error)
        if batch is None:
            raise StopIteration

        buffer = _pack_tensors(tensors)
        if buffer.numel():
            dist.broadcast(buffer, src=self._source_rank, group=self._group)
        return batch

    def _broadcast_object(self, obj: Any) -> Any:
        object_list = [obj]
        dist.broadcast_object_list(
            object_list,
            src=self._source_rank,
            group=self._group,
            device=self._device,
        )
        return object_list[0]

    def _fill_header(self, status: int, tensors: list[torch.Tensor | None]) -> None:
        assert self._header is not None and self._header_cpu is not None
        staging = self._header_cpu
        staging[0] = status
        for slot, tensor in enumerate(tensors):
            base = 1 + slot * _HEADER_SLOT
            if tensor is None:
                staging[base : base + _HEADER_SLOT] = -1
                continue
            # validated in _prepare_batch: failing here would skip the error status
            staging[base] = _DTYPE_TO_CODE[tensor.dtype]
            staging[base + 1] = tensor.ndim
            for dim, size in enumerate(tensor.shape):
                staging[base + 2 + dim] = size
        self._header.copy_(staging)

    def _prepare_batch(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
    ) -> tuple[dict[str, Any], torch.Tensor]:
        if not isinstance(batch, tuple) or len(batch) != 2:
            raise ValueError(
                "Replicated dataloader batches must be tuple[dict[str, Tensor], Tensor]"
            )

        input_dict, labels = batch
        if not isinstance(input_dict, dict):
            raise ValueError("Replicated dataloader inputs must be a dict")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("Replicated dataloader labels must be a Tensor")

        prepared: dict[str, Any] = {}
        for key, value in input_dict.items():
            if not isinstance(key, str):
                raise ValueError("Replicated dataloader input keys must be strings")
            if value is not None and not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Replicated dataloader input {key!r} must be a Tensor or None"
                )
            if value is not None and key not in self._owner_only_keys:
                self._validate_wire_tensor(key, value)
            prepared[key] = self._to_device(value)
        self._validate_wire_tensor("labels", labels)
        prepared_labels = self._to_device(labels)
        assert prepared_labels is not None
        return prepared, prepared_labels

    @staticmethod
    def _validate_wire_tensor(name: str, tensor: torch.Tensor) -> None:
        if tensor.dtype not in _DTYPE_TO_CODE:
            raise ValueError(
                f"Unsupported dtype {tensor.dtype} for {name!r} in replicated dataloader batch"
            )
        if tensor.ndim > _HEADER_MAX_NDIM:
            raise ValueError(
                f"Tensor rank {tensor.ndim} of {name!r} exceeds transport limit {_HEADER_MAX_NDIM}"
            )

    def _to_device(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.to(self._device, non_blocking=True).contiguous()

    def _receiver_step(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        assert self._header is not None and self._keys is not None
        dist.broadcast(self._header, src=self._source_rank, group=self._group)
        header = self._header.tolist()

        status = header[0]
        if status == _STATUS_EOF:
            raise StopIteration
        if status == _STATUS_ERROR:
            _raise_owner_error(self._broadcast_object(None), None)
        if status != _STATUS_BATCH:
            raise RuntimeError(f"Unknown replicated dataloader status {status}")

        specs: list[tuple[torch.dtype, tuple[int, ...]] | None] = []
        for slot in range(len(self._broadcast_keys) + 1):
            base = 1 + slot * _HEADER_SLOT
            code, ndim = header[base], header[base + 1]
            if ndim < 0:
                specs.append(None)
            else:
                specs.append(
                    (_WIRE_DTYPES[code], tuple(header[base + 2 : base + 2 + ndim]))
                )

        values = _unpack_tensors(
            specs,
            self._device,
            lambda buffer: dist.broadcast(
                buffer, src=self._source_rank, group=self._group
            ),
        )

        labels = values[-1]
        assert labels is not None, "labels are validated as a Tensor on the owner"
        inputs = dict(
            zip(
                self._broadcast_keys,
                cast(list[torch.Tensor], values[:-1]),
                strict=True,
            )
        )
        return inputs, labels

    def state_dict(self) -> dict[str, Any]:
        if not self.is_owner:
            return {}
        assert self._dataloader is not None
        return self._dataloader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not self.is_owner:
            return
        assert self._dataloader is not None
        self._dataloader.load_state_dict(state_dict)


def _wire_order(dtypes: list[torch.dtype | None]) -> list[int]:
    """Non-None slots by descending itemsize so every typed view into fused uint8 buffer stays aligned"""
    present = [(i, dtype) for i, dtype in enumerate(dtypes) if dtype is not None]
    present.sort(key=lambda pair: -pair[1].itemsize)
    return [i for i, _dtype in present]


def _pack_tensors(tensors: list[torch.Tensor | None]) -> torch.Tensor:
    """Fuse tensors into one flat uint8 buffer for a single broadcast"""
    order = _wire_order([t.dtype if t is not None else None for t in tensors])
    chunks: list[torch.Tensor] = []
    for i in order:
        tensor = tensors[i]
        assert tensor is not None
        chunks.append(tensor.view(-1).view(torch.uint8))
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.uint8)


def _unpack_tensors(
    specs: list[tuple[torch.dtype, tuple[int, ...]] | None],
    device: torch.device,
    receive: Callable[[torch.Tensor], Any],
) -> list[torch.Tensor | None]:
    """Receive fused uint8 buffer and view it back into typed tensors"""
    order = _wire_order([spec[0] if spec is not None else None for spec in specs])
    nbytes = [math.prod(spec[1]) * spec[0].itemsize if spec else 0 for spec in specs]
    buffer = torch.empty(sum(nbytes), dtype=torch.uint8, device=device)
    if buffer.numel():
        receive(buffer)

    values: list[torch.Tensor | None] = [None] * len(specs)
    offset = 0
    for i in order:
        spec = specs[i]
        assert spec is not None
        dtype, shape = spec
        values[i] = buffer[offset : offset + nbytes[i]].view(dtype).view(shape)
        offset += nbytes[i]
    return values


def _error_status(error: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _raise_owner_error(status: Any, owner_error: Exception | None) -> None:
    error_type = "Exception"
    error_message = ""
    if isinstance(status, dict):
        error_type = status.get("error_type", error_type)
        error_message = status.get("error_message", error_message)
    raise RuntimeError(
        f"Replicated dataloader owner raised {error_type}: {error_message}"
    ) from owner_error


# Canonical dataloading-mesh order. ``batch`` is absent on purpose: replicating
# across it would give every data-parallel rank the same samples.
_REPLICATION_AXES = ("pp", "cp", "tp")


def _replication_mesh(
    parallel_dims: "ParallelDims", axes: tuple[str, ...]
) -> DeviceMesh | None:
    """Flatten the ranks that must share one batch into a 1D mesh.

    Degree-1 axes are dropped before slicing, because ``get_mesh`` rejects
    them. The flattened local rank 0 is the rank whose coordinates on these
    axes are all zero inside one ``batch`` coordinate.
    """
    unknown = [axis for axis in axes if axis not in _REPLICATION_AXES]
    if unknown:
        raise ValueError(
            f"replicate_batch_over={axes} is not supported; "
            f"only subsets of {_REPLICATION_AXES} are"
        )
    active = [
        axis
        for axis in _REPLICATION_AXES
        if axis in axes and getattr(parallel_dims, axis) > 1
    ]
    if not active:
        return None
    mesh = parallel_dims.get_mesh(active if len(active) > 1 else active[0])
    if mesh.ndim == 1:
        return mesh
    return mesh._flatten("_".join(active))


def build_replicated_dataloader(
    config: BaseDataLoader.Config,
    build_fn: Callable[[], BaseDataLoader],
    *,
    parallel_dims: "ParallelDims",
    device: torch.device,
    owner_only_keys: Iterable[str] = (),
) -> BaseDataLoader:
    """Build the real dataloader on one owner per batch coordinate and wrap everyone else.

    ``config.replicate_batch_over`` names the axes that must see identical
    batches. The owner is local rank 0 of that group; other ranks only receive.
    """
    axes = config.replicate_batch_over
    if not axes:
        return build_fn()
    mesh = _replication_mesh(parallel_dims, axes)
    if mesh is None:
        return build_fn()

    dataloader = build_fn() if mesh.get_local_rank() == 0 else None
    return ReplicatedBatchDataLoader(
        dataloader,
        mesh=mesh,
        device=device,
        owner_only_keys=owner_only_keys,
    )


class ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader that is aware of distributed data parallelism.

    This dataloader is used to load data in a distributed data parallel fashion. It also
    utilizes ``torchdata.stateful_dataloader.StatefulDataLoader`` to implement the necessary
    methods such as ``__iter__``.

    Args:
        dataset (IterableDataset): The dataset to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        **kwargs: Additional keyword arguments passed to StatefulDataLoader (e.g.,
            batch_size, collate_fn, num_workers, persistent_workers, prefetch_factor,
            pin_memory).
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = ParallelAwareDataloaderConfig

    dp_rank: int
    dp_world_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        **kwargs: Any,
    ) -> None:
        self._validate_kwargs(kwargs)

        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

        super().__init__(dataset, **kwargs)

    @staticmethod
    def _validate_kwargs(kwargs: dict[str, Any]) -> None:
        """Validate and sanitize kwargs passed to the dataloader.

        Args:
            kwargs: Dictionary of keyword arguments to validate. This dict is
                modified in-place to remove invalid combinations.

        Raises:
            ValueError: If 'dataset' is in kwargs or if any invalid kwargs are passed.
        """
        if "dataset" in kwargs:
            raise ValueError(
                "'dataset' should not be passed in kwargs; "
                "it must be provided as the first positional argument."
            )

        sig = inspect.signature(StatefulDataLoader.__init__)
        valid_kwargs = frozenset(
            name for name in sig.parameters.keys() if name not in ("self", "dataset")
        )
        invalid_kwargs = set(kwargs.keys()) - valid_kwargs
        if invalid_kwargs:
            raise ValueError(
                f"Invalid dataloader kwargs: {invalid_kwargs}. "
                f"Valid kwargs are: {sorted(valid_kwargs)}"
            )

        # persistent_workers and prefetch_factor are only valid when num_workers > 0.
        # Removing them here if num_workers is 0 to avoid StatefulDataLoader errors
        if kwargs.get("num_workers", 0) == 0:
            kwargs.pop("persistent_workers", None)
            kwargs.pop("prefetch_factor", None)

    def state_dict(self) -> dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However,
            # we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(StatefulDataLoader.state_dict(self)),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # State being empty is valid.
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                "expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        # We don't have to use pickle as DCP will serialize the state_dict. However, we have to
        # keep this for backward compatibility.
        StatefulDataLoader.load_state_dict(
            self, pickle.loads(state_dict[self._rank_id])
        )
