"""Bounded native numerical reference recorder for the CUDA graph campaign.

This module is opt-in through CampaignNumericalProbeConfig. It is deliberately
limited to a native S0 and one or ten observed updates; it does not change
training math. Tensor payloads are read back and checked on CPU after each
update.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import json
import math
import marshal
import os
import shutil
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DTensor

from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.callbacks.sources.base import canonical_fqn
from torchtitan.config import Configurable
from torchtitan.config.schema.experiments.cuda_graph_campaign.numerical_probe import (
    CampaignNumericalProbeConfig,
)
from torchtitan.experiments.cuda_graph_campaign.continuation import (
    CampaignRNGContinuationObserver,
)


_SCHEMA = 1
_CAP_BYTES = 700 * 1024**3
_FREE_RESERVE_BYTES = 32 * 1024**3
_WRITE_CHUNK_BYTES = 8 * 1024**2
_STATE_FAMILIES = ("exp_avg", "exp_avg_sq", "step")
_GROUP_FIELDS = frozenset(
    {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "initial_lr",
        "decoupled_weight_decay",
    }
)


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported probe metadata value {type(value).__name__}")


def _atomic_json(
    path: Path, value: Any, *, budget: _PayloadBudget | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if budget is not None:
        budget.reserve(len(encoded))
    with temp.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _hash_header(
    *, identity: str, family: str, phase: str, tensor: torch.Tensor
) -> bytes:
    header = {
        "identity": identity,
        "family": family,
        "phase": phase,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\0"


def _local_tensor(
    value: torch.Tensor, *, grad_placements: tuple[Any, ...] | None = None
) -> torch.Tensor:
    if isinstance(value, DTensor):
        local = value.to_local(grad_placements=grad_placements)
        if isinstance(local, AsyncCollectiveTensor):
            local = local.wait()
        return local
    if grad_placements is not None:
        raise TypeError("a DTensor gradient is required for explicit placements")
    return value


def _storage_span(shape: tuple[int, ...], stride: tuple[int, ...], offset: int) -> int:
    if len(shape) != len(stride) or offset < 0:
        raise ValueError("invalid tensor layout metadata")
    if any(size < 0 or step < 0 for size, step in zip(shape, stride)):
        raise ValueError("negative sizes or strides are unsupported")
    if any(size == 0 for size in shape):
        return 0
    dims = sorted(
        ((step, size) for size, step in zip(shape, stride) if size > 1),
        key=lambda pair: pair[0],
    )
    covered = 1
    for step, size in dims:
        if step < covered:
            raise ValueError(
                "overlapping or zero-stride parameter layouts are unsupported"
            )
        covered = step * size
    return offset + 1 + sum((size - 1) * step for size, step in zip(shape, stride))


def _tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
    local = _local_tensor(tensor)
    shape = tuple(int(dim) for dim in local.shape)
    stride = tuple(int(step) for step in local.stride())
    offset = int(local.storage_offset())
    span = _storage_span(shape, stride, offset)
    return {
        "shape": list(shape),
        "stride": list(stride),
        "storage_offset": offset,
        "storage_span": span,
        "dtype": str(local.dtype),
        "device": str(local.device),
        "layout": str(local.layout),
        "requires_grad": bool(local.requires_grad),
        "empty": local.numel() == 0,
    }


def _logical_cpu(tensor: torch.Tensor) -> torch.Tensor:
    local = _local_tensor(tensor)
    if local.layout != torch.strided or local.is_complex():
        raise ValueError(
            f"unsupported tensor layout/dtype: {local.layout}, {local.dtype}"
        )
    _storage_span(tuple(local.shape), tuple(local.stride()), local.storage_offset())
    return local.detach().to(device="cpu", memory_format=torch.contiguous_format)


def _raw_bytes(tensor: torch.Tensor) -> memoryview:
    """View contiguous CPU tensor storage as bytes, including bfloat16 and FP8."""
    cpu = _logical_cpu(tensor)
    return memoryview(cpu.reshape(-1).contiguous().view(torch.uint8).numpy())


def _logical_digest(tensor: torch.Tensor, header: bytes) -> str:
    digest = hashlib.sha256(header)
    view = _raw_bytes(tensor)
    for start in range(0, len(view), _WRITE_CHUNK_BYTES):
        digest.update(view[start : start + _WRITE_CHUNK_BYTES])
    return digest.hexdigest()


class _PayloadBudget:
    """A locked counter includes retained source/run files and all rank writes."""

    def __init__(self, output_dir: Path, roots: tuple[Path, ...]):
        self.output_dir = output_dir
        self.counter = output_dir / ".payload-bytes"
        self.roots = roots

    def reserve(self, byte_count: int) -> None:
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("payload byte count must be a nonnegative integer")
        with self.counter.open("a+", encoding="ascii") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            previous = int(stream.read() or "0")
            projected = previous + byte_count
            if projected > _CAP_BYTES:
                raise OSError(
                    f"probe payload cap exceeded: {projected} > {_CAP_BYTES} bytes"
                )
            free_bytes = shutil.disk_usage(self.output_dir.parent).free
            actual = _tree_bytes(self.roots, exclude=self.counter)
            pending = max(0, projected - actual)
            if free_bytes - pending < _FREE_RESERVE_BYTES:
                raise OSError(
                    f"probe would leave less than {_FREE_RESERVE_BYTES} free bytes"
                )
            stream.seek(0)
            stream.truncate()
            stream.write(str(projected))
            stream.flush()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def reconcile(self) -> int:
        actual = _tree_bytes(self.roots, exclude=self.counter)
        if actual > _CAP_BYTES:
            raise OSError(
                f"total live probe payload exceeds {_CAP_BYTES} bytes: {actual}"
            )
        free_bytes = shutil.disk_usage(self.output_dir.parent).free
        if free_bytes < _FREE_RESERVE_BYTES:
            raise OSError(
                f"probe storage has less than {_FREE_RESERVE_BYTES} free bytes"
            )
        with self.counter.open("r+", encoding="ascii") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            stream.truncate()
            stream.write(str(actual))
            stream.flush()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return actual


def _tree_bytes(roots: tuple[Path, ...], *, exclude: Path | None = None) -> int:
    files: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = (root,)
        else:
            candidates = (
                Path(directory) / name
                for directory, _, names in os.walk(root)
                for name in names
            )
        for path in candidates:
            resolved = path.resolve()
            if exclude is not None and resolved == exclude.resolve():
                continue
            if path.is_file():
                files.add(resolved)
    return sum(path.stat().st_size for path in files)


def _filesystem_info(path: Path) -> dict[str, Any]:
    stats = os.statvfs(path)
    usage = shutil.disk_usage(path)
    mount_point = None
    filesystem = None
    resolved = path.resolve()
    try:
        with Path("/proc/self/mountinfo").open(encoding="utf-8") as stream:
            candidates = []
            for line in stream:
                fields = line.split()
                point = Path(fields[4].replace("\\040", " ")).resolve()
                try:
                    resolved.relative_to(point)
                except ValueError:
                    continue
                separator = fields.index("-")
                candidates.append((len(str(point)), point, fields[separator + 1]))
            if candidates:
                _, point, filesystem = max(candidates, key=lambda value: value[0])
                mount_point = str(point)
    except OSError:
        pass
    return {
        "device": os.stat(path).st_dev,
        "mount_point": mount_point,
        "filesystem": filesystem,
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "available_inodes": stats.f_favail,
    }


class _TensorPackWriter:
    """Append compact logical bytes, one tensor at a time, then publish."""

    def __init__(
        self,
        final_path: Path,
        budget: _PayloadBudget,
        *,
        pre_reserved_bytes: int = 0,
        pre_reserved_families: frozenset[str] = frozenset(),
    ):
        self.final_path = final_path
        self.partial_path = final_path.with_name(final_path.name + ".partial")
        self.budget = budget
        self.remaining_reserved_bytes = pre_reserved_bytes
        self.pre_reserved_families = pre_reserved_families
        self.stream = self.partial_path.open("xb")
        self.entries: dict[str, dict[str, Any]] = {}
        self.payload_digest = hashlib.sha256()
        self.closed = False

    def add(
        self,
        key: str,
        tensor: torch.Tensor | None,
        *,
        identity: str,
        family: str,
        phase: str,
    ) -> dict[str, Any]:
        if key in self.entries:
            raise ValueError(f"duplicate payload key {key!r}")
        if tensor is None:
            entry = {
                "status": "none",
                "identity": identity,
                "family": family,
                "phase": phase,
                "sha256": None,
                "offset": None,
                "nbytes": 0,
            }
            self.entries[key] = entry
            return entry

        layout = _tensor_layout(tensor)
        cpu = _logical_cpu(tensor)
        byte_count = cpu.numel() * cpu.element_size()
        if family in self.pre_reserved_families:
            if byte_count > self.remaining_reserved_bytes:
                self.budget.reserve(byte_count - self.remaining_reserved_bytes)
                self.remaining_reserved_bytes = 0
            else:
                self.remaining_reserved_bytes -= byte_count
        else:
            self.budget.reserve(byte_count)
        raw = _raw_bytes(cpu)
        header = _hash_header(identity=identity, family=family, phase=phase, tensor=cpu)
        digest = hashlib.sha256(header)
        offset = self.stream.tell()
        for start in range(0, len(raw), _WRITE_CHUNK_BYTES):
            chunk = raw[start : start + _WRITE_CHUNK_BYTES]
            digest.update(chunk)
            self.payload_digest.update(chunk)
            self.stream.write(chunk)
        entry = {
            "status": "tensor",
            "identity": identity,
            "family": family,
            "phase": phase,
            "sha256": digest.hexdigest(),
            "offset": offset,
            "nbytes": byte_count,
            **layout,
        }
        self.entries[key] = entry
        return entry

    def close(self) -> None:
        if self.closed:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.partial_path, self.final_path)
        self.closed = True

    def abort(self) -> None:
        if not self.closed:
            self.stream.close()
            self.closed = True


class _TensorPackReader:
    """Bounded CPU-only logical tensor readback and hash verification."""

    def __init__(self, path: Path, entries: Mapping[str, Mapping[str, Any]]):
        self.stream = path.open("rb")
        self.entries = entries

    def __enter__(self) -> _TensorPackReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def read(self, key: str) -> torch.Tensor | None:
        entry = self.entries[key]
        if entry["status"] == "none":
            return None
        if entry["status"] != "tensor":
            raise ValueError(f"unsupported payload status for {key}")
        self.stream.seek(int(entry["offset"]))
        raw = bytearray(self.stream.read(int(entry["nbytes"])))
        if len(raw) != int(entry["nbytes"]):
            raise EOFError(f"incomplete tensor payload for {key}")
        dtype_name = str(entry["dtype"]).removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"unsupported payload dtype {entry['dtype']!r}")
        flat = (
            torch.frombuffer(raw, dtype=dtype) if raw else torch.empty(0, dtype=dtype)
        )
        shape = tuple(int(size) for size in entry["shape"])
        stride = tuple(int(step) for step in entry["stride"])
        offset = int(entry["storage_offset"])
        span = int(entry["storage_span"])
        if flat.numel() != math.prod(shape):
            raise ValueError(f"tensor payload size does not match metadata for {key}")
        if _storage_span(shape, stride, offset) != span:
            raise ValueError(f"tensor payload layout metadata is invalid for {key}")
        header = _hash_header(
            identity=str(entry["identity"]),
            family=str(entry["family"]),
            phase=str(entry["phase"]),
            tensor=flat.reshape(shape),
        )
        digest = hashlib.sha256(header)
        view = memoryview(raw)
        for start in range(0, len(raw), _WRITE_CHUNK_BYTES):
            digest.update(view[start : start + _WRITE_CHUNK_BYTES])
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"tensor payload hash mismatch for {key}")
        logical = flat.reshape(shape)
        if _logical_digest(logical, header) != entry["sha256"]:
            raise ValueError(f"tensor logical readback mismatch for {key}")
        return logical

    def verify_all(self) -> int:
        verified = 0
        expected_offset = 0
        for key, entry in self.entries.items():
            if entry["status"] == "none":
                self.read(key)
                verified += 1
        tensor_entries = sorted(
            (
                (key, entry)
                for key, entry in self.entries.items()
                if entry["status"] == "tensor"
            ),
            key=lambda item: int(item[1]["offset"]),
        )
        for key, entry in tensor_entries:
            if int(entry["offset"]) != expected_offset:
                raise ValueError(f"tensor payload offsets are not contiguous at {key}")
            expected_offset += int(entry["nbytes"])
            self.read(key)
            verified += 1
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() != expected_offset:
            raise ValueError("tensor payload file has missing or trailing bytes")
        return verified

    def close(self) -> None:
        self.stream.close()


def _snapshot_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _snapshot_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_cpu(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_cpu(item) for item in value]
    if value is None or type(value) in (str, bytes, bool, int, float):
        return value
    raise TypeError(f"unsupported native state value {type(value).__name__}")


def _capture_input_tensor(
    writer: _TensorPackWriter,
    value: torch.Tensor | None,
    *,
    key: str,
    identity: str,
    phase: str,
) -> Any:
    """Capture a target input tensor without retaining its device storage."""
    if value is not None and not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor or None for prepared input {key}")
    if value is None:
        return {"status": "none"}
    return writer.add(
        key,
        value,
        identity=identity,
        family="input_tensor",
        phase=phase,
    )


def _capture_mask_callable(
    writer: _TensorPackWriter,
    function: Any,
    *,
    key: str,
    identity: str,
    phase: str,
) -> dict[str, Any]:
    """Record the pinned causal/document mask callable identity and closure data."""
    if not isinstance(function, types.FunctionType):
        raise TypeError(f"unsupported FlexAttention mask callable at {key}")
    qualname = function.__qualname__
    digest = hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    closure = () if function.__closure__ is None else function.__closure__
    record: dict[str, Any] = {
        "module": function.__module__,
        "qualname": qualname,
        "code_sha256": digest,
        "closure": [],
        "interpretation": "metadata_only_not_reconstructibility_evidence",
    }
    if qualname == "and_masks.<locals>.and_mask":
        if len(closure) != 1 or not isinstance(closure[0].cell_contents, tuple):
            raise ValueError("unsupported pinned FlexAttention and_masks closure")
        record["closure"] = [
            _capture_mask_callable(
                writer,
                child,
                key=f"{key}/mask-{index}",
                identity=f"{identity}/mask-{index}",
                phase=phase,
            )
            for index, child in enumerate(closure[0].cell_contents)
        ]
    elif qualname == "get_document_mask_mod.<locals>.document_mask":
        if len(closure) != 1 or not isinstance(closure[0].cell_contents, torch.Tensor):
            raise ValueError("unsupported pinned document-mask closure")
        record["closure"] = [
            _capture_input_tensor(
                writer,
                closure[0].cell_contents,
                key=f"{key}/document-ids",
                identity=f"{identity}/document-ids",
                phase=phase,
            )
        ]
    elif qualname == "get_fixed_block_mask_mod.<locals>.blocked_mask_mod":
        if len(closure) != 1 or type(closure[0].cell_contents) is not int:
            raise ValueError("unsupported pinned fixed-block mask closure")
        record["closure"] = [{"fixed_block_size": closure[0].cell_contents}]
    elif qualname == "get_causal_mask_mod.<locals>._causal_mask":
        if closure:
            raise ValueError("causal mask callable unexpectedly has a closure")
    else:
        raise ValueError(f"unsupported pinned FlexAttention mask callable {qualname!r}")
    return record


def _capture_block_mask(
    writer: _TensorPackWriter,
    mask: Any,
    *,
    key: str,
    identity: str,
    phase: str,
) -> dict[str, Any]:
    """Capture the installed BlockMask tensor/context fields used by this target."""
    if (
        type(mask).__module__ != "torch.nn.attention.flex_attention"
        or type(mask).__name__ != "BlockMask"
    ):
        raise TypeError(
            f"unsupported attention mask object at {key}: {type(mask).__name__}"
        )
    tensor_fields = (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
        "dq_write_order",
        "dq_write_order_full",
        "dq_kv_order",
    )
    expected_fields = set(tensor_fields) | {
        "seq_lengths",
        "dq_kv_order_spt",
        "BLOCK_SIZE",
        "mask_mod",
    }
    if set(vars(mask)) != expected_fields:
        raise ValueError(
            "installed FlexAttention BlockMask fields differ from pinned API"
        )
    fields = {}
    for name in tensor_fields:
        fields[name] = _capture_input_tensor(
            writer,
            getattr(mask, name),
            key=f"{key}/{name}",
            identity=f"{identity}/{name}",
            phase=phase,
        )
    for name in ("seq_lengths", "BLOCK_SIZE"):
        value = getattr(mask, name)
        if not isinstance(value, tuple) or any(type(item) is not int for item in value):
            raise TypeError(f"unsupported FlexAttention BlockMask field {name}")
        fields[name] = list(value)
    spt = mask.dq_kv_order_spt
    if spt is not None and type(spt) is not bool:
        raise TypeError("unsupported FlexAttention BlockMask dq_kv_order_spt")
    fields["dq_kv_order_spt"] = spt
    fields["mask_mod"] = _capture_mask_callable(
        writer,
        mask.mask_mod,
        key=f"{key}/mask_mod",
        identity=f"{identity}/mask_mod",
        phase=phase,
    )
    return {"type": "torch.nn.attention.flex_attention.BlockMask", "fields": fields}


def _rng_json(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(snapshot)
    result["python_numpy_state"] = snapshot["python_numpy_state"].hex()
    for name in ("torch_cpu_state", "torch_cuda_state"):
        tensor = snapshot[name].detach().to(device="cpu").contiguous()
        result[name] = tensor.numpy().tobytes().hex()
        result[name + "_shape"] = list(tensor.shape)
    return result


def _rng_from_json(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **snapshot,
        "python_numpy_state": bytes.fromhex(snapshot["python_numpy_state"]),
        "torch_cpu_state": torch.frombuffer(
            bytearray.fromhex(snapshot["torch_cpu_state"]), dtype=torch.uint8
        )
        .clone()
        .reshape(snapshot["torch_cpu_state_shape"]),
        "torch_cuda_state": torch.frombuffer(
            bytearray.fromhex(snapshot["torch_cuda_state"]), dtype=torch.uint8
        )
        .clone()
        .reshape(snapshot["torch_cuda_state_shape"]),
    }


def _same_rng(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["schema_version"] == right["schema_version"]
        and left["global_rank"] == right["global_rank"]
        and left["world_size"] == right["world_size"]
        and left["device_type"] == right["device_type"]
        and left["device_index"] == right["device_index"]
        and left["python_numpy_state"] == right["python_numpy_state"]
        and torch.equal(left["torch_cpu_state"], right["torch_cpu_state"])
        and torch.equal(left["torch_cuda_state"], right["torch_cuda_state"])
    )


def _rank_and_world() -> tuple[int, int]:
    if not dist.is_initialized():
        raise RuntimeError(
            "campaign_numerical_probe requires initialized distributed training"
        )
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 8:
        raise ValueError(
            f"campaign_numerical_probe requires eight ranks, got {world_size}"
        )
    return rank, world_size


def _output_path(config: CampaignNumericalProbeConfig, dump_folder: str) -> Path:
    path = Path(config.output_dir)
    if path.is_absolute():
        raise ValueError(
            "campaign_numerical_probe.output_dir must be relative to dump_folder"
        )
    return (Path(dump_folder) / path).resolve()


def _source_s0_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if not path.is_absolute() or path.name != "step-1":
        raise ValueError("observe requires an absolute native step-1 checkpoint path")
    return path.parent / "numerical_probe"


def _named_parameters(model_parts: tuple[nn.Module, ...]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    canonical_owners: dict[tuple[int, str], int] = {}
    for part_index, model in enumerate(model_parts):
        for name, param in model.named_parameters(remove_duplicate=False):
            fqn = canonical_fqn(name)
            if not fqn:
                raise ValueError("empty canonical parameter name")
            key = (part_index, fqn)
            owner = canonical_owners.get(key)
            if owner is not None and owner != id(param):
                raise ValueError(f"canonical parameter-name collision: {key}")
            canonical_owners[key] = id(param)
            record = by_id.get(id(param))
            if record is None:
                record = {
                    "part_index": part_index,
                    "canonical_fqn": fqn,
                    "parameter": param,
                    "names": [],
                }
                by_id[id(param)] = record
            elif record["part_index"] != part_index:
                raise ValueError("one parameter is owned by multiple model parts")
            record["names"].append(name)

    storage_owners: dict[tuple[str, int], int] = {}
    for record in by_id.values():
        param = record["parameter"]
        local = _local_tensor(param)
        if local.numel() == 0:
            continue
        storage = local.untyped_storage()
        storage_key = (str(local.device), storage.data_ptr())
        other = storage_owners.get(storage_key)
        if other is not None and other != id(param):
            raise ValueError(
                "distinct parameters share storage; overlapping or tied views are unsupported"
            )
        storage_owners[storage_key] = id(param)
    return sorted(
        by_id.values(),
        key=lambda record: (record["part_index"], record["canonical_fqn"]),
    )


def _mesh_metadata(param: torch.Tensor) -> dict[str, Any]:
    if not isinstance(param, DTensor):
        return {"mesh_axes": [], "mesh_ranks": [], "placements": []}
    mesh = param.device_mesh
    ranks = mesh.mesh.detach().to(device="cpu").tolist()
    return {
        "mesh_axes": list(mesh.mesh_dim_names or ()),
        "mesh_ranks": ranks,
        "placements": [
            type(placement).__name__ + ":" + str(placement)
            for placement in param.placements
        ],
    }


def _group_map(state: ObserverState) -> dict[int, dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    for group in state.optimizer_groups():
        name = group.optimizer_type.__name__
        if group.optimizer_type is not torch.optim.AdamW or name != "AdamW":
            raise TypeError(f"unsupported numerical-probe optimizer {name}")
        if set(group.hparams) - _GROUP_FIELDS:
            raise ValueError(
                f"unsupported AdamW group fields: {sorted(set(group.hparams) - _GROUP_FIELDS)}"
            )
        hparams = dict(group.hparams)
        if (
            hparams.get("fused") is not True
            or hparams.get("foreach") not in (False, None)
            or hparams.get("amsgrad") is not False
            or hparams.get("maximize") is not False
            or hparams.get("capturable") is not False
            or hparams.get("differentiable") is not False
            or hparams.get("decoupled_weight_decay") is not True
        ):
            raise ValueError(f"unsupported native AdamW options: {hparams}")
        for param_index, param in enumerate(group.params):
            previous = groups.get(id(param))
            if previous is not None:
                raise ValueError("a parameter is owned by multiple optimizer groups")
            groups[id(param)] = {
                "optimizer_name": name,
                "part_index": group.part_index,
                "group_index": group.group_index,
                "param_index": param_index,
                "hparams": _json_value(hparams),
                "optimizer_type": group.optimizer_type,
            }
    return groups


def _parameter_record(
    record: Mapping[str, Any], group: Mapping[str, Any] | None, rank: int
) -> dict[str, Any]:
    param = record["parameter"]
    local = _local_tensor(param)
    result = {
        "rank": rank,
        "part_index": record["part_index"],
        "canonical_fqn": record["canonical_fqn"],
        "original_names": list(record["names"]),
        "global_shape": list(param.shape),
        "local": _tensor_layout(local),
        "mesh": _mesh_metadata(param),
        "requires_grad": bool(param.requires_grad),
        "group": None,
    }
    if group is not None:
        result["group"] = {
            key: value for key, value in group.items() if key != "optimizer_type"
        }
    elif param.requires_grad:
        raise ValueError(
            f"trainable parameter has no optimizer owner: {record['canonical_fqn']}"
        )
    return result


def _buffer_inventory(model_parts: tuple[nn.Module, ...]) -> list[dict[str, Any]]:
    """Record registered buffers and direct tensor attributes, without recursion."""
    result = []
    seen: set[tuple[int, str]] = set()
    for part_index, model in enumerate(model_parts):
        for module_name, module in model.named_modules():
            prefix = canonical_fqn(module_name)
            for name, tensor in module.named_buffers(recurse=False):
                key = (id(module), name)
                if key in seen:
                    continue
                seen.add(key)
                local = _local_tensor(tensor)
                result.append(
                    {
                        "part_index": part_index,
                        "name": f"{prefix}.{name}" if prefix else name,
                        "kind": "buffer",
                        "persistent": name not in module._non_persistent_buffers_set,
                        "layout": _tensor_layout(local),
                        "sha256": _logical_digest(
                            local,
                            _hash_header(
                                identity=f"{prefix}.{name}",
                                family="buffer",
                                phase="inventory",
                                tensor=local,
                            ),
                        ),
                    }
                )
            for name, value in vars(module).items():
                if name.startswith("_") or not isinstance(value, torch.Tensor):
                    continue
                key = (id(module), name)
                if key in seen:
                    continue
                seen.add(key)
                local = _local_tensor(value)
                entry = {
                    "part_index": part_index,
                    "name": f"{prefix}.{name}" if prefix else name,
                    "kind": "direct_tensor_attribute",
                    "persistent": False,
                    "layout": _tensor_layout(local),
                    "sha256": _logical_digest(
                        local,
                        _hash_header(
                            identity=f"{prefix}.{name}",
                            family="direct_tensor",
                            phase="inventory",
                            tensor=local,
                        ),
                    ),
                }
                if name == "aux_loss_scale" and getattr(module, "_aux_loss_kinds", ()):
                    entry["reconstructed_at"] = "AuxLossTracker.weigh"
                result.append(entry)
    return result


def _reconstructed_aux_scale(item: Mapping[str, Any]) -> bool:
    return (
        item.get("kind") == "direct_tensor_attribute"
        and item.get("name", "").rsplit(".", 1)[-1] == "aux_loss_scale"
        and item.get("reconstructed_at") == "AuxLossTracker.weigh"
    )


def _same_loaded_buffer_inventory(
    source: list[dict[str, Any]], current: list[dict[str, Any]]
) -> bool:
    if len(source) != len(current):
        return False
    for expected, observed in zip(source, current):
        if expected == observed:
            continue
        if not (
            _reconstructed_aux_scale(expected) and _reconstructed_aux_scale(observed)
        ):
            return False
        expected_layout = {
            key: value for key, value in expected.items() if key != "sha256"
        }
        observed_layout = {
            key: value for key, value in observed.items() if key != "sha256"
        }
        if expected_layout != observed_layout:
            return False
    return True


def _router_state(model_parts: tuple[nn.Module, ...]) -> list[dict[str, Any]]:
    state = []
    for part_index, model in enumerate(model_parts):
        for module_name, module in model.named_modules():
            counts = getattr(module, "tokens_per_expert_E", None)
            bias = getattr(module, "expert_bias_E", None)
            if not isinstance(counts, torch.Tensor):
                continue
            prefix = canonical_fqn(module_name)
            state.append(
                {
                    "part_index": part_index,
                    "fqn": prefix,
                    "counts": _logical_digest(
                        counts,
                        _hash_header(
                            identity=prefix,
                            family="router_counts",
                            phase="boundary",
                            tensor=counts,
                        ),
                    ),
                    "bias": None
                    if bias is None
                    else _logical_digest(
                        bias,
                        _hash_header(
                            identity=prefix,
                            family="router_bias",
                            phase="boundary",
                            tensor=bias,
                        ),
                    ),
                    "counts_layout": _tensor_layout(counts),
                    "bias_layout": None if bias is None else _tensor_layout(bias),
                }
            )
    return state


class CampaignNumericalProbe(Configurable, Observer):
    """Opt-in recorder for direct native tensor references."""

    Config: TypeAlias = CampaignNumericalProbeConfig

    def __init__(
        self,
        config: Config,
        *,
        rng_continuation: CampaignRNGContinuationObserver,
    ) -> None:
        self.config = config
        self.rng_continuation = rng_continuation
        self.rank, self.world_size = _rank_and_world()
        self.output_dir: Path | None = None
        self.budget: _PayloadBudget | None = None
        self._trainer: Any = None
        self._hook_handles: list[Any] = []
        self._aux_loss_weigh_owner: Any = None
        self._aux_loss_weigh_previous: Any = None
        self._aux_loss_weigh_had_instance = False
        self._aux_loss_weigh_wrapped = False
        self._rng_instance_value: Any = None
        self._rng_had_instance_value = False
        self._rng_wrapped = False
        self._rng_serialized: list[dict[str, Any]] = []
        self._current_microbatch_index: int | None = None
        self._current: dict[str, Any] | None = None
        self._active_writer: _TensorPackWriter | None = None
        self._seen_inner_optimizers: set[int] = set()
        self._group_param_keys: dict[tuple[int, int, int], str] = {}
        self._prior_p_hashes: dict[str, str] = {}
        self._storage_info: dict[str, Any] = {}
        self._account_roots: tuple[Path, ...] = ()
        self._phase_history: list[dict[str, Any]] = []
        self._rng_restore_checked = False
        self._source_rng_serialization: dict[str, Any] | None = None
        self._next_observed_step = 2
        self._local_error: str | None = None

    @property
    def current_microbatch_index(self) -> int | None:
        return self._current_microbatch_index

    def bind_trainer(
        self, trainer: Any, rng_continuation: CampaignRNGContinuationObserver
    ) -> None:
        if self._trainer is not None:
            raise RuntimeError("campaign_numerical_probe is already bound")
        if rng_continuation is not self.rng_continuation:
            raise ValueError("campaign_numerical_probe received a different RNG owner")
        if self.rank != dist.get_rank() or self.world_size != dist.get_world_size():
            raise RuntimeError("campaign_numerical_probe rank ownership changed")
        self.output_dir = _output_path(self.config, trainer.config.dump_folder)
        if self.output_dir.exists():
            raise FileExistsError(
                f"campaign_numerical_probe output already exists: {self.output_dir}"
            )
        if self.config.mode == "create_s0":
            expected = (
                Path(trainer.config.dump_folder)
                / trainer.config.checkpoint.folder
                / "numerical_probe"
            ).resolve()
            if self.output_dir != expected:
                raise ValueError(
                    "create_s0 output_dir must be checkpoint.folder/numerical_probe"
                )
        if self.config.mode == "observe":
            if trainer.config.checkpoint.initial_load_path is None:
                raise ValueError("observe requires checkpoint.initial_load_path")
        optimizers = getattr(trainer.optimizers, "optimizers", None)
        if not isinstance(optimizers, list) or not optimizers:
            raise TypeError("numerical probe requires native inner optimizer instances")
        try:
            self._trainer = trainer
            if self.config.mode == "create_s0":
                self._bind_rng_serialization(rng_continuation.rng_state)
            else:
                self._bind_aux_loss_weigh(trainer.aux_loss)
                for optimizer in optimizers:
                    if type(optimizer) is not torch.optim.AdamW:
                        raise TypeError(
                            f"unsupported native optimizer {type(optimizer).__name__}"
                        )
                    pre_hooks = getattr(optimizer, "_optimizer_step_pre_hooks", None)
                    post_hooks = getattr(optimizer, "_optimizer_step_post_hooks", None)
                    if pre_hooks is None or post_hooks is None:
                        raise RuntimeError(
                            "installed AdamW hook registry is unavailable"
                        )
                    if pre_hooks or post_hooks:
                        raise ValueError(
                            "native AdamW has pre-existing hooks; exact probe is unsupported"
                        )
                    self._hook_handles.append(
                        optimizer.register_step_pre_hook(self._inner_optimizer_pre_hook)
                    )
        except Exception:
            self.close()
            raise

    def _bind_aux_loss_weigh(self, tracker: Any) -> None:
        original = getattr(tracker, "weigh", None)
        states = getattr(tracker, "_states", None)
        if not callable(original) or not isinstance(states, list):
            raise TypeError("observe requires the native AuxLossTracker interface")
        if len(states) != len(self._trainer.model_parts):
            raise ValueError("AuxLossTracker states differ from native model parts")
        self._aux_loss_weigh_owner = tracker
        self._aux_loss_weigh_had_instance = "weigh" in vars(tracker)
        if self._aux_loss_weigh_had_instance:
            self._aux_loss_weigh_previous = vars(tracker)["weigh"]

        def observed_weigh(
            *, valid_tokens: torch.Tensor, global_valid_tokens: torch.Tensor
        ) -> None:
            result = original(
                valid_tokens=valid_tokens,
                global_valid_tokens=global_valid_tokens,
            )
            self._capture_aux_loss_weigh(tracker)
            return result

        tracker.weigh = observed_weigh
        self._aux_loss_weigh_wrapped = True

    def _bind_rng_serialization(self, rng_state: Any) -> None:
        name = "state_dict"
        self._rng_had_instance_value = name in rng_state.__dict__
        if self._rng_had_instance_value:
            self._rng_instance_value = rng_state.__dict__[name]
        original = rng_state.state_dict

        def observed_state_dict() -> dict[str, Any]:
            value = original()
            self._rng_serialized.append(copy.deepcopy(value))
            return value

        setattr(rng_state, name, observed_state_dict)
        self._rng_wrapped = True
        self._rng_state_owner = rng_state

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        if self._aux_loss_weigh_wrapped:
            if self._aux_loss_weigh_had_instance:
                self._aux_loss_weigh_owner.weigh = self._aux_loss_weigh_previous
            else:
                del self._aux_loss_weigh_owner.weigh
            self._aux_loss_weigh_wrapped = False
        if self._rng_wrapped:
            if self._rng_had_instance_value:
                self._rng_state_owner.state_dict = self._rng_instance_value
            else:
                del self._rng_state_owner.state_dict
            self._rng_wrapped = False
        self._abort_active_writer()

    def _abort_active_writer(self) -> None:
        if self._active_writer is not None:
            writer = self._active_writer
            self._active_writer = None
            try:
                writer.abort()
            except Exception:
                pass

    def _best_effort_failure_record(
        self, filename: str, status: dict[str, Any]
    ) -> None:
        if self.output_dir is None or self.budget is None:
            return
        try:
            _atomic_json(
                self.output_dir / filename,
                status,
                budget=self.budget,
            )
        except Exception:
            pass

    def _ensure_output(
        self,
        *,
        account_roots: tuple[Path, ...],
        expected_payload_bytes: int = 0,
        full_cap_admission: bool = False,
    ) -> None:
        assert self.output_dir is not None
        roots = tuple(dict.fromkeys(path.resolve() for path in account_roots))
        local_status: dict[str, Any] = {"rank": self.rank, "ok": True}
        if self.rank == 0:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=False)
                info = _filesystem_info(self.output_dir)
                baseline = _tree_bytes(roots)
                if baseline > _CAP_BYTES:
                    raise OSError(
                        f"existing validation files exceed {_CAP_BYTES} bytes"
                    )
                if info["available_inodes"] < 256:
                    raise OSError("probe storage has fewer than 256 available inodes")
                free_need = (
                    max(0, _CAP_BYTES - baseline)
                    if full_cap_admission
                    else expected_payload_bytes
                )
                if info["free_bytes"] < free_need + _FREE_RESERVE_BYTES:
                    raise OSError(
                        "probe storage lacks the projected payload plus 32-GiB reserve"
                    )
                check_path = self.output_dir / ".storage-write-check"
                with check_path.open("xb") as stream:
                    stream.write(b"campaign numerical probe storage check\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                if (
                    check_path.read_bytes()
                    != b"campaign numerical probe storage check\n"
                ):
                    raise OSError("probe storage write/read check failed")
                check_path.unlink()
                info["baseline_live_bytes"] = baseline
                info["cap_bytes"] = _CAP_BYTES
                info["free_reserve_bytes"] = _FREE_RESERVE_BYTES
                info["expected_payload_bytes"] = expected_payload_bytes
                info["full_cap_admission"] = full_cap_admission
                info["account_roots"] = [str(path) for path in roots]
                _atomic_json(self.output_dir / ".storage.json", info)
                counter = self.output_dir / ".payload-bytes"
                with counter.open("x", encoding="ascii") as stream:
                    stream.write(str(_tree_bytes(roots, exclude=counter)))
                    stream.flush()
                    os.fsync(stream.fileno())
                local_status["storage_info"] = info
            except Exception as error:
                local_status = {
                    "rank": self.rank,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        statuses: list[Any] = [None] * self.world_size
        dist.all_gather_object(statuses, local_status)
        failures = [status for status in statuses if not status.get("ok")]
        if failures:
            raise RuntimeError(f"probe storage admission failed: {failures}")
        self._account_roots = roots
        self._storage_info = json.loads(
            (self.output_dir / ".storage.json").read_text(encoding="utf-8")
        )
        self.budget = _PayloadBudget(self.output_dir, roots)

    def _writer(
        self,
        filename: str,
        *,
        pre_reserved_bytes: int = 0,
        pre_reserved_families: frozenset[str] = frozenset(),
    ) -> _TensorPackWriter:
        if self.budget is None or self.output_dir is None:
            raise RuntimeError("probe output is not initialized")
        return _TensorPackWriter(
            self.output_dir / filename,
            self.budget,
            pre_reserved_bytes=pre_reserved_bytes,
            pre_reserved_families=pre_reserved_families,
        )

    def _publish_phase(
        self,
        phase: str,
        local_status: dict[str, Any],
        *,
        complete: bool = False,
    ) -> None:
        self._phase_history.append({"phase": phase, **local_status})
        statuses: list[Any] = [None] * self.world_size
        dist.all_gather_object(statuses, local_status)
        histories: list[Any] = [None] * self.world_size
        dist.all_gather_object(histories, self._phase_history)
        failures = [status for status in statuses if not status.get("ok")]
        publish_error = None
        if self.rank == 0:
            root = {
                "schema_version": _SCHEMA,
                "mode": self.config.mode,
                "phase": phase,
                "world_size": self.world_size,
                "updates": self.config.updates,
                "ranks": statuses,
                "phase_history_by_rank": histories,
                "complete": False,
                "storage": self._storage_info,
            }
            if self.budget is None:
                publish_error = "storage budget unavailable; manifest omitted"
            else:
                try:
                    _atomic_json(
                        self.output_dir / "manifest.json", root, budget=self.budget
                    )
                    self.budget.reconcile()
                    if not failures and complete:
                        root["complete"] = True
                        _atomic_json(
                            self.output_dir / "manifest.json", root, budget=self.budget
                        )
                        self.budget.reconcile()
                except Exception as error:
                    publish_error = f"{type(error).__name__}: {error}"
                    root["complete"] = False
                    root["error"] = publish_error
                    if failures:
                        root["phase_failures"] = failures
                    try:
                        _atomic_json(
                            self.output_dir / "manifest.failure.json",
                            root,
                            budget=self.budget,
                        )
                    except Exception:
                        pass
        phase_error = None if not failures else f"phase failures: {failures}"
        if publish_error is not None and phase_error is not None:
            error_detail = (
                f"{phase_error}; manifest publication failed: {publish_error}"
            )
        else:
            error_detail = publish_error or phase_error
        publish_status = {
            "rank": self.rank,
            "ok": not failures and (self.rank != 0 or publish_error is None),
            "error": error_detail,
        }
        final_statuses: list[Any] = [None] * self.world_size
        dist.all_gather_object(final_statuses, publish_status)
        dist.barrier()
        if any(not status.get("ok") for status in final_statuses):
            raise RuntimeError(f"numerical probe {phase} failed: {final_statuses}")

    def _inventory(
        self, state: ObserverState, *, phase: str
    ) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        records = _named_parameters(state.model_parts)
        groups = _group_map(state)
        report_records = []
        for record in records:
            group = groups.get(id(record["parameter"]))
            item = _parameter_record(record, group, self.rank)
            state_entries = state.optimizer_state(record["parameter"])
            keys = sorted(state_entries)
            if set(keys) - set(_STATE_FAMILIES):
                raise ValueError(
                    f"unsupported native AdamW state keys for {item['canonical_fqn']}: {keys}"
                )
            if keys and set(keys) != set(_STATE_FAMILIES):
                raise ValueError(
                    f"partial native AdamW state for {item['canonical_fqn']}: {keys}"
                )
            item["state_keys"] = keys
            item["inventory_phase"] = phase
            item["state_steps"] = {}
            values = {"P": record["parameter"], **state_entries}
            item["family_hashes"] = {}
            for family in ("P", *_STATE_FAMILIES):
                value = values.get(family)
                if value is None:
                    item["family_hashes"][family] = None
                    continue
                local = _local_tensor(value)
                hash_phase = {
                    "P": "parameter_endpoint",
                    "exp_avg": "optimizer_state",
                    "exp_avg_sq": "optimizer_state",
                    "step": "optimizer_step",
                }[family]
                item["family_hashes"][family] = _logical_digest(
                    local,
                    _hash_header(
                        identity=f"{item['part_index']}:{item['canonical_fqn']}",
                        family=family,
                        phase=hash_phase,
                        tensor=local,
                    ),
                )
            for key, value in state_entries.items():
                if key == "step":
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise TypeError("native AdamW step must be a scalar tensor")
                    local = _local_tensor(value)
                    item["state_steps"][key] = {
                        "dtype": str(local.dtype),
                        "device": str(local.device),
                        "sha256": _logical_digest(
                            local,
                            _hash_header(
                                identity=item["canonical_fqn"],
                                family="step",
                                phase="optimizer_step",
                                tensor=local,
                            ),
                        ),
                    }
                elif not isinstance(value, torch.Tensor):
                    raise TypeError(f"native AdamW state {key!r} is not a tensor")
            report_records.append(
                {
                    "meta": item,
                    "parameter": record["parameter"],
                    "state": state_entries,
                }
            )
        return report_records, groups

    def _s0_report(self, state: ObserverState) -> dict[str, Any]:
        records, _ = self._inventory(state, phase="s0")
        return {
            "schema_version": _SCHEMA,
            "rank": self.rank,
            "world_size": self.world_size,
            "step": state.step,
            "dump_folder": str(Path(state.config.dump_folder).resolve()),
            "parameters": [entry["meta"] for entry in records],
            "buffers": _buffer_inventory(state.model_parts),
            "router_state": _router_state(state.model_parts),
        }

    def _native_state(self) -> dict[str, Any]:
        trainer = self._trainer
        callbacks = {
            key: value.state_dict()
            for key, value in trainer.callbacks.stateful_objects().items()
            if key != self.rng_continuation.state_key
        }
        return _snapshot_cpu(
            {
                "trainer": trainer.state_dict(),
                "dataloader": trainer.dataloader.state_dict(),
                "lr_scheduler": trainer.lr_schedulers.state_dict(),
                "callbacks": callbacks,
            }
        )

    def _save_small_state(self, filename: str, value: Any) -> dict[str, Any]:
        assert self.output_dir is not None and self.budget is not None
        buffer = io.BytesIO()
        torch.save(value, buffer)
        encoded = buffer.getvalue()
        self.budget.reserve(len(encoded))
        path = self.output_dir / filename
        partial = path.with_name(path.name + ".partial")
        with partial.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        return {
            "file": filename,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    @staticmethod
    def _state_equal(left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return (
                left.dtype == right.dtype
                and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left, right)
            )
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return left.keys() == right.keys() and all(
                CampaignNumericalProbe._state_equal(left[key], right[key])
                for key in left
            )
        if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
            return len(left) == len(right) and all(
                CampaignNumericalProbe._state_equal(a, b) for a, b in zip(left, right)
            )
        return type(left) is type(right) and left == right

    def _estimate_update_bytes(self, state: ObserverState) -> int:
        """Conservatively estimate direct tensor bytes for one native update."""
        total = 0
        for record in _named_parameters(state.model_parts):
            param = record["parameter"]
            local = _local_tensor(param)
            param_bytes = local.numel() * local.element_size()
            total += param_bytes  # post-update P, including frozen parameters
            if not param.requires_grad:
                continue
            total += 2 * param_bytes  # preclip and postclip gradients
            native_state = state.optimizer_state(param)
            keys = set(native_state)
            if keys and keys != set(_STATE_FAMILIES):
                raise ValueError(
                    f"partial native AdamW state at {record['canonical_fqn']}: {sorted(keys)}"
                )
            if not keys:
                # A parameter unused in S0 may receive its first gradient in the
                # observed window, creating two moment tensors and one step.
                total += 2 * param_bytes + 8
            else:
                total += sum(
                    _local_tensor(native_state[name]).numel()
                    * _local_tensor(native_state[name]).element_size()
                    for name in _STATE_FAMILIES
                )
        return total + 64  # loss, token count, and total norm scalars

    def after_checkpoint_save(self, state: ObserverState) -> None:
        if self.config.mode != "create_s0":
            return
        status: dict[str, Any]
        try:
            if state.step != 1 or not state.last_saved_checkpoint:
                raise ValueError(
                    "create_s0 must observe the completed native step-1 save"
                )
            expected = (
                Path(state.config.dump_folder)
                / state.config.checkpoint.folder
                / "step-1"
            ).resolve()
            if Path(state.last_saved_checkpoint).resolve() != expected:
                raise ValueError(
                    "native S0 checkpoint path differs from the resolved configuration"
                )
            if state.device.type == "cuda":
                torch.cuda.synchronize(state.device)
            self._ensure_output(
                account_roots=(Path(state.config.dump_folder),),
            )
            if len(self._rng_serialized) != 1:
                raise RuntimeError(
                    "native S0 serialization must call RNGContinuationState.state_dict exactly once"
                )
            report = self._s0_report(state)
            report["checkpoint_path"] = str(expected)
            report["native_rng_serialization"] = _rng_json(self._rng_serialized[0])
            report["native_state"] = self._save_small_state(
                f"rank-{self.rank:02d}.s0-native-state.pt", self._native_state()
            )
            _atomic_json(
                self.output_dir / f"rank-{self.rank:02d}.s0.json",
                report,
                budget=self.budget,
            )
            status = {
                "rank": self.rank,
                "ok": True,
                "report": f"rank-{self.rank:02d}.s0.json",
                "native_state_bytes": report["native_state"]["bytes"],
                "checkpoint_path": str(expected),
            }
        except Exception as error:
            status = {
                "rank": self.rank,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
            self._best_effort_failure_record(
                f"rank-{self.rank:02d}.s0.failure.json", status
            )
        try:
            self._publish_phase("s0", status, complete=True)
        finally:
            self.close()
        if not status["ok"]:
            raise RuntimeError(f"native S0 capture failed: {status}")

    def after_load(self, state: ObserverState) -> None:
        if self.config.mode != "observe":
            return
        checkpoint_path = state.config.checkpoint.initial_load_path
        source_dir = None
        pre_status: dict[str, Any] = {"rank": self.rank, "ok": True}
        source_report = None
        current_report = None
        native_state = None
        try:
            if checkpoint_path is None:
                raise ValueError("observe requires a native S0 load path")
            source_dir = _source_s0_path(checkpoint_path)
            if state.step != 1:
                raise ValueError(f"observe requires restored step 1, got {state.step}")
            source_manifest = json.loads(
                (source_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                source_manifest.get("mode") != "create_s0"
                or source_manifest.get("phase") != "s0"
                or not source_manifest.get("complete")
            ):
                raise ValueError(
                    "source S0 probe manifest is incomplete or has wrong mode"
                )
            rank_status = next(
                (
                    item
                    for item in source_manifest["ranks"]
                    if item.get("rank") == self.rank
                ),
                None,
            )
            if rank_status is None or not rank_status.get("ok"):
                raise ValueError("source S0 rank report is missing or invalid")
            source_report = json.loads(
                (source_dir / rank_status["report"]).read_text(encoding="utf-8")
            )
            self._source_rng_serialization = _rng_from_json(
                source_report["native_rng_serialization"]
            )
            if (
                Path(source_report["checkpoint_path"]).resolve()
                != Path(checkpoint_path).resolve()
            ):
                raise ValueError(
                    "source S0 manifest checkpoint path does not match initial_load_path"
                )
            if (
                _file_sha256(source_dir / source_report["native_state"]["file"])
                != source_report["native_state"]["sha256"]
            ):
                raise ValueError("source S0 native state record is corrupt")
            pending_rng = self.rng_continuation.pending_restore_snapshot()
            if pending_rng is None or not _same_rng(
                pending_rng, self._source_rng_serialization
            ):
                raise ValueError(
                    "loaded pending RNG differs from native S0 serialization output"
                )
            current_report = self._s0_report(state)
            expected_records = {
                (item["part_index"], item["canonical_fqn"]): item
                for item in source_report["parameters"]
            }
            for item in current_report["parameters"]:
                key = (item["part_index"], item["canonical_fqn"])
                if expected_records.get(key) != item:
                    raise ValueError(f"restored S0 state hash mismatch: {key}")
            if len(expected_records) != len(current_report["parameters"]):
                raise ValueError(
                    "restored S0 parameter inventory differs from native S0"
                )
            if not _same_loaded_buffer_inventory(
                source_report["buffers"], current_report["buffers"]
            ):
                raise ValueError("restored S0 model buffer inventory differs")
            if source_report["router_state"] != current_report["router_state"]:
                raise ValueError("restored S0 router inventory differs")
            source_state_path = source_dir / source_report["native_state"]["file"]
            source_state = torch.load(
                source_state_path, map_location="cpu", weights_only=False
            )
            native_state = self._native_state()
            if not self._state_equal(source_state, native_state):
                raise ValueError(
                    "loaded trainer/loader/scheduler/callback state differs from S0"
                )
            pre_status["estimated_update_bytes"] = self._estimate_update_bytes(state)
            pre_status["source_dump_folder"] = source_report["dump_folder"]
        except Exception as error:
            pre_status = {
                "rank": self.rank,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        statuses: list[Any] = [None] * self.world_size
        dist.all_gather_object(statuses, pre_status)
        pre_failures = [status for status in statuses if not status.get("ok")]
        if pre_failures:
            failed_status = {
                "rank": self.rank,
                "ok": False,
                "error": str(pre_failures),
            }
            self._best_effort_failure_record(
                f"rank-{self.rank:02d}.s0-verified.failure.json", failed_status
            )
            try:
                self._publish_phase("s0_verified", failed_status)
            finally:
                self.close()
            raise RuntimeError(f"native S0 verification failed: {pre_failures}")

        verified_status: dict[str, Any]
        try:
            expected_payload = (
                sum(int(status["estimated_update_bytes"]) for status in statuses)
                * self.config.updates
            )
            self._ensure_output(
                account_roots=(
                    Path(statuses[0]["source_dump_folder"]),
                    Path(state.config.dump_folder),
                ),
                expected_payload_bytes=expected_payload,
                full_cap_admission=self.config.updates == 10,
            )
            assert (
                source_report is not None
                and current_report is not None
                and native_state is not None
                and source_dir is not None
                and checkpoint_path is not None
            )
            current_report["source_checkpoint_path"] = str(
                Path(checkpoint_path).resolve()
            )
            current_report["source_s0_dir"] = str(source_dir.resolve())
            current_report["pending_rng_matches_serialized_output"] = True
            current_report["native_state_matches_s0"] = True
            source_scales = {
                (item["part_index"], item["name"]): item["sha256"]
                for item in source_report["buffers"]
                if _reconstructed_aux_scale(item)
            }
            current_report["reconstructed_aux_loss_scales_at_load"] = [
                {
                    "part_index": item["part_index"],
                    "name": item["name"],
                    "source_s0_sha256": source_scales[
                        (item["part_index"], item["name"])
                    ],
                    "loaded_sha256": item["sha256"],
                    "producer": "AuxLossTracker.weigh",
                }
                for item in current_report["buffers"]
                if _reconstructed_aux_scale(item)
            ]
            self._prior_p_hashes = {
                f"{item['part_index']}:{item['canonical_fqn']}": item["family_hashes"][
                    "P"
                ]
                for item in current_report["parameters"]
            }
            current_report["native_state"] = self._save_small_state(
                f"rank-{self.rank:02d}.loaded-native-state.pt", native_state
            )
            _atomic_json(
                self.output_dir / f"rank-{self.rank:02d}.s0-verified.json",
                current_report,
                budget=self.budget,
            )
            verified_status = {
                "rank": self.rank,
                "ok": True,
                "report": f"rank-{self.rank:02d}.s0-verified.json",
                "native_state_bytes": current_report["native_state"]["bytes"],
            }
        except Exception as error:
            self._abort_active_writer()
            verified_status = {
                "rank": self.rank,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
            self._best_effort_failure_record(
                f"rank-{self.rank:02d}.s0-verified.failure.json", verified_status
            )
        try:
            self._publish_phase("s0_verified", verified_status)
        finally:
            if not verified_status["ok"]:
                self.close()
        if not verified_status["ok"]:
            raise RuntimeError(f"native S0 verification failed: {verified_status}")

    def batch_start(self, state: ObserverState) -> None:
        if self.config.mode != "observe":
            return
        if not 2 <= state.step <= self.config.updates + 1:
            return
        error = None
        try:
            if state.step != self._next_observed_step:
                raise RuntimeError(
                    f"expected numerical probe step {self._next_observed_step}, got {state.step}"
                )
            if self._current is not None:
                raise RuntimeError("previous numerical probe update did not finish")
            estimated_bytes = self._estimate_update_bytes(state)
            if self.budget is None:
                raise RuntimeError("probe output budget is not initialized")
            self.budget.reserve(estimated_bytes)
            self._current = {
                "schema_version": _SCHEMA,
                "rank": self.rank,
                "step": state.step,
                "microbatches": [],
                "losses": [],
                "parameters": {},
                "rng_restored_checked": self._rng_restore_checked,
                "estimated_tensor_bytes": estimated_bytes,
            }
            self._local_error = None
            self._seen_inner_optimizers.clear()
            self._active_writer = self._writer(
                f"rank-{self.rank:02d}.step-{state.step:05d}.tensorpack",
                pre_reserved_bytes=estimated_bytes,
                pre_reserved_families=frozenset(
                    {"gradient", "P", "exp_avg", "exp_avg_sq", "step", "total_norm"}
                ),
            )
        except Exception as caught:
            error = caught
        self._collective_check("batch start", error)

    def microbatch_start(self, state: ObserverState) -> None:
        if self._current is not None:
            self._current_microbatch_index = state.microbatch_idx

    def capture_prepared_batch(
        self,
        *,
        step: int,
        microbatch_index: int | None,
        prepared: tuple[Any, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]],
    ) -> None:
        if (
            self.config.mode != "observe"
            or self._current is None
            or self._local_error is not None
        ):
            return
        try:
            if step != self._current["step"]:
                raise ValueError(
                    "prepared input step differs from active numerical probe step"
                )
            if microbatch_index is None or self._active_writer is None:
                raise RuntimeError(
                    "prepared batch arrived without a microbatch boundary"
                )
            if any(
                item["index"] == microbatch_index
                for item in self._current["microbatches"]
            ):
                raise ValueError(f"duplicate prepared microbatch {microbatch_index}")
            inputs, labels, extra_inputs, extra_kwargs = prepared
            identity = f"step:{step}:microbatch:{microbatch_index}"
            fields = {
                "input": _capture_input_tensor(
                    self._active_writer,
                    inputs,
                    key=f"{identity}:input",
                    identity=f"{identity}:input",
                    phase="actual_forward_input",
                ),
                "labels": _capture_input_tensor(
                    self._active_writer,
                    labels,
                    key=f"{identity}:labels",
                    identity=f"{identity}:labels",
                    phase="actual_forward_input",
                ),
                "extra_inputs": {},
                "extra_kwargs": {},
            }
            for name, value in sorted(extra_inputs.items()):
                fields["extra_inputs"][name] = _capture_input_tensor(
                    self._active_writer,
                    value,
                    key=f"{identity}:extra_inputs:{name}",
                    identity=f"{identity}:extra_inputs:{name}",
                    phase="actual_forward_input",
                )
            for name, value in sorted(extra_kwargs.items()):
                key = f"{identity}:extra_kwargs:{name}"
                if name == "attention_masks" and isinstance(value, torch.Tensor):
                    fields["extra_kwargs"][name] = _capture_input_tensor(
                        self._active_writer,
                        value,
                        key=key,
                        identity=key,
                        phase="actual_forward_input",
                    )
                elif name == "attention_masks" and isinstance(value, dict):
                    fields["extra_kwargs"][name] = {
                        mask_name: _capture_block_mask(
                            self._active_writer,
                            value[mask_name],
                            key=f"{key}:{mask_name}",
                            identity=f"{key}:{mask_name}",
                            phase="actual_forward_input",
                        )
                        for mask_name in sorted(value)
                    }
                elif name == "attention_masks" and value is not None:
                    fields["extra_kwargs"][name] = _capture_block_mask(
                        self._active_writer,
                        value,
                        key=key,
                        identity=key,
                        phase="actual_forward_input",
                    )
                else:
                    fields["extra_kwargs"][name] = _capture_input_tensor(
                        self._active_writer,
                        value,
                        key=key,
                        identity=key,
                        phase="actual_forward_input",
                    )
            self._current["microbatches"].append(
                {"index": microbatch_index, "inputs": fields}
            )
        except Exception as caught:
            self._remember_local_error("prepared input capture", caught)

    def _capture_aux_loss_weigh(self, tracker: Any) -> None:
        if self.config.mode != "observe" or self._current is None:
            return
        try:
            if self._current_microbatch_index is None:
                raise RuntimeError("native auxiliary scale has no active microbatch")
            index = self._current_microbatch_index
            microbatch = next(
                item for item in self._current["microbatches"] if item["index"] == index
            )
            if "native_aux_loss_scale" in microbatch:
                raise ValueError("native auxiliary scale was recorded twice")
            states = tracker._states
            if len(states) != len(self._trainer.model_parts):
                raise ValueError(
                    "AuxLossTracker state count changed during observation"
                )
            parts = []
            for part_index, (model, aux_state) in enumerate(
                zip(self._trainer.model_parts, states)
            ):
                scale = aux_state.scale
                module_names = []
                for module_name, module in model.named_modules():
                    if not getattr(module, "_aux_loss_kinds", ()):
                        continue
                    if getattr(module, "aux_loss_scale", None) is not scale:
                        raise ValueError(
                            "native auxiliary module is not wired to its tracker scale"
                        )
                    module_names.append(canonical_fqn(module_name) or "<root>")
                part = {"part_index": part_index, "module_names": module_names}
                if module_names:
                    if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
                        raise TypeError(
                            "native auxiliary scale must be a scalar tensor"
                        )
                    part["value"] = float(scale.detach().item())
                parts.append(part)
            microbatch["native_aux_loss_scale"] = {
                "producer": "AuxLossTracker.weigh",
                "boundary": "after_weigh_before_forward",
                "parts": parts,
            }
        except Exception as error:
            self._remember_local_error(
                "native AuxLossTracker.weigh verification", error
            )

    def after_dataloader(self, state: ObserverState) -> None:
        if (
            self.config.mode != "observe"
            or self._current is None
            or self._rng_restore_checked
            or self._local_error is not None
        ):
            return
        try:
            actual = self.rng_continuation.rng_state.state_dict()
            expected = self.rng_continuation.pending_restore_snapshot()
            if expected is not None:
                raise RuntimeError(
                    "RNG continuation restore did not consume its pending snapshot"
                )
            if self._source_rng_serialization is None or not _same_rng(
                actual, self._source_rng_serialization
            ):
                raise ValueError(
                    "first post-fetch RNG state differs from actual native S0 serialization output"
                )
            self._current["rng_restored_checked"] = True
        except Exception as caught:
            self._remember_local_error("first-fetch RNG check", caught)
        else:
            self._rng_restore_checked = True

    def after_forward(self, state: ObserverState) -> None:
        if (
            self.config.mode != "observe"
            or self._current is None
            or self._local_error is not None
        ):
            return
        try:
            if state.loss is None or state.global_valid_tokens is None:
                raise ValueError("observe requires native loss and valid-token tensors")
            if self._active_writer is None:
                raise RuntimeError("observe writer is not active")
            index = state.microbatch_idx
            if index is None:
                raise ValueError("native after_forward event has no microbatch index")
            identity = f"step:{state.step}:microbatch:{index}"
            loss = self._active_writer.add(
                f"{identity}:loss",
                state.loss,
                identity=identity,
                family="loss_sum",
                phase="after_forward",
            )
            valid_tokens = self._active_writer.add(
                f"{identity}:global_valid_tokens",
                state.global_valid_tokens,
                identity=identity,
                family="valid_tokens",
                phase="after_forward",
            )
            auxiliary = []
            for split, aux_state in enumerate(self._trainer.aux_loss._states):
                auxiliary.append(
                    {
                        "split": split,
                        "names": list(aux_state.names),
                        "scale": _logical_digest(
                            aux_state.scale,
                            _hash_header(
                                identity=identity,
                                family="aux_scale",
                                phase="after_forward",
                                tensor=aux_state.scale,
                            ),
                        ),
                        "totals": _logical_digest(
                            aux_state.totals,
                            _hash_header(
                                identity=identity,
                                family="aux_totals",
                                phase="after_forward",
                                tensor=aux_state.totals,
                            ),
                        ),
                        "scale_layout": _tensor_layout(aux_state.scale),
                        "totals_layout": _tensor_layout(aux_state.totals),
                    }
                )
            self._current["losses"].append(
                {
                    "index": index,
                    "loss": loss,
                    "global_valid_tokens": valid_tokens,
                    "auxiliary": auxiliary,
                }
            )
        except Exception as caught:
            self._remember_local_error("forward output capture", caught)

    def before_optimizer_step(self, state: ObserverState) -> None:
        if self.config.mode != "observe" or self._current is None:
            return
        error = (
            RuntimeError(self._local_error) if self._local_error is not None else None
        )
        try:
            if error is not None:
                raise error
            if state.step != self._current["step"]:
                raise ValueError(
                    "preclip event differs from active numerical probe step"
                )
            if self._active_writer is None:
                raise RuntimeError("observe writer is not active")
            if state.gradient_accumulation_steps != 4:
                raise ValueError(
                    "numerical probe requires the exact four-microbatch accumulation path"
                )
            if len(self._current["microbatches"]) != state.gradient_accumulation_steps:
                raise ValueError("prepared input capture missed a gradient microbatch")
            if any(
                "native_aux_loss_scale" not in item
                for item in self._current["microbatches"]
            ):
                raise ValueError("native auxiliary scale capture missed a microbatch")
            if len(self._current["losses"]) != state.gradient_accumulation_steps:
                raise ValueError("forward capture missed a gradient microbatch")
            if [item["index"] for item in self._current["microbatches"]] != list(
                range(4)
            ):
                raise ValueError(
                    "prepared input microbatch order differs from native order"
                )
            if [item["index"] for item in self._current["losses"]] != list(range(4)):
                raise ValueError("forward microbatch order differs from native order")
            if state.device.type == "cuda":
                torch.cuda.synchronize(state.device)
            reports, _ = self._inventory(state, phase="preclip")
            records = {}
            for entry in reports:
                meta, param = entry["meta"], entry["parameter"]
                identity = f"{meta['part_index']}:{meta['canonical_fqn']}"
                parameter_hash = meta["family_hashes"]["P"]
                previous_hash = self._prior_p_hashes.get(identity)
                if previous_hash is None or previous_hash != parameter_hash:
                    raise ValueError(
                        f"parameter endpoint is discontinuous at {identity}"
                    )
                grad = param.grad
                if grad is not None and isinstance(grad, DTensor):
                    grad = _local_tensor(grad, grad_placements=tuple(grad.placements))
                elif grad is not None:
                    grad = _local_tensor(grad)
                gradient = self._active_writer.add(
                    f"{identity}:grad_preclip",
                    grad,
                    identity=identity,
                    family="gradient",
                    phase="preclip",
                )
                records[identity] = {
                    "part_index": meta["part_index"],
                    "canonical_fqn": meta["canonical_fqn"],
                    "group": meta["group"],
                    "state_keys": meta["state_keys"],
                    "state_hashes": {
                        key: value
                        for key, value in meta["family_hashes"].items()
                        if key != "P"
                    },
                    "parameter_hash": parameter_hash,
                    "gradient": gradient,
                }
            self._current["parameters"] = records
            self._current["used_groups"] = [
                {
                    "part_index": group.part_index,
                    "group_index": group.group_index,
                    "optimizer_name": group.optimizer_name,
                    "hparams": _json_value(dict(group.hparams)),
                }
                for group in state.optimizer_groups()
            ]
            self._current["router_preclip"] = _router_state(state.model_parts)
            self._current["buffer_preclip"] = _buffer_inventory(state.model_parts)
        except Exception as caught:
            error = caught
        self._collective_check("preclip capture", error)

    def _inner_optimizer_pre_hook(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if self._current is None or self.config.mode != "observe":
            return None
        error = None
        try:
            if id(optimizer) in self._seen_inner_optimizers:
                raise RuntimeError("native optimizer pre-hook fired more than once")
            self._seen_inner_optimizers.add(id(optimizer))
            optimizer_index = next(
                index
                for index, candidate in enumerate(self._trainer.optimizers.optimizers)
                if candidate is optimizer
            )
            postclip = {}
            if self._active_writer is None:
                raise RuntimeError("observe writer is not active")
            for group_index, group in enumerate(optimizer.param_groups):
                for param_index, param in enumerate(group["params"]):
                    record = next(
                        (
                            item
                            for item in self._current["parameters"].values()
                            if item["group"]
                            and item["group"]["part_index"] == optimizer_index
                            and item["group"]["group_index"] == group_index
                            and item["group"]["param_index"] == param_index
                        ),
                        None,
                    )
                    if record is None:
                        raise ValueError(
                            "inner AdamW parameter does not match preclip inventory"
                        )
                    identity = f"{record['part_index']}:{record['canonical_fqn']}"
                    grad = param.grad
                    if grad is not None:
                        if isinstance(grad, DTensor):
                            grad = _local_tensor(
                                grad, grad_placements=tuple(grad.placements)
                            )
                        else:
                            grad = _local_tensor(grad)
                    entry = self._active_writer.add(
                        f"{identity}:grad_postclip",
                        grad,
                        identity=identity,
                        family="gradient",
                        phase="postclip",
                    )
                    postclip[identity] = {
                        "gradient": entry,
                        "lr_used": _json_value(group["lr"]),
                    }
            self._current.setdefault("postclip", {}).update(postclip)
            self._current["router_after_container_hook"] = _router_state(
                tuple(self._trainer.model_parts)
            )
        except Exception as caught:
            error = caught
        self._collective_check("postclip optimizer hook", error)
        return None

    def after_optimizer_step(self, state: ObserverState) -> None:
        if self.config.mode != "observe" or self._current is None:
            return
        error = None
        try:
            if state.step != self._current["step"]:
                raise ValueError(
                    "post-update event differs from active numerical probe step"
                )
            if state.grad_norm is None or self._active_writer is None:
                raise ValueError(
                    "native optimizer event is missing its norm or payload writer"
                )
            if len(self._seen_inner_optimizers) != len(
                self._trainer.optimizers.optimizers
            ):
                raise RuntimeError("not every native inner AdamW pre-hook fired")
            if state.device.type == "cuda":
                torch.cuda.synchronize(state.device)
            self._current["total_norm"] = self._active_writer.add(
                "native:total_norm",
                state.grad_norm,
                identity="native",
                family="total_norm",
                phase="native_clip_result",
            )
            self._current["updated_parameters"] = {}
            groups = _group_map(state)
            for record in _named_parameters(state.model_parts):
                param = record["parameter"]
                group = groups.get(id(param))
                identity = f"{record['part_index']}:{record['canonical_fqn']}"
                state_entries = state.optimizer_state(param)
                keys = set(state_entries)
                if keys and keys != set(_STATE_FAMILIES):
                    raise ValueError(
                        f"partial AdamW state at {identity}: {sorted(keys)}"
                    )
                payload = {}
                hashes = {}
                for family, value, phase in (
                    ("P", param, "parameter_endpoint"),
                    (
                        "exp_avg",
                        state_entries.get("exp_avg"),
                        "optimizer_state",
                    ),
                    (
                        "exp_avg_sq",
                        state_entries.get("exp_avg_sq"),
                        "optimizer_state",
                    ),
                    ("step", state_entries.get("step"), "optimizer_step"),
                ):
                    local = None if value is None else _local_tensor(value)
                    entry = self._active_writer.add(
                        f"{identity}:{family}",
                        local,
                        identity=identity,
                        family=family,
                        phase=phase,
                    )
                    payload[family] = entry
                    hashes[family] = entry["sha256"]
                self._current["updated_parameters"][identity] = {
                    "state_keys": sorted(keys),
                    "payload": payload,
                    "hashes": hashes,
                    "group": None
                    if group is None
                    else {
                        "part_index": group["part_index"],
                        "group_index": group["group_index"],
                        "param_index": group["param_index"],
                        "hparams": group["hparams"],
                    },
                }
                self._prior_p_hashes[identity] = hashes["P"]
            self._current["router_post_update"] = _router_state(state.model_parts)
            self._current["buffer_post_update"] = _buffer_inventory(state.model_parts)
            self._current["next_groups"] = [
                {
                    "part_index": group.part_index,
                    "group_index": group.group_index,
                    "hparams": _json_value(dict(group.hparams)),
                }
                for group in state.optimizer_groups()
            ]
        except Exception as caught:
            error = caught
        self._collective_check("post-update capture", error)

    def batch_end(self, state: ObserverState) -> None:
        if self.config.mode != "observe" or self._current is None:
            return
        step = self._current["step"]
        status: dict[str, Any]
        try:
            if state.step != step:
                raise ValueError("batch_end differs from active numerical probe step")
            if not self._current.get("rng_restored_checked"):
                raise RuntimeError("first-fetch RNG restoration was not verified")
            if state.loss is None or self._active_writer is None:
                raise ValueError("native batch_end state is incomplete")
            if not self._current.get("updated_parameters"):
                raise RuntimeError("native post-update tensors were not captured")
            self._current["loss_sum"] = self._active_writer.add(
                "native:loss_sum",
                state.loss,
                identity="native",
                family="loss_sum",
                phase="batch_end",
            )
            self._current["ntokens_seen"] = int(state.ntokens_seen)
            self._current["global_valid_tokens"] = _tensor_layout(
                state.global_valid_tokens
            )
            self._current["native_state"] = self._save_small_state(
                f"rank-{self.rank:02d}.step-{step:05d}.native-state.pt",
                self._native_state(),
            )
            self._current["buffers_at_batch_end"] = _buffer_inventory(state.model_parts)
            self._active_writer.close()
            self._current["payload_file"] = self._active_writer.final_path.name
            self._current["payload_sha256"] = (
                self._active_writer.payload_digest.hexdigest()
            )
            self._current["payload_entries"] = self._active_writer.entries
            self._current["payload_bytes"] = sum(
                int(entry["nbytes"]) for entry in self._active_writer.entries.values()
            )
            reader = _TensorPackReader(
                self._active_writer.final_path, self._active_writer.entries
            )
            try:
                self._current["readback"] = {
                    "verified_entries": reader.verify_all(),
                    "file_bytes": self._active_writer.final_path.stat().st_size,
                    "payload_sha256": self._current["payload_sha256"],
                }
            finally:
                reader.close()
            _atomic_json(
                self.output_dir / f"rank-{self.rank:02d}.step-{step:05d}.json",
                self._current,
                budget=self.budget,
            )
            status = {
                "rank": self.rank,
                "ok": True,
                "step": step,
                "report": f"rank-{self.rank:02d}.step-{step:05d}.json",
                "payload": self._active_writer.final_path.name,
                "payload_bytes": self._current["payload_bytes"],
                "payload_sha256": self._current["payload_sha256"],
                "readback_entries": self._current["readback"]["verified_entries"],
            }
        except Exception as error:
            status = {
                "rank": self.rank,
                "step": step,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
            self._best_effort_failure_record(
                f"rank-{self.rank:02d}.step-{step:05d}.failure.json", status
            )
            self._abort_active_writer()
        self._active_writer = None
        self._current = None
        try:
            self._publish_phase(
                f"step_{step}",
                status,
                complete=status.get("ok", False) and step == self.config.updates + 1,
            )
        finally:
            if not status.get("ok"):
                self.close()
        if not status.get("ok"):
            raise RuntimeError(f"numerical probe step {step} failed: {status}")
        if status.get("ok"):
            self._next_observed_step += 1
        if status.get("ok") and step == self.config.updates + 1:
            self.close()

    def _collective_check(self, phase: str, error: Exception | None) -> None:
        local = {
            "rank": self.rank,
            "ok": error is None,
            "error": None if error is None else f"{type(error).__name__}: {error}",
        }
        statuses: list[Any] = [None] * self.world_size
        dist.all_gather_object(statuses, local)
        failures = [status for status in statuses if not status.get("ok")]
        if failures:
            self._abort_active_writer()
            step = None if self._current is None else self._current["step"]
            failure_status = {
                "rank": self.rank,
                "ok": False,
                "error": str(failures),
            }
            try:
                self._publish_phase(
                    f"step_{step}_{phase}" if step is not None else phase,
                    failure_status,
                )
            except Exception as publish_error:
                self._current = None
                self.close()
                raise RuntimeError(
                    f"numerical probe {phase} failed: {failures}; "
                    f"failure publication: {publish_error}"
                ) from publish_error
            self._current = None
            self.close()
            raise RuntimeError(f"numerical probe {phase} failed: {failures}")

    def _remember_local_error(self, phase: str, error: Exception) -> None:
        if self._local_error is None:
            self._local_error = f"{phase}: {type(error).__name__}: {error}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_WRITE_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
