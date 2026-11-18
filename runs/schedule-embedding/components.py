#!/usr/bin/env python3
"""Measure staged row-owner embeddings on the captured production crop.

The default path only prints a CPU-only plan. CUDA/NCCL execution requires
``--execute`` and exactly eight ranks. Each invocation measures one immutable
prefix of the row-owner optimization ladder in a fresh process group.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[2]
GIFT = ROOT / "gift"
WORLD_SIZE = 8
VOCAB = 128256
WIDTH = 4096
BATCH = 2
SEQUENCE = 4096
WARMUP = 10
SAMPLES = 10
STAGE_ORDER = ("compact", "owner_add", "owner_lookup", "routing")
STAGE_LADDER = tuple(tuple(STAGE_ORDER[:index]) for index in range(len(STAGE_ORDER) + 1))
FIXTURE_ROOT = (
    ROOT
    / "runs/production-data-abcd/artifacts/raw/proddata-component-capture-001/native-inputs"
)
D_CACHE_ROOT = ROOT / "runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache"
RAW_ROOT = ROOT / "runs/schedule-embedding/artifacts/raw"
EXPECTED_PYTHON = Path("/opt/torchtitan-venv/bin/python3.12")

BASE_COMPONENTS = ROOT / "runs/ffn-row-owner/components.py"
FIXTURE_MODULE = ROOT / "runs/production-data-abcd/component_fixtures.py"
OWNER_MODULE_PATH = GIFT / "torchtitan/distributed/gfsdp/row_owner_embedding.py"
KERNEL_MODULE_PATH = GIFT / "torchtitan/distributed/gfsdp/_row_owner_embedding_kernels.py"
SOURCE_FILES = (
    GIFT / "torchtitan/distributed/gfsdp/fully_shard.py",
    GIFT / "torchtitan/distributed/gfsdp/_state.py",
    GIFT / "torchtitan/distributed/gfsdp/_param_group.py",
    GIFT / "torchtitan/distributed/gfsdp/_collectives.py",
    GIFT / "torchtitan/distributed/gfsdp/_common.py",
    GIFT / "torchtitan/distributed/gfsdp/_tensor.py",
    OWNER_MODULE_PATH,
    KERNEL_MODULE_PATH,
    BASE_COMPONENTS,
    FIXTURE_MODULE,
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_index(stages: tuple[str, ...]) -> int:
    try:
        return STAGE_LADDER.index(stages)
    except ValueError as exc:
        expected = " -> ".join("+".join(stage) if stage else "D0" for stage in STAGE_LADDER)
        raise ValueError(f"--stages must select one ordered ladder prefix: {expected}") from exc


def _stage_id(stages: tuple[str, ...]) -> str:
    index = _stage_index(stages)
    return "d0" if index == 0 else f"d{index}-" + "-".join(stages)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--execute",
        action="store_true",
        help="run the explicit eight-rank CUDA/NCCL component measurement",
    )
    result.add_argument(
        "--stages",
        nargs="*",
        choices=STAGE_ORDER,
        default=[],
        help="ordered prefix: compact owner_add owner_lookup routing",
    )
    result.add_argument(
        "--input-fixtures",
        type=Path,
        default=FIXTURE_ROOT,
        help="captured native first-microbatch inputs, one receipt per rank",
    )
    result.add_argument(
        "--output",
        type=Path,
        help="fresh output directory below runs/schedule-embedding/artifacts/raw",
    )
    return result


def validate(args: argparse.Namespace) -> tuple[str, ...]:
    stages = tuple(args.stages)
    _stage_index(stages)
    if args.execute and args.output is None:
        raise ValueError("--execute requires a fresh --output directory")
    return stages


def validate_world_size(world: int) -> None:
    if world != WORLD_SIZE:
        raise ValueError(f"production component execution requires WORLD_SIZE={WORLD_SIZE}; found {world}")


def fixture_manifest(root: Path) -> list[dict[str, Any]]:
    """Validate the small capture receipts without importing Torch or hashing 2.5 GiB."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"production component fixture directory is missing or unsafe: {root}")
    root = root.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for rank in range(WORLD_SIZE):
        rank_dir = root / f"rank{rank}"
        receipt_path = rank_dir / "receipt.json"
        tensor_path = rank_dir / "tensors.pt"
        if rank_dir.is_symlink() or receipt_path.is_symlink() or tensor_path.is_symlink():
            raise ValueError(f"fixture rank {rank} must not traverse symlinks")
        if not rank_dir.is_dir() or not receipt_path.is_file() or not tensor_path.is_file():
            raise FileNotFoundError(f"fixture rank {rank} is incomplete: {rank_dir}")
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        expected_shape = [BATCH, SEQUENCE]
        if (
            receipt.get("schema_version") != 1
            or receipt.get("rank") != rank
            or receipt.get("world_size") != WORLD_SIZE
            or receipt.get("microbatch") != "first complete native forward_backward_step"
            or receipt.get("input_ids_shape") != expected_shape
            or not receipt.get("input_ids_int64_sha256")
            or not receipt.get("tensors_sha256")
            or not receipt.get("source_head")
        ):
            raise ValueError(f"fixture rank {rank} receipt does not describe the production B2x4096 capture")
        rows.append(
            {
                "rank": rank,
                "world_size": receipt["world_size"],
                "microbatch": receipt["microbatch"],
                "tokens": BATCH * SEQUENCE,
                "input_ids_shape": expected_shape,
                "input_ids_int64_sha256": receipt["input_ids_int64_sha256"],
                "tensor_payload_sha256": receipt["tensors_sha256"],
                "fixture_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "capture_source_head": receipt["source_head"],
                "payload_hash_rechecked": False,
                "owner_distribution": None,
            }
        )
    if len({row["input_ids_int64_sha256"] for row in rows}) != WORLD_SIZE:
        raise ValueError("production fixture rank-local ID hashes must be distinct")
    return rows


def plan(args: argparse.Namespace) -> dict[str, Any]:
    stages = validate(args)
    return {
        "execution": bool(args.execute),
        "execute_flag_required_for_cuda": True,
        "stage_id": _stage_id(stages),
        "optimization_stages": list(stages),
        "stage_ladder": [list(stage) for stage in STAGE_LADDER],
        "shape": {
            "vocabulary": VOCAB,
            "width": WIDTH,
            "batch": BATCH,
            "sequence": SEQUENCE,
            "tokens_per_rank": BATCH * SEQUENCE,
            "dtype_contract": "FP32 owner master and returned local gradient; BF16 lookup and source coalescing",
        },
        "topology": {
            "world_size": WORLD_SIZE,
            "backend": "NCCL",
            "device": "one H100 per rank",
            "fresh_process_per_stage_tuple": True,
        },
        "warmup": WARMUP,
        "samples": SAMPLES,
        "fixture_root": str(Path(args.input_fixtures).resolve()),
        "fixtures": fixture_manifest(args.input_fixtures),
        "compiler_cache": {
            "policy": "reuse validated D cache",
            "root": str(D_CACHE_ROOT),
            "TORCHINDUCTOR_CACHE_DIR": str(D_CACHE_ROOT / "torchinductor"),
            "TRITON_CACHE_DIR_pattern": str(D_CACHE_ROOT / "triton/rank_{rank}"),
            "set_before_torch_or_triton_import": True,
        },
        "timers": {
            "forward": "joined CUDA event span, host call, end synchronization, synchronized wall",
            "backward": "separate joined CUDA event span after forward completion; includes routed gradient exchange and autograd accumulation",
            "integrated": "one ordinary model(ids).backward(gradient) call with joined streams",
            "memory": "absolute allocated/reserved peaks after per-call reset; gradient zeroing and barriers excluded",
            "profiling": "one extra forward/backward pair after timing, split into separate raw forward/backward traces",
        },
        "temporary_bytes_accounting": {
            "full_native_bf16_gradient_table": VOCAB * WIDTH * 2,
            "compact_native_output_per_unique_row": WIDTH * 2,
            "required_owned_fp32_gradient_per_rank": (VOCAB // WORLD_SIZE) * WIDTH * 4,
            "native_fp32_partial_scratch": None,
            "native_fp32_partial_note": "CUDA embedding backward internal partial allocation is not surfaced by the native API; allocator peaks are retained, exact bytes remain unknown.",
            "route_state_and_borrowed_scratch": "computed from the actual rank-local input, captured route splits, and first excluded warmup; scratch views are compared with the existing private lm_head buffer",
        },
        "output": None if args.output is None else str(Path(args.output).resolve()),
    }


def _safe_directory(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    return path.resolve(strict=True)


def configure_compiler_caches(rank: int) -> dict[str, str]:
    """Route to the established D cache before importing Torch or Triton."""
    root = _safe_directory(D_CACHE_ROOT, "saved D compiler cache")
    inductor = _safe_directory(root / "torchinductor", "saved D TorchInductor cache")
    triton = _safe_directory(root / "triton" / f"rank_{rank}", "saved D rank-local Triton cache")
    paths = {"TORCHINDUCTOR_CACHE_DIR": str(inductor), "TRITON_CACHE_DIR": str(triton)}
    for key, value in paths.items():
        os.environ[key] = value
    return {"root": str(root), "rank": rank, **paths}


def _import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_snapshot() -> dict[str, Any]:
    files = {}
    for path in SOURCE_FILES:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"required measurement source is missing or unsafe: {path}")
        files[str(path.relative_to(ROOT))] = _sha256(path)
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    return {"revision": head, "files_sha256": files}


@contextmanager
def staged_factory(row_owner_class: type, stages: tuple[str, ...]) -> Iterator[None]:
    """Set the diagnostic prefix only during the shared builder call."""
    descriptor = row_owner_class.__dict__.get("from_embedding")
    if descriptor is None or not isinstance(descriptor, classmethod):
        raise RuntimeError("RowOwnerEmbedding.from_embedding is not the expected classmethod")
    original = descriptor.__func__

    def from_embedding(
        cls: type,
        embedding: Any,
        dense_mesh: Any,
        *,
        shard_alignment: int = 8,
        optimize: bool = False,
        optimization_stages: tuple[str, ...] | None = None,
    ) -> Any:
        if optimize or optimization_stages is not None:
            raise RuntimeError("component builder must not override its staged factory selection")
        if shard_alignment != 8:
            raise RuntimeError("component builder must retain the canonical row shard alignment")
        return original(cls, embedding, dense_mesh, optimization_stages=stages)

    row_owner_class.from_embedding = classmethod(from_embedding)
    try:
        yield
    finally:
        row_owner_class.from_embedding = descriptor


@contextmanager
def first_warmup_observer(
    owner: Any,
    row_owner_module: Any,
    stages: tuple[str, ...],
    observations: dict[str, Any],
) -> Iterator[None]:
    """Capture dispatch, route shapes, and borrowed scratch on the excluded first warmup."""
    original_route = owner._route
    original_scratch = owner._scratch
    original_coalesce = row_owner_module._native_source_coalesce
    had_route = "_route" in vars(owner)
    had_scratch = "_scratch" in vars(owner)
    observations.setdefault("route", [])
    observations.setdefault("scratch", [])
    observations.setdefault("coalescing", [])
    observations.setdefault("kernel_calls", [])
    scratch_index = [0]
    owner_storage_ptrs = {
        buffer.untyped_storage().data_ptr()
        for group in owner._sync_groups
        if group.grad_buffer_ctx is not None
        and hasattr(group.grad_buffer_ctx, "buffers")
        for buffer in group.grad_buffer_ctx.buffers.values()
    }
    row_owner_module._native_source_coalesce = original_coalesce

    def observe_route(ids: Any) -> Any:
        result = original_route(ids)
        unique, inverse, requested, send, receive = result
        observations["route"].append(
            {
                "input_tokens": int(ids.numel()),
                "input_shape": list(ids.shape),
                "unique_count": int(unique.numel()),
                "inverse_count": int(inverse.numel()),
                "requested_count": int(requested.numel()),
                "send_counts_by_owner": [int(value) for value in send],
                "receive_counts_by_source": [int(value) for value in receive],
            }
        )
        return result

    @contextmanager
    def observe_scratch(size: int) -> Iterator[Any]:
        with original_scratch(size) as view:
            storage = view.untyped_storage()
            operation = "forward" if scratch_index[0] == 0 else "backward"
            scratch_index[0] += 1
            observations["scratch"].append(
                {
                    "scope": "first_warmup_integrated",
                    "operation": operation,
                    "requested_bytes": int(size),
                    "view_bytes": int(view.numel() * view.element_size()),
                    "storage_bytes": int(storage.nbytes()),
                    "storage_offset_bytes": int(view.storage_offset() * view.element_size()),
                    "aliases_owner_gfsdp_gradient_storage": storage.data_ptr() in owner_storage_ptrs,
                }
            )
            yield view

    def observe_coalesce(
        grad_bf16: Any,
        indices: Any,
        unique: Any,
        num_weights: int,
        *,
        compact_rows: int | None = None,
        output: Any = None,
    ) -> Any:
        result = original_coalesce(
            grad_bf16,
            indices,
            unique,
            num_weights,
            compact_rows=compact_rows,
            output=output,
        )
        observations["coalescing"].append(
            {
                "input_tokens": int(indices.numel()),
                "num_weights_argument": int(num_weights),
                "compact_rows_argument": None if compact_rows is None else int(compact_rows),
                "full_native_bf16_table_bytes": 0 if compact_rows is not None else VOCAB * WIDTH * 2,
                "native_return_shape": list(result.shape),
                "native_return_dtype": str(result.dtype),
                "native_return_bytes": int(result.numel() * result.element_size()),
                "native_fp32_partial_scratch_bytes": None,
            }
        )
        return result

    owner._route = observe_route
    owner._scratch = observe_scratch
    row_owner_module._native_source_coalesce = observe_coalesce
    kernel_module = None
    kernel_originals: dict[str, Callable[..., Any]] = {}
    kernel_names = {
        "owner_add": ("owner_add_bf16_",),
        "owner_lookup": ("owner_lookup_bf16",),
        "routing": ("validate_and_clamp_ids", "build_routing_metadata"),
    }
    try:
        if any(stage in stages for stage in kernel_names):
            kernel_module = __import__(
                "torchtitan.distributed.gfsdp._row_owner_embedding_kernels",
                fromlist=["_row_owner_embedding_kernels"],
            )
            for stage in stages:
                for name in kernel_names.get(stage, ()):
                    original_call = getattr(kernel_module, name)
                    kernel_originals[name] = original_call

                    def observe_kernel(
                        *args: Any,
                        _name: str = name,
                        _call: Callable[..., Any] = original_call,
                        **kwargs: Any,
                    ) -> Any:
                        observations["kernel_calls"].append(_name)
                        return _call(*args, **kwargs)

                    setattr(kernel_module, name, observe_kernel)
        yield
    finally:
        if kernel_module is not None:
            for name, original_call in kernel_originals.items():
                setattr(kernel_module, name, original_call)
        row_owner_module._native_source_coalesce = original_coalesce
        if had_route:
            owner._route = original_route
        else:
            delattr(owner, "_route")
        if had_scratch:
            owner._scratch = original_scratch
        else:
            delattr(owner, "_scratch")


def _allocator_counters(torch: Any) -> dict[str, int | None]:
    stats = torch.cuda.memory_stats()
    return {
        "allocation_retries": stats.get("num_alloc_retries"),
        "ooms": stats.get("num_ooms"),
    }


def _live_bytes(torch: Any) -> dict[str, int]:
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def _local_tensor(tensor: Any) -> Any:
    return tensor._local_tensor if hasattr(tensor, "_local_tensor") else tensor


def _gradient_state(torch: Any, owner: Any) -> dict[str, Any]:
    gradient = owner.weight.grad
    if gradient is None:
        return {"present": False}
    local = _local_tensor(gradient)
    if gradient.dtype != torch.float32 or local.dtype != torch.float32:
        raise RuntimeError("row-owner returned gradient must remain FP32")
    return {
        "present": True,
        "dtype": str(local.dtype),
        "local_shape": list(local.shape),
        "logical_shape": list(gradient.shape),
        "local_numel": int(local.numel()),
        "local_bytes": int(local.numel() * local.element_size()),
        "storage_bytes": int(local.untyped_storage().nbytes()),
    }


def _reset_gradient(torch: Any, owner: Any) -> dict[str, Any]:
    gradient = owner.weight.grad
    if gradient is not None:
        if gradient.dtype != torch.float32:
            raise RuntimeError("refusing to reset a non-FP32 accumulated owner gradient")
        gradient.zero_()
    torch.cuda.synchronize()
    return {
        "outside_timed_region": True,
        "live_bytes_after_reset": _live_bytes(torch),
        "gradient": _gradient_state(torch, owner),
    }


def _measured_call(torch: Any, helper: Any, fn: Callable[[], Any], comm_ctx: Any) -> tuple[Any, dict[str, Any]]:
    torch.cuda.reset_peak_memory_stats()
    starting_bytes = _live_bytes(torch)
    counters_before = _allocator_counters(torch)
    result, timers = helper.timed_call(fn, comm_ctx)
    counters_after = _allocator_counters(torch)
    return result, {
        **timers,
        "starting_bytes": starting_bytes,
        "allocator_counters_before": counters_before,
        "allocator_counters_after": counters_after,
        "allocator_retry_delta": (
            None
            if counters_before["allocation_retries"] is None or counters_after["allocation_retries"] is None
            else counters_after["allocation_retries"] - counters_before["allocation_retries"]
        ),
        "allocator_oom_delta": (
            None
            if counters_before["ooms"] is None or counters_after["ooms"] is None
            else counters_after["ooms"] - counters_before["ooms"]
        ),
    }


def _source_coalescing_budget(observations: dict[str, Any]) -> dict[str, Any]:
    calls = observations.get("coalescing", [])
    if not calls:
        return {"status": "not_observed"}
    call = calls[0]
    compact = call["compact_rows_argument"] is not None
    return {
        "dispatch": "compact_native_output" if compact else "full_native_table_then_unique_gather",
        "native_output_bytes": call["native_return_bytes"],
        "native_full_bf16_table_bytes": call["full_native_bf16_table_bytes"],
        "native_fp32_partial_scratch_bytes": None,
        "native_fp32_partial_note": "internal CUDA embedding-backward scratch is not exposed by the native API; allocator peaks cover its allocation, exact bytes remain unknown",
    }


def temporary_bytes_accounting(
    route: dict[str, Any],
    scratch: list[dict[str, Any]],
    coalescing: dict[str, Any],
    stages: tuple[str, ...],
) -> dict[str, Any]:
    tokens = route["input_tokens"]
    unique = route["unique_count"]
    requested = route["requested_count"]
    compact_dispatch = coalescing.get("dispatch") == "compact_native_output"
    saved_route_bytes = (tokens + requested + (0 if compact_dispatch else unique)) * 8
    scratch_bytes = [row["requested_bytes"] for row in scratch]
    route_transient = {
        "clamped_or_safe_ids": tokens * 8,
        "inverse_ids": 0 if compact_dispatch else tokens * 8,
        "unique_ids": unique * 8 if compact_dispatch else 0,
        "local_count_metadata": (WORLD_SIZE + 1) * 8,
        "gathered_count_matrix": WORLD_SIZE * (WORLD_SIZE + 1) * 8,
        "fallback_owner_index": 0 if "routing" in stages else unique * 8,
        "fallback_owner_count_ones": 0 if "routing" in stages else unique * 8,
    }
    saved_tensor_metadata = [
        {
            "name": "inverse" if compact_dispatch else "input_ids",
            "shape": [tokens] if compact_dispatch else route.get("input_shape", [tokens]),
            "dtype": "int64",
            "bytes": tokens * 8,
            "storage_note": (
                "inverse tensor saved by value"
                if compact_dispatch
                else "original input IDs retained by reference"
            ),
        }
    ]
    if not compact_dispatch:
        saved_tensor_metadata.append(
            {"name": "unique_ids", "shape": [unique], "dtype": "int64", "bytes": unique * 8}
        )
    saved_tensor_metadata.append(
        {"name": "local_requested_rows", "shape": [requested], "dtype": "int64", "bytes": requested * 8}
    )
    return {
        "rank_local": {
            "input_tokens": tokens,
            "unique_rows": unique,
            "rows_requested_from_this_owner": requested,
            "saved_route_tensor_bytes": {
                "autograd_saved_tensors": saved_tensor_metadata,
                "ids_or_compact_inverse": tokens * 8,
                "unique_ids": 0 if compact_dispatch else unique * 8,
                "requested_local_rows": requested * 8,
                "total": saved_route_bytes,
                "lifetime": "forward autograd context through backward; input-ID storage is retained by reference when compact is inactive",
            },
            "forward_route_transient_bytes": {
                **route_transient,
                "known_tensor_total": sum(route_transient.values()),
                "scope_note": "known int64 routing/count tensors only; excludes short-lived validity boolean temporaries and allocator reuse",
            },
            "forward_output_bf16_bytes": tokens * WIDTH * 2,
            "required_owned_fp32_gradient_bytes": (VOCAB // WORLD_SIZE) * WIDTH * 4,
            "native_source_coalescing": coalescing,
            "owner_scratch_views_first_warmup": scratch,
            "scratch_max_requested_bytes": max(scratch_bytes, default=0),
            "scratch_storage_is_borrowed": all(
                row["aliases_owner_gfsdp_gradient_storage"] for row in scratch
            ),
            "stage_prefix": list(stages),
            "native_fp32_partial_scratch_bytes": None,
            "native_fp32_partial_note": "allocator peaks are measured; the CUDA native op does not expose exact partial-scratch bytes",
        },
        "aggregation": "rank-local values retained for all eight ranks; memory is not summed across ranks",
    }


def _trace_summary(path: Path) -> dict[str, Any]:
    trace = json.loads(path.read_text())
    events = trace.get("traceEvents", [])
    gpu_events: list[dict[str, Any]] = []
    kernels: list[dict[str, Any]] = []
    copies: list[dict[str, Any]] = []
    memsets: list[dict[str, Any]] = []
    for event in events:
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
            continue
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", ""))
        lowered_name = name.lower()
        is_kernel = "kernel" in category or "nccldevkernel" in lowered_name
        is_memset = "memset" in category or "memset" in lowered_name
        is_copy = "memcpy" in category or "copy" in category or "memcpy" in lowered_name
        if not (is_kernel or is_copy or is_memset):
            continue
        row = {"name": name, "cat": category, "ts": float(event["ts"]), "dur": float(event["dur"])}
        gpu_events.append(row)
        if is_kernel:
            kernels.append(row)
        elif is_memset:
            memsets.append(row)
        elif is_copy:
            copies.append(row)
    nccl = [row for row in kernels if "nccl" in row["name"].lower()]
    local = [row for row in kernels if row not in nccl]
    intervals = sorted((row["ts"], row["ts"] + row["dur"]) for row in gpu_events)
    union = 0.0
    end = None
    for start, stop in intervals:
        if end is None or start > end:
            union += stop - start
            end = stop
        elif stop > end:
            union += stop - end
            end = stop
    names: dict[str, int] = {}
    for row in kernels:
        names[row["name"]] = names.get(row["name"], 0) + 1
    first = intervals[0][0] if intervals else None
    last = max((stop for _, stop in intervals), default=None)
    return {
        "raw_event_count": len(events),
        "kernel_count": len(kernels),
        "local_kernel_count": len(local),
        "nccl_kernel_count": len(nccl),
        "copy_count": len(copies),
        "memset_count": len(memsets),
        "kernel_duration_sum_ms": sum(row["dur"] for row in kernels) / 1000,
        "local_kernel_duration_sum_ms": sum(row["dur"] for row in local) / 1000,
        "nccl_kernel_duration_sum_ms": sum(row["dur"] for row in nccl) / 1000,
        "copy_duration_sum_ms": sum(row["dur"] for row in copies) / 1000,
        "memset_duration_sum_ms": sum(row["dur"] for row in memsets) / 1000,
        "device_event_interval_union_ms": union / 1000,
        "device_first_to_last_span_ms": None if first is None else (last - first) / 1000,
        "kernel_names": dict(sorted(names.items())),
        "scope_note": "profiled event sums may overlap; union is elapsed device-event coverage, not operation latency",
    }


def _profile_phase(
    torch: Any,
    helper: Any,
    label: str,
    fn: Callable[[], Any],
    comm_ctx: Any,
    observer: Any,
    rank: int,
    output: Path,
) -> tuple[Any, dict[str, Any]]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        with torch.profiler.record_function(f"row_owner_component::{label}"):
            result = fn()
            helper.join_streams(comm_ctx)
            torch.cuda.synchronize()
    stage_records = observer.collect()
    trace_path = output / f"rank{rank}-{label}.trace.json"
    profile.export_chrome_trace(str(trace_path))
    return result, {
        "trace": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "device_activity": _trace_summary(trace_path),
        "row_owner_stage_observer": stage_records,
    }


def _prepare_output(torch: Any, dist: Any, output: Path, rank: int, device: int) -> Path:
    raw_root = RAW_ROOT.resolve(strict=False)
    if output.is_absolute():
        selected = output
    else:
        selected = ROOT / output
    selected = selected.resolve(strict=False)
    if selected == raw_root or not selected.is_relative_to(raw_root):
        raise ValueError(f"--output must be a child of the ignored artifact area: {raw_root}")
    current = selected
    while current != raw_root:
        if current.exists() and current.is_symlink():
            raise ValueError(f"--output must not traverse symlinks: {current}")
        current = current.parent
    status: list[Any] = [None]
    if rank == 0:
        try:
            RAW_ROOT.mkdir(parents=True, exist_ok=True)
            if RAW_ROOT.is_symlink():
                raise ValueError(f"raw artifact area must not be a symlink: {RAW_ROOT}")
            selected.mkdir(parents=True, exist_ok=False)
            status[0] = {"ok": True}
        except Exception as exc:
            status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(status, src=0, device=torch.device("cuda", device))
    if not status[0]["ok"]:
        raise FileExistsError(f"refusing to reuse component output: {status[0]['error']}")
    dist.barrier()
    return selected


def _rank_max_aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) != WORLD_SIZE or sorted(row["rank"] for row in reports) != list(range(WORLD_SIZE)):
        raise RuntimeError("component run did not collect one completed receipt from all eight ranks")
    phase_names = ("forward", "backward", "integrated")
    aggregate: dict[str, Any] = {"rank_count": WORLD_SIZE, "phases": {}, "memory": {}}
    timers = ("cuda_elapsed_span_ms", "host_call_ms", "host_end_sync_ms", "synchronized_wall_ms")
    for phase in phase_names:
        phase_result: dict[str, Any] = {"rank_max_by_sample": {}, "rank_local_medians": {}}
        count = len(reports[0]["measurements"][phase])
        for timer in timers:
            rank_max = [
                max(report["measurements"][phase][sample][timer] for report in reports)
                for sample in range(count)
            ]
            phase_result["rank_max_by_sample"][timer] = {
                "samples": rank_max,
                "median": statistics.median(rank_max),
                "min": min(rank_max),
                "max": max(rank_max),
                "spread": max(rank_max) - min(rank_max),
            }
            phase_result["rank_local_medians"][timer] = [
                statistics.median(report["measurements"][phase][sample][timer] for sample in range(count))
                for report in reports
            ]
        for memory_name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            per_rank = [
                max(sample[memory_name] for sample in report["measurements"][phase])
                for report in reports
            ]
            phase_result[memory_name] = {
                "per_rank": per_rank,
                "rank_max": max(per_rank),
            }
        aggregate["phases"][phase] = phase_result
    aggregate["profile"] = [
        {"rank": report["rank"], "phases": report["profile"]} for report in reports
    ]
    aggregate["rank_max_note"] = "rank maxima are computed sample-wise; ranks are not summed; event duration sums may overlap"
    return aggregate


def execute(args: argparse.Namespace) -> None:
    if not args.execute:
        raise ValueError("CUDA/NCCL execution requires --execute")
    stages = validate(args)
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"execute with {EXPECTED_PYTHON}; found {sys.executable}")
    world = int(os.environ.get("WORLD_SIZE", "0"))
    validate_world_size(world)
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if not 0 <= rank < WORLD_SIZE or local_rank != rank:
        raise ValueError(f"single-node execution requires matching RANK/LOCAL_RANK, found {rank}/{local_rank}")
    fixture_manifest(args.input_fixtures)
    cache = configure_compiler_caches(rank)
    source_before = source_snapshot()
    sys.path.insert(0, str(GIFT))

    # Torch and Triton are imported only after the validated rank-local caches
    # have been placed in the process environment.
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from torch.distributed.device_mesh import init_device_mesh

    torch.cuda.set_device(local_rank)
    gpu_name = torch.cuda.get_device_name(local_rank)
    if "H100" not in gpu_name:
        raise RuntimeError(f"the production component plan requires H100; found {gpu_name}")
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    output = None
    try:
        output = _prepare_output(torch, dist, Path(args.output), rank, local_rank)
        helper = _import_file("row_owner_component_helpers", BASE_COMPONENTS)
        fixture_module = _import_file("row_owner_component_fixtures", FIXTURE_MODULE)
        captured, fixture_receipt = fixture_module.load_fixture(args.input_fixtures, rank)
        cpu_ids = captured["input_ids"]
        ids_sha256 = hashlib.sha256(cpu_ids.contiguous().numpy().tobytes()).hexdigest()
        if ids_sha256 != fixture_receipt["input_ids_int64_sha256"]:
            raise RuntimeError("loaded production input hash does not match its native capture receipt")
        if cpu_ids.dtype != torch.int64 or list(cpu_ids.shape) != [BATCH, SEQUENCE]:
            raise RuntimeError("loaded fixture is not the production B2x4096 signed-int64 input")
        traffic = helper.traffic(cpu_ids)
        traffic["input_ids_int64_sha256"] = ids_sha256
        input_ids = cpu_ids.to(device=f"cuda:{local_rank}")
        del captured, cpu_ids
        gradient_generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(2048 + rank)
        upstream = torch.randn(
            (BATCH, SEQUENCE, WIDTH),
            device=f"cuda:{local_rank}",
            dtype=torch.bfloat16,
            generator=gradient_generator,
        )

        row_owner_module = __import__(
            "torchtitan.distributed.gfsdp.row_owner_embedding",
            fromlist=["RowOwnerEmbedding"],
        )
        mesh = init_device_mesh("cuda", (WORLD_SIZE,), mesh_dim_names=("fsdp",))
        with staged_factory(row_owner_module.RowOwnerEmbedding, stages):
            model, comm_ctx = helper.build_embedding(mesh, "owner", seed=2048)
        owner = model.tok_embeddings
        if tuple(owner.optimization_stages) != stages:
            raise RuntimeError(
                f"constructed embedding stage marker mismatch: {owner.optimization_stages} != {stages}"
            )
        if owner.weight.dtype != torch.float32 or tuple(owner.weight.shape) != (VOCAB, WIDTH):
            raise RuntimeError("row-owner must retain the production FP32 [V,D] master")
        if model.lm_head.weight.dtype != torch.float32 or tuple(model.lm_head.weight.shape) != (VOCAB, WIDTH):
            raise RuntimeError("component root must retain the untied FP32 production lm_head")
        gradient_buffers = helper.gradient_buffer_inventory(model)
        if not gradient_buffers["owner_scratch_aliases_private_lm_head"]:
            raise RuntimeError("row-owner scratch must alias existing private lm_head GFSDP storage")
        if gradient_buffers["unique_storage_count"] != 1:
            raise RuntimeError("row-owner component unexpectedly created a second GFSDP gradient storage")

        observations: dict[str, Any] = {"scope": "first_warmup_integrated"}
        warmups = []
        for index in range(WARMUP):
            reset = (
                _reset_gradient(torch, owner)
                if index == 0
                else {"outside_timed_region": True, "performed": False}
            )
            torch.cuda.reset_peak_memory_stats()
            start_bytes = _live_bytes(torch)
            if index == 0:
                with first_warmup_observer(owner, row_owner_module, stages, observations):
                    _, timers = helper.timed_call(lambda: model(input_ids).backward(upstream), comm_ctx)
            else:
                _, timers = helper.timed_call(lambda: model(input_ids).backward(upstream), comm_ctx)
            gradient_state = _gradient_state(torch, owner)
            if not gradient_state["present"]:
                raise RuntimeError("ordinary autograd did not allocate the returned owner gradient")
            warmups.append(
                {
                    "index": index,
                    "first_execution": index == 0,
                    "instrumented_for_dispatch_and_memory_only": index == 0,
                    "reset": reset,
                    "starting_bytes": start_bytes,
                    **timers,
                    "gradient": gradient_state,
                }
            )
        torch.cuda.synchronize()
        if len(observations.get("route", [])) != 1 or len(observations.get("coalescing", [])) != 1:
            raise RuntimeError("first warmup did not observe exactly one forward route and source coalescing call")
        scratch_views = observations.get("scratch", [])
        if (
            len(scratch_views) != 2
            or [view["operation"] for view in scratch_views] != ["forward", "backward"]
            or any(not view["aliases_owner_gfsdp_gradient_storage"] for view in scratch_views)
        ):
            raise RuntimeError("first warmup scratch views did not both borrow existing GFSDP gradient storage")

        route = observations["route"][0]
        owner_distribution = list(traffic["unique_requests_by_owner"])
        per_rank_input = {
            **fixture_receipt,
            "input_ids_int64_sha256": ids_sha256,
            "tokens": int(input_ids.numel()),
            "owner_distribution": owner_distribution,
            "traffic": traffic,
            "route_observation": route,
        }
        # The receipt carries a large source manifest; the component report
        # retains only the provenance needed to identify this exact input.
        per_rank_input = {
            key: per_rank_input[key]
            for key in (
                "schema_version", "rank", "world_size", "microbatch", "source_head",
                "capture_sha256", "tensors_sha256", "input_ids_shape",
                "input_ids_int64_sha256", "tokens", "owner_distribution", "traffic",
                "route_observation",
            )
            if key in per_rank_input
        }
        coalescing_budget = _source_coalescing_budget(observations)
        memory_budget = temporary_bytes_accounting(
            route, observations["scratch"], coalescing_budget, stages
        )
        memory_lifecycle = {
            "first_execution": warmups[0],
            "remaining_warmup_count": WARMUP - 1,
            "warmup_memory": {
                "steady_warmup_count": WARMUP - 1,
                "max_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in warmups[1:]),
                "max_peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in warmups[1:]),
            },
            "gradient_reset_is_outside_timed_region": True,
            "gradient_after_warmup": _gradient_state(torch, owner),
        }

        measurements: dict[str, list[dict[str, Any]]] = {
            "forward": [], "backward": [], "integrated": []
        }
        for sample in range(SAMPLES):
            dist.barrier()
            reset = _reset_gradient(torch, owner)
            output_value, forward = _measured_call(
                torch, helper, lambda: model(input_ids), comm_ctx
            )
            _, backward = _measured_call(
                torch, helper, lambda output=output_value: output.backward(upstream), comm_ctx
            )
            gradient_state = _gradient_state(torch, owner)
            if not gradient_state["present"]:
                raise RuntimeError("separate backward failed to return the ordinary FP32 owner gradient")
            measurements["forward"].append(
                {"sample": sample, "reset": reset, **forward}
            )
            measurements["backward"].append(
                {
                    "sample": sample,
                    "starting_bytes": backward["starting_bytes"],
                    "gradient": gradient_state,
                    **backward,
                }
            )
            del output_value
            torch.cuda.synchronize()

        for sample in range(SAMPLES):
            dist.barrier()
            reset = _reset_gradient(torch, owner)
            _, integrated = _measured_call(
                torch,
                helper,
                lambda: model(input_ids).backward(upstream),
                comm_ctx,
            )
            gradient_state = _gradient_state(torch, owner)
            if not gradient_state["present"]:
                raise RuntimeError("integrated forward/backward failed to accumulate an FP32 owner gradient")
            measurements["integrated"].append(
                {"sample": sample, "reset": reset, "gradient": gradient_state, **integrated}
            )

        # Profiling is a separate representative pair after all unprofiled
        # samples. The stage observer and profiler never enter timing samples.
        dist.barrier()
        _reset_gradient(torch, owner)
        observer = helper.StageObserver()
        owner.stage_observer = observer
        profiled_output, forward_profile = _profile_phase(
            torch,
            helper,
            "forward",
            lambda: model(input_ids),
            comm_ctx,
            observer,
            rank,
            output,
        )
        dist.barrier()
        _, backward_profile = _profile_phase(
            torch,
            helper,
            "backward",
            lambda output=profiled_output: output.backward(upstream),
            comm_ctx,
            observer,
            rank,
            output,
        )
        owner.stage_observer = None
        del profiled_output

        source_after = source_snapshot()
        if source_after != source_before:
            raise RuntimeError("relevant source files changed while this component stage was running")
        report = {
            "schema_version": 1,
            "stage_id": _stage_id(stages),
            "optimization_stages_requested": list(stages),
            "optimization_stages_constructed": list(owner.optimization_stages),
            "rank": rank,
            "world_size": WORLD_SIZE,
            "command": sys.argv,
            "output": str(output),
            "runtime": {
                "python": sys.version,
                "executable": str(Path(sys.executable).resolve()),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": list(torch.cuda.nccl.version()),
                "gpu": gpu_name,
                "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(local_rank).total_memory),
                "hostname": os.uname().nodename,
            },
            "compiler_cache": cache,
            "source_before": source_before,
            "source_after": source_after,
            "fixture": per_rank_input,
            "gfsdp_gradient_buffer_inventory": gradient_buffers,
            "dispatch_observation_first_warmup": observations,
            "temporary_bytes_accounting": memory_budget,
            "warmup": warmups,
            "memory_lifecycle": memory_lifecycle,
            "measurements": measurements,
            "profile": {"forward": forward_profile, "backward": backward_profile},
            "scope_notes": {
                "forward": "production IDs; owner lookup, count exchange, routed value exchange and BF16 reconstruction included",
                "backward": "incoming BF16 gradient; source coalescing, routed gradient exchange, ordered owner accumulation and standard autograd accumulation included",
                "integrated": "ordinary autograd forward plus backward; no full lm_head compute; private head gradient storage remains resident as scratch backing in every stage",
                "host_call": "dispatch, Python/autograd, submission and internal host waits inside each call",
                "cuda_event": "joined compute, all-gather and reduce-scatter stream span; includes event queue gaps and is not kernel-duration sum",
                "memory": "absolute per-rank peaks include resident GFSDP storage; zeroing and barriers happen outside timers",
                "precision": "FP32 master and returned gradient, BF16 lookup/coalesced values and BF16 routed payloads; no result/gradient export",
            },
        }
        all_reports: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(all_reports, report)
        completed = [item for item in all_reports if item is not None]
        if len(completed) != WORLD_SIZE:
            raise RuntimeError("not every rank completed the component measurements")
        _atomic_write_json(output / f"rank{rank}.json", report)
        if rank == 0:
            aggregate = {
                "schema_version": 1,
                "stage_id": _stage_id(stages),
                "optimization_stages": list(stages),
                "ranks": completed,
                "aggregate": _rank_max_aggregate(completed),
            }
            _atomic_write_json(output / "all-ranks.json", aggregate)
        dist.barrier()
    except BaseException as exc:
        if output is not None:
            failure = {
                "rank": rank,
                "stage_id": _stage_id(stages),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "source_before": source_before,
            }
            try:
                _atomic_write_json(output / f"rank{rank}-failed.json", failure)
            except Exception:
                pass
        raise
    finally:
        dist.destroy_process_group()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    validate(args)
    if args.execute:
        execute(args)
    else:
        print(json.dumps(plan(args), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
