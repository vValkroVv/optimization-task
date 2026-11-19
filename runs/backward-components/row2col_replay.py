"""Replay native EP8 expert-wgrad operands through both row2col implementations."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import file_digest
import importlib.metadata
import itertools
import json
from math import prod
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns
from typing import Callable

import torch


BLOCK = 128
NUM_SMS = 116
SHAPES = {"w13": (16, 3072, 4096), "w2": (16, 4096, 1536)}
PROJECTIONS = ("w2", "w13")  # Expert backward order.
ROLES = ("w13_x", "w2_dy", "w13_dy", "w2_x")
SLOT_SIZE = sum(map(prod, SHAPES.values()))
ROOT = Path(__file__).resolve().parents[2]
BUILD_TOOLS = ROOT / "runs/efsdp-optimizations/direct-dw"
NATIVE_SOURCE_PIN = "585bd50a641857ccae8b5ec691bb15791c9529e7"
PINNED_WRAPPER_SHA256 = (
    "3839e99e3f3de2af14fbc0562d5d155f4d93fcc3b116a2a6077c488ef9cf72ae"
)
WRITER_KERNEL = "sm90_k_grouped_fp8_gemm_rounded_f32_impl"
CONVERTER_KERNELS = {
    "cudacherry": "requantize_grouped_kernel",
    "triton": "_rowwise_to_columnwise_deepgemm_layout_kernel",
}
CUDA_EVENT_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}

Operand = tuple[torch.Tensor, torch.Tensor]
Converter = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Operand]


@dataclass
class Fixture:
    metadata: dict
    padded_sizes: torch.Tensor
    m_indices: torch.Tensor
    roles: dict[str, Operand]
    offsets: tuple[int, int]
    raw_sizes: list[int] | None = None

    @property
    def used_rows(self):
        return sum(self.padded_sizes.tolist())

    def describe(self):
        return {
            "metadata": self.metadata,
            "padded_sizes": self.padded_sizes.tolist(),
            "raw_sizes": self.raw_sizes,
            "capacity_rows": self.m_indices.numel(),
            "consumed_rows": self.used_rows,
            "unused_capacity_rows": self.m_indices.numel() - self.used_rows,
            "slot_offsets": dict(zip(SHAPES, self.offsets, strict=True)),
            "roles": {
                name: {
                    "data": layout(data),
                    "scale": layout(scale),
                    "scale_copy_required": scale.stride(0) != 1,
                    "native_data_copy_required": not data.is_contiguous(),
                }
                for name, (data, scale) in self.roles.items()
            },
        }


def sha256(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def layout(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "address_mod_256": tensor.data_ptr() % 256,
    }


def tensor_bytes(tensor):
    return tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor


def restore(tensor, offset, device):
    """Copy represented values to a view with the captured strides and offset."""
    strides = tensor.stride()
    span = 0
    if tensor.numel():
        span = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, strides, strict=True)
        )
    storage = torch.empty(offset + span, dtype=tensor.dtype, device=device)
    result = storage.as_strided(tensor.shape, strides, offset)
    tensor_bytes(result).copy_(tensor_bytes(tensor))
    return result


def expected_indices(padded_sizes, capacity):
    used_rows = sum(padded_sizes)
    if any(size < 0 or size % BLOCK for size in padded_sizes):
        raise ValueError("Padded expert sizes must be nonnegative and 128-aligned")
    if capacity < used_rows or capacity % BLOCK:
        raise ValueError("Capacity must cover the expert segments and be 128-aligned")
    return torch.tensor(
        [expert for expert, size in enumerate(padded_sizes) for _ in range(size)]
        + [-1] * (capacity - used_rows),
        dtype=torch.int32,
    )


def require_cpu_tensor(tensor, shape, dtypes, name):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.shape != shape
        or tensor.dtype not in dtypes
    ):
        raise ValueError(f"Invalid captured {name}: expected CPU {shape} in {dtypes}")


def load_fixture(path, device):
    payload = torch.load(path, weights_only=True, map_location="cpu")
    metadata = payload["metadata"]
    if (
        payload.get("schema") != 1
        or metadata.get("diagnostic_only") is not True
        or metadata.get("world_size") != 8
        or metadata.get("gradient_accumulation") != 32
        or metadata.get("num_sms") != NUM_SMS
        or metadata.get("torch_version") != str(torch.__version__)
        or metadata.get("cuda_version") != torch.version.cuda
    ):
        raise ValueError(
            "Fixture must come from the matching native EP8 capture environment"
        )
    roles = {}
    offsets = []
    padded_sizes = m_indices = None
    raw_sizes = None
    for projection, shape in SHAPES.items():
        row = payload["row2col"][projection]
        if tuple(row["weight_shape"]) != shape or row["slot_numel"] != SLOT_SIZE:
            raise ValueError(
                f"Wrong captured {projection} slot shape, expected {shape}"
            )
        offsets.append(row["direct_dw_offset"])
        sizes = row["aligned_sizes"]
        require_cpu_tensor(sizes, (shape[0],), (torch.int32,), "aligned_sizes")
        indices = row["m_indices"]
        capacity = indices.numel()
        require_cpu_tensor(indices, (capacity,), (torch.int32,), "m_indices")
        if not sizes.is_contiguous() or not indices.is_contiguous():
            raise ValueError("Replay requires contiguous native int32 routing")
        if padded_sizes is None:
            padded_sizes, m_indices = sizes, indices
            if not torch.equal(indices, expected_indices(sizes.tolist(), capacity)):
                raise ValueError(
                    "m_indices does not describe the expert segments and capacity tail"
                )
            if not sum(sizes.tolist()):
                raise ValueError("Native timing needs a fixture with some routed rows")
        elif not torch.equal(sizes, padded_sizes) or not torch.equal(
            indices, m_indices
        ):
            raise ValueError("The w13 and w2 captures must share the same routing")
        if row.get("raw_sizes") is not None:
            raw = row["raw_sizes"]
            raw = raw.tolist() if isinstance(raw, torch.Tensor) else list(raw)
            if (
                len(raw) != shape[0]
                or any(not isinstance(count, int) or count < 0 for count in raw)
                or [(count + BLOCK - 1) // BLOCK * BLOCK for count in raw]
                != sizes.tolist()
                or (raw_sizes is not None and raw_sizes != raw)
            ):
                raise ValueError("Captured raw expert sizes disagree with padded sizes")
            raw_sizes = raw
        for role, width in (("x", shape[2]), ("dy", shape[1])):
            data_key, scale_key = f"{role}_data", f"{role}_scales"
            data, scales = row[data_key], row[scale_key]
            require_cpu_tensor(
                data, (capacity, width), (torch.uint8, torch.float8_e4m3fn), data_key
            )
            require_cpu_tensor(
                scales, (capacity, width // BLOCK), (torch.float32,), scale_key
            )
            restored = []
            for key, tensor in ((data_key, data), (scale_key, scales)):
                offset = row.get("storage_offsets", {}).get(
                    key, tensor.storage_offset()
                )
                actual = restore(tensor, offset, device)
                address = row.get("address_mod_256", {}).get(key)
                if address is not None and actual.data_ptr() % 256 != address:
                    raise ValueError(
                        f"Address alignment changed for {projection} {key}"
                    )
                restored.append(actual)
            roles[f"{projection}_{role}"] = tuple(restored)
    if tuple(offsets) not in ((0, prod(SHAPES["w13"])), (prod(SHAPES["w2"]), 0)):
        raise ValueError(
            "Captured offsets do not address the shared FP32 direct dW slot"
        )
    return Fixture(metadata, padded_sizes, m_indices, roles, tuple(offsets), raw_sizes)


def mixed_fixture(native):
    """Take finite represented values from native inputs and give them edge routing."""
    raw = [0, 127, 0, 141, 0, 385, *([0] * 10)]
    padded = [(size + BLOCK - 1) // BLOCK * BLOCK for size in raw]
    used_rows = sum(padded)
    capacity = used_rows + 2 * BLOCK
    roles = {}
    for name, operand in native.roles.items():
        restored = []
        for tensor in operand:
            strides = tensor.stride()
            # A larger diagnostic row count must not overlap adjacent scale columns.
            if strides[0] == 1:
                strides = (1, max(capacity, strides[1]))
            restored.append(
                torch.empty_strided(
                    (capacity, tensor.shape[1]),
                    strides,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            )
        data, scale = restored
        # Positive one is finite in E4M3FN. Use distinct finite tail values.
        bits = tensor_bytes(data)
        bits.fill_(0x38)
        scale.fill_(0.25)
        bits[:used_rows].zero_()
        # Select only native routed rows, including when a native case is small.
        source_rows = native.used_rows
        position = 0
        source_bits, source_scale = tensor_bytes(operand[0]), operand[1]
        for count, aligned in zip(raw, padded, strict=True):
            if count:
                rows = torch.arange(count, device=data.device) % source_rows
                bits[position : position + count].copy_(source_bits[rows])
                scale[position : position + count].copy_(source_scale[rows])
            position += aligned
        roles[name] = data, scale
    return Fixture(
        {"kind": "synthetic_routing_real_values"},
        torch.tensor(padded, dtype=torch.int32),
        expected_indices(padded, capacity),
        roles,
        native.offsets,
        raw,
    )


def configure_kernels(build_root, device):
    sys.path.insert(0, str(BUILD_TOOLS))
    from direct_dw import load_direct_dw
    import deep_gemm
    import triton
    from torchtitan.ops.float8._cudacherry import requantize_rowwise_to_columnwise
    import cudacherry.quantization.row2col_dq as native_module
    from torchtitan.ops.float8.direct_expert_dw import (
        acquire_direct_expert_dw,
        configure_direct_expert_dw,
        write_direct_expert_dw,
    )
    from torchtitan.ops.float8.triton_kernels import row2col_deepgemm_quantization

    extension = load_direct_dw(build_root, num_sms=NUM_SMS)
    configure_direct_expert_dw(extension.k_grouped_fp8_gemm_nt_contiguous_rounded_f32)
    deep_gemm.set_num_sms(NUM_SMS)
    native_binary = Path(native_module._ext().__file__).resolve()
    triton_module = Path(row2col_deepgemm_quantization.__file__).resolve()
    native_wrapper = Path(native_module.__file__).resolve()
    wrapper_digest = sha256(native_wrapper)
    properties = torch.cuda.get_device_properties(device)
    identity = {
        "device": properties.name,
        "device_total_sms": properties.multi_processor_count,
        "device_memory_bytes": properties.total_memory,
        "direct_dw_sm_budget": NUM_SMS,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "allocator_config": os.environ.get(
            "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        ),
        "deep_gemm": importlib.metadata.version("deep-gemm"),
        "direct_dw_build": str(build_root.resolve()),
        "direct_dw_manifest_sha256": sha256(build_root / "manifest.json"),
        "cudacherry": importlib.metadata.version("cudacherry"),
        "cudacherry_image_commit": os.environ.get("CUDACHERRY_COMMIT"),
        "cudacherry_source_pin": NATIVE_SOURCE_PIN,
        "cudacherry_wrapper": {
            "path": str(native_wrapper),
            "sha256": wrapper_digest,
            "matches_pinned_source": wrapper_digest == PINNED_WRAPPER_SHA256,
        },
        "cudacherry_binary": {
            "path": str(native_binary),
            "sha256": sha256(native_binary),
        },
        "triton_wrapper": {"path": str(triton_module), "sha256": sha256(triton_module)},
        "replay_sha256": sha256(__file__),
    }
    converters = {
        "cudacherry": requantize_rowwise_to_columnwise,
        "triton": row2col_deepgemm_quantization.row2col_requantization_deepgemm_layout_fn,
    }
    return converters, acquire_direct_expert_dw, write_direct_expert_dw, identity


class Replay:
    def __init__(self, fixture, converter, writer, slot):
        self.fixture = fixture
        self.converter = converter
        self.writer = writer
        self.slot = slot
        self.indices = fixture.m_indices.to(slot.device)
        self.sizes = fixture.padded_sizes.to(slot.device)
        self.offsets = dict(zip(SHAPES, fixture.offsets, strict=True))
        self.rows = fixture.used_rows

    def role(self, name):
        data, scales = self.fixture.roles[name]
        data, scales = self.converter(data, self.indices, self.sizes, scales)
        return data.view(torch.float8_e4m3fn), scales.mT

    def write(self, projection, x, dy):
        self.writer(
            *dy,
            *x,
            self.slot,
            self.offsets[projection],
            None,
            self.fixture.padded_sizes,
            self.sizes,
        )

    def output(self, projection):
        shape = SHAPES[projection]
        return self.slot.narrow(0, self.offsets[projection], prod(shape)).view(shape)

    def projection(self, name, profile=False):
        def region(label):
            return torch.profiler.record_function(label) if profile else nullcontext()

        with region(f"row2col/{name}_x"):
            x = self.role(f"{name}_x")
        with region(f"row2col/{name}_dy"):
            dy = self.role(f"{name}_dy")
        with region(f"row2col/{name}_writer"):
            self.write(name, x, dy)


def check_output_layout(operand, width, capacity):
    data, scales = operand
    if (
        data.shape != (width * capacity,)
        or data.dtype != torch.float8_e4m3fn
        or not data.is_contiguous()
        or scales.dtype != torch.float32
        or scales.shape != (width, capacity // BLOCK)
        or scales.stride() != (1, width)
        or data.data_ptr() % 16
        or scales.data_ptr() % 16
    ):
        raise ValueError("A row2col arm produced an invalid DeepGEMM layout")


def count_differences(actual, expected):
    """Compare raw representations, including negative zeros and NaN payloads."""
    if torch.equal(actual, expected):
        return 0
    return int(torch.count_nonzero(actual != expected).item())


def compare_role(actual, expected, fixture, width):
    actual_bits, expected_bits = (
        tensor_bytes(operand[0]) for operand in (actual, expected)
    )
    actual_scales, expected_scales = (
        operand[1].view(torch.int32) for operand in (actual, expected)
    )
    cursor = 0
    by_expert = []
    for expert, padded in enumerate(fixture.padded_sizes.tolist()):
        if not padded:
            by_expert.append({"expert": expert, "padded_rows": 0})
            continue
        first, last = cursor * width, (cursor + padded) * width
        group_a = actual_bits[first:last].view(width, padded)
        group_b = expected_bits[first:last].view(width, padded)
        prefix = slice(cursor // BLOCK, (cursor + padded) // BLOCK)
        raw = None if fixture.raw_sizes is None else fixture.raw_sizes[expert]
        by_expert.append(
            {
                "expert": expert,
                "padded_rows": padded,
                "fp8_mismatches": count_differences(group_a, group_b),
                "scale_mismatches": count_differences(
                    actual_scales[:, prefix], expected_scales[:, prefix]
                ),
                "fp8_padding_mismatches": None
                if raw is None
                else count_differences(group_a[:, raw:], group_b[:, raw:]),
                "scales_covering_padding_mismatches": None
                if raw is None
                else count_differences(
                    actual_scales[:, (cursor + raw) // BLOCK : prefix.stop],
                    expected_scales[:, (cursor + raw) // BLOCK : prefix.stop],
                ),
            }
        )
        cursor += padded
    return {
        "compared_fp8_bytes": cursor * width,
        "compared_fp32_scales": cursor // BLOCK * width,
        "excluded_capacity_rows": fixture.m_indices.numel() - cursor,
        "fp8_mismatches": sum(row.get("fp8_mismatches", 0) for row in by_expert),
        "scale_mismatches": sum(row.get("scale_mismatches", 0) for row in by_expert),
        "by_expert": by_expert,
    }


def check_parity(fixture, converters, acquire, writer, slot):
    # Use the same physical slot for both arms. A reference snapshot also serves
    # the identical-input repeat before trying the candidate.
    replays = {
        name: Replay(fixture, converter, writer, slot)
        for name, converter in converters.items()
    }

    def convert(replay):
        converted = {name: replay.role(name) for name in ROLES}
        for name, operand in converted.items():
            check_output_layout(
                operand, fixture.roles[name][0].shape[1], fixture.m_indices.numel()
            )
        return converted

    a, b = (convert(replay) for replay in replays.values())
    roles = {
        name: compare_role(b[name], a[name], fixture, fixture.roles[name][0].shape[1])
        for name in ROLES
    }

    def write_pair(arm, pair, sentinel):
        slot.view(torch.int32).fill_(sentinel)
        lease = acquire(slot, fixture.offsets)
        try:
            for name in PROJECTIONS:
                replays[arm].write(name, pair[f"{name}_x"], pair[f"{name}_dy"])
            lease.finish()
        finally:
            lease.abort()

    def slot_bits():
        return {
            name: replays["cudacherry"].output(name).view(torch.int32)
            for name in PROJECTIONS
        }

    def compare_slot(reference):
        return {
            name: [
                count_differences(expert_a, expert_b)
                for expert_a, expert_b in zip(bits, reference[name], strict=True)
            ]
            for name, bits in slot_bits().items()
        }

    write_pair("cudacherry", a, 0x7FC11111)
    reference = {name: view.clone() for name, view in slot_bits().items()}
    write_pair("cudacherry", a, 0x7FC22222)
    identical_input = compare_slot(reference)
    write_pair("triton", b, 0x7FC33333)
    candidate = compare_slot(reference)
    empty_expert_nonzero = {
        name: {
            expert: int(torch.count_nonzero(bits[expert]).item())
            for expert, size in enumerate(fixture.padded_sizes.tolist())
            if size == 0
        }
        for name, bits in slot_bits().items()
    }
    return {
        "roles": roles,
        "identical_input_dw_mismatches_by_expert": identical_input,
        "candidate_dw_mismatches_by_expert": candidate,
        "empty_expert_nonzero_bits": empty_expert_nonzero,
        "pass": (
            all(
                row["fp8_mismatches"] == row["scale_mismatches"] == 0
                for row in roles.values()
            )
            and not any(map(any, identical_input.values()))
            and not any(map(any, candidate.values()))
            and not any(any(row.values()) for row in empty_expert_nonzero.values())
        ),
    }


def measure_scope(operations, prepare, device, warmup, samples):
    orders = tuple(
        itertools.permutations(("cudacherry", "cudacherry_control", "triton"))
    )
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    trials = []
    for trial in range(-warmup, samples):
        order = orders[trial % len(orders)]
        measurement = {}
        for arm in order:
            lease = prepare()
            try:
                start.record()
                cpu_start = perf_counter_ns()
                operations[arm]()
                cpu_submit = perf_counter_ns()
                stop.record()
                stop.synchronize()
                cpu_complete = perf_counter_ns()
                measurement[arm] = {
                    "cuda_ms": start.elapsed_time(stop),
                    "host_submit_ms": (cpu_submit - cpu_start) / 1e6,
                    "host_to_cuda_completion_ms": (cpu_complete - cpu_start) / 1e6,
                }
            finally:
                if lease is not None:
                    lease.abort()
        if trial >= 0:
            trials.append({"order": order, "milliseconds": measurement})

    memory = {}
    for arm, operation in operations.items():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        lease = prepare()
        try:
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            operation()
            torch.cuda.synchronize(device)
            memory[arm] = {
                "initial_allocated_bytes": allocated,
                "initial_reserved_bytes": reserved,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        finally:
            if lease is not None:
                lease.abort()

    medians = {
        arm: {
            key: median(trial["milliseconds"][arm][key] for trial in trials)
            for key in ("cuda_ms", "host_submit_ms", "host_to_cuda_completion_ms")
        }
        for arm in operations
    }
    pairs = [trial["milliseconds"] for trial in trials]
    paired_control_noise_ms = median(
        abs(row["cudacherry"]["cuda_ms"] - row["cudacherry_control"]["cuda_ms"])
        for row in pairs
    )
    paired_delta_ms = median(
        row["triton"]["cuda_ms"]
        - (row["cudacherry"]["cuda_ms"] + row["cudacherry_control"]["cuda_ms"]) / 2
        for row in pairs
    )
    return {
        "samples": trials,
        "medians_ms": medians,
        "paired_candidate_delta_cuda_ms": paired_delta_ms,
        "median_absolute_identical_control_delta_cuda_ms": paired_control_noise_ms,
        "memory": memory,
        "candidate_faster_than_control_noise": paired_delta_ms
        < -paired_control_noise_ms,
        "candidate_no_higher_peak_memory": all(
            memory["triton"][key] <= memory["cudacherry"][key]
            for key in ("peak_allocated_bytes", "peak_reserved_bytes")
        ),
    }


def read_trace(path, arm, fixture):
    events = json.loads(path.read_text())["traceEvents"]
    regions = {
        event["name"].removeprefix("row2col/"): event
        for event in events
        if event.get("cat") == "user_annotation"
        and event["name"].startswith("row2col/")
    }
    launches = {}
    for event in events:
        if event.get("cat") not in {"cuda_runtime", "cuda_driver"}:
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            launches.setdefault(correlation, []).append(event)

    def inside(event, region):
        return (event["pid"], event["tid"]) == (
            region["pid"],
            region["tid"],
        ) and region["ts"] <= event["ts"] < region["ts"] + region["dur"]

    copies = {
        name: [
            event
            for event in events
            if event.get("cat") == "cpu_op"
            and event["name"] == "aten::copy_"
            and inside(event, region)
        ]
        for name, region in regions.items()
    }
    scale_copies = {}
    for name, (_, scale) in fixture.roles.items():
        copied_shape = list(reversed(scale.shape))
        scale_copies[name] = [
            event
            for event in copies.get(name, ())
            if event.get("args", {}).get("Input Dims", [])[:2] == [copied_shape] * 2
        ]
    rows = {name: {"gpu": []} for name in (*ROLES, "w13_writer", "w2_writer")}
    for event in events:
        if event.get("cat") not in CUDA_EVENT_CATEGORIES:
            continue
        args = event.get("args", {})
        event_launches = launches.get(args.get("correlation"), [])
        matches = {
            name
            for launch in event_launches
            for name, region in regions.items()
            if inside(launch, region)
        }
        if len(matches) == 1:
            name = matches.pop()
            rows[name]["gpu"].append(
                {
                    "name": event["name"],
                    "category": event["cat"],
                    "duration_us": event["dur"],
                    "grid": args.get("grid"),
                    "stream": args.get("stream"),
                    "input_scale_copy": any(
                        inside(launch, copy)
                        for launch in event_launches
                        for copy in scale_copies.get(name, ())
                    ),
                }
            )
    activation_errors = []
    for name, row in rows.items():
        marker = WRITER_KERNEL if name.endswith("_writer") else CONVERTER_KERNELS[arm]
        matching = [event for event in row["gpu"] if marker in event["name"]]
        if len(matching) != 1:
            activation_errors.append(
                f"{name}: expected one {marker}, found {len(matching)}"
            )
        elif name.endswith("_writer") and matching[0]["grid"] != [NUM_SMS, 1, 1]:
            activation_errors.append(
                f"{name}: expected a 116-SM launch grid, saw {matching[0]['grid']}"
            )
        row["aten_copy_calls"] = len(copies.get(name, ()))
        if name in fixture.roles:
            scale = fixture.roles[name][1]
            copy_expected = scale.stride(0) != 1
            copy_count = len(scale_copies[name])
            row["scale_copy_calls"] = copy_count
            row["scale_copy_gpu_duration_us"] = sum(
                event["duration_us"]
                for event in row["gpu"]
                if event["input_scale_copy"]
            )
            if copy_count != int(copy_expected) or (
                copy_expected and not row["scale_copy_gpu_duration_us"]
            ):
                activation_errors.append(
                    f"{name}: expected {int(copy_expected)} scale copies with GPU attribution, "
                    f"found {copy_count} with {row['scale_copy_gpu_duration_us']} us"
                )
        row["gpu_duration_sum_us"] = sum(event["duration_us"] for event in row["gpu"])
    return {
        "path": str(path),
        "sha256": sha256(path),
        "scopes": rows,
        "activation_errors": activation_errors,
    }


def measure_fixture(
    fixture, converters, acquire, writer, slot, output, warmup, samples
):
    device = slot.device
    replays = {
        name: Replay(fixture, converter, writer, slot)
        for name, converter in converters.items()
    }
    scopes = {}
    for name in (*ROLES, *PROJECTIONS):
        if name in ROLES:

            def prepare():
                return None

            operations = {
                arm: (lambda replay=replay: replay.role(name))
                for arm, replay in replays.items()
            }
        else:

            def prepare():
                return acquire(slot, fixture.offsets)

            operations = {
                arm: (lambda replay=replay: replay.projection(name))
                for arm, replay in replays.items()
            }
        operations["cudacherry_control"] = operations["cudacherry"]
        scopes[name] = measure_scope(operations, prepare, device, warmup, samples)
        print(f"{name}: {scopes[name]['medians_ms']}", flush=True)

    traces = {}
    for arm, replay in replays.items():
        torch.cuda.synchronize(device)
        lease = acquire(slot, fixture.offsets)
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profile:
                for name in PROJECTIONS:
                    replay.projection(name, profile=True)
                lease.finish()
                torch.cuda.synchronize(device)
        finally:
            lease.abort()
        trace = output / f"{arm}_trace.json"
        profile.export_chrome_trace(str(trace))
        traces[arm] = read_trace(trace, arm, fixture)
    return scopes, traces


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=6, help="Warmup trials per scope")
    parser.add_argument(
        "--samples",
        type=int,
        default=18,
        help="Measured trials per scope, in a multiple of six matched orders",
    )
    args = parser.parse_args()
    if args.warmup < 6 or args.samples < 6 or args.samples % 6:
        parser.error(
            "Warmup and samples must each be at least 6. Samples must be a multiple of 6."
        )
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if (
        "H100" not in properties.name
        or properties.major != 9
        or properties.multi_processor_count < NUM_SMS
    ):
        raise RuntimeError(
            f"Replay needs an H100 with at least 116 SMs, found {properties}"
        )
    args.output.mkdir(parents=True)
    converters, acquire, writer, identity = configure_kernels(args.build_root, device)
    slot = torch.empty(SLOT_SIZE, dtype=torch.float32, device=device)
    result = {
        "runtime": identity,
        "scope": (
            "Single-H100 warmed eager CUDA-event spans and host timings; "
            "each invocation includes converter allocations and kernel launches. "
            "Input tensors and the shared direct-dW slot stay resident. "
            "Allocator peaks follow an untimed cache clear."
        ),
        "warmup_trials": args.warmup,
        "samples_per_scope": args.samples,
        "native": [],
    }
    for number, path in enumerate(args.fixture):
        fixture = load_fixture(path, device)
        directory = args.output / f"native-{number}"
        directory.mkdir()
        parity = check_parity(fixture, converters, acquire, writer, slot)
        print(
            f"{path}: native bitwise parity {'passed' if parity['pass'] else 'failed'}",
            flush=True,
        )
        if number == 0:
            edge = mixed_fixture(fixture)
            edge_parity = check_parity(edge, converters, acquire, writer, slot)
            result["mixed_edge"] = {"fixture": edge.describe(), "parity": edge_parity}
            print(
                f"Mixed routing edge parity {'passed' if edge_parity['pass'] else 'failed'}",
                flush=True,
            )
            del edge
        scopes, traces = measure_fixture(
            fixture,
            converters,
            acquire,
            writer,
            slot,
            directory,
            args.warmup,
            args.samples,
        )
        result["native"].append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "fixture": fixture.describe(),
                "parity": parity,
                "scopes": scopes,
                "traces": traces,
            }
        )
        del fixture
    result["passes_component_gate"] = result["mixed_edge"]["parity"]["pass"] and all(
        row["parity"]["pass"]
        and not any(trace["activation_errors"] for trace in row["traces"].values())
        and all(
            row["scopes"][name]["candidate_faster_than_control_noise"]
            and row["scopes"][name]["candidate_no_higher_peak_memory"]
            for name in PROJECTIONS
        )
        for row in result["native"]
    )
    (args.output / "row2col.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passes_component_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
