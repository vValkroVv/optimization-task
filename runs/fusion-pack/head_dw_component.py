"""Compare pitched BF16 LM-head dW with GFSDP's native copy and add."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runs/backward-components"))
from folded_ce import FLAG, CompileConfig, MTPLoss, load_fixture, sha256  # noqa: E402

VOCAB = 128256
WIDTH = 4096
TOKENS = 8192
FIXTURE_SHA256 = "15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727"


@triton.jit
def merge_head_dw(staged, output, width: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, width)
    base = row.to(tl.int64) * width
    first = tl.load(staged + 2 * base + column).to(tl.float32)
    second = tl.load(staged + 2 * base + width + column).to(tl.float32)
    # Every warp must finish reading both BF16 halves before any FP32 store
    # can overwrite another warp's input. Distinct rows occupy disjoint bytes.
    tl.debug_barrier()
    tl.store(output + base + column, first + second)


def bits_equal(left, right):
    int_dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32}
    return torch.equal(
        left.view(int_dtype[left.dtype]), right.view(int_dtype[right.dtype])
    )


def capture_operands(path):
    fixture, metadata = load_fixture(path)
    if metadata["lm_head_compiled_in_native"]:
        raise ValueError("The native fixture must use an eager head boundary")
    hidden = fixture["hidden"].detach().requires_grad_()
    with torch.device("meta"):
        head = torch.nn.Linear(WIDTH, VOCAB, bias=False)
    head.weight = torch.nn.Parameter(fixture["weight"])
    loss = MTPLoss(
        MTPLoss.Config(
            num_chunks=fixture["n_chunks"], loss_weights=fixture["head_weights"][1:]
        ),
        compile_config=CompileConfig(
            enable=True, components=["loss"], backend=metadata["compile_backend"]
        ),
    )
    loss.set_lm_head(head)
    activations, logit_grads, native_dw = [], [], []

    def receive(name, tensor, shape, target):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"Unexpected {name}: {tensor.shape}, {tensor.dtype}, {tensor.stride()}"
            )
        target.append(tensor.detach())

    def save_head_operands(_head, inputs, logits):
        receive("hidden", inputs[0], (4, 2048, WIDTH), activations)

        def save_logit_grad(grad):
            receive("logit gradient", grad, (4, 2048, VOCAB), logit_grads)

        logits.register_hook(save_logit_grad)

    def save_native_dw(weight):
        receive("weight gradient", weight.grad, (VOCAB, WIDTH), native_dw)
        weight.grad = None

    with (
        head.register_forward_hook(save_head_operands),
        head.weight.register_post_accumulate_grad_hook(save_native_dw),
    ):
        targets = loss.pack_targets(
            fixture["labels"], {"positions": fixture["positions"]}
        )
        value = loss(hidden, targets, fixture["denominator"])
        value.backward()
    if any(len(chunks) != 2 for chunks in (activations, logit_grads, native_dw)):
        raise RuntimeError("The eager head did not run both native chunk gradients")
    inputs = tuple(
        (dy.view(TOKENS, VOCAB).t(), x.view(TOKENS, WIDTH))
        for x, dy in zip(activations, logit_grads, strict=True)
    )
    return (
        inputs,
        native_dw,
        {
            "fixture_capture": metadata,
            "folded_ce_switch": os.environ[FLAG],
            "loss": float(value.detach()),
            "denominator": float(fixture["denominator"]),
            "operands": [
                [
                    {
                        "shape": list(t.shape),
                        "stride": list(t.stride()),
                        "dtype": str(t.dtype),
                    }
                    for t in chunk
                ]
                for chunk in inputs
            ],
        },
    )


class Component:
    def __init__(self, operands):
        self.operands = operands
        self.output = torch.empty((VOCAB, WIDTH), dtype=torch.float32, device="cuda")
        self.staged = self.output.view(torch.bfloat16)
        self.halves = self.staged[:, :WIDTH], self.staged[:, WIDTH:]

    def control(self):
        for i, (dy, x) in enumerate(self.operands):
            gradient = torch.mm(dy, x)
            combine = torch._foreach_copy_ if i == 0 else torch._foreach_add_
            combine([self.output], [gradient])
            del gradient

    def pitched(self):
        for (dy, x), dest in zip(self.operands, self.halves, strict=True):
            torch.mm(dy, x, out=dest)
        self.merge()

    def merge(self):
        return merge_head_dw[(VOCAB,)](self.staged, self.output, WIDTH, num_warps=4)

    def check_numerics(self, native_dw):
        expected = torch.empty_like(self.output)
        torch._foreach_copy_([expected], [native_dw[0]])
        torch._foreach_add_([expected], [native_dw[1]])
        self.control()
        control_matches = bits_equal(expected, self.output)
        chunk_matches = []
        for (dy, x), dest, native in zip(
            self.operands, self.halves, native_dw, strict=True
        ):
            torch.mm(dy, x, out=dest)
            chunk_matches.append(bits_equal(native, dest))
        kernel = self.merge()
        merged_matches = bits_equal(expected, self.output)
        ptx = kernel.asm["ptx"]
        barrier_present = "bar.sync" in ptx or "barrier.sync" in ptx
        return {
            "parity_reference": "eager bias-free Linear backward, native BF16 dW with FP32 foreach copy and add",
            "elements_per_chunk": self.output.numel(),
            "control_fp32_bitwise": control_matches,
            "each_pitched_bf16_chunk_bitwise": chunk_matches,
            "pitched_fp32_bitwise": merged_matches,
            "merge_ptx_cta_barrier": barrier_present,
            "passed": control_matches
            and all(chunk_matches)
            and merged_matches
            and barrier_present,
        }


def sample(run, start, end):
    time_begin = time.perf_counter_ns()
    start.record()
    run()
    end.record()
    enqueue_ms = (time.perf_counter_ns() - time_begin) / 1e6
    wait_begin = time.perf_counter_ns()
    end.synchronize()
    wait_ms = (time.perf_counter_ns() - wait_begin) / 1e6
    return {
        "cuda_event_ms": start.elapsed_time(end),
        "wall_ms": (time.perf_counter_ns() - time_begin) / 1e6,
        "host_enqueue_ms": enqueue_ms,
        "host_wait_ms": wait_ms,
    }


def paired_timing(control, pitched, count):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    observations = {"control": [], "control_repeat": [], "pitched": []}
    calls = {"control": control, "control_repeat": control, "pitched": pitched}
    orders = (
        ("control", "pitched", "control_repeat"),
        ("control_repeat", "pitched", "control"),
    )
    for repeat in range(count):
        for label in orders[repeat % 2]:
            observations[label].append(sample(calls[label], start, end))
    return {
        "samples": observations,
        "median_ms": {
            label: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
            for label, rows in observations.items()
        },
        "median_pitched_minus_control_midpoint_event_ms": statistics.median(
            p["cuda_event_ms"] - (a["cuda_event_ms"] + b["cuda_event_ms"]) / 2
            for p, a, b in zip(
                *[
                    observations[label]
                    for label in ("pitched", "control", "control_repeat")
                ],
                strict=True,
            )
        ),
        "median_absolute_control_pair_event_delta_ms": statistics.median(
            abs(a["cuda_event_ms"] - b["cuda_event_ms"])
            for a, b in zip(
                observations["control"], observations["control_repeat"], strict=True
            )
        ),
    }


def memory(run):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_allocated = torch.cuda.memory_allocated()
    starting_reserved = torch.cuda.memory_reserved()
    stats_before = torch.cuda.memory_stats()
    run()
    torch.cuda.synchronize()
    stats_after = torch.cuda.memory_stats()
    return {
        "starting_allocated_bytes": starting_allocated,
        "starting_reserved_bytes": starting_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocation_retries": stats_after["num_alloc_retries"]
        - stats_before["num_alloc_retries"],
        "ooms": stats_after["num_ooms"] - stats_before["num_ooms"],
    }


def graph_timing(component, count):
    control_graph, pitched_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(control_graph):
        component.control()
    with torch.cuda.graph(pitched_graph):
        component.pitched()
    for _ in range(3):
        control_graph.replay()
        pitched_graph.replay()
    torch.cuda.synchronize()
    return paired_timing(control_graph.replay, pitched_graph.replay, count)


def summarize_trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    host = [e for e in events if e.get("cat") in {"cpu_op", "user_annotation"}]
    owner_by_id = {
        e["args"]["External id"]: e for e in host if "External id" in e.get("args", {})
    }
    summary = {}
    for label in ("control", "pitched"):
        submit = next(e for e in host if e["name"] == f"head_dw_submit_{label}")
        wait = next(e for e in host if e["name"] == f"head_dw_wait_{label}")
        device = []
        for event in events:
            category = event.get("cat")
            if category not in {"kernel", "gpu_memset", "gpu_memcpy"}:
                continue
            owner = owner_by_id.get(event["args"].get("External id"))
            if (
                owner is None
                or not submit["ts"] <= owner["ts"] < submit["ts"] + submit["dur"]
            ):
                continue
            device.append(
                {
                    "category": category,
                    "owner": owner["name"],
                    "name": event["name"],
                    "stream": event["args"]["stream"],
                    "grid": event["args"].get("grid"),
                    "block": event["args"].get("block"),
                    "start_us": event["ts"],
                    "duration_us": event["dur"],
                }
            )
        device.sort(key=lambda e: e["start_us"])
        gemms = [
            e for e in device if e["category"] == "kernel" and e["owner"] == "aten::mm"
        ]
        if len(gemms) != 2 or len({e["stream"] for e in device}) != 1:
            raise ValueError(
                f"Expected two mm kernels on one stream for {label}: {device}"
            )
        span = (
            max(e["start_us"] + e["duration_us"] for e in device)
            - device[0]["start_us"]
        )
        kernel_sum = sum(e["duration_us"] for e in device if e["category"] == "kernel")
        copy_sum = sum(e["duration_us"] for e in device if e["category"] != "kernel")
        summary[label] = {
            "host_submit_us": submit["dur"],
            "host_wait_us": wait["dur"],
            "device_span_us": span,
            "device_kernel_us": kernel_sum,
            "device_gemm_us": sum(e["duration_us"] for e in gemms),
            "device_memcpy_memset_us": copy_sum,
            "device_inter_event_gap_us": span - kernel_sum - copy_sum,
            "gemm_signatures": [
                {k: e[k] for k in ("name", "grid", "block")} for e in gemms
            ],
            "device_events": device,
        }
    summary["same_gemm_signatures"] = (
        summary["control"]["gemm_signatures"] == summary["pitched"]["gemm_signatures"]
    )
    return summary


def profile(component, path):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for label in ("control", "pitched"):
            with torch.profiler.record_function(f"head_dw_submit_{label}"):
                getattr(component, label)()
            with torch.profiler.record_function(f"head_dw_wait_{label}"):
                torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    return summarize_trace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 2:
        parser.error("--warmup must be positive and --samples must be at least two")
    if torch.cuda.device_count() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        parser.error(
            "This campaign uses physical GPU 2 alone via CUDA_VISIBLE_DEVICES=2"
        )
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2048)
    os.environ[FLAG] = "1"
    fixture_sha256 = sha256(args.fixture)
    if fixture_sha256 != FIXTURE_SHA256:
        raise ValueError(f"Unexpected native fixture SHA-256: {fixture_sha256}")

    operands, native_dw, capture = capture_operands(args.fixture)
    component = Component(operands)
    numerical = component.check_numerics(native_dw)
    del native_dw
    print(f"numerics={numerical}", flush=True)
    for _ in range(args.warmup):
        component.control()
        component.pitched()
    torch.cuda.synchronize()
    memory_peaks = {
        label: memory(getattr(component, label)) for label in ("control", "pitched")
    }
    eager = paired_timing(component.control, component.pitched, args.samples)
    print(f"eager_median_ms={eager['median_ms']} memory={memory_peaks}", flush=True)
    graph_error = None
    try:
        graph = graph_timing(component, args.samples)
    except RuntimeError as exc:
        graph = None
        graph_error = str(exc)
    trace_path = args.output.with_suffix(".trace.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_error = None
    try:
        traced = profile(component, trace_path)
    except (RuntimeError, ValueError) as exc:
        traced = None
        trace_error = str(exc)

    memory_passed = all(
        memory_peaks["pitched"][key] <= memory_peaks["control"][key]
        for key in ("peak_allocated_bytes", "peak_reserved_bytes")
    ) and all(
        not peak["allocation_retries"] and not peak["ooms"]
        for peak in memory_peaks.values()
    )
    speed_passed = (
        eager["median_ms"]["pitched"]["wall_ms"]
        < min(
            eager["median_ms"][label]["wall_ms"]
            for label in ("control", "control_repeat")
        )
        and eager["median_pitched_minus_control_midpoint_event_ms"]
        < -eager["median_absolute_control_pair_event_delta_ms"]
    )
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "component_sha256": sha256(Path(__file__)),
        "fixture": str(args.fixture),
        "fixture_sha256": fixture_sha256,
        "capture": capture,
        "torch_version": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "shape": {"vocab": VOCAB, "width": WIDTH, "tokens_per_chunk": TOKENS},
        "destination": {"slot_bytes": component.output.nbytes, "bf16_pitch": 2 * WIDTH},
        "eager_scope": "Two GEMMs and native foreach copy/add or pitched FP32 merge, no dX, loss or collectives. Event span includes submission gaps. Wall span includes the host wait.",
        "graph_scope": "Same device work with allocated output and captured allocator/dispatch operations excluded from replay. Warm-cache diagnostic with shared real inputs.",
        "trace_scope": "CPU submission and wait regions, GPU GEMM, kernels, copies/memsets and same-stream inter-event gaps; no communication.",
        "warmups_per_arm": args.warmup,
        "numerical": numerical,
        "memory": memory_peaks,
        "eager": eager,
        "graph": graph,
        "graph_error": graph_error,
        "trace": traced,
        "trace_path": str(trace_path),
        "trace_error": trace_error,
        "gate": {
            "numeric": numerical["passed"],
            "memory": memory_passed,
            "elapsed": speed_passed,
            "trace": traced is not None,
            "passed": numerical["passed"]
            and memory_passed
            and speed_passed
            and traced is not None,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"gate={report['gate']} {args.output}", flush=True)
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
