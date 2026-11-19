"""Sweep workspace-free cuBLASLt variants for the staged LM-head backward."""

import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runs/fusion-pack"))
from head_dw_component import (  # noqa: E402
    Component,
    FLAG,
    bits_equal,
    capture_operands,
    memory,
    paired_timing,
    sha256,
)

CONFIG_FIELDS = (
    "algorithm_id",
    "tile_id",
    "split_k",
    "reduction_scheme",
    "cta_swizzle",
    "custom_option",
    "stages_id",
    "inner_shape_id",
    "cluster_shape_id",
    "heuristic_rank",
)
BASELINE = "ab62730a9fabd7c95b69fd07607f59dac0e7ae03"
STAGED_SOURCE = "gift/torchtitan/distributed/gfsdp/_staged_head_dw.py"


def load_library(output):
    wheel = Path(torch.__file__).resolve().parent.parent / "nvidia/cu13"
    cublas = wheel / "lib/libcublasLt.so.13"
    source = Path(__file__).with_name("head_lt.cpp")
    build = output / "head_lt.so"
    subprocess.run(
        [
            "c++",
            "-O2",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-fPIC",
            "-shared",
            f"-I{wheel / 'include'}",
            "-I/usr/local/cuda/include",
            str(source),
            str(cublas),
            f"-Wl,-rpath,{cublas.parent}",
            "-o",
            str(build),
        ],
        check=True,
    )
    library = ct.CDLL(str(build))
    signatures = {
        "version": (ct.c_size_t, []),
        "create": (ct.c_void_p, [ct.c_int64] * 6 + [ct.c_int, ct.c_uint32, ct.c_int]),
        "destroy": (None, [ct.c_void_p]),
        "counts": (None, [ct.c_void_p, ct.POINTER(ct.c_int)]),
        "info": (ct.c_float, [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_int)]),
        "run": (ct.c_int, [ct.c_void_p, ct.c_int] + [ct.c_void_p] * 4),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(library, f"head_lt_{name}")
        function.restype = result
        function.argtypes = arguments
    return library, cublas


class Gemm:
    def __init__(self, library, name, inputs, outputs, expected, trans_b):
        self.library = library
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.expected = expected
        self.trans_b = trans_b
        left, right = inputs[0]
        output = outputs[0]
        # A row-major output is C^T in cuBLAS's column-major formulation.
        self.problem = (
            output.size(1),
            output.size(0),
            left.size(1),
            right.stride(0),
            left.stride(1 if trans_b else 0),
            output.stride(0),
        )
        self.alignment = min(
            (t.data_ptr() & -t.data_ptr()) for pair in inputs for t in pair
        )
        self.alignment = min(
            self.alignment, *((t.data_ptr() & -t.data_ptr()) for t in outputs), 256
        )
        self.handle = library.head_lt_create(*self.problem, trans_b, self.alignment, 96)
        if not self.handle:
            raise RuntimeError(f"Could not initialize {name} cuBLASLt plan")
        counts = (ct.c_int * 3)()
        library.head_lt_counts(self.handle, counts)
        self.counts = dict(
            zip(
                ("exposed_ids", "validated_configs", "zero_workspace_configs"),
                counts,
                strict=True,
            )
        )
        self.candidates = []
        for i in range(counts[2]):
            config = (ct.c_int * len(CONFIG_FIELDS))()
            waves = library.head_lt_info(self.handle, i, config)
            self.candidates.append(
                {
                    "index": i,
                    **dict(zip(CONFIG_FIELDS, config, strict=True)),
                    "waves": waves,
                }
            )

    def close(self):
        if self.handle is not None:
            self.library.head_lt_destroy(self.handle)
            self.handle = None

    def describe(self):
        return {
            "column_major_m_n_k_lda_ldb_ldc": self.problem,
            "column_major_trans_a": "N",
            "column_major_trans_b": "T" if self.trans_b else "N",
            "minimum_pointer_alignment_bytes": self.alignment,
            "output_shapes": [list(t.shape) for t in self.outputs],
            "output_strides": [list(t.stride()) for t in self.outputs],
            "output_storage_offsets": [t.storage_offset() for t in self.outputs],
            **self.counts,
        }

    def baseline(self, chunk, output=None):
        return torch.mm(
            *self.inputs[chunk], out=self.outputs[chunk] if output is None else output
        )

    def launch(self, index, chunk, output=None):
        b, a = self.inputs[chunk]
        out = self.outputs[chunk] if output is None else output
        return self.library.head_lt_run(
            self.handle,
            index,
            a.data_ptr(),
            b.data_ptr(),
            out.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )

    def run(self, index, chunk, output=None):
        status = self.launch(index, chunk, output)
        if status:
            raise RuntimeError(
                f"{self.name} candidate {index} returned cuBLAS status {status}"
            )

    def check(self, chunk):
        return bits_equal(self.expected[chunk], self.outputs[chunk])


def sample_gpu(call, start, end):
    start.record()
    status = call()
    end.record()
    end.synchronize()
    return status, start.elapsed_time(end)


def sweep(gemm, repetitions):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for chunk in range(2):
        gemm.baseline(chunk)
        if not gemm.check(chunk):
            raise RuntimeError(f"{gemm.name} native control failed at chunk {chunk}")
    controls, rows = [], []
    for candidate in gemm.candidates:
        i = candidate["index"]
        if i % 8 == 0:
            controls.append(
                [
                    sample_gpu(lambda: gemm.baseline(chunk), start, end)[1]
                    for chunk in range(2)
                ]
            )
        compare, times, status = [], [], 0
        for chunk in range(2):
            status, _ = sample_gpu(lambda: gemm.launch(i, chunk), start, end)
            if status:
                break
            elapsed = []
            for _ in range(repetitions):
                status, timing = sample_gpu(lambda: gemm.launch(i, chunk), start, end)
                if status:
                    break
                elapsed.append(timing)
            if status:
                break
            times.append(elapsed)
            compare.append(gemm.check(chunk))
        rows.append(
            {
                **candidate,
                "status": status,
                "chunk_ms": times,
                "chunk_bitwise": compare,
                "total_median_ms": sum(statistics.median(row) for row in times)
                if not status
                else None,
            }
        )
    rows.sort(
        key=lambda row: (row["status"] != 0, row["total_median_ms"] or float("inf"))
    )
    print(
        f"{gemm.name}: {gemm.describe()}, {sum(not row['status'] for row in rows)} launches supported, "
        f"{sum(all(row['chunk_bitwise']) and not row['status'] for row in rows)} bitwise",
        flush=True,
    )
    return {
        "problem": gemm.describe(),
        "interleaved_control_chunk_ms": controls,
        "ranking": rows,
    }


def trace(calls, path):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for name, call in calls.items():
            with torch.profiler.record_function(f"head_lt_submit_{name}"):
                call()
            with torch.profiler.record_function(f"head_lt_wait_{name}"):
                torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    launches = {
        event["args"]["correlation"]: event
        for event in events
        if event.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in event.get("args", {})
    }
    result = {}
    for name in calls:
        submit = next(
            event for event in events if event.get("name") == f"head_lt_submit_{name}"
        )
        wait = next(
            event for event in events if event.get("name") == f"head_lt_wait_{name}"
        )
        device = []
        for event in events:
            if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            owner = launches.get(event["args"].get("correlation"))
            if owner is None or (owner["pid"], owner["tid"]) != (
                submit["pid"],
                submit["tid"],
            ):
                continue
            if submit["ts"] <= owner["ts"] < submit["ts"] + submit["dur"]:
                device.append(
                    {
                        "name": event["name"],
                        "kind": event["cat"],
                        "ts_us": event["ts"],
                        "dur_us": event["dur"],
                        "stream": event["args"].get("stream"),
                        "grid": event["args"].get("grid"),
                        "block": event["args"].get("block"),
                    }
                )
        if not device or len({event["stream"] for event in device}) != 1:
            raise RuntimeError(
                f"Expected a single-stream replay trace for {name}, got {device}"
            )
        device.sort(key=lambda event: event["ts_us"])
        span = (
            max(event["ts_us"] + event["dur_us"] for event in device)
            - device[0]["ts_us"]
        )
        kernels = sum(event["dur_us"] for event in device if event["kind"] == "kernel")
        copies = sum(event["dur_us"] for event in device if event["kind"] != "kernel")
        result[name] = {
            "host_submit_us": submit["dur"],
            "host_wait_us": wait["dur"],
            "device_span_us": span,
            "kernel_us": kernels,
            "copy_memset_us": copies,
            "device_inter_event_gap_us": span - kernels - copies,
            "events": device,
        }
    return result


def production_pair(
    component, dx, dw, dx_index=None, dw_index=None, preallocate_dx=False
):
    for chunk in range(2):
        if dx_index is None:
            gradient = (
                dx.baseline(chunk) if preallocate_dx else torch.mm(*dx.inputs[chunk])
            )
        else:
            gradient = (
                dx.outputs[chunk]
                if preallocate_dx
                else torch.empty_like(dx.outputs[chunk])
            )
            dx.run(dx_index, chunk, gradient)
        if dw_index is None:
            dw.baseline(chunk)
        else:
            dw.run(dw_index, chunk)
        del gradient
    component.merge()


def rank_final(gemm, report, count):
    rankings = report["ranking"]
    valid = [row for row in rankings if not row["status"]]
    parity = [row for row in valid if all(row["chunk_bitwise"])]
    selected = {row["index"]: row for row in valid[:3] + parity[:3]}
    for i, row in selected.items():

        def control():
            for chunk in range(2):
                gemm.baseline(chunk)

        def candidate():
            for chunk in range(2):
                gemm.run(i, chunk)

        row["paired_eager_preallocated"] = paired_timing(control, candidate, count)
    return min(
        parity[:3],
        key=lambda row: row["paired_eager_preallocated"][
            "median_pitched_minus_control_midpoint_event_ms"
        ],
        default=None,
    )


def fixture_weight(path):
    raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return raw["ce"]["weight"].cuda()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--screen-repetitions", type=int, default=2)
    parser.add_argument("--paired-repetitions", type=int, default=12)
    args = parser.parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or torch.cuda.device_count() != 1
    ):
        parser.error(
            "Select authorized physical GPU 0 with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0"
        )
    if args.screen_repetitions < 1 or args.paired_repetitions < 2:
        parser.error("At least one screen and two paired repetitions are required")
    if sha256(args.fixture) != args.fixture_sha256:
        parser.error("Fixture SHA-256 differs from the supplied receipt")
    original = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{STAGED_SOURCE}"], cwd=ROOT
    )
    if original != (ROOT / STAGED_SOURCE).read_bytes():
        parser.error("Staged head source differs from the pinned production baseline")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.manual_seed(2048)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ[FLAG] = "1"
    library, cublas = load_library(args.output)
    operands, native_dw, capture = capture_operands(args.fixture)
    weight = fixture_weight(args.fixture)
    component = Component(operands)
    numerical = component.check_numerics(native_dw)
    if not numerical["passed"]:
        raise RuntimeError(f"Staged production reference mismatch: {numerical}")

    dx_inputs = tuple((dy.t(), weight) for dy, _ in operands)
    native_dx = tuple(torch.mm(*inputs) for inputs in dx_inputs)
    dx_outputs = tuple(torch.empty_like(out) for out in native_dx)
    # The contiguous dW pitch is diagnostic. The pitched run writes both
    # disjoint production offsets directly into the shared FP32 output slot.
    dx = Gemm(library, "dx", dx_inputs, dx_outputs, native_dx, False)
    dw = Gemm(library, "dw_pitched", operands, component.halves, native_dw, True)
    contiguous = Gemm(
        library,
        "dw_contiguous",
        operands,
        tuple(torch.empty_like(out) for out in native_dw),
        native_dw,
        True,
    )
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "staged_source_revision": BASELINE,
        "staged_source_sha256": sha256(ROOT / STAGED_SOURCE),
        "script_sha256": sha256(Path(__file__)),
        "cpp_sha256": sha256(Path(__file__).with_name("head_lt.cpp")),
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": args.fixture_sha256,
        "capture": capture,
        "staged_reference_checks": numerical,
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "toolkit": torch.version.cuda,
        "cublas_lt_version": library.head_lt_version(),
        "cublas_lt_library": str(cublas),
        "cublas_lt_library_sha256": sha256(cublas),
        "gpu": str(torch.cuda.get_device_properties(0)),
        "blas_preferred_backend": str(torch.backends.cuda.preferred_blas_library()),
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "workspace_environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUBLASLT_WORKSPACE_SIZE",
                "TORCH_CUBLASLT_UNIFIED_WORKSPACE",
                "DISABLE_ADDMM_CUDA_LT",
            )
        },
        "candidate_workspace_bytes": 0,
        "screen_repetitions": args.screen_repetitions,
        "paired_repetitions": args.paired_repetitions,
        "roles": {},
    }
    report_path = args.output / "result.json"
    try:
        for gemm in (dx, dw, contiguous):
            role = sweep(gemm, args.screen_repetitions)
            winner = rank_final(gemm, role, args.paired_repetitions)
            role["fastest_bitwise_candidate_index"] = (
                winner["index"] if winner else None
            )
            report["roles"][gemm.name] = role
            report_path.write_text(json.dumps(report, indent=2) + "\n")

        del contiguous.outputs, native_dx, native_dw
        eligible = {}
        calls = {"baseline": lambda: production_pair(component, dx, dw)}
        for gemm in (dx, dw):
            role = report["roles"][gemm.name]
            best = role["fastest_bitwise_candidate_index"]
            if best is not None:
                row = next(row for row in role["ranking"] if row["index"] == best)
                timing = row["paired_eager_preallocated"]
                if (
                    timing["median_pitched_minus_control_midpoint_event_ms"]
                    < -timing["median_absolute_control_pair_event_delta_ms"]
                ):
                    eligible[gemm.name] = best
            fastest = next((row for row in role["ranking"] if not row["status"]), None)
            for label, index in (
                ("fastest", fastest["index"] if fastest else None),
                ("bitwise", best),
            ):
                if index is not None:
                    calls[f"{gemm.name}_{label}"] = lambda g=gemm, i=index: [
                        g.run(i, chunk) for chunk in range(2)
                    ]
            calls[f"{gemm.name}_control"] = lambda g=gemm: [
                g.baseline(chunk) for chunk in range(2)
            ]
        report["eligible_bitwise_candidates"] = eligible
        if eligible:
            dx_index, dw_index = eligible.get("dx"), eligible.get("dw_pitched")

            def selected():
                production_pair(component, dx, dw, dx_index, dw_index)

            calls["candidate"] = selected
            baseline = calls["baseline"]
            baseline()
            expected_merge = torch.empty_like(component.output)
            expected_merge.copy_(component.output)
            selected()
            report["selected_fp32_dW_merge_bitwise"] = bits_equal(
                expected_merge, component.output
            )
            del expected_merge
            report["matched_eager"] = paired_timing(
                baseline, selected, args.paired_repetitions
            )
            report["matched_memory"] = {
                label: memory(calls[label]) for label in ("baseline", "candidate")
            }
            try:
                graphs = (torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph())
                for graph, (selected_dx, selected_dw) in zip(
                    graphs, ((None, None), (dx_index, dw_index)), strict=True
                ):
                    with torch.cuda.graph(graph):
                        production_pair(
                            component,
                            dx,
                            dw,
                            selected_dx,
                            selected_dw,
                            preallocate_dx=True,
                        )
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                report["matched_graph_replay"] = paired_timing(
                    graphs[0].replay, graphs[1].replay, args.paired_repetitions
                )
            except RuntimeError as error:
                report["matched_graph_error"] = str(error)
        else:
            report["matched_memory"] = {"baseline": memory(calls["baseline"])}

        trace_path = args.output / "profile.trace.json"
        try:
            report["trace"] = trace(calls, trace_path)
            report["trace_sha256"] = sha256(trace_path)
        except (RuntimeError, ValueError) as error:
            report["trace_error"] = str(error)
    finally:
        torch.cuda.synchronize()
        for gemm in (dx, dw, contiguous):
            gemm.close()
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
