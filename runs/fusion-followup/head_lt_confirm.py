"""Confirm shortlisted pitched dW cuBLASLt algorithms with balanced launch order."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

import torch

from head_lt_sweep import (
    BASELINE,
    CONFIG_FIELDS,
    FLAG,
    ROOT,
    STAGED_SOURCE,
    Component,
    Gemm,
    bits_equal,
    borrow_existing_workspace,
    capture_operands,
    fixture_weight,
    load_library,
    memory,
    production_pair,
    sha256,
    trace,
)
from head_dw_component import sample


def balanced_timing(calls, cycles):
    names = list(calls)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    observations = []
    for cycle in range(cycles):
        order = names if cycle % 2 == 0 else list(reversed(names))
        for position in range(len(order)):
            rotated = order[position:] + order[:position]
            measurements = {name: sample(calls[name], start, end) for name in rotated}
            observations.append({"order": rotated, "samples": measurements})
    timing_fields = tuple(observations[0]["samples"]["control"])
    medians = {
        name: {
            field: statistics.median(
                row["samples"][name][field] for row in observations
            )
            for field in timing_fields
        }
        for name in names
    }
    control_pair_noise = statistics.median(
        abs(
            row["samples"]["control"]["cuda_event_ms"]
            - row["samples"]["repeat"]["cuda_event_ms"]
        )
        for row in observations
    )
    paired = {}
    for name in names:
        if name in ("control", "repeat"):
            continue
        deltas = [
            row["samples"][name]["cuda_event_ms"]
            - (
                row["samples"]["control"]["cuda_event_ms"]
                + row["samples"]["repeat"]["cuda_event_ms"]
            )
            / 2
            for row in observations
        ]
        paired[name] = {
            "delta_event_ms": statistics.median(deltas),
            "rounds_faster": sum(delta < 0 for delta in deltas),
            "balanced_cycle_medians_event_ms": [
                statistics.median(deltas[i : i + len(names)])
                for i in range(0, len(deltas), len(names))
            ],
        }
    return {
        "median_ms": medians,
        "median_absolute_control_pair_event_delta_ms": control_pair_noise,
        "paired": paired,
        "observations": observations,
    }


def graph_calls(calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls.values():
            call()
    stream.synchronize()
    graphs = {}
    for name, call in calls.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        graphs[name] = graph
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    return {name: graph.replay for name, graph in graphs.items()}


def verify_shortlist(role, gemm):
    config_fields = (*CONFIG_FIELDS, "workspace_bytes")
    shortlist = [
        row
        for row in role["ranking"]
        if "paired_eager_preallocated" in row and all(row["chunk_bitwise"])
    ]
    shortlist.sort(
        key=lambda row: row["paired_eager_preallocated"][
            "median_pitched_minus_control_midpoint_event_ms"
        ]
    )
    if not shortlist:
        raise RuntimeError("The screen has no bitwise pitched dW candidate to confirm")
    for row in shortlist:
        fresh = gemm.candidates[row["index"]]
        if any(fresh[key] != row[key] for key in config_fields):
            raise RuntimeError(
                f"Rebuilt candidate {row['index']} differs from the screen"
            )
    return {
        f"tile_{row['tile_id']}": {key: row[key] for key in (*config_fields, "index")}
        for row in shortlist
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=6)
    args = parser.parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or torch.cuda.device_count() != 1
    ):
        parser.error(
            "Select physical GPU 0 with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0"
        )
    if args.cycles < 2 or args.cycles % 2:
        parser.error("Use an even number of at least two cycles")
    screen = json.loads(args.screen.read_text())
    fixture = Path(screen["fixture"])
    if sha256(fixture) != screen["fixture_sha256"]:
        parser.error("The retained fixture differs from the screen")
    if sha256(ROOT / STAGED_SOURCE) != screen["staged_source_sha256"]:
        parser.error(f"The staged head differs from the screen at {BASELINE}")
    if sha256(Path(__file__).with_name("head_lt.cpp")) != screen["cpp_sha256"]:
        parser.error("The cuBLASLt search code differs from the screen")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.manual_seed(2048)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ[FLAG] = "1"
    library, cublas = load_library(args.output)
    if (
        sha256(cublas) != screen["cublas_lt_library_sha256"]
        or torch.version.git_version != screen["torch_git"]
    ):
        raise RuntimeError("The Torch or cuBLASLt version differs from the screen")
    if str(torch.cuda.get_device_properties(0)) != screen["gpu"]:
        raise RuntimeError("The physical GPU differs from the screen")
    operands, native_dw, _ = capture_operands(fixture)
    component = Component(operands)
    workspace = borrow_existing_workspace(library)
    if workspace != screen["shared_workspace_limit_bytes"]:
        raise RuntimeError("The existing workspace differs from the screen")
    weight = fixture_weight(fixture)
    dx_inputs = tuple((dy.t(), weight) for dy, _ in operands)
    dx_output = torch.empty_like(operands[0][1])
    dx = Gemm(library, "dx", dx_inputs, (dx_output, dx_output), (), False, workspace)
    dw = Gemm(
        library, "dw_pitched", operands, component.halves, native_dw, True, workspace
    )
    shortlist = verify_shortlist(screen["roles"]["dw_pitched"], dw)
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "confirm_sha256": sha256(Path(__file__)),
        "screen": str(args.screen.resolve()),
        "screen_sha256": sha256(args.screen),
        "fixture_sha256": screen["fixture_sha256"],
        "shortlist": shortlist,
        "cycles": args.cycles,
        "gpu": screen["gpu"],
        "parity": {},
    }
    output = args.output / "result.json"
    try:
        expected = torch.empty_like(component.output)
        torch._foreach_copy_([expected], [native_dw[0]])
        torch._foreach_add_([expected], [native_dw[1]])
        for name, row in shortlist.items():
            chunk_parity = []
            for chunk in range(2):
                dw.run(row["index"], chunk)
                chunk_parity.append(dw.check(chunk))
            component.merge()
            passed = all(chunk_parity) and bits_equal(component.output, expected)
            report["parity"][name] = {"bf16_chunks": chunk_parity, "fp32_merge": passed}
            if not passed:
                raise RuntimeError(f"The rebuilt {name} lost bitwise agreement")
        dw.expected = None
        del native_dw, expected

        def baseline():
            production_pair(component, dx, dw)

        calls = {
            "control": baseline,
            **{
                name: (
                    lambda i=row["index"]: production_pair(
                        component, dx, dw, dw_index=i
                    )
                )
                for name, row in shortlist.items()
            },
            "repeat": baseline,
        }
        for call in calls.values():
            call()
        torch.cuda.synchronize()
        report["eager"] = balanced_timing(calls, args.cycles)
        report["memory"] = {
            name: memory(call) for name, call in calls.items() if name != "repeat"
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Eager replay complete: {report['eager']['paired']}", flush=True)

        graph_inputs = {
            "control": lambda: production_pair(component, dx, dw, preallocate_dx=True),
            **{
                name: (
                    lambda i=row["index"]: production_pair(
                        component, dx, dw, dw_index=i, preallocate_dx=True
                    )
                )
                for name, row in shortlist.items()
            },
            "repeat": lambda: production_pair(component, dx, dw, preallocate_dx=True),
        }
        try:
            report["graph"] = balanced_timing(graph_calls(graph_inputs), args.cycles)
        except RuntimeError as error:
            report["graph_error"] = str(error)
        print(
            f"Graph replay complete: {report.get('graph', {}).get('paired')}",
            flush=True,
        )
        trace_path = args.output / "profile.trace.json"
        trace_calls = {}
        for i, order in enumerate(
            (list(calls), list(reversed(calls)), list(calls)[2:] + list(calls)[:2])
        ):
            trace_calls.update({f"{name}_{i}": calls[name] for name in order})
        try:
            report["trace"] = trace(trace_calls, trace_path)
            report["trace_sha256"] = sha256(trace_path)
        except (RuntimeError, ValueError) as error:
            report["trace_error"] = str(error)
    finally:
        torch.cuda.synchronize()
        dx.close()
        dw.close()
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
