"""Time expert SwiGLU backward row tiles and the next native FP8 dgrad on real data."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import time
from types import SimpleNamespace

import deep_gemm
import torch
import triton
from torchtitan.ops.float8 import deep_gemm_handle as grouped
from torchtitan.ops.float8.triton_kernels import fused_swiglu_quantization as swiglu


ARMS = {"control": (32, 4), "row16": (16, 4), "row16_w2": (16, 2), "row16_w8": (16, 8)}
DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def bitwise(left, right, rows=None):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    for i in range(0, left.size(0), 1024):
        x, y = left[i : i + 1024].contiguous(), right[i : i + 1024].contiguous()
        x, y = x.view(torch.uint8), y.view(torch.uint8)
        if rows is not None:
            selected = rows[i : i + 1024]
            x, y = x[selected], y[selected]
        if not torch.equal(x, y):
            return False
    return True


def device_time(fn, repeats):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    begin = time.perf_counter_ns()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    submitted = time.perf_counter_ns()
    end.synchronize()
    completed = time.perf_counter_ns()
    return {
        "event_ms": start.elapsed_time(end) / repeats,
        "wall_ms": (completed - begin) * 1e-6 / repeats,
        "submit_ms": (submitted - begin) * 1e-6 / repeats,
        "wait_ms": (completed - submitted) * 1e-6 / repeats,
    }


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("fixture", type=Path)
    args.add_argument("out", type=Path)
    opt = args.parse_args()
    opt.out.mkdir(parents=True, exist_ok=False)
    with opt.fixture.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    captured = torch.load(opt.fixture, map_location="cpu", weights_only=True, mmap=True)
    d = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in captured.items()}
    required = {
        "weight",
        "weight_scale",
        "m_indices",
        "dgrad",
        "x",
        "p",
        "dy",
        "limit",
        "grad_data",
        "grad_scale",
    }
    assert required <= d.keys(), required - d.keys()
    deep_gemm.set_num_sms(116)
    x, p, dy = d["x"], d["p"], d["dy"]
    width = x.shape[1] // 2
    assert width == 1536 and x.shape[0] % 64 == 0
    ctx = SimpleNamespace(
        saved_tensors=(x, p),
        input_tensor_shape=x.shape,
        swiglu_limit=d["limit"],
        use_edges=True,
    )

    def compute(with_consumer=True):
        grad, _, probs, *_ = swiglu.FusedQuantizedSwiGLU.backward(ctx, dy)
        data, scale = grad._data, grad._scale
        if not with_consumer:
            return data, scale, probs
        dx = grouped._m_grouped_fp8_gemm_nt_contiguous(
            data, scale, d["weight"], d["weight_scale"], d["m_indices"]
        )
        return data, scale, probs, dx

    results = {
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "fixture": str(opt.fixture),
        "fixture_sha256": digest,
        "shape": list(x.shape),
        "clamp": d["limit"],
        "triton": triton.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "deepgemm_sms": 116,
        "arms": {},
    }

    def select(name):
        rows, warps = ARMS[name]
        swiglu.BWD_ROW_BLOCK = rows
        swiglu.STATIC_TRITON_PARAMS_BWD[width] = {"num_warps": warps, "maxnreg": 128}

    select("control")
    reference = compute()
    valid = d["m_indices"] >= 0
    results["native_byte_parity"] = dict(
        data=bitwise(reference[0], d["grad_data"]),
        scale=bitwise(reference[1], d["grad_scale"]),
        dgrad=bitwise(reference[3], d["dgrad"], valid),
    )
    assert all(results["native_byte_parity"].values()), results["native_byte_parity"]
    for arm in ARMS:
        select(arm)
        outputs = compute()
        exact = [
            bitwise(v, r, valid if i == 3 else None)
            for i, (v, r) in enumerate(zip(outputs, reference))
        ]
        results["arms"][arm] = {
            "tile": ARMS[arm],
            "bitwise_data_scale_dprobs_dx": exact,
        }
        del outputs
    del reference
    for k in ("grad_data", "grad_scale", "dgrad"):
        del d[k]
    for arm, result in results["arms"].items():
        select(arm)
        for _ in range(5):
            compute()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        compute()
        torch.cuda.synchronize()
        result["peak_allocated"] = torch.cuda.max_memory_allocated()
        result["peak_reserved"] = torch.cuda.max_memory_reserved()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            with torch.profiler.record_function(f"expert_swiglu_to_dgrad_{arm}"):
                compute()
            torch.cuda.synchronize()
        filename = opt.out / f"{arm}.trace.json"
        prof.export_chrome_trace(str(filename))
        entries = json.loads(filename.read_text())["traceEvents"]
        result["device_events"] = [
            {
                "name": e["name"],
                "us": e["dur"],
                "registers_per_thread": e.get("args", {}).get("registers per thread"),
                "grid": e.get("args", {}).get("grid"),
            }
            for e in entries
            if e.get("cat") in DEVICE_CATEGORIES
        ]
    for scope, include_consumer in (("swiglu", False), ("through_dgrad", True)):
        graphs = {}
        for arm in ARMS:
            select(arm)
            for _ in range(3):
                compute(include_consumer)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compute(include_consumer)
            graphs[arm] = graph
        for arm in (*ARMS, *reversed(ARMS)):
            select(arm)
            graph = graphs[arm]
            graph.replay()
            torch.cuda.synchronize()
            eager = [
                device_time(lambda: compute(include_consumer), 1) for _ in range(15)
            ]
            replay = [device_time(graph.replay, 12)["event_ms"] for _ in range(15)]
            block = {k: median(v[k] for v in eager) for k in eager[0]}
            block["graph_ms"] = median(replay)
            results["arms"][arm].setdefault(scope, []).append(block)
            print(scope, arm, block, flush=True)
        del graphs, graph
        gc.collect()
        torch.cuda.empty_cache()
    (opt.out / "result.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
