"""Replay native expert SwiGLU backward and its two FP8 scale consumers."""

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter_ns
from types import SimpleNamespace

import deep_gemm
import torch
import triton
from torchtitan.ops.float8 import deep_gemm_handle
from torchtitan.ops.float8._cudacherry import requantize_rowwise_to_columnwise
from torchtitan.ops.float8.triton_kernels import fused_swiglu_quantization as swiglu


FLAG = "GIFT_MOE_SWIGLU_BWD_MN_MAJOR_SCALE"
NATIVE_SHA256 = "4b3c072c701a76d65ce9360e34acae6ee4271c6a50afd39703bc774636929e0f"
NUM_SMS = 116
ARMS = ("baseline", "candidate")
ORDER = ("baseline", "candidate", "candidate", "baseline")
DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def sha256(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def select(arm):
    os.environ[FLAG] = str(int(arm == "candidate"))


def bitwise(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        )
    )


def load_case(path, stress):
    case = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    fields = ("x", "p", "dy", "weight", "weight_scale", "m_indices")
    indices = case["m_indices"]
    experts = case["weight"].shape[0]
    sizes = torch.bincount(indices[indices >= 0].long(), minlength=experts)
    used = int(sizes.sum())
    tail = indices.numel() - used
    grouped = torch.repeat_interleave(torch.arange(experts), sizes)
    if (
        sizes.numel() != experts
        or bool((sizes % 128 != 0).any())
        or tail % 128
        or not torch.equal(indices[:used], grouped)
        or not bool((indices[used:] == -1).all())
    ):
        raise ValueError(
            "The replay expects padded expert groups followed by -1 tail rows"
        )
    gpu = {name: case[name].cuda() for name in fields}
    if stress:
        # Double the expert count and the used rows by duplicating native groups.
        # Add a small unused tail when the capture contains only packed rows.
        extra_tail = tail or 256
        for name in ("x", "p", "dy"):
            tensor = gpu[name]
            padding = tensor[used:] if tail else torch.zeros_like(tensor[:extra_tail])
            gpu[name] = torch.cat((tensor[:used], tensor[:used], padding))
        indices = gpu["m_indices"]
        gpu["m_indices"] = torch.cat(
            (
                indices[:used],
                indices[:used] + experts,
                indices.new_full((extra_tail,), -1),
            )
        )
        for name in ("weight", "weight_scale"):
            gpu[name] = gpu[name].repeat(2, 1, 1)
        sizes = sizes.repeat(2)
        used *= 2
    else:
        gpu.update(
            {name: case[name].cuda() for name in ("grad_data", "grad_scale", "dgrad")}
        )
    gpu["sizes"] = sizes.to(device="cuda", dtype=torch.int32)
    gpu["limit"] = case["limit"]
    return gpu, used, sizes.tolist()


def make_replay(case, *, pack_gradient=False):
    x, p, dy = case["x"], case["p"], case["dy"]
    weight, scales = case["weight"], case["weight_scale"]
    indices, sizes = case["m_indices"], case["sizes"]
    ctx = SimpleNamespace(
        saved_tensors=(x, p),
        swiglu_limit=case["limit"],
        use_edges=True,
    )

    def producer():
        if pack_gradient:
            grad, _, dprobs, *_ = swiglu.FusedQuantizedSwiGLU.backward(ctx, dy)
            return grad._data, grad._scale, dprobs
        return swiglu._swiglu_quantized_backward(dy, x, p, case["limit"])

    def chain(return_packed=False):
        data, scale, dprobs = producer()
        dgrad = deep_gemm_handle._m_grouped_fp8_gemm_nt_contiguous(
            data, scale, weight, scales, indices
        )
        column_data, column_scales = requantize_rowwise_to_columnwise(
            data, indices, sizes, scale
        )
        consumed = dprobs, dgrad, column_data, column_scales
        return (data, scale, *consumed) if return_packed else consumed

    return producer, chain


def check_parity(chain, used, case):
    outputs = {}
    proof = {}
    rows, width = case["x"].shape
    for arm in ARMS:
        select(arm)
        data, scale, dprobs, dgrad, column_data, column_scales = chain(True)
        if arm == "baseline":
            assert scale.stride() == (width // 128, 1), scale.stride()
        else:
            assert scale.stride() == (1, rows), scale.stride()
        transformed = deep_gemm._C.get_mn_major_tma_aligned_tensor(scale)
        aliases = transformed.data_ptr() == scale.data_ptr()
        assert aliases == (arm == "candidate"), (
            arm,
            scale.stride(),
            transformed.stride(),
        )
        proof[arm] = {
            "scale_stride": list(scale.stride()),
            "deep_gemm_aliases_input": aliases,
        }
        outputs[arm] = (
            data,
            scale,
            dprobs,
            dgrad[:used],
            column_data[: used * width],
            column_scales[: used // 128],
        )
    names = ("row_fp8", "row_scale", "dprobs", "dgrad", "column_fp8", "column_scale")
    proof["ab_exact"] = dict(
        zip(names, map(bitwise, outputs["baseline"], outputs["candidate"]), strict=True)
    )
    assert all(proof["ab_exact"].values()), proof
    if "grad_data" in case:
        proof["capture_exact"] = {
            "row_fp8": bitwise(outputs["baseline"][0], case["grad_data"]),
            "row_scale": bitwise(outputs["baseline"][1], case["grad_scale"]),
            "dgrad": bitwise(outputs["baseline"][3], case["dgrad"][:used]),
        }
        assert all(proof["capture_exact"].values()), proof
    return proof


def time_calls(call, repeats):
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin = perf_counter_ns()
    start.record()
    for _ in range(repeats):
        call()
    stop.record()
    submitted = perf_counter_ns()
    stop.synchronize()
    finished = perf_counter_ns()
    return {
        "event_us": start.elapsed_time(stop) * 1e3 / repeats,
        "host_submit_us": (submitted - begin) * 1e-3 / repeats,
        "host_wait_us": (finished - submitted) * 1e-3 / repeats,
        "completion_us": (finished - begin) * 1e-3 / repeats,
    }


def time_scope(call, samples, replays):
    graphs = {}
    for arm in ARMS:
        select(arm)
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[arm] = graph
    blocks = []
    for arm in ORDER:
        select(arm)
        graph = graphs[arm]
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        compiled = [time_calls(call, 1) for _ in range(samples)]
        replay = [time_calls(graph.replay, replays) for _ in range(samples)]
        blocks.append(
            {
                "arm": arm,
                "compiled_calls": compiled,
                "compiled_medians_us": {
                    key: median(row[key] for row in compiled) for key in compiled[0]
                },
                "graph_calls": replay,
                "graph_medians_us": {
                    key: median(row[key] for row in replay) for key in replay[0]
                },
            }
        )
    return blocks


def profile_chain(chain, output):
    results = {}
    for arm in ARMS:
        select(arm)
        chain()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
        chain()
        torch.cuda.synchronize()
        memory = {
            "start_allocated_bytes": start_allocated,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_active_bytes": torch.cuda.memory_stats()["active_bytes.all.peak"],
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            chain()
            torch.cuda.synchronize()
        trace = output / f"{arm}.trace.json"
        profile.export_chrome_trace(str(trace))
        device = [
            {
                "category": event["cat"],
                "name": event["name"],
                "us": event["dur"],
                "stream": event["args"].get("stream"),
                "grid": event["args"].get("grid"),
                "registers_per_thread": event["args"].get("registers per thread"),
            }
            for event in json.loads(trace.read_text())["traceEvents"]
            if event.get("cat") in DEVICE_CATEGORIES
        ]
        results[arm] = {
            "memory": memory,
            "device_events": device,
            "trace_sha256": sha256(trace),
        }
    return results


def sac_parity():
    from torchtitan.config import ActivationCheckpointConfig
    from torchtitan.distributed.activation_checkpoint import _apply_ac_to_block
    from torchtitan.ops.float8.grouped_gemm import GroupedGemmFp8Wrapper
    from torchtitan.ops.float8.quantize_expert_weights import quantize_expert_weight
    from torchtitan.ops.float8.triton_kernels.utils import build_m_indices

    rows, dim, hid, groups = 256, 4096, 1536, [128, 0, 128]
    sizes_cpu = torch.tensor(groups, dtype=torch.int32)
    sizes = sizes_cpu.cuda()
    indices = build_m_indices(sizes, num_rows=rows)
    wrapper = GroupedGemmFp8Wrapper(num_sms=NUM_SMS)

    class Experts(torch.nn.Module):
        def forward(self, x, p, w13, w2):
            layout = {
                "tensor_groups_sizes": groups,
                "tensor_groups_sizes_cpu": sizes_cpu,
                "tensor_groups_sizes_padded": sizes,
                "m_indices": indices,
            }
            fc1, edge = wrapper.group_gemm(
                x,
                weights=quantize_expert_weight(w13),
                grad_edge=x,
                fp8_grad_output=True,
                **layout,
            )
            edge, payload = swiglu.fused_swiglu_quantized_fn(
                edge, fc1, p, swiglu_limit=4.0
            )
            output, _ = wrapper.group_gemm(
                payload, weights=quantize_expert_weight(w2), grad_edge=edge, **layout
            )
            return output

    torch.manual_seed(2048)
    x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    p = torch.rand(rows, device="cuda")
    w13 = (
        torch.randn((len(groups), 2 * hid, dim), device="cuda", dtype=torch.bfloat16)
        * 0.02
    )
    w2 = (
        torch.randn((len(groups), dim, hid), device="cuda", dtype=torch.bfloat16) * 0.02
    )
    dy = torch.randn_like(x)
    sac = _apply_ac_to_block(
        Experts(), ActivationCheckpointConfig(), mode="save:experts_up"
    )
    compiled = torch.compile(sac, fullgraph=True)
    results = {}
    for arm in ARMS:
        select(arm)
        leaves = [tensor.detach().requires_grad_() for tensor in (x, p, w13, w2)]
        out = compiled(*leaves)
        out.backward(dy)
        results[arm] = (out.detach(), *(tensor.grad for tensor in leaves))
    names = ("output", "dinput", "dprobs", "dw13", "dw2")
    exact = dict(zip(names, map(bitwise, *(results[arm] for arm in ARMS)), strict=True))
    assert all(exact.values()), exact
    return exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--ep4-stress", action="store_true")
    parser.add_argument("--sac-parity", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.replays) < 1:
        parser.error("sample and replay counts must be positive")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("The selected DeepGEMM path requires Hopper")
    args.output.mkdir(parents=True, exist_ok=False)
    fixture_sha256 = sha256(args.fixture)
    if fixture_sha256 != NATIVE_SHA256:
        raise ValueError(
            f"Fixture differs from the native data-002 capture: {fixture_sha256}"
        )
    deep_gemm.set_num_sms(NUM_SMS)
    case, used, sizes = load_case(args.fixture, args.ep4_stress)
    producer, chain = make_replay(case)
    _, eager_chain = make_replay(case, pack_gradient=True)
    compiled_producer = torch.compile(producer, fullgraph=True)
    compiled_chain = torch.compile(chain, fullgraph=True)
    result = {
        "case": "ep4-shaped duplicated EP8 groups"
        if args.ep4_stress
        else "native EP8 rank 0",
        "fixture": str(args.fixture),
        "fixture_sha256": fixture_sha256,
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": sha256(__file__),
        "producer_sha256": sha256(swiglu.__file__),
        "cudacherry_wrapper_sha256": sha256(
            requantize_rowwise_to_columnwise.__code__.co_filename
        ),
        "deep_gemm": deep_gemm.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "deep_gemm_sms": NUM_SMS,
        "x_shape": list(case["x"].shape),
        "x_dtype": str(case["x"].dtype),
        "scale_plane_bytes": case["x"].numel() // 128 * 4,
        "padded_expert_rows": sizes,
        "used_rows": used,
        "unused_rows": case["x"].shape[0] - used,
        "clamp": case["limit"],
        "samples_per_block": args.samples,
        "replays_per_sample": args.replays,
        "timer_scope": "one CUDA stream, reused inputs; compiled call versus warmed CUDA Graph replay",
    }

    def write_result():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    result["exact_parity"] = {"eager": check_parity(eager_chain, used, case)}
    write_result()
    result["exact_parity"]["compiled"] = check_parity(compiled_chain, used, case)
    write_result()
    result["compiled_profile"] = profile_chain(compiled_chain, args.output)
    write_result()
    for scope, call in (
        ("producer", compiled_producer),
        ("producer_through_consumers", compiled_chain),
    ):
        blocks = time_scope(call, args.samples, args.replays)
        result[scope] = blocks
        write_result()
        print(
            scope,
            [(row["arm"], row["graph_medians_us"]["event_us"]) for row in blocks],
            flush=True,
        )
    if args.sac_parity:
        result["compiled_sac_parity"] = sac_parity()
        write_result()


if __name__ == "__main__":
    original_flag = os.environ.get(FLAG)
    try:
        main()
    finally:
        if original_flag is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = original_flag
