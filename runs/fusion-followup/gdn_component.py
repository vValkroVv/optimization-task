"""Replay native short-convolution and QKV projection backward on one GPU."""

import argparse
from collections import Counter
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint


LABELS = ("input_gradient", "projection_weight_gradient", "conv_weight_gradient")


def digest(path):
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def fixture_on_device(path):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if digest(path) != receipt["fixture_sha256"]:
        raise ValueError("The fixture hash does not match the native capture receipt")
    fixture = torch.load(path, weights_only=True, map_location="cpu")
    if tuple(fixture["qkv"].shape) != tuple(receipt["qkv_shape"]):
        raise ValueError("QKV shape differs from the native capture receipt")
    return {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in fixture.items()
    }, receipt


def synthetic_boundary(tokens, *, packed=False):
    # Boundary diagnostics only. Native captured tensors supply all timings.
    torch.manual_seed(23)

    def tensor(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.15

    return {
        "x": tensor(1, tokens, 256),
        "proj_weight": tensor(1024, 256),
        "conv_weight": tensor(1024, 1, 4),
        "grad_q": tensor(1, tokens, 2, 128),
        "grad_k": tensor(1, tokens, 2, 128),
        "grad_v": tensor(1, tokens, 4, 128),
        "cu_seqlens": torch.tensor(
            [0, 1, 3, 68, tokens], device="cuda", dtype=torch.int32
        )
        if packed
        else None,
        "num_heads": 2,
        "num_v_heads": 4,
        "normalize_qk": True,
        "eps": 1e-6,
    }


def bytes_equal(left, right):
    if left is None or right is None:
        return left is right
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def comparisons(names, baseline, candidate):
    result = {}
    for name, left, right in zip(names, baseline, candidate, strict=True):
        equal = bytes_equal(left, right)
        result[name] = {"bitwise": equal}
        if not equal and left is not None and right is not None:
            left_bits = left.contiguous().view(torch.uint8)
            right_bits = right.contiguous().view(torch.uint8)
            result[name]["different_bytes"] = int(
                torch.count_nonzero(left_bits != right_bits)
            )
    return result


def assert_bits(result):
    if not all(check["bitwise"] for check in result.values()):
        raise ValueError(f"Byte comparison failed: {result}")


def gradient_inputs(fixture):
    return tuple(fixture[name] for name in ("grad_q", "grad_k", "grad_v"))


def make_backward(fixture, quantized, *, needs_wgrad=True):
    from torchtitan.models.common.attention.linear.gdn_short_conv import (
        gdn_short_conv_backward,
        gdn_short_conv_backward_fp8,
    )
    from torchtitan.ops.float8._cudacherry import quantize_vec128
    from torchtitan.ops.float8.cuda_gemm import FP8Gemm, prepare_for_saving
    from torchtitan.ops.float8.fp8_tensor import (
        Float8BlockwiseQTensor,
        InputLayoutType,
    )

    x, weight, qkv = (fixture[name] for name in ("x", "proj_weight", "qkv"))
    saved = (quantize_vec128(x), weight) if needs_wgrad else (weight,)
    tensors, metadata = prepare_for_saving(*saved)
    conv_args = (
        qkv,
        fixture["conv_weight"],
        *gradient_inputs(fixture),
        fixture["cu_seqlens"],
        fixture["num_heads"],
        fixture["num_v_heads"],
        fixture["normalize_qk"],
        fixture["eps"],
    )

    def backward():
        if quantized:
            data, scales, grad_conv_weight = gdn_short_conv_backward_fp8(*conv_args)
            gradient = Float8BlockwiseQTensor(
                data,
                scales,
                qkv.dtype,
                InputLayoutType.ROWWISE,
                1,
                scale_block_first=True,
            )
        else:
            gradient, grad_conv_weight = gdn_short_conv_backward(*conv_args)
        saved_context = SimpleNamespace(
            needs_wgrad=needs_wgrad,
            tensor_objects=metadata,
            saved_tensors=tensors,
            quantize_weight_tail=False,
            act_requires_grad=True,
            dtype=qkv.dtype,
            inp_cur_shape=x.shape,
            w_shape=weight.shape,
            has_residual=False,
        )
        dx, dw, *_ = FP8Gemm.backward(saved_context, gradient)
        return dx, dw, grad_conv_weight

    return backward, conv_args


def make_module(fixture, *, quantized, amp, compile_sac, freeze=False):
    from torchtitan.config.schema.models.common.nn_modules import Conv1dConfig
    from torchtitan.distributed.activation_checkpoint import _sac_context_fn
    from torchtitan.models.common.attention.qkv import LinearAttnQKV
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    x, weight = fixture["x"], fixture["proj_weight"]
    out_features, in_features = weight.shape
    opts = LinearAttnQKV.Config(
        head_k_dim=128,
        head_v_dim=128,
        num_k_heads=fixture["num_heads"],
        num_v_heads=fixture["num_v_heads"],
        normalize_qk=fixture["normalize_qk"],
        eps=fixture["eps"],
        qkv_proj=FP8Linear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        ),
        conv=Conv1dConfig(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=4,
            groups=out_features,
            padding=3,
            bias=False,
        ),
    )
    model = opts.build().to(device=x.device, dtype=x.dtype)
    with torch.no_grad():
        model.qkv_proj.weight.copy_(weight)
        model.qkv_conv1d.weight.copy_(fixture["conv_weight"])
    model.qkv_proj.weight.requires_grad_(not freeze)
    x = x.detach().requires_grad_()
    incoming = gradient_inputs(fixture)
    cu = fixture["cu_seqlens"]
    qkv_forward = partial(
        model,
        fuse_conv_backward=True,
        quantize_conv_backward=quantized,
    )

    def forward(value):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            if compile_sac:
                return checkpoint(
                    qkv_forward,
                    value,
                    cu,
                    use_reentrant=False,
                    context_fn=partial(_sac_context_fn, (), ()),
                    early_stop=False,
                )
            return qkv_forward(value, cu)

    call = torch.compile(forward, fullgraph=True) if compile_sac else forward

    def forward_backward():
        outputs = call(x)
        if freeze:
            dx, dconv = torch.autograd.grad(
                outputs,
                (x, model.qkv_conv1d.weight),
                incoming,
            )
            return (*outputs, dx, None, dconv)
        return (
            *outputs,
            *torch.autograd.grad(
                outputs,
                (x, model.qkv_proj.weight, model.qkv_conv1d.weight),
                incoming,
            ),
        )

    return forward_backward, model


def trace(call, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    events = sorted(
        (
            event
            for event in json.loads(path.read_text())["traceEvents"]
            if event.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")
        ),
        key=lambda event: event["ts"],
    )
    if len({event["args"]["stream"] for event in events}) != 1:
        raise ValueError("Expected all component events on one CUDA stream")
    kernels = [event for event in events if event["cat"] == "kernel"]
    return {
        "path": str(path),
        "sha256": digest(path),
        "counts": dict(Counter(event["cat"] for event in events)),
        "device_span_us": events[-1]["ts"] + events[-1]["dur"] - events[0]["ts"],
        "device_gap_us": sum(
            max(0.0, right["ts"] - left["ts"] - left["dur"])
            for left, right in zip(events, events[1:])
        ),
        "kernel_time_us": sum(event["dur"] for event in kernels),
        "copy_and_memset_time_us": sum(
            event["dur"] for event in events if event["cat"] != "kernel"
        ),
        "kernels": [
            [event["name"], event["args"]["grid"], event["args"]["block"], event["dur"]]
            for event in kernels
        ],
    }


def pair_timing(calls, samples, *, graph_replays=20):
    order = ("baseline", "candidate", "control")
    calls = {**calls, "control": calls["baseline"]}
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    graphs = {}
    worker = torch.cuda.Stream()
    worker.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(worker):
        for name in order:
            for _ in range(5):
                calls[name]()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=worker):
                calls[name]()
            graph.replay()
            graphs[name] = graph
    worker.synchronize()

    series = (
        "graph_us",
        "event_elapsed_us",
        "host_call_us",
        "event_record_wait_us",
        "wall_us",
    )
    data = {name: {key: [] for key in series} for name in order}
    for i in range(samples):
        for name in order if i % 2 == 0 else reversed(order):
            begin.record()
            for _ in range(graph_replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            data[name]["graph_us"].append(
                begin.elapsed_time(end) * 1000 / graph_replays
            )

            begin.record()
            t0 = time.perf_counter()
            calls[name]()
            t1 = time.perf_counter()
            end.record()
            end.synchronize()
            t2 = time.perf_counter()
            data[name]["event_elapsed_us"].append(begin.elapsed_time(end) * 1000)
            data[name]["host_call_us"].append((t1 - t0) * 1e6)
            data[name]["event_record_wait_us"].append((t2 - t1) * 1e6)
            data[name]["wall_us"].append((t2 - t0) * 1e6)
    return {
        name: {
            key: {"samples": ints, "median": statistics.median(ints)}
            for key, ints in result.items()
        }
        for name, result in data.items()
    }


def memory(call):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_allocated = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()
    out = call()
    torch.cuda.synchronize()
    result = {
        "start_allocated_bytes": start_allocated,
        "start_reserved_bytes": start_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del out
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output / "inductor")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose only the assigned GPU")
    from torchtitan.models.common.attention.linear.gdn_short_conv import (
        _gdn_short_conv_backward_kernel,
        gdn_short_conv_backward,
        gdn_short_conv_backward_fp8,
    )
    from torchtitan.models.common.attention.linear.gdn_short_conv_fp8 import (
        gdn_short_conv_backward_fp8_kernel,
    )
    from torchtitan.ops.float8._cudacherry import quantize_vec128

    if args.smoke:
        fixture = synthetic_boundary(128, packed=True)
        receipt = {"seed": 23, "scope": "synthetic packed-boundary diagnostics"}
    else:
        fixture, receipt = fixture_on_device(args.fixture)
    result = {
        "fixture": str(args.fixture) if args.fixture else None,
        "receipt": receipt,
        "device": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    summary = output / "result.json"

    def save():
        summary.write_text(json.dumps(result, indent=2) + "\n")

    base_module, base_model = make_module(
        fixture,
        quantized=False,
        compile_sac=False,
        amp=False,
    )
    with torch.no_grad():
        qkv = base_model.qkv_proj(fixture["x"])
    if "qkv" in fixture:
        result["capture_projection_bitwise"] = bytes_equal(qkv, fixture["qkv"])
        save()
        if not result["capture_projection_bitwise"]:
            raise ValueError("The reconstructed projection differs from captured QKV")
    else:
        fixture["qkv"] = qkv
    del qkv
    base_call, conv_args = make_backward(fixture, False)
    fused_call, _ = make_backward(fixture, True)
    dx, dw = gdn_short_conv_backward(*conv_args)
    native_codes, native_scales, _, _ = quantize_vec128(dx, pack=False)
    fused_codes, fused_scales, fused_dw = gdn_short_conv_backward_fp8(*conv_args)
    result["epilogue_bits"] = comparisons(
        ("codes", "scales", "conv_weight_gradient"),
        (native_codes, native_scales, dw),
        (fused_codes, fused_scales, fused_dw),
    )
    result["kernels"] = [
        {
            "name": kernel.name,
            "registers": kernel.n_regs,
            "spills": kernel.n_spills,
            "shared_bytes": kernel.metadata.shared,
            "warps": kernel.metadata.num_warps,
        }
        for fn in (_gdn_short_conv_backward_kernel, gdn_short_conv_backward_fp8_kernel)
        for device in fn.device_caches.values()
        for kernel in device[0].values()
    ]
    save()
    assert_bits(result["epilogue_bits"])
    del dx, dw, native_codes, native_scales, fused_codes, fused_scales, fused_dw
    result["backward_bits"] = comparisons(LABELS, base_call(), fused_call())
    save()
    assert_bits(result["backward_bits"])

    result["module_bits"] = {}
    if not args.smoke:
        calls = {"baseline": base_call, "candidate": fused_call}
        result["backward_graph_and_synchronized_calls"] = pair_timing(
            calls, args.samples
        )
        result["backward_memory"] = {name: memory(call) for name, call in calls.items()}
        result["backward_trace"] = {
            name: trace(call, output / f"backward-{name}.json")
            for name, call in calls.items()
        }
        save()

    for mode, compile_sac, amp, freeze in (
        ("eager", False, False, False),
        ("compiled_sac", True, False, False),
        ("amp_frozen_projection", False, True, True),
    ):
        stock = base_module
        if compile_sac or amp:
            stock, _ = make_module(
                fixture,
                quantized=False,
                compile_sac=compile_sac,
                amp=amp,
                freeze=freeze,
            )
        fused, _ = make_module(
            fixture,
            quantized=True,
            compile_sac=compile_sac,
            amp=amp,
            freeze=freeze,
        )
        bits = comparisons(("q", "k", "v", *LABELS), stock(), fused())
        result["module_bits"][mode] = bits
        save()
        assert_bits(bits)
        if not args.smoke and mode == "compiled_sac":
            result["compiled_sac_memory"] = {
                name: memory(call)
                for name, call in (("baseline", stock), ("candidate", fused))
            }
            result["compiled_sac_trace"] = {
                name: trace(call, output / f"compiled-{name}.json")
                for name, call in (("baseline", stock), ("candidate", fused))
            }
            save()

    if args.smoke:
        fixture = synthetic_boundary(129)
        stock, _ = make_module(fixture, quantized=False, compile_sac=False, amp=False)
        fused, _ = make_module(fixture, quantized=True, compile_sac=False, amp=False)
        bits = comparisons(("q", "k", "v", *LABELS), stock(), fused())
        result["module_bits"]["misaligned_fallback"] = bits
        save()
        assert_bits(bits)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fixture", type=Path)
    inputs.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    run(parser.parse_args())
