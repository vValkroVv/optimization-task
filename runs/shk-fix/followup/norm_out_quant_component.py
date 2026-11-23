"""Replay the resident ordinary CuTe norm epilogue and native FP8 row quant."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import torch

from norm_f1_component import load_variant, measure_pairs
from norm_pair_tile_reuse_component import prepare
from norm_skip import NATIVE_SHAPE, digest, mismatched_bytes, restore

OUTPUTS = ("bf16_output", "rho", "p", "fp8_data", "fp32_scale")
REFERENCE_REVISION = "d3e0febbf5bf4a08c7f2d9499e96eaddced5e510"
VENDOR_SOURCES = {
    "quantization/blockwise.py": "7fbe0417ed3a775be0e11171e09de4b733a0b0f5acffd5b42fcc7617f4247094",
    "csrc/quantization/blockwise/row_quantization_kernel.cu": (
        "e62efe4691fc3bd3fdea79443ca1150a0928c0994d6a4dfda1cb9abae26ecbbd"
    ),
    "csrc/common/warp_reduce.cuh": "8624651ab1d8a16fd84c7ad8aa4830726536a1d3a56cf07687190cc7a775c1fd",
    "csrc/common/fp8_math.cuh": "f37372e154312bfe89f4a18b6299188d8080db5878f8afa52127d94bd541c5d8",
    "csrc/common/scale.cuh": "5178434b5892b726d7c20f708795835d3713cfdda0757e5e1cce07d2a35440d0",
    "_build_spec.py": "20e143204d2a27bd0a380536b74a60926a3d74cae7cf7117e65d73f8972a6d6f",
}


def compile_ptx(candidate, eps, gating_weight, output):
    """Compile for Hopper with symbolic tensor descriptors and no kernel launch."""
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_stream

    def compile_kernel(_key, make_kernel, args):
        return cute.compile(
            make_kernel(),
            *args,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --gpu-arch sm_90a --keep-ptx --keep-cubin",
        )

    compiled = candidate.compile_norm_out_quant(
        eps, gating_weight, compile_kernel=compile_kernel
    )
    ptx, cubin = compiled.__ptx__, compiled.__cubin__
    if not ptx or not isinstance(cubin, bytes) or not cubin:
        raise RuntimeError("Expected SM90a PTX and CUBIN")
    for instruction in ("shfl.sync.bfly.b32", "cvt.rn.satfinite.e4m3x2.f32"):
        if instruction not in ptx:
            raise RuntimeError(f"Missing quantization instruction {instruction}")
    with output.with_suffix(".ptx").open("x") as handle:
        handle.write(ptx)
    with output.with_suffix(".cubin").open("xb") as handle:
        handle.write(cubin)


def make_calls(source, p_part, ss_part, weight, down, f2, fused, quantize):
    rows, hidden = source.shape
    if rows % 4 or hidden != 4096:
        raise ValueError("Expected unpadded physical FP8 data and scale planes")

    def buffers():
        outputs = (
            torch.empty_like(source),
            torch.empty(rows, device=source.device, dtype=torch.float32),
            torch.empty((rows, 16), device=source.device, dtype=torch.float32),
        )
        return outputs, (source, weight, down, p_part, ss_part, *outputs)

    baseline_outputs, baseline_args = buffers()
    fused_outputs, fused_args = buffers()
    words = torch.empty((rows, hidden // 4), device=source.device, dtype=torch.int32)
    scales = torch.empty((hidden // 128, rows), device=source.device, dtype=torch.float32)
    data = words.view(torch.float8_e4m3fn)

    def baseline(_):
        f2(*baseline_args, source)
        return (*baseline_outputs, *quantize(baseline_outputs[0], True))

    def candidate(_):
        fused(*fused_args, words, scales)
        return (*fused_outputs, data, scales)

    for index, output in enumerate((*baseline_outputs, *fused_outputs, words, scales)):
        output.view(torch.uint8).fill_(index + 1)
    return {"A": baseline, "B": candidate}


def verify(runs):
    reference = runs["A"](None)
    result = mismatched_bytes(OUTPUTS, reference, runs["B"](None))
    if any(result.values()):
        raise RuntimeError(f"Changed norm/quant bytes {result}")
    return result


def capture(runs):
    """Keep each graph's output allocation resident for device-only comparison."""
    graphs, retained_outputs = {}, {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for run in runs.values():
            run(None)
    stream.synchronize()
    for arm, run in runs.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            retained_outputs[arm] = run(None)
        graphs[arm] = graph
    torch.cuda.synchronize()

    def bind(arm):
        def replay(_):
            graphs[arm].replay()
            return retained_outputs[arm]

        return replay

    return {arm: bind(arm) for arm in runs}


def profile(runs, calls):
    results = {}
    for arm, run in runs.items():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as trace:
            for _ in range(calls):
                run(None)
                torch.cuda.synchronize()
        kernels = sorted(
            (event for event in trace.events() if event.device_type.name == "CUDA"),
            key=lambda event: event.time_range.start,
        )
        expected = ("GnFwdOut_", "quantize_vec128_kernel") if arm == "A" else ("GnFwdOut_",)
        if len(kernels) != len(expected) * calls or any(
            name not in event.name for name, event in zip(expected * calls, kernels, strict=True)
        ):
            raise RuntimeError(f"Unexpected kernels in {arm}: {[e.name for e in kernels]}")
        results[arm] = [
            {"name": e.name, "duration_us": e.time_range.elapsed_us()} for e in kernels
        ]
    return results


@torch.no_grad()
def replay(args, forward, candidate):
    from torchtitan.ops.float8 import _cudacherry
    from cudacherry._loader import load_family

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    vendor_package = Path(_cudacherry.cudacherry.__file__).resolve().parent
    vendor_hashes = {name: digest(vendor_package / name) for name in VENDOR_SOURCES}
    if vendor_hashes != VENDOR_SOURCES:
        raise RuntimeError(f"CudaCherry row quantizer differs from the pinned sources {vendor_hashes}")
    vendor = load_family("quantize")
    quantize = torch.ops.cudacherry.quantize_vec128.default
    fixture = torch.load(args.fixture, map_location="cpu", mmap=True, weights_only=True)
    if fixture["torch_revision"] != torch.version.git_version:
        raise RuntimeError("Fixture and replay use different PyTorch revisions")
    record = fixture["records"]["plain"]
    x = restore(record["tensors"]["x"], record["layouts"]["x"])
    if tuple(x.shape) != NATIVE_SHAPE or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise RuntimeError("Expected captured contiguous BF16 input")
    hidden = x.shape[-1]
    x = x.view(-1, hidden)
    rank = record["tensors"]["up"].shape[0]
    sms = forward.resolve_num_sms(device)
    chunks = forward.pick_n_hc(x.shape[0], hidden, sms)
    if (hidden, rank, sms, chunks) != (4096, 16, 116, 4):
        raise RuntimeError("Fixture does not select the native ordinary norm tile")

    eps, gating_weight = record["eps"], record["gating_weight"]
    prep, f1, f2, splits = forward._fwd_plan(
        hidden,
        rank,
        tuple(record["tensors"][key].dtype for key in ("up", "weight", "down")),
        eps,
        gating_weight,
        device,
        chunks,
        sms,
    )
    if splits != 8:
        raise RuntimeError("Expected the native eight-way F1 split")
    w_eff, weight, down = prepare(prep, record)
    fused = candidate.compile_norm_out_quant(eps, gating_weight)

    def inputs(source, norm_weight):
        rows = source.shape[0]
        p_part, ss_part = forward.workspace(device, [(splits, rows, rank), (splits, rows)])
        f1(source, w_eff, p_part, ss_part)
        return make_calls(source, p_part, ss_part, norm_weight, down, f2, fused, quantize)

    edge = x[:68].clone()
    edge[0].zero_()
    edge[1].fill_(-0.0)
    edge[2].fill_(float("nan"))
    edge[3, 64:128].fill_(float("inf"))
    edge[4, :128].fill_(2.0**-130)
    edge_weight = weight.clone()
    edge_weight[128] = float("inf")
    edge_weight[129] = float("-inf")
    parity = {"68_rows_special_values": verify(inputs(edge, edge_weight))}
    native = inputs(x, weight)
    parity["8192_rows_captured"] = verify(native)
    torch.cuda.synchronize()

    graphs = capture(native)
    parity["8192_rows_graph_replay"] = verify(graphs)
    receipt = {
        "scope": "resident_norm_f2_and_row_quant_under_cuda_graph_replay",
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "cudacherry_reference_revision": REFERENCE_REVISION,
        "cudacherry_package": str(vendor_package),
        "cudacherry_sources_sha256": vendor_hashes,
        "cudacherry_extension_sha256": digest(Path(vendor.__file__)),
        "fixture_sha256": digest(args.fixture),
        "runner_sha256": digest(Path(__file__)),
        "baseline_sha256": digest(Path(forward.__file__)),
        "candidate_sha256": digest(args.candidate),
        "shape": list(NATIVE_SHAPE),
        "eps": eps,
        "gating_weight": gating_weight,
        "outputs": {
            name: {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)}
            for name, t in zip(OUTPUTS, graphs["A"](None), strict=True)
        },
        "mismatched_bytes": parity,
        "pairs": measure_pairs(graphs, args),
        "profiles": profile(graphs, args.profile_calls),
    }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--profile-calls", type=int, default=8)
    args = parser.parse_args()
    if not args.compile_only and args.fixture is None:
        parser.error("--fixture is required for resident replay")
    if min(args.warmup, args.samples, args.pairs, args.profile_calls) < 1:
        parser.error("Replay counts must be positive")

    from torchtitan.models.common.norms.cute_gated_norm import forward

    candidate = load_variant(forward, args.candidate, "forward_norm_out_quant")
    if args.compile_only:
        compile_ptx(candidate, 1e-6, 2.0, args.output)
    else:
        replay(args, forward, candidate)


if __name__ == "__main__":
    main()
