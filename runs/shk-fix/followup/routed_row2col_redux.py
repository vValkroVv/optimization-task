"""Compare stock, private control and SM90 warp-max CudaCherry row2col."""

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess
from unittest.mock import patch

from routed_dw import probe
from routed_row2col_component import CHECKOUT, ROLE_PAIRS, SOURCES, verify_outputs
from routed_swiglu_fwd_order_component import summarize


DONOR_REVISION = "d3e0febbf5bf4a08c7f2d9499e96eaddced5e510"
INSTALLED_EXTENSION_SHA = (
    "1515ae9f465a12f422865ce880bc2e713e97af7f477dd8f2488d0f0394a0d435"
)
SOURCE_DIR = "csrc/quantization/row2col_dq"
KERNEL = "requantize_grouped_kernel"
PRIVATE = {"A": "shk_row2col_control", "B": "shk_row2col_redux"}
MAX_TREE = """    // Warp-level reduction: 5 butterfly steps cover all 32 lanes.
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        #pragma unroll
        for (int c = 0; c < 16; ++c)
            amax[c] = fmaxf(amax[c],
                            __shfl_xor_sync(0xFFFFFFFF, amax[c], off));
    }
"""
HARDWARE_MAX = """    // Local fmaxf starts at +0 and ignores NaNs from fabsf.
    // Unsigned FP32 bits preserve the order of these nonnegative maxima.
    #pragma unroll
    for (int c = 0; c < 16; ++c)
        amax[c] = __uint_as_float(__reduce_max_sync(0xFFFFFFFFu, __float_as_uint(amax[c])));
"""


def transform_kernel(source, name, redux):
    if source.count(MAX_TREE) != 1 or source.count(KERNEL) != 2:
        raise ValueError("The donor row2col warp reduction or launcher has changed")
    source = source.replace(KERNEL, f"{KERNEL}_{name}")
    source = re.sub(r"\bvoid row2col_dq\(", f"void {name}(", source)
    return source.replace(MAX_TREE, HARDWARE_MAX) if redux else source


def transform_binding(source, name):
    # The op's basename/schema stay identical within its private namespace.
    source = source.replace("(cudacherry,", f"({name},")
    return source.replace("void row2col_dq(", f"void {name}(").replace(
        "&row2col_dq", f"&{name}"
    )


def bind_python(source, path, name, native):
    call = "torch.ops.cudacherry.row2col_dq("
    if source.count(call) != 1:
        raise ValueError("The installed row2col Python dispatcher has changed")
    namespace = vars(native).copy()
    exec(
        compile(
            source.replace(call, f"torch.ops.{name}.row2col_dq("), str(path), "exec"
        ),
        namespace,
    )
    return namespace["requantize_rowwise_to_columnwise"]


def disassembly(tool, extension, output):
    if tool is None:
        return {"status": "cuobjdump unavailable"}
    with output.open("x") as handle:
        subprocess.run([tool, "--dump-sass", str(extension)], stdout=handle, check=True)
    sass = output.read_text()
    redux = re.findall(r"\bREDUX(?:\.[A-Z0-9_]+)*\b", sass)
    return {
        "status": "checked",
        "sha256": probe.sha256(output),
        "redux": len(redux),
        "redux_opcodes": sorted(set(redux)),
        "shuffles": len(re.findall(r"\bSHFL(?:\.[A-Z0-9_]+)*\b", sass)),
    }


def build(args, native, torch):
    from cudacherry import _build_spec
    from torch.utils.cpp_extension import CUDA_HOME, load

    revision = subprocess.check_output(
        ["git", "-C", str(args.donor), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != DONOR_REVISION:
        raise RuntimeError(f"Unexpected CudaCherry donor revision {revision}")
    package = Path(native.__file__).resolve().parents[1]
    donor = args.donor / "src/cudacherry"
    for root in (donor, package):
        if {rel: probe.sha256(root / rel) for rel in SOURCES} != SOURCES:
            raise RuntimeError(f"CudaCherry source identities differ at {root}")
    header = "csrc/common/scale.cuh"
    if probe.sha256(package / header) != probe.sha256(donor / header):
        raise RuntimeError("Installed and donor row2col scale helpers differ")
    installed = Path(native._ext().__file__).resolve()
    if probe.sha256(installed) != INSTALLED_EXTENSION_SHA:
        raise RuntimeError("Unexpected installed CudaCherry row2col extension")

    family = next(spec for spec in _build_spec.FAMILIES if spec.key == "requantize")
    flags = family.nvcc(_build_spec.build_arch_flags("9.0"))
    if flags != [
        "-O3",
        "-gencode=arch=compute_90,code=sm_90",
        "--use_fast_math",
        "-lineinfo",
        "--ptxas-options=-v",
        "-std=c++20",
    ]:
        raise RuntimeError(f"Unexpected CudaCherry build flags {flags}")
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None and CUDA_HOME:
        local = Path(CUDA_HOME) / "bin/cuobjdump"
        if local.is_file():
            cuobjdump = str(local)

    original_cu = (donor / SOURCE_DIR / "row2col_kernel.cu").read_text()
    original_binding = (donor / SOURCE_DIR / "bindings.cpp").read_text()
    python = Path(native.__file__).read_text()
    wrappers = {"installed": native.requantize_rowwise_to_columnwise}
    ops = {"installed": torch.ops.cudacherry.row2col_dq.default}
    evidence = {}
    for arm, name in PRIVATE.items():
        root = args.output / name
        root.mkdir()
        cu, cpp = root / "row2col_kernel.cu", root / "bindings.cpp"
        cu.write_text(transform_kernel(original_cu, name, arm == "B"))
        cpp.write_text(transform_binding(original_binding, name))
        module = load(
            name=name,
            sources=[str(cpp), str(cu)],
            extra_cflags=[*family.cxx, "-std=c++20"],
            extra_cuda_cflags=flags,
            extra_include_paths=[str(donor / "csrc")],
            build_directory=str(root),
            verbose=True,
        )
        extension = Path(module.__file__)
        wrappers[arm] = bind_python(python, Path(native.__file__), name, native)
        ops[arm] = getattr(torch.ops, name).row2col_dq.default
        evidence[arm] = {
            "cuda_sha256": probe.sha256(cu),
            "binding_sha256": probe.sha256(cpp),
            "extension_sha256": probe.sha256(extension),
            "build_ninja_sha256": probe.sha256(root / "build.ninja"),
            "sass": disassembly(cuobjdump, extension, root / "kernel.sass"),
        }
    if cuobjdump and (
        evidence["A"]["sass"]["redux"]
        or not any(".MAX" in op for op in evidence["B"]["sass"]["redux_opcodes"])
    ):
        raise RuntimeError(f"Unexpected REDUX disassembly: {evidence}")
    return (
        wrappers,
        ops,
        {
            "donor_revision": revision,
            "sources_sha256": SOURCES,
            "scale_header_sha256": probe.sha256(donor / header),
            "installed_extension_sha256": INSTALLED_EXTENSION_SHA,
            "script_sha256": probe.sha256(__file__),
            "nvcc_flags": flags,
            "cxx_flags": [*family.cxx, "-std=c++20"],
            "private_builds": evidence,
        },
    )


def make_compiled_paths(wrappers, weight_shape, wgrad, torch):
    def operations(converter):
        def convert(index):
            def call(inputs):
                operands, indices, counts = inputs
                return converter(
                    operands[index][0], indices, counts, operands[index][1]
                )

            return call

        def chain(inputs):
            operands, indices, counts, counts_cpu = inputs
            converted = []
            for data, scale in operands:
                packed, packed_scale = converter(data, indices, counts, scale)
                converted.append((packed.view(torch.float8_e4m3fn), packed_scale.mT))
            return wgrad(
                *converted, weight_shape, counts.device, None, counts_cpu, counts
            )

        return {"dy": convert(0), "x": convert(1), "through_wgrad": chain}

    return {
        arm: {
            scope: torch.compile(
                fn,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            for scope, fn in operations(wrapper).items()
        }
        for arm, wrapper in wrappers.items()
    }


def check_special(ops, device, torch):
    # Two live expert tiles and a full, unused capacity tile. The inputs
    # include every E4M3 byte, nonfinite scales, signed zero and FP32 FTZ cases.
    rows, width, used = 384, 256, 256
    data = (
        torch.arange(rows * width, device=device, dtype=torch.int32)
        .to(torch.uint8)
        .view(rows, width)
    )
    scale_bits = [
        0,
        -2147483648,
        1,
        0x007FFFFF,
        0x00800000,
        0x3F800000,
        0x7F7FFFFF,
        0x7F800000,
        0x7FC00000,
    ]
    scales = (
        torch.tensor(scale_bits, dtype=torch.int32, device=device)
        .repeat(86)[: rows * 2]
        .view(2, rows)
        .mT.view(torch.float32)
    )
    counts = torch.tensor([128, 0, 128], dtype=torch.int32, device=device)
    indices = torch.arange(rows, dtype=torch.int32, device=device) // 128 * 2
    indices[used:] = -1

    expected = None
    for arm, op in ops.items():
        data_store = torch.full(
            (rows * width + 32,), 0xA5, dtype=torch.uint8, device=device
        )
        scale_store = torch.full(
            (rows // 128 * width + 8,), 0x5A5A5A5A, dtype=torch.int32, device=device
        )
        packed = data_store[16:-16].view(torch.float8_e4m3fn)
        packed_scale = scale_store[4:-4].view(torch.float32).view(-1, width)
        op(data, scales, counts, indices, packed, packed_scale)
        for output, offset, live, canary in (
            (data_store, 16, used * width, 0xA5),
            (scale_store, 4, used // 128 * width, 0x5A5A5A5A),
        ):
            if not (
                (output[:offset] == canary).all()
                & (output[offset + live :] == canary).all()
            ).item():
                raise RuntimeError(f"{arm} touched a guard or the unused capacity tail")
        actual = (
            packed.view(torch.uint8)[: used * width],
            packed_scale[: used // 128].view(torch.int32),
        )
        if expected is None:
            expected = actual
        elif not all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True)):
            raise RuntimeError(f"{arm} differs on row2col special-value bits")
    return {
        "all_e4m3_encodings": True,
        "fp32_scale_bit_patterns": scale_bits,
        "tail_and_guard_canaries": True,
    }


def launch_profile(operation, inputs, name, arm, scope, device, args, harness):
    path = args.output / f"{name}-{scope}-{arm}.trace.json"
    summary = harness.trace_operation(lambda: operation(inputs), device, path)
    events = json.loads(path.read_bytes())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    converters = [event for event in kernels if KERNEL in event["name"]]
    wgrad = [event for event in kernels if probe.BASELINE_KERNEL in event["name"]]
    suffix = f"_{PRIVATE[arm]}" if arm in PRIVATE else "("
    expected = (2, 1) if scope == "through_wgrad" else (1, 0)
    if (len(converters), len(wgrad)) != expected or any(
        KERNEL + suffix not in event["name"] for event in converters
    ):
        raise RuntimeError(
            f"{name} / {scope} / {arm} unexpected row2col or wgrad launches"
        )
    geometry = []
    for event in converters + wgrad:
        launch = event["args"]
        geometry.append({"grid": launch["grid"], "block": launch["block"]})
    return {"trace": str(path), "geometry": geometry, **summary}


def measure_scope(
    paths,
    inputs,
    name,
    scope,
    device,
    args,
    harness,
    torch,
    measure,
    device_memory_usage,
):
    for arm in ("installed", "A", "B"):
        for _ in range(args.warmup):
            paths[arm][scope](inputs)
        torch.cuda.synchronize(device)

    blocks = []
    for repeat in range(args.pairs):
        order = ("installed", "A", "B") if repeat % 2 == 0 else ("B", "A", "installed")
        arms = {}
        for arm in order:
            block = measure(paths[arm][scope], inputs, args.samples)
            if block["allocator_retries"]:
                raise RuntimeError(f"Allocator retry in {name} / {scope} / {arm}")
            _, block["gift_device_peak_gib"] = device_memory_usage(
                torch.cuda.memory_stats()
            )
            arms[arm] = block
        blocks.append({"order": order, "arms": arms})

    profiles = {
        arm: launch_profile(
            paths[arm][scope], inputs, name, arm, scope, device, args, harness
        )
        for arm in ("installed", "A", "B")
    }
    control = profiles["installed"]
    if any(
        item["geometry"] != control["geometry"]
        or len(item["kernels"]) != len(control["kernels"])
        or len(item["device_memsets"]) != len(control["device_memsets"])
        for item in profiles.values()
    ):
        raise RuntimeError(f"Launch geometry or census changed in {name} / {scope}")
    samples = [
        {"arms": {arm: item["median"] for arm, item in block["arms"].items()}}
        for block in blocks
    ]
    return {
        "blocks": blocks,
        "block_median_candidate_minus_control_ms": {
            field: median(
                block["arms"]["B"]["median"][field]
                - block["arms"]["A"]["median"][field]
                for block in blocks
            )
            for field in blocks[0]["arms"]["A"]["median"]
        },
        "peak_by_arm": {
            arm: max(block["arms"][arm]["gift_device_peak_gib"] for block in blocks)
            for arm in ("installed", "A", "B")
        },
        "paired_statistics": {
            label: {
                field: summarize(samples, reference, trial, field)
                for field in ("cuda_event_ms", "wall_ms")
            }
            for label, reference, trial in (
                ("build_anchor", "installed", "A"),
                ("redux", "A", "B"),
            )
        },
        "profiles": profiles,
    }


def run(args):
    import torch

    from dense_weight_scale import measure
    from torchtitan.ops.float8 import deep_gemm_handle
    from torchtitan.tools.utils import device_memory_usage

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one SM90 GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    if not Path(deep_gemm_handle.__file__).resolve().is_relative_to(CHECKOUT / "gift"):
        raise RuntimeError("The run imported GIFT from outside the task checkout")
    if os.environ.get("GIFT_ROUTED_DW_BF16_OVERWRITE", "0") != "0":
        raise RuntimeError(
            "Use the same installed stock BF16 wgrad for all row2col arms"
        )
    if probe.sha256(args.fixture) != probe.FIXTURE_SHA256:
        raise RuntimeError("Expected the captured native EP8 rank-0 fixture")
    native = importlib.import_module("cudacherry.quantization.row2col_dq")
    harness = probe.reference_module(
        args.reference_root / "runs/efsdp-optimizations/direct-dw", "verify_direct_dw"
    )
    replay = probe.reference_module(
        args.reference_root / "runs/backward-components", "row2col_replay"
    )
    legacy = deep_gemm_handle._load_deep_gemm()
    if probe.sha256(legacy._C.__file__) != probe.BASELINE_SHA256:
        raise RuntimeError("Unexpected installed stock DeepGEMM binary")
    legacy.set_num_sms(probe.NUM_SMS)
    fixture = replay.load_fixture(args.fixture, device)
    wrappers, ops, build_evidence = build(args, native, torch)
    record = {
        "build": build_evidence,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "fixture_sha256": probe.FIXTURE_SHA256,
        "fixture_metadata": fixture.metadata,
        "deepgemm_binary_sha256": probe.BASELINE_SHA256,
        "deepgemm_num_sms": probe.NUM_SMS,
        "compile": {"backend": "inductor", "fullgraph": True, "cudagraphs": False},
        "timing_scope": "one synchronized compiled call per sample, CUDA event may include host launch gaps",
        "warmups": args.warmup,
        "calls_per_arm_block": args.samples,
        "pairs": args.pairs,
        "cases": {},
        "complete": False,
    }
    count_values = fixture.padded_sizes.tolist()
    layouts = {"native-ep8": count_values}
    if args.derived:
        layouts["ep4-shaped-derived"] = probe.derived_counts(count_values)

    with torch.no_grad():
        record["special_parity"] = check_special(ops, device, torch)
        for topology, values in layouts.items():
            indices = replay.expected_indices(values, fixture.m_indices.numel()).to(
                device
            )
            counts_cpu = torch.tensor(values, dtype=torch.int32, pin_memory=True)
            counts = counts_cpu.to(device, non_blocking=True)
            for (projection, roles), shape in zip(
                ROLE_PAIRS.items(), probe.shapes(len(values)), strict=True
            ):
                name = f"{topology}-{projection}"
                operands = tuple(fixture.roles[role] for role in roles)
                converter_inputs = operands, indices, counts
                chain_inputs = *converter_inputs, counts_cpu
                torch._dynamo.reset()
                paths = make_compiled_paths(
                    wrappers,
                    torch.Size(shape),
                    deep_gemm_handle.routed_bf16_wgrad,
                    torch,
                )
                case = {
                    "role_shapes": [list(data.shape) for data, _ in operands],
                    "weight_shape": shape,
                    "expert_sizes": values,
                    "parity": {},
                    "timings": {},
                }
                # Allow first compilation of the unchanged stock GEMM during
                # parity, then make a cache miss fatal in the timed run.
                with patch.dict(os.environ, {"DG_CACHE_FORCE_SKIP": "1"}):
                    for scope, (data, _) in zip(("dy", "x"), operands, strict=True):
                        case["parity"][scope] = {}
                        for baseline, trial in (("installed", "A"), ("A", "B")):
                            comparable = {
                                arm: {
                                    "converter": paths[key][scope],
                                    "through_wgrad": paths[key]["through_wgrad"],
                                }
                                for arm, key in (("A", baseline), ("B", trial))
                            }
                            case["parity"][scope][f"{baseline}-{trial}"] = (
                                verify_outputs(
                                    comparable,
                                    converter_inputs,
                                    chain_inputs,
                                    sum(values),
                                    data.shape[1],
                                    torch,
                                )
                            )
                with torch._dynamo.config.patch(error_on_recompile=True):
                    for scope, inputs in (
                        ("dy", converter_inputs),
                        ("x", converter_inputs),
                        ("through_wgrad", chain_inputs),
                    ):
                        case["timings"][scope] = measure_scope(
                            paths,
                            inputs,
                            name,
                            scope,
                            device,
                            args,
                            harness,
                            torch,
                            measure,
                            device_memory_usage,
                        )
                record["cases"][name] = case
                print(
                    f"{name}: {case['timings']['through_wgrad']['block_median_candidate_minus_control_ms']}",
                    flush=True,
                )
                (args.output / "component.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
    record["complete"] = True
    (args.output / "component.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("reference-root", "donor", "fixture", "output"):
        parser.add_argument(f"--{option}", type=probe.absolute_path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument(
        "--derived",
        action="store_true",
        help="also replay the native rows split over 32 experts",
    )
    args = parser.parse_args()
    if min(args.warmup, args.samples) < 1 or args.pairs < 2 or args.pairs % 2:
        parser.error("Require positive warmup and samples and an even pair count")
    args.output.mkdir(parents=True, exist_ok=False)
    temporary = args.output / "tmp"
    temporary.mkdir()
    with patch.dict(
        os.environ,
        {
            "TMPDIR": str(temporary),
            "CUDACHERRY_DISABLE_JIT": "1",
            "USE_CPP_20_STANDARD": "1",
            "MAX_JOBS": "1",
            "DG_CACHE_FORCE_SKIP": "0",
        },
    ):
        os.environ.pop("DG_JIT_SHARED_CACHE_DIR", None)
        run(args)


if __name__ == "__main__":
    main()
