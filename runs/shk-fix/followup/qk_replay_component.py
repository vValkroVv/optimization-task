"""Compare stock and Q/K-only compiled VJP replay on native GDN short-conv inputs."""

import argparse
import json
from math import isfinite
from pathlib import Path
import platform

import torch
import triton

from dense_dy_quant_transpose import digest, measure
from dense_dy_qkv_fixture import QKV_SHAPE, load_capture
from routed_swiglu_fwd_order_component import summarize


HEADS = 16
VALUE_HEADS = 32
HEAD_DIM = 128
WINDOW = 4
TOKEN_TILE = 64
VALUE_GROUPS = VALUE_HEADS // HEADS
GROUP_WIDTH = (2 + VALUE_GROUPS) * HEAD_DIM
CAPTURE_SHA256 = "7e244453fe8d7de975438cc9584a112e687832fd06896bf4db47bc63eb4c52f3"
FLA_SHA256 = "5abe9c02fa191f58df29031b52f023f780d7c5d4b84cf2d1f3c77397a5e881cf"
GIFT_SHA256 = "f595ecaf1a83fcb0dcfab11a7836638e3f34080406642b8a99a6679b90292201"


def load_inputs(path):
    data, receipt = load_capture(path)
    eps = data["eps"]
    if (
        receipt["fixture_sha256"] != CAPTURE_SHA256
        or data["num_heads"] != HEADS
        or data["num_v_heads"] != VALUE_HEADS
        or data["normalize_qk"] is not True
        or data["cu_seqlens"] is not None
        or not isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("Expected the normalized, fixed-length production GDN capture")

    batch, tokens, width = QKV_SHAPE
    shapes = {
        "qkv": (QKV_SHAPE,),
        "conv_weight": ((width, WINDOW), (width, 1, WINDOW)),
        "grad_q": ((batch, tokens, HEADS, HEAD_DIM),),
        "grad_k": ((batch, tokens, HEADS, HEAD_DIM),),
        "grad_v": ((batch, tokens, VALUE_HEADS, HEAD_DIM),),
    }
    operands = []
    inputs = {}
    for name, allowed in shapes.items():
        tensor = data[name]
        dtypes = (
            (torch.bfloat16, torch.float32)
            if name == "conv_weight"
            else (torch.bfloat16,)
        )
        if tensor.dtype not in dtypes or tuple(tensor.shape) not in allowed:
            raise ValueError(
                f"Expected captured {name} with shape in {allowed} and dtype in {dtypes}"
            )
        tensor = tensor.to("cuda")
        operands.append(tensor)
        inputs[name] = {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
        }
    return (tuple(operands), float(eps)), {
        "path": str(path.resolve()),
        "capture_receipt": receipt,
        "operands": inputs,
        "eps": eps,
    }


def weight_matrix(weight):
    if weight.ndim == 3:
        weight = weight[:, 0, :]
    return weight.contiguous()


def replay_grid(segments):
    batch, tokens, _ = QKV_SHAPE
    return HEADS * segments, triton.cdiv(tokens, TOKEN_TILE), batch


def replay_qk(x, weight, eps):
    from fla.ops.common.fused_conv_qkv import _qkv_conv1d_unpack_kernel

    batch, tokens, width = x.shape
    q = x.new_empty((batch, tokens, HEADS, HEAD_DIM))
    k = torch.empty_like(q)
    rstd_q = x.new_empty((batch, tokens, HEADS), dtype=torch.float32)
    rstd_k = torch.empty_like(rstd_q)
    # The two segments select only Q and K. The V and initial-state pointers
    # are unreachable. Keep the original packed stride that includes V.
    _qkv_conv1d_unpack_kernel[replay_grid(2)](
        x,
        weight,
        x,
        q,
        k,
        q,
        rstd_q,
        rstd_k,
        B=batch,
        T=tokens,
        BT_=triton.cdiv(tokens, 8192),
        H=HEADS,
        HV=VALUE_HEADS,
        HD=HEAD_DIM,
        HVD=HEAD_DIM,
        W=WINDOW,
        D=width,
        GROUP_WIDTH=GROUP_WIDTH,
        VALUE_GROUPS=VALUE_GROUPS,
        NUM_SEGMENTS=2,
        BLOCK_D=HEAD_DIM,
        BT=TOKEN_TILE,
        HAS_INITIAL_STATE=False,
        NORMALIZE_QK=True,
        SAVE_RSTD=True,
        EPS=eps,
    )
    return q, k, rstd_q, rstd_k


@torch.library.custom_op("shk_qk_replay::vjp", mutates_args=())
def qk_vjp(
    x: torch.Tensor,
    weight: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.common.fused_conv_qkv import _fused_qkv_conv1d_unpack_backward_via_fla

    x = x.contiguous()
    matrix = weight_matrix(weight)
    q, k, rstd_q, rstd_k = replay_qk(x, matrix, eps)
    dx, dw, _ = _fused_qkv_conv1d_unpack_backward_via_fla(
        x=x,
        weight=matrix,
        dq=dq.contiguous(),
        dk=dk.contiguous(),
        dv=dv.contiguous(),
        q=q,
        k=k,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        num_heads=HEADS,
        num_v_heads=VALUE_HEADS,
        head_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        initial_state=None,
        cp_context=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        normalize_qk=True,
        bt=TOKEN_TILE,
    )
    return dx, dw[:, None, :] if weight.ndim == 3 else dw


@qk_vjp.register_fake
def _fake_qk_vjp(x, weight, dq, dk, dv, eps):
    return torch.empty_like(x), torch.empty_like(weight)


def compile_pair(fla_vjp, eps):
    def control(operands):
        x, weight, dq, dk, dv = operands
        dx, dw = fla_vjp(
            "qkv_conv1d",
            [x, weight],
            [dq, dk, dv],
            [None],
            [HEADS, VALUE_HEADS, HEAD_DIM, HEAD_DIM, True, eps],
            "",
        )
        return dx, dw

    def candidate(operands):
        return qk_vjp(*operands, eps)

    return {
        name: torch.compile(
            operation,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        for name, operation in (("A", control), ("B", candidate))
    }


def compare_results(labels, control, candidate):
    result = {}
    for name, a, b in zip(labels, control, candidate, strict=True):
        metadata_equal = (a.dtype, a.shape, a.stride()) == (
            b.dtype,
            b.shape,
            b.stride(),
        )
        bit_dtype = torch.int32 if a.dtype == torch.float32 else torch.int16
        result[name] = {
            "metadata_equal": metadata_equal,
            "mismatched_elements": (
                int(
                    torch.count_nonzero(
                        a.contiguous().view(bit_dtype) != b.contiguous().view(bit_dtype)
                    )
                )
                if metadata_equal
                else None
            ),
        }
    return result


def compare_replay(fla, x, weight, eps):
    x = x.contiguous()
    weight = weight_matrix(weight)
    batch, tokens, _ = x.shape
    rstd_q = x.new_empty((batch, tokens, HEADS), dtype=torch.float32)
    rstd_k = torch.empty_like(rstd_q)
    q, k, _ = fla._fused_qkv_conv1d_unpack_forward(
        x,
        weight,
        num_heads=HEADS,
        num_v_heads=VALUE_HEADS,
        head_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        normalize_qk=True,
        eps=eps,
        bt=TOKEN_TILE,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
    )
    return compare_results(
        ("q", "k", "rstd_q", "rstd_k"),
        (q, k, rstd_q, rstd_k),
        replay_qk(x, weight, eps),
    )


def parse_profile(events, arm):
    kernels = sorted(
        (event for event in events if event.get("cat") == "kernel"),
        key=lambda event: event["ts"],
    )
    details = [
        {
            "name": event["name"],
            "grid": event["args"].get("grid"),
            "block": event["args"].get("block"),
            "duration_us": event["dur"],
        }
        for event in kernels
    ]
    copies = [
        {
            "name": event["name"],
            "bytes": event["args"].get("bytes"),
            "duration_us": event["dur"],
        }
        for event in events
        if event.get("cat") in {"gpu_memcpy", "gpu_memset"}
    ]
    replay = [item for item in details if "_qkv_conv1d_unpack_kernel" in item["name"]]
    pack = [item for item in details if "_qkv_bwd_pack_kernel" in item["name"]]
    op_name = "torchtitan::fla_vjp" if arm == "A" else "shk_qk_replay::vjp"
    dispatches = sum(event.get("name") == op_name for event in events)
    segments = 2 + VALUE_GROUPS if arm == "A" else 2
    first = tail = kernels[0]["ts"] if kernels else 0.0
    busy = 0.0
    for event in kernels:
        end = event["ts"] + event["dur"]
        busy += max(0.0, end - max(tail, event["ts"]))
        tail = max(tail, end)
    span = tail - first
    return {
        "activation_verified": (
            dispatches == 1
            and len(replay) == len(pack) == 1
            and replay[0]["grid"] == list(replay_grid(segments))
            and replay[0]["block"] == [128, 1, 1]
            and pack[0]["grid"] == list(replay_grid(2 + VALUE_GROUPS))
            and details.index(replay[0]) < details.index(pack[0])
            and any("causal_conv1d_fwd_kernel" in item["name"] for item in details)
            and any("causal_conv1d_bwd_kernel" in item["name"] for item in details)
        ),
        "op_dispatches": dispatches,
        "kernels": details,
        "copies": copies,
        "kernel_sum_us": sum(item["duration_us"] for item in details),
        "kernel_union_us": busy,
        "kernel_interval_span_us": span,
        "kernel_interval_gaps_us": max(0.0, span - busy),
        "replay_us": replay[0]["duration_us"] if len(replay) == 1 else None,
    }


def trace(operation, operands, arm, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        result = operation(operands)
        torch.cuda.synchronize()
    del result
    profiler.export_chrome_trace(str(path))
    return {
        **parse_profile(json.loads(path.read_bytes())["traceEvents"], arm),
        "path": str(path),
        "sha256": digest(path),
    }


def unchanged_other_launches(a, b):
    def signature(profile):
        return [
            (item["name"], item["grid"], item["block"])
            for item in profile["kernels"]
            if "_qkv_conv1d_unpack_kernel" not in item["name"]
        ]

    def transfers(profile):
        return [(item["name"], item["bytes"]) for item in profile["copies"]]

    return signature(a) == signature(b) and transfers(a) == transfers(b)


def paired_measurements(pair, operands, args, device_memory_usage):
    pairs = []
    for index in range(args.pairs):
        order = ("A", "B", "A_control", "A_same")
        if index % 2:
            order = order[::-1]
        arms = {}
        for arm in order:
            operation = pair["B" if arm == "B" else "A"]
            torch.cuda.empty_cache()
            for _ in range(4):
                operation(operands)
            block = measure(operation, operands, args.samples)
            _, block["gift_device_peak_gib"] = device_memory_usage(
                torch.cuda.memory_stats()
            )
            arms[arm] = block
        pairs.append({"order": order, "arms": arms})
        a, b = (arms[arm]["medians"]["cuda_event_ms"] for arm in ("A", "B"))
        print(f"Pair {index + 1} CUDA event A {a:.6f} ms, B {b:.6f} ms", flush=True)

    paired_medians = [
        {"arms": {arm: block["medians"] for arm, block in pair["arms"].items()}}
        for pair in pairs
    ]
    return {
        "scope": (
            "Repeated standalone compiled VJP on resident inputs. CUDA events bracket "
            "the submitted call on one stream. Host submission and host drain are "
            "measured separately, with synchronization after each call."
        ),
        "warmup_per_arm": args.warmup,
        "samples_per_arm_per_pair": args.samples,
        "pairs": pairs,
        "peak_by_arm": {
            arm: {
                field: max(pair["arms"][arm][field] for pair in pairs)
                for field in (
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "gift_device_peak_gib",
                )
            }
            for arm in ("A", "B", "A_control", "A_same")
        },
        "statistics": {
            label: {
                field: summarize(paired_medians, control, trial, field)
                for field in paired_medians[0]["arms"]["A"]
            }
            for label, control, trial in (
                ("AB", "A", "B"),
                ("AA", "A_control", "A_same"),
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=16)
    args = parser.parse_args()
    trace_paths = {
        arm: args.output.with_suffix(f".{arm}.trace.json") for arm in ("A", "B")
    }
    if any(path.exists() for path in (args.output, *trace_paths.values())):
        parser.error("Specify a fresh output path")
    if args.warmup < 1 or args.samples < 1 or args.pairs < 2:
        parser.error("Require positive warmup and samples and at least two pairs")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    torch.cuda.set_device(0)

    from fla.ops.common import fused_conv_qkv
    from torchtitan.models.common.attention.linear import fla_custom_ops
    from torchtitan.tools.utils import device_memory_usage

    implementation = {}
    for name, module, expected in (
        ("fla", fused_conv_qkv, FLA_SHA256),
        ("gift", fla_custom_ops, GIFT_SHA256),
    ):
        path = str(Path(module.__file__).resolve())
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(
                f"Expected source SHA-256 {expected} for {path}, got {actual}"
            )
        implementation[name] = {"path": path, "sha256": actual}

    (operands, eps), provenance = load_inputs(args.capture)
    pair = compile_pair(fla_custom_ops.fla_vjp, eps)
    parity = {
        **compare_results(("dx", "dweight"), pair["A"](operands), pair["B"](operands)),
        **compare_replay(fused_conv_qkv, *operands[:2], eps),
    }
    exact = all(
        item["metadata_equal"] and item["mismatched_elements"] == 0
        for item in parity.values()
    )
    receipt = {
        "scope": "Fixed-length, normalized, CP1, state-free GDN QKV convolution VJP",
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "implementation": implementation,
        "harness_sha256": digest(__file__),
        "inputs": provenance,
        "compile": {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "cudagraphs": False,
        },
        "forward_and_saved_tensors": "Original GIFT forward and its saves are outside the VJP cut",
        "removed_replay_output_bytes": QKV_SHAPE[0]
        * QKV_SHAPE[1]
        * VALUE_HEADS
        * HEAD_DIM
        * 2,
        "parity": parity,
        "bitwise": exact,
    }
    if exact:
        torch._dynamo.config.error_on_recompile = True
        for operation in pair.values():
            for _ in range(args.warmup):
                operation(operands)
            torch.cuda.synchronize()
        profiles = {
            arm: trace(pair[arm], operands, arm, trace_paths[arm]) for arm in pair
        }
        receipt["profiles"] = profiles
        receipt["unchanged_other_launches"] = unchanged_other_launches(
            profiles["A"], profiles["B"]
        )
        receipt["activation_verified"] = receipt["unchanged_other_launches"] and all(
            profile["activation_verified"] for profile in profiles.values()
        )
        if receipt["activation_verified"]:
            receipt["timing"] = paired_measurements(
                pair, operands, args, device_memory_usage
            )

    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    if not exact:
        raise RuntimeError(
            f"Q/K-only replay or convolution VJP bitwise parity failed. See {args.output}"
        )
    if not receipt["activation_verified"]:
        raise RuntimeError(
            f"Unexpected kernel geometry, launches or transfers. See {args.output}"
        )


if __name__ == "__main__":
    main()
