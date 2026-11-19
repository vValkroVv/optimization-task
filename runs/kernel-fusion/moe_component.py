"""Compare staged and fused expert SwiGLU backward on one Hopper GPU."""

import argparse
import hashlib
import inspect
import json
import statistics
from pathlib import Path

import torch
import triton

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import _apply_ac_to_block
from torchtitan.ops.float8._cudacherry import requantize_rowwise_to_columnwise
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor
from torchtitan.ops.float8.grouped_gemm import GroupedGemmFp8Wrapper
from torchtitan.ops.float8.quantize_expert_weights import quantize_expert_weight
from torchtitan.ops.float8.triton_kernels.fused_swiglu_quantization import (
    _swiglu_quantized_backward,
    fused_swiglu_quantized_fn,
)
from torchtitan.ops.float8.triton_kernels.row_quantization import (
    row_quantization_1x128_fn,
)
from torchtitan.ops.float8.triton_kernels.utils import build_m_indices


def bf16_randn(*shape):
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16)


def packed_backward(x, probs, dy, limit, indices, counts, fused):
    if fused:
        row, dprobs, col = _swiglu_quantized_backward(
            dy, x, probs, limit, indices, counts
        )
    else:
        row, dprobs, _ = _swiglu_quantized_backward(dy, x, probs, limit)
        col = requantize_rowwise_to_columnwise(row[0], indices, counts, row[1])
    return row[0], row[1], col[0], col[1], dprobs


def parity(names, reference, candidate):
    result = {}
    for name, expected, actual in zip(names, reference, candidate, strict=True):
        assert expected.shape == actual.shape, (name, expected.shape, actual.shape)
        if expected.element_size() == actual.element_size() == 1:
            expected = expected.view(torch.uint8)
            actual = actual.view(torch.uint8)
        mismatch = expected.ne(actual)
        result[name] = {
            "mismatched": mismatch.count_nonzero().item(),
            "elements": mismatch.numel(),
            "max_abs": (expected.float() - actual.float()).abs().max().item(),
        }
        if result[name]["mismatched"]:
            index = mismatch.reshape(-1).to(torch.uint8).argmax().item()
            result[name]["first_mismatch"] = {
                "index": index,
                "staged": expected.reshape(-1)[index].item(),
                "fused": actual.reshape(-1)[index].item(),
            }
    return result


def check(result):
    mismatched = [name for name, values in result.items() if values["mismatched"]]
    if mismatched:
        raise AssertionError(f"Numerical mismatch in {mismatched}")


def measure_graph(fns, repeats):
    graphs = []
    for fn in fns:
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs.append(graph)
    samples = [[], []]
    for _ in range(repeats):
        for graph, times in zip(graphs, samples, strict=True):
            start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(20):
                graph.replay()
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop) * 1000 / 20)
    return [
        {"samples_us": times, "median_us": statistics.median(times)}
        for times in samples
    ]


def standalone(groups, physical_rows, hidden, limit, use_compile, repeats, case):
    nvalid = sum(groups)
    x = bf16_randn(physical_rows, hidden * 2)
    probs = torch.rand(physical_rows, device="cuda")
    dy = (
        bf16_randn(hidden, physical_rows).mT
        if physical_rows != nvalid
        else bf16_randn(physical_rows, hidden)
    )
    # Include zero-probability rows within groups and in the fixed-capacity tail.
    probs[::17] = 0.0
    probs[nvalid:] = 0.0
    dy[nvalid:] = 0.0
    if physical_rows != nvalid:
        # Compare the scale floor on a wholly zero tile in a nonempty expert.
        dy[:128] = 0.0
    counts = torch.tensor(groups, device="cuda", dtype=torch.int32)
    indices = build_m_indices(counts, num_rows=physical_rows)

    def baseline():
        return packed_backward(x, probs, dy, limit, indices, counts, False)

    def candidate():
        return packed_backward(x, probs, dy, limit, indices, counts, True)

    if use_compile:
        baseline = torch.compile(baseline, fullgraph=True)
        candidate = torch.compile(candidate, fullgraph=True)
    names = ("row_fp8", "row_scale", "column_fp8", "column_scale", "dprobs")

    def used_outputs(result):
        row, row_scale, col, col_scale, dprobs = result
        return (
            row,
            row_scale,
            col[: nvalid * 2 * hidden],
            col_scale[: nvalid // 128],
            dprobs,
        )

    case["parity"] = parity(names, used_outputs(baseline()), used_outputs(candidate()))
    check(case["parity"])
    if repeats:
        case["cuda_graph"] = dict(
            zip(
                ("staged", "fused"),
                measure_graph((baseline, candidate), repeats),
                strict=True,
            )
        )


def expert_pair(
    wrapper, x_edge, probs, x_data, x_scale, w13, w2, indices, counts, counts_cpu, fused
):
    x = Float8BlockwiseQTensor(x_data, x_scale, torch.bfloat16, scale_block_first=False)
    layout = {
        "tensor_groups_sizes": None,
        "tensor_groups_sizes_cpu": counts_cpu,
        "tensor_groups_sizes_padded": counts,
        "m_indices": indices,
    }
    fc1, edge = wrapper.group_gemm(
        x,
        weights=quantize_expert_weight(w13),
        grad_edge=x_edge,
        fp8_grad_output=not fused,
        swiglu_probs=probs if fused else None,
        swiglu_limit=4.0,
        **layout,
    )
    if not fused:
        edge, fc1 = fused_swiglu_quantized_fn(edge, fc1, probs, swiglu_limit=4.0)
    out, _ = wrapper.group_gemm(
        fc1,
        weights=quantize_expert_weight(w2),
        grad_edge=edge,
        **layout,
    )
    return out


class GroupedExpertsProbe(torch.nn.Module):
    def __init__(self, wrapper, fused):
        super().__init__()
        self.wrapper = wrapper
        self.fused = fused

    def forward(self, *args):
        return expert_pair(self.wrapper, *args, self.fused)


def expert_pair_parity(args, use_compile, case):
    groups = [args.rows // (128 * 16) * 128] * 16
    groups[0] += args.rows - sum(groups)
    counts_cpu = torch.tensor(groups, dtype=torch.int32, pin_memory=True)
    counts = counts_cpu.cuda(non_blocking=True)
    wrapper = GroupedGemmFp8Wrapper(num_sms=116)
    indices = build_m_indices(counts, num_rows=args.rows)
    x_data, x_scale = row_quantization_1x128_fn(bf16_randn(args.rows, args.dim))
    x_edge = (
        torch.zeros((), device="cuda", dtype=torch.bfloat16)
        .expand(args.rows, args.dim)
        .requires_grad_()
    )
    probs = torch.rand(args.rows, device="cuda", requires_grad=True)
    w13 = (bf16_randn(16, 2 * args.hidden, args.dim) * 0.02).requires_grad_()
    w2 = (bf16_randn(16, args.dim, args.hidden) * 0.02).requires_grad_()
    dy = bf16_randn(args.rows, args.dim)

    results = []
    case["memory"] = {}
    for fused in (False, True):
        model = GroupedExpertsProbe(wrapper, fused)
        if use_compile:
            model = _apply_ac_to_block(
                model,
                ActivationCheckpointConfig(selective_save=[]),
                mode="save:experts_up",
            )
            model = torch.compile(model, fullgraph=True)

        # Warm the same calls once so allocator counts exclude compilation/JIT.
        for measured in (False, True):
            # Eager grouped GEMM releases the saved quantized operands.
            input_data, input_scale = x_data.clone(), x_scale.clone()
            torch.cuda.synchronize()
            allocated_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = model(
                x_edge,
                probs,
                input_data,
                input_scale,
                w13,
                w2,
                indices,
                counts,
                counts_cpu,
            )
            forward_peak = torch.cuda.max_memory_allocated()
            backward_start = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            # Use identical incoming gradients and populate leaf .grad, which
            # also works with compiled non-reentrant selective checkpointing.
            out.backward(dy)
            torch.cuda.synchronize()
            if measured:
                case["memory"]["fused" if fused else "staged"] = dict(
                    allocated_before=allocated_before,
                    forward_peak_allocated=forward_peak,
                    backward_start_allocated=backward_start,
                    backward_peak_allocated=torch.cuda.max_memory_allocated(),
                    backward_peak_reserved=torch.cuda.max_memory_reserved(),
                )
                loss = (out.float() * dy.float()).mean().detach()
                results.append(
                    tuple(
                        t.detach().cpu()
                        for t in (out, loss, x_edge.grad, probs.grad, w13.grad, w2.grad)
                    )
                )
            for leaf in (x_edge, probs, w13, w2):
                leaf.grad = None
            del out

    names = ("output", "loss", "dinput", "dprobs", "dw13", "dw2")
    case["parity"] = parity(names, *results)
    check(case["parity"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=63_872)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--grouped-only", action="store_true")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("The selected DeepGEMM path requires Hopper")
    if args.rows < 128 * 16 or args.rows % 128:
        parser.error("--rows must contain at least one 128-row tile per local expert")
    torch.manual_seed(123)
    vendor = Path(inspect.getfile(requantize_rowwise_to_columnwise))
    results = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "vendor_wrapper": str(vendor),
        "vendor_wrapper_sha256": hashlib.sha256(vendor.read_bytes()).hexdigest(),
        "rows": args.rows,
        "dim": args.dim,
        "hidden": args.hidden,
        "input": "synthetic diagnostic with median production trace row count",
        "graph_replays_per_sample": 20,
    }
    groups = [args.rows // 16 // 128 * 128] * 16
    groups[0] += args.rows - sum(groups)
    try:
        if not args.grouped_only:
            for name, groups, rows, limit, compiled, repeats in (
                ("padded_and_empty_groups", [128, 0, 256], 512, 4.0, False, 0),
                ("production_shape_eager", groups, args.rows, 0.0, False, args.repeats),
                (
                    "production_shape_compiled",
                    groups,
                    args.rows,
                    4.0,
                    True,
                    args.repeats,
                ),
            ):
                print(name, flush=True)
                results[name] = {"padded_groups": groups, "physical_rows": rows}
                standalone(
                    groups, rows, args.hidden, limit, compiled, repeats, results[name]
                )
        for compiled in (False, True):
            name = "grouped_pair_compile_and_sac" if compiled else "grouped_pair_eager"
            print(name, flush=True)
            results[name] = {}
            expert_pair_parity(args, compiled, results[name])
    finally:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
