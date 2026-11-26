"""Benchmark one standalone folded CE route on a preprojected model chunk."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time
import traceback

import torch

from matrix_fixture import SHAPES, VOCAB, load_inputs
from vendor_ops import IGNORE_INDEX, VENDOR_VARIANTS, VendorUnavailable, make_vendor_ce


ROWS = {("100b", 2): 8192, ("100b", 8): 2048, ("700b", 8): 3072}
VARIANTS = ("baseline", "folded", "cuda13", "liger-early", *VENDOR_VARIANTS)
GIB = 1 << 30


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def make_route(args, variant):
    if variant == "liger-early":
        from liger_early import make_liger_early_ce
        route = make_liger_early_ce()
        return route, route.backend_receipt
    if variant in VENDOR_VARIANTS:
        route = make_vendor_ce(variant)
        return route, route.backend_receipt
    if variant == "cuda13":
        from torchtitan.ops import exact_folded_ce

        if args.exact_so is None or not args.exact_so.is_file():
            raise VendorUnavailable("Generalized cuda13 requires a built shared library at --exact-so")
        binary = args.exact_so.resolve()
        digest = args.exact_sha256 or file_sha256(binary)
        route = exact_folded_ce.ExactFoldedCrossEntropy(str(binary), library_sha256=digest)
        return route, {
            "backend": "GIFT generalized cuda13",
            "binary": str(binary),
            "binary_sha256": route.library_sha256,
            "source_sha256": file_sha256(exact_folded_ce.__file__),
        }
    if variant == "baseline":
        from torchtitan.components import loss

        source = loss
        op = torch.compile(loss.cross_entropy_loss, backend="inductor", fullgraph=True)

        def route(logits, labels, weights, denominator):
            return tuple(op(x, y) for x, y in zip(logits.chunk(2), labels.chunk(2), strict=True))

        backend = "GIFT FP32-input PyTorch CE, compiled separately per head"
    else:
        from torchtitan.experiments.four_fusion import stable_ce

        source = stable_ce
        op = torch.compile(stable_ce.folded_cross_entropy, backend="inductor", fullgraph=True)

        def route(logits, labels, weights, denominator):
            return op(logits, labels, 2)

        backend = "GIFT historical stable folded CE, compiled"
    return route, {"backend": backend, "source": source.__file__, "source_sha256": file_sha256(source.__file__)}


@torch.no_grad()
def project_chunk(args):
    from torchtitan.components.loss import MTPLoss

    record = load_inputs(args.case, args.fixture, args.extra_fixture)
    hidden = record["hidden"]
    weight = record["weight"]
    target = record["labels"]
    positions = record["positions"]
    weights = tuple(record["head_weights"])
    denominator = record["denominator"]
    denominator_value = float(denominator)
    if not math.isfinite(denominator_value) or denominator_value <= 0:
        raise ValueError("Expected a finite positive global denominator")
    if isinstance(denominator, torch.Tensor):
        if denominator.ndim != 0:
            raise ValueError("Expected a scalar global denominator")
        denominator = denominator.cuda()
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or target.dtype != torch.int64:
        raise ValueError("Expected captured BF16 hidden/weight and int64 labels")
    if positions is not None and positions.shape != target.shape:
        raise ValueError("Expected aligned packed positions")
    if record.get("ignore_index", IGNORE_INDEX) != IGNORE_INDEX:
        raise ValueError("Unexpected ignore index")

    shifted = MTPLoss._shift_labels(target, 1, positions)
    width = target.size(1) // args.chunks
    interval = slice(args.part * width, (args.part + 1) * width)
    targets = torch.cat((target, shifted))[:, interval].contiguous()
    invalid = (targets != IGNORE_INDEX) & ((targets < 0) | (targets >= VOCAB))
    if invalid.any():
        raise ValueError("Captured targets outside production vocabulary")
    activations = hidden.flatten(0, 1)[:, interval].contiguous().cuda()
    logits = torch.nn.functional.linear(activations, weight.cuda())
    if logits.numel() != ROWS[args.case, args.chunks] * VOCAB:
        raise RuntimeError(f"Unexpected projected chunk shape {logits.shape}")
    receipt = {
        "path": str(args.fixture.resolve()),
        "file_bytes": args.fixture.stat().st_size,
        "provenance": record["provenance"],
        "chunk_index": args.part,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_bytes": logits.numel() * logits.element_size(),
        "projected_vocab_stride_4093_sample_sha256": tensor_sha256(logits[:, :, ::4093]),
        "target_sha256": tensor_sha256(targets),
        "valid_rows_by_head": [int((head != IGNORE_INDEX).sum()) for head in targets.chunk(2)],
        "head_weights": list(weights),
        "global_denominator": denominator_value,
        "denominator_type": str(denominator.dtype) if isinstance(denominator, torch.Tensor) else type(denominator).__name__,
    }
    return logits, targets.cuda(), weights, denominator, receipt


def evaluate(route, logits, labels, weights, denominator):
    main, mtp = route(logits, labels, weights, denominator)
    if main.shape != () or mtp.shape != () or main.dtype != torch.float32 or mtp.dtype != torch.float32:
        raise RuntimeError("CE must return two FP32 scalar sums")
    weighted = (main / denominator) * weights[0] + (mtp / denominator) * weights[1]
    (grad,) = torch.autograd.grad(weighted, logits)
    if grad.shape != logits.shape or grad.dtype != torch.bfloat16 or not grad.is_contiguous():
        raise RuntimeError("CE gradient must match the contiguous BF16 logits contract")
    return main.detach(), mtp.detach(), weighted.detach(), grad.detach()


def elapsed_series(times):
    return {"samples_ms": times, "median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times)}


def measure(route, input_data, working, warmup, samples):
    logits, labels, weights, denominator = input_data
    for _ in range(warmup):
        working.detach().copy_(logits)
        evaluate(route, working, labels, weights, denominator)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_stats()
    torch.cuda.reset_peak_memory_stats()

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    host_submit, host_wait, wall, event = [], [], [], []
    for _ in range(samples):
        working.detach().copy_(logits)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        start.record()
        result = evaluate(route, working, labels, weights, denominator)
        end.record()
        submitted = time.perf_counter()
        end.synchronize()
        finished = time.perf_counter()
        host_submit.append((submitted - begin) * 1000)
        host_wait.append((finished - submitted) * 1000)
        wall.append((finished - begin) * 1000)
        event.append(start.elapsed_time(end))
        aliases = result[-1].data_ptr() == working.data_ptr()
        del result

    from torchtitan.tools.utils import device_memory_usage

    stats = torch.cuda.memory_stats()
    external, total = device_memory_usage(stats)
    return {
        "scope": "single-stream CE forward, two FP32 head sums, weighted normalization and external BF16 dZ backward; input restoration and projection excluded",
        "cuda_event_submission_inclusive": elapsed_series(event),
        "host_submission": elapsed_series(host_submit),
        "host_final_sync": elapsed_series(host_wait),
        "wall_synchronized": elapsed_series(wall),
        "gradient_aliases_working_logits": aliases,
        "memory": {
            "scope": "timed eager invocations only, including retained pristine logits and the working allocation",
            "starting_allocated_bytes": before["allocated_bytes.all.current"],
            "starting_reserved_bytes": before["reserved_bytes.all.current"],
            "peak_allocated_bytes": stats["allocated_bytes.all.peak"],
            "peak_reserved_bytes": stats["reserved_bytes.all.peak"],
            "gift_external_gib": external,
            "gift_total_peak_gib": total,
            "allocator_retries": stats["num_alloc_retries"] - before["num_alloc_retries"],
        },
    }


def graph_measure(route, input_data, working, warmup, samples):
    logits, labels, weights, denominator = input_data
    start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    # CUDA-13 caches and checks its FP32 scales per stream. Populate that
    # cache before capture so the measured capture never initializes it.
    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            working.detach().copy_(logits)
            evaluate(route, working, labels, weights, denominator)
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        working.detach().copy_(logits)
        start.record()
        captured_result = evaluate(route, working, labels, weights, denominator)
        end.record()
    elapsed = []
    for index in range(samples + 1):
        graph.replay()
        end.synchronize()
        if index > 0:
            elapsed.append(start.elapsed_time(end))
    return {
        "status": "measured",
        "scope": "same-stream warm-cache CUDA Graph replay span; CE, two head sums, normalization and dZ; events exclude restoration and Python submission, include device gaps",
        "capture_stream_warmup": warmup,
        "discarded_priming_replays": 1,
        "gradient_aliases_working_logits": captured_result[-1].data_ptr() == working.data_ptr(),
        "elapsed": elapsed_series(elapsed),
    }


def oracle_scales(logits, weights, denominator):
    prototype = logits.new_zeros(2, dtype=torch.float32, requires_grad=True)
    main, mtp = prototype.unbind()
    weighted = (main / denominator) * weights[0] + (mtp / denominator) * weights[1]
    return torch.autograd.grad(weighted, prototype)[0]


@torch.no_grad()
def compare_dense(logits, labels, scales, candidate, reference):
    """Use FP64 for every logit, with fixed-size tiles to bound temporary memory."""
    diagnostics = []
    for x, y, got, ref, scale in zip(
        logits.flatten(0, 1).chunk(2), labels.flatten().chunk(2),
        candidate.flatten(0, 1).chunk(2), reference.flatten(0, 1).chunk(2), scales, strict=True,
    ):
        sums = x.new_zeros(11, dtype=torch.float64)
        maxima = x.new_zeros(2, dtype=torch.float64)
        for block, targets, output, control in zip(x.split(32), y.split(32), got.split(32), ref.split(32), strict=True):
            valid = targets != IGNORE_INDEX
            safe = targets.masked_fill(~valid, 0).unsqueeze(1)
            values = block.double()
            values.sub_(values.amax(dim=1, keepdim=True))
            row_losses = (values.logsumexp(dim=1, keepdim=True) - values.gather(1, safe)).squeeze(1)
            row_losses.masked_fill_(~valid, 0)
            exact = values.softmax(dim=1)
            exact.scatter_add_(1, safe, -valid[:, None].double())
            exact.mul_(scale.double())
            exact.masked_fill_(~valid[:, None], 0)
            error = output.double() - exact
            control_error = control.double() - exact
            ideal_bits = exact.to(torch.bfloat16).view(torch.int16)
            output_bits, control_bits = output.view(torch.int16), control.view(torch.int16)
            sums += torch.stack((
                row_losses.sum(), exact.square().sum(), error.square().sum(), control_error.square().sum(),
                (output_bits != control_bits).sum(), (output_bits != ideal_bits).sum(),
                (control_bits != ideal_bits).sum(), (~output.isfinite()).sum(), (~control.isfinite()).sum(),
                ((output_bits != 0) & ~valid[:, None]).sum(), ((control_bits != 0) & ~valid[:, None]).sum(),
            ))
            maxima = torch.maximum(maxima, torch.stack((error.abs().max(), control_error.abs().max())))

        raw, norm, err, ref_err, diff, rounded, ref_rounded, nonfinite, ref_nonfinite, ignore, ref_ignore = sums.tolist()
        max_abs, ref_max_abs = maxima.tolist()
        diagnostics.append({
            "oracle_raw_fp64_sum": raw,
            "elements": got.numel(),
            "bit_diff_vs_folded": int(diff),
            "candidate": {
                "relative_l2_vs_fp64": math.sqrt(err / norm) if norm else 0.0,
                "max_abs_vs_fp64": max_abs,
                "bit_diff_vs_once_rounded_oracle": int(rounded),
                "nonfinite": int(nonfinite),
                "ignored_dz_not_positive_zero": int(ignore),
            },
            "folded": {
                "relative_l2_vs_fp64": math.sqrt(ref_err / norm) if norm else 0.0,
                "max_abs_vs_fp64": ref_max_abs,
                "bit_diff_vs_once_rounded_oracle": int(ref_rounded),
                "nonfinite": int(ref_nonfinite),
                "ignored_dz_not_positive_zero": int(ref_ignore),
            },
        })
    return diagnostics


def verify(args, route, input_data, working):
    logits, labels, weights, denominator = input_data
    working.detach().copy_(logits)
    output = evaluate(route, working, labels, weights, denominator)
    if args.variant == "folded":
        control = output
        reference_receipt = None
    else:
        folded, reference_receipt = make_route(args, "folded")
        ref_input = logits.clone().requires_grad_(True)
        control = evaluate(folded, ref_input, labels, weights, denominator)
    scales = oracle_scales(logits, weights, denominator)
    diagnostics = compare_dense(logits, labels, scales, output[-1], control[-1])
    absolute_errors = []
    for head, got, ref in zip(diagnostics, output[:2], control[:2], strict=True):
        expected = head["oracle_raw_fp64_sum"]
        for key, value in (("candidate", got), ("folded", ref)):
            head[key].update(raw_fp32_sum=float(value), raw_sum_abs_error=abs(float(value) - expected))
        absolute_errors.append(head["candidate"]["raw_sum_abs_error"] <= head["folded"]["raw_sum_abs_error"])

    oracle_weighted = sum(h["oracle_raw_fp64_sum"] * w / float(denominator) for h, w in zip(diagnostics, weights, strict=True))
    weighted_errors = {key: abs(float(result[2]) - oracle_weighted) for key, result in (("candidate", output), ("folded", control))}
    valid = all(
        head["candidate"]["nonfinite"] == 0 and head["candidate"]["ignored_dz_not_positive_zero"] == 0
        and head["folded"]["nonfinite"] == 0 and head["folded"]["ignored_dz_not_positive_zero"] == 0
        for head in diagnostics
    ) and all(math.isfinite(float(value)) for value in (*output[:3], *control[:3]))
    dz_no_regression = all(
        head["candidate"][metric] <= head["folded"][metric]
        for head in diagnostics for metric in ("relative_l2_vs_fp64", "max_abs_vs_fp64")
    )
    return {
        "oracle_scope": "all rows and columns of the timed BF16 input; FP64 centered CE, FP64 sums and FP64 VJP using the exact FP32 GIFT upstream scale",
        "accepted_reference_route": reference_receipt,
        "incoming_fp32_scales": scales.tolist(),
        "heads": diagnostics,
        "weighted_loss": float(output[2]),
        "weighted_loss_folded": float(control[2]),
        "weighted_loss_oracle_fp64": oracle_weighted,
        "weighted_loss_abs_errors": weighted_errors,
        "bitwise_folded": all(
            torch.equal(got.view(torch.int32), ref.view(torch.int32))
            for got, ref in zip(output[:3], control[:3], strict=True)
        ) and all(head["bit_diff_vs_folded"] == 0 for head in diagnostics),
        "strict_non_regression": {
            "finite_and_ignored_positive_zero": valid,
            "raw_head_sums": absolute_errors,
            "weighted_loss": weighted_errors["candidate"] <= weighted_errors["folded"],
            "dz_relative_l2_and_max_abs": dz_no_regression,
            "passed": valid and all(absolute_errors) and weighted_errors["candidate"] <= weighted_errors["folded"] and dz_no_regression,
        },
    }


def run(args, receipt):
    import triton

    if not torch.cuda.is_available():
        raise VendorUnavailable("Standalone CE measurement requires a CUDA GPU")
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(args.gpu)
    receipt["environment"] = {
        "host": socket.gethostname(), "gpu": props.name, "gpu_uuid": str(props.uuid),
        "gpu_capacity_gib": props.total_memory / GIB, "torch": str(torch.__version__),
        "torch_git": torch.version.git_version, "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    route, identity = make_route(args, args.variant)
    receipt["route"] = identity
    logits, labels, weights, denominator, fixture = project_chunk(args)
    receipt["fixture"] = fixture
    input_data = logits, labels, weights, denominator
    working = torch.empty_like(logits, requires_grad=True)
    if not args.graph_only:
        receipt["timing"] = measure(route, input_data, working, args.warmup, args.samples)
        receipt["numerical"] = verify(args, route, input_data, working)
    if args.graph or args.graph_only:
        try:
            receipt["graph"] = graph_measure(
                route, input_data, working, args.warmup if args.graph_only else 1, args.samples,
            )
        except (RuntimeError, TypeError) as error:
            receipt["graph"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    if args.variant == "cuda13":
        receipt["route"]["scales_checked"] = route.scales_checked
    receipt["status"] = receipt["graph"]["status"] if args.graph_only else "measured"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=SHAPES)
    parser.add_argument("--chunks", required=True, type=int, choices=(2, 8))
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--extra-fixture", type=Path)
    parser.add_argument("--exact-so", type=Path)
    parser.add_argument("--exact-sha256")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--graph", action="store_true", help="also measure a controlled CUDA Graph replay span")
    parser.add_argument("--graph-only", action="store_true", help="measure only graph replay after a complete eager receipt")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (args.case, args.chunks) not in ROWS or not 0 <= args.part < args.chunks:
        parser.error("Unknown case/chunk combination or chunk index outside case")
    if args.warmup < 1 or args.samples < 1:
        parser.error("Warmup and sample counts must be positive")

    receipt = {
        "argv": sys.argv,
        "case": args.case,
        "chunks": args.chunks,
        "variant": args.variant,
        "warmup": args.warmup,
        "samples": args.samples,
        "graph_only": args.graph_only,
        "source_hashes": {
            "ce_bench.py": file_sha256(__file__),
            "vendor_ops.py": file_sha256(Path(__file__).with_name("vendor_ops.py")),
            "matrix_fixture.py": file_sha256(Path(__file__).with_name("matrix_fixture.py")),
        },
    }
    try:
        run(args, receipt)
    except VendorUnavailable as error:
        receipt.update(status="unavailable", error=str(error))
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{args.case} C{args.chunks} {args.variant}: {receipt['status']} ({args.output})", flush=True)
    return 1 if receipt["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
