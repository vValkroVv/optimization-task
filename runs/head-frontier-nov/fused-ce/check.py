"""Test and time the in-place CE on projected real-data captures."""

import argparse
import json
from pathlib import Path
from statistics import median

import torch
import triton

import fused
import replay
from torchtitan.components.loss import MTPLoss
from torchtitan.experiments.four_fusion import stable_ce


def projected_chunk(chunks, part):
    capture = torch.load(replay.FIXTURE, map_location="cpu", weights_only=True, mmap=True)["ce"]
    labels = capture["labels"].cuda()
    positions = capture["positions"]
    if positions is not None:
        positions = positions.cuda()
    shifted = MTPLoss._shift_labels(labels, 1, positions)
    target = torch.cat((labels, shifted))
    width = labels.shape[-1] // chunks
    interval = slice(part * width, (part + 1) * width)
    hidden = capture["hidden"].flatten(0, 1)[:, interval].contiguous().cuda()
    weight = capture["weight"].cuda().to(torch.bfloat16)
    logits = torch.nn.functional.linear(hidden, weight).view(-1, weight.shape[0])
    denominator = capture["denominator"]
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.cuda()
    target = target[:, interval].contiguous().view(-1)
    original_ignores = int((target == -100).sum())
    target[0] = -100
    target[1] = 0
    target[len(target) // 2] = -100
    target[-1] = weight.shape[0] - 1
    scales = fused.incoming_scales(denominator, capture["head_weights"], logits)
    with torch.enable_grad():
        raw_loss = torch.zeros(2, device="cuda", dtype=torch.float32, requires_grad=True)
        x, y = raw_loss.unbind()
        if denominator is not None:
            x, y = x / denominator, y / denominator
        native_scale, = torch.autograd.grad(x * capture["head_weights"][0] + y * capture["head_weights"][1], raw_loss)
    assert torch.equal(native_scale, scales)
    meta = {
        "denominator_dtype": str(denominator.dtype) if isinstance(denominator, torch.Tensor) else type(denominator).__name__,
        "denominator": float(denominator) if denominator is not None else None,
        "incoming_fp32_scales": scales.tolist(),
        "original_ignored_rows": original_ignores,
        "ignore_rows_after_injection": int((target == -100).sum()),
        "shape": list(logits.shape),
    }
    return logits, target, scales, meta


def native(logits, targets, losses, maxima, logs, scales):
    rows, vocab = logits.shape
    for slab in (0, 1):
        interval = slice(slab * (rows // 2), (slab + 1) * (rows // 2))
        x, y, loss, maximum, log_denom = (a[interval] for a in (logits, targets, losses, maxima, logs))
        stable_ce.forward_kernel[(len(y),)](
            x, y, loss, maximum, log_denom, vocab, 4096,
            num_warps=4, enable_fp_fusion=False, debug=True,
        )
    for slab in (0, 1):
        interval = slice(slab * (rows // 2), (slab + 1) * (rows // 2))
        x, y, maximum, log_denom = (a[interval] for a in (logits, targets, maxima, logs))
        stable_ce._launch_backward(x, y, maximum, log_denom, scales[slab], x)


def differences(got, want):
    mismatched, max_absolute, squared, norm = 0, 0.0, 0.0, 0.0
    raw_dtype = torch.int16 if got.dtype == torch.bfloat16 else torch.int32
    for part in range(0, len(got), 64):
        x = got[part:part + 64]
        y = want[part:part + 64]
        mismatched += int((x.view(raw_dtype) != y.view(raw_dtype)).sum(dtype=torch.int64))
        delta = x.float() - y.float()
        max_absolute = max(max_absolute, float(delta.abs().max()))
        squared += float(delta.square().sum())
        norm += float(y.float().square().sum())
    return {"mismatched": mismatched, "elements": got.numel(),
            "max_absolute": max_absolute, "relative_l2": (squared / norm) ** 0.5}


def oracle(logits, dlogits, loss, target, scales):
    total_rows = len(target)
    gpus_rows = [0, 1, 2, total_rows // 2, total_rows // 2 + 1, total_rows - 1]
    values = logits[gpus_rows].double()
    exact_loss = values.logsumexp(1)
    exact_dx = values.softmax(1)
    for index, row in enumerate(gpus_rows):
        label = int(target[row])
        if label == -100:
            exact_loss[index] = 0
            exact_dx[index].zero_()
        else:
            exact_loss[index] -= values[index, label]
            exact_dx[index, label] -= 1
            exact_dx[index] *= scales[int(row >= total_rows // 2)].double()
    return {
        "checked_rows": gpus_rows,
        "max_fp64_row_loss_error": float((exact_loss - loss[gpus_rows].double()).abs().max()),
        "relative_l2_vs_fp64": float(torch.linalg.vector_norm(dlogits[gpus_rows].double() - exact_dx)
                                    / torch.linalg.vector_norm(exact_dx)),
    }


def time_kernels(route, working, logits, rounds=10, captured=False):
    """Measure same-stream CE after restoring pristine BF16 logits."""
    start, end = (torch.cuda.Event(enable_timing=True, external=captured) for _ in range(2))

    def region():
        working.copy_(logits)
        start.record()
        route()
        end.record()

    if captured:
        region()
        end.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            region()
        execute = graph.replay
    else:
        execute = region
    windows = []
    for _ in range(rounds):
        execute()
        end.synchronize()
        windows.append(start.elapsed_time(end))
    return windows


@torch.no_grad()
def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    logits, targets, scales, meta = projected_chunk(args.chunks, args.part)
    scratch = [logits.new_empty((len(targets),), dtype=torch.float32) for _ in range(3)]
    maxima, logs, losses = scratch
    reference = logits.clone()
    native(reference, targets, losses, maxima, logs, scales)
    native_losses = losses.clone()
    working = logits.clone()
    route_native = lambda: native(working, targets, losses, maxima, logs, scales)
    result = {
        "mode_scope": (
            "CUDA event same-stream span; full in-place CE, restored inputs outside window, excludes head sums"
        ),
        "cuda_graph_replay": args.graph,
        "runtime": {"cuda": torch.version.cuda, "torch": torch.__version__, "triton": triton.__version__,
                    "device": torch.cuda.get_device_name(0)},
        "chunks": args.chunks, "part": args.part,
        "capture": str(replay.FIXTURE), **meta,
        "native": {"oracle": oracle(logits, reference, native_losses, targets, scales)},
        "configs": {},
    }
    samples = time_kernels(route_native, working, logits, captured=args.graph)
    result["native"].update(ms=median(samples), samples_ms=samples)
    for name in args.methods:
        launch = fused.Launcher(name)

        def candidate():
            launch(working, targets, losses, scales)

        working.copy_(logits)
        candidate()
        torch.cuda.synchronize()
        grad_delta = differences(working, reference)
        loss_delta = differences(losses, native_losses)
        assert torch.all(working[targets == -100] == 0)
        for a in (native_losses, losses):
            assert torch.isfinite(a).all()
        quality = oracle(logits, working, losses, targets, scales)
        native_samples = time_kernels(route_native, working, logits, rounds=4, captured=args.graph)
        new_samples = time_kernels(candidate, working, logits, captured=args.graph)
        receipt = dict(
            ms=median(new_samples), samples_ms=new_samples,
            paired_native_ms=median(native_samples), paired_native_samples_ms=native_samples,
            dlogits=grad_delta, row_loss=loss_delta, oracle=quality,
        )
        result["configs"][name] = receipt
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(name, receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=int, choices=(2, 8))
    parser.add_argument("output", type=Path)
    parser.add_argument("methods", nargs="+")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--graph", action="store_true")
    run(parser.parse_args())
