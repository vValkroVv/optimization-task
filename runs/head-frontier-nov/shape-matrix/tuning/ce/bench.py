"""Compare consuming CE binaries on production-tokenizer logits and test their ABI."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from ce_bench import elapsed_series, evaluate, project_chunk
from torchtitan.ops.exact_folded_ce import ExactFoldedCrossEntropy
from torchtitan.tools.utils import device_memory_usage


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bit_differences(a, b):
    storage_type = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
    a, b = a.view(storage_type), b.view(storage_type)
    counts = [(left != right).sum() for left, right in zip(a.split(128), b.split(128), strict=True)]
    return int(torch.stack(counts).sum())


def eager(routes, pristine, working, call, warmup, samples):
    times = {name: [] for name in routes}
    names = list(routes)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for cycle in range(warmup + samples):
        if cycle == warmup:
            torch.cuda.empty_cache()
        shift = cycle % len(names)
        for name in names[shift:] + names[:shift]:
            working.detach().copy_(pristine)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            start.record()
            result = call(routes[name])
            stop.record()
            submitted = time.perf_counter()
            stop.synchronize()
            finished = time.perf_counter()
            if cycle >= warmup:
                if result[-1].data_ptr() != working.data_ptr():
                    raise RuntimeError(f"{name} failed to consume its logits in place")
                stats = torch.cuda.memory_stats()
                times[name].append(dict(wall_ms=(finished - begin) * 1000,
                                        submit_ms=(submitted - begin) * 1000,
                                        wait_ms=(finished - submitted) * 1000,
                                        event_ms=start.elapsed_time(stop),
                                        total_gib=device_memory_usage(stats)[1],
                                        allocation_retries=stats["num_alloc_retries"]))
            del result
    return times


def graph(routes, pristine, working, call, samples):
    captures = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name, route in routes.items():
            working.detach().copy_(pristine)
            call(route)
            stream.synchronize()
            start, stop = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                working.detach().copy_(pristine)
                start.record()
                result = call(route)
                stop.record()
            captures[name] = (capture, start, stop, result)
        times = {name: [] for name in routes}
        names = list(routes)
        for cycle in range(samples + 1):
            shift = cycle % len(names)
            for name in names[shift:] + names[:shift]:
                capture, start, stop, _ = captures[name]
                capture.replay()
                stop.synchronize()
                if cycle:
                    times[name].append(start.elapsed_time(stop))
    return times


def native(path, logits, labels, scales):
    launcher = ctypes.CDLL(str(path)).centered_ce_launch_folded_v1
    pointer = ctypes.c_void_p
    launcher.argtypes = [pointer] * 4 + [ctypes.c_int] * 3 + [pointer]
    launcher.restype = ctypes.c_int
    loss = torch.empty(labels.shape, device=logits.device, dtype=torch.float32)
    status = launcher(logits.data_ptr(), labels.data_ptr(), loss.data_ptr(), scales.data_ptr(),
                      labels.shape[1], logits.shape[-1], scales.numel(),
                      torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f"{path} returned {status}")
    return loss


def runtime_parity(control, candidate):
    generator = torch.Generator(device="cuda").manual_seed(82)
    results = []
    # Both storage alignments, all cluster tiers and changing signed/zero scales.
    for vocabulary, heads, rows, offset in (
        (7, 1, 7, 0), (24, 3, 15, 1), (16376, 5, 11, 0), (16385, 4, 9, 0),
        (32768, 2, 53, 0), (32769, 3, 15, 0), (65537, 4, 9, 0),
        (123456, 2, 13, 0), (131072, 7, 19, 0),
    ):
        shape = (heads, rows, vocabulary)
        pristine = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        pristine[0, 0, :] = -0.0
        pristine[0, 0, -1] = 0.0
        pristine[0, 2, :] = -20.0
        pristine[0, 2, -1] = 20.0
        labels = torch.randint(vocabulary, (heads, rows), generator=generator, device="cuda")
        labels[:, :3] = torch.tensor([0, -100, vocabulary - 1], device="cuda")
        scales = torch.linspace(-0.025, 0.035, heads, device="cuda")
        scales[0] = 0.0

        def set_up():
            logits = torch.empty(pristine.numel() + offset, device="cuda", dtype=pristine.dtype)[offset:]
            return logits.view_as(pristine).copy_(pristine)

        a, b = set_up(), set_up()
        base_loss = native(control, a, labels, scales)
        got_loss = native(candidate, b, labels, scales)
        row_diff = bit_differences(got_loss, base_loss)
        dz_diff = bit_differences(b.flatten(0, 1), a.flatten(0, 1))
        results.append(dict(shape=shape, offset=offset, row_loss_differences=row_diff,
                            dz_differences=dz_diff))
    return results


def wrapper_edge(control, candidate):
    generator = torch.Generator(device="cuda").manual_seed(17)
    heads, rows, vocabulary = 11, 13, 16385
    shape = (heads, rows, vocabulary)
    pristine = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(vocabulary, (heads, rows), generator=generator, device="cuda")
    labels[:, :3] = torch.tensor([0, -100, vocabulary - 1], device="cuda")
    cases = (
        (tuple((head - 5) / 7 for head in range(heads)), 2387.0),
        (tuple((head - 5) / 7 for head in range(heads)), 871.0),
        (tuple((4 - head) / 11 for head in range(heads)), 871.0),
    )

    def invoke(route, weights, denominator):
        storage = torch.empty(pristine.numel() + 1, device="cuda", dtype=pristine.dtype)
        logits = storage[1:].view_as(pristine).copy_(pristine).requires_grad_()
        assert logits.data_ptr() % 16 == 2
        sums = route(logits, labels, weights, denominator)
        weighted = sum((term / denominator) * weight for term, weight in zip(sums, weights, strict=True))
        (dz,) = torch.autograd.grad(weighted, logits)
        assert dz.data_ptr() == logits.data_ptr() and route.scales_checked
        return torch.stack((*sums, weighted)), dz.flatten(0, 1)

    results = []
    for weights, denominator in cases:
        a, da = invoke(control, weights, denominator)
        b, db = invoke(candidate, weights, denominator)
        results.append(dict(shape=shape, storage_offset_bf16=1, weights=weights, denominator=denominator,
                            final_loss_differences=bit_differences(a, b),
                            dz_differences=bit_differences(da, db)))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--chunks", type=int, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidate", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--wrapper-edge", action="store_true")
    args = p.parse_args()
    args.part = args.chunks - 1
    args.extra_fixture = None
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._inductor.config.triton.cudagraphs = False
    pristine, labels, weights, denominator, fixture = project_chunk(args)
    working = torch.empty_like(pristine, requires_grad=True)
    paths = {"control": args.control.resolve(), "control-repeat": args.control.resolve()}
    paths.update((path.stem, path.resolve()) for path in args.candidate)
    hashes = {name: digest(path) for name, path in paths.items()}
    routes = {name: ExactFoldedCrossEntropy(str(path), library_sha256=hashes[name])
              for name, path in paths.items()}

    def call(route):
        return evaluate(route, working, labels, weights, denominator)

    samples = eager(routes, pristine, working, call, args.warmup, args.samples)
    device_spans = graph(routes, pristine, working, call, args.samples) if args.graph else {}
    assert all(route.scales_checked for route in routes.values())

    def output(route):
        working.detach().copy_(pristine)
        *losses, gradient = call(route)
        return torch.stack(losses), gradient.flatten(0, 1).clone()

    control_loss, control_dz = output(routes["control"])
    loss_scales = torch.tensor(weights, device=pristine.device, dtype=torch.float32) / denominator
    working.detach().copy_(pristine)
    shaped_labels = labels.reshape(len(weights), -1)
    control_rows = native(paths["control"], working.detach(), shaped_labels, loss_scales)
    results = {}
    for name, route in routes.items():
        losses, dz = output(route)
        working.detach().copy_(pristine)
        rows = native(paths[name], working.detach(), shaped_labels, loss_scales)
        results[name] = dict(
            path=str(paths[name]), sha256=hashes[name], samples=samples[name],
            graph_elapsed=elapsed_series(device_spans[name]) if name in device_spans else None,
            medians={metric: statistics.median(sample[metric] for sample in samples[name])
                     for metric in ("wall_ms", "submit_ms", "wait_ms", "event_ms")},
            gift_total_peak_gib=max(sample["total_gib"] for sample in samples[name]),
            final_loss_differences=bit_differences(losses, control_loss),
            dz_differences=bit_differences(dz, control_dz),
            raw_row_loss_differences=bit_differences(rows, control_rows),
            runtime_parity=runtime_parity(paths["control"], paths[name]) if name not in
            ("control", "control-repeat") else [],
        )
    if args.wrapper_edge:
        for name, route in routes.items():
            if name not in ("control", "control-repeat"):
                results[name]["wrapper_edge"] = wrapper_edge(routes["control"], route)
    output = dict(fixture=fixture, gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  torch_version=torch.__version__, warmup=args.warmup,
                  timing_scope="Restoration and projection excluded. Eager wall, Python submission, "
                               "final wait and event spans contain consuming CE, separate FP32 head "
                               "sums, normalized loss and BF16 dZ autograd. Graph spans use the same "
                               "operation on a single stream and exclude Python submission.",
                  results=results)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
