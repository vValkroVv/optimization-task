"""Confirm the general consuming CE choice on the remaining B1 chunk shapes."""

import argparse
import importlib.util
from pathlib import Path

import torch

from measure import GRID, ROOT, chunk_input, region
from ce_bench import evaluate, tensor_sha256
from torchtitan.ops import exact_folded_ce

spec = importlib.util.spec_from_file_location("ce_region_tuning", GRID / "region-tuning/ce/bench_long.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
from bench import bit_differences, digest, native


def graph_samples(calls, restore, rounds):
    captures = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name, call in calls.items():
            restore()
            call()
            stream.synchronize()
            start, stop = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                restore()
                start.record()
                captured = call()
                stop.record()
            captures[name] = (graph, start, stop, captured)
        times = {name: [] for name in captures}
        for i in range(rounds + 1):
            for name in region.groups_by_index(list(captures), i):
                graph, start, stop, _ = captures[name]
                graph.replay()
                stop.synchronize()
                if i:
                    times[name].append(start.elapsed_time(stop))
    return times


def measure(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._inductor.config.triton.cudagraphs = False
    hidden, weight, targets, fixture = chunk_input(args.model, args.seq, args.chunks,
                                                  ROOT / "inputs" / f"{args.model}.pt")
    with torch.no_grad():
        working = torch.nn.functional.linear(hidden, weight)
    output_bytes = working.numel() * working.element_size()
    # The largest C2 case cannot hold a pristine logits copy on device.
    # Restore it using an out GEMM before the start marker.
    use_saved_copy = torch.cuda.mem_get_info()[0] >= output_bytes + (1 << 30)
    pristine = working.clone() if use_saved_copy else None
    if use_saved_copy:
        del hidden, weight
    working.requires_grad_(True)

    def restore():
        buffer = working.detach()
        if pristine is not None:
            buffer.copy_(pristine)
        else:
            torch.mm(hidden.flatten(0, 1), weight.T, out=buffer.flatten(0, 1))

    fixture.update(model=args.model, chunks=args.chunks, sequence=args.seq,
                   logits_shape=list(working.shape), logits_bytes=output_bytes,
                   label_sha256=tensor_sha256(targets))
    weights, denominator = tuple(fixture["head_weights"]), fixture["denominator"]
    libs = {name: previous.LIBRARIES[name] for name in ("tuned", "cuda13", "unrolled")}
    libs = {"tuned": libs["tuned"], "tuned-repeat": libs["tuned"], **libs}
    paths = {name: ROOT / path for name, (path, _) in libs.items()}
    hashes = {name: digest(path) for name, path in paths.items()}
    assert all(hashes[name] == expected for name, (_, expected) in libs.items())
    routes = {name: exact_folded_ce.ExactFoldedCrossEntropy(str(path), library_sha256=hashes[name])
              for name, path in paths.items()}

    def call(route):
        result = evaluate(route, working, targets, weights, denominator)
        if result[-1].data_ptr() != working.data_ptr():
            raise RuntimeError("Expected consuming CE to alias its logits")
        return result

    def bind(route):
        return lambda: call(route)

    calls = {name: bind(route) for name, route in routes.items()}
    print(f"{args.model} C{args.chunks} S{args.seq} CE: {list(working.shape)}, "
          f"restoration {'copy' if use_saved_copy else 'projection'}", flush=True)
    timer = region.Timer()
    torch.cuda.empty_cache()
    samples = {name: [] for name in calls}
    for i in range(args.warmup + args.rounds):
        for name in region.groups_by_index(list(calls), i):
            restore()
            torch.cuda.synchronize()
            result, sample = timer.sample(calls[name])
            if i >= args.warmup:
                samples[name].append(sample)
            del result
    torch.cuda.empty_cache()
    print(f"{args.model} C{args.chunks} S{args.seq} CE: completed eager", flush=True)
    device = graph_samples(calls, restore, args.graph_rounds)
    scales = torch.tensor(weights, device=working.device, dtype=torch.float32) / denominator
    restore()
    *sums, dz = call(routes["tuned"])
    reference_sums = torch.stack(sums)
    ref_on_cpu = torch.cuda.mem_get_info()[0] < output_bytes + (1 << 30)
    reference_dz = dz.flatten(0, 1).cpu() if ref_on_cpu else dz.flatten(0, 1).clone()
    del dz

    def rows(name):
        restore()
        return native(paths[name], working.detach(), targets, scales)

    reference_rows = rows("tuned")
    results = {}
    for name, route in routes.items():
        restore()
        *sums, dz = call(route)
        # Compare the exact full BF16 representation in a bounded tile.
        from measure import bits_equal
        dz_equal = bits_equal(dz.flatten(0, 1), reference_dz)
        results[name] = dict(
            sha256=hashes[name], samples=samples[name], graph_ms=device[name],
            medians={key: region.median(s[key] for s in samples[name])
                     for key in ("wall_ms", "submit_ms", "wait_ms", "event_ms")},
            graph_median_ms=region.median(device[name]),
            gift_total_peak_gib=max(s["gift_total_peak_gib"] for s in samples[name]),
            full_dz_bitwise=dz_equal, checked_dz_elements=dz.numel(),
            raw_row_loss_differences=bit_differences(rows(name), reference_rows),
            checked_row_losses=reference_rows.numel(),
            final_loss_differences=bit_differences(torch.stack(sums), reference_sums),
            scales_checked=route.scales_checked,
        )
    source_paths = (Path(__file__).resolve(), Path(__file__).with_name("measure.py"),
                    GRID / "region-tuning/ce/bench_long.py", ROOT / "tuning/ce/bench.py",
                    ROOT / "ce_bench.py", Path(exact_folded_ce.__file__))
    region.save(args.out, dict(
        fixture=fixture, gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=torch.__version__, torch_git=torch.version.git_version, torch_cuda=torch.version.cuda,
        source_hashes={str(p): digest(p) for p in source_paths}, results=results,
        reference_dz_device=str(reference_dz.device),
        untimed_restore="copy of pristine logits" if use_saved_copy else "out GEMM into working logits",
        timing_scope="Single stream consuming CE with two FP32 head sums and weighted autograd "
        "returning aliased BF16 dZ. Input restore completes before the CUDA start event. "
        "Graph event spans exclude Python. Parity and restore are excluded from measurements.",
        memory_scope="GIFT total over eager CE. Includes the working logits and the pristine "
        "copy or live hidden and weight for the restore GEMM.",
    ))
    if any(not r["full_dz_bitwise"] or r["raw_row_loss_differences"] or r["final_loss_differences"]
           or not r["scales_checked"] for r in results.values()):
        raise RuntimeError("CE comparison or activation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("100b", "700b"), required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--chunks", type=int, choices=(2, 8), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--graph-rounds", type=int, default=9)
    measure(parser.parse_args())
