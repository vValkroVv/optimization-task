"""Replay one final C8 CE chunk from a B1 length-grid fixture."""

import argparse
import os
from pathlib import Path
import socket
import statistics
import sys

import torch


GRID = Path(__file__).resolve().parents[2]
ROOT = GRID.parent
sys.path[:0] = [str(ROOT / "tuning/ce"), str(GRID), str(ROOT)]

from bench import bit_differences, digest, eager, graph, native
from ce_bench import elapsed_series, evaluate, tensor_sha256
from matrix_fixture import VOCAB, load_inputs
from replay import IGNORE_INDEX, resize_inputs, write_json
from torchtitan.components.loss import MTPLoss
from torchtitan.ops import exact_folded_ce


CHUNKS = 8
LENGTHS = (32768, 65536, 131072, 262144)
LIBRARIES = {
    "tuned": ("tuning/ce/centered-tuned.so", "d3343cd518a05a6d5bea58b0493fa33373e040f1ce3dd5864b14c03033528929"),
    "cuda13": ("native-ce/centered-folded-v1.so", "17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8"),
    "threads512": ("tuning/ce/centered-tune-11.so", "2bf8cb6954ba978288700c82d0bbf79e1315ea7f58b35129e889afa276babae1"),
    "unrolled": ("tuning/ce/centered-tune-67.so", "931ccc9d71f36b88418aea7cb42efd52367206e3eb0cbebfcdbcd3104b99037e"),
}


@torch.no_grad()
def project_final_chunk(model, seq):
    fixture = ROOT / "inputs" / f"{model}.pt"
    record = load_inputs(model, fixture)
    # Only CE operands are needed. The hidden z-loss has no logits VJP.
    record.pop("z_hidden", None)
    record, valid = resize_inputs(record, 1, seq, torch.device("cuda"))
    target, positions = record["labels"], record["positions"]
    shifted = MTPLoss._shift_labels(target, 1, positions)
    interval = slice(-seq // CHUNKS, None)
    labels = torch.cat((target, shifted))[:, interval].contiguous()
    hidden = record.pop("hidden").flatten(0, 1)[:, interval].contiguous()
    weight = record["weight"].cuda()
    pristine = torch.nn.functional.linear(hidden, weight)
    if (pristine.shape != (2, seq // CHUNKS, VOCAB) or pristine.dtype != torch.bfloat16
            or not pristine.is_contiguous()):
        raise RuntimeError(f"Unexpected projected chunk: {pristine.shape}, {pristine.dtype}")
    receipt = dict(
        path=str(fixture), file_bytes=fixture.stat().st_size, provenance=record["provenance"],
        model=model, batch=1, seq=seq, chunks=CHUNKS, chunk_index=CHUNKS - 1,
        folded_rows=2 * seq // CHUNKS, logits_shape=list(pristine.shape), logits_dtype=str(pristine.dtype),
        logits_bytes=pristine.numel() * pristine.element_size(), input_width=hidden.size(-1),
        valid_sequence_rows_by_head=[valid, int(shifted.ne(IGNORE_INDEX).sum())],
        valid_chunk_rows_by_head=labels.ne(IGNORE_INDEX).sum(dim=1).tolist(),
        head_weights=record["head_weights"], global_denominator=record["denominator"],
        label_sha256=tensor_sha256(labels),
        projected_vocab_stride_4093_sample_sha256=tensor_sha256(pristine[..., ::4093]),
    )
    return pristine, labels.cuda(), tuple(record["head_weights"]), record["denominator"], receipt


def verify(paths, routes, pristine, working, labels, weights, denominator, call):
    scales = torch.tensor(weights, device=working.device, dtype=torch.float32) / denominator

    def invoke(name):
        working.detach().copy_(pristine)
        *sums, dz = call(routes[name])
        if dz.data_ptr() != working.data_ptr():
            raise RuntimeError(f"{name} returned a non-aliasing gradient")
        return torch.stack(sums), dz.flatten(0, 1)

    def row_losses(name):
        working.detach().copy_(pristine)
        return native(paths[name], working.detach(), labels, scales)

    control_sums, dz = invoke("tuned")
    # Retain a single untimed full reference. The comparison allocates just one
    # 128-row temporary mask at a time and checks the entire BF16 representation.
    control_dz = dz.clone()
    control_rows = row_losses("tuned")
    comparisons = {}
    for name in routes:
        sums, dz = invoke(name)
        dz_diff = bit_differences(dz, control_dz)
        comparisons[name] = dict(
            raw_row_loss_differences=bit_differences(row_losses(name), control_rows),
            final_loss_differences=bit_differences(sums, control_sums),
            dz_differences=dz_diff,
            checked_dz_elements=control_dz.numel(),
            checked_row_losses=control_rows.numel(),
            sums_and_weighted_loss=sums.tolist(),
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("100b", "700b"), required=True)
    parser.add_argument("--seq", type=int, choices=LENGTHS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 2:
        parser.error("Expected at least one warmup and two samples")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch._inductor.config.triton.cudagraphs = False
    pristine, labels, weights, denominator, fixture = project_final_chunk(args.model, args.seq)
    working = torch.empty_like(pristine, requires_grad=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"{args.model} S={args.seq} {list(pristine.shape)}", flush=True)

    paths = {"tuned": ROOT / LIBRARIES["tuned"][0], "tuned-repeat": ROOT / LIBRARIES["tuned"][0]}
    paths.update((name, ROOT / lib[0]) for name, lib in LIBRARIES.items() if name != "tuned")
    hashes = {name: digest(path) for name, path in paths.items()}
    for name, (_, expected) in LIBRARIES.items():
        if hashes[name] != expected:
            raise RuntimeError(f"Unexpected digest for {paths[name]}: {hashes[name]}")
    routes = {name: exact_folded_ce.ExactFoldedCrossEntropy(str(path), library_sha256=hashes[name])
              for name, path in paths.items()}

    def call(route):
        return evaluate(route, working, labels, weights, denominator)

    samples = eager(routes, pristine, working, call, args.warmup, args.samples)
    print(f"{args.model} S={args.seq} completed eager timing", flush=True)
    device_spans = graph(routes, pristine, working, call, args.samples)
    if not all(route.scales_checked for route in routes.values()):
        raise RuntimeError("The compiled CE did not verify the prepared VJP scales")
    print(f"{args.model} S={args.seq} completed graph timing", flush=True)
    comparisons = verify(paths, routes, pristine, working, labels, weights, denominator, call)
    results = {}
    for name in routes:
        results[name] = dict(
            path=str(paths[name]), sha256=hashes[name], samples=samples[name],
            graph_elapsed=elapsed_series(device_spans[name]),
            medians={metric: statistics.median(sample[metric] for sample in samples[name])
                     for metric in ("wall_ms", "submit_ms", "wait_ms", "event_ms")},
            gift_total_peak_gib=max(sample["total_gib"] for sample in samples[name]),
            **comparisons[name],
        )

    source_paths = (Path(__file__).resolve(), ROOT / "tuning/ce/bench.py", ROOT / "ce_bench.py",
                    GRID / "replay.py", ROOT / "matrix_fixture.py", Path(exact_folded_ce.__file__))
    output = dict(
        fixture=fixture, host=socket.gethostname(), gpu_name=torch.cuda.get_device_name(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        gpu_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_version=str(torch.__version__),
        torch_revision=torch.version.git_version, torch_cuda=torch.version.cuda, warmup=args.warmup,
        samples=args.samples, arm_order=list(routes), sources_sha256={str(p): digest(p) for p in source_paths},
        timing_scope="Single stream from restored Z through consuming CE, two separate FP32 head sums, "
                     "weighted normalization and autograd returning the in-place BF16 dZ. Projection and "
                     "restoration are excluded from all timed spans. Graph timings omit host submission.",
        memory_scope="Maximum eager GIFT total. The pristine and working logits stay live; projections, "
                     "graph pools and the untimed parity reference are outside the measured window.",
        results=results,
    )
    write_json(args.output, output)
    print(f"{args.model} S={args.seq} wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
