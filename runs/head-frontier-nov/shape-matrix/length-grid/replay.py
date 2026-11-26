"""Replay a shape-matrix LM head with a different local batch and sequence."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "tuning"
IGNORE_INDEX = -100


def resize_hidden(source, batch, seq, device):
    """Fill the destination from seed rows without a host-sized replica."""
    base_batch, base_seq, width = source.shape[-3:]
    if (batch, seq) == (base_batch, base_seq):
        return source.to(device)

    result = torch.empty((*source.shape[:-3], batch, seq, width), dtype=source.dtype, device=device)
    tile_seq = min(seq, base_seq)
    full_tiles, tail = divmod(seq, tile_seq)
    full_seq = full_tiles * tile_seq
    for row in range(min(batch, base_batch)):
        seed = source[..., row, :tile_seq, :].to(device)
        dest = result[..., row::base_batch, :, :]
        dest[..., :full_seq, :].unflatten(-2, (full_tiles, tile_seq)).copy_(
            seed.unsqueeze(-3).unsqueeze(-3)
        )
        if tail:
            dest[..., full_seq:, :].copy_(seed[..., :tail, :].unsqueeze(-3))
    return result


def resize_inputs(record, batch, seq, device):
    base_labels, base_positions = record["labels"], record["positions"]
    base_batch, base_seq = base_labels.shape
    base_valid = int(base_labels.ne(IGNORE_INDEX).sum())
    if base_valid == 0 or record.get("ignore_index", IGNORE_INDEX) != IGNORE_INDEX:
        raise ValueError("Expected supervised targets with the replay's ignore index")
    if not base_labels[:, -1].eq(IGNORE_INDEX).all() or not base_positions[:, 0].eq(0).all():
        raise ValueError("Expected each base row to start a packed sequence and end with an ignored label")
    denominator_scale = float(record["denominator"]) / base_valid
    if denominator_scale <= 0 or not denominator_scale.is_integer():
        raise ValueError("Expected a positive integer fixture denominator multiplier")

    rows = torch.arange(batch) % base_batch
    tokens = torch.arange(seq) % base_seq
    labels = base_labels[rows[:, None], tokens]
    positions = base_positions[rows[:, None], tokens]
    # Original row ends already mask every full tile boundary.
    labels[:, -1] = IGNORE_INDEX
    valid_tokens = int(labels.ne(IGNORE_INDEX).sum())
    if valid_tokens == 0:
        raise ValueError("The resized fixture has no supervised targets")

    record = dict(record, labels=labels, positions=positions, denominator=valid_tokens * denominator_scale)
    record["hidden"] = resize_hidden(record["hidden"], batch, seq, device)
    if "z_hidden" in record:
        record["z_hidden"] = resize_hidden(record["z_hidden"], batch, seq, device)
    return record, valid_tokens


def check_environment(args):
    candidate = args.variant == "cuda13"
    expected = {
        "GIFT_DIRECT_BF16_DHIDDEN": str(int(candidate)),
        "GIFT_FUSED_HIDDEN_Z_LOSS": str(int(candidate and args.model == "100b")),
        "GIFT_STABLE_CE_FOLDED_BACKWARD": str(int(candidate)),
        "GIFT_STAGED_HEAD_DW": "0",
        "GIFT_EXPERIMENT_LM_HEAD_MATRIX": "generic" if candidate else "",
        "GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS": "",
    }
    for key, value in expected.items():
        if os.environ.get(key, "") != value:
            raise ValueError(f"{args.variant} requires {key}={value!r}")
    library = os.environ.get("GIFT_EXACT_FOLDED_CE_LIBRARY")
    digest = os.environ.get("GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256")
    if bool(library) != candidate or bool(digest) != candidate:
        raise ValueError(f"Unexpected exact CE library configuration for {args.variant}")
    if os.environ.get("HEAD_REPLAY_DHIDDEN_DIGEST") == "1":
        raise ValueError("The length replay must not export the hidden-gradient tensor")
    return digest


def install_candidate(writers):
    from head_run import install_linear_ops
    from torchtitan.distributed.gfsdp import _staged_head_dw as staged
    from tuning.lt.lt_writer import make_dw_writer

    catalog = TUNING / "lt/results/calibrated-catalog.json"

    def factory(*unused):
        writer = make_dw_writer(policy="replay", receipt=catalog)
        writers.append(writer)
        return writer

    staged.make_head_dw_writer = factory
    return install_linear_ops(["linear"], TUNING / "linear-plans.json")


def candidate_activation(args, output, counts, policy, writers, digest):
    # head_bench runs one probe, one allocator prime and one final clean backward.
    steps = args.warmup + args.samples + 3
    chunks = steps * args.chunks
    by_shape = {part: chunks - count for part, count in counts.items()}
    linear = policy.calls
    if any(count < 0 for count in by_shape.values()) or sum(linear.values()) != sum(counts.values()):
        raise RuntimeError(f"Inconsistent linear routes: {counts}, {linear}")
    routes = {
        "candidate": linear["candidate"],
        "fallback": sum(by_shape.values()) + linear["fallback"],
        "descriptor_fallback": linear["fallback"],
        "uncalibrated_shape": by_shape,
    }
    if len(writers) != 1:
        raise RuntimeError(f"Expected one native dW writer, got {len(writers)}")
    plan = writers[0].receipt()
    actual_shape = (output["rows"], output["vocab"], output["width"])
    if {tuple(entry["shape"]) for entry in plan["plans"]} != {actual_shape}:
        raise RuntimeError(f"Native dW did not receive the expected shape {actual_shape}")
    # The catalog counts tuned writes. The lease writes once per chunk, with
    # beta zero for the first chunk and beta one for the rest.
    totals = (steps, steps * (args.chunks - 1))
    native_calls = [
        total - selected for total, selected in zip(totals, plan["tuned_calls_beta0_beta1"], strict=True)
    ]
    if any(count < 0 for count in native_calls):
        raise RuntimeError(f"Inconsistent dW routes: {plan}")
    plan["native_generic_calls_beta0_beta1"] = native_calls
    return dict(ce=digest, ce_route="cuda13-consuming", operations=["ce", "dw", "linear"],
                counts=counts, linear_route=routes, plans=[plan])


def write_json(path, value):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value, allow_nan=False) + "\n")
    partial.replace(path)


def run(args):
    rank = int(os.environ["RANK"])
    args.out.mkdir(parents=True, exist_ok=True)
    phase = "setup"
    try:
        digest = check_environment(args)
        # The remote tuning directory holds the preserved timer used by head_run.py.
        sys.path[:0] = [str(TUNING), str(ROOT)]
        import head_bench
        from matrix_fixture import load_inputs

        dimensions = {}

        def load_resized_inputs(model, fixture):
            nonlocal phase
            phase = "fixture"
            from torchtitan.components.loss import MTPLoss

            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            original = load_inputs(model, fixture)
            base_batch, base_seq = original["labels"].shape
            record, valid = resize_inputs(original, args.batch, args.seq, device)
            labels, positions = record["labels"], record["positions"]
            mtp_labels = MTPLoss._shift_labels(labels, 1, positions)
            dimensions.update(
                batch=labels.size(0), seq=labels.size(1), physical_tokens=labels.numel(),
                valid_tokens=valid, valid_mtp_tokens=int(mtp_labels.ne(IGNORE_INDEX).sum()),
                denominator=record["denominator"], hidden_shape=list(record["hidden"].shape),
                hidden_dtype=str(record["hidden"].dtype), weight_dtype=str(record["weight"].dtype),
                label_dtype=str(labels.dtype), position_dtype=str(positions.dtype),
                fixture_mapping=dict(base_seq=base_seq, rows=list(range(min(args.batch, base_batch)))),
            )
            if "z_hidden" in record:
                dimensions["z_hidden_shape"] = list(record["z_hidden"].shape)
            phase = "head-replay"
            return record

        writers = []
        if digest:
            counts, policy = install_candidate(writers)
        head_bench.load_inputs = load_resized_inputs
        args.reference = None
        phase = "head-replay"
        head_bench.run(args)

        phase = "receipts"
        path = args.out / f"rank-{rank}.json"
        output = json.loads(path.read_text())
        heads, batch, seq, width = dimensions["hidden_shape"]
        if (heads, batch, seq, width) != (2, args.batch, args.seq, output["width"]):
            raise RuntimeError(f"Unexpected replay tensor shape: {dimensions}")
        if output["rows"] != heads * batch * seq // args.chunks or len(output["measured"]) != args.samples:
            raise RuntimeError(f"Unexpected replay shape or sample count on rank {rank}")
        if not {"loss_ce", "loss_mtp"} <= output["components"].keys():
            raise RuntimeError(f"Missing head loss components on rank {rank}")
        if digest:
            if not output["bf16_direct"]:
                raise RuntimeError("The candidate did not activate the direct BF16 hidden gradient")
            activation = candidate_activation(args, output, counts, policy, writers, digest)
        else:
            if output["bf16_direct"] or any(output["staged_chunks"]):
                raise RuntimeError("The baseline activated a candidate gradient path")
            activation = dict(ce=None, ce_route="compiled-aten-per-head", operations=[], counts={},
                              linear_route={}, plans=[])
        write_json(path, dict(output, **dimensions))
        write_json(args.out / f"activation-{rank}.json", activation)
    except torch.OutOfMemoryError as error:
        if "cuda out of memory" in str(error).lower():
            write_json(args.out / f"oom-{rank}.json", dict(rank=rank, phase=phase, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("100b", "700b"), required=True)
    parser.add_argument("--variant", choices=("baseline", "cuda13"), required=True)
    for name in ("batch", "seq", "chunks", "warmup", "samples"):
        parser.add_argument(f"--{name}", type=int, required=True)
    for name in ("fixture", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.seq < 2 or args.chunks < 1 or args.seq % args.chunks:
        parser.error("batch must be positive and seq must be at least two and divisible by chunks")
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be nonnegative and samples must be positive")
    run(args)


if __name__ == "__main__":
    main()
