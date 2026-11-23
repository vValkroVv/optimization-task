"""Replay only the ordered owner-side reduction after the BF16 exchange."""

import argparse
from datetime import timedelta
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import runpy
from statistics import median
import time

import torch
from torch import distributed as dist

from row_owner_route_numeric import PASSES, SOURCE_ROOT, WORLD, check_runtime


KERNEL_REVISION = "3860b1eb5625d5bf9aba72d00a8961d6ca755ce9"
KERNEL_SHA256 = "a8474c8da675e48e6a36d6fbbe566315c8a3420a11b30f18576ee4edba43fc1c"
TRITON_BUILD = "3.8.0+git3f6e4113"
ROWS, WIDTH, SEED = 16032, 4096, 2048
WARMUP, SAMPLES = 10, 10
ARMS = ("A1", "B1", "A2", "B2")
STOCK_WORDS = {
    "ordered_cancellation": "3f800000",
    "negative_zero_plus_negative_zero": "80000000",
    "positive_bf16_subnormal": "00000000",
    "negative_bf16_subnormal": "00000000",
}


def load_kernel(path):
    if sha256(path.read_bytes()).hexdigest() != KERNEL_SHA256:
        raise ValueError(
            "archived Triton kernel source does not match the pinned Git blob"
        )
    if version("triton") != TRITON_BUILD:
        raise RuntimeError(f"expected the pinned Triton wheel {TRITON_BUILD}")
    import owner_add_ftz_kernel

    candidate = Path(owner_add_ftz_kernel.__file__)
    return owner_add_ftz_kernel.owner_add_bf16_, {
        "archived_reference": {"revision": KERNEL_REVISION, "sha256": KERNEL_SHA256},
        "candidate": {
            "file": candidate.name,
            "sha256": sha256(candidate.read_bytes()).hexdigest(),
            "fp32_add": "add.rn.ftz.f32",
        },
    }


def digest(tensor):
    return sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    ).hexdigest()


def route(unique):
    """Sorted signed IDs make each owner's BF16 rows a contiguous split."""
    if unique.numel() and not (0 <= unique[0] <= unique[-1] < ROWS * WORLD):
        raise ValueError("captured IDs are outside the production embedding table")
    boundaries = torch.searchsorted(unique, torch.arange(WORLD + 1) * ROWS)
    return tuple(boundaries.diff().tolist())


def expected_received(uniques, rank):
    donor_rows = [ids[(ids // ROWS) == rank] - rank * ROWS for ids in uniques]
    return tuple(len(rows) for rows in donor_rows), torch.cat(donor_rows)


def wire_fixture(input_root, rank, device):
    reference = json.loads(
        Path(__file__).with_name("embedding-replay-result.json").read_text()
    )
    load = runpy.run_path(
        str(SOURCE_ROOT / "runs/production-data-abcd/component_fixtures.py")
    )["load_fixture"]
    capture, receipt = load(input_root, rank)
    if receipt["input_ids_int64_sha256"] != reference["input_ids_sha256_by_rank"][rank]:
        raise ValueError("captured input differs from the accepted replay fixture")
    unique = torch.unique(capture.pop("input_ids"), sorted=True)
    del capture
    send = route(unique)
    all_ids = [None] * WORLD
    dist.all_gather_object(all_ids, unique.tolist())
    receive, expected = expected_received(
        [torch.tensor(ids, dtype=torch.int64) for ids in all_ids], rank
    )
    actual = torch.empty(sum(receive), dtype=torch.int64, device=device)
    dist.all_to_all_single(actual, unique.to(device), list(receive), list(send))
    actual.sub_(rank * ROWS)
    if not torch.equal(actual.cpu(), expected):
        raise RuntimeError("received IDs differ from sorted owner slices in rank order")
    payloads, wire_receipt = [], []
    for seed in (SEED + rank, SEED + WORLD + rank):
        generator = torch.Generator(device=device).manual_seed(seed)
        outgoing = torch.randn(
            (len(unique), WIDTH),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        incoming = outgoing.new_empty((sum(receive), WIDTH))
        dist.all_to_all_single(incoming, outgoing, list(receive), list(send))
        payloads.append(incoming)
        wire_receipt.append(
            {
                "seed": seed,
                "sent_bf16_sha256": digest(outgoing),
                "received_bf16_sha256": digest(incoming),
            }
        )
    cursor = 0
    for count in receive:
        donor_rows = expected[cursor : cursor + count]
        if not bool((donor_rows[1:] > donor_rows[:-1]).all()):
            raise RuntimeError("a donor slice must be strictly increasing")
        cursor += count
    return (
        actual,
        payloads,
        receive,
        {
            "input_ids_int64_sha256": receipt["input_ids_int64_sha256"],
            "tensors_sha256": receipt["tensors_sha256"],
            "send_split_rows": send,
            "receive_split_rows": receive,
            "received_local_rows_int64_sha256": digest(expected),
            "received_distinct_owner_rows": expected.unique().numel(),
            "bf16_wire": wire_receipt,
        },
    )


def reduce_donors(destination, local_rows, received, counts, scratch, fused):
    offset = 0
    for count in counts:
        if count:
            rows = local_rows[offset : offset + count]
            values = received[offset : offset + count]
            if fused is None:
                widened = scratch[:count]
                widened.copy_(values)
                destination.index_add_(0, rows, widened)
            else:
                fused(destination, rows, values)
        offset += count


def adversarial(device, fused):
    counts = (2, 0, 1, 1, 2, 0, 0, 1)
    rows = torch.tensor([0, 2, 0, 0, 0, 1, 2], dtype=torch.int64)
    payload = torch.zeros((len(rows), WIDTH), dtype=torch.bfloat16)
    initial = torch.zeros((4, WIDTH), dtype=torch.float32)
    payload_words = payload.view(torch.uint16)
    initial_words = initial.view(torch.uint32)
    payload[[0, 2, 3, 4], 0] = payload.new_tensor([2**24, 1, -(2**24), 1])
    payload_words[5, 1], initial_words[1, 1] = 0x8000, 0x80000000
    payload_words[0, 2], payload_words[0, 3] = 0x0001, 0x8001
    probes = {
        "ordered_cancellation": (0, 0),
        "negative_zero_plus_negative_zero": (1, 1),
        "positive_bf16_subnormal": (0, 2),
        "negative_bf16_subnormal": (0, 3),
    }
    # Row 1 receives just one donor row. Seed both operands by bits so copies
    # into the fixture preserve zeros, subnormals and NaN payloads.
    single_donor = {
        "negative_zero_plus_positive_zero": (0x80000000, 0x0000),
        "positive_zero_plus_negative_zero": (0x00000000, 0x8000),
        "negative_zero_plus_negative_bf16_subnormal": (0x80000000, 0x8001),
        "positive_fp32_subnormal_input": (0x00000001, 0x0000),
        "negative_fp32_subnormal_input": (0x80000001, 0x8000),
        "negative_bf16_subnormal_against_normal": (0x00800000, 0x807F),
        "positive_bf16_subnormal_against_normal": (0x80800000, 0x007F),
        "positive_fp32_result_underflow": (0x00800001, 0x8080),
        "negative_fp32_result_underflow": (0x80800001, 0x0080),
        "positive_infinity": (0x3F800000, 0x7F80),
        "negative_infinity": (0xBF800000, 0xFF80),
        "opposite_infinities": (0x7F800000, 0xFF80),
        "positive_bf16_quiet_nan": (0x3F800000, 0x7FC1),
        "negative_bf16_quiet_nan": (0x3F800000, 0xFFC7),
        "bf16_signaling_nan": (0x3F800000, 0x7F81),
        "fp32_nan_payload": (0x7FC12345, 0x3F80),
        "two_nan_payloads": (0xFFC12345, 0x7FC1),
    }
    for column, (name, (start_word, donor_word)) in enumerate(
        single_donor.items(), start=4
    ):
        initial_words[1, column] = start_word
        payload_words[5, column] = donor_word
        probes[name] = (1, column)

    rows, payload, initial = rows.to(device), payload.to(device), initial.to(device)
    scratch = initial.new_empty((max(counts), WIDTH))
    results = {}
    for label, kernel in (("A", None), ("B", fused)):
        out = initial.clone()
        reduce_donors(out, rows, payload, counts, scratch, kernel)
        results[label] = out.view(torch.int32).cpu()
    mismatches = (results["A"] != results["B"]).nonzero()

    def words_at(row, column):
        return {
            "row": row,
            "column": column,
            **{
                arm: f"{int(words[row, column]) & 0xFFFFFFFF:08x}"
                for arm, words in results.items()
            },
        }

    return {
        "identical_f32_words": len(mismatches) == 0,
        "probes": {
            name: words_at(row, column) for name, (row, column) in probes.items()
        },
        "mismatch_count": len(mismatches),
        "first_mismatches": [
            words_at(row, column) for row, column in mismatches[:16].tolist()
        ],
    }


def gradient_receipts(rows, inputs, counts, scratch, fused):
    parameter = torch.empty(
        (ROWS, WIDTH), dtype=torch.float32, device=rows.device, requires_grad=True
    )
    records = {}
    for case, incoming in zip(PASSES, (inputs[0], inputs[1], inputs[0]), strict=True):
        if case == "reset":
            parameter.grad = None
        gradient = torch.zeros_like(parameter)
        reduce_donors(gradient, rows, incoming, counts, scratch, fused)
        # An explicit leaf gradient uses ordinary autograd AccumulateGrad.
        torch.autograd.backward((parameter,), (gradient,))
        records[case] = digest(parameter.grad)
        del gradient
    return records


def measure(target, rows, received, counts, scratch, fused):
    for _ in range(WARMUP):
        target.zero_()
        reduce_donors(target, rows, received, counts, scratch, fused)
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    samples = []
    for _ in range(SAMPLES):
        target.zero_()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        begin = time.perf_counter()
        start.record()
        reduce_donors(target, rows, received, counts, scratch, fused)
        end.record()
        end.synchronize()
        samples.append(
            {
                "synchronized_window_ms": (time.perf_counter() - begin) * 1000,
                "cuda_event_span_ms": start.elapsed_time(end),
                "allocated_before_bytes": allocated,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated()
                - allocated,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    return samples


def save(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def verify_parity(receipts):
    for receipt in receipts:
        a, b = [receipt["gradient_sha256"][arm] for arm in "AB"]
        adversarial_result = receipt["adversarial"]
        observed = {
            name: adversarial_result["probes"][name]["A"] for name in STOCK_WORDS
        }
        if observed != STOCK_WORDS:
            raise RuntimeError(
                "stock CUDA index_add changed adversarial words; see parity.json"
            )
        if (
            a != b
            or a["initial"] != a["reset"]
            or a["initial"] == a["accumulated"]
            or not adversarial_result["identical_f32_words"]
            or not receipt["payload_unchanged"]
        ):
            raise RuntimeError("owner-add changed gradient bits; see parity.json")


def replay(args, rank, device, fused, kernel_receipt):
    props = torch.cuda.get_device_properties(device)
    if (
        props.multi_processor_count != 132
        or torch.are_deterministic_algorithms_enabled()
    ):
        raise RuntimeError(
            "expected a 132-SM H100 and Original A's native index_add dispatch"
        )
    rows, payloads, counts, fixture = wire_fixture(args.fixture, rank, device)
    # One fixed stand-in for the existing borrowed FP32 view. Both arms share
    # its residency, but only A reads or writes it. No head bucket is built.
    scratch = torch.empty((max(counts), WIDTH), dtype=torch.float32, device=device)
    record = {
        "rank": rank,
        "fixture": fixture,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "triton": version("triton"),
            "gpu": props.name,
            "sm_count": props.multi_processor_count,
            "deterministic_algorithms": False,
        },
        "kernel": kernel_receipt,
        "adversarial": adversarial(device, fused),
        "gradient_sha256": {
            "A": gradient_receipts(rows, payloads, counts, scratch, None),
            "B": gradient_receipts(rows, payloads, counts, scratch, fused),
        },
        "nonempty_donors": sum(count > 0 for count in counts),
        "scratch_bytes_by_arm": {"A": scratch.nbytes, "B": 0},
        "resident_scratch_stand_in_bytes": scratch.nbytes,
    }
    record["payload_unchanged"] = all(
        digest(payload) == item["received_bf16_sha256"]
        for payload, item in zip(payloads, fixture["bf16_wire"], strict=True)
    )
    parity = [None] * WORLD
    dist.all_gather_object(parity, record)
    if rank == 0:
        save(args.output / "parity.json", {"ranks": parity})
    verify_parity(parity)

    del payloads[1:]
    torch.cuda.empty_cache()
    target = torch.empty((ROWS, WIDTH), dtype=torch.float32, device=device)
    record["arms"] = {
        arm: measure(
            target,
            rows,
            payloads[0],
            counts,
            scratch,
            fused if arm.startswith("B") else None,
        )
        for arm in ARMS
    }
    save(args.output / f"rank{rank}.json", record)
    all_ranks = [None] * WORLD
    dist.all_gather_object(all_ranks, record["arms"])
    if rank == 0:
        summary = {}
        for arm in ARMS:
            records = [row[arm] for row in all_ranks]
            summary[arm] = {}
            for metric in records[0][0]:
                series = [
                    max(sample[metric] for sample in group)
                    for group in zip(*records, strict=True)
                ]
                summary[arm][metric] = {
                    "rank_max_samples": series,
                    "median": median(series),
                    "max": max(series),
                }
        comparisons = {}
        for metric in ("synchronized_window_ms", "cuda_event_span_ms"):
            spans = {arm: summary[arm][metric]["median"] for arm in ARMS}
            comparisons[metric] = {
                f"{test}_over_{control}": spans[test] / spans[control]
                for control, test in (("A1", "B1"), ("A2", "B2"), ("A1", "A2"))
            }
        save(
            args.output / "summary.json",
            {
                "scope": "owner reduction only; excludes initialization, NCCL, source coalescing, autograd and full training",
                "all_rank_cuda_fp32_gradient_bytes_identical": True,
                "all_rank_adversarial_f32_words_identical": True,
                "kernel": kernel_receipt,
                "warmup_and_samples_per_arm": [WARMUP, SAMPLES],
                "serial_arm_order": ARMS,
                "scratch_is_resident_in_both_arms": True,
                "arms": summary,
                "median_ratios": comparisons,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--archive-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = check_runtime()
    fused, kernel_receipt = load_kernel(args.archive_kernel)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        replay(args, rank, device, fused, kernel_receipt)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
