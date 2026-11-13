#!/usr/bin/env python3
"""Compare native and stable CE on a task-local, synthetic production-shape replay.

Run after the authorized GPU slot is idle:
    PYTHONDONTWRITEBYTECODE=1 python runs/four-fusion/ce-numerical-replay.py \
        --output runs/four-fusion/artifacts/raw/tasktmp/ce-numerical-replay.json

The inputs are seeded BF16 logits, not captured model values. This isolates
local CE forward/backward with the two MTP heads, native normalization order,
and the production two-chunk labels; it is not a full-model or convergence
check. It follows MTPLoss._chunk_loss scaling/order without changing the logits
producer. Reported differences are measurements, with no universal tolerance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "runs/four-fusion/artifacts/raw/tasktmp"
CACHE_ROOT = RAW_ROOT / "compile-cache"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_ROOT / "torchinductor")
os.environ["TRITON_CACHE_DIR"] = str(CACHE_ROOT / "triton")
sys.path.insert(0, str(ROOT / "gift"))

import torch  # noqa: E402
import triton  # noqa: E402

from torchtitan.components.loss import cross_entropy_loss as native_ce  # noqa: E402
from torchtitan.experiments.four_fusion.stable_ce import (  # noqa: E402
    cross_entropy_loss as stable_ce,
)

SEED = 2048
BATCH = 2
SEQUENCE = 4096
CHUNK_SEQUENCE = SEQUENCE // 2
HEADS = 2
VOCAB = 128256
IGNORE_INDEX = -100
GLOBAL_VALID_TOKENS = 2_096_640.0
MTP_WEIGHT = 0.3
ROW_TILE = 8


def _scalar_bits(value: torch.Tensor) -> int:
    return int(value.detach().view(torch.int32).cpu().item()) & 0xFFFFFFFF


def _ordered_bf16(value: torch.Tensor) -> torch.Tensor:
    bits = value.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where((bits & 0x8000) != 0, 0xFFFF - bits, bits + 0x8000)


def _ulp_bins(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    distance = (_ordered_bf16(left) - _ordered_bf16(right)).abs()
    return torch.stack(
        (
            (distance == 0).sum(),
            (distance == 1).sum(),
            (distance > 1).sum(),
        )
    )


def _run_variant(
    ce_fn: Any, folded_logits: torch.Tensor, folded_labels: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    if folded_logits.shape != (HEADS * BATCH, CHUNK_SEQUENCE, VOCAB):
        raise AssertionError(f"Unexpected folded logits shape: {folded_logits.shape}")
    if folded_logits.dtype != torch.bfloat16:
        raise AssertionError(f"Expected BF16 logits, got {folded_logits.dtype}")
    if (
        folded_labels.shape != (HEADS * BATCH, CHUNK_SEQUENCE)
        or folded_labels.dtype != torch.int64
    ):
        raise AssertionError("Expected folded int64 targets matching the logits rows")
    leaf = folded_logits.detach().requires_grad_(True)
    storage_ptr = leaf.untyped_storage().data_ptr()
    main_raw = ce_fn(leaf[:BATCH], folded_labels[:BATCH])
    mtp_raw = ce_fn(leaf[BATCH:], folded_labels[BATCH:])
    for raw in (main_raw, mtp_raw):
        if raw.ndim != 0 or raw.dtype != torch.float32:
            raise AssertionError(
                f"CE must return an FP32 scalar; got {raw.shape}/{raw.dtype}"
            )
        if not bool(torch.isfinite(raw).item()):
            raise AssertionError("CE returned a non-finite scalar loss")
        raw.retain_grad()

    # Keep the native _chunk_loss / MTPLoss operation order exactly: divide,
    # multiply by the head weight, then add the two heads before backward.
    main_weighted = (main_raw / GLOBAL_VALID_TOKENS) * 1.0
    mtp_weighted = (mtp_raw / GLOBAL_VALID_TOKENS) * MTP_WEIGHT
    chunk_loss = main_weighted + mtp_weighted
    if not bool(torch.isfinite(chunk_loss).item()):
        raise AssertionError("Weighted chunk loss is non-finite")
    chunk_loss.backward(torch.ones_like(chunk_loss))
    if (
        leaf.grad is None
        or leaf.grad.shape != leaf.shape
        or leaf.grad.dtype != torch.bfloat16
    ):
        raise AssertionError("CE backward did not return BF16 logits gradients")
    scales = (main_raw.grad, mtp_raw.grad)
    if any(scale is None or scale.dtype != torch.float32 for scale in scales):
        raise AssertionError(
            "Expected retained FP32 incoming gradients for both CE sums"
        )

    result = {
        "raw_loss": [float(main_raw.detach().item()), float(mtp_raw.detach().item())],
        "weighted_loss": [
            float(main_weighted.detach().item()),
            float(mtp_weighted.detach().item()),
        ],
        "chunk_loss": float(chunk_loss.detach().item()),
        "incoming_scale": [
            {
                "value": float(scale.detach().item()),
                "fp32_bits": f"0x{_scalar_bits(scale):08x}",
            }
            for scale in scales
        ],
        "storage_ptr": storage_ptr,
    }
    return leaf.grad.detach(), result, chunk_loss.detach()


def _row_location(row: int, chunk_index: int) -> dict[str, int | str] | None:
    if row < 0:
        return None
    rows_per_head = BATCH * CHUNK_SEQUENCE
    head = row // rows_per_head
    within_head = row % rows_per_head
    return {
        "head": "main" if head == 0 else "mtp1",
        "batch": within_head // CHUNK_SEQUENCE,
        "sequence": chunk_index * CHUNK_SEQUENCE + within_head % CHUNK_SEQUENCE,
    }


def _gradient_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    labels: torch.Tensor,
    chunk_index: int,
) -> dict[str, Any]:
    if (
        baseline.shape != candidate.shape
        or baseline.dtype != torch.bfloat16
        or candidate.dtype != torch.bfloat16
    ):
        raise AssertionError("Baseline/candidate BF16 gradient shape or dtype changed")
    vocab = baseline.shape[-1]
    rows = labels.numel()
    baseline = baseline.reshape(-1, vocab)
    candidate = candidate.reshape(-1, vocab)
    if baseline.shape[0] != rows:
        raise AssertionError("Gradient rows do not match folded labels")
    labels = labels.reshape(-1)
    device = baseline.device
    bins = torch.zeros(3, dtype=torch.int64, device=device)
    rel_sq = torch.zeros((), dtype=torch.float64, device=device)
    ref_sq = torch.zeros_like(rel_sq)
    max_abs = torch.zeros((), dtype=torch.float32, device=device)
    max_abs_index = torch.full((), -1, dtype=torch.int64, device=device)
    worst_row_rel = torch.zeros((), dtype=torch.float64, device=device)
    worst_row_index = torch.full((), -1, dtype=torch.int64, device=device)
    nonfinite = torch.zeros((), dtype=torch.int64, device=device)
    ignored_nonzero = torch.zeros((), dtype=torch.int64, device=device)

    for start in range(0, rows, ROW_TILE):
        stop = min(start + ROW_TILE, rows)
        left, right = baseline[start:stop], candidate[start:stop]
        difference = left.float() - right.float()
        abs_difference = difference.abs()
        local_max, local_flat = abs_difference.reshape(-1).max(dim=0)
        take_max = local_max > max_abs
        max_abs = torch.maximum(max_abs, local_max)
        max_abs_index = torch.where(take_max, start * vocab + local_flat, max_abs_index)

        row_diff_sq = difference.to(torch.float64).square().sum(dim=1)
        row_ref_sq = left.to(torch.float64).square().sum(dim=1)
        rel_sq += row_diff_sq.sum()
        ref_sq += row_ref_sq.sum()
        safe_ref_sq = row_ref_sq.clamp_min(torch.finfo(torch.float64).tiny)
        row_relative = torch.sqrt(row_diff_sq / safe_ref_sq)
        row_relative = torch.where(
            (row_ref_sq == 0) & (row_diff_sq == 0), 0.0, row_relative
        )
        row_relative = torch.where(
            (row_ref_sq == 0) & (row_diff_sq > 0), float("inf"), row_relative
        )
        local_row_rel, local_row = row_relative.max(dim=0)
        take_row = local_row_rel > worst_row_rel
        worst_row_rel = torch.maximum(worst_row_rel, local_row_rel)
        worst_row_index = torch.where(take_row, start + local_row, worst_row_index)

        bins += _ulp_bins(left, right)
        nonfinite += (~torch.isfinite(left)).sum() + (~torch.isfinite(right)).sum()
        ignored = labels[start:stop] == IGNORE_INDEX
        ignored_nonzero += torch.count_nonzero(left[ignored])
        ignored_nonzero += torch.count_nonzero(right[ignored])

    if int(nonfinite.item()) != 0:
        raise AssertionError("Non-finite BF16 logits gradients were observed")
    if int(ignored_nonzero.item()) != 0:
        raise AssertionError("Ignored-token logits gradients must be exactly zero")
    expected_count = baseline.numel()
    if int(bins.sum().item()) != expected_count:
        raise AssertionError("ULP buckets did not cover every BF16 gradient element")
    global_relative = torch.sqrt(
        rel_sq / ref_sq.clamp_min(torch.finfo(torch.float64).tiny)
    )
    abs_flat = int(max_abs_index.item())
    row_flat = int(worst_row_index.item())
    return {
        "elements": expected_count,
        "ulp_distance_counts": {
            "zero": int(bins[0].item()),
            "one": int(bins[1].item()),
            "greater_than_one": int(bins[2].item()),
        },
        "max_abs_difference": float(max_abs.item()),
        "max_abs_location": {
            "row": _row_location(abs_flat // vocab, chunk_index),
            "vocab_column": abs_flat % vocab if abs_flat >= 0 else None,
        },
        "relative_l2": float(global_relative.item()),
        "worst_row_relative_l2": float(worst_row_rel.item()),
        "worst_row": _row_location(row_flat, chunk_index),
        "ignored_gradient_nonzero_count_both_variants": int(ignored_nonzero.item()),
    }


def _calibration_rows(labels: torch.Tensor, metrics: dict[str, Any]) -> list[int]:
    row_count = labels.numel()
    rows = {round(i * (row_count - 1) / 11) for i in range(12)}
    rows.update(
        {
            0,
            CHUNK_SEQUENCE - 1,
            CHUNK_SEQUENCE,
            2 * CHUNK_SEQUENCE - 1,
            2 * CHUNK_SEQUENCE,
            3 * CHUNK_SEQUENCE - 1,
            3 * CHUNK_SEQUENCE,
            row_count - 1,
        }
    )
    labels_flat = labels.reshape(-1)
    ignored = torch.nonzero(labels_flat == IGNORE_INDEX).flatten().tolist()
    rows.update(ignored)
    for key in ("max_abs_location", "worst_row"):
        location = (
            metrics[key].get("row") if key == "max_abs_location" else metrics[key]
        )
        if location is not None:
            head = 0 if location["head"] == "main" else 1
            rows.add(
                head * BATCH * CHUNK_SEQUENCE
                + location["batch"] * CHUNK_SEQUENCE
                + location["sequence"] % CHUNK_SEQUENCE
            )
    if len(rows) > 32:
        raise AssertionError(f"Calibration selected {len(rows)} rows; bound is 32")
    return sorted(rows)


def _reference_gradient(
    logits: torch.Tensor,
    labels: torch.Tensor,
    rows: torch.Tensor,
    incoming_scales: list[float],
) -> torch.Tensor:
    selected = logits.reshape(-1, VOCAB).index_select(0, rows).to(torch.float64)
    targets = labels.reshape(-1).index_select(0, rows)
    maximum = selected.max(dim=1, keepdim=True).values
    centered = selected - maximum
    log_denominator = torch.log(torch.exp(centered).sum(dim=1, keepdim=True))
    probabilities = torch.exp(centered - log_denominator)
    valid = targets != IGNORE_INDEX
    safe_targets = targets.clamp_min(0).unsqueeze(1)
    probabilities.scatter_add_(1, safe_targets, -valid.to(torch.float64).unsqueeze(1))
    probabilities.mul_(valid.to(torch.float64).unsqueeze(1))
    head_ids = torch.div(rows, BATCH * CHUNK_SEQUENCE, rounding_mode="floor")
    scales = torch.tensor(incoming_scales, dtype=torch.float64, device=logits.device)
    probabilities.mul_(scales.index_select(0, head_ids).unsqueeze(1))
    return probabilities


def _calibrate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    baseline_grad: torch.Tensor,
    candidate_grad: torch.Tensor,
    baseline_scales: list[float],
    candidate_scales: list[float],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    selected_rows = _calibration_rows(labels, metrics)
    rows = torch.tensor(selected_rows, dtype=torch.int64, device=logits.device)
    oracle_base = _reference_gradient(logits, labels, rows, baseline_scales)
    oracle_candidate = _reference_gradient(logits, labels, rows, candidate_scales)
    ideal_base = oracle_base.to(torch.bfloat16)
    ideal_candidate = oracle_candidate.to(torch.bfloat16)
    actual_base = baseline_grad.reshape(-1, VOCAB).index_select(0, rows)
    actual_candidate = candidate_grad.reshape(-1, VOCAB).index_select(0, rows)

    def summarize(
        actual: torch.Tensor, oracle: torch.Tensor, ideal: torch.Tensor
    ) -> dict[str, Any]:
        error = actual.to(torch.float64) - oracle
        ulp_counts = _ulp_bins(actual, ideal)
        return {
            "fp64_reference_max_abs_error": float(error.abs().max().item()),
            "fp64_reference_relative_l2": float(
                torch.sqrt(error.square().sum() / oracle.square().sum()).item()
            ),
            "ideal_bf16_ulp_counts": {
                "zero": int(ulp_counts[0].item()),
                "one": int(ulp_counts[1].item()),
                "greater_than_one": int(ulp_counts[2].item()),
            },
        }

    return {
        "selected_rows": selected_rows,
        "rows_include": "spread, first/last, batch/head boundaries, all ignored rows, and worst changed rows",
        "baseline": summarize(actual_base, oracle_base, ideal_base),
        "candidate": summarize(actual_candidate, oracle_candidate, ideal_candidate),
        "forward_fp64_subset": "omitted; full-input CE loss values are compared directly",
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_ROOT / "ce-numerical-replay.json",
        help="JSON receipt path (default: ignored task-local raw artifact)",
    )
    args = parser.parse_args()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    label_generator = torch.Generator(device="cpu").manual_seed(SEED)
    logits_generator = torch.Generator(device=device).manual_seed(SEED)
    main_labels = torch.randint(
        VOCAB, (BATCH, SEQUENCE), generator=label_generator, dtype=torch.int64
    )
    main_labels[:, -1] = IGNORE_INDEX
    mtp_labels = torch.cat(
        (main_labels[:, 1:], torch.full_like(main_labels[:, :1], IGNORE_INDEX)), dim=1
    )
    if not torch.equal(
        mtp_labels[:, CHUNK_SEQUENCE - 1], main_labels[:, CHUNK_SEQUENCE]
    ):
        raise AssertionError("MTP targets must shift across the outer chunk boundary")
    if not torch.all(mtp_labels[:, -2:] == IGNORE_INDEX):
        raise AssertionError(
            "MTP targets at the final two sequence positions must be ignored"
        )
    folded_labels = torch.cat((main_labels, mtp_labels), dim=0).to(device)
    if GLOBAL_VALID_TOKENS != 512 * (SEQUENCE - 1):
        raise AssertionError("Global denominator no longer matches the stated baseline")

    baseline_compiled = torch.compile(native_ce, backend="inductor")
    candidate_compiled = torch.compile(stable_ce, backend="inductor")
    total_loss = {
        "baseline": torch.zeros((), dtype=torch.float32, device=device),
        "candidate": torch.zeros((), dtype=torch.float32, device=device),
    }
    chunk_records = []

    for chunk_index in range(2):
        start = chunk_index * CHUNK_SEQUENCE
        stop = start + CHUNK_SEQUENCE
        chunk_labels = folded_labels[:, start:stop]
        logits = torch.randn(
            (HEADS * BATCH, CHUNK_SEQUENCE, VOCAB),
            generator=logits_generator,
            device=device,
            dtype=torch.bfloat16,
        )
        original_storage = logits.untyped_storage().data_ptr()
        baseline_grad, baseline_result, baseline_chunk_loss = _run_variant(
            baseline_compiled, logits, chunk_labels
        )
        candidate_grad, candidate_result, candidate_chunk_loss = _run_variant(
            candidate_compiled, logits, chunk_labels
        )
        if (
            baseline_result["storage_ptr"] != original_storage
            or candidate_result["storage_ptr"] != original_storage
        ):
            raise AssertionError("Variants did not reuse the identical logits storage")
        baseline_scale_bits = [
            entry["fp32_bits"] for entry in baseline_result["incoming_scale"]
        ]
        candidate_scale_bits = [
            entry["fp32_bits"] for entry in candidate_result["incoming_scale"]
        ]
        if baseline_scale_bits != candidate_scale_bits:
            raise AssertionError(
                "Baseline and candidate incoming FP32 scales differ bitwise"
            )
        for head_index, weight in enumerate((1.0, MTP_WEIGHT)):
            expected_scale = (
                torch.tensor(weight, dtype=torch.float32, device=device)
                / GLOBAL_VALID_TOKENS
            )
            if _scalar_bits(expected_scale) != int(baseline_scale_bits[head_index], 16):
                raise AssertionError(
                    "Recorded incoming scale does not match native backward order"
                )

        metrics = _gradient_metrics(
            baseline_grad, candidate_grad, chunk_labels, chunk_index
        )
        scales_base = [entry["value"] for entry in baseline_result["incoming_scale"]]
        scales_candidate = [
            entry["value"] for entry in candidate_result["incoming_scale"]
        ]
        calibration = _calibrate(
            logits,
            chunk_labels,
            baseline_grad,
            candidate_grad,
            scales_base,
            scales_candidate,
            metrics,
        )
        total_loss["baseline"] = total_loss["baseline"] + baseline_chunk_loss
        total_loss["candidate"] = total_loss["candidate"] + candidate_chunk_loss
        chunk_records.append(
            {
                "chunk": chunk_index + 1,
                "sequence_range": [start, stop],
                "valid_labels_by_head": [
                    int(
                        (
                            chunk_labels[head * BATCH : (head + 1) * BATCH]
                            != IGNORE_INDEX
                        )
                        .sum()
                        .item()
                    )
                    for head in range(HEADS)
                ],
                "baseline": baseline_result,
                "candidate": candidate_result,
                "absolute_raw_loss_differences": [
                    abs(left - right)
                    for left, right in zip(
                        baseline_result["raw_loss"], candidate_result["raw_loss"]
                    )
                ],
                "absolute_weighted_head_loss_differences": [
                    abs(left - right)
                    for left, right in zip(
                        baseline_result["weighted_loss"],
                        candidate_result["weighted_loss"],
                    )
                ],
                "absolute_weighted_chunk_loss_difference": abs(
                    baseline_result["chunk_loss"] - candidate_result["chunk_loss"]
                ),
                "gradient": metrics,
                "fp64_gradient_calibration": calibration,
            }
        )
        del (
            baseline_grad,
            candidate_grad,
            baseline_chunk_loss,
            candidate_chunk_loss,
            logits,
        )
        print(f"Finished numerical replay chunk {chunk_index + 1}/2", flush=True)

    output = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "torch_cuda_build": torch.version.cuda,
            "triton": triton.__version__,
            "device": torch.cuda.get_device_name(device),
            "git_head": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "source_sha256": {
                "native_loss": _hash(ROOT / "gift/torchtitan/components/loss.py"),
                "stable_ce": _hash(
                    ROOT / "gift/torchtitan/experiments/four_fusion/stable_ce.py"
                ),
                "replay_script": _hash(Path(__file__).resolve()),
            },
            "torch_compile": {
                "backend": "inductor",
                "other_options": "native defaults",
            },
            "cache_dirs": {
                "torchinductor": os.environ["TORCHINDUCTOR_CACHE_DIR"],
                "triton": os.environ["TRITON_CACHE_DIR"],
            },
        },
        "case": {
            "seed": SEED,
            "input": "synthetic BF16 normal logits with standard deviation 1; generated separately per chunk",
            "shape_per_chunk": [HEADS * BATCH, CHUNK_SEQUENCE, VOCAB],
            "dtype": "bfloat16",
            "batch": BATCH,
            "sequence": SEQUENCE,
            "vocabulary": VOCAB,
            "chunks": 2,
            "folded_order": "main batch then MTP batch, matching MTPLoss.flatten(0, 1)",
            "mtp_label_shift": "shifted across full sequence before chunk slicing; last target ignored",
            "global_valid_tokens_python_float": GLOBAL_VALID_TOKENS,
            "head_weights": [1.0, MTP_WEIGHT],
            "normalization_order": "(raw_sum / global_valid_tokens) * head_weight; main + MTP",
            "unit_upstream": True,
        },
        "results": {
            "chunks": chunk_records,
            "baseline_weighted_total_loss": float(total_loss["baseline"].item()),
            "candidate_weighted_total_loss": float(total_loss["candidate"].item()),
            "absolute_total_loss_difference": abs(
                float(total_loss["baseline"].item())
                - float(total_loss["candidate"].item())
            ),
        },
        "interpretation_limit": "CE-only synthetic replay; measured differences need precision-aware review and are not a PASS/tolerance claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
