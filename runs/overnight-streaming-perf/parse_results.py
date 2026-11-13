#!/usr/bin/env python3
"""Validate per-rank measurement sidecars and summarize one native run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any

WORLD_SIZE = 8
MICROBATCHES_PER_UPDATE = 32
LOCAL_PHYSICAL_TOKENS_PER_UPDATE = 2 * 4096 * MICROBATCHES_PER_UPDATE
GLOBAL_SUPERVISED_TOKENS_PER_UPDATE = 2_096_640
GIB = 1024**3
PROFILE_RANKS = [0, 1]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median_s": None, "min_s": None, "max_s": None}
    return {
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def _host_cadence(records: list[dict[str, Any]], first_step: int, last_step: int) -> dict[str, Any]:
    by_rank: dict[str, Any] = {}
    groups: dict[str, list[float]] = {"ordinary": [], "monitoring": []}
    for record in records:
        rank = str(record["rank"])
        selected = [
            item
            for item in record.get("cadence", [])
            if first_step <= int(item["step"]) <= last_step and item.get("interval_s") is not None
        ]
        intervals = [float(item["interval_s"]) for item in selected]
        for item in selected:
            key = "monitoring" if item["monitoring_step"] else "ordinary"
            groups[key].append(float(item["interval_s"]))
        by_rank[rank] = {
            "intervals": [
                {
                    "step": int(item["step"]),
                    "interval_s": float(item["interval_s"]),
                    "monitoring_step": bool(item["monitoring_step"]),
                    "window_boundary": item.get("window_boundary"),
                }
                for item in selected
            ],
            "summary": _quantiles(intervals),
        }
    return {
        "scope": (
            "host interval between native Profiler.step returns; adapter does not "
            "synchronize each update, and the final interval includes the end-boundary drain"
        ),
        "by_rank": by_rank,
        "ordinary_steps": _quantiles(groups["ordinary"]),
        "monitoring_steps": _quantiles(groups["monitoring"]),
    }


def _expected_profile_spec(target_update: int, warmup_updates: int) -> dict[str, Any]:
    return {
        "target_update": target_update,
        "warmup_updates": warmup_updates,
        "profile_ranks": PROFILE_RANKS,
        "settings": {
            "profiler.enable_profiling": True,
            "profiler.enable_memory_snapshot": False,
            "profiler.profile_ranks": PROFILE_RANKS,
            "profiler.save_traces_folder": "profiling/traces",
            "profiler.record_shapes": False,
            "profiler.profile_freq": 2,
            "profiler.profiler_warmup": 1,
            "profiler.profiler_active": 1,
            "profiler.profiler_repeat": 1,
            "profiler.profiler_skip_first": target_update - 2,
            "profiler.profiler_skip_first_wait": 0,
        },
        "native_hardcoded": {"with_stack": False, "profile_memory": False},
        "trace_directory": f"native/profiling/traces/iteration_{target_update}",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_int64_digest(value: int) -> str:
    return hashlib.sha256(struct.pack("=q", value)).hexdigest()


def _has_nonfinite_float(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        encoded_float = value.get("float")
        if isinstance(encoded_float, str):
            try:
                if not math.isfinite(float(encoded_float)):
                    return True
            except ValueError:
                pass
        return any(_has_nonfinite_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_nonfinite_float(item) for item in value)
    return False


def _exact_mismatches(left: Any, right: Any, *, limit: int = 20) -> list[str]:
    """Return bounded paths where two same-rank receipts differ exactly."""
    differences: list[str] = []

    def visit(a: Any, b: Any, path: str) -> None:
        if len(differences) >= limit:
            return
        if type(a) is not type(b):
            differences.append(path)
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                child = f"{path}.{key}"
                if key not in a or key not in b:
                    differences.append(child)
                else:
                    visit(a[key], b[key], child)
                if len(differences) >= limit:
                    return
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(f"{path}.length")
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                visit(item_a, item_b, f"{path}[{index}]")
                if len(differences) >= limit:
                    return
        elif a != b:
            differences.append(path)

    visit(left, right, "$receipt")
    return differences


def compare_native_numerical_receipts(
    left_path: Path, right_path: Path
) -> dict[str, Any]:
    """Compare every emitted field of two receipts for the same rank exactly."""
    left = _load_json(Path(left_path))
    right = _load_json(Path(right_path))
    differences = _exact_mismatches(left, right)
    if len(differences) == 20:
        differences.append("additional mismatch paths omitted")
    return {"equal": not differences, "mismatch_paths": differences}


def _native_numerics_summary(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    manifest = {
        "schema": "native-numerical-receipt-v1",
        "capture_updates": [1, 2, 50],
        "expected_microbatches": 32,
        "scope": "diagnostic only; no TPS",
    }
    if metadata.get("native_numerics") != manifest:
        errors.append("run metadata native_numerics manifest is missing or differs")
    if (
        metadata.get("steps") != 50
        or metadata.get("warmup_updates") != 0
        or metadata.get("window_updates") != 0
        or metadata.get("monitor_interval") != 50
        or metadata.get("profile_capture") is not None
    ):
        errors.append("run metadata does not match the exclusive native-numerics mode")

    before_hashes = metadata.get("config_hashes_before", {})
    before_hash = before_hashes.get("native_numerics_sha256") if isinstance(before_hashes, dict) else None
    after_path = run_dir / "config-hashes-final.json"
    try:
        after_hashes = _load_json(after_path)
    except ValueError as exc:
        after_hashes = {}
        errors.append(f"native-numerics final config hashes: {exc}")
    after_hash = after_hashes.get("native_numerics_sha256")
    if (
        not isinstance(before_hash, str)
        or len(before_hash) != 64
        or not isinstance(after_hash, str)
        or len(after_hash) != 64
        or before_hash != after_hash
    ):
        errors.append("native-numerics module hash is missing or changed during the run")

    measurement_dir = run_dir / "measurement"
    paths = sorted(measurement_dir.glob("native-numerics-rank*.json"))
    expected_names = {f"native-numerics-rank{rank}.json" for rank in range(WORLD_SIZE)}
    if {path.name for path in paths} != expected_names or len(paths) != WORLD_SIZE:
        errors.append(
            f"expected exactly eight native-numerics receipts rank0..rank7; found {[p.name for p in paths]}"
        )

    receipts: dict[int, dict[str, Any]] = {}
    for path in paths:
        try:
            receipt = _load_json(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        rank = receipt.get("rank")
        if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
            errors.append(f"{path.name} has an invalid rank")
            continue
        if rank in receipts:
            errors.append(f"duplicate native-numerics receipt for rank {rank}")
        receipts[rank] = receipt

    expected_coverage = {
        "input_batches": 32,
        "microbatches": 32,
        "summary_calls": 1,
        "clip_calls": 1,
        "optimizer_calls": 1,
        "batch_end_calls": 1,
    }
    expected_denominator = 2_096_640.0
    updates = (1, 2, 50)
    def only(records: list[dict[str, Any]], *, update: int, phase: str, name: str, microbatch: int | None = None) -> list[dict[str, Any]]:
        return [
            item for item in records
            if item.get("update") == update
            and item.get("phase") == phase
            and item.get("name") == name
            and item.get("microbatch") == microbatch
        ]

    for rank, receipt in sorted(receipts.items()):
        expected_receipt_keys = {
            "schema", "complete", "rank", "capture_updates", "expected_microbatches",
            "hash_semantics", "clip_norm_semantics", "ema_state", "callback_state_scope",
            "training_metadata", "positions_semantics", "microbatch_capture", "scope",
            "coverage", "observations",
        }
        if set(receipt) != expected_receipt_keys:
            errors.append(f"rank {rank} receipt fields differ from schema v1")
        if receipt.get("schema") != "native-numerical-receipt-v1" or receipt.get("complete") is not True:
            errors.append(f"rank {rank} receipt schema or completion marker is invalid")
        if receipt.get("scope") != (
            "Parameter, gradient, optimizer-state and update-summary copies to CPU "
            "synchronize and are diagnostic instrumentation, not TPS."
        ):
            errors.append(f"rank {rank} receipt scope differs from the diagnostic contract")
        if receipt.get("capture_updates") != list(updates) or receipt.get("expected_microbatches") != 32:
            errors.append(f"rank {rank} receipt capture contract differs")
        training = receipt.get("training_metadata")
        if not isinstance(training, dict) or (
            type(training.get("initial_step")) is not int or training.get("initial_step") != 0
            or training.get("mtp_loss_weights") != [0.3]
            or type(training.get("initial_num_chunks")) is not int or training.get("initial_num_chunks") != 2
        ):
            errors.append(f"rank {rank} initial step, MTP weight, or initial chunk count differs")

        coverage = receipt.get("coverage")
        observations = receipt.get("observations")
        if not isinstance(coverage, dict) or not isinstance(observations, list):
            errors.append(f"rank {rank} coverage or observations have the wrong type")
            continue
        if set(coverage) != {str(step) for step in updates}:
            errors.append(f"rank {rank} coverage does not contain exactly updates 1, 2, and 50")
        for update in updates:
            counts = coverage.get(str(update))
            if not isinstance(counts, dict) or counts != expected_coverage or any(
                type(value) is not int for value in counts.values()
            ):
                errors.append(f"rank {rank} update {update} call counts differ from the contract")

        identities: set[tuple[int, str, str, int | None]] = set()
        valid_observations = True
        for item in observations:
            if not isinstance(item, dict):
                valid_observations = False
                break
            update = item.get("update")
            phase = item.get("phase")
            name = item.get("name")
            microbatch = item.get("microbatch")
            if (
                type(update) is not int or update not in {0, *updates}
                or not isinstance(phase, str) or not phase
                or not isinstance(name, str) or not name
                or (microbatch is not None and (type(microbatch) is not int or not 0 <= microbatch < 32))
            ):
                valid_observations = False
                break
            key = (update, phase, name, microbatch)
            if key in identities:
                errors.append(f"rank {rank} has duplicate observation {key!r}")
            identities.add(key)
            required_tensor_phases = {
                "dataloader_input", "microbatch_output", "clip_result",
                "post_optimizer_parameter",
            }
            optional_gradient_phases = {"grad_preclip", "grad_postclip"}
            if phase in required_tensor_phases | optional_gradient_phases:
                present = item.get("present")
                dtype = item.get("dtype")
                if "present" not in item or type(present) is not bool:
                    valid_observations = False
                    break
                if phase in required_tensor_phases and present is not True:
                    valid_observations = False
                    break
                if present and (not isinstance(dtype, str) or not dtype.startswith("torch.")):
                    valid_observations = False
                    break
                if not present and phase in optional_gradient_phases and item.get("shape") is not None:
                    valid_observations = False
                    break
            if "present" in item:
                present = item.get("present")
                digest = item.get("sha256")
                finite = item.get("is_finite")
                dtype = item.get("dtype")
                if type(present) is not bool or (finite is not None and type(finite) is not bool):
                    valid_observations = False
                    break
                if present and (not isinstance(dtype, str) or not dtype):
                    valid_observations = False
                    break
                if present and dtype.startswith("torch.") and (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or finite is False
                ):
                    valid_observations = False
                    break
                if present and dtype.startswith("torch."):
                    floating_dtype = dtype.startswith(
                        ("torch.float", "torch.bfloat", "torch.complex")
                    )
                    if (floating_dtype and finite is not True) or (
                        not floating_dtype and finite is not None
                    ):
                        valid_observations = False
                        break
                if present and dtype.startswith("torch.") and (
                    type(item.get("rank")) is not int
                    or not isinstance(item.get("source_device_type"), str)
                    or not isinstance(item.get("global_shape"), list)
                    or not isinstance(item.get("local_shape"), list)
                    or not isinstance(item.get("local_stride"), list)
                    or type(item.get("local_numel")) is not int
                    or "mesh" not in item
                    or not isinstance(item.get("placements"), list)
                ):
                    valid_observations = False
                    break
                if present and dtype.startswith("python.float") and (
                    type(item.get("value")) is not float
                    or not math.isfinite(item["value"])
                    or finite is not True
                ):
                    valid_observations = False
                    break
                if not present and any(item.get(key) is not None for key in ("dtype", "sha256", "is_finite")):
                    valid_observations = False
                    break
                if type(item.get("rank")) is int and item["rank"] != rank:
                    valid_observations = False
                    break
                for shape_key in ("global_shape", "local_shape", "local_stride"):
                    shape = item.get(shape_key)
                    if shape is not None and (
                        not isinstance(shape, list)
                        or any(type(dim) is not int for dim in shape)
                    ):
                        valid_observations = False
                        break
                if not valid_observations:
                    break
            if any(isinstance(value, float) and not math.isfinite(value) for value in item.values()):
                valid_observations = False
                break
        if not valid_observations:
            errors.append(f"rank {rank} observations contain malformed or non-finite records")
            continue
        after_step = [
            item for item in observations
            if item.get("update") == 0
            and item.get("phase") == "after_load"
            and item.get("name") == "trainer.step"
        ]
        if (
            len(after_step) != 1
            or after_step[0].get("microbatch") is not None
            or type(after_step[0].get("value")) is not int
            or after_step[0].get("value") != 0
        ):
            errors.append(f"rank {rank} after-load step is not zero")
        for update in updates:
            step_rows = [
                item for item in observations
                if item.get("update") == update
                and item.get("phase") == "batch_end"
                and item.get("name") == "trainer.step"
            ]
            if (
                len(step_rows) != 1
                or step_rows[0].get("microbatch") is not None
                or type(step_rows[0].get("value")) is not int
                or step_rows[0].get("value") != update
            ):
                errors.append(f"rank {rank} captured trainer.step does not match update {update}")
        expected_counter_values = {0: 0, 1: 2_096_640, 2: 4_193_280, 50: 104_832_000}
        for step, value in expected_counter_values.items():
            phase = "after_load" if step == 0 else "batch_end"
            counter = only(observations, update=step, phase=phase, name="trainer.ntokens_seen")
            if len(counter) != 1:
                errors.append(f"rank {rank} counter receipt missing at update {step}")
                continue
            item = counter[0]
            if (
                item.get("dtype") != "torch.int64"
                or item.get("source_device_type") != "cuda"
                or item.get("global_shape") != []
                or item.get("local_shape") != []
                or item.get("local_numel") != 1
                or item.get("sha256") != _native_int64_digest(value)
                or item.get("is_finite") is not None
            ):
                errors.append(f"rank {rank} int64 ntokens_seen receipt differs at update {step}")

        for update, chunks in ((1, 2), (2, 2), (50, 32)):
            metadata_rows = [
                item for item in observations
                if item.get("update") == update and item.get("phase") == "microbatch_metadata"
            ]
            if len(metadata_rows) != 32 or {item.get("microbatch") for item in metadata_rows} != set(range(32)):
                errors.append(f"rank {rank} update {update} lacks 32 unique forward metadata rows")
            if any(type(item.get("active_num_chunks")) is not int or item.get("active_num_chunks") != chunks for item in metadata_rows):
                errors.append(f"rank {rank} update {update} active chunk count differs")
            if any(
                not isinstance(item.get("global_valid_tokens"), dict)
                or item["global_valid_tokens"].get("dtype") != "python.float"
                or type(item["global_valid_tokens"].get("value")) is not float
                or item["global_valid_tokens"].get("value") != expected_denominator
                for item in metadata_rows
            ):
                errors.append(f"rank {rank} update {update} microbatch denominator differs")
            labels = [
                item for item in observations
                if item.get("update") == update and item.get("phase") == "dataloader_input"
                and item.get("name") == "labels"
            ]
            inputs = [
                item for item in observations
                if item.get("update") == update and item.get("phase") == "dataloader_input"
                and item.get("name", "").startswith("input.")
            ]
            outputs = [
                item for item in observations
                if item.get("update") == update and item.get("phase") == "microbatch_output"
                and item.get("name") == "loss.local_normalized"
            ]
            if len(labels) != 32 or {item.get("microbatch") for item in labels} != set(range(32)):
                errors.append(f"rank {rank} update {update} lacks 32 unique input-label receipts")
            if len(inputs) < 32 or {item.get("microbatch") for item in inputs} != set(range(32)):
                errors.append(f"rank {rank} update {update} lacks input receipts for all microbatches")
            if len(outputs) != 32 or {item.get("microbatch") for item in outputs} != set(range(32)):
                errors.append(f"rank {rank} update {update} lacks 32 unique forward-loss receipts")
            summary_denominator = only(
                observations, update=update, phase="update_summary", name="global_valid_tokens"
            )
            if len(summary_denominator) != 1 or (
                summary_denominator[0].get("dtype") != "python.float"
                or summary_denominator[0].get("value") != expected_denominator
            ):
                errors.append(f"rank {rank} update {update} summary denominator differs")
            if len(only(observations, update=update, phase="clip_result", name="clip.total_norm")) != 1:
                errors.append(f"rank {rank} update {update} gradient clipping receipt is missing")
            if len(only(observations, update=update, phase="callback_state", name="callbacks.batch_end")) != 1:
                errors.append(f"rank {rank} update {update} batch-end receipt is missing")
            if not any(item.get("update") == update and item.get("phase") == "optimizer_group" for item in observations):
                errors.append(f"rank {rank} update {update} optimizer receipt is missing")
            if not any(item.get("update") == update and item.get("phase") == "post_optimizer_parameter" for item in observations):
                errors.append(f"rank {rank} update {update} optimizer parameter receipt is missing")
            if not any(item.get("update") == update and item.get("phase") == "grad_preclip" for item in observations):
                errors.append(f"rank {rank} update {update} preclip gradient receipts are missing")
            if not any(item.get("update") == update and item.get("phase") == "grad_postclip" for item in observations):
                errors.append(f"rank {rank} update {update} postclip gradient receipts are missing")
        if any(_has_nonfinite_float(item) for item in observations):
            errors.append(f"rank {rank} observations contain non-finite JSON numbers")
    return {
        "schema_version": 1,
        "status": "complete" if not errors else "incomplete",
        "run_id": metadata.get("run_id"),
        "mode": "native_numerics",
        "scope": "diagnostic only; no TPS",
        "interpretation": "receipt validation only; not A/A equivalence, convergence, or EMA-state evidence",
        "rank_count": len(receipts),
        "capture_updates": list(updates),
        "expected_microbatches": 32,
        "errors": errors,
    }


def _profile_update_summary(
    run_dir: Path,
    by_rank: dict[int, dict[str, Any]],
    ordered: list[dict[str, Any]],
    metadata: dict[str, Any],
    run_id: str | None,
    errors: list[str],
) -> dict[str, Any]:
    from compiler_stats import (
        deep_gemm_guard_manifest,
        validate_deep_gemm_guard_receipt,
    )

    steps = int(metadata.get("steps", 0))
    warmup = int(metadata.get("warmup_updates", 0))
    window = int(metadata.get("window_updates", 0))
    monitor_interval = int(metadata.get("monitor_interval", 0))
    spec = metadata.get("profile_capture")

    if not isinstance(spec, dict):
        errors.append("run metadata lacks a profile_capture object")
        target_update = steps
    else:
        try:
            target_update = int(spec["target_update"])
        except (KeyError, TypeError, ValueError):
            target_update = steps
            errors.append("profile_capture target_update is malformed")

    if window != 0:
        errors.append("profile-update runs require window_updates == 0")
    if monitor_interval != 50:
        errors.append("profile-update runs require monitor_interval == 50")
    if warmup <= 0:
        errors.append("profile-update runs require warmup_updates > 0")
    if target_update < 2:
        errors.append("profile target update must be at least 2")
    if target_update <= warmup:
        errors.append("profile target update must be greater than warmup_updates")
    if target_update % 50 == 0:
        errors.append("profile target update must not be a monitoring update")
    if steps != target_update:
        errors.append("profile target update must equal the complete training step count")

    expected_spec = _expected_profile_spec(target_update, warmup)
    if spec != expected_spec:
        errors.append("profile_capture settings do not match the required native schedule")
    expected_guard_manifest = deep_gemm_guard_manifest(
        enabled=True, activate_after_update=warmup
    )
    if metadata.get("deep_gemm_compilation_guard") != expected_guard_manifest:
        errors.append("run metadata DeepGEMM guard manifest is missing or differs from the required policy")

    per_rank_counts: dict[str, Any] = {}
    guard_by_rank: dict[str, Any] = {}
    for rank, record in sorted(by_rank.items()):
        if record.get("measurement_status") != "disabled" or record.get("measurement") is not None:
            errors.append(f"rank {rank} did not leave window timing disabled")
        cadence_steps = [int(item["step"]) for item in record.get("cadence", [])]
        if cadence_steps != list(range(1, steps + 1)):
            errors.append(f"rank {rank} cadence does not cover updates 1..{steps}")
        if monitor_interval > 0:
            for item in record.get("cadence", []):
                step = int(item["step"])
                expected_monitoring = step % monitor_interval == 0
                if item.get("monitoring_step") is not expected_monitoring:
                    errors.append(
                        f"rank {rank} monitoring cadence is invalid at update {step}"
                    )

        guard_receipt = record.get("compiler_guard")
        guard_by_rank[str(rank)] = guard_receipt
        for detail in validate_deep_gemm_guard_receipt(
            guard_receipt,
            activation_after_update=warmup,
            end_after_update=target_update,
            activation_sync_calls=1,
        ):
            errors.append(f"rank {rank} DeepGEMM guard: {detail}")

        receipt = record.get("profile_capture")
        if not isinstance(receipt, dict):
            errors.append(f"rank {rank} has no profile_capture receipt")
            continue
        if type(receipt.get("target_update")) is not int or receipt.get("target_update") != target_update:
            errors.append(f"rank {rank} profile target differs from run metadata")
        if type(receipt.get("warmup_updates")) is not int or receipt.get("warmup_updates") != warmup:
            errors.append(f"rank {rank} profile warmup differs from run metadata")
        for key, expected in (
            ("microbatch_calls", MICROBATCHES_PER_UPDATE),
            ("train_step_calls", 1),
            ("optimizer_step_calls", 1),
            ("native_profiler_build_calls", 1),
        ):
            if type(receipt.get(key)) is not int or receipt.get(key) != expected:
                errors.append(f"rank {rank} profile receipt {key}={receipt.get(key)!r}, expected {expected}")
        expected_constructed = rank in PROFILE_RANKS
        if receipt.get("native_profiler_created") is not expected_constructed:
            errors.append(
                f"rank {rank} native profiler constructed={receipt.get('native_profiler_created')!r}; "
                f"expected {expected_constructed}"
            )
        per_rank_counts[str(rank)] = {
            "microbatch_calls": receipt.get("microbatch_calls"),
            "train_step_calls": receipt.get("train_step_calls"),
            "optimizer_step_calls": receipt.get("optimizer_step_calls"),
            "native_profiler_build_calls": receipt.get("native_profiler_build_calls"),
            "native_profiler_created": receipt.get("native_profiler_created"),
        }

    trace_root = run_dir / "native/profiling/traces"
    expected_paths = {
        trace_root / f"iteration_{target_update}" / f"rank{rank}_trace.json"
        for rank in PROFILE_RANKS
    }
    observed_paths = (
        {path for path in trace_root.rglob("*_trace.json") if path.is_file()}
        if trace_root.is_dir()
        else set()
    )
    if observed_paths != expected_paths:
        expected_names = sorted(str(path.relative_to(run_dir)) for path in expected_paths)
        observed_names = sorted(
            str(path.relative_to(run_dir)) for path in observed_paths
        )
        errors.append(
            f"expected exactly selected-rank traces {expected_names}, found {observed_names}"
        )

    traces: dict[str, Any] = {}
    for rank in PROFILE_RANKS:
        path = trace_root / f"iteration_{target_update}" / f"rank{rank}_trace.json"
        if not path.is_file():
            continue
        marker_counts: dict[str, int] = {}
        non_complete_markers: set[str] = set()
        try:
            with path.open(encoding="utf-8") as handle:
                trace = json.load(handle)
            events = trace.get("traceEvents") if isinstance(trace, dict) else None
            if not isinstance(events, list):
                raise ValueError("traceEvents must be a list")
            for event in events:
                if isinstance(event, dict):
                    name = event.get("name")
                    if isinstance(name, str) and name.startswith("task.profile_update."):
                        marker_counts[name] = marker_counts.get(name, 0) + 1
                        if event.get("ph") != "X":
                            non_complete_markers.add(name)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"rank {rank} trace is not a valid Kineto JSON trace: {exc}")
            continue

        expected_markers = {
            f"task.profile_update.{target_update}.microbatch.rank{rank}": MICROBATCHES_PER_UPDATE,
            f"task.profile_update.{target_update}.train_step.rank{rank}": 1,
            f"task.profile_update.{target_update}.optimizer_step.rank{rank}": 1,
        }
        if marker_counts != expected_markers:
            errors.append(
                f"rank {rank} trace task.profile_update marker set/counts {marker_counts!r}; "
                f"expected exactly {expected_markers!r}"
            )
        if non_complete_markers:
            errors.append(
                f"rank {rank} trace task.profile_update markers are not complete events: "
                f"{sorted(non_complete_markers)}"
            )
        traces[str(rank)] = {
            "path": str(path.relative_to(run_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "marker_counts": marker_counts,
        }

    return {
        "schema_version": 1,
        "status": "complete" if not errors else "incomplete",
        "run_id": run_id,
        "mode": "profile_update",
        "training_steps": steps,
        "measurement_status": "disabled",
        "profile_capture": {
            "target_update": target_update,
            "warmup_updates": warmup,
            "profile_ranks": PROFILE_RANKS,
            "settings": expected_spec["settings"],
            "native_hardcoded": expected_spec["native_hardcoded"],
            "call_counts_by_rank": per_rank_counts,
            "traces_by_rank": traces,
        },
        "deep_gemm_compilation_guard": {
            "status": "complete" if len(guard_by_rank) == WORLD_SIZE and not any(
                "DeepGEMM guard" in error for error in errors
            ) else "incomplete",
            "manifest": metadata.get("deep_gemm_compilation_guard"),
            "by_rank": guard_by_rank,
        },
        "parameter_census_by_rank": {
            str(record["rank"]): record.get("parameter_census") for record in ordered
        },
        "errors": errors,
    }


def summarize_records(
    records: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    errors: list[str] = []
    expected_steps = int(metadata.get("steps", records[0].get("expected_steps", 0) if records else 0))
    warmup = int(metadata.get("warmup_updates", 0))
    window = int(metadata.get("window_updates", 0))
    monitor_interval = int(metadata.get("monitor_interval", 0))

    by_rank: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            rank = int(record["rank"])
        except (KeyError, TypeError, ValueError):
            errors.append("rank sidecar has no integer rank")
            continue
        if rank in by_rank:
            errors.append(f"duplicate sidecar for rank {rank}")
        by_rank[rank] = record

    expected_ranks = set(range(WORLD_SIZE))
    if set(by_rank) != expected_ranks:
        errors.append(
            f"expected rank sidecars {sorted(expected_ranks)}, found {sorted(by_rank)}"
        )
    ordered = [by_rank[rank] for rank in sorted(by_rank)]

    for rank, record in sorted(by_rank.items()):
        if int(record.get("world_size", -1)) != WORLD_SIZE:
            errors.append(f"rank {rank} reports world size {record.get('world_size')}")
        if record.get("training_status") != "complete":
            errors.append(f"rank {rank} training status is {record.get('training_status')!r}")
        if int(record.get("expected_steps", -1)) != expected_steps:
            errors.append(f"rank {rank} expected-step count does not match run metadata")
        if int(record.get("completed_steps", -1)) != expected_steps:
            errors.append(f"rank {rank} completed {record.get('completed_steps')} updates, expected {expected_steps}")

    if not ordered:
        return {
            "schema_version": 1,
            "status": "incomplete",
            "run_id": metadata.get("run_id"),
            "mode": "unknown",
            "errors": errors or ["no rank sidecars were produced"],
        }

    hosts = {record.get("host_name") for record in ordered}
    if len(hosts) != 1 or None in hosts:
        errors.append("rank monotonic clocks are not verified on one shared host")

    run_id = metadata.get("run_id", ordered[0].get("run_id"))
    if any(record.get("run_id") != run_id for record in ordered):
        errors.append("rank sidecars do not share one run ID")

    if metadata.get("profile_capture") is not None:
        if run_dir is None:
            errors.append("profile trace validation requires the run directory")
            return {
                "schema_version": 1,
                "status": "incomplete",
                "run_id": run_id,
                "mode": "profile_update",
                "errors": errors,
            }
        return _profile_update_summary(
            run_dir,
            by_rank,
            ordered,
            metadata,
            run_id,
            errors,
        )
    if any(record.get("profile_capture") is not None for record in ordered):
        errors.append("rank profile receipts have no matching run-level profile_capture metadata")

    if window == 0:
        for rank, record in sorted(by_rank.items()):
            if record.get("measurement_status") != "disabled":
                errors.append(f"rank {rank} did not disable window timing")
            cadence_steps = [int(item["step"]) for item in record.get("cadence", [])]
            if cadence_steps != list(range(1, expected_steps + 1)):
                errors.append(f"rank {rank} cadence does not cover updates 1..{expected_steps}")
            if record.get("capture_microbatches"):
                expected = {str(step): MICROBATCHES_PER_UPDATE for step in range(1, expected_steps + 1)}
                if record.get("microbatch_calls_by_step") != expected:
                    errors.append(f"rank {rank} did not execute 32 forward/backward calls per update")
        summary = {
            "schema_version": 1,
            "status": "complete" if not errors else "incomplete",
            "run_id": run_id,
            "mode": "cadence_only",
            "training_steps": expected_steps,
            "measurement_status": "disabled",
            "adapter_cuda_synchronizations": 0,
            "rank_cadence": _host_cadence(ordered, 1, expected_steps),
            "parameter_census_by_rank": {
                str(record["rank"]): record.get("parameter_census") for record in ordered
            },
            "errors": errors,
        }
        return summary

    if window != 50:
        errors.append(f"unsupported measured window length {window}; expected 50")
    if monitor_interval != 50:
        errors.append("measured windows require monitor interval 50")
    if expected_steps != warmup + window:
        errors.append("measured run must stop at the end of its complete window")

    from compiler_stats import (
        compare_compilation_snapshots,
        deep_gemm_guard_manifest,
        validate_deep_gemm_guard_receipt,
    )

    expected_guard_manifest = deep_gemm_guard_manifest(
        enabled=True, activate_after_update=warmup
    )
    if metadata.get("deep_gemm_compilation_guard") != expected_guard_manifest:
        errors.append("run metadata DeepGEMM guard manifest is missing or differs from the required policy")

    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    local_elapsed: dict[int, float] = {}
    physical_local: dict[int, int] = {}
    supervised_global: dict[int, int] = {}
    measurements: dict[int, dict[str, Any]] = {}
    compilation_by_rank: dict[int, dict[str, Any]] = {}
    comparison_receipt_ranks: set[int] = set()
    compiler_metadata_by_rank: dict[int, dict[str, Any]] = {}
    wall_boundary_ranks: set[int] = set()
    guard_by_rank: dict[int, dict[str, Any]] = {}
    expected_monitoring_updates = [
        step
        for step in range(warmup + 1, warmup + window + 1)
        if monitor_interval and step % monitor_interval == 0
    ]

    for rank, record in sorted(by_rank.items()):
        measurement = record.get("measurement")
        if record.get("measurement_status") != "complete" or not isinstance(measurement, dict):
            errors.append(f"rank {rank} has no complete measurement window")
            continue
        measurements[rank] = measurement
        guard_receipt = record.get("compiler_guard")
        guard_errors = validate_deep_gemm_guard_receipt(
            guard_receipt,
            activation_after_update=warmup,
            end_after_update=warmup + window,
            activation_sync_calls=0,
        )
        if isinstance(guard_receipt, dict):
            guard_by_rank[rank] = guard_receipt
        for detail in guard_errors:
            errors.append(f"rank {rank} DeepGEMM guard: {detail}")
        observer = measurement.get("compilation_observer")
        if not isinstance(observer, dict):
            errors.append(f"rank {rank} has no before/after compiler snapshots")
        else:
            source_metadata = observer.get("source_metadata")
            compiler_metadata_by_rank[rank] = source_metadata
            comparison = compare_compilation_snapshots(
                observer.get("before"), observer.get("after"), source_metadata
            )
            compilation_by_rank[rank] = comparison
            if observer.get("comparison") != comparison:
                errors.append(f"rank {rank} stored compiler comparison differs from its boundary snapshots")
            else:
                comparison_receipt_ranks.add(rank)
            if comparison["status"] == "changes_observed":
                errors.append(
                    f"rank {rank} observed compiler/cache state growth in "
                    f"{comparison['changed_sources']}"
                )
            elif comparison["status"] != "observed_stable":
                detail = "; ".join(comparison.get("errors", [])) or "state is inconclusive"
                errors.append(f"rank {rank} compiler observation is incomplete: {detail}")
        wall_start = measurement.get("start_wall_time_ns")
        wall_end = measurement.get("end_wall_time_ns")
        if type(wall_start) is not int or type(wall_end) is not int or wall_end <= wall_start:
            errors.append(f"rank {rank} has invalid wall-clock window boundary timestamps")
        else:
            wall_boundary_ranks.add(rank)
        if int(measurement.get("warmup_updates", -1)) != warmup:
            errors.append(f"rank {rank} warmup boundary differs from run metadata")
        if int(measurement.get("window_updates", -1)) != window:
            errors.append(f"rank {rank} window length differs from run metadata")
        if int(measurement.get("cuda_synchronize_calls", -1)) != 2:
            errors.append(f"rank {rank} did not use exactly two adapter boundary synchronizations")
        if measurement.get("monitoring_updates") != expected_monitoring_updates:
            errors.append(f"rank {rank} monitoring updates differ from the expected window cadence")
        try:
            start_ns = int(measurement["start_monotonic_ns"])
            end_ns = int(measurement["end_monotonic_ns"])
            starts[rank] = start_ns
            ends[rank] = end_ns
            local_elapsed[rank] = (end_ns - start_ns) / 1_000_000_000
            stored_elapsed = float(measurement["local_elapsed_s"])
            if end_ns <= start_ns or abs(local_elapsed[rank] - stored_elapsed) > 1e-6:
                errors.append(f"rank {rank} has invalid local window timestamps")
            physical_local[rank] = int(measurement["physical_input_tokens_local"])
            supervised_global[rank] = int(measurement["supervised_tokens_global"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"rank {rank} measurement fields are malformed: {exc}")

    if set(measurements) != expected_ranks:
        errors.append("complete measurements are not present for all eight ranks")
    if set(compilation_by_rank) != expected_ranks:
        errors.append("compiler before/after observations are not present for all eight ranks")
    metadata_consistent = bool(compiler_metadata_by_rank) and len(
        {json.dumps(value, sort_keys=True) for value in compiler_metadata_by_rank.values()}
    ) == 1
    if compiler_metadata_by_rank and not metadata_consistent:
        errors.append("compiler source metadata differs across ranks")
    compilation_status = (
        "incomplete"
        if (
            not metadata_consistent
            or wall_boundary_ranks != expected_ranks
            or comparison_receipt_ranks != expected_ranks
        )
        else "observed_stable"
        if set(compilation_by_rank) == expected_ranks
        and all(item.get("status") == "observed_stable" for item in compilation_by_rank.values())
        else "changes_observed"
        if any(item.get("status") == "changes_observed" for item in compilation_by_rank.values())
        else "incomplete"
    )
    if set(starts) == expected_ranks and set(ends) == expected_ranks:
        start_skew_ns = max(starts.values()) - min(starts.values())
        end_skew_ns = max(ends.values()) - min(ends.values())
        max_local_elapsed = max(local_elapsed.values())
        min_local_elapsed = min(local_elapsed.values())
        global_envelope_ns = max(ends.values()) - min(starts.values())
        global_envelope_s = global_envelope_ns / 1_000_000_000
        if global_envelope_s < max_local_elapsed:
            errors.append("same-host global envelope is shorter than a rank-local window")
    else:
        start_skew_ns = end_skew_ns = global_envelope_ns = 0
        global_envelope_s = max_local_elapsed = min_local_elapsed = 0.0

    expected_local_physical = window * LOCAL_PHYSICAL_TOKENS_PER_UPDATE
    if physical_local and any(value != expected_local_physical for value in physical_local.values()):
        errors.append(
            "physical label-storage tokens differ from full-row baseline geometry: "
            f"expected {expected_local_physical} per rank, found {physical_local}"
        )
    if supervised_global and len(set(supervised_global.values())) != 1:
        errors.append("Trainer.ntokens_seen deltas differ across ranks; they are global counters")
    expected_supervised = window * GLOBAL_SUPERVISED_TOKENS_PER_UPDATE
    if supervised_global and next(iter(supervised_global.values())) != expected_supervised:
        errors.append(
            f"global supervised-token delta is not {expected_supervised}: {supervised_global}"
        )

    physical_global = sum(physical_local.values()) if len(physical_local) == WORLD_SIZE else 0
    supervised_once = next(iter(supervised_global.values())) if len(supervised_global) == WORLD_SIZE and len(set(supervised_global.values())) == 1 else 0
    physical_global_tps = physical_global / global_envelope_s if global_envelope_s > 0 else None
    supervised_global_tps = supervised_once / global_envelope_s if global_envelope_s > 0 else None

    peak_keys = (
        "allocated_bytes.all.peak",
        "active_bytes.all.peak",
        "reserved_bytes.all.peak",
    )
    memory_by_rank: dict[str, Any] = {}
    memory_maxima: dict[str, int | None] = {}
    retry_deltas: dict[str, Any] = {}
    for key in peak_keys:
        observed = {
            rank: int(measurements[rank].get("memory", {}).get("peak_bytes", {})[key])
            for rank in measurements
            if key in measurements[rank].get("memory", {}).get("peak_bytes", {})
        }
        memory_maxima[key] = max(observed.values()) if observed else None
    for rank, measurement in measurements.items():
        memory = measurement.get("memory", {})
        counters = memory.get("allocator_counter_deltas", {})
        retry_deltas[str(rank)] = counters
        memory_by_rank[str(rank)] = {
            "peak_bytes": memory.get("peak_bytes", {}),
            "peak_gib": {
                key: value / GIB
                for key, value in memory.get("peak_bytes", {}).items()
            },
            "allocator_counters_start": memory.get("allocator_counters_start", {}),
            "allocator_counters_end": memory.get("allocator_counters_end", {}),
            "allocator_counter_deltas": counters,
        }

    for rank, record in sorted(by_rank.items()):
        census = record.get("parameter_census")
        if not isinstance(census, dict) or "trainable" not in census or "frozen" not in census:
            errors.append(f"rank {rank} lacks trainable/frozen parameter census")

    summary = {
        "schema_version": 1,
        "status": "complete" if not errors else "incomplete",
        "run_id": run_id,
        "mode": "complete_window",
        "host": next(iter(hosts)) if len(hosts) == 1 else None,
        "world_size": WORLD_SIZE,
        "training_steps": expected_steps,
        "window": {
            "warmup_updates": warmup,
            "start_after_update": warmup,
            "end_after_update": warmup + window,
            "updates": window,
            "monitor_interval": monitor_interval,
            "monitoring_updates": expected_monitoring_updates,
            "global_envelope_convention": "max(rank end monotonic timestamp) - min(rank start timestamp)",
            "primary_elapsed_s": global_envelope_s,
            "max_local_elapsed_s": max_local_elapsed,
            "min_local_elapsed_s": min_local_elapsed,
            "start_skew_ns": start_skew_ns,
            "end_skew_ns": end_skew_ns,
            "start_skew_s": start_skew_ns / 1_000_000_000,
            "end_skew_s": end_skew_ns / 1_000_000_000,
            "per_rank": {
                str(rank): {
                    "start_monotonic_ns": starts.get(rank),
                    "end_monotonic_ns": ends.get(rank),
                    "local_elapsed_s": local_elapsed.get(rank),
                    "physical_input_tokens_local": physical_local.get(rank),
                    "supervised_tokens_global": supervised_global.get(rank),
                }
                for rank in sorted(expected_ranks)
            },
        },
        "tokens": {
            "physical_input_tokens_global": physical_global,
            "physical_input_tokens_per_gpu": physical_global / WORLD_SIZE,
            "supervised_tokens_global": supervised_once,
            "supervised_tokens_per_gpu": supervised_once / WORLD_SIZE,
            "physical_input_tokens_per_second_global": physical_global_tps,
            "physical_input_tokens_per_second_per_gpu": (
                physical_global_tps / WORLD_SIZE if physical_global_tps is not None else None
            ),
            "supervised_tokens_per_second_global": supervised_global_tps,
            "supervised_tokens_per_second_per_gpu": (
                supervised_global_tps / WORLD_SIZE if supervised_global_tps is not None else None
            ),
        },
        "host_boundary_cadence": _host_cadence(ordered, warmup + 1, warmup + window),
        "memory": {
            "maximum_peak_bytes_across_ranks": memory_maxima,
            "maximum_peak_gib_across_ranks": {
                key: None if value is None else value / GIB
                for key, value in memory_maxima.items()
            },
            "by_rank": memory_by_rank,
            "allocator_counter_deltas_by_rank": retry_deltas,
        },
        "compilation_observer": {
            "status": compilation_status,
            "coverage_status": "partial",
            "observed_stable": compilation_status == "observed_stable",
            "source_metadata_by_rank": {
                str(rank): value for rank, value in sorted(compiler_metadata_by_rank.items())
            },
            "by_rank": {
                str(rank): value for rank, value in sorted(compilation_by_rank.items())
            },
            "unverified": [
                "asynchronous compiler worker completion",
                "DeepGEMM native JIT paths outside the tested grouped APIs",
            ],
            "limits": [
                "Asynchronous compiler workers may outlive the parent timing scope; this observer provides partial evidence.",
                "Triton cache changes may represent a disk-cache load or specialization.",
                "DeepGEMM cache misses are rejected for the verified binary; runtime guard behavior was probed through m_grouped and k_grouped.",
            ],
        },
        "deep_gemm_compilation_guard": {
            "status": "complete" if len(guard_by_rank) == WORLD_SIZE and not any(
                "DeepGEMM guard" in error for error in errors
            ) else "incomplete",
            "manifest": metadata.get("deep_gemm_compilation_guard"),
            "by_rank": {
                str(rank): value for rank, value in sorted(guard_by_rank.items())
            },
        },
        "parameter_census_by_rank": {
            str(record["rank"]): record.get("parameter_census") for record in ordered
        },
        "errors": errors,
    }
    return summary


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    try:
        metadata = _load_json(run_dir / "metadata.json")
    except ValueError:
        metadata = {}
    if "native_numerics" in metadata or any(
        (run_dir / "measurement").glob("native-numerics-rank*.json")
    ):
        return _native_numerics_summary(run_dir, metadata)
    records: list[dict[str, Any]] = []
    measurement_dir = run_dir / "measurement"
    if measurement_dir.is_dir():
        for path in sorted(measurement_dir.glob("rank-*.json")):
            try:
                records.append(_load_json(path))
            except ValueError as exc:
                return {
                    "schema_version": 1,
                    "status": "incomplete",
                    "run_id": metadata.get("run_id"),
                    "mode": "unknown",
                    "errors": [str(exc)],
                }
    return summarize_records(records, metadata, run_dir=run_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary = summarize_run(args.run_dir)
    output = args.output or args.run_dir / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
