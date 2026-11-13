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


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _comparison_structure_mismatches(
    left: Any,
    right: Any,
    *,
    allowed_float_roots: frozenset[str] = frozenset(),
    ignored_root_fields: frozenset[str] = frozenset(),
) -> list[str]:
    """Keep types and structure exact while permitting selected float values."""
    differences: list[str] = []

    def visit(a: Any, b: Any, path: str, allow_float_difference: bool = False) -> None:
        if type(a) is not type(b):
            differences.append(path)
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if path == "$receipt" and key in ignored_root_fields:
                    continue
                child = f"{path}.{key}"
                if key not in a or key not in b:
                    differences.append(child)
                else:
                    visit(
                        a[key],
                        b[key],
                        child,
                        allow_float_difference
                        or (path == "$receipt" and key in allowed_float_roots),
                    )
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(f"{path}.length")
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                visit(item_a, item_b, f"{path}[{index}]", allow_float_difference)
        elif (
            allow_float_difference
            and type(a) is float
            and math.isfinite(a)
            and math.isfinite(b)
        ):
            return
        elif a != b:
            differences.append(path)

    visit(left, right, "$receipt")
    return differences


_OPTIMIZER_MOMENT_SUFFIXES = (
    ".optimizer.exp_avg",
    ".optimizer.exp_avg_sq",
    ".optimizer.max_exp_avg_sq",
    ".optimizer.momentum_buffer",
)


def _native_float_observation_allowed(observation: dict[str, Any]) -> bool:
    """Allow drift only for selected mathematical values emitted by known hooks."""
    phase = observation.get("phase")
    name = observation.get("name")
    if not isinstance(name, str):
        return False
    if phase in {"microbatch_output", "grad_preclip", "grad_postclip", "post_optimizer_parameter"}:
        return (
            (phase == "microbatch_output" and name == "loss.local_normalized")
            or (phase in {"grad_preclip", "grad_postclip"} and name.endswith(".grad"))
            or phase == "post_optimizer_parameter"
        )
    if phase in {"optimizer_state", "optimizer_state_value"}:
        return name.endswith(_OPTIMIZER_MOMENT_SUFFIXES)
    if phase == "loss_components":
        return name.startswith("loss.components.")
    if phase == "aux_loss_state":
        return name.startswith("aux_loss.states.") and name.endswith((".totals", ".scale"))
    if phase == "clip_result":
        return name == "clip.total_norm"
    if phase in {"router_preclip", "router_post_optimizer", "router_batch_end"}:
        return name.endswith((".expert_bias_E", ".tokens_per_expert_E"))
    if phase == "callback_state":
        return name in {"callbacks.batch_end.loss", "callbacks.batch_end.grad_norm"}
    if phase == "monitor_publish":
        metric_name = observation.get("metric_name")
        return (
            isinstance(metric_name, str)
            and name.startswith("publish[")
            and name.endswith(f".{metric_name}")
        )
    return False


def _native_allowed_float_roots(observation: dict[str, Any]) -> frozenset[str]:
    if "value" not in observation:
        return frozenset()
    phase = observation.get("phase")
    name = observation.get("name")
    if phase == "loss_components" and isinstance(name, str) and name.startswith("loss.components."):
        return frozenset({"value"})
    if phase == "aux_loss_state" and isinstance(name, str) and name.startswith("aux_loss.states.") and name.endswith((".totals", ".scale")):
        return frozenset({"value"})
    if phase == "clip_result" and name == "clip.total_norm":
        return frozenset({"value"})
    if phase == "callback_state" and name in {
        "callbacks.batch_end.loss",
        "callbacks.batch_end.grad_norm",
    }:
        return frozenset({"value"})
    if phase == "monitor_publish" and _native_float_observation_allowed(observation):
        return frozenset({"value"})
    if phase == "optimizer_state_value" and isinstance(name, str) and name.endswith(_OPTIMIZER_MOMENT_SUFFIXES):
        return frozenset({"value"})
    return frozenset()


def _native_observation_requires_exact(observation: dict[str, Any]) -> bool:
    phase = observation.get("phase")
    name = observation.get("name")
    if name in {
        "global_valid_tokens",
        "trainer.step",
        "trainer.ntokens_seen",
        "callbacks.batch_end.global_valid_tokens",
    }:
        return True
    if phase in {"optimizer_state", "optimizer_state_value"} and isinstance(name, str):
        return name.endswith(".optimizer.step")
    return phase in {"optimizer_group", "optimizer_state_keys", "scheduler_state", "scheduler_after_load"}


_TENSOR_METRIC_FIELDS = {
    "count",
    "exact_element_match_count",
    "exact_element_match",
    "reference_sha256",
    "current_sha256",
    "exact_hash_match",
    "nonfinite_reference_count",
    "nonfinite_current_count",
    "nonfinite_mismatch_count",
    "finite_pair_count",
    "max_abs_error",
    "max_abs_error_logical_flat_index",
    "rms_error",
    "reference_rms",
    "current_rms",
    "relative_l2",
    "relative_l2_status",
    "zero_reference_mismatch_count",
    "zero_reference_max_abs_error",
    "floating_metrics_use_fp64_diagnostics",
}


def _native_tensor_metric_errors(
    metrics: Any,
    reference: dict[str, Any],
    current: dict[str, Any],
    *,
    require_exact: bool = False,
) -> list[str]:
    """Validate complete streamed metrics and bind them to both observations."""
    errors: list[str] = []
    if not isinstance(metrics, dict) or set(metrics) != _TENSOR_METRIC_FIELDS:
        return ["metrics fields are missing or unexpected"]

    reference_digest = reference.get("sha256")
    current_digest = current.get("sha256")
    for label, digest in (
        ("reference", reference_digest),
        ("current", current_digest),
        ("metrics reference", metrics.get("reference_sha256")),
        ("metrics current", metrics.get("current_sha256")),
    ):
        if not _valid_sha256(digest):
            errors.append(f"{label} tensor digest is invalid")
    if metrics.get("reference_sha256") != reference_digest:
        errors.append("metrics reference digest differs from the A observation")
    if metrics.get("current_sha256") != current_digest:
        errors.append("metrics current digest differs from the B observation")

    reference_count = reference.get("local_numel")
    current_count = current.get("local_numel")
    count = metrics.get("count")
    exact_count = metrics.get("exact_element_match_count")
    for label, observation in (("reference", reference), ("current", current)):
        shape = observation.get("local_shape")
        if (
            not isinstance(shape, list)
            or any(type(size) is not int or size < 0 for size in shape)
            or type(observation.get("local_numel")) is not int
            or math.prod(shape) != observation.get("local_numel")
        ):
            errors.append(f"{label} tensor local shape and logical count disagree")
    if (
        type(reference_count) is not int
        or type(current_count) is not int
        or reference_count < 0
        or current_count != reference_count
        or type(count) is not int
        or count != reference_count
    ):
        errors.append("metrics logical count differs from the A/B local tensor shape")
        logical_count = 0
    else:
        logical_count = count
    if type(exact_count) is not int or not 0 <= exact_count <= logical_count:
        errors.append("exact element count is invalid")
        exact_count = -1
    exact_elements = metrics.get("exact_element_match")
    if type(exact_elements) is not bool or exact_elements != (exact_count == logical_count):
        errors.append("exact element flag differs from the exact element count")

    exact_hash = metrics.get("exact_hash_match")
    digests_equal = (
        _valid_sha256(reference_digest)
        and _valid_sha256(current_digest)
        and reference_digest == current_digest
    )
    if type(exact_hash) is not bool or exact_hash != digests_equal:
        errors.append("exact hash flag differs from the A/B observation digests")
    if type(exact_count) is int and exact_hash != (exact_count == logical_count):
        errors.append("whole-tensor hash and elementwise exactness disagree")
    if metrics.get("floating_metrics_use_fp64_diagnostics") is not True:
        errors.append("floating diagnostic precision marker is missing")

    dtype = reference.get("dtype")
    floating = isinstance(dtype, str) and dtype.startswith(
        ("torch.float", "torch.bfloat", "torch.complex")
    )
    nonfinite_names = (
        "nonfinite_reference_count",
        "nonfinite_current_count",
        "nonfinite_mismatch_count",
    )
    nonfinite_counts = []
    for name in nonfinite_names:
        value = metrics.get(name)
        if type(value) is not int or not 0 <= value <= logical_count:
            errors.append(f"{name} is not a valid logical element count")
            nonfinite_counts.append(-1)
        else:
            nonfinite_counts.append(value)

    floating_fields = (
        "max_abs_error",
        "rms_error",
        "reference_rms",
        "current_rms",
        "relative_l2",
        "zero_reference_max_abs_error",
    )
    if not floating:
        for name in nonfinite_names:
            if metrics.get(name) != 0:
                errors.append(f"nonfloating tensor has nonzero {name}")
        if type(reference.get("is_finite")) is not type(None) or type(current.get("is_finite")) is not type(None):
            errors.append("nonfloating tensor finite metadata must be null")
        for name in floating_fields + ("finite_pair_count", "max_abs_error_logical_flat_index", "zero_reference_mismatch_count"):
            if metrics.get(name) is not None:
                errors.append(f"nonfloating tensor has unexpected {name}")
        if metrics.get("relative_l2_status") != "not_floating_point":
            errors.append("nonfloating tensor relative L2 status is invalid")
        if exact_count != logical_count or not exact_hash:
            errors.append("integer or other nonfloating tensor differs from its reference")
        return errors

    if type(reference.get("is_finite")) is not bool or type(current.get("is_finite")) is not bool:
        errors.append("floating tensor finite metadata is missing")
    elif reference.get("is_finite") is not True or current.get("is_finite") is not True:
        errors.append("captured floating tensors must be finite on both sides")
    if any(value != 0 for value in nonfinite_counts):
        errors.append("captured floating tensor metrics contain nonfinite elements")

    finite_pairs = metrics.get("finite_pair_count")
    if type(finite_pairs) is not int or not 0 <= finite_pairs <= logical_count:
        errors.append("finite pair count is invalid")
        finite_pairs = -1
    elif all(value >= 0 for value in nonfinite_counts):
        numerator = 2 * logical_count - sum(nonfinite_counts)
        if numerator < 0 or numerator % 2 or finite_pairs != numerator // 2:
            errors.append("finite pair count disagrees with nonfinite counts")
    if finite_pairs < 0:
        return errors

    if require_exact and (exact_count != logical_count or exact_hash is not True):
        errors.append("this floating tensor is an exact counter or optimizer-step state")

    floating_values: dict[str, float | None] = {}
    for name in floating_fields:
        value = metrics.get(name)
        if value is None and (
            name == "relative_l2"
            or (finite_pairs == 0 and name in {"max_abs_error", "rms_error", "reference_rms", "current_rms"})
        ):
            floating_values[name] = None
        elif type(value) is not float or not math.isfinite(value) or value < 0:
            errors.append(f"{name} must be a finite nonnegative float or its documented null")
            floating_values[name] = None
        else:
            floating_values[name] = value

    max_index = metrics.get("max_abs_error_logical_flat_index")
    zero_mismatches = metrics.get("zero_reference_mismatch_count")
    zero_max = floating_values["zero_reference_max_abs_error"]
    if finite_pairs > 0:
        if floating_values["max_abs_error"] is None or max_index is None:
            errors.append("finite pairs lack maximum error evidence")
        elif type(max_index) is not int or not 0 <= max_index < logical_count:
            errors.append("maximum-error logical index is invalid")
        if any(floating_values[name] is None for name in ("rms_error", "reference_rms", "current_rms")):
            errors.append("finite pairs lack RMS error evidence")
        elif (
            floating_values["rms_error"] is not None
            and floating_values["max_abs_error"] is not None
            and floating_values["rms_error"]
            > floating_values["max_abs_error"] * (1.0 + 1e-12) + 1e-300
        ):
            errors.append("RMS error exceeds maximum absolute error")
    elif any(
        metrics.get(name) is not None
        for name in ("max_abs_error", "max_abs_error_logical_flat_index", "rms_error", "reference_rms", "current_rms")
    ):
        errors.append("tensor without finite pairs has finite error metrics")

    if type(zero_mismatches) is not int or not 0 <= zero_mismatches <= max(finite_pairs, 0):
        errors.append("zero-reference mismatch count is invalid")
    elif type(zero_max) is float and (
        (zero_mismatches == 0 and zero_max != 0.0)
        or (zero_mismatches > 0 and zero_max <= 0.0)
        or (
            floating_values["max_abs_error"] is not None
            and zero_max > floating_values["max_abs_error"]
        )
    ):
        errors.append("zero-reference error metrics are inconsistent")

    status = metrics.get("relative_l2_status")
    relative = floating_values["relative_l2"]
    if logical_count == 0:
        if status != "empty_tensor" or relative is not None:
            errors.append("empty tensor relative L2 status is invalid")
    elif finite_pairs == 0:
        if status != "no_finite_pairs" or relative is not None:
            errors.append("tensor without finite pairs has invalid relative L2 status")
    elif (
        floating_values["reference_rms"] is None
        or floating_values["rms_error"] is None
    ):
        errors.append("relative L2 cannot be checked without finite RMS metrics")
    elif floating_values["reference_rms"] == 0.0:
        if floating_values["rms_error"] == 0.0:
            if status != "zero_reference_zero_error" or relative != 0.0:
                errors.append("zero-reference zero-error metrics are inconsistent")
        elif status != "zero_reference_nonzero_error" or relative is not None:
            errors.append("zero-reference nonzero-error metrics are inconsistent")
    else:
        if status != "defined" or relative is None:
            errors.append("relative L2 metric is missing for a nonzero reference")
        elif not math.isclose(
            relative,
            floating_values["rms_error"] / floating_values["reference_rms"],
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            errors.append("relative L2 differs from the reported RMS metrics")
    return errors


def _numeric_comparison_has_exact_mismatch(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("kind") != "float" and "kind" in value:
            return True
        return any(_numeric_comparison_has_exact_mismatch(item) for item in value.values())
    if isinstance(value, list):
        return any(_numeric_comparison_has_exact_mismatch(item) for item in value)
    return False


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


def _native_reference_summary(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate rank-local reference export or streamed comparison evidence."""
    errors: list[str] = []
    spec = metadata.get("native_reference")
    if not isinstance(spec, dict):
        return {
            "schema_version": 1,
            "status": "incomplete",
            "run_id": metadata.get("run_id"),
            "mode": "native_reference",
            "errors": ["native_reference metadata is missing"],
        }
    mode = spec.get("mode")
    capture_update = spec.get("capture_update")
    if mode not in {"export", "compare"} or type(capture_update) is not int or capture_update not in {1, 2, 50}:
        errors.append("reference mode or selected capture update is invalid")
    prefix_updates = list(range(1, capture_update + 1)) if type(capture_update) is int and capture_update > 0 else []
    if (
        spec.get("schema") != "native-numerical-calibration-v1"
        or spec.get("prefix_updates") != prefix_updates
        or spec.get("expected_microbatches") != MICROBATCHES_PER_UPDATE
        or spec.get("monitor_interval") != 50
        or metadata.get("steps") != capture_update
        or metadata.get("warmup_updates") != 0
        or metadata.get("window_updates") != 0
        or metadata.get("monitor_interval") != 50
        or metadata.get("profile_capture") is not None
    ):
        errors.append("reference-run contract is not bounded to the selected diagnostic prefix")
    if not isinstance(spec.get("reference_namespace"), str):
        errors.append("reference namespace path is missing")

    before_hashes = metadata.get("config_hashes_before", {})
    try:
        after_hashes = _load_json(run_dir / "config-hashes-final.json")
    except ValueError as exc:
        after_hashes = {}
        errors.append(f"reference source hashes: {exc}")
    for name in ("native_numerics_sha256", "native_numerics_payload_sha256"):
        before = before_hashes.get(name) if isinstance(before_hashes, dict) else None
        after = after_hashes.get(name)
        if not isinstance(before, str) or len(before) != 64 or before != after:
            errors.append(f"{name} is missing or changed during the run")

    measurement_dir = run_dir / "measurement"
    paths = sorted(measurement_dir.glob("native-numerics-rank*.json"))
    expected_names = {f"native-numerics-rank{rank}.json" for rank in range(WORLD_SIZE)}
    if {path.name for path in paths} != expected_names or len(paths) != WORLD_SIZE:
        errors.append(f"expected eight calibration receipts; found {[path.name for path in paths]}")

    receipts: dict[int, dict[str, Any]] = {}
    total_payload_bytes = 0
    total_payload_count = 0
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
            errors.append(f"duplicate calibration receipt for rank {rank}")
        receipts[rank] = receipt
        if (
            receipt.get("schema") != "native-numerical-calibration-v1"
            or receipt.get("complete") is not True
            or receipt.get("rank") != rank
            or receipt.get("capture_updates") != [capture_update]
            or receipt.get("prefix_updates") != prefix_updates
            or receipt.get("expected_microbatches") != MICROBATCHES_PER_UPDATE
        ):
            errors.append(f"rank {rank} calibration receipt schema or capture contract differs")

        expected_coverage = {
            "input_batches": 32,
            "microbatches": 32,
            "summary_calls": 1,
            "clip_calls": 1,
            "optimizer_calls": 1,
            "batch_end_calls": 1,
        }
        coverage = receipt.get("coverage")
        observations = receipt.get("observations")
        if (
            not isinstance(coverage, dict)
            or coverage != {str(capture_update): expected_coverage}
            or not isinstance(observations, list)
        ):
            errors.append(f"rank {rank} selected-update coverage or observations are malformed")
            continue

        training = receipt.get("training_metadata")
        if not isinstance(training, dict) or (
            training.get("initial_step") != 0
            or training.get("mtp_loss_weights") != [0.3]
            or training.get("initial_num_chunks") != 2
            or not isinstance(training.get("resolved_config_sha256"), str)
            or len(training.get("resolved_config_sha256", "")) != 64
            or not isinstance(training.get("ownership"), list)
            or not isinstance(training.get("deterministic_backend_flags"), dict)
        ):
            errors.append(f"rank {rank} initial config or ownership gate is incomplete")

        identities: dict[tuple[Any, ...], dict[str, Any]] = {}
        for item in observations:
            if not isinstance(item, dict):
                errors.append(f"rank {rank} has a non-object observation")
                continue
            key = (
                item.get("rank"), item.get("update"), item.get("phase"),
                item.get("name"), item.get("microbatch"),
            )
            if (
                type(key[0]) is not int or key[0] != rank
                or type(key[1]) is not int or key[1] not in {0, *prefix_updates}
                or not isinstance(key[2], str) or not key[2]
                or not isinstance(key[3], str) or not key[3]
                or (key[4] is not None and (type(key[4]) is not int or not 0 <= key[4] < 32))
            ):
                errors.append(f"rank {rank} has an invalid observation identity")
                continue
            if key in identities:
                errors.append(f"rank {rank} has duplicate observation identity {key!r}")
            identities[key] = item
            if type(item.get("present")) is not bool:
                errors.append(f"rank {rank} observation {key!r} lacks present metadata")
            if (
                item.get("present") is True
                and item.get("phase") != "monitor_record"
                and str(item.get("dtype", "")).startswith("torch.")
            ):
                if (
                    not isinstance(item.get("sha256"), str)
                    or len(item["sha256"]) != 64
                    or type(item.get("local_numel")) is not int
                    or not isinstance(item.get("local_shape"), list)
                    or not isinstance(item.get("local_stride"), list)
                    or not isinstance(item.get("placements"), list)
                    or "mesh" not in item
                    or "owner" not in item
                    or not isinstance(item.get("aliases"), list)
                ):
                    errors.append(f"rank {rank} tensor observation {key!r} lacks shard metadata")
        for update in prefix_updates:
            labels = [
                item for key, item in identities.items()
                if key[1] == update and key[2] == "dataloader_input" and key[3] == "labels"
            ]
            inputs = [
                item for key, item in identities.items()
                if key[1] == update and key[2] == "dataloader_input" and str(key[3]).startswith("input.")
            ]
            if len(labels) != 32 or {item.get("microbatch") for item in labels} != set(range(32)):
                errors.append(f"rank {rank} update {update} lacks 32 CPU label hashes")
            if len(inputs) < 32 or {item.get("microbatch") for item in inputs} != set(range(32)):
                errors.append(f"rank {rank} update {update} lacks all-prefix CPU input hashes")

        chunk_count = 32 if capture_update == 50 else 2
        selected_metadata = [
            item for key, item in identities.items()
            if key[1] == capture_update and key[2] == "microbatch_metadata"
        ]
        if len(selected_metadata) != 32 or any(
            type(item.get("active_num_chunks")) is not int
            or item.get("active_num_chunks") != chunk_count
            for item in selected_metadata
        ):
            errors.append(f"rank {rank} update {capture_update} chunk count differs from {chunk_count}")
        selected_losses = [
            item for key, item in identities.items()
            if key[1] == capture_update and key[2] == "microbatch_output"
        ]
        if len(selected_losses) != 32:
            errors.append(f"rank {rank} selected update lacks 32 detached scalar losses")
        for phase in ("grad_preclip", "grad_postclip", "post_optimizer_parameter"):
            if not any(key[1] == capture_update and key[2] == phase for key in identities):
                errors.append(f"rank {rank} selected update lacks {phase} observations")

        rng = [item for key, item in identities.items() if key[2] == "fit_start_rng"]
        if {item.get("name") for item in rng} != {
            "python.random", "numpy.random", "torch.cpu", "torch.cuda.current_rank"
        } or any(not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64 for item in rng):
            errors.append(f"rank {rank} fit_start RNG inventory is incomplete")

        monitor = receipt.get("monitor_attestation")
        if capture_update == 50:
            if (
                not isinstance(monitor, dict)
                or monitor.get("status") != "captured"
                or monitor.get("event") != "batch_end"
                or monitor.get("update") != 50
                or monitor.get("emitter_calls") != 1
                or monitor.get("record_count", 0) <= 0
                or not monitor.get("publish_calls")
                or any(not item.get("scalar") or item.get("reduce_mode") != "online" for item in monitor.get("records", []))
            ):
                errors.append(f"rank {rank} monitor50 emitter attestation is incomplete or uncovered")
        elif (
            not isinstance(monitor, dict)
            or monitor.get("status") != "not_applicable"
            or any(key[2] in {"monitor_record", "monitor_publish"} for key in identities)
        ):
            errors.append(f"rank {rank} fabricated monitor observations for a nonmonitor update")

        reference = receipt.get("reference")
        if not isinstance(reference, dict) or reference.get("mode") != mode or reference.get("capture_update") != capture_update:
            errors.append(f"rank {rank} receipt reference result differs from run mode")
            continue
        namespace = Path(spec.get("reference_namespace", ""))
        rank_dir = namespace / f"rank-{rank:02d}"
        manifest_path = rank_dir / "manifest.json"
        try:
            reference_manifest = _load_json(manifest_path)
        except ValueError as exc:
            errors.append(f"rank {rank} reference manifest: {exc}")
            continue
        if (
            reference_manifest.get("schema") != "native-numerical-reference-v1"
            or reference_manifest.get("complete") is not True
            or type(reference_manifest.get("rank")) is not int
            or reference_manifest.get("rank") != rank
            or reference_manifest.get("capture_update") != capture_update
            or reference_manifest.get("prefix_updates") != prefix_updates
        ):
            errors.append(f"rank {rank} reference manifest schema, rank, or selected prefix differs")
            continue
        if (
            reference_manifest.get("monitor_attestation") != receipt.get("monitor_attestation")
            or reference_manifest.get("training_metadata") != receipt.get("training_metadata")
            or reference_manifest.get("coverage") != receipt.get("coverage")
        ):
            errors.append(f"rank {rank} reference manifest attestation differs from its receipt")
        manifest_items = reference_manifest.get("observations")
        if not isinstance(manifest_items, list):
            errors.append(f"rank {rank} reference observations are malformed")
            continue
        manifest_ids = set()
        manifest_by_id: dict[tuple[Any, ...], dict[str, Any]] = {}
        payload_ids = set()
        for entry in manifest_items:
            if not isinstance(entry, dict) or not isinstance(entry.get("metadata"), dict):
                errors.append(f"rank {rank} reference contains a malformed observation")
                continue
            item = entry["metadata"]
            item_key = (item.get("rank"), item.get("update"), item.get("phase"), item.get("name"), item.get("microbatch"))
            if (
                type(item_key[0]) is not int
                or type(item_key[1]) is not int
                or not isinstance(item_key[2], str)
                or not isinstance(item_key[3], str)
                or (item_key[4] is not None and type(item_key[4]) is not int)
            ):
                errors.append(f"rank {rank} reference has an invalid observation identity")
                continue
            if item_key in manifest_ids:
                errors.append(f"rank {rank} reference has duplicate observation {item_key!r}")
            manifest_ids.add(item_key)
            manifest_by_id[item_key] = item
            payload_file = entry.get("payload_file")
            gated_observation = item_key[1] == 0 or item_key[2] == "dataloader_input"
            floating_payload = (
                payload_file is not None
                and str(item.get("dtype", "")).startswith(
                    ("torch.float", "torch.bfloat", "torch.complex")
                )
            )
            numeric_drift_allowed = (
                mode == "compare"
                and not gated_observation
                and _native_float_observation_allowed(item)
            )
            allowed_float_roots = (
                _native_allowed_float_roots(item)
                if numeric_drift_allowed and payload_file is None
                else frozenset()
            )
            ignored_payload_fields = (
                frozenset({"sha256", "is_finite"})
                if numeric_drift_allowed and floating_payload
                else frozenset()
            )
            current_item = identities.get(item_key)
            if current_item is None:
                errors.append(f"rank {rank} reference observation is absent from its receipt at {item_key!r}")
            elif _comparison_structure_mismatches(
                current_item,
                item,
                allowed_float_roots=allowed_float_roots,
                ignored_root_fields=ignored_payload_fields,
            ):
                errors.append(f"rank {rank} reference metadata differs from its rank receipt at {item_key!r}")
            if numeric_drift_allowed and floating_payload and current_item is not None:
                for label, observation in (("reference", item), ("current", current_item)):
                    if not _valid_sha256(observation.get("sha256")):
                        errors.append(f"rank {rank} selected floating {label} digest is invalid at {item_key!r}")
                    if type(observation.get("is_finite")) is not bool:
                        errors.append(f"rank {rank} selected floating {label} finite flag is invalid at {item_key!r}")
            if payload_file is not None:
                payload_ids.add(item_key)
                relative = Path(payload_file)
                payload_path = (rank_dir / relative).resolve()
                if relative.is_absolute() or ".." in relative.parts or rank_dir.resolve() not in payload_path.parents:
                    errors.append(f"rank {rank} reference payload escapes its namespace")
                else:
                    if type(entry.get("nbytes")) is not int or entry["nbytes"] < 0:
                        errors.append(f"rank {rank} reference payload size metadata is invalid at {item_key!r}")
                    else:
                        total_payload_bytes += entry["nbytes"]
                        total_payload_count += 1
                        if mode == "export" and (
                            not payload_path.is_file()
                            or payload_path.stat().st_size != entry["nbytes"]
                        ):
                            errors.append(f"rank {rank} reference payload is missing or truncated at {item_key!r}")
                        if mode == "compare" and payload_path.exists():
                            errors.append(f"rank {rank} comparison left a reference payload unconsumed at {item_key!r}")
        if manifest_ids != set(identities):
            errors.append(f"rank {rank} reference key inventory differs from its receipt")
        digest = _sha256_file(manifest_path)
        if reference_manifest.get("training_metadata") != receipt.get("training_metadata"):
            errors.append(f"rank {rank} reference initial training metadata differs")
        if reference_manifest.get("coverage") != receipt.get("coverage"):
            errors.append(f"rank {rank} reference selected coverage differs")

        if mode == "export":
            expected_payload_ids = {
                key for key, item in identities.items()
                if key[1] == capture_update and key[2] != "dataloader_input"
                and key[2] != "monitor_record"
                and item.get("present") is True and str(item.get("dtype", "")).startswith("torch.")
            }
            if payload_ids != expected_payload_ids:
                errors.append(f"rank {rank} exported payload set differs from selected tensor inventory")
            if reference.get("manifest") != str(manifest_path):
                errors.append(f"rank {rank} receipt points to another export manifest")
        else:
            comparison_path = Path(reference.get("manifest", ""))
            try:
                comparison_manifest = _load_json(comparison_path)
            except ValueError as exc:
                errors.append(f"rank {rank} comparison manifest: {exc}")
                continue
            if (
                comparison_manifest.get("schema") != "native-numerical-comparison-v1"
                or comparison_manifest.get("complete") is not True
                or comparison_manifest.get("rank") != rank
                or comparison_manifest.get("capture_update") != capture_update
                or comparison_manifest.get("reference_manifest_sha256") != digest
                or comparison_manifest.get("initial_gate") != "exact_match"
                or comparison_manifest.get("prefix_input_gate") != "all_hashes_equal"
            ):
                errors.append(f"rank {rank} comparison manifest or gates are incomplete")
                continue
            rows = comparison_manifest.get("observations")
            gated_ids = {
                key for key in identities
                if key[1] == 0
                or (key[2] == "dataloader_input" and key[1] <= capture_update)
            }
            expected_result_ids = set(identities) - gated_ids
            if (
                not isinstance(rows, list)
                or len(rows) != len(expected_result_ids)
                or comparison_manifest.get("observation_count") != len(identities)
                or comparison_manifest.get("payload_comparison_count") != len(payload_ids)
                or comparison_manifest.get("scalar_comparison_count") != len(expected_result_ids - payload_ids)
            ):
                errors.append(f"rank {rank} comparison result inventory is incomplete")
                continue
            row_ids = set()
            result_paths = set()
            from numerics_payload import _numeric_tree

            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("key"), dict):
                    errors.append(f"rank {rank} comparison has a malformed observation result")
                    continue
                result_key = row["key"]
                item_key = tuple(
                    result_key.get(field)
                    for field in ("rank", "update", "phase", "name", "microbatch")
                )
                if (
                    type(item_key[0]) is not int
                    or type(item_key[1]) is not int
                    or not isinstance(item_key[2], str)
                    or not isinstance(item_key[3], str)
                    or (item_key[4] is not None and type(item_key[4]) is not int)
                ):
                    errors.append(f"rank {rank} comparison has an invalid result identity")
                    continue
                if item_key in row_ids or item_key not in expected_result_ids:
                    errors.append(f"rank {rank} comparison has an unexpected or duplicate result {item_key!r}")
                row_ids.add(item_key)
                expected_key = {
                    field: value
                    for field, value in zip(
                        ("rank", "update", "phase", "name", "microbatch"), item_key
                    )
                }
                if _exact_mismatches(result_key, expected_key, limit=1):
                    errors.append(f"rank {rank} comparison result key is not canonical at {item_key!r}")
                relative_result = row.get("result")
                if not isinstance(relative_result, str) or not relative_result:
                    errors.append(f"rank {rank} comparison result path is invalid at {item_key!r}")
                    continue
                result_path = (comparison_path.parent / relative_result).resolve()
                if comparison_path.parent.resolve() not in result_path.parents:
                    errors.append(f"rank {rank} comparison result escapes its namespace")
                    continue
                if result_path in result_paths:
                    errors.append(f"rank {rank} comparison reuses a result file at {item_key!r}")
                result_paths.add(result_path)
                try:
                    result = _load_json(result_path)
                except ValueError as exc:
                    errors.append(f"rank {rank} comparison result: {exc}")
                    continue
                result_digest = row.get("result_sha256")
                if (
                    not _valid_sha256(result_digest)
                    or _sha256_file(result_path) != result_digest
                ):
                    errors.append(f"rank {rank} comparison result digest differs at {item_key!r}")
                if (
                    result.get("schema") != "native-numerical-comparison-v1"
                    or _exact_mismatches(result.get("key"), result_key, limit=1)
                ):
                    errors.append(f"rank {rank} comparison result identity differs")
                if result.get("metadata_equal") is not True:
                    errors.append(f"rank {rank} comparison metadata gate is missing at {item_key!r}")

                reference_item = manifest_by_id.get(item_key)
                current_item = identities.get(item_key)
                if reference_item is None or current_item is None:
                    errors.append(f"rank {rank} comparison has no A/B observations at {item_key!r}")
                    continue
                if item_key in payload_ids:
                    metric_errors = _native_tensor_metric_errors(
                        result.get("metrics"),
                        reference_item,
                        current_item,
                        require_exact=_native_observation_requires_exact(reference_item),
                    )
                    if result.get("comparison_eligible") is not True:
                        metric_errors.append("tensor result is not marked comparison eligible")
                    if any(
                        field in result
                        for field in ("numeric_comparison", "reference_value", "current_value", "reason")
                    ):
                        metric_errors.append("tensor result contains fields for another result kind")
                    errors.extend(
                        f"rank {rank} comparison at {item_key!r}: {message}"
                        for message in metric_errors
                    )
                elif (
                    reference_item.get("phase") == "monitor_publish"
                    and reference_item.get("resource_observation") is True
                ):
                    reference_value = reference_item.get("value")
                    current_value = current_item.get("value")
                    if (
                        result.get("comparison_eligible") is not False
                        or current_item.get("resource_observation") is not True
                        or type(reference_value) is not float
                        or not math.isfinite(reference_value)
                        or type(current_value) is not float
                        or not math.isfinite(current_value)
                        or type(result.get("reference_value")) is not float
                        or result.get("reference_value") != reference_value
                        or type(result.get("current_value")) is not float
                        or result.get("current_value") != current_value
                        or result.get("reason")
                        != "timing, throughput, remaining-time, or memory observation"
                        or any(
                            field in result
                            for field in ("metrics", "numeric_comparison")
                        )
                    ):
                        errors.append(f"rank {rank} resource observation result is invalid at {item_key!r}")
                else:
                    expected_numeric = _numeric_tree(reference_item, current_item)
                    if (
                        _has_nonfinite_float(reference_item)
                        or _has_nonfinite_float(current_item)
                        or (
                            _native_observation_requires_exact(reference_item)
                            and _comparison_structure_mismatches(
                                reference_item, current_item
                            )
                        )
                        or result.get("comparison_eligible") is not True
                        or "metrics" in result
                        or any(
                            field in result
                            for field in ("reference_value", "current_value", "reason")
                        )
                        or _exact_mismatches(
                            result.get("numeric_comparison"), expected_numeric, limit=1
                        )
                        or _numeric_comparison_has_exact_mismatch(expected_numeric)
                    ):
                        errors.append(f"rank {rank} scalar or exact result is invalid at {item_key!r}")
            if row_ids != expected_result_ids:
                errors.append(f"rank {rank} comparison result keys differ from nongate receipt inventory")
            try:
                initial_gate = _load_json(comparison_path.parent / "initial-gates.json")
                prefix_gate = _load_json(comparison_path.parent / "prefix-inputs.json")
            except ValueError as exc:
                errors.append(f"rank {rank} comparison gates: {exc}")
                continue
            expected_input_ids = {
                key for key in identities
                if key[2] == "dataloader_input" and key[1] <= capture_update
            }
            prefix_rows = prefix_gate.get("observations")
            prefix_ids = {
                tuple(row.get("key", {}).get(field) for field in ("rank", "update", "phase", "name", "microbatch"))
                for row in prefix_rows or []
                if isinstance(row, dict) and isinstance(row.get("key"), dict)
            }
            prefix_hashes_match = all(
                isinstance(row, dict)
                and isinstance(row.get("key"), dict)
                and row.get("equal") is True
                and identities.get(
                    tuple(row["key"].get(field) for field in ("rank", "update", "phase", "name", "microbatch"))
                , {}).get("sha256") == row.get("sha256")
                for row in prefix_rows or []
            )
            initial_count = sum(key[1] == 0 for key in identities)
            if (
                initial_gate.get("complete") is not True
                or initial_gate.get("gate") != "initial_state"
                or initial_gate.get("rank") != rank
                or initial_gate.get("observation_count") != initial_count
                or initial_gate.get("training_metadata_equal") is not True
                or prefix_gate.get("complete") is not True
                or prefix_gate.get("gate") != "all_prefix_cpu_inputs"
                or prefix_gate.get("rank") != rank
                or prefix_gate.get("updates") != prefix_updates
                or prefix_gate.get("observation_count") != len(expected_input_ids)
                or prefix_ids != expected_input_ids
                or not prefix_hashes_match
            ):
                errors.append(f"rank {rank} initial or all-prefix input gate receipt is incomplete")

    return {
        "schema_version": 1,
        "status": "complete" if not errors else "incomplete",
        "run_id": metadata.get("run_id"),
        "mode": f"native_reference_{mode}",
        "scope": "diagnostic calibration only; no TPS or numerical tolerance acceptance",
        "rank_count": len(receipts),
        "capture_update": capture_update,
        "prefix_updates": prefix_updates,
        "expected_microbatches": 32,
        "payload_count": total_payload_count,
        "payload_bytes": total_payload_bytes,
        "numerical_verdict": "calibration metrics only; A/A differences require later interpretation",
        "errors": errors,
    }


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
        gpu_projection_counts: dict[str, int] = {}
        invalid_marker_categories: set[str] = set()
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
                        category = event.get("cat")
                        if category == "user_annotation":
                            counts = marker_counts
                        elif category == "gpu_user_annotation":
                            counts = gpu_projection_counts
                        else:
                            invalid_marker_categories.add(
                                "missing" if "cat" not in event or category is None else repr(category)
                            )
                            continue
                        counts[name] = counts.get(name, 0) + 1
                        if event.get("ph") != "X":
                            non_complete_markers.add(f"{category}:{name}")
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
                f"rank {rank} trace CPU task.profile_update marker set/counts {marker_counts!r}; "
                f"expected exactly {expected_markers!r}"
            )
        unexpected_gpu_markers = set(gpu_projection_counts) - set(expected_markers)
        if unexpected_gpu_markers:
            errors.append(
                f"rank {rank} trace GPU task.profile_update markers contain unexpected names: "
                f"{sorted(unexpected_gpu_markers)}"
            )
        if invalid_marker_categories:
            errors.append(
                f"rank {rank} trace task.profile_update markers have missing or unsupported "
                f"categories: {sorted(invalid_marker_categories)}"
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
            "gpu_projection_counts": gpu_projection_counts,
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
    if "native_reference" in metadata:
        return _native_reference_summary(run_dir, metadata)
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
