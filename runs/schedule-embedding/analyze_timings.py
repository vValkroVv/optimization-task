#!/usr/bin/env python3
"""Analyze completed seven-variant native timing receipts.

The extractor understands the existing all-rank native receipt schema. The
campaign CLI is deliberately stricter: all seven final rows need authenticated
004 timing adjudications and complete 140-update, 40/100-window receipts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

LABELS = ("A", "B", "C", "D0", "S", "SE", "SE-fused")
RANKS = tuple(range(8))
EXPECTED_STEPS = 140
EXPECTED_WARMUP = 40
EXPECTED_WINDOW = 100
EXPECTED_PHYSICAL_TOKENS = 209_715_200
EXPECTED_SUPERVISED_TOKENS = 209_664_000
LOCAL_PHYSICAL_TOKENS = EXPECTED_PHYSICAL_TOKENS // len(RANKS)
STAGE_RECEIPT_CAPTURE = "after_trainer_init_before_first_optimizer_update"
HISTORICAL_SE_FIXTURE_RUN_ID = "schedule-embedding-se-timing-001"
RAW_FAIL_VARIANTS = frozenset(("A", "B", "C", "D0", "S"))
STAGE_MODULE_COUNTS = {label: (0 if label in {"A", "B", "C"} else 1) for label in LABELS}
EVIDENCE_ROOT = Path(__file__).resolve().parent
RAW_ROOT = EVIDENCE_ROOT / "artifacts/raw"
REPAIR_PLAN_PATH = RAW_ROOT / "newlaunchrecords/repair-004-plan.json"
REPAIR_PLAN_SHA256 = "1a193606963e2f3c7086a6afb1d6d56ad78692dae54121e257c05c18fc8a5968"
SOURCE_FREEZE_SHA256 = "3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd"
NULLABLE_STAGE_GUARD = "resolved config lacks parallelism.gfsdp_row_owner_embedding_optimization_stages"
NULLABLE_STAGE_FAILURE = f"RuntimeError: {NULLABLE_STAGE_GUARD}"
ADJUDICATION_ROOT = RAW_ROOT / "adjudication-140-004"
ADJUDICATOR_METHOD_PATH = EVIDENCE_ROOT / "adjudicate.py"
STAGE_TUPLES = {
    "A": (),
    "B": (),
    "C": (),
    "D0": (),
    "S": (),
    "SE": ("compact", "owner_add", "owner_lookup"),
    "SE-fused": ("compact", "owner_add", "owner_lookup", "routing"),
}
INCREMENTAL_COMPARISONS = (
    ("B/A", "B", "A"),
    ("C/B", "C", "B"),
    ("D0/C", "D0", "C"),
    ("S/D0", "S", "D0"),
    ("SE/S", "SE", "S"),
    ("SE-fused/SE", "SE-fused", "SE"),
    ("SE-fused/S", "SE-fused", "S"),
)

GC_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d[,.]\d+)"
    r".*?\[GC\]\s+(?P<event>Initial GC collection|Performing periodic GC collection)"
    r" took (?P<seconds>[0-9]+(?:\.[0-9]+)?) seconds",
)
STAGE_PATTERN = re.compile(r"row_owner_embedding optimizations enabled stages=(\([^\n]*?\))")
EXPLICIT_UPDATE_PATTERN = re.compile(r"\b(?:update|step)\s*[=: ]\s*(\d+)\b", re.IGNORECASE)


class ReceiptError(ValueError):
    """Raised when receipt files cannot be parsed or a campaign gate fails."""


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"cannot read JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReceiptError(f"expected a JSON object in {path}")
    return value


def _read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _absolute_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return str(Path(value).expanduser().resolve())


def _expected_authenticated_runtime_files(source_freeze: dict[str, Any]) -> dict[str, str] | None:
    """Reconstruct the explicit helper manifest authenticated by the freeze."""
    verified: dict[str, str] = {}
    helpers = source_freeze.get("executed_helpers")
    if not isinstance(helpers, dict) or not helpers:
        return None
    for entry in helpers.values():
        if not isinstance(entry, dict):
            return None
        path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(path, str) or not Path(path).is_absolute():
            return None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        verified[path] = digest

    schedule_sources = source_freeze.get("schedule_source_sha256")
    if not isinstance(schedule_sources, dict) or not schedule_sources:
        return None
    for relative, digest in schedule_sources.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            return None
        verified[str((EVIDENCE_ROOT.parents[1] / relative).resolve())] = digest

    flash_identity = source_freeze.get("flash_source_identity")
    bridge_sources = flash_identity.get("bridge_sources") if isinstance(flash_identity, dict) else None
    if not isinstance(bridge_sources, dict) or not bridge_sources:
        return None
    for relative, digest in bridge_sources.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            return None
        verified[str((EVIDENCE_ROOT.parents[1] / relative).resolve())] = digest
    return verified


def validate_adjudication_record(
    record: dict[str, Any],
    *,
    label: str,
    expected_run_id: str,
    raw_run_path: Path,
    raw_status: str | None,
    raw_exit_code: str | None,
    raw_failure: dict[str, Any] | None,
    raw_process: dict[str, Any] | None,
    raw_completion: dict[str, Any] | None,
    raw_summary: dict[str, Any] | None,
    metadata: dict[str, Any],
    source_final: dict[str, Any] | None,
    config_hashes_final: dict[str, Any] | None,
    summary: dict[str, Any],
    summary_path: Path,
    summary_sha256: str,
    adjudication_path: Path,
    expected_adjudication_path: Path,
    plan: dict[str, Any],
    plan_path: Path,
    plan_sha256: str,
    manifest_method_sha256: str | None,
    current_method_sha256: str,
    current_raw_hashes: dict[str, str],
    expected_source_freeze_sha256: str = SOURCE_FREEZE_SHA256,
    expected_plan_sha256: str = REPAIR_PLAN_SHA256,
) -> list[str]:
    """Validate the uniform timing proof for either supported wrapper outcome."""
    errors: list[str] = []
    _require(record.get("schema_version") == 1, errors, "adjudication schema_version must be 1")
    _require(record.get("status") == "ADJUDICATED_PASS", errors, "adjudication status is not ADJUDICATED_PASS")
    _require(record.get("adjudication_error") is None, errors, "adjudication contains an error")
    _require(record.get("mode") in (None, "timing"), errors, "adjudication is not a timing proof")
    _require(record.get("variant") == label, errors, "adjudication variant differs from manifest label")
    _require(record.get("run_id") == expected_run_id, errors, "adjudication run_id differs from the pinned plan")
    _require(metadata.get("run_id") == expected_run_id, errors, "raw metadata run_id differs from the pinned plan")
    _require(_absolute_path(record.get("raw_attempt")) == str(raw_run_path.resolve()), errors,
             "adjudication raw_attempt differs from listed raw run path")
    _require(_absolute_path(record.get("adjudication_path")) == str(adjudication_path.resolve()), errors,
             "adjudication_path in record differs from manifest proof path")
    _require(adjudication_path.resolve() == expected_adjudication_path.resolve(), errors,
             "adjudication path does not use the pinned timing namespace")

    _require(record.get("raw_receipts_unchanged") is True, errors,
             "adjudication does not confirm unchanged raw receipts")
    recorded_raw_hashes = record.get("raw_receipt_hashes")
    _require(isinstance(recorded_raw_hashes, dict), errors, "adjudication raw receipt hashes are missing")
    if isinstance(recorded_raw_hashes, dict):
        _require(recorded_raw_hashes == current_raw_hashes, errors,
                 "adjudication raw receipt hashes differ from current raw files")

    _require(record.get("original_wrapper_status") == raw_status, errors,
             "adjudication original wrapper status differs from raw status.txt")
    _require(record.get("original_exit_code") == (0 if raw_status == "PASS" else 1), errors,
             "adjudication original wrapper exit code differs from raw exit-code.txt")
    _require(record.get("original_failure") == raw_failure, errors,
             "adjudication original_failure differs from raw failure.json")
    _require(isinstance(raw_process, dict), errors, "raw process receipt is missing")
    _require(record.get("original_process") == raw_process, errors,
             "adjudication original_process differs from raw process.json")
    _require(isinstance(raw_process, dict) and raw_process.get("child_exit_code") == 0, errors,
             "native child did not exit zero")
    _require(isinstance(raw_process, dict) and raw_process.get("timeout") is False, errors,
             "native child process receipt reports a timeout")
    _require(isinstance(raw_process, dict) and raw_process.get("orphaned_process_group") is False, errors,
             "native child process group is orphaned")
    _require(isinstance(raw_process, dict) and raw_process.get("process_group_stopped") is True, errors,
             "native child process group was not stopped")

    if raw_status == "PASS":
        _require(label in {"SE", "SE-fused"}, errors,
                 "raw PASS is only expected for explicit-selector SE variants")
        _require(raw_exit_code == "0", errors, "raw PASS wrapper exit code must remain 0")
        _require(raw_failure is None, errors, "raw PASS attempt must not contain failure.json")
        _require(isinstance(raw_completion, dict), errors, "raw PASS completion receipt is missing")
        if isinstance(raw_completion, dict):
            _require(raw_completion.get("status") == "PASS", errors,
                     "raw completion receipt does not report PASS")
            _require(raw_completion.get("source_unchanged") is True, errors,
                     "raw completion receipt does not confirm unchanged source")
            _require(raw_completion.get("config_unchanged") is True, errors,
                     "raw completion receipt does not confirm unchanged config")
            resolved_hash = current_raw_hashes.get("native/config/resolved_config.yaml")
            _require(raw_completion.get("resolved_config_sha256") == resolved_hash, errors,
                     "raw completion resolved-config hash differs from raw YAML")
        _require(raw_summary == summary, errors,
                 "raw PASS parser summary differs from adjudicator parser summary")
        _require(record.get("known_strict_guard_failure") is None, errors,
                 "raw PASS proof must not claim a strict nullable-guard failure")
        _require(record.get("nullable_exception") is None, errors,
                 "raw PASS proof must not claim a nullable-selector exception")
    elif raw_status == "FAIL":
        _require(label in RAW_FAIL_VARIANTS, errors,
                 "raw FAIL is only accepted for A/B/C/D0/S nullable-selector variants")
        _require(raw_exit_code == "1", errors, "known raw FAIL wrapper exit code must remain 1")
        _require(raw_completion is None, errors,
                 "known nullable-guard failure must not have a fabricated completion receipt")
        _require(raw_summary is None, errors,
                 "known nullable-guard failure must not have a raw parser summary")
        _require(record.get("known_strict_guard_failure") == NULLABLE_STAGE_GUARD, errors,
                 "adjudication does not identify the exact nullable-stage guard")
        _require(isinstance(raw_failure, dict) and raw_failure.get("status") == "FAIL", errors,
                 "raw failure receipt is missing or does not report FAIL")
        _require(isinstance(raw_failure, dict) and raw_failure.get("error") == NULLABLE_STAGE_FAILURE, errors,
                 "raw failure is not the known nullable-stage guard defect")
        _require(isinstance(raw_failure, dict) and raw_failure.get("timed_out") is False, errors,
                 "raw failure receipt indicates a timeout")
        nullable = record.get("nullable_exception")
        _require(isinstance(nullable, dict), errors, "adjudication nullable-exception proof is missing")
        if isinstance(nullable, dict):
            _require(nullable.get("field") == "parallelism.gfsdp_row_owner_embedding_optimization_stages",
                     errors, "nullable-exception proof names the wrong field")
            evidence = nullable.get("evidence")
            expected_evidence = {
                "typed_default_none",
                "typed_annotation_nullable",
                "resolved_yaml_omits_selector",
                "explicit_child_null_override",
                "native_typed_decode_is_none",
                "all_rank_module_stage_receipts_valid",
            }
            _require(isinstance(evidence, dict) and set(evidence) == expected_evidence
                     and all(value is True for value in evidence.values()), errors,
                     "nullable-exception evidence is incomplete or false")
    else:
        errors.append("raw wrapper status must be PASS or the exact accepted nullable-guard FAIL")

    native = record.get("actual_native_completion")
    _require(isinstance(native, dict), errors, "adjudication native-completion proof is missing")
    if isinstance(native, dict):
        _require(native.get("status") == "complete", errors, "adjudicated native completion is not complete")
        _require(native.get("training_steps") == EXPECTED_STEPS, errors,
                 "adjudicated native completion does not contain 140 updates")
        _require(native.get("errors") == [], errors,
                 "adjudicated parser summary errors must use the canonical empty-list schema")
        for field in ("status", "training_steps", "window", "tokens", "memory", "errors"):
            _require(native.get(field) == summary.get(field), errors,
                     f"adjudication native-completion {field} differs from parser summary")

    _require(_absolute_path(record.get("parser_summary_path")) == str(summary_path.resolve()), errors,
             "parser_summary_path does not name the sibling external parser summary")
    _require(record.get("parser_summary_sha256") == summary_sha256, errors,
             "parser summary hash differs from the external summary file")
    _require(summary.get("run_id") == expected_run_id, errors,
             "external parser summary run_id differs from the pinned plan")

    _require(_absolute_path(record.get("launch_plan_path")) == str(plan_path.resolve()), errors,
             "adjudication launch-plan path differs from the pinned repair plan")
    _require(record.get("launch_plan_sha256") == expected_plan_sha256 == plan_sha256, errors,
             "adjudication launch-plan hash differs from the pinned repair plan")
    _require(plan.get("source_freeze_sha256") == expected_source_freeze_sha256, errors,
             "repair plan source freeze differs from the pinned source freeze")
    _require(record.get("source_freeze_sha256") == expected_source_freeze_sha256, errors,
             "adjudication source-freeze hash differs from the pinned source freeze")
    source_freeze = record.get("source_freeze")
    _require(isinstance(source_freeze, dict), errors, "adjudication source-freeze content is missing")
    if isinstance(source_freeze, dict):
        _require(_json_sha256(source_freeze) == expected_source_freeze_sha256, errors,
                 "adjudication source-freeze content does not match the pinned hash")
        _require(source_freeze.get("gift_source_snapshot") == metadata.get("source_snapshot_before"), errors,
                 "raw source snapshot differs from the pinned source freeze")
        base_config_hashes = source_freeze.get("base_config_hashes")
        config_before = metadata.get("config_hashes_before")
        _require(isinstance(base_config_hashes, dict) and isinstance(config_before, dict), errors,
                 "source/config freeze identity receipts are missing")
        if isinstance(base_config_hashes, dict) and isinstance(config_before, dict):
            for key, value in base_config_hashes.items():
                _require(config_before.get(key) == value, errors,
                         f"raw config hash {key} differs from source freeze")
        flash_identity = source_freeze.get("flash_source_identity")
        bridge_sources = flash_identity.get("bridge_sources") if isinstance(flash_identity, dict) else None
        _require(isinstance(config_before, dict)
                 and config_before.get("flash_qla_bridge_sources") == bridge_sources, errors,
                 "raw FlashQLA bridge config hashes differ from source freeze")
        expected_runtime = _expected_authenticated_runtime_files(source_freeze)
        _require(expected_runtime is not None, errors,
                 "source freeze lacks explicit executed-helper or bridge-source hashes")
        _require(record.get("authenticated_runtime_files") == expected_runtime, errors,
                 "adjudication authenticated runtime-file manifest differs from source freeze")
        expected_stages = record.get("validated_stage_receipts")
        expected_module_count = STAGE_MODULE_COUNTS[label]
        _require(isinstance(expected_stages, dict)
                 and expected_stages.get("rank_count") == 8
                 and expected_stages.get("module_count_per_rank") == expected_module_count,
                 errors, "adjudication stage proof has the wrong rank or module count")
        names_by_rank = expected_stages.get("module_names_by_rank") if isinstance(expected_stages, dict) else None
        _require(isinstance(names_by_rank, dict)
                 and set(names_by_rank) == {str(rank) for rank in RANKS},
                 errors, "adjudication stage proof does not cover all eight ranks")
        if isinstance(names_by_rank, dict):
            for rank in RANKS:
                names = names_by_rank.get(str(rank))
                _require(isinstance(names, list) and len(names) == expected_module_count,
                         errors, f"adjudication rank {rank} module count differs from the variant")

    _require(isinstance(source_final, dict) and source_final == metadata.get("source_snapshot_before"), errors,
             "raw source snapshot before/after differs")
    _require(isinstance(config_hashes_final, dict)
             and config_hashes_final == metadata.get("config_hashes_before"), errors,
             "raw configuration hashes before/after differ")
    _require(metadata.get("source_head_before") == (source_final or {}).get("head"), errors,
             "raw source HEAD before/after differs")

    valid_digest = re.compile(r"[0-9a-f]{64}")
    _require(isinstance(manifest_method_sha256, str) and bool(valid_digest.fullmatch(manifest_method_sha256)),
             errors, "manifest adjudicator_method_sha256 is missing or malformed")
    _require(record.get("adjudicator_method_sha256") == manifest_method_sha256, errors,
             "adjudication method hash differs from manifest pin")
    _require(current_method_sha256 == manifest_method_sha256, errors,
             "current adjudicator method hash differs from manifest pin")
    return errors


def percentile(values: Iterable[float], q: float) -> float:
    """Return a linear-interpolated percentile for q in [0, 1]."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile q must be in [0, 1]")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot summarize an empty series")
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p10": percentile(numbers, 0.10),
        "p90": percentile(numbers, 0.90),
        "min": min(numbers),
        "max": max(numbers),
    }


def primary_window_metrics(
    starts_ns: dict[int, int],
    ends_ns: dict[int, int],
    physical_tokens: int,
    supervised_tokens: int,
    measured_updates: int,
) -> dict[str, float | int]:
    """Compute the CUDA-drained same-host envelope and its token rates."""
    if set(starts_ns) != set(ends_ns) or not starts_ns:
        raise ValueError("window starts and ends must cover the same non-empty ranks")
    start_ns = min(starts_ns.values())
    end_ns = max(ends_ns.values())
    elapsed_s = (end_ns - start_ns) / 1_000_000_000
    if elapsed_s <= 0 or measured_updates <= 0:
        raise ValueError("window envelope and measured update count must be positive")
    return {
        "global_envelope_s": elapsed_s,
        "mean_update_s": elapsed_s / measured_updates,
        "physical_tokens_global": physical_tokens,
        "supervised_tokens_global": supervised_tokens,
        "physical_tokens_per_second_global": physical_tokens / elapsed_s,
        "supervised_tokens_per_second_global": supervised_tokens / elapsed_s,
        "envelope_start_monotonic_ns": start_ns,
        "envelope_end_monotonic_ns": end_ns,
    }


def validate_cadence_steps(records: list[dict[str, Any]], expected_steps: Iterable[int]) -> list[str]:
    observed = [record.get("step") for record in records]
    expected = list(expected_steps)
    if observed == expected:
        return []
    return [f"cadence steps must be {expected[0]}..{expected[-1]} in order; observed {observed}"]


def _rank_map(value: Any) -> dict[int, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, Any] = {}
    for key, item in value.items():
        try:
            result[int(key)] = item
        except (TypeError, ValueError):
            continue
    return result


def _rank_hint(path: Path, line: str) -> int | None:
    for pattern in (r"\[rank(\d+)\]", r"\[default(\d+)\]"):
        match = re.search(pattern, line)
        if match:
            return int(match.group(1))
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part.startswith("attempt_") and index + 1 < len(parts):
            candidate = parts[index + 1]
            if candidate.isdigit():
                return int(candidate)
    return None


def _extract_logs(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    gc_events: dict[tuple[Any, ...], dict[str, Any]] = {}
    stage_markers: dict[int, dict[str, Any]] = {
        rank: {"tuples": [], "occurrences": []} for rank in RANKS
    }
    log_hashes: dict[str, str] = {}
    for path in sorted((run_root / "logs").rglob("*.log")) if (run_root / "logs").exists() else []:
        relative = str(path.relative_to(run_root))
        content = path.read_bytes()
        log_hashes[relative] = hashlib.sha256(content).hexdigest()
        for line_number, raw_line in enumerate(content.decode("utf-8", errors="replace").splitlines(), 1):
            line = raw_line.rstrip()
            rank = _rank_hint(path, line)
            stage_match = STAGE_PATTERN.search(line)
            if stage_match:
                try:
                    stages = ast.literal_eval(stage_match.group(1))
                except (SyntaxError, ValueError):
                    stages = None
                if isinstance(stages, tuple) and all(isinstance(stage, str) for stage in stages):
                    if rank in stage_markers:
                        entry = stage_markers[rank]
                        stage_list = list(stages)
                        if stage_list not in entry["tuples"]:
                            entry["tuples"].append(stage_list)
                        entry["occurrences"].append(
                            {"stages": stage_list, "log": relative, "line": line_number}
                        )
            gc_match = GC_PATTERN.search(line)
            if gc_match:
                explicit_update = EXPLICIT_UPDATE_PATTERN.search(line)
                event = {
                    "log_timestamp": gc_match.group("timestamp"),
                    "rank": rank,
                    "event": gc_match.group("event"),
                    "duration_s": float(gc_match.group("seconds")),
                    "explicit_update": int(explicit_update.group(1)) if explicit_update else None,
                    "line": line,
                    "sources": [],
                }
                key = (
                    event["log_timestamp"],
                    rank,
                    event["event"],
                    event["duration_s"],
                    event["explicit_update"],
                )
                existing = gc_events.setdefault(key, event)
                existing["sources"].append({"log": relative, "line": line_number})
    return (
        {str(rank): stage_markers[rank] for rank in RANKS},
        {"events": list(gc_events.values()), "event_count": len(gc_events)},
        log_hashes,
    )


def _stage_actual_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    modules = receipt.get("modules", [])
    tuples = []
    for module in modules if isinstance(modules, list) else []:
        stages = module.get("optimization_stages") if isinstance(module, dict) else None
        if isinstance(stages, list) and stages not in tuples:
            tuples.append(stages)
    return {
        "source": "native_stage_selection_receipt",
        "tuples": tuples,
        "expected_stages": receipt.get("expected_stages"),
        "expected_module_count": receipt.get("expected_module_count"),
        "modules": modules,
        "source_sha256": receipt.get("source_sha256"),
    }


def _is_stage_selector(key: Any) -> bool:
    normalized = str(key).lower()
    return "stage" in normalized and (
        "embedding" in normalized or "optimization" in normalized
    ) or normalized in {"optimization_stages", "embedding_stages"}


def _controls_without_stage_selector(controls: Any) -> Any:
    if not isinstance(controls, dict):
        return controls
    return {key: value for key, value in controls.items() if not _is_stage_selector(key)}


def _stage_selector_fields(controls: Any) -> dict[str, Any]:
    if not isinstance(controls, dict):
        return {}
    return {key: value for key, value in controls.items() if _is_stage_selector(key)}


def validate_stage_selection_receipts(run: dict[str, Any], label: str) -> list[str]:
    """Validate the authoritative per-rank constructor-stage receipts."""
    errors: list[str] = []
    run_id = run.get("run_id") or "unknown"
    prefix = f"{label} ({run_id})"
    if label not in STAGE_MODULE_COUNTS:
        return [f"{prefix}: unsupported timing variant for stage receipt validation"]
    receipts = _rank_map(run.get("embedding_stage_selection_receipts_by_rank"))
    _require(set(receipts) == set(RANKS), errors,
             f"{prefix}: embedding-stage-selection receipts must cover ranks 0..7")
    frozen_source_hashes = run.get("expected_frozen_schedule_source_sha256")
    _require(isinstance(frozen_source_hashes, dict) and bool(frozen_source_hashes), errors,
             f"{prefix}: frozen schedule-source hash map is missing")
    if isinstance(frozen_source_hashes, dict):
        _require(all(
            isinstance(path, str)
            and isinstance(digest, str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", digest))
            for path, digest in frozen_source_hashes.items()
        ), errors, f"{prefix}: frozen schedule-source hash map is malformed")
    expected_count = STAGE_MODULE_COUNTS[label]
    for rank in RANKS:
        receipt = receipts.get(rank)
        if not isinstance(receipt, dict):
            continue
        rank_prefix = f"{prefix}: rank {rank} stage-selection receipt"
        _require(receipt.get("schema_version") == 1, errors, f"{rank_prefix} schema_version must be 1")
        _require(receipt.get("run_id") == run_id, errors, f"{rank_prefix} run_id differs")
        _require(receipt.get("rank") == rank, errors, f"{rank_prefix} rank field differs")
        _require(receipt.get("world_size") == 8, errors, f"{rank_prefix} world_size must be 8")
        _require(_normal_label(str(receipt.get("variant", ""))) == _normal_label(label), errors,
                 f"{rank_prefix} variant does not match {label}")
        source_hash = receipt.get("source_sha256")
        _require(isinstance(source_hash, dict), errors,
                 f"{rank_prefix} source_sha256 must be a path-to-digest map")
        if isinstance(source_hash, dict):
            _require(source_hash == frozen_source_hashes, errors,
                     f"{rank_prefix} source_sha256 differs from frozen schedule sources")
            _require(all(
                isinstance(path, str)
                and isinstance(digest, str)
                and bool(re.fullmatch(r"[0-9a-f]{64}", digest))
                for path, digest in source_hash.items()
            ), errors, f"{rank_prefix} source_sha256 map is malformed")
        _require(receipt.get("capture_stage") == STAGE_RECEIPT_CAPTURE, errors,
                 f"{rank_prefix} capture_stage is invalid")
        _require(type(receipt.get("device_reads")) is int and receipt.get("device_reads") == 0, errors,
                 f"{rank_prefix} device_reads must be 0")
        expected_stages = list(STAGE_TUPLES[label])
        _require(receipt.get("expected_stages") == expected_stages, errors,
                 f"{rank_prefix} expected_stages must be {expected_stages!r}")
        expected_count = receipt.get("expected_module_count")
        modules = receipt.get("modules")
        _require(type(expected_count) is int and expected_count == STAGE_MODULE_COUNTS[label], errors,
                 f"{rank_prefix} expected_module_count must be {STAGE_MODULE_COUNTS[label]}")
        _require(isinstance(modules, list), errors, f"{rank_prefix} modules must be a list")
        if not isinstance(modules, list):
            continue
        _require(len(modules) == STAGE_MODULE_COUNTS[label], errors,
                 f"{rank_prefix} modules must contain {STAGE_MODULE_COUNTS[label]} entries")
        names = []
        for module in modules:
            if not isinstance(module, dict):
                errors.append(f"{rank_prefix} has a malformed module entry")
                continue
            name = module.get("name")
            _require(isinstance(name, str) and bool(name), errors, f"{rank_prefix} module name is missing")
            if isinstance(name, str):
                names.append(name)
            _require(module.get("optimization_stages") == expected_stages, errors,
                     f"{rank_prefix} module {name!r} stages differ from expected {expected_stages!r}")
        _require(len(names) == len(set(names)), errors, f"{rank_prefix} has duplicate module names")
    return errors


def _keys_from(snapshot: Any, source: str, field: str) -> set[str]:
    if not isinstance(snapshot, dict):
        return set()
    source_data = snapshot.get("sources", {}).get(source, {})
    value = source_data.get(field, {}) if isinstance(source_data, dict) else {}
    return {str(key) for key in value} if isinstance(value, dict) else set()


def _cache_boundary(rank_record: dict[str, Any]) -> dict[str, Any]:
    observer = rank_record.get("measurement", {}).get("compilation_observer")
    if not isinstance(observer, dict):
        return {"available": False, "sources": {}}
    before = observer.get("before")
    after = observer.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {"available": False, "sources": {}, "source_metadata": observer.get("source_metadata")}
    source_fields = {
        "triton": ("kernel_entry_ids", "kernel_cache_ids"),
        "cute": ("entries",),
    }
    sources: dict[str, Any] = {}
    for source, fields in source_fields.items():
        diffs = {}
        for field in fields:
            old = _keys_from(before, source, field)
            new = _keys_from(after, source, field)
            diffs[field] = {
                "added": sorted(new - old),
                "removed": sorted(old - new),
                "added_count": len(new - old),
                "removed_count": len(old - new),
            }
        sources[source] = diffs
    return {
        "available": True,
        "before_errors": before.get("snapshot_errors", []),
        "after_errors": after.get("snapshot_errors", []),
        "comparison": observer.get("comparison", {}),
        "source_metadata": observer.get("source_metadata", {}),
        "sources": sources,
    }


def _file_ref(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": _sha256(path)}


def _pinned_repair_plan() -> tuple[dict[str, Any], str]:
    path = REPAIR_PLAN_PATH
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"pinned repair plan is missing or unsafe: {path}")
    digest = _sha256(path)
    if digest != REPAIR_PLAN_SHA256:
        raise ReceiptError(f"repair plan hash differs from pinned value: {digest}")
    plan = _read_json(path)
    if (
        plan is None
        or plan.get("schema_version") != 1
        or plan.get("source_freeze_sha256") != SOURCE_FREEZE_SHA256
        or plan.get("retained_A_run_id") != "schedule-embedding-a-timing-140-003"
    ):
        raise ReceiptError("repair plan does not pin the accepted source freeze and retained A")
    return plan, digest


def _run_id_from_repair_plan(plan: dict[str, Any], label: str) -> str:
    if label == "A":
        run_id = plan.get("retained_A_run_id")
    else:
        matches = [
            row for row in plan.get("runs", [])
            if isinstance(row, dict) and row.get("name") == f"{label}-timing"
        ]
        run_id = matches[0].get("run_id") if len(matches) == 1 else None
    if not isinstance(run_id, str) or not run_id:
        raise ReceiptError(f"repair plan has no unique timing run for {label}")
    return run_id


def _current_raw_receipt_hashes(root: Path, wrapper_status: str) -> dict[str, str]:
    """Hash the same bounded raw-receipt set as adjudicate.py, without logs."""
    fixed_names = (
        "config/experiment.yaml",
        "config-hashes-final.json",
        "exit-code.txt",
        "failure.json",
        "flash-qla-bridge.json",
        "metadata.json",
        "native/config/resolved_config.yaml",
        "process.json",
        "schedule-embedding-bridge.json",
        "source-final.json",
        "completion.json",
        "summary.json",
        "status.txt",
    )
    rank_names = tuple(
        relative
        for rank in range(8)
        for relative in (
            f"measurement/rank-{rank:02d}.json",
            f"measurement/embedding-stage-selection-rank-{rank:02d}.json",
            f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json",
            f"measurement/triton-startup-rank-{rank:02d}.json",
        )
    )
    required = {
        "config/experiment.yaml",
        "config-hashes-final.json",
        "exit-code.txt",
        "flash-qla-bridge.json",
        "metadata.json",
        "native/config/resolved_config.yaml",
        "process.json",
        "schedule-embedding-bridge.json",
        "source-final.json",
        "status.txt",
        *(relative
          for rank in range(8)
          for relative in (
              f"measurement/rank-{rank:02d}.json",
              f"measurement/embedding-stage-selection-rank-{rank:02d}.json",
              f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json",
              f"measurement/triton-startup-rank-{rank:02d}.json",
          )),
    }
    if wrapper_status == "FAIL":
        required.add("failure.json")
    elif wrapper_status == "PASS":
        required.update(("completion.json", "summary.json"))
    else:
        raise ReceiptError(f"unsupported raw wrapper status for timing proof: {wrapper_status!r}")
    result: dict[str, str] = {}
    for relative in (*fixed_names, *rank_names):
        path = root / relative
        if not path.exists():
            if relative in required:
                raise ReceiptError(f"adjudicated raw receipt is missing: {path}")
            continue
        if path.is_symlink() or not path.is_file():
            raise ReceiptError(f"adjudicated raw receipt is unsafe: {path}")
        result[relative] = _sha256(path)
    return result


def _load_adjudication(
    raw_root: Path,
    *,
    label: str,
    adjudication_path: Path,
    manifest_method_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, plan_sha256 = _pinned_repair_plan()
    expected_run_id = _run_id_from_repair_plan(plan, label)
    expected_raw_path = (RAW_ROOT / expected_run_id).resolve()
    if raw_root.resolve() != expected_raw_path:
        raise ReceiptError(f"listed raw run path is not the plan-pinned attempt {expected_raw_path}")

    expected_adjudication_path = ADJUDICATION_ROOT / label / "adjudication.json"
    if adjudication_path.is_symlink() or adjudication_path.resolve() != expected_adjudication_path.resolve():
        raise ReceiptError(f"adjudication path must name the pinned proof {expected_adjudication_path}")
    if not adjudication_path.is_file():
        raise ReceiptError(f"adjudication record is missing: {adjudication_path}")
    record = _read_json(adjudication_path)
    if record is None:
        raise ReceiptError(f"adjudication record is empty: {adjudication_path}")

    summary_path_value = record.get("parser_summary_path")
    if not isinstance(summary_path_value, str):
        raise ReceiptError("adjudication parser_summary_path is missing")
    summary_path = Path(summary_path_value)
    expected_summary_path = adjudication_path.parent / "summary.json"
    if summary_path.is_symlink() or summary_path.resolve() != expected_summary_path.resolve():
        raise ReceiptError(f"parser summary path must name the sibling file {expected_summary_path}")
    if not summary_path.is_file():
        raise ReceiptError(f"external parser summary is missing: {summary_path}")
    summary = _read_json(summary_path)
    if summary is None:
        raise ReceiptError(f"external parser summary is empty: {summary_path}")
    summary_sha256 = _sha256(summary_path)

    raw_status_path = raw_root / "status.txt"
    raw_exit_path = raw_root / "exit-code.txt"
    try:
        raw_status = raw_status_path.read_text(encoding="utf-8").strip()
        raw_exit_code = raw_exit_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ReceiptError(f"cannot read raw wrapper status/exit code: {exc}") from exc
    raw_failure = _read_json(raw_root / "failure.json")
    raw_process = _read_json(raw_root / "process.json")
    raw_completion = _read_json(raw_root / "completion.json")
    raw_summary = _read_json(raw_root / "summary.json")
    metadata = _read_json(raw_root / "metadata.json")
    source_final = _read_json(raw_root / "source-final.json")
    config_hashes_final = _read_json(raw_root / "config-hashes-final.json")
    if metadata is None:
        raise ReceiptError("adjudicated raw metadata receipt is missing")

    try:
        current_raw_hashes = _current_raw_receipt_hashes(raw_root, raw_status)
        current_method_sha256 = _sha256(ADJUDICATOR_METHOD_PATH)
    except OSError as exc:
        raise ReceiptError(f"cannot authenticate adjudication inputs: {exc}") from exc
    errors = validate_adjudication_record(
        record,
        label=label,
        expected_run_id=expected_run_id,
        raw_run_path=raw_root,
        raw_status=raw_status,
        raw_exit_code=raw_exit_code,
        raw_failure=raw_failure,
        raw_process=raw_process,
        raw_completion=raw_completion,
        raw_summary=raw_summary,
        metadata=metadata,
        source_final=source_final,
        config_hashes_final=config_hashes_final,
        summary=summary,
        summary_path=summary_path,
        summary_sha256=summary_sha256,
        adjudication_path=adjudication_path,
        expected_adjudication_path=expected_adjudication_path,
        plan=plan,
        plan_path=REPAIR_PLAN_PATH,
        plan_sha256=plan_sha256,
        manifest_method_sha256=manifest_method_sha256,
        current_method_sha256=current_method_sha256,
        current_raw_hashes=current_raw_hashes,
        expected_source_freeze_sha256=SOURCE_FREEZE_SHA256,
        expected_plan_sha256=REPAIR_PLAN_SHA256,
    )
    if errors:
        raise ReceiptError("adjudication proof rejected:\n- " + "\n- ".join(errors))

    proof = {
        "status": record["status"],
        "variant": record["variant"],
        "run_id": record["run_id"],
        "adjudication_path": str(adjudication_path.resolve()),
        "adjudication_record_sha256": _sha256(adjudication_path),
        "raw_attempt": record["raw_attempt"],
        "original_wrapper_status": record["original_wrapper_status"],
        "original_exit_code": record["original_exit_code"],
        "raw_wrapper_status": raw_status,
        "raw_wrapper_exit_code": raw_exit_code,
        "original_failure": record["original_failure"],
        "known_strict_guard_failure": record.get("known_strict_guard_failure"),
        "nullable_exception": record.get("nullable_exception"),
        "raw_receipts_unchanged": record["raw_receipts_unchanged"],
        "raw_summary_matches_external": raw_status == "PASS" and raw_summary == summary,
        "raw_receipt_hashes": record["raw_receipt_hashes"],
        "launch_plan_path": record["launch_plan_path"],
        "launch_plan_sha256": record["launch_plan_sha256"],
        "source_freeze_sha256": record["source_freeze_sha256"],
        "adjudicator_method_sha256": record["adjudicator_method_sha256"],
        "parser_summary_path": record["parser_summary_path"],
        "parser_summary_sha256": record["parser_summary_sha256"],
        "authenticated_runtime_files": record["authenticated_runtime_files"],
        "source_freeze_schedule_source_sha256": record["source_freeze"]["schedule_source_sha256"],
        "validated_stage_receipts": record.get("validated_stage_receipts"),
    }
    return summary, proof


def extract_receipt(
    run_path: str | Path,
    *,
    adjudication_path: str | Path | None = None,
    label: str | None = None,
    adjudicator_method_sha256: str | None = None,
) -> dict[str, Any]:
    """Extract a run without applying the seven-run 140-update acceptance gate."""
    root = Path(run_path).expanduser().resolve()
    if not root.is_dir():
        raise ReceiptError(f"run path is not a directory: {root}")
    metadata = _read_json(root / "metadata.json")
    if metadata is None:
        raise ReceiptError(f"run {root} is missing required receipt: metadata.json")
    raw_summary_path = root / "summary.json"
    raw_summary_present = raw_summary_path.is_file()
    raw_summary = _read_json(raw_summary_path) if adjudication_path is None else None
    adjudication = None
    if adjudication_path is None:
        if raw_summary is None:
            raise ReceiptError(f"run {root} is missing required receipt: summary.json")
        summary = raw_summary
        acceptance_mode = "native_completion"
    else:
        if label not in LABELS:
            raise ReceiptError("adjudicated extraction requires a known manifest label")
        summary, adjudication = _load_adjudication(
            root,
            label=label,
            adjudication_path=Path(adjudication_path).expanduser().resolve(),
            manifest_method_sha256=adjudicator_method_sha256,
        )
        acceptance_mode = (
            "adjudicated_raw_pass"
            if adjudication["raw_wrapper_status"] == "PASS"
            else "adjudicated_nullable_guard_fail"
        )

    optional_names = (
        "completion.json",
        "source-final.json",
        "config-hashes-final.json",
        "schedule-embedding-bridge.json",
        "flash-qla-bridge.json",
    )
    optional = {name: _read_json(root / name) for name in optional_names}
    sidecars: dict[int, dict[str, Any]] = {}
    sidecar_paths: dict[int, Path] = {}
    missing_rank_sidecars = []
    for rank in RANKS:
        path = root / "measurement" / f"rank-{rank:02d}.json"
        value = _read_json(path)
        if value is None:
            missing_rank_sidecars.append(rank)
        else:
            sidecars[rank] = value
            sidecar_paths[rank] = path

    bootstrap: dict[int, dict[str, Any]] = {}
    for rank in RANKS:
        path = root / "measurement" / f"flash-qla-bootstrap-rank-{rank:02d}.json"
        value = _read_json(path)
        if value is not None:
            bootstrap[rank] = value

    stage_receipts: dict[int, dict[str, Any]] = {}
    missing_stage_receipts = []
    for rank in RANKS:
        path = root / "measurement" / f"embedding-stage-selection-rank-{rank:02d}.json"
        value = _read_json(path)
        if value is None:
            missing_stage_receipts.append(rank)
        else:
            stage_receipts[rank] = value

    window = summary.get("window", {})
    start_after = window.get("start_after_update", metadata.get("warmup_updates"))
    end_after = window.get("end_after_update", metadata.get("steps"))
    cadence_by_rank: dict[str, list[dict[str, Any]]] = {}
    measured_by_rank: dict[str, list[dict[str, Any]]] = {}
    window_by_rank: dict[str, dict[str, Any]] = {}
    cache_by_rank: dict[str, dict[str, Any]] = {}
    for rank in RANKS:
        sidecar = sidecars.get(rank)
        if sidecar is None:
            continue
        cadence = sidecar.get("cadence", [])
        if not isinstance(cadence, list):
            cadence = []
        cadence_by_rank[str(rank)] = cadence
        measured_by_rank[str(rank)] = [
            item
            for item in cadence
            if isinstance(item, dict)
            and isinstance(start_after, int)
            and isinstance(end_after, int)
            and start_after < item.get("step", -1) <= end_after
        ]
        measurement = sidecar.get("measurement")
        if isinstance(measurement, dict):
            window_by_rank[str(rank)] = measurement
        cache_by_rank[str(rank)] = _cache_boundary(sidecar)

    logged_stages, gc, log_hashes = _extract_logs(root)
    stage_source_hashes_by_json = {
        json.dumps(value.get("source_sha256"), sort_keys=True): value.get("source_sha256")
        for value in stage_receipts.values()
        if isinstance(value.get("source_sha256"), dict)
    }
    stage_source_hashes = list(stage_source_hashes_by_json.values())
    if not missing_stage_receipts:
        actual_stages = {
            str(rank): _stage_actual_from_receipt(stage_receipts[rank]) for rank in RANKS
        }
    elif (
        metadata.get("steps") == 20
        and summary.get("run_id") == HISTORICAL_SE_FIXTURE_RUN_ID
        and not stage_receipts
    ):
        actual_stages = {
            rank: {**record, "source": "historical_20_step_log_fallback"}
            for rank, record in logged_stages.items()
        }
    else:
        actual_stages = {
            str(rank): (
                _stage_actual_from_receipt(stage_receipts[rank])
                if rank in stage_receipts
                else {"source": "missing_native_stage_receipt", "tuples": [], "modules": []}
            )
            for rank in RANKS
        }
    refs: dict[str, dict[str, str]] = {}
    fixed_paths = [root / "metadata.json"]
    if raw_summary_present:
        fixed_paths.append(root / "summary.json")
    fixed_paths += [root / name for name in optional_names]
    fixed_paths += [root / "exit-code.txt", root / "status.txt"]
    fixed_paths += list(sidecar_paths.values())
    fixed_paths += [
        root / "measurement" / f"flash-qla-bootstrap-rank-{rank:02d}.json"
        for rank in bootstrap
    ]
    fixed_paths += [
        root / "measurement" / f"embedding-stage-selection-rank-{rank:02d}.json"
        for rank in stage_receipts
    ]
    for path in fixed_paths:
        if path.is_file():
            refs[str(path.relative_to(root))] = {"sha256": _sha256(path)}
    refs.update({path: {"sha256": digest} for path, digest in log_hashes.items()})

    token_data = summary.get("tokens", {})
    source_before = metadata.get("source_snapshot_before")
    source_final = optional["source-final.json"]
    config_before = metadata.get("config_hashes_before")
    config_final = optional["config-hashes-final.json"]
    window_starts = {}
    window_ends = {}
    for rank, measurement in ((int(rank), value) for rank, value in window_by_rank.items()):
        try:
            window_starts[rank] = int(measurement["start_monotonic_ns"])
            window_ends[rank] = int(measurement["end_monotonic_ns"])
        except (KeyError, TypeError, ValueError):
            continue
    actual_physical = sum(
        int(value.get("physical_input_tokens_local", 0))
        for value in window_by_rank.values()
        if isinstance(value, dict)
    )
    supervised_values = {
        int(value["supervised_tokens_global"])
        for value in window_by_rank.values()
        if isinstance(value, dict) and value.get("supervised_tokens_global") is not None
    }
    actual_supervised = next(iter(supervised_values)) if len(supervised_values) == 1 else None
    try:
        primary = primary_window_metrics(
            window_starts,
            window_ends,
            actual_physical,
            actual_supervised or 0,
            int(window.get("updates", metadata.get("window_updates", 0))),
        )
    except (ValueError, TypeError):
        primary = None

    memory = summary.get("memory", {})
    memory_by_rank = _rank_map(memory.get("by_rank"))
    allocator_by_rank = _rank_map(memory.get("allocator_counter_deltas_by_rank"))
    memory_by_rank_normalized = {
        str(rank): {
            "peak_bytes": record.get("peak_bytes", {}),
            "peak_gib": record.get("peak_gib", {}),
            "allocator_counter_deltas": record.get("allocator_counter_deltas", allocator_by_rank.get(rank, {})),
            "allocator_counters_start": record.get("allocator_counters_start", {}),
            "allocator_counters_end": record.get("allocator_counters_end", {}),
        }
        for rank, record in sorted(memory_by_rank.items())
        if isinstance(record, dict)
    }

    return {
        "run_path": str(root),
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "mode": summary.get("mode"),
        "acceptance_mode": acceptance_mode,
        "raw_status": _read_text_or_none(root / "status.txt"),
        "raw_exit_code": _read_text_or_none(root / "exit-code.txt"),
        "raw_failure": _read_json(root / "failure.json"),
        "raw_process": _read_json(root / "process.json"),
        "raw_summary_present": raw_summary_present,
        "adjudication": adjudication,
        "summary_training_steps": summary.get("training_steps"),
        "summary_world_size": summary.get("world_size"),
        "metadata": metadata,
        "summary": summary,
        "completion": optional["completion.json"],
        "source_final": source_final,
        "config_hashes_final": config_final,
        "schedule_embedding_bridge": optional["schedule-embedding-bridge.json"],
        "flash_qla_bridge": optional["flash-qla-bridge.json"],
        "rank_sidecars": {str(rank): record for rank, record in sorted(sidecars.items())},
        "missing_rank_sidecars": missing_rank_sidecars,
        "bootstrap_by_rank": {str(rank): value for rank, value in sorted(bootstrap.items())},
        "embedding_stage_selection_receipts_by_rank": {
            str(rank): value for rank, value in sorted(stage_receipts.items())
        },
        "missing_embedding_stage_selection_receipts": missing_stage_receipts,
        "embedding_stage_source_sha256": stage_source_hashes[0] if len(stage_source_hashes) == 1 else None,
        "expected_frozen_schedule_source_sha256": (
            adjudication.get("source_freeze_schedule_source_sha256") if adjudication else None
        ),
        "cadence_by_rank": cadence_by_rank,
        "measured_cadence_by_rank": measured_by_rank,
        "window_measurement_by_rank": window_by_rank,
        "primary": primary,
        "tokens_from_summary": token_data,
        "actual_physical_tokens_from_rank_measurements": actual_physical,
        "actual_supervised_tokens_from_rank_measurements": actual_supervised,
        "memory_by_rank": memory_by_rank_normalized,
        "memory_maximum_peak_bytes_from_parser": memory.get("maximum_peak_bytes_across_ranks", {}),
        "allocator_counter_deltas_by_rank_from_parser": {
            str(rank): value for rank, value in sorted(allocator_by_rank.items())
        },
        "cache_observations_by_rank": cache_by_rank,
        "cache_summary": summary.get("compilation_observer", {}),
        "actual_embedding_stages_by_rank": actual_stages,
        "embedding_stage_log_observations_by_rank": logged_stages,
        "gc": gc,
        "log_hashes": log_hashes,
        "raw_refs": refs,
        "identity": {
            "source_snapshot_before": source_before,
            "source_final": source_final,
            "config_hashes_before": config_before,
            "config_hashes_final": config_final,
            "resolved_config_sha256": (optional["completion.json"] or {}).get("resolved_config_sha256")
            or ((adjudication or {}).get("raw_receipt_hashes", {}).get("native/config/resolved_config.yaml")),
            "input_data": metadata.get("input_data"),
            "seed": metadata.get("seed"),
            "runtime": metadata.get("runtime"),
            "host": summary.get("host"),
            "parallelism_features": metadata.get("requested_parallelism_features"),
            "num_prefetch_batches": metadata.get("num_prefetch_batches"),
            "gfsdp_reuse_bf16_expert_weights": metadata.get("gfsdp_reuse_bf16_expert_weights"),
            "compiler_cache_policy": metadata.get("compiler_cache_policy"),
            "compiler_cache_producer_run_id": metadata.get("compiler_cache_producer_run_id"),
            "argv": metadata.get("argv"),
            "schedule_embedding_bridge": optional["schedule-embedding-bridge.json"],
            "flash_qla_bridge": optional["flash-qla-bridge.json"],
        },
        "observed_window_updates": {
            "start_after_update": start_after,
            "end_after_update": end_after,
            "updates": window.get("updates", metadata.get("window_updates")),
            "first_observed_cache_update_interval": (
                f"({start_after},{end_after}]" if isinstance(start_after, int) and isinstance(end_after, int) else None
            ),
        },
    }


def _require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_run_receipt(run: dict[str, Any], label: str) -> list[str]:
    """Return strict acceptance failures for one listed matrix run."""
    errors: list[str] = []
    metadata = run["metadata"]
    summary = run["summary"]
    raw_completion = run.get("completion")
    completion = raw_completion or {}
    adjudication = run.get("adjudication") or {}
    acceptance_mode = run.get("acceptance_mode")
    adjudicated = acceptance_mode in {
        "adjudicated_raw_pass",
        "adjudicated_nullable_guard_fail",
    }
    window = summary.get("window", {})
    run_id = run.get("run_id") or run["run_path"]
    prefix = f"{label} ({run_id})"

    _require(summary.get("status") == "complete", errors, f"{prefix}: summary status is not complete")
    _require(summary.get("mode") == "complete_window", errors, f"{prefix}: summary mode is not complete_window")
    _require(summary.get("errors") == [], errors,
             f"{prefix}: parser summary errors must use the canonical empty-list schema")
    _require(summary.get("training_steps") == EXPECTED_STEPS, errors, f"{prefix}: summary steps must be 140")
    _require(metadata.get("steps") == EXPECTED_STEPS, errors, f"{prefix}: metadata steps must be 140")
    _require(metadata.get("warmup_updates") == EXPECTED_WARMUP, errors, f"{prefix}: metadata warmup must be 40")
    _require(metadata.get("window_updates") == EXPECTED_WINDOW, errors, f"{prefix}: metadata window must be 100")
    _require(metadata.get("run_id") == run_id, errors, f"{prefix}: metadata run_id differs from summary")
    _require(summary.get("world_size") == 8, errors, f"{prefix}: summary world_size must be 8")
    _require(window.get("warmup_updates") == 40, errors, f"{prefix}: summary warmup must be 40")
    _require(window.get("start_after_update") == 40, errors, f"{prefix}: window must start after update 40")
    _require(window.get("end_after_update") == 140, errors, f"{prefix}: window must end after update 140")
    _require(window.get("updates") == 100, errors, f"{prefix}: summary window must contain 100 updates")
    _require(window.get("monitoring_updates") == [], errors, f"{prefix}: monitored updates must be empty")
    _require(window.get("monitor_interval") == 0, errors, f"{prefix}: summary monitor interval must be zero")
    _require(metadata.get("monitor_interval") == 0, errors, f"{prefix}: metadata monitor interval must be zero")
    _require(metadata.get("profiler") == "disabled", errors, f"{prefix}: profiler must be disabled")
    _require(metadata.get("memory_snapshots") == "disabled", errors, f"{prefix}: memory snapshots must be disabled")
    _require(metadata.get("checkpointing") == "disabled", errors, f"{prefix}: checkpointing must be disabled")
    _require(metadata.get("validation") == "disabled", errors, f"{prefix}: validation must be disabled")
    _require("disabled" in str(metadata.get("detailed_monitoring", "")).lower(), errors,
             f"{prefix}: detailed monitoring must be disabled")

    if adjudicated:
        raw_failure = run.get("raw_failure")
        _require(adjudication.get("status") == "ADJUDICATED_PASS", errors,
                 f"{prefix}: external adjudication proof is not accepted")
        _require(adjudication.get("raw_receipts_unchanged") is True, errors,
                 f"{prefix}: adjudication does not preserve raw receipts")
        _require(isinstance(run.get("raw_process"), dict)
                 and run["raw_process"].get("child_exit_code") == 0
                 and run["raw_process"].get("timeout") is False
                 and run["raw_process"].get("orphaned_process_group") is False
                 and run["raw_process"].get("process_group_stopped") is True,
                 errors, f"{prefix}: native child process proof is incomplete")
        if acceptance_mode == "adjudicated_raw_pass":
            _require(label in {"SE", "SE-fused"}, errors,
                     f"{prefix}: raw PASS is only expected for explicit-selector SE variants")
            _require(run.get("raw_status") == "PASS", errors,
                     f"{prefix}: raw PASS status was not retained")
            _require(run.get("raw_exit_code") == "0", errors,
                     f"{prefix}: raw PASS exit code 0 was not retained")
            _require(raw_failure is None and adjudication.get("original_failure") is None, errors,
                     f"{prefix}: raw PASS proof must retain no failure receipt")
            _require(isinstance(raw_completion, dict) and completion.get("status") == "PASS", errors,
                     f"{prefix}: raw PASS completion receipt is missing")
            _require(completion.get("source_unchanged") is True
                     and completion.get("config_unchanged") is True, errors,
                     f"{prefix}: raw PASS completion identity checks failed")
            _require(run.get("raw_summary_present") is True
                     and adjudication.get("raw_summary_matches_external") is True, errors,
                     f"{prefix}: raw PASS parser summary differs from external summary")
            _require(adjudication.get("known_strict_guard_failure") is None
                     and adjudication.get("nullable_exception") is None, errors,
                     f"{prefix}: raw PASS proof must not claim the nullable exception")
            resolved_hash = completion.get("resolved_config_sha256")
            _require(bool(re.fullmatch(r"[0-9a-f]{64}", str(resolved_hash))), errors,
                     f"{prefix}: raw PASS resolved config SHA256 is missing or malformed")
            _require(run["identity"].get("resolved_config_sha256") == resolved_hash, errors,
                     f"{prefix}: raw PASS resolved config hash differs from adjudication proof")
        else:
            _require(label in RAW_FAIL_VARIANTS, errors,
                     f"{prefix}: raw FAIL is only accepted for A/B/C/D0/S")
            _require(run.get("raw_status") == "FAIL", errors,
                     f"{prefix}: raw wrapper FAIL status was not retained")
            _require(run.get("raw_exit_code") == "1", errors,
                     f"{prefix}: raw wrapper exit code 1 was not retained")
            _require(isinstance(raw_failure, dict)
                     and raw_failure.get("error") == NULLABLE_STAGE_FAILURE
                     and raw_failure.get("status") == "FAIL"
                     and raw_failure.get("timed_out") is False, errors,
                     f"{prefix}: raw wrapper failure is not the known nullable-stage guard")
            _require(raw_completion is None and run.get("raw_summary_present") is False, errors,
                     f"{prefix}: nullable-guard raw FAIL must have no completion or summary")
            _require(adjudication.get("known_strict_guard_failure") == NULLABLE_STAGE_GUARD
                     and isinstance(adjudication.get("nullable_exception"), dict), errors,
                     f"{prefix}: nullable-guard proof is missing its exact exception")
            _require(bool(re.fullmatch(r"[0-9a-f]{64}", str(
                run["identity"].get("resolved_config_sha256", "")
            ))), errors, f"{prefix}: adjudicated resolved config SHA256 is missing or malformed")
    else:
        _require(completion.get("status") == "PASS", errors, f"{prefix}: native completion receipt is not PASS")
        _require(completion.get("source_unchanged") is True, errors,
                 f"{prefix}: source changed during native run")
        _require(completion.get("config_unchanged") is True, errors,
                 f"{prefix}: configuration changed during native run")
        _require(bool(re.fullmatch(r"[0-9a-f]{64}", str(completion.get("resolved_config_sha256", "")))), errors,
                 f"{prefix}: resolved config SHA256 is missing or malformed")
    before_source = metadata.get("source_snapshot_before")
    final_source = run.get("source_final")
    _require(isinstance(before_source, dict) and before_source == final_source, errors,
             f"{prefix}: source snapshot before/after differs or is missing")
    _require(metadata.get("source_head_before") == (final_source or {}).get("head"), errors,
             f"{prefix}: source HEAD before/after differs")
    before_config = metadata.get("config_hashes_before")
    final_config = run.get("config_hashes_final")
    _require(isinstance(before_config, dict) and before_config == final_config, errors,
             f"{prefix}: config hashes before/after differ or are missing")
    exit_code = run.get("raw_exit_code")
    if not adjudicated:
        _require(exit_code == "0", errors, f"{prefix}: native launcher exit code is not zero")

    _require(not run["missing_rank_sidecars"], errors,
             f"{prefix}: rank sidecars are missing for ranks {run['missing_rank_sidecars']}")
    sidecars = _rank_map(run.get("rank_sidecars"))
    _require(set(sidecars) == set(RANKS), errors, f"{prefix}: rank sidecars do not cover ranks 0..7")
    cadence_by_rank = _rank_map(run.get("cadence_by_rank"))
    _require(set(cadence_by_rank) == set(RANKS), errors, f"{prefix}: cadence does not cover ranks 0..7")
    summary_window_ranks = _rank_map(window.get("per_rank"))
    _require(set(summary_window_ranks) == set(RANKS), errors, f"{prefix}: parser window does not cover ranks 0..7")
    summary_cadence_ranks = _rank_map(summary.get("host_boundary_cadence", {}).get("by_rank"))
    _require(set(summary_cadence_ranks) == set(RANKS), errors,
             f"{prefix}: parser cadence summary does not cover ranks 0..7")

    start_ns: dict[int, int] = {}
    end_ns: dict[int, int] = {}
    physical_by_rank: dict[int, int] = {}
    supervised_by_rank: dict[int, int] = {}
    for rank in RANKS:
        sidecar = sidecars.get(rank)
        if not isinstance(sidecar, dict):
            continue
        rank_prefix = f"{prefix}: rank {rank}"
        _require(sidecar.get("rank") == rank, errors, f"{rank_prefix} sidecar rank field differs")
        _require(sidecar.get("world_size") == 8, errors, f"{rank_prefix} sidecar world_size must be 8")
        _require(sidecar.get("run_id") == run_id, errors, f"{rank_prefix} sidecar run_id differs")
        _require(sidecar.get("training_status") == "complete", errors, f"{rank_prefix} training did not complete")
        _require(sidecar.get("completed_steps") == 140 and sidecar.get("expected_steps") == 140,
                 errors, f"{rank_prefix} did not complete all 140 updates")
        _require(sidecar.get("measurement_status") == "complete", errors,
                 f"{rank_prefix} measurement is not complete")
        cadence = cadence_by_rank.get(rank, [])
        if isinstance(cadence, list):
            errors.extend(f"{rank_prefix}: {message}" for message in validate_cadence_steps(cadence, range(1, 141)))
            _require(len(cadence) == 140, errors, f"{rank_prefix} cadence must retain all 140 updates")
            if len(cadence) >= EXPECTED_STEPS:
                _require(cadence[39].get("window_boundary") == "start", errors,
                         f"{rank_prefix} update 40 lacks the start-window boundary marker")
                _require(cadence[139].get("window_boundary") == "end", errors,
                         f"{rank_prefix} update 140 lacks the end-window boundary marker")
                for item in cadence[1:]:
                    _require(isinstance(item.get("interval_s"), (int, float)) and item["interval_s"] > 0,
                             errors, f"{rank_prefix} has a missing/nonpositive cadence interval at update {item.get('step')}")
        measurement = sidecar.get("measurement")
        if not isinstance(measurement, dict):
            errors.append(f"{rank_prefix} measurement object is missing")
            continue
        for key, expected in (
            ("warmup_updates", 40),
            ("window_updates", 100),
            ("start_after_update", 40),
            ("end_after_update", 140),
            ("cuda_synchronize_calls", 2),
        ):
            _require(measurement.get(key) == expected, errors,
                     f"{rank_prefix} {key} must be {expected}")
        _require(measurement.get("monitoring_updates") == [], errors,
                 f"{rank_prefix} measurement monitoring updates must be empty")
        try:
            start_ns[rank] = int(measurement["start_monotonic_ns"])
            end_ns[rank] = int(measurement["end_monotonic_ns"])
            physical_by_rank[rank] = int(measurement["physical_input_tokens_local"])
            supervised_by_rank[rank] = int(measurement["supervised_tokens_global"])
            local_seconds = (end_ns[rank] - start_ns[rank]) / 1_000_000_000
            _require(end_ns[rank] > start_ns[rank], errors, f"{rank_prefix} timestamps are not increasing")
            _require(abs(local_seconds - float(measurement["local_elapsed_s"])) <= 1e-6,
                     errors, f"{rank_prefix} local elapsed time disagrees with timestamps")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{rank_prefix} window measurement fields are malformed: {exc}")
        _require(physical_by_rank.get(rank) == LOCAL_PHYSICAL_TOKENS, errors,
                 f"{rank_prefix} physical token count must be {LOCAL_PHYSICAL_TOKENS}")
        _require(supervised_by_rank.get(rank) == EXPECTED_SUPERVISED_TOKENS, errors,
                 f"{rank_prefix} supervised token count must be {EXPECTED_SUPERVISED_TOKENS}")
        parsed_window = summary_window_ranks.get(rank)
        if isinstance(parsed_window, dict) and rank in start_ns and rank in end_ns:
            _require(parsed_window.get("start_monotonic_ns") == start_ns[rank], errors,
                     f"{rank_prefix} parser start timestamp disagrees with raw measurement")
            _require(parsed_window.get("end_monotonic_ns") == end_ns[rank], errors,
                     f"{rank_prefix} parser end timestamp disagrees with raw measurement")
            _require(parsed_window.get("physical_input_tokens_local") == physical_by_rank[rank], errors,
                     f"{rank_prefix} parser physical tokens disagree with raw measurement")
            _require(parsed_window.get("supervised_tokens_global") == supervised_by_rank[rank], errors,
                     f"{rank_prefix} parser supervised tokens disagree with raw measurement")
        parser_steps = summary_cadence_ranks.get(rank, {}).get("intervals", [])
        if isinstance(parser_steps, list):
            expected_measured_steps = list(range(41, 141))
            errors.extend(f"{rank_prefix}: parser {message}" for message in validate_cadence_steps(parser_steps, expected_measured_steps))
            raw_measured = {
                item.get("step"): item.get("interval_s")
                for item in cadence if isinstance(item, dict) and item.get("step") in range(41, 141)
            }
            for parser_record in parser_steps:
                step = parser_record.get("step")
                raw_interval = raw_measured.get(step)
                parsed_interval = parser_record.get("interval_s")
                _require(
                    isinstance(raw_interval, (int, float))
                    and isinstance(parsed_interval, (int, float))
                    and abs(raw_interval - parsed_interval) <= 1e-9,
                    errors,
                    f"{rank_prefix} parser cadence differs from raw sidecar at update {step}",
                )

    _require(set(start_ns) == set(RANKS) and set(end_ns) == set(RANKS), errors,
             f"{prefix}: raw window timestamps do not cover all eight ranks")
    if set(physical_by_rank) == set(RANKS):
        _require(sum(physical_by_rank.values()) == EXPECTED_PHYSICAL_TOKENS, errors,
                 f"{prefix}: measured physical tokens must total {EXPECTED_PHYSICAL_TOKENS}")
    _require(len(set(supervised_by_rank.values())) == 1, errors,
             f"{prefix}: all ranks must report the same supervised token total")
    _require(set(supervised_by_rank.values()) == {EXPECTED_SUPERVISED_TOKENS}, errors,
             f"{prefix}: measured supervised tokens must total {EXPECTED_SUPERVISED_TOKENS}")
    tokens = summary.get("tokens", {})
    _require(tokens.get("physical_input_tokens_global") == EXPECTED_PHYSICAL_TOKENS, errors,
             f"{prefix}: parser physical tokens must be {EXPECTED_PHYSICAL_TOKENS}")
    _require(tokens.get("supervised_tokens_global") == EXPECTED_SUPERVISED_TOKENS, errors,
             f"{prefix}: parser supervised tokens must be {EXPECTED_SUPERVISED_TOKENS}")
    if set(start_ns) == set(RANKS) and set(end_ns) == set(RANKS):
        expected_primary = primary_window_metrics(
            start_ns, end_ns, EXPECTED_PHYSICAL_TOKENS, EXPECTED_SUPERVISED_TOKENS, EXPECTED_WINDOW
        )
        _require(abs(float(window.get("primary_elapsed_s", -1)) - expected_primary["global_envelope_s"]) <= 1e-6,
                 errors, f"{prefix}: parser primary time differs from same-host global envelope")
        for field, expected in (
            ("physical_input_tokens_per_second_global", expected_primary["physical_tokens_per_second_global"]),
            ("supervised_tokens_per_second_global", expected_primary["supervised_tokens_per_second_global"]),
        ):
            _require(isinstance(tokens.get(field), (int, float)) and abs(tokens[field] - expected) <= 1e-6,
                     errors, f"{prefix}: parser {field} disagrees with the global envelope")
        _require(run.get("primary") is not None, errors, f"{prefix}: primary window metrics could not be derived")

    memory_by_rank = _rank_map(run.get("memory_by_rank"))
    allocator_by_rank = _rank_map(run.get("allocator_counter_deltas_by_rank_from_parser"))
    _require(set(memory_by_rank) == set(RANKS), errors, f"{prefix}: parser memory peaks do not cover ranks 0..7")
    _require(set(allocator_by_rank) == set(RANKS), errors,
             f"{prefix}: parser allocator deltas do not cover ranks 0..7")
    required_peak_fields = ("allocated_bytes.all.peak", "reserved_bytes.all.peak")
    for rank in RANKS:
        record = memory_by_rank.get(rank, {})
        peak_bytes = record.get("peak_bytes", {})
        counters = allocator_by_rank.get(rank, {})
        for field in required_peak_fields:
            _require(isinstance(peak_bytes.get(field), int) and peak_bytes[field] >= 0, errors,
                     f"{prefix}: rank {rank} lacks exact {field} memory peak")
        for field in ("num_alloc_retries", "num_ooms"):
            _require(isinstance(counters.get(field), int) and counters[field] >= 0, errors,
                     f"{prefix}: rank {rank} lacks {field} delta")
    parser_memory_maxima = run.get("memory_maximum_peak_bytes_from_parser", {})
    for field in required_peak_fields:
        values = [
            memory_by_rank[rank].get("peak_bytes", {}).get(field)
            for rank in RANKS if rank in memory_by_rank
        ]
        if len(values) == len(RANKS) and all(isinstance(value, int) for value in values):
            _require(parser_memory_maxima.get(field) == max(values), errors,
                     f"{prefix}: parser maximum {field} disagrees with per-rank peaks")

    _require(metadata.get("seed") == 2048, errors, f"{prefix}: seed must be 2048")
    input_data = metadata.get("input_data")
    _require(isinstance(input_data, dict), errors, f"{prefix}: input identity is missing")
    if isinstance(input_data, dict):
        for field in ("dataset_sha256", "data_receipt_sha256", "tokenizer_file_sha256"):
            _require(bool(input_data.get(field)), errors, f"{prefix}: input identity lacks {field}")
    _require(metadata.get("observe_compiler_cache") is True, errors,
             f"{prefix}: compiler cache boundary observer is not enabled")
    cache_summary = run.get("cache_summary", {})
    cache_ranks = _rank_map(cache_summary.get("by_rank"))
    _require(set(cache_ranks) == set(RANKS), errors, f"{prefix}: compiler observer summary misses ranks")
    cache_observations = _rank_map(run.get("cache_observations_by_rank"))
    _require(set(cache_observations) == set(RANKS), errors, f"{prefix}: compiler boundary snapshots miss ranks")
    for rank in RANKS:
        _require(cache_observations.get(rank, {}).get("available") is True, errors,
                 f"{prefix}: rank {rank} lacks compiler boundary snapshots")
    interval = run.get("observed_window_updates", {}).get("first_observed_cache_update_interval")
    _require(interval == "(40,140]", errors, f"{prefix}: cache boundary interval must be (40,140]")

    bootstrap = _rank_map(run.get("bootstrap_by_rank"))
    _require(set(bootstrap) == set(RANKS), errors, f"{prefix}: native bootstrap receipts do not cover all ranks")
    for rank in RANKS:
        record = bootstrap.get(rank, {})
        _require(record.get("rank") == rank and record.get("native_main_completed") is True,
                 errors, f"{prefix}: rank {rank} native bootstrap did not confirm completion")

    errors.extend(validate_stage_selection_receipts(run, label))

    schedule_bridge = run.get("schedule_embedding_bridge")
    if label in ("SE", "SE-fused"):
        _require(isinstance(schedule_bridge, dict), errors,
                 f"{prefix}: schedule-embedding runtime control receipt is missing")
        _require(isinstance(run.get("flash_qla_bridge"), dict), errors,
                 f"{prefix}: native feature-control receipt is missing")
        controls = schedule_bridge.get("controls") if isinstance(schedule_bridge, dict) else None
        _require(isinstance(controls, dict), errors,
                 f"{prefix}: schedule-embedding controls are missing")
        if isinstance(controls, dict):
            _require(controls.get("gfsdp_row_owner_embedding_optimizations") is False, errors,
                     f"{prefix}: aggregate embedding optimization boolean must be false for explicit stage selection")
            selectors = _stage_selector_fields(controls)
            expected_stages = list(STAGE_TUPLES[label])
            _require(bool(selectors), errors,
                     f"{prefix}: explicit embedding stage selector list is missing from controls")
            for key, stages in selectors.items():
                _require(stages == expected_stages, errors,
                         f"{prefix}: control {key!r} must select {expected_stages!r}")
    if isinstance(schedule_bridge, dict) and schedule_bridge.get("variant") is not None:
        bridge_label = _normal_label(str(schedule_bridge["variant"]))
        _require(bridge_label == _normal_label(label), errors,
                 f"{prefix}: schedule-embedding bridge names {schedule_bridge['variant']!r}")

    return errors


def _normal_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _cadence_analysis(run: dict[str, Any]) -> dict[str, Any]:
    cadence = _rank_map(run.get("cadence_by_rank", {}))
    measured = _rank_map(run["measured_cadence_by_rank"])
    first_step = int(run["observed_window_updates"]["start_after_update"]) + 1
    last_step = int(run["observed_window_updates"]["end_after_update"])
    full_ranks: dict[str, dict[int, float]] = {}
    full_series: dict[str, list[float | None]] = {}
    for rank in RANKS:
        rows = cadence.get(rank, [])
        values = {
            int(row["step"]): float(row["interval_s"])
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("step"), int)
            and isinstance(row.get("interval_s"), (int, float))
        }
        full_ranks[str(rank)] = values
        full_series[str(rank)] = [values.get(step) for step in range(1, last_step + 1)]
    measured_ranks: dict[str, dict[int, float]] = {}
    for rank in RANKS:
        rows = measured.get(rank, [])
        measured_ranks[str(rank)] = {
            int(row["step"]): float(row["interval_s"])
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("step"), int)
            and isinstance(row.get("interval_s"), (int, float))
        }
    all_values = [value for row in measured_ranks.values() for value in row.values()]
    rank_summaries = {
        rank: summarize(list(values.values())) for rank, values in measured_ranks.items() if values
    }
    windows = {
        "first20": (first_step, min(first_step + 19, last_step)),
        "last20": (max(first_step, last_step - 19), last_step),
    }
    rank_windows: dict[str, Any] = {}
    rank_changes = []
    for rank, values in measured_ranks.items():
        rank_windows[rank] = {}
        first_values = []
        last_values = []
        for name, (start, end) in windows.items():
            sample = [values[step] for step in range(start, end + 1) if step in values]
            rank_windows[rank][name] = {
                "updates": [start, end],
                "statistics": summarize(sample) if sample else None,
            }
            if name == "first20":
                first_values = sample
            else:
                last_values = sample
        if first_values and last_values:
            before = statistics.median(first_values)
            after = statistics.median(last_values)
            change_pct = (after / before - 1.0) * 100.0
            rank_changes.append({
                "rank": int(rank),
                "first20_median_s": before,
                "last20_median_s": after,
                "change_percent": change_pct,
                "direction": "slower" if change_pct > 0 else "faster" if change_pct < 0 else "unchanged",
            })

    bins = []
    bin_start = first_step
    while bin_start <= last_step:
        bin_end = min(bin_start + 9, last_step)
        values = [
            rank_values[step]
            for rank_values in measured_ranks.values()
            for step in range(bin_start, bin_end + 1)
            if step in rank_values
        ]
        per_rank = {}
        for rank, rank_values in measured_ranks.items():
            sample = [rank_values[step] for step in range(bin_start, bin_end + 1) if step in rank_values]
            per_rank[rank] = summarize(sample) if sample else None
        bins.append({
            "updates": [bin_start, bin_end],
            "all_rank_statistics": summarize(values) if values else None,
            "per_rank_statistics": per_rank,
        })
        bin_start = bin_end + 1

    warmup_bins = []
    for start, end in ((11, 20), (21, 30), (31, 40)):
        all_bin_values = [
            values[step]
            for values in full_ranks.values()
            for step in range(start, end + 1)
            if step in values
        ]
        per_rank = {
            rank: summarize([values[step] for step in range(start, end + 1) if step in values])
            if any(step in values for step in range(start, end + 1)) else None
            for rank, values in full_ranks.items()
        }
        warmup_bins.append({
            "updates": [start, end],
            "all_rank_statistics": summarize(all_bin_values) if all_bin_values else None,
            "per_rank_statistics": per_rank,
        })

    startup_values = [
        {"update": step, "rank": int(rank), "interval_s": values[step]}
        for rank, values in full_ranks.items()
        for step in range(1, 11)
        if step in values
    ]
    startup_summary = {
        "updates": [1, 10],
        "available_interval_updates": sorted({item["update"] for item in startup_values}),
        "statistics": summarize(item["interval_s"] for item in startup_values) if startup_values else None,
        "largest_interval": max(startup_values, key=lambda item: item["interval_s"]) if startup_values else None,
        "note": "Update 1 has no preceding cadence interval; all available update 1-10 intervals remain included.",
    }
    full_per_update = []
    for step in range(1, last_step + 1):
        values = {rank: row[step] for rank, row in full_ranks.items() if step in row}
        full_per_update.append({
            "update": step,
            "all_rank_statistics": summarize(values.values()) if values else None,
            "slowest_ranks": sorted(rank for rank, value in values.items() if value == max(values.values()))
            if values else [],
        })

    per_update = []
    straggler_counts = {str(rank): 0 for rank in RANKS}
    tie_updates = 0
    for step in range(first_step, last_step + 1):
        values = {rank: rank_values[step] for rank, rank_values in measured_ranks.items() if step in rank_values}
        if not values:
            continue
        maximum = max(values.values())
        leaders = sorted(rank for rank, value in values.items() if value == maximum)
        if len(leaders) > 1:
            tie_updates += 1
        for rank in leaders:
            straggler_counts[str(rank)] += 1
        per_update.append({
            "update": step,
            "all_rank_statistics": summarize(values.values()),
            "straggler_ranks": leaders,
        })
    direction_counts = {
        direction: sum(record["direction"] == direction for record in rank_changes)
        for direction in ("slower", "faster", "unchanged")
    }
    all_values = [value for row in measured_ranks.values() for value in row.values()]
    full_values = [value for row in full_ranks.values() for value in row.values()]
    return {
        "percentile_method": "linear interpolation at q*(n-1)",
        "full_training_updates": [1, last_step],
        "full_training_statistics": summarize(full_values) if full_values else None,
        "full_training_series_by_rank": full_series,
        "full_training_per_update_statistics": full_per_update,
        "startup_updates_1_10": startup_summary,
        "warmup_ten_update_bins": warmup_bins,
        "measured_updates": [first_step, last_step],
        "all_rank_statistics": summarize(all_values) if all_values else None,
        "per_rank_statistics": rank_summaries,
        "full_measured_series_by_rank": {
            rank: [values.get(step) for step in range(first_step, last_step + 1)]
            for rank, values in measured_ranks.items()
        },
        "ten_update_bins": bins,
        "first20_vs_last20": {
            "window_updates": {key: list(value) for key, value in windows.items()},
            "per_rank": rank_windows,
            "per_rank_median_changes": rank_changes,
            "direction_counts": direction_counts,
        },
        "synchronized_trend_and_stragglers": {
            "per_update_all_rank_statistics_and_stragglers": per_update,
            "straggler_rank_leader_counts": straggler_counts,
            "updates_with_tied_stragglers": tie_updates,
            "rank_first_last_direction_counts": direction_counts,
        },
    }


def compare_runs(candidate: dict[str, Any], reference: dict[str, Any], name: str) -> dict[str, Any]:
    candidate_primary = candidate["primary"]
    reference_primary = reference["primary"]
    if not candidate_primary or not reference_primary:
        raise ValueError(f"cannot compare runs without primary window metrics: {name}")
    throughput_fields = ("physical_tokens_per_second_global", "supervised_tokens_per_second_global")
    throughput = {}
    for field in throughput_fields:
        ratio = candidate_primary[field] / reference_primary[field]
        throughput[field] = {"ratio": ratio, "change_percent": (ratio - 1.0) * 100.0}
    time_ratio = candidate_primary["mean_update_s"] / reference_primary["mean_update_s"]
    memory_changes = {}
    for key in ("allocated_bytes.all.peak", "reserved_bytes.all.peak"):
        candidate_value = max(
            record.get("peak_bytes", {}).get(key, 0)
            for record in candidate["memory_by_rank"].values()
        )
        reference_value = max(
            record.get("peak_bytes", {}).get(key, 0)
            for record in reference["memory_by_rank"].values()
        )
        ratio = candidate_value / reference_value if reference_value else None
        memory_changes[key] = {
            "candidate_max_bytes": candidate_value,
            "reference_max_bytes": reference_value,
            "candidate_max_gib": candidate_value / (1024 ** 3),
            "reference_max_gib": reference_value / (1024 ** 3),
            "ratio": ratio,
            "change_percent": (ratio - 1.0) * 100.0 if ratio is not None else None,
        }
    return {
        "comparison": name,
        "candidate": candidate["label"],
        "reference": reference["label"],
        "throughput": throughput,
        "mean_update_time": {
            "candidate_s": candidate_primary["mean_update_s"],
            "reference_s": reference_primary["mean_update_s"],
            "ratio": time_ratio,
            "change_percent": (time_ratio - 1.0) * 100.0,
        },
        "memory_maxima": memory_changes,
        "allocator_deltas": {
            "candidate_by_rank": candidate["allocator_counter_deltas_by_rank_from_parser"],
            "reference_by_rank": reference["allocator_counter_deltas_by_rank_from_parser"],
        },
    }


def _common_campaign_errors(runs: dict[str, dict[str, Any]]) -> list[str]:
    errors = []
    first = runs[LABELS[0]]
    first_source = first["identity"].get("source_snapshot_before")
    first_input = first["identity"].get("input_data")
    for label in LABELS[1:]:
        run = runs[label]
        _require(run["identity"].get("source_snapshot_before") == first_source, errors,
                 f"{label}: source snapshot differs from A")
        _require(run["identity"].get("source_final") == first["identity"].get("source_final"), errors,
                 f"{label}: final source identity differs from A")
        _require(run["identity"].get("input_data") == first_input, errors,
                 f"{label}: input/tokenizer identity differs from A")
        _require(run["identity"].get("seed") == first["identity"].get("seed"), errors,
                 f"{label}: seed differs from A")

    se = runs["SE"]
    fused = runs["SE-fused"]
    for field in (
        "input_data", "seed", "runtime", "parallelism_features", "num_prefetch_batches",
        "gfsdp_reuse_bf16_expert_weights", "compiler_cache_policy", "compiler_cache_producer_run_id",
    ):
        _require(se["identity"].get(field) == fused["identity"].get(field), errors,
                 f"SE and SE-fused differ in fixed configuration field {field}")
    se_flash = se["identity"].get("flash_qla_bridge") or {}
    fused_flash = fused["identity"].get("flash_qla_bridge") or {}
    for field in ("backend_variant", "routing_metadata", "cross_entropy", "capture_mode"):
        _require(se_flash.get(field) == fused_flash.get(field), errors,
                 f"SE and SE-fused differ in FlashQLA control {field}")
    se_schedule = se["identity"].get("schedule_embedding_bridge") or {}
    fused_schedule = fused["identity"].get("schedule_embedding_bridge") or {}
    _require(se_schedule.get("source_sha256") == fused_schedule.get("source_sha256"), errors,
             "SE and SE-fused schedule-embedding source hashes differ")
    _require(_controls_without_stage_selector(se_schedule.get("controls"))
             == _controls_without_stage_selector(fused_schedule.get("controls")), errors,
             "SE and SE-fused non-stage schedule-embedding controls differ")
    _require(se["embedding_stage_source_sha256"] == fused["embedding_stage_source_sha256"], errors,
             "SE and SE-fused stage-selection source hashes differ")
    return errors


def _published_run(run: dict[str, Any]) -> dict[str, Any]:
    """Drop raw nested summaries after retaining the fields needed for review."""
    schedule_bridge = run.get("schedule_embedding_bridge") or {}
    flash_bridge = run.get("flash_qla_bridge") or {}
    compiler_summary = run.get("cache_summary") or {}
    summary_by_rank = _rank_map(compiler_summary.get("by_rank"))
    raw_cache = _rank_map(run.get("cache_observations_by_rank"))
    cache_by_rank = {}
    for rank in RANKS:
        observed = raw_cache.get(rank, {})
        parsed = summary_by_rank.get(rank, {})
        cache_by_rank[str(rank)] = {
            "status": parsed.get("status"),
            "errors": parsed.get("errors", []),
            "source_coverage": parsed.get("source_coverage", {}),
            "source_statuses": {
                source: value.get("status")
                for source, value in parsed.get("sources", {}).items()
                if isinstance(value, dict)
            },
            "triton_summary_changed_keys": parsed.get("sources", {}).get("triton", {}).get("changed_keys", []),
            "raw_boundary_snapshot_errors": {
                "before": observed.get("before_errors", []),
                "after": observed.get("after_errors", []),
            },
            "key_differences": observed.get("sources", {}),
        }
    rank_windows = {}
    for rank, measurement in run.get("window_measurement_by_rank", {}).items():
        rank_windows[rank] = {
            key: measurement.get(key)
            for key in (
                "start_monotonic_ns", "end_monotonic_ns", "start_wall_time_ns", "end_wall_time_ns",
                "local_elapsed_s", "physical_input_tokens_local", "supervised_tokens_global",
                "warmup_updates", "window_updates", "start_after_update", "end_after_update",
                "cuda_synchronize_calls",
            )
        }
    return {
        "label": run["label"],
        "run_id": run["run_id"],
        "run_path": run["run_path"],
        "status": run["status"],
        "mode": run["mode"],
        "acceptance": {
            "mode": run["acceptance_mode"],
            "effective_native_summary_status": run["status"],
            "raw_wrapper_status": run.get("raw_status"),
            "raw_wrapper_exit_code": run.get("raw_exit_code"),
            "raw_failure": run.get("raw_failure"),
            "adjudication_proof": run.get("adjudication"),
        },
        "primary": run["primary"],
        "window_rank_measurements": rank_windows,
        "all_rank_cadence_records": run["cadence_by_rank"],
        "cadence_analysis": run["cadence_analysis"],
        "memory": {
            "scope": "native parser timing-window peaks; not a constructor-inclusive lifecycle peak",
            "by_rank": run["memory_by_rank"],
            "maximum_peak_bytes_across_ranks": run["memory_maximum_peak_bytes_from_parser"],
            "allocator_counter_deltas_by_rank": run["allocator_counter_deltas_by_rank_from_parser"],
        },
        "identity": {
            "source_snapshot_before": run["identity"].get("source_snapshot_before"),
            "source_final": run["identity"].get("source_final"),
            "config_hashes_before": run["identity"].get("config_hashes_before"),
            "config_hashes_final": run["identity"].get("config_hashes_final"),
            "resolved_config_sha256": run["identity"].get("resolved_config_sha256"),
            "input_data": run["identity"].get("input_data"),
            "seed": run["identity"].get("seed"),
            "runtime": run["identity"].get("runtime"),
            "host": run["identity"].get("host"),
        },
        "configuration_controls": {
            "requested_parallelism_features": run["identity"].get("parallelism_features"),
            "num_prefetch_batches": run["identity"].get("num_prefetch_batches"),
            "gfsdp_reuse_bf16_expert_weights": run["identity"].get("gfsdp_reuse_bf16_expert_weights"),
            "compiler_cache_policy": run["identity"].get("compiler_cache_policy"),
            "compiler_cache_producer_run_id": run["identity"].get("compiler_cache_producer_run_id"),
            "schedule_embedding_bridge": {
                key: schedule_bridge.get(key)
                for key in ("variant", "controls", "controls_sha256", "variant_sha256", "source_sha256", "config_hash_guard")
            } if schedule_bridge else None,
            "flash_qla_bridge": {
                key: flash_bridge.get(key)
                for key in ("backend_variant", "routing_metadata", "cross_entropy", "capture_mode", "source_identity")
            } if flash_bridge else None,
            "native_argv": run["identity"].get("argv"),
            "actual_embedding_stages_by_rank": run["actual_embedding_stages_by_rank"],
            "embedding_stage_selection_receipts_by_rank": run.get(
                "embedding_stage_selection_receipts_by_rank", {}
            ),
        },
        "compiler_cache_observations": {
            "status": compiler_summary.get("status"),
            "coverage_status": compiler_summary.get("coverage_status"),
            "coverage_limits": compiler_summary.get("limits", []),
            "unverified": compiler_summary.get("unverified", []),
            "first_observation_interval": run["observed_window_updates"].get("first_observed_cache_update_interval"),
            "semantics": run["cache_observation_semantics"],
            "per_rank": cache_by_rank,
        },
        "gc": run["gc"],
        "raw_artifact_sha256": run["raw_refs"],
    }


def analyze_campaign(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("runs")
    if not isinstance(rows, list):
        raise ReceiptError("manifest must contain a runs list")
    labels = [row.get("label") for row in rows if isinstance(row, dict)]
    if labels != list(LABELS):
        raise ReceiptError(f"manifest runs must be in this exact order: {', '.join(LABELS)}")
    manifest_path = Path(manifest.get("_manifest_path", ".")).resolve()
    missing_proofs = [
        str(row.get("label")) for row in rows
        if not isinstance(row.get("adjudication_path"), str) or not row.get("adjudication_path")
    ]
    if missing_proofs:
        raise ReceiptError(
            "seven-run campaign not accepted:\n- every final timing row requires its 004 adjudication_path; "
            f"missing for {', '.join(missing_proofs)}"
        )
    method_sha256 = manifest.get("adjudicator_method_sha256")
    if (
        not isinstance(method_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", method_sha256)
    ):
        raise ReceiptError("seven-run campaign not accepted: manifest must pin a 64-character adjudicator_method_sha256")
    extracted: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        label = str(row["label"])
        run_path = Path(str(row.get("path", ""))).expanduser()
        if not run_path.is_absolute():
            run_path = manifest_path.parent / run_path
        adjudication_value = row.get("adjudication_path")
        adjudication_path = None
        if adjudication_value:
            adjudication_path = Path(str(adjudication_value)).expanduser()
            if not adjudication_path.is_absolute():
                adjudication_path = manifest_path.parent / adjudication_path
        try:
            run = extract_receipt(
                run_path,
                adjudication_path=adjudication_path,
                label=label,
                adjudicator_method_sha256=method_sha256,
            )
            run["label"] = label
            extracted[label] = run
            errors.extend(validate_run_receipt(run, label))
        except ReceiptError as exc:
            errors.append(f"{label}: {exc}")
    if len(extracted) == len(LABELS):
        errors.extend(_common_campaign_errors(extracted))
    if errors:
        raise ReceiptError("seven-run campaign not accepted:\n- " + "\n- ".join(errors))

    for run in extracted.values():
        run["cadence_analysis"] = _cadence_analysis(run)
        run["cache_observation_semantics"] = (
            "Boundary deltas show keys newly observed in the stated interval. They do not identify an exact update, "
            "distinguish compilation from a cache load, or prove that a pre-existing key executed."
        )

    published_runs = {label: _published_run(extracted[label]) for label in LABELS}

    incremental = [compare_runs(extracted[candidate], extracted[reference], name)
                   for name, candidate, reference in INCREMENTAL_COMPARISONS]
    vs_a = [compare_runs(extracted[label], extracted["A"], f"{label}/A") for label in LABELS[1:]]
    return {
        "schema_version": 1,
        "status": "complete",
        "campaign_order": list(LABELS),
        "acceptance": {
            "warmup_updates": EXPECTED_WARMUP,
            "measured_updates": EXPECTED_WINDOW,
            "total_updates": EXPECTED_STEPS,
            "world_size": len(RANKS),
            "physical_tokens_global": EXPECTED_PHYSICAL_TOKENS,
            "supervised_tokens_global": EXPECTED_SUPERVISED_TOKENS,
            "primary_timer": "CUDA-drained single-host envelope: max(rank end monotonic timestamp) - min(rank start monotonic timestamp)",
            "primary_mean_update": "global envelope / 100 measured optimizer updates",
            "percentile_method": "linear interpolation at q*(n-1)",
        },
        "comparisons": {"incremental": incremental, "versus_A": vs_a},
        "runs": published_runs,
        "notes": [
            "The full 41-140 global envelope is primary; tail windows are supporting cadence analysis only.",
            "All 140 update cadence records are retained and plotted; update 1 has no preceding interval, and startup intervals are not discarded.",
            "Warmup cadence bins 11-20, 21-30, and 31-40 are supporting diagnostics for the early cadence hump.",
            "Per-update host cadence is a host interval after native Profiler.step and is not device-only time.",
            "Cache keys newly present after update 40 are bounded to first observation in (40,140]; no exact compile time is inferred.",
            "An existing process or disk cache entry does not establish execution. Use actual activation and the separate profile for path execution.",
            "GC update attribution is reported only when an update number appears explicitly in a native log line.",
        ],
        "plot_status": {"status": "pending", "artifacts": []},
    }


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def markdown_fragment(result: dict[str, Any]) -> str:
    lines = [
        "## Seven-run native timing results",
        "",
        "Primary time is the CUDA-drained single-host envelope across updates 41–140; mean update time is that envelope divided by 100. Physical and supervised TPS use the parser-validated global token totals.",
        "",
        "| Variant | Receipt acceptance | Physical TPS | Supervised TPS | Mean update (s) | Max allocated (GiB) | Max reserved (GiB) | Retries / OOMs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        run = result["runs"][label]
        acceptance = run["acceptance"]
        if acceptance["mode"].startswith("adjudicated_"):
            receipt_status = (
                f"ADJUDICATED_PASS (raw {acceptance['raw_wrapper_status']}/"
                f"{acceptance['raw_wrapper_exit_code']})"
            )
        else:
            receipt_status = f"native {acceptance['raw_wrapper_status']}/{acceptance['raw_wrapper_exit_code']}"
        primary = run["primary"]
        memory = run["memory"]
        maxima = memory["maximum_peak_bytes_across_ranks"]
        allocated = maxima["allocated_bytes.all.peak"] / (1024 ** 3)
        reserved = maxima["reserved_bytes.all.peak"] / (1024 ** 3)
        counters = memory["allocator_counter_deltas_by_rank"]
        retries = sum(value.get("num_alloc_retries", 0) for value in counters.values())
        ooms = sum(value.get("num_ooms", 0) for value in counters.values())
        lines.append(
            f"| {label} | {receipt_status} | {primary['physical_tokens_per_second_global']:.3f} | "
            f"{primary['supervised_tokens_per_second_global']:.3f} | {primary['mean_update_s']:.6f} | "
            f"{allocated:.3f} | {reserved:.3f} | {retries} / {ooms} |"
        )
    if any(
        result["runs"][label]["acceptance"]["mode"].startswith("adjudicated_")
        for label in LABELS
    ):
        lines.extend([
            "",
            "ADJUDICATED_PASS rows retain the original raw wrapper FAIL/exit code and cite an external adjudication record, parser-summary hash, pinned repair-plan hash, source-freeze hash, method hash, and unchanged raw-receipt hashes in the JSON output.",
        ])
    lines.extend([
        "",
        "### Incremental and cumulative comparisons",
        "",
        "Each throughput and memory value is candidate/reference; update-time values are candidate/reference, so negative time change is lower elapsed time.",
        "",
        "| Comparison | Physical TPS Δ | Supervised TPS Δ | Mean update time Δ | Max allocated Δ | Max reserved Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    comparison_rows = result["comparisons"]["incremental"] + result["comparisons"]["versus_A"]
    for row in comparison_rows:
        physical = row["throughput"]["physical_tokens_per_second_global"]["change_percent"]
        supervised = row["throughput"]["supervised_tokens_per_second_global"]["change_percent"]
        time_change = row["mean_update_time"]["change_percent"]
        allocated = row["memory_maxima"]["allocated_bytes.all.peak"]["change_percent"]
        reserved = row["memory_maxima"]["reserved_bytes.all.peak"]["change_percent"]
        lines.append(
            f"| {row['comparison']} | {_format_pct(physical)} | {_format_pct(supervised)} | "
            f"{_format_pct(time_change)} | {_format_pct(allocated)} | {_format_pct(reserved)} |"
        )
    lines.extend([
        "",
        "### Cadence, cache and GC evidence",
        "",
        "The JSON retains all 140 cadence records on every rank, full measured updates 41–140, per-rank first-20/last-20 statistics, measured ten-update bins, warmup bins 11–20/21–30/31–40, and per-update rank stragglers. The full-run plot includes startup intervals; update 1 is null because no preceding cadence exists. These host cadence intervals are not device-only times.",
        "",
        "Cache boundary changes identify keys newly observed in (40,140]; they do not locate an exact compile update, distinguish compile from load, or establish execution for a pre-existing key. GC events are extracted from native logs; update attribution is left blank unless logged explicitly.",
        "",
        f"Plot generation: {result['plot_status']['status']}.",
    ])
    if result["plot_status"].get("reason"):
        lines.append(f"Reason: {result['plot_status']['reason']}.")
    return "\n".join(lines) + "\n"


def _make_plots(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "unavailable", "reason": "matplotlib is not installed; no plot files were written", "artifacts": []}

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    palette = {label: plt.get_cmap("tab10")(index) for index, label in enumerate(LABELS)}

    fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    for index, label in enumerate(LABELS):
        axis = axes.flat[index]
        analysis = result["runs"][label]["cadence_analysis"]
        series = analysis["full_measured_series_by_rank"]
        steps = list(range(41, 141))
        by_step = [[series[str(rank)][position] for rank in RANKS] for position in range(100)]
        medians = [statistics.median(values) for values in by_step]
        maxima = [max(values) for values in by_step]
        axis.plot(steps, medians, color=palette[label], label="rank median")
        axis.plot(steps, maxima, color="black", alpha=0.55, linewidth=0.9, label="slowest rank")
        axis.set_title(label)
        axis.set_ylabel("Host cadence (s)")
        axis.grid(True, alpha=0.25)
    axes.flat[-1].axis("off")
    axes.flat[6].set_xlabel("Optimizer update")
    fig.suptitle("Measured per-update host cadence, updates 41–140")
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, output_dir, "per-update-cadence"))
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    for index, label in enumerate(LABELS):
        axis = axes.flat[index]
        analysis = result["runs"][label]["cadence_analysis"]
        series = analysis["full_training_series_by_rank"]
        steps = list(range(1, 141))
        by_step = [
            [series[str(rank)][position] for rank in RANKS if series[str(rank)][position] is not None]
            for position in range(140)
        ]
        medians = [statistics.median(values) if values else None for values in by_step]
        maxima = [max(values) if values else None for values in by_step]
        axis.plot(steps, medians, color=palette[label], label="rank median")
        axis.plot(steps, maxima, color="black", alpha=0.55, linewidth=0.9, label="slowest rank")
        startup = analysis["startup_updates_1_10"]["largest_interval"]
        if startup is not None:
            axis.scatter([startup["update"]], [startup["interval_s"]], marker="*", s=90, color="red", zorder=4)
            axis.annotate(
                f"startup max u{startup['update']} r{startup['rank']}",
                (startup["update"], startup["interval_s"]),
                xytext=(5, 6), textcoords="offset points", fontsize=7,
            )
        axis.set_title(label)
        axis.set_ylabel("Host cadence (s)")
        axis.grid(True, alpha=0.25)
    axes.flat[-1].axis("off")
    axes.flat[6].set_xlabel("Optimizer update")
    fig.suptitle("All 140 updates; startup intervals retained (update 1 has no prior interval)")
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, output_dir, "all-updates-cadence"))
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    warmup_centers = [15, 25, 35]
    for label in LABELS:
        bins = result["runs"][label]["cadence_analysis"]["warmup_ten_update_bins"]
        medians = [item["all_rank_statistics"]["median"] for item in bins]
        low = [item["all_rank_statistics"]["p10"] for item in bins]
        high = [item["all_rank_statistics"]["p90"] for item in bins]
        axis.plot(warmup_centers, medians, marker="o", color=palette[label], label=label)
        axis.fill_between(warmup_centers, low, high, color=palette[label], alpha=0.08)
    axis.set_xticks(warmup_centers, ["11–20", "21–30", "31–40"])
    axis.set_xlabel("Optimizer updates (ten-update warmup bin)")
    axis.set_ylabel("Host cadence (s)")
    axis.set_title("Warmup cadence bins across all ranks")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=4)
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, output_dir, "warmup-bins-11-40"))
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 6))
    for label in LABELS:
        bins = result["runs"][label]["cadence_analysis"]["ten_update_bins"]
        centers = [statistics.mean(item["updates"]) for item in bins]
        medians = [item["all_rank_statistics"]["median"] for item in bins]
        low = [item["all_rank_statistics"]["p10"] for item in bins]
        high = [item["all_rank_statistics"]["p90"] for item in bins]
        axis.plot(centers, medians, marker="o", color=palette[label], label=label)
        axis.fill_between(centers, low, high, color=palette[label], alpha=0.08)
    axis.set_xlabel("Optimizer update (ten-update bin center)")
    axis.set_ylabel("Host cadence (s)")
    axis.set_title("Ten-update cadence bins across all ranks")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=4)
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, output_dir, "ten-update-bins"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for index, label in enumerate(LABELS):
        per_rank = result["runs"][label]["cadence_analysis"]["first20_vs_last20"]["per_rank"]
        for rank in RANKS:
            first = per_rank[str(rank)]["first20"]["statistics"]["median"]
            last = per_rank[str(rank)]["last20"]["statistics"]["median"]
            axes[0].plot([index - 0.08, index + 0.08], [first, last], color=palette[label], alpha=0.7)
            axes[0].scatter([index - 0.08, index + 0.08], [first, last], color=palette[label], s=13)
    axes[0].set_xticks(range(len(LABELS)), LABELS, rotation=25)
    axes[0].set_ylabel("Per-rank median cadence (s)")
    axes[0].set_title("Updates 41–60 to 121–140, by rank")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(
        range(len(LABELS)),
        [result["runs"][label]["cadence_analysis"]["first20_vs_last20"]["direction_counts"]["slower"] for label in LABELS],
        color=[palette[label] for label in LABELS],
    )
    axes[1].set_xticks(range(len(LABELS)), LABELS, rotation=25)
    axes[1].set_ylabel("Ranks with higher last-20 median")
    axes[1].set_title("Direction shared across ranks")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, output_dir, "first20-vs-last20"))
    plt.close(fig)
    return {"status": "generated", "matplotlib_version": matplotlib.__version__, "artifacts": artifacts}


def _save_figure(fig: Any, output_dir: Path, stem: str) -> list[dict[str, str]]:
    outputs = []
    for extension in ("png", "svg"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=160 if extension == "png" else None, bbox_inches="tight")
        outputs.append({"path": str(path), "sha256": _sha256(path)})
    return outputs


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest is None:
        raise ReceiptError(f"manifest does not exist: {path}")
    manifest["_manifest_path"] = str(path.resolve())
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="JSON with seven ordered {label, path, adjudication_path} entries and a pinned adjudicator_method_sha256")
    parser.add_argument("--json-out", required=True, type=Path,
                        help="write the complete JSON result here after all strict gates pass")
    parser.add_argument("--markdown-out", type=Path,
                        help="write the Markdown report fragment here; defaults to stdout")
    parser.add_argument("--plots-dir", type=Path,
                        help="plot output directory; defaults beside --json-out")
    args = parser.parse_args(argv)
    try:
        result = analyze_campaign(_load_manifest(args.manifest))
    except ReceiptError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    json_path = args.json_out.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = args.plots_dir.expanduser().resolve() if args.plots_dir else json_path.parent / f"{json_path.stem}-plots"
    result["plot_status"] = _make_plots(result, plots_dir)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    fragment = markdown_fragment(result)
    if args.markdown_out:
        markdown_path = args.markdown_out.expanduser().resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(fragment, encoding="utf-8")
    else:
        sys.stdout.write(fragment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
