#!/usr/bin/env python3
"""Validate completed native timing attempts without changing their raw receipts."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, get_args, get_type_hints

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
RAW_ROOT = EVIDENCE / "artifacts/raw"
OUTPUT_ROOT = RAW_ROOT / "adjudication-140-004"
DIAGNOSTIC_OUTPUT_ROOT = RAW_ROOT / "adjudication-diagnostic-004"
EXPECTED_FREEZE_SHA256 = "3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd"
EXPECTED_PLAN_PATH = RAW_ROOT / "newlaunchrecords/repair-004-plan.json"
EXPECTED_PLAN_SHA256 = "1a193606963e2f3c7086a6afb1d6d56ad78692dae54121e257c05c18fc8a5968"
RETAINED_FREEZE_PATH = OUTPUT_ROOT.parent / "adjudication-140-003/A-recovery-003/adjudication.json"
EXPECTED_CAMPAIGN_SHA256 = "57c6a6b1dc527dd7581a5fa4e363965443dab98ad2d3226ae70aafa85b13dee8"
STAGES_FIELD = "gfsdp_row_owner_embedding_optimization_stages"
STRICT_NULLABLE_GUARD_ERROR = f"resolved config lacks parallelism.{STAGES_FIELD}"
KNOWN_NULLABLE_GUARD_ERROR = f"RuntimeError: {STRICT_NULLABLE_GUARD_ERROR}"
VARIANTS = ("A", "B", "C", "D0", "S", "SE", "SE-fused")
REPAIR_VARIANTS = ("B", "C", "D0", "S", "SE", "SE-fused")
REPAIR_TIMING_NAMES = tuple(f"{variant}-timing" for variant in REPAIR_VARIANTS)
REPAIR_DIAGNOSTIC_NAMES = tuple(f"{variant}-diagnostic" for variant in ("D0", "S", "SE", "SE-fused"))
TIMING_CADENCE = (140, 40, 100)
DIAGNOSTIC_CADENCE = (21, 20, 0)
DIAGNOSTIC_VARIANTS = ("D0", "S", "SE", "SE-fused")
DIAGNOSTIC_PROFILE_TAIL_STATUS = "PENDING_PERFETTO"
GLOBAL_PHYSICAL_TOKENS = 209_715_200
GLOBAL_SUPERVISED_TOKENS = 209_664_000
LOCAL_PHYSICAL_TOKENS = GLOBAL_PHYSICAL_TOKENS // 8
SAVED_D_CACHE = ROOT / "runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache"
EXPECTED_EXECUTED_HELPERS = {
    "production_data_contract": ROOT / "runs/production-data-abcd/data_contract.py",
    "overnight_runner": ROOT / "runs/overnight-streaming-perf/run.py",
    "overnight_measure": ROOT / "runs/overnight-streaming-perf/measure.py",
    "overnight_parser": ROOT / "runs/overnight-streaming-perf/parse_results.py",
    "compiler_observer": ROOT / "runs/overnight-streaming-perf/compiler_stats.py",
    "streaming_filelock": ROOT / "runs/overnight-streaming-perf/streaming_filelock.py",
    "historical_campaign_plan": ROOT / "runs/production-data-abcd/campaign-plan.json",
    "original_capacity_snapshot": ROOT / "runs/capacity-group-allreduce/capacity.original.py",
}
EXPECTED_FLASH_BRIDGE_FILES = (
    "runs/flash-qla/capture.py",
    "runs/flash-qla/measure.py",
    "runs/flash-qla/run.py",
)
EXPECTED_SCHEDULE_SOURCE_FILES = (
    "runs/schedule-embedding/campaign.py",
    "runs/schedule-embedding/measure.py",
    "runs/schedule-embedding/run.py",
)


def _expected_controls(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise AdjudicationError(f"unsupported timing variant {variant!r}")
    row_owner = variant not in {"A", "B", "C"}
    stages = {
        "SE": ["compact", "owner_add", "owner_lookup"],
        "SE-fused": ["compact", "owner_add", "owner_lookup", "routing"],
    }.get(variant)
    return {
        "gfsdp_split_mtp_ffn": row_owner,
        "gfsdp_split_layer0_ffn": row_owner,
        "gfsdp_row_owner_embeddings": row_owner,
        "gfsdp_mtp_transition_prefetch": variant in {"S", "SE", "SE-fused"},
        "gfsdp_row_owner_embedding_optimizations": False,
        STAGES_FIELD: stages,
    }


class AdjudicationError(RuntimeError):
    """The raw attempt does not satisfy the frozen campaign contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise AdjudicationError(f"required raw receipt is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _option_values(argv: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                raise AdjudicationError(f"{option} lacks a value in the pinned plan")
            values.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.partition("=")[2])
    return values


def _check_plan_argv(
    row: dict[str, Any], *, variant: str, mode: str, expected_run_id: str
) -> list[str]:
    argv = row.get("argv")
    if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
        raise AdjudicationError(f"{row.get('name')} lacks an argv string list")
    if (
        row.get("name") != f"{variant}-{mode}"
        or row.get("run_id") != expected_run_id
        or argv[:2]
        != [
            "/opt/torchtitan-venv/bin/python3.12",
            str(EVIDENCE / "run.py"),
        ]
        or _option_values(argv, "--variant") != [variant]
        or _option_values(argv, "--run-id") != [expected_run_id]
    ):
        raise AdjudicationError(f"{row.get('name')} argv identity differs from repair plan")
    expected_cadence = TIMING_CADENCE if mode == "timing" else DIAGNOSTIC_CADENCE
    for option, expected in zip(("--steps", "--warmup", "--window"), expected_cadence):
        if _option_values(argv, option) != [str(expected)]:
            raise AdjudicationError(f"{row.get('name')} {option} differs from repair protocol")
    if _option_values(argv, "--output-root") != [str(RAW_ROOT)]:
        raise AdjudicationError(f"{row.get('name')} output root differs from repair protocol")
    if _option_values(argv, "--compiler-cache-root") != [str(SAVED_D_CACHE)]:
        raise AdjudicationError(f"{row.get('name')} cache path differs from repair protocol")
    if mode == "timing":
        if any(option in argv for option in ("--profile-update", "--profile-microbatches")):
            raise AdjudicationError(f"{row.get('name')} timing argv unexpectedly enables profiling")
    elif (
        _option_values(argv, "--profile-update") != ["21"]
        or _option_values(argv, "--profile-microbatches") != ["3"]
    ):
        raise AdjudicationError(f"{row.get('name')} diagnostic profile differs from update 21")
    return argv


def _validate_repair_plan(plan: dict[str, Any]) -> None:
    if (
        plan.get("schema_version") != 1
        or plan.get("source_freeze_sha256") != EXPECTED_FREEZE_SHA256
        or plan.get("retained_A_run_id") != "schedule-embedding-a-timing-140-003"
        or plan.get("campaign_py_sha256") != EXPECTED_CAMPAIGN_SHA256
    ):
        raise AdjudicationError("repair plan does not pin the accepted source freeze and retained A")
    rows = plan.get("runs")
    if not isinstance(rows, list) or tuple(row.get("name") for row in rows) != REPAIR_TIMING_NAMES:
        raise AdjudicationError("repair plan timing entries differ from the six required 004 runs")
    for variant, row in zip(REPAIR_VARIANTS, rows):
        run_id = f"schedule-embedding-{variant.lower()}-timing-140-004"
        argv = _check_plan_argv(row, variant=variant, mode="timing", expected_run_id=run_id)
        expected_backend = "fla" if variant == "B" else "external_gate"
        for option, expected in (
            ("--gdn-backend", expected_backend),
            ("--routing-metadata", "fused"),
            ("--cross-entropy", "stable"),
            ("--num-prefetch-batches", "32"),
            ("--epoch-size", "102400"),
        ):
            if _option_values(argv, option) != [expected]:
                raise AdjudicationError(f"{row['name']} {option} differs from accepted recipe")
        if ("--gfsdp-reuse-bf16-expert-weights" in argv) is not True:
            raise AdjudicationError(f"{row['name']} lost BF16 expert-weight reuse")
        controls = _expected_controls(variant)
        for field, flag in (
            ("gfsdp_split_mtp_ffn", "--gfsdp-split-mtp-ffn"),
            ("gfsdp_split_layer0_ffn", "--gfsdp-split-layer0-ffn"),
            ("gfsdp_row_owner_embeddings", "--gfsdp-row-owner-embeddings"),
        ):
            if (flag in argv) is not controls[field]:
                raise AdjudicationError(f"{row['name']} {flag} differs from expected controls")

    diagnostics = plan.get("diagnostics")
    if (
        not isinstance(diagnostics, list)
        or tuple(row.get("name") for row in diagnostics) != REPAIR_DIAGNOSTIC_NAMES
    ):
        raise AdjudicationError("repair plan diagnostics differ from the four update-21 profiles")
    for row in diagnostics:
        variant = str(row["name"]).removesuffix("-diagnostic")
        run_id = f"schedule-embedding-{variant.lower()}-diagnostic-140-004"
        _check_plan_argv(row, variant=variant, mode="diagnostic", expected_run_id=run_id)


def _load_pinned_plan(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or path.resolve(strict=True) != EXPECTED_PLAN_PATH.resolve(strict=True):
        raise AdjudicationError(f"--plan must name the immutable repair plan {EXPECTED_PLAN_PATH}")
    plan_sha256 = _sha256_file(path)
    if plan_sha256 != EXPECTED_PLAN_SHA256:
        raise AdjudicationError(f"repair plan hash differs: {plan_sha256}")
    plan = _read_json(path)
    _validate_repair_plan(plan)
    return plan, plan_sha256


def _load_retained_freeze(plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
    record = _read_json(RETAINED_FREEZE_PATH)
    if (
        record.get("status") != "ADJUDICATED_PASS"
        or record.get("variant") != "A"
        or record.get("run_id") != plan.get("retained_A_run_id")
        or record.get("raw_receipts_unchanged") is not True
        or record.get("source_freeze_sha256") != EXPECTED_FREEZE_SHA256
    ):
        raise AdjudicationError("retained A evidence does not authenticate the pinned source freeze")
    freeze = record.get("source_freeze")
    if not isinstance(freeze, dict) or _json_digest(freeze) != EXPECTED_FREEZE_SHA256:
        raise AdjudicationError("retained A source-freeze JSON does not hash to the pinned freeze")
    schedule_hashes = freeze.get("schedule_source_sha256", {})
    if schedule_hashes.get("runs/schedule-embedding/campaign.py") != plan.get("campaign_py_sha256"):
        raise AdjudicationError("repair plan campaign hash differs from retained native freeze")
    return freeze, EXPECTED_FREEZE_SHA256


def _authenticate_frozen_files(freeze: dict[str, Any]) -> dict[str, str]:
    """Hash the frozen runtime/parser dependencies before importing any of them."""
    expected_files = _frozen_file_hash_manifest(freeze)
    verified: dict[str, str] = {}
    for path, expected_digest in expected_files.items():
        actual_digest = _sha256_file(Path(path))
        if actual_digest != expected_digest:
            if path in {str(ROOT / relative) for relative in EXPECTED_FLASH_BRIDGE_FILES}:
                relative = Path(path).relative_to(ROOT)
                raise AdjudicationError(f"frozen FlashQLA bridge source changed: {relative}")
            if path in {str(ROOT / relative) for relative in EXPECTED_SCHEDULE_SOURCE_FILES}:
                relative = Path(path).relative_to(ROOT)
                raise AdjudicationError(f"frozen schedule source changed: {relative}")
            name = next(
                key for key, helper_path in EXPECTED_EXECUTED_HELPERS.items()
                if str(helper_path) == path
            )
            raise AdjudicationError(f"frozen helper hash mismatch: {name}")
        verified[path] = actual_digest
    if verified[str(ROOT / "runs/schedule-embedding/campaign.py")] != EXPECTED_CAMPAIGN_SHA256:
        raise AdjudicationError("campaign.py differs from the immutable launch plan")
    return verified


def _frozen_file_hash_manifest(freeze: dict[str, Any]) -> dict[str, str]:
    """Return the exact retained path/hash map without reading current files."""
    expected: dict[str, str] = {}
    helpers = freeze.get("executed_helpers")
    if not isinstance(helpers, dict) or set(helpers) != set(EXPECTED_EXECUTED_HELPERS):
        raise AdjudicationError("retained freeze has an incomplete executed-helper manifest")
    for name, expected_path in EXPECTED_EXECUTED_HELPERS.items():
        entry = helpers.get(name)
        expected_digest = entry.get("sha256") if isinstance(entry, dict) else None
        recorded_path = entry.get("path") if isinstance(entry, dict) else None
        if recorded_path != str(expected_path):
            raise AdjudicationError(f"retained freeze helper path differs for {name}")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise AdjudicationError(f"retained freeze has an invalid helper hash for {name}")
        expected[str(expected_path)] = expected_digest

    schedule_hashes = freeze.get("schedule_source_sha256")
    if not isinstance(schedule_hashes, dict) or set(schedule_hashes) != set(EXPECTED_SCHEDULE_SOURCE_FILES):
        raise AdjudicationError("retained freeze has an incomplete schedule-source manifest")
    for relative in EXPECTED_SCHEDULE_SOURCE_FILES:
        digest = schedule_hashes[relative]
        if not isinstance(digest, str) or len(digest) != 64:
            raise AdjudicationError(f"retained freeze has an invalid schedule-source hash: {relative}")
        expected[str(ROOT / relative)] = digest

    flash_identity = freeze.get("flash_source_identity")
    bridge_hashes = flash_identity.get("bridge_sources") if isinstance(flash_identity, dict) else None
    if not isinstance(bridge_hashes, dict) or set(bridge_hashes) != set(EXPECTED_FLASH_BRIDGE_FILES):
        raise AdjudicationError("retained freeze has an incomplete FlashQLA bridge manifest")
    for relative in EXPECTED_FLASH_BRIDGE_FILES:
        digest = bridge_hashes[relative]
        if not isinstance(digest, str) or len(digest) != 64:
            raise AdjudicationError(f"retained freeze has an invalid FlashQLA bridge hash: {relative}")
        expected[str(ROOT / relative)] = digest
    if expected[str(ROOT / "runs/schedule-embedding/campaign.py")] != EXPECTED_CAMPAIGN_SHA256:
        raise AdjudicationError("campaign.py differs from the immutable launch plan")
    return expected


def _plan_run(
    plan: dict[str, Any], variant: str, mode: str = "timing"
) -> tuple[str, list[str] | None]:
    if mode not in {"timing", "diagnostic"}:
        raise AdjudicationError(f"unsupported adjudication mode {mode!r}")
    if mode == "timing" and variant == "A":
        return str(plan["retained_A_run_id"]), None
    if mode == "diagnostic" and variant not in DIAGNOSTIC_VARIANTS:
        raise AdjudicationError(f"{variant} has no pinned diagnostic profile")
    rows = plan["runs"] if mode == "timing" else plan["diagnostics"]
    row_name = f"{variant}-{mode}"
    row = next((item for item in rows if item["name"] == row_name), None)
    if row is None:
        raise AdjudicationError(f"repair plan has no {mode} entry for {variant}")
    return str(row["run_id"]), list(row["argv"])


def _require_all_timing_attempts_terminal(plan: dict[str, Any]) -> None:
    run_ids = [str(plan["retained_A_run_id"]), *(str(row["run_id"]) for row in plan["runs"])]
    for run_id in run_ids:
        attempt = RAW_ROOT / run_id
        status_path = attempt / "status.txt"
        if not status_path.is_file() or status_path.read_text(encoding="utf-8").strip() == "RUNNING":
            raise AdjudicationError("defer adjudication until every planned timing attempt is terminal")
        status = status_path.read_text(encoding="utf-8").strip()
        if status not in {"PASS", "FAIL", "TIMEOUT", "INTERRUPTED"}:
            raise AdjudicationError(f"{run_id} has nonterminal or unsupported status {status!r}")
        process_path = attempt / "process.json"
        if process_path.is_file():
            process = _read_json(process_path)
            if (
                process.get("process_group_stopped") is not True
                or process.get("orphaned_process_group") is True
            ):
                raise AdjudicationError(f"{run_id} still has an active/orphaned process group")


def _require_diagnostic_attempt_terminal(run_id: str) -> None:
    attempt = RAW_ROOT / run_id
    status_path = attempt / "status.txt"
    if not status_path.is_file():
        raise AdjudicationError("defer diagnostic adjudication until the requested attempt is terminal")
    status = status_path.read_text(encoding="utf-8").strip()
    if status == "RUNNING":
        raise AdjudicationError("defer diagnostic adjudication until the requested attempt is terminal")
    if status not in {"PASS", "FAIL", "TIMEOUT", "INTERRUPTED"}:
        raise AdjudicationError(f"{run_id} has nonterminal or unsupported status {status!r}")
    process_path = attempt / "process.json"
    if not process_path.is_file():
        raise AdjudicationError(f"{run_id} lacks a terminal process receipt")
    process = _read_json(process_path)
    if (
        process.get("process_group_stopped") is not True
        or process.get("orphaned_process_group") is True
    ):
        raise AdjudicationError(f"{run_id} still has an active/orphaned process group")


def _require_all_diagnostic_attempts_terminal(plan: dict[str, Any]) -> None:
    rows = plan.get("diagnostics")
    if (
        not isinstance(rows, list)
        or tuple(row.get("name") for row in rows) != REPAIR_DIAGNOSTIC_NAMES
    ):
        raise AdjudicationError("repair plan diagnostics differ from the four required profiles")
    for variant, row in zip(DIAGNOSTIC_VARIANTS, rows):
        expected_run_id = f"schedule-embedding-{variant.lower()}-diagnostic-140-004"
        if row.get("run_id") != expected_run_id:
            raise AdjudicationError(f"{variant} diagnostic run ID differs from the pinned plan")
        _require_diagnostic_attempt_terminal(expected_run_id)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdjudicationError(f"could not read JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdjudicationError(f"expected a JSON object in {path}")
    return value


def _write_json_new(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(payload)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AdjudicationError(f"cannot load read-only helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Load the frozen campaign helpers in the task's pinned source context."""
    sys.dont_write_bytecode = True
    gift_root = ROOT / "gift"
    if str(gift_root) not in sys.path:
        sys.path.insert(0, str(gift_root))
    campaign = _load_module(EVIDENCE / "campaign.py", "_schedule_adjudication_campaign")
    schedule_runner = _load_module(EVIDENCE / "run.py", "_schedule_adjudication_runner")
    flash_runner = _load_module(
        ROOT / "runs/flash-qla/run.py", "_schedule_adjudication_flash_runner"
    )
    schedule_measure = _load_module(
        EVIDENCE / "measure.py", "_schedule_adjudication_measure"
    )
    flash_measure = _load_module(
        ROOT / "runs/flash-qla/measure.py", "_schedule_adjudication_flash_measure"
    )
    base = flash_runner._load_base_runner()
    native_measure = flash_measure._load_base_measure()
    return campaign, schedule_runner, flash_runner, schedule_measure, base, native_measure


def _dotted_get(mapping: dict[str, Any], dotted: str) -> tuple[bool, Any]:
    current: Any = mapping
    for component in dotted.split("."):
        if not isinstance(current, dict) or component not in current:
            return False, None
        current = current[component]
    return True, current


def _check_expected_config(
    resolved: dict[str, Any],
    expected: dict[str, Any],
    *,
    nullable_stage_evidence: dict[str, Any] | None = None,
) -> str | None:
    """Match the runner guard, with one evidence-backed nullable default case."""
    accepted_missing: str | None = None
    for dotted, expected_value in expected.items():
        present, actual = _dotted_get(resolved, dotted)
        if not present:
            if (
                dotted == f"parallelism.{STAGES_FIELD}"
                and expected_value is None
                and nullable_stage_evidence
                and all(nullable_stage_evidence.values())
            ):
                accepted_missing = dotted
                continue
            raise AdjudicationError(f"resolved config lacks {dotted}")
        if actual != expected_value:
            raise AdjudicationError(
                f"resolved config {dotted}={actual!r}, expected {expected_value!r}"
            )
    return accepted_missing


def _strict_guard_error(resolved: dict[str, Any], expected: dict[str, Any]) -> str | None:
    try:
        _check_expected_config(resolved, expected)
    except AdjudicationError as exc:
        return str(exc)
    return None


def _check_stage_receipts(
    receipts: list[dict[str, Any]],
    *,
    run_id: str,
    variant: str,
    source_hashes: dict[str, str],
    expected_stages: list[str],
) -> dict[str, Any]:
    expected_count = 0 if variant in {"A", "B", "C"} else 1
    if len(receipts) != 8:
        raise AdjudicationError(f"expected 8 stage receipts, found {len(receipts)}")
    module_names: dict[int, list[str]] = {}
    for rank, record in enumerate(receipts):
        modules = record.get("modules")
        if (
            record.get("run_id") != run_id
            or record.get("rank") != rank
            or record.get("world_size") != 8
            or record.get("variant") != variant
            or record.get("source_sha256") != source_hashes
            or record.get("capture_stage")
            != "after_trainer_init_before_first_optimizer_update"
            or record.get("device_reads") != 0
            or record.get("expected_module_count") != expected_count
            or record.get("expected_stages") != expected_stages
            or not isinstance(modules, list)
            or len(modules) != expected_count
        ):
            raise AdjudicationError(f"rank {rank} stage-selection receipt is invalid")
        names = []
        for module in modules:
            name = module.get("name")
            if not isinstance(name, str) or not name:
                raise AdjudicationError(f"rank {rank} stage receipt has an unnamed module")
            if module.get("optimization_stages") != expected_stages:
                raise AdjudicationError(f"rank {rank} resolved embedding stages differ")
            names.append(name)
        if len(set(names)) != len(names):
            raise AdjudicationError(f"rank {rank} stage receipt repeats a module name")
        module_names[rank] = names
    return {"rank_count": 8, "module_count_per_rank": expected_count, "module_names_by_rank": module_names}


def _typed_nullable_default() -> dict[str, Any]:
    from torchtitan.config.configs import ParallelismConfig

    fields = {field.name: field for field in dataclasses.fields(ParallelismConfig)}
    field = fields.get(STAGES_FIELD)
    if field is None:
        raise AdjudicationError(f"native ParallelismConfig lacks {STAGES_FIELD}")
    annotation = get_type_hints(ParallelismConfig).get(STAGES_FIELD)
    nullable = annotation is not None and type(None) in get_args(annotation)
    if field.default is not None or not nullable:
        raise AdjudicationError("native stage selector is not a nullable None-default field")
    return {
        "config_class": f"{ParallelismConfig.__module__}.{ParallelismConfig.__qualname__}",
        "source_file": str(Path(sys.modules[ParallelismConfig.__module__].__file__).resolve()),
        "default_is_none": True,
        "annotation_accepts_none": True,
    }


def _load_raw_hash_paths(attempt: Path, *, mode: str = "timing") -> list[Path]:
    if mode not in {"timing", "diagnostic"}:
        raise AdjudicationError(f"unsupported raw receipt hash mode {mode!r}")
    paths = [
        attempt / "config/experiment.yaml",
        attempt / "config-hashes-final.json",
        attempt / "exit-code.txt",
        attempt / "failure.json",
        attempt / "flash-qla-bridge.json",
        attempt / "metadata.json",
        attempt / "native/config/resolved_config.yaml",
        attempt / "process.json",
        attempt / "schedule-embedding-bridge.json",
        attempt / "source-final.json",
        attempt / "completion.json",
        attempt / "summary.json",
        attempt / "status.txt",
    ]
    for rank in range(8):
        paths.extend(
            attempt / "measurement" / name
            for name in (
                f"rank-{rank:02d}.json",
                f"embedding-stage-selection-rank-{rank:02d}.json",
                f"flash-qla-bootstrap-rank-{rank:02d}.json",
                f"triton-startup-rank-{rank:02d}.json",
            )
        )
    if mode == "diagnostic":
        paths.extend(
            [
                attempt / "preflight/task-tmpdir.json",
                attempt / "task-tmpdir-cleanup.json",
                *(
                    attempt
                    / f"native/profiling/traces/iteration_{DIAGNOSTIC_CADENCE[0]}"
                    / f"rank{rank}_trace.json"
                    for rank in (0, 1)
                ),
            ]
        )
    return [path for path in paths if path.exists()]


def _raw_hashes(paths: list[Path], attempt: Path) -> dict[str, str]:
    return {str(path.relative_to(attempt)): _sha256_file(path) for path in paths}


def _timing_adjudication_path(variant: str) -> Path:
    return OUTPUT_ROOT / variant / "adjudication.json"


def _require_all_timing_adjudications_pass(
    plan: dict[str, Any],
    freeze: dict[str, Any],
    plan_sha256: str,
) -> None:
    """Require the seven reviewed timing proofs and their current receipts."""
    expected_runtime_hashes = _frozen_file_hash_manifest(freeze)
    current_method_sha256 = _sha256_file(Path(__file__).resolve())
    records: list[tuple[str, Path, Path, dict[str, Any]]] = []
    for variant in VARIANTS:
        run_id, _argv = _plan_run(plan, variant, "timing")
        attempt = RAW_ROOT / run_id
        record_path = _timing_adjudication_path(variant)
        record = _read_json(record_path)
        if (
            record.get("status") != "ADJUDICATED_PASS"
            or record.get("variant") != variant
            or record.get("run_id") != run_id
            or record.get("original_wrapper_status") not in {"PASS", "FAIL"}
            or record.get("original_exit_code")
            != (0 if record.get("original_wrapper_status") == "PASS" else 1)
            or record.get("raw_receipts_unchanged") is not True
            or record.get("source_freeze_sha256") != EXPECTED_FREEZE_SHA256
            or record.get("source_freeze") != freeze
            or record.get("adjudicator_method_sha256") != current_method_sha256
            or record.get("launch_plan_path") != str(EXPECTED_PLAN_PATH)
            or record.get("launch_plan_sha256") != plan_sha256
            or record.get("authenticated_runtime_files") != expected_runtime_hashes
        ):
            raise AdjudicationError(f"{variant} lacks a matching ADJUDICATED_PASS timing proof")
        launch_plan_path = record.get("launch_plan_path")
        launch_plan_digest = record.get("launch_plan_sha256")
        if (
            not isinstance(launch_plan_path, str)
            or not isinstance(launch_plan_digest, str)
            or len(launch_plan_digest) != 64
        ):
            raise AdjudicationError(f"{variant} timing proof lacks launch-plan provenance")
        source_provenance = record.get("source_freeze_provenance")
        if (
            not isinstance(source_provenance, dict)
            or source_provenance.get("retained_record") != str(RETAINED_FREEZE_PATH)
            or source_provenance.get("current_full_source_walk_recomputed") is not False
        ):
            raise AdjudicationError(f"{variant} timing proof source-freeze provenance differs")

        try:
            recorded_attempt = Path(record["raw_attempt"]).resolve(strict=True)
            expected_attempt = attempt.resolve(strict=True)
        except (KeyError, OSError) as exc:
            raise AdjudicationError(f"{variant} timing proof has no existing raw attempt") from exc
        if recorded_attempt != expected_attempt:
            raise AdjudicationError(f"{variant} timing proof points to another raw attempt")
        if record.get("adjudication_path") != str(record_path.resolve(strict=True)):
            raise AdjudicationError(f"{variant} timing proof path differs from its immutable location")
        summary_path = record_path.parent / "summary.json"
        try:
            recorded_summary_path = Path(record["parser_summary_path"]).resolve(strict=True)
            expected_summary_path = summary_path.resolve(strict=True)
        except (KeyError, OSError) as exc:
            raise AdjudicationError(f"{variant} timing proof lacks its parser summary") from exc
        if recorded_summary_path != expected_summary_path:
            raise AdjudicationError(f"{variant} timing parser-summary path differs")
        records.append((variant, attempt, summary_path, record))

    # Do not hash raw timing receipts or summaries until all seven proofs are admitted.
    for variant, attempt, summary_path, record in records:
        if (attempt / "status.txt").read_text(encoding="utf-8").strip() != record[
            "original_wrapper_status"
        ]:
            raise AdjudicationError(f"{variant} original timing status differs from its proof")
        if int((attempt / "exit-code.txt").read_text(encoding="utf-8").strip()) != record[
            "original_exit_code"
        ]:
            raise AdjudicationError(f"{variant} original timing exit code differs from its proof")

        raw_hashes = record.get("raw_receipt_hashes")
        if not isinstance(raw_hashes, dict) or raw_hashes != _raw_hashes(
            _load_raw_hash_paths(attempt), attempt
        ):
            raise AdjudicationError(f"{variant} original timing receipts changed after adjudication")
        summary_digest = _sha256_file(summary_path)
        if record.get("parser_summary_sha256") != summary_digest:
            raise AdjudicationError(f"{variant} timing parser-summary provenance differs")


def _parse_parent_recipe(
    variant: str,
    campaign: Any,
    schedule_runner: Any,
    flash_runner: Any,
    base: Any,
    planned_argv: list[str] | None,
) -> tuple[Any, Any, dict[str, Any]]:
    argv = planned_argv or campaign._run_argv(f"{variant}-timing")
    parsed_variant, forwarded = schedule_runner._parse_variant_args(argv[2:])
    if parsed_variant != variant:
        raise AdjudicationError("frozen campaign variant parser changed the selected variant")
    bridge, forwarded = flash_runner._parse_bridge_args(forwarded)
    args = base._parse_args(forwarded)
    recipe_variant = variant if variant in {"A", "B", "C"} else "D"
    recipe = next(
        row
        for row in campaign._read_plan()["runs"]
        if row.get("variant") == recipe_variant and row.get("role") == "timing"
    )
    expected = recipe.get("expected", {})
    if (
        bridge.gdn_backend != expected.get("backend")
        or bridge.routing_metadata != expected.get("routing")
        or bridge.cross_entropy != expected.get("ce")
        or args.num_prefetch_batches != expected.get("prefetch")
        or args.gfsdp_reuse_bf16_expert_weights is not expected.get("reuse")
        or base._native_feature_values(args)
        != {
            campaign.SPLIT_MTP_FIELD: expected.get("three"),
            campaign.SPLIT_LAYER0_FIELD: expected.get("three"),
            campaign.ROW_OWNER_FIELD: expected.get("three"),
        }
    ):
        raise AdjudicationError(f"{variant} historical base recipe changed")
    return args, bridge, expected


def _validate_bootstrap(
    receipts: list[dict[str, Any]],
    *,
    variant: str,
    controls: dict[str, Any],
    source_hashes: dict[str, str],
) -> None:
    if len(receipts) != 8:
        raise AdjudicationError(f"expected 8 capacity/bootstrap receipts, found {len(receipts)}")
    for rank, receipt in enumerate(receipts):
        schedule = receipt.get("schedule_embedding", {})
        control = schedule.get("capacity_control", {})
        final = schedule.get("capacity_control_final_check", {})
        if (
            receipt.get("rank") != rank
            or receipt.get("local_rank") != rank
            or receipt.get("native_main_completed") is not True
            or schedule.get("variant") != variant
            or schedule.get("controls") != controls
            or schedule.get("source_sha256") != source_hashes
            or control.get("module_name")
            != "torchtitan.models.common.moe.dispatch.capacity"
            or control.get("module_sha256")
            != "2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120"
            or control.get("loaded_before_native_imports") is not True
            or final.get("registered_same_object") is not True
            or final.get("module_path_is_original_snapshot") is not True
            or final.get("current_capacity_file_unchanged") is not True
        ):
            raise AdjudicationError(f"rank {rank} capacity/variant bootstrap receipt is invalid")


def _check_source_config_identity(
    *,
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    config_before: dict[str, Any],
    config_after: dict[str, Any],
    freeze: dict[str, Any],
) -> None:
    if source_before != freeze["gift_source_snapshot"]:
        raise AdjudicationError("attempt source snapshot differs from the current source freeze")
    if (
        source_after.get("tracked_and_untracked_sha256")
        != source_before.get("tracked_and_untracked_sha256")
        or source_after.get("head") != source_before.get("head")
    ):
        raise AdjudicationError("final source snapshot differs from the run-start snapshot")
    if config_after != config_before:
        raise AdjudicationError("final config hashes differ from the run-start config hashes")
    if any(
        config_before.get(key) != value
        for key, value in freeze["base_config_hashes"].items()
    ):
        raise AdjudicationError("base configuration/input hashes differ from the source freeze")
    if (
        config_before.get("flash_qla_bridge_sources")
        != freeze["flash_source_identity"]["bridge_sources"]
    ):
        raise AdjudicationError("FlashQLA bridge source hashes differ from the source freeze")


def _check_input_cache_metadata(
    metadata: dict[str, Any], expected_input: dict[str, Any], cache_root: Path
) -> None:
    if metadata.get("input_data") != expected_input:
        raise AdjudicationError("attempt production input receipt differs from current verified inputs")
    if (
        metadata.get("compiler_cache_policy") != "reuse"
        or Path(metadata.get("compiler_cache_root", "")) != cache_root
        or metadata.get("compiler_cache_producer_run_id") != "proddata-d-timing-001"
    ):
        raise AdjudicationError("attempt compiler-cache policy, path or producer differs")


def _validate_timing_summary(summary: dict[str, Any]) -> None:
    if (
        summary.get("status") != "complete"
        or summary.get("errors") != []
        or summary.get("training_steps") != TIMING_CADENCE[0]
        or summary.get("world_size") != 8
        or summary.get("mode") != "complete_window"
    ):
        raise AdjudicationError("native parser did not report a complete 140-update timing run")
    window = summary.get("window", {})
    tokens = summary.get("tokens", {})
    if (
        window.get("warmup_updates") != TIMING_CADENCE[1]
        or window.get("start_after_update") != TIMING_CADENCE[1]
        or window.get("end_after_update") != TIMING_CADENCE[0]
        or window.get("updates") != TIMING_CADENCE[2]
        or window.get("global_envelope_convention")
        != "max(rank end monotonic timestamp) - min(rank start timestamp)"
        or tokens.get("physical_input_tokens_global") != GLOBAL_PHYSICAL_TOKENS
        or tokens.get("supervised_tokens_global") != GLOBAL_SUPERVISED_TOKENS
    ):
        raise AdjudicationError("native parser summary differs from the 40+100 token window")
    per_rank = window.get("per_rank", {})
    expected_ranks = {str(rank) for rank in range(8)}
    if set(per_rank) != expected_ranks:
        raise AdjudicationError("native parser summary lacks per-rank timing/token evidence")
    for rank, values in per_rank.items():
        if (
            values.get("physical_input_tokens_local") != LOCAL_PHYSICAL_TOKENS
            or values.get("supervised_tokens_global") != GLOBAL_SUPERVISED_TOKENS
        ):
            raise AdjudicationError(f"rank {rank} parser token counters differ from protocol")
    memory = summary.get("memory", {}).get("by_rank", {})
    if set(memory) != expected_ranks:
        raise AdjudicationError("native parser summary lacks all eight memory records")
    for rank, values in memory.items():
        peaks = values.get("peak_bytes", {})
        deltas = values.get("allocator_counter_deltas", {})
        if not {"allocated_bytes.all.peak", "reserved_bytes.all.peak"}.issubset(peaks):
            raise AdjudicationError(f"rank {rank} memory peak fields are incomplete")
        if not {"num_alloc_retries", "num_ooms"}.issubset(deltas):
            raise AdjudicationError(f"rank {rank} allocator deltas are incomplete")


def _validate_diagnostic_wrapper_receipts(
    *,
    variant: str,
    status: str,
    exit_code: int,
    failure: dict[str, Any] | None,
    completion: dict[str, Any] | None,
    raw_summary: dict[str, Any] | None,
    summary: dict[str, Any],
) -> str | None:
    if status == "PASS":
        if (
            exit_code != 0
            or failure is not None
            or completion is None
            or completion.get("status") != "PASS"
            or completion.get("source_unchanged") is not True
            or completion.get("config_unchanged") is not True
            or raw_summary != summary
        ):
            raise AdjudicationError("original diagnostic PASS wrapper receipts are inconsistent")
        return None
    if status == "FAIL" and variant in {"D0", "S"}:
        if (
            exit_code != 1
            or failure is None
            or failure.get("status") != "FAIL"
            or failure.get("error") != KNOWN_NULLABLE_GUARD_ERROR
            or failure.get("timed_out") is not False
            or completion is not None
            or raw_summary is not None
        ):
            raise AdjudicationError(
                "diagnostic FAIL is not the exact pre-parser nullable-stage guard defect"
            )
        return KNOWN_NULLABLE_GUARD_ERROR
    raise AdjudicationError(f"unsupported original diagnostic wrapper status {status!r}")


def _validate_diagnostic_task_tmpdir_cleanup(
    attempt: Path,
    *,
    run_id: str,
    metadata: dict[str, Any],
    completion: dict[str, Any] | None,
) -> None:
    task_tmpdir = metadata.get("task_tmpdir")
    path = task_tmpdir.get("path") if isinstance(task_tmpdir, dict) else None
    if not isinstance(path, str) or not path:
        raise AdjudicationError("diagnostic metadata lacks its task temporary directory")
    preflight = _read_json(attempt / "preflight/task-tmpdir.json")
    cleanup = _read_json(attempt / "task-tmpdir-cleanup.json")
    tmp_path = Path(path)
    if (
        preflight.get("run_id") != run_id
        or preflight.get("path") != path
        or cleanup.get("path") != path
        or cleanup.get("removed") is not True
        or tmp_path.exists()
        or tmp_path.is_symlink()
    ):
        raise AdjudicationError("diagnostic task temporary directory cleanup is not proven")
    if completion is not None and (
        completion.get("task_tmpdir") != path
        or completion.get("task_tmpdir_removed") is not True
    ):
        raise AdjudicationError("diagnostic completion does not prove task TMPDIR cleanup")


def _validate_diagnostic_summary(
    summary: dict[str, Any], metadata: dict[str, Any], parser: Any
) -> dict[str, Any]:
    if (
        summary.get("status") != "complete"
        or summary.get("mode") != "profile_update"
        or summary.get("training_steps") != DIAGNOSTIC_CADENCE[0]
        or summary.get("measurement_status") != "disabled"
        or summary.get("errors") != []
    ):
        raise AdjudicationError("native parser did not accept the complete update-21 profile run")
    if not callable(getattr(parser, "_expected_profile_spec", None)):
        raise AdjudicationError("frozen parser lacks its native profile-spec validator")
    expected_profile = parser._expected_profile_spec(
        DIAGNOSTIC_CADENCE[0], DIAGNOSTIC_CADENCE[1]
    )
    expected_profile["profile_microbatches"] = 3
    if metadata.get("profile_capture") != expected_profile:
        raise AdjudicationError("diagnostic metadata lacks the full frozen profile specification")

    capture = summary.get("profile_capture")
    expected_ranks = {str(rank) for rank in range(8)}
    if not isinstance(capture, dict) or (
        capture.get("target_update") != DIAGNOSTIC_CADENCE[0]
        or capture.get("warmup_updates") != DIAGNOSTIC_CADENCE[1]
        or capture.get("profile_ranks") != [0, 1]
        or capture.get("settings") != expected_profile["settings"]
        or capture.get("native_hardcoded") != expected_profile["native_hardcoded"]
    ):
        raise AdjudicationError("native parser profile capture differs from its metadata recipe")

    calls = capture.get("call_counts_by_rank", {})
    if not isinstance(calls, dict) or set(calls) != expected_ranks:
        raise AdjudicationError("native parser lacks profile call-count proof for all eight ranks")
    for rank in range(8):
        counts = calls[str(rank)]
        expected = {
            "microbatch_calls": 32,
            "train_step_calls": 1,
            "optimizer_step_calls": 1,
            "native_profiler_build_calls": 1,
            "native_profiler_created": rank in (0, 1),
        }
        if not isinstance(counts, dict) or any(
            counts.get(key) != value for key, value in expected.items()
        ):
            raise AdjudicationError(f"rank {rank} native profile call counters differ")

    traces = capture.get("traces_by_rank", {})
    if not isinstance(traces, dict) or set(traces) != {"0", "1"}:
        raise AdjudicationError("native parser lacks the two selected original traces")
    expected_trace_hashes: dict[str, str] = {}
    for rank in (0, 1):
        trace = traces[str(rank)]
        expected_path = f"native/profiling/traces/iteration_21/rank{rank}_trace.json"
        expected_markers = {
            f"task.profile_update.21.microbatch.rank{rank}": 3,
            f"task.profile_update.21.train_step.rank{rank}": 1,
        }
        gpu_counts = trace.get("gpu_projection_counts") if isinstance(trace, dict) else None
        if (
            not isinstance(trace, dict)
            or trace.get("path") != expected_path
            or type(trace.get("size_bytes")) is not int
            or trace["size_bytes"] <= 0
            or not isinstance(trace.get("sha256"), str)
            or len(trace["sha256"]) != 64
            or trace.get("marker_counts") != expected_markers
            or not isinstance(gpu_counts, dict)
            or gpu_counts.get(f"task.profile_update.21.microbatch.rank{rank}", 0) < 3
        ):
            raise AdjudicationError(f"rank {rank} native parser trace proof is incomplete")
        expected_trace_hashes[expected_path] = trace["sha256"]

    return {
        "status": "NATIVE_PROFILE_PARSER_PASS",
        "mode": summary["mode"],
        "training_steps": summary["training_steps"],
        "profile_capture": capture,
        "trace_tail_coverage": {
            "status": DIAGNOSTIC_PROFILE_TAIL_STATUS,
            "overall_acceptance_complete": False,
            "required_follow_up": (
                "Reconcile original CPU launches/flows with GPU work and verify the third-microbatch "
                "asynchronous tail through the completed device drain in Perfetto."
            ),
        },
        "expected_original_trace_hashes": expected_trace_hashes,
    }


def _check_original_trace_hashes(
    proof: dict[str, Any], raw_hashes: dict[str, str]
) -> None:
    expected = proof.get("expected_original_trace_hashes")
    if not isinstance(expected, dict) or set(expected) != {
        "native/profiling/traces/iteration_21/rank0_trace.json",
        "native/profiling/traces/iteration_21/rank1_trace.json",
    }:
        raise AdjudicationError("native parser proof lacks both original diagnostic trace hashes")
    if any(raw_hashes.get(path) != digest for path, digest in expected.items()):
        raise AdjudicationError("native parser trace hashes differ from the original trace bytes")


def _validate(
    variant: str,
    run_id: str,
    planned_argv: list[str] | None,
    attempt: Path,
    summary: dict[str, Any],
    runtime: tuple[Any, Any, Any, Any, Any, Any],
    freeze: dict[str, Any],
    freeze_sha256: str,
    *,
    mode: str = "timing",
    parser: Any | None = None,
    raw_hashes_before: dict[str, str] | None = None,
) -> dict[str, Any]:
    if mode not in {"timing", "diagnostic"}:
        raise AdjudicationError(f"unsupported adjudication mode {mode!r}")
    if mode == "timing" and variant not in VARIANTS:
        raise AdjudicationError(f"unsupported timing variant {variant!r}")
    if mode == "diagnostic" and variant not in DIAGNOSTIC_VARIANTS:
        raise AdjudicationError(f"{variant} has no pinned diagnostic profile")
    campaign, schedule_runner, flash_runner, _schedule_measure, base, native_measure = runtime
    cadence = TIMING_CADENCE if mode == "timing" else DIAGNOSTIC_CADENCE
    metadata = _read_json(attempt / "metadata.json")
    process = _read_json(attempt / "process.json")
    status = (attempt / "status.txt").read_text(encoding="utf-8").strip()
    exit_code = int((attempt / "exit-code.txt").read_text(encoding="utf-8").strip())
    failure_path = attempt / "failure.json"
    failure = _read_json(failure_path) if failure_path.is_file() else None
    completion_path = attempt / "completion.json"
    completion = _read_json(completion_path) if completion_path.is_file() else None
    raw_summary_path = attempt / "summary.json"
    raw_summary = _read_json(raw_summary_path) if raw_summary_path.is_file() else None
    source_final = _read_json(attempt / "source-final.json")
    config_final = _read_json(attempt / "config-hashes-final.json")
    bridge = _read_json(attempt / "schedule-embedding-bridge.json")
    flash_bridge = _read_json(attempt / "flash-qla-bridge.json")
    resolved_path = attempt / "native/config/resolved_config.yaml"
    resolved = base._load_yaml(resolved_path)

    if metadata.get("run_id") != run_id or metadata.get("argv", [None])[0] != "/opt/torchtitan-venv/bin/python3.12":
        raise AdjudicationError("attempt metadata identity or pinned Python differs")
    if (metadata.get("steps"), metadata.get("warmup_updates"), metadata.get("window_updates")) != cadence:
        raise AdjudicationError(
            "attempt metadata is not the fixed 140/40/100 timing protocol"
            if mode == "timing"
            else "attempt metadata is not the fixed 21/20/0 diagnostic protocol"
        )
    if (
        process.get("child_exit_code") != 0
        or process.get("timeout") is not False
        or process.get("orphaned_process_group") is not False
        or process.get("process_group_stopped") is not True
    ):
        raise AdjudicationError("native child did not complete with a clean stopped process group")
    if mode == "diagnostic":
        strict_error = _validate_diagnostic_wrapper_receipts(
            variant=variant,
            status=status,
            exit_code=exit_code,
            failure=failure,
            completion=completion,
            raw_summary=raw_summary,
            summary=summary,
        )
        _validate_diagnostic_task_tmpdir_cleanup(
            attempt,
            run_id=run_id,
            metadata=metadata,
            completion=completion,
        )
    elif status == "PASS":
        if (
            exit_code != 0
            or failure is not None
            or completion is None
            or completion.get("status") != "PASS"
            or completion.get("source_unchanged") is not True
            or completion.get("config_unchanged") is not True
        ):
            raise AdjudicationError("original PASS wrapper receipts are inconsistent")
        if raw_summary != summary:
            raise AdjudicationError("original summary differs from the frozen parser result")
    elif status == "FAIL":
        if (
            exit_code != 1
            or failure is None
            or failure.get("status") != "FAIL"
            or failure.get("error") != KNOWN_NULLABLE_GUARD_ERROR
            or failure.get("timed_out") is not False
            or completion is not None
        ):
            raise AdjudicationError("wrapper failure is not the known nullable-stage guard defect")
    elif mode == "timing":
        raise AdjudicationError(f"unsupported original wrapper status {status!r}")

    config_before = metadata.get("config_hashes_before", {})
    _check_source_config_identity(
        source_before=metadata.get("source_snapshot_before", {}),
        source_after=source_final,
        config_before=config_before,
        config_after=config_final,
        freeze=freeze,
    )

    controls = _expected_controls(variant)
    if campaign._controls(variant) != controls:
        raise AdjudicationError("loaded campaign controls differ from the repair-plan selector contract")
    if (
        bridge.get("variant") != variant
        or bridge.get("controls") != controls
        or bridge.get("source_sha256") != freeze["schedule_source_sha256"]
        or flash_bridge.get("source_identity") != freeze["flash_source_identity"]
        or flash_bridge.get("run_id") != run_id
    ):
        raise AdjudicationError("schedule/Flash bridge identity differs from the source freeze")
    guard = bridge.get("config_hash_guard", {})
    if (
        len(guard) != 5
        or any(config_before.get(key) != value for key, value in guard.items())
        or config_before.get("schedule_embedding_variant_sha256") != bridge.get("variant_sha256")
        or config_before.get("schedule_embedding_controls_sha256") != bridge.get("controls_sha256")
    ):
        raise AdjudicationError("schedule/embedding config hash guard does not match the bridge")
    if metadata.get("gfsdp_buffer_inventory") is not False:
        raise AdjudicationError("attempt enabled GFSDP buffer inventory")

    expected_input = base._verify_ready_inputs(ROOT)
    _check_input_cache_metadata(metadata, expected_input, campaign.SAVED_D_CACHE)
    base._validate_completed_compiler_cache_root(campaign.SAVED_D_CACHE)
    base._verify_triton_startup_receipts(
        {"measurement": attempt / "measurement"}, metadata
    )

    argv = metadata.get("argv")
    if not isinstance(argv, list) or str((EVIDENCE / "measure.py").resolve()) not in argv:
        raise AdjudicationError("native child command does not invoke schedule-embedding measure.py")
    if "--nproc_per_node=8" not in argv:
        raise AdjudicationError("native child command does not launch eight ranks")
    measured, training_argv = native_measure._parse_args(
        argv[argv.index(str((EVIDENCE / "measure.py").resolve())) + 1 :]
    )
    if (measured.steps, measured.warmup, measured.window) != cadence:
        raise AdjudicationError(
            "actual child parser changed the 140/40/100 cadence"
            if mode == "timing"
            else "actual child parser changed the 21/20/0 diagnostic cadence"
        )
    if measured.gfsdp_buffer_inventory:
        raise AdjudicationError("attempt enabled buffer inventory")
    if mode == "timing" and measured.profile_update is not None:
        raise AdjudicationError("timing child enabled profiling")
    if mode == "diagnostic" and (
        measured.profile_update != DIAGNOSTIC_CADENCE[0]
        or measured.profile_microbatches != 3
    ):
        raise AdjudicationError("diagnostic child does not profile three microbatches at update 21")
    for field, value in controls.items():
        option = f"--parallelism.{field}"
        if campaign._option_values(argv, option) != [campaign._control_argument(value)]:
            raise AdjudicationError(f"actual child config differs for {field}")

    parent_args, flash_args, recipe_expected = _parse_parent_recipe(
        variant, campaign, schedule_runner, flash_runner, base, planned_argv
    )
    prefetch_values = campaign._option_values(
        training_argv, "--dataloader.num_prefetch_batches"
    )
    if prefetch_values != [str(recipe_expected.get("prefetch"))]:
        raise AdjudicationError("actual native command differs from the historical prefetch recipe")
    parallelism = campaign._native_parallelism_config(training_argv)
    if (
        flash_bridge.get("backend_variant") != recipe_expected.get("backend")
        or flash_bridge.get("routing_metadata") != recipe_expected.get("routing")
        or flash_bridge.get("cross_entropy") != recipe_expected.get("ce")
        or parallelism.gfsdp_reuse_bf16_expert_weights is not recipe_expected.get("reuse")
    ):
        raise AdjudicationError("actual native command or Flash bridge differs from its historical recipe")
    if flash_args.gdn_backend != recipe_expected.get("backend"):
        raise AdjudicationError("frozen Flash parser differs from the historical base recipe")

    resolved_stage = getattr(parallelism, STAGES_FIELD)
    expected_stages = list(controls[STAGES_FIELD] or [])
    expected_typed_stage = tuple(expected_stages) if controls[STAGES_FIELD] is not None else None
    if resolved_stage != expected_typed_stage:
        raise AdjudicationError(
            f"typed native selector resolved to {resolved_stage!r}, expected {expected_typed_stage!r}"
        )

    stage_receipts = [
        _read_json(attempt / f"measurement/embedding-stage-selection-rank-{rank:02d}.json")
        for rank in range(8)
    ]
    stage_summary = _check_stage_receipts(
        stage_receipts,
        run_id=run_id,
        variant=variant,
        source_hashes=freeze["schedule_source_sha256"],
        expected_stages=expected_stages,
    )

    expectations = schedule_runner._variant_config_expectations(
        base._resolved_config_expectations, variant
    )(parent_args)
    strict_error = _strict_guard_error(resolved, expectations)
    nullable_default = _typed_nullable_default() if controls[STAGES_FIELD] is None else None
    explicit_null = campaign._option_values(
        argv, f"--parallelism.{STAGES_FIELD}"
    ) == ["null"]
    selector_present, _selector_value = _dotted_get(
        resolved, f"parallelism.{STAGES_FIELD}"
    )
    nullable_evidence = {
        "typed_default_none": nullable_default is not None,
        "typed_annotation_nullable": nullable_default is not None,
        "resolved_yaml_omits_selector": not selector_present,
        "explicit_child_null_override": explicit_null,
        "native_typed_decode_is_none": resolved_stage is None,
        "all_rank_module_stage_receipts_valid": len(stage_receipts) == 8,
    }
    if strict_error is None:
        if status == "FAIL":
            raise AdjudicationError("raw wrapper failed even though the strict resolved guard passed")
        missing_nullable = _check_expected_config(resolved, expectations)
    elif strict_error == STRICT_NULLABLE_GUARD_ERROR:
        if status != "FAIL" or controls[STAGES_FIELD] is not None:
            raise AdjudicationError("nullable-stage strict guard failure conflicts with raw status")
        missing_nullable = _check_expected_config(
            resolved,
            expectations,
            nullable_stage_evidence=nullable_evidence,
        )
        if missing_nullable != f"parallelism.{STAGES_FIELD}":
            raise AdjudicationError("strict guard failure was not the one nullable stage field")
    else:
        raise AdjudicationError(f"strict resolved-config guard failed: {strict_error}")

    # Reuse the runner's remaining resolved-config checks for callbacks, paths and input selection.
    base._verify_resolved_config(
        {"native": attempt / "native"},
        parent_args,
    )
    for field, value in controls.items():
        present, actual = _dotted_get(resolved, f"parallelism.{field}")
        if field == STAGES_FIELD and not present and value is None:
            continue
        if not present or actual != value:
            raise AdjudicationError(
                f"resolved parallelism.{field}={actual!r}, expected {value!r}"
            )

    bootstrap = [
        _read_json(attempt / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json")
        for rank in range(8)
    ]
    _validate_bootstrap(
        bootstrap,
        variant=variant,
        controls=controls,
        source_hashes=freeze["schedule_source_sha256"],
    )
    if summary.get("run_id") != run_id:
        raise AdjudicationError("native parser summary identifies another attempt")
    diagnostic_proof = None
    if mode == "timing":
        _validate_timing_summary(summary)
        _validate_rank_cadence(attempt, run_id)
    else:
        if parser is None:
            raise AdjudicationError("diagnostic adjudication lacks the authenticated native parser")
        diagnostic_proof = _validate_diagnostic_summary(summary, metadata, parser)
        if raw_hashes_before is None:
            raise AdjudicationError("diagnostic adjudication lacks pre-parser raw receipt hashes")
        _check_original_trace_hashes(diagnostic_proof, raw_hashes_before)
        _validate_rank_cadence(attempt, run_id, expected_steps=DIAGNOSTIC_CADENCE[0])

    if status == "PASS" and completion is not None:
        resolved_digest = _sha256_file(resolved_path)
        if completion.get("resolved_config_sha256") != resolved_digest:
            raise AdjudicationError("original completion resolved-config digest differs")
    if mode == "diagnostic":
        assert diagnostic_proof is not None
        return {
            "mode": "diagnostic",
            "variant": variant,
            "run_id": run_id,
            "original_wrapper_status": status,
            "original_exit_code": exit_code,
            "original_failure": failure,
            "original_process": process,
            "known_strict_guard_failure": strict_error,
            "nullable_exception": (
                {
                    "field": f"parallelism.{STAGES_FIELD}",
                    "reason": "typed None default was explicitly passed as null, decoded as None, and omitted by native YAML serialization; all eight module-stage receipts validate the selected empty-stage behavior",
                    "evidence": nullable_evidence,
                }
                if missing_nullable is not None
                else None
            ),
            "actual_native_completion": {
                "status": summary["status"],
                "training_steps": summary["training_steps"],
                "errors": summary["errors"],
            },
            "native_capture_status": diagnostic_proof["status"],
            "native_capture_proof": diagnostic_proof,
            "validated_stage_receipts": stage_summary,
            "validation_checks": [
                "seven reviewed timing adjudications, pinned source/config/input freeze, and original-receipt hashes",
                "diagnostic direct argv, 21/20/0 cadence, and update-21/three-microbatch profile recipe",
                "clean native exit, wrapper status, immutable source/config receipts, and task TMPDIR cleanup",
                "original-capacity overlay and per-rank bootstrap receipts",
                "typed selector decode, resolved config, and per-rank module-stage receipts",
                "all eight rank updates 1..21 and frozen native profile-parser checks",
                "production tokenizer/data contract and compiler-cache producer/path",
                "Triton startup receipts, source/config immutability, bridge identity, and original trace hashes",
            ],
        }

    return {
        "variant": variant,
        "run_id": run_id,
        "original_wrapper_status": status,
        "original_exit_code": exit_code,
        "original_failure": failure,
        "original_process": process,
        "known_strict_guard_failure": strict_error,
        "nullable_exception": (
            {
                "field": f"parallelism.{STAGES_FIELD}",
                "reason": "typed None default was explicitly passed as null, decoded as None, and omitted by native YAML serialization; all eight module-stage receipts validate the selected empty-stage behavior",
                "evidence": nullable_evidence,
            }
            if missing_nullable is not None
            else None
        ),
        "actual_native_completion": {
            "status": summary["status"],
            "training_steps": summary["training_steps"],
            "window": summary["window"],
            "tokens": summary["tokens"],
            "memory": summary["memory"],
            "errors": summary["errors"],
        },
        "validated_stage_receipts": stage_summary,
        "validation_checks": [
            "frozen seven-variant source/config/input freeze",
            "historical A/B/C/D timing recipe and actual child parser",
            "140/40/100 native cadence and clean child process group",
            "original-capacity overlay and per-rank bootstrap receipts",
            "typed selector decode, resolved config, and per-rank module-stage receipts",
            "all eight rank cadence, timing, token, memory, retry, and OOM contracts",
            "production tokenizer/data contract and compiler-cache producer/path",
            "Triton startup receipts, source/config immutability, bridge identity",
        ],
    }


def _validate_rank_cadence(
    attempt: Path, run_id: str, *, expected_steps: int = TIMING_CADENCE[0]
) -> None:
    expected_cadence = list(range(1, expected_steps + 1))
    for rank in range(8):
        record = _read_json(attempt / f"measurement/rank-{rank:02d}.json")
        if (
            record.get("rank") != rank
            or record.get("world_size") != 8
            or record.get("run_id") != run_id
            or record.get("training_status") != "complete"
            or record.get("completed_steps") != expected_steps
            or [int(item["step"]) for item in record.get("cadence", [])] != expected_cadence
        ):
            raise AdjudicationError(f"rank {rank} cadence does not cover updates 1..{expected_steps}")


def _refresh_index(index_path: Path, record: dict[str, Any]) -> None:
    existing: dict[str, Any] = {
        "schema_version": 1,
        "source_freeze_sha256": EXPECTED_FREEZE_SHA256,
        "adjudications": {},
    }
    if index_path.exists():
        existing = _read_json(index_path)
    adjudications = existing.setdefault("adjudications", {})
    variant = record["variant"]
    current = adjudications.get(variant, [])
    if isinstance(current, dict):
        current = [current]
    if any(item.get("adjudication_path") == record["adjudication_path"] for item in current):
        raise AdjudicationError(f"index already contains this {variant} adjudication path")
    if record.get("mode") == "diagnostic":
        current.append({
            "mode": "diagnostic",
            "status": record["status"],
            "native_capture_status": record.get("native_capture_status"),
            "trace_tail_coverage_status": record.get("native_capture_proof", {})
            .get("trace_tail_coverage", {})
            .get("status"),
            "run_id": record["run_id"],
            "original_wrapper_status": record["original_wrapper_status"],
            "original_exit_code": record["original_exit_code"],
            "source_freeze_sha256": record["source_freeze_sha256"],
            "launch_plan_sha256": record["launch_plan_sha256"],
            "adjudicator_method_sha256": record["adjudicator_method_sha256"],
            "adjudication_path": record["adjudication_path"],
            "parser_summary_path": record["parser_summary_path"],
            "parser_summary_sha256": record["parser_summary_sha256"],
            "raw_attempt": record["raw_attempt"],
            "raw_receipt_hash_count": len(record["raw_receipt_hashes"]),
        })
    else:
        current.append({
            "status": record["status"],
            "run_id": record["run_id"],
            "original_wrapper_status": record["original_wrapper_status"],
            "original_exit_code": record["original_exit_code"],
            "source_freeze_sha256": record["source_freeze_sha256"],
            "launch_plan_sha256": record["launch_plan_sha256"],
            "adjudicator_method_sha256": record["adjudicator_method_sha256"],
            "adjudication_path": record["adjudication_path"],
            "parser_summary_path": record["parser_summary_path"],
            "parser_summary_sha256": record["parser_summary_sha256"],
            "raw_attempt": record["raw_attempt"],
            "raw_receipt_hash_count": len(record["raw_receipt_hashes"]),
            "physical_input_tokens_global": record.get("actual_native_completion", {}).get("tokens", {}).get("physical_input_tokens_global"),
            "supervised_tokens_global": record.get("actual_native_completion", {}).get("tokens", {}).get("supervised_tokens_global"),
            "primary_elapsed_s": record.get("actual_native_completion", {}).get("window", {}).get("primary_elapsed_s"),
        })
    adjudications[variant] = current
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(".json.tmp")
    payload = (json.dumps(existing, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(index_path)


def adjudicate(
    variant: str,
    attempt: Path,
    output_dir: Path,
    plan_path: Path,
    *,
    mode: str = "timing",
) -> dict[str, Any]:
    if mode not in {"timing", "diagnostic"}:
        raise AdjudicationError(f"unsupported adjudication mode {mode!r}")
    if mode == "timing" and variant not in VARIANTS:
        raise AdjudicationError(f"unsupported timing variant {variant!r}")
    if mode == "diagnostic" and variant not in DIAGNOSTIC_VARIANTS:
        raise AdjudicationError(f"{variant} has no pinned diagnostic profile")
    plan, plan_sha256 = _load_pinned_plan(plan_path)
    freeze, freeze_sha256 = _load_retained_freeze(plan)
    run_id, planned_argv = _plan_run(plan, variant, mode)
    expected_attempt = RAW_ROOT / run_id
    attempt = attempt.resolve(strict=True)
    if attempt != expected_attempt.resolve(strict=True):
        raise AdjudicationError(f"run directory must be the plan-pinned attempt {expected_attempt}")
    output_dir = output_dir.resolve()
    output_root = OUTPUT_ROOT if mode == "timing" else DIAGNOSTIC_OUTPUT_ROOT
    if output_dir != (output_root / variant).resolve():
        raise AdjudicationError(
            f"output directory must be the unique {mode} plan-004 path {output_root / variant}"
        )
    if output_dir.exists():
        raise AdjudicationError(f"refusing to overwrite existing adjudication output {output_dir}")

    # Keep parsing, cache validation and data validation out of every active timing window.
    _require_all_timing_attempts_terminal(plan)
    if mode == "diagnostic":
        _require_all_diagnostic_attempts_terminal(plan)
        _require_all_timing_adjudications_pass(plan, freeze, plan_sha256)
    authenticated_runtime_files = _authenticate_frozen_files(freeze)
    paths = _load_raw_hash_paths(attempt, mode=mode)
    raw_hashes_before = _raw_hashes(paths, attempt)
    runtime = _load_runtime()

    # The parser is read-only; its derived summary is written only under the new output directory.
    overnight_helpers = ROOT / "runs/overnight-streaming-perf"
    if str(overnight_helpers) not in sys.path:
        sys.path.insert(0, str(overnight_helpers))
    parser = _load_module(
        ROOT / "runs/overnight-streaming-perf/parse_results.py",
        "_schedule_adjudication_parser",
    )
    summary = parser.summarize_run(attempt)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    summary_sha256 = _write_json_new(summary_path, summary)
    try:
        validated = _validate(
            variant,
            run_id,
            planned_argv,
            attempt,
            summary,
            runtime,
            freeze,
            freeze_sha256,
            mode=mode,
            parser=parser,
            raw_hashes_before=raw_hashes_before,
        )
        status = (
            "ADJUDICATED_PASS"
            if mode == "timing"
            else "DIAGNOSTIC_PARSER_PASS_TAIL_PENDING"
        )
        error = None
    except Exception as exc:
        validated = {
            "variant": variant,
            "run_id": run_id,
            "original_wrapper_status": (attempt / "status.txt").read_text(encoding="utf-8").strip(),
            "original_exit_code": int((attempt / "exit-code.txt").read_text(encoding="utf-8").strip()),
        }
        status = "ADJUDICATION_REJECTED"
        error = f"{type(exc).__name__}: {exc}"
    raw_hashes_after = _raw_hashes(paths, attempt)
    raw_unchanged = raw_hashes_before == raw_hashes_after
    record = {
        "schema_version": 1,
        "status": status if raw_unchanged else "ADJUDICATION_REJECTED",
        **({"mode": "diagnostic"} if mode == "diagnostic" else {}),
        **validated,
        "adjudication_error": error if raw_unchanged else "raw attempt receipts changed during adjudication",
        "raw_attempt": str(attempt),
        "raw_receipt_hashes": raw_hashes_before,
        "raw_receipts_unchanged": raw_unchanged,
        "source_freeze_sha256": freeze_sha256,
        "source_freeze": freeze,
        "source_freeze_provenance": {
            "retained_record": str(RETAINED_FREEZE_PATH),
            "current_full_source_walk_recomputed": False,
            "historical_scan_timing": "not recorded; no duration inferred",
        },
        "launch_plan_path": str(EXPECTED_PLAN_PATH),
        "launch_plan_sha256": plan_sha256,
        "adjudicator_method_sha256": _sha256_file(Path(__file__).resolve()),
        "authenticated_runtime_files": authenticated_runtime_files,
        "parser_summary_path": str(summary_path.resolve()),
        "parser_summary_sha256": summary_sha256,
        "adjudication_path": str((output_dir / "adjudication.json").resolve()),
    }
    adjudication_sha256 = _write_json_new(output_dir / "adjudication.json", record)
    record["adjudication_sha256"] = adjudication_sha256
    _refresh_index(output_root / "compact-adjudication-index.json", record)
    return record


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("timing", "diagnostic"), default="timing")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        record = adjudicate(
            args.variant,
            args.run_dir,
            args.output_dir,
            args.plan,
            mode=args.mode,
        )
    except Exception as exc:
        print(json.dumps({"status": "ADJUDICATION_REJECTED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps({
        "status": record["status"],
        "variant": record["variant"],
        "run_id": record["run_id"],
        "source_freeze_sha256": record["source_freeze_sha256"],
        "launch_plan_sha256": record["launch_plan_sha256"],
        "adjudicator_method_sha256": record["adjudicator_method_sha256"],
        "parser_summary_path": record["parser_summary_path"],
        "parser_summary_sha256": record["parser_summary_sha256"],
        "adjudication_path": record["adjudication_path"],
        "adjudication_sha256": record["adjudication_sha256"],
    }, sort_keys=True))
    return 0 if record["status"] in {
        "ADJUDICATED_PASS",
        "DIAGNOSTIC_PARSER_PASS_TAIL_PENDING",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
