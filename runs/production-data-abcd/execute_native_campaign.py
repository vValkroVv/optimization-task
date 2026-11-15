#!/usr/bin/env python3
"""Execute the fixed native A/B/C/D campaign serially after a root GPU lease."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = TASK_ROOT / "runs/production-data-abcd"
PLAN_PATH = CAMPAIGN_DIR / "campaign-plan.json"
READINESS_PATH = CAMPAIGN_DIR / "input-readiness.json"
RAW_ROOT = CAMPAIGN_DIR / "artifacts/raw"
ORCHESTRATION_ROOT = RAW_ROOT / "native-abcd-orchestration"
PYTHON = Path("/opt/torchtitan-venv/bin/python3.12")
GIFT_ROOT = TASK_ROOT / "gift"

EXPECTED_RUNS = (
    ("proddata-a-timing-001", "A", "timing"),
    ("proddata-b-timing-001", "B", "timing"),
    ("proddata-c-timing-001", "C", "timing"),
    ("proddata-d-timing-001", "D", "timing"),
    ("proddata-a-diagnostic-001", "A", "diagnostic"),
    ("proddata-b-diagnostic-001", "B", "diagnostic"),
    ("proddata-c-diagnostic-001", "C", "diagnostic"),
    ("proddata-d-diagnostic-001", "D", "diagnostic"),
)

FREEZE_FILES = (
    "gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml",
    "runs/overnight-streaming-perf/mds-dataloader.yaml",
    "runs/production-data-abcd/data_contract.py",
    "runs/production-data-abcd/data-002.json",
    "runs/production-data-abcd/data-002-loader-summary.json",
    "runs/production-data-abcd/data-002-loader-data002-full-depth0.json",
    "runs/production-data-abcd/data-002-loader-data002-full-depth32.json",
    "runs/production-data-abcd/integration-review.md",
    "runs/overnight-streaming-perf/run.py",
    "runs/overnight-streaming-perf/measure.py",
    "runs/overnight-streaming-perf/parse_results.py",
    "runs/overnight-streaming-perf/compiler_stats.py",
    "runs/flash-qla/run.py",
    "runs/flash-qla/measure.py",
    "runs/production-data-abcd/execute_native_campaign.py",
    "runs/production-data-abcd/launch_native_campaign.sh",
)

_stop_signal: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _log(message: str) -> None:
    line = f"{_utc_now()} {message}"
    print(line, flush=True)
    log_path = ORCHESTRATION_ROOT / "campaign.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=TASK_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _gift_source_snapshot() -> dict[str, Any]:
    tracked = _git("ls-files", "-z", "--cached", "--", "gift/")
    untracked = _git(
        "ls-files", "-z", "--others", "--exclude-standard", "--", "gift/"
    )
    tracked_paths = sorted(path for path in tracked.split(b"\0") if path)
    untracked_paths = sorted(path for path in untracked.split(b"\0") if path)

    def manifest(paths: list[bytes]) -> str:
        digest = hashlib.sha256()
        for raw_path in paths:
            relative = raw_path.decode("utf-8", errors="strict")
            path = TASK_ROOT / relative
            digest.update(raw_path + b"\0")
            if not path.exists() and not path.is_symlink():
                digest.update(b"missing\0")
            elif path.is_symlink():
                digest.update(b"symlink\0")
                digest.update(os.readlink(path).encode("utf-8") + b"\0")
            else:
                digest.update(f"{path.stat().st_mode & 0o7777:o}\0".encode())
                digest.update(_sha256(path).encode("ascii") + b"\0")
        return digest.hexdigest()

    status = _git(
        "status", "--porcelain=v1", "--untracked-files=all", "--", "gift/"
    )
    return {
        "head": _git("rev-parse", "HEAD").decode().strip(),
        "tracked_file_count": len(tracked_paths),
        "untracked_file_count": len(untracked_paths),
        "tracked_sha256": manifest(tracked_paths),
        "untracked_sha256": manifest(untracked_paths),
        "tracked_and_untracked_sha256": manifest(
            sorted(set(tracked_paths).union(untracked_paths))
        ),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _plan_fingerprint(plan: dict[str, Any]) -> str:
    normalized = dict(plan)
    normalized.pop("status", None)
    runs = []
    for record in plan["runs"]:
        normalized_record = dict(record)
        normalized_record.pop("status", None)
        runs.append(normalized_record)
    normalized["runs"] = runs
    return _canonical_sha256(normalized)


def _option(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"campaign argv is missing {name}") from exc


def _validate_plan(plan: dict[str, Any], *, check_artifacts: bool) -> None:
    if plan.get("task_root") != str(TASK_ROOT):
        raise RuntimeError("campaign plan task_root differs from the active task")
    if plan.get("command_round_trip", {}).get("status") != "PASS":
        raise RuntimeError("campaign plan command round-trip is not PASS")
    if plan.get("command_round_trip", {}).get("runs") != 8:
        raise RuntimeError("campaign command round-trip does not cover eight runs")
    if _git("rev-parse", "HEAD").decode().strip() != plan.get("source_head"):
        raise RuntimeError("task HEAD differs from the prepared campaign source_head")
    records = plan.get("runs")
    if not isinstance(records, list) or len(records) != len(EXPECTED_RUNS):
        raise RuntimeError("campaign plan does not contain the fixed eight runs")
    for record, (run_id, variant, role) in zip(records, EXPECTED_RUNS, strict=True):
        if (record.get("run_id"), record.get("variant"), record.get("role")) != (
            run_id,
            variant,
            role,
        ):
            raise RuntimeError(f"campaign ordering differs at {run_id}")
        if record.get("status") != "not_started":
            raise RuntimeError(f"{run_id} already has status {record.get('status')!r}")
        if record.get("acquisition_env") != "unset":
            raise RuntimeError(f"{run_id} does not require component acquisition unset")
        argv = record.get("argv")
        if not isinstance(argv, list) or not argv or argv[0] != str(PYTHON):
            raise RuntimeError(f"{run_id} has an unexpected executable")
        if Path(argv[1]).resolve() != (CAMPAIGN_DIR.parent / "flash-qla/run.py").resolve():
            raise RuntimeError(f"{run_id} does not use the prepared native bridge")
        if _option(argv, "--run-id") != run_id:
            raise RuntimeError(f"{run_id} argv selects a different run ID")
        if Path(_option(argv, "--output-root")).resolve() != RAW_ROOT.resolve():
            raise RuntimeError(f"{run_id} output root differs from the task raw root")
        expected_steps = "20" if role == "timing" else "21"
        if _option(argv, "--steps") != expected_steps:
            raise RuntimeError(f"{run_id} has an unexpected step count")
        expected_mode = ("10", "10") if role == "timing" else ("20", "0")
        if (_option(argv, "--warmup"), _option(argv, "--window")) != expected_mode:
            raise RuntimeError(f"{run_id} has an unexpected warmup/window")
        if check_artifacts and (RAW_ROOT / run_id).exists():
            raise RuntimeError(f"refusing to overwrite existing run root: {RAW_ROOT / run_id}")


def _frozen_identity(readiness: dict[str, Any]) -> dict[str, Any]:
    evidence = readiness.get("evidence", {})
    tokenizer_dir = Path(evidence.get("tokenizer_path", ""))
    files = {relative: _sha256(TASK_ROOT / relative) for relative in FREEZE_FILES}
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = tokenizer_dir / filename
        files[str(path)] = _sha256(path)
    dataset_root = Path(evidence.get("dataset_root", ""))
    for relative in ("train/index.json",):
        path = dataset_root / relative
        files[str(path)] = _sha256(path)
    return {
        "gift_source_snapshot": _gift_source_snapshot(),
        "files_sha256": files,
        "readiness_sha256": _sha256(READINESS_PATH),
        "campaign_plan_execution_fingerprint": _plan_fingerprint(_read_json(PLAN_PATH)),
    }


def _validate_readiness(readiness: dict[str, Any]) -> None:
    if readiness.get("status") != "PASS":
        raise RuntimeError("actual-input readiness is not PASS")
    evidence = readiness.get("evidence", {})
    for key in (
        "tokenizer_file_sha256",
        "data_receipt_sha256",
        "dataset_sha256",
        "mds_index_sha256",
        "row_manifest_sha256",
        "cpu_loader_receipt_sha256",
    ):
        if not evidence.get(key):
            raise RuntimeError(f"input-readiness lacks {key}")
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = Path(evidence["tokenizer_path"]) / filename
        if _sha256(path) != evidence["tokenizer_file_sha256"].get(filename):
            raise RuntimeError(f"production tokenizer hash changed: {path}")
    if evidence.get("dataset_root") != str(
        TASK_ROOT / "runs/production-data-abcd/artifacts/raw/data-002"
    ):
        raise RuntimeError("input readiness points to a different data-002 root")


def _validate_binding_receipt(
    receipt: dict[str, Any],
    *,
    run_id: str,
    rank: int,
    expected: dict[str, Any],
    expected_head: str,
) -> dict[str, Any]:
    routing_modules = {
        "original": "torchtitan.distributed.deepep.routing",
        "fused": "torchtitan.experiments.four_fusion.routing_metadata",
    }
    ce_modules = {
        "original": "torchtitan.components.loss",
        "stable": "torchtitan.experiments.four_fusion.stable_ce",
    }
    route = receipt.get("routing_metadata_callable", {})
    ce = receipt.get("cross_entropy_callable", {})
    constructed = receipt.get("constructed_loss_callables", [])
    identity = receipt.get("source_identity", {})
    installer = receipt.get("installer_identity", {})
    if (
        receipt.get("rank") != rank
        or receipt.get("backend_variant") != expected["backend"]
        or installer.get("variant") != expected["backend"]
        or receipt.get("routing_metadata") != expected["routing"]
        or route.get("module") != routing_modules[expected["routing"]]
        or route.get("qualname") != "indices_to_map"
        or receipt.get("cross_entropy") != expected["ce"]
        or ce.get("module") != ce_modules[expected["ce"]]
        or ce.get("qualname") != "cross_entropy_loss"
        or receipt.get("capture_mode") != "disabled"
        or receipt.get("native_main_completed") is not True
        or identity.get("task_git_head") != expected_head
        or len(constructed) != 1
        or constructed[0].get("class") != "torchtitan.components.loss.MTPLoss"
        or constructed[0].get("callable") != ce
    ):
        raise RuntimeError(f"{run_id} rank {rank} native binding receipt failed")
    return {
        "rank": rank,
        "backend": receipt["backend_variant"],
        "routing": receipt["routing_metadata"],
        "routing_callable_module": route["module"],
        "cross_entropy": receipt["cross_entropy"],
        "cross_entropy_callable_module": ce["module"],
        "constructed_loss_class": constructed[0]["class"],
        "constructed_loss_callable_count": 1,
    }


def _validate_completed_capture_receipts(plan: dict[str, Any]) -> dict[str, Any]:
    run_id = "proddata-component-capture-001"
    expected = {"backend": "external_gate", "routing": "fused", "ce": "stable"}
    capture_root = RAW_ROOT / run_id / "measurement"
    checks = []
    for rank in range(8):
        path = capture_root / f"flash-qla-bootstrap-rank-{rank:02d}.json"
        checks.append(
            {
                **_validate_binding_receipt(
                    _read_json(path),
                    run_id=run_id,
                    rank=rank,
                    expected=expected,
                    expected_head=plan["source_head"],
                ),
                "receipt_path": str(path),
                "receipt_sha256": _sha256(path),
            }
        )
    return {"run_id": run_id, "status": "PASS", "ranks": checks}


def _gpu_idle_receipt() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is unavailable in the leased environment")
    gpu_query = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    devices = []
    for line in gpu_query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise RuntimeError(f"cannot parse nvidia-smi device row: {line!r}")
        devices.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
            }
        )
    if len(devices) != 8 or any("H100" not in item["name"] for item in devices):
        raise RuntimeError(f"leased host is not the expected eight-H100 allocation: {devices}")
    app_query = subprocess.run(
        [
            executable,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    active = [line.strip() for line in app_query.stdout.splitlines() if line.strip()]
    if active:
        raise RuntimeError(f"compute applications occupy the leased GPUs: {active}")
    return {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "devices": devices,
        "compute_applications": active,
        "checked_at_utc": _utc_now(),
    }


def _process_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _failure_evidence(run_id: str, bridge_pid: int | None) -> dict[str, Any]:
    run_root = RAW_ROOT / run_id
    evidence: dict[str, Any] = {
        "run_id": run_id,
        "attempt_dir": str(run_root),
        "bridge_pid": bridge_pid,
        "bridge_process_group_exists": (
            _process_group_exists(bridge_pid) if bridge_pid else None
        ),
    }
    for filename in (
        "failure.json",
        "process.json",
        "source-final.json",
        "config-hashes-final.json",
    ):
        path = run_root / filename
        if path.is_file():
            evidence[filename] = _read_json(path)
    status_path = run_root / "status.txt"
    if status_path.is_file():
        evidence["status.txt"] = status_path.read_text(encoding="utf-8").strip()
    process = evidence.get("process.json", {})
    group_id = process.get("process_group_id")
    if isinstance(group_id, int) and group_id > 0:
        evidence["native_process_group_exists_after_exit"] = _process_group_exists(group_id)
    else:
        evidence["native_process_group_exists_after_exit"] = None
    evidence["launcher_log"] = str(run_root / "logs/launcher.log")
    return evidence


def _write_plan_statuses(fingerprint: str, statuses: dict[str, str], overall: str) -> None:
    plan = _read_json(PLAN_PATH)
    if _plan_fingerprint(plan) != fingerprint:
        raise RuntimeError("campaign plan commands changed during execution")
    for record in plan["runs"]:
        if record["run_id"] in statuses:
            record["status"] = statuses[record["run_id"]]
    plan["status"] = overall
    _write_json_atomic(PLAN_PATH, plan)


def _load_run_files(run_id: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_root = RAW_ROOT / run_id
    metadata = _read_json(run_root / "metadata.json")
    completion = _read_json(run_root / "completion.json")
    summary = _read_json(run_root / "summary.json")
    return run_root, metadata, completion, summary


def _validate_finished_run(record: dict[str, Any], exit_code: int) -> dict[str, Any]:
    run_id = record["run_id"]
    role = record["role"]
    expected = record["expected"]
    if exit_code != 0:
        raise RuntimeError(f"{run_id} bridge exited with code {exit_code}")
    run_root, metadata, completion, summary = _load_run_files(run_id)
    expected_steps = 20 if role == "timing" else 21
    expected_warmup, expected_window = (10, 10) if role == "timing" else (20, 0)
    expected_features = {
        "gfsdp_row_owner_embeddings": bool(expected["three"]),
        "gfsdp_split_layer0_ffn": bool(expected["three"]),
        "gfsdp_split_mtp_ffn": bool(expected["three"]),
    }
    if completion.get("status") != "PASS" or not completion.get("source_unchanged"):
        raise RuntimeError(f"{run_id} lacks a passing completion/source receipt")
    if completion.get("config_unchanged") is not True:
        raise RuntimeError(f"{run_id} canonical or generated configuration changed")
    if metadata.get("run_id") != run_id or metadata.get("steps") != expected_steps:
        raise RuntimeError(f"{run_id} metadata does not match the fixed plan")
    if (
        metadata.get("warmup_updates") != expected_warmup
        or metadata.get("window_updates") != expected_window
        or metadata.get("gfsdp_reuse_bf16_expert_weights") is not bool(expected["reuse"])
        or metadata.get("num_prefetch_batches") != expected["prefetch"]
        or metadata.get("requested_parallelism_features") != expected_features
        or metadata.get("requested_epoch_size") != 102400
        or metadata.get("seed") != 2048
        or metadata.get("scheduler_total_steps") != 65000
        or metadata.get("scheduler_warmup_steps") != 2000
        or metadata.get("monitor_interval") != 0
        or metadata.get("detailed_monitoring")
        != "disabled; callbacks omitted; ordinary two-chunk loss"
        or metadata.get("memory_snapshots") != "disabled"
        or metadata.get("observe_compiler_cache") is not (role == "timing")
    ):
        raise RuntimeError(f"{run_id} fixed workload/feature metadata differs from the plan")
    if metadata.get("profiler") != ("disabled" if role == "timing" else "enabled for one update on global ranks 0 and 1"):
        raise RuntimeError(f"{run_id} profiler activation differs from the plan")
    profile_spec = metadata.get("profile_capture")
    if role == "timing":
        if profile_spec is not None:
            raise RuntimeError(f"{run_id} timing run unexpectedly contains a profile schedule")
    elif (
        not isinstance(profile_spec, dict)
        or profile_spec.get("target_update") != 21
        or profile_spec.get("warmup_updates") != 20
        or profile_spec.get("profile_ranks") != [0, 1]
        or profile_spec.get("profile_microbatches") != 3
        or profile_spec.get("trace_directory") != "native/profiling/traces/iteration_21"
    ):
        raise RuntimeError(f"{run_id} diagnostic profile schedule differs from the plan")
    if metadata.get("checkpointing") != "disabled" or metadata.get("validation") != "disabled":
        raise RuntimeError(f"{run_id} unexpectedly enabled checkpointing or validation")
    if summary.get("status") != "complete":
        raise RuntimeError(f"{run_id} rank summary is incomplete: {summary.get('errors')}")
    if summary.get("training_steps") != expected_steps:
        raise RuntimeError(f"{run_id} summary does not cover all configured updates")
    if role == "timing" and summary.get("world_size") != 8:
        raise RuntimeError(f"{run_id} timing summary does not cover all eight ranks")

    input_data = metadata.get("input_data", {})
    evidence = _read_json(READINESS_PATH).get("evidence", {})
    for key in (
        "dataset_root",
        "dataset_sha256",
        "mds_index_sha256",
        "row_manifest_sha256",
        "tokenizer_path",
        "tokenizer_file_sha256",
    ):
        if input_data.get(key) != evidence.get(key):
            raise RuntimeError(f"{run_id} actual input identity differs at {key}")

    bootstrap = []
    for rank in range(8):
        path = run_root / "measurement" / f"flash-qla-bootstrap-rank-{rank:02d}.json"
        receipt = _read_json(path)
        bootstrap.append(
            {
                **_validate_binding_receipt(
                    receipt,
                    run_id=run_id,
                    rank=rank,
                    expected=expected,
                    expected_head=_read_json(PLAN_PATH)["source_head"],
                ),
                "receipt_path": str(path),
                "receipt_sha256": _sha256(path),
            }
        )

    rank_records = []
    for rank in range(8):
        path = run_root / "measurement" / f"rank-{rank:02d}.json"
        rank_record = _read_json(path)
        if (
            rank_record.get("training_status") != "complete"
            or rank_record.get("completed_steps") != expected_steps
            or rank_record.get("world_size") != 8
        ):
            raise RuntimeError(f"{run_id} rank {rank} did not finish the complete workload")
        if role == "timing" and rank_record.get("measurement_status") != "complete":
            raise RuntimeError(f"{run_id} rank {rank} did not complete its timing window")
        if role == "diagnostic":
            profile = rank_record.get("profile_capture", {})
            expected_collected = 3 if rank in (0, 1) else 0
            if (
                profile.get("target_update") != 21
                or profile.get("profile_microbatches") != 3
                or profile.get("collected_microbatches") != expected_collected
                or profile.get("optimizer_step_calls") != 1
            ):
                raise RuntimeError(f"{run_id} rank {rank} did not complete its bounded diagnostic")
        rank_records.append(
            {
                "rank": rank,
                "path": str(path),
                "sha256": _sha256(path),
                "cadence": rank_record.get("cadence", []),
                "memory": rank_record.get("memory", {}),
            }
        )

    if role == "timing":
        window = summary.get("window", {})
        if (
            summary.get("mode") != "complete_window"
            or window.get("warmup_updates") != 10
            or window.get("updates") != 10
            or summary.get("compilation_observer", {}).get("status") == "disabled"
            or set(summary.get("compilation_observer", {}).get("by_rank", {}))
            != {str(rank) for rank in range(8)}
        ):
            raise RuntimeError(f"{run_id} timing summary does not contain ten measured updates")
        if metadata.get("compiler_cache_policy") != "fresh":
            raise RuntimeError(f"{run_id} timing run did not use a fresh compiler cache")
        timing = {
            "window": summary["window"],
            "tokens": summary["tokens"],
            "memory": summary["memory"],
            "host_boundary_cadence": summary["host_boundary_cadence"],
            "compilation_observer": summary.get("compilation_observer"),
        }
        trace_receipts: list[dict[str, Any]] = []
    else:
        expected_cache_producer = f"proddata-{record['variant'].lower()}-timing-001"
        if (
            metadata.get("compiler_cache_policy") != "reuse"
            or metadata.get("compiler_cache_producer_run_id") != expected_cache_producer
        ):
            raise RuntimeError(f"{run_id} diagnostic did not reuse its matching timing cache")
        profile_capture = summary.get("profile_capture", {})
        if (
            summary.get("mode") != "profile_update"
            or profile_capture.get("target_update") != 21
            or profile_capture.get("warmup_updates") != 20
            or set(profile_capture.get("traces_by_rank", {})) != {"0", "1"}
            or set(profile_capture.get("call_counts_by_rank", {}))
            != {str(rank) for rank in range(8)}
        ):
            raise RuntimeError(f"{run_id} diagnostic trace summary is incomplete")
        timing = {"profile_capture": profile_capture}
        trace_receipts = []
        for rank_key, trace in sorted(profile_capture["traces_by_rank"].items()):
            trace_path = run_root / trace["path"]
            if _sha256(trace_path) != trace.get("sha256"):
                raise RuntimeError(f"{run_id} original trace hash differs from parser receipt")
            trace_receipts.append(
                {
                    "rank": int(rank_key),
                    "path": str(trace_path),
                    "sha256": trace["sha256"],
                    "size_bytes": trace["size_bytes"],
                    "gpu_projection_counts": trace.get("gpu_projection_counts", {}),
                    "marker_counts": trace.get("marker_counts", {}),
                }
            )

    return {
        "run_id": run_id,
        "variant": record["variant"],
        "role": role,
        "status": "complete",
        "argv": record["argv"],
        "child_torchrun_argv": record["child_torchrun_argv"],
        "attempt_dir": str(run_root),
        "completion": completion,
        "metadata_sha256": _sha256(run_root / "metadata.json"),
        "summary_sha256": _sha256(run_root / "summary.json"),
        "resolved_config_sha256": completion.get("resolved_config_sha256"),
        "cache": {
            "policy": metadata.get("compiler_cache_policy"),
            "root": metadata.get("compiler_cache_root"),
            "producer_run_id": metadata.get("compiler_cache_producer_run_id"),
        },
        "input_identity": {
            "dataset_root": input_data.get("dataset_root"),
            "dataset_sha256": evidence.get("dataset_sha256"),
            "tokenizer_path": input_data.get("tokenizer_path"),
            "tokenizer_file_sha256": evidence.get("tokenizer_file_sha256"),
        },
        "bindings_by_rank": bootstrap,
        "summary": timing,
        "rank_series_and_memory": rank_records,
        "original_traces": trace_receipts,
    }


def _on_signal(signum: int, _frame: Any) -> None:
    global _stop_signal
    _stop_signal = signum


def _wait_for_child(process: subprocess.Popen[Any], run_id: str) -> int:
    while process.poll() is None:
        if _stop_signal is not None:
            _log(f"received signal {_stop_signal}; sending SIGINT to owned bridge PID {process.pid}")
            process.send_signal(signal.SIGINT)
            try:
                return process.wait(timeout=120)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"{run_id} bridge PID {process.pid} did not exit after SIGINT; inspect its owned process group"
                ) from exc
        time.sleep(1)
    return int(process.returncode)


def _run_campaign(root_lease_ref: str) -> int:
    plan = _read_json(PLAN_PATH)
    _validate_plan(plan, check_artifacts=True)
    readiness = _read_json(READINESS_PATH)
    _validate_readiness(readiness)
    identity = _frozen_identity(readiness)
    if identity["gift_source_snapshot"]["head"] != plan["source_head"]:
        raise RuntimeError("current task source HEAD differs from the campaign plan")
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise RuntimeError(f"approved runtime is unavailable: {PYTHON}")
    if not RAW_ROOT.is_dir() or RAW_ROOT.is_symlink():
        raise RuntimeError(f"verified task raw root is missing or unsafe: {RAW_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("CUDA_VISIBLE_DEVICES is nonempty; refusing a partial or remapped launch")

    ORCHESTRATION_ROOT.mkdir(mode=0o700, exist_ok=False)
    (ORCHESTRATION_ROOT / "run-logs").mkdir(mode=0o700)
    campaign_id = f"native-abcd-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    plan_fingerprint = _plan_fingerprint(plan)
    statuses = {record["run_id"]: "not_started" for record in plan["runs"]}
    result = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "status": "running",
        "root_gpu_lease_ref": root_lease_ref,
        "hostname": socket.gethostname(),
        "started_at_utc": _utc_now(),
        "plan_sha256_before_status_updates": _sha256(PLAN_PATH),
        "plan_execution_fingerprint": plan_fingerprint,
        "frozen_identity_at_start": identity,
        "runs": [],
    }
    state_path = ORCHESTRATION_ROOT / "campaign-state.json"
    results_path = ORCHESTRATION_ROOT / "native-campaign-results.json"
    _write_json_new(state_path, result)
    _write_json_new(results_path, result)
    _log(f"campaign={campaign_id} lease_ref={root_lease_ref} host={socket.gethostname()}")
    _log("campaign plan and actual input readiness validated; checking leased GPUs")

    try:
        idle_receipt = _gpu_idle_receipt()
        result["initial_gpu_idle_receipt"] = idle_receipt
        _write_json_atomic(state_path, result)
        _write_json_atomic(results_path, result)
        for index, record in enumerate(plan["runs"], start=1):
            if _stop_signal is not None:
                raise RuntimeError(f"campaign interrupted by signal {_stop_signal}")
            current_identity = _frozen_identity(_read_json(READINESS_PATH))
            if current_identity != identity:
                raise RuntimeError("frozen source/config/data/tokenizer identity changed before a run")
            gpu_receipt = _gpu_idle_receipt()
            run_id = record["run_id"]
            statuses[run_id] = "running"
            _write_plan_statuses(plan_fingerprint, statuses, f"running; run {index}/8 {run_id}")
            log_path = ORCHESTRATION_ROOT / "run-logs" / f"{run_id}.log"
            _log(f"starting {index}/8 {run_id}; exact prepared argv; GPUs idle")
            with log_path.open("xb") as log_handle:
                process = subprocess.Popen(
                    record["argv"],
                    cwd=GIFT_ROOT,
                    env={
                        key: value
                        for key, value in os.environ.items()
                        if key != "ABCD_COMPONENT_FIXTURE_DIR"
                    },
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                run_entry = {
                    "run_id": run_id,
                    "variant": record["variant"],
                    "role": record["role"],
                    "status": "running",
                    "pid": process.pid,
                    "process_group_id": process.pid,
                    "log_path": str(log_path),
                    "gpu_prelaunch_receipt": gpu_receipt,
                    "started_at_utc": _utc_now(),
                    "acquisition_environment_variable_unset": True,
                }
                result["runs"].append(run_entry)
                _write_json_atomic(state_path, result)
                _write_json_atomic(results_path, result)
                exit_code = _wait_for_child(process, run_id)
            run_entry["exit_code"] = exit_code
            run_entry["finished_at_utc"] = _utc_now()
            if exit_code != 0:
                run_entry["failure_evidence"] = _failure_evidence(run_id, process.pid)
                raise RuntimeError(
                    f"{run_id} failed with exit code {exit_code}; no retry or later run will be launched"
                )
            if _process_group_exists(process.pid):
                run_entry["failure_evidence"] = _failure_evidence(run_id, process.pid)
                raise RuntimeError(
                    f"{run_id} bridge process group survived successful bridge exit; no next run will launch"
                )
            validated_run = _validate_finished_run(record, exit_code)
            after_run_identity = _frozen_identity(_read_json(READINESS_PATH))
            run_entry["frozen_identity_after_run"] = after_run_identity
            if after_run_identity != identity:
                raise RuntimeError(f"source/config/data identity changed during {run_id}")
            run_entry.update(validated_run)
            run_entry["frozen_identity_unchanged_after_run"] = True
            run_entry["status"] = "complete"
            statuses[run_id] = "complete"
            result["completed_runs"] = index
            _write_plan_statuses(
                plan_fingerprint,
                statuses,
                f"in progress; {index}/8 complete",
            )
            _write_json_atomic(state_path, result)
            _write_json_atomic(results_path, result)
            _log(f"completed {index}/8 {run_id}; next run remains serial")

        final_identity = _frozen_identity(_read_json(READINESS_PATH))
        if final_identity != identity:
            raise RuntimeError("frozen source/config/data identity changed by campaign close")
        result["final_frozen_identity"] = final_identity
        result["status"] = "complete"
        result["finished_at_utc"] = _utc_now()
        _write_plan_statuses(plan_fingerprint, statuses, "complete; all eight requested runs passed")
        _write_json_atomic(state_path, result)
        _write_json_atomic(results_path, result)
        _log("all eight fixed native runs completed and passed per-run validation")
        return 0
    except BaseException as exc:
        result["status"] = "interrupted" if _stop_signal else "failed"
        result["failure"] = f"{type(exc).__name__}: {exc}"
        result["finished_at_utc"] = _utc_now()
        for run_entry in result["runs"]:
            if run_entry.get("status") == "running":
                run_entry["status"] = "failed"
                run_entry["failure"] = result["failure"]
                if "failure_evidence" not in run_entry:
                    run_entry["failure_evidence"] = _failure_evidence(
                        run_entry["run_id"], run_entry.get("pid")
                    )
                statuses[run_entry["run_id"]] = "failed"
        try:
            _write_plan_statuses(plan_fingerprint, statuses, f"failed; {result['failure']}")
            _write_json_atomic(state_path, result)
            _write_json_atomic(results_path, result)
            _log(f"STOP: {result['failure']}; no automatic retry")
        except Exception as record_error:
            print(f"failed to persist campaign failure state: {record_error}", file=sys.stderr)
        return 130 if _stop_signal else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="validate the prepared plan and inputs without querying GPUs or creating run directories",
    )
    parser.add_argument(
        "--root-gpu-lease-ref",
        help="explicit root lease reference authorizing reuse of the existing eight-H100 allocation",
    )
    args = parser.parse_args(argv)
    if args.review_only:
        plan = _read_json(PLAN_PATH)
        _validate_plan(plan, check_artifacts=True)
        readiness = _read_json(READINESS_PATH)
        _validate_readiness(readiness)
        frozen_identity = _frozen_identity(readiness)
        capture_receipts = _validate_completed_capture_receipts(plan)
        print(
            json.dumps(
                {
                    "status": "READY_FOR_ROOT_GPU_LEASE",
                    "frozen_identity_sha256": _canonical_sha256(frozen_identity),
                    "existing_native_capture_binding_check": capture_receipts,
                    "runs": [
                        {
                            "run_id": record["run_id"],
                            "variant": record["variant"],
                            "role": record["role"],
                            "argv": record["argv"],
                        }
                        for record in plan["runs"]
                    ],
                    "gpu_queried": False,
                    "run_directories_created": False,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if not args.root_gpu_lease_ref or not args.root_gpu_lease_ref.strip():
        parser.error("execution requires --root-gpu-lease-ref after root explicitly grants the lease")
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    return _run_campaign(args.root_gpu_lease_ref.strip())


if __name__ == "__main__":
    sys.exit(main())
