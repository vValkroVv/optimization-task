#!/usr/bin/env python3
"""Print or execute the frozen seven-configuration timing and trace campaign."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
RAW = EVIDENCE / "artifacts/raw"
PLAN_PATH = ROOT / "runs/production-data-abcd/campaign-plan.json"
SAVED_D_CACHE = ROOT / "runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache"
PYTHON = "/opt/torchtitan-venv/bin/python3.12"
FLASH_RUNNER = ROOT / "runs/flash-qla/run.py"
FLASH_MEASURE = ROOT / "runs/flash-qla/measure.py"
RUNNER = EVIDENCE / "run.py"
MEASURE = EVIDENCE / "measure.py"
RECEIPTS = RAW / "campaign-140-receipts.jsonl"

VARIANTS = ("A", "B", "C", "D0", "S", "SE", "SE-fused")
TIMING_RUNS = tuple(f"{variant}-timing" for variant in VARIANTS)
DIAGNOSTIC_VARIANTS = ("D0", "S", "SE", "SE-fused")
DIAGNOSTIC_RUNS = tuple(f"{variant}-diagnostic" for variant in DIAGNOSTIC_VARIANTS)
RUN_NAMES = (*TIMING_RUNS, *DIAGNOSTIC_RUNS)
RUN_IDS = {name: f"schedule-embedding-{name.lower()}-140-003" for name in RUN_NAMES}
TIMING_CADENCE = (140, 40, 100)
DIAGNOSTIC_CADENCE = (21, 20, 0)
GLOBAL_PHYSICAL_TOKENS = 209_715_200
GLOBAL_SUPERVISED_TOKENS = 209_664_000
LOCAL_PHYSICAL_TOKENS = GLOBAL_PHYSICAL_TOKENS // 8
SPLIT_MTP_FIELD = "gfsdp_split_mtp_ffn"
SPLIT_LAYER0_FIELD = "gfsdp_split_layer0_ffn"
ROW_OWNER_FIELD = "gfsdp_row_owner_embeddings"
SCHEDULING_FIELD = "gfsdp_mtp_transition_prefetch"
EMBEDDING_FIELD = "gfsdp_row_owner_embedding_optimizations"
STAGES_FIELD = "gfsdp_row_owner_embedding_optimization_stages"
EMBEDDING_STAGES = {
    "SE": ["compact", "owner_add", "owner_lookup"],
    "SE-fused": ["compact", "owner_add", "owner_lookup", "routing"],
}
CAPACITY_CONTROL_SHA256 = "2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120"
BASE_D_FLAGS = (
    "--gfsdp-reuse-bf16-expert-weights",
    "--gfsdp-split-mtp-ffn",
    "--gfsdp-split-layer0-ffn",
    "--gfsdp-row-owner-embeddings",
)

_active: subprocess.Popen[bytes] | None = None
_requested_signal: int | None = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_modules() -> tuple[Any, Any]:
    runner = _load_module(RUNNER, "_schedule_embedding_campaign_runner")
    flash = _load_module(FLASH_RUNNER, "_schedule_embedding_campaign_flash_runner")
    return runner, flash


def _read_plan() -> dict[str, Any]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan.get("command_round_trip", {}).get("status") != "PASS":
        raise RuntimeError("production D campaign plan lacks a passing command round-trip")
    return plan


def _option_values(argv: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                raise RuntimeError(f"{option} lacks a value")
            values.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.partition("=")[2])
    return values


def _replace_option(argv: list[str], option: str, value: str) -> list[str]:
    result = list(argv)
    matches = [i for i, token in enumerate(result) if token == option]
    equals = [i for i, token in enumerate(result) if token.startswith(f"{option}=")]
    if len(matches) + len(equals) > 1:
        raise RuntimeError(f"expected at most one {option} in D campaign recipe")
    if matches:
        if matches[0] + 1 >= len(result):
            raise RuntimeError(f"{option} lacks a value")
        result[matches[0] + 1] = value
    elif equals:
        result[equals[0]] = f"{option}={value}"
    else:
        result.extend((option, value))
    return result


def _base_argv(variant: str, mode: str) -> list[str]:
    if mode not in {"timing", "diagnostic"}:
        raise ValueError(f"unsupported mode {mode!r}")
    if mode == "diagnostic" and variant not in DIAGNOSTIC_VARIANTS:
        raise ValueError(f"{variant} has no requested diagnostic profile")
    recipe_variant = variant if variant in {"A", "B", "C"} else "D"
    rows = [
        row for row in _read_plan().get("runs", [])
        if row.get("variant") == recipe_variant and row.get("role") == mode
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one existing {recipe_variant} {mode} recipe")
    argv = list(rows[0].get("argv", []))
    if len(argv) < 3 or argv[:2] != [PYTHON, str(FLASH_RUNNER)]:
        raise RuntimeError(f"{recipe_variant} recipe does not use the pinned Python and FlashQLA runner")
    expected = rows[0].get("expected", {})
    if recipe_variant == "D":
        if any(flag not in argv for flag in BASE_D_FLAGS):
            raise RuntimeError("D recipe lost one of its existing feature controls")
        if (
            _option_values(argv, "--gdn-backend") != ["external_gate"]
            or _option_values(argv, "--routing-metadata") != ["fused"]
            or _option_values(argv, "--cross-entropy") != ["stable"]
            or _option_values(argv, "--num-prefetch-batches") != ["32"]
            or _option_values(argv, "--epoch-size") != ["102400"]
            or expected.get("three") is not True
        ):
            raise RuntimeError("D recipe controls differ from the accepted production plan")
    elif (
        expected.get("three") is not False
        or any(flag in argv for flag in BASE_D_FLAGS[1:])
        or _option_values(argv, "--epoch-size") != ["102400"]
    ):
        raise RuntimeError(f"historical {recipe_variant} recipe no longer matches the ABCD plan")
    if mode == "timing":
        for option, value in zip(("--steps", "--warmup", "--window"), TIMING_CADENCE):
            argv = _replace_option(argv, option, str(value))
    elif tuple(
        int(_option_values(argv, option)[0])
        for option in ("--steps", "--warmup", "--window")
    ) != DIAGNOSTIC_CADENCE:
        raise RuntimeError("D diagnostic recipe differs from the update-21 protocol")
    return argv


def _run_argv(name: str) -> list[str]:
    if name not in RUN_IDS:
        raise ValueError(f"unknown run name {name!r}")
    variant, mode = name.rsplit("-", maxsplit=1)
    argv = _base_argv(variant, mode)
    argv[1] = str(RUNNER)
    argv[2:2] = ["--variant", variant]
    argv = _replace_option(argv, "--run-id", RUN_IDS[name])
    argv = _replace_option(argv, "--output-root", str(RAW))
    argv = _replace_option(argv, "--compiler-cache-root", str(SAVED_D_CACHE))
    return argv


def _controls(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported schedule/embedding variant {variant!r}")
    row_owner = variant not in {"A", "B", "C"}
    return {
        SPLIT_MTP_FIELD: row_owner,
        SPLIT_LAYER0_FIELD: row_owner,
        ROW_OWNER_FIELD: row_owner,
        SCHEDULING_FIELD: variant in {"S", "SE", "SE-fused"},
        EMBEDDING_FIELD: False,
        STAGES_FIELD: EMBEDDING_STAGES.get(variant),
    }


def _control_argument(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ", ".join(value) + "]"
    raise TypeError(f"unsupported control value {value!r}")


def _native_parallelism_config(training_argv: list[str]) -> Any:
    from torchtitan.config.cli import load_document
    from torchtitan.config.configs import ParallelismConfig
    from torchtitan.config.decode import decode
    from torchtitan.config.overrides import parse_argv

    config_path, raw_overrides = parse_argv(training_argv)
    document, _ = load_document(config_path, raw_overrides)
    return decode(
        ParallelismConfig,
        document.data.get("parallelism", {}),
        path="parallelism",
    )


def _dry_check_one(name: str, runner: Any, flash: Any, base: Any, native_measure: Any) -> dict[str, Any]:
    variant, mode = name.rsplit("-", maxsplit=1)
    argv = _run_argv(name)
    parsed_variant, forwarded = runner._parse_variant_args(argv[2:])
    if parsed_variant != variant:
        raise RuntimeError(f"{name}: variant parser changed the selected variant")
    bridge, forwarded = flash._parse_bridge_args(forwarded)
    args = base._parse_args(forwarded)
    recipe_variant = variant if variant in {"A", "B", "C"} else "D"
    recipe = next(
        row
        for row in _read_plan()["runs"]
        if row.get("variant") == recipe_variant and row.get("role") == mode
    )
    recipe_expected = recipe["expected"]
    if (bridge.gdn_backend, bridge.routing_metadata, bridge.cross_entropy) != (
        recipe_expected["backend"],
        recipe_expected["routing"],
        recipe_expected["ce"],
    ):
        raise RuntimeError(f"{name}: FlashQLA parser changed the historical recipe")
    expected_legacy_controls = {
        SPLIT_MTP_FIELD: recipe_expected["three"],
        SPLIT_LAYER0_FIELD: recipe_expected["three"],
        ROW_OWNER_FIELD: recipe_expected["three"],
    }
    if (
        args.gfsdp_reuse_bf16_expert_weights is not recipe_expected["reuse"]
        or args.num_prefetch_batches != recipe_expected["prefetch"]
        or args.epoch_size != 102400
        or base._native_feature_values(args) != expected_legacy_controls
    ):
        raise RuntimeError(f"{name}: base parser changed historical recipe controls")
    cadence = TIMING_CADENCE if mode == "timing" else DIAGNOSTIC_CADENCE
    if (args.steps, args.warmup, args.window) != cadence:
        raise RuntimeError(f"{name}: unexpected runner cadence")
    if args.gfsdp_buffer_inventory:
        raise RuntimeError(f"{name}: profile and timing runs must not collect buffer inventory")
    if mode == "diagnostic" and (args.profile_update, args.profile_microbatches) != (21, 3):
        raise RuntimeError(f"{name}: diagnostic must profile update 21 for three microbatches")
    if mode == "timing" and args.profile_update is not None:
        raise RuntimeError(f"{name}: timing run must not enable profiling")

    controls = _controls(variant)
    with tempfile.TemporaryDirectory(prefix="schedule-command-", dir=RAW) as temporary:
        root = Path(temporary)
        paths = {
            "logs": root / "logs",
            "measurement": root / "measurement",
            "config": root / "config",
            "native": root / "native",
        }
        paths["config"].mkdir()
        base._write_experiment_config(paths["config"] / "experiment.yaml")
        command = runner._build_command_hook(
            flash._build_command_hook, base, variant
        )(args, paths)
        measure_path = str(MEASURE.resolve())
        if command.count(measure_path) != 1:
            raise RuntimeError(f"{name}: child command did not select schedule measure.py")
        for field, value in controls.items():
            option = f"--parallelism.{field}"
            if _option_values(command, option) != [_control_argument(value)]:
                raise RuntimeError(f"{name}: generated command has the wrong {option}")

        expected = runner._variant_config_expectations(
            base._resolved_config_expectations, variant
        )(args)
        for field, value in controls.items():
            if expected.get(f"parallelism.{field}") != value:
                raise RuntimeError(f"{name}: resolved-config guard lost {field}")
        hashes = runner._build_config_hashes_hook(
            flash._build_config_hashes_hook, base, variant
        )()
        if any(
            not isinstance(hashes.get(key), str) or len(hashes[key]) != 64
            for key in (
                "schedule_embedding_variant_sha256",
                "schedule_embedding_controls_sha256",
                "schedule_embedding_run_py_sha256",
                "schedule_embedding_measure_py_sha256",
                "schedule_embedding_campaign_py_sha256",
            )
        ):
            raise RuntimeError(f"{name}: source/config hash guard is incomplete")

        measure_index = command.index(measure_path)
        measured, training_argv = native_measure._parse_args(
            command[measure_index + 1 :]
        )
        if (measured.steps, measured.warmup, measured.window) != cadence:
            raise RuntimeError(f"{name}: measurement parser disagrees with the run recipe")
        if measured.gfsdp_buffer_inventory:
            raise RuntimeError(f"{name}: measurement parser enabled buffer inventory")
        if mode == "diagnostic" and (
            measured.profile_update != 21 or measured.profile_microbatches != 3
        ):
            raise RuntimeError(f"{name}: measurement parser lost the diagnostic profile")
        parallelism_config = _native_parallelism_config(training_argv)

    for field, value in controls.items():
        actual = getattr(parallelism_config, field)
        expected_value = (
            tuple(value) if field == STAGES_FIELD and value is not None else value
        )
        if actual != expected_value:
            raise RuntimeError(
                f"{name}: native config round trip resolved {field}={actual!r}, "
                f"expected {expected_value!r}"
            )
    return {
        "name": name,
        "steps_warmup_window": cadence,
        "feature_values": controls,
        "resolved_parallelism": dataclasses.asdict(parallelism_config),
        "child_measure_path": measure_path,
        "child_argv_tokens": len(command),
        "child_argv_sha256": _sha256_bytes(
            json.dumps(command, separators=(",", ":")).encode("utf-8")
        ),
        "training_override_count": len(training_argv) - 1,
    }


def _dry_check_capacity(measure: Any) -> dict[str, Any]:
    capacity_path = measure.CURRENT_CAPACITY_PATH
    before = hashlib.sha256(capacity_path.read_bytes()).hexdigest()
    module, identity = measure._load_original_capacity_module()
    import torch

    result = module._branchless_capacity_mask(
        torch.tensor([0.9, 0.7, 0.8, 0.6]),
        torch.tensor([0, 0, 1, 0], dtype=torch.int64),
        torch.ones(4, dtype=torch.bool),
        torch.tensor([3, 1], dtype=torch.int64),
        capacity=2,
    )
    if result.tolist() != [True, True, True, False]:
        raise RuntimeError(f"original capacity function returned {result.tolist()}")
    parent = importlib.import_module("torchtitan.models.common.moe.dispatch")
    imported = importlib.import_module(measure.CAPACITY_MODULE_NAME)
    after = hashlib.sha256(capacity_path.read_bytes()).hexdigest()
    if imported is not module or sys.modules[measure.CAPACITY_MODULE_NAME] is not module:
        raise RuntimeError("package import replaced the original capacity module")
    parent_capacity = getattr(parent, "capacity", None)
    if parent_capacity is not None and parent_capacity is not module:
        raise RuntimeError("dispatch parent points to another capacity module")
    if before != after:
        raise RuntimeError("current contributor capacity.py bytes changed")
    return {
        "status": "PASS",
        "module": identity,
        "function_result": result.tolist(),
        "current_capacity_sha256_before": before,
        "current_capacity_sha256_after": after,
    }


def _dry_check_main_hooks(runner: Any, flash: Any, base: Any) -> list[dict[str, Any]]:
    original_runner_loader = runner._load_flash_runner
    original_flash_loader = flash._load_base_runner
    original_main = base.main
    original_environment = base._run_environment
    results: list[dict[str, Any]] = []
    active_paths: dict[str, Path] = {}

    def stub_main(argv: list[str]) -> int:
        args = base._parse_args(argv)
        command = base._command(args, active_paths)
        hashes = base._config_hashes()
        env = base._run_environment(active_paths)
        expected = base._resolved_config_expectations(args)
        variant = env.get("TASK_SCHEDULE_EMBEDDING_VARIANT", "")
        controls = _controls(variant)
        if command.count(str(MEASURE.resolve())) != 1:
            raise RuntimeError("runner.main hook test did not replace the measure script")
        for field, value in controls.items():
            option = f"--parallelism.{field}"
            if _option_values(command, option) != [_control_argument(value)]:
                raise RuntimeError(f"runner.main hook test generated the wrong {option}")
            if expected.get(f"parallelism.{field}") != value:
                raise RuntimeError(f"runner.main hook test installed the wrong config guard for {field}")
        try:
            identity = json.loads(env["TASK_SCHEDULE_EMBEDDING_IDENTITY"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("runner.main hook test did not install a valid variant identity") from exc
        bridge = json.loads(
            (active_paths["attempt"] / "schedule-embedding-bridge.json").read_text(encoding="utf-8")
        )
        if identity.get("variant") != variant or bridge.get("variant") != variant:
            raise RuntimeError("runner.main hook test variant differs across command and environment hooks")
        if any(hashes.get(key) != value for key, value in bridge.get("config_hash_guard", {}).items()):
            raise RuntimeError("runner.main hook test source/config hash guard differs")
        results.append(
            {
                "variant": variant,
                "measure_path": str(MEASURE.resolve()),
                "control_values": controls,
                "config_hash_guard_keys": sorted(bridge["config_hash_guard"]),
            }
        )
        return 0

    runner._load_flash_runner = lambda: flash
    flash._load_base_runner = lambda: base
    base.main = stub_main
    try:
        for variant in VARIANTS:
            with tempfile.TemporaryDirectory(prefix="schedule-hooks-", dir=EVIDENCE) as temporary:
                attempt = Path(temporary) / "attempt"
                attempt.mkdir()
                cache = attempt / "cache"
                cache.mkdir()
                active_paths = {
                    "attempt": attempt,
                    "cache": cache,
                    "logs": attempt / "logs",
                    "measurement": attempt / "measurement",
                    "config": attempt / "config",
                    "native": attempt / "native",
                }
                base._run_environment = lambda _paths: {"TASK_HOOK_TEST": "true"}
                argv = _run_argv(f"{variant}-timing")
                if runner.main(argv[2:]) != 0:
                    raise RuntimeError(f"runner.main hook test failed for {variant}")
    finally:
        runner._load_flash_runner = original_runner_loader
        flash._load_base_runner = original_flash_loader
        base.main = original_main
        base._run_environment = original_environment
    if [item["variant"] for item in results] != list(VARIANTS):
        raise RuntimeError("runner.main hook test did not exercise all seven variants")
    return results


def _dry_check_stage_receipts(measure: Any) -> list[dict[str, Any]]:
    import torch.nn as nn
    from torchtitan.distributed.gfsdp.row_owner_embedding import RowOwnerEmbedding

    def owner_with_stages(stages: list[str]) -> RowOwnerEmbedding:
        owner = object.__new__(RowOwnerEmbedding)
        nn.Module.__init__(owner)
        owner._optimization_stages = tuple(stages)
        return owner

    receipts = []
    for variant in VARIANTS:
        with tempfile.TemporaryDirectory(prefix="stage-receipt-", dir=RAW) as temporary:
            measurement = Path(temporary)
            part = nn.Module()
            expected_stages = EMBEDDING_STAGES.get(variant, [])
            if variant not in {"A", "B", "C"}:
                part.add_module("tok_embeddings", owner_with_stages(expected_stages))

            class DummyMeasurement:
                def __init__(
                    self,
                    *,
                    trainer: Any,
                    output_dir: Path,
                    run_id: str,
                    rank: int,
                ):
                    self.trainer = trainer
                    self.output_dir = output_dir
                    self.run_id = run_id
                    self.rank = rank

            base = type("DryBase", (), {"WindowMeasurement": DummyMeasurement})
            original_init = DummyMeasurement.__init__
            flash = type(
                "DryFlash",
                (),
                {
                    "_write_bootstrap_receipt": staticmethod(lambda **_kwargs: {}),
                    "_update_bootstrap_receipt": staticmethod(lambda *_args: None),
                    "_load_base_measure": staticmethod(lambda: base),
                },
            )
            identity = {"source_sha256": {"dry_check": "test"}}
            measure._patch_flash_measure(
                flash,
                module=None,
                capacity_identity={},
                identity=identity,
                variant=variant,
            )
            resolved_base = flash._load_base_measure()
            trainer = type("DryTrainer", (), {"model_parts": [part]})()
            resolved_base.WindowMeasurement(
                trainer=trainer,
                output_dir=measurement,
                run_id=f"dry-{variant}",
                rank=0,
            )
            if resolved_base.WindowMeasurement.__init__ is not original_init:
                raise RuntimeError(f"{variant}: stage-selection hook did not restore __init__")
            path = measurement / "embedding-stage-selection-rank-00.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                record.get("variant") != variant
                or record.get("expected_stages") != expected_stages
                or record.get("capture_stage")
                != "after_trainer_init_before_first_optimizer_update"
                or record.get("device_reads") != 0
            ):
                raise RuntimeError(f"{variant}: native stage receipt hook dry check failed")
            receipts.append({"variant": variant, "record": record})
    return receipts


def _dry_checks() -> dict[str, Any]:
    if sys.executable != PYTHON:
        raise RuntimeError(f"run dry checks with {PYTHON}; found {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("dry checks require CUDA_VISIBLE_DEVICES='' to hide CUDA")
    sys.path.insert(0, str(ROOT / "gift"))
    runner, flash = _load_runner_modules()
    base = flash._load_base_runner()
    measure = _load_module(MEASURE, "_schedule_embedding_campaign_measure")
    capacity = _dry_check_capacity(measure)
    flash_measure = _load_module(FLASH_MEASURE, "_schedule_embedding_campaign_flash_measure")
    native_measure = flash_measure._load_base_measure()
    checks = [
        _dry_check_one(f"{variant}-timing", runner, flash, base, native_measure)
        for variant in VARIANTS
    ]
    checks.extend(
        _dry_check_one(name, runner, flash, base, native_measure)
        for name in DIAGNOSTIC_RUNS
    )
    by_name = {item["name"]: item for item in checks}
    se_config = dict(by_name["SE-timing"]["resolved_parallelism"])
    fused_config = dict(by_name["SE-fused-timing"]["resolved_parallelism"])
    se_config.pop(STAGES_FIELD)
    fused_config.pop(STAGES_FIELD)
    if se_config != fused_config:
        raise RuntimeError("SE and SE-fused differ outside embedding routing stage selection")
    if by_name["SE-timing"]["feature_values"][STAGES_FIELD] != [
        "compact", "owner_add", "owner_lookup"
    ] or by_name["SE-fused-timing"]["feature_values"][STAGES_FIELD] != [
        "compact", "owner_add", "owner_lookup", "routing"
    ]:
        raise RuntimeError("SE/SE-fused do not resolve the required stage selections")
    main_hooks = _dry_check_main_hooks(runner, flash, base)
    stage_receipts = _dry_check_stage_receipts(measure)
    if len(checks) != 11 or len(stage_receipts) != 7:
        raise RuntimeError("dry check did not cover all seven timing and four diagnostic commands")
    return {
        "status": "PASS",
        "gpu_execution": "none",
        "cuda_visible_devices": "",
        "capacity": capacity,
        "commands": checks,
        "stage_receipt_hook": stage_receipts,
        "runner_main_hooks": main_hooks,
    }


def _plan(names: list[str]) -> dict[str, Any]:
    return {
        "execute_with": "campaign.py --execute --runs <ordered names>",
        "saved_d_cache": str(SAVED_D_CACHE),
        "runs": [
            {
                "name": name,
                "run_id": RUN_IDS[name],
                "argv": _run_argv(name),
            }
            for name in names
        ],
    }


def _append_receipt(value: dict[str, Any]) -> None:
    if RECEIPTS.is_symlink():
        raise RuntimeError(f"campaign receipt file is a symlink: {RECEIPTS}")
    RAW.mkdir(parents=True, exist_ok=True)
    with RECEIPTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _previous_run_ids() -> set[str]:
    if not RECEIPTS.exists():
        return set()
    return {
        value["run_id"]
        for value in (json.loads(line) for line in RECEIPTS.read_text().splitlines() if line)
        if isinstance(value.get("run_id"), str)
    }


def _preflight_idle_h100s() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty for the eight-GPU D recipe")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.splitlines()
    rows = [tuple(field.strip() for field in row.split(",")) for row in gpu]
    if len(rows) != 8 or any(len(row) != 3 or "H100" not in row[1] or row[2] != "0" for row in rows):
        raise RuntimeError(f"expected eight idle H100 GPUs, found {rows}")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if apps and "No running processes found" not in apps:
        raise RuntimeError(f"GPU compute applications are present: {apps}")
    return {"gpus": rows, "compute_apps": apps or "none"}


def _group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_group_exit(group_id: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _group_exists(group_id):
            return True
        time.sleep(0.2)
    return not _group_exists(group_id)


def _stop_owned_group(process: subprocess.Popen[bytes]) -> bool:
    for sig, seconds in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if not _group_exists(process.pid):
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        _wait_group_exit(process.pid, seconds)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return not _group_exists(process.pid)


def _run_ids_from_receipts() -> set[str]:
    try:
        return _previous_run_ids()
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"campaign receipt file is malformed: {RECEIPTS}") from exc


def _expected_features(variant: str) -> dict[str, Any]:
    return _controls(variant)


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required campaign input is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _campaign_freeze(base: Any, flash: Any, runner: Any, measure: Any) -> dict[str, Any]:
    source_snapshot = base.source_snapshot(ROOT)
    base_hashes = base._config_hashes()
    flash_identity = flash._source_identity(base)
    schedule_hashes = runner._source_hashes()
    capacity_path = Path(measure.ORIGINAL_CAPACITY_PATH)
    capacity_head_sha256 = _sha256_bytes(
        _git("show", f"HEAD:{measure.CAPACITY_RELATIVE_PATH}")
    )
    capacity_snapshot_sha256 = _sha256_file(capacity_path)
    if (
        capacity_snapshot_sha256 != CAPACITY_CONTROL_SHA256
        or capacity_head_sha256 != CAPACITY_CONTROL_SHA256
    ):
        raise RuntimeError("original-capacity snapshot or HEAD bytes differ from the verified control")
    helper_paths = {
        "production_data_contract": Path(base.PRODUCTION_DATA_DIR) / "data_contract.py",
        "overnight_runner": Path(base.HELPER_DIR) / "run.py",
        "overnight_measure": Path(base.HELPER_DIR) / "measure.py",
        "overnight_parser": Path(base.HELPER_DIR) / "parse_results.py",
        "compiler_observer": Path(base.HELPER_DIR) / "compiler_stats.py",
        "streaming_filelock": Path(base.HELPER_DIR) / "streaming_filelock.py",
        "historical_campaign_plan": PLAN_PATH,
        "original_capacity_snapshot": capacity_path,
    }
    executed_helpers = {
        name: {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
        for name, path in helper_paths.items()
    }
    return {
        "gift_source_snapshot": source_snapshot,
        "base_config_hashes": base_hashes,
        "flash_source_identity": flash_identity,
        "schedule_source_sha256": schedule_hashes,
        "executed_helpers": executed_helpers,
        "capacity_head_sha256": capacity_head_sha256,
        "capacity_snapshot_sha256": capacity_snapshot_sha256,
    }


def _freeze_sha256(freeze: dict[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _validate_finished(
    name: str, argv: list[str], freeze: dict[str, Any], base: Any
) -> dict[str, Any]:
    variant, mode = name.rsplit("-", maxsplit=1)
    run_id = RUN_IDS[name]
    attempt = RAW / run_id
    completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
    process = json.loads((attempt / "process.json").read_text(encoding="utf-8"))
    metadata = json.loads((attempt / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
    bridge = json.loads((attempt / "schedule-embedding-bridge.json").read_text(encoding="utf-8"))
    controls = _expected_features(variant)
    steps, warmup, window = TIMING_CADENCE if mode == "timing" else DIAGNOSTIC_CADENCE
    if (
        completion.get("status") != "PASS"
        or completion.get("source_unchanged") is not True
        or completion.get("config_unchanged") is not True
        or process.get("process_group_stopped") is not True
        or process.get("orphaned_process_group") is True
        or metadata.get("run_id") != run_id
        or metadata.get("steps") != steps
        or metadata.get("warmup_updates") != warmup
        or metadata.get("window_updates") != window
        or summary.get("status") != "complete"
        or summary.get("training_steps") != steps
        or summary.get("errors")
        or metadata.get("compiler_cache_policy") != "reuse"
        or Path(metadata.get("compiler_cache_root", "")) != SAVED_D_CACHE
        or metadata.get("compiler_cache_producer_run_id") != "proddata-d-timing-001"
        or bridge.get("variant") != variant
        or bridge.get("controls") != controls
        or metadata.get("gfsdp_buffer_inventory") is not False
    ):
        raise RuntimeError(f"{run_id} completion, metadata, summary, or variant receipt failed")
    if metadata.get("source_snapshot_before") != freeze["gift_source_snapshot"]:
        raise RuntimeError(f"{run_id} source snapshot differs from the campaign freeze")
    config_hashes = metadata.get("config_hashes_before", {})
    config_guard = bridge.get("config_hash_guard", {})
    if any(
        config_hashes.get(key) != value
        for key, value in freeze["base_config_hashes"].items()
    ):
        raise RuntimeError(f"{run_id} base configuration/input hashes differ from the freeze")
    if config_hashes.get("flash_qla_bridge_sources") != freeze["flash_source_identity"]["bridge_sources"]:
        raise RuntimeError(f"{run_id} FlashQLA bridge hashes differ from the freeze")
    if (
        len(config_guard) != 5
        or any(config_hashes.get(key) != value for key, value in config_guard.items())
        or bridge.get("source_sha256") != freeze["schedule_source_sha256"]
    ):
        raise RuntimeError(f"{run_id} schedule/embedding source or variant hash guard differs")
    flash_bridge = json.loads((attempt / "flash-qla-bridge.json").read_text(encoding="utf-8"))
    if flash_bridge.get("source_identity") != freeze["flash_source_identity"]:
        raise RuntimeError(f"{run_id} FlashQLA source identity differs from the campaign freeze")
    actual = metadata.get("argv", [])
    if str(MEASURE.resolve()) not in actual or "--nproc_per_node=8" not in actual:
        raise RuntimeError(f"{run_id} actual child command does not use schedule measure.py on 8 ranks")
    for field, value in controls.items():
        if _option_values(actual, f"--parallelism.{field}") != [_control_argument(value)]:
            raise RuntimeError(f"{run_id} actual child config differs for {field}")
    resolved = base._load_yaml(attempt / "native/config/resolved_config.yaml")
    for field, value in controls.items():
        actual_value = resolved.get("parallelism", {}).get(field, object())
        if actual_value != value:
            raise RuntimeError(
                f"{run_id} resolved parallelism.{field}={actual_value!r}, expected {value!r}"
            )

    cadence_steps = list(range(1, steps + 1))
    rank_paths = []
    stage_paths = []
    for rank in range(8):
        path = attempt / f"measurement/rank-{rank:02d}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("rank") != rank
            or record.get("world_size") != 8
            or record.get("training_status") != "complete"
            or record.get("completed_steps") != steps
            or [int(item["step"]) for item in record.get("cadence", [])]
            != cadence_steps
        ):
            raise RuntimeError(f"{run_id} rank {rank} cadence does not cover all {steps} updates")
        rank_paths.append(str(path))

        stage_path = attempt / f"measurement/embedding-stage-selection-rank-{rank:02d}.json"
        stage_receipt = json.loads(stage_path.read_text(encoding="utf-8"))
        expected_stages = controls[STAGES_FIELD] or []
        expected_count = 0 if variant in {"A", "B", "C"} else 1
        actual_stages = [
            item.get("optimization_stages") for item in stage_receipt.get("modules", [])
        ]
        if (
            stage_receipt.get("run_id") != run_id
            or stage_receipt.get("rank") != rank
            or stage_receipt.get("world_size") != 8
            or stage_receipt.get("variant") != variant
            or stage_receipt.get("source_sha256") != freeze["schedule_source_sha256"]
            or stage_receipt.get("capture_stage")
            != "after_trainer_init_before_first_optimizer_update"
            or stage_receipt.get("device_reads") != 0
            or len(actual_stages) != expected_count
            or actual_stages != [expected_stages] * expected_count
        ):
            raise RuntimeError(f"{run_id} rank {rank} resolved embedding stages differ")
        stage_paths.append(str(stage_path))

    if mode == "diagnostic":
        profile = metadata.get("profile_capture", {})
        if (
            profile.get("target_update") != 21
            or profile.get("profile_ranks") != [0, 1]
            or profile.get("profile_microbatches") != 3
        ):
            raise RuntimeError(f"{run_id} diagnostic metadata differs from the rank-0/1 recipe")
    for rank in range(8):
        path = attempt / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        schedule = receipt.get("schedule_embedding", {})
        control = schedule.get("capacity_control", {})
        final = schedule.get("capacity_control_final_check", {})
        if (
            schedule.get("variant") != variant
            or schedule.get("controls") != controls
            or control.get("module_name") != "torchtitan.models.common.moe.dispatch.capacity"
            or control.get("module_sha256") != CAPACITY_CONTROL_SHA256
            or control.get("loaded_before_native_imports") is not True
            or final.get("registered_same_object") is not True
            or final.get("module_path_is_original_snapshot") is not True
            or final.get("current_capacity_file_unchanged") is not True
            or schedule.get("source_sha256") != bridge.get("source_sha256")
        ):
            raise RuntimeError(f"{run_id} rank {rank} capacity/variant receipt is invalid")
    if mode == "timing":
        measured = summary.get("window", {})
        tokens = summary.get("tokens", {})
        if (
            summary.get("mode") != "complete_window"
            or measured.get("updates") != 100
            or measured.get("warmup_updates") != 40
            or measured.get("start_after_update") != 40
            or measured.get("end_after_update") != 140
            or measured.get("global_envelope_convention")
            != "max(rank end monotonic timestamp) - min(rank start timestamp)"
            or tokens.get("physical_input_tokens_global") != GLOBAL_PHYSICAL_TOKENS
            or tokens.get("supervised_tokens_global") != GLOBAL_SUPERVISED_TOKENS
        ):
            raise RuntimeError(f"{run_id} summary does not contain the exact 40+100 token window")
        per_rank = measured.get("per_rank", {})
        if set(per_rank) != {str(rank) for rank in range(8)}:
            raise RuntimeError(f"{run_id} summary lacks per-rank window evidence")
        for rank, values in per_rank.items():
            if (
                values.get("physical_input_tokens_local") != LOCAL_PHYSICAL_TOKENS
                or values.get("supervised_tokens_global") != GLOBAL_SUPERVISED_TOKENS
            ):
                raise RuntimeError(f"{run_id} rank {rank} token counters differ from the protocol")
        memory = summary.get("memory", {}).get("by_rank", {})
        if set(memory) != {str(rank) for rank in range(8)}:
            raise RuntimeError(f"{run_id} summary lacks per-rank memory peaks")
        for rank, values in memory.items():
            peaks = values.get("peak_bytes", {})
            retries = values.get("allocator_counter_deltas", {})
            if not {"allocated_bytes.all.peak", "reserved_bytes.all.peak"}.issubset(peaks):
                raise RuntimeError(f"{run_id} rank {rank} memory peaks are incomplete")
            if not {"num_alloc_retries", "num_ooms"}.issubset(retries):
                raise RuntimeError(f"{run_id} rank {rank} allocator deltas are incomplete")
    return {
        "name": name,
        "run_id": run_id,
        "argv": argv,
        "child_torchrun_argv": actual,
        "completion": completion,
        "metadata_sha256": hashlib.sha256((attempt / "metadata.json").read_bytes()).hexdigest(),
        "summary": summary,
        "rank_measurements": rank_paths,
        "stage_selection_by_rank": stage_paths,
    }


def _passed_timing_receipts(
    previous_run_ids: set[str], freeze_sha256: str
) -> set[str]:
    if not RECEIPTS.exists():
        return set()
    passed = set()
    for line in RECEIPTS.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if (
            record.get("event") == "run_complete"
            and record.get("name") in TIMING_RUNS
            and record.get("run_id") in previous_run_ids
            and record.get("source_freeze_sha256") == freeze_sha256
        ):
            passed.add(record["name"])
    return passed


def _execute(names: list[str]) -> int:
    global _active
    if sys.executable != PYTHON:
        raise RuntimeError(f"execute with {PYTHON}; found {sys.executable}")
    if os.environ.get("ABCD_COMPONENT_FIXTURE_DIR"):
        raise RuntimeError("ABCD_COMPONENT_FIXTURE_DIR must be unset")
    if not SAVED_D_CACHE.is_dir() or SAVED_D_CACHE.is_symlink():
        raise RuntimeError(f"saved D compiler cache is missing or unsafe: {SAVED_D_CACHE}")
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise RuntimeError("--runs must be a nonempty list without duplicate names")
    if tuple(names) not in (TIMING_RUNS, DIAGNOSTIC_RUNS, RUN_NAMES):
        raise RuntimeError(
            "execute the seven timing runs in order, the four diagnostics in order, "
            "or the full ordered list"
        )
    RAW.mkdir(parents=True, exist_ok=True)
    previous = _run_ids_from_receipts()
    for name in names:
        run_id = RUN_IDS[name]
        if run_id in previous or (RAW / run_id).exists() or (RAW / f"{run_id}-bridge.log").exists():
            raise RuntimeError(f"refusing to overwrite or repeat output for {run_id}")

    runner, flash = _load_runner_modules()
    base = flash._load_base_runner()
    base._validate_completed_compiler_cache_root(SAVED_D_CACHE)
    measure = _load_module(MEASURE, "_schedule_embedding_campaign_measure_freeze")
    freeze = _campaign_freeze(base, flash, runner, measure)
    freeze_sha256 = _freeze_sha256(freeze)
    timing_passed: set[str] = set()
    if tuple(names) == DIAGNOSTIC_RUNS:
        passed = _passed_timing_receipts(previous, freeze_sha256)
        if not set(TIMING_RUNS).issubset(passed):
            missing = sorted(set(TIMING_RUNS).difference(passed))
            raise RuntimeError(
                "update-21 profiles require all seven timing receipts from this round "
                f"and source freeze; missing {missing}"
            )
        timing_passed.update(passed)
    campaign_id = f"schedule-embedding-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    argv_by_name = {name: _run_argv(name) for name in names}
    _append_receipt(
        {
            "event": "campaign_begin",
            "campaign_id": campaign_id,
            "runs": names,
            "argv_by_run": argv_by_name,
            "source_freeze": freeze,
            "source_freeze_sha256": freeze_sha256,
            "saved_d_cache": str(SAVED_D_CACHE),
            "saved_d_cache_producer": "proddata-d-timing-001",
        }
    )

    failures: dict[str, str] = {}
    completed: set[str] = set()
    try:
        for name in names:
            if _requested_signal is not None:
                raise InterruptedError(f"campaign received signal {_requested_signal}")
            if name in DIAGNOSTIC_RUNS and not set(TIMING_RUNS).issubset(
                timing_passed
            ):
                missing = sorted(set(TIMING_RUNS).difference(timing_passed))
                reason = f"diagnostics skipped because timing did not pass: {missing}"
                _append_receipt(
                    {
                        "event": "diagnostics_skipped",
                        "campaign_id": campaign_id,
                        "reason": reason,
                    }
                )
                failures["diagnostics"] = reason
                break
            if _campaign_freeze(base, flash, runner, measure) != freeze:
                raise RuntimeError("campaign source, configuration, inputs, or helper bytes changed")
            argv = argv_by_name[name]
            run_id = RUN_IDS[name]
            bridge_log = RAW / f"{run_id}-bridge.log"
            try:
                inventory = _preflight_idle_h100s()
                _append_receipt(
                    {
                        "event": "run_begin",
                        "campaign_id": campaign_id,
                        "name": name,
                        "run_id": run_id,
                        "argv": argv,
                        "gpu_preflight": inventory,
                        "source_freeze_sha256": freeze_sha256,
                    }
                )
                print(f"BEGIN {name} {run_id}", flush=True)
                with bridge_log.open("xb") as log:
                    _active = subprocess.Popen(
                        argv,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        while True:
                            try:
                                code = _active.wait(timeout=1)
                                break
                            except subprocess.TimeoutExpired:
                                if _requested_signal is not None:
                                    raise InterruptedError(
                                        f"campaign received signal {_requested_signal}"
                                    )
                        if not _wait_group_exit(_active.pid, 10):
                            _stop_owned_group(_active)
                            raise RuntimeError(
                                f"owned runner process group for {run_id} did not exit"
                            )
                        if code != 0:
                            raise RuntimeError(f"{run_id} runner exited with status {code}")
                    except BaseException:
                        _stop_owned_group(_active)
                        raise
                    finally:
                        _active = None
                if _campaign_freeze(base, flash, runner, measure) != freeze:
                    raise RuntimeError("campaign source, configuration, inputs, or helpers changed during the run")
                result = _validate_finished(name, argv, freeze, base)
                result["bridge_log"] = str(bridge_log)
                result["source_freeze_sha256"] = freeze_sha256
                _append_receipt({"event": "run_complete", "campaign_id": campaign_id, **result})
                completed.add(name)
                if name in TIMING_RUNS:
                    timing_passed.add(name)
                print(f"PASS {name} {run_id}", flush=True)
            except InterruptedError:
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                failures[name] = message
                _append_receipt(
                    {
                        "event": "run_fail",
                        "campaign_id": campaign_id,
                        "name": name,
                        "run_id": run_id,
                        "bridge_log": str(bridge_log),
                        "error": message,
                        "source_freeze_sha256": freeze_sha256,
                    }
                )
                print(f"FAIL {name} {run_id}: {message}", file=sys.stderr, flush=True)
                if _campaign_freeze(base, flash, runner, measure) != freeze:
                    raise RuntimeError("campaign source, configuration, inputs, or helpers changed after a failed run")

        status = "PASS" if not failures else "FAIL"
        _append_receipt(
            {
                "event": "campaign_complete",
                "campaign_id": campaign_id,
                "status": status,
                "runs": names,
                "completed": sorted(completed),
                "failures": failures,
                "source_freeze_sha256": freeze_sha256,
            }
        )
        return 0 if not failures else 1
    except BaseException as exc:
        if _active is not None:
            _stop_owned_group(_active)
            _active = None
        _append_receipt(
            {"event": "campaign_fail", "campaign_id": campaign_id, "error": f"{type(exc).__name__}: {exc}"}
        )
        print(f"campaign: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def _on_signal(signum: int, _frame: Any) -> None:
    global _requested_signal
    _requested_signal = signum


def main(argv: list[str] | None = None) -> int:
    global RUN_IDS
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true", help="execute only the explicit --runs list")
    parser.add_argument("--runs", nargs="+", choices=RUN_NAMES, help="ordered names such as D0-timing S-timing SE-timing")
    parser.add_argument("--dry-check", action="store_true", help="check parsers and original capacity import without GPU execution")
    parser.add_argument(
        "--round",
        type=int,
        default=3,
        help="new campaign round suffix (must be 3 or greater; IDs use the 140 namespace)",
    )
    args = parser.parse_args(argv)
    if args.round < 3:
        parser.error("--round must be 3 or greater to preserve existing attempts")
    RUN_IDS = {
        name: f"schedule-embedding-{name.lower()}-140-{args.round:03d}"
        for name in RUN_NAMES
    }
    if args.dry_check and args.execute:
        parser.error("--dry-check cannot be combined with --execute")
    if args.execute and not args.runs:
        parser.error("--execute requires --runs")
    try:
        if args.dry_check:
            print(json.dumps(_dry_checks(), sort_keys=True, indent=2))
            return 0
        if not args.execute:
            names = RUN_NAMES if args.runs is None else args.runs
            print(json.dumps(_plan(list(names)), sort_keys=True, indent=2))
            return 0
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        return _execute(args.runs)
    except Exception as exc:
        print(f"campaign: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
