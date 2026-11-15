"""Native-runner coverage for independent FFN and row-owner controls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

RUNNER_DIR = Path(__file__).resolve().parents[1]
TASK_ROOT = RUNNER_DIR.parents[1]
sys.path.insert(0, str(RUNNER_DIR))

import measure
import run as runner


FEATURES = (
    ("gfsdp_split_mtp_ffn", "--gfsdp-split-mtp-ffn"),
    ("gfsdp_split_layer0_ffn", "--gfsdp-split-layer0-ffn"),
    ("gfsdp_row_owner_embeddings", "--gfsdp-row-owner-embeddings"),
)
BRIDGE_RUN_PATH = TASK_ROOT / "runs/flash-qla/run.py"
RAW_ROOT = TASK_ROOT / "runs/ffn-row-owner/artifacts/raw"
PLAN_PATH = TASK_ROOT / "runs/ffn-row-owner/native-plan.json"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def flash_bridge() -> Any:
    return _load_module(BRIDGE_RUN_PATH, "_test_ffn_row_owner_flash_bridge")


def _outer_args(
    run_id: str,
    feature_flags: tuple[str, ...] = (),
    *,
    diagnostic: bool = False,
) -> list[str]:
    if diagnostic:
        duration = [
            "--steps",
            "21",
            "--warmup",
            "20",
            "--window",
            "0",
            "--profile-update",
            "21",
            "--profile-microbatches",
            "3",
        ]
    else:
        duration = ["--steps", "20", "--warmup", "10", "--window", "10"]
    return [
        "--run-id",
        run_id,
        *duration,
        "--num-prefetch-batches",
        "32",
        "--epoch-size",
        "102400",
        "--gfsdp-reuse-bf16-expert-weights",
        *feature_flags,
        "--output-root",
        str(RAW_ROOT),
    ]


def _build_child(
    flash_bridge: Any,
    outer_args: list[str],
) -> tuple[list[str], Any, list[str], Any, list[str]]:
    bridge_args, forwarded = flash_bridge._parse_bridge_args(
        ["--gdn-backend", "external_gate", *outer_args]
    )
    args = runner._parse_args(forwarded)
    attempt = RAW_ROOT / args.run_id
    paths = {
        "attempt": attempt,
        "logs": attempt / "logs",
        "measurement": attempt / "measurement",
        "native": attempt / "native",
        "config": attempt / "config",
    }
    command = runner._command(args, paths)
    helper_start = command.index(str((runner.HELPER_DIR / "measure.py").resolve())) + 1
    helper_args, native_args = measure._parse_args(command[helper_start:])
    return forwarded, bridge_args, command, helper_args, native_args


def _native_overrides(native_args: list[str]) -> dict[str, str]:
    assert len(native_args[1:]) % 2 == 0
    return {
        option.removeprefix("--"): value
        for option, value in zip(native_args[1::2], native_args[2::2], strict=True)
    }


def _resolved_config(expectations: dict[str, Any], native: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for dotted, value in expectations.items():
        current = config
        components = dotted.split(".")
        for component in components[:-1]:
            current = current.setdefault(component, {})
        current[components[-1]] = value
    config.setdefault("callbacks", {})["ema"] = {"type": "ema"}
    config.setdefault("dataloader", {})["type"] = "streaming_text"
    config["dump_folder"] = str(native.resolve())
    config["hf_assets_path"] = str(runner.TOKENIZER.resolve())
    return config


def test_defaults_keep_new_controls_off_and_native_command_unchanged(
    tmp_path: Path,
    flash_bridge: Any,
) -> None:
    args = runner._parse_args(["--run-id", "default"])
    assert runner._native_feature_values(args) == {field: False for field, _ in FEATURES}
    assert args.gfsdp_buffer_inventory is False
    overrides = runner._training_overrides(args, {"native": tmp_path / "native"})
    assert not any("gfsdp_split_" in item or "gfsdp_row_owner" in item for item in overrides)

    _, bridge_args, command, helper_args, native_args = _build_child(
        flash_bridge,
        ["--run-id", "default", "--output-root", str(RAW_ROOT)],
    )
    assert bridge_args.gdn_backend == "external_gate"
    assert helper_args.gfsdp_reuse_bf16_expert_weights is False
    assert helper_args.profile_update is None
    assert "--profile-update" not in command
    assert not any("gfsdp_split_" in item or "gfsdp_row_owner" in item for item in native_args)


def test_gfsdp_buffer_inventory_is_an_explicit_one_update_measurement_diagnostic(
    flash_bridge: Any,
) -> None:
    outer_args = [
        "--run-id",
        "memory-control",
        "--steps",
        "1",
        "--warmup",
        "0",
        "--window",
        "1",
        "--num-prefetch-batches",
        "32",
        "--epoch-size",
        "102400",
        "--gfsdp-reuse-bf16-expert-weights",
        "--gfsdp-buffer-inventory",
        "--output-root",
        str(RAW_ROOT),
    ]
    _, bridge_args, command, helper_args, native_args = _build_child(
        flash_bridge, outer_args
    )

    assert bridge_args.gdn_backend == "external_gate"
    assert helper_args.gfsdp_buffer_inventory is True
    assert helper_args.steps == 1 and helper_args.warmup == 0 and helper_args.window == 1
    assert "--gfsdp-buffer-inventory" in command
    assert "--profiler.enable_profiling" in command
    overrides = _native_overrides(native_args)
    assert overrides["profiler.enable_profiling"] == "false"
    assert overrides["profiler.enable_memory_snapshot"] == "false"

    with pytest.raises(SystemExit):
        runner._parse_args(
            ["--run-id", "invalid-window", "--steps", "1", "--warmup", "0", "--window", "1"]
        )
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--run-id",
                "invalid-inventory",
                "--steps",
                "2",
                "--warmup",
                "0",
                "--window",
                "2",
                "--gfsdp-buffer-inventory",
            ]
        )


@pytest.mark.parametrize(("field", "flag"), FEATURES)
def test_each_control_reaches_its_typed_native_override_independently(
    field: str,
    flag: str,
    flash_bridge: Any,
) -> None:
    run_id = f"single-{field}"
    outer_args = _outer_args(run_id, (flag,))
    forwarded, bridge_args, command, helper_args, native_args = _build_child(
        flash_bridge, outer_args
    )
    parsed = runner._parse_args(forwarded)
    expected_features = {name: name == field for name, _ in FEATURES}
    assert bridge_args.gdn_backend == "external_gate"
    assert forwarded == outer_args
    assert runner._native_feature_values(parsed) == expected_features
    assert helper_args.gfsdp_reuse_bf16_expert_weights is True
    assert helper_args.profile_update is None
    assert "--profiler.enable_profiling" in command
    overrides = _native_overrides(native_args)
    for name, enabled in expected_features.items():
        option = f"parallelism.{name}"
        assert (overrides.get(option) == "true") is enabled
        assert runner._resolved_config_expectations(parsed)[option] is enabled
    assert overrides["parallelism.gfsdp_reuse_bf16_expert_weights"] == "true"
    assert overrides["dataloader.num_prefetch_batches"] == "32"
    assert overrides["dataloader.dataset.epoch_size"] == "102400"
    assert overrides["profiler.enable_profiling"] == "false"
    assert overrides["profiler.enable_memory_snapshot"] == "false"


def test_combined_ab_common_config_and_three_microbatch_diagnostic(
    tmp_path: Path,
    flash_bridge: Any,
) -> None:
    all_flags = tuple(flag for _, flag in FEATURES)
    a_forwarded, a_bridge, a_command, a_helper, a_native = _build_child(
        flash_bridge, _outer_args("ffn-row-owner-a-001")
    )
    b_forwarded, b_bridge, b_command, b_helper, b_native = _build_child(
        flash_bridge, _outer_args("ffn-row-owner-b-001", all_flags)
    )
    assert a_bridge.gdn_backend == b_bridge.gdn_backend == "external_gate"
    assert a_forwarded == _outer_args("ffn-row-owner-a-001")
    assert b_forwarded == _outer_args("ffn-row-owner-b-001", all_flags)
    assert (a_helper.steps, a_helper.warmup, a_helper.window) == (20, 10, 10)
    assert (b_helper.steps, b_helper.warmup, b_helper.window) == (20, 10, 10)
    assert a_helper.profile_update is b_helper.profile_update is None
    assert "--profile-update" not in a_command + b_command
    assert "--capture-microbatches" not in a_command + b_command
    assert "--bf16-cache-counters" not in a_command + b_command
    assert "--observe-compiler-cache" not in a_command + b_command

    a_overrides = _native_overrides(a_native)
    b_overrides = _native_overrides(b_native)
    feature_options = {f"parallelism.{field}" for field, _ in FEATURES}
    a_common = {key: value for key, value in a_overrides.items() if key not in feature_options | {"dump_folder"}}
    b_common = {key: value for key, value in b_overrides.items() if key not in feature_options | {"dump_folder"}}
    assert a_common == b_common
    assert not feature_options.intersection(a_overrides)
    assert {key: b_overrides[key] for key in feature_options} == {
        key: "true" for key in feature_options
    }
    assert a_common["parallelism.gfsdp_reuse_bf16_expert_weights"] == "true"
    assert a_common["dataloader.num_prefetch_batches"] == "32"
    assert a_common["dataloader.dataset.epoch_size"] == "102400"
    assert a_common["training.steps"] == "20"
    assert a_common["lr_scheduler.total_steps"] == "65000"
    assert a_common["lr_scheduler.warmup_steps"] == "2000"
    assert a_common["debug.seed"] == "2048"
    assert a_common["profiler.enable_profiling"] == "false"
    assert a_common["profiler.enable_memory_snapshot"] == "false"
    assert runner._native_feature_values(runner._parse_args(b_forwarded)) == {
        field: True for field, _ in FEATURES
    }
    for forwarded in (a_forwarded, b_forwarded):
        common_expectations = runner._resolved_config_expectations(
            runner._parse_args(forwarded)
        )
        assert common_expectations["training.local_batch_size"] == 2
        assert common_expectations["training.global_batch_size"] == 512
        assert common_expectations["training.seq_len"] == 4096
        assert common_expectations["lr_scheduler.total_steps"] == 65000
        assert common_expectations["lr_scheduler.warmup_steps"] == 2000
        assert 512 // (2 * 8) == 32

    smoke_args = runner._parse_args(
        [
            "--run-id",
            "ffn-row-owner-smoke-001",
            "--steps",
            "1",
            "--warmup",
            "0",
            "--window",
            "0",
            "--num-prefetch-batches",
            "32",
            "--epoch-size",
            "102400",
            "--gfsdp-reuse-bf16-expert-weights",
            *all_flags,
            "--output-root",
            str(RAW_ROOT),
        ]
    )
    assert (smoke_args.steps, smoke_args.warmup, smoke_args.window) == (1, 0, 0)
    assert runner._native_feature_values(smoke_args) == {field: True for field, _ in FEATURES}
    assert runner._training_overrides(smoke_args, {"native": tmp_path / "smoke"})

    diagnostic_outer = _outer_args("ffn-row-owner-diagnostic-001", all_flags, diagnostic=True)
    _, diagnostic_bridge, diagnostic_command, diagnostic_helper, diagnostic_native = _build_child(
        flash_bridge, diagnostic_outer
    )
    diagnostic_overrides = _native_overrides(diagnostic_native)
    assert diagnostic_bridge.gdn_backend == "external_gate"
    assert (
        diagnostic_helper.steps,
        diagnostic_helper.warmup,
        diagnostic_helper.window,
        diagnostic_helper.profile_update,
        diagnostic_helper.profile_microbatches,
    ) == (21, 20, 0, 21, 3)
    assert "--capture-microbatches" not in diagnostic_command
    assert "--bf16-cache-counters" not in diagnostic_command
    assert diagnostic_overrides["profiler.enable_profiling"] == "true"
    assert diagnostic_overrides["profiler.enable_memory_snapshot"] == "false"
    assert diagnostic_overrides["profiler.profile_ranks"] == "[0, 1]"
    assert diagnostic_overrides["profiler.profiler_skip_first"] == "19"
    assert all(diagnostic_overrides[f"parallelism.{field}"] == "true" for field, _ in FEATURES)


def test_resolved_config_checks_all_three_defaults_and_selected_values(
    tmp_path: Path,
) -> None:
    args = runner._parse_args(
        [
            "--run-id",
            "resolved",
            "--gfsdp-split-mtp-ffn",
            "--gfsdp-row-owner-embeddings",
        ]
    )
    native = tmp_path / "native"
    config_dir = native / "config"
    config_dir.mkdir(parents=True)
    import yaml

    resolved = _resolved_config(
        runner._resolved_config_expectations(args), native
    )
    (config_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved), encoding="utf-8"
    )
    runner._verify_resolved_config({"native": native}, args)

    resolved["parallelism"]["gfsdp_row_owner_embeddings"] = False
    (config_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="parallelism.gfsdp_row_owner_embeddings"):
        runner._verify_resolved_config({"native": native}, args)


def test_campaign_plan_commands_match_runner_and_measurement_hooks(
    flash_bridge: Any,
) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert plan["task_root"] == str(TASK_ROOT.resolve())
    assert plan["raw_output_root"]["path"] == str(RAW_ROOT.resolve())
    assert plan["raw_output_root"]["git_ignored"] is True
    assert plan["status"] == "prepared; no GPU command launched"

    for run_plan in plan["runs"]:
        outer_argv = run_plan["argv"]
        assert outer_argv[0] == str(runner.PYTHON)
        assert outer_argv[1] == str(BRIDGE_RUN_PATH)
        bridge_args, forwarded = flash_bridge._parse_bridge_args(outer_argv[2:])
        assert bridge_args.gdn_backend == "external_gate"
        args = runner._parse_args(forwarded)
        assert args.run_id == run_plan["id"]

        attempt = RAW_ROOT / args.run_id
        paths = {
            "attempt": attempt,
            "logs": attempt / "logs",
            "measurement": attempt / "measurement",
            "native": attempt / "native",
            "config": attempt / "config",
        }
        assert run_plan["native_config_overrides"] == runner._training_overrides(args, paths)
        child_argv = runner._command(args, paths)
        assert run_plan["child_torchrun_argv"] == child_argv
        helper_start = child_argv.index(
            str((runner.HELPER_DIR / "measure.py").resolve())
        ) + 1
        helper_args, native_args = measure._parse_args(child_argv[helper_start:])
        assert run_plan["helper_parse"] == {
            "run_id": helper_args.run_id,
            "steps": helper_args.steps,
            "warmup": helper_args.warmup,
            "window": helper_args.window,
            "profile_update": helper_args.profile_update,
            "profile_microbatches": helper_args.profile_microbatches,
            "gfsdp_reuse_bf16_expert_weights": helper_args.gfsdp_reuse_bf16_expert_weights,
        }
        assert native_args[1:] == run_plan["native_config_overrides"]
        assert run_plan["requested_parallelism_features"] == runner._native_feature_values(args)
