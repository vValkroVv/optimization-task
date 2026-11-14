"""Runner contracts for native epoch size and microbatch capture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure
import run as runner


def _child_config(args: argparse.Namespace, tmp_path: Path):
    from torchtitan.config import run_pre_submit
    from torchtitan.config.build import prepare_document
    from torchtitan.config.cli import load_document
    from torchtitan.config.save import dump_yaml, get_typed_document

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    experiment = config_dir / "experiment.yaml"
    runner._write_experiment_config(experiment)
    paths = {
        "config": config_dir,
        "logs": tmp_path / "logs",
        "measurement": tmp_path / "measurement",
        "native": tmp_path / "native",
    }
    command = runner._command(args, paths)
    helper_start = command.index(str(runner.HELPER_DIR / "measure.py")) + 1
    helper_args, native_args = measure._parse_args(command[helper_start:])
    pairs = [
        (key.removeprefix("--"), value)
        for key, value in zip(native_args[1::2], native_args[2::2])
    ]
    resolved = run_pre_submit(native_args[0], pairs, world_size=8)
    assert resolved.ok, resolved.report.errors
    document, _ = load_document(native_args[0], pairs)
    prepared = prepare_document(document)
    native_config = paths["native"] / "config"
    native_config.mkdir(parents=True)
    (native_config / "resolved_config.yaml").write_text(
        dump_yaml(get_typed_document(prepared.trainer_config, prepared.model_section)),
        encoding="utf-8",
    )
    assert runner._verify_resolved_config(paths, args)
    return helper_args, native_args, resolved.data


def _write_resolved_fixture(
    native_dir: Path, args: argparse.Namespace, epoch_size: int | None
) -> None:
    import yaml

    values: dict[str, object] = {
        "dataloader.num_prefetch_batches": args.num_prefetch_batches,
        "dataloader.dataset.epoch_size": epoch_size,
        "dataloader.type": "streaming_text",
        "training.steps": args.steps,
        "training.local_batch_size": 2,
        "training.global_batch_size": 512,
        "training.seq_len": 4096,
        "lr_scheduler.total_steps": 65000,
        "lr_scheduler.warmup_steps": 2000,
        "debug.seed": 2048,
        "loss.num_chunks": 2,
        "metrics.log_freq": 1,
        "checkpoint.enable": False,
        "validator.enable": False,
        "compile.enable": True,
        "activation_checkpoint.mode": "selective",
        "parallelism.expert_parallel_degree": 8,
        "parallelism.gfsdp_reuse_bf16_expert_weights": (
            args.gfsdp_reuse_bf16_expert_weights
        ),
        "parallelism.pipeline_parallel_degree": 1,
        "parallelism.tensor_parallel_degree": 1,
        "parallelism.context_parallel_degree": 1,
        "fp8.sparse_use_grouped_gemm": True,
        "fp8.deep_gemm_warmup_on_start": True,
        "dump_folder": str(native_dir.resolve()),
        "hf_assets_path": str(runner.TOKENIZER.resolve()),
        "callbacks.ema": {},
        **runner._profile_config_values(args),
    }
    resolved: dict[str, object] = {}
    for dotted, value in values.items():
        current = resolved
        components = dotted.split(".")
        for component in components[:-1]:
            current = current.setdefault(component, {})  # type: ignore[assignment]
        current[components[-1]] = value

    config_dir = native_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved), encoding="utf-8"
    )


def test_epoch_size_cli_validation_and_existing_run_schedules() -> None:
    default = runner._parse_args(["--run-id", "default"])
    assert default.epoch_size is None
    assert default.capture_microbatches is False

    selected = runner._parse_args(
        [
            "--run-id",
            "epoch",
            "--epoch-size",
            "51200",
            "--capture-microbatches",
            "--gfsdp-reuse-bf16-expert-weights",
        ]
    )
    assert selected.epoch_size == 51200
    assert selected.capture_microbatches is True
    assert selected.gfsdp_reuse_bf16_expert_weights is True

    for value in ("0", "-1", "not-an-int"):
        with pytest.raises(SystemExit):
            runner._parse_args(["--run-id", "invalid", "--epoch-size", value])

    tps = runner._parse_args(
        ["--run-id", "tps", "--steps", "60", "--warmup", "20", "--window", "40"]
    )
    assert (tps.steps, tps.warmup, tps.window) == (60, 20, 40)

    diagnostic = runner._parse_args(
        [
            "--run-id",
            "diagnostic",
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
    )
    assert (diagnostic.steps, diagnostic.warmup, diagnostic.window) == (21, 20, 0)
    assert (diagnostic.profile_update, diagnostic.profile_microbatches) == (21, 3)


def test_child_receives_native_epoch_override_and_default_stays_null(
    tmp_path: Path,
) -> None:
    selected = runner._parse_args(
        [
            "--run-id",
            "selected",
            "--steps",
            "60",
            "--warmup",
            "20",
            "--window",
            "40",
            "--epoch-size",
            "51200",
            "--capture-microbatches",
            "--gfsdp-reuse-bf16-expert-weights",
        ]
    )
    helper_args, native_args, resolved = _child_config(selected, tmp_path / "selected")
    assert helper_args.capture_microbatches is True
    assert (helper_args.steps, helper_args.warmup, helper_args.window) == (60, 20, 40)
    assert helper_args.gfsdp_reuse_bf16_expert_weights is True
    assert "--dataloader.dataset.epoch_size" in native_args
    assert resolved["dataloader"]["dataset"]["epoch_size"] == 51200
    assert resolved["parallelism"]["gfsdp_reuse_bf16_expert_weights"] is True

    default = runner._parse_args(["--run-id", "default"])
    default_overrides = runner._training_overrides(default, {"native": tmp_path / "default"})
    assert "--dataloader.dataset.epoch_size" not in default_overrides
    default_helper, default_native_args, default_resolved = _child_config(
        default, tmp_path / "default"
    )
    assert default_helper.capture_microbatches is False
    assert "--capture-microbatches" not in default_native_args
    assert default_resolved["dataloader"]["dataset"].get("epoch_size") is None


def test_resolved_config_checks_requested_epoch_size(tmp_path: Path) -> None:
    args = runner._parse_args(["--run-id", "resolved", "--epoch-size", "51200"])
    native_dir = tmp_path / "native"
    _write_resolved_fixture(native_dir, args, epoch_size=51200)
    paths = {"native": native_dir}
    assert runner._verify_resolved_config(paths, args)

    _write_resolved_fixture(native_dir, args, epoch_size=25600)
    with pytest.raises(RuntimeError, match="dataloader.dataset.epoch_size"):
        runner._verify_resolved_config(paths, args)


@pytest.mark.parametrize(
    ("options", "expected_epoch_size", "expected_capture", "expected_feature", "expected_counters"),
    [
        ([], None, False, False, False),
        (
            ["--epoch-size", "51200", "--capture-microbatches", "--gfsdp-reuse-bf16-expert-weights"],
            51200,
            True,
            True,
            False,
        ),
        (
            [
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
                "--gfsdp-reuse-bf16-expert-weights",
                "--bf16-cache-counters",
            ],
            None,
            False,
            True,
            True,
        ),
    ],
)
def test_run_metadata_records_requested_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    expected_epoch_size: int | None,
    expected_capture: bool,
    expected_feature: bool,
    expected_counters: bool,
) -> None:
    root = tmp_path / "attempt"
    config_dir = root / "config"
    config_dir.mkdir(parents=True)
    paths = {
        "attempt": root,
        "logs": root / "logs",
        "preflight": root / "preflight",
        "config": config_dir,
        "native": root / "native",
        "measurement": root / "measurement",
        "cache": root / "cache",
        "shared_hf_cache": root / "shared-hf-cache",
        "cache_torchinductor": root / "cache/torchinductor",
        "cache_triton": root / "cache/triton",
        "cache_deep_gemm": root / "cache/deep-gemm",
        "cache_hybridep": root / "cache/hybridep",
        "cache_deep_gemm_ranks": root / "cache/deep-gemm/ranks",
    }
    source = {"head": "head", "tracked_and_untracked_sha256": "same"}
    metadata: dict[str, object] = {}

    monkeypatch.setattr(runner, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(runner, "verify_task_root", lambda _root: None)
    monkeypatch.setattr(runner, "_make_run_dirs", lambda _run_id, _output=None: paths)
    monkeypatch.setattr(runner, "_data_root", lambda _root: tmp_path)
    monkeypatch.setattr(runner, "_filesystem_status", lambda _paths, _mds: {})
    monkeypatch.setattr(
        runner, "_make_task_tmpdir", lambda: (tmp_path / "task-tmp", {})
    )
    monkeypatch.setattr(runner, "_run_environment", lambda _paths: {})
    monkeypatch.setattr(runner, "source_snapshot", lambda _root: dict(source))
    monkeypatch.setattr(runner, "_config_hashes", lambda: {"config": "same"})
    monkeypatch.setattr(runner, "_runtime_preflight", lambda _env, _paths: {})
    monkeypatch.setattr(runner, "_launch", lambda *_args: (1, False, 123, False, True))
    monkeypatch.setattr(runner, "_write_new_text", lambda *_args: None)
    monkeypatch.setattr(runner, "_write_atomic_json", lambda *_args: None)
    monkeypatch.setattr(runner, "_write_replace_json", lambda *_args: None)
    monkeypatch.setattr(runner, "_write_replace_text", lambda *_args: None)
    monkeypatch.setattr(runner, "_cleanup_task_tmpdir", lambda _path: None)

    def capture_json(path: Path, value: object) -> None:
        if path.name == "metadata.json":
            assert isinstance(value, dict)
            metadata.update(value)

    monkeypatch.setattr(runner, "_write_new_json", capture_json)

    result = runner.main(["--run-id", "metadata", *options])
    assert result == 1  # The mocked child exits nonzero after metadata is written.
    assert metadata["requested_epoch_size"] == expected_epoch_size
    assert metadata["capture_microbatches"] is expected_capture
    assert metadata["gfsdp_reuse_bf16_expert_weights"] is expected_feature
    assert metadata["bf16_cache_counters"] is expected_counters
