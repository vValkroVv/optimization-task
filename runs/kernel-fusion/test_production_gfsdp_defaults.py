"""Check production CLI defaults through the native command and config bridge."""

from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


RUNNER = Path(__file__).resolve().with_name("run.py")
HEAD = "a" * 40
FIELDS = (
    "parallelism.gfsdp_retain_tail_weights",
    "parallelism.gfsdp_fp8_expert_weights",
    "parallelism.gfsdp_direct_expert_dw",
)


@pytest.fixture
def cli(monkeypatch, tmp_path):
    spec = spec_from_file_location("_production_gfsdp_test_runner", RUNNER)
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)
    root = runner.ROOT
    load = runner.load
    native_path = Path("runs/capacity-selector/native.py")
    native = load(root / native_path, "_production_gfsdp_test_native")
    native_load = native._load
    measured = json.loads(
        (root / "runs/efsdp-optimizations/direct-dw-receipts.json").read_text()
    )["timing"]["candidate"]["gift_snapshot_sha256"]
    snapshot = {
        "head": HEAD,
        "untracked_file_count": 0,
        "tracked_and_untracked_sha256": measured,
    }
    git = Mock()
    git_output = {"rev-parse": f"{HEAD}\n", "status": ""}
    git.check_output.side_effect = lambda command, **_: git_output[command[3]]
    fixed_recipe = Mock()
    emitted = {}

    def load_schedule(path, name):
        schedule = native_load(path, name)
        flash = schedule._load_flash_runner()
        base = flash._load_base_runner()

        def inspect_native(argv):
            args = base._parse_args(argv)
            paths = {
                name: tmp_path / name
                for name in ("measurement", "logs", "config", "native")
            }
            emitted["command"] = base._command(args, paths)
            emitted["config"] = base._resolved_config_expectations(args)
            emitted["hashes"] = base._config_hashes()
            emitted["snapshot"] = base.source_snapshot(tmp_path)
            return 0

        monkeypatch.setattr(base, "main", inspect_native)
        monkeypatch.setattr(base, "source_snapshot", lambda _root: snapshot)
        monkeypatch.setattr(base, "_config_hashes", lambda: {"input": "historical"})
        monkeypatch.setattr(flash, "_load_base_runner", lambda: base)
        monkeypatch.setattr(schedule, "_load_flash_runner", lambda: flash)
        return schedule

    def load_runner(path, name):
        relative = path.relative_to(tmp_path)
        return native if relative == native_path else load(root / relative, name)

    monkeypatch.setattr(native, "_load", load_schedule)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "load", load_runner)
    monkeypatch.setattr(runner, "subprocess", git)
    monkeypatch.setattr(runner, "enforce_fixed_recipe", fixed_recipe)
    for name in (
        *runner.SWITCHES.values(),
        runner.DIRECT_DW_BUILD_ENV,
        runner.DIRECT_DW_NUM_SMS_ENV,
        "TORCH_COMPILE_DISABLE",
        "TORCHDYNAMO_DISABLE",
    ):
        monkeypatch.setenv(name, "stale")

    build = tmp_path / "build"
    build.mkdir()
    manifest = build / "manifest.json"
    manifest.write_text("{}\n")

    def invoke(variant, mode, *arguments):
        monkeypatch.setattr(
            sys, "argv", [str(RUNNER), variant, mode, "test-run", *arguments]
        )
        return runner.main()

    return SimpleNamespace(
        invoke=invoke,
        runner=runner,
        git=git,
        git_output=git_output,
        snapshot=snapshot,
        fixed_recipe=fixed_recipe,
        emitted=emitted,
        manifest=manifest,
        native=native,
    )


@pytest.mark.parametrize(
    ("mode", "arguments", "expected"),
    [
        ("timing-140", (), (True, True, True)),
        ("timing", ("--source-revision", HEAD), (False, False, False)),
        (
            "trace",
            ("--source-revision", HEAD, "--gfsdp-optimization", "fp8-expert-weights"),
            (False, True, False),
        ),
        (
            "timing",
            ("--source-revision", HEAD, "--gfsdp-optimization", "direct-expert-dw"),
            (False, False, True),
        ),
    ],
    ids=["canonical", "explicit-control", "explicit-fp8", "explicit-direct"],
)
def test_production_routes_exact_gfsdp_flags(cli, mode, arguments, expected):
    if arguments:
        cli.snapshot["tracked_and_untracked_sha256"] = "b" * 64
    if expected[-1]:
        arguments = (*arguments, "--direct-dw-build", str(cli.manifest.parent))

    assert cli.invoke("production", mode, *arguments) == 0

    runner = cli.runner
    cli.git.run.assert_called_once_with(
        [
            "git",
            "-C",
            str(runner.ROOT),
            "merge-base",
            "--is-ancestor",
            runner.PRODUCTION_REVISION,
            HEAD,
        ],
        check=True,
    )
    command = cli.emitted["command"]
    rank = command.index(str(RUNNER))
    assert command[rank + 1 : rank + 3] == ["rank", "production"]
    _, overrides = runner.training_overrides(command)
    assert overrides["--compile.enable"] == "true"
    assert [overrides[f"--{field}"] for field in FIELDS] == [
        str(value).lower() for value in expected
    ]
    assert [cli.emitted["config"][field] for field in FIELDS] == list(expected)
    assert cli.emitted["snapshot"] == cli.snapshot
    assert cli.native.S_SOURCE_SHA256 == cli.snapshot["tracked_and_untracked_sha256"]
    control = (
        runner.FIXED_TRACE_CONTROL if mode == "trace" else runner.FIXED_TIMING_CONTROL
    )
    _, control_path, flags, fixed_mode = cli.fixed_recipe.call_args.args
    assert control_path == runner.HERE / "artifacts/raw" / control
    assert flags == dict(zip(FIELDS, expected, strict=True))
    assert fixed_mode == mode
    hashes = cli.emitted["hashes"]
    assert hashes["input"] == "historical"
    assert (
        hashes["kernel_fusion_runner_sha256"]
        == hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    )
    if expected[-1]:
        assert (
            hashes["direct_dw_build_manifest_sha256"]
            == hashlib.sha256(cli.manifest.read_bytes()).hexdigest()
        )
        assert os.environ[runner.DIRECT_DW_BUILD_ENV] == str(cli.manifest.parent)
    else:
        assert "direct_dw_build_manifest_sha256" not in hashes
        assert runner.DIRECT_DW_BUILD_ENV not in os.environ


@pytest.mark.parametrize(
    ("variant", "arguments", "error"),
    [
        ("production", (), "--direct-dw-build is required"),
        ("production", ("--direct-dw-build", "missing"), "needs a verified build"),
        (
            "production",
            ("--gfsdp-optimization", "retain-tail-weights"),
            "GFSDP optimizations require a committed --source-revision",
        ),
        (
            "baseline",
            ("--source-revision", HEAD, "--gfsdp-optimization", "fp8-expert-weights"),
            "Follow-up revisions require the production recipe",
        ),
    ],
)
def test_invalid_production_arguments_fail_before_preflight(
    cli, capsys, variant, arguments, error
):
    with pytest.raises(SystemExit) as result:
        cli.invoke(variant, "timing", *arguments)

    assert result.value.code == 2
    assert error in capsys.readouterr().err
    cli.git.check_output.assert_not_called()
    assert not cli.emitted


def test_canonical_production_rejects_unmeasured_gift_snapshot(cli):
    cli.snapshot["tracked_and_untracked_sha256"] = "b" * 64

    with pytest.raises(
        RuntimeError, match="GIFT source differs from the measured GFSDP source"
    ):
        cli.invoke(
            "production", "timing", "--direct-dw-build", str(cli.manifest.parent)
        )

    cli.fixed_recipe.assert_not_called()
    assert not cli.emitted


def test_canonical_production_rejects_uncommitted_runner_or_gift_changes(cli):
    cli.git_output["status"] = " M runs/kernel-fusion/run.py\n"

    with pytest.raises(ValueError, match="Follow-up source has uncommitted changes"):
        cli.invoke(
            "production", "timing", "--direct-dw-build", str(cli.manifest.parent)
        )

    cli.fixed_recipe.assert_not_called()
    assert not cli.emitted
