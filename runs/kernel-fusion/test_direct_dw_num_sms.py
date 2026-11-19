import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNNER_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("kernel_fusion_runner_test", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def controls_for(*enabled):
    return {
        field: name in enabled
        for name, field in runner.GFSDP_CONTROLS.items()
    }


def fixed_recipe_base(tmp_path):
    resolved_path = tmp_path / "native/config/resolved_config.yaml"
    expected = {
        "dump_folder": "control-output",
        "training": {"steps": 20},
        "parallelism": {
            "gfsdp_retain_tail_weights": False,
            "gfsdp_fp8_expert_weights": False,
            "gfsdp_direct_expert_dw": False,
        },
        "fp8": {"sparse_deep_gemm_num_sms": 116},
    }
    metadata = {
        "source_snapshot_before": {
            "tracked_and_untracked_sha256": runner.PRODUCTION_SOURCE_SHA256
        },
        "config_hashes_before": {
            "canonical_experiment_sha256": "canonical",
            "generated_experiment_sha256": "generated",
        },
        "argv": ["python", "run.py", "--", "experiment.yaml", "--dump_folder", "control"],
    }
    captured = {}

    def load_json(path):
        if path.name == "completion.json":
            return {"status": "PASS", "resolved_config_sha256": "resolved"}
        return metadata

    def sha256_file(path):
        return "resolved" if path == resolved_path else "generated"

    def launch(command, env, *_args):
        captured["command"] = command
        captured["rank_environment"] = env
        return 0

    base = SimpleNamespace(
        _load_json=load_json,
        sha256_file=sha256_file,
        _config_hashes=lambda: {
            "canonical_experiment_sha256": "canonical",
            "generated_experiment_sha256": "generated",
        },
        _load_yaml=lambda _path: copy.deepcopy(expected),
        _launch=launch,
        _verify_resolved_config=lambda _paths, _args: "resolved",
    )
    return base, expected, captured


def launch_command(controls):
    command = ["python", "run.py", "--", "experiment.yaml", "--dump_folder", "candidate"]
    for dotted, enabled in controls.items():
        command.extend((f"--{dotted}", str(enabled).lower()))
    return command


@pytest.mark.parametrize(
    "enabled",
    [
        ("direct-expert-dw",),
        ("direct-expert-dw", "fp8-expert-weights", "retain-tail-weights"),
    ],
)
def test_fixed_recipe_budget_reaches_child_environment(tmp_path, monkeypatch, enabled):
    base, _, captured = fixed_recipe_base(tmp_path)
    controls = controls_for(*enabled)
    monkeypatch.setenv(runner.DIRECT_DW_NUM_SMS_ENV, "999")
    runner.enforce_fixed_recipe(base, tmp_path, controls, "timing")

    assert os.environ[runner.DIRECT_DW_NUM_SMS_ENV] == "116"
    base._launch(launch_command(controls), os.environ.copy(), {}, 0)
    assert captured["rank_environment"][runner.DIRECT_DW_NUM_SMS_ENV] == "116"


@pytest.mark.parametrize(
    "enabled",
    [(), ("retain-tail-weights",), ("fp8-expert-weights",)],
)
def test_other_gfsdp_controls_do_not_set_direct_dw_budget(tmp_path, monkeypatch, enabled):
    base, _, _ = fixed_recipe_base(tmp_path)
    monkeypatch.setenv(runner.DIRECT_DW_NUM_SMS_ENV, "999")

    runner.enforce_fixed_recipe(base, tmp_path, controls_for(*enabled), "timing")

    assert runner.DIRECT_DW_NUM_SMS_ENV not in os.environ


def prepare_rank_environment(monkeypatch, *, build=None, num_sms=None):
    import os

    monkeypatch.setattr(runner, "load", lambda *_args: SimpleNamespace(run_rank=lambda _args: None))
    monkeypatch.setattr(sys, "argv", ["run.py", "rank", runner.PRODUCTION])
    for switch in runner.SWITCHES.values():
        if switch in runner.variant_switches(runner.PRODUCTION):
            monkeypatch.setenv(switch, "1")
        else:
            monkeypatch.delenv(switch, raising=False)
    if build is None:
        os.environ.pop(runner.DIRECT_DW_BUILD_ENV, None)
    else:
        monkeypatch.setenv(runner.DIRECT_DW_BUILD_ENV, build)
    if num_sms is None:
        os.environ.pop(runner.DIRECT_DW_NUM_SMS_ENV, None)
    else:
        monkeypatch.setenv(runner.DIRECT_DW_NUM_SMS_ENV, num_sms)


def test_rank_configures_direct_dw_with_recipe_budget(monkeypatch, capsys):
    seen = []
    prepare_rank_environment(monkeypatch, build="/task/build", num_sms="116")
    monkeypatch.setattr(
        runner,
        "configure_direct_dw_rank",
        lambda build, num_sms: seen.append((build, num_sms)),
    )

    assert runner.main() == 0
    assert seen == [("/task/build", 116)]
    assert "direct_dw_num_sms=116" in capsys.readouterr().out


def test_rank_without_direct_dw_build_skips_extension_setup(monkeypatch):
    prepare_rank_environment(monkeypatch)
    monkeypatch.setattr(
        runner,
        "configure_direct_dw_rank",
        lambda *_args: pytest.fail("configured without a direct dW build"),
    )

    assert runner.main() == 0


@pytest.mark.parametrize("num_sms", [None, "0", "invalid"])
def test_direct_dw_rank_rejects_missing_or_invalid_budget(monkeypatch, num_sms):
    prepare_rank_environment(monkeypatch, build="/task/build", num_sms=num_sms)
    monkeypatch.setattr(runner, "configure_direct_dw_rank", lambda *_args: pytest.fail("configured"))

    with pytest.raises(RuntimeError, match=runner.DIRECT_DW_NUM_SMS_ENV):
        runner.main()


def test_parent_entry_clears_stale_direct_dw_environment(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "--help"])
    monkeypatch.setenv(runner.DIRECT_DW_BUILD_ENV, "/stale/build")
    monkeypatch.setenv(runner.DIRECT_DW_NUM_SMS_ENV, "999")

    with pytest.raises(SystemExit) as result:
        runner.main()

    assert result.value.code == 0
    assert runner.DIRECT_DW_BUILD_ENV not in os.environ
    assert runner.DIRECT_DW_NUM_SMS_ENV not in os.environ
