"""Every launched experiment exists and has a readable golden snapshot."""

import importlib.util
import shlex
import subprocess
from pathlib import Path

import pytest

from tests.integration_tests import run_tests as launcher
from tests.integration_tests.features import build_features_test_list
from tests.integration_tests.flux import build_flux_test_list, FLUX_CONFIG_FILE
from tests.integration_tests.h100 import build_h100_tests_list
from tests.integration_tests.models import build_model_tests_list
from tests.integration_tests.run_tests import _split_config_file, DEFAULT_CONFIG_FILE
from tests.unit_tests.config_golden import load_golden
from torchtitan.config import run_pre_submit
from torchtitan.config.overrides import parse_argv
from torchtitan.experiments.graph_trainer.tests.integration_tests import (
    build_graph_trainer_autoparallel_h100_test_list,
    build_graph_trainer_autoparallel_test_list,
    build_graph_trainer_h100_test_list,
    build_graph_trainer_test_list,
)
from torchtitan.experiments.graph_trainer.tests.run_precompile_tests import (
    _build_precompile_tests,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The launch catalogue is runtime-free; importing the FT package would also
# import torchft and the training runtime, which enumeration does not require.
_ft_spec = importlib.util.spec_from_file_location(
    "ft_integration_definitions",
    REPO_ROOT / "torchtitan/experiments/ft/tests/integration_tests.py",
)
assert _ft_spec is not None and _ft_spec.loader is not None
_ft_definitions = importlib.util.module_from_spec(_ft_spec)
_ft_spec.loader.exec_module(_ft_definitions)
build_ft_test_list = _ft_definitions.build_ft_test_list
FT_CONFIG_FILE = _ft_definitions.FT_CONFIG_FILE

LEGACY_TOKENS = ("--module", "--config ", "--config=")


def _launch_catalogue():
    suites = (
        (DEFAULT_CONFIG_FILE, build_features_test_list()),
        (DEFAULT_CONFIG_FILE, build_model_tests_list()),
        (DEFAULT_CONFIG_FILE, build_h100_tests_list()),
        (FLUX_CONFIG_FILE, build_flux_test_list()),
        (FT_CONFIG_FILE, build_ft_test_list()),
        (DEFAULT_CONFIG_FILE, build_graph_trainer_test_list()),
        (DEFAULT_CONFIG_FILE, build_graph_trainer_h100_test_list()),
        (DEFAULT_CONFIG_FILE, build_graph_trainer_autoparallel_test_list()),
        (DEFAULT_CONFIG_FILE, build_graph_trainer_autoparallel_h100_test_list()),
    )
    for config_file, suite in suites:
        for test in suite:
            yield config_file, test


def _launched_config_files() -> set[str]:
    files = {DEFAULT_CONFIG_FILE, FLUX_CONFIG_FILE, FT_CONFIG_FILE}
    for default, test in _launch_catalogue():
        for variant in test.override_args:
            config_file, rest = _split_config_file(variant)
            for arg in rest:
                assert not arg.startswith(LEGACY_TOKENS), (
                    f"{test.test_name}: legacy `--module/--config` launch "
                    f"string {arg!r}; name the experiment YAML with "
                    "`--config-file` instead"
                )
            files.add(config_file or default)
    for test in _build_precompile_tests():
        files.add(test.config_file)
        assert test.config_file in test.precompile_command, test.test_name
        # Single process: the command declares the world the artifact is
        # compiled for, and the preflight checks the degrees against it.
        assert "WORLD_SIZE=" in test.precompile_command, test.test_name
    return files


@pytest.mark.parametrize("default,test", list(_launch_catalogue()))
def test_launched_commands_pass_pre_submit(monkeypatch, default, test):
    commands = []

    def capture(command, **kwargs):
        commands.append(shlex.split(command))
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(launcher, "_run_cmd", capture)
    monkeypatch.setattr(launcher, "_emit_block", lambda *args: None)
    launcher.run_single_test(test, "/tmp/integration-output", default)
    for command in commands:
        if "./run_train.sh" not in command:
            continue  # The generation variant runs a different entrypoint.
        index = command.index("./run_train.sh")
        env = dict(token.split("=", 1) for token in command[:index])
        config_file, overrides = parse_argv([env["CONFIG_FILE"], *command[index + 1 :]])
        result = run_pre_submit(config_file, overrides, world_size=int(env["NGPU"]))
        assert result.ok, (test.test_name, command, result.report.to_dict())


@pytest.mark.parametrize(
    "test", _build_precompile_tests(), ids=lambda test: test.test_name
)
def test_precompile_commands_pass_pre_submit(test):
    command = shlex.split(test.precompile_command)
    index = command.index(test.config_file)
    precompile_world_size = int(
        next(
            token.split("=", 1)[1]
            for token in command[:index]
            if token.startswith("WORLD_SIZE=")
        )
    )
    for argv, world_size in (
        (command[index:], precompile_world_size),
        (
            [
                test.config_file,
                "--dump_folder",
                f"/tmp/integration-output/{test.test_name}",
                *shlex.split(" ".join(test.override_args)),
            ],
            test.ngpu,
        ),
    ):
        config_file, overrides = parse_argv(argv)
        result = run_pre_submit(config_file, overrides, world_size=world_size)
        assert result.ok, result.report.to_dict()


def test_split_config_file():
    assert _split_config_file(["--a 1", "--config-file x.yaml", "--b"]) == (
        "x.yaml",
        ["--a 1", "--b"],
    )
    assert _split_config_file(["--config-file=x.yaml --c 2"]) == ("x.yaml", ["--c 2"])
    assert _split_config_file(["--a 1"]) == (None, ["--a 1"])


def test_launched_configs_exist_and_are_pinned():
    pinned = {
        key.removeprefix("experiments/")
        for key in load_golden()
        if key.startswith("experiments/")
    }
    for config_file in sorted(_launched_config_files()):
        assert (REPO_ROOT / config_file).is_file(), config_file
        assert Path(config_file).name in pinned, (
            f"{config_file} is launched by an integration suite but has no "
            "readable golden snapshot (tests/unit_tests/golden_configs/experiments/)"
        )
