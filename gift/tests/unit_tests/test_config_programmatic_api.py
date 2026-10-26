"""Programmatic loading shares validation with the CLI without exiting."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torchtitan.config import load_config
from torchtitan.config import build, cli
from torchtitan.config.errors import ConfigError, ConfigSchemaError
from torchtitan.config.overrides import parse_argv
from torchtitan.config.validation import Issue, RuntimeInfo, ValidationReport


@pytest.fixture
def runtime_builder(monkeypatch):
    def builder(section, model_tree):
        return SimpleNamespace(model=model_tree, name="llama3")

    monkeypatch.setattr(build, "_build_model_spec", builder)
    monkeypatch.setattr(RuntimeInfo, "from_env", lambda: RuntimeInfo())
    monkeypatch.setattr(
        cli, "preflight", lambda *args: ValidationReport(stage="in_container")
    )
    monkeypatch.setattr(
        cli,
        "validate_typed_config",
        lambda *args: ValidationReport(stage="in_container"),
    )
    return builder


def test_pathlike_and_mixed_layers_preserve_order_and_caller_data(runtime_builder):
    from pathlib import Path

    layer = {"optimizer": {"lr": 0.002}, "training": {"steps": 5}}
    original = deepcopy(layer)
    loaded = load_config(
        Path("configs/experiments/llama3_debug.yaml"),
        overrides=["--optimizer.lr", "0.001", layer, "--optimizer.lr", "0.003"],
    )
    assert loaded.source_document["optimizer"]["lr"] == 0.003
    assert loaded.source_document["training"]["steps"] == 5
    assert loaded.applied_overrides["optimizer.lr"] == 0.003
    assert layer == original


def test_programmatic_open_mapping_null_resets_the_key(runtime_builder):
    loaded = load_config(
        "configs/experiments/sft_llama3_debug.yaml",
        overrides=[
            {"dataloader": {"load_dataset_kwargs": {"option": "old"}}},
            {"dataloader": {"load_dataset_kwargs": {"option": None}}},
        ],
    )
    assert "option" not in loaded.source_document["dataloader"]["load_dataset_kwargs"]


def test_missing_config_raises_without_logger_or_exit(runtime_builder, tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_preflight_error_is_raised_without_exit(runtime_builder, monkeypatch):
    def failed(*args):
        report = ValidationReport(stage="in_container")
        report.error("dump_folder", "unwritable")
        return report

    monkeypatch.setattr(cli, "preflight", failed)
    with pytest.raises(ConfigError, match="dump_folder"):
        load_config("configs/experiments/llama3_debug.yaml")


def test_returned_report_keeps_both_warning_sources(runtime_builder, monkeypatch):
    def typed(*args):
        return ValidationReport(
            stage="in_container",
            issues=[Issue(path="training", message="example", severity="warning")],
        )

    monkeypatch.setattr(cli, "validate_typed_config", typed)
    monkeypatch.setattr(
        cli,
        "merge_type_change_issues",
        lambda document: [
            Issue(path="dataloader", message="type changed", severity="warning")
        ],
    )
    loaded = load_config(
        "configs/experiments/llama3_debug.yaml",
    )
    assert {issue.path for issue in loaded.preflight_report.warnings} == {
        "training",
        "dataloader",
    }


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_invalid_explicit_world_size_is_config_error(
    runtime_builder, monkeypatch, value
):
    monkeypatch.setenv("WORLD_SIZE", value)
    with pytest.raises(ConfigError, match="WORLD_SIZE"):
        load_config("configs/experiments/llama3_debug.yaml")


def test_cli_wrapper_still_logs_and_exits(runtime_builder, monkeypatch):
    def failed(document):
        raise ConfigSchemaError("optimizer.lr: invalid")

    monkeypatch.setattr(build, "prepare_document", failed)
    logger = Mock()
    with pytest.raises(SystemExit) as exc:
        cli.load_trainer_config(
            ["configs/experiments/llama3_debug.yaml"], logger=logger
        )
    assert exc.value.code == 1
    logger.error.assert_called_once()


@pytest.mark.parametrize(
    "overrides", ["--optimizer.lr 1", {"optimizer": {"lr": 1}}, [object()]]
)
def test_malformed_override_container_is_rejected(runtime_builder, overrides):
    with pytest.raises(ConfigError, match="overrides"):
        load_config("configs/experiments/llama3_debug.yaml", overrides=overrides)


def test_bare_flag_before_file_has_clear_grammar_error():
    with pytest.raises(ConfigError, match="config file path must precede"):
        parse_argv(["--debug.print_config", "experiment.yaml"])


@pytest.mark.parametrize("arg", ["--checkpoint.interval 0", "--a .b=1", "--=1"])
def test_override_path_with_whitespace_is_malformed(arg):
    with pytest.raises(
        ConfigSchemaError, match=r"malformed override argument .*--a\.b=value"
    ):
        parse_argv(["experiment.yaml", arg])


def test_override_value_may_contain_spaces():
    assert parse_argv(["experiment.yaml", "--a.b=x y"])[1] == [("a.b", "x y")]


def test_marker_can_be_used_in_programmatic_cli_tokens(runtime_builder):
    loaded = load_config(
        "configs/experiments/llama3_debug.yaml",
        overrides=["--dataloader", "!replace {type: hf_text_interleaved}"],
    )
    assert loaded.source_document["dataloader"] == {"type": "hf_text_interleaved"}
    assert not loaded.preflight_report.warnings


def test_nested_readonly_mapping_is_supported(runtime_builder):
    from types import MappingProxyType

    loaded = load_config(
        "configs/experiments/llama3_debug.yaml",
        overrides=[{"optimizer": MappingProxyType({"lr": 0.002})}],
    )
    assert loaded.source_document["optimizer"]["lr"] == 0.002


def test_cyclic_override_has_a_config_error(runtime_builder):
    layer = {}
    layer["loop"] = layer
    with pytest.raises(ConfigError, match="cyclic"):
        load_config("configs/experiments/llama3_debug.yaml", overrides=[layer])
