"""Public entry points prepare a model once and share its pristine tree."""

from unittest.mock import Mock

import pytest

from torchtitan.config import build, pre_submit, run_pre_submit
from torchtitan.config.schema.components.lora import LoRAConverterConfig
from torchtitan.config.schema.models.llama3.model import Llama3ModelConfig
from torchtitan.config.sections import ModelSection

EXPERIMENT = "configs/experiments/llama3_debug.yaml"


@pytest.fixture
def preparation_calls(monkeypatch):
    import importlib
    from torchtitan.config import save

    calls = {"decode": 0, "materialize": 0, "tree": None, "validated_tree": None}
    decode = build.decode
    materialize = Llama3ModelConfig.materialize
    validate = LoRAConverterConfig.validate_pipeline

    def counted_decode(cls, *args, **kwargs):
        if cls is ModelSection:
            calls["decode"] += 1
        return decode(cls, *args, **kwargs)

    def counted_materialize(config):
        calls["materialize"] += 1
        tree = materialize(config)
        calls["tree"] = calls["tree"] or tree  # the run's, not a reload's
        return tree

    def counted_validate(config, tree, previous, trainer_config=None):
        calls["validated_tree"] = calls["validated_tree"] or tree
        return validate(config, tree, previous, trainer_config)

    monkeypatch.setattr(build, "decode", counted_decode)
    monkeypatch.setattr(save, "decode", counted_decode)
    monkeypatch.setattr(
        importlib.import_module("torchtitan.config.decode"), "decode", counted_decode
    )
    monkeypatch.setattr(Llama3ModelConfig, "materialize", counted_materialize)
    monkeypatch.setattr(LoRAConverterConfig, "validate_pipeline", counted_validate)
    return calls


@pytest.mark.parametrize("pattern,valid", [(".*w1$", True), (".*wo$", False)])
def test_pre_submit_shares_one_tree_with_lora_fp8_validation(
    preparation_calls, pattern, valid
):
    result = run_pre_submit(
        EXPERIMENT,
        [
            ("model.config.converters", "[{type: lora, target_modules: [wo]}]"),
            ("fp8.dense_include_linear_layers", pattern),
        ],
    )
    assert result.ok is valid
    # Once for the run, once more for the resolved-YAML round trip of a valid one.
    assert (
        preparation_calls["decode"]
        == preparation_calls["materialize"]
        == (2 if valid else 1)
    )
    assert preparation_calls["validated_tree"] is preparation_calls["tree"]
    if not valid:
        assert result.report.errors[0].path == "model.config.converters[0]"


def test_combined_print_modes_reuse_validated_preparation(preparation_calls, capsys):
    assert pre_submit.main([EXPERIMENT, "--print-resolved", "--print-layers"]) == 0
    # The run plus the round-trip reload; printing prepares nothing more.
    assert preparation_calls["decode"] == preparation_calls["materialize"] == 2
    output = capsys.readouterr().out
    assert "config_version: 1" in output
    assert "layers:" in output


def test_torch_free_trainer_builder_keeps_public_return_contract(preparation_calls):
    from torchtitan.config import load_document
    from torchtitan.config.schema.trainer import TrainerConfig

    document, _ = load_document(EXPERIMENT)
    trainer = build.build_trainer_config(document, with_model_spec=False)
    assert isinstance(trainer, TrainerConfig)
    assert trainer.model_spec is None
    assert preparation_calls["decode"] == preparation_calls["materialize"] == 1


def test_loaded_bundle_decodes_source_and_checks_serialized_text_once(
    preparation_calls, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from torchtitan.config import load_document, save
    from torchtitan.config.validation import RuntimeInfo, ValidationReport

    document, _ = load_document(EXPERIMENT)
    prepared = build.prepare_document(document)
    prepared.trainer_config.dump_folder = str(tmp_path)
    loaded = SimpleNamespace(
        trainer_config=prepared.trainer_config,
        source_document=document.data,
        runtime=RuntimeInfo(),
        applied_overrides={},
        import_graph=document.graph,
        preflight_report=ValidationReport(stage="in_container"),
    )
    preparation_calls["decode"] = preparation_calls["materialize"] = 0
    monkeypatch.setattr(save, "collect_metadata", lambda **kwargs: {})
    path = save.save_loaded_run_bundle(loaded, logger=Mock())
    # Decode the live source and serialized model once each; only the latter
    # needs materialization for the round-trip validation.
    assert preparation_calls["decode"] == 2
    assert preparation_calls["materialize"] == 1
    assert path == str(tmp_path / "config/resolved_config.yaml")
    assert (tmp_path / "config/source_config.yaml").is_file()
    assert len(list((tmp_path / "config/launches").iterdir())) == 1


def test_public_bundle_decodes_source_and_serialized_model_once_each(
    preparation_calls, tmp_path, monkeypatch
):
    from torchtitan.config import load_document, save
    from torchtitan.config.validation import RuntimeInfo

    document, _ = load_document(EXPERIMENT)
    trainer = build.prepare_document(document).trainer_config
    preparation_calls["decode"] = preparation_calls["materialize"] = 0
    monkeypatch.setattr(save, "collect_metadata", lambda **kwargs: {})
    path = save.save_run_bundle(
        trainer,
        document.data,
        dest=str(tmp_path),
        command=["train"],
        runtime=RuntimeInfo(),
    )
    assert preparation_calls["decode"] == 2
    assert preparation_calls["materialize"] == 1
    assert path == str(tmp_path / "resolved_config.yaml")


def test_loaded_bundle_reads_current_source_model(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from torchtitan.config import cli, load_config, save
    from torchtitan.config.validation import ValidationReport

    monkeypatch.setattr(
        build,
        "_build_model_spec",
        lambda section, tree: SimpleNamespace(model=tree, name="llama3"),
    )
    monkeypatch.setattr(
        cli, "preflight", lambda *args: ValidationReport(stage="in_container")
    )
    loaded = load_config(EXPERIMENT)
    loaded.trainer_config.dump_folder = str(tmp_path)
    # The source document is mutable; serialize its current model consistently.
    loaded.source_document["model"]["config"]["dim"] = 512
    loaded.source_document["model"]["config"]["rope"]["dim"] = 32
    monkeypatch.setattr(save, "collect_metadata", lambda **kwargs: {})
    save.save_loaded_run_bundle(loaded, logger=Mock())
    from torchtitan.config.loader import parse_yaml_value

    source = parse_yaml_value(
        (tmp_path / "config/source_config.yaml").read_text(), source="source"
    )
    resolved = parse_yaml_value(
        (tmp_path / "config/resolved_config.yaml").read_text(), source="resolved"
    )
    assert source["model"]["config"]["dim"] == resolved["model"]["config"]["dim"] == 512
    assert (
        source["model"]["config"]["rope"]["dim"]
        == resolved["model"]["config"]["rope"]["dim"]
        == 32
    )


def test_invalid_combined_print_modes_preserve_error_and_report(tmp_path, capsys):
    from torchtitan.config.loader import parse_yaml_value

    report = tmp_path / "report.yaml"
    result = pre_submit.main(
        [
            EXPERIMENT,
            "--model.config.rope.theta",
            ".nan",
            "--print-merged",
            "--print-resolved",
            "--report",
            str(report),
        ]
    )
    assert result == 1
    output = capsys.readouterr()
    assert "config_version: 1" in output.out
    assert "model.config.rope.theta" in output.err
    assert "AttributeError" not in output.err
    payload = parse_yaml_value(report.read_text(), source="report")
    assert any(
        issue["path"] == "model.config.rope.theta" for issue in payload["issues"]
    )
