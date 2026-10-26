"""Structured type-change diagnostics and explicit YAML mapping replacement."""

import pytest

from torchtitan.config.cli import load_document
from torchtitan.config.errors import ConfigFileError
from torchtitan.config.loader import load_raw_document, parse_yaml_value, ReplaceMapping
from torchtitan.config.pre_submit import main, run_pre_submit
from torchtitan.config.save import dump_yaml
from torchtitan.config.validation import merge_type_change_issues


@pytest.mark.parametrize(
    "experiment",
    [
        "ft_llama3_debug.yaml",
        "gift_4_debug.yaml",
        "gift_4_700b.yaml",
        "graph_deepseek_v3_debug.yaml",
        "graph_deepseek_v3_debug_flex_attn.yaml",
        "graph_deepseek_v3_debug_hybridep.yaml",
        "graph_llama3_8b.yaml",
        "graph_llama3_debug.yaml",
        "graph_llama3_debug_flex_attn.yaml",
        "graph_qwen3_debug.yaml",
        "graph_qwen3_moe_debug.yaml",
    ],
)
def test_intentional_subclass_transition_is_quiet(experiment):
    document, _ = load_document("configs/experiments/" + experiment)
    assert merge_type_change_issues(document) == []


def test_type_switch_of_a_marked_field_still_warns():
    """``qkv_linear`` is ``Derived``: its ``Annotated`` marker must not hide
    the family from the type-switch diagnostics."""
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            (
                "model.config.blocks.dense.attention.qkv_linear",
                "{type: separate, wq: {bias: true}}",
            ),
            ("model.config.blocks.dense.attention.qkv_linear.type", "fused"),
        ],
    )
    (issue,) = result.report.warnings
    assert issue.path == "model.config.blocks.dense.attention.qkv_linear"
    assert "wq" in issue.message and "!replace" in issue.message


def test_sibling_warning_is_structured_even_on_decode_error():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml", [("dataloader.type", "streaming_text")]
    )
    assert not result.ok
    assert len(result.report.warnings) == 1
    issue = result.report.warnings[0]
    assert issue.path == "dataloader"
    assert "dataset" in issue.message and "debug.yaml" in issue.message
    assert "!replace" in issue.message


def test_sibling_warning_is_counted_and_written(tmp_path, capsys):
    report = tmp_path / "report.yaml"
    assert (
        main(
            [
                "configs/experiments/llama3_debug.yaml",
                "--dataloader.type",
                "hf_chat",
                "--dataloader.dataset_path",
                "json",
                "--dataloader.sample_processor.type",
                "question_answer",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert "1 warning(s)" in output.out
    data = parse_yaml_value(report.read_text(), source=str(report))
    assert len([i for i in data["issues"] if i["severity"] == "warning"]) == 1


def test_nested_inline_replacement_discards_imported_values(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "outer: {part: {type: first, stale: 1, keep: 2}}\n"
    )
    run = tmp_path / "run.yaml"
    run.write_text(
        "imports: [base.yaml]\nouter:\n  part: !replace {type: second, keep: 3}\n"
    )
    document = load_raw_document(str(run))
    assert document.data == {"outer": {"part": {"type": "second", "keep": 3}}}
    assert not document.type_changes


def test_cli_replacement_resets_section_in_one_value():
    document, overrides = load_document(
        "configs/experiments/llama3_debug.yaml",
        [("dataloader", "!replace {type: hf_chat, dataset: custom}")],
    )
    assert document.data["dataloader"] == {"type": "hf_chat", "dataset": "custom"}
    assert not document.type_changes
    assert isinstance(overrides["dataloader"], ReplaceMapping)
    restored = parse_yaml_value(dump_yaml(overrides), source="overrides")
    assert isinstance(restored["dataloader"], ReplaceMapping)


def test_replacement_survives_import_chain_and_later_overrides(tmp_path):
    (tmp_path / "a.yaml").write_text("outer: {part: {old: 1}}\n")
    (tmp_path / "b.yaml").write_text(
        "imports: [a.yaml]\nouter: {part: !replace {new: 2}}\n"
    )
    run = tmp_path / "run.yaml"
    run.write_text("imports: [b.yaml]\nouter: {part: {last: 3}}\n")
    assert load_raw_document(str(run)).data == {
        "outer": {"part": {"new": 2, "last": 3}}
    }


def test_existing_null_then_cli_mapping_still_replaces():
    document, _ = load_document(
        "configs/experiments/llama3_debug.yaml",
        [("dataloader", "null"), ("dataloader", "{type: hf_chat, dataset: custom}")],
    )
    assert document.data["dataloader"] == {"type": "hf_chat", "dataset": "custom"}
    assert not document.type_changes


@pytest.mark.parametrize(
    "text,message",
    [
        ("!replace 1", "!replace requires a mapping"),
        ("!replace [1, 2]", "!replace requires a mapping"),
        ("!replce {a: 1}", "unknown YAML tag.*!replace"),
    ],
)
def test_invalid_replacement_tags_have_actionable_errors(text, message):
    with pytest.raises(ConfigFileError, match=message):
        parse_yaml_value(text, source="test")


def test_container_routes_type_warning_through_logger(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from torchtitan.config import build, cli
    from torchtitan.config.validation import ValidationReport

    monkeypatch.setattr(
        build,
        "_build_model_spec",
        lambda section, tree: SimpleNamespace(model=tree, name="llama3"),
    )
    monkeypatch.setattr(
        cli, "preflight", lambda *args: ValidationReport(stage="in_container")
    )
    monkeypatch.setattr(
        cli,
        "validate_typed_config",
        lambda *args: ValidationReport(stage="in_container"),
    )
    logger = Mock()
    loaded = cli.load_trainer_config(
        [
            "configs/experiments/llama3_debug.yaml",
            "--model.config.rope.type",
            "cos_sin",
            "--model.config.rope.scaling",
            "!replace {type: none}",
        ],
        logger=logger,
    )
    assert len(loaded.preflight_report.warnings) == 1
    logger.warning.assert_called_once()
    assert "model.config.rope:" in logger.warning.call_args.args[0]


def test_unrelated_family_type_names_cannot_suppress_warning(tmp_path):
    # "default" exists in several registries: use this field's loss family.
    root = tmp_path / "base.yaml"
    root.write_text("config_version: 1\nloss: {type: cross_entropy, bogus: 1}\n")
    document = load_raw_document(str(root), overrides=[{"loss": {"type": "mtp"}}])
    assert len(merge_type_change_issues(document)) == 1


def test_replacement_after_a_switch_removes_stale_warning(tmp_path):
    root = tmp_path / "base.yaml"
    root.write_text("dataloader: {type: hf_text, dataset: old}\n")
    replacement = parse_yaml_value("!replace {type: hf_chat}", source="test")
    document = load_raw_document(
        str(root),
        overrides=[{"dataloader": {"type": "hf_chat"}}, {"dataloader": replacement}],
    )
    assert document.type_changes == ()


@pytest.mark.parametrize("replacement", [None, "new_dataset"])
def test_later_dataset_override_does_not_warn_about_old_source(replacement):
    document = load_raw_document(
        "configs/experiments/llama3_debug.yaml",
        overrides=[
            {"dataloader": {"type": "hf_chat"}},
            {"dataloader": {"dataset": replacement}},
        ],
    )
    assert merge_type_change_issues(document) == []


def test_switch_back_to_original_type_does_not_warn():
    document = load_raw_document(
        "configs/experiments/llama3_debug.yaml",
        overrides=[
            {"dataloader": {"type": "hf_chat"}},
            {"dataloader": {"type": "hf_text"}},
        ],
    )
    assert merge_type_change_issues(document) == []


def test_malformed_trainer_is_a_schema_error_despite_type_events():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [("trainer", "[bad]"), ("dataloader.type", "hf_chat")],
    )
    assert not result.ok
    assert "trainer" in str(result.report.errors)


def test_open_kwargs_type_is_not_a_component_switch():
    document = load_raw_document(
        "configs/experiments/sft_llama3_debug.yaml",
        overrides=[
            {"dataloader": {"load_dataset_kwargs": {"type": "first", "value": 1}}},
            {"dataloader": {"load_dataset_kwargs": {"type": "second"}}},
        ],
    )
    assert merge_type_change_issues(document) == []


def test_root_replacement_also_applies_in_section_import(tmp_path):
    (tmp_path / "base.yaml").write_text("dataloader: {old: 1}\n")
    (tmp_path / "replacement.yaml").write_text(
        "!replace\nimports: [base.yaml]\ndataloader: {new: 2}\n"
    )
    root = tmp_path / "run.yaml"
    root.write_text("imports: {dataloader: replacement.yaml}\n")
    assert load_raw_document(str(root)).data == {"dataloader": {"new": 2}}


def test_replacement_alias_is_not_consumed_in_another_branch(tmp_path):
    root = tmp_path / "run.yaml"
    root.write_text("one: &fresh !replace {value: 3}\ntwo: *fresh\n")
    assert load_raw_document(str(root)).data == {
        "one": {"value": 3},
        "two": {"value": 3},
    }


@pytest.mark.parametrize("value", ["[]", "{}"])
def test_non_string_trainer_type_is_a_schema_error_with_type_events(value):
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [("trainer.type", value), ("dataloader.type", "hf_chat")],
    )
    assert not result.ok
    assert "trainer.type" in str(result.report.errors)
