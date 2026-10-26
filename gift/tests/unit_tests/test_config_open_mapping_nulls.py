"""Null mapping values reset keys regardless of the selected field schema."""

from pathlib import Path

import pytest

from torchtitan.config.build import build_trainer_config
from torchtitan.config.cli import load_document
from torchtitan.config.loader import load_raw_document
from torchtitan.config.save import check_round_trip, dump_yaml, get_typed_document


def _experiment(tmp_path, options):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        dump_yaml(
            {
                "config_version": 1,
                "imports": [
                    str(Path("configs/experiments/sft_llama3_debug.yaml").resolve())
                ],
                "dataloader": {"load_dataset_kwargs": options},
            }
        )
    )
    return path


def test_nested_mapping_nulls_are_resets_in_both_loaders(tmp_path):
    path = _experiment(
        tmp_path,
        {"option": None, "nested": {"nullable": None}, "items": [{"value": None}]},
    )
    raw = load_raw_document(str(path))
    document, _ = load_document(str(path))
    assert raw.data == document.data
    kwargs = document.data["dataloader"]["load_dataset_kwargs"]
    assert "option" not in kwargs
    assert "nested" not in kwargs
    assert kwargs["items"] == [{}]


def test_cli_null_resets_open_mapping_key_and_config_field(tmp_path):
    path = _experiment(tmp_path, {"option": "old"})
    document, _ = load_document(
        str(path),
        [("dataloader.load_dataset_kwargs.option", "null"), ("optimizer.lr", "null")],
    )
    assert "option" not in document.data["dataloader"]["load_dataset_kwargs"]
    assert "lr" not in document.data["optimizer"]


def test_open_mapping_can_be_reset_as_a_whole(tmp_path):
    path = _experiment(tmp_path, {"option": "old"})
    document, _ = load_document(str(path), [("dataloader.load_dataset_kwargs", "null")])
    assert "load_dataset_kwargs" not in document.data["dataloader"]


def test_reset_kwargs_survives_resolved_bundle_round_trip(tmp_path):
    document, _ = load_document(
        "configs/experiments/sft_llama3_debug.yaml",
        [("dataloader.load_dataset_kwargs.split", "null")],
    )
    config = build_trainer_config(document, with_model_spec=False)
    resolved = get_typed_document(config, document.data["model"])
    assert "split" not in config.dataloader.load_dataset_kwargs
    assert (
        resolved["dataloader"]["load_dataset_kwargs"]
        == config.dataloader.load_dataset_kwargs
    )
    check_round_trip(config, document.data, resolved, str(tmp_path))


@pytest.mark.parametrize("trainer", ["default", "unknown"])
def test_raw_merge_does_not_depend_on_schema_or_component_type(tmp_path, trainer):
    path = tmp_path / "raw.yaml"
    path.write_text(
        dump_yaml(
            {
                "config_version": 1,
                "trainer": {"type": trainer},
                "payload": {"option": "old", "nested": {"nullable": None}},
                "entries": [{"options": {"nullable": None}}],
            }
        )
    )
    document, _ = load_document(str(path), [("payload.option", "null")])
    assert document.data["payload"] == {}
    assert document.data["entries"] == [{}]
