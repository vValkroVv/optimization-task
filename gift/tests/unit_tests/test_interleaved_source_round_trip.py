"""Interleaved dataset descriptions stay plain values through run bundles."""

import dataclasses
import logging
from pathlib import Path

import pytest
import yaml

from torchtitan.config import load_raw_document
from torchtitan.config.build import prepare_document
from torchtitan.config.cli import LoadedConfig
from torchtitan.config.configurable import family_of
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.loader import ReplaceMapping
from torchtitan.config.pre_submit import run_pre_submit
from torchtitan.config.save import (
    get_typed_document,
    save_loaded_run_bundle,
    save_run_bundle,
)
from torchtitan.config.schema.datasets.hf.text_datasets import (
    ChatDataSource,
    HFDataSource,
)
from torchtitan.config.validation import RuntimeInfo, ValidationReport


@pytest.fixture(params=["hf_text_interleaved", "hf_chat_interleaved"])
def loader_data(request):
    if request.param == "hf_text_interleaved":
        sources = [
            {
                "dataset": "c4",
                "dataset_path": "datasets/first",
                "infinite": False,
                "weight": 0.25,
            },
            {
                "dataset": "custom",
                "dataset_path": "datasets/second",
                "infinite": False,
                "weight": 0.75,
            },
        ]
    else:
        sources = [
            {
                "dataset_path": "json",
                "load_dataset_kwargs": {
                    "data_files": ["first.jsonl"],
                    "split": "train",
                },
                "sample_processor": {
                    "type": "question_answer",
                    "question_key": "prompt",
                    "answer_key": "response",
                },
                "infinite": False,
                "weight": 0.25,
            },
            {
                "dataset_path": "chat/second",
                "load_dataset_kwargs": {"streaming": True},
                "sample_processor": {"type": "split_reasoning", "separator": "####"},
                "infinite": False,
                "weight": 0.75,
            },
        ]
    return {
        "type": request.param,
        "sources": sources,
        "seed": 17,
        "stopping_strategy": "all_exhausted",
    }


def _raw(loader):
    return load_raw_document(
        "configs/experiments/llama3_debug.yaml",
        overrides=[{"dataloader": ReplaceMapping(loader)}],
    )


def test_sources_are_values_outside_the_dataloader_family():
    for source in (HFDataSource, ChatDataSource):
        assert dataclasses.is_dataclass(source)
        assert family_of(source) is None
        assert not hasattr(source, "owner")


def test_interleaved_sources_prepare_serialize_decode_and_save(tmp_path, loader_data):
    raw = _raw(loader_data)
    prepared = prepare_document(raw)
    loader = prepared.trainer_config.dataloader
    assert [s.weight for s in loader.sources] == [0.25, 0.75]
    assert all(not s.infinite for s in loader.sources)
    resolved = get_typed_document(prepared.trainer_config, raw.data["model"])
    sources = resolved["dataloader"]["sources"]
    assert all("type" not in source for source in sources)
    for actual, written in zip(sources, loader_data["sources"], strict=True):
        for key, value in written.items():
            if key == "sample_processor":
                assert all(
                    actual[key][name] == expected for name, expected in value.items()
                )
            else:
                assert actual[key] == value

    path = save_run_bundle(
        prepared.trainer_config,
        raw.data,
        dest=str(tmp_path / "bundle"),
        command=["python3", "train.py"],
        runtime=RuntimeInfo(),
    )
    saved = yaml.safe_load(Path(path).read_text())
    assert saved["dataloader"] == resolved["dataloader"]
    reloaded = prepare_document(load_raw_document(path))
    assert reloaded.trainer_config.dataloader == loader
    result = run_pre_submit(path)
    assert result.ok, result.report.issues

    prepared.trainer_config.dump_folder = str(tmp_path / "launch")
    loaded = LoadedConfig(
        trainer_config=prepared.trainer_config,
        config_path="configs/experiments/llama3_debug.yaml",
        source_document=raw.data,
        import_graph=raw.graph,
        applied_overrides={},
        runtime=RuntimeInfo(),
        preflight_report=ValidationReport(stage="pre_submit"),
    )
    saved_path = save_loaded_run_bundle(loaded, logger=logging.getLogger(__name__))
    assert saved_path is not None
    assert (
        yaml.safe_load(Path(saved_path).read_text())["dataloader"]
        == resolved["dataloader"]
    )


@pytest.mark.parametrize("ignored_field", ["num_workers", "pin_memory"])
def test_source_rejects_loader_settings_it_cannot_apply(loader_data, ignored_field):
    loader_data["sources"][0][ignored_field] = 1
    with pytest.raises(
        ConfigSchemaError, match=rf"sources\[0\].{ignored_field}: unknown key"
    ):
        prepare_document(_raw(loader_data))


def test_chat_source_still_requires_a_dataset_path():
    data = {
        "type": "hf_chat_interleaved",
        "sources": [{"sample_processor": {"type": "question_answer"}}],
    }
    with pytest.raises(ConfigSchemaError, match="dataset_path"):
        prepare_document(_raw(data))
