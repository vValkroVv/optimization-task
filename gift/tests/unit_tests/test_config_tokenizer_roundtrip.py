"""Every tokenizer schema can be saved and decoded without importing runtimes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torchtitan.config.build import build_trainer_config
from torchtitan.config.cli import load_document
from torchtitan.config.configurable import to_plain
from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigFileError
from torchtitan.config.registry import families, import_schema
from torchtitan.config.save import (
    check_round_trip,
    get_typed_document,
    save_loaded_run_bundle,
)
from torchtitan.config.schema.components.tokenizer import BaseTokenizerConfig


def test_all_registered_tokenizers_round_trip():
    import_schema()
    examples = {
        "huggingface": {},
        "multimodal": dict(
            image_token="image",
            video_token="video",
            vision_start_token="start",
            vision_end_token="end",
            pad_token="pad",
        ),
        "flux": {},
    }
    family = families()["tokenizer"]
    assert set(examples) == set(family.types)
    for name, fields in examples.items():
        config = decode(BaseTokenizerConfig, {"type": name, **fields})
        assert to_plain(config)["type"] == name
        assert decode(BaseTokenizerConfig, to_plain(config)) == config


def test_flux_resolved_document_round_trip(tmp_path, monkeypatch):
    import torchtitan.config.save as save

    document, _ = load_document("configs/experiments/flux_debug.yaml")
    config = build_trainer_config(document, with_model_spec=False)
    resolved = get_typed_document(config, document.data["model"])
    assert resolved["tokenizer"]["type"] == "flux"
    check_round_trip(config, document.data, resolved, str(tmp_path))
    monkeypatch.setattr(save, "collect_metadata", lambda **kwargs: {})
    path = save.save_run_bundle(
        config, document.data, dest=str(tmp_path), command=[], runtime=None
    )
    reloaded, _ = load_document(path)
    assert build_trainer_config(reloaded, with_model_spec=False) == config


def test_serialization_failure_is_logged_as_config_error(monkeypatch, tmp_path):
    import torchtitan.config.save as save

    document, _ = load_document("configs/experiments/flux_debug.yaml")
    config = build_trainer_config(document, with_model_spec=False)
    config.dump_folder = str(tmp_path)
    loaded = SimpleNamespace(trainer_config=config, source_document=document.data)
    logger = Mock()

    def fail(value):
        raise ValueError("unregistered runtime-only schema")

    monkeypatch.setattr(save, "to_plain", fail)
    with pytest.raises(SystemExit) as error:
        save_loaded_run_bundle(loaded, logger=logger)
    assert error.value.code == 1
    assert isinstance(error.value.__cause__, ConfigFileError)
    assert "unregistered runtime-only schema" in str(error.value.__cause__)
    logger.error.assert_called_once()
