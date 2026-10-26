"""Config-level guard: dataloaders built on DataWorld need the data mesh,
whose process group cannot be created under comm.mode='torchcomms'"""

import pytest

pytest.importorskip("streaming")
pytest.importorskip("tree_sitter")
pytest.importorskip("torchtitan.trainer")

from torchtitan.components.validate import Validator
from torchtitan.config.configs import CommConfig
from torchtitan.config.schema.components.loss import ChunkedCELossConfig
from torchtitan.config.schema.datasets.hf.text_datasets import (
    HuggingFaceTextDataLoaderConfig,
)
from torchtitan.datasets.streaming.text.dataloader import StreamingTextDataLoader
from torchtitan.trainer import Trainer


def _config(**sections) -> Trainer.Config:
    sections.setdefault("dataloader", HuggingFaceTextDataLoaderConfig())
    return Trainer.Config(loss=ChunkedCELossConfig(), **sections)


def test_torchcomms_rejects_data_world_train_dataloader():
    with pytest.raises(ValueError, match="torchcomms"):
        _config(
            dataloader=StreamingTextDataLoader.Config(),
            comm=CommConfig(mode="torchcomms"),
        )


def test_torchcomms_rejects_data_world_validation_dataloader():
    with pytest.raises(ValueError, match="torchcomms"):
        _config(
            comm=CommConfig(mode="torchcomms"),
            validator=Validator.Config(
                enable=True, dataloader=StreamingTextDataLoader.Config()
            ),
        )


def test_torchcomms_guard_passes_unused_data_world_combinations():
    _config(dataloader=StreamingTextDataLoader.Config())
    _config(comm=CommConfig(mode="torchcomms"))
    _config(
        comm=CommConfig(mode="torchcomms"),
        validator=Validator.Config(dataloader=StreamingTextDataLoader.Config()),
    )
