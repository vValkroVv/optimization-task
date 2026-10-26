"""
Unit tests for the trainer-facing contract of StreamingTextDataLoader: it must
be a drop-in replacement for the HF dataloader when built through the exact
trainer kwargs (trainer.py:441-461)
"""

import pytest

pytest.importorskip("streaming")
pytest.importorskip("tree_sitter")

import torch

from tests.unit_tests.streaming_helpers import (
    make_sequential_rows,
    make_text_dataloader,
    StubTokenizer,
    write_tokenized_mds,
)
from torchtitan.components.dataloader import build_replicated_dataloader
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.data_world import DataWorld
from torchtitan.config.schema.datasets.streaming.text.configs import (
    StreamingTextDatasetConfig,
    TextStreamConfig,
)
from torchtitan.datasets.streaming.text.dataloader import StreamingTextDataLoader


def _single_rank_parallel_dims():
    return ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
    )


def _trainer_config(local_dir):
    return StreamingTextDataLoader.Config(
        dataset=StreamingTextDatasetConfig(
            streams=[TextStreamConfig(local=str(local_dir))],
            shuffle=False,
            num_canonical_nodes=1,
        ),
        infinite=False,
        num_workers=0,
    )


def _build_like_trainer(config):
    """Mirror of the dataloader build in trainer.py (data mesh=None)"""
    dataloader_kwargs = {
        "tokenizer": StubTokenizer(),
        "seq_len": 4,
        "local_batch_size": 2,
        "snapshot_every_n_steps": 8,
    }
    assert config.use_data_world
    dataloader_kwargs["data_world"] = DataWorld.from_mesh(None, role="train")
    return build_replicated_dataloader(
        config,
        lambda: config.build(**dataloader_kwargs),
        parallel_dims=_single_rank_parallel_dims(),
        device=torch.device("cpu"),
    )


def test_classvar_contract():
    """The flags the trainer and validator dispatch on: data_world build path,
    PP batch replication, validation-loop reiteration"""
    config = StreamingTextDataLoader.Config()

    assert config.use_data_world is True
    assert config.replicate_batch_over == ("pp",)
    assert config.supports_reiteration is True


def test_builds_and_fetches_through_trainer_contract(tmp_path):
    write_tokenized_mds(tmp_path, make_sequential_rows(8, row_len=4))

    loader = _build_like_trainer(_trainer_config(tmp_path))

    # pp disabled: the loader comes back unwrapped
    assert isinstance(loader, StreamingTextDataLoader)
    inputs, labels = next(iter(loader))  # unpacked exactly like the train loop
    assert inputs["input"].shape == labels.shape == (2, 4)


def test_stateful_resume_through_trainer_contract(tmp_path, monkeypatch):
    """The checkpointer holds the loader as a Stateful: state from one
    trainer-built loader must resume another trainer-built one"""
    write_tokenized_mds(tmp_path, make_sequential_rows(8, row_len=4))

    # distinct run ids give the restarted "job" its own shm namespace
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "job_before_restart")
    loader = _build_like_trainer(_trainer_config(tmp_path))
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected = [inputs["input"].tolist() for inputs, _ in iterator]

    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "job_after_restart")
    resumed = _build_like_trainer(_trainer_config(tmp_path))
    resumed.load_state_dict(state)
    assert [inputs["input"].tolist() for inputs, _ in resumed] == expected


def test_pipeline_wrapper_passes_through_without_pp(tmp_path):
    write_tokenized_mds(tmp_path, make_sequential_rows(2, row_len=4))
    loader = make_text_dataloader(tmp_path, seq_len=4, local_batch_size=1)

    wrapped = build_replicated_dataloader(
        _trainer_config(tmp_path),
        lambda: loader,
        parallel_dims=_single_rank_parallel_dims(),
        device=torch.device("cpu"),
    )

    assert wrapped is loader
