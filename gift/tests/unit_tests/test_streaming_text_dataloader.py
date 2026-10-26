"""
Unit tests for StreamingTextDataLoader

Covers the Titan -> streaming world translation (data is partitioned by the
data-loading DataWorld, and the checkpoint sample counter is global), the
end-to-end dataset -> loader -> collator batch contract, drop_last and
infinite epochs
"""

import itertools

import pytest

pytest.importorskip("streaming")
pytest.importorskip("tree_sitter")

import torch

from tests.unit_tests.streaming_helpers import (
    install_single_process_shm_gather,
    make_sequential_rows,
    make_text_dataloader,
    write_tokenized_mds,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config.schema.datasets.streaming.text.configs import (
    StreamingTextDatasetConfig,
    TextStreamConfig,
)
from torchtitan.config.schema.datasets.streaming.text.dataloader import (
    _stream_fim_rates,
)


def test_state_dict_scales_num_samples_by_dp_world_size(tmp_path, monkeypatch):
    """
    Checkpoint sample counter must be global: per-rank samples times
    dp_world_size, stored under the rank-independent dp_rank_0 key
    """

    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(32, row_len=4))
    loader = make_text_dataloader(
        tmp_path, seq_len=4, local_batch_size=2, dp_rank=0, dp_world_size=4
    )

    consumed = list(itertools.islice(iter(loader), 3))
    assert len(consumed) == 3

    state = loader.state_dict()
    assert set(state) == {"dp_rank_0"}
    assert state["dp_rank_0"]["sample_in_epoch"] == 3 * 2 * 4


def test_partitioning_follows_data_world_not_env(tmp_path, monkeypatch):
    """
    Ranks split the dataset by DataWorld coordinates; the global env vars
    (which describe the full TP/PP/CP mesh) must not leak into partitioning
    """

    install_single_process_shm_gather(monkeypatch)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    write_tokenized_mds(tmp_path, make_sequential_rows(16, row_len=4))

    rank_samples = []
    for dp_rank in range(2):
        loader = make_text_dataloader(
            tmp_path, seq_len=4, local_batch_size=2, dp_rank=dp_rank, dp_world_size=2
        )
        first_tokens = {
            int(token) for inputs, _ in loader for token in inputs["input"][:, 0]
        }
        assert len(first_tokens) == 8
        rank_samples.append(first_tokens)

    assert rank_samples[0].isdisjoint(rank_samples[1])
    assert rank_samples[0] | rank_samples[1] == {1000 + i for i in range(16)}


def test_partitioning_shuffled_four_ranks(tmp_path, monkeypatch):
    """Shuffled four-rank partitioning still covers the dataset exactly once"""

    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(16, row_len=4))

    sample_ids = []
    for dp_rank in range(4):
        loader = make_text_dataloader(
            tmp_path,
            seq_len=4,
            local_batch_size=2,
            dp_rank=dp_rank,
            dp_world_size=4,
            shuffle=True,
        )
        rank_ids = [
            int(token) for inputs, _ in loader for token in inputs["input"][:, 0]
        ]
        assert len(rank_ids) == 4
        sample_ids += rank_ids

    # no duplicates, no gaps across ranks
    assert sorted(sample_ids) == [1000 + i for i in range(16)]
    # shuffle actually reordered the samples (deterministic under the pinned seed)
    assert sample_ids != sorted(sample_ids)


def test_batches_invariant_to_num_workers(tmp_path):
    """Worker processes must not change the batch stream"""

    write_tokenized_mds(tmp_path, make_sequential_rows(16, row_len=4))

    def epoch_batches(num_workers):
        loader = make_text_dataloader(
            tmp_path,
            seq_len=4,
            local_batch_size=2,
            shuffle=True,
            num_workers=num_workers,
        )
        return [inputs["input"].tolist() for inputs, _ in loader]

    assert epoch_batches(0) == epoch_batches(2)


@pytest.mark.parametrize(
    "dp_world_size,local_batch_size,num_workers",
    [(2, 4, 0), (4, 2, 0), (2, 4, 2), (1, 8, 2)],
)
def test_global_batch_invariant_to_topology(
    tmp_path, monkeypatch, dp_world_size, local_batch_size, num_workers
):
    """
    The per-optimizer-step global sample multiset must not depend on how the
    global batch is split across ranks, workers and local batches.
    num_canonical_nodes is pinned to 1 in the factory so the shuffle order does
    not change with dp_world_size
    """

    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(32, row_len=4))

    def step_sample_ids(world_size, local_batch, workers):
        per_rank = []
        for dp_rank in range(world_size):
            loader = make_text_dataloader(
                tmp_path,
                seq_len=4,
                local_batch_size=local_batch,
                dp_rank=dp_rank,
                dp_world_size=world_size,
                shuffle=True,
                num_workers=workers,
            )
            per_rank.append(
                [[row[0] for row in inputs["input"].tolist()] for inputs, _ in loader]
            )
        num_steps = len(per_rank[0])
        assert all(len(rank_steps) == num_steps for rank_steps in per_rank)
        return [
            sorted(sample for rank_steps in per_rank for sample in rank_steps[step])
            for step in range(num_steps)
        ]

    reference = step_sample_ids(1, 8, 0)
    assert len(reference) == 4  # 32 rows / global batch 8
    assert step_sample_ids(dp_world_size, local_batch_size, num_workers) == reference


def test_e2e_batch_format_and_content(tmp_path):
    rows = make_sequential_rows(4, row_len=6)
    write_tokenized_mds(tmp_path, rows)
    loader = make_text_dataloader(tmp_path, seq_len=6, local_batch_size=2)

    inputs, labels = next(iter(loader))

    assert set(inputs) == {"input", "positions"}
    assert inputs["input"].shape == labels.shape == (2, 6)
    assert inputs["input"].dtype == labels.dtype == torch.int64
    assert inputs["positions"].dtype == torch.int32
    assert torch.equal(inputs["input"], torch.tensor(rows[:2], dtype=torch.int64))
    assert torch.equal(labels[:, :-1], inputs["input"][:, 1:])
    assert (labels[:, -1] == IGNORE_INDEX).all()
    # rows contain no eos, so positions count up from every row start
    assert torch.equal(
        inputs["positions"], torch.arange(6, dtype=torch.int32).repeat(2, 1)
    )


@pytest.mark.parametrize(
    "drop_last,expected_batch_sizes",
    [(True, [4, 4]), (False, [4, 4, 2])],
)
def test_drop_last_controls_partial_batch(tmp_path, drop_last, expected_batch_sizes):
    write_tokenized_mds(tmp_path, make_sequential_rows(10, row_len=4))
    loader = make_text_dataloader(
        tmp_path, seq_len=4, local_batch_size=4, drop_last=drop_last
    )
    assert [len(labels) for _, labels in loader] == expected_batch_sizes


def test_infinite_replays_epochs_in_order(tmp_path):
    write_tokenized_mds(tmp_path, make_sequential_rows(6, row_len=4))
    loader = make_text_dataloader(
        tmp_path, seq_len=4, local_batch_size=2, infinite=True
    )

    first_tokens = [
        int(inputs["input"][0, 0]) for inputs, _ in itertools.islice(iter(loader), 6)
    ]

    # 6 rows / batch 2 = 3 batches per epoch; the second epoch replays the first
    assert first_tokens[:3] == first_tokens[3:]


def test_infinite_raises_when_epochs_stay_empty(tmp_path):
    write_tokenized_mds(tmp_path, make_sequential_rows(2, row_len=4))
    loader = make_text_dataloader(
        tmp_path, seq_len=4, local_batch_size=4, drop_last=True, infinite=True
    )
    with pytest.raises(RuntimeError, match="no batches in two consecutive epochs"):
        next(iter(loader))


def test_build_requires_tokenizer_eos_id(tmp_path):
    write_tokenized_mds(tmp_path, make_sequential_rows(2, row_len=4))
    with pytest.raises(ValueError, match="eos_id"):
        make_text_dataloader(
            tmp_path, seq_len=4, local_batch_size=1, tokenizer=object()
        )


def test_stream_fim_rates_resolution():
    """
    Per-stream fim_rates override the dataset-level ones, and disabling
    syntax-aware FIM zeroes that rate everywhere
    """

    config = StreamingTextDatasetConfig(
        streams=[
            TextStreamConfig(local="/data/a"),
            TextStreamConfig(
                local="/data/b",
                fim_rates={"vanilla_fim": 0.5, "syntax_aware_fim": 0.2},
            ),
        ],
        fim_rates={"vanilla_fim": 0.1, "syntax_aware_fim": 0.3},
        enable_syntax_aware_fim=False,
    )

    stream_rates, default_rates = _stream_fim_rates(config)

    assert default_rates == {"vanilla_fim": 0.1, "syntax_aware_fim": 0.0}
    assert stream_rates == [
        {"vanilla_fim": 0.1, "syntax_aware_fim": 0.0},
        {"vanilla_fim": 0.5, "syntax_aware_fim": 0.0},
    ]
