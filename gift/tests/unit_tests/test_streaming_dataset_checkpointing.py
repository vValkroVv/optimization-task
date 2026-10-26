"""
Unit tests for streaming dataloader checkpoint resume

The saved state is global and rank-identical ({"dp_rank_0": ...} without a
world_size field), which is what makes resuming on a different dp_world_size
legal. Covers the canonical resume shape (consume N -> state_dict -> fresh
loader -> load_state_dict -> identical continuation), resharding, chained
resume offset arithmetic and the epoch boundary with dataloader workers
"""

import itertools

import pytest

pytest.importorskip("streaming")
pytest.importorskip("tree_sitter")

from tests.unit_tests.streaming_helpers import (
    install_single_process_shm_gather,
    make_sequential_rows,
    make_text_dataloader,
    write_tokenized_mds,
)


def _take(iterator, num_batches):
    """Next num_batches as comparable (input rows, label rows) fingerprints"""
    return [
        (inputs["input"].tolist(), labels.tolist())
        for inputs, labels in itertools.islice(iterator, num_batches)
    ]


def _sample_ids(batches):
    """First token of every row identifies the sample (see make_sequential_rows)"""
    return [row[0] for input_rows, _ in batches for row in input_rows]


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("dp_world_size", [1, 2, 4])
def test_resume_reproduces_next_batches(tmp_path, monkeypatch, shuffle, dp_world_size):
    """
    Canonical mid-epoch resume: a fresh loader fed the saved state must
    reproduce the original continuation bit for bit on every rank
    """
    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(32, row_len=4))

    states = []
    for dp_rank in range(dp_world_size):
        loader = make_text_dataloader(
            tmp_path,
            seq_len=4,
            local_batch_size=2,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            shuffle=shuffle,
        )
        iterator = iter(loader)
        assert len(_take(iterator, 2)) == 2
        states.append(loader.state_dict())
        expected = _take(iterator, 2)
        assert len(expected) == 2

        resumed = make_text_dataloader(
            tmp_path,
            seq_len=4,
            local_batch_size=2,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            shuffle=shuffle,
        )
        resumed.load_state_dict(states[dp_rank])
        assert _take(iter(resumed), 2) == expected

    # the state is global: every rank writes the identical payload
    assert all(state == states[0] for state in states)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("world_from,world_to", [(16, 8), (2, 4)])
def test_resume_with_changed_world_size(
    tmp_path, monkeypatch, shuffle, world_from, world_to
):
    """
    Resharding resume must continue the global sample cursor: nothing is
    consumed twice and nothing is skipped (the HF loader forbids resharding;
    streaming supports it because the state carries no world_size)
    """
    install_single_process_shm_gather(monkeypatch)
    num_rows = 64
    write_tokenized_mds(tmp_path, make_sequential_rows(num_rows, row_len=4))

    consumed = []
    state = None
    for dp_rank in range(world_from):
        loader = make_text_dataloader(
            tmp_path,
            seq_len=4,
            local_batch_size=1,
            dp_rank=dp_rank,
            dp_world_size=world_from,
            shuffle=shuffle,
        )
        consumed += _sample_ids(_take(iter(loader), 2))
        if dp_rank == 0:
            state = loader.state_dict()

    for dp_rank in range(world_to):
        resumed = make_text_dataloader(
            tmp_path,
            seq_len=4,
            local_batch_size=1,
            dp_rank=dp_rank,
            dp_world_size=world_to,
            shuffle=shuffle,
        )
        resumed.load_state_dict(state)
        # drain the rest of the epoch on the new world size
        consumed += _sample_ids(_take(iter(resumed), num_rows))

    # one full epoch exactly once: no duplicates, no gaps
    assert sorted(consumed) == [1000 + i for i in range(num_rows)]


def test_resume_after_resume_advances_the_cursor(tmp_path):
    """
    The second snapshot must count from the restored offset (restored + newly
    yielded samples), so chained checkpoint -> resume -> checkpoint -> resume
    stays aligned with a continuous run
    """
    write_tokenized_mds(tmp_path, make_sequential_rows(16, row_len=4))

    def build():
        return make_text_dataloader(
            tmp_path, seq_len=4, local_batch_size=2, shuffle=True
        )

    reference = _take(iter(build()), 8)

    first = build()
    first_iterator = iter(first)
    assert _take(first_iterator, 2) == reference[:2]
    state_first = first.state_dict()

    second = build()
    second.load_state_dict(state_first)
    second_iterator = iter(second)
    assert _take(second_iterator, 2) == reference[2:4]
    state_second = second.state_dict()
    # 4 restored + 4 newly yielded samples, not a counter reset
    assert state_second["dp_rank_0"]["sample_in_epoch"] == 8

    third = build()
    third.load_state_dict(state_second)
    assert _take(iter(third), 4) == reference[4:8]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_at_epoch_boundary(tmp_path, num_workers):
    """
    A snapshot taken exactly at the epoch end must resume into the next
    epoch, also when batches come through dataloader worker processes
    """
    write_tokenized_mds(tmp_path, make_sequential_rows(8, row_len=4))

    def build():
        return make_text_dataloader(
            tmp_path,
            seq_len=4,
            local_batch_size=2,
            shuffle=True,
            infinite=True,
            num_workers=num_workers,
        )

    reference = _take(iter(build()), 8)
    # py1br reshuffles every epoch, so a wrong epoch after resume is visible
    assert reference[:4] != reference[4:]

    loader = build()
    iterator = iter(loader)
    assert _take(iterator, 4) == reference[:4]
    state = loader.state_dict()

    resumed = build()
    resumed.load_state_dict(state)
    assert _take(iter(resumed), 4) == reference[4:8]
