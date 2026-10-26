"""
Unit tests for StreamingTextDataset: decode, transform plumbing and guards
"""

import pickle

import pytest

pytest.importorskip("streaming")

import torch
from streaming.base import Stream

from tests.unit_tests.streaming_helpers import (
    make_sequential_rows,
    write_tokenized_mds,
)
from torchtitan.datasets.streaming.text.datasets import StreamingTextDataset


class RecordingTransform:
    """Records the kwargs the dataset passes in, returns tokens unchanged"""

    def __init__(self):
        self.calls = []

    def __call__(self, tokens, *, stream_id, idx, epoch):
        self.calls.append((stream_id, idx, epoch))
        return tokens


def _make_dataset(local_dirs, **kwargs):
    kwargs.setdefault("seq_len", 4)
    streams = [Stream(local=str(local_dir)) for local_dir in local_dirs]
    return StreamingTextDataset(streams=streams, **kwargs)


def test_getitem_decodes_and_truncates_to_seq_len(tmp_path):
    write_tokenized_mds(tmp_path, [[1, 2, 3, 4, 5]])
    dataset = _make_dataset([tmp_path], seq_len=3)
    assert torch.equal(dataset[0]["tokens"], torch.tensor([1, 2, 3]))


def test_transforms_receive_stream_id_idx_and_pinned_epoch(tmp_path):
    """FIM reproducibility hinges on this plumbing: the transform must see the
    stream the sample belongs to, its global index and the pinned epoch"""
    for name in ("s0", "s1"):
        write_tokenized_mds(tmp_path / name, make_sequential_rows(4, row_len=4))
    transform = RecordingTransform()
    dataset = _make_dataset(
        [tmp_path / "s0", tmp_path / "s1"],
        transforms=[transform],
        transforms_epoch=5,
    )

    dataset[0]
    dataset[6]  # samples 4..7 belong to the second stream

    assert transform.calls == [(0, 0, 5), (1, 6, 5)]


def test_transform_epoch_follows_consumed_epoch_when_not_pinned(tmp_path):
    write_tokenized_mds(tmp_path, make_sequential_rows(2, row_len=4))
    transform = RecordingTransform()
    dataset = _make_dataset(
        [tmp_path], transforms=[transform], batch_size=1, num_canonical_nodes=1
    )

    def epoch_of_first_sample():
        transform.calls.clear()
        dataset[0]
        return transform.calls[0][2]

    assert epoch_of_first_sample() == 0  # nothing consumed yet
    list(iter(dataset))
    # next_epoch was bumped to 1, but the epoch being consumed is next_epoch - 1
    assert epoch_of_first_sample() == 0
    list(iter(dataset))
    assert epoch_of_first_sample() == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"seq_len": 0}, "seq_len must be positive"),
        ({"transforms_epoch": -1}, "must be non-negative"),
    ],
)
def test_invalid_args_raise(tmp_path, kwargs, match):
    write_tokenized_mds(tmp_path, [[1, 2, 3, 4]])
    with pytest.raises(ValueError, match=match):
        _make_dataset([tmp_path], **kwargs)


def test_custom_token_field(tmp_path):
    write_tokenized_mds(tmp_path, [[7, 8, 9, 10]], token_field="code_tokens")
    dataset = _make_dataset([tmp_path], token_field="code_tokens")
    assert dataset[0]["tokens"].tolist() == [7, 8, 9, 10]


def test_dataset_pickle_roundtrip_for_spawn_workers(tmp_path):
    write_tokenized_mds(tmp_path, [[1, 2, 3, 4]])
    dataset = _make_dataset([tmp_path])
    clone = pickle.loads(pickle.dumps(dataset))
    assert torch.equal(clone[0]["tokens"], dataset[0]["tokens"])


def test_ncn_multiplier_excludes_num_canonical_nodes(tmp_path):
    write_tokenized_mds(tmp_path, [[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="not both"):
        _make_dataset([tmp_path], num_canonical_nodes=1, ncn_multiplier=2)
