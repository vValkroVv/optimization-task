"""
Unit tests for TextPretrainCollator block_causal positions

Includes the equivalence check against the HF dataset path: the HF dataset
normalizes every window to start at position 0 (_normalize_positions), the
collator restarts at every row start, so the same token stream must produce
identical positions on doc-aligned and mid-document windows alike
"""

import itertools

import pytest
import torch

from torchtitan.datasets.streaming.text.collators import TextPretrainCollator

EOS = 99


def test_positions_restart_after_each_eos():
    collator = TextPretrainCollator(seq_len=5, eos_token_id=EOS)

    inputs, _ = collator(
        [
            {"tokens": torch.tensor([7, 8, EOS, 9, 10])},
            {"tokens": torch.tensor([EOS, 7, 8, EOS, 9])},
        ]
    )

    # a new document starts right after every eos (the eos belongs to its doc)
    assert inputs["positions"].tolist() == [[0, 1, 2, 0, 1], [0, 0, 1, 2, 0]]


@pytest.mark.parametrize(
    "samples,match",
    [
        ([], "received no samples"),
        ([{"other": torch.tensor([1, 2, 3, 4])}], "missing 'tokens'"),
        ([{"tokens": [1, 2, 3, 4]}], "must be a torch.Tensor"),
        ([{"tokens": torch.ones(2, 4, dtype=torch.long)}], "must be 1D"),
    ],
)
def test_invalid_samples_raise(samples, match):
    collator = TextPretrainCollator(seq_len=4, eos_token_id=EOS)
    with pytest.raises(ValueError, match=match):
        collator(samples)


def test_positions_match_hf_dataset_on_all_window_types(monkeypatch):
    """
    Feed the windows produced by the real HF dataset back through the
    collator: positions must be identical (up to int32 vs int64) both for
    windows starting at a document boundary and mid-document
    """
    datasets = pytest.importorskip("datasets")

    from tests.unit_tests.test_dataloader import DummyTokenizer
    from torchtitan.datasets.hf.configs import DatasetConfig
    from torchtitan.datasets.hf.text_datasets import (
        DATASETS,
        HuggingFaceTextDataLoader,
    )

    # short docs give doc-aligned windows; the long one spans several windows
    # and yields mid-document window starts
    texts = ["abc", "0123456789ABCDEFGH", "xy", "hello world", "tail"]
    monkeypatch.setitem(
        DATASETS,
        "collator_equivalence_test",
        DatasetConfig(
            path="unused",
            loader=lambda path: datasets.Dataset.from_dict({"text": texts}),
            sample_processor=lambda sample: sample["text"],
        ),
    )

    tokenizer = DummyTokenizer()
    seq_len = 8
    hf_loader = HuggingFaceTextDataLoader.Config(
        dataset="collator_equivalence_test", infinite=False
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        seq_len=seq_len,
        local_batch_size=1,
    )
    collator = TextPretrainCollator(seq_len=seq_len, eos_token_id=tokenizer.eos_id)

    window_kinds = set()
    for inputs, _ in itertools.islice(iter(hf_loader), 6):
        hf_row, hf_positions = inputs["input"][0], inputs["positions"][0]
        collated, _ = collator([{"tokens": hf_row}])
        assert torch.equal(collated["positions"][0].to(torch.int64), hf_positions)
        is_aligned = int(hf_row[0]) == tokenizer.bos_id
        window_kinds.add("doc_aligned" if is_aligned else "mid_doc")

    assert window_kinds == {"doc_aligned", "mid_doc"}
