from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HELPER_DIR = Path(__file__).resolve().parents[1]
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

import prepare_data
from prepare_data import TokenPacker, make_mds_sample, tokenizer_contract
from torchtitan.datasets.streaming.core.mds_decode import decode_tokenized_mds_sample


def test_token_packer_keeps_document_tokens_contiguous_and_rows_nonoverlapping():
    packer = TokenPacker(seq_len=4)

    assert list(packer.append([1, 2])) == []
    first_rows = list(packer.append([3, 4, 5, 6, 7]))
    assert [row.tolist() for row in first_rows] == [[1, 2, 3, 4]]
    assert packer.pending_tokens == 3

    second_rows = list(packer.append([8]))
    assert [row.tolist() for row in second_rows] == [[5, 6, 7, 8]]
    assert packer.pending_tokens == 0


def test_mds_sample_uses_little_endian_uint32_and_dtype_code_one():
    row = np.arange(4096, dtype=np.uint32)

    sample = make_mds_sample(row)
    decoded = decode_tokenized_mds_sample(sample, max_length=4096)

    assert sample["dtype"] == 1
    assert len(sample["tokens"]) == 4096 * 4
    assert sample["tokens"] == row.astype("<u4").tobytes()
    assert decoded.dtype == torch.int64
    np.testing.assert_array_equal(decoded.numpy(), row.astype(np.int64))


def test_mds_sample_rejects_wrong_shape_and_out_of_vocabulary_ids():
    with pytest.raises(ValueError, match="shape"):
        make_mds_sample(np.zeros(4095, dtype=np.uint32))
    row = np.zeros(4096, dtype=np.uint32)
    row[-1] = 128256
    with pytest.raises(ValueError, match="canonical vocabulary"):
        make_mds_sample(row)


def test_loaded_canonical_tokenizer_keeps_single_bos_eos_behavior():
    tokenizer, receipt = tokenizer_contract()

    assert receipt["vocab_size"] == 128256
    assert receipt["special_token_ids"] == {
        "bos": 1,
        "eos": 2,
        "unk": 0,
        "fim_prefix": 128253,
        "fim_middle": 128254,
        "fim_suffix": 128255,
    }
    assert tokenizer.encode("", add_bos=True, add_eos=True) == [1, 2]
    assert tokenizer.hf_adds_bos and tokenizer.hf_adds_eos


def test_source_candidates_merges_multiple_directories_in_stable_order(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "corpus"
    for relative in ("single.jsonl.zst", "crawl-a/02.jsonl.zst", "crawl-b/01.jsonl.zst"):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source shard")
    ignored = source_root / "crawl-a/notes.txt"
    ignored.write_text("not a source shard", encoding="utf-8")

    monkeypatch.setattr(prepare_data, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(
        prepare_data,
        "SOURCE_RECIPES",
        {
            "first": {"files": ("single.jsonl.zst",), "max_files": None},
            "second": {
                "directories": ("crawl-b", "crawl-a"),
                "suffix": ".zst",
                "max_files": None,
            },
        },
    )
    monkeypatch.setattr(prepare_data, "DOMAIN_ORDER", ("first", "second"))

    candidates = prepare_data.source_candidates()

    assert tuple(candidates) == ("first", "second")
    assert candidates["second"] == [
        source_root / "crawl-a/02.jsonl.zst",
        source_root / "crawl-b/01.jsonl.zst",
    ]
