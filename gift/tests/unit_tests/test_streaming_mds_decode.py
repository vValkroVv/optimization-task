"""Unit tests for decoding tokenized MDS byte fields (llm-foundry convention)"""

import numpy as np
import pytest
import torch

from torchtitan.datasets.streaming.core.mds_decode import (
    decode_tokenized_mds_sample,
    DTYPE_CODE_TO_NUMPY_DTYPE,
)


@pytest.mark.parametrize("code,np_dtype", sorted(DTYPE_CODE_TO_NUMPY_DTYPE.items()))
def test_decodes_every_dtype_code(code, np_dtype):
    values = [0, 1, 42, 65535]
    sample = {"tokens": np.asarray(values, dtype=np_dtype).tobytes(), "dtype": code}

    tokens = decode_tokenized_mds_sample(sample)

    assert tokens.dtype == torch.int64
    assert tokens.tolist() == values


def test_dtype_code_defaults_to_int64():
    sample = {"tokens": np.asarray([-5, 7], dtype=np.int64).tobytes()}
    assert decode_tokenized_mds_sample(sample).tolist() == [-5, 7]


def test_max_length_truncates():
    sample = {"tokens": np.arange(10, dtype=np.uint32).tobytes(), "dtype": 1}
    tokens = decode_tokenized_mds_sample(sample, max_length=4)
    assert tokens.tolist() == [0, 1, 2, 3]


def test_custom_field_name():
    sample = {"labels": np.asarray([11], dtype=np.int64).tobytes()}
    tokens = decode_tokenized_mds_sample(sample, field="labels")
    assert tokens.tolist() == [11]


def test_missing_field_raises():
    with pytest.raises(ValueError, match="Missing tokenized MDS field 'tokens'"):
        decode_tokenized_mds_sample({"dtype": 1})


def test_unknown_dtype_code_raises():
    with pytest.raises(ValueError, match="Unsupported dtype code 9"):
        decode_tokenized_mds_sample({"tokens": b"", "dtype": 9})
