from typing import Any

import numpy as np
import torch


DTYPE_CODE_TO_NUMPY_DTYPE = {
    0: np.uint16,
    1: np.uint32,
    2: np.int64,
    3: np.int32,
}


def decode_tokenized_mds_sample(
    sample: dict[str, Any],
    *,
    field: str = "tokens",
    max_length: int | None = None,
) -> torch.Tensor:
    """Decode a tokenized MDS byte field into a torch.long tensor, using the
    llm-foundry dtype-code convention (the 'dtype' column, int64 by default)."""
    if field not in sample:
        raise ValueError(f"Missing tokenized MDS field {field!r}")

    code = int(sample.get("dtype", 2))
    if code not in DTYPE_CODE_TO_NUMPY_DTYPE:
        raise ValueError(
            f"Unsupported dtype code {code} in MDS sample. "
            f"Supported codes are {sorted(DTYPE_CODE_TO_NUMPY_DTYPE)}"
        )

    tokens = np.frombuffer(sample[field], dtype=DTYPE_CODE_TO_NUMPY_DTYPE[code])
    if max_length is not None:
        tokens = tokens[:max_length]
    return torch.from_numpy(tokens.astype(np.int64))
