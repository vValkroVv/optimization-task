from typing import Any

import numpy as np
from streaming.base import StreamingDataset

from torchtitan.datasets.streaming.core.mds_decode import decode_tokenized_mds_sample


class StreamingTextDataset(StreamingDataset):
    """Streaming dataset for already-tokenized text MDS samples"""

    def __init__(
        self,
        *,
        seq_len: int,
        token_field: str = "tokens",
        transforms: list[Any] | None = None,
        transforms_epoch: int | None = None,
        **streaming_dataset_kwargs: Any,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if transforms_epoch is not None and transforms_epoch < 0:
            raise ValueError(
                f"transforms_epoch must be non-negative, got {transforms_epoch}"
            )

        super().__init__(**streaming_dataset_kwargs)
        self.seq_len = seq_len
        self.token_field = token_field
        self.transforms = list(transforms or [])
        self.transforms_epoch = transforms_epoch

    def _stream_id(self, sample_idx: int) -> int:
        """Index of the configured stream the sample belongs to"""
        shard_id, _ = self.spanner[sample_idx]
        return int(self.stream_per_shard[shard_id])

    def __getitem__(
        self,
        at: int | list[int] | np.ndarray | slice,
    ) -> Any:
        if not isinstance(at, (int, np.integer)):
            return super().__getitem__(at)

        sample_idx = int(at)
        sample = super().__getitem__(sample_idx)
        tokens = decode_tokenized_mds_sample(
            sample,
            field=self.token_field,
            max_length=self.seq_len,
        )

        if self.transforms:
            if self.transforms_epoch is not None:
                epoch = self.transforms_epoch
            else:
                # the epoch being consumed is next_epoch - 1 (bumped at iteration start)
                epoch = max(int(self.next_epoch) - 1, 0)
            stream_id = self._stream_id(sample_idx)
            for transform in self.transforms:
                tokens = transform(
                    tokens, stream_id=stream_id, idx=sample_idx, epoch=epoch
                )

        return {"tokens": tokens[: self.seq_len]}
