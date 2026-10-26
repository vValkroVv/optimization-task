"""DataStream resume: no lost/duplicate idx, deterministic order, error unwind."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torchtitan.datasets.kiwi.configs import TextDataStreamConfig
from torchtitan.datasets.kiwi.data_streams import TextDataStream

N_SHARDS = 4
ROWS_PER_SHARD = 5
EPOCH_LEN = N_SHARDS * ROWS_PER_SHARD
N_TAKE = EPOCH_LEN * 2 + 3
BREAK_AT = 7


def _write_shards(root: Path) -> Path:
    paths: list[str] = []
    for shard in range(N_SHARDS):
        rows = []
        for local in range(ROWS_PER_SHARD):
            doc = shard * ROWS_PER_SHARD + local
            tokens = np.array([doc, *range(1, 8)], dtype=np.int64)
            rows.append({"tokens": tokens.tobytes(), "dtype": 2})
        path = root / f"shard_{shard}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        paths.append(str(path))
    listing = root / "shards.txt"
    listing.write_text("\n".join(paths) + "\n")
    return listing


def _stream(
    listing: Path,
    *,
    n_threads: int = 3,
) -> TextDataStream:
    cfg = TextDataStreamConfig(
        name="text",
        data_file=str(listing),
        num_reader_threads=n_threads,
        prefetch_size=32,
    )
    return TextDataStream(cfg, dp_rank=0, dp_world_size=1)


def _mark(sample: dict) -> int:
    return int(sample["text"][0].item())


def _take(
    stream: TextDataStream,
    n: int,
    *,
    start: int = 0,
    skip_idx: set[int] | None = None,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, payload in stream.iter_samples(start=start, skip_idx=skip_idx):
        assert payload is not None
        out.append((idx, _mark(payload)))
        if len(out) >= n:
            break
    return out


def test_data_stream_resume_after_error_and_deterministic_order(tmp_path: Path) -> None:
    listing = _write_shards(tmp_path)

    full_a = _take(_stream(listing), N_TAKE)
    full_b = _take(_stream(listing), N_TAKE)
    assert full_a == full_b
    idxs = [idx for idx, _ in full_a]
    assert idxs == list(range(N_TAKE))

    seen: list[tuple[int, int]] = []
    with pytest.raises(RuntimeError, match="interrupt"):
        for idx, payload in _stream(listing):
            assert payload is not None
            seen.append((idx, _mark(payload)))
            if idx == BREAK_AT:
                raise RuntimeError("interrupt")

    assert [idx for idx, _ in seen] == list(range(BREAK_AT + 1))

    resumed = _take(_stream(listing), N_TAKE - (BREAK_AT + 1), start=BREAK_AT + 1)
    stitched = seen + resumed
    assert stitched == full_a
    stitched_idxs = [idx for idx, _ in stitched]
    assert stitched_idxs == list(range(N_TAKE))
    assert len(set(stitched_idxs)) == N_TAKE
    assert [mark for _, mark in stitched] == [mark for _, mark in full_a]


def test_stream_cursor_compacts_watermark() -> None:
    from torchtitan.datasets.kiwi.kiwi_dataloader import StreamCursor

    cursor = StreamCursor()
    for idx in (0, 1, 2, 3, 6, 8, 9):
        cursor.ack(idx)
    assert cursor.watermark == 4
    assert cursor.skip == {6, 8, 9}


def test_data_stream_skips_acked_indices(tmp_path: Path) -> None:
    listing = _write_shards(tmp_path)
    full = _take(_stream(listing), N_TAKE)
    skip = {6, 8, 9}
    watermark = 4
    resumed = _take(
        _stream(listing),
        N_TAKE - watermark - len(skip),
        start=watermark,
        skip_idx=skip,
    )
    resumed_idxs = [idx for idx, _ in resumed]
    assert 4 in resumed_idxs and 5 in resumed_idxs and 7 in resumed_idxs
    assert not skip.intersection(resumed_idxs)
    eaten = set(range(watermark)) | skip | set(resumed_idxs)
    assert eaten == set(range(N_TAKE))
    by_idx = {idx: mark for idx, mark in full}
    for idx, mark in resumed:
        assert mark == by_idx[idx]


def test_data_stream_small_prefetch_buffer(tmp_path: Path) -> None:
    listing = _write_shards(tmp_path)
    full = _take(_stream(listing), N_TAKE)
    cfg = TextDataStreamConfig(
        name="text",
        data_file=str(listing),
        num_reader_threads=2,
        prefetch_size=1,
    )
    tight = _take(TextDataStream(cfg, dp_rank=0, dp_world_size=1), N_TAKE)
    assert tight == full
