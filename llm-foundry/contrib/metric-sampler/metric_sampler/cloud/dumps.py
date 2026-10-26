from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class DumpSlice:
    """One (point, layer)'s sampled tokens: `values` is float32 [tokens, hidden_size]."""

    dtype: str
    values: np.ndarray
    token_index: np.ndarray
    token_id: np.ndarray
    loss: np.ndarray
    stream_id: np.ndarray


@dataclass(frozen=True)
class ActivationDump:
    step: int
    rank: int
    slices: dict[tuple[str, int], DumpSlice]


def read_dump(path: Path) -> ActivationDump:
    table = pq.read_table(path)

    step = table.column("step")[0].as_py()
    rank = table.column("rank")[0].as_py()
    points = table.column("point").to_numpy(zero_copy_only=False)
    layers = table.column("layer").to_numpy()
    token_index = table.column("token_index").to_numpy()
    token_id = table.column("token_id").to_numpy()
    loss = table.column("loss").to_numpy()
    stream_id = table.column("stream_id").to_numpy()
    dtype = table.column("dtype").to_numpy(zero_copy_only=False)

    # FixedSizeList -> flat child array -> reshape: a view onto the same buffer, not a copy.
    fixed_size_list = table.column("values").combine_chunks()
    hidden_size = fixed_size_list.type.list_size
    flat_values = fixed_size_list.values.to_numpy(zero_copy_only=False).reshape(-1, hidden_size)

    slices = {}
    for point, layer in sorted(set(zip(points.tolist(), layers.tolist()))):
        mask = (points == point) & (layers == layer)
        slices[(point, layer)] = DumpSlice(
            dtype=str(dtype[mask][0]),
            values=flat_values[mask],
            token_index=token_index[mask],
            token_id=token_id[mask],
            loss=loss[mask],
            stream_id=stream_id[mask],
        )
    return ActivationDump(step=int(step), rank=int(rank), slices=slices)
