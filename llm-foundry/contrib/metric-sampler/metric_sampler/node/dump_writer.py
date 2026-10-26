from __future__ import annotations

import numpy as np
import pyarrow as pa

from metric_sampler.capture.dump_buffer import (
    DTYPE_ITEMSIZES,
    DTYPE_NAMES,
    DTYPE_NOT_CAPTURED,
    DumpBuffer,
    decode_dump_entry,
)
from metric_sampler.contract.raw_layout import AttemptDir
from metric_sampler.node.parquet_files import VALUES_LEAF_COLUMN, write_parquet_atomically


def _dump_schema(hidden: int) -> pa.Schema:
    return pa.schema(
        [
            ("step", pa.int64()),
            ("rank", pa.int32()),
            ("microbatch", pa.int16()),
            ("point", pa.string()),
            ("layer", pa.int32()),
            ("token_index", pa.int32()),
            ("token_id", pa.int32()),
            ("loss", pa.float32()),
            ("stream_id", pa.int32()),
            ("dtype", pa.string()),
            ("values", pa.list_(pa.float32(), hidden)),
        ]
    )


class DumpWriter:
    """Writes the activation dumps the ranks have ready in their dump buffers as one Parquet file per rank and step,
    then hands each buffer back to its rank."""

    def __init__(self, attempt_dir: AttemptDir, dump_buffers: dict[int, DumpBuffer]) -> None:
        self._attempt_dir = attempt_dir
        self._dump_buffers = dump_buffers

    def write_ready_dumps(self) -> None:
        for rank, buffer in self._dump_buffers.items():
            if not buffer.is_ready:
                continue
            self._write_dump(rank, buffer)
            buffer.release()

    def _write_dump(self, rank: int, buffer: DumpBuffer) -> None:
        step = buffer.step
        tokens = buffer.tokens
        hidden = buffer.hidden
        dtype_codes = buffer.entry_dtype_codes()
        captured = [
            (index, entry) for index, entry in enumerate(buffer.entries) if dtype_codes[index] != DTYPE_NOT_CAPTURED
        ]
        if not captured:
            return

        token_ids = buffer.token_ids()
        token_loss = buffer.token_loss()
        stream_ids = buffer.stream_ids()

        entry_count = len(captured)
        points = np.array([entry.point for _, entry in captured])
        layers = np.array([entry.layer for _, entry in captured], dtype=np.int32)
        dtype_names = np.array([DTYPE_NAMES[dtype_codes[index]] for index, _ in captured])

        values_blocks = []
        for index, _ in captured:
            code = int(dtype_codes[index])
            itemsize = DTYPE_ITEMSIZES[code]
            raw = buffer.entry_bytes(index)[: tokens * hidden * itemsize]
            values_blocks.append(decode_dump_entry(raw, code, tokens, hidden))
        values_flat = np.concatenate(values_blocks, axis=0).reshape(-1)

        row_count = entry_count * tokens
        table = pa.table(
            {
                "step": np.full(row_count, step, dtype=np.int64),
                "rank": np.full(row_count, rank, dtype=np.int32),
                "microbatch": np.zeros(row_count, dtype=np.int16),
                "point": pa.array(np.repeat(points, tokens), type=pa.string()),
                "layer": np.repeat(layers, tokens),
                "token_index": np.tile(np.arange(tokens, dtype=np.int32), entry_count),
                "token_id": np.tile(token_ids, entry_count),
                "loss": np.tile(token_loss, entry_count).astype(np.float32),
                "stream_id": np.tile(stream_ids, entry_count),
                "dtype": pa.array(np.repeat(dtype_names, tokens), type=pa.string()),
                "values": pa.FixedSizeListArray.from_arrays(pa.array(values_flat, type=pa.float32()), hidden),
            },
            schema=_dump_schema(hidden),
        )
        dump_path = self._attempt_dir.dump_path(step, rank)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_atomically(dump_path, table, use_dictionary=False, use_byte_stream_split=[VALUES_LEAF_COLUMN])
