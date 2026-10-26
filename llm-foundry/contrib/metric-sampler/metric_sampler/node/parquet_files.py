from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from metric_sampler.contract.raw_layout import write_atomically

# The leaf column of the `values` list, which Parquet encodings are named by.
VALUES_LEAF_COLUMN = "values.list.element"
_COMPRESSION = "zstd"


def write_parquet_atomically(path: Path, table: pa.Table, **write_options: object) -> None:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=_COMPRESSION, **write_options)
    write_atomically(path, sink.getvalue().to_pybytes())
