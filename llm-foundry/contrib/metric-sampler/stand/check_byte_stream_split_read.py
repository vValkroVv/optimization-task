from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_ROWS = 20000
_WIDTH = 192
_ROWS_PER_STEP = 100
_RANKS = 8
_EMPTY_ARRAY_PERIOD = 17
_ROW_GROUP_ROWS = 7000
_DATA_PAGE_BYTES = 64 * 1024
_VALUES_COLUMN_INDEX = 2
_SEED = 0
_COUNT_LIMIT = 10000
_FLOAT_SCALE = 0.01
# Encoding name -> whether the file stores the values in byte streams.
_ENCODINGS = {"plain": False, "split": True}
_STRUCTURES = {
    "uint32": "step UInt32, rank Int16, values Array(UInt32)",
    "float32": "step UInt32, rank Int16, values Array(Float32)",
}
_SUMMARIES = {
    "uint32": "count(), sum(cityHash64(*)), sum(arraySum(arrayMap(x -> toUInt64(x), values)))",
    "float32": "count(), sum(cityHash64(*)), sum(length(values))",
}
_READERS = [
    ("the Arrow reader", "input_format_parquet_use_native_reader_v3 = 0"),
    ("the native reader v3", "input_format_parquet_use_native_reader_v3 = 1"),
]
_UINT32_EDGES = [0, 1, 255, 256, 65535, 65536, 2**31 - 1, 2**31, 2**32 - 1]
_FLOAT32_EDGES = [0.0, -0.0, 1e-45, 3.4028235e38, -3.4028235e38, np.inf, -np.inf, np.nan]


def _table(kind: str, generator: np.random.Generator) -> tuple[pa.Table, int]:
    """The rows and the exact sum of the values (UInt32) or their number (Float32)."""
    if kind == "uint32":
        values = generator.integers(0, _COUNT_LIMIT, size=(_ROWS, _WIDTH)).astype(np.uint32)
        values[:, : len(_UINT32_EDGES)] = _UINT32_EDGES
        value_type = pa.uint32()
    else:
        values = generator.normal(0.0, _FLOAT_SCALE, size=(_ROWS, _WIDTH)).astype(np.float32)
        values[:, : len(_FLOAT32_EDGES)] = _FLOAT32_EDGES
        value_type = pa.float32()
    arrays = [row.tolist() if index % _EMPTY_ARRAY_PERIOD else [] for index, row in enumerate(values)]
    table = pa.table(
        {
            "step": pa.array(np.arange(_ROWS) // _ROWS_PER_STEP, type=pa.uint32()),
            "rank": pa.array(np.arange(_ROWS) % _RANKS, type=pa.int16()),
            "values": pa.array(arrays, type=pa.list_(value_type)),
        }
    )
    kept = values[[index for index in range(_ROWS) if index % _EMPTY_ARRAY_PERIOD]]
    return table, int(kept.astype(np.uint64).sum()) if kind == "uint32" else int(kept.size)


def _check_sql(directory: Path, kind: str, reader: tuple[str, str]) -> str:
    name, setting = reader
    reads = {
        encoding: f"SELECT ({_SUMMARIES[kind]}) FROM "
                  f"file('{directory}/{encoding}_{kind}.parquet', 'Parquet', '{_STRUCTURES[kind]}')"
        for encoding in _ENCODINGS
    }
    return (
        f"SELECT '{kind} with {name}' AS what, ({reads['plain']}) AS plain, ({reads['split']}) AS split, "
        f"plain = split AS identical SETTINGS {setting} FORMAT Vertical;\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write byte-stream-split test files and the SQL that compares their reads."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clickhouse-dir", type=Path, help="where the server sees --out (default: --out)")
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(_SEED)
    checks = []
    for kind in _STRUCTURES:
        table, expected = _table(kind, generator)
        for encoding, split in _ENCODINGS.items():
            path = arguments.out / f"{encoding}_{kind}.parquet"
            pq.write_table(
                table, path, compression="zstd", use_dictionary=False, row_group_size=_ROW_GROUP_ROWS,
                data_page_size=_DATA_PAGE_BYTES,
                use_byte_stream_split=["values.list.element"] if split else False,
            )
            chunk = pq.read_metadata(path).row_group(0).column(_VALUES_COLUMN_INDEX)
            assert ("BYTE_STREAM_SPLIT" in chunk.encodings) is split, path
        print(f"{kind}: {_ROWS} rows, "
              f"expected {'sum of values' if kind == 'uint32' else 'number of values'} {expected}")
        checks.extend(_check_sql(arguments.clickhouse_dir or arguments.out, kind, reader) for reader in _READERS)
    (arguments.out / "check.sql").write_text("".join(checks))
    print(f"wrote {len(_STRUCTURES) * len(_ENCODINGS)} Parquet files and {arguments.out / 'check.sql'}")


if __name__ == "__main__":
    main()
