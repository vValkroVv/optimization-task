from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metric_sampler.cloud.dumps import read_dump

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema(
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
        ("values", pa.list_(pa.float32(), 3)),
    ]
)

# (point, layer, token_index, token_id, loss, stream_id, dtype, values); rows are interleaved
# across points/layers on purpose, since read_dump must group them rather than rely on order.
_ROWS = [
    ("resid", 0, 0, 11, 0.5, 0, "bf16", [1.0, 2.0, 3.0]),
    ("mlp_out", 0, 0, 11, 0.5, 0, "fp32", [10.0, 11.0, 12.0]),
    ("resid", 0, 1, 12, 0.25, 1, "bf16", [4.0, 5.0, 6.0]),
    ("resid", 3, 0, 11, 0.5, 0, "bf16", [7.0, 8.0, 9.0]),
]


def _write_dump(path: Path) -> None:
    table = pa.table(
        {
            "step": [1200] * len(_ROWS),
            "rank": [3] * len(_ROWS),
            "microbatch": [0] * len(_ROWS),
            "point": [row[0] for row in _ROWS],
            "layer": [row[1] for row in _ROWS],
            "token_index": [row[2] for row in _ROWS],
            "token_id": [row[3] for row in _ROWS],
            "loss": [row[4] for row in _ROWS],
            "stream_id": [row[5] for row in _ROWS],
            "dtype": [row[6] for row in _ROWS],
            "values": [row[7] for row in _ROWS],
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd", use_byte_stream_split=["values"])


def test_read_dump_groups_rows_by_point_and_layer_with_exact_values(tmp_path: Path) -> None:
    path = tmp_path / "rank=00003.parquet"
    _write_dump(path)

    dump = read_dump(path)

    assert dump.step == 1200
    assert dump.rank == 3
    assert set(dump.slices) == {("resid", 0), ("resid", 3), ("mlp_out", 0)}

    residual_layer_0 = dump.slices[("resid", 0)]
    assert residual_layer_0.dtype == "bf16"
    assert residual_layer_0.values.dtype == np.float32
    assert residual_layer_0.values.shape == (2, 3)
    np.testing.assert_array_equal(residual_layer_0.token_index, [0, 1])
    np.testing.assert_array_equal(residual_layer_0.token_id, [11, 12])
    np.testing.assert_allclose(residual_layer_0.loss, [0.5, 0.25], rtol=1e-6)
    np.testing.assert_array_equal(residual_layer_0.stream_id, [0, 1])
    np.testing.assert_allclose(residual_layer_0.values, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    residual_layer_3 = dump.slices[("resid", 3)]
    assert residual_layer_3.dtype == "bf16"
    np.testing.assert_array_equal(residual_layer_3.token_index, [0])
    np.testing.assert_allclose(residual_layer_3.values, [[7.0, 8.0, 9.0]])

    mlp_output = dump.slices[("mlp_out", 0)]
    assert mlp_output.dtype == "fp32"
    np.testing.assert_allclose(mlp_output.values, [[10.0, 11.0, 12.0]])
