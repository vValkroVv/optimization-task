from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metric_sampler.capture.layout import Layout
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import SCHEMA_VERSION, Family, MetricSpec
from metric_sampler.contract.raw_layout import AttemptDir, RawRoot
from metric_sampler.node.spool import Spool
from metric_sampler.node.step_merger import Row
from metric_sampler.node.window_counters import WindowCounters
from metric_sampler.node.window_writer import WindowWriter
from sber_tests.unit.node.conftest import named

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")
_SCORES = metric_by_name("router/scores_mean")
_ASSIGNMENTS = metric_by_name("dispatcher/assignments_before_capacity")
_BASE_TYPES = {
    "node": pa.string(), "step": pa.uint32(), "rank": pa.int16(), "microbatch": pa.int16(), "layer": pa.int16(),
    "metric_id": pa.uint16(), "source_metric_id": pa.uint16(), "contributors": pa.uint8(),
    "expected_contributors": pa.uint8(), "identical": pa.bool_(), "verbose": pa.bool_(),
}
_FLOAT_TYPES = {**_BASE_TYPES, "values": pa.list_(pa.float32())}
_EXPERT_COUNTS_TYPES = {**_BASE_TYPES, "alias_metric_id": pa.uint16(), "values": pa.list_(pa.uint32())}


class _Setup:
    """A writer of node `n0` in attempt `a1` of run `exp/run`, with the spool and the counters it works with."""

    def __init__(
        self, tmp_path: Path, ranks: list[int] | None = None, absent_ranks: list[int] | None = None,
        nodes_in_job: int = 4, capture_interval: int = 1,
    ) -> None:
        self._tmp_path = tmp_path
        self.raw_root = RawRoot(tmp_path / "raw")
        output_dir = self.raw_root.path / "exp" / "run"
        self.attempt_dir = AttemptDir(path=output_dir / "attempt=a1", exp="exp", run="run", attempt="a1")
        self.node_directory = self.attempt_dir.path / "node=n0"
        self.ranks = [0, 1] if ranks is None else ranks
        self.absent_ranks = [] if absent_ranks is None else absent_ranks
        self.nodes_in_job = nodes_in_job
        self.capture_interval = capture_interval
        self.layouts = {rank: Layout([(_LOSS, -1, -1, 1)]) for rank in self.ranks}
        self.counters = WindowCounters(self.ranks)
        self.spool = Spool(self.node_directory, self.layouts, {"node": "n0", "capture_interval": capture_interval})
        self.writer = WindowWriter(
            "n0", self.node_directory, self.attempt_dir, self.raw_root, self.nodes_in_job, self.ranks,
            self.absent_ranks, self.capture_interval, self.spool, self.counters,
        )

    def restart(self) -> _Setup:
        """A new writer over the spool this one left behind, as a restarted node process has."""
        return _Setup(self._tmp_path, self.ranks, self.absent_ranks, self.nodes_in_job, self.capture_interval)

    def window(self, first_step: int, last_step: int) -> Path:
        return self.node_directory / f"w{first_step:08d}-{last_step:08d}"

    def collected_window(self, first_step: int, last_step: int) -> Path:
        return (
            self.raw_root.path / "_collected" / "exp" / "run" / "attempt=a1" / "node=n0"
            / f"w{first_step:08d}-{last_step:08d}"
        )


def _row(
    step: int, metric: MetricSpec, values: Sequence[float], rank: int = 0, layer: int = -1, microbatch: int = -1,
    source_metric_id: int = 0, contributors: int = 1, expected_contributors: int = 1, identical: bool = False,
    verbose: bool = False, alias_metric_id: int = 0,
) -> Row:
    dtype = np.uint32 if metric.family is Family.EXPERT_COUNTS else np.float32
    return Row(
        step, rank, microbatch, layer, metric.metric_id, source_metric_id, contributors, expected_contributors,
        identical, verbose, np.array(values, dtype=dtype), alias_metric_id,
    )


def _add_loss_steps(setup: _Setup, steps: Sequence[int]) -> None:
    for step in steps:
        setup.writer.add_step(step, [_row(step, _LOSS, [float(step)])], [])


def _persist_slots(setup: _Setup, steps: Sequence[int]) -> None:
    """The spooled slots of rank 0, which the writer discards once their steps are published."""
    layout = setup.layouts[0]
    payload = np.zeros(layout.words + 1 + 1, dtype=np.int32)
    for step in steps:
        setup.spool.persist_slot(0, step, 0, False, layout, payload)


def _spooled_steps(setup: _Setup) -> list[str]:
    return sorted(path.name for path in (setup.node_directory / "_spool").glob("*.zst"))


def _manifest(paths: list[Path]) -> dict:
    return json.loads(named(paths, "manifest.json").read_text())


def _block(path: Path) -> Path:
    """Makes `path` a directory, which no file can replace until it is removed."""
    path.mkdir(parents=True)
    return path


def test_a_window_holds_one_file_per_family_and_a_manifest_describing_it(tmp_path) -> None:
    setup = _Setup(tmp_path, ranks=[0, 1, 2], absent_ranks=[3], nodes_in_job=16, capture_interval=4)
    setup.writer.add_step(
        4,
        [_row(4, _SCORES, [0.5, 1.5], rank=1, layer=2), _row(4, _LOSS, [2.5]), _row(4, _COUNTS, [3, 4], layer=2)],
        [],
    )
    setup.writer.add_step(8, [_row(8, _LOSS, [3.5], rank=2)], [1, 2])
    setup.counters.dropped_steps_per_rank[0] = 3
    setup.counters.torn_slots_per_rank[1] = 1
    setup.counters.late_slots_per_rank[2] = 2
    setup.counters.identical_violations = 5

    written = setup.writer.write_window(2)

    window = setup.window(4, 8)
    assert sorted(written) == [
        window / "expert_counts.parquet", window / "manifest.json", window / "router_vectors.parquet",
        window / "scalars.parquet",
    ]
    assert written[-1] == window / "manifest.json"
    manifest = json.loads((window / "manifest.json").read_text())
    assert datetime.fromisoformat(manifest.pop("written_at")).tzinfo is not None
    assert manifest == {
        "schema_version": SCHEMA_VERSION,
        "attempt": "a1",
        "node": "n0",
        "nodes_in_job": 16,
        "ranks": [0, 1, 2],
        "absent_ranks": [3],
        "first_step": 4,
        "last_step": 8,
        "steps": [4, 8],
        "capture_interval": 4,
        "rows_per_family": {"scalars": 2, "expert_counts": 1, "router_vectors": 1},
        "missing_ranks": {"8": [1, 2]},
        "dropped_steps_per_rank": {"0": 3, "1": 0, "2": 0},
        "torn_slots_per_rank": {"0": 0, "1": 1, "2": 0},
        "late_slots_per_rank": {"0": 0, "1": 0, "2": 2},
        "identical_violations": 5,
    }


def test_writing_a_window_starts_the_counters_over_and_keeps_the_steps_it_did_not_take(tmp_path) -> None:
    setup = _Setup(tmp_path)
    _add_loss_steps(setup, [1, 2, 3])
    _persist_slots(setup, [1, 2, 3])
    setup.counters.dropped_steps_per_rank[0] = 3
    setup.counters.identical_violations = 1

    written = setup.writer.write_window(2)

    assert _manifest(written)["steps"] == [1, 2]
    assert setup.writer.buffered_step_count == 1
    assert (setup.counters.dropped_steps_per_rank, setup.counters.identical_violations) == ({0: 0, 1: 0}, 0)
    assert _spooled_steps(setup) == ["step=00000003-rank=00000.zst"]
    assert setup.spool.committed_steps == {1, 2}
    assert not setup.spool.has_intent()

    written = setup.writer.write_window(1)

    assert _manifest(written)["steps"] == [3]
    assert setup.writer.buffered_step_count == 0
    assert _spooled_steps(setup) == []
    assert setup.spool.committed_steps == {1, 2, 3}


def test_a_window_takes_the_oldest_steps_first_and_sorts_every_family_file_by_its_keys(tmp_path) -> None:
    setup = _Setup(tmp_path)
    setup.writer.add_step(
        2,
        [_row(2, _LOSS, [0.5], rank=1), _row(2, _LOSS, [1.5], rank=0), _row(2, _COUNTS, [1], layer=1, microbatch=0)],
        [],
    )
    setup.writer.add_step(
        1,
        [
            _row(1, _COUNTS, [2], layer=1, microbatch=1), _row(1, _COUNTS, [3], layer=0, microbatch=1),
            _row(1, _COUNTS, [4], layer=1, microbatch=0, rank=1), _row(1, _COUNTS, [5], layer=1, microbatch=0),
            _row(1, _COUNTS, [6], layer=1, microbatch=0, source_metric_id=_SCORES.metric_id),
        ],
        [],
    )

    written = setup.writer.write_window(2)

    rows = pq.read_table(named(written, "expert_counts.parquet"), partitioning=None).to_pylist()
    assert [
        (row["step"], row["source_metric_id"], row["layer"], row["microbatch"], row["rank"], row["values"])
        for row in rows
    ] == [
        (1, 0, 0, 1, 0, [3]), (1, 0, 1, 0, 0, [5]), (1, 0, 1, 0, 1, [4]), (1, 0, 1, 1, 0, [2]),
        (1, _SCORES.metric_id, 1, 0, 0, [6]),
        (2, 0, 1, 0, 0, [1]),
    ]
    assert _manifest(written)["steps"] == [1, 2]
    scalar_rows = pq.read_table(named(written, "scalars.parquet"), partitioning=None).to_pylist()
    assert [row["rank"] for row in scalar_rows] == [0, 1]


def test_every_family_file_has_the_contract_columns_values_and_encodings(tmp_path) -> None:
    setup = _Setup(tmp_path)
    setup.writer.add_step(
        3,
        [
            _row(3, _COUNTS, [1, 2, 3, 4], rank=2, layer=5, expected_contributors=2),
            _row(3, _COUNTS, list(range(0, 768, 3)), rank=-1, layer=6, contributors=2, expected_contributors=2),
            _row(3, _ASSIGNMENTS, [], rank=2, layer=5, expected_contributors=2, alias_metric_id=_COUNTS.metric_id),
            _row(3, _SCORES, [0.5, -1.5, 2.25], expected_contributors=1),
            _row(3, _SCORES, np.linspace(-1, 1, 64).tolist(), rank=1, identical=True, verbose=True),
            _row(3, _LOSS, [0.75]),
        ],
        [],
    )

    written = setup.writer.write_window(1)

    for name, types, family in (
        ("expert_counts.parquet", _EXPERT_COUNTS_TYPES, Family.EXPERT_COUNTS),
        ("router_vectors.parquet", _FLOAT_TYPES, Family.ROUTER_VECTORS),
        ("scalars.parquet", _FLOAT_TYPES, Family.SCALARS),
    ):
        path = named(written, name)
        table = pq.read_table(path, partitioning=None)
        assert dict(zip(table.schema.names, table.schema.types)) == types
        assert list(table.schema.names) == list(types)
        assert set(table.column("node").to_pylist()) == {"n0"}
        metadata = pq.ParquetFile(path).metadata
        values = next(
            metadata.row_group(0).column(i) for i in range(metadata.num_columns)
            if metadata.row_group(0).column(i).path_in_schema.startswith("values.")
        )
        assert not values.has_dictionary_page
        assert ("BYTE_STREAM_SPLIT" in values.encodings) is (family is not Family.EXPERT_COUNTS)

    expert_rows = pq.read_table(named(written, "expert_counts.parquet"), partitioning=None).to_pylist()
    assert [(row["rank"], row["layer"], row["alias_metric_id"], row["values"]) for row in expert_rows] == [
        (2, 5, 0, [1, 2, 3, 4]),
        (-1, 6, 0, list(range(0, 768, 3))),
        (2, 5, _COUNTS.metric_id, []),
    ]
    vector_rows = pq.read_table(named(written, "router_vectors.parquet"), partitioning=None).to_pylist()
    assert [(row["rank"], row["identical"], row["verbose"], len(row["values"])) for row in vector_rows] == [
        (0, False, False, 3), (1, True, True, 64),
    ]
    assert vector_rows[0]["values"] == [0.5, -1.5, 2.25]
    assert vector_rows[1]["values"] == np.linspace(-1, 1, 64, dtype=np.float32).tolist()


@pytest.mark.parametrize("failure_stage", ["family file", "manifest", "commit"])
def test_a_publication_that_failed_is_retried_with_every_buffered_step_intact(tmp_path, failure_stage) -> None:
    setup = _Setup(tmp_path)
    setup.writer.add_step(1, [_row(1, _COUNTS, [1, 2, 3, 4])], [])
    _persist_slots(setup, [1])
    blocked = _block(
        {
            "family file": setup.window(1, 1) / "expert_counts.parquet",
            "manifest": setup.window(1, 1) / "manifest.json",
            "commit": setup.node_directory / "_spool" / "committed.json",
        }[failure_stage]
    )
    with pytest.raises(OSError):
        setup.writer.write_window(1)
    assert setup.writer.buffered_step_count == 1
    assert _spooled_steps(setup) == ["step=00000001-rank=00000.zst"]
    blocked.rmdir()

    written = setup.writer.write_window(1)

    rows = pq.read_table(named(written, "expert_counts.parquet"), partitioning=None).to_pylist()
    assert [(row["step"], row["rank"], row["values"]) for row in rows] == [(1, 0, [1, 2, 3, 4])]
    assert setup.writer.buffered_step_count == 0
    assert _spooled_steps(setup) == []
    assert setup.spool.committed_steps == {1}
    assert not setup.spool.has_intent()


def test_a_restarted_writer_publishes_the_window_its_predecessor_made_durable_but_did_not_publish(tmp_path) -> None:
    setup = _Setup(tmp_path)
    _add_loss_steps(setup, [1])
    blocked = _block(setup.window(1, 1) / "manifest.json")
    with pytest.raises(OSError):
        setup.writer.write_window(1)
    blocked.rmdir()
    intent = setup.spool.read_intent()

    restarted = setup.restart()
    assert restarted.spool.has_intent()
    steps, written = restarted.writer.finish_publication()

    assert steps == [1]
    assert written == [setup.window(1, 1) / "scalars.parquet", setup.window(1, 1) / "manifest.json"]
    assert (setup.window(1, 1) / "manifest.json").read_bytes() == intent
    assert restarted.spool.committed_steps == {1}
    assert not restarted.spool.has_intent()


def test_a_window_the_collector_already_moved_is_acknowledged_without_publishing_it_again(tmp_path) -> None:
    setup = _Setup(tmp_path)
    _add_loss_steps(setup, [1])
    blocked = _block(setup.node_directory / "_spool" / "committed.json")
    with pytest.raises(OSError):
        setup.writer.write_window(1)
    blocked.rmdir()
    collected = setup.collected_window(1, 1)
    collected.parent.mkdir(parents=True)
    shutil.move(setup.window(1, 1), collected)

    steps, written = setup.restart().writer.finish_publication()

    assert steps == [1]
    assert written == [collected / "scalars.parquet", collected / "manifest.json"]
    assert not setup.window(1, 1).exists()
    assert json.loads((setup.node_directory / "_spool" / "committed.json").read_bytes()) == [1]


def test_a_window_already_committed_is_not_published_again_when_its_intent_is_found(tmp_path) -> None:
    setup = _Setup(tmp_path)
    _add_loss_steps(setup, [1])
    manifest_path = named(setup.writer.write_window(1), "manifest.json")
    setup.spool.write_intent(manifest_path.read_bytes())
    shutil.rmtree(setup.window(1, 1))

    steps, written = setup.restart().writer.finish_publication()

    assert (steps, written) == ([1], [])
    assert not setup.window(1, 1).exists()
    assert not setup.spool.has_intent()


def test_a_published_manifest_that_differs_from_the_intent_is_never_overwritten(tmp_path) -> None:
    setup = _Setup(tmp_path)
    _add_loss_steps(setup, [1])
    _persist_slots(setup, [1])
    blocked = _block(setup.node_directory / "_spool" / "committed.json")
    with pytest.raises(OSError):
        setup.writer.write_window(1)
    blocked.rmdir()
    (setup.window(1, 1) / "manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="differs from its durable intent"):
        setup.restart().writer.finish_publication()

    assert (setup.window(1, 1) / "manifest.json").read_text() == "{}"
    assert setup.spool.has_intent()
    assert setup.spool.committed_steps == set()
    assert _spooled_steps(setup) == ["step=00000001-rank=00000.zst"]


def test_a_manifest_is_not_published_before_every_family_file_it_names_exists(tmp_path) -> None:
    setup = _Setup(tmp_path)
    _add_loss_steps(setup, [1])
    blocked = _block(setup.window(1, 1) / "manifest.json")
    with pytest.raises(OSError):
        setup.writer.write_window(1)
    blocked.rmdir()
    (setup.window(1, 1) / "scalars.parquet").unlink()

    with pytest.raises(FileNotFoundError):
        setup.restart().writer.finish_publication()

    assert not (setup.window(1, 1) / "manifest.json").exists()
    assert setup.spool.has_intent()
