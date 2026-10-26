from __future__ import annotations

import json
import os
import shutil
import struct
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metric_sampler.capture.dump_buffer import DumpEntry
from metric_sampler.capture.layout import Layout
from metric_sampler.contract.catalog import METRICS, metric_by_name
from metric_sampler.contract.metrics import ACTIVATION_COUNTS_WIDTH, INT32_MAX
from metric_sampler.contract.raw_layout import RawRoot
from metric_sampler.island.collector import Collector
from metric_sampler.node.aggregator import NodeAggregator, RankDumpBuffer, RankRing
from sber_tests.unit.island.conftest import FakeSession
from sber_tests.unit.node.conftest import fill_dump_buffer, full_step, named, rank_rings, read_rows, write_step

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")
_SCORES = metric_by_name("router/scores_mean")
_BIAS = metric_by_name("router/expert_bias_before_update")
_DEAD_EXPERTS = metric_by_name("summary/dead_experts")
_MAX_EXPERT_SHARE = metric_by_name("summary/max_expert_share")
# Above this a float32 can no longer hold every integer.
_FLOAT32_EXACT_INTEGER_LIMIT = 2**24
_DUMP_TOKENS = 4
_DUMP_HIDDEN = 3
_DUMP_ENTRIES = (DumpEntry("residual", 0), DumpEntry("final_norm", -1))


def _aggregator(
    rank_rings: list[RankRing],
    output_dir: Path,
    node: str = "n0",
    window_steps: int = 1,
    rank_timeout_s: float = 5.0,
    nodes_in_job: int = 4,
    primary_node: bool = True,
    clock: Callable[[], float] = lambda: 0.0,
    absent_ranks: Sequence[int] = (),
    dump_rings: tuple[RankDumpBuffer, ...] = (),
    capture_interval: int = 1,
) -> NodeAggregator:
    """A node aggregator of attempt `a1`; the clock stands still unless a test moves it."""
    return NodeAggregator(
        rank_rings, output_dir, node, "a1", window_steps, rank_timeout_s, nodes_in_job, primary_node,
        absent_ranks=absent_ranks, clock=clock, dump_rings=dump_rings,
        capture_interval=capture_interval,
    )


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("recover", [False, True])
def test_exact_activation_counts_survive_ring_spool_and_parquet_and_replace_the_moments_counters(
    shm_rings, tmp_path, verbose: bool, recover: bool,
) -> None:
    moments = metric_by_name("act/residual/moments")
    counts = metric_by_name("act/residual/counts")
    layout = Layout([
        (metric, layer, microbatch, width)
        for layer in (0, 2) for microbatch in (0, 1)
        for metric, width in ((counts, ACTIVATION_COUNTS_WIDTH), (moments, 4))
    ])
    rings = [shm_rings(layout, slot_count=1) for _ in range(2)]
    aggregator = _aggregator(rank_rings(*rings), tmp_path, window_steps=10, nodes_in_job=1)
    expected_counts = {}
    expected_moments = {}
    for rank, ring in enumerate(rings):
        values = {}
        for layer in (0, 2):
            for microbatch in (0, 1):
                nonfinite = _FLOAT32_EXACT_INTEGER_LIMIT + 1 + rank * 100 + layer * 10 + microbatch
                finite = _FLOAT32_EXACT_INTEGER_LIMIT + 3 + rank * 100 + layer * 10 + microbatch
                vector = [nonfinite, finite] + [finite - 1, 1] + [0] * 62 + [0] * 63 + [7]
                values[(counts.name, layer, microbatch)] = vector
                values[(moments.name, layer, microbatch)] = [1.25, 4.5, 0.0, 0.0]
                identity = (rank, layer, microbatch)
                expected_counts[identity] = vector
                expected_moments[identity] = [1.25, 4.5, float(np.float32(nonfinite)), float(np.float32(finite))]
        ring.verbose_until_step = 75 if verbose else -1
        write_step(ring, layout, 75, 0, values)
    assert aggregator.poll_once() == []
    node = tmp_path / "attempt=a1" / "node=n0"
    assert len(list((node / "_spool").glob("*.zst"))) == 2
    paths = NodeAggregator.recover(node) if recover else aggregator.flush()

    count_table = pq.ParquetFile(named(paths, "expert_counts.parquet")).read()
    moment_table = pq.ParquetFile(named(paths, "router_vectors.parquet")).read()
    assert count_table.schema.field("values").type == pa.list_(pa.uint32())
    assert moment_table.schema.field("values").type == pa.list_(pa.float32())
    for table, metric, expected in (
        (count_table, counts, expected_counts), (moment_table, moments, expected_moments),
    ):
        rows = table.to_pylist()
        assert len(rows) == 8
        assert {(row["rank"], row["layer"], row["microbatch"]): row["values"] for row in rows} == expected
        assert all(
            (row["metric_id"], row["step"], row["source_metric_id"], row["contributors"], row["expected_contributors"],
             row["identical"], row["verbose"]) == (metric.metric_id, 75, 0, 1, 2, False, verbose)
            for row in rows
        )
    assert not list((node / "_spool").glob("*.zst"))
    assert NodeAggregator.recover(node) == []


def test_ten_step_windows_recover_the_last_five_of_75_steps(shm_rings, tmp_path) -> None:
    layout = Layout([(_COUNTS, 0, microbatch, 2) for microbatch in range(2)])
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    aggregator = _aggregator(rank_rings(*rings), tmp_path, window_steps=10, nodes_in_job=1)
    for step in range(1, 76):
        for rank, ring in enumerate(rings):
            write_step(ring, layout, step, step - 1, {
                (_COUNTS.name, 0, microbatch): [step + rank, microbatch] for microbatch in range(2)
            })
        aggregator.poll_once()
    node_dir = tmp_path / "attempt=a1" / "node=n0"
    assert len(list(node_dir.glob("w*/manifest.json"))) == 7
    assert len(list((node_dir / "_spool").glob("*.zst"))) == 10

    recovered = NodeAggregator.recover(node_dir)

    assert named(recovered, "manifest.json").parent.name == "w00000071-00000075"
    rows = []
    for path in sorted(node_dir.glob("w*/expert_counts.parquet")):
        rows.extend(pq.ParquetFile(path).read().to_pylist())
    assert [(row["step"], row["rank"], row["microbatch"], row["values"]) for row in rows] == [
        (step, rank, microbatch, [step + rank, microbatch])
        for step in range(1, 76) for microbatch in range(2) for rank in range(2)
    ]
    assert not list((node_dir / "_spool").glob("*.zst"))
    assert NodeAggregator.recover(node_dir) == []


def test_restart_after_spool_write_preserves_values_without_counting_the_ring_twice(
    shm_rings, layout, tmp_path,
) -> None:
    ring = shm_rings(layout, slot_count=1)
    aggregator = _aggregator(rank_rings(ring), tmp_path, window_steps=10, nodes_in_job=1)
    values = full_step(1.5, [0, INT32_MAX, 9, 4], [0.1, 0.2, 0.3, 0.4], [0.5] * 4)
    slot = write_step(ring, layout, 75, 0, values, dropped_before=3)
    assert aggregator.poll_once() == []
    ring.mark_ready(slot, 75, 0, 3)

    restarted = _aggregator(rank_rings(ring), tmp_path, window_steps=10, nodes_in_job=1)
    paths = restarted.flush()

    counts = read_rows(paths, "expert_counts.parquet")
    assert [row["values"] for row in counts] == [[0, INT32_MAX, 9, 4]]
    manifest = json.loads(named(paths, "manifest.json").read_text())
    assert manifest["dropped_steps_per_rank"] == {"0": 3}
    assert manifest["steps"] == [75]
    assert ring.ready_slots() == []


def test_a_restarted_node_publishes_the_window_its_predecessor_left_unpublished_with_its_first_poll(
    shm_rings, layout: Layout, tmp_path,
) -> None:
    ring = shm_rings(layout, slot_count=2)
    aggregator = _aggregator(rank_rings(ring), tmp_path, nodes_in_job=1)
    write_step(ring, layout, 1, 0, full_step(3.5, [1, 2, 3, 4], [0.5] * 4, [0.5] * 4))
    node_dir = tmp_path / "attempt=a1" / "node=n0"
    committed = node_dir / "_spool" / "committed.json"
    committed.mkdir()
    with pytest.raises(OSError):
        aggregator.poll_once()
    committed.rmdir()
    write_step(ring, layout, 2, 1, full_step(4.5, [1, 2, 3, 4], [0.5] * 4, [0.5] * 4))

    restarted = _aggregator(rank_rings(ring), tmp_path, nodes_in_job=1)
    written = restarted.poll_once()

    assert [json.loads(path.read_text())["steps"] for path in written if path.name == "manifest.json"] == [[1], [2]]
    assert json.loads(committed.read_text()) == [1, 2]
    assert not (node_dir / "_spool" / "publish.json").exists()
    assert not list((node_dir / "_spool").glob("*.zst"))


@pytest.mark.parametrize("failure_stage", ["spool", "parquet", "manifest", "commit"])
def test_failed_durable_write_keeps_statistics_available_for_retry(shm_rings, layout, tmp_path, failure_stage) -> None:
    ring = shm_rings(layout, slot_count=1)
    aggregator = _aggregator(rank_rings(ring), tmp_path, nodes_in_job=1)
    values = full_step(3.5, [1, 2, 3, 4], [0.1] * 4, [0.5] * 4)
    write_step(ring, layout, 1, 0, values)
    node_dir = tmp_path / "attempt=a1" / "node=n0"
    relative = {
        "spool": "_spool/step=00000001-rank=00000.zst",
        "parquet": "w00000001-00000001/expert_counts.parquet",
        "manifest": "w00000001-00000001/manifest.json",
        "commit": "_spool/committed.json",
    }[failure_stage]
    blocked = node_dir / relative
    blocked.mkdir(parents=True)
    with pytest.raises(OSError):
        aggregator.poll_once()
    assert bool(ring.ready_slots()) is (failure_stage == "spool")
    blocked.rmdir()

    paths = aggregator.flush()

    rows = read_rows(paths, "expert_counts.parquet")
    assert [(row["step"], row["rank"], row["values"]) for row in rows] == [(1, 0, [1, 2, 3, 4])]
    assert not list((node_dir / "_spool").glob("*.zst"))


@pytest.mark.parametrize("spooled_interval", [4, None], ids=["capture interval", "spool before capture intervals"])
def test_a_window_records_the_capture_interval_also_when_recovered(
    shm_rings, layout, tmp_path, spooled_interval,
) -> None:
    ring = shm_rings(layout, slot_count=1)
    aggregator = _aggregator(rank_rings(ring), tmp_path, window_steps=10, nodes_in_job=1, capture_interval=4)
    write_step(ring, layout, 8, 0, full_step())
    aggregator.poll_once()
    node_dir = tmp_path / "attempt=a1" / "node=n0"
    if spooled_interval is None:
        path = node_dir / "_spool" / "metadata.json"
        config = json.loads(path.read_text())
        del config["capture_interval"]
        path.write_text(json.dumps(config))

    paths = NodeAggregator.recover(node_dir)

    manifest = json.loads(named(paths, "manifest.json").read_text())
    assert (manifest["steps"], manifest["capture_interval"]) == ([8], spooled_interval or 1)


def test_recovery_removes_spool_left_after_a_committed_window(shm_rings, layout, tmp_path) -> None:
    ring = shm_rings(layout, slot_count=1)
    aggregator = _aggregator(rank_rings(ring), tmp_path, window_steps=10, nodes_in_job=1)
    write_step(ring, layout, 1, 0, full_step())
    aggregator.poll_once()
    node_dir = tmp_path / "attempt=a1" / "node=n0"
    path = next((node_dir / "_spool").glob("*.zst"))
    payload = path.read_bytes()
    aggregator.flush()
    path.write_bytes(payload)

    assert NodeAggregator.recover(node_dir) == []
    assert not path.exists()
    assert len(list(node_dir.glob("w*/manifest.json"))) == 1


@pytest.mark.parametrize("invalid", ["node", "layout", "payload"])
def test_recovery_rejects_incompatible_or_corrupt_spool(shm_rings, layout, tmp_path, invalid) -> None:
    ring = shm_rings(layout, slot_count=1)
    aggregator = _aggregator(rank_rings(ring), tmp_path, window_steps=10, nodes_in_job=1)
    write_step(ring, layout, 1, 0, full_step())
    aggregator.poll_once()
    node_dir = tmp_path / "attempt=a1" / "node=n0"
    if invalid == "node":
        path = node_dir / "_spool" / "metadata.json"
        config = json.loads(path.read_text())
        config["node"] = "another-node"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="does not match"):
            NodeAggregator.recover(node_dir)
    elif invalid == "layout":
        with pytest.raises(ValueError, match="configuration changed"):
            _aggregator(rank_rings(ring), tmp_path, window_steps=5, nodes_in_job=1)
    else:
        path = next((node_dir / "_spool").glob("*.zst"))
        data = path.read_bytes()
        path.write_bytes(struct.pack("<Q", layout.words + 1) + data[8:])
        with pytest.raises(ValueError, match="invalid payload length"):
            NodeAggregator.recover(node_dir)


def test_recovery_records_missing_ranks_for_a_partial_final_step(shm_rings, layout, tmp_path) -> None:
    rings = [shm_rings(layout, slot_count=1) for _ in range(2)]
    aggregator = _aggregator(rank_rings(*rings), tmp_path, window_steps=10, nodes_in_job=1)
    write_step(rings[0], layout, 75, 0, full_step())
    assert aggregator.poll_once() == []

    paths = NodeAggregator.recover(tmp_path / "attempt=a1" / "node=n0")

    manifest = json.loads(named(paths, "manifest.json").read_text())
    assert manifest["steps"] == [75]
    assert manifest["ranks"] == [0, 1]
    assert manifest["missing_ranks"] == {"75": [1]}


def test_recovery_acknowledges_a_published_window_already_moved_by_the_collector(
    shm_rings, tmp_path, target,
) -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])
    ring = shm_rings(layout, slot_count=1)
    raw_root = RawRoot(tmp_path / "raw")
    output = raw_root.path / "exp" / "run"
    aggregator = _aggregator(rank_rings(ring), output, node="node0", nodes_in_job=1)
    write_step(ring, layout, 75, 0, {(_LOSS.name, -1, -1): [3.5]})
    node = output / "attempt=a1" / "node=node0"
    committed = node / "_spool" / "committed.json"
    committed.mkdir()

    with pytest.raises(OSError):
        aggregator.poll_once()

    window = node / "w00000075-00000075"
    intent = (node / "_spool" / "publish.json").read_bytes()
    assert (window / "manifest.json").read_bytes() == intent
    session = FakeSession(stats={"scalars": (1, 75, 75)})
    collector = Collector(session, target, collected_retention_s=10**9)
    receipts = collector.collect_cycle(raw_root)
    assert len(receipts) == 1 and receipts[0].rows == 1
    assert not window.exists()
    query_count = len(session.queries)
    committed.rmdir()

    recovered = NodeAggregator.recover(node)

    collected = raw_root.path / "_collected" / "exp" / "run" / "attempt=a1" / "node=node0" / window.name
    assert recovered == [collected / "scalars.parquet", collected / "manifest.json"]
    assert (collected / "manifest.json").read_bytes() == intent
    assert json.loads(committed.read_bytes()) == [75]
    assert not window.exists()
    assert not (node / "_spool" / "publish.json").exists()
    assert not list((node / "_spool").glob("*.zst"))
    assert collector.collect_cycle(raw_root) == []
    assert len(session.queries) == query_count
    assert NodeAggregator.recover(node) == []
    assert pq.ParquetFile(collected / "scalars.parquet").read().to_pylist()[0]["values"] == [3.5]


def test_recovery_finishes_an_intent_not_yet_published(shm_rings, tmp_path) -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])
    ring = shm_rings(layout, slot_count=1)
    aggregator = _aggregator(rank_rings(ring), tmp_path, node="node0", nodes_in_job=1)
    write_step(ring, layout, 1, 0, {(_LOSS.name, -1, -1): [7.0]})
    node = tmp_path / "attempt=a1" / "node=node0"
    manifest = node / "w00000001-00000001" / "manifest.json"
    manifest.mkdir(parents=True)
    with pytest.raises(OSError):
        aggregator.poll_once()
    manifest.rmdir()
    intent = (node / "_spool" / "publish.json").read_bytes()

    recovered = NodeAggregator.recover(node)

    assert manifest in recovered
    assert manifest.read_bytes() == intent
    assert pq.ParquetFile(manifest.parent / "scalars.parquet").read().to_pylist()[0]["values"] == [7.0]
    assert json.loads((node / "_spool" / "committed.json").read_bytes()) == [1]


def test_recovery_does_not_republish_an_acknowledged_window_after_retention(shm_rings, tmp_path) -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])
    ring = shm_rings(layout, slot_count=1)
    aggregator = _aggregator(rank_rings(ring), tmp_path, node="node0", nodes_in_job=1)
    write_step(ring, layout, 1, 0, {(_LOSS.name, -1, -1): [7.0]})
    manifest = named(aggregator.poll_once(), "manifest.json")
    node = manifest.parent.parent
    (node / "_spool" / "publish.json").write_bytes(manifest.read_bytes())
    shutil.rmtree(manifest.parent)

    assert NodeAggregator.recover(node) == []

    assert not manifest.parent.exists()
    assert not (node / "_spool" / "publish.json").exists()
    assert json.loads((node / "_spool" / "committed.json").read_bytes()) == [1]


def test_recovery_rejects_a_published_manifest_that_differs_from_the_intent(shm_rings, tmp_path) -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])
    ring = shm_rings(layout, slot_count=1)
    aggregator = _aggregator(rank_rings(ring), tmp_path, node="node0", nodes_in_job=1)
    write_step(ring, layout, 1, 0, {(_LOSS.name, -1, -1): [7.0]})
    node = tmp_path / "attempt=a1" / "node=node0"
    committed = node / "_spool" / "committed.json"
    committed.mkdir()
    with pytest.raises(OSError):
        aggregator.poll_once()
    committed.rmdir()
    manifest = node / "w00000001-00000001" / "manifest.json"
    manifest.write_text("{}")

    with pytest.raises(ValueError, match="differs from its durable intent"):
        NodeAggregator.recover(node)

    assert manifest.read_text() == "{}"
    assert not committed.exists()
    assert (node / "_spool" / "publish.json").exists()
    assert len(list((node / "_spool").glob("*.zst"))) == 1


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
def test_recovery_preserves_primary_node_and_verbose_row_semantics(
    shm_rings, layout, tmp_path, primary, verbose,
) -> None:
    rings = [shm_rings(layout, slot_count=1) for _ in range(2)]
    paths_by_mode = []
    for mode in ("live", "recovery"):
        for rank, ring in enumerate(rings):
            ring.verbose_until_step = 80 if verbose else -1
            write_step(ring, layout, 75, 0, full_step(rank + 1.0, [rank + 1] * 4, [0.1] * 4, [0.5] * 4))
        aggregator = _aggregator(
            rank_rings(*rings), tmp_path / mode, window_steps=10,
            nodes_in_job=1, primary_node=primary,
        )
        aggregator.poll_once()
        paths_by_mode.append(
            aggregator.flush() if mode == "live" else NodeAggregator.recover(tmp_path / mode / "attempt=a1" / "node=n0")
        )
    for family in ("scalars.parquet", "expert_counts.parquet", "router_vectors.parquet"):
        live_rows, recovered_rows = [
            read_rows(paths, family) for paths in paths_by_mode
        ]
        assert live_rows == recovered_rows
        assert all(row["verbose"] is verbose for row in recovered_rows if row["source_metric_id"] == 0)
        if family == "router_vectors.parquet":
            bias_rows = [row for row in recovered_rows if row["metric_id"] == _BIAS.metric_id]
            assert len(bias_rows) == (2 if verbose else 1) if primary else len(bias_rows) == 0


def test_preserves_two_ranks_by_key(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=4)
    ring1 = shm_rings(layout, slot_count=4)
    write_step(ring0, layout, step=1, seq=0, values=full_step(1.0, [1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], [0.5] * 4))
    write_step(ring1, layout, step=1, seq=0, values=full_step(2.0, [10, 20, 30, 40], [0.5, 0.6, 0.7, 0.8], [0.5] * 4))

    aggregator = _aggregator(rank_rings(ring0, ring1), tmp_path)
    written = aggregator.poll_once()

    window_dir = tmp_path / "attempt=a1" / "node=n0" / "w00000001-00000001"
    assert set(written) == {
        window_dir / "expert_counts.parquet",
        window_dir / "router_vectors.parquet",
        window_dir / "scalars.parquet",
        window_dir / "manifest.json",
    }

    expert_rows = pq.read_table(window_dir / "expert_counts.parquet", partitioning=None).to_pylist()
    assert expert_rows == [
        {
            "node": "n0", "step": 1, "rank": rank, "microbatch": -1, "layer": 0, "metric_id": _COUNTS.metric_id,
            "source_metric_id": 0, "contributors": 1, "expected_contributors": 2, "identical": False,
            "verbose": False, "alias_metric_id": 0, "values": values,
        }
        for rank, values in ((0, [1, 2, 3, 4]), (1, [10, 20, 30, 40]))
    ]

    vector_rows = pq.read_table(window_dir / "router_vectors.parquet", partitioning=None).to_pylist()
    scores_rows = [row for row in vector_rows if row["metric_id"] == _SCORES.metric_id]
    bias_row = next(row for row in vector_rows if row["metric_id"] == _BIAS.metric_id)
    assert [(row["rank"], row["values"]) for row in scores_rows] == [
        (0, pytest.approx([0.1, 0.2, 0.3, 0.4])), (1, pytest.approx([0.5, 0.6, 0.7, 0.8])),
    ]
    assert [row["contributors"] for row in scores_rows] == [1, 1]
    assert [row["expected_contributors"] for row in scores_rows] == [2, 2]
    assert bias_row["identical"] is True
    assert bias_row["values"] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert all(row["node"] == "n0" for row in vector_rows)

    scalar_rows = pq.read_table(window_dir / "scalars.parquet", partitioning=None).to_pylist()
    loss_rows = [row for row in scalar_rows if row["metric_id"] == _LOSS.metric_id]
    scalars_by_rank = {row["rank"]: row["values"][0] for row in loss_rows}
    assert scalars_by_rank[0] == pytest.approx(1.0)
    assert scalars_by_rank[1] == pytest.approx(2.0)
    assert all(row["node"] == "n0" for row in scalar_rows)

    dead_by_rank = {row["rank"]: row["values"][0] for row in scalar_rows if row["metric_id"] == _DEAD_EXPERTS.metric_id}
    share_by_rank = {
        row["rank"]: row["values"][0] for row in scalar_rows if row["metric_id"] == _MAX_EXPERT_SHARE.metric_id
    }
    assert dead_by_rank == {0: pytest.approx(0.0), 1: pytest.approx(0.0)}
    assert share_by_rank[0] == pytest.approx(0.4)
    assert share_by_rank[1] == pytest.approx(0.4)

    manifest = json.loads((window_dir / "manifest.json").read_text())
    assert manifest["ranks"] == [0, 1]
    assert manifest["steps"] == [1]
    assert manifest["capture_interval"] == 1
    assert manifest["first_step"] == 1
    assert manifest["last_step"] == 1
    assert manifest["missing_ranks"] == {}
    assert manifest["identical_violations"] == 0
    assert manifest["dropped_steps_per_rank"] == {"0": 0, "1": 0}
    assert manifest["torn_slots_per_rank"] == {"0": 0, "1": 0}
    assert manifest["late_slots_per_rank"] == {"0": 0, "1": 0}
    assert manifest["rows_per_family"] == {"expert_counts": 2, "router_vectors": 3, "scalars": 6}

    metrics_registry = json.loads((tmp_path / "attempt=a1" / "metrics.json").read_text())
    assert len(metrics_registry) == len(METRICS)
    assert metrics_registry[0]["name"] == METRICS[0].name


def test_expected_contributors_reflects_layouts_even_with_a_missing_rank(shm_rings, tmp_path) -> None:
    layout0 = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (_COUNTS, 1, -1, 4)])
    layout1 = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4)])
    ring0 = shm_rings(layout0, slot_count=2)
    ring1 = shm_rings(layout1, slot_count=2)
    clock_box = [0.0]
    aggregator = _aggregator(rank_rings(ring0, ring1), tmp_path, rank_timeout_s=1.0, clock=lambda: clock_box[0])

    write_step(
        ring0, layout0, step=1, seq=0,
        values={
            ("loss/total", -1, -1): [1.0],
            ("router/tokens_per_expert", 0, -1): [1, 2, 3, 4],
            ("router/tokens_per_expert", 1, -1): [5, 6, 7, 8],
        },
    )
    assert aggregator.poll_once() == []

    clock_box[0] = 10.0
    with pytest.raises(TimeoutError, match="waiting for live ranks"):
        aggregator.poll_once()
    written = aggregator.flush()

    expert_rows = read_rows(written, "expert_counts.parquet")
    rows_by_layer = {row["layer"]: row for row in expert_rows}
    assert rows_by_layer[0]["expected_contributors"] == 2
    assert rows_by_layer[0]["contributors"] == 1
    assert rows_by_layer[1]["expected_contributors"] == 1
    assert rows_by_layer[1]["contributors"] == 1

    manifest = json.loads(named(written, "manifest.json").read_text())
    assert manifest["missing_ranks"] == {"1": [1]}


def test_incomplete_step_stays_pending_until_timeout(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=2)
    ring1 = shm_rings(layout, slot_count=2)
    clock_box = [0.0]
    aggregator = _aggregator(rank_rings(ring0, ring1), tmp_path, clock=lambda: clock_box[0])

    write_step(ring0, layout, step=1, seq=0, values=full_step())
    assert aggregator.poll_once() == []

    clock_box[0] = 10.0
    with pytest.raises(TimeoutError, match="waiting for live ranks"):
        aggregator.poll_once()
    written = aggregator.flush()
    assert written

    manifest = json.loads(named(written, "manifest.json").read_text())
    assert manifest["missing_ranks"] == {"1": [1]}
    scalar_rows = read_rows(written, "scalars.parquet")
    loss_rows = [row for row in scalar_rows if row["metric_id"] == _LOSS.metric_id]
    assert len(loss_rows) == 1
    assert loss_rows[0]["rank"] == 0


def test_the_manifest_reports_what_each_rank_dropped_tore_and_delivered_late_and_each_window_starts_over(
    shm_rings, layout: Layout, tmp_path,
) -> None:
    ring0 = shm_rings(layout, slot_count=4)
    ring1 = shm_rings(layout, slot_count=4)
    clock_box = [0.0]
    aggregator = _aggregator(rank_rings(ring0, ring1), tmp_path, rank_timeout_s=1.0, clock=lambda: clock_box[0])
    write_step(ring0, layout, step=1, seq=0, values=full_step(), dropped_before=3)
    write_step(ring1, layout, step=1, seq=0, values=full_step(), trailer=999)
    assert aggregator.poll_once() == []
    clock_box[0] = 10.0
    with pytest.raises(TimeoutError, match="waiting for live ranks"):
        aggregator.poll_once()

    first = json.loads(named(aggregator.flush(), "manifest.json").read_text())

    assert first["steps"] == [1]
    assert first["missing_ranks"] == {"1": [1]}
    assert first["dropped_steps_per_rank"] == {"0": 3, "1": 0}
    assert first["torn_slots_per_rank"] == {"0": 0, "1": 1}
    assert first["late_slots_per_rank"] == {"0": 0, "1": 0}
    write_step(ring1, layout, step=1, seq=1, values=full_step())
    write_step(ring0, layout, step=2, seq=1, values=full_step())
    write_step(ring1, layout, step=2, seq=2, values=full_step())

    second = json.loads(named(aggregator.poll_once(), "manifest.json").read_text())

    assert second["steps"] == [2]
    assert second["dropped_steps_per_rank"] == {"0": 0, "1": 0}
    assert second["torn_slots_per_rank"] == {"0": 0, "1": 0}
    assert second["late_slots_per_rank"] == {"0": 0, "1": 1}


def test_the_manifest_reports_the_job_setup_the_process_was_told_about(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    write_step(ring0, layout, step=4, seq=0, values=full_step())
    aggregator = _aggregator(rank_rings(ring0), tmp_path, nodes_in_job=16, absent_ranks=[2, 1], capture_interval=4)

    manifest = json.loads(named(aggregator.poll_once(), "manifest.json").read_text())

    assert (manifest["node"], manifest["attempt"], manifest["nodes_in_job"]) == ("n0", "a1", 16)
    assert (manifest["ranks"], manifest["absent_ranks"], manifest["capture_interval"]) == ([0], [1, 2], 4)


def test_only_the_primary_node_writes_the_attempt_registries(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    attempt_dir = tmp_path / "attempt=a1"

    _aggregator(rank_rings(ring0), tmp_path / "secondary", primary_node=False)
    _aggregator(rank_rings(ring0), tmp_path, primary_node=True)

    assert sorted(path.name for path in (tmp_path / "secondary" / "attempt=a1").iterdir()) == ["node=n0"]
    assert sorted(path.name for path in attempt_dir.iterdir()) == ["metrics.json", "node=n0", "streams.json"]
    assert len(json.loads((attempt_dir / "metrics.json").read_text())) == len(METRICS)


@pytest.mark.parametrize("stream_names", [("all",), ("all", "code", "web")])
def test_the_streams_registry_is_written_from_the_stream_names_of_the_rank_layouts(
    shm_rings, tmp_path, stream_names,
) -> None:
    layout = Layout([(_LOSS, -1, -1, 1)], stream_names=stream_names)
    rings = [shm_rings(layout, slot_count=1) for _ in range(2)]

    _aggregator(rank_rings(*rings), tmp_path)

    assert json.loads((tmp_path / "attempt=a1" / "streams.json").read_text()) == [
        {"stream_id": index, "name": name} for index, name in enumerate(stream_names)
    ]


def test_window_splits_when_more_steps_arrive_than_window_size(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=8)
    ring1 = shm_rings(layout, slot_count=8)
    for step in range(1, 5):
        write_step(
            ring0, layout, step=step, seq=step - 1,
            values=full_step(float(step), [step] * 4, [0.1 * step] * 4, [0.5] * 4),
        )
        write_step(
            ring1, layout, step=step, seq=step - 1,
            values=full_step(float(step), [step] * 4, [0.1 * step] * 4, [0.5] * 4),
        )

    aggregator = _aggregator(rank_rings(ring0, ring1), tmp_path, window_steps=2)
    written = aggregator.poll_once()

    window_dirs = sorted({path.parent.name for path in written if path.name == "manifest.json"})
    assert window_dirs == ["w00000001-00000002", "w00000003-00000004"]


def test_flush_writes_a_partial_window(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=4)
    ring1 = shm_rings(layout, slot_count=4)
    aggregator = _aggregator(rank_rings(ring0, ring1), tmp_path, window_steps=5)

    for step in (1, 2):
        write_step(
            ring0, layout, step=step, seq=step - 1, values=full_step()
        )
        write_step(
            ring1, layout, step=step, seq=step - 1, values=full_step()
        )

    assert aggregator.poll_once() == []

    written = aggregator.flush()
    window_dir = tmp_path / "attempt=a1" / "node=n0" / "w00000001-00000002"
    assert any(path == window_dir / "manifest.json" for path in written)


def test_flush_with_nothing_pending_returns_empty_list(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=2)
    aggregator = _aggregator(rank_rings(ring0), tmp_path, window_steps=5)

    assert aggregator.flush() == []


def test_run_exits_when_stop_already_requested(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=2)
    ring0.request_stop()
    aggregator = _aggregator(rank_rings(ring0), tmp_path, window_steps=5)

    aggregator.run(poll_interval_s=0.001, rank_pids=[os.getpid()])


def test_run_exits_when_no_rank_process_is_alive(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=2)
    aggregator = _aggregator(rank_rings(ring0), tmp_path, window_steps=5)

    aggregator.run(poll_interval_s=0.001, rank_pids=[999999])


def test_run_stops_via_the_should_stop_callable(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=2)
    aggregator = _aggregator(rank_rings(ring0), tmp_path, window_steps=5)

    aggregator.run(poll_interval_s=0.001, rank_pids=[os.getpid()], should_stop=lambda: True)

    assert ring0.stop_requested is False


def test_run_sleeps_between_polls_until_stopped(shm_rings, layout: Layout, tmp_path, monkeypatch) -> None:
    ring0 = shm_rings(layout, slot_count=2)
    aggregator = _aggregator(rank_rings(ring0), tmp_path, window_steps=5)
    sleep_calls = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        ring0.request_stop()

    monkeypatch.setattr("metric_sampler.node.aggregator.time.sleep", fake_sleep)

    aggregator.run(poll_interval_s=0.02, rank_pids=[os.getpid()])

    assert sleep_calls == [0.02]


def test_mismatched_stream_names_across_ranks_raises_value_error(shm_rings, tmp_path) -> None:
    layout0 = Layout([(_LOSS, -1, -1, 1)], stream_names=("all",))
    layout1 = Layout([(_LOSS, -1, -1, 1)], stream_names=("all", "code"))
    ring0 = shm_rings(layout0, slot_count=1)
    ring1 = shm_rings(layout1, slot_count=1)

    with pytest.raises(ValueError):
        _aggregator(rank_rings(ring0, ring1), tmp_path)


def test_poll_once_collects_ready_dumps_after_metrics(dump_buffers, layout: Layout, shm_rings, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=2)
    write_step(ring0, layout, step=1, seq=0, values=full_step())
    buffer = dump_buffers(_DUMP_ENTRIES, _DUMP_TOKENS, _DUMP_HIDDEN)
    zeros = np.zeros((_DUMP_TOKENS, _DUMP_HIDDEN), dtype=np.float32)
    fill_dump_buffer(
        buffer, step=1, token_ids=[0, 1, 2, 3], loss=[0.0] * 4, stream_ids=[0] * 4,
        entries_values=[(2, zeros), (2, zeros)],
    )

    aggregator = _aggregator(rank_rings(ring0), tmp_path, dump_rings=(RankDumpBuffer(0, buffer),))
    aggregator.poll_once()

    assert (tmp_path / "attempt=a1" / "dumps" / "step=00000001" / "rank=00000.parquet").exists()
