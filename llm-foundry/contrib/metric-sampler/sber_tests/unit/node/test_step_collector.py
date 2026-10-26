from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.ring import ShmRing
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.node.spool import Spool
from metric_sampler.node.step_collector import PendingStep, StepCollector
from metric_sampler.node.window_counters import WindowCounters
from sber_tests.unit.node.conftest import full_step, write_step

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")
_SPOOL_METADATA = {"node": "n0", "capture_interval": 1}
_TIMEOUT_S = 5.0


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _collector(
    tmp_path: Path,
    rings: list[ShmRing],
    layout: Layout,
    clock: Callable[[], float] = lambda: 0.0,
    rank_count: int | None = None,
) -> tuple[StepCollector, WindowCounters, Spool]:
    """A collector over the rings of ranks 0, 1, ... and a spool of its own under `tmp_path`. The ranks are those
    of the rings unless `rank_count` says how many a node without rings has."""
    layouts = dict.fromkeys(range(len(rings) if rank_count is None else rank_count), layout)
    counters = WindowCounters(layouts)
    spool = Spool(tmp_path / "node", layouts, _SPOOL_METADATA)
    return StepCollector(dict(enumerate(rings)), layouts, spool, counters, clock, _TIMEOUT_S), counters, spool


def _settle(collector: StepCollector, force: bool = False) -> dict[int, PendingStep]:
    settled = {}
    for step, pending in collector.settleable_steps(force):
        settled[step] = pending
        collector.settle(step)
    return settled


def _values(pending: PendingStep, metric_name: str, layout: Layout, layer: int = -1, microbatch: int = -1) -> dict:
    metric_id = layout.slot(metric_name, layer, microbatch).metric.metric_id
    return {rank: slots[(metric_id, layer, microbatch)][1].tolist() for rank, slots in pending.ranks.items()}


def test_a_step_every_rank_delivered_is_settleable_with_each_ranks_words_and_typed_values(
    shm_rings, layout: Layout, tmp_path,
) -> None:
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    collector, _, _ = _collector(tmp_path, rings, layout)
    write_step(rings[0], layout, step=1, seq=0, values=full_step(1.5, counts=[1, 2, 3, 4]))
    write_step(rings[1], layout, step=1, seq=0, values=full_step(2.5, counts=[10, 20, 30, 40]))

    collector.collect_ready_slots()
    settled = _settle(collector)

    assert list(settled) == [1]
    pending = settled[1]
    assert (pending.verbose, collector.missing_ranks(pending)) == (False, [])
    assert _values(pending, "loss/total", layout) == {0: [1.5], 1: [2.5]}
    assert _values(pending, "router/tokens_per_expert", layout, layer=0) == {0: [1, 2, 3, 4], 1: [10, 20, 30, 40]}
    loss_words, loss_typed = pending.ranks[0][(_LOSS.metric_id, -1, -1)]
    counts_words, counts_typed = pending.ranks[0][(_COUNTS.metric_id, 0, -1)]
    assert (loss_words.dtype, loss_typed.dtype, counts_typed.dtype) == (np.int32, np.float32, np.int32)
    assert loss_words.tolist() == np.array([1.5], dtype=np.float32).view(np.int32).tolist()
    assert counts_words.tolist() == [1, 2, 3, 4]
    assert all(ring.ready_slots() == [] for ring in rings)
    assert _settle(collector) == {}


def test_a_step_a_rank_recorded_nothing_in_still_counts_as_delivered_by_that_rank(
    shm_rings, layout: Layout, tmp_path,
) -> None:
    ring = shm_rings(layout, slot_count=1)
    collector, _, _ = _collector(tmp_path, [ring], layout)
    write_step(ring, layout, step=1, seq=0, values={})

    collector.collect_ready_slots()

    assert {step: pending.ranks for step, pending in _settle(collector).items()} == {1: {0: {}}}


def test_a_torn_slot_is_dropped_and_counted_and_leaves_its_rank_missing(shm_rings, layout: Layout, tmp_path) -> None:
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    clock = _Clock()
    collector, counters, _ = _collector(tmp_path, rings, layout, clock)
    write_step(rings[0], layout, step=7, seq=0, values=full_step(), trailer=999)
    write_step(rings[1], layout, step=7, seq=0, values=full_step(2.0))

    collector.collect_ready_slots()

    assert counters.torn_slots_per_rank == {0: 1, 1: 0}
    assert all(ring.ready_slots() == [] for ring in rings)
    assert _settle(collector) == {}
    clock.now = 10.0
    with pytest.raises(TimeoutError, match=r"step 7 is waiting for live ranks \[0\]"):
        _settle(collector)
    settled = _settle(collector, force=True)
    assert list(settled) == [7]
    assert (list(settled[7].ranks), collector.missing_ranks(settled[7])) == ([1], [0])


def test_a_slot_arriving_after_its_step_settled_is_dropped_and_counted_as_late(
    shm_rings, layout: Layout, tmp_path,
) -> None:
    rings = [shm_rings(layout, slot_count=4) for _ in range(2)]
    collector, counters, _ = _collector(tmp_path, rings, layout)
    write_step(rings[0], layout, step=5, seq=0, values=full_step())
    collector.collect_ready_slots()
    assert list(_settle(collector, force=True)) == [5]

    write_step(rings[1], layout, step=5, seq=0, values=full_step(2.0))
    collector.collect_ready_slots()

    assert counters.late_slots_per_rank == {0: 0, 1: 1}
    assert rings[1].ready_slots() == []
    assert _settle(collector, force=True) == {}


def test_a_slot_of_a_rank_that_already_delivered_the_step_is_released_and_not_counted_twice(
    shm_rings, layout: Layout, tmp_path,
) -> None:
    ring = shm_rings(layout, slot_count=2)
    collector, counters, _ = _collector(tmp_path, [ring], layout)
    slot = write_step(ring, layout, step=75, seq=0, values=full_step(), dropped_before=3)
    collector.collect_ready_slots()
    ring.mark_ready(slot, step=75, seq=0, dropped_before=3)

    collector.collect_ready_slots()

    assert counters.dropped_steps_per_rank == {0: 3}
    assert ring.ready_slots() == []
    assert list(_settle(collector)) == [75]


def test_the_steps_a_rank_dropped_before_its_slots_are_counted_per_rank(shm_rings, layout: Layout, tmp_path) -> None:
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    collector, counters, _ = _collector(tmp_path, rings, layout)
    write_step(rings[0], layout, step=1, seq=0, values=full_step(), dropped_before=3)
    write_step(rings[0], layout, step=2, seq=1, values=full_step(), dropped_before=2)
    write_step(rings[1], layout, step=1, seq=0, values=full_step())

    collector.collect_ready_slots()

    assert counters.dropped_steps_per_rank == {0: 5, 1: 0}


def test_pending_steps_keep_their_values_when_ring_slots_are_reused(shm_rings, tmp_path) -> None:
    layout = Layout([(_LOSS, -1, 0, 1)])
    rings = [shm_rings(layout, slot_count=1) for _ in range(2)]
    collector, _, _ = _collector(tmp_path, rings, layout)
    for step in (1, 2):
        write_step(rings[0], layout, step, step - 1, {("loss/total", -1, 0): [float(step)]})
        collector.collect_ready_slots()
        assert _settle(collector) == {}
    write_step(rings[1], layout, 1, 0, {("loss/total", -1, 0): [10.0]})
    collector.collect_ready_slots()
    write_step(rings[1], layout, 2, 1, {("loss/total", -1, 0): [20.0]})
    collector.collect_ready_slots()

    settled = _settle(collector)

    assert {step: _values(pending, "loss/total", layout, microbatch=0) for step, pending in settled.items()} == {
        1: {0: [1.0], 1: [10.0]}, 2: {0: [2.0], 1: [20.0]},
    }


@pytest.mark.parametrize(("verbose_until_step", "verbose"), [(5, True), (3, True), (2, False), (-1, False)])
def test_a_step_is_verbose_when_a_rank_asked_for_it_and_stays_so_across_a_restart(
    shm_rings, layout: Layout, tmp_path, verbose_until_step, verbose,
) -> None:
    ring = shm_rings(layout, slot_count=1)
    ring.verbose_until_step = verbose_until_step
    collector, _, _ = _collector(tmp_path, [ring], layout)
    write_step(ring, layout, step=3, seq=0, values=full_step())
    collector.collect_ready_slots()

    restarted, _, _ = _collector(tmp_path, [], layout, rank_count=1)
    restarted.restore_spooled_slots()

    assert [pending.verbose for pending in _settle(collector).values()] == [verbose]
    assert [pending.verbose for pending in _settle(restarted).values()] == [verbose]


def test_a_restarted_collector_takes_back_the_spooled_slots_and_the_steps_they_dropped(
    shm_rings, layout: Layout, tmp_path,
) -> None:
    rings = [shm_rings(layout, slot_count=1) for _ in range(2)]
    collector, _, _ = _collector(tmp_path, rings, layout)
    write_step(rings[0], layout, step=75, seq=0, values=full_step(1.5, counts=[0, 2**31 - 1, 9, 4]), dropped_before=3)
    collector.collect_ready_slots()

    restarted, counters, _ = _collector(tmp_path, [], layout, rank_count=2)
    restarted.restore_spooled_slots()
    settled = _settle(restarted)

    assert list(settled) == [75]
    assert _values(settled[75], "loss/total", layout) == {0: [1.5]}
    assert _values(settled[75], "router/tokens_per_expert", layout, layer=0) == {0: [0, 2**31 - 1, 9, 4]}
    assert restarted.missing_ranks(settled[75]) == [1]
    assert counters.dropped_steps_per_rank == {0: 3, 1: 0}


def test_the_steps_a_spool_committed_are_settled_so_their_slots_are_late(shm_rings, layout: Layout, tmp_path) -> None:
    ring = shm_rings(layout, slot_count=2)
    collector, counters, spool = _collector(tmp_path, [ring], layout)
    spool.commit([4])
    collector.restore_spooled_slots()
    write_step(ring, layout, step=4, seq=0, values=full_step())
    write_step(ring, layout, step=5, seq=1, values=full_step())

    collector.collect_ready_slots()

    assert counters.late_slots_per_rank == {0: 1}
    assert list(_settle(collector)) == [5]


def test_an_incomplete_step_stays_pending_until_the_rank_timeout_and_is_then_an_error(
    shm_rings, layout: Layout, tmp_path,
) -> None:
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    clock = _Clock()
    collector, _, _ = _collector(tmp_path, rings, layout, clock)
    write_step(rings[0], layout, step=1, seq=0, values=full_step())
    collector.collect_ready_slots()

    assert _settle(collector) == {}
    clock.now = _TIMEOUT_S
    assert _settle(collector) == {}
    clock.now = _TIMEOUT_S + 1.0
    with pytest.raises(TimeoutError, match=r"metric sampler step 1 is waiting for live ranks \[1\]"):
        _settle(collector)

    settled = _settle(collector, force=True)
    assert list(settled) == [1]
    assert collector.missing_ranks(settled[1]) == [1]


def test_the_steps_before_a_timed_out_one_settle_before_the_error(shm_rings, layout: Layout, tmp_path) -> None:
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    clock = _Clock()
    collector, _, _ = _collector(tmp_path, rings, layout, clock)
    for ring in rings:
        write_step(ring, layout, step=1, seq=0, values=full_step())
    write_step(rings[0], layout, step=2, seq=1, values=full_step())
    collector.collect_ready_slots()
    clock.now = _TIMEOUT_S + 1.0
    settled = []

    with pytest.raises(TimeoutError, match="step 2 is waiting"):
        for step, _ in collector.settleable_steps():
            settled.append(step)
            collector.settle(step)

    assert settled == [1]


def test_a_stopped_rank_with_no_ready_slots_is_not_waited_for(shm_rings, layout: Layout, tmp_path) -> None:
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    rings[1].request_stop()
    clock = _Clock()
    collector, _, _ = _collector(tmp_path, rings, layout, clock)
    write_step(rings[0], layout, step=1, seq=0, values=full_step())
    clock.now = 1.0

    collector.collect_ready_slots()
    settled = _settle(collector)

    assert list(settled) == [1]
    assert collector.missing_ranks(settled[1]) == [1]


def test_a_stopped_rank_that_still_has_a_ready_slot_is_waited_for(shm_rings, layout: Layout, tmp_path) -> None:
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    collector, _, _ = _collector(tmp_path, rings, layout)
    write_step(rings[0], layout, step=1, seq=0, values=full_step())
    collector.collect_ready_slots()
    write_step(rings[1], layout, step=1, seq=0, values=full_step())
    rings[1].request_stop()

    assert _settle(collector) == {}

    collector.collect_ready_slots()
    assert list(_settle(collector)) == [1]


def test_a_stopped_rank_that_already_delivered_does_not_delay_the_step(shm_rings, layout: Layout, tmp_path) -> None:
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    collector, _, _ = _collector(tmp_path, rings, layout)
    for ring in rings:
        write_step(ring, layout, step=1, seq=0, values=full_step())
    rings[1].request_stop()

    collector.collect_ready_slots()
    settled = _settle(collector)

    assert list(settled) == [1]
    assert collector.missing_ranks(settled[1]) == []


def test_a_rank_whose_owner_process_died_is_not_waited_for(shm_rings, layout: Layout, tmp_path, monkeypatch) -> None:
    rings = [shm_rings(layout, slot_count=2) for _ in range(2)]
    collector, _, _ = _collector(tmp_path, rings, layout)
    monkeypatch.setattr("metric_sampler.node.step_collector.os.kill", _every_process_is_gone)
    write_step(rings[0], layout, step=1, seq=0, values=full_step())

    collector.collect_ready_slots()
    settled = _settle(collector)

    assert list(settled) == [1]
    assert collector.missing_ranks(settled[1]) == [1]


def test_without_rings_every_rank_missing_from_a_restored_step_counts_as_gone(
    shm_rings, layout: Layout, tmp_path,
) -> None:
    ring = shm_rings(layout, slot_count=1)
    collector, _, _ = _collector(tmp_path, [ring], layout)
    write_step(ring, layout, step=75, seq=0, values=full_step())
    collector.collect_ready_slots()
    layouts = {0: layout, 1: layout}
    spool = Spool(tmp_path / "node", layouts, _SPOOL_METADATA)

    restarted = StepCollector({}, layouts, spool, WindowCounters(layouts), lambda: 0.0, _TIMEOUT_S)
    restarted.restore_spooled_slots()
    settled = _settle(restarted)

    assert list(settled) == [75]
    assert restarted.missing_ranks(settled[75]) == [1]


def test_the_collector_reports_whether_every_ring_requested_stop(shm_rings, layout: Layout, tmp_path) -> None:
    rings = [shm_rings(layout, slot_count=1) for _ in range(2)]
    collector, _, _ = _collector(tmp_path, rings, layout)
    assert not collector.all_ranks_stopped()

    rings[0].request_stop()
    assert not collector.all_ranks_stopped()

    rings[1].request_stop()
    assert collector.all_ranks_stopped()


@pytest.mark.parametrize(
    ("rank_pids", "expected"),
    [([os.getpid()], True), ([999999], False), ([999999, os.getpid()], True), ([], False)],
)
def test_ranks_are_alive_while_any_of_their_processes_is(
    shm_rings, layout: Layout, tmp_path, rank_pids, expected,
) -> None:
    collector, _, _ = _collector(tmp_path, [shm_rings(layout, slot_count=1)], layout)

    assert collector.any_rank_alive(rank_pids) is expected


@pytest.mark.parametrize("alive", [True, False])
def test_a_rank_is_probed_with_signal_zero_at_the_operating_system_process_boundary(
    shm_rings, layout: Layout, tmp_path, monkeypatch, alive,
) -> None:
    # Signal zero is a liveness probe on Linux but a console signal on Windows.
    probes = []

    def probe(pid: int, signal: int) -> None:
        probes.append((pid, signal))
        if not alive:
            raise ProcessLookupError

    monkeypatch.setattr("metric_sampler.node.step_collector.os.kill", probe)
    collector, _, _ = _collector(tmp_path, [shm_rings(layout, slot_count=1)], layout)

    assert collector.any_rank_alive([123456]) is alive
    assert probes == [(123456, 0)]


def _every_process_is_gone(pid: int, signal: int) -> None:
    raise ProcessLookupError
