import itertools
from dataclasses import replace

import pytest

from node_runner.src.inference.engine import EngineLoad, EngineUnit
from node_runner.src.metrics.metrics_manager import (
    EVALUATED,
    DynamicScheduler,
    EngineSlot,
)
from sber_tests.unit.metrics.scheduler_fakes import (
    IDLE,
    FakeEngine,
    FakeOpLog,
    engines,
    task,
    units,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_cooldown")]

LAYOUTS = [
    pytest.param(p, d, pn, dn, jn, id=f"{p}p-{pn}nodes-{d}d-{dn}nodes-judge{jn}")
    for p, d, pn, dn, jn in itertools.product(
        range(1, 6), range(1, 6), range(1, 5), range(1, 5), range(1, 7)
    )
    if p * pn + d * dn <= 24 and (p * pn + d * dn) % jn == 0
]
LAYOUT_ARGS = "prefill,decode,prefill_nodes,decode_nodes,judge_nodes"


@pytest.fixture
def make_pd_scheduler(log):
    def make(prefill, decode, prefill_nodes, decode_nodes, judge_nodes, judges=1):
        split = prefill * prefill_nodes
        nnodes = split + decode * decode_nodes
        prefill_units = [
            replace(u, prefill_instances=1)
            for u in units(
                *(
                    tuple(range(r, r + prefill_nodes))
                    for r in range(0, split, prefill_nodes)
                )
            )
        ]
        decode_units = [
            replace(u, decode_instances=1)
            for u in units(
                *(
                    tuple(range(r, r + decode_nodes))
                    for r in range(split, nnodes, decode_nodes)
                )
            )
        ]
        anchor = EngineUnit(
            ranks=prefill_units[0].ranks + decode_units[0].ranks,
            worker_urls=prefill_units[0].worker_urls + decode_units[0].worker_urls,
            prefill_instances=1,
            decode_instances=1,
        )
        evaluated_units = [anchor] + [
            replace(u, standalone=False) for u in decode_units[1:] + prefill_units[1:]
        ]
        judge_units = units(
            *(tuple(range(r, r + judge_nodes)) for r in range(0, nnodes, judge_nodes))
        )
        evaluated = FakeEngine(log, EVALUATED, evaluated_units)
        judge_engines = [
            FakeEngine(log, f"judge{i}", judge_units) for i in range(judges)
        ]
        tasks = [
            task(log, judge.key, (EVALUATED, judge.key)) for judge in judge_engines
        ]
        return DynamicScheduler(
            FakeOpLog(log), engines(evaluated, *judge_engines), tasks
        )

    return make


def _counts(slot):
    return (
        sum(u.prefill_instances for u in slot.up),
        sum(u.decode_instances for u in slot.up),
    )


def _assert_serving_pair_and_disjoint_ranks(scheduler):
    assert min(_counts(scheduler._slots[EVALUATED])) >= 1
    ranks = [r for slot in scheduler._slots.values() for u in slot.up for r in u.ranks]
    assert len(ranks) == len(set(ranks))


@pytest.mark.parametrize("phase", ["initial", "transfer"])
@pytest.mark.parametrize(
    f"{LAYOUT_ARGS},expected",
    [
        pytest.param(2, 4, 2, 1, 2, (2, 2), id="decode-majority-loses-two-instances"),
        pytest.param(4, 2, 1, 2, 2, (2, 2), id="prefill-majority-loses-two-instances"),
        pytest.param(6, 2, 1, 3, 2, (4, 2), id="unequal-three-node-decode"),
        pytest.param(2, 6, 3, 1, 2, (2, 4), id="unequal-three-node-prefill"),
        pytest.param(6, 2, 4, 4, 1, (5, 2), id="four-node-groups"),
        pytest.param(2, 6, 4, 4, 1, (2, 5), id="four-node-groups-mirrored"),
    ],
)
def test_larger_side_donates_even_when_more_instances_are_needed(
    make_pd_scheduler,
    phase,
    prefill,
    decode,
    prefill_nodes,
    decode_nodes,
    judge_nodes,
    expected,
):
    scheduler = make_pd_scheduler(
        prefill, decode, prefill_nodes, decode_nodes, judge_nodes
    )
    evaluated = scheduler._slots[EVALUATED]

    if phase == "initial":
        scheduler._grow_onto_free_units()
    else:
        evaluated.up = list(evaluated.units)
        scheduler._take_units_of_other_engines()
        scheduler._advance_drains()
        scheduler._advance_drains()
    scheduler._attach_started_units()

    assert _counts(evaluated) == expected
    assert scheduler._slots["judge0"].up
    _assert_serving_pair_and_disjoint_ranks(scheduler)


def test_multiple_judges_fill_one_sacrificed_group(make_pd_scheduler):
    scheduler = make_pd_scheduler(6, 2, 4, 4, 1, judges=3)

    scheduler._grow_onto_free_units()
    scheduler._attach_started_units()

    assert _counts(scheduler._slots[EVALUATED]) == (5, 2)
    judge_ranks = {
        r
        for key, slot in scheduler._slots.items()
        if key != EVALUATED
        for u in slot.up
        for r in u.ranks
    }
    assert judge_ranks == {20, 21, 22, 23}
    assert all(slot.up for slot in scheduler._slots.values())
    _assert_serving_pair_and_disjoint_ranks(scheduler)


@pytest.mark.parametrize("mirrored", [False, True])
def test_each_completed_transfer_recomputes_the_larger_side(
    make_pd_scheduler, mirrored
):
    layout = (2, 6, 3, 1, 2) if mirrored else (6, 2, 1, 3, 2)
    expected = [(2, 4), (2, 2), (1, 2)] if mirrored else [(4, 2), (2, 2), (2, 1)]
    scheduler = make_pd_scheduler(*layout)
    evaluated = scheduler._slots[EVALUATED]
    judge = scheduler._slots["judge0"]
    evaluated.up = list(evaluated.units)

    for counts in expected:
        evaluated.idle_probes = EngineSlot.UNDERLOAD_PROBES
        judge.overloaded_probes = EngineSlot.OVERLOAD_PROBES
        scheduler._take_units_of_other_engines()
        scheduler._advance_drains()
        scheduler._advance_drains()
        scheduler._attach_started_units()
        assert _counts(evaluated) == counts
        _assert_serving_pair_and_disjoint_ranks(scheduler)

    evaluated.idle_probes = EngineSlot.UNDERLOAD_PROBES
    judge.overloaded_probes = EngineSlot.OVERLOAD_PROBES
    scheduler._take_units_of_other_engines()
    assert not scheduler._drains


@pytest.mark.parametrize("judges", [1, 2, 3])
@pytest.mark.parametrize(LAYOUT_ARGS, LAYOUTS)
def test_initial_placement_retains_each_role_without_overlapping_judges(
    make_pd_scheduler, prefill, decode, prefill_nodes, decode_nodes, judge_nodes, judges
):
    scheduler = make_pd_scheduler(
        prefill, decode, prefill_nodes, decode_nodes, judge_nodes, judges
    )

    scheduler._grow_onto_free_units()
    scheduler._attach_started_units()

    _assert_serving_pair_and_disjoint_ranks(scheduler)


@pytest.mark.parametrize(LAYOUT_ARGS, LAYOUTS)
def test_repeated_transfers_preserve_roles_and_follow_the_current_majority(
    make_pd_scheduler, prefill, decode, prefill_nodes, decode_nodes, judge_nodes
):
    scheduler = make_pd_scheduler(
        prefill, decode, prefill_nodes, decode_nodes, judge_nodes
    )
    evaluated = scheduler._slots[EVALUATED]
    judge = scheduler._slots["judge0"]
    evaluated.up = list(evaluated.units)

    for _ in range(len(judge.units) + 1):
        evaluated.idle_probes = EngineSlot.UNDERLOAD_PROBES
        judge.overloaded_probes = EngineSlot.OVERLOAD_PROBES
        before_p, before_d = _counts(evaluated)
        majority_available = False
        for candidate in scheduler._missing(judge):
            donors = scheduler._donors(candidate, judge)
            if donors is None:
                continue
            given = donors.get(EVALUATED, [])
            if before_p > before_d:
                majority_available |= bool(given) and all(
                    u.decode_instances == 0 for u in given
                )
            elif before_d > before_p:
                majority_available |= bool(given) and all(
                    u.prefill_instances == 0 for u in given
                )

        scheduler._take_units_of_other_engines()
        _assert_serving_pair_and_disjoint_ranks(scheduler)
        if not scheduler._drains:
            break
        after_p, after_d = _counts(evaluated)
        if majority_available:
            if before_p > before_d:
                assert after_d == before_d
            else:
                assert after_p == before_p
        scheduler._advance_drains()
        scheduler._advance_drains()
        scheduler._attach_started_units()
        assert not scheduler._drains
        _assert_serving_pair_and_disjoint_ranks(scheduler)
    else:
        pytest.fail("node transfers did not converge")


@pytest.mark.parametrize(
    "load",
    [
        pytest.param(None, id="unknown-load"),
        pytest.param(EngineLoad(1, 0, 0, 0), id="running-request"),
        pytest.param(EngineLoad(0, 1, 0, 0), id="queued-request"),
        pytest.param(EngineLoad(0, 0, 0, 0, kv_in_transfer=1), id="kv-transfer"),
    ],
)
@pytest.mark.parametrize("mirrored", [False, True])
def test_failed_drain_restores_the_pd_groups_and_retry_uses_the_majority(
    make_pd_scheduler, load, mirrored
):
    layout = (2, 6, 3, 1, 2) if mirrored else (6, 2, 1, 3, 2)
    scheduler = make_pd_scheduler(*layout)
    evaluated = scheduler._slots[EVALUATED]
    evaluated.up = list(evaluated.units)
    original = set(evaluated.up)
    evaluated.engine._load = load

    scheduler._take_units_of_other_engines()
    (drain,) = scheduler._drains
    given = set(drain.donors[EVALUATED])
    assert _counts(evaluated) == ((2, 4) if mirrored else (4, 2))
    scheduler._advance_drains()
    assert scheduler._drains == [drain]
    drain.deadline = 0
    scheduler._advance_drains()

    assert set(evaluated.up) == original
    assert _counts(evaluated) == layout[:2]
    assert not scheduler._drains
    assert not scheduler._slots["judge0"].up

    evaluated.engine._load = IDLE
    scheduler._take_units_of_other_engines()
    (retry,) = scheduler._drains
    assert set(retry.donors[EVALUATED]) == given
    scheduler._advance_drains()
    scheduler._advance_drains()
    scheduler._attach_started_units()
    assert _counts(evaluated) == ((2, 4) if mirrored else (4, 2))
    assert scheduler._slots["judge0"].up
    assert not scheduler._drains
    _assert_serving_pair_and_disjoint_ranks(scheduler)
