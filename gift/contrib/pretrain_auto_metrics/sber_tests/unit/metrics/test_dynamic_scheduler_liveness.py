import itertools
from dataclasses import replace
import time

import pytest

from node_runner.src.metrics.metrics_manager import EVALUATED, DynamicScheduler
from sber_tests.unit.metrics.scheduler_fakes import (
    BUSY,
    IDLE,
    NODELESS,
    OVERLOADED,
    SERVING,
    FakeEngine,
    FakeOpLog,
    assert_clean_finish,
    assert_in_order,
    engines,
    nodes,
    run_scheduler,
    set_cooldown,
    task,
    units,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fast_clock")]

HANG_TIMEOUT = 1.5


@pytest.mark.parametrize(
    "judge_units",
    [
        pytest.param(nodes(1), id="single-node"),
        pytest.param(
            [*units((1, 2)), *units((3,), standalone=False)], id="pd-with-extra"
        ),
    ],
)
def test_opposite_judge_orders_share_nodes_without_hanging(log, judge_units):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0)),
        FakeEngine(log, "judge-a", judge_units, tail=judge_units),
        FakeEngine(log, "judge-b", nodes(1)),
    )
    first_a = task(log, "first@a", (EVALUATED, "judge-a"))
    first_b = task(log, "first@b", (EVALUATED, "judge-b"), after=first_a)
    second_b = task(log, "second@b", (EVALUATED, "judge-b"))
    second_a = task(log, "second@a", (EVALUATED, "judge-a"), after=second_b)
    tasks = [
        first_a,
        first_b,
        task(log, "first@a-again", (EVALUATED, "judge-a"), after=first_b),
        second_b,
        second_a,
        task(log, "second@b-again", (EVALUATED, "judge-b"), after=second_a),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks, timeout=HANG_TIMEOUT)

    assert all(t.state == "done" for t in tasks)
    assert_in_order(log, ("end", "first@a"), ("start", "first@b"))
    assert_in_order(log, ("end", "second@b"), ("start", "second@a"))
    assert_clean_finish(log)


def _ups_with_rank(events: list[tuple], rank: int) -> list[tuple]:
    return [
        event
        for event in events
        if event[0] == "up" and any(rank in unit for unit in event[2])
    ]


@pytest.mark.parametrize(
    "failed_key, failed_rank, expected_started",
    [
        pytest.param("judge", 3, ["lm_eval"], id="judge-first-unit"),
        pytest.param(EVALUATED, 1, ["tau2@judge", "lm_eval"], id="evaluated-unit"),
    ],
)
def test_failed_unit_is_stopped_once_and_its_node_is_never_used_again(
    log, failed_key, failed_rank, expected_started
):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    tasks = [
        task(log, "tau2@judge", (EVALUATED, "judge")),
        task(log, "lm_eval", (EVALUATED,), seconds=0.3),
    ]
    failed = ("up", failed_key, (failed_rank,))

    run_scheduler(FakeOpLog(log, fail={failed}), fakes, tasks)

    assert sorted(log.started()) == sorted(expected_started)
    first_up = _ups_with_rank(log.events, failed_rank)[0]
    ops = log.of("up", "down")
    after_failure = ops[ops.index(first_up) + 1 :]
    ops_on_failed_node = [
        op for op in after_failure if any(failed_rank in unit for unit in op[2])
    ]
    assert ops_on_failed_node == [("down", failed_key, ((failed_rank,),))]
    assert ("attach", failed_key, ((failed_rank,),)) not in log.events
    assert_clean_finish(log)


def test_judge_that_never_starts_is_tried_once_and_skips_only_its_tasks(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1)),
        FakeEngine(log, "judge", nodes(0, 1)),
    )
    tasks = [
        task(log, "tau2@judge", (EVALUATED, "judge")),
        task(log, "bfcl", (EVALUATED,)),
    ]
    fail = {("up", "judge", (0,)), ("up", "judge", (1,))}

    run_scheduler(FakeOpLog(log, fail=fail), fakes, tasks)

    assert log.started() == ["bfcl"]
    assert log.of("up", key="judge") == [("up", "judge", ((1,),))]
    assert_clean_finish(log)


@pytest.mark.parametrize(
    "evaluated_units", [NODELESS, nodes(0, 1, 2)], ids=["nodeless", "on-nodes"]
)
def test_evaluated_failing_everywhere_skips_all_its_tasks_and_finishes(
    log, evaluated_units
):
    fakes = engines(
        FakeEngine(log, EVALUATED, evaluated_units),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    tasks = [
        task(log, "lm_eval", (EVALUATED,)),
        task(log, "tau2@judge", (EVALUATED, "judge")),
    ]
    fail = {("up", EVALUATED, unit.ranks) for unit in evaluated_units}

    run_scheduler(FakeOpLog(log, fail=fail), fakes, tasks)

    assert log.started() == []
    assert_clean_finish(log)


def test_stage_chained_after_a_skipped_stage_is_skipped_too(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1)),
        FakeEngine(log, "judge", nodes(0, 1, 2)),
    )
    dredd = task(log, "dredd", (EVALUATED,))
    tasks = [dredd, task(log, "dredd@judge", ("judge",), after=dredd)]
    fail = {("up", EVALUATED, (0,)), ("up", EVALUATED, (1,))}

    run_scheduler(FakeOpLog(log, fail=fail), fakes, tasks)

    assert log.started() == []
    assert log.of("up", key="judge") == []


def test_failed_down_is_not_retried_and_the_scheduler_finishes(log):
    fakes = engines(FakeEngine(log, EVALUATED, nodes(0, 1)))
    fail = {("down", EVALUATED, (0,))}

    run_scheduler(FakeOpLog(log, fail=fail), fakes, [task(log, "lm", (EVALUATED,))])

    assert log.of("down") == [("down", EVALUATED, ((0,), (1,)))]
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_node_whose_engine_did_not_stop_is_never_reused(log):
    fakes, tasks = _rebalance(log, IDLE)

    run_scheduler(FakeOpLog(log, fail={("down", EVALUATED, (2,))}), fakes, tasks)

    assert ("down", EVALUATED, ((2,),)) in log.events
    assert ("up", "judge", ((2,),)) not in log.events
    assert_clean_finish(log)


def test_judge_keeps_growing_when_every_evaluated_unit_is_quarantined(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0)),
        FakeEngine(log, "judge", nodes(1, 2)),
    )
    warm = task(log, "warm", ("judge",), seconds=0.5)
    tasks = [
        warm,
        task(log, "generate", (EVALUATED,)),
        task(log, "after_warm", (EVALUATED,), after=warm),
    ]

    run_scheduler(FakeOpLog(log, fail={("up", EVALUATED, (0,))}), fakes, tasks)

    assert log.started() == ["warm"]
    assert_clean_finish(log)


def test_pd_anchor_failure_stops_the_extras_and_leaves_their_nodes_usable(log):
    pd_units = [*units((0, 1)), *units((2,), standalone=False)]
    fakes = engines(
        FakeEngine(log, EVALUATED, pd_units),
        FakeEngine(log, "judge", nodes(2, 3)),
    )
    tasks = [
        task(log, "lm_eval", (EVALUATED,)),
        task(log, "score", ("judge",), seconds=0.3),
    ]

    run_scheduler(FakeOpLog(log, fail={("up", EVALUATED, (0, 1))}), fakes, tasks)

    assert log.of("up", "down", key=EVALUATED) == [
        ("up", EVALUATED, ((0, 1), (2,))),
        ("down", EVALUATED, ((0, 1), (2,))),
    ]
    assert log.of("attach", key=EVALUATED) == []
    assert log.started() == ["score"]
    assert ("up", "judge", ((2,),)) in log.events
    assert_clean_finish(log)


def test_unit_refused_by_the_router_is_dropped_and_the_engine_keeps_serving(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2), attach_fails=((1,),)),
        FakeEngine(log, "judge", nodes(1, 3)),
    )
    lm_eval = task(log, "lm_eval", (EVALUATED,), seconds=0.2)
    tasks = [lm_eval, task(log, "score", ("judge",), after=lm_eval)]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(
        log,
        ("attach", EVALUATED, ((1,),)),
        ("detach", EVALUATED, ((1,),)),
        ("down", EVALUATED, ((1,),)),
    )
    assert _ups_with_rank(log.events, 1) == [("up", EVALUATED, ((0,), (1,), (2,)))]
    assert log.started() == ["lm_eval", "score"]
    assert_clean_finish(log)


def test_judge_whose_only_unit_is_refused_skips_its_tasks(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1)),
        FakeEngine(log, "judge", nodes(1), attach_fails=((1,),)),
    )
    tasks = [
        task(log, "tau2@judge", (EVALUATED, "judge")),
        task(log, "lm_eval", (EVALUATED,)),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert log.started() == ["lm_eval"]
    assert_clean_finish(log)


def _rebalance(log, evaluated_load):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3), load=evaluated_load),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3), load=OVERLOADED),
    )
    return fakes, [task(log, "tau2@judge", (EVALUATED, "judge"), seconds=0.4)]


def _drain(sequence):
    drain_loads = itertools.chain(sequence, itertools.repeat(IDLE))
    return lambda probed: next(drain_loads) if probed == ((2,),) else IDLE


@pytest.mark.usefixtures("no_cooldown")
def test_donor_goes_down_only_after_two_consecutive_empty_drain_probes(log):
    fakes, tasks = _rebalance(log, _drain([BUSY, IDLE, BUSY, IDLE, IDLE]))

    run_scheduler(FakeOpLog(log), fakes, tasks)

    detach = ("detach", EVALUATED, ((2,),))
    down = ("down", EVALUATED, ((2,),))
    assert_in_order(log, detach, down, ("up", "judge", ((2,),)))
    drain_probes = log.between(detach, down).count(("load", EVALUATED, ((2,),)))
    assert drain_probes == 5
    assert_clean_finish(log)


def test_engine_whose_load_cannot_be_read_still_serves_its_tasks(log):
    fakes = engines(FakeEngine(log, EVALUATED, nodes(0, 1), load=None))

    run_scheduler(FakeOpLog(log), fakes, [task(log, "lm_eval", (EVALUATED,), 0.2)])

    assert log.started() == ["lm_eval"]
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_donor_does_not_go_down_while_kv_is_in_transfer(log):
    kv_in_transfer = replace(IDLE, kv_in_transfer=3)
    fakes, tasks = _rebalance(log, _drain([kv_in_transfer, IDLE, IDLE]))

    run_scheduler(FakeOpLog(log), fakes, tasks)

    detach = ("detach", EVALUATED, ((2,),))
    down = ("down", EVALUATED, ((2,),))
    drain_probes = log.between(detach, down).count(("load", EVALUATED, ((2,),)))
    assert drain_probes == 3
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_drain_timeout_puts_the_unit_back_and_does_not_grow_the_receiver(
    log, monkeypatch
):
    monkeypatch.setattr(DynamicScheduler, "DRAIN_TIMEOUT_SECONDS", 0.05)
    fakes, tasks = _rebalance(log, _drain(itertools.repeat(BUSY)))

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(
        log,
        ("detach", EVALUATED, ((2,),)),
        ("attach", EVALUATED, ((2,),)),
    )
    assert ("down", EVALUATED, ((2,),)) not in log.events
    assert ("up", "judge", ((2,),)) not in log.events
    assert log.started() == ["tau2@judge"]
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_move_aborted_by_a_drain_timeout_succeeds_once_the_donor_drains(
    log, monkeypatch
):
    monkeypatch.setattr(DynamicScheduler, "DRAIN_TIMEOUT_SECONDS", 0.05)

    def evaluated_load(probed):
        if probed != ((2,),):
            return IDLE
        reattached = log.events.count(("attach", EVALUATED, ((2,),))) >= 2
        return IDLE if reattached else BUSY

    fakes, tasks = _rebalance(log, evaluated_load)

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(
        log,
        ("detach", EVALUATED, ((2,),)),
        ("attach", EVALUATED, ((2,),)),
        ("detach", EVALUATED, ((2,),)),
        ("down", EVALUATED, ((2,),)),
        ("up", "judge", ((2,),)),
    )
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
@pytest.mark.parametrize(
    "drain_loads",
    [
        pytest.param(
            lambda: itertools.chain([None], itertools.repeat(BUSY)), id="once"
        ),
        pytest.param(lambda: itertools.repeat(None), id="always"),
    ],
)
def test_failed_load_probes_do_not_stop_a_unit_with_requests_in_flight(
    log, monkeypatch, drain_loads
):
    monkeypatch.setattr(DynamicScheduler, "DRAIN_TIMEOUT_SECONDS", 0.05)
    fakes, tasks = _rebalance(log, _drain(drain_loads()))

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert ("down", EVALUATED, ((2,),)) not in log.events
    assert ("up", "judge", ((2,),)) not in log.events
    assert_clean_finish(log)


def test_releasing_a_whole_engine_does_not_wait_for_a_drain(log, monkeypatch):
    monkeypatch.setattr(DynamicScheduler, "DRAIN_TIMEOUT_SECONDS", 30)
    fakes = engines(FakeEngine(log, EVALUATED, nodes(0, 1), load=BUSY))

    run_scheduler(FakeOpLog(log), fakes, [task(log, "lm", (EVALUATED,))], timeout=3)

    assert_clean_finish(log)


def _forced_eviction_that_never_drains(log, monkeypatch, drain_timeout):
    monkeypatch.setattr(DynamicScheduler, "DRAIN_TIMEOUT_SECONDS", drain_timeout)
    fakes = engines(
        FakeEngine(
            log,
            EVALUATED,
            nodes(0, 1, 2, 3),
            load=lambda probed: BUSY if len(probed) == 1 else IDLE,
        ),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    dredd = task(log, "dredd", (EVALUATED,))
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.8),
        dredd,
        task(log, "dredd@judge", ("judge",), after=dredd),
        task(log, "bfcl", (EVALUATED,), after=dredd),
    ]
    return fakes, tasks


def test_forced_eviction_that_timed_out_is_not_retried_every_tick(log, monkeypatch):
    fakes, tasks = _forced_eviction_that_never_drains(log, monkeypatch, 0.05)

    run_scheduler(FakeOpLog(log), fakes, tasks)

    attempts = log.before(("end", "lm_eval")).count(("detach", EVALUATED, ((3,),)))
    assert attempts <= 2


def test_ready_task_is_admitted_before_a_forced_eviction_in_the_same_tick(
    log, monkeypatch
):
    fakes, tasks = _forced_eviction_that_never_drains(log, monkeypatch, 0.6)

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert log.time(("start", "bfcl")) - log.time(("end", "dredd")) < 0.3


def test_task_that_becomes_ready_during_a_drain_does_not_wait_for_it(log, monkeypatch):
    fakes, tasks = _forced_eviction_that_never_drains(log, monkeypatch, 0.8)
    pre = task(log, "pre", (EVALUATED,), seconds=0.15)
    tasks += [pre, task(log, "x", (EVALUATED,), after=pre)]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert log.time(("start", "x")) - log.time(("end", "pre")) < 0.3


@pytest.mark.usefixtures("no_cooldown")
@pytest.mark.parametrize(
    ("judge_a_load", "moves"),
    [
        pytest.param(
            lambda probed: BUSY if probed == ((3,),) else IDLE,
            True,
            id="one-donor-never-drains",
        ),
        pytest.param(lambda probed: SERVING, False, id="one-donor-is-not-idle"),
    ],
)
def test_move_with_two_donors_needs_both_idle_and_both_drained(
    log, monkeypatch, judge_a_load, moves
):
    monkeypatch.setattr(DynamicScheduler, "DRAIN_TIMEOUT_SECONDS", 0.05)
    judge_b_units = units((2, 3), (5,))
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2), load=IDLE),
        FakeEngine(log, "judge-a", nodes(3, 4), load=judge_a_load),
        FakeEngine(
            log, "judge-b", judge_b_units, tail=judge_b_units[::-1], load=OVERLOADED
        ),
    )
    tasks = [
        task(log, "tau2@a", (EVALUATED, "judge-a"), seconds=0.5),
        task(log, "tau2@b", (EVALUATED, "judge-b"), seconds=0.5),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    # Once either task ends, releasing its engine is legitimate.
    before_end = log.before(log.of("end")[0])
    assert ("down", "judge-a", ((3,),)) not in before_end
    assert ("down", EVALUATED, ((2,),)) not in before_end
    detaches = [event for event in before_end if event[0] == "detach"]
    if moves:
        assert_in_order(
            log, ("detach", "judge-a", ((3,),)), ("attach", "judge-a", ((3,),))
        )
        assert_in_order(
            log, ("detach", EVALUATED, ((2,),)), ("attach", EVALUATED, ((2,),))
        )
    else:
        assert detaches == []
    assert_clean_finish(log)


def test_donor_that_just_grew_is_not_drained_before_its_cooldown(log, monkeypatch):
    cooldown = 0.3
    set_cooldown(monkeypatch, cooldown)
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3), load=IDLE),
        FakeEngine(
            log,
            "judge-a",
            nodes(0, 1, 2, 3),
            load=lambda probed: OVERLOADED if len(probed) == 1 else IDLE,
        ),
        FakeEngine(log, "judge-b", nodes(0, 1, 2, 3), load=OVERLOADED),
    )
    tasks = [
        task(log, "tau2@a", (EVALUATED, "judge-a"), seconds=1.5),
        task(log, "tau2@b", (EVALUATED, "judge-b"), seconds=1.5),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks, timeout=5)

    grown = [event for event in log.of("attach", key="judge-a")][1:]
    for attach in grown:
        (unit,) = attach[2]
        detach = ("detach", "judge-a", (unit,))
        if detach in log.events:
            assert log.time(detach) - log.time(attach) >= cooldown
    assert_clean_finish(log)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_task_that_raises_is_done_and_the_others_still_run(log):
    fakes = engines(FakeEngine(log, EVALUATED, nodes(0, 1)))
    broken = task(log, "broken", (EVALUATED,), error=RuntimeError("shooter crashed"))
    tasks = [
        broken,
        task(log, "bfcl", (EVALUATED,)),
        task(log, "next", (EVALUATED,), after=broken),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert log.started() == ["broken", "bfcl", "next"]
    assert_clean_finish(log)


def _raising_load(probed):
    raise RuntimeError("get_load failed")


@pytest.mark.parametrize(
    "fake_kwargs",
    [{"load": _raising_load}, {"detach_raises": True}],
    ids=["load-raises", "detach-raises"],
)
def test_engine_error_surfaces_and_still_publishes_finish(log, fake_kwargs):
    fakes = engines(FakeEngine(log, EVALUATED, nodes(0, 1), **fake_kwargs))

    with pytest.raises(RuntimeError):
        run_scheduler(FakeOpLog(log), fakes, [task(log, "lm", (EVALUATED,))])

    assert log.events[-1] == ("finish",)


def test_parallel_judges_never_take_every_node_before_evaluated_starts(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1)),
        FakeEngine(log, "judge-a", nodes(0, 1)),
        FakeEngine(log, "judge-b", nodes(0, 1)),
    )
    tasks = [
        task(log, "tau2@a", (EVALUATED, "judge-a")),
        task(log, "forge@b", (EVALUATED, "judge-b")),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks, timeout=HANG_TIMEOUT)

    assert sorted(log.started()) == ["forge@b", "tau2@a"]


def test_pinning_task_runs_when_an_evaluated_unit_is_quarantined(log):
    fakes = engines(FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)))
    tasks = [
        task(log, "acc_len", (EVALUATED,), pins=True),
        task(log, "lm_eval", (EVALUATED,)),
    ]

    run_scheduler(
        FakeOpLog(log, fail={("up", EVALUATED, (3,))}),
        fakes,
        tasks,
        timeout=HANG_TIMEOUT,
    )

    assert sorted(log.started()) == ["acc_len", "lm_eval"]


def test_task_whose_judge_can_never_be_placed_is_skipped(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1)),
        FakeEngine(log, "judge", units((0, 1))),
    )
    gen = task(log, "gen", (EVALUATED,))
    tasks = [gen, task(log, "score", ("judge",), after=gen)]

    run_scheduler(
        FakeOpLog(log, fail={("up", EVALUATED, (1,))}),
        fakes,
        tasks,
        timeout=HANG_TIMEOUT,
    )

    assert log.started() == ["gen"]
    assert_clean_finish(log)


def test_parallel_task_finishes_when_quarantine_leaves_evaluated_one_unit(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1)),
        FakeEngine(log, "judge", nodes(0, 1)),
    )
    lm_eval = task(log, "lm_eval", (EVALUATED,))
    tasks = [lm_eval, task(log, "tau2@judge", (EVALUATED, "judge"), after=lm_eval)]

    run_scheduler(
        FakeOpLog(log, fail={("up", EVALUATED, (1,))}),
        fakes,
        tasks,
        timeout=HANG_TIMEOUT,
    )

    assert log.started()[0] == "lm_eval"
    assert_clean_finish(log)


EVALUATED_LAYOUTS = {
    "nodeless": NODELESS,
    "1-node": nodes(0, 1, 2, 3),
    "2-node": units((0, 1), (2, 3)),
}
JUDGES = {
    "api-judge": ("api:judge", NODELESS),
    "node-judge": ("judge", nodes(0, 1, 2, 3)),
}
FAILURES = [
    "none",
    "first-unit-up",
    "last-unit-up",
    "all-units-up",
    "judge-up",
    "attach",
    "drain-never-empties",
]


def _matrix():
    for combination in itertools.product(EVALUATED_LAYOUTS, JUDGES, FAILURES):
        yield pytest.param(*combination, id="-".join(combination))


@pytest.mark.parametrize("layout, judge, failure", list(_matrix()))
def test_every_failure_combination_terminates_cleanly(
    log, monkeypatch, layout, judge, failure
):
    set_cooldown(monkeypatch, 0)
    monkeypatch.setattr(DynamicScheduler, "DRAIN_TIMEOUT_SECONDS", 0.05)
    evaluated_units = EVALUATED_LAYOUTS[layout]
    judge_key, judge_units = JUDGES[judge]
    evaluated_ranks = [unit.ranks for unit in evaluated_units]
    fail = {
        "first-unit-up": {("up", EVALUATED, evaluated_ranks[0])},
        "last-unit-up": {("up", EVALUATED, evaluated_ranks[-1])},
        "all-units-up": {("up", EVALUATED, ranks) for ranks in evaluated_ranks},
        "judge-up": {("up", judge_key, unit.ranks) for unit in judge_units},
    }.get(failure, set())
    attach_fails = (evaluated_ranks[0],) if failure == "attach" else ()
    evaluated_load = IDLE
    judge_load = IDLE
    if failure == "drain-never-empties":
        overloaded_until = time.monotonic() + 0.3
        evaluated_load = lambda probed: BUSY if len(probed) == 1 else IDLE  # noqa: E731
        judge_load = lambda probed: (  # noqa: E731
            OVERLOADED if time.monotonic() < overloaded_until else IDLE
        )
    fakes = engines(
        FakeEngine(
            log,
            EVALUATED,
            evaluated_units,
            load=evaluated_load,
            attach_fails=attach_fails,
        ),
        FakeEngine(log, judge_key, judge_units, load=judge_load),
    )
    gen = task(log, "gen", (EVALUATED,))
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.2),
        task(log, "tau2@judge", (EVALUATED, judge_key), seconds=0.2),
        gen,
        task(log, "gen:scoring", (judge_key,), after=gen),
        task(log, "acc_len", (EVALUATED,), pins=True),
    ]

    run_scheduler(FakeOpLog(log, fail=fail), fakes, tasks, timeout=HANG_TIMEOUT + 1)

    assert_clean_finish(log)
    every_task = sorted(t.evaluator for t in tasks)
    if failure in ("none", "drain-never-empties"):
        assert sorted(log.started()) == every_task
    if failure in ("first-unit-up", "last-unit-up") and layout == "1-node":
        assert sorted(log.started()) == every_task
    if failure == "judge-up":
        assert set(log.started()) == {"lm_eval", "gen", "acc_len"}
