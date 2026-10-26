import itertools
import time

import pytest

from node_runner.src.inference.engine import EngineLoad
from node_runner.src.metrics.metrics_manager import (
    EVALUATED,
    DynamicScheduler,
    EngineSlot,
)
from sber_tests.unit.metrics.scheduler_fakes import (
    BUSY,
    IDLE,
    NODELESS,
    OVERLOADED,
    QUEUE_OVERLOADED,
    SERVING,
    FakeEngine,
    FakeOpLog,
    Log,
    assert_clean_finish,
    assert_in_order,
    assert_no_unit_restarted,
    engines,
    nodes,
    run_scheduler,
    set_cooldown,
    task,
    units,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fast_clock")]

ALL4 = ((0,), (1,), (2,), (3,))


def _ops(log: Log, key: str | None = None) -> list[tuple]:
    return log.of("up", "down", "finish", key=key)


def test_evaluated_boots_all_its_units_in_one_batch_and_stops_them_together(log):
    evaluated = FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3))

    run_scheduler(FakeOpLog(log), engines(evaluated), [task(log, "lm", (EVALUATED,))])

    assert _ops(log) == [
        ("up", EVALUATED, ALL4),
        ("down", EVALUATED, ALL4),
        ("finish",),
    ]
    assert log.of("attach") == [("attach", EVALUATED, (unit,)) for unit in ALL4]
    assert log.started() == ["lm"]


def test_parallel_judge_starts_on_one_tail_unit_and_evaluated_on_the_rest(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )

    run_scheduler(
        FakeOpLog(log), fakes, [task(log, "tau2@judge", (EVALUATED, "judge"))]
    )

    assert _ops(log) == [
        ("up", "judge", ((3,),)),
        ("up", EVALUATED, ((0,), (1,), (2,))),
        ("down", EVALUATED, ((0,), (1,), (2,))),
        ("down", "judge", ((3,),)),
        ("finish",),
    ]
    assert_in_order(log, ("attach", "judge", ((3,),)), ("start", "tau2@judge"))
    assert_in_order(log, ("attach", EVALUATED, ((2,),)), ("start", "tau2@judge"))


def test_fast_judge_on_worker_nodes_joins_its_router_before_slow_evaluated(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1)),
        FakeEngine(log, "judge", nodes(0, 1)),
    )
    tasks = [
        task(log, "lm_eval", (EVALUATED,)),
        task(log, "tau2@judge", (EVALUATED, "judge")),
    ]

    run_scheduler(
        FakeOpLog(log, boot_seconds={EVALUATED: 0.6, "judge": 0.05}), fakes, tasks
    )

    assert_in_order(log, ("attach", "judge", ((1,),)), ("attach", EVALUATED, ((0,),)))
    assert_clean_finish(log)


def test_evaluated_serves_only_after_every_unit_of_its_boot_is_up(log):
    fakes = engines(FakeEngine(log, EVALUATED, nodes(0, 1)))

    run_scheduler(
        FakeOpLog(log, boot_seconds={(EVALUATED, (0,)): 0.05, (EVALUATED, (1,)): 0.4}),
        fakes,
        [task(log, "lm_eval", (EVALUATED,))],
    )

    assert log.of("attach")[:2] == [
        ("attach", EVALUATED, ((0,),)),
        ("attach", EVALUATED, ((1,),)),
    ]
    assert (
        log.time(("attach", EVALUATED, ((0,),)))
        - log.time(("up", EVALUATED, ((0,), (1,))))
        >= 0.4
    )
    assert_clean_finish(log)


def test_pd_judge_extra_that_boots_first_waits_for_its_anchor(log):
    pd_units = [*units((0,)), *units((1,), standalone=False)]
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(2)),
        FakeEngine(log, "judge", pd_units, tail=pd_units),
    )

    run_scheduler(
        FakeOpLog(log, boot_seconds={("judge", (0,)): 0.4, ("judge", (1,)): 0.05}),
        fakes,
        [task(log, "tau2", (EVALUATED, "judge"))],
    )

    assert_in_order(log, ("attach", "judge", ((0,),)), ("attach", "judge", ((1,),)))
    assert log.of("up", key="judge") == [("up", "judge", ((0,), (1,)))]
    assert_clean_finish(log)


@pytest.mark.parametrize(
    ("evaluated_units", "judge_units", "boot_seconds"),
    [
        pytest.param(NODELESS, nodes(0, 1), {"judge": 0.3}, id="slow-judge"),
        pytest.param(nodes(0, 1), NODELESS, {EVALUATED: 0.3}, id="slow-evaluated"),
    ],
)
def test_parallel_task_starts_only_once_both_engines_serve(
    log, evaluated_units, judge_units, boot_seconds
):
    fakes = engines(
        FakeEngine(log, EVALUATED, evaluated_units),
        FakeEngine(log, "judge", judge_units),
    )

    run_scheduler(
        FakeOpLog(log, boot_seconds=boot_seconds),
        fakes,
        [task(log, "tau2@judge", (EVALUATED, "judge"))],
    )

    last_attach = log.of("attach")[-1]
    assert log.index(last_attach) < log.index(("start", "tau2@judge"))
    assert_clean_finish(log)


def test_judge_serves_from_its_first_ready_unit(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    gen = task(log, "gen", (EVALUATED,))
    tasks = [gen, task(log, "score", ("judge",), after=gen)]

    run_scheduler(
        FakeOpLog(log, boot_seconds={("judge", (3,)): 0.05, "judge": 0.5}),
        fakes,
        tasks,
    )

    assert log.time(("start", "score")) < log.time(("attach", "judge", ((0,),)))
    assert_clean_finish(log)


def test_one_bad_node_in_a_judge_batch_does_not_skip_its_tasks(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    gen = task(log, "gen", (EVALUATED,))
    tasks = [gen, task(log, "score", ("judge",), after=gen)]

    run_scheduler(
        FakeOpLog(
            log,
            fail={("up", "judge", (3,))},
            boot_seconds={("judge", (3,)): 0.0, "judge": 0.3},
        ),
        fakes,
        tasks,
    )

    assert log.started() == ["gen", "score"]
    assert ("attach", "judge", ((3,),)) not in log.events
    assert_clean_finish(log)


def test_finish_waits_for_a_unit_that_is_still_booting(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    tasks = [
        task(log, "lm_eval", (EVALUATED,)),
        task(log, "tau2@judge", (EVALUATED, "judge")),
    ]

    run_scheduler(
        FakeOpLog(
            log,
            fail={("up", EVALUATED, (rank,)) for rank in (0, 1, 2)},
            boot_seconds={"judge": 0.3},
        ),
        fakes,
        tasks,
    )

    assert log.started() == []
    assert_clean_finish(log)


def test_parallel_task_whose_judge_always_overlaps_evaluated_is_skipped(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1)),
        FakeEngine(log, "judge", units((0, 1))),
    )
    tasks = [
        task(log, "lm_eval", (EVALUATED,)),
        task(log, "tau2@judge", (EVALUATED, "judge")),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert log.started() == ["lm_eval"]
    assert log.of("up", key="judge") == []
    assert_clean_finish(log)


def test_multi_node_judge_starts_on_its_tail_unit(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, units((0, 1), (2, 3))),
        FakeEngine(log, "judge", units((0, 1), (2, 3))),
    )

    run_scheduler(
        FakeOpLog(log), fakes, [task(log, "tau2@judge", (EVALUATED, "judge"))]
    )

    assert _ops(log)[:2] == [("up", "judge", ((2, 3),)), ("up", EVALUATED, ((0, 1),))]
    assert_clean_finish(log)


def test_two_parallel_judges_take_separate_free_nodes_and_run_concurrently(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge-a", nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge-b", nodes(0, 1, 2, 3)),
    )
    tasks = [
        task(log, "tau2@a", (EVALUATED, "judge-a"), seconds=0.3),
        task(log, "forge@b", (EVALUATED, "judge-b"), seconds=0.3),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert _ops(log)[:3] == [
        ("up", "judge-a", ((3,),)),
        ("up", "judge-b", ((2,),)),
        ("up", EVALUATED, ((0,), (1,))),
    ]
    assert log.time(("start", "forge@b")) < log.time(("end", "tau2@a"))
    assert_clean_finish(log)


def test_scoring_judge_is_booted_only_after_its_inference_stage(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    dredd = task(log, "dredd", (EVALUATED,), seconds=0.2)
    tasks = [dredd, task(log, "dredd@judge", ("judge",), after=dredd)]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert _ops(log)[0] == ("up", EVALUATED, ALL4)
    first_judge_op = _ops(log, "judge")[0]
    assert log.index(first_judge_op) > log.index(("end", "dredd"))
    assert_clean_finish(log)


def test_stopped_judge_boots_all_its_free_units_in_one_batch(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    dredd = task(log, "dredd", (EVALUATED,))
    tasks = [dredd, task(log, "dredd@judge", ("judge",), after=dredd)]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert log.of("up", key="judge") == [("up", "judge", ((3,), (2,), (1,), (0,)))]


def test_engines_of_the_first_tick_boot_concurrently(log):
    boot_seconds = 0.3
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    started_at = time.monotonic()

    run_scheduler(
        FakeOpLog(log, boot_seconds={EVALUATED: boot_seconds, "judge": boot_seconds}),
        fakes,
        [task(log, "tau2@judge", (EVALUATED, "judge"))],
    )

    assert log.time(("start", "tau2@judge")) - started_at < 1.5 * boot_seconds


def test_full_pipeline_runs_every_task_and_never_boots_an_unused_engine(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
        FakeEngine(log, "api:judge", NODELESS),
        FakeEngine(log, "unused", nodes(0, 1, 2, 3)),
    )
    dredd = task(log, "dredd", (EVALUATED,))
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.2),
        dredd,
        task(log, "dredd:scoring", ("api:judge",), after=dredd),
        task(log, "tau2@judge", (EVALUATED, "judge")),
        task(log, "acc_len", (EVALUATED,), pins=True),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert sorted(log.started()) == sorted(t.evaluator for t in tasks)
    assert _ops(log, "api:judge") == [
        ("up", "api:judge", ((),)),
        ("down", "api:judge", ((),)),
    ]
    assert [event for event in log.events if "unused" in event] == []
    assert_clean_finish(log)


def test_external_evaluated_engine_is_started_and_stopped_for_its_tasks(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, NODELESS),
        FakeEngine(log, "judge", nodes(0, 1)),
    )
    tasks = [
        task(log, "lm_eval", (EVALUATED,)),
        task(log, "tau2@judge", (EVALUATED, "judge")),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert sorted(log.started()) == ["lm_eval", "tau2@judge"]
    assert _ops(log, EVALUATED) == [
        ("up", EVALUATED, ((),)),
        ("down", EVALUATED, ((),)),
    ]
    assert_clean_finish(log)


def test_multi_node_judge_unit_takes_the_evaluated_unit_under_it_once(log):
    evaluated = FakeEngine(
        log,
        EVALUATED,
        units((0, 1), (2, 3)),
        load=lambda probed: IDLE if probed == ((2, 3),) else BUSY,
    )
    fakes = engines(evaluated, FakeEngine(log, "judge", units((0, 1), (2, 3))))
    dredd = task(log, "dredd", (EVALUATED,))
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.4),
        dredd,
        task(log, "dredd@judge", ("judge",), after=dredd),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(
        log,
        ("detach", EVALUATED, ((2, 3),)),
        ("down", EVALUATED, ((2, 3),)),
        ("up", "judge", ((2, 3),)),
        ("start", "dredd@judge"),
    )
    assert_clean_finish(log)


def test_first_judge_evicts_a_drained_tail_unit_of_busy_evaluated_even_in_cooldown(
    log,
):
    evaluated = FakeEngine(
        log,
        EVALUATED,
        nodes(0, 1, 2, 3),
        load=lambda probed: IDLE if probed == ((3,),) else BUSY,
    )
    fakes = engines(evaluated, FakeEngine(log, "judge", nodes(0, 1, 2, 3)))
    dredd = task(log, "dredd", (EVALUATED,))
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.6),
        dredd,
        task(log, "dredd@judge", ("judge",), after=dredd),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(
        log,
        ("end", "dredd"),
        ("detach", EVALUATED, ((3,),)),
        ("down", EVALUATED, ((3,),)),
        ("up", "judge", ((3,),)),
        ("start", "dredd@judge"),
        ("end", "lm_eval"),
    )
    evaluated_downs_while_busy = log.of(
        "down", key=EVALUATED, before=("end", "lm_eval")
    )
    assert evaluated_downs_while_busy == [("down", EVALUATED, ((3,),))]
    assert_clean_finish(log)


def test_forced_eviction_prefers_a_free_unit_over_draining_a_busy_one(log):
    evaluated = FakeEngine(
        log,
        EVALUATED,
        nodes(0, 1, 3),
        load=lambda probed: IDLE if probed == ((3,),) else BUSY,
    )
    fakes = engines(
        evaluated,
        FakeEngine(log, "judge-x", nodes(2)),
        FakeEngine(log, "judge-y", nodes(2, 3)),
    )
    x = task(log, "x", (EVALUATED, "judge-x"))
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.6),
        x,
        task(log, "y", ("judge-y",), after=x),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert ("up", "judge-y", ((2,),)) in log.events
    assert log.of("detach", key=EVALUATED, before=("end", "lm_eval")) == []


@pytest.mark.usefixtures("no_cooldown")
def test_second_judge_never_evicts_busy_evaluated_and_waits_for_the_first(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3), load=BUSY),
        FakeEngine(log, "judge-a", nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge-b", nodes(0, 1, 2, 3)),
    )
    gen = task(log, "gen", (EVALUATED,), seconds=0.02)
    tasks = [
        task(log, "tau2@a", (EVALUATED, "judge-a"), seconds=0.4),
        task(log, "lm_eval", (EVALUATED,), seconds=0.8),
        gen,
        task(log, "score@b", ("judge-b",), after=gen),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert log.of("detach", "down", key=EVALUATED, before=("end", "lm_eval")) == []
    assert_in_order(
        log,
        ("end", "tau2@a"),
        ("down", "judge-a", ((3,),)),
        ("up", "judge-b", ((3,),)),
        ("start", "score@b"),
    )
    assert_clean_finish(log)


def test_nodeless_judge_never_takes_or_evicts_nodes(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3), load=BUSY),
        FakeEngine(log, "api:judge", NODELESS),
    )
    dredd = task(log, "dredd", (EVALUATED,), seconds=0.02)
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.4),
        dredd,
        task(log, "dredd:scoring", ("api:judge",), after=dredd),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert _ops(log, "api:judge") == [
        ("up", "api:judge", ((),)),
        ("down", "api:judge", ((),)),
    ]
    assert _ops(log, EVALUATED) == [("up", EVALUATED, ALL4), ("down", EVALUATED, ALL4)]
    assert log.time(("start", "dredd:scoring")) < log.time(("end", "lm_eval"))
    assert_clean_finish(log)


def test_donor_keeps_its_last_unit_while_it_has_work(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0), load=BUSY),
        FakeEngine(log, "judge", nodes(0)),
    )
    dredd = task(log, "dredd", (EVALUATED,), seconds=0.02)
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.3),
        dredd,
        task(log, "dredd@judge", ("judge",), after=dredd),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(
        log,
        ("end", "lm_eval"),
        ("detach", EVALUATED, ((0,),)),
        ("down", EVALUATED, ((0,),)),
        ("up", "judge", ((0,),)),
        ("start", "dredd@judge"),
    )
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_overloaded_evaluated_never_takes_the_last_unit_of_a_judge_mid_task(log):
    fakes = engines(
        FakeEngine(
            log,
            EVALUATED,
            nodes(0, 1),
            load=lambda probed: IDLE if probed == ((1,),) else OVERLOADED,
        ),
        FakeEngine(log, "judge", nodes(0, 1), load=IDLE),
    )
    dredd = task(log, "dredd", (EVALUATED,), seconds=0.02)
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.8),
        dredd,
        task(log, "dredd@judge", ("judge",), after=dredd, seconds=0.5),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    judge_downs = log.of("down", key="judge")
    assert judge_downs[0] not in log.before(("end", "dredd@judge"))
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
@pytest.mark.parametrize(
    "judge_load",
    [
        pytest.param(OVERLOADED, id="kv-full"),
        pytest.param(QUEUE_OVERLOADED, id="too-many-requests"),
    ],
)
def test_overloaded_judge_takes_a_drained_unit_from_idle_evaluated(log, judge_load):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3), load=IDLE),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3), load=judge_load),
    )

    run_scheduler(
        FakeOpLog(log),
        fakes,
        [task(log, "tau2@judge", (EVALUATED, "judge"), seconds=0.4)],
    )

    assert_in_order(
        log,
        ("detach", EVALUATED, ((2,),)),
        ("down", EVALUATED, ((2,),)),
        ("up", "judge", ((2,),)),
        ("attach", "judge", ((2,),)),
    )
    assert_clean_finish(log)


def _alternating(*loads):
    cycle = itertools.cycle(loads)
    return lambda probed: next(cycle)


@pytest.mark.usefixtures("no_cooldown")
@pytest.mark.parametrize(
    "evaluated_load, judge_load",
    [
        pytest.param(BUSY, OVERLOADED, id="donor-busy"),
        pytest.param(
            IDLE,
            _alternating(OVERLOADED, IDLE),
            id="overload-spikes-are-not-consecutive",
        ),
        pytest.param(
            IDLE,
            _alternating(OVERLOADED, OVERLOADED, SERVING),
            id="a-normal-probe-breaks-the-overload-streak",
        ),
        pytest.param(
            IDLE,
            EngineLoad(running=10, waiting=2, capacity=0, token_usage=0.92),
            id="judge-exactly-at-the-overload-limits",
        ),
        pytest.param(
            EngineLoad(running=0, waiting=5, capacity=0, token_usage=0.1),
            OVERLOADED,
            id="donor-with-a-queue-is-not-idle",
        ),
    ],
)
def test_no_rebalance_without_steady_overload_and_an_idle_donor(
    log, evaluated_load, judge_load
):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3), load=evaluated_load),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3), load=judge_load),
    )

    run_scheduler(
        FakeOpLog(log),
        fakes,
        [task(log, "tau2@judge", (EVALUATED, "judge"), seconds=0.4)],
    )

    assert _ops(log) == [
        ("up", "judge", ((3,),)),
        ("up", EVALUATED, ((0,), (1,), (2,))),
        ("down", EVALUATED, ((0,), (1,), (2,))),
        ("down", "judge", ((3,),)),
        ("finish",),
    ]


@pytest.mark.usefixtures("no_cooldown")
def test_donor_gives_a_unit_only_after_enough_idle_probes(log):
    busy_probes = 10
    evaluated_probes = itertools.count(1)

    def evaluated_load(probed):
        if len(probed) == 1:
            return IDLE
        return BUSY if next(evaluated_probes) <= busy_probes else IDLE

    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3), load=evaluated_load),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3), load=OVERLOADED),
    )

    run_scheduler(
        FakeOpLog(log),
        fakes,
        [task(log, "tau2@judge", (EVALUATED, "judge"), seconds=0.5)],
    )

    probes_before_give = log.before(("down", EVALUATED, ((2,),))).count(
        ("load", EVALUATED, ((0,), (1,), (2,)))
    )
    assert probes_before_give == busy_probes + EngineSlot.UNDERLOAD_PROBES


def _rebalance_gaps(log: Log, key: str, until: tuple) -> list[float]:
    times = [
        log.times[index]
        for index, event in enumerate(log.before(until))
        if event[:2] in {("up", key), ("down", key)}
    ]
    return [later - earlier for earlier, later in zip(times, times[1:])]


@pytest.mark.parametrize("flip_seconds", [None, 0.25], ids=["steady", "flipping"])
def test_rebalance_moves_are_at_least_one_cooldown_apart(
    log, monkeypatch, flip_seconds
):
    cooldown = 0.3
    set_cooldown(monkeypatch, cooldown)
    began = time.monotonic()

    def judge_hungry() -> bool:
        if flip_seconds is None:
            return True
        return int((time.monotonic() - began) / flip_seconds) % 2 == 0

    fakes = engines(
        FakeEngine(
            log,
            EVALUATED,
            nodes(0, 1, 2, 3),
            load=lambda probed: IDLE if judge_hungry() else OVERLOADED,
        ),
        FakeEngine(
            log,
            "judge",
            nodes(0, 1, 2, 3),
            load=lambda probed: OVERLOADED if judge_hungry() else IDLE,
        ),
    )

    run_scheduler(
        FakeOpLog(log),
        fakes,
        [task(log, "tau2@judge", (EVALUATED, "judge"), seconds=1.2)],
    )

    end = ("end", "tau2@judge")
    assert ("down", EVALUATED, ((2,),)) in log.before(end), "never rebalanced"
    for key in (EVALUATED, "judge"):
        assert all(gap >= cooldown for gap in _rebalance_gaps(log, key, end)), key
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_overloaded_evaluated_takes_a_unit_back_from_an_idle_judge(log):
    judge_grew = []

    def judge_load(probed):
        if len(probed) == 2:
            judge_grew.append(True)
        return IDLE if judge_grew else OVERLOADED

    fakes = engines(
        FakeEngine(
            log,
            EVALUATED,
            nodes(0, 1),
            load=lambda probed: OVERLOADED if judge_grew else IDLE,
        ),
        FakeEngine(log, "judge", nodes(1, 2), load=judge_load),
    )

    run_scheduler(
        FakeOpLog(log),
        fakes,
        [task(log, "tau2@judge", (EVALUATED, "judge"), seconds=0.6)],
    )

    assert_in_order(
        log,
        ("up", "judge", ((2,),)),
        ("up", EVALUATED, ((0,), (1,))),
        ("down", EVALUATED, ((1,),)),
        ("up", "judge", ((1,),)),
        ("detach", "judge", ((1,),)),
        ("down", "judge", ((1,),)),
        ("up", EVALUATED, ((1,),)),
        ("end", "tau2@judge"),
    )
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_most_overloaded_engine_claims_the_contested_unit_first(log):
    judge_b_probes = itertools.count(1)
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1), load=IDLE),
        FakeEngine(
            log,
            "judge-b",
            nodes(1, 3),
            load=lambda probed: OVERLOADED if next(judge_b_probes) > 3 else IDLE,
        ),
        FakeEngine(log, "judge-a", nodes(1, 2), load=OVERLOADED),
    )
    tasks = [
        task(log, "tau2@a", (EVALUATED, "judge-a"), seconds=0.4),
        task(log, "tau2@b", (EVALUATED, "judge-b"), seconds=0.4),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    judge_a_takes = ("up", "judge-a", ((1,),))
    assert judge_a_takes in log.events
    assert ("up", "judge-b", ((1,),)) not in log.before(judge_a_takes)
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_pd_judge_keeps_its_anchor_while_its_task_runs(log):
    pd_units = [*units((1, 2)), *units((3,), standalone=False)]
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1), load=OVERLOADED),
        FakeEngine(log, "judge", pd_units, tail=pd_units, load=IDLE),
    )

    run_scheduler(
        FakeOpLog(log),
        fakes,
        [task(log, "tau2@judge", (EVALUATED, "judge"), seconds=0.5)],
    )

    assert ("down", "judge", ((1, 2),)) not in log.before(("end", "tau2@judge"))
    assert_clean_finish(log)


def test_judges_of_one_evaluator_never_overlap_when_a_middle_one_is_skipped(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "ja", nodes(0, 1, 2, 3)),
        FakeEngine(log, "jb", units((0, 1, 2, 3))),
        FakeEngine(log, "jc", nodes(0, 1, 2, 3)),
    )
    with_ja = task(log, "tau2@ja", (EVALUATED, "ja"), seconds=0.3)
    with_jb = task(log, "tau2@jb", (EVALUATED, "jb"), after=with_ja)
    tasks = [
        with_ja,
        with_jb,
        task(log, "tau2@jc", (EVALUATED, "jc"), after=with_jb),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert log.started() == ["tau2@ja", "tau2@jc"]
    assert_in_order(log, ("end", "tau2@ja"), ("start", "tau2@jc"))
    assert_clean_finish(log)


def test_two_judges_ready_together_evict_evaluated_only_once(log):
    evaluated = FakeEngine(
        log,
        EVALUATED,
        nodes(0, 1, 2, 3),
        load=lambda probed: IDLE if len(probed) == 1 else BUSY,
    )
    fakes = engines(
        evaluated,
        FakeEngine(log, "judge-a", nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge-b", nodes(0, 1, 2, 3)),
    )
    gen_a = task(log, "gen_a", (EVALUATED,), seconds=0.02)
    gen_b = task(log, "gen_b", (EVALUATED,), seconds=0.02)
    tasks = [
        task(log, "lm_eval", (EVALUATED,), seconds=0.8),
        gen_a,
        gen_b,
        task(log, "score@a", ("judge-a",), after=gen_a),
        task(log, "score@b", ("judge-b",), after=gen_b),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    evaluated_detaches = log.of("detach", key=EVALUATED, before=("end", "lm_eval"))
    assert len(evaluated_detaches) == 1
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_pd_donor_gives_extras_but_never_its_anchor_while_working(log):
    pd_units = [*units((0,)), *units((1,), (2,), standalone=False)]
    fakes = engines(
        FakeEngine(log, EVALUATED, pd_units, load=IDLE),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3), load=OVERLOADED),
    )

    run_scheduler(
        FakeOpLog(log),
        fakes,
        [task(log, "tau2@judge", (EVALUATED, "judge"), seconds=0.5)],
    )

    given = [
        event[2]
        for event in log.of("down", key=EVALUATED, before=("end", "tau2@judge"))
    ]
    assert given == [((2,),), ((1,),)]
    assert_clean_finish(log)


def test_finished_judge_is_released_at_once_and_its_node_goes_to_evaluated(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge-later", nodes(0, 1, 2, 3)),
    )
    lm_eval = task(log, "lm_eval", (EVALUATED,), seconds=0.5)
    tasks = [
        task(log, "tau2@judge", (EVALUATED, "judge")),
        lm_eval,
        task(log, "score", ("judge-later",), after=lm_eval),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(
        log,
        ("end", "tau2@judge"),
        ("down", "judge", ((3,),)),
        ("up", EVALUATED, ((3,),)),
        ("attach", EVALUATED, ((3,),)),
        ("end", "lm_eval"),
    )
    assert _ops(log, "judge-later")[0] not in log.before(("end", "lm_eval"))
    assert_clean_finish(log)


def test_engine_is_not_restarted_between_its_tasks(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    tau2 = task(log, "tau2@judge", (EVALUATED, "judge"))
    lm_eval = task(log, "lm_eval", (EVALUATED,))
    tasks = [
        tau2,
        task(log, "forge@judge", (EVALUATED, "judge")),
        task(log, "tau2:scoring", ("judge",), after=tau2),
        lm_eval,
        task(log, "bfcl", (EVALUATED,), after=lm_eval),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert sorted(log.started()) == sorted(t.evaluator for t in tasks)
    assert_no_unit_restarted(log)
    assert_clean_finish(log)


@pytest.mark.usefixtures("no_cooldown")
def test_judge_waiting_for_a_chained_stage_keeps_its_unit(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3), load=OVERLOADED),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    dredd = task(log, "dredd", (EVALUATED,), seconds=0.4)
    tasks = [
        task(log, "tau2@judge", (EVALUATED, "judge")),
        dredd,
        task(log, "dredd@judge", ("judge",), after=dredd),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_no_unit_restarted(log)
    assert log.of("down", key="judge")[0] not in log.before(("end", "dredd"))
    assert_clean_finish(log)


def test_pd_evaluated_boots_anchor_and_extras_in_one_batch_and_attaches_anchor_first(
    log,
):
    pd_units = [
        *units((2,), standalone=False),
        *units((0, 1)),
        *units((3,), standalone=False),
    ]
    fakes = engines(FakeEngine(log, EVALUATED, pd_units))

    run_scheduler(FakeOpLog(log), fakes, [task(log, "lm_eval", (EVALUATED,))])

    assert log.of("up") == [("up", EVALUATED, ((2,), (0, 1), (3,)))]
    assert log.of("attach") == [
        ("attach", EVALUATED, ((0, 1),)),
        ("attach", EVALUATED, ((2,),)),
        ("attach", EVALUATED, ((3,),)),
    ]
    assert log.started() == ["lm_eval"]


def test_stopped_pd_judge_boots_its_anchor_and_extras_in_one_batch(log):
    pd_units = [*units((2, 3)), *units((1,), standalone=False)]
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0)),
        FakeEngine(log, "judge", pd_units, tail=pd_units),
    )
    tasks = [task(log, "score", ("judge",), seconds=0.2), task(log, "lm", (EVALUATED,))]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert log.of("up", key="judge") == [("up", "judge", ((2, 3), (1,)))]
    assert_clean_finish(log)


def test_pinning_task_waits_until_evaluated_has_every_unit_back(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    tasks = [
        task(log, "tau2@judge", (EVALUATED, "judge"), seconds=0.3),
        task(log, "acc_len", (EVALUATED,), pins=True),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(
        log,
        ("end", "tau2@judge"),
        ("up", EVALUATED, ((3,),)),
        ("start", "acc_len"),
    )
    assert_clean_finish(log)


def test_running_pinning_task_blocks_every_move_of_evaluated(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3), load=BUSY),
        FakeEngine(log, "judge", nodes(0, 1, 2, 3)),
    )
    dredd = task(log, "dredd", (EVALUATED,), seconds=0.1)
    tasks = [
        dredd,
        task(log, "acc_len", (EVALUATED,), seconds=0.4, pins=True),
        task(log, "dredd@judge", ("judge",), after=dredd),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(
        log,
        ("start", "acc_len"),
        ("end", "dredd"),
        ("end", "acc_len"),
        ("down", EVALUATED, ALL4),
        ("up", "judge", ((3,), (2,), (1,), (0,))),
    )
    assert log.of("detach", key=EVALUATED) == [("detach", EVALUATED, ALL4)]
    assert_clean_finish(log)


def test_admits_every_ready_task_in_the_same_tick(log, monkeypatch):
    tick = 0.15
    monkeypatch.setattr(DynamicScheduler, "PROBE_SECONDS", tick)
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0)),
        FakeEngine(log, "api:judge", NODELESS),
        FakeEngine(log, "api:other", NODELESS),
    )
    tasks = [
        task(log, "lm_eval", (EVALUATED,)),
        task(log, "bfcl", (EVALUATED,)),
        task(log, "tau2@api", (EVALUATED, "api:judge")),
        task(log, "score@other", ("api:other",)),
    ]
    began = time.monotonic()

    run_scheduler(FakeOpLog(log), fakes, tasks)

    first_start = log.time(("start", "tau2@api"))
    tick_of = {
        name: round((log.time(("start", name)) - first_start) / tick)
        for name in log.started()
    }
    assert tick_of == {"tau2@api": 0, "score@other": 0, "lm_eval": 0, "bfcl": 0}
    assert first_start - began < tick
    assert_clean_finish(log)


@pytest.mark.parametrize(
    "task_keys, overloaded_key, waits",
    [
        pytest.param((EVALUATED,), EVALUATED, True, id="plain-task-evaluated-busy"),
        pytest.param(
            (EVALUATED, "api:judge"), EVALUATED, False, id="parallel-ignores-evaluated"
        ),
        pytest.param(
            (EVALUATED, "api:judge"), "api:judge", True, id="parallel-judge-busy"
        ),
        pytest.param(("api:judge",), "api:judge", True, id="scoring-judge-busy"),
    ],
)
@pytest.mark.parametrize("overload", [OVERLOADED, QUEUE_OVERLOADED])
def test_task_is_not_admitted_to_an_overloaded_engine(
    log, task_keys, overloaded_key, waits, overload
):
    overload_seconds = 0.4
    began = time.monotonic()

    def load(key):
        def probe(probed):
            still_overloaded = time.monotonic() - began < overload_seconds
            if key == overloaded_key and still_overloaded:
                return overload
            return IDLE

        return probe

    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0), load=load(EVALUATED)),
        FakeEngine(log, "api:judge", NODELESS, load=load("api:judge")),
    )
    gen = task(log, "gen", (EVALUATED,))
    tasks = [
        task(log, "warm", ("api:judge",), seconds=overload_seconds + 0.1),
        gen,
        task(log, "next", task_keys, after=gen),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    started_after_overload = log.time(("start", "next")) - began >= overload_seconds
    assert started_after_overload is waits
    assert_clean_finish(log)


def test_chained_stages_never_overlap(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge-a", nodes(0, 1, 2, 3)),
        FakeEngine(log, "judge-b", nodes(0, 1, 2, 3)),
    )
    tau2_a = task(log, "tau2@a", (EVALUATED, "judge-a"))
    dredd = task(log, "dredd", (EVALUATED,))
    tasks = [
        tau2_a,
        task(log, "tau2@b", (EVALUATED, "judge-b"), after=tau2_a),
        dredd,
        task(log, "dredd@a", ("judge-a",), after=dredd),
    ]

    run_scheduler(FakeOpLog(log), fakes, tasks)

    assert_in_order(log, ("end", "tau2@a"), ("start", "tau2@b"))
    assert_in_order(log, ("end", "dredd"), ("start", "dredd@a"))
    assert_clean_finish(log)


def test_no_tasks_finishes_without_touching_any_engine(log):
    fakes = engines(
        FakeEngine(log, EVALUATED, nodes(0, 1)),
        FakeEngine(log, "judge", nodes(0, 1)),
    )

    run_scheduler(FakeOpLog(log), fakes, [], timeout=1)

    assert log.events == [("finish",)]


def test_finish_is_published_within_a_tick_after_the_last_task(log, monkeypatch):
    tick = 0.2
    monkeypatch.setattr(DynamicScheduler, "PROBE_SECONDS", tick)
    fakes = engines(FakeEngine(log, EVALUATED, NODELESS))

    run_scheduler(FakeOpLog(log), fakes, [task(log, "lm", (EVALUATED,))])

    assert log.time(("finish",)) - log.time(("end", "lm")) < tick + 0.1
