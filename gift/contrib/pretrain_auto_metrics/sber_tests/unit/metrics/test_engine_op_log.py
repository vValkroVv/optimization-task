import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from tenacity import wait_fixed

from node_runner.src import orkestration_manager as om_module
from node_runner.src.inference.engine import NODELESS_UNIT, EngineLoad, EngineUnit
from node_runner.src.metrics.metrics_manager import (
    EVALUATED,
    AutoMetricsManager,
    DynamicScheduler,
    EngineOpLog,
    EngineSlot,
    Stage,
    Task,
)
from node_runner.src.orkestration_manager import FastApiOrkestrationManager

pytestmark = pytest.mark.unit

JOIN_SECONDS = 3


def _fail(action):
    raise RuntimeError(f"{action} failed")


class _Engine:
    def __init__(self, rank, key, calls, on_call=None, units=()):
        self.rank, self.key, self.calls, self.on_call = rank, key, calls, on_call
        self._units = list(units)

    def start(self):
        self._call("up")

    def stop(self):
        self._call("down")

    def _call(self, action):
        self.calls.append((self.rank, self.key, action))
        if self.on_call:
            self.on_call(action)

    def units(self, from_tail=False):
        return self._units[::-1] if from_tail else self._units

    def load(self, units):
        return EngineLoad(0, 0, capacity=8, token_usage=0.0)

    def attach(self, units):
        pass

    def detach(self, units):
        pass


class _Cluster:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, nnodes: int, apps: list):
        self._monkeypatch = monkeypatch
        self._apps = apps
        self.calls: list[tuple] = []
        self.oms = [self._setup(rank, nnodes) for rank in range(nnodes)]
        self.followers: list[threading.Thread] = []

    def _setup(self, rank: int, nnodes: int) -> FastApiOrkestrationManager:
        om = FastApiOrkestrationManager()
        om.setup(
            node_name="mpimaster-0" if rank == 0 else f"mpiworker-{rank - 1}",
            orkestration_port=1,
            global_nnodes=nnodes,
            global_node_rank=rank,
            global_dist_init_host="master",
        )
        if rank == 0:
            self.client = TestClient(self._apps[-1])
            client = self.client
            self._monkeypatch.setattr(
                om_module,
                "requests",
                SimpleNamespace(
                    get=lambda url, timeout: client.get(urlsplit(url).path),
                    post=lambda url, json, timeout: client.post(
                        urlsplit(url).path, json=json
                    ),
                ),
            )
        return om

    def op_log(self, rank: int, hooks=None, keys=("judge",)) -> EngineOpLog:
        hooks = hooks or {}
        engines = {key: _Engine(rank, key, self.calls, hooks.get(rank)) for key in keys}
        return EngineOpLog(self.oms[rank], engines)

    def follow(self, *ranks, hooks=None, keys=("judge",)) -> None:
        for rank in ranks:
            op_log = self.op_log(rank, hooks, keys)
            thread = threading.Thread(target=op_log.follow, daemon=True)
            thread.start()
            self.followers.append(thread)

    def published_ops(self) -> list[dict]:
        ops = []
        while (resp := self.client.get(f"/ops/{len(ops)}")).status_code == 200:
            ops.append(resp.json())
        return ops

    def finish_and_join(self) -> None:
        self.oms[0].publish_op({"action": "finish"})
        for thread in self.followers:
            thread.join(timeout=JOIN_SECONDS)
        assert not any(t.is_alive() for t in self.followers), "worker hung"

    def calls_of(self, rank: int) -> list[tuple]:
        return [(key, action) for r, key, action in self.calls if r == rank]


@pytest.fixture
def make_cluster(monkeypatch: pytest.MonkeyPatch):
    apps: list = []

    def config(app, **kwargs):
        apps.append(app)
        return SimpleNamespace(app=app, **kwargs)

    monkeypatch.setattr(om_module.uvicorn, "Config", config)
    monkeypatch.setattr(
        om_module.uvicorn, "Server", lambda config: SimpleNamespace(run=lambda: None)
    )
    monkeypatch.setattr(
        FastApiOrkestrationManager.fetch_op.retry, "wait", wait_fixed(0.005)
    )
    monkeypatch.setattr(
        FastApiOrkestrationManager.wait_barrier.retry, "wait", wait_fixed(0.005)
    )
    monkeypatch.setattr(EngineOpLog, "POLL_SECONDS", 0.005)
    clusters: list[_Cluster] = []
    threads_before = set(threading.enumerate())

    def make(nnodes: int) -> _Cluster:
        cluster = _Cluster(monkeypatch, nnodes, apps)
        clusters.append(cluster)
        return cluster

    yield make
    for cluster in clusters:
        cluster.oms[0].publish_op({"action": "finish"})
    for thread in set(threading.enumerate()) - threads_before:
        thread.join(timeout=JOIN_SECONDS)


def _unit(*ranks) -> EngineUnit:
    return EngineUnit(ranks=ranks, worker_urls=())


def _in_thread(fn, *args) -> tuple[threading.Thread, list]:
    result: list = []
    thread = threading.Thread(target=lambda: result.append(fn(*args)), daemon=True)
    thread.start()
    return thread, result


def _joined(thread: threading.Thread, result: list):
    thread.join(timeout=JOIN_SECONDS)
    assert not thread.is_alive(), "master hung"
    return result[0]


def _outcomes(op_log: EngineOpLog, action, units_by_engine) -> dict[str, list[bool]]:
    pending = op_log.start(action, units_by_engine)
    deadline = time.monotonic() + JOIN_SECONDS
    while not all(op_log.finished(op) for op in pending):
        assert time.monotonic() < deadline, "master hung"
        time.sleep(0.005)
    outcomes: dict[str, list[bool]] = {key: [] for key in units_by_engine}
    for op in pending:
        outcomes[op.engine_key].append(op.succeeded)
    return outcomes


@pytest.mark.parametrize("action", ["up", "down"])
def test_multi_node_units_are_published_and_applied_on_every_rank(make_cluster, action):
    cluster = make_cluster(4)
    cluster.follow(1, 2, 3)

    result = _outcomes(cluster.op_log(0), action, {"judge": [_unit(0, 1), _unit(2, 3)]})
    cluster.finish_and_join()

    assert result == {"judge": [True, True]}
    assert cluster.published_ops()[:2] == [
        {"action": action, "engine": "judge", "ranks": [0, 1]},
        {"action": action, "engine": "judge", "ranks": [2, 3]},
    ]
    assert sorted(cluster.calls) == [(r, "judge", action) for r in range(4)]


def test_every_unit_is_published_before_the_master_boots_its_own(make_cluster):
    cluster = make_cluster(2)
    seen_when_master_booted = []
    master_hook = {
        0: lambda _: seen_when_master_booted.append(
            cluster.client.get("/ops/1").status_code
        )
    }
    cluster.follow(1)

    result = _outcomes(
        cluster.op_log(0, master_hook), "up", {"judge": [_unit(0), _unit(1)]}
    )
    cluster.finish_and_join()

    assert result == {"judge": [True, True]}
    assert seen_when_master_booted == [200]


def test_units_of_every_engine_in_one_call_boot_concurrently(make_cluster):
    cluster = make_cluster(4)
    all_booting = threading.Barrier(4, timeout=JOIN_SECONDS)
    hooks = {rank: lambda _: all_booting.wait() for rank in range(4)}
    keys = ("judge", EVALUATED)
    cluster.follow(1, 2, 3, hooks=hooks, keys=keys)
    units = {"judge": [_unit(1), _unit(2)], EVALUATED: [_unit(3), _unit(0)]}

    result = _outcomes(cluster.op_log(0, hooks, keys), "up", units)
    cluster.finish_and_join()

    assert result == {"judge": [True, True], EVALUATED: [True, True]}


def test_start_returns_before_the_unit_is_up_and_finished_reports_it_later(
    make_cluster,
):
    cluster = make_cluster(2)
    booted = threading.Event()
    cluster.follow(1, hooks={1: lambda _: booted.wait(JOIN_SECONDS)})
    op_log = cluster.op_log(0)

    (op,) = op_log.start("up", {"judge": [_unit(1)]})
    finished_while_booting = op_log.finished(op)
    booted.set()
    deadline = time.monotonic() + JOIN_SECONDS
    while not op_log.finished(op) and time.monotonic() < deadline:
        time.sleep(0.01)
    cluster.finish_and_join()

    assert finished_while_booting is False
    assert op.succeeded is True


def test_start_boots_the_master_share_before_returning(make_cluster):
    cluster = make_cluster(2)
    master_booted = threading.Event()
    keys = ("judge", EVALUATED)
    cluster.follow(1, keys=keys)
    op_log = cluster.op_log(0, {0: lambda _: master_booted.wait(JOIN_SECONDS)}, keys)

    thread, result = _in_thread(
        op_log.start, "up", {"judge": [_unit(1)], EVALUATED: [_unit(0)]}
    )
    deadline = time.monotonic() + JOIN_SECONDS
    while cluster.calls_of(1) != [("judge", "up")] and time.monotonic() < deadline:
        time.sleep(0.01)
    returned_while_master_boots = not thread.is_alive()
    master_booted.set()
    _joined(thread, result)
    cluster.finish_and_join()

    assert returned_while_master_boots is False


def test_run_returns_only_after_every_rank_applied_the_op(make_cluster):
    cluster = make_cluster(2)
    stopped = threading.Event()
    cluster.follow(1, hooks={1: lambda _: stopped.wait()})

    thread, result = _in_thread(cluster.op_log(0).run, "down", {"judge": [_unit(1)]})
    time.sleep(0.2)
    returned_before_stop = not thread.is_alive()
    stopped.set()
    _joined(thread, result)
    cluster.finish_and_join()

    assert returned_before_stop is False
    assert cluster.calls_of(1) == [("judge", "down")]


def test_master_does_not_boot_units_without_its_rank(make_cluster):
    cluster = make_cluster(3)
    cluster.follow(1, 2)

    result = _outcomes(cluster.op_log(0), "up", {"judge": [_unit(1), _unit(2)]})
    cluster.finish_and_join()

    assert result == {"judge": [True, True]}
    assert cluster.calls_of(0) == []


@pytest.mark.parametrize("fails", [False, True])
def test_nodeless_unit_is_applied_on_the_master_without_publishing(make_cluster, fails):
    cluster = make_cluster(2)
    op_log = cluster.op_log(0, {0: _fail} if fails else None)

    results = [_outcomes(op_log, a, {"judge": [NODELESS_UNIT]}) for a in ("up", "down")]

    assert results == [{"judge": [not fails]}, {"judge": [not fails]}]
    assert cluster.calls_of(0) == [("judge", "up"), ("judge", "down")]
    assert cluster.published_ops() == []


def test_failed_master_unit_is_reported_not_raised(make_cluster):
    cluster = make_cluster(2)

    result = _outcomes(cluster.op_log(0, {0: _fail}), "up", {"judge": [_unit(0)]})

    assert result == {"judge": [False]}


@pytest.mark.parametrize("action", ["up", "down"])
def test_failed_worker_unit_fails_alone_and_its_worker_keeps_following(
    make_cluster, action
):
    cluster = make_cluster(3)
    fail_once = iter([_fail])
    cluster.follow(1, 2, hooks={2: lambda a: next(fail_once, lambda _: None)(a)})
    op_log = cluster.op_log(0)

    first = _outcomes(op_log, action, {"judge": [_unit(1), _unit(2)]})
    second = _outcomes(op_log, action, {"judge": [_unit(1), _unit(2)]})
    cluster.finish_and_join()

    assert (first, second) == ({"judge": [True, False]}, {"judge": [True, True]})
    assert cluster.calls_of(2) == [("judge", action), ("judge", action)]


def test_unit_with_one_failed_rank_fails_only_after_every_rank_finished(
    make_cluster,
):
    cluster = make_cluster(3)
    rank_1_booted = threading.Event()
    cluster.follow(1, 2, hooks={1: lambda _: rank_1_booted.wait(), 2: _fail})
    op_log = cluster.op_log(0)

    (op,) = op_log.start("up", {"judge": [_unit(1, 2)]})
    time.sleep(0.2)
    finished_while_rank_1_boots = op_log.finished(op)
    rank_1_booted.set()
    deadline = time.monotonic() + JOIN_SECONDS
    while not op_log.finished(op) and time.monotonic() < deadline:
        time.sleep(0.01)
    cluster.finish_and_join()

    assert finished_while_rank_1_boots is False
    assert op.succeeded is False


def test_unknown_engine_on_a_worker_is_a_failed_unit_not_a_dead_worker(
    make_cluster,
):
    cluster = make_cluster(2)
    cluster.follow(1, keys=("evaluated",))

    result = _outcomes(cluster.op_log(0), "up", {"judge": [_unit(1)]})
    still_following = cluster.followers[0].is_alive()

    assert (result, still_following) == ({"judge": [False]}, True)
    cluster.finish_and_join()


def test_worker_applies_only_its_own_ops_in_publish_order(make_cluster):
    cluster = make_cluster(4)
    master = cluster.oms[0]
    cluster.follow(2, 3, keys=("judge", "eval"))

    master.publish_op({"action": "up", "engine": "judge", "ranks": [1]})
    master.publish_op({"action": "up", "engine": "judge", "ranks": [2]})
    master.publish_op({"action": "down", "engine": "eval", "ranks": [2, 3]})
    cluster.finish_and_join()

    assert cluster.calls_of(2) == [("judge", "up"), ("eval", "down")]
    assert cluster.calls_of(3) == [("eval", "down")]


def test_finish_is_a_terminal_op_that_stops_followers(make_cluster):
    cluster = make_cluster(2)
    cluster.follow(1)

    cluster.op_log(0).finish()
    cluster.followers[0].join(timeout=JOIN_SECONDS)

    assert not cluster.followers[0].is_alive()
    assert cluster.published_ops() == [{"action": "finish"}]
    assert cluster.calls == []


def test_follower_waits_for_ops_not_yet_published(make_cluster):
    cluster = make_cluster(2)
    cluster.follow(1)

    time.sleep(0.1)
    alive_before_any_op = cluster.followers[0].is_alive()
    result = _outcomes(cluster.op_log(0), "up", {"judge": [_unit(1)]})
    cluster.finish_and_join()

    assert alive_before_any_op is True
    assert result == {"judge": [True]}
    assert cluster.calls_of(1) == [("judge", "up")]


def test_follower_reports_failure_and_still_arrives(make_cluster):
    cluster = make_cluster(2)
    master = cluster.oms[0]
    cluster.follow(1, hooks={1: _fail})

    master.publish_op({"action": "up", "engine": "judge", "ranks": [1]})
    cluster.finish_and_join()

    assert master.barrier_released("ENGINE_OP_0_FAIL") is True
    assert master.barrier_released("ENGINE_OP_0") is True


def test_follower_does_not_arrive_at_barriers_of_ops_for_other_ranks(make_cluster):
    cluster = make_cluster(4)
    master = cluster.oms[0]
    cluster.follow(3)

    master.publish_op({"action": "up", "engine": "judge", "ranks": [1]})
    cluster.finish_and_join()

    assert master.barrier_released("ENGINE_OP_0") is False
    assert cluster.calls == []


@pytest.fixture
def fast_scheduler(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DynamicScheduler, "PROBE_SECONDS", 0.005)
    monkeypatch.setattr(EngineSlot, "COOLDOWN_SECONDS", 0)


def _task(name: str, engine_keys: tuple[str, ...], run) -> Task:
    return Task(name, Stage(engine_keys, run), after=None)


def _pipelines(cluster: _Cluster, tasks: list[Task], master_load=None):
    units = {EVALUATED: [_unit(0), _unit(1)], "judge": [_unit(2)]}
    managers = []
    for rank, om in enumerate(cluster.oms):
        manager = AutoMetricsManager()
        manager.orkestration_manager = om
        manager.run_config = SimpleNamespace(
            autometrics=SimpleNamespace(
                globals=SimpleNamespace(send_metrics_at_end=False)
            )
        )
        manager.engines_by_key = {
            key: _Engine(rank, key, cluster.calls, units=key_units)
            for key, key_units in units.items()
        }
        if rank == 0:
            manager.tasks = tasks
            if master_load:
                manager.engines_by_key[EVALUATED].load = master_load
        managers.append(manager)
    return [_in_thread(manager.run_pipeline) for manager in managers]


def test_master_and_workers_run_the_pipeline_to_completion(
    make_cluster, fast_scheduler
):
    cluster = make_cluster(3)
    ran: list[str] = []
    tasks = [
        _task("gen", (EVALUATED,), lambda: ran.append("gen")),
        _task("judged", (EVALUATED, "judge"), lambda: ran.append("judged")),
    ]

    threads = _pipelines(cluster, tasks)
    for thread, _ in threads:
        thread.join(timeout=JOIN_SECONDS)

    assert not any(thread.is_alive() for thread, _ in threads), "pipeline hung"
    assert sorted(ran) == ["gen", "judged"]
    assert [op["action"] for op in cluster.published_ops()].count("finish") == 1
    for rank in range(3):
        for key in (EVALUATED, "judge"):
            actions = [a for k, a in cluster.calls_of(rank) if k == key]
            assert actions == ["up", "down"] * (len(actions) // 2), (rank, key)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_workers_stop_following_when_the_master_scheduler_crashes(
    make_cluster, fast_scheduler
):
    cluster = make_cluster(3)

    def broken_load(units):
        raise RuntimeError("router is gone")

    tasks = [_task("gen", (EVALUATED,), lambda: None)]
    (master, _), *workers = _pipelines(cluster, tasks, master_load=broken_load)
    master.join(timeout=JOIN_SECONDS)
    for thread, _ in workers:
        thread.join(timeout=0.5)

    assert not master.is_alive()
    assert not any(thread.is_alive() for thread, _ in workers), "workers hung"
