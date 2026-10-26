import threading
from dataclasses import replace
import time

import pytest

from node_runner.src.inference.engine import NODELESS_UNIT, EngineLoad, EngineUnit
from node_runner.src.metrics.metrics_manager import (
    DynamicScheduler,
    EngineSlot,
    PendingOp,
    Stage,
    Task,
)

TICK = 0.01
UNIT_CAPACITY = 10
IDLE = EngineLoad(running=0, waiting=0, capacity=0, token_usage=0.0)
BUSY = EngineLoad(running=40, waiting=0, capacity=0, token_usage=0.5)
OVERLOADED = EngineLoad(running=1, waiting=10, capacity=0, token_usage=0.95)
SERVING = EngineLoad(running=8, waiting=1, capacity=0, token_usage=0.5)
QUEUE_OVERLOADED = EngineLoad(running=10, waiting=3, capacity=0, token_usage=0.1)
NODELESS = [NODELESS_UNIT]


def set_cooldown(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    monkeypatch.setattr(EngineSlot, "COOLDOWN_SECONDS", seconds)


class Log:
    def __init__(self):
        self._lock = threading.Lock()
        self.events: list[tuple] = []
        self.times: list[float] = []

    def add(self, *event) -> None:
        with self._lock:
            self.events.append(event)
            self.times.append(time.monotonic())

    def of(
        self, *kinds: str, key: str | None = None, before: tuple | None = None
    ) -> list[tuple]:
        events = list(self.events) if before is None else self.before(before)
        return [
            event
            for event in events
            if event[0] in kinds and (key is None or event[1:2] == (key,))
        ]

    def between(self, start: tuple, end: tuple) -> list[tuple]:
        return self.events[self.index(start) : self.index(end)]

    def index(self, event: tuple) -> int:
        return self.events.index(event)

    def time(self, event: tuple) -> float:
        return self.times[self.events.index(event)]

    def before(self, event: tuple) -> list[tuple]:
        return self.events[: self.index(event)]

    def started(self) -> list[str]:
        return [name for _, name in self.of("start")]


def ranks(units: list[EngineUnit]) -> tuple[tuple[int, ...], ...]:
    return tuple(unit.ranks for unit in units)


def units(*groups: tuple[int, ...], standalone: bool = True) -> list[EngineUnit]:
    return [
        EngineUnit(
            ranks=group, worker_urls=(f"http://w{group}",), standalone=standalone
        )
        for group in groups
    ]


def nodes(*node_ranks: int) -> list[EngineUnit]:
    return units(*[(rank,) for rank in node_ranks])


class FakeEngine:
    def __init__(
        self,
        log: Log,
        key: str,
        engine_units: list[EngineUnit],
        *,
        tail: list[EngineUnit] | None = None,
        load=IDLE,
        attach_fails: tuple[tuple[int, ...], ...] = (),
        detach_raises: bool = False,
    ):
        self.key = key
        self._log = log
        self._units = engine_units
        self._tail = engine_units[::-1] if tail is None else tail
        self._load = load
        self._attach_fails = set(attach_fails)
        self._detach_raises = detach_raises

    def units(self, from_tail: bool = False) -> list[EngineUnit]:
        return list(self._tail if from_tail else self._units)

    def load(self, engine_units: list[EngineUnit]) -> EngineLoad | None:
        probed = ranks(engine_units)
        self._log.add("load", self.key, probed)
        load = self._load(probed) if callable(self._load) else self._load
        if load is None:
            return None
        return replace(load, capacity=UNIT_CAPACITY * len(engine_units))

    def attach(self, engine_units: list[EngineUnit]) -> None:
        attached = ranks(engine_units)
        self._log.add("attach", self.key, attached)
        if any(unit in self._attach_fails for unit in attached):
            raise RuntimeError("router refused the worker")

    def detach(self, engine_units: list[EngineUnit]) -> None:
        self._log.add("detach", self.key, ranks(engine_units))
        if self._detach_raises:
            raise RuntimeError("router is gone")


def engines(*fakes: FakeEngine) -> dict[str, FakeEngine]:
    return {fake.key: fake for fake in fakes}


class FakeOpLog:
    def __init__(self, log: Log, fail=(), boot_seconds=None):
        self._log = log
        self._fail = set(fail)
        self._boot_seconds = boot_seconds or {}
        self._ready: dict[int, tuple[float, bool]] = {}

    def start(self, action, units_by_engine) -> list[PendingOp]:
        began = time.monotonic()
        pending = []
        for key, engine_units in units_by_engine.items():
            self._log.add(action, key, ranks(engine_units))
            engine_boot = self._boot_seconds.get(key, 0.0)
            for unit in engine_units:
                boot = self._boot_seconds.get((key, unit.ranks), engine_boot)
                op = PendingOp(key, unit, index=None)
                self._ready[id(op)] = (began + boot, self._ok(action, key, unit))
                pending.append(op)
        return pending

    def finished(self, op: PendingOp) -> bool:
        ready_at, ok = self._ready[id(op)]
        if time.monotonic() < ready_at:
            return False
        op.succeeded = ok
        return True

    def run(self, action, units_by_engine) -> list[PendingOp]:
        ops = []
        for key, engine_units in units_by_engine.items():
            self._log.add(action, key, ranks(engine_units))
            for unit in engine_units:
                ok = self._ok(action, key, unit)
                ops.append(PendingOp(key, unit, index=None, succeeded=ok))
        return ops

    def _ok(self, action, key, unit) -> bool:
        return (action, key, unit.ranks) not in self._fail

    def finish(self) -> None:
        self._log.add("finish")


def task(
    log: Log,
    name: str,
    engine_keys: tuple[str, ...],
    seconds: float = 0.05,
    after: Task | None = None,
    pins: bool = False,
    error: Exception | None = None,
) -> Task:

    def run():
        log.add("start", name)
        time.sleep(seconds)
        log.add("end", name)
        if error is not None:
            raise error

    stage = Stage(engine_keys, run, pins_workers=pins)
    return Task(name, stage, after)


def run_scheduler(op_log, engine_by_key, tasks, timeout: float = 5.0) -> None:
    errors: list[BaseException] = []

    def target():
        try:
            DynamicScheduler(op_log, engine_by_key, tasks).run()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "scheduler hung"
    if errors:
        raise errors[0]


def assert_in_order(log: Log, *expected: tuple) -> None:
    position = -1
    for event in expected:
        try:
            position = log.events.index(event, position + 1)
        except ValueError:
            pytest.fail(f"{event} not found after position {position}: {log.events}")


def assert_clean_finish(log: Log) -> None:
    ops = log.of("up", "down", "finish")
    assert ops.count(("finish",)) == 1
    assert ops[-1] == ("finish",)
    up: set[tuple] = set()
    for action, key, op_ranks in ops[:-1]:
        for unit in op_ranks:
            if action == "up":
                assert (key, unit) not in up, f"{key} {unit} booted twice"
                up.add((key, unit))
            else:
                assert (key, unit) in up, f"{key} {unit} stopped while down"
                up.remove((key, unit))
    assert up == set(), "engine units left up"
    finish = log.index(("finish",))
    for name in log.started():
        assert log.index(("end", name)) < finish


def assert_no_unit_restarted(log: Log) -> None:
    unit_ops = [
        (action, key, unit)
        for action, key, op_ranks in log.of("up", "down")
        for unit in op_ranks
    ]
    assert len(unit_ops) == len(set(unit_ops)), "a unit was restarted"
