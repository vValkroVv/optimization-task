import pytest

from node_runner.src.metrics.metrics_manager import DynamicScheduler, EngineOpLog
from sber_tests.unit.metrics.scheduler_fakes import TICK, Log, set_cooldown


@pytest.fixture
def fast_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DynamicScheduler, "PROBE_SECONDS", TICK)
    monkeypatch.setattr(DynamicScheduler, "DRAIN_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(EngineOpLog, "POLL_SECONDS", TICK)


@pytest.fixture
def no_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    set_cooldown(monkeypatch, 0)


@pytest.fixture
def log() -> Log:
    return Log()
