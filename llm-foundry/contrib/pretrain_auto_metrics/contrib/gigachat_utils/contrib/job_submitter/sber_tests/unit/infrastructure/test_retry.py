from types import SimpleNamespace

import pytest

from job_submitter.infrastructure.retry import RetryPolicy

pytestmark = pytest.mark.unit


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(
        "job_submitter.infrastructure.retry.time.monotonic", lambda: clock.now
    )
    monkeypatch.setattr("job_submitter.infrastructure.retry.time.sleep", sleep)
    return clock


def test_retry_recovers_on_last_attempt_and_preserves_arguments(clock):
    calls = []

    @RetryPolicy(
        exceptions=(OSError,), delay=lambda attempt, error: 0.5 * 2 ** (attempt - 1)
    )
    def operation(value, *, suffix):
        calls.append((value, suffix))
        if len(calls) < 3:
            raise OSError("unavailable")
        return value + suffix

    assert operation("a", suffix="b") == "ab"
    assert calls == [("a", "b")] * 3
    assert clock.sleeps == [0.5, 1.0]
    assert operation.__name__ == "operation"


@pytest.mark.parametrize("decorate", [False, True], ids=["run", "decorator"])
def test_retry_policy_reuses_settings_for_bound_method_calls(clock, decorate):
    class Writer:
        def __init__(self):
            self.calls = []

        def write(self, value, *, suffix):
            self.calls.append((value, suffix))
            if len(self.calls) % 2:
                raise OSError("unavailable")
            return value + suffix

    writer = Writer()
    policy = RetryPolicy(
        exceptions=(OSError,), attempts=2, delay=lambda attempt, error: 1
    )
    if decorate:
        write = policy(writer.write)
        assert write("a", suffix="b") == "ab"
        assert write("c", suffix="d") == "cd"
    else:
        assert policy.run(writer.write, "a", suffix="b") == "ab"
        assert policy.run(writer.write, "c", suffix="d") == "cd"

    assert writer.calls == [("a", "b"), ("a", "b"), ("c", "d"), ("c", "d")]
    assert clock.sleeps == [1, 1]


@pytest.mark.parametrize("attempts", [1, 3])
def test_retry_raises_last_error_without_extra_call_or_sleep(clock, attempts):
    errors = []

    @RetryPolicy(
        exceptions=(OSError,), attempts=attempts, delay=lambda attempt, error: 1
    )
    def operation():
        error = OSError(f"failure {len(errors) + 1}")
        errors.append(error)
        raise error

    with pytest.raises(OSError) as caught:
        operation()

    assert len(errors) == attempts
    assert caught.value is errors[-1]
    assert clock.sleeps == [1] * (attempts - 1)


def test_retry_does_not_catch_unlisted_errors(clock):
    @RetryPolicy(exceptions=(OSError,), delay=lambda attempt, error: 1)
    def operation():
        raise ValueError("invalid input")

    with pytest.raises(ValueError, match="invalid input"):
        operation()

    assert clock.sleeps == []


def test_retry_predicate_can_reject_error(clock):
    calls = []

    @RetryPolicy(
        exceptions=(OSError,),
        delay=lambda attempt, error: 1,
        should_retry=lambda error: str(error) == "temporary",
    )
    def operation():
        calls.append(None)
        raise OSError("permanent")

    with pytest.raises(OSError, match="permanent"):
        operation()

    assert len(calls) == 1
    assert clock.sleeps == []


def test_retry_adds_bounded_jitter(clock, monkeypatch):
    calls = []

    def uniform(lower, upper):
        assert (lower, upper) == (0, 1)
        return 0.25

    monkeypatch.setattr("job_submitter.infrastructure.retry.random.uniform", uniform)

    @RetryPolicy(
        exceptions=(OSError,),
        delay=lambda attempt, error: 5 * 2 ** (attempt - 1),
        jitter_seconds=1,
    )
    def operation():
        calls.append(None)
        raise OSError("unavailable")

    with pytest.raises(OSError):
        operation()

    assert len(calls) == 3
    assert clock.sleeps == [5.25, 10.25]


def test_zero_delay_retries_immediately_even_with_jitter(clock, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "job_submitter.infrastructure.retry.random.uniform",
        lambda lower, upper: pytest.fail("immediate retry must not add jitter"),
    )

    @RetryPolicy(
        exceptions=(OSError,), delay=lambda attempt, error: 0, jitter_seconds=1
    )
    def operation():
        calls.append(None)
        if len(calls) == 1:
            raise OSError("retry immediately")
        return "ok"

    assert operation() == "ok"
    assert len(calls) == 2
    assert clock.sleeps == []


@pytest.mark.parametrize("duration", [5, 6, 10])
def test_retry_does_not_wait_when_next_attempt_would_exceed_budget(clock, duration):
    calls = []

    @RetryPolicy(
        exceptions=(OSError,),
        delay=lambda attempt, error: 5,
        timeout_seconds=10,
    )
    def operation():
        calls.append(None)
        clock.now += duration
        raise OSError("unavailable")

    with pytest.raises(OSError):
        operation()

    assert len(calls) == 1
    assert clock.sleeps == []


def test_retry_recovers_within_time_budget(clock):
    calls = []

    @RetryPolicy(
        exceptions=(OSError,),
        delay=lambda attempt, error: 1,
        timeout_seconds=10,
    )
    def operation():
        calls.append(None)
        clock.now += 2
        if len(calls) == 1:
            raise OSError("unavailable")
        return "ok"

    assert operation() == "ok"
    assert clock.now == 5
    assert clock.sleeps == [1]


def test_retry_checks_budget_again_after_sleep(clock, monkeypatch):
    calls = []

    def oversleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += 10

    monkeypatch.setattr("job_submitter.infrastructure.retry.time.sleep", oversleep)

    @RetryPolicy(
        exceptions=(OSError,),
        delay=lambda attempt, error: 1,
        timeout_seconds=5,
    )
    def operation():
        calls.append(None)
        raise OSError("unavailable")

    with pytest.raises(OSError):
        operation()

    assert len(calls) == 1
    assert clock.sleeps == [1]


def test_retry_budget_is_fresh_for_each_call(clock):
    calls = []

    @RetryPolicy(
        exceptions=(OSError,),
        attempts=2,
        delay=lambda attempt, error: 1,
        timeout_seconds=5,
    )
    def operation():
        calls.append(None)
        if len(calls) % 2:
            raise OSError("unavailable")
        return "ok"

    assert operation() == "ok"
    clock.now += 100
    assert operation() == "ok"
    assert len(calls) == 4
    assert clock.sleeps == [1, 1]


def test_retry_counts_jitter_toward_time_budget(clock, monkeypatch):
    monkeypatch.setattr(
        "job_submitter.infrastructure.retry.random.uniform", lambda lower, upper: 1
    )
    calls = []

    @RetryPolicy(
        exceptions=(OSError,),
        delay=lambda attempt, error: 4,
        jitter_seconds=1,
        timeout_seconds=5,
    )
    def operation():
        calls.append(None)
        raise OSError("unavailable")

    with pytest.raises(OSError):
        operation()

    assert len(calls) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("result", [None, False, "ok"])
def test_retry_returns_success_even_when_call_outlasts_budget(clock, result):
    @RetryPolicy(
        exceptions=(OSError,),
        delay=lambda attempt, error: 1,
        timeout_seconds=5,
    )
    def operation():
        clock.now += 10
        return result

    assert operation() == result
    assert clock.now == 10
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "options",
    [
        {"attempts": 0},
        {"jitter_seconds": -1},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1},
    ],
)
def test_retry_rejects_invalid_policy(options):
    with pytest.raises(ValueError):
        RetryPolicy(exceptions=(OSError,), delay=lambda attempt, error: 1, **options)
