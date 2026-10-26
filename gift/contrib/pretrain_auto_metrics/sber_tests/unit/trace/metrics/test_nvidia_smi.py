import subprocess
import threading

import pytest

import node_runner.src.trace.metrics.nvidia_smi as nvidia_smi_module
from node_runner.src.trace.metrics.model import NVIDIA_SMI_METRICS, MetricSample
from node_runner.src.trace.metrics.nvidia_smi import NVIDIA_SMI_QUERY, NvidiaSmiSampler

pytestmark = pytest.mark.unit

CELLS = {
    "utilization.gpu": "55",
    "utilization.memory": "12",
    "memory.used": "4096",
    "memory.total": "81920",
    "temperature.gpu": "61",
    "power.draw": "250.50",
    "enforced.power.limit": "400",
    "clocks.sm": "1410",
    "clocks_throttle_reasons.active": "0x0000000000000001",
}


class _FakeNvidiaSmi:
    """Stands in for the nvidia-smi binary, one answer per call."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.commands: list[list[str]] = []

    def run(self, cmd, capture_output, text, timeout):
        self.commands.append(cmd)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _StuckThread:
    """A sampler thread that ignores the stop event."""

    def join(self, timeout=None) -> None:
        return None

    def is_alive(self) -> bool:
        return True


def _row(index: str = "0", uuid: str = "GPU-abc", **cells: str) -> str:
    return ", ".join([index, uuid, *dict(CELLS, **cells).values()])


def _output(*rows: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="".join(f"{row}\n" for row in rows),
        stderr="",
    )


def _failure(stderr: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=9, stdout="", stderr=stderr
    )


def _sample_for(samples: list[MetricSample], metric: str) -> MetricSample:
    return next(sample for sample in samples if sample.metric == metric)


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    records: dict[str, list[str]] = {"info": [], "warning": [], "error": []}
    for level in records:
        monkeypatch.setattr(nvidia_smi_module.Logger, level, records[level].append)
    return records


@pytest.fixture
def written(monkeypatch: pytest.MonkeyPatch) -> list[list[MetricSample]]:
    batches: list[list[MetricSample]] = []
    monkeypatch.setattr(nvidia_smi_module.MetricsStream, "write", batches.append)
    return batches


@pytest.fixture
def nvidia_smi(monkeypatch: pytest.MonkeyPatch):
    def _install(*answers) -> _FakeNvidiaSmi:
        fake = _FakeNvidiaSmi(*answers)
        monkeypatch.setattr(nvidia_smi_module.subprocess, "run", fake.run)
        return fake

    return _install


@pytest.fixture
def nvidia_smi_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nvidia_smi_module.shutil, "which", lambda name: f"/usr/bin/{name}"
    )


class TestQuery:
    def test_a_successful_query_asks_nvidia_smi_for_the_tracked_metrics(
        self, nvidia_smi
    ):
        fake = nvidia_smi(_output(_row()))

        rows = NvidiaSmiSampler()._query()

        assert rows == [_row()]
        assert fake.commands == [
            [
                "nvidia-smi",
                f"--query-gpu={','.join(NVIDIA_SMI_QUERY)}",
                "--format=csv,noheader,nounits",
            ]
        ]

    def test_a_non_zero_exit_code_is_reported_once(self, nvidia_smi, logged):
        nvidia_smi(_failure("  driver mismatch  "), _failure("driver mismatch"))
        sampler = NvidiaSmiSampler()

        assert sampler._query() is None
        assert sampler._query() is None
        assert logged["warning"] == [
            "gpu metrics unavailable, nvidia-smi failed: driver mismatch"
        ]

    @pytest.mark.parametrize(
        "error",
        [
            OSError("nvidia-smi vanished"),
            subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0),
        ],
    )
    def test_a_broken_call_is_reported_once(self, nvidia_smi, logged, error):
        nvidia_smi(error, error)
        sampler = NvidiaSmiSampler()

        assert sampler._query() is None
        assert sampler._query() is None
        assert len(logged["warning"]) == 1
        assert logged["warning"][0].startswith("gpu metrics unavailable")

    def test_recovery_is_reported_once_nvidia_smi_answers_again(
        self, nvidia_smi, logged
    ):
        nvidia_smi(_failure("boom"), _output(_row()), _output(_row()))
        sampler = NvidiaSmiSampler()

        assert sampler._query() is None
        assert sampler._query() == [_row()]
        assert sampler._query() == [_row()]
        assert len(logged["warning"]) == 1
        assert logged["info"] == ["gpu metrics available again"]


class TestSamples:
    def test_a_row_becomes_one_node_sample_per_tracked_metric(self):
        samples = NvidiaSmiSampler()._samples([_row()])

        assert [sample.metric for sample in samples] == list(
            NVIDIA_SMI_METRICS.values()
        )
        assert {
            "log_profile": "node",
            "model_name": None,
            "source": "nvidia_smi",
            "endpoint": "gpu:0",
            "metric": "nvidia_smi:memory_used_mib",
            "kind": "gauge",
            "labels": {"uuid": "GPU-abc"},
            "value": 4096.0,
        }.items() <= _sample_for(
            samples, "nvidia_smi:memory_used_mib"
        ).model_dump().items()

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("55", 55.0),
            ("250.50", 250.5),
            ("0x0000000000000004", 4.0),
            ("[N/A]", None),
            ("[Not Supported]", None),
            ("", None),
            ("N/A", None),
        ],
    )
    def test_an_unreadable_cell_is_skipped_instead_of_sampled(self, raw, expected):
        row = _row(**{"utilization.gpu": raw})

        samples = NvidiaSmiSampler()._samples([row])

        values = [
            sample.value
            for sample in samples
            if sample.metric == "nvidia_smi:utilization_gpu"
        ]
        assert values == ([] if expected is None else [expected])

    def test_every_gpu_of_one_query_gets_its_own_endpoint_and_one_timestamp(self):
        samples = NvidiaSmiSampler()._samples(
            [_row(index="0", uuid="GPU-a"), _row(index="1", uuid="GPU-b")]
        )

        assert {(sample.endpoint, sample.labels["uuid"]) for sample in samples} == {
            ("gpu:0", "GPU-a"),
            ("gpu:1", "GPU-b"),
        }
        assert len({sample.ts for sample in samples}) == 1


class TestTick:
    def test_a_successful_query_is_written_to_the_metrics_stream(
        self, nvidia_smi, written
    ):
        nvidia_smi(_output(_row()))

        NvidiaSmiSampler()._tick()

        assert [sample.metric for sample in written[0]] == list(
            NVIDIA_SMI_METRICS.values()
        )

    def test_a_failed_query_writes_nothing(self, nvidia_smi, logged, written):
        nvidia_smi(_failure("boom"))

        NvidiaSmiSampler()._tick()

        assert written == []


class TestThreadLifecycle:
    def test_a_node_without_nvidia_smi_never_starts_a_thread(
        self, monkeypatch: pytest.MonkeyPatch, logged
    ):
        monkeypatch.setattr(nvidia_smi_module.shutil, "which", lambda name: None)
        sampler = NvidiaSmiSampler()

        sampler.start()

        assert sampler._thread is None
        assert logged["warning"] == ["gpu metrics disabled: nvidia-smi not found"]

    def test_starting_an_already_running_sampler_keeps_the_first_thread(
        self, monkeypatch: pytest.MonkeyPatch, nvidia_smi_installed
    ):
        sampler = NvidiaSmiSampler()
        monkeypatch.setattr(sampler, "_tick", lambda: None)

        sampler.start()
        first_thread = sampler._thread
        sampler.start()
        sampler.stop()

        assert sampler._thread is first_thread

    def test_stopping_a_sampler_that_never_started_is_a_noop(self):
        NvidiaSmiSampler().stop()  # must not raise

    def test_the_loop_ticks_until_it_is_stopped(
        self, monkeypatch: pytest.MonkeyPatch, nvidia_smi_installed
    ):
        monkeypatch.setattr(nvidia_smi_module, "SCRAPE_INTERVAL_SECONDS", 0.01)
        sampler = NvidiaSmiSampler()
        ticked = threading.Event()
        monkeypatch.setattr(sampler, "_tick", ticked.set)

        sampler.start()
        assert ticked.wait(timeout=5)
        sampler.stop()

        assert sampler._thread.is_alive() is False

    def test_a_failing_tick_does_not_kill_the_loop(
        self, monkeypatch: pytest.MonkeyPatch, nvidia_smi_installed, logged
    ):
        monkeypatch.setattr(nvidia_smi_module, "SCRAPE_INTERVAL_SECONDS", 0.01)
        sampler = NvidiaSmiSampler()
        ticks: list[None] = []
        survived = threading.Event()

        def _tick() -> None:
            ticks.append(None)
            if len(ticks) == 1:
                raise RuntimeError("boom")
            survived.set()

        monkeypatch.setattr(sampler, "_tick", _tick)

        sampler.start()
        assert survived.wait(timeout=5)
        sampler.stop()

        assert logged["warning"] == ["gpu metrics sample tick failed: boom"]

    def test_a_thread_that_outlives_stop_is_reported(self, logged):
        sampler = NvidiaSmiSampler()
        sampler._thread = _StuckThread()

        sampler.stop()

        assert "did not stop in time" in logged["error"][0]
