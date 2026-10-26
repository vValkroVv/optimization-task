import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import node_runner.src.trace.metrics.stream as stream_module
from node_runner.src.trace.metrics.model import MetricSample
from node_runner.src.trace.metrics.stream import MetricsStream

pytestmark = pytest.mark.unit


def _sample(metric: str, value: float = 1.0) -> MetricSample:
    return MetricSample(
        log_profile="evaluated",
        model_name=None,
        source="engine",
        endpoint=":30001",
        metric=metric,
        kind="gauge",
        labels={},
        ts=1710000123.4,
        value=value,
    )


def _setup(artifacts_path: Path) -> None:
    MetricsStream.setup(
        SimpleNamespace(
            job_id="job-a",
            artifacts_path=artifacts_path,
            node_name="mpimaster-0",
        )
    )


@pytest.fixture(autouse=True)
def unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MetricsStream, "_path", None)
    monkeypatch.setattr(MetricsStream, "_done_path", None)


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    _setup(tmp_path)
    return Path(tmp_path, "traces", "mpimaster-0")


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reported: list[str] = []
    monkeypatch.setattr(stream_module.Logger, "warning", reported.append)
    return reported


class TestSamples:
    def test_samples_are_appended_one_json_object_per_line(self, trace_dir: Path):
        MetricsStream.write([_sample("a"), _sample("b")])
        MetricsStream.write([_sample("c")])

        lines = Path(trace_dir, "metrics-job-a.jsonl").read_text().splitlines()
        assert [json.loads(line)["metric"] for line in lines] == ["a", "b", "c"]

    def test_writing_before_setup_is_a_noop(self):
        MetricsStream.write([_sample("a")])  # must not raise

    def test_setup_that_cannot_create_the_trace_dir_is_a_noop(
        self, tmp_path: Path, warnings: list[str]
    ):
        not_a_dir = Path(tmp_path, "artifacts")
        not_a_dir.write_text("")

        _setup(not_a_dir)
        MetricsStream.write([_sample("a")])
        MetricsStream.close()

        assert "metrics stream disabled" in warnings[0]

    def test_an_empty_batch_leaves_no_file(self, trace_dir: Path):
        MetricsStream.write([])

        assert not Path(trace_dir, "metrics-job-a.jsonl").exists()

    def test_a_non_finite_sample_is_omitted_and_the_rest_of_the_batch_is_kept(
        self, trace_dir: Path
    ):
        MetricsStream.write(
            [
                _sample("ok"),
                _sample("sglang:fwd_occupancy", float("nan")),
                _sample("sglang:fwd_occupancy", float("inf")),
                _sample("sglang:fwd_occupancy", float("-inf")),
            ]
        )

        text = Path(trace_dir, "metrics-job-a.jsonl").read_text()
        assert "NaN" not in text
        assert "Infinity" not in text
        assert [json.loads(line)["metric"] for line in text.splitlines()] == ["ok"]

    def test_a_batch_of_only_non_finite_samples_leaves_no_file(self, trace_dir: Path):
        MetricsStream.write([_sample("sglang:fwd_occupancy", float("nan"))])

        assert not Path(trace_dir, "metrics-job-a.jsonl").exists()

    def test_a_batch_it_cannot_write_is_dropped_with_a_warning(
        self, trace_dir: Path, warnings: list[str]
    ):
        Path(trace_dir, "metrics-job-a.jsonl").mkdir()

        MetricsStream.write([_sample("a")])

        assert "1 rows" in warnings[0]


class TestCompletionMarker:
    def test_marker_is_written_next_to_the_samples(self, trace_dir: Path):
        marker = Path(trace_dir, "metrics-job-a.done")
        assert not marker.exists()

        MetricsStream.close()

        assert marker.exists()

    def test_closing_before_setup_is_a_noop(self):
        MetricsStream.close()  # must not raise

    def test_a_marker_it_cannot_write_is_reported(
        self, trace_dir: Path, warnings: list[str]
    ):
        trace_dir.rmdir()

        MetricsStream.close()

        assert "marker not written" in warnings[0]
