"""Tests for the job_dispatcher JD metrics client."""

from pathlib import Path
from typing import Any, Dict

import pytest
from tenacity import wait_fixed

from node_runner.src.config_model import ModelInfoConfig
from node_runner.src.job_dispatcher import JD
from node_runner.src.job_dispatcher import metrics as jd_metrics
from node_runner.src.logger import Logger
from sber_tests.test_support import set_env

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_requests_post(monkeypatch: pytest.MonkeyPatch) -> Dict[str, int]:
    """Fixture mocking requests.post and tracking calls."""
    called: Dict[str, int] = {"post": 0, "raise_for_status": 0}

    class _MockResponse:
        def raise_for_status(self) -> None:
            called["raise_for_status"] += 1

    def _mock_post(*args: Any, **kwargs: Any) -> _MockResponse:
        called["post"] += 1
        return _MockResponse()

    monkeypatch.setattr(jd_metrics.requests, "post", _mock_post)
    return called


@pytest.fixture
def mock_logger(monkeypatch: pytest.MonkeyPatch) -> Dict[str, list]:
    """Fixture mocking logger and capturing log messages."""
    logs: Dict[str, list] = {"info": [], "error": [], "warning": []}

    monkeypatch.setattr(Logger, "info", logs["info"].append)
    monkeypatch.setattr(Logger, "error", logs["error"].append)
    monkeypatch.setattr(Logger, "warning", logs["warning"].append)

    return logs


@pytest.fixture
def online(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JD, "internet_is_available", lambda: True)


class TestSendMetrics:
    @pytest.mark.parametrize(
        ("has_internet", "expected_url"),
        [
            (True, "https://jdmetrics.sberdevices.ru/api/ml_models/add_metrics"),
            (False, "https://dj.static.space/api/ml_models/add_metrics"),
        ],
    )
    def test_posts_to_jd_reachable_from_current_network(
        self,
        has_internet: bool,
        expected_url: str,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        posted_urls = []

        class _MockResponse:
            def raise_for_status(self) -> None:
                pass

        def _mock_post(url: str, **kwargs: Any) -> _MockResponse:
            posted_urls.append(url)
            return _MockResponse()

        monkeypatch.setattr(JD, "internet_is_available", lambda: has_internet)
        monkeypatch.setattr(jd_metrics.requests, "post", _mock_post)

        assert JD.send_metrics({"a": 1}, None) is True

        assert posted_urls == [expected_url]

    def test_success(
        self,
        online: None,
        mock_requests_post: Dict[str, int],
        mock_logger: Dict[str, list],
    ):
        assert JD.send_metrics({"a": 1}, None) is True

        assert mock_requests_post["post"] == 1
        assert mock_requests_post["raise_for_status"] == 1
        assert "Metrics successfully sent to JD" in mock_logger["info"]

    def test_a_replay_does_not_report_to_jd(
        self,
        mock_requests_post: Dict[str, int],
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(JD, "reports_to_jd", False)

        assert JD.send_metrics({"a": 1}, None) is False

        assert mock_requests_post["post"] == 0
        assert (
            "Answers come from a recording; skip sending metrics to jd"
            in mock_logger["info"]
        )

    def test_adds_checkpoint_history_from_hf_checkpoint_artifacts(
        self,
        online: None,
        mock_requests_post: Dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        jd_data = [{"checkpoint_info": {"id": "checkpoint-id"}}]
        body = {"a": 1}
        collected_paths = []

        def collect_checkpoint_jd_data(path: Path) -> list:
            collected_paths.append(path)
            return jd_data

        monkeypatch.setattr(
            JD, "collect_checkpoint_jd_data", collect_checkpoint_jd_data
        )

        JD.send_metrics(body, tmp_path / "hf_checkpoint")

        assert collected_paths == [tmp_path / "hf_checkpoint" / "artifacts"]
        assert body["checkpoint_history"] == jd_data
        assert mock_requests_post["post"] == 1

    def test_skips_missing_checkpoint_history(
        self,
        online: None,
        mock_requests_post: Dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        body = {"a": 1}
        monkeypatch.setattr(JD, "collect_checkpoint_jd_data", lambda path: None)

        JD.send_metrics(body, tmp_path)

        assert "checkpoint_history" not in body
        assert mock_requests_post["post"] == 1

    def test_rejected_metrics_are_logged_with_the_response_body(
        self,
        online: None,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A JD rejection is logged with status and body before the retry kicks in."""

        class _RejectingResponse:
            status_code = 422
            text = "validation failed"

            def raise_for_status(self) -> None:
                raise jd_metrics.requests.HTTPError("422")

        monkeypatch.setattr(
            jd_metrics.requests, "post", lambda *a, **kw: _RejectingResponse()
        )
        monkeypatch.setattr(JD.send_metrics.retry, "wait", wait_fixed(0))

        assert JD.send_metrics({"a": 1}, None) is False
        assert "JD rejected metrics (422): validation failed" in mock_logger["error"]

    def test_exhausted_retries_are_logged(
        self,
        online: None,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A request failing every retry is logged, not silently dropped."""

        def _failing_post(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("jd is down")

        monkeypatch.setattr(jd_metrics.requests, "post", _failing_post)
        monkeypatch.setattr(JD.send_metrics.retry, "wait", wait_fixed(0))

        assert JD.send_metrics({"a": 1}, None) is False
        assert any(
            "after 3 attempt(s)" in message and "jd is down" in message
            for message in mock_logger["error"]
        )


class TestSaveMetricsJson:
    def test_success(self, tmp_path: Path, mock_logger: Dict[str, list]):
        JD.save_metrics_json({"acc": 0.1}, 3, tmp_path)

        assert (tmp_path / "metrics_step_3.json").exists()
        assert len(mock_logger["info"]) > 0

    def test_error_is_logged(
        self,
        tmp_path: Path,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            jd_metrics.json,
            "dump",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        JD.save_metrics_json({"acc": 0.1}, 4, tmp_path)

        assert len(mock_logger["error"]) > 0


class TestShooterVersion:
    def test_reads_file(self, tmp_path: Path):
        ver_file = tmp_path / "ver.txt"
        ver_file.write_text("1.2.3\n")

        assert JD.get_shooter_version(str(ver_file)) == "1.2.3"

    def test_missing_file_falls_back(
        self, tmp_path: Path, mock_logger: Dict[str, list]
    ):
        result = JD.get_shooter_version(str(tmp_path / "missing.txt"))

        assert result == "PAM_FILE"
        assert len(mock_logger["error"]) > 0


class TestBodyHelpers:
    def test_custom_model_name_from_env(self):
        with set_env(CUSTOM_MODEL_NAME="custom_model_name"):
            result = JD.get_custom_model_name_if_exist("old_model_name")
            assert result == "custom_model_name"

    def test_set_autocompare_id_from_env_in_body(self):
        with set_env(AUTOCOMPARE_ID="12345"):
            body = {"existing": "value"}
            JD.set_autocompare_id_in_body(body)
            assert body == {"existing": "value", "autocompare_id": "12345"}

    def test_custom_model_name_without_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CUSTOM_MODEL_NAME", raising=False)

        assert JD.get_custom_model_name_if_exist("original_name") == "original_name"

    def test_set_autocompare_id_without_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AUTOCOMPARE_ID", raising=False)
        body = {"existing": "value"}

        JD.set_autocompare_id_in_body(body)

        assert body == {"existing": "value"}


class TestGenerateMetricsJson:
    @pytest.mark.parametrize(
        ("step_metrics", "expected"),
        [
            ({"acc": 0.5, "f1": 0.0}, {"acc": 0.5, "f1": 0.0}),
            ({"acc": 0.5, "missing": None}, {"acc": 0.5}),
            ({"acc": 0.5, "broken": float("nan")}, {"acc": 0.5}),
            ({"acc": 0.5, "broken": float("inf")}, {"acc": 0.5}),
            ({"acc": 0.5, "broken": float("-inf")}, {"acc": 0.5}),
        ],
    )
    def test_drops_none_and_non_finite_values(
        self, step_metrics: Dict[str, Any], expected: Dict[str, float]
    ):
        assert JD.generate_metrics_json({1: step_metrics}) == {1: expected}

    def test_keeps_every_step(self):
        result = JD.generate_metrics_json({1: {"acc": 0.1}, 2: {"acc": 0.2}})

        assert result == {1: {"acc": 0.1}, 2: {"acc": 0.2}}


class TestCanSendMetrics:
    @staticmethod
    def _args(tmp_path: Path, **overrides) -> Dict[str, Any]:
        args = {
            "metrics": {"acc": 0.5},
            "hf_checkpoint_path": tmp_path / "hf",
            "metrics_dir": tmp_path / "metrics",
            "model_info": ModelInfoConfig(model_class="sft", size="7b", username="u"),
        }
        args.update(overrides)
        return args

    def test_passes_when_everything_is_set(self, tmp_path: Path):
        assert JD.can_send_metrics(**self._args(tmp_path)) is True

    @pytest.mark.parametrize(
        ("missing", "expected_warning"),
        [
            ("metrics", "No metrics to report"),
            ("hf_checkpoint_path", "hf_checkpoint_path is not set"),
            ("metrics_dir", "metrics_dir is not set"),
            ("model_info", "model_info is not set"),
        ],
    )
    def test_reports_the_first_missing_prerequisite(
        self,
        mock_logger: Dict[str, list],
        tmp_path: Path,
        missing: str,
        expected_warning: str,
    ):
        empty = {} if missing == "metrics" else None

        result = JD.can_send_metrics(**self._args(tmp_path, **{missing: empty}))

        assert result is False
        assert mock_logger["warning"] == [
            f"{expected_warning}, skipping JD metrics send"
        ]
