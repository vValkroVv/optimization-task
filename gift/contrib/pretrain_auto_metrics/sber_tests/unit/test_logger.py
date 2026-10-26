from pathlib import Path
import logging

import pytest
from node_runner.src.logger import AnonymizingFormatter, Logger, create_stdout_logger

pytestmark = pytest.mark.unit


class TestLoggerSetup:
    def setup_method(self):
        Logger.logger = None
        Logger.is_slurm = False
        Logger.log_dir = None
        Logger.node_log_dir = None
        Logger.inference_log_file = None
        Logger.router_log_file = None
        Logger.autometrics_log_file = None
        Logger.convertation_log_file = None
        Logger.node_log_file = None
        Logger.obm_proxy_log_file = None

    def test_paths_are_initialized(self, tmp_path: Path):
        Logger.setup(str(tmp_path), "test", is_master=True)
        assert Logger.log_dir == Path(tmp_path, "logs")
        assert Logger.node_log_dir == Path(tmp_path, "logs", "test")
        assert Logger.inference_log_file == Path(
            tmp_path, "logs", "test", "inference.log"
        )
        assert Logger.node_log_file == Path(tmp_path, "logs", "test", "test.log")
        assert Logger.obm_proxy_log_file == Path(
            tmp_path, "logs", "test", "obm_proxy.log"
        )
        assert Logger.convertation_log_file == Path(
            tmp_path, "logs", "test", "convertation.log"
        )
        assert Logger.autometrics_log_file == Path(
            tmp_path, "logs", "test", "autometrics.log"
        )
        assert Logger.router_log_file == Path(tmp_path, "logs", "test", "router.log")

    def test_each_judge_logs_to_its_own_files(self, tmp_path: Path):
        Logger.setup(str(tmp_path), "test", is_master=True)
        node_log_dir = Path(tmp_path, "logs", "test")

        assert Logger.inference_log_file_for("evaluated") == Logger.inference_log_file
        assert Logger.router_log_file_for("evaluated") == Logger.router_log_file
        assert Logger.inference_log_file_for("org/judge") == (
            node_log_dir / "org_judge_inference.log"
        )
        assert Logger.router_log_file_for("org/judge") == (
            node_log_dir / "org_judge_router.log"
        )

    def test_error_is_noop_before_setup(self):
        Logger.error("no logger to write to")  # must not raise

    def test_setup_returns_early_when_logger_already_created(self, tmp_path: Path):
        Logger.setup(str(tmp_path), "test", is_master=False)
        first_logger = Logger.logger
        Logger.setup(str(tmp_path), "another", is_master=True)
        assert Logger.logger is first_logger

    def test_warning_and_error_clean_message_when_slurm(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        captured = {"warning": [], "error": []}

        class _Dummy:
            def info(self, _msg):
                pass

            def warning(self, msg):
                captured["warning"].append(msg)

            def error(self, msg):
                captured["error"].append(msg)

        monkeypatch.setattr(Logger, "logger", _Dummy())
        monkeypatch.setattr(Logger, "is_slurm", True)
        Logger.warning("sber warning")
        Logger.error("sber error")
        assert captured["warning"] == ["papaya warning"]
        assert captured["error"] == ["papaya error"]

    def test_info_cleans_message_when_slurm(self, monkeypatch: pytest.MonkeyPatch):
        captured = {"info": []}

        class _Dummy:
            def info(self, msg):
                captured["info"].append(msg)

        monkeypatch.setattr(Logger, "logger", _Dummy())
        monkeypatch.setattr(Logger, "is_slurm", True)
        Logger.info("sber info")
        assert captured["info"] == ["papaya info"]

    def test_message_not_cleaned_when_not_slurm(self, monkeypatch: pytest.MonkeyPatch):
        captured = {"info": []}

        class _Dummy:
            def info(self, msg):
                captured["info"].append(msg)

        monkeypatch.setattr(Logger, "logger", _Dummy())
        monkeypatch.setattr(Logger, "is_slurm", False)
        Logger.info("sber info")
        assert captured["info"] == ["sber info"]


class TestStdoutLoggerFactory:
    def test_anonymizing_formatter_cleans_formatted_record(self):
        formatter = AnonymizingFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="sber token",
            args=(),
            exc_info=None,
        )

        assert formatter.format(record) == "papaya token"

    def test_create_stdout_logger_replaces_handlers_and_disables_propagation(self):
        logger = logging.getLogger("sber-tests-stdout")
        logger.addHandler(logging.NullHandler())

        created = create_stdout_logger("sber-tests-stdout", level=logging.WARNING)

        assert created is logger
        assert created.level == logging.WARNING
        assert created.propagate is False
        assert len(created.handlers) == 1
        assert isinstance(created.handlers[0].formatter, AnonymizingFormatter)
