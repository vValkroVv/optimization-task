"""Tests for the SubprocessManager class."""

import subprocess
import signal
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from node_runner.src.subprocess_manager import (
    SubprocessManager,
    SubprocessManagerError,
)

pytestmark = pytest.mark.unit

# ============================================================================
# Constants for test configuration
# ============================================================================

DEFAULT_TEST_COMMAND = ["echo", "test"]
DEFAULT_TEST_CWD = "/tmp"
DEFAULT_LOG_FILENAME = "run.log"
TEST_RUN_NAME = "test_run"
UNICODE_CONTENT = "Привет мир 🌍\nHello world\n"
ERROR_LOG_PREFIX = "error_"
UNICODE_LOG_FILENAME = "unicode.log"
RESOURCE_LOG_FILENAME = "resource.log"

# Exit code constants
EXIT_CODE_SUCCESS = 0
EXIT_CODE_GENERIC_ERROR = 1
EXIT_CODE_UNKNOWN_ERROR = -1
EXIT_CODE_OUT_OF_RANGE = 255
EXIT_CODE_SIGINT = 130  # 128 + 2
EXIT_CODE_SIGKILL = 137  # 128 + 9
EXIT_CODE_SIGSEGV = 139  # 128 + 11

# Test data constants
TEST_LOG_CONTENT = "LINE 1\nsber failed\n"
TEST_PROCESS_NAME = "test-process"
COMPLEX_SCRIPT_NAME = "script.py"
FAILING_PROCESS_SCRIPT = "failing_process.py"
RESOURCE_INTENSIVE_SCRIPT = "resource_intensive.py"
DATA_PROCESSING_RUN_NAME = "data-processing"

# Signal descriptions
SIGINT_DESCRIPTION = "SIGINT (Ctrl+C)"
SIGKILL_DESCRIPTION = "SIGKILL (OOM)"
SIGSEGV_DESCRIPTION = "SIGSEGV (segfault)"
UNKNOWN_ERROR_DESCRIPTION = "unknown error"
OUT_OF_RANGE_DESCRIPTION = "out of range"

# ============================================================================
# Mock classes for testing
# ============================================================================


class MockProcess:
    """Mock implementation of a subprocess for testing."""

    def __init__(self, poll_result: Optional[int] = None, returncode: int = 0):
        self._poll_result = poll_result
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self.raise_timeout_once = False

    def poll(self) -> Optional[int]:
        return self._poll_result

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: Optional[float] = None) -> int:
        self.wait_calls += 1
        if self.raise_timeout_once:
            self.raise_timeout_once = False
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        self._poll_result = self.returncode
        return self.returncode


# ============================================================================
# Helper functions
# ============================================================================


def create_log_file_with_content(tmp_path: Path, filename: str, content: str) -> Path:
    """Helper to create a log file with specified content."""
    log_path = tmp_path / filename
    log_path.write_text(content, encoding="utf-8")
    return log_path


def assert_log_contains_messages(
    log_capture: Dict[str, list], expected_keywords: list[str], log_level: str = "info"
) -> None:
    """Helper to assert that log contains expected keywords."""
    log_messages = " ".join(log_capture[log_level])
    for keyword in expected_keywords:
        assert keyword in log_messages, f"Keyword '{keyword}' not found in logs"


# ============================================================================
# Pytest fixtures
# ============================================================================


@pytest.fixture
def mock_process() -> MockProcess:
    """Fixture providing a mock process instance."""
    return MockProcess()


@pytest.fixture
def mock_process_with_error() -> MockProcess:
    """Fixture providing a mock process that will raise TimeoutExpired."""
    process = MockProcess()
    process.raise_timeout_once = True
    return process


@pytest.fixture
def mock_log_capture(monkeypatch: pytest.MonkeyPatch) -> Dict[str, list]:
    """Fixture capturing log messages."""
    logs: Dict[str, list] = {"info": [], "error": []}

    def _capture_info(msg: str) -> None:
        logs["info"].append(msg)

    def _capture_error(msg: str) -> None:
        logs["error"].append(msg)

    monkeypatch.setattr("node_runner.src.subprocess_manager.Logger.info", _capture_info)
    monkeypatch.setattr(
        "node_runner.src.subprocess_manager.Logger.error", _capture_error
    )

    return logs


@pytest.fixture
def subprocess_manager_with_log(tmp_path: Path) -> SubprocessManager:
    """Fixture providing SubprocessManager with log file."""
    log_path = tmp_path / DEFAULT_LOG_FILENAME
    return SubprocessManager(DEFAULT_TEST_COMMAND, log_path, DEFAULT_TEST_CWD)


@pytest.fixture
def subprocess_manager_without_log() -> SubprocessManager:
    """Fixture providing SubprocessManager without log file."""
    return SubprocessManager(DEFAULT_TEST_COMMAND, None, DEFAULT_TEST_CWD)


@pytest.fixture
def subprocess_manager_factory():
    """Factory fixture for creating SubprocessManager with custom parameters."""

    def _create_manager(
        command: list[str] = None,
        log_path: Optional[Path] = None,
        cwd: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> SubprocessManager:
        return SubprocessManager(
            command or DEFAULT_TEST_COMMAND,
            log_path,
            cwd or DEFAULT_TEST_CWD,
            run_name=run_name,
        )

    return _create_manager


@pytest.fixture
def mock_popen(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Fixture mocking subprocess.Popen and capturing calls."""
    calls: Dict[str, Any] = {}
    mock_proc = MockProcess()

    def _mock_popen(cmd, stdout=None, stderr=None, cwd=None, **kwargs):
        calls["cmd"] = cmd
        calls["stdout"] = stdout
        calls["stderr"] = stderr
        calls["cwd"] = cwd
        calls["kwargs"] = kwargs
        return mock_proc

    monkeypatch.setattr(subprocess, "Popen", _mock_popen)
    calls["process"] = mock_proc
    return calls


class TestSubprocessManagerLogTail:
    """Tests for log file tailing functionality."""

    def test_return_last_lines_no_file(
        self, subprocess_manager_without_log: SubprocessManager
    ) -> None:
        """Test handling when no log file is specified."""
        result = subprocess_manager_without_log._return_last_lines_of_log_file(None)
        assert result == "No log file specified"

    def test_return_last_lines_missing_file(
        self,
        subprocess_manager_without_log: SubprocessManager,
        tmp_path: Path,
    ) -> None:
        """Test handling when log file does not exist."""
        missing_file = tmp_path / "missing.log"
        result = subprocess_manager_without_log._return_last_lines_of_log_file(
            missing_file
        )
        assert result == f"Log file {missing_file} does not exist"

    def test_return_last_lines_normalizes_and_replaces_sber(
        self,
        subprocess_manager_with_log: SubprocessManager,
        tmp_path: Path,
    ) -> None:
        """Test log normalization and text replacement."""
        log_path = tmp_path / DEFAULT_LOG_FILENAME
        log_content = "LINE 1\nsber failed\n"
        log_path.write_text(log_content, encoding="utf-8")

        lines_to_return = 10
        result = subprocess_manager_with_log._return_last_lines_of_log_file(
            log_path, lines_to_return
        )

        # Normalized to lowercase
        assert "line 1" in result
        # 'sber' replaced with 'papaya'
        assert "papaya failed" in result
        # Original text not present
        assert "sber" not in result


class TestSubprocessManagerStart:
    """Tests for starting subprocesses."""

    def test_process_starts_successfully_with_log_file(
        self,
        mock_popen: Dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that a process can be successfully started with log file."""
        log_path = tmp_path / DEFAULT_LOG_FILENAME
        python_command = ["python", "-c", "print('Hello, World!')"]
        manager = SubprocessManager(python_command, log_path, DEFAULT_TEST_CWD)

        # Start the process
        manager.start()

        # Verify process was started
        assert manager.process is not None
        assert manager.is_alive() is True

        # Verify command was passed correctly
        assert mock_popen["cmd"] == python_command
        assert mock_popen["cwd"] == DEFAULT_TEST_CWD

        # Cleanup
        manager._cleanup()

    def test_process_starts_successfully_without_log_file(
        self,
        mock_popen: Dict[str, Any],
    ) -> None:
        """Test that a process can be successfully started without log file."""
        manager = SubprocessManager(DEFAULT_TEST_COMMAND, None, DEFAULT_TEST_CWD)

        # Start the process
        manager.start()

        # Verify process was started
        assert manager.process is not None
        assert manager.is_alive() is True

        # Verify command was passed correctly
        assert mock_popen["cmd"] == DEFAULT_TEST_COMMAND
        assert mock_popen["cwd"] == DEFAULT_TEST_CWD

        # Cleanup
        manager._cleanup()

    def test_custom_env_passed_to_popen_with_log_file(
        self,
        mock_popen: Dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that custom env is forwarded to Popen when log file is used."""
        log_path = tmp_path / DEFAULT_LOG_FILENAME
        custom_env = {"PYTHONPATH": "/custom/path", "MY_VAR": "value"}
        manager = SubprocessManager(
            DEFAULT_TEST_COMMAND, log_path, DEFAULT_TEST_CWD, env=custom_env
        )

        manager.start()

        assert mock_popen["kwargs"].get("env") == custom_env
        manager._cleanup()

    def test_custom_env_passed_to_popen_without_log_file(
        self,
        mock_popen: Dict[str, Any],
    ) -> None:
        """Test that custom env is forwarded to Popen when no log file is used."""
        custom_env = {"PYTHONPATH": "/custom/path"}
        manager = SubprocessManager(
            DEFAULT_TEST_COMMAND, None, DEFAULT_TEST_CWD, env=custom_env
        )

        manager.start()

        assert mock_popen["kwargs"].get("env") == custom_env
        manager._cleanup()

    def test_default_env_is_none(self) -> None:
        """Test that default env is None (inherit from parent process)."""
        manager = SubprocessManager(DEFAULT_TEST_COMMAND, None, DEFAULT_TEST_CWD)
        assert manager.env is None

    def test_process_with_run_name_logs_correctly(
        self,
        mock_popen: Dict[str, Any],
        mock_log_capture: Dict[str, list],
    ) -> None:
        """Test that process with run name is identifiable in logs."""
        run_name = "test-process"
        manager = SubprocessManager(
            DEFAULT_TEST_COMMAND, None, DEFAULT_TEST_CWD, run_name=run_name
        )

        manager.start()
        manager.wait_for_completion()

        # Verify run name appears in log messages
        log_messages = " ".join(mock_log_capture["info"])
        assert run_name in log_messages

        # Cleanup
        manager._cleanup()


class TestSubprocessManagerStop:
    """Tests for stopping subprocesses."""

    def test_can_gracefully_stop_running_process(
        self,
        mock_process: MockProcess,
        tmp_path: Path,
    ):
        """Test that a running process can be gracefully stopped."""
        log_path = tmp_path / "run.log"
        manager = SubprocessManager(["long_running_task"], log_path, None)
        manager._log_fp = log_path.open("a", encoding="utf-8")
        manager.process = mock_process

        # Process should be alive before stop
        assert manager.is_alive() is True

        # Gracefully stop the process
        manager.graceful_stop()

        # Verify process was terminated
        assert mock_process.terminated is True
        assert manager.exit_code == 0
        assert manager._log_fp is None  # Log file should be closed

        # Process should not be alive after stop
        assert manager.is_alive() is False

    def test_graceful_stop_falls_back_to_kill_on_timeout(
        self,
        mock_process: MockProcess,
        tmp_path: Path,
    ):
        """Test that graceful stop kills process if it doesn't terminate in time."""
        log_path = tmp_path / "run.log"
        manager = SubprocessManager(["stubborn_task"], log_path, None)
        manager._log_fp = log_path.open("a", encoding="utf-8")

        # Mock process to timeout on wait()
        mock_process.raise_timeout_once = True
        manager.process = mock_process

        # Gracefully stop should handle timeout
        manager.graceful_stop()

        # Verify both terminate and kill were called
        assert mock_process.terminated is True  # First try terminate
        assert mock_process.killed is True  # Then kill after timeout
        assert manager.exit_code == 0
        assert manager._log_fp is None

    def test_can_force_kill_running_process(
        self,
        mock_process: MockProcess,
        tmp_path: Path,
    ):
        """Test that a running process can be force killed."""
        log_path = tmp_path / "run.log"
        manager = SubprocessManager(["unresponsive_task"], log_path, None)
        manager._log_fp = log_path.open("a", encoding="utf-8")
        manager.process = mock_process

        # Force kill the process
        manager.kill()

        # Verify process was killed (not just terminated)
        assert mock_process.killed is True
        assert manager.exit_code == 0
        assert manager._log_fp is None

        # Process should not be alive
        assert manager.is_alive() is False

    def test_signal_process_group_no_process_is_noop(self):
        manager = SubprocessManager(["echo"], None, None)
        manager.process = None
        manager._signal_process_group(signal.SIGTERM)

    def test_signal_process_group_uses_killpg(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        manager = SubprocessManager(["echo"], None, None)

        class _Process:
            pid = 12345

        manager.process = _Process()
        called: Dict[str, Any] = {}

        monkeypatch.setattr(
            "node_runner.src.subprocess_manager.os.getpgid", lambda _pid: 777
        )

        def _killpg(process_group_id, sig):
            called["process_group_id"] = process_group_id
            called["signal"] = sig

        monkeypatch.setattr("node_runner.src.subprocess_manager.os.killpg", _killpg)
        manager._signal_process_group(signal.SIGTERM)
        assert called["process_group_id"] == 777
        assert called["signal"] == signal.SIGTERM

    def test_stop_methods_do_nothing_when_process_not_running(
        self,
        mock_process: MockProcess,
    ):
        """Test that stop methods are safe to call on non-running process."""
        manager = SubprocessManager(["echo"], None, None)
        mock_process._poll_result = 0  # Process already finished
        manager.process = mock_process

        # Both stop methods should be safe to call
        manager.graceful_stop()
        manager.kill()

        # Exit code should remain None (process wasn't running)
        assert manager.exit_code is None
        # Process should not have been terminated or killed
        assert not mock_process.terminated
        assert not mock_process.killed


class TestSubprocessManagerResultHandling:
    """Tests for handling subprocess results and errors."""

    def test_can_wait_for_process_completion(
        self,
        mock_process: MockProcess,
        mock_log_capture: Dict[str, list],
    ):
        """Test that waiting for process completion works correctly."""
        run_name = "data-processing-job"
        manager = SubprocessManager(["process_data.py"], None, None, run_name=run_name)
        manager.process = mock_process

        # Wait for process to complete
        manager.wait_for_completion()

        # Verify exit code is set
        assert manager.exit_code == 0
        # Verify completion is logged with run name
        assert any(run_name in msg for msg in mock_log_capture["info"])
        # Process should not be alive after completion
        assert manager.is_alive() is False

    def test_waiting_for_completion_is_safe_without_process(
        self,
        mock_log_capture: Dict[str, list],
    ):
        """Test that waiting for completion doesn't crash when no process is running."""
        manager = SubprocessManager(["echo"], None, None, run_name="test")

        # Should not raise any exception
        manager.wait_for_completion()

        # Exit code should remain None
        assert manager.exit_code is None
        # No log messages should be generated
        assert len(mock_log_capture["info"]) == 0

    def test_error_is_raised_when_process_fails(
        self,
        mock_log_capture: Dict[str, list],
    ):
        """Test that an error is raised when process exits with failure code."""
        run_name = "failing-task"
        manager = SubprocessManager(
            ["failing_script.py"], None, None, run_name=run_name
        )
        manager.exit_code = 1  # Simulate process failure

        # Should raise appropriate error
        with pytest.raises(SubprocessManagerError, match=f"Process {run_name} failed"):
            manager.raise_for_exit_code()

        # Note: raise_for_exit_code() only logs errors when log_file exists
        # Without log_file, it just raises exception without logging
        # So we don't check for log messages here

    def test_no_error_is_raised_when_process_succeeds(
        self,
        mock_log_capture: Dict[str, list],
    ):
        """Test that no error is raised when process exits successfully."""
        manager = SubprocessManager(
            ["successful_task.py"], None, None, run_name="success"
        )
        manager.exit_code = 0

        # Should not raise any exception
        manager.raise_for_exit_code()

        # No error should be logged
        assert len(mock_log_capture["error"]) == 0

    def test_error_message_includes_log_tail_when_available(
        self,
        mock_log_capture: Dict[str, list],
        tmp_path: Path,
    ):
        """Test that error messages include relevant log tail when log file exists."""
        log_path = tmp_path / "error.log"
        error_message = "Segmentation fault at address 0x1234"
        log_path.write_text(
            f"Starting process...\n{error_message}\nProcess crashed", encoding="utf-8"
        )

        run_name = "crashing-process"
        manager = SubprocessManager(["crash.py"], log_path, None, run_name=run_name)
        manager.exit_code = 139  # SIGSEGV

        # Error should include log tail (note: text is normalized to lowercase)
        with pytest.raises(
            SubprocessManagerError, match="segmentation fault at address 0x1234"
        ):
            manager.raise_for_exit_code()

        # Error should be logged with tail information
        assert any("log tail" in msg for msg in mock_log_capture["error"])

    def test_successful_process_completion_is_logged_correctly(
        self,
        mock_log_capture: Dict[str, list],
        tmp_path: Path,
    ):
        """Test that successful process completion generates appropriate logs."""
        log_path = tmp_path / "success.log"
        run_name = "successful-job"
        manager = SubprocessManager(["job.py"], log_path, None, run_name=run_name)
        manager.exit_code = 0

        manager.log_error()

        # Success should be logged
        assert any("completed successfully" in msg for msg in mock_log_capture["info"])
        # No errors should be logged
        assert len(mock_log_capture["error"]) == 0

    def test_failed_process_generates_error_logs(
        self,
        mock_log_capture: Dict[str, list],
        tmp_path: Path,
    ):
        """Test that failed process generates appropriate error logs."""
        log_path = tmp_path / "failure.log"
        error_content = "Out of memory error\nCannot allocate buffer"
        log_path.write_text(error_content, encoding="utf-8")

        run_name = "memory-intensive-job"
        manager = SubprocessManager(
            ["memory_hog.py"], log_path, None, run_name=run_name
        )
        manager.exit_code = 137  # SIGKILL (OOM)

        manager.log_error()

        # Error should be logged
        assert any("failed" in msg.lower() for msg in mock_log_capture["error"])
        # Log tail should be included in error message
        error_logs = " ".join(mock_log_capture["error"])
        assert "memory" in error_logs.lower() or "error" in error_logs.lower()

    def test_is_alive_accurately_reports_process_state(self, mock_process: MockProcess):
        """Test that is_alive method correctly reports process liveness."""
        manager = SubprocessManager(["monitored_process.py"], None, None)

        # Initially no process
        assert manager.is_alive() is False

        # Process starts
        manager.process = mock_process
        assert manager.is_alive() is True

        # Process finishes
        mock_process._poll_result = 0
        assert manager.is_alive() is False

        # Edge case: process object exists but poll returns None (zombie?)
        mock_process._poll_result = None
        assert manager.is_alive() is True


class TestSubprocessManagerRealWorldScenarios:
    """Tests for real-world usage scenarios of SubprocessManager."""

    def test_long_running_process_with_periodic_status_checks(
        self,
        mock_popen: Dict[str, Any],
        mock_process: MockProcess,
        mock_log_capture: Dict[str, list],
    ):
        """Test monitoring a long-running process with periodic status checks."""
        run_name = "training-job"
        manager = SubprocessManager(
            ["train_model.py", "--epochs", "100"], None, "/workspace", run_name=run_name
        )

        # Start the process (uses mock_popen via fixture)
        manager.start()
        assert manager.process is mock_popen["process"]
        assert manager.is_alive() is True

        # Simulate periodic status checks
        for _ in range(3):
            assert manager.is_alive() is True

        # Complete the process
        manager.wait_for_completion()
        assert manager.exit_code == 0
        assert manager.is_alive() is False

        # Verify completion is logged
        assert any(run_name in msg for msg in mock_log_capture["info"])

    def test_process_with_standard_output_capturing(
        self,
        mock_popen: Dict[str, Any],
        tmp_path: Path,
    ):
        """Test process that generates standard output."""
        log_path = tmp_path / "output.log"
        manager = SubprocessManager(
            [
                "python",
                "-c",
                "import sys; print('Progress: 50%'); sys.stderr.write('Warning: low memory\\n')",
            ],
            log_path,
            "/tmp",
        )

        manager.start()
        manager.wait_for_completion()

        # Verify process was started with correct redirects
        assert mock_popen["stdout"] is mock_popen["stderr"]
        assert mock_popen["stdout"] is not None

    def test_graceful_shutdown_of_critical_process(
        self,
        mock_process: MockProcess,
        tmp_path: Path,
        mock_log_capture: Dict[str, list],
    ):
        """Test graceful shutdown of a critical process that must clean up resources."""
        log_path = tmp_path / "critical.log"
        run_name = "database-migration"
        manager = SubprocessManager(
            ["migrate_database.py"], log_path, "/data", run_name=run_name
        )
        manager._log_fp = log_path.open("a", encoding="utf-8")
        manager.process = mock_process

        # Process is already running (we set it manually)
        assert manager.is_alive() is True

        # Request graceful shutdown
        manager.graceful_stop()

        # Verify graceful termination was attempted
        assert mock_process.terminated is True
        assert manager.exit_code == 0
        assert manager._log_fp is None

        # Note: graceful_stop() doesn't log completion, only wait_for_completion() does
        # So we don't check for log messages here

    def test_handling_of_unresponsive_process(
        self,
        mock_process: MockProcess,
        tmp_path: Path,
        mock_log_capture: Dict[str, list],
    ):
        """Test handling of a process that becomes unresponsive."""
        log_path = tmp_path / "hang.log"
        run_name = "unresponsive-service"
        manager = SubprocessManager(
            ["hang_server.py"], log_path, "/srv", run_name=run_name
        )
        manager._log_fp = log_path.open("a", encoding="utf-8")

        # Mock process to timeout (simulating unresponsiveness)
        mock_process.raise_timeout_once = True
        manager.process = mock_process

        # Attempt graceful stop (should fall back to kill)
        manager.graceful_stop()

        # Verify both terminate and kill were called
        assert mock_process.terminated is True
        assert mock_process.killed is True
        assert manager.exit_code == 0

        # Note: graceful_stop() doesn't log completion
        # So we don't check for log messages here

    def test_concurrent_process_management(
        self,
        mock_popen: Dict[str, Any],
    ):
        """Test managing multiple processes concurrently."""
        processes = []

        # Start multiple processes
        for i in range(3):
            manager = SubprocessManager(
                [f"worker_{i}.py", f"--id={i}"],
                None,
                f"/workers/{i}",
                run_name=f"worker-{i}",
            )
            manager.start()
            processes.append(manager)

        # Verify all processes were started
        for manager in processes:
            assert manager.process is not None
            assert manager.is_alive() is True

        # Cleanup
        for manager in processes:
            manager._cleanup()

    def test_process_with_custom_working_directory(
        self,
        mock_popen: Dict[str, Any],
        tmp_path: Path,
    ):
        """Test process that requires specific working directory."""
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        manager = SubprocessManager(
            ["./build.sh", "--release"],
            None,
            str(working_dir),
            run_name="build-process",
        )

        manager.start()

        # Verify process was started in correct directory
        assert mock_popen["cwd"] == str(working_dir)

        manager._cleanup()

    def test_error_handling_for_missing_executable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mock_log_capture: Dict[str, list],
    ):
        """Test error handling when process executable is missing."""

        # Mock Popen to raise FileNotFoundError
        def mock_popen_raises(*args, **kwargs):
            raise FileNotFoundError(
                "[Errno 2] No such file or directory: 'nonexistent'"
            )

        monkeypatch.setattr("subprocess.Popen", mock_popen_raises)

        manager = SubprocessManager(
            ["nonexistent_executable", "--option"],
            tmp_path / "error.log",
            "/tmp",
            run_name="missing-exe",
        )

        # Starting should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            manager.start()

        # Process should not be set
        assert manager.process is None
        # Exit code should not be set
        assert manager.exit_code is None

        # Attempting to wait should be safe (no process)
        manager.wait_for_completion()
        assert manager.exit_code is None

        # Note: start() doesn't log errors, it just raises exceptions
        # So we don't check for log messages here

    def test_process_with_complex_command_line(
        self,
        mock_popen: Dict[str, Any],
    ) -> None:
        """Test process with complex command line arguments."""
        complex_command = [
            "python",
            COMPLEX_SCRIPT_NAME,
            "--input",
            "data.csv",
            "--output",
            "results.json",
            "--verbose",
            "--workers",
            "4",
            "--batch-size",
            "32",
        ]

        manager = SubprocessManager(
            complex_command, None, "/project", run_name=DATA_PROCESSING_RUN_NAME
        )

        manager.start()

        # Verify complex command was passed correctly
        assert mock_popen["cmd"] == complex_command

        manager._cleanup()

    def test_resource_cleanup_after_process_completion(
        self,
        mock_process: MockProcess,
        tmp_path: Path,
    ) -> None:
        """Test that resources are properly cleaned up after process completion."""
        log_path = tmp_path / RESOURCE_LOG_FILENAME

        # Open log file manually to track it
        log_file = log_path.open("a", encoding="utf-8")
        assert not log_file.closed

        manager = SubprocessManager([RESOURCE_INTENSIVE_SCRIPT], log_path, "/tmp")
        manager._log_fp = log_file
        manager.process = mock_process

        # Complete the process
        manager.wait_for_completion()

        # Verify resources are cleaned up
        assert manager._log_fp is None  # File pointer should be closed
        assert manager.exit_code == EXIT_CODE_SUCCESS

        # Original file should be closed
        assert log_file.closed


class TestSubprocessManagerEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_return_last_lines_with_unicode_content(
        self,
        subprocess_manager_with_log: SubprocessManager,
        tmp_path: Path,
    ) -> None:
        """Test log tailing with Unicode content."""
        log_path = create_log_file_with_content(
            tmp_path, UNICODE_LOG_FILENAME, UNICODE_CONTENT
        )

        lines_to_return = 5
        result = subprocess_manager_with_log._return_last_lines_of_log_file(
            log_path, lines_to_return
        )

        # Result is already normalized to lowercase by _return_last_lines_of_log_file
        assert "привет мир" in result  # Already lowercase
        assert "hello world" in result  # Already lowercase

    def test_start_with_empty_command_list(
        self,
        mock_popen: Dict[str, Any],
    ):
        """Test starting process with empty command list."""
        manager = SubprocessManager([], None, "/tmp")

        manager.start()

        assert mock_popen["cmd"] == []

    @pytest.mark.parametrize(
        "exit_code, expected_signal",
        [
            (EXIT_CODE_UNKNOWN_ERROR, UNKNOWN_ERROR_DESCRIPTION),
            (EXIT_CODE_OUT_OF_RANGE, OUT_OF_RANGE_DESCRIPTION),
            (EXIT_CODE_SIGINT, SIGINT_DESCRIPTION),
            (EXIT_CODE_SIGKILL, SIGKILL_DESCRIPTION),
            (EXIT_CODE_SIGSEGV, SIGSEGV_DESCRIPTION),
        ],
    )
    def test_error_handling_for_various_exit_codes(
        self,
        exit_code: int,
        expected_signal: str,
        mock_log_capture: Dict[str, list],
        tmp_path: Path,
    ) -> None:
        """Test error handling for various process exit codes and signals."""
        run_name = f"process-exit-{exit_code}"
        log_filename = f"{ERROR_LOG_PREFIX}{exit_code}.log"
        log_content = f"Process failed with exit code {exit_code}"
        log_path = create_log_file_with_content(tmp_path, log_filename, log_content)

        # With log file, errors should be logged
        manager = SubprocessManager(
            [FAILING_PROCESS_SCRIPT], log_path, None, run_name=run_name
        )
        manager.exit_code = exit_code

        # Error should be raised
        with pytest.raises(SubprocessManagerError):
            manager.raise_for_exit_code()

        # Error should be logged when log file exists
        error_logs = " ".join(mock_log_capture["error"])
        assert run_name in error_logs
        assert "failed" in error_logs.lower()
        assert "log tail" in error_logs.lower()

    def test_wait_for_completion_with_already_finished_process(
        self,
        mock_process: MockProcess,
        mock_log_capture: Dict[str, list],
    ):
        """Test waiting for completion when process is already finished.

        Even though the process is already finished (poll_result=0),
        wait_for_completion() still calls wait() to ensure the exit code
        is properly retrieved and cleanup is performed.
        """
        manager = SubprocessManager(["echo"], None, None, run_name="test")
        mock_process._poll_result = 0  # Already finished
        mock_process.returncode = 0
        manager.process = mock_process

        manager.wait_for_completion()

        assert manager.exit_code == 0
        # wait() is always called to get exit code and perform cleanup
        assert mock_process.wait_calls == 1
        # Verify log message was recorded
        assert any("test" in msg for msg in mock_log_capture["info"])

    def test_graceful_stop_with_already_closed_log_file(
        self,
        mock_process: MockProcess,
        tmp_path: Path,
    ):
        """Test graceful stop when log file is already closed."""
        log_path = tmp_path / "test.log"
        manager = SubprocessManager(["echo"], log_path, None)
        manager._log_fp = log_path.open("a", encoding="utf-8")
        manager._log_fp.close()  # Simulate already closed
        manager.process = mock_process

        manager.graceful_stop()

        assert mock_process.terminated is True
        assert manager.exit_code == 0

    def test_log_error_with_missing_log_file(
        self,
        mock_log_capture: Dict[str, list],
        tmp_path: Path,
    ):
        """Test logging error when log file doesn't exist."""
        missing_log = tmp_path / "missing.log"
        manager = SubprocessManager(["echo"], missing_log, None, run_name="test")
        manager.exit_code = 1

        manager.log_error()

        # Should still log error without crashing
        assert any(
            "process test failed" in msg.lower() for msg in mock_log_capture["error"]
        )
