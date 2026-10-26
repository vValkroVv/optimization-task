import os
import signal
import subprocess
import time
from base64 import b64decode
from collections import deque
from pathlib import Path
from typing import Optional, TextIO

from .logger import Logger


class SubprocessManagerError(Exception):
    pass


class SubprocessManager:
    def __init__(
        self,
        bash_cmd: list[str],
        log_file: Optional[Path],
        cwd: Optional[str],
        run_name: str = "",
        env: Optional[dict[str, str]] = None,
    ):
        self.bash_cmd = bash_cmd
        self.process: Optional[subprocess.Popen] = None
        self.log_file = log_file
        self.cwd = cwd
        self.exit_code: Optional[int] = None
        self.run_name = run_name
        self._log_fp: Optional[TextIO] = None
        self.env = env
        self._started_at: Optional[float] = None

    def _return_last_lines_of_log_file(self, filepath: Optional[Path], n: int = 100):
        if not filepath:
            return "No log file specified"

        path = Path(filepath)
        if path.exists():
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                last_lines = deque[str](f, maxlen=n)
            joined_lines = "\n".join(last_lines)
            joined_lines = joined_lines.lower().replace(
                b64decode(b"c2Jlcg==").decode("utf-8"), "papaya"
            )
            return joined_lines
        else:
            return f"Log file {filepath} does not exist"

    def start(self):
        if self.log_file:
            self._log_fp = self.log_file.open("a", encoding="utf-8", errors="ignore")
            self.process = subprocess.Popen(
                self.bash_cmd,
                stdout=self._log_fp,
                stderr=self._log_fp,
                cwd=self.cwd,
                env=self.env,
                start_new_session=True,
            )
        else:
            self.process = subprocess.Popen(
                self.bash_cmd,
                cwd=self.cwd,
                env=self.env,
                start_new_session=True,
            )
        self._started_at = time.monotonic()

    def _format_runtime(self) -> str:
        if self._started_at is None:
            return "unknown"
        elapsed = max(0.0, time.monotonic() - self._started_at)
        return f"{elapsed:.2f}s"

    def _cleanup(self):
        if self._log_fp and not self._log_fp.closed:
            self._log_fp.close()
        self._log_fp = None

    def _signal_process_group(self, sig: signal.Signals) -> None:
        if not self.process:
            return
        try:
            process_group_id = os.getpgid(self.process.pid)
            os.killpg(process_group_id, sig)
        except (AttributeError, ProcessLookupError):
            if sig == signal.SIGTERM:
                self.process.terminate()
            else:
                self.process.kill()

    def graceful_stop(self):
        if not self.process or self.process.poll() is not None:
            return

        try:
            self._signal_process_group(signal.SIGTERM)
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._signal_process_group(signal.SIGKILL)
            self.process.wait(timeout=20)
        finally:
            self.exit_code = self.process.returncode
            self._cleanup()

    def kill(self):
        if not self.process or self.process.poll() is not None:
            return
        self._signal_process_group(signal.SIGKILL)
        self.process.wait(timeout=20)
        self.exit_code = self.process.returncode
        self._cleanup()

    def wait_for_completion(self):
        if not self.process:
            return

        self.process.wait()
        self.exit_code = self.process.returncode
        self._cleanup()

        Logger.info(
            f"Process {self.run_name} completed with exit code {self.exit_code} in {self._format_runtime()}"
        )

    def is_successful(self):
        return self.exit_code == 0

    def raise_for_exit_code(self):
        if not self.is_successful():
            if self.log_file:
                log_tail = self._return_last_lines_of_log_file(self.log_file, 100)
                Logger.error(
                    f"Process {self.run_name} failed with log tail: {log_tail}"
                )
                raise SubprocessManagerError(log_tail)
            else:
                raise SubprocessManagerError(f"Process {self.run_name} failed")

    def log_error(self):
        if not self.is_successful():
            Logger.error(self._return_last_lines_of_log_file(self.log_file, 100))
            Logger.error(f"Process {self.run_name} failed")
        else:
            Logger.info(f"Process {self.run_name} completed successfully")

    def is_alive(self):
        return bool(self.process and self.process.poll() is None)
