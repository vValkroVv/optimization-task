from abc import ABC
import shlex
import shutil
from typing import Optional
from enum import Enum
import os
import subprocess
import sys
import threading
from multiprocessing.pool import ApplyResult
from typing import Dict, IO
import logging
from datetime import datetime


class JobStatus(Enum):
    SUCCESS = 0
    FAILED = 1
    RUNNING = 2
    PENDING = 3


class JobHandle(ABC):

    def status(self) -> JobStatus:
        ...

    def wait(self) -> None:
        ...


class LocalJob:
    def __init__(
            self,
            train_script_path: str | os.PathLike,
            config_path: str | os.PathLike,
            processes_per_worker: int,
            env_variables: Dict[str, str],
            log_dir: Optional[str | os.PathLike],
    ):
        self.train_script_path = str(train_script_path)
        self.config_path = str(config_path)
        self.processes_per_worker = processes_per_worker
        self.env_variables = env_variables
        self.log_dir = log_dir
        self._temp_paths: set[str] = set()

    def _setup_pytest_logger(self):
        pytest_log_path = os.path.join(self.log_dir, "pytest_log.txt")

        self.pytest_logger = logging.getLogger(f"pytest_job_{id(self)}")
        self.pytest_logger.setLevel(logging.INFO)

        self.pytest_logger.handlers.clear()

        file_handler = logging.FileHandler(pytest_log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt='%(asctime)s [PYTEST] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self.pytest_logger.addHandler(file_handler)
        self.pytest_logger.propagate = False

    def _log_pytest(self, level: str, message: str):
        if hasattr(self, 'pytest_logger'):
            getattr(self.pytest_logger, level.lower())(message)

    def run(self) -> None:
        self._log_pytest("info", f"Starting ML job with config: {self.config_path}")
        self._log_pytest("info", f"Processes per worker: {self.processes_per_worker}")
        self._log_pytest("info", f"Log directory: {self.log_dir}")

        try:
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                self._setup_pytest_logger()
                stderr_file = open(os.path.join(self.log_dir, "stderr.txt"), "w")
                stdout_file = open(os.path.join(self.log_dir, "stdout.txt"), "w")
                self._log_pytest("info", "Created ML job log files: stderr.txt, stdout.txt")
            else:
                stderr_file = subprocess.DEVNULL
                stdout_file = subprocess.DEVNULL
                self._log_pytest("warning", "No log directory specified, ML logs will be discarded")

            train_script = self._make_torchrun_command()
            env_variables = self._make_env_variables()

            self._log_pytest("info", f"Executing command: {train_script}")

            process = subprocess.Popen(
                train_script,
                env=env_variables,
                shell=True,
                stdout=stdout_file,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            self._log_pytest("info", f"ML process started with PID: {process.pid}")

            line_count = 0
            for line in process.stderr:
                line_count += 1
                if stderr_file != subprocess.DEVNULL:
                    stderr_file.write(line)
                    stderr_file.flush()

                sys.stdout.write(f"[TEST] {line}")
                sys.stdout.flush()

                if line_count % 100 == 0:
                    self._log_pytest("debug", f"Processed {line_count} stderr lines from ML job")

            return_code = process.wait()

            self._log_pytest("info", f"ML process finished with return code: {return_code}")
            self._log_pytest("info", f"Total stderr lines processed: {line_count}")

            if return_code != 0:
                error_msg = f"Job exited with non-zero exit code: {return_code}. Config path: {self.config_path}"
                self._log_pytest("error", error_msg)
                raise ValueError(error_msg)
            else:
                self._log_pytest("info", "ML job completed successfully")

        except Exception as e:
            self._log_pytest("error", f"Exception occurred: {str(e)}")
            if 'process' in locals() and process.poll() is None:
                self._log_pytest("warning", "Terminating ML process due to exception")
                process.terminate()
                process.wait()
                self._log_pytest("info", "ML process terminated")
            raise e
        finally:
            if self.log_dir and stderr_file and stderr_file != subprocess.DEVNULL:
                stderr_file.close()
                self._log_pytest("info", "Closed stderr.txt file")
            if self.log_dir and stdout_file and stdout_file != subprocess.DEVNULL:
                stdout_file.close()
                self._log_pytest("info", "Closed stdout.txt file")

            self._cleanup_temp_paths()
            self._log_pytest("info", "LocalJob.run() completed")

    def _make_env_variables(self):
        dttm = str(datetime.now().strftime('%Y%m%d%H%M%S%f'))
        pytorch_cache = f"/tmp/.cache/{dttm}"
        triton_cache = f"/tmp/.cache/triton/{dttm}"
        temp_dir = f"/tmp/{dttm}"
        self._temp_paths.update({pytorch_cache, triton_cache, temp_dir})
        cuda_visible_devices = {
            "CUDA_VISIBLE_DEVICES": ",".join(
                [str(i) for i in range(self.processes_per_worker)]
            )
        }
        temp_env = {
            "PYTORCH_KERNEL_CACHE_PATH": pytorch_cache,
            "TRITON_CACHE_DIR": triton_cache,
            "TRITON_HOME": triton_cache,
            "TEMPDIR": temp_dir,
        }
        env_vars = self.env_variables | temp_env | cuda_visible_devices
        self._log_pytest("debug", f"CUDA_VISIBLE_DEVICES set to: {cuda_visible_devices['CUDA_VISIBLE_DEVICES']}")
        return env_vars

    def _cleanup_temp_paths(self):
        for temp_path in self._temp_paths:
            normalized = os.path.abspath(temp_path)
            if not normalized.startswith("/tmp/") or normalized == "/tmp":
                continue
            shutil.rmtree(normalized, ignore_errors=True)

    def _make_torchrun_command(self):
        command = f"torchrun --nproc_per_node {self.processes_per_worker} {self.train_script_path} {self.config_path} use_clearml=False"
        self._log_pytest("debug", f"Generated torchrun command: {command}")
        return command


class LocalJobHandle(JobHandle):
    def __init__(self, handle: ApplyResult):
        self._handle = handle

    def status(self) -> JobStatus:
        if self._handle.ready():
            try:
                self._handle.get(timeout=0)
                return JobStatus.SUCCESS
            except:
                return JobStatus.FAILED
        else:
            return JobStatus.RUNNING

    def wait(self):
        return self._handle.wait()


class VerboseLocalJobHandle(LocalJobHandle):
    def __init__(self, apply_result, log_dir=None):
        super().__init__(apply_result)
        self._log_dir = log_dir
        self._last_log_position = {}

    def get(self, timeout=None):
        print(f"[EXECUTOR] Starting nested job via LocalJob...", flush=True)

        iteration = 0
        while not self.is_ready():
            iteration += 1
            print(f"[EXECUTOR] Nested job running... iteration {iteration}", flush=True)

            self._stream_all_logs()
            time.sleep(3)

        result = super().get(timeout)
        print(f"[EXECUTOR] Nested job completed with result: {result}", flush=True)

        self._stream_all_logs(final=True)
        return result

    def _stream_all_logs(self, final=False):
        if not self._log_dir or not os.path.exists(self._log_dir):
            return

        try:
            log_path = Path(self._log_dir)
            for log_file in log_path.rglob("*"):
                if log_file.is_file() and log_file.stat().st_size > 0:
                    self._stream_file(log_file, final)

        except Exception as e:
            print(f"[EXECUTOR] Error reading logs: {e}", flush=True)

    def _stream_file(self, log_file, final=False):
        try:
            file_path = str(log_file)

            start_pos = self._last_log_position.get(file_path, 0)

            with open(log_file, 'r') as f:
                f.seek(start_pos)
                new_content = f.read()

                if new_content:
                    print(f"[LOG {log_file.name}]:", flush=True)
                    print(new_content, flush=True)
                    print("-" * 40, flush=True)

                    self._last_log_position[file_path] = f.tell()

        except Exception as e:
            print(f"[EXECUTOR] Error streaming {log_file}: {e}", flush=True)
