from typing import Optional
import os
import subprocess
import shutil
from typing import Dict, Set
from threading import Lock, Condition
from concurrent.futures import Future
import socket
from datetime import datetime
from random import randint
from .job import JobStatus, JobHandle
from time import sleep

TCP_BARRIER_PORT = randint(2000, 65000)
class ResourcePool:
    def __init__(self, total_resources: int):
        self._total_resources = total_resources
        self._available_indices = set(range(total_resources))
        self._lock = Lock()

    def acquire(self, count: int) -> Set[int]:
        while len(self._available_indices) < count:
            sleep(0.1) # TODO
        with self._lock:
            acquired = set(sorted(self._available_indices)[:count])
            self._available_indices -= acquired
        return acquired

    def release(self, indices: Set[int]):
        with self._lock:
            self._available_indices.update(indices)

    @property
    def available(self) -> int:
        with self._lock:
            return len(self._available_indices)

class LocalJob:
    def __init__(self,
        train_script_path: str | os.PathLike,
        config_path: str | os.PathLike,
        processes_per_worker: int,
        env_variables: Dict[str, str],
        log_dir: Optional[str | os.PathLike],
        gpu_indices: Set[int],
        resource_pool):  

        self.train_script_path = str(train_script_path)
        self.config_path = str(config_path)
        self.processes_per_worker = processes_per_worker
        self.env_variables = env_variables
        self.log_dir = log_dir
        self.gpu_indices = gpu_indices
        self._resource_pool = resource_pool
        self._temp_paths: set[str] = set()

    def run(self) -> None:
        try:
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                stdout = open(os.path.join(self.log_dir, "stdout.txt"), "w")
                stderr = open(os.path.join(self.log_dir, "stderr.txt"), "w")
            else:
                stdout = None
                stderr = None

            train_script = self._make_torchrun_command()
            env_variables = self._make_env_variables()
            # print(f"RUNNING subprocess for JOB: {self.config_path} with GPUs: {','.join(map(str, sorted(self.gpu_indices)))}")
            subprocess.run(
                train_script,
                env=env_variables,
                shell=True,
                check=True,
                stdout=stdout,
                stderr=stderr,
            )
            # print(f"FINISHED subprocess for JOB: {self.config_path} with GPUs: {','.join(map(str, sorted(self.gpu_indices)))}")
        except subprocess.CalledProcessError:
            print(f"ERROR RAISED {self.config_path}")
            # raise ValueError(f"Job exited with non-zero exit code. Config path: {self.config_path}")
        except Exception as e:
            print("EXCEPTION: {e}")
        finally:
            self._resource_pool.release(self.gpu_indices)
            # print(f"RELEASING RESOURCES: {self.config_path=} {self.gpu_indices=}")
            if self.log_dir:
                stdout.flush()
                stderr.flush()
                stdout.close()
                stderr.close()
            self._cleanup_temp_paths()

    def _make_env_variables(self):
        global TCP_BARRIER_PORT
        TCP_BARRIER_PORT += 1
        dttm = str(datetime.now().strftime('%Y%m%d%H%M%S%f'))
        pytorch_cache = f"/tmp/.cache/{dttm}"
        triton_cache = f"/tmp/.cache/triton/{dttm}"
        temp_dir = f"/tmp/{dttm}"
        self._temp_paths.update({pytorch_cache, triton_cache, temp_dir})
        return {
            **self.env_variables,
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, sorted(self.gpu_indices))),
            "META_LOGS_DIR": self.log_dir,
            "PYTORCH_KERNEL_CACHE_PATH": pytorch_cache,
            "TRITON_CACHE_DIR": triton_cache,
            "TRITON_HOME": triton_cache,
            "STREAMING_LOCALS": dttm,
            "TCP_BARRIER_PORT": str(TCP_BARRIER_PORT), 
            "TEMPDIR": temp_dir,
            "PATH": os.environ.get("PATH", "") + ":/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin",
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
        }

    def _cleanup_temp_paths(self):
        for temp_path in self._temp_paths:
            normalized = os.path.abspath(temp_path)
            if not normalized.startswith("/tmp/") or normalized == "/tmp":
                continue
            shutil.rmtree(normalized, ignore_errors=True)

    def _make_torchrun_command(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))  
        free_port = sock.getsockname()[1]  
        sock.close() 

        return (
            f"torchrun --nproc_per_node {self.processes_per_worker} "
            f"--rdzv_backend=c10d --rdzv_endpoint=localhost:{free_port} "
            f"{self.train_script_path} {self.config_path} use_clearml=False"
        )

class LocalJobHandle(JobHandle):
    def __init__(self, future: Future, gpu_indices: Set[int]=None):
        self._future = future
        self._gpu_indices = gpu_indices
        self._status = JobStatus.PENDING

    def get_gpu_indices(self) -> Set[int]:
        return self._gpu_indices

    def status(self) -> JobStatus:
        if self._status == JobStatus.PENDING and self._future.done():
            try:
                self._future.result()
                self._status = JobStatus.SUCCESS
            except Exception:
                self._status = JobStatus.FAILED
        return self._status

    def wait(self, timeout: Optional[float] = None) -> None:
        result = self._future.result(timeout=timeout)
        # submit_job returns an outer handle whose future resolves to the inner
        # handle; the inner handle's future is the actual job.run (returns None).
        if isinstance(result, LocalJobHandle):
            result._future.result(timeout=timeout)
