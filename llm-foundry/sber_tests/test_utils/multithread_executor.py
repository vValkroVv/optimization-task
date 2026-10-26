from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from queue import Queue
import torch
from typing import Dict, Optional, List
from .job_multi import LocalJob, LocalJobHandle, ResourcePool
import threading
import os


class Executor:
    def __init__(self, train_script_path: str, code_libraries: list, verbose: bool = False):
        self._resource_pool = ResourcePool(torch.cuda.device_count())
        self._job_queue = Queue()
        self._thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        self._train_script_path = train_script_path
        self._verbose = verbose
        self._default_env_variables = {
            "PYTHONPATH": ":".join([str(p) for p in code_libraries]),
            "NNODES": "1",
            "NODE_RANK": "0",
            "LOCAL_RANK": "0",
            "REUSE_SOCKETS": "1",
            "USE_FA3": str(os.getenv("USE_FA3", "1")),
        }

        self._active_jobs: List[LocalJobHandle] = []
        self._active_jobs_lock = threading.Lock()

        self._current_batch_name: Optional[str] = None
        self._auto_barrier_enabled = True

        if self._verbose:
            print(f"Created pool with {torch.cuda.device_count()} GPUs")

        self._start_worker()

    def _start_worker(self):
        def worker():
            while True:
                try:
                    job_config = self._job_queue.get()

                    required_gpus = job_config["processes_per_worker"]

                    gpu_indices = self._resource_pool.acquire(required_gpus)

                    job = LocalJob(
                        train_script_path=self._train_script_path,
                        config_path=job_config["config_path"],
                        processes_per_worker=required_gpus,
                        env_variables=job_config["env_vars"],
                        log_dir=job_config["log_dir"],
                        gpu_indices=gpu_indices,
                        resource_pool=self._resource_pool
                    )

                    if self._verbose:
                        print(f"Started job: {job_config['config_path']} "
                              f"acquiring GPUs {gpu_indices}/{job_config['processes_per_worker']}")

                    future = self._thread_pool.submit(job.run)

                    handle = LocalJobHandle(future, gpu_indices)

                    job_config["callback"](handle)

                    if self._verbose:
                        print(f"Job submitted to thread pool: {job_config['config_path']}")

                except Exception as e:
                    print(f"WORKER EXCEPTION: {e}")
                    import traceback
                    traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()

        if self._verbose:
            print("Worker started and awaiting jobs")

    def submit_job(self, config_path: str, processes_per_worker: int,
                   env_vars: Optional[Dict] = None, log_dir: str = '',
                   batch_name: Optional[str] = None) -> LocalJobHandle:
        if self._auto_barrier_enabled and batch_name is not None:
            if self._current_batch_name is not None and self._current_batch_name != batch_name:
                if self._verbose:
                    print(f"\n{'=' * 60}")
                    print(f"Auto-barrier: switching from '{self._current_batch_name}' to '{batch_name}'")
                    print(f"{'=' * 60}\n")
                self.wait_current_batch()
            self._current_batch_name = batch_name

        future = Future()

        def callback(handle):
            future.set_result(handle)
            with self._active_jobs_lock:
                self._active_jobs.append(handle)

        self._job_queue.put({
            "config_path": config_path,
            "processes_per_worker": processes_per_worker,
            "env_vars": self._default_env_variables | (env_vars or {}),
            "log_dir": log_dir,
            "callback": callback
        })

        return LocalJobHandle(future)

    def wait_current_batch(self, timeout: Optional[float] = None) -> List[LocalJobHandle]:
        with self._active_jobs_lock:
            jobs_to_wait = self._active_jobs.copy()
            self._active_jobs.clear()

        if self._verbose:
            print(f"Waiting for batch of {len(jobs_to_wait)} jobs to complete...")

        completed_jobs = []

        for i, handle in enumerate(jobs_to_wait):
            try:
                handle.wait(timeout=timeout)
                completed_jobs.append(handle)

                if self._verbose:
                    print(f"Job {i + 1}/{len(jobs_to_wait)} completed")

            except TimeoutError:
                if self._verbose:
                    print(f"Timeout waiting for job {i + 1}/{len(jobs_to_wait)}")
                raise
            except Exception as e:
                print(f"Error waiting for job {i + 1}/{len(jobs_to_wait)}: {e}")

        if self._verbose:
            print(f"Batch complete: {len(completed_jobs)}/{len(jobs_to_wait)} jobs finished\n")

        return completed_jobs

    def enable_auto_barrier(self):
        self._auto_barrier_enabled = True
        if self._verbose:
            print("Auto-barrier enabled")

    def disable_auto_barrier(self):
        self._auto_barrier_enabled = False
        if self._verbose:
            print("Auto-barrier disabled")

    def get_active_jobs_count(self) -> int:
        with self._active_jobs_lock:
            return len(self._active_jobs)

    def get_queue_size(self) -> int:
        return self._job_queue.qsize()

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        if self._verbose:
            print("Shutting down executor...")

        if wait:
            self.wait_current_batch()

        self._thread_pool.shutdown(wait=wait, cancel_futures=cancel_futures)

        if self._verbose:
            print("Executor shutdown complete")
