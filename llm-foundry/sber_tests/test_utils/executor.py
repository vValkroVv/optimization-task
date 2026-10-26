from typing import Dict, List, Optional
import os
import time
from multiprocessing import Pool
from torch.cuda import device_count
from sber_tests.test_utils.job import LocalJob, LocalJobHandle, VerboseLocalJobHandle


class Executor:
    def __init__(
            self,
            train_script_path: str | os.PathLike,
            code_libraries: List[str | os.PathLike],
            verbose: bool = False,
    ) -> None:
        self._train_script_path = train_script_path
        self._verbose = verbose
        self._default_env_variables = {
            "PYTHONPATH": ":".join([str(p) for p in code_libraries]),
            "NNODES": "1",
            "NODE_RANK": "0",
            "LOCAL_RANK": "0",
            "REUSE_SOCKETS": "1",
        }
        self._pool = Pool(processes=1)

    def submit_job(
            self,
            config_path: str | os.PathLike,
            processes_per_worker: int,
            env_variables: Optional[Dict[str, str]] = None,
            log_dir: Optional[str | os.PathLike] = None,
    ) -> LocalJobHandle:
        assert processes_per_worker <= device_count(), f"Requested GPU count larger than available ({processes_per_worker} > {device_count()})"

        env_variables = env_variables or dict()
        env_variables = os.environ | self._default_env_variables | env_variables

        if self._verbose and not log_dir:
            log_dir = f"/tmp/nested_job_logs_{int(time.time())}"
            os.makedirs(log_dir, exist_ok=True)
            print(f"[EXECUTOR] Auto-created log_dir: {log_dir}", flush=True)

        job = LocalJob(
            self._train_script_path,
            config_path,
            processes_per_worker,
            env_variables,
            log_dir,
        )

        apply_result = self._pool.apply_async(job.run)

        if self._verbose:
            print(f"[EXECUTOR] Submitting job in verbose mode with log_dir: {log_dir}", flush=True)
            return VerboseLocalJobHandle(apply_result, log_dir)
        else:
            return LocalJobHandle(apply_result)

    def close(self):
        self._pool.close()
