import time
from pathlib import Path
from typing import Optional

from .logger import Logger
from .subprocess_manager import SubprocessManager


class SubprocessPoolRunner:
    """
    Запускает команды в виде пула процессов фиксированного размера.

    cmds_to_run: dict[run_name, cmd_list]
    max_parallel: сколько процессов держать одновременно
    """

    def __init__(
        self,
        cmds_to_run: dict[str, list[str]],
        max_parallel: int,
        autometrics_log_file: Path,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")

        self.cmds_to_run = list(cmds_to_run.items())
        self.max_parallel = max_parallel
        self.autometrics_log_file = autometrics_log_file
        self.cwd = cwd
        self.env = env

        self._pending_idx: int = 0
        self._active: dict[str, SubprocessManager] = {}

    def _start_next_if_possible(self) -> None:
        while len(self._active) < self.max_parallel and self._pending_idx < len(
            self.cmds_to_run
        ):
            run_name, cmd_list = self.cmds_to_run[self._pending_idx]
            self._pending_idx += 1

            Logger.info(f"Starting PoolRunner task {run_name} with command: {cmd_list}")
            sm = SubprocessManager(
                cmd_list,
                self.autometrics_log_file,
                self.cwd,
                run_name,
                env=self.env,
            )
            sm.start()
            self._active[run_name] = sm

    def _finalize_finished(self) -> None:
        finished: list[str] = []

        for run_name, sm in self._active.items():
            if not sm.is_alive():
                if sm.process is not None and sm.exit_code is None:
                    sm.wait_for_completion()

                sm.log_error()
                finished.append(run_name)

        for run_name in finished:
            self._active.pop(run_name, None)

    def _has_work_left(self) -> bool:
        return bool(self._active) or self._pending_idx < len(self.cmds_to_run)

    def run_all(self, poll_interval: float = 5.0) -> None:
        start_time = time.time()
        Logger.info(
            f"Starting PoolRunner with {len(self.cmds_to_run)} runs, "
            f"max_parallel={self.max_parallel} tasks"
        )

        self._start_next_if_possible()

        if not self._active and self._pending_idx >= len(self.cmds_to_run):
            Logger.info("No PoolRunner tasks to execute")
            return

        while self._has_work_left():
            self._finalize_finished()
            self._start_next_if_possible()
            if self._has_work_left():
                time.sleep(poll_interval)

        Logger.info(
            f"All PoolRunner runs completed in {time.time() - start_time:.2f} seconds"
        )
