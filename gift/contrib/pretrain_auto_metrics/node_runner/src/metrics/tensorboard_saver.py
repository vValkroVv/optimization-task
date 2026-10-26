import fcntl
import logging
import os
import re
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from ..job_dispatcher.submetrics import MetricValue, is_number
from ..logger import Logger


class EmptyDirectoryError(Exception):
    pass


class TensorBoardSaver:
    _FIXED_EVENT_FILENAME = "events.out.tfevents.autometrics"

    def __init__(
        self, checkpoint_path: Path, tensorboard_path: Path | None, run_name: str
    ):
        self.checkpoint_path = checkpoint_path
        self.run_name = run_name
        self.step = self._get_tensorboard_step(checkpoint_path)
        self.metrics_tensorboard = (
            Path(tensorboard_path, run_name, "metrics")
            if tensorboard_path is not None
            else None
        )
        if self.metrics_tensorboard is not None:
            os.makedirs(self.metrics_tensorboard, exist_ok=True)

    # The training step a checkpoint directory names, per framework:
    # llm-foundry writes ``ep0_ba1000``, torchtitan writes ``step-1000``.
    _STEP_PATTERNS = (r"ep\d+_ba(\d+)", r"step-(\d+)")

    @staticmethod
    def _get_tensorboard_step(checkpoint_path: Path) -> int:
        """The step to plot these metrics at.

        Falling back to 0 is not a neutral default: every checkpoint of a run
        then lands on the same point and overwrites the one before it, leaving
        a board that shows whichever was measured last instead of a curve. So
        say so when the name is not recognized.
        """
        # Not just the leaf: the HF-export path hands us
        # ``<step-800>/artifacts/hf_checkpoint``, so the step is two levels up.
        # Walking from the leaf outwards takes the innermost match, which is the
        # checkpoint's own step rather than something from the run's path.
        for part in reversed(checkpoint_path.parts):
            for pattern in TensorBoardSaver._STEP_PATTERNS:
                match = re.search(pattern, part)
                if match:
                    return int(match.group(1))
        Logger.warning(
            f"Cannot read a training step out of {str(checkpoint_path)!r}: "
            f"none of {TensorBoardSaver._STEP_PATTERNS} match. Plotting at step "
            "0, where every other unrecognized checkpoint also lands."
        )
        return 0

    def _save_to_tensorboard(
        self, result_metrics: dict[str, float], tb_path: Path, step: int
    ) -> None:
        writer = SummaryWriter(log_dir=tb_path)
        for metric_name, metric_value in result_metrics.items():
            writer.add_scalar(self._format_metric_name(metric_name), metric_value, step)
        writer.close()

    @staticmethod
    def _format_metric_name(metric_name: str) -> str:
        if metric_name.split("/")[0] == "mmlu_pro":
            return f"{metric_name}"
        return f"autometrics/{metric_name}"

    @contextmanager
    def _tensorboard_write_lock(self) -> Iterator[None]:
        lock_path = self.metrics_tensorboard / ".tensorboard_saver.lock"
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=15),
        after=after_log(Logger, logging.INFO),
        retry=retry_if_exception_type(FileExistsError),
        reraise=True,
    )
    def _create_new_run_dir(self) -> Path:
        run_dir = self.metrics_tensorboard / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True)
        return run_dir

    def _iter_tensorboard_run_dirs(self) -> list[Path]:
        if not self.metrics_tensorboard.exists():
            return []
        return sorted(
            path
            for path in self.metrics_tensorboard.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    def _collect_latest_scalars(self) -> dict[str, dict[int, float]]:
        scalars_by_tag: dict[str, dict[int, float]] = {}
        for run_dir in self._iter_tensorboard_run_dirs():
            ea = event_accumulator.EventAccumulator(
                str(run_dir),
                size_guidance={event_accumulator.SCALARS: 0},
            )
            ea.Reload()
            tags = ea.Tags().get("scalars", [])

            for tag in tags:
                by_step = scalars_by_tag.setdefault(tag, {})
                for ev in ea.Scalars(tag):
                    by_step[ev.step] = ev.value
        return scalars_by_tag

    def _write_sorted_scalars(
        self,
        run_dir: Path,
        scalars_by_tag: dict[str, dict[int, float]],
    ) -> None:
        writer = SummaryWriter(log_dir=str(run_dir))
        try:
            for tag, by_step in scalars_by_tag.items():
                for step in sorted(by_step.keys()):
                    writer.add_scalar(tag, float(by_step[step]), step)
            writer.flush()
        finally:
            writer.close()

        event_files = sorted(
            run_dir.glob("events.out.tfevents.*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not event_files:
            return

        fixed_event_file = run_dir / self._FIXED_EVENT_FILENAME
        os.replace(event_files[0], fixed_event_file)

        for extra_file in event_files[1:]:
            extra_file.unlink(missing_ok=True)

    def _cleanup_previous_timestamp_dirs(self, current_run_dir: Path) -> None:
        for run_dir in self.metrics_tensorboard.iterdir():
            if run_dir.name >= current_run_dir.name:
                continue
            shutil.rmtree(run_dir, ignore_errors=True)

    def _add_metrics_to_scalars(
        self,
        scalars_by_tag: dict[str, dict[int, float]],
        metrics_results: dict[str, float],
    ) -> None:
        for metric_name, metric_value in metrics_results.items():
            tag = self._format_metric_name(metric_name)
            scalars_by_tag.setdefault(tag, {})[self.step] = metric_value

    def _dir_is_empty(self, run_dir: Path) -> bool:
        return not any(run_dir.iterdir())

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(EmptyDirectoryError),
    )
    def _wait_for_all_dirs_to_be_written(self) -> None:
        for run_dir in self._iter_tensorboard_run_dirs():
            if not run_dir.exists():
                continue
            if self._dir_is_empty(run_dir):
                raise EmptyDirectoryError(f"Directory {run_dir} is empty")

    def save_metrics_to_tensorboard(self, metrics_results: dict[str, MetricValue]):
        metrics_results = {
            key: value for key, value in metrics_results.items() if is_number(value)
        }
        if self.metrics_tensorboard is None or not metrics_results:
            return
        with self._tensorboard_write_lock():
            self._wait_for_all_dirs_to_be_written()
            scalars_by_tag = self._collect_latest_scalars()
            self._add_metrics_to_scalars(scalars_by_tag, metrics_results)
            current_run_dir = self._create_new_run_dir()
            Logger.info(f"Saving metrics to tensorboard on path: {current_run_dir}")
            self._write_sorted_scalars(current_run_dir, scalars_by_tag)
            self._cleanup_previous_timestamp_dirs(current_run_dir)
            Logger.info("Previous tensorboards cleared successfully")
