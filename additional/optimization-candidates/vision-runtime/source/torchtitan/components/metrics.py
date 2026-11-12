# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import socket
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.summary_pb2 import Summary

from torchtitan.tools.utils import (
    device_capacity_report,
    device_module,
    reset_peak_stats,
)
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, NoColor


# Default size limit (MiB) past which the TensorBoard event file is rotated.
TB_ROTATE_MIB_DEFAULT = 64
WRITER_MAX_QUEUE = 1000


class BaseLogger:
    """Logger that does nothing, used when logging is disabled."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Free-form text under ``tag`` (run records such as the callback schedule)."""
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    def __init__(
        self,
        log_dir: str,
        tag: str | None = None,
        *,
        rotate_mib: int = 0,
    ) -> None:
        self.tag = tag
        self._log_dir = Path(log_dir)
        self._rotate_bytes = rotate_mib * 1024 * 1024
        self.writer = SummaryWriter(log_dir, max_queue=WRITER_MAX_QUEUE)
        # The file the writer is appending to; captured once per event file.
        self._event_file: Path | None = None
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        values = []
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            if isinstance(v, (int, float)):
                value = Summary.Value(  # pyrefly: ignore [missing-attribute]
                    tag=tag, simple_value=float(v)
                )
                values.append(value)
            else:
                self.writer.add_scalar(tag, v, step)
        if values:
            self.writer._get_file_writer().add_summary(  # pyrefly: ignore [missing-attribute]
                Summary(value=values),  # pyrefly: ignore [not-callable]
                global_step=step,
            )
        self._close_event_file_if_oversized()

    def _current_event_file(self) -> Path | None:
        """The event file being written, discovered from the log directory."""
        if self._event_file is None:
            files = self._log_dir.glob("events.out.tfevents.*")
            self._event_file = max(files, key=os.path.getmtime, default=None)
        return self._event_file

    def _close_event_file_if_oversized(self) -> None:
        """Close the writer once its event file outgrows the size limit."""
        if not self._rotate_bytes:
            return
        try:
            event_file = self._current_event_file()
            if event_file is None:
                return
            size = event_file.stat().st_size
        except OSError as exc:
            logger.warning(
                f"TensorBoard event file size probe failed ({exc}); "
                "keeping the current file"
            )
            return
        if size >= self._rotate_bytes:
            self.writer.close()
            self._event_file = None

    def log_text(self, tag: str, text: str, step: int) -> None:
        tag = tag if self.tag is None else f"{self.tag}/{tag}"
        # Markdown renderer: keep the table's spacing.
        self.writer.add_text(tag, "```\n" + text + "\n```", step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases."""

    def __init__(
        self,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        # Import wandb here to avoid startup import
        import wandb

        self.wandb = wandb
        self.tag = tag

        wandb_log_dir = os.getenv("WANDB_DIR") or log_dir
        os.makedirs(wandb_log_dir, exist_ok=True)
        tags = os.getenv("WANDB_RUN_TAGS")
        wandb_tags = (
            [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else None
        )

        self.wandb.init(
            entity=os.getenv("WANDB_TEAM", None),
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            name=os.getenv("WANDB_RUN_NAME", None),
            id=os.getenv("WANDB_RUN_ID", None),
            notes=os.getenv("WANDB_RUN_NOTES", None),
            tags=wandb_tags,
            group=os.getenv("WANDB_RUN_GROUP", None),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE", None),
            resume_from=os.getenv("WANDB_RESUME_FROM", None),
            fork_from=os.getenv("WANDB_FORK_FROM", None),
            dir=wandb_log_dir,
            config=config_dict,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wandb.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


class LoggerContainer(BaseLogger):
    """Container to call all loggers enabled in the job config."""

    def __init__(self) -> None:
        self._loggers: list[BaseLogger] = []

    def add_logger(self, logger_instance: BaseLogger) -> None:
        self._loggers.append(logger_instance)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log(metrics, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log_text(tag, text, step)

    @property
    def number_of_loggers(self) -> int:
        return len(self._loggers)

    def close(self) -> None:
        for logger_instance in self._loggers:
            logger_instance.close()


def ensure_pp_loss_visible(
    *, parallel_dims: ParallelDims, pp_schedule: str, color: Color | NoColor
) -> None:
    """
    Ensures that the loss is visible on the console for pipeline-parallel training.

    For pipeline-parallel training, the loss is only visible on the last pipeline stage.
    This function checks if the appropriate rank is included in the LOG_RANK environment
    variable and warns if it's not.
    """

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return

    # Calculate the rank where loss is visible (first rank of the last pipeline stage)
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    loss_visible_rank = (world_size // pp_size) * (pp_size - 1)

    # The console filter reads GIFT_LOG_RANK with a LOG_RANK fallback
    # (tools/logging.py); check the same value the filter will use.
    env_logged_ranks = os.environ.get(
        "GIFT_LOG_RANK", os.environ.get("LOG_RANK", "")
    ).split(",")
    if env_logged_ranks == [""]:
        env_logged_ranks = []

    if str(loss_visible_rank) not in env_logged_ranks:
        logger.warning(
            f"{color.red}Pipeline Parallel loss is not visible. "
            f"Please add {color.yellow}rank {loss_visible_rank}{color.red} "
            f"to the GIFT_LOG_RANK environment variable.{color.reset}"
        )


def _get_metrics_rank(
    *,
    parallel_dims: ParallelDims,
    pp_schedule: str,
) -> int:
    """
    Determines which rank should log metrics.

    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel 'ZBVZeroBubble' schedule
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)


class MetricsProcessor(Configurable):
    """Metrics processor to processes the metrics and log metrics.

    The current MetricsProcessor log some metrics to STDOUT and some metrics to
    TensorBoard or WandB.

    Args:
        config (Config): Metrics configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        dump_folder (str): Base folder for log output.
        pp_schedule (str): Pipeline parallel schedule name.
        ft_enable (bool): Whether fault tolerance is enabled.
        ft_replica_id (int): Fault tolerance replica ID.
        config_dict (dict | None): Full job config dict for WandB logging.
        tag (str | None): Tag to use for TensorBoard or WandB. Defaults to None.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to log metrics to TensorBoard, in iterations"""

        enable_tensorboard: bool = False
        """Whether to log metrics to TensorBoard"""

        disable_color_printing: bool = False
        """Whether to disable color printing in logs"""

        save_tb_folder: str = "tb"
        """Folder to dump TensorBoard states"""

        tb_rotate_mib: int = TB_ROTATE_MIB_DEFAULT
        """Start a new event file once the current one exceeds this many MiB."""

        tb_folder_per_launch: bool = False
        """Whether each launch gets its own timestamped subfolder of
        ``save_tb_folder``. Off by default: a run is one curve, and a relaunch
        continues it, so splitting a resumed run across sibling event
        directories is what has to be asked for."""

        save_for_all_ranks: bool = False
        """
        Whether to save TensorBoard/Wandb metrics only for rank 0 or for all ranks.
        When this option is False and pipeline_parallel_degree is > 1, the metrics
        component uses the 0th rank of the last stage pipeline group, which is the
        only stage that computes loss metrics.
        """

        enable_wandb: bool = False
        """Whether to log metrics to Weights & Biases"""

        save_rank_memory_stats: bool = False
        """Write allocator statistics on every rank at log_freq to memory/rankN.jsonl.

        Independent of TensorBoard/W&B rank selection. Peaks cover the interval
        since the previous metrics log, including all accumulated microbatches.
        """

    config: Config
    logger: BaseLogger
    parallel_dims: ParallelDims
    color: utils.NoColor | utils.Color

    optimizers: OptimizersContainer | None
    lr_schedulers: LRSchedulersContainer | None
    model_parts: list[torch.nn.Module] | None

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        dump_folder: str = "./outputs",
        pp_schedule: str = "1F1B",
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
        num_flops_per_token: int = 0,
        has_quantization: bool = False,
    ):
        self.logger = self._build_metric_logger(
            config=config,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
            pp_schedule=pp_schedule,
            ft_enable=ft_enable,
            ft_replica_id=ft_replica_id,
            config_dict=config_dict,
            tag=tag,
        )
        self.parallel_dims = parallel_dims
        self.config = config
        self._rank_memory_file = None
        if config.save_rank_memory_stats:
            self._memory_rank = torch.distributed.get_rank()
            self._memory_host = socket.gethostname()
            memory_dir = Path(dump_folder) / "memory"
            if ft_enable:
                memory_dir /= f"replica_{ft_replica_id}"
            memory_dir.mkdir(parents=True, exist_ok=True)
            self._rank_memory_file = (
                memory_dir / f"rank{self._memory_rank}.jsonl"
            ).open("a", buffering=1)
        # used for colorful printing
        self.color = utils.NoColor() if config.disable_color_printing else utils.Color()

        reset_peak_stats()

        self.ntokens_since_last_log = 0
        self.data_loading_times: list[float] = []
        self._time_last_log = time.perf_counter()
        self._peak_flops: float | None = None

        self.num_flops_per_token = num_flops_per_token
        self.has_quantization = has_quantization
        # These have to be set later: they are built after the processor.
        self.optimizers = None
        self.lr_schedulers = None
        self.model_parts = None

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.config.log_freq == 0

    def interval_metrics(self) -> dict[str, float]:
        """Throughput of the interval that just ended, resetting its counters.

        tps/tflops/mfu are per device (divided by the non-data-parallel size),
        matching the historical convention.
        """
        time_delta = time.perf_counter() - self._time_last_log
        self._time_last_log = time.perf_counter()

        data_loading_times = self.data_loading_times
        ntokens = self.ntokens_since_last_log
        self.ntokens_since_last_log = 0
        self.data_loading_times = []

        if time_delta <= 0:
            return {}

        tps = ntokens / (time_delta * self.parallel_dims.non_data_parallel_size)
        metrics: dict[str, float] = {
            "tps": tps,
            "time/train_step": time_delta / self.config.log_freq,
        }
        if self.num_flops_per_token > 0:
            metrics["throughput/device/tflops"] = self.num_flops_per_token * tps / 1e12
            if not self.has_quantization:
                if self._peak_flops is None:
                    self._peak_flops = utils.get_peak_flops(
                        device_module.get_device_name(device_module.current_device())
                    )
                metrics["throughput/device/mfu"] = (
                    100 * self.num_flops_per_token * tps / self._peak_flops
                )
        if data_loading_times:
            metrics["time/data_loading"] = sum(data_loading_times) / len(
                data_loading_times
            )
            metrics["time/data_loading_pct"] = (
                100 * sum(data_loading_times) / time_delta
            )
        return metrics

    def log_extra_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        if metrics:
            self.logger.log(metrics, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        self.logger.log_text(tag, text, step)

    def _build_metric_logger(
        self,
        *,
        config: Config,
        parallel_dims: ParallelDims,
        dump_folder: str,
        pp_schedule: str,
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> BaseLogger:
        """
        Build an appropriate metric logger based on configuration.
        """
        # Log initial config state
        logger.debug(
            f"Building logger with config: wandb={config.enable_wandb}, "
            f"tensorboard={config.enable_tensorboard}"
        )

        # Check if any logging backend is enabled
        has_logging_enabled = config.enable_tensorboard or config.enable_wandb

        # Determine if this rank should log
        should_log = has_logging_enabled
        if (not config.save_for_all_ranks) and should_log:
            metrics_rank = _get_metrics_rank(
                parallel_dims=parallel_dims, pp_schedule=pp_schedule
            )
            should_log = torch.distributed.get_rank() == metrics_rank

        logger.debug(
            f"Logging decision: has_logging_enabled={has_logging_enabled}, should_log={should_log}"
        )

        if not should_log:
            logger.debug("Returning BaseLogger due to should_log=False")
            return BaseLogger()

        # Setup logging directory
        base_log_dir = os.path.join(dump_folder, config.save_tb_folder)
        if config.tb_folder_per_launch:
            base_log_dir = os.path.join(
                base_log_dir, datetime.now().strftime("%Y%m%d-%H%M")
            )

        if ft_enable:
            base_log_dir = os.path.join(
                base_log_dir,
                f"replica_{ft_replica_id}",
            )

        if config.save_for_all_ranks:
            base_log_dir = os.path.join(
                base_log_dir, f"rank_{torch.distributed.get_rank()}"
            )

        # Create logger container
        logger_container = LoggerContainer()

        # Create loggers in priority order
        if config.enable_wandb:
            logger.debug("Attempting to create WandB logger")
            try:
                wandb_logger = WandBLogger(
                    base_log_dir, config_dict=config_dict, tag=tag
                )
                logger_container.add_logger(wandb_logger)
            except Exception as e:
                if "No module named 'wandb'" in str(e):
                    logger.error(
                        "Failed to create WandB logger: No module named 'wandb'. Please install it using 'pip install wandb'."
                    )
                else:
                    logger.error(f"Failed to create WandB logger: {e}")

        if config.enable_tensorboard:
            logger.debug("Creating TensorBoard logger")
            tensorboard_logger = TensorBoardLogger(
                base_log_dir, tag, rotate_mib=config.tb_rotate_mib
            )
            logger_container.add_logger(tensorboard_logger)

        if logger_container.number_of_loggers == 0:
            logger.debug("No loggers enabled, returning an empty LoggerContainer")
        return logger_container

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float | None,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        """
        Log training metrics including loss, throughput, and memory statistics.

        Args:
            step: Current training step
            global_avg_loss: Globally reduced training objective. Its denominator
                follows the configured loss normalization.
            global_max_loss: Maximum local loss across all ranks, or ``None``
                when the objective cannot be reconstructed with one local denominator
            grad_norm: Gradient norm after clipping
            extra_metrics: Optional additional metrics to log

        """
        capacity, memory_gib = device_capacity_report()
        interval = self.interval_metrics()
        tps = interval.pop("tps", float("nan"))
        metrics = {
            "loss/train/total": global_avg_loss,
            "l2_norm/grad/global": grad_norm,
            **capacity,
            **interval,
        }
        if global_max_loss is not None:
            metrics["loss/train/global_max"] = global_max_loss

        if extra_metrics:
            metrics.update(extra_metrics)

        if self._rank_memory_file is not None:
            self._rank_memory_file.write(
                json.dumps(
                    {
                        "step": step,
                        "rank": self._memory_rank,
                        "hostname": self._memory_host,
                        "time_ns": time.time_ns(),
                        **capacity,
                        **interval,
                    }
                )
                + "\n"
            )
        self.logger.log(metrics, step)
        reset_peak_stats()

        color = self.color
        memory_pct = capacity["memory/max_reserved_pct"]
        tflops = interval.get("throughput/device/tflops", float("nan"))
        mfu = interval.get("throughput/device/mfu")
        mfu_str = f"{mfu:.2f}%" if mfu is not None else "N/A"
        logger.info(
            f"{color.red}step: {step:2}  "
            f"{color.green}loss: {global_avg_loss:8.5f}  "
            f"{color.orange}grad_norm: {grad_norm:7.4f}  "
            f"{color.turquoise}memory: {memory_gib:5.2f}GiB({memory_pct:.2f}%)  "
            f"{color.blue}tps: {tps:,.0f}  "
            f"{color.cyan}tflops: {tflops:,.2f}  "
            f"{color.magenta}mfu: {mfu_str}{color.reset}"
        )

    def log_validation(
        self, loss: float, step: int, extra_metrics: dict[str, Any] | None = None
    ):
        memory_metrics, max_reserved_gib = device_capacity_report("memory/valid")

        metrics = {
            # Same tag families as the training path (loss/train/total etc.).
            "loss/valid/total": loss,
            **memory_metrics,
        }

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.yellow}validate step: {step:2}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.turquoise}memory: {max_reserved_gib:5.2f}GiB"
            f"({memory_metrics['memory/valid/max_reserved_pct']:.2f}%){color.reset}"
        )

        reset_peak_stats()

    def close(self):
        if self._rank_memory_file is not None:
            self._rank_memory_file.close()
        self.logger.close()
