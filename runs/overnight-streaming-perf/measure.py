#!/usr/bin/env python3
"""Task-local, end-of-update measurements around native TorchTitan training."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1
PEAK_STAT_KEYS = (
    "allocated_bytes.all.peak",
    "active_bytes.all.peak",
    "reserved_bytes.all.peak",
)
ALLOCATOR_COUNTER_KEYS = ("num_alloc_retries", "num_ooms")


class IncompleteWindowError(RuntimeError):
    """Raised when native training ends before its requested measurement window."""


class DeepGemmCacheRedirect:
    """Redirect the active task warmup helper to one fresh run-owned cache."""

    def __init__(
        self,
        warmup_module: Any,
        trainer_module: Any,
        cache_root: Path,
        *,
        task_root: Path | None = None,
    ) -> None:
        self.warmup_module = warmup_module
        self.trainer_module = trainer_module
        self.cache_root = cache_root.resolve()
        self.task_root = (task_root or Path(__file__).resolve().parents[2]).resolve()
        self._original_rank_cache_dir = warmup_module._rank_cache_dir
        self._original_setup = trainer_module.setup_deep_gemm_cache
        self._installed = False

    def __enter__(self) -> "DeepGemmCacheRedirect":
        try:
            self.cache_root.relative_to(self.task_root)
        except ValueError as exc:
            raise RuntimeError(
                f"DeepGEMM rank cache must stay inside the task tree: {self.cache_root}"
            ) from exc
        if not self.cache_root.is_dir():
            raise RuntimeError(f"DeepGEMM rank cache root is missing: {self.cache_root}")

        def task_rank_cache_dir(local_rank: int) -> str:
            rank = int(local_rank)
            if not 0 <= rank < 8:
                raise RuntimeError(f"unexpected local rank for the single-node run: {rank}")
            return str(self.cache_root / f"rank_{rank}" / "deepgemm") + os.sep

        def guarded_setup(local_rank: int, model_config: Any, ep_degree: int, fp8: Any) -> Any:
            if getattr(fp8, "deep_gemm_warmup_on_start", False) is not True:
                raise RuntimeError(
                    "task-local cache redirect requires the fixed DeepGEMM warmup path"
                )
            if os.environ.get("PROPAGATE_DG_CACHE") or os.environ.get(
                "PROPAGATE_DG_CACHE_DIR"
            ):
                raise RuntimeError(
                    "PROPAGATE_DG_CACHE paths bypass the task-local rank cache"
                )
            return self._original_setup(local_rank, model_config, ep_degree, fp8)

        self.warmup_module._rank_cache_dir = task_rank_cache_dir
        self.trainer_module.setup_deep_gemm_cache = guarded_setup
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._installed:
            self.warmup_module._rank_cache_dir = self._original_rank_cache_dir
            self.trainer_module.setup_deep_gemm_cache = self._original_setup
            self._installed = False


def parameter_census(model_parts: Any) -> dict[str, Any]:
    """Count local parameter shards by trainability and dtype without reading values."""
    totals: dict[str, dict[str, Any]] = {
        key: {"tensor_count": 0, "local_numel": 0, "by_dtype": {}}
        for key in ("trainable", "frozen")
    }
    seen: set[int] = set()

    if model_parts is None:
        raise RuntimeError("trainer.model_parts is unavailable for the parameter census")

    parts = model_parts if isinstance(model_parts, (list, tuple)) else [model_parts]
    for part in parts:
        for parameter in part.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)

            local_parameter = parameter
            to_local = getattr(parameter, "to_local", None)
            if callable(to_local):
                local_parameter = to_local()
            elif hasattr(parameter, "_local_tensor"):
                # GIFT's RaggedShardDTensor exposes its already-local shard here.
                local_parameter = parameter._local_tensor

            category = "trainable" if parameter.requires_grad else "frozen"
            dtype = str(parameter.dtype)
            category_totals = totals[category]
            category_totals["tensor_count"] += 1
            local_numel = int(local_parameter.numel())
            category_totals["local_numel"] += local_numel
            category_totals["by_dtype"][dtype] = (
                category_totals["by_dtype"].get(dtype, 0) + local_numel
            )

    return totals


class WindowMeasurement:
    """Collect one local rank's full-update window and lightweight cadence data."""

    def __init__(
        self,
        *,
        trainer: Any,
        run_id: str,
        output_dir: Path,
        expected_steps: int,
        warmup_updates: int,
        window_updates: int,
        monitor_interval: int,
        rank: int,
        world_size: int,
        cuda_synchronize: Callable[[], None],
        memory_stats: Callable[[], dict[str, int]],
        capture_microbatches: bool,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.trainer = trainer
        self.run_id = run_id
        self.output_dir = output_dir
        self.expected_steps = expected_steps
        self.warmup_updates = warmup_updates
        self.window_updates = window_updates
        self.monitor_interval = monitor_interval
        self.rank = rank
        self.world_size = world_size
        self.cuda_synchronize = cuda_synchronize
        self.memory_stats = memory_stats
        self.capture_microbatches = capture_microbatches
        self.clock_ns = clock_ns

        self.host_name = socket.gethostname()
        self.parameter_census = parameter_census(trainer.model_parts)
        self.metrics_processor = trainer.metrics_processor
        self._physical_tokens_flushed = 0
        self._last_step = 0
        self._last_timestamp_ns: int | None = None
        self._cadence: list[dict[str, Any]] = []
        self._microbatch_calls: dict[int, int] = defaultdict(int)
        self._sync_count = 0
        self._window_started = False
        self._window_finished = False
        self._window_start_ns: int | None = None
        self._window_end_ns: int | None = None
        self._physical_start: int | None = None
        self._physical_end: int | None = None
        self._supervised_start: int | None = None
        self._supervised_end: int | None = None
        self._allocator_start: dict[str, int | None] | None = None
        self._allocator_end: dict[str, int | None] | None = None
        self._peak_bytes: dict[str, int] = {}
        self._failure_reason: str | None = None
        self._written = False

    def note_microbatch_complete(self, step: int) -> None:
        if self.capture_microbatches:
            self._microbatch_calls[step] += 1

    def before_interval_metrics(self, metrics_processor: Any) -> None:
        if (
            metrics_processor is self.metrics_processor
            and self._window_started
            and not self._window_finished
        ):
            self._physical_tokens_flushed += int(
                metrics_processor.ntokens_since_last_log
            )

    def before_native_memory_reset(self) -> None:
        if self._window_started and not self._window_finished:
            self._sample_memory()

    @staticmethod
    def _counter_snapshot(stats: dict[str, int]) -> dict[str, int | None]:
        return {
            key: int(stats[key]) if key in stats else None
            for key in ALLOCATOR_COUNTER_KEYS
        }

    def _sample_memory(self) -> dict[str, int]:
        stats = self.memory_stats()
        for key in PEAK_STAT_KEYS:
            value = stats.get(key)
            if value is not None:
                self._peak_bytes[key] = max(self._peak_bytes.get(key, 0), int(value))
        return stats

    def _physical_total(self) -> int:
        return self._physical_tokens_flushed + int(
            self.metrics_processor.ntokens_since_last_log
        )

    @staticmethod
    def _synced_integer(value: Any) -> int:
        item = getattr(value, "item", None)
        return int(item() if callable(item) else value)

    def _start_window(self) -> int:
        self.cuda_synchronize()
        self._sync_count += 1
        start_ns = self.clock_ns()
        stats = self._sample_memory()
        self._allocator_start = self._counter_snapshot(stats)
        self._physical_start = self._physical_total()
        self._supervised_start = self._synced_integer(self.trainer.ntokens_seen)
        self._window_started = True
        self._window_start_ns = start_ns
        return start_ns

    def _end_window(self) -> int:
        self.cuda_synchronize()
        self._sync_count += 1
        end_ns = self.clock_ns()
        stats = self._sample_memory()
        self._allocator_end = self._counter_snapshot(stats)
        self._physical_end = self._physical_total()
        self._supervised_end = self._synced_integer(self.trainer.ntokens_seen)
        self._window_finished = True
        self._window_end_ns = end_ns
        return end_ns

    def on_profiler_step(self, step: int) -> None:
        if step != self._last_step + 1:
            self._failure_reason = (
                f"expected native update {self._last_step + 1}, observed {step}"
            )
        self._last_step = step

        boundary = None
        if self.window_updates and step == self.warmup_updates:
            boundary = "start"
            timestamp_ns = self._start_window()
        elif self.window_updates and step == self.warmup_updates + self.window_updates:
            boundary = "end"
            timestamp_ns = self._end_window()
        else:
            timestamp_ns = self.clock_ns()

        previous_ns = self._last_timestamp_ns
        self._cadence.append(
            {
                "step": step,
                "monotonic_ns": timestamp_ns,
                "interval_s": (
                    None
                    if previous_ns is None
                    else (timestamp_ns - previous_ns) / 1_000_000_000
                ),
                "monitoring_step": step % self.monitor_interval == 0,
                "window_boundary": boundary,
            }
        )
        self._last_timestamp_ns = timestamp_ns

        if self.window_started and not self.window_finished and boundary != "start":
            self._sample_memory()

    @property
    def window_started(self) -> bool:
        return self._window_started

    @property
    def window_finished(self) -> bool:
        return self._window_finished

    def _window_result(self) -> dict[str, Any] | None:
        if not self.window_updates:
            return None
        if not self._window_started or not self._window_finished:
            return None
        assert self._window_start_ns is not None and self._window_end_ns is not None
        assert self._physical_start is not None and self._physical_end is not None
        assert self._supervised_start is not None and self._supervised_end is not None
        assert self._allocator_start is not None and self._allocator_end is not None

        retry_deltas: dict[str, int | None] = {}
        for key in ALLOCATOR_COUNTER_KEYS:
            start_value = self._allocator_start[key]
            end_value = self._allocator_end[key]
            retry_deltas[key] = (
                None
                if start_value is None or end_value is None
                else end_value - start_value
            )

        return {
            "warmup_updates": self.warmup_updates,
            "window_updates": self.window_updates,
            "start_after_update": self.warmup_updates,
            "end_after_update": self.warmup_updates + self.window_updates,
            "start_monotonic_ns": self._window_start_ns,
            "end_monotonic_ns": self._window_end_ns,
            "local_elapsed_s": (self._window_end_ns - self._window_start_ns)
            / 1_000_000_000,
            "monitoring_updates": [
                step
                for step in range(
                    self.warmup_updates + 1,
                    self.warmup_updates + self.window_updates + 1,
                )
                if step % self.monitor_interval == 0
            ],
            "physical_input_tokens_local": self._physical_end - self._physical_start,
            "supervised_tokens_global": self._supervised_end
            - self._supervised_start,
            "cuda_synchronize_calls": self._sync_count,
            "memory": {
                "peak_bytes": dict(self._peak_bytes),
                "allocator_counters_start": self._allocator_start,
                "allocator_counters_end": self._allocator_end,
                "allocator_counter_deltas": retry_deltas,
            },
        }

    def finish(self, *, training_succeeded: bool, error: BaseException | None = None) -> str | None:
        if self._written:
            return self._failure_reason
        self._written = True
        completed_steps = int(self.trainer.step)
        training_complete = training_succeeded and completed_steps == self.expected_steps
        if training_succeeded and not training_complete:
            self._failure_reason = self._failure_reason or (
                f"native training completed at update {completed_steps}; "
                f"expected {self.expected_steps}"
            )

        measurement_status = "disabled"
        if self.window_updates:
            measurement_status = (
                "complete"
                if (
                    training_complete
                    and self._failure_reason is None
                    and self._window_result() is not None
                )
                else "incomplete"
            )
            if measurement_status != "complete":
                self._failure_reason = self._failure_reason or (
                    "native training ended before a complete measurement window"
                )

        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "host_name": self.host_name,
            "training_status": "complete" if training_complete else "incomplete",
            "expected_steps": self.expected_steps,
            "completed_steps": completed_steps,
            "training_error": (
                None if error is None else f"{type(error).__name__}: {error}"
            ),
            "measurement_status": measurement_status,
            "measurement": self._window_result(),
            "parameter_census": self.parameter_census,
            "cadence": self._cadence,
            "microbatch_calls_by_step": {
                str(step): count
                for step, count in sorted(self._microbatch_calls.items())
            },
            "capture_microbatches": self.capture_microbatches,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"rank-{self.rank:02d}.json"
        temporary = destination.with_suffix(".json.tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, indent=2)
            handle.write("\n")
        temporary.replace(destination)
        return self._failure_reason


class NativeAdapter:
    """Install temporary wrappers around the native trainer/profiler classes."""

    def __init__(
        self,
        *,
        trainer_class: type,
        profiler_class: type,
        metrics_class: type,
        metrics_module: Any,
        options: dict[str, Any],
        cuda_synchronize: Callable[[], None],
        memory_stats: Callable[[], dict[str, int]],
        deep_gemm_cache_redirect: DeepGemmCacheRedirect | None = None,
    ) -> None:
        self.trainer_class = trainer_class
        self.profiler_class = profiler_class
        self.metrics_class = metrics_class
        self.metrics_module = metrics_module
        self.options = options
        self.cuda_synchronize = cuda_synchronize
        self.memory_stats = memory_stats
        self.deep_gemm_cache_redirect = deep_gemm_cache_redirect
        self.active: WindowMeasurement | None = None
        self._originals: list[tuple[Any, str, Any]] = []

    def _replace(self, owner: Any, name: str, value: Any) -> None:
        self._originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def __enter__(self) -> "NativeAdapter":
        if self.deep_gemm_cache_redirect is not None:
            self.deep_gemm_cache_redirect.__enter__()
        original_train = self.trainer_class.train
        original_profiler_step = self.profiler_class.step
        original_forward_backward = self.trainer_class.forward_backward_step

        def tracked_train(trainer: Any, *args: Any, **kwargs: Any) -> Any:
            if self.active is not None:
                raise RuntimeError("nested native Trainer.train() is unsupported")
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            if world_size != 8:
                raise RuntimeError(
                    f"expected world size 8, got {world_size}"
                )
            self.active = WindowMeasurement(
                trainer=trainer,
                rank=rank,
                world_size=world_size,
                cuda_synchronize=self.cuda_synchronize,
                memory_stats=self.memory_stats,
                **self.options,
            )
            succeeded = False
            try:
                result = original_train(trainer, *args, **kwargs)
                succeeded = True
            except BaseException as exc:
                self.active.finish(training_succeeded=False, error=exc)
                raise
            else:
                reason = self.active.finish(training_succeeded=succeeded)
                if reason is not None:
                    raise IncompleteWindowError(reason)
                return result
            finally:
                self.active = None

        def tracked_profiler_step(profiler: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_profiler_step(profiler, *args, **kwargs)
            if self.active is not None:
                self.active.on_profiler_step(int(self.active.trainer.step))
            return result

        def tracked_forward_backward(trainer: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_forward_backward(trainer, *args, **kwargs)
            if self.active is not None and trainer is self.active.trainer:
                self.active.note_microbatch_complete(int(trainer.step))
            return result

        self._replace(self.trainer_class, "train", tracked_train)
        self._replace(self.profiler_class, "step", tracked_profiler_step)
        if self.options["capture_microbatches"]:
            self._replace(
                self.trainer_class, "forward_backward_step", tracked_forward_backward
            )

        if self.options["window_updates"]:
            original_interval_metrics = self.metrics_class.interval_metrics
            original_reset_peak_stats = self.metrics_module.reset_peak_stats

            def tracked_interval_metrics(
                metrics_processor: Any, *args: Any, **kwargs: Any
            ) -> Any:
                if self.active is not None:
                    self.active.before_interval_metrics(metrics_processor)
                return original_interval_metrics(metrics_processor, *args, **kwargs)

            def tracked_reset_peak_stats(*args: Any, **kwargs: Any) -> Any:
                if self.active is not None:
                    self.active.before_native_memory_reset()
                return original_reset_peak_stats(*args, **kwargs)

            self._replace(self.metrics_class, "interval_metrics", tracked_interval_metrics)
            self._replace(self.metrics_module, "reset_peak_stats", tracked_reset_peak_stats)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        if self.deep_gemm_cache_redirect is not None:
            self.deep_gemm_cache_redirect.__exit__(exc_type, exc_value, traceback)


def _split_training_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    try:
        separator = argv.index("--")
    except ValueError as exc:
        raise SystemExit("measure.py requires `--` before native train arguments") from exc
    return argv[:separator], argv[separator + 1 :]


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    helper_argv, train_argv = _split_training_argv(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--monitor-interval", type=int, required=True)
    parser.add_argument("--capture-microbatches", action="store_true")
    args = parser.parse_args(helper_argv)
    if not train_argv:
        parser.error("a native experiment config and overrides must follow `--`")
    if args.window < 0 or args.warmup < 0 or args.steps <= 0:
        parser.error("steps must be positive; warmup and window must be nonnegative")
    if args.monitor_interval <= 0:
        parser.error("monitor interval must be positive")
    if args.window and (args.warmup <= 0 or args.steps != args.warmup + args.window):
        parser.error("measured runs require warmup>0 and steps == warmup + window")
    if not args.window and args.warmup:
        parser.error("warmup must be zero when measurement window is disabled")
    return args, train_argv


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, train_argv = _parse_args(argv)

    import torch
    import torchtitan.components.metrics as metrics_module
    import torchtitan.tools.profiler as profiler_module
    import torchtitan.trainer as trainer_module
    from torchtitan.ops.warmup import deep_gemm_fast_warmup

    device_module = __import__("torchtitan.tools.utils", fromlist=["device_module"])
    options = {
        "run_id": args.run_id,
        "output_dir": args.measurement_dir,
        "expected_steps": args.steps,
        "warmup_updates": args.warmup,
        "window_updates": args.window,
        "monitor_interval": args.monitor_interval,
        "capture_microbatches": args.capture_microbatches,
    }
    adapter = NativeAdapter(
        trainer_class=trainer_module.Trainer,
        profiler_class=profiler_module.Profiler,
        metrics_class=metrics_module.MetricsProcessor,
        metrics_module=metrics_module,
        options=options,
        cuda_synchronize=torch.cuda.synchronize,
        memory_stats=device_module.device_module.memory_stats,
        deep_gemm_cache_redirect=DeepGemmCacheRedirect(
            deep_gemm_fast_warmup,
            trainer_module,
            Path(os.environ["TASK_DEEP_GEMM_RANK_CACHE_ROOT"]),
        ),
    )
    with adapter:
        from torchtitan.train import main as native_main

        native_main(train_argv)


if __name__ == "__main__":
    main()
