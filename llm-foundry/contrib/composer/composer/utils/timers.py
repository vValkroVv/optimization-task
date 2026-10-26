# Based on
# https://github.com/huggingface/nanotron/blob/main/src/nanotron/logging/timers.py

import time
import functools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class TimerType(Enum):
    CPU = "cpu"  # Regular CPU timer (uses time.time())
    CUDA = "cuda"  # CUDA-aware timer (uses CUDA events)


@dataclass
class TimerRecord:
    """
    Records timing information for a single timer.

    By default, uses CUDA events for timing GPU operations, which provides more accurate
    measurements of GPU execution time without forcing CPU-GPU synchronization.

    For CPU-only operations, you can use CPU-based timing by specifying timer_type=TimerType.CPU.
    """

    name: str
    timer_type: TimerType = TimerType.CPU   

    # timer_level = 0 - timer is disabled
    # timer_level = 1 - compute only "avg_ms" and "calls"
    # timer_level = 2 - compute "avg_ms", "calls", "min_ms" and "max_ms"
    # timer_level = 3 - compute "avg_ms", "calls", "min_ms", "max_ms" and can be used for inter-rank aggregation
    _timer_level: int = 1
    is_above_global_threshold: bool = False

    enabled: bool = True  # Allow individual timer to be enabled/disabled

    start_time: float = 0.0
    end_time: float = 0.0
    running: bool = False
    cuda_sync: bool = False  # Option to add CUDA synchronization for more accurate timings

    # For CPU timers we still track total_time
    _cpu_total_time: float = 0.0
    _cpu_durations: List[float] = field(default_factory=list)  # in ms

    # CUDA specific fields
    _cuda_events: List[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    _current_start_event: Optional[torch.cuda.Event] = None

    def __enter__(self):
        """Context manager support: Start the timer when entering a context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support: End the timer when exiting a context."""
        self.end()
        return False  # Don't suppress exceptions

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

    def start(self) -> "TimerRecord":
        """Start the timer."""
        if self.name == "dummy" or not self.is_active():  # disabled
            return self

        if self.running:
            logger.warning(f"Timer '{self.name}' already running. Restarting.")

        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available():
                # Synchronize before starting timing if requested
                if self.cuda_sync:
                    torch.cuda.synchronize()
                # Create a new start event - we'll create the end event when end() is called
                self._current_start_event = torch.cuda.Event(enable_timing=True)
                self._current_start_event.record()
            else:
                logger.warning("CUDA timer requested but CUDA is not available. Falling back to CPU timer.")
                self.timer_type = TimerType.CPU
                self.start_time = time.time()
        else:
            self.start_time = time.time()

        self.running = True
        return self

    def end(self) -> None:
        """End the timer, but don't compute elapsed time yet."""

        if self.name == "dummy" or not self.is_active():  # disabled
            return

        if not self.running:
            logger.warning(f"Timer '{self.name}' was not running. Ignoring end call.")
            return

        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available() and self._current_start_event is not None:
                # Synchronize before ending timing if requested
                if self.cuda_sync:
                    torch.cuda.synchronize()
                # Create and record an end event
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()

                # Store the start/end event pair for later querying
                self._cuda_events.append((self._current_start_event, end_event))
                self._current_start_event = None
            else:
                logger.warning("CUDA timer end called but CUDA events are not available.")
                self.timer_type = TimerType.CPU
                self.end_time = time.time()
                elapsed_s = self.end_time - self.start_time
                self._cpu_total_time += elapsed_s
                self._cpu_durations.append(elapsed_s * 1000.0)
        else:
            self.end_time = time.time()
            elapsed_s = self.end_time - self.start_time
            self._cpu_total_time += elapsed_s
            self._cpu_durations.append(elapsed_s * 1000.0)

        self.running = False

    def is_active(self) -> bool:
        """Check if the timer is active based on its level and the global timers threshold."""
        return self.enabled and self.timer_level > 0

    def compare_with_global_timer_threshold(self):
        """Update whether timer level is above global threshold."""
        self.is_above_global_threshold = self._timer_level > gigatimer.timers_threshold

    def set_timer_level(self, timer_level: int) -> None:
        """
        Set timer level and update is_above_global_threshold accordingly.
        
        timer_level = 0 - timer is disabled
        timer_level = 1 - compute only "avg_ms" and "calls"
        timer_level = 2 - compute "avg_ms", "calls", "min_ms" and "max_ms"
        timer_level = 3 - compute "avg_ms", "calls", "min_ms", "max_ms" and can be used for inter-rank aggregation
        """
        if timer_level not in (0, 1, 2, 3):
            raise ValueError(f"`timer_level` must be one of {{0, 1, 2, 3}}, got {timer_level}.")
        self._timer_level = timer_level
        self.compare_with_global_timer_threshold()

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = 0.0
        self.end_time = 0.0
        self.running = False
        self._cpu_total_time = 0.0
        self._cpu_durations = []
        self._cuda_events = []
        self._current_start_event = None

    def get_history_durations(self, timer_type: TimerType) -> List[float]:
        """Return per-call durations from selected timer type."""
        if timer_type == TimerType.CPU:
            return list(self._cpu_durations)

        durations = []
        for start_event, end_event in self._cuda_events:
            end_event.synchronize()
            durations.append(start_event.elapsed_time(end_event))
        return durations
    
    @property
    def timer_level(self) -> int:
        if self.is_above_global_threshold:
            return gigatimer.timers_threshold
        return self._timer_level

    def _check_stat_can_be_calculated(self, threshold: int) -> bool:
        return self.is_active() and self.timer_level >= threshold

    def get_calls(self) -> int:
        """Return number of completed calls in current history window."""
        if not self._check_stat_can_be_calculated(threshold=1): return 0
        
        if self.timer_type == TimerType.CPU:
            return len(self._cpu_durations)
        return len(self._cuda_events)

    def get_avg_ms(self, durations: List) -> float:
        """Return average call duration from selected timer type."""
        if not self._check_stat_can_be_calculated(threshold=1) or not durations: return -1.0
        return sum(durations) / len(durations)

    def get_min_ms(self, durations: List) -> float:
        """Return minimum call duration from selected timer type."""
        if not self._check_stat_can_be_calculated(threshold=2) or not durations: return -1.0
        return min(durations)

    def get_max_ms(self, durations: List) -> float:
        """Return maximum call duration from selected timer type."""
        if not self._check_stat_can_be_calculated(threshold=2) or not durations: return -1.0
        return max(durations)
    

class MetricsHandler:
    """Prepare timer metrics and rollups from aggregated timer stats."""

    def __init__(self, parse_timer_name: Callable[[str], tuple[str, str]]) -> None:
        self._parse_timer_name = parse_timer_name
        self._cumulative_namespace_ms: Dict[str, float] = {}

    def reset_cumulative_rollups(self) -> None:
        """Reset cumulative namespace sums."""
        self._cumulative_namespace_ms = {}

    def filter_active_timers_stats(
        self,
        timer_stats: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Keep timers with calls > 0 and drop incorrect min/max fields."""
        filtered_timer_stats = {}
        for timer_name, stats in timer_stats.items():
            calls = stats["calls"]
            if calls <= 0.0:
                continue

            filtered_stats = {"calls": calls, "avg_ms": stats["avg_ms"]}
            if stats["min_ms"] >= 0.0: filtered_stats["min_ms"] = stats["min_ms"]
            if stats["max_ms"] >= 0.0: filtered_stats["max_ms"] = stats["max_ms"]
            filtered_timer_stats[timer_name] = filtered_stats

        return filtered_timer_stats

    def process_metrics(
        self,
        timer_stats: Dict[str, Dict[str, float]],
        suffix: Optional[str] = None,
    ) -> Dict[str, float]:
        """Prepare metrics dict from per-timer aggregates."""
        metrics = {}
        for timer_name, stats in sorted(timer_stats.items()):
            calls = stats["calls"]
            
            namespace, name = self._parse_timer_name(timer_name)
            metric_prefix = f"{namespace}/{name}" if suffix is None else f"{namespace}/{name}_{suffix}"

            metrics[f"{metric_prefix}/calls"] = calls
            if "avg_ms" in stats:
                metrics[f"{metric_prefix}/avg_ms"] = stats["avg_ms"]
            if "min_ms" in stats:
                metrics[f"{metric_prefix}/min_ms"] = stats["min_ms"]
            if "max_ms" in stats:
                metrics[f"{metric_prefix}/max_ms"] = stats["max_ms"]
        return metrics

    def rollup_interval_sums(
        self,
        timer_stats: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Prepare interval sums by namespace from per-timer stats."""
        interval_namespace_ms = {}

        for timer_name, stats in timer_stats.items():
            if "avg_ms" not in stats:
                continue

            namespace, _ = self._parse_timer_name(timer_name)
            interval_sum_ms = stats["avg_ms"] * stats["calls"]
            interval_namespace_ms[namespace] = interval_namespace_ms.get(namespace, 0.0) + interval_sum_ms

        return interval_namespace_ms

    def update_cumulative_namespaces_rollups(
        self,
        interval_namespace_ms: Dict[str, float],
    ) -> None:
        """Add interval sums to cumulative namespace sums."""
        for namespace, interval_sum_ms in interval_namespace_ms.items():
            self._cumulative_namespace_ms[namespace] = (
                self._cumulative_namespace_ms.get(namespace, 0.0) + interval_sum_ms
            )

    def process_rollup_metrics(
        self,
        interval_namespace_ms: Dict[str, float],
    ) -> Dict[str, float]:
        """Prepare rollup metrics namespace interval and cumulative."""
        metrics = {}
        for namespace in sorted(interval_namespace_ms.keys()):
            metrics[f"{namespace}/sum_ms_interval"] = interval_namespace_ms[namespace]
            metrics[f"{namespace}/sum_ms_cumulative"] = self._cumulative_namespace_ms.get(namespace, 0.0)
        return metrics

    def process_namespace_metrics(
        self,
        timer_stats: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Prepare per-timer metrics and namespace rollups for one interval."""
        metrics = self.process_metrics(timer_stats)
        interval_namespace_ms = self.rollup_interval_sums(timer_stats)
        self.update_cumulative_namespaces_rollups(interval_namespace_ms)
        metrics.update(self.process_rollup_metrics(interval_namespace_ms))
        return metrics


class InterrankAggregationHandler:
    """Aggregate timer statistics across ranks."""

    @staticmethod
    def aggregate_timers_stats(
        timer_stats: Dict[str, Dict[str, float]],
        process_group=None,
    ) -> Dict[str, Dict[str, float]]:
        if not dist.is_available():
            raise RuntimeError("timers_aggregate_global requires torch.distributed to be available.")
        if not dist.is_initialized():
            raise RuntimeError("timers_aggregate_global requires torch.distributed to be initialized.")

        world_size = dist.get_world_size(group=process_group)

        local_timer_names = sorted(timer_stats.keys())
        gathered_names = [None] * world_size
        dist.all_gather_object(gathered_names, local_timer_names, group=process_group)
        global_timer_names = sorted({name for rank_names in gathered_names for name in rank_names})
        
        if len(global_timer_names) == 0:
            return {}

        local_sum_ms, local_calls, local_min, local_max = [], [], [], []
        local_min_presence, local_max_presence = [], []
        for timer_name in global_timer_names:
            if timer_name in timer_stats:
                current_timer_stats = timer_stats[timer_name]
                calls = current_timer_stats["calls"]
                has_min = "min_ms" in current_timer_stats
                has_max = "max_ms" in current_timer_stats
                avg_ms = current_timer_stats["avg_ms"]
                min_ms = current_timer_stats["min_ms"] if has_min else float("inf")
                max_ms = current_timer_stats["max_ms"] if has_max else float("-inf")
            else:
                calls = 0.0
                has_min = False
                has_max = False
                avg_ms = 0.0
                min_ms = float("inf")
                max_ms = float("-inf")

            local_calls.append(calls)
            local_sum_ms.append(avg_ms * calls)
            local_min.append(min_ms)
            local_max.append(max_ms)
            local_min_presence.append(1.0 if has_min else 0.0)
            local_max_presence.append(1.0 if has_max else 0.0)

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        calls_tensor = torch.tensor(local_calls, dtype=torch.float64, device=device)
        sum_ms_tensor = torch.tensor(local_sum_ms, dtype=torch.float64, device=device)
        min_tensor = torch.tensor(local_min, dtype=torch.float64, device=device)
        max_tensor = torch.tensor(local_max, dtype=torch.float64, device=device)
        presence_tensor = torch.tensor(
            [local_min_presence, local_max_presence],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(calls_tensor, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(sum_ms_tensor, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN, group=process_group)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX, group=process_group)
        dist.all_reduce(presence_tensor, op=dist.ReduceOp.SUM, group=process_group)

        global_aggregates = {}
        for idx, timer_name in enumerate(global_timer_names):
            calls = calls_tensor[idx].item()
            if calls <= 0.0:
                continue

            has_min = presence_tensor[0, idx].item() > 0.0
            has_max = presence_tensor[1, idx].item() > 0.0
            aggregated_timer_stats = {
                "calls": calls,
                "avg_ms": sum_ms_tensor[idx].item() / calls,
            }
            if has_min:
                aggregated_timer_stats["min_ms"] = min_tensor[idx].item()
            if has_max:
                aggregated_timer_stats["max_ms"] = max_tensor[idx].item()
            global_aggregates[timer_name] = aggregated_timer_stats

        return global_aggregates


class Timers:
    """
    A collection of timers for tracking execution time in Nanotron.

    Timers are stored in a namespaces:
    "{namespace: TimerRecord}".
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timers, cls).__new__(cls)
            cls._instance.timers_by_namespace: Dict[str, List[TimerRecord]] = {}
            cls._instance.interval: int = 10
            cls._instance.timers_threshold: int = 1
            cls._instance.metrics_handler = MetricsHandler(parse_timer_name=cls._instance.parse_timer_name)
            cls._instance.interrank_aggregation_handler = InterrankAggregationHandler()
        return cls._instance

    def set_interval(self, interval: int) -> None:
        """Set timer aggregation interval in steps."""
        if interval < 1:
            raise ValueError(f"'interval' must be >= 1, got {interval}.")
        self.interval = interval

    def set_timers_threshold(self, timers_threshold: int) -> None:
        """Set global timers threshold.

        Timers with timer_level above this threshold won't be active.
        """
        if timers_threshold not in (0, 1, 2, 3):
            raise ValueError(f"`timers_threshold` must be one of {{0, 1, 2, 3}}, got {timers_threshold}.")
        
        self.timers_threshold = timers_threshold
        for _, timer in self.iter_timers():
            if timers_threshold == 0 and timer.running:
                timer.running = False
                timer.start_time = 0.0
                timer.end_time = 0.0
                timer._current_start_event = None
            timer.compare_with_global_timer_threshold()

    def get_local_namespaces(self) -> List[str]:
        """Get list of timers namespaceses on the current rank"""
        return sorted(self.timers_by_namespace.keys())

    def get_namespaces_by_process_group(self, process_group=None) -> List[str]:
        """
        Get list of timers namespaces across ranks in this process group.
        If process_group is None, gets list of global namespaces across all ranks.
        """
        local_namespaces = self.get_local_namespaces()
        if not (dist.is_available() and dist.is_initialized()):
            return local_namespaces

        world_size = dist.get_world_size(group=process_group)
        gathered_namespaces = [None] * world_size
        dist.all_gather_object(gathered_namespaces, local_namespaces, group=process_group)
        return sorted({
            namespace
            for rank_namespaces in gathered_namespaces
            for namespace in rank_namespaces
        })
    
    @staticmethod
    def validate_timer_name(name: str) -> None:
        """Validate timer name format "<namespace>.<timer_name>"."""
        if not name:
            raise ValueError("Timer name must be non-empty.")

        parts = name.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Timer name '{name}' must have at least 2 parts: "
                "<namespace>.<timer_name>."
            )
        if any(part == "" for part in parts):
            raise ValueError(f"Timer name '{name}' contains empty path segments.")

        timer_name = ".".join(parts[1:])
        if timer_name == "":
            raise ValueError(f"Timer name '{name}' has empty timer leaf.")

    @staticmethod
    def parse_timer_name(name: str) -> tuple[str, str]:
        """Parse timer name into "(namespace, timer_name)"."""
        Timers.validate_timer_name(name)
        parts = name.split(".")
        namespace = parts[0]
        timer_name = ".".join(parts[1:])
        return namespace, timer_name

    @staticmethod
    def default_name_from_callable(func: Any) -> str:
        """Build default timer name for decorator-style usage without explicit name."""
        qualname_parts = func.__qualname__.split(".")
        group = qualname_parts[-2] if len(qualname_parts) >= 2 else "global"
        return f"timers_default.{group}.{func.__name__}"

    def get_timer(self, namespace: str, timer_name: str) -> Optional[TimerRecord]:
        """Extract a timer by its namespace and name from timers_by_namespace. Returns None if not found."""
        namespace_timers = self.get_timers_by_namespace(namespace)
        full_name = f"{namespace}.{timer_name}"
        for timer in namespace_timers:
            if timer.name == full_name:
                return timer
        return None

    def get_timers_by_namespace(self, namespace: str) -> List[TimerRecord]:
        """Return timers from this namespace."""
        return self.timers_by_namespace.get(namespace, [])

    def register_timer(self, namespace: str, timer: TimerRecord) -> None:
        """Register a timer in timers_by_namespace by its namespace and name, ensuring no duplicates."""
        timers = self.timers_by_namespace.setdefault(namespace, [])
        for existing in timers:
            if existing.name == timer.name:
                return
        timers.append(timer)

    def iter_timers(self):
        """Iterate over all timers."""
        for _, timers in self.timers_by_namespace.items():
            for timer in timers:
                yield timer.name, timer

    def __call__(
        self,
        name: Optional[Union[str, Callable]] = None,
        timer_type: Union[TimerType, str] = TimerType.CPU,
        timer_level: int = 1,
        cuda_sync: bool = False,
        enabled: bool = True,
    ) -> Union[TimerRecord, Callable]:
        if isinstance(timer_type, str):
            timer_type = TimerType(timer_type)

        def decorate(func):
            timer_name = self.default_name_from_callable(func)
            return self._create_timer_decorator(timer_name, timer_type, timer_level, cuda_sync, enabled)(func)

        if callable(name):
            return decorate(func)

        if name is None:
            return decorate

        namespace, timer_name = self.parse_timer_name(name)
        existing_timer = self.get_timer(namespace, timer_name)
        if existing_timer is not None:
            if existing_timer.timer_type != timer_type or existing_timer.cuda_sync != cuda_sync:
                raise ValueError(
                    f"Timer '{name}' already exists with different settings.\n"
                    f"Existing: type={existing_timer.timer_type}, cuda_sync={existing_timer.cuda_sync}, enabled={existing_timer.enabled}\n"
                    f"New: type={timer_type}, cuda_sync={cuda_sync}, enabled={enabled}"
                )
            existing_timer.enabled = enabled
            existing_timer.set_timer_level(timer_level)
            return existing_timer

        timer_record = TimerRecord(
            name=name,
            timer_type=timer_type,
            cuda_sync=cuda_sync,
            enabled=enabled,
        )
        timer_record.set_timer_level(timer_level)
        self.register_timer(namespace, timer_record)
        return timer_record

    def _create_timer_decorator(
            self, name: str, timer_type: TimerType=TimerType.CUDA,
            timer_level: int=1, cuda_sync: bool=False, enabled: bool=True
        ):
        """Create a decorator that times the execution of a function."""
        timer = self(name, timer_type, timer_level, cuda_sync, enabled)

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not timer.is_active():
                    return func(*args, **kwargs)
                with timer:
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def reset_all(self) -> None:
        for _, timer in self.iter_timers():
            timer.reset()
        self.reset_cumulative_rollups()

    def reset(self, namespace: str, name: str) -> None:
        timer = self.get_timer(namespace, name)
        if timer is not None:
            timer.reset()

    def reset_cumulative_rollups(self) -> None:
        """Reset cumulative namespace sums."""
        self.metrics_handler.reset_cumulative_rollups()

    def timer_aggregate_local(self, timer: Optional[TimerRecord]) -> Dict[str, float]:
        """Aggregate stats for one timer on the current rank and reset this timers."""
        result = {"avg_ms": 0.0, "calls": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        if timer is None or not timer.is_active():
            return result

        result["calls"] = timer.get_calls()
        durations = timer.get_history_durations(timer.timer_type)
        result["avg_ms"] = timer.get_avg_ms(durations)
        result["min_ms"] = timer.get_min_ms(durations)
        result["max_ms"] = timer.get_max_ms(durations)
        
        timer.reset()
        return result

    def timers_aggregate_local(self, timer_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """Aggregates statistics for a list of timer names on the current rank."""
        aggregated_statistics = {}
        for timer_name in timer_names:
            namespace, name = self.parse_timer_name(timer_name)
            timer = self.get_timer(namespace, name)
            aggregated_statistics[timer_name] = self.timer_aggregate_local(timer)
        return aggregated_statistics

    def timers_aggregate_global(
        self,
        timer_stats: Dict[str, Dict[str, float]],
        process_group=None,
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate per-timer stats across ranks in a process group.

        Args:
            timer_stats (Dict[str, Dict[str, float]]): Local per-timer metrics
            process_group: Optional process group. "None" means the default global group.

        Returns:
            Dict[str, Dict[str, float]]: Per-timer global metrics with keys:
                "avg_ms", "min_ms", "max_ms", "calls".
        """
        return self.interrank_aggregation_handler.aggregate_timers_stats(
            timer_stats,
            process_group=process_group,
        )

    def build_timers_metrics_by_namespace(self, namespace: str, process_group=None) -> Dict[str, float]:
        """Logs timers aggregations of one timers namespace and returns ready-to-log metrics."""
        if self.timers_threshold < 1:
            return {}

        namespace_timers = self.get_timers_by_namespace(namespace)
        timer_names = sorted({timer.name for timer in namespace_timers})
        interrank_timer_names = sorted({
            timer.name
            for timer in namespace_timers
            if timer.timer_level == 3
        })
        timer_stats = self.timers_aggregate_local(timer_names)
        timer_stats = self.metrics_handler.filter_active_timers_stats(timer_stats)

        metrics = self.metrics_handler.process_namespace_metrics(timer_stats)

        if self.timers_threshold == 3:
            interrank_timer_stats = {
                timer_name: timer_stats[timer_name]
                for timer_name in interrank_timer_names
                if timer_name in timer_stats
            }
            global_timer_aggregates = self.timers_aggregate_global(
                interrank_timer_stats,
                process_group=process_group,
            )
            metrics.update(self.metrics_handler.process_metrics(global_timer_aggregates, suffix='global'))
        return metrics

    def build_all_timers_metrics(self, process_group=None) -> Dict[str, float]:
        """Aggregate timers of all namespaces and return ready-to-log metrics."""
        if self.timers_threshold < 1:
            return {}

        if self.timers_threshold < 3:
            namespaces = self.get_local_namespaces()
        else:
            namespaces = self.get_namespaces_by_process_group(process_group=process_group)

        metrics = {}
        for namespace in namespaces:
            metrics.update(self.build_timers_metrics_by_namespace(namespace, process_group=process_group))

        return metrics


# Create a singleton instance
gigatimer = Timers()


def callback_timer(timer_type=TimerType.CPU, timer_level=1):
    """
    Callback-aware timer decorator with automatic naming.
    
    timers_threshold is an upper bound on timer levels to aggregate statistics:
    timer_level = 0 - all timers disabled, 
    timer_level = 1 - aggregates avg and calls, 
    timer_level = 2 - aggregates avg, calls, min and max, 
    timer_level = 3 - aggregates avg, calls, min, max and interrank aggregation
    if timer_level > timers_threshold, timer_level will be decreased to timers_threshold
    
    """

    def decorator(fn):
        qualname_parts = [part for part in fn.__qualname__.split(".")]
        cls_name = qualname_parts[-2] if len(qualname_parts) >= 2 else "UnknownCallback"
        method_name = fn.__name__
        name = f"timers_callback.{cls_name}.{method_name}"
        gigatimer.validate_timer_name(name)
        return gigatimer(name, timer_type=timer_type, timer_level=timer_level)(fn)

    return decorator
