import os
from typing import Any, TypeAlias

import torch.distributed as dist

from torchtitan.config.schema.callbacks.builtin.system_monitor import (
    SystemMetricsMonitorConfig,
)
from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.config import Configurable
from torchtitan.tools.utils import device_memory_usage, device_module


class SystemMetricsMonitor(Configurable, Observer):
    """Logs everything about the machine: host, GPU, and the device allocator.

    Two independent metric families, each with its own cadence because they
    cost different amounts and llm-f runs them as separate callbacks:

    - ``memory/*`` (``memory_log_interval``) — allocator counters from
      ``device_module.memory_stats()``, rank-local, no collectives, no
      third-party dependency. Device totals include non-PyTorch allocations
      reported by ``mem_get_info()``. Byte counters become GiB;
      count counters stay counts.
    - ``system_metrics/*`` (``host_log_interval``) — psutil host counters and
      NVML GPU counters. Host metrics are collected once per node by the local
      leader rank (LOCAL_RANK 0) and all-gathered; nodes are numbered by the
      global rank of their leader, so ``node0`` is the node hosting rank 0.
      GPU metrics are collected by each rank for its own device and tagged with
      the global rank. Memory sizes are GiB; percentages and counters retain
      their units.

    An interval of 0 turns its family off.

    Example:
        ``callbacks["system_metrics"] = SystemMetricsMonitor.Config(
        memory_log_interval=10)`` in the preset

    Logged metrics:
        - ``memory/current_allocated_mem``
        - ``memory/current_active_mem``
        - ``memory/current_inactive_mem``
        - ``memory/current_reserved_mem``
        - ``memory/peak_allocated_mem``
        - ``memory/peak_active_mem``
        - ``memory/peak_inactive_mem``
        - ``memory/peak_reserved_mem``
        - ``memory/alloc_retries``
        - ``memory/non_torch_mem``
        - ``memory/total_peak_mem``
        - ``system_metrics/cpu_percentage/node{node_idx}``
        - ``system_metrics/cpu_memory_<field>/node{node_idx}``
        - ``system_metrics/swap_memory_<field>/node{node_idx}``
        - ``system_metrics/disk_memory_<field>/node{node_idx}``
        - ``system_metrics/disk_io_<disk>/<field>/node{node_idx}``
        - ``system_metrics/network_<field>/node{node_idx}``
        - ``system_metrics/gpu_memory_<field>/device{global_rank}`` when NVML is available
        - ``system_metrics/gpu_percentage/device{global_rank}`` when NVML is available
        - ``system_metrics/gpu_memory_percentage/device{global_rank}`` when NVML is available
        - ``system_metrics/gpu_temperature/device{global_rank}`` when NVML is available
    """

    _MEMORY_KEYS = {
        "allocated_bytes.all.current": "current_allocated_mem",
        "active_bytes.all.current": "current_active_mem",
        "inactive_split_bytes.all.current": "current_inactive_mem",
        "reserved_bytes.all.current": "current_reserved_mem",
        "allocated_bytes.all.peak": "peak_allocated_mem",
        "active_bytes.all.peak": "peak_active_mem",
        "inactive_split_bytes.all.peak": "peak_inactive_mem",
        "reserved_bytes.all.peak": "peak_reserved_mem",
        "num_alloc_retries": "alloc_retries",
    }

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = SystemMetricsMonitorConfig

    def __init__(self, config: Config) -> None:
        self.config = config
        # (pynvml, device_handle), filled in on the first GPU read.
        self._nvml: tuple[Any, Any] | None = None
        self._nvml_usable = bool(config.host_log_interval)
        if config.host_log_interval:
            try:
                import psutil
            except ImportError as exc:
                raise ValueError(
                    "SystemMetricsMonitor host metrics are enabled but psutil "
                    "is not installed. Install it or set callbacks.system_metrics.host_log_interval=0."
                ) from exc

            psutil.cpu_percent()

    def batch_end(self, state: ObserverState) -> None:
        memory_interval = self.config.memory_log_interval
        host_interval = self.config.host_log_interval

        metrics: dict[str, float] = {}
        if memory_interval and state.step % memory_interval == 0:
            metrics.update(self.get_memory_report())
        if host_interval and state.step % host_interval == 0:
            metrics.update(self.get_host_report())

        if metrics:
            state.record_all(metrics)

    @staticmethod
    def _to_gib(memory_in_bytes: int | float) -> float:
        return float(f"{memory_in_bytes / (1 << 30):.5g}")

    @classmethod
    def get_memory_report(cls) -> dict[str, float]:
        memory_stats = device_module.memory_stats()
        report: dict[str, float] = {}
        for torch_name, tag_name in cls._MEMORY_KEYS.items():
            if torch_name not in memory_stats:
                continue
            value = memory_stats[torch_name]
            if "bytes" in torch_name:
                report[f"memory/{tag_name}"] = cls._to_gib(value)
            else:
                report[f"memory/{tag_name}"] = float(value)
        non_torch_gib, total_peak_gib = device_memory_usage(memory_stats)
        report["memory/non_torch_mem"] = non_torch_gib
        report["memory/total_peak_mem"] = total_peak_gib
        return report

    def get_host_report(self) -> dict[str, float]:
        """Read this rank's counters, share them world-wide, name the tags."""
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        is_node_leader = int(os.environ.get("LOCAL_RANK", 0)) == 0

        payload = {
            "rank": rank,
            "node": self._read_node_counters() if is_node_leader else {},
            "gpu": self._read_gpu_counters(),
        }

        if distributed:
            gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, payload)
            payloads = [item for item in gathered if item is not None]
        else:
            payloads = [payload]

        report: dict[str, float] = {}
        leaders = sorted(
            (item for item in payloads if item["node"]), key=lambda item: item["rank"]
        )

        for node_idx, leader in enumerate(leaders):
            for name, value in leader["node"].items():
                report[f"system_metrics/{name}/node{node_idx}"] = float(value)
        for item in payloads:
            for name, value in item["gpu"].items():
                report[f"system_metrics/{name}/device{item['rank']}"] = float(value)
        return report

    def _read_node_counters(self) -> dict[str, float]:
        import psutil

        metrics: dict[str, float] = {}
        metrics["cpu_percentage"] = float(psutil.cpu_percent())

        for name, value in psutil.virtual_memory()._asdict().items():
            metrics[f"cpu_memory_{name}"] = (
                float(value) if name == "percent" else self._to_gib(value)
            )

        for name, value in psutil.swap_memory()._asdict().items():
            metrics[f"swap_memory_{name}"] = (
                self._to_gib(value)
                if name in ("total", "used", "free")
                else float(value)
            )

        for name, value in psutil.disk_usage(os.sep)._asdict().items():
            metrics[f"disk_memory_{name}"] = (
                float(value) if name == "percent" else self._to_gib(value)
            )

        disk_io_counters = psutil.disk_io_counters(perdisk=True) or {}
        for disk_name, disk_stats in disk_io_counters.items():
            for field_name, value in disk_stats._asdict().items():
                metrics[f"disk_io_{disk_name}/{field_name}"] = float(value)

        for name, value in psutil.net_io_counters()._asdict().items():
            metrics[f"network_{name}"] = float(value)

        return metrics

    def _read_gpu_counters(self) -> dict[str, float]:
        """NVML counters for this rank's own device, {} when NVML is unusable."""
        if not self._nvml_usable:
            return {}

        if self._nvml is None:
            try:
                import pynvml

                pynvml.nvmlInit()
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(local_rank))
            except Exception:
                # No NVML on this host: give up for good instead of retrying
                # the import on every logged step.
                self._nvml_usable = False
                return {}

        pynvml, handle = self._nvml
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU,
            )
        except Exception:
            return {}

        return {
            "gpu_memory_total": self._to_gib(memory.total),
            "gpu_memory_free": self._to_gib(memory.free),
            "gpu_memory_used": self._to_gib(memory.used),
            "gpu_percentage": float(utilization.gpu),
            "gpu_memory_percentage": float(utilization.memory),
            "gpu_temperature": float(temperature),
        }
