# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""System metrics monitor callback."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, Optional

import psutil

import torch.distributed as tdist

from composer.core import Callback, Event, State
from composer.loggers import Logger
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)

__all__ = ['SystemMetricsMonitor']


class SystemMetricsMonitor(Callback):
    """Track system metrics."""

    def __init__(
            self,
            log_interval_batches: int = 10,
            log_gpu_metrics: bool = False,
            log_cpu_percentage: bool = True,
            log_cpu_memory: bool = True,
            log_swap_memory: bool = True,
            log_disk_memory: bool = True,
            log_disk_io: bool = True,
            log_network_io: bool = True,
        ) -> None:
        super().__init__()

        self.log_interval_batches = log_interval_batches
        if self.log_interval_batches <= 0:
            raise ValueError(f'log_interval_batches must be > 0, got {self.log_interval_batches}')

        self.log_gpu_metrics = log_gpu_metrics
        if self.log_gpu_metrics:
            try:
                import pynvml
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group='pynvml',
                                                    conda_package='pynvml',
                                                    conda_channel='conda-forge') from e
            pynvml.nvmlInit()

        self.log_cpu_percentage = log_cpu_percentage
        self.log_cpu_memory = log_cpu_memory
        self.log_swap_memory = log_swap_memory
        self.log_disk_memory = log_disk_memory
        self.log_disk_io = log_disk_io
        self.log_network_io = log_network_io

        self.node_leader_group = None

    @property
    def log_system_cpu_metrics(self) -> bool:
        return self.log_cpu_percentage or self.log_cpu_memory or self.log_swap_memory or self.log_disk_memory or self.log_disk_io or self.log_network_io

    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.INIT:
            if self.log_system_cpu_metrics:
                self.node_leader_group = dist.get_node_leader_group()
            return
        elif event == Event.BATCH_START:
            step = state.timestamp.batch.value + 1
        elif event == Event.EVAL_BATCH_START:
            step = state.eval_timestamp.batch.value + 1
        elif event == Event.PREDICT_BATCH_START:
            step = state.predict_timestamp.batch.value + 1
        else:
            return

        if step % self.log_interval_batches != 0:
            return

        system_metrics = {}
        if self.log_system_cpu_metrics and dist.get_local_rank() == 0:
            local_node_metrics = self._compute_node_metrics(node=dist.get_global_rank() // dist.get_local_world_size())
            group_size = tdist.get_world_size(group=self.node_leader_group) if self.node_leader_group is not None else 1
            all_node_metrics = self._gather_metrics(
                local_node_metrics,
                group=self.node_leader_group,
                group_size=group_size,
            )
            system_metrics.update(all_node_metrics)
            system_metrics.update(self._compute_internode_stats(all_node_metrics))

        if self.log_gpu_metrics:
            local_gpu_metrics = self._compute_gpu_metrics()
            all_gpu_metrics = self._gather_metrics(
                local_gpu_metrics,
                group=None,
                group_size=dist.get_world_size(),
            )
            system_metrics.update(all_gpu_metrics)

        if system_metrics and dist.get_global_rank() == 0:
            logger.log_metrics(system_metrics)

    def _gather_metrics(
        self,
        local_metrics: Dict[str, float],
        group: Optional[tdist.ProcessGroup],
        group_size: int,
    ) -> Dict[str, float]:
        if not (dist.is_available() and dist.is_initialized()) or group_size <= 1:
            return local_metrics

        gathered_metrics = [None] * group_size
        tdist.all_gather_object(gathered_metrics, local_metrics, group=group)

        all_metrics = {
            key: value
            for local_metrics in gathered_metrics
            for key, value in local_metrics.items()
        }
        return all_metrics
    
    def _compute_internode_stats(self, all_node_metrics: Dict[str, float]) -> Dict[str, float]:
        grouped_values: defaultdict[str, list[float]] = defaultdict(list)
        for key, value in all_node_metrics.items():
            if key.startswith('node') and '_' in key and isinstance(value, (int, float)):
                grouped_values[key.split('_', 1)[1]].append(float(value))

        return {
            f'system_{agg_name}/{metric_name}': agg_fn(values)
            for metric_name, values in grouped_values.items()
            for agg_name, agg_fn in {
                'avg': lambda xs: sum(xs) / len(xs),
                'max': max,
            }.items()
        }

    def _compute_gpu_metrics(self) -> Dict[str, float]:
        """
        Metrics for the GPU:
            - Memory (total, free, used)
            - GPU percentage
            - Memory percentage
            - GPU temperature
        """
        gpu_metrics: Dict[str, float] = {}

        import pynvml
        
        local_rank = dist.get_local_rank()
        global_rank = dist.get_global_rank()
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_metrics[f'system_metrics/gpu_memory_total/device{global_rank}'] = memory.total
        gpu_metrics[f'system_metrics/gpu_memory_free/device{global_rank}'] = memory.free
        gpu_metrics[f'system_metrics/gpu_memory_used/device{global_rank}'] = memory.used
        device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_metrics[f'system_metrics/gpu_percentage/device{global_rank}'] = device_utilization.gpu
        gpu_metrics[f'system_metrics/gpu_memory_percentage/device{global_rank}'] = device_utilization.memory
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_metrics[f'system_metrics/gpu_temperature/device{global_rank}'] = temperature

        return gpu_metrics

    def _compute_node_metrics(self, node: int) -> Dict[str, float]:
        """
        Metrics for the node:
            - CPU percentage
            - CPU memory (total, available, percent, used, free)
            - Swap memory (total, used, free, percent, sin, sout)
            - Disk memory (total, used, free, percent)
            - Disk IO (read_count, write_count, read_bytes, write_bytes, read_time
            - Network IO (bytes_sent, bytes_recv, packets_sent, packets_recv, errin, errout, dropin, dropout)
        """

        node_metrics = {}

        if self.log_cpu_percentage:
            cpu_percent = psutil.cpu_percent()
            node_metrics[f'system_metrics/cpu_percentage/node{node}'] = cpu_percent
        
        if self.log_cpu_memory:
            system_memory = psutil.virtual_memory()._asdict()
            for k, v in system_memory.items():
                node_metrics[f'system_metrics/cpu_memory_{k}/node{node}'] = v

        if self.log_swap_memory:
            swap_memory = psutil.swap_memory()._asdict()
            for k, v in swap_memory.items():
                node_metrics[f'system_metrics/swap_memory_{k}/node{node}'] = v
        
        if self.log_disk_memory:
            disk_usage = psutil.disk_usage(os.sep)._asdict()
            for k, v in disk_usage.items():
                node_metrics[f'system_metrics/disk_memory_{k}/node{node}'] = v
        
        if self.log_disk_io:
            disk_io_counters = psutil.disk_io_counters(perdisk=True)
            for disk_name, disk_stats in disk_io_counters.items():
                for field_name, value in disk_stats._asdict().items():
                    node_metrics[f'system_metrics/disk_io_{disk_name}/{field_name}/node{node}'] = value

        if self.log_network_io:
            network_usage = psutil.net_io_counters()._asdict()
            for k, v in network_usage.items():
                node_metrics[f'system_metrics/network_{k}/node{node}'] = v

        return node_metrics
