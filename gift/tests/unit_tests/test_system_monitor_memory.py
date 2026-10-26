import ctypes
import logging
import sys
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torchtitan.callbacks.builtin.system_monitor import SystemMetricsMonitor
from torchtitan.components.metrics import MetricsProcessor, TensorBoardLogger
from torchtitan.tools import utils


def _counters(**values):
    return namedtuple("Counters", values)(**values)


def test_host_memory_sizes_use_gib_without_rescaling_io(monkeypatch):
    gib = 1 << 30
    psutil = SimpleNamespace(
        cpu_percent=lambda: 12.5,
        virtual_memory=lambda: _counters(
            total=64 * gib, available=16 * gib, used=48 * gib, percent=75.0
        ),
        swap_memory=lambda: _counters(
            total=8 * gib,
            used=2 * gib,
            free=6 * gib,
            percent=25.0,
            sin=3 * gib,
            sout=4 * gib,
        ),
        disk_usage=lambda _: _counters(
            total=1024 * gib, used=256 * gib, free=768 * gib, percent=25.0
        ),
        disk_io_counters=lambda **_: {
            "sda": _counters(
                read_count=3,
                write_count=4,
                read_bytes=gib,
                write_bytes=2 * gib,
                read_time=15,
                write_time=25,
                busy_time=50,
            )
        },
        net_io_counters=lambda: _counters(
            bytes_sent=3 * gib, bytes_recv=4 * gib, packets_sent=10, packets_recv=20
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    monitor = SystemMetricsMonitor(
        SystemMetricsMonitor.Config(host_log_interval=0, memory_log_interval=1)
    )

    assert monitor._read_node_counters() == {
        "cpu_percentage": 12.5,
        "cpu_memory_total": 64.0,
        "cpu_memory_available": 16.0,
        "cpu_memory_used": 48.0,
        "cpu_memory_percent": 75.0,
        "swap_memory_total": 8.0,
        "swap_memory_used": 2.0,
        "swap_memory_free": 6.0,
        "swap_memory_percent": 25.0,
        "swap_memory_sin": 3 * gib,
        "swap_memory_sout": 4 * gib,
        "disk_memory_total": 1024.0,
        "disk_memory_used": 256.0,
        "disk_memory_free": 768.0,
        "disk_memory_percent": 25.0,
        "disk_io_sda/read_count": 3,
        "disk_io_sda/write_count": 4,
        "disk_io_sda/read_bytes": gib,
        "disk_io_sda/write_bytes": 2 * gib,
        "disk_io_sda/read_time": 15,
        "disk_io_sda/write_time": 25,
        "disk_io_sda/busy_time": 50,
        "network_bytes_sent": 3 * gib,
        "network_bytes_recv": 4 * gib,
        "network_packets_sent": 10,
        "network_packets_recv": 20,
    }


def test_nvml_memory_sizes_use_gib_without_rescaling_utilization():
    gib = 1 << 30
    nvml = SimpleNamespace(
        nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(
            total=80 * gib, free=60 * gib, used=20 * gib
        ),
        nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=42, memory=55),
        nvmlDeviceGetTemperature=lambda *_: 67,
        NVML_TEMPERATURE_GPU=0,
    )
    monitor = SystemMetricsMonitor(
        SystemMetricsMonitor.Config(host_log_interval=0, memory_log_interval=1)
    )
    monitor._nvml_usable = True
    monitor._nvml = (nvml, object())

    assert monitor._read_gpu_counters() == {
        "gpu_memory_total": 80.0,
        "gpu_memory_free": 60.0,
        "gpu_memory_used": 20.0,
        "gpu_percentage": 42.0,
        "gpu_memory_percentage": 55.0,
        "gpu_temperature": 67.0,
    }


@pytest.mark.parametrize("validation", [False, True], ids=["train", "validation"])
def test_console_memory_matches_logged_total_peak(
    monkeypatch, caplog, tmp_path, validation
):
    gib = 1 << 30
    readings = iter([(66 * gib, 80 * gib), (65 * gib, 80 * gib)])
    device = SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda _: SimpleNamespace(total_memory=80 * gib),
        memory_stats=lambda: {
            "active_bytes.all.peak": 11 * gib,
            "reserved_bytes.all.current": 12 * gib,
            "reserved_bytes.all.peak": 18 * gib,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        },
        mem_get_info=lambda: next(readings),
        reset_peak_memory_stats=lambda: None,
    )
    monkeypatch.setattr(utils, "device_module", device)
    monkeypatch.setattr(
        "torchtitan.callbacks.builtin.system_monitor.device_module", device
    )
    monkeypatch.setattr(utils, "_capacity_bytes", None)
    processor = MetricsProcessor.__new__(MetricsProcessor)
    log_dir = str(tmp_path / "tb")
    processor.logger = TensorBoardLogger(log_dir)
    processor.color = utils.NoColor()
    monkeypatch.setattr(processor, "interval_metrics", lambda: {"tps": 100.0})

    monitor = SystemMetricsMonitor(
        SystemMetricsMonitor.Config(host_log_interval=0, memory_log_interval=1)
    )
    try:
        with caplog.at_level(logging.INFO):
            monitor.batch_end(
                SimpleNamespace(
                    step=10,
                    record_all=lambda metrics: processor.log_extra_metrics(10, metrics),
                )
            )
            if validation:
                processor.log_validation(loss=1.0, step=10)
            else:
                processor.log(
                    step=10, global_avg_loss=1.0, global_max_loss=1.0, grad_norm=0.5
                )
    finally:
        processor.close()

    events = EventAccumulator(log_dir).Reload()
    prefix = "memory/valid" if validation else "memory/train"
    for name, expected in [("non_torch_mem", 3.0), ("total_peak_mem", 21.0)]:
        assert [(e.step, e.value) for e in events.Scalars(f"{prefix}/{name}")] == [
            (10, expected)
        ]
        assert [(e.step, e.value) for e in events.Scalars(f"memory/{name}")] == [
            (10, expected - 1.0)
        ]
    capacity_prefix = "memory/valid" if validation else "memory"
    assert events.Scalars(f"{capacity_prefix}/max_reserved_pct")[0].value == 22.5
    assert events.Scalars(f"{capacity_prefix}/max_active_pct")[0].value == 13.75
    assert not any(tag.endswith("/total_peak_pct") for tag in events.Tags()["scalars"])
    assert "memory: 21.00GiB  non_torch: 3.00GiB" in caplog.text


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("builtin", [False, True], ids=["callback", "builtin"])
def test_total_peak_includes_reserved_cache_and_direct_cuda_memory(builtin):
    torch.cuda.empty_cache()
    tensor = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    del tensor
    torch.cuda.reset_peak_memory_stats()
    reserved_peak = torch.cuda.max_memory_reserved()
    assert reserved_peak > torch.cuda.max_memory_allocated()
    prefix = "memory/train" if builtin else "memory"
    before = (
        utils.device_capacity_report()[0]
        if builtin
        else SystemMetricsMonitor.get_memory_report()
    )
    assert before[f"{prefix}/total_peak_mem"] == pytest.approx(
        reserved_peak / (1 << 30) + before[f"{prefix}/non_torch_mem"], rel=1e-4
    )

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    pointer = ctypes.c_uint64()
    size = 64 << 20
    assert cuda.cuMemAlloc_v2(ctypes.byref(pointer), size) == 0
    try:
        after = (
            utils.device_capacity_report()[0]
            if builtin
            else SystemMetricsMonitor.get_memory_report()
        )
        assert torch.cuda.max_memory_reserved() == reserved_peak
        for name in ("non_torch_mem", "total_peak_mem"):
            assert after[f"{prefix}/{name}"] == pytest.approx(
                before[f"{prefix}/{name}"] + size / (1 << 30), rel=1e-4
            )
    finally:
        assert cuda.cuMemFree_v2(pointer) == 0
        torch.cuda.empty_cache()
