"""Record read-only NVML samples on GPUs 0–7 until the matrix finishes."""
import argparse
import csv
import importlib.metadata
import json
import socket
import sys
import time
from pathlib import Path

import pynvml as nvml


REASONS = {
    "idle": nvml.nvmlClocksEventReasonGpuIdle,
    "sw_power_cap": nvml.nvmlClocksEventReasonSwPowerCap,
    "hw_slowdown": nvml.nvmlClocksEventReasonHwSlowdown,
    "sw_thermal": nvml.nvmlClocksEventReasonSwThermalSlowdown,
    "hw_thermal": nvml.nvmlClocksEventReasonHwThermalSlowdown,
    "power_brake": nvml.nvmlClocksEventReasonHwPowerBrakeSlowdown,
}
FIELDS = {
    "memory_c": nvml.NVML_FI_DEV_MEMORY_TEMP,
    "sw_power_counter_ns": nvml.NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP,
    "sw_thermal_counter_ns": nvml.NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN,
    "hw_thermal_counter_ns": nvml.NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN,
    "power_brake_counter_ns": nvml.NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--duration", type=float, default=21600.0)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    assert args.duration > 0 and args.interval > 0
    nvml.nvmlInit()
    handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]
    assert len(handles) == 8, "Expected eight physical GPUs"
    started = time.monotonic()
    metadata = {
        "host": socket.gethostname(),
        "pid": __import__("os").getpid(),
        "argv": sys.argv,
        "started_utc_ns": time.time_ns(),
        "python": sys.executable,
        "nvml": nvml.nvmlSystemGetNVMLVersion(),
        "driver": nvml.nvmlSystemGetDriverVersion(),
        "pynvml_provider": {
            name: importlib.metadata.version(name)
            for name in importlib.metadata.packages_distributions().get("pynvml", [])
        },
        "gpu_uuids": [nvml.nvmlDeviceGetUUID(h) for h in handles],
        "supported_masks": [nvml.nvmlDeviceGetSupportedClocksThrottleReasons(h) for h in handles],
        "field_status_by_gpu": [
            {key: value.nvmlReturn for key, value in zip(
                FIELDS, nvml.nvmlDeviceGetFieldValues(h, list(FIELDS.values())), strict=True
            )}
            for h in handles
        ],
        "interval_s": args.interval,
        "duration_limit_s": args.duration,
        "reason_bits": REASONS,
    }
    stop = args.out.with_suffix(".stop")
    marker = args.out.with_suffix(".finished.json")
    series = 0
    last_alert = {}
    last_mask = [0] * len(handles)
    with args.out.open("x", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "series", "utc_ns", "counter_utc_ns", "gpu", "temp_c", "sm_mhz", "watts",
            "util_pct", "mem_used_mib", "mask", *REASONS, "thermal_counter_ns", *FIELDS,
        ))
        with args.out.with_suffix(".meta.json").open("x") as meta_stream:
            json.dump(metadata, meta_stream, indent=2)
        while not stop.exists() and time.monotonic() - started < args.duration:
            sample_started = time.monotonic()
            for i, handle in enumerate(handles):
                mask = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                mask_ns = time.time_ns()
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000
                util = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                used = nvml.nvmlDeviceGetMemoryInfo(handle).used / 1048576
                event = nvml.nvmlDeviceGetViolationStatus(handle, nvml.NVML_PERF_POLICY_THERMAL)
                fields = nvml.nvmlDeviceGetFieldValues(handle, list(FIELDS.values()))
                values = [
                    (f.value.uiVal if key == "memory_c" else f.value.ullVal)
                    if f.nvmlReturn == nvml.NVML_SUCCESS else ""
                    for key, f in zip(FIELDS, fields, strict=True)
                ]
                writer.writerow((
                    series, mask_ns, event.referenceTime * 1000, i, temp, clock, power, util, used,
                    mask, *(int(bool(mask & flag)) for flag in REASONS.values()),
                    event.violationTime, *values,
                ))
                alert = mask & (REASONS["sw_thermal"] | REASONS["hw_thermal"] | REASONS["power_brake"])
                if alert != last_mask[i] or (alert and sample_started - last_alert.get(i, 0) >= 5):
                    print(json.dumps({
                        "gpu": i, "utc_ns": mask_ns, "alert_mask": alert, "mask": mask,
                        "temp_c": temp, "hbm_c": values[0], "sm_mhz": clock,
                        "util_pct": util, "thermal_counter_ns": event.violationTime,
                    }), flush=True)
                    last_alert[i] = sample_started
                    last_mask[i] = alert
            series += 1
            time.sleep(max(0, args.interval - (time.monotonic() - sample_started)))
    nvml.nvmlShutdown()
    with marker.open("x") as marker_stream:
        json.dump({
            "samples_per_gpu": series,
            "ended_utc_ns": time.time_ns(),
            "stop": "stop_file" if stop.exists() else "duration",
        }, marker_stream, indent=2)


if __name__ == "__main__":
    main()
