"""Read all GPU thermal counters and flags during one eight-rank head diagnostic."""
import csv
import sys
import time
from pathlib import Path

import pynvml as n

out = Path(sys.argv[1])
stop = out.with_suffix(".stop")
n.nvmlInit()
handles = [n.nvmlDeviceGetHandleByIndex(i) for i in range(n.nvmlDeviceGetCount())]
assert len(handles) == 8
with out.open("x", buffering=1) as stream:
    w = csv.writer(stream)
    w.writerow(("utc_ns", "gpu", "temp_c", "memory_c", "sm_mhz", "watts", "util_pct", "hw_thermal", "sw_thermal", "power_brake", "sw_power_cap", "thermal_counter_ns"))
    for _ in range(4800):
        for i, h in enumerate(handles):
            mask = n.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            mem = n.nvmlDeviceGetFieldValues(h, [n.NVML_FI_DEV_MEMORY_TEMP])[0]
            temp = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
            clock = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
            power = n.nvmlDeviceGetPowerUsage(h)
            util = n.nvmlDeviceGetUtilizationRates(h).gpu
            event = n.nvmlDeviceGetViolationStatus(h, n.NVML_PERF_POLICY_THERMAL)
            w.writerow((event.referenceTime*1000, i, temp, mem.value.uiVal if mem.nvmlReturn == 0 else "", clock, round(power/1000, 2), util,
                        *(int(bool(mask & flag)) for flag in (n.nvmlClocksEventReasonHwThermalSlowdown,n.nvmlClocksEventReasonSwThermalSlowdown,n.nvmlClocksEventReasonHwPowerBrakeSlowdown,n.nvmlClocksEventReasonSwPowerCap)),event.violationTime))
        if stop.exists():
            break
        time.sleep(0.19)
