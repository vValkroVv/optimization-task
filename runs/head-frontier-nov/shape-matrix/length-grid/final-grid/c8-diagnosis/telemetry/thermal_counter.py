"""Sample the per-device NVML thermal-policy time counter. All calls are read-only."""
import csv
from pathlib import Path
import time

import pynvml as nvml

nvml.nvmlInit()
devices = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]
path = Path(__file__).parent / time.strftime("thermal-%Y%m%dT%H%M%SZ.csv", time.gmtime())
with path.open("x", buffering=1) as stream:
    writer = csv.writer(stream)
    writer.writerow(["physical_gpu", "nvml_utc_us", "thermal_violation_ns"])
    for _ in range(2000):
        for i, dev in enumerate(devices):
            q = nvml.nvmlDeviceGetViolationStatus(dev, nvml.NVML_PERF_POLICY_THERMAL)
            writer.writerow([i, q.referenceTime, q.violationTime])
        time.sleep(0.5)
