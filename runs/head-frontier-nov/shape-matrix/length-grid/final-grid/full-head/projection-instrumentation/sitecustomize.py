"""Record the preserved timer's timestamps after each measured sample stops."""

import atexit
import json
import os
from pathlib import Path
import sys
import time


if "LOCAL_RANK" in os.environ and "--out" in sys.argv:
    import torch

    memory_stats = torch.cuda.memory_stats
    rank = int(os.environ["LOCAL_RANK"])
    output = Path(sys.argv[sys.argv.index("--out") + 1]) / f"timing-{rank}.json"
    samples = []

    def timed_memory_stats(*args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_code.co_name == "run" and caller.f_code.co_filename.endswith("/tuning/head_bench.py"):
            utc_offset = time.time_ns() - time.perf_counter_ns()
            samples.append({f"{label}_utc_ns": round(caller.f_locals[label] * 1e9) + utc_offset
                            for label in ("start", "submitted", "stop")})
        return memory_stats(*args, **kwargs)

    def save():
        if samples:
            output.write_text(json.dumps(samples) + "\n")

    torch.cuda.memory_stats = timed_memory_stats
    atexit.register(save)
