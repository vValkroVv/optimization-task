"""Profile one synchronized sample in the unchanged final-grid rank replay."""

import json
import os
from pathlib import Path
import sys
import time

import torch


FULL_HEAD = Path(__file__).resolve().parents[2] / "full-head"
sys.path.insert(0, str(FULL_HEAD))
import rank_replay


def is_timer(frame):
    return frame.f_code.co_name == "run" and frame.f_code.co_filename.endswith("/tuning/head_bench.py")


def main():
    output = Path(sys.argv[sys.argv.index("--out") + 1])
    rank = int(os.environ["RANK"])
    profile_sample = int(os.environ.get("C8_PHASE_PROFILE_SAMPLE", "0"))
    reset_base = torch.cuda.reset_peak_memory_stats
    stats_base = torch.cuda.memory_stats
    profiler = None
    marker = None
    timings = []

    def reset(*args, **kwargs):
        nonlocal marker
        result = reset_base(*args, **kwargs)
        if profiler is not None and is_timer(sys._getframe(1)):
            marker = torch.profiler.record_function("c8_phase_measured_call")
            marker.__enter__()
        return result

    completed_profiles = []

    def stats(*args, **kwargs):
        nonlocal profiler, marker
        caller = sys._getframe(1)
        if not is_timer(caller):
            return stats_base(*args, **kwargs)
        utc_offset = time.time_ns() - time.perf_counter_ns()
        profiled = marker is not None
        if profiled:
            marker.__exit__(None, None, None)
            profiler.__exit__(None, None, None)
            completed_profiles.append(profiler)
            marker = None
            profiler = None

        result = stats_base(*args, **kwargs)
        timings.append({
            **{f"{key}_utc_ns": round(caller.f_locals[key] * 1e9) + utc_offset
               for key in ("start", "submitted", "stop")},
            "profiled": profiled,
            **{key: result[key] for key in ("allocated_bytes.all.peak", "reserved_bytes.all.peak",
                                            "num_alloc_retries")},
        })
        if len(timings) + 1 == profile_sample:
            profiler = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
            ])
            profiler.__enter__()
        return result

    torch.cuda.reset_peak_memory_stats = reset
    torch.cuda.memory_stats = stats
    rank_replay.main()
    visible_uuid = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank]
    runtime_uuid = str(torch.cuda.get_device_properties(rank).uuid)
    normalize_uuid = lambda uuid: uuid.lower().removeprefix("gpu-").replace("-", "")
    if normalize_uuid(visible_uuid) != normalize_uuid(runtime_uuid):
        raise RuntimeError(f"Logical CUDA device {rank} did not resolve to {visible_uuid}: {runtime_uuid}")
    (output / f"physical-device-{rank}.json").write_text(
        json.dumps(dict(logical_rank=rank, visible_uuid=visible_uuid, runtime_uuid=runtime_uuid)) + "\n")
    (output / f"timing-{rank}.json").write_text(json.dumps(timings) + "\n")
    if profile_sample:
        if len(completed_profiles) != 1 or not timings[profile_sample - 1]["profiled"]:
            raise RuntimeError("The replay did not run the requested profiler sample")
        completed_profiles[0].export_chrome_trace(str(output / f"trace-{rank}.json"))


if __name__ == "__main__":
    main()
