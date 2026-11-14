#!/usr/bin/env python3
"""Read native FlashQLA artifacts without importing training code or touching GPUs.

Usage: python3 analyze_native.py /path/to/raw/fqla-native-fla-001 [...]
JSON goes to stdout. No files are written. Incomplete runs remain incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

WORLD = 8
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"step:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+).*?tps:\s*([\d,]+)")
ACTIVATION = re.compile(r"GDN_BACKEND_ACTIVATED variant=(\w+) grad_enabled=(True|False)")


def stats(values):
    values = list(values)
    if not values:
        return None
    ordered = sorted(values)
    def q(p):
        i = (len(ordered) - 1) * p
        lo = int(i)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (i - lo)
    med = statistics.median(values)
    return dict(n=len(values), median=med, min=min(values), max=max(values),
                p10=q(.1), p90=q(.9), mad=statistics.median(abs(x-med) for x in values),
                first20_median=statistics.median(values[:20]),
                last20_median=statistics.median(values[-20:]),
                last20_vs_first20_percent=(100*(statistics.median(values[-20:])/statistics.median(values[:20])-1)
                                          if len(values) >= 40 and statistics.median(values[:20]) else None))


def analyze(run):
    errors, unavailable = [], []
    def load(name, required=True):
        path = run / name
        if not path.is_file():
            if required:
                errors.append("missing " + name)
            return {}
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected object")
            return value
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
            return {}
    def require(condition, message):
        if not condition:
            errors.append(message)

    metadata = load("metadata.json")
    completion = load("completion.json")
    summary = load("summary.json")
    bridge = load("flash-qla-bridge.json")
    steps = metadata.get("steps", 0)
    warmup = metadata.get("warmup_updates", 0)
    window = metadata.get("window_updates", 0)
    expected_id = run.name
    require(metadata.get("run_id") == expected_id, "metadata run_id mismatch")
    require(completion.get("status") == "PASS", "completion not PASS")
    require(completion.get("source_unchanged") is True, "source unchanged not established")
    require(completion.get("config_unchanged") is True, "config unchanged not established")
    require(summary.get("status") == "complete" and not summary.get("errors"), "native parser did not accept run")
    exit_path = run / "exit-code.txt"
    require(exit_path.is_file() and exit_path.read_text().strip() == "0", "missing/nonzero process exit code")
    require(isinstance(steps, int) and steps > 0, "invalid expected step count")
    require(isinstance(warmup, int) and isinstance(window, int) and window >= 0, "invalid window")
    if not isinstance(steps, int) or not isinstance(warmup, int) or not isinstance(window, int):
        steps, warmup, window = 0, 0, 0
    if window:
        require(steps == warmup + window, "window does not reach final update")
    variant = bridge.get("backend_variant")
    require(variant in ("fla", "external_gate", "fused_gate"), "missing/invalid backend")
    records, bootstraps, rank_logs, cadence, memory = {}, {}, {}, {}, {}
    for rank in range(WORLD):
        key = str(rank)
        rec = load(f"measurement/rank-{rank:02d}.json")
        boot = load(f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json")
        if rec:
            records[key] = rec
            require(rec.get("schema_version") == 1 and rec.get("rank") == rank and rec.get("world_size") == WORLD,
                    f"rank {rank} schema/rank/world mismatch")
            require(rec.get("run_id") == expected_id and rec.get("expected_steps") == steps
                    and rec.get("completed_steps") == steps and rec.get("training_status") == "complete"
                    and rec.get("training_error") is None, f"rank {rank} training incomplete")
            series = rec.get("cadence", [])
            require([x.get("step") for x in series] == list(range(1, steps+1)),
                    f"rank {rank} cadence not contiguous")
            selected = [dict(step=x["step"], interval_s=x["interval_s"]) for x in series
                        if warmup < x.get("step", 0) <= warmup+window and x.get("interval_s") is not None]
            cadence[key] = dict(series=selected, seconds=stats(x["interval_s"] for x in selected))
            if window:
                m = rec.get("measurement") or {}
                require(rec.get("measurement_status") == "complete" and bool(m), f"rank {rank} measurement incomplete")
                require(m.get("start_after_update") == warmup and m.get("end_after_update") == steps
                        and m.get("window_updates") == window and m.get("cuda_synchronize_calls") == 2,
                        f"rank {rank} window/synchronization mismatch")
                require(len(selected) == window, f"rank {rank} measured cadence length mismatch")
                mem = m.get("memory", {})
                memory[key] = dict(peak_bytes=mem.get("peak_bytes"), peak_gib={
                    k: v/1024**3 for k, v in (mem.get("peak_bytes") or {}).items()},
                    allocator_counter_deltas=mem.get("allocator_counter_deltas"))
        if boot:
            bootstraps[key] = boot
            require(boot.get("rank") == rank and boot.get("backend_variant") == variant
                    and boot.get("capture_mode") == bridge.get("capture_mode"),
                    f"rank {rank} bootstrap mismatch")
            require(boot.get("source_identity") == bridge.get("source_identity"),
                    f"rank {rank} bootstrap source identity mismatch")
        # torchrun attempt_0 is required: ambiguous retries must not be silently merged.
        files = sorted((run / "logs/torchrun").glob(f"*/attempt_*/{rank}/stdout.log"))
        require(len(files) == 1, f"rank {rank} stdout absent/ambiguous")
        observed, markers = [], set()
        for path in files:
            for line in path.read_text(errors="replace").splitlines():
                text = ANSI.sub("", line)
                markers.update(ACTIVATION.findall(text))
                match = STEP.search(text)
                if match:
                    step, loss, grad, tps = match.groups()
                    observed.append(dict(step=int(step), loss=float(loss), grad_norm=float(grad),
                                         displayed_physical_tps_per_gpu=int(tps.replace(",", ""))))
        require(all(math.isfinite(x["loss"]) and math.isfinite(x["grad_norm"]) for x in observed),
                f"rank {rank} nonfinite loss/grad norm")
        # Normal rank filtering can hide step logs. Rank0 is the numerical reporting source.
        if rank == 0:
            require([x["step"] for x in observed] == list(range(1, steps+1)), "rank0 step logs not complete/unique")
        native = bridge.get("capture_mode") == "disabled"
        if native:
            require((variant, "True") in markers and (variant, "False") in markers,
                    f"rank {rank} forward/grad-enabled activation not observed")
            require(all(v == variant for v, _ in markers), f"rank {rank} unexpected active backend")
        rank_logs[key] = dict(paths=[str(p.relative_to(run)) for p in files],
                              activation=sorted([dict(variant=v, grad_enabled=g == "True") for v,g in markers],
                                                key=lambda x:(x["variant"],x["grad_enabled"])),
                              step_count=len(observed))
        if rank == 0:
            numerical = observed

    config = run / "native/config/resolved_config.yaml"
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else None
    require(config_hash is not None and completion.get("resolved_config_sha256") == config_hash,
            "resolved config missing or hash not accepted by completion")
    throughput = None
    if window and len(records) == WORLD and all(r.get("measurement") for r in records.values()):
        measurements = [r["measurement"] for r in records.values()]
        require(len({r.get("host_name") for r in records.values()}) == 1, "monotonic envelope spans different hosts")
        elapsed = (max(m["end_monotonic_ns"] for m in measurements)
                   - min(m["start_monotonic_ns"] for m in measurements))/1e9
        physical = sum(m["physical_input_tokens_local"] for m in measurements)
        supervised = {m["supervised_tokens_global"] for m in measurements}
        require(len(supervised) == 1, "global supervised counts disagree")
        require(elapsed > 0, "nonpositive integrated elapsed")
        require(all(m["physical_input_tokens_local"] == window*2*4096*32 for m in measurements),
                "physical token geometry differs from native recipe")
        require(supervised == {window*2096640}, "supervised token geometry differs from native recipe")
        for rank, m in enumerate(measurements):
            require(abs(m["local_elapsed_s"]-(m["end_monotonic_ns"]-m["start_monotonic_ns"])/1e9) < 1e-6,
                    f"rank {rank} elapsed disagrees with timestamps")
        if elapsed > 0 and len(supervised) == 1:
            supervised_once = next(iter(supervised))
            throughput = dict(global_envelope_s=elapsed, physical_tokens_global=physical,
                              supervised_tokens_global=supervised_once,
                              physical_tps_global=physical/elapsed, physical_tps_per_gpu=physical/elapsed/WORLD,
                              supervised_tps_global=supervised_once/elapsed,
                              supervised_tps_per_gpu=supervised_once/elapsed/WORLD)
    if not window:
        unavailable.append("capture/cadence-only artifact: no measured window or throughput")
    if not memory:
        unavailable.append("no completed per-rank measured memory sidecars")
    native_series = [x for x in numerical if warmup < x["step"] <= warmup+window]
    result = dict(schema=1, run_id=expected_id, run_path=str(run), status="complete" if not errors else "incomplete",
                  measurement_status=("complete" if not errors and window else "disabled" if not errors else "incomplete"),
                  errors=errors, unavailable=unavailable, expected=dict(steps=steps,warmup=warmup,window=window),
                  variant=variant, capture_mode=bridge.get("capture_mode"),
                  completed_rank_sidecars=len(records), bootstraps=len(bootstraps),
                  identity=dict(source=bridge.get("source_identity"),metadata_source=metadata.get("source_snapshot_before"),
                                config_hashes=metadata.get("config_hashes_before"),resolved_config_sha256=config_hash,
                                resolved_config_path=str(config),completion=completion,
                                installer_by_rank={k:v.get("installer_identity") for k,v in bootstraps.items()}),
                  configuration=dict(argv=metadata.get("argv"),seed=metadata.get("seed"),
                                     scheduler_total_steps=metadata.get("scheduler_total_steps"),
                                     scheduler_warmup_steps=metadata.get("scheduler_warmup_steps"),
                                     compiler_cache_root=metadata.get("compiler_cache_root"),
                                     runtime=metadata.get("runtime")),
                  activation_and_logs=rank_logs, throughput=throughput if not errors else None,
                  cadence_by_rank=cadence, memory_by_rank=memory,
                  numerical_rank0=dict(series=numerical,scope="rounded native stdout metrics; short trajectory, not convergence or elementwise parity"),
                  displayed_tps_rank0=dict(series=native_series,
                      stats=stats(x["displayed_physical_tps_per_gpu"] for x in native_series),
                      scope="rounded per-GPU physical native logging-interval TPS, separate from synchronized integrated window"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    results = []
    for path in args.runs:
        try:
            result = analyze(path.resolve())
            json.dumps(result, allow_nan=False)
            results.append(result)
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, ZeroDivisionError) as exc:
            results.append(dict(run_path=str(path),status="incomplete",errors=[f"artifact/schema error: {exc}"]))
    print(json.dumps(dict(schema=1, runs=results, interpretation={
        "integrated_latency": "seconds; same-host max rank end minus min rank start; drained boundaries",
        "cadence": "seconds; rank-local host intervals between Profiler.step returns; no per-update drain; last includes end drain",
        "device_compute": "unavailable", "host_work_attribution": "unavailable",
        "host_sync_and_gpu_idle_waits": "unavailable as separate attribution",
        "copies_and_communication": "unavailable as separate attribution",
        "comparison": "single sequence descriptive observations only; no paired controls or general speedup/convergence claim",
        "memory": "allocator peaks sampled before native resets; inspect boundary semantics in preparation notes; bytes and GiB",
        "drift": "last20/first20 median minus one; reported only with at least40 samples; descriptive, not causal"
    }), allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
