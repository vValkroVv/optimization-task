#!/usr/bin/env python3
"""Collect and verify the four native Original A prefetch runs."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import tempfile

import yaml


REVISION = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
CASES = ((8, 0, "e8-off"), (8, 32, "e8-on"), (4, 0, "e4-off"), (4, 32, "e4-on"))
WORLD = 8
WARMUP = 10
STEPS = 20
EPOCH_SIZE = 102_400
PHYSICAL_TOKENS = (STEPS - WARMUP) * 512 * 4096
ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRIC = re.compile(
    r"\bstep:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)\s+"
    r"memory:\s*([^\s]+?)\s*GiB.*?tps:\s*([\d,]+)"
)
MESH = re.compile(r"Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=(\d+)\b")
ACTIVE_AXES = re.compile(r"Successfully created meshes with active dimensions: (\[[^\n]*\])")
CONSOLE_FIELDS = ["update", "loss", "grad_norm", "total_gpu_memory_gib", "local_tps"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    document = json.loads(path.read_text())
    require(isinstance(document, dict), f"Expected JSON object at {path}")
    return document


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_picks(observed, expected, label):
    require(isinstance(observed, dict), f"{label} is missing or malformed")
    differences = [key for key, value in expected.items() if observed.get(key) != value]
    require(not differences, f"{label} differs at {', '.join(differences)}")


def rank_console(attempt, rank, ep):
    logs = list((attempt / "logs/torchrun").glob(f"*/attempt_*/{rank}/stdout.log"))
    require(len(logs) == 1 and logs[0].parents[1].name == "attempt_0",
            f"Rank {rank} native stdout missing or ambiguous")
    log = ANSI.sub("", logs[0].read_text())
    require(MESH.findall(log) == [str(ep)], f"Rank {rank} dense/expert mesh mismatch")
    axes = ACTIVE_AXES.findall(log)
    require(len(axes) == 1, f"Rank {rank} active mesh axes missing or ambiguous")
    active = ast.literal_eval(axes[0])
    require({"fsdp", "ep"} <= set(active) and ("efsdp" in active) == (ep == 4),
            f"Rank {rank} logged unexpected active mesh axes {active}")
    rows = [[int(step), float(loss), float(norm), float(memory), int(tps.replace(",", ""))]
            for step, loss, norm, memory, tps in METRIC.findall(log)]
    require([row[0] for row in rows] == list(range(1, STEPS + 1)),
            f"Rank {rank} native stdout must contain each update 1–{STEPS} once")
    require(all(math.isfinite(value) for row in rows for value in row[1:])
            and all(row[3] > 0 and row[4] > 0 for row in rows[WARMUP:]),
            f"Rank {rank} logged invalid metrics")
    return {"stdout": str(logs[0].relative_to(attempt)), "mesh_axes": active, "updates": rows}


def rank_bootstrap(attempt, rank, bridge, gift):
    boot = read_json(attempt / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json")
    verify_picks(boot, {
        "rank": rank, "local_rank": rank, "cuda_device": rank,
        "backend_variant": "fla", "routing_metadata": "original", "cross_entropy": "original",
        "capture_mode": "disabled", "native_main_completed": True,
        "source_identity": bridge["source_identity"],
    }, f"Rank {rank} Original A bootstrap")
    require(boot["installer_identity"]["variant"] == "fla" and
            "flash_qla" not in boot["installer_identity"], f"Rank {rank} installed another GDN backend")
    for field, relative in (("routing_metadata_callable", "distributed/deepep/routing.py"),
                            ("cross_entropy_callable", "components/loss.py")):
        expected = gift / "torchtitan" / relative
        require(Path(boot[field]["file"]) == expected and boot[field]["sha256"] == sha256(expected),
                f"Rank {rank} resolved unexpected {field}")
    losses = boot["constructed_loss_callables"]
    require(losses and all(loss["callable"] == boot["cross_entropy_callable"] for loss in losses),
            f"Rank {rank} did not construct the original loss")


def rank_measurement(attempt, rank, depth, host):
    record = read_json(attempt / f"measurement/rank-{rank:02d}.json")
    verify_picks(record, {
        "rank": rank, "world_size": WORLD, "run_id": attempt.name, "host_name": host,
        "expected_steps": STEPS, "completed_steps": STEPS,
        "training_status": "complete", "measurement_status": "complete",
    }, f"Rank {rank} training receipt")
    boundaries = record["training_boundary_receipts"]
    before = boundaries["before_window"]
    for phase, epoch, next_epoch in (("setup", boundaries["before_train"], 0),
                                      ("warmup", before["epoch"], 1),
                                      ("finished", boundaries["after_train"], 1)):
        verify_picks(epoch, {"dataset_available": True, "epoch_size": EPOCH_SIZE,
                            "length_samples_per_rank": EPOCH_SIZE // WORLD, "next_epoch": next_epoch},
                     f"Rank {rank} {phase} epoch")
    queue = before["prefetch_queue"]
    if depth:
        verify_picks(queue, {
            "iterator_present": True, "snapshot_available": True, "capacity": depth,
            "producer_thread_alive": True, "producer_thread_name": "streaming-batch-prefetch",
        }, f"Rank {rank} active prefetch queue")
        require(type(queue["ready_count"]) is int and 0 <= queue["ready_count"] <= depth,
                f"Rank {rank} reported invalid prefetch occupancy")
    else:
        verify_picks(queue, {"iterator_present": False, "snapshot_available": False},
                     f"Rank {rank} disabled prefetch queue")

    window = record["measurement"]
    verify_picks(window, {"warmup_updates": WARMUP, "window_updates": STEPS - WARMUP,
                          "start_after_update": WARMUP, "end_after_update": STEPS,
                          "physical_input_tokens_local": PHYSICAL_TOKENS // WORLD,
                          "monitoring_updates": [], "cuda_synchronize_calls": 2},
                 f"Rank {rank} measurement window")
    start, end = window["start_monotonic_ns"], window["end_monotonic_ns"]
    require(type(start) is int and type(end) is int and end > start,
            f"Rank {rank} measurement timestamps invalid")
    return window, boundaries


def collect_arm(attempt, ep, depth):
    launch = read_json(attempt / "prefetch-launch.json")
    verify_picks(launch, {"source_revision": REVISION, "ep": ep, "depth": depth}, "Launch identity")
    for name in ("source_sha256", "resolved_reference_sha256", "adapter_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", launch[name]), f"Missing verified launch {name}")
    completion = read_json(attempt / "completion.json")
    verify_picks(completion, {"status": "PASS", "source_unchanged": True, "config_unchanged": True},
                 "Native completion")
    resolved_path = attempt / "native/config/resolved_config.yaml"
    require(sha256(resolved_path) == completion["resolved_config_sha256"], "Resolved config changed after launch")
    config = yaml.safe_load(resolved_path.read_text())
    require(config["dataloader"]["num_prefetch_batches"] == depth
            and config["dataloader"]["dataset"]["epoch_size"] == EPOCH_SIZE
            and config["parallelism"]["expert_parallel_degree"] == ep
            and config["training"]["steps"] == STEPS
            and Path(config["dump_folder"]).resolve() == attempt / "native", "Resolved config differs from the arm")
    # These are the only intended differences between resolved configs in the matrix.
    config.pop("dump_folder")
    config["dataloader"].pop("num_prefetch_batches")
    config["parallelism"].pop("expert_parallel_degree")

    metadata = read_json(attempt / "metadata.json")
    verify_picks(metadata, {"run_id": attempt.name, "source_head_before": REVISION,
                            "steps": STEPS, "warmup_updates": WARMUP, "window_updates": STEPS - WARMUP,
                            "num_prefetch_batches": depth, "requested_epoch_size": EPOCH_SIZE,
                            "profiler": "disabled", "checkpointing": "disabled", "validation": "disabled",
                            "memory_snapshots": "disabled", "observe_compiler_cache": True}, "Native metadata")
    verify_picks(metadata["source_snapshot_before"], {
        "head": REVISION, "tracked_and_untracked_sha256": launch["source_sha256"],
        "untracked_file_count": 0, "git_status_sha256": hashlib.sha256(b"").hexdigest(),
    }, "Clean GIFT source")
    gift = Path(metadata["task_root"]) / "gift"
    require(Path(metadata["runtime"]["packages"]["torchtitan"]["path"]) == gift / "torchtitan/__init__.py",
            "Torchtitan did not import from the clean A checkout")

    bridge = read_json(attempt / "flash-qla-bridge.json")
    require(bridge["source_identity"]["task_git_head"] == REVISION, "Bridge source does not match Original A")
    summary = read_json(attempt / "summary.json")
    verify_picks(summary, {"status": "complete", "mode": "complete_window", "world_size": WORLD,
                           "training_steps": STEPS, "run_id": attempt.name}, "Native summary")
    verify_picks(summary["window"], {"start_after_update": WARMUP, "end_after_update": STEPS,
                                     "updates": STEPS - WARMUP}, "Native timing window")
    require(bool(summary["host"]), "Native summary lacks a common rank host")

    ranks = {}
    windows = []
    for rank in range(WORLD):
        rank_bootstrap(attempt, rank, bridge, gift)
        window, boundaries = rank_measurement(attempt, rank, depth, summary["host"])
        windows.append(window)
        ranks[str(rank)] = rank_console(attempt, rank, ep) | {"training_boundaries": boundaries}

    start = min(window["start_monotonic_ns"] for window in windows)
    end = max(window["end_monotonic_ns"] for window in windows)
    elapsed = (end - start) / 1e9
    physical = sum(window["physical_input_tokens_local"] for window in windows)
    supervised = {window["supervised_tokens_global"] for window in windows}
    require(supervised == {summary["tokens"]["supervised_tokens_global"]} and next(iter(supervised)) > 0,
            "Ranks and summary disagree on the valid-target token count")
    valid = next(iter(supervised))
    require(physical == summary["tokens"]["physical_input_tokens_global"] == PHYSICAL_TOKENS
            and math.isclose(elapsed, summary["window"]["primary_elapsed_s"])
            and math.isclose(physical / elapsed, summary["tokens"]["physical_input_tokens_per_second_global"])
            and math.isclose(valid / elapsed, summary["tokens"]["supervised_tokens_per_second_global"]),
            "Native token, elapsed or TPS arithmetic mismatch")
    retries = {str(rank): window["memory"]["allocator_counter_deltas"] for rank, window in enumerate(windows)}
    require(retries == summary["memory"]["allocator_counter_deltas_by_rank"], "Allocator counters differ from rank receipts")

    memory, update, rank = max((row[3], row[0], int(rank)) for rank, receipt in ranks.items()
                               for row in receipt["updates"][WARMUP:])
    step_tps = [sum(receipt["updates"][index][4] for receipt in ranks.values()) for index in range(WARMUP, STEPS)]
    return launch | {
        "status": "complete", "attempt": str(attempt), "ep": ep, "efsdp": WORLD // ep,
        "depth": depth, "host": summary["host"], "data_path": metadata["mds_data_root"],
        "compiler_cache": {"policy": metadata["compiler_cache_policy"], "root": metadata["compiler_cache_root"],
                           "producer": metadata["compiler_cache_producer_run_id"]},
        "compiler_observer": summary["compilation_observer"],
        "allocator_counter_deltas_by_rank": retries,
        "resolved_config_sha256": completion["resolved_config_sha256"],
        "window_elapsed_s": elapsed, "physical_tokens_global": physical, "valid_target_tokens_global": valid,
        "window_global_tps": physical / elapsed, "window_global_valid_target_tps": valid / elapsed,
        "step_global_tps": step_tps, "median_step_global_tps": median(step_tps),
        "last_vs_first_five_step_tps_percent": 100 * (median(step_tps[5:]) / median(step_tps[:5]) - 1),
        "total_gpu_memory_gib": memory, "memory_max_rank": rank, "memory_max_update": update,
        "console_fields": CONSOLE_FIELDS, "per_rank": ranks,
    }, config


def write_atomic(path, result):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("tag")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        parser.error("tag must be a native run-id prefix")
    raw = args.raw_root.resolve(strict=True)
    arms, configs = {}, {}
    for ep, depth, arm in CASES:
        attempt = raw / f"{args.tag}-{arm}"
        failure = attempt / "failure.json"
        try:
            if failure.is_file():
                arms[arm] = {"status": "failed", "attempt": str(attempt), "failure": read_json(failure)}
            elif not attempt.is_dir():
                arms[arm] = {"status": "missing", "attempt": str(attempt)}
            else:
                arms[arm], configs[arm] = collect_arm(attempt, ep, depth)
        except (OSError, TypeError, ValueError, KeyError, SyntaxError, yaml.YAMLError) as error:
            arms[arm] = {"status": "invalid", "attempt": str(attempt), "error": str(error)}

    issues = []
    if configs and any(config != next(iter(configs.values())) for config in configs.values()):
        issues.append("Resolved configs differ beyond topology, prefetch depth and dump folder")
    completed = [row for row in arms.values() if row["status"] == "complete"]
    for key in ("source_sha256", "resolved_reference_sha256", "adapter_sha256", "host", "data_path"):
        if len({row[key] for row in completed}) > 1:
            issues.append(f"Arms disagree on {key}")

    comparisons = {}
    if not issues:
        for ep in (8, 4):
            off, on = arms[f"e{ep}-off"], arms[f"e{ep}-on"]
            if off["status"] == on["status"] == "complete":
                comparisons[f"ep{ep}/efsdp{WORLD // ep}"] = {
                    "window_global_tps_delta_percent": 100 * (on["window_global_tps"] / off["window_global_tps"] - 1),
                    "median_step_global_tps_delta_percent": 100 * (on["median_step_global_tps"] / off["median_step_global_tps"] - 1),
                    "total_gpu_memory_delta_gib": round(on["total_gpu_memory_gib"] - off["total_gpu_memory_gib"], 2),
                }
    complete = len(completed) == len(CASES) and not issues
    result = {
        "status": "complete" if complete else "incomplete", "tag": args.tag,
        "warmup_updates": WARMUP, "measured_updates": [WARMUP + 1, STEPS],
        "arms": arms, "comparisons": comparisons, "issues": issues,
    }
    destination = raw / f"{args.tag}-result.json"
    write_atomic(destination, result)
    print(f"{result['status']}: {destination}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
