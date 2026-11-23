#!/usr/bin/env python3
"""Verify and compare the four unprofiled original A and expert GFSDP runs."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

import yaml


RANKS = set(range(8))
UPDATES = tuple(range(1, 21))
MEASURED = tuple(range(11, 21))
RUN_PREFIX = "efg-n22-07-final"
EP8_CANDIDATE = f"{RUN_PREFIX}-ep8-b"
EP8_CANDIDATE_RETRY = f"{EP8_CANDIDATE}-retry"
FLAGS = {
    "bf16-cache": "gfsdp_reuse_bf16_expert_weights",
    "direct-dw": "gfsdp_direct_expert_dw",
    "fp8-cache": "gfsdp_fp8_expert_weights",
    "tail-retention": "gfsdp_retain_tail_weights",
}
GROUPS = {f"layers.{layer}[efsdp]" for layer in range(1, 8)}
CONSOLE_FIELDS = ["update", "loss", "grad_norm", "total_gpu_memory_gib", "rank_tps"]
COMPACT_MARKER = (
    "GFSDP compact BF16 owner transport active: "
    "slot_bytes=1207959552 receive_bytes=301989888"
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text())


def verify_devices(path):
    devices = {}
    with path.open(newline="") as source:
        for row in csv.reader(source, skipinitialspace=True):
            if not row or row[0].strip() == "index":
                continue
            index = int(row[0])
            uuids = [value.strip() for value in row if value.strip().startswith("GPU-")]
            require(index not in devices and len(uuids) == 1, f"Invalid GPU row {row}")
            devices[index] = uuids[0]
    require(
        devices.keys() == RANKS and len(set(devices.values())) == 8,
        "Device receipt must identify GPU indices 0–7 and eight distinct UUIDs",
    )
    return devices


def verify_config(path, ep, treatment, expected_sha):
    content = path.read_bytes()
    actual_sha = hashlib.sha256(content).hexdigest()
    require(actual_sha == expected_sha, "Resolved configuration SHA-256 mismatch")
    config = yaml.safe_load(content)
    require(
        config["compile"]["enable"] is True
        and {"model", "loss"} <= set(config["compile"]["components"])
        and config["training"]["steps"] == 20,
        "Expected compiled model and loss with a twenty-update run",
    )
    require(
        all(config[name]["enable"] is False for name in ("checkpoint", "validator"))
        and config["profiler"]["enable_profiling"] is False
        and config["profiler"]["enable_memory_snapshot"] is False,
        "Expected unprofiled timing with validation and checkpoint saving off",
    )
    del config["dump_folder"]
    parallelism = config["parallelism"]
    require(
        parallelism.pop("expert_parallel_degree") == ep,
        "Unexpected resolved EP degree",
    )
    for field in FLAGS.values():
        require(
            parallelism.pop(field, False) is treatment,
            f"Unexpected resolved value for {field}",
        )
    return config


def verify_activation(summary, ep, treatment, marker_path):
    expected_requested = sorted(FLAGS) if treatment else []
    require(
        summary["ep"] == ep and summary["requested"] == expected_requested,
        "Unexpected Expert GFSDP request in summary",
    )
    receipts = summary["per_rank_activation"]
    require(len(receipts) == 8, "Expected activation receipts from eight ranks")
    for rank, receipt in enumerate(receipts):
        require(
            receipt["ep"] == ep and receipt["requested"] == expected_requested,
            f"Unexpected Expert GFSDP request on rank {rank}",
        )
        groups = receipt["groups"]
        require(
            len(groups) == 7 and {group["fqn"] for group in groups} == GROUPS,
            f"Invalid expert group list on rank {rank}",
        )
        for group in groups:
            require(
                group["efsdp"] == 8 // ep
                and all(
                    group[flag] is (treatment and flag != "bf16-cache")
                    for flag in FLAGS
                ),
                f"Unexpected Expert GFSDP activation on rank {rank}: {group}",
            )

    marker_ranks = []
    for line in marker_path.read_text().splitlines():
        rank, marker = ANSI_ESCAPE.sub("", line).split("\t", maxsplit=1)
        require(
            marker.endswith(COMPACT_MARKER), f"Invalid compact marker on rank {rank}"
        )
        marker_ranks.append(int(rank))
    expected_ranks = sorted(RANKS) if ep == 4 and treatment else []
    require(
        sorted(marker_ranks) == expected_ranks, "Unexpected compact marker coverage"
    )
    return expected_ranks


def derive_console_metrics(summary):
    require(summary["console_fields"] == CONSOLE_FIELDS, "Unknown console row layout")
    console = summary["per_rank_console"]
    require(set(console) == {str(rank) for rank in RANKS}, "Missing console ranks")
    for rank, rows in console.items():
        require(
            tuple(row[0] for row in rows) == UPDATES,
            f"Missing or out-of-order updates on rank {rank}",
        )
        for update, loss, grad_norm, memory, tps in rows:
            require(
                all(math.isfinite(value) for value in (loss, grad_norm, memory, tps))
                and memory > 0
                and tps > 0,
                f"Invalid console metrics on rank {rank}, update {update}",
            )
    peak_memory, peak_update, peak_rank = max(
        (console[str(rank)][step - 1][3], step, rank)
        for rank in RANKS
        for step in MEASURED
    )
    require(
        (peak_memory, peak_update, peak_rank)
        == (
            summary["total_gpu_memory_gib"],
            summary["memory_max_update"],
            summary["memory_max_rank"],
        ),
        "Native total GPU memory peak differs from rank and update records",
    )
    steps = [
        {
            "update": step,
            "global_tps": sum(console[str(rank)][step - 1][4] for rank in RANKS),
            "peak_total_gpu_memory_gib": max(
                console[str(rank)][step - 1][3] for rank in RANKS
            ),
        }
        for step in MEASURED
    ]
    median_tps = median(step["global_tps"] for step in steps)
    require(median_tps == summary["median_step_global_tps"], "Step TPS median mismatch")
    window_tps = summary["window_global_tps"]
    require(math.isfinite(window_tps) and window_tps > 0, "Invalid window TPS")
    return {
        "window_global_tps": window_tps,
        "median_step_global_tps": median_tps,
        "peak_total_gpu_memory_gib": peak_memory,
        "peak_update": peak_update,
        "peak_rank": peak_rank,
        "measured_steps": steps,
    }


def verify_run(path, ep, treatment, expected_source):
    metadata = load_json(path / "metadata.json")
    completion = load_json(path / "completion.json")
    source_after = load_json(path / "source-final.json")
    summary = load_json(path / "expert-gfsdp-summary.json")
    require(
        (path / "status.txt").read_text().strip() == completion["status"] == "PASS"
        and metadata["run_id"] == summary["run_id"] == path.name,
        "Missing successful completion for this run",
    )
    require(
        (metadata["steps"], metadata["warmup_updates"], metadata["window_updates"])
        == (20, 10, 10),
        "Unexpected warmup or measurement window",
    )
    require(
        metadata["compiler_cache_policy"] == summary["compiler_cache_policy"] == "fresh"
        and metadata["compiler_cache_producer_run_id"] is None
        and metadata["compiler_cache_root"] == summary["compiler_cache_root"],
        "Expected a fresh compiler cache for this arm",
    )
    digest_key = "tracked_and_untracked_sha256"
    require(
        metadata["source_snapshot_before"][digest_key]
        == source_after[digest_key]
        == summary["gift_source_sha256"]
        == expected_source
        and completion["source_unchanged"] is True,
        "Unexpected GIFT source snapshot",
    )
    config_sha = completion["resolved_config_sha256"]
    require(
        config_sha == summary["resolved_config_sha256"]
        and completion["config_unchanged"] is True,
        "Configuration digest differs between receipts",
    )
    config = verify_config(
        path / "native/config/resolved_config.yaml", ep, treatment, config_sha
    )
    compact_ranks = verify_activation(
        summary, ep, treatment, path / "compact-markers.tsv"
    )
    compiler = summary["compiler_observer"]
    return (
        config,
        metadata["input_data"],
        {
            **derive_console_metrics(summary),
            "ep": ep,
            "efsdp": 8 // ep,
            "requested_expert_features": summary["requested"],
            "compact_bf16_owner_ranks": compact_ranks,
            "gift_source_sha256": expected_source,
            "resolved_config_sha256": config_sha,
            "compiler_cache_root": summary["compiler_cache_root"],
            "compiler_observer": {
                key: compiler.get(key)
                for key in ("status", "coverage_status", "observed_stable")
            },
        },
    )


def compare(root, ep8_candidate_run=EP8_CANDIDATE):
    pairs = {
        8: (f"{RUN_PREFIX}-ep8-a", ep8_candidate_run),
        4: (f"{RUN_PREFIX}-ep4-a", f"{RUN_PREFIX}-ep4-b"),
    }
    expected_runs = {}
    for ep, (baseline_id, candidate_id) in pairs.items():
        expected_runs[baseline_id] = (ep, False)
        expected_runs[candidate_id] = (ep, True)
    require(
        (root / "matrix.exit").read_text().strip() == "0", "Matrix did not exit zero"
    )
    devices = verify_devices(root / "devices.csv")
    expected_sources = load_json(root / "source-digests.json")
    require(
        expected_sources.keys() == expected_runs.keys()
        and all(
            re.fullmatch("[0-9a-f]{64}", digest) for digest in expected_sources.values()
        ),
        "Expected GIFT SHA-256 digests keyed by the four exact run IDs",
    )
    configs, inputs, runs = {}, {}, {}
    for run_id, (ep, treatment) in expected_runs.items():
        try:
            configs[run_id], inputs[run_id], runs[run_id] = verify_run(
                root / run_id, ep, treatment, expected_sources[run_id]
            )
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            raise ValueError(f"{run_id}: {error}") from error
    reference = pairs[8][0]
    for run_id in expected_runs:
        require(
            configs[run_id] == configs[reference], f"{run_id}: resolved config differs"
        )
        require(inputs[run_id] == inputs[reference], f"{run_id}: input receipt differs")
    require(
        len({row["compiler_cache_root"] for row in runs.values()})
        == len(expected_runs),
        "The arms must have distinct compiler cache roots",
    )
    comparisons = {}
    for ep, (baseline_id, candidate_id) in pairs.items():
        baseline = runs[baseline_id]
        candidate = runs[candidate_id]
        comparisons[f"ep{ep}"] = {
            "window_tps_change_pct": 100
            * (candidate["window_global_tps"] / baseline["window_global_tps"] - 1),
            "median_step_tps_change_pct": 100
            * (
                candidate["median_step_global_tps"] / baseline["median_step_global_tps"]
                - 1
            ),
            "peak_total_gpu_memory_change_gib": round(
                candidate["peak_total_gpu_memory_gib"]
                - baseline["peak_total_gpu_memory_gib"],
                2,
            ),
        }
    return {
        "status": "PASS",
        "gpu_index_to_uuid": devices,
        "measured_updates": MEASURED,
        "input_data": inputs[reference],
        "runs": runs,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument(
        "--ep8-candidate-run",
        choices=(EP8_CANDIDATE, EP8_CANDIDATE_RETRY),
        default=EP8_CANDIDATE,
        help="Select the successful EP8 candidate attempt",
    )
    parser.add_argument(
        "--output", type=Path, help="Write JSON to this path instead of stdout"
    )
    args = parser.parse_args()
    try:
        result = json.dumps(
            compare(args.receipts, args.ep8_candidate_run), indent=2, allow_nan=False
        )
        if args.output is None:
            print(result)
        else:
            args.output.write_text(result + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
