#!/usr/bin/env python3
"""Run four native, unprofiled Linear-fixes arms with 10 warmup and 10 measured updates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import subprocess
import sys

import acceptance
from acceptance import RAW, WORLD, case_path, digest, read_json, require, write_json
from profile_matrix import check_gpus


SCRIPT = Path(__file__).resolve()
ORDER = ((8, "baseline"), (8, "combined"), (4, "combined"), (4, "baseline"))
STEPS = 20
WARMUP = 10
TOKENS = (STEPS - WARMUP) * 512 * 4096


def configure_acceptance():
    acceptance.ORDER = ORDER
    acceptance.STEPS = STEPS
    acceptance.WARMUP = WARMUP
    acceptance.TOKENS = TOKENS


def campaign_path(path):
    path = path.resolve(strict=True)
    require(path.parent == RAW.resolve(), "Campaign must be a direct child of the Linear-fixes raw directory")
    return path


def activation_hashes(path):
    return {
        f"ep{ep}-{arm}": digest(path / "activation" / f"ep{ep}-{arm}.verified.json")
        for ep, arm in ORDER
    }


def dry_config_hashes(path):
    hashes = {}
    for ep, arm in ORDER:
        name = f"ep{ep}-{arm}"
        receipt = read_json(path / "dry" / f"{name}.verified.json")
        config = path / "dry" / name / "native/config/resolved_config.yaml"
        hashes[name] = digest(config)
        require(hashes[name] == receipt["resolved_config_sha256"], f"Dry config changed for {name}")
    return hashes


def plan(path):
    return {
        "order": [list(arm) for arm in ORDER],
        "training_updates": STEPS,
        "warmup_updates": WARMUP,
        "measured_physical_tokens_global": TOKENS,
        "inputs_sha256": digest(path / "inputs.json"),
        "activation_sha256": activation_hashes(path),
        "dry_configs_sha256": dry_config_hashes(path),
        "adapter_sha256": digest(SCRIPT),
        "acceptance_sha256": digest(Path(acceptance.__file__)),
    }


def preflight(path, prior):
    require(path != prior, "Use a fresh timing campaign directory")
    require(digest(path / "inputs.json") == digest(prior / "inputs.json"),
            "Timing inputs differ from the verified acceptance inputs")
    require(activation_hashes(path) == activation_hashes(prior),
            "Copied activation receipts differ from prior acceptance")
    require(acceptance.main(["preflight", str(path)]) == 0, "Timing preflight failed")
    require(acceptance.main(["verify-ready", str(path)]) == 0, "Timing preflight receipts failed verification")
    write_json(path / "timing-plan.json", plan(path) | {"prior_activation_campaign": str(prior)})


def verify_ready(path):
    expected = plan(path)
    observed = read_json(path / "timing-plan.json")
    require(all(observed.get(key) == value for key, value in expected.items()),
            "Timing plan, activation receipts, or resolved configs changed after preflight")
    require(acceptance.main(["verify-ready", str(path)]) == 0, "Timing preflight receipts failed verification")


def run_arm(path, ep, arm):
    verify_ready(path)
    check_gpus(path / f"ep{ep}-{arm}.gpus.csv")
    result = acceptance.main(["run", str(path), str(ep), arm])
    if result:
        return result
    return acceptance.main(["summarize", str(path), str(ep), arm])


def arm_result(path, ep, arm):
    attempt = case_path(path, ep, arm)
    summary = read_json(attempt / "acceptance-summary.json")
    fields = (
        "attempt", "window_elapsed_s", "physical_input_tokens", "window_global_tps",
        "median_step_global_tps", "compiler_observer_status", "total_gpu_memory_gib",
        "memory_max_rank", "memory_max_update", "allocator_counter_deltas_by_rank",
    )
    rows = summary["per_rank_console"]
    step_tps = [sum(rows[str(rank)][step - 1][4] for rank in range(WORLD))
                for step in range(WARMUP + 1, STEPS + 1)]
    midpoint = len(step_tps) // 2
    drift = 100 * (median(step_tps[midpoint:]) / median(step_tps[:midpoint]) - 1)
    return {key: summary[key] for key in fields} | {
        "ep": ep, "efsdp": WORLD // ep, "arm": arm, "step_global_tps": step_tps,
        "last_vs_first_five_step_tps_percent": drift,
    }


def summarize(path, successes):
    rows = {(ep, arm): arm_result(path, ep, arm) for ep, arm in successes}
    comparisons = []
    for ep in (8, 4):
        baseline = rows.get((ep, "baseline"))
        combined = rows.get((ep, "combined"))
        if baseline and combined:
            comparisons.append({
                "ep": ep,
                "combined_vs_baseline_tps_percent": acceptance.percentage(combined, baseline),
                "combined_minus_baseline_total_gpu_memory_gib": (
                    combined["total_gpu_memory_gib"] - baseline["total_gpu_memory_gib"]
                ),
            })
    unavailable = [list(case) for case in ORDER if case not in rows]
    unstable = [list(case) for case, row in rows.items()
                if row["compiler_observer_status"] != "observed_stable"]
    write_json(path / "timing-matrix.json", {
        "warmup_updates": WARMUP,
        "measured_updates": [WARMUP + 1, STEPS],
        "arms": list(rows.values()), "comparisons": comparisons,
        "unavailable": unavailable, "compiler_stability_unverified": unstable,
    })
    return int(bool(unavailable or unstable))


def matrix(path):
    verify_ready(path)
    successes = []
    with (path / "matrix.tsv").open("x", buffering=1) as progress:
        print("ep\tarm\tstatus\ttimestamp_utc", file=progress)
        for ep, arm in ORDER:
            print(f"{ep}\t{arm}\tSTART\t{datetime.now(timezone.utc).isoformat()}", file=progress)
            log_path = path / f"ep{ep}-{arm}.runner.log"
            with log_path.open("x") as log:
                result = subprocess.run(
                    [sys.executable, "-Bu", str(SCRIPT), "run", str(path), str(ep), arm],
                    stdout=log, stderr=subprocess.STDOUT,
                )
            if result.returncode == 0:
                successes.append((ep, arm))
            status = "PASS" if result.returncode == 0 else "FAIL"
            print(f"{ep}\t{arm}\t{status}\t{datetime.now(timezone.utc).isoformat()}", file=progress)
            print(f"EP{ep} {arm} {status} ({log_path})", flush=True)
    return summarize(path, successes)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("preflight", "run", "matrix"):
        command = modes.add_parser(mode)
        command.add_argument("campaign", type=Path)
        if mode == "preflight":
            command.add_argument("prior_acceptance", type=Path)
        if mode == "run":
            command.add_argument("ep", type=int, choices=(8, 4))
            command.add_argument("arm", choices=("baseline", "combined"))
    args = parser.parse_args(argv)
    path = campaign_path(args.campaign)
    configure_acceptance()
    if args.mode == "preflight":
        preflight(path, campaign_path(args.prior_acceptance))
        return 0
    if args.mode == "run":
        return run_arm(path, args.ep, args.arm)
    return matrix(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
