#!/usr/bin/env python3
"""Capture three native microbatches for dense GFSDP mechanism analysis."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

import campaign


SCRIPT = Path(__file__).resolve()
UPDATE = campaign.WARMUP + 1
MICROBATCHES = 3
RANKS = [0, 1]
ARMS = ("mtp", "mtp-transition", "all")
PROFILE = {
    "enable_profiling": True,
    "enable_memory_snapshot": False,
    "profile_ranks": RANKS,
    "save_traces_folder": "profiling/traces",
    "record_shapes": False,
    "profile_freq": 2,
    "profiler_warmup": 1,
    "profiler_active": 1,
    "profiler_repeat": 1,
    "profiler_skip_first": UPDATE - 2,
    "profiler_skip_first_wait": 0,
}


class DiagnosticCase(campaign.NativeCase):
    def base_args(self, run_id, timeout):
        argv = super().base_args(run_id, timeout)
        argv[argv.index("--window") + 1] = "0"
        argv.remove("--observe-compiler-cache")
        return [*argv, "--profile-update", str(UPDATE), "--profile-microbatches", str(MICROBATCHES)]

    def check_native_command(self, argv):
        _, actual = campaign.native_config_and_overrides(argv)
        profiler_flags = {key: value for key, value in actual.items() if key.startswith("--profiler.")}
        campaign.check_equal(
            {f"--profiler.{key}": str(value).lower() for key, value in PROFILE.items()},
            profiler_flags,
            "Native profiler CLI",
        )
        _, reference = campaign.native_config_and_overrides(self.reference_meta["argv"])
        begin = argv.index("--") + 2
        options = dict(zip(argv[begin::2], argv[begin + 1::2], strict=True))
        original_a = {key: value for key, value in options.items() if not key.startswith("--profiler.")}
        original_a.update({key: value for key, value in reference.items() if key.startswith("--profiler.")})
        return super().check_native_command(argv[:begin] + [
            item for pair in original_a.items() for item in pair
        ])

    def check_resolved(self, paths):
        load_yaml = self.base._load_yaml

        def reference_with_profiler(path):
            config = load_yaml(path)
            if path == self.reference_cfg:
                config["profiler"].update(PROFILE)
            return config

        self.base._load_yaml = reference_with_profiler
        try:
            return super().check_resolved(paths)
        finally:
            self.base._load_yaml = load_yaml

    def summarize(self, attempt):
        completion = campaign.read_json(attempt / "completion.json")
        metadata = campaign.read_json(attempt / "metadata.json")
        summary = campaign.read_json(attempt / "summary.json")
        config = attempt / "native/config/resolved_config.yaml"
        campaign.require(
            completion["status"] == "PASS" and completion["source_unchanged"]
            and completion["config_unchanged"] and completion["resolved_config_sha256"] == campaign.digest(config),
            f"Native diagnostic completion check failed in {attempt}",
        )
        self.check_native_command(metadata["argv"])
        self.check_resolved({"native": attempt / "native"})
        self.check_input_hashes(metadata["config_hashes_before"])
        campaign.require(metadata["source_head_before"] == self.expected_head
                         and metadata["source_snapshot_before"]["tracked_and_untracked_sha256"] == self.expected_hash,
                         "Diagnostic launch source identity differs")

        profile = summary["profile_capture"]
        campaign.require(
            summary["status"] == "complete" and summary["mode"] == "profile_update"
            and summary["measurement_status"] == "disabled" and summary["training_steps"] == UPDATE
            and metadata["profile_capture"]["profile_microbatches"] == MICROBATCHES
            and profile["target_update"] == UPDATE and profile["warmup_updates"] == campaign.WARMUP
            and profile["profile_ranks"] == RANKS and set(profile["traces_by_rank"]) == {str(i) for i in RANKS},
            "Native diagnostic scope differs",
        )
        bridge = campaign.read_json(attempt / "flash-qla-bridge.json")
        route_hash = campaign.digest(self.root / "gift/torchtitan/distributed/deepep/routing.py")
        activations = {}
        for rank in range(campaign.WORLD):
            receipt = campaign.read_json(attempt / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json")
            campaign.verify_bridge(receipt, bridge, rank, route_hash)
            _, activations[rank] = campaign.read_rank(attempt, rank, self.ep, self.features)

        return {
            "purpose": "mechanism analysis",
            "attempt": str(attempt), "ep": self.ep, "efsdp": campaign.WORLD // self.ep,
            "arm": self.arm, "source_head": self.expected_head, "source_sha256": self.expected_hash,
            "reference": str(self.reference), "input_hashes": self.input_hashes,
            "resolved_config_sha256": campaign.digest(config),
            "diagnostic_driver_sha256": campaign.digest(SCRIPT),
            "native_driver_sha256": campaign.digest(campaign.SCRIPT),
            "update": UPDATE, "warmup_updates": campaign.WARMUP,
            "microbatches_per_rank": MICROBATCHES, "features": self.features,
            "effective_paths_by_rank": activations, "traces_by_rank": profile["traces_by_rank"],
        }


def run(args):
    # This process has exactly one capture. The same native checks now expect
    # its shorter update count while the timing entry retains its fixed window.
    campaign.STEPS = UPDATE
    case = DiagnosticCase(args, args.ep, args.arm)
    case.preflight(args.timeout_seconds)
    print(f"PASS source, real inputs and profiler CLI for EP{args.ep} {args.arm}", flush=True)
    if args.preflight_only:
        return 0

    root = args.campaign.resolve()
    campaign.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", root.name), "Invalid campaign name")
    output = root / "diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    key = f"ep{args.ep}-{args.arm}"
    run_id = f"{root.name}-diagnostic-{key}-{uuid4().hex[:8]}"
    verified = output / f"{key}.json"
    with (root / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.require(not verified.exists(), f"A verified capture already exists at {verified}")
        campaign.check_gpus(output / f"{run_id}.gpus.csv")
        print(f"START {run_id} ({case.raw / run_id})", flush=True)
        status = case.launch(run_id, args.timeout_seconds)
        if status:
            return status
        evidence = case.summarize(case.raw / run_id)
        campaign.write_json(verified, evidence)
        print(f"PASS {verified}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="same output directory as the native timing matrix")
    parser.add_argument("ep", type=int, choices=(8, 4))
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--preflight-only", action="store_true")
    campaign.add_native_args(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
