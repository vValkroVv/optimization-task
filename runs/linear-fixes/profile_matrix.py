#!/usr/bin/env python3
"""Capture native update 11 on all eight ranks for the Linear-fixes A/B matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from acceptance import (
    BASE_RUNNER, FLASH_RUNNER, MEASURE, RAW, ROOT, WORLD, Campaign, digest,
    flags, load_module, native_config_and_overrides, read_json, require, resolve,
    set_ep, write_json,
)


SCRIPT = Path(__file__).resolve()
ORDER = ((8, "baseline"), (8, "combined"), (4, "baseline"), (4, "combined"))
RANKS = list(range(WORLD))
WARMUP, UPDATE = 10, 11
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


class ProfileCampaign(Campaign):
    def check_native_command(self, argv, ep, steps):
        start = argv.index("--") + 2
        _, profile_args = native_config_and_overrides(argv)
        approved = {f"--profiler.{key}": str(value).lower() for key, value in PROFILE.items()}
        require(
            {key: val for key, val in profile_args.items() if key.startswith("--profiler.")}
            == approved,
            "Native profiler CLI differs from the approved full-update settings",
        )
        _, reference = native_config_and_overrides(self.reference_metadata["argv"])
        non_profile = {
            k: v for k, v in zip(argv[start::2], argv[start + 1::2], strict=True)
            if not k.startswith("--profiler.")
        }
        non_profile.update({k: v for k, v in reference.items() if k.startswith("--profiler.")})
        normalized = argv[:start] + [item for pair in non_profile.items() for item in pair]
        super().check_native_command(normalized, ep, steps)

    def check_resolved(self, paths, ep, steps):
        base = self.base
        load_yaml = base._load_yaml
        reference_path = self.reference / "native/config/resolved_config.yaml"

        def load_with_approved_profile(path):
            config = load_yaml(path)
            if path == reference_path:
                config["profiler"].update(PROFILE)
            return config

        base._load_yaml = load_with_approved_profile
        try:
            super().check_resolved(paths, ep, steps)
        finally:
            base._load_yaml = load_yaml

    def receipt(self, ep, arm):
        return super().receipt(ep, arm) | {"runner_sha256": digest(SCRIPT), "profile": PROFILE}

    def forward(self, ep, arm):
        argv = [
            "--num-prefetch-batches", "0", "--epoch-size", "102400",
            "--run-id", f"{self.path.name}-ep{ep}-{arm}", "--output-root", str(RAW),
            "--steps", str(UPDATE), "--warmup", str(WARMUP), "--window", "0",
            "--profile-update", str(UPDATE), "--timeout-seconds", "7200",
        ]
        if self.cache:
            argv.extend(("--compiler-cache-root", str(self.cache)))
        return argv


class AllRankMeasureLoader(importlib.machinery.SourceFileLoader):
    """Extend only the task observer's five fixed rank predicates in memory."""

    def get_code(self, fullname):
        source = self.get_data(self.path)
        predicate = b"active.rank in (0, 1)"
        require(source.count(predicate) == 5, "Native measurement rank-predicate contract changed")
        return self.source_to_code(source.replace(predicate, b"active.rank in range(8)"), self.path)


def load_all_rank_measure(path):
    name = "_linear_all_rank_measure"
    loader = AllRankMeasureLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def measure(arm, argv):
    expected = flags(arm)
    require({name: os.environ.get(name) for name in expected} == expected,
            "Profiling worker switch environment differs from its launch arm")
    bridge = load_module(MEASURE, "_linear_profile_worker_bridge")
    bridge._load_base_measure = lambda: load_all_rank_measure(bridge.BASE_MEASURE_PATH)
    bridge.main(argv)


def command_hook(campaign, bridge, ep, arm):
    build_command = bridge._build_command_hook

    def build(base):
        command = build_command(base)

        def checked(args, paths):
            argv = set_ep(command(args, paths), ep)
            i = argv.index(str(MEASURE))
            argv[i:i + 1] = [str(SCRIPT), "measure", arm]
            campaign.check_native_command(argv, ep, UPDATE)
            return argv

        return checked

    return build


def dry_arm(campaign, ep, arm):
    base = campaign.base
    attempt = campaign.path / "profile-preflight" / f"ep{ep}-{arm}"
    attempt.mkdir(parents=True, exist_ok=False)
    paths = {key: attempt / key for key in ("config", "native", "logs", "measurement")}
    for path in paths.values():
        path.mkdir()
    base._write_experiment_config(paths["config"] / "experiment.yaml")
    bridge = load_module(FLASH_RUNNER, "_linear_profile_bridge")
    argv = command_hook(campaign, bridge, ep, arm)(base)(
        base._parse_args(campaign.forward(ep, arm)), paths
    )
    env = os.environ | flags(arm) | {
        "PYTHONPATH": os.pathsep.join((str(base.GIFT_ROOT), str(base.GIFT_ROOT / "contrib/flash-linear-attention"))),
        "WORLD_SIZE": str(WORLD), "LOCAL_WORLD_SIZE": str(WORLD),
        "RANK": "0", "LOCAL_RANK": "0", "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (attempt / "resolve.log").open("x") as log:
        result = subprocess.run(
            [str(base.PYTHON), str(SCRIPT), "resolve", *argv[argv.index("--") + 1:]],
            cwd=base.GIFT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    require(result.returncode == 0, f"Native config preflight failed; see {log.name}")
    campaign.check_resolved(paths, ep, UPDATE)
    write_json(attempt / "verified.json", campaign.receipt(ep, arm) | {
        "argv": argv,
        "resolved_config_sha256": digest(paths["native"] / "config/resolved_config.yaml"),
    })


def check_gpus(path):
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader",
    ], text=True)
    with path.open("x") as handle:
        handle.write(inventory)
    rows = list(csv.reader(inventory.splitlines(), skipinitialspace=True))
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
    ], text=True)
    require(len(rows) == WORLD and all(len(row) == 4 and row[2] == "0 MiB" for row in rows)
            and len({row[1] for row in rows}) == WORLD and not apps.strip(),
            f"GPU preflight requires eight free GPUs; see {path}")


def run_arm(campaign, ep, arm):
    base = campaign.base
    dry_path = campaign.path / "profile-preflight" / f"ep{ep}-{arm}"
    dry = read_json(dry_path / "verified.json")
    receipt = campaign.receipt(ep, arm)
    require(all(dry.get(key) == val for key, val in receipt.items()) and
            dry["resolved_config_sha256"] == digest(dry_path / "native/config/resolved_config.yaml"),
            "Profile preflight identity differs")
    campaign.verify_inputs()
    snapshot = base.source_snapshot
    base.source_snapshot = lambda root: campaign.verify_snapshot(snapshot, root)
    base.source_snapshot(ROOT)
    base.RAW_ROOT.mkdir(parents=True, exist_ok=True)
    original_expect = base._resolved_config_expectations
    base._resolved_config_expectations = lambda args: original_expect(args) | {
        "parallelism.expert_parallel_degree": ep,
    }
    original_config = base._verify_resolved_config

    def verify_config(paths, args):
        config_hash = original_config(paths, args)
        campaign.check_resolved(paths, ep, UPDATE)
        return config_hash

    base._verify_resolved_config = verify_config
    original_cache_runtime = base._assert_runtime_compatible
    base._assert_runtime_compatible = lambda producer, current, field="runtime": (
        campaign.verify_cache_runtime(original_cache_runtime, producer, current)
        if field == "runtime" else original_cache_runtime(producer, current, field)
    )

    def parser():
        module = load_module(base.HELPER_DIR / "parse_results.py", "_linear_profile_parser")
        module.PROFILE_RANKS = RANKS
        return module

    base._load_parser = parser
    bridge = load_module(FLASH_RUNNER, "_linear_profile_bridge")
    bridge._load_base_runner = lambda: base
    build_command = command_hook(campaign, bridge, ep, arm)
    original_environment = bridge._build_environment_hook

    def build_environment(base, **kwargs):
        environment = original_environment(base, **kwargs)
        return lambda paths: environment(paths) | flags(arm)

    def record_command(base):
        command = build_command(base)

        def record(args, paths):
            argv = command(args, paths)
            write_json(paths["attempt"] / "profile-launch.json", receipt | {"argv": argv, "arm": arm})
            return argv

        return record

    write_metadata = base._write_new_json

    def corrected_metadata(path, value):
        if path.name == "metadata.json":
            value["profiler"] = "enabled for one update on all eight global ranks"
        return write_metadata(path, value)

    base._write_new_json = corrected_metadata
    bridge._build_command_hook = record_command
    bridge._build_environment_hook = build_environment
    check_gpus(campaign.path / f"ep{ep}-{arm}.gpus.csv")
    return bridge.main([
        "--gdn-backend", "fla", "--routing-metadata", "original", "--cross-entropy", "original",
        *campaign.forward(ep, arm),
    ])


def matrix(path):
    progress_path = path / "profile-matrix.tsv"
    failed = False
    with progress_path.open("x", buffering=1) as progress:
        print("ep\tarm\tstatus\ttimestamp_utc", file=progress)
        for ep, arm in ORDER:
            print(f"{ep}\t{arm}\tSTART\t{datetime.now(timezone.utc).isoformat()}", file=progress)
            log_path = path / f"ep{ep}-{arm}.profile.log"
            with log_path.open("x") as log:
                result = subprocess.run(
                    [sys.executable, "-Bu", str(SCRIPT), "run", str(path), str(ep), arm],
                    stdout=log, stderr=subprocess.STDOUT,
                )
            status = "PASS" if result.returncode == 0 else "FAIL"
            print(f"{ep}\t{arm}\t{status}\t{datetime.now(timezone.utc).isoformat()}", file=progress)
            print(f"EP{ep} {arm}: {status}; {log_path}", flush=True)
            failed |= result.returncode != 0
    return int(failed)


def main(argv):
    if argv[:1] == ["measure"]:
        measure(argv[1], argv[2:])
        return 0
    if argv[:1] == ["resolve"]:
        resolve(argv[1:])
        return 0
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    for mode in ("preflight", "run", "matrix"):
        command = modes.add_parser(mode)
        command.add_argument("campaign", type=Path)
        if mode == "run":
            command.add_argument("ep", type=int, choices=(8, 4))
            command.add_argument("arm", choices=("baseline", "combined"))
    args = cli.parse_args(argv)
    path = args.campaign.resolve(strict=True)
    require(path.parent == RAW.resolve(), "Campaign must be a direct child of the linear-fixes raw directory")
    if args.mode == "matrix":
        return matrix(path)
    base = load_module(BASE_RUNNER, "_linear_profile_native_runner")
    base.PROFILE_RANKS = RANKS
    campaign = ProfileCampaign(path, base)
    if args.mode == "run":
        return run_arm(campaign, args.ep, args.arm)
    campaign.verify_inputs()
    campaign.verify_snapshot(base.source_snapshot, ROOT)
    load_all_rank_measure(MEASURE.parents[1] / "overnight-streaming-perf/measure.py")
    for ep, arm in ORDER:
        dry_arm(campaign, ep, arm)
        print(f"Profile preflight verified EP{ep} {arm}", flush=True)
    campaign.verify_snapshot(base.source_snapshot, ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
