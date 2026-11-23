#!/usr/bin/env python3
"""Run matched native Original A and dense GFSDP comparisons on eight GPUs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import re
import runpy
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4


SCRIPT = Path(__file__).resolve()
ORIGINAL_A = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
BASE = Path("runs/overnight-streaming-perf/run.py")
BRIDGE = Path("runs/flash-qla/run.py")
DATA = Path("runs/production-data-abcd/artifacts/raw/data-002")
DATA_RECEIPT = Path("runs/production-data-abcd/data-002.json")
INPUT_HASHES = (
    "canonical_experiment_sha256", "mds_fragment_sha256", "tokenizer_json_sha256",
    "tokenizer_config_sha256", "special_tokens_map_sha256", "data_receipt_sha256",
    "mds_index_sha256", "row_manifest_sha256", "cpu_loader_receipt_sha256",
)

LAYER0 = "gfsdp_split_layer0_ffn"
MTP = "gfsdp_split_mtp_ffn"
ROW_OWNER = "gfsdp_row_owner_embeddings"
TRANSITION = "gfsdp_mtp_transition_prefetch"
FEATURES = (LAYER0, MTP, ROW_OWNER, TRANSITION)
ARMS = {
    "a": (), "layer0": (LAYER0,), "mtp": (MTP,), "row-owner": (ROW_OWNER,),
    "mtp-transition": (MTP, TRANSITION), "all": FEATURES,
}
FINAL = ((8, "a"), (8, "all"), (4, "all"), (4, "a"))
PER_FIX = (
    (8, "a"), (8, "layer0"), (8, "row-owner"), (8, "mtp"), (8, "mtp-transition"),
    (4, "mtp-transition"), (4, "mtp"), (4, "row-owner"), (4, "layer0"), (4, "a"),
)
WORLD, STEPS, WARMUP = 8, 20, 10
TOKENS = (STEPS - WARMUP) * 512 * 4096
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(
    r"\bstep:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)\s+"
    r"memory:\s*(\S+)\s*GiB.*?tps:\s*([\d,]+)"
)
MESH = re.compile(
    r"Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, "
    r"cp=1, tp=1, ep=(\d+)\b"
)
MESH_AXES = re.compile(r"Successfully created meshes with active dimensions: (\[[^\n]*\])")
GFSDP_LOG_TAG = "dense-gfsdp-activation "
MARKERS = {
    LAYER0: "GFSDP FFN overlap active: layers.0.feed_forward ",
    MTP: "GFSDP FFN overlap active: mtp.layers.0.block.feed_forward ",
    ROW_OWNER: "row_owner_embedding active fqn=tok_embeddings.weight ",
    TRANSITION: "GFSDP buffer region transition active: ",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def differences(expected, actual, prefix=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key in expected and key in actual:
                yield from differences(expected[key], actual[key], name)
            else:
                yield name
    elif expected != actual:
        yield prefix


def check_equal(expected, actual, label):
    differing = list(differences(expected, actual))
    require(not differing, f"{label} differs at {differing}")


def native_config_and_overrides(argv):
    config_path, *switches = argv[argv.index("--") + 1:]
    overrides = dict(zip(switches[::2], switches[1::2], strict=True))
    require(len(overrides) * 2 == len(switches), "Native CLI repeats an override")
    overrides.pop("--dump_folder")
    return Path(config_path), overrides


class NativeCase:
    def __init__(self, args, ep, arm):
        self.ep = ep
        self.arm = arm
        self.features = dict.fromkeys(FEATURES, False) | dict.fromkeys(ARMS[arm], True)
        self.uses_candidate = self.features[TRANSITION]
        self.root = (args.candidate_root if self.uses_candidate else args.original_root).resolve(strict=True)
        self.reference = args.reference.resolve(strict=True)
        self.reference_cfg = self.reference / "native/config/resolved_config.yaml"
        self.reference_meta = read_json(self.reference / "metadata.json")
        completion = read_json(self.reference / "completion.json")
        require(completion["status"] == "PASS" and completion["source_unchanged"]
                and completion["config_unchanged"]
                and self.reference_meta["source_head_before"] == ORIGINAL_A
                and completion["resolved_config_sha256"] == digest(self.reference_cfg),
                "Reference must be a completed, verified Original A run")
        self.expected_head = args.candidate_head if self.uses_candidate else ORIGINAL_A
        self.expected_hash = (
            args.candidate_source_sha256 if self.uses_candidate
            else self.reference_meta["source_snapshot_before"]["tracked_and_untracked_sha256"]
        )
        require(re.fullmatch(r"[0-9a-f]{40}", self.expected_head)
                and re.fullmatch(r"[0-9a-f]{64}", self.expected_hash),
                "Expected a pinned head and source fingerprint")
        self.base = load_module(self.root / BASE, "_dense_gfsdp_native")
        self.flash = load_module(self.root / BRIDGE, "_dense_gfsdp_bridge")
        self.base.verify_task_root(self.root)
        self.base.source_snapshot = self.checked_snapshot(self.base.source_snapshot)
        self.base.source_snapshot(self.root)
        self.reuse_original_inputs()
        self.input_hashes = self.check_input_hashes(self.base._config_hashes())
        self.config_hashes = self.base._config_hashes
        self.base._config_hashes = self.verified_hashes
        self.raw = self.base.RAW_ROOT

    def checked_snapshot(self, original):
        def snapshot(root):
            value = original(root)
            require(value["head"] == self.expected_head
                    and value["tracked_and_untracked_sha256"] == self.expected_hash,
                    f"Unexpected source fingerprint in {root}")
            if not self.uses_candidate:
                clean = self.base._git(root, "status", "--porcelain", "--untracked-files=all",
                                       "--", "gift/").stdout
                require(not clean, "Original A GIFT tree must be pristine")
            return value

        return snapshot

    def reuse_original_inputs(self):
        base = self.base
        input_root = Path(self.reference_meta["task_root"]).resolve(strict=True)
        base.DATA_FRAGMENT = input_root / BASE.parent / "mds-dataloader.yaml"
        base.DATASET_ROOT = input_root / DATA
        base.DATA_RECEIPT = input_root / DATA_RECEIPT
        base.PREPARATION_STATUS = base.DATA_RECEIPT.with_name("data-002-preparation-status.json")
        base.LOADER_SUMMARY = base.DATA_RECEIPT.with_name("data-002-loader-summary.json")
        base.ROW_MANIFEST = base.DATASET_ROOT / "data-002-rows.jsonl"
        configured = base._configured_path
        inside = base._inside

        def configured_path(value, *, base, label):
            path = configured(value, base=base, label=label)
            if label == "canonical dataset stream local" and path == (self.root / DATA).resolve():
                return self.base.DATASET_ROOT.resolve()
            return path

        def inside_path(root, path, label):
            if label == "MDS data path" and path.resolve() == base.DATASET_ROOT.resolve():
                return path.resolve(strict=True)
            return inside(root, path, label)

        base._configured_path = configured_path
        base._inside = inside_path

    def check_input_hashes(self, current):
        reference = self.reference_meta["config_hashes_before"]
        expected = {key: reference[key] for key in INPUT_HASHES}
        check_equal(expected, {key: current[key] for key in INPUT_HASHES}, "Original A inputs")
        return expected

    def verified_hashes(self):
        hashes = self.config_hashes()
        self.check_input_hashes(hashes)
        return hashes

    def check_native_command(self, argv):
        config_path, actual = native_config_and_overrides(argv)
        _, expected = native_config_and_overrides(self.reference_meta["argv"])
        require(expected.get("--dataloader.num_prefetch_batches") == "0"
                and expected.get("--dataloader.dataset.epoch_size") == "102400"
                and expected.get("--profiler.enable_profiling") == "false",
                "Reference does not carry the Original A native workload settings")
        expected["--training.steps"] = str(STEPS)
        if self.ep == 4:
            expected["--parallelism.expert_parallel_degree"] = "4"
        for field, enabled in self.features.items():
            if enabled:
                expected[f"--parallelism.{field}"] = "true"
        check_equal(expected, actual, "Native CLI")
        check_equal({
            "config_version": 1,
            "imports": [str(self.base.BASE_CONFIG.resolve()), str(self.base.DATA_FRAGMENT.resolve())],
            "callbacks": {"monitoring": None, "system_metrics": None, "timers": None},
            "loss": {"num_chunks_on_monitoring_steps": None},
        }, self.base._load_yaml(config_path), "Native config imports")

    def check_resolved(self, paths):
        base = self.base
        expected = base._load_yaml(self.reference_cfg)
        actual = base._load_yaml(paths["native"] / "config/resolved_config.yaml")
        expected["dump_folder"] = str(paths["native"].resolve())
        expected["training"]["steps"] = STEPS
        expected["parallelism"]["expert_parallel_degree"] = self.ep
        for field in FEATURES:
            require(expected["parallelism"].get(field, False) is False,
                    f"Reference enables {field}")
        fields = FEATURES if self.uses_candidate else FEATURES[:-1]
        expected["parallelism"].update({field: self.features[field] for field in fields})
        expected_stream = expected["dataloader"]["dataset"]["streams"][0]
        actual_stream = actual["dataloader"]["dataset"]["streams"][0]
        reference_stream = Path(expected_stream["local"])
        if not reference_stream.is_absolute():
            reference_stream = Path(self.reference_meta["task_root"]) / "gift" / reference_stream
        require(reference_stream.resolve() == base.DATASET_ROOT.resolve(),
                "Reference resolved different MDS data")
        require(base._configured_path(actual_stream["local"], base=base.GIFT_ROOT, label="active MDS stream")
                == base.DATASET_ROOT.resolve(), "Active run resolved different MDS data")
        expected_stream["local"] = actual_stream["local"]
        check_equal(expected, actual, "Resolved Original A configuration")

    def launch(self, run_id, timeout):
        base, flash = self.base, self.flash
        replace_worker = flash._build_command_hook
        verify_config = base._verify_resolved_config
        expectations = base._resolved_config_expectations

        def build_command(implementation):
            command = replace_worker(implementation)

            def build(args, paths):
                argv = command(args, paths)
                if self.ep == 4:
                    argv.extend(("--parallelism.expert_parallel_degree", "4"))
                if self.features[TRANSITION]:
                    argv.extend((f"--parallelism.{TRANSITION}", "true"))
                self.check_native_command(argv)
                worker = self.root / BRIDGE.with_name("measure.py")
                index = argv.index(str(worker))
                argv[index:index + 1] = [str(SCRIPT), "worker", str(worker)]
                return argv

            return build

        def verify(paths, args):
            result = verify_config(paths, args)
            self.check_resolved(paths)
            return result

        base._resolved_config_expectations = lambda args: expectations(args) | {
            "parallelism.expert_parallel_degree": self.ep,
        }
        base._verify_resolved_config = verify
        flash._load_base_runner = lambda: base
        flash._build_command_hook = build_command
        return flash.main([
            "--gdn-backend", "fla", "--routing-metadata", "original", "--cross-entropy", "original",
            *self.base_args(run_id, timeout),
        ])

    def base_args(self, run_id, timeout):
        return [
            "--run-id", run_id, "--output-root", str(self.raw),
            "--steps", str(STEPS), "--warmup", str(WARMUP), "--window", str(STEPS - WARMUP),
            "--epoch-size", "102400", "--num-prefetch-batches", "0", "--observe-compiler-cache",
            "--timeout-seconds", str(timeout),
            *(f"--{field.replace('_', '-')}" for field in FEATURES[:-1] if self.features[field]),
        ]

    def preflight(self, timeout):
        self.base._verify_ready_inputs(self.root)
        with TemporaryDirectory(prefix="dense-gfsdp-preflight-") as temporary:
            folder = Path(temporary)
            paths = {key: folder / key for key in ("config", "native", "measurement", "logs")}
            paths["config"].mkdir()
            self.base._write_experiment_config(paths["config"] / "experiment.yaml")
            args = self.base._parse_args(self.base_args("dense-gfsdp-preflight", timeout))
            argv = self.flash._build_command_hook(self.base)(args, paths)
            if self.ep == 4:
                argv.extend(("--parallelism.expert_parallel_degree", "4"))
            if self.features[TRANSITION]:
                argv.extend((f"--parallelism.{TRANSITION}", "true"))
            self.check_native_command(argv)

    def summarize(self, attempt):
        completion = read_json(attempt / "completion.json")
        summary = read_json(attempt / "summary.json")
        metadata = read_json(attempt / "metadata.json")
        require(completion["status"] == "PASS" and completion["source_unchanged"]
                and completion["config_unchanged"], f"Native completion check failed in {attempt}")
        config = attempt / "native/config/resolved_config.yaml"
        require(digest(config) == completion["resolved_config_sha256"], "Resolved config changed")
        self.check_resolved({"native": attempt / "native"})
        self.check_input_hashes(metadata["config_hashes_before"])
        require(metadata["source_head_before"] == self.expected_head
                and metadata["source_snapshot_before"]["tracked_and_untracked_sha256"] == self.expected_hash,
                "Native launch source identity differs")
        require(summary["status"] == "complete" and summary["world_size"] == WORLD
                and summary["training_steps"] == STEPS
                and summary["window"]["start_after_update"] == WARMUP
                and summary["window"]["end_after_update"] == STEPS
                and summary["tokens"]["physical_input_tokens_global"] == TOKENS,
                "Native timing window is incomplete")
        elapsed = summary["window"]["primary_elapsed_s"]
        require(math.isfinite(elapsed) and elapsed > 0, "Invalid native timing interval")
        tps = TOKENS / elapsed
        require(math.isclose(tps, summary["tokens"]["physical_input_tokens_per_second_global"]),
                "Native timing denominator differs")
        bridge = read_json(attempt / "flash-qla-bridge.json")
        route_hash = digest(self.root / "gift/torchtitan/distributed/deepep/routing.py")
        logs, activation = {}, {}
        for rank in range(WORLD):
            receipt = read_json(attempt / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json")
            verify_bridge(receipt, bridge, rank, route_hash)
            logs[rank], activation[rank] = read_rank(attempt, rank, self.ep, self.features)
        result = measured_result(logs)
        counters = summary["memory"]["allocator_counter_deltas_by_rank"]
        require(len(counters) == WORLD and all(
            value.get(key) == 0 for value in counters.values() for key in ("num_alloc_retries", "num_ooms")
        ), "Allocator retries or OOMs in the measured window")
        return result | {
            "attempt": str(attempt), "ep": self.ep, "efsdp": WORLD // self.ep, "arm": self.arm,
            "features": self.features, "effective_paths_by_rank": activation,
            "per_rank_console": logs, "elapsed_s": elapsed, "window_global_tps": tps,
            "compilation_status": summary["compilation_observer"]["status"],
            "allocator_counter_deltas_by_rank": counters,
            "source_head": self.expected_head, "source_sha256": self.expected_hash,
            "input_hashes": self.input_hashes, "resolved_config_sha256": digest(config),
        }


def verify_bridge(receipt, bridge, rank, route_hash):
    require(
        receipt["rank"] == rank and receipt["backend_variant"] == "fla"
        and receipt["routing_metadata"] == "original" and receipt["cross_entropy"] == "original"
        and receipt["capture_mode"] == "disabled" and receipt["native_main_completed"]
        and receipt["source_identity"] == bridge["source_identity"]
        and receipt["routing_metadata_callable"]["sha256"] == route_hash
        and receipt["installer_identity"]["variant"] == "fla"
        and "flash_qla" not in receipt["installer_identity"] and receipt["constructed_loss_callables"]
        and all(entry["callable"] == receipt["cross_entropy_callable"]
                for entry in receipt["constructed_loss_callables"]),
        f"Rank {rank} lacks active Original A FLA/routing/CE receipts",
    )


def read_rank(attempt, rank, ep, features):
    paths = list((attempt / "logs/torchrun").glob(f"*/attempt_*/{rank}/stdout.log"))
    require(len(paths) == 1 and paths[0].parents[1].name == "attempt_0",
            f"Rank {rank} stdout missing or has restarted workers")
    log = ANSI.sub("", paths[0].read_text())
    require(MESH.findall(log) == [str(ep)], f"Rank {rank} has a different mesh")
    meshes = MESH_AXES.findall(log)
    require(len(meshes) == 1 and "'fsdp'" in meshes[0] and "'ep'" in meshes[0]
            and ("'efsdp'" in meshes[0]) == (ep == 4), f"Rank {rank} lacks matching mesh axes")
    entries = list(STEP.finditer(log))
    rows = [
        [int(step), float(loss), float(norm), float(mem), int(tps.replace(",", ""))]
        for step, loss, norm, mem, tps in (entry.groups() for entry in entries)
    ]
    require([row[0] for row in rows] == list(range(1, STEPS + 1))
            and all(math.isfinite(value) for row in rows for value in row[1:])
            and all(row[3] > 0 and row[4] > 0 for row in rows), f"Invalid step metrics for rank {rank}")
    activation = {}
    for field, marker in MARKERS.items():
        if field in (ROW_OWNER, TRANSITION):
            marker = GFSDP_LOG_TAG + marker
        hits = [line.strip() for line in log.splitlines() if marker in line]
        require(len(hits) == int(features[field]), f"Rank {rank} activation mismatch for {field}")
        if hits:
            require(log.index(marker) < entries[0].start(), f"Rank {rank} activated {field} after update 1")
        activation[field] = hits[0] if hits else None
    return rows, activation


def measured_result(logs):
    memory, rank, update = max(
        (row[3], rank, row[0]) for rank, rows in logs.items() for row in rows[WARMUP:]
    )
    step_tps = [sum(rows[index][4] for rows in logs.values()) for index in range(WARMUP, STEPS)]
    midpoint = len(step_tps) // 2
    return {
        "total_gift_memory_gib": memory, "memory_max_rank": rank, "memory_max_update": update,
        "median_step_global_tps": median(step_tps), "step_global_tps": step_tps,
        "last_vs_first_five_step_tps_percent": 100 * (
            median(step_tps[midpoint:]) / median(step_tps[:midpoint]) - 1
        ),
    }


def worker(argv):
    path, *forwarded = argv
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(GFSDP_LOG_TAG + "%(message)s"))
    handler.addFilter(lambda record: record.getMessage().startswith(
        (MARKERS[ROW_OWNER], MARKERS[TRANSITION])
    ))
    logger = logging.getLogger("gfsdp")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    sys.argv = [path, *forwarded]
    runpy.run_path(path, run_name="__main__")


def check_gpus(path):
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader",
    ], text=True)
    path.write_text(inventory)
    rows = list(csv.reader(inventory.splitlines(), skipinitialspace=True))
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
    ], text=True)
    require(len(rows) == WORLD and all(len(row) == 4 and row[2] == "0 MiB" for row in rows)
            and len({row[1] for row in rows}) == WORLD and not apps.strip(),
            f"Expected eight free GPUs; see {path}")


def compare(results, suite):
    pairs = [("a", "all")] if suite == "final" else [
        ("a", "layer0"), ("a", "mtp"), ("a", "row-owner"), ("mtp", "mtp-transition"),
    ]
    summaries = []
    for ep in (8, 4):
        for control_arm, candidate_arm in pairs:
            control, candidate = (results.get(f"ep{ep}-{arm}") for arm in (control_arm, candidate_arm))
            if control is None or candidate is None:
                continue
            stable = control["compilation_status"] == candidate["compilation_status"] == "observed_stable"
            summaries.append({
                "ep": ep, "efsdp": WORLD // ep, "control": control_arm, "candidate": candidate_arm,
                "measured_compilation_stable": stable,
                "median_step_tps_percent": 100 * (
                    candidate["median_step_global_tps"] / control["median_step_global_tps"] - 1
                ),
                "window_tps_percent": 100 * (candidate["window_global_tps"] / control["window_global_tps"] - 1),
                "total_gift_memory_delta_gib": candidate["total_gift_memory_gib"] - control["total_gift_memory_gib"],
            })
    return summaries


def matrix(args):
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    require(args.original_root.resolve() != args.candidate_root.resolve(),
            "Use distinct Original A and candidate checkouts")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", campaign.name), "Invalid campaign name")
    suite = campaign / args.suite
    suite.mkdir(exist_ok=True)
    cases = FINAL if args.suite == "final" else PER_FIX
    keys = {f"ep{ep}-{arm}" for ep, arm in cases}
    require(set(args.case) <= keys, "Unknown case in this suite")
    plan = {
        "original_root": str(args.original_root.resolve()), "candidate_root": str(args.candidate_root.resolve()),
        "reference": str(args.reference.resolve()), "candidate_head": args.candidate_head,
        "candidate_source_sha256": args.candidate_source_sha256, "driver_sha256": digest(SCRIPT),
        "reference_metadata_sha256": digest(args.reference.resolve() / "metadata.json"),
        "order": [list(case) for case in cases], "warmup": WARMUP, "steps": STEPS,
    }
    plan_file = suite / "plan.json"
    with (campaign / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plan_file.exists():
            check_equal(plan, read_json(plan_file), "Campaign restart plan")
        else:
            write_json(plan_file, plan)

        results = {}
        for ep, arm in cases:
            key = f"ep{ep}-{arm}"
            case_file = suite / f"{key}.json"
            if case_file.exists():
                results[key] = read_json(case_file)
            elif not args.case or key in args.case:
                run_id = f"{campaign.name}-{args.suite}-{key}-{uuid4().hex[:8]}"
                gpu_csv = suite / f"{run_id}.gpus.csv"
                check_gpus(gpu_csv)
                command = [
                    sys.executable, "-Bu", str(SCRIPT), "arm", str(suite), str(ep), arm,
                    run_id, "--original-root", str(args.original_root),
                    "--candidate-root", str(args.candidate_root), "--reference", str(args.reference),
                    "--candidate-head", args.candidate_head,
                    "--candidate-source-sha256", args.candidate_source_sha256,
                    "--timeout-seconds", str(args.timeout_seconds),
                ]
                print(f"START {key} {run_id}", flush=True)
                with (suite / f"{run_id}.runner.log").open("x") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                with (suite / "attempts.jsonl").open("a") as handle:
                    print(json.dumps({
                        "case": key, "run_id": run_id, "returncode": completed.returncode,
                        "time": datetime.now(timezone.utc).isoformat(), "gpu_preflight": str(gpu_csv),
                    }), file=handle)
                print(f"{'PASS' if completed.returncode == 0 else 'FAIL'} {key} {run_id}", flush=True)
                if completed.returncode == 0:
                    results[key] = read_json(case_file)
            write_json(suite / "results.json", {
                "measured_updates": [WARMUP + 1, STEPS], "arms": results,
                "comparisons": compare(results, args.suite), "pending": sorted(keys - results.keys()),
            })
        selected = set(args.case) or keys
        return int(not selected <= results.keys() or any(
            results[key]["compilation_status"] != "observed_stable" for key in selected
        ))


def launch_arm(args):
    case = NativeCase(args, args.ep, args.arm)
    status = case.launch(args.run_id, args.timeout_seconds)
    if status:
        return status
    result = case.summarize(case.raw / args.run_id)
    write_json(args.suite_path / f"ep{args.ep}-{args.arm}.json", result)
    return 0


def add_native_args(parser):
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--candidate-source-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)


def main(argv):
    if argv[:1] == ["worker"]:
        worker(argv[1:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    show_source = subparsers.add_parser("source")
    show_source.add_argument("root", type=Path)
    run = subparsers.add_parser("matrix")
    run.add_argument("campaign", type=Path)
    run.add_argument("--suite", choices=("final", "per-fix"), required=True)
    run.add_argument("--case", action="append", default=[], help="limit to named cases, for example ep4-all")
    add_native_args(run)
    dry = subparsers.add_parser("preflight", help="verify frozen source, real inputs and generated CLI")
    dry.add_argument("ep", type=int, choices=(8, 4))
    dry.add_argument("arm", choices=ARMS)
    add_native_args(dry)
    arm = subparsers.add_parser("arm")
    arm.add_argument("suite_path", type=Path)
    arm.add_argument("ep", type=int, choices=(8, 4))
    arm.add_argument("arm", choices=ARMS)
    arm.add_argument("run_id")
    add_native_args(arm)
    args = parser.parse_args(argv)
    if args.mode == "source":
        root = args.root.resolve(strict=True)
        base = load_module(root / BASE, "_dense_gfsdp_source")
        base.verify_task_root(root)
        print(json.dumps(base.source_snapshot(root), indent=2))
        return 0
    if args.mode == "preflight":
        case = NativeCase(args, args.ep, args.arm)
        case.preflight(args.timeout_seconds)
        print(f"PASS source, real inputs and native CLI for EP{args.ep} {args.arm}")
        return 0
    return matrix(args) if args.mode == "matrix" else launch_arm(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
