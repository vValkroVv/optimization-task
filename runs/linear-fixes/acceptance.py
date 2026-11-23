#!/usr/bin/env python3
"""Run or summarize Linear-fixes acceptance on the Original A crop."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from copy import deepcopy
import os
from pathlib import Path
import re
import subprocess
from statistics import median
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "artifacts/raw"
BASE_RUNNER = ROOT / "runs/overnight-streaming-perf/run.py"
FLASH_RUNNER = ROOT / "runs/flash-qla/run.py"
MEASURE = FLASH_RUNNER.with_name("measure.py")
ACTIVATE = HERE / "activation.py"
ORIGINAL_A = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
DATA_RELATIVE = Path("runs/production-data-abcd/artifacts/raw/data-002")
FRAGMENT_RELATIVE = Path("runs/overnight-streaming-perf/mds-dataloader.yaml")
SWITCHES = (
    "GIFT_GDN_DIRECT_FLA_BACKWARD",
    "GIFT_GDN_FUSE_SHORT_CONV_BWD",
    "GIFT_GDN_FUSE_OUTPUT_NORM_FP8",
    "GIFT_MLA_FUSED_ROPE",
    "GIFT_GDN_DIRECT_OUTPUT_NORM_BWD",
)
ARMS = {
    "baseline": "00000",
    "scan": "10000",
    "short-conv": "01000",
    "fp8": "00100",
    "norm-backward": "00101",
    "mla": "00010",
    "combined": "11111",
}
ORDER = (
    (8, "baseline"), (4, "baseline"),
    (8, "scan"), (4, "short-conv"), (8, "short-conv"), (4, "scan"),
    (8, "fp8"), (8, "norm-backward"), (4, "fp8"), (4, "norm-backward"),
    (8, "mla"), (4, "mla"), (8, "combined"), (4, "combined"),
)
ACTIVATION_ARMS = ("baseline", "norm-backward", "combined")
WORLD = 8
STEPS = 60
WARMUP = 30
TOKENS = 62_914_560
ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRIC = re.compile(
    r"\bstep:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)\s+"
    r"memory:\s*(\S+)\s*GiB.*?tps:\s*([\d,]+)"
)
MESH = re.compile(
    r"Building device mesh with parallelism: "
    r"pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=(\d+)\b"
)
CREATED_MESH = re.compile(r"Successfully created meshes with active dimensions: (\[[^\n]*\])")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def flags(arm):
    return dict(zip(SWITCHES, ARMS[arm], strict=True))


def case_path(campaign, ep, arm, mode="run"):
    tag = campaign.name if mode == "run" else f"{campaign.name}-{mode}"
    return RAW / f"{tag}-ep{ep}-{arm}"


def native_config_and_overrides(argv):
    split = argv.index("--") + 1
    config = Path(argv[split])
    items = argv[split + 1:]
    require(len(items) % 2 == 0, "Native CLI contains an unpaired override")
    overrides = dict(zip(items[::2], items[1::2], strict=True))
    require(len(overrides) * 2 == len(items), "Native CLI repeats an override")
    overrides.pop("--dump_folder")
    return config, overrides


def differing_config_keys(expected, actual, prefix=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                yield name
            else:
                yield from differing_config_keys(expected[key], actual[key], name)
    elif expected != actual:
        yield prefix


class Campaign:
    def __init__(self, path, base):
        self.path = path
        self.base = base
        self.inputs = read_json(path / "inputs.json")
        self.inputs_hash = digest(path / "inputs.json")
        self.source_hash = self.inputs["source_sha256"]
        self.head = self.inputs.get("source_head", ORIGINAL_A)
        require(re.fullmatch(r"[0-9a-f]{64}", self.source_hash), "Missing approved source hash")
        self.reference = Path(self.inputs["reference"]).resolve(strict=True)
        self.input_root = Path(self.inputs.get("input_root", ROOT)).resolve(strict=True)
        self.cache = self.inputs.get("cache")
        if self.cache:
            self.cache = Path(self.cache).resolve(strict=True)

        self.reference_metadata = read_json(self.reference / "metadata.json")
        reference_done = read_json(self.reference / "completion.json")
        config = self.reference / "native/config/resolved_config.yaml"
        require(
            reference_done["status"] == "PASS"
            and self.reference_metadata["source_head_before"] == ORIGINAL_A
            and digest(config) == reference_done["resolved_config_sha256"],
            "Original A reference is not a completed, verified run",
        )
        self.ref_hashes = self.reference_metadata["config_hashes_before"]
        self.reference_root = Path(self.reference_metadata["task_root"])
        self.configure_paths()

    def configure_paths(self):
        base = self.base
        base.DATA_FRAGMENT = self.input_root / FRAGMENT_RELATIVE
        base.DATASET_ROOT = self.input_root / DATA_RELATIVE
        base.DATA_RECEIPT = self.input_root / "runs/production-data-abcd/data-002.json"
        base.PREPARATION_STATUS = base.DATA_RECEIPT.with_name("data-002-preparation-status.json")
        base.LOADER_SUMMARY = base.DATA_RECEIPT.with_name("data-002-loader-summary.json")
        base.ROW_MANIFEST = base.DATASET_ROOT / "data-002-rows.jsonl"

        resolve_configured = base._configured_path
        inside = base._inside

        def resolve_path(value, *, base: Path, label):
            selected = resolve_configured(value, base=base, label=label)
            if label == "canonical dataset stream local" and selected == (ROOT / DATA_RELATIVE).resolve():
                return self.base.DATASET_ROOT.resolve()
            return selected

        def inside_path(root, path, label):
            if label == "MDS data path" and path.resolve() == base.DATASET_ROOT.resolve():
                return path.resolve()
            return inside(root, path, label)

        base._configured_path = resolve_path
        base._inside = inside_path

    def verify_snapshot(self, original, root):
        snapshot = original(root)
        require(
            snapshot["head"] == self.head
            and snapshot["tracked_and_untracked_sha256"] == self.source_hash,
            "GIFT source differs from the approved frozen snapshot",
        )
        return snapshot

    def verify_inputs(self):
        base = self.base
        verified = base._verify_ready_inputs(ROOT)
        current = base._config_hashes()
        for key in ("mds_index_sha256", "row_manifest_sha256", "canonical_experiment_sha256",
                    "tokenizer_json_sha256", "tokenizer_config_sha256", "special_tokens_map_sha256"):
            require(current[key] == self.ref_hashes[key], f"Original A input differs at {key}")

        # A staged copy needs local stream paths and the matching loader receipt link.
        for key, staged in (
            ("mds_fragment_sha256", base.DATA_FRAGMENT),
            ("data_receipt_sha256", base.DATA_RECEIPT),
            ("cpu_loader_receipt_sha256", base.LOADER_SUMMARY),
        ):
            if current[key] == self.ref_hashes[key]:
                continue
            original = self.reference_root / staged.relative_to(self.input_root)
            require(digest(original) == self.ref_hashes[key], f"Original A input hash differs at {original}")
            if key == "mds_fragment_sha256":
                expected = base._load_yaml(original)
                actual = base._load_yaml(staged)
                expected["dataloader"]["dataset"]["streams"][0]["local"] = str(base.DATASET_ROOT)
            else:
                expected = read_json(original)
                actual = read_json(staged)
                if key == "data_receipt_sha256":
                    expected["dataset"]["root"] = str(base.DATASET_ROOT)
                    expected["dataset"]["row_manifest"]["path"] = str(base.ROW_MANIFEST)
                else:
                    expected["data_receipt_sha256"] = current["data_receipt_sha256"]
                    expected["dataset"]["root"] = str(base.DATASET_ROOT)
                    expected["dataset"]["row_manifest"]["path"] = str(base.ROW_MANIFEST)
                expected = self.relocate_input_paths(expected)
                actual = self.relocate_input_paths(actual)
            differing = list(differing_config_keys(expected, actual))
            require(not differing, f"Staged input differs from Original A in {staged.name}: {differing}")
        return verified

    def relocate_input_paths(self, value):
        if isinstance(value, dict):
            return {key: self.relocate_input_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.relocate_input_paths(item) for item in value]
        if isinstance(value, str) and (value == str(self.reference_root) or value.startswith(f"{self.reference_root}/")):
            return str(self.input_root) + value[len(str(self.reference_root)):]
        return value

    def verify_cache_runtime(self, original, producer, current):
        normalized = deepcopy(producer)
        old = self.reference_root / "gift/torchtitan/__init__.py"
        new = self.base.GIFT_ROOT / "torchtitan/__init__.py"
        package = normalized["packages"]["torchtitan"]
        require(package["path"] in {str(old), str(new)},
                f"Unknown compiler cache producer torchtitan path: {package['path']}")
        package["path"] = str(new)
        return original(normalized, current)

    def check_native_command(self, argv, ep, steps):
        _, expected = native_config_and_overrides(self.reference_metadata["argv"])
        config, actual = native_config_and_overrides(argv)
        expected["--training.steps"] = str(steps)
        if ep != 8:
            expected["--parallelism.expert_parallel_degree"] = str(ep)
        differing = list(differing_config_keys(expected, actual))
        require(not differing, f"Native CLI differs from Original A at {differing}")
        document = self.base._load_yaml(config)
        require(
            document == {
                "config_version": 1,
                "imports": [str(self.base.BASE_CONFIG.resolve()), str(self.base.DATA_FRAGMENT.resolve())],
                "callbacks": {"monitoring": None, "system_metrics": None, "timers": None},
                "loss": {"num_chunks_on_monitoring_steps": None},
            },
            "Generated native config changed beyond the approved import paths",
        )

    def check_resolved(self, paths, ep, steps):
        resolved_path = paths["native"] / "config/resolved_config.yaml"
        expected = self.base._load_yaml(self.reference / "native/config/resolved_config.yaml")
        actual = self.base._load_yaml(resolved_path)
        expected["dump_folder"] = str(paths["native"].resolve())
        expected["training"]["steps"] = steps
        expected["parallelism"]["expert_parallel_degree"] = ep
        expected_stream = expected["dataloader"]["dataset"]["streams"][0]
        actual_stream = actual["dataloader"]["dataset"]["streams"][0]
        old_stream = Path(expected_stream["local"])
        if not old_stream.is_absolute():
            old_stream = self.reference_root / "gift" / old_stream
        approved_streams = {
            self.base.DATASET_ROOT.resolve(),
            (self.reference_root / DATA_RELATIVE).resolve(),
        }
        require(old_stream.resolve() in approved_streams, f"Unknown Original A stream: {old_stream}")
        current_stream = self.base._configured_path(
            actual_stream["local"], base=self.base.GIFT_ROOT, label="resolved data stream local"
        )
        require(current_stream == self.base.DATASET_ROOT.resolve(), "Active dataset path differs")
        expected_stream["local"] = actual_stream["local"]
        differing = list(differing_config_keys(expected, actual))
        require(not differing, f"Resolved config differs from Original A at {differing}")

    def receipt(self, ep, arm):
        return {
            "ep": ep,
            "efsdp": WORLD // ep,
            "fsdp": WORLD,
            "switches": flags(arm),
            "source_head": self.head,
            "source_sha256": self.source_hash,
            "original_a_reference": str(self.reference),
            "input_root": str(self.input_root),
            "inputs_sha256": self.inputs_hash,
        }


def forwarded_args(campaign, attempt, mode):
    activate = mode == "activate"
    argv = [
        "--num-prefetch-batches", "0", "--epoch-size", "102400",
        "--run-id", attempt.name, "--output-root", str(RAW),
        "--steps", "1" if activate else str(STEPS),
        "--warmup", "0" if activate else str(WARMUP),
        "--window", "0" if activate else str(STEPS - WARMUP),
        "--timeout-seconds", "7200",
    ]
    if campaign.cache:
        argv.extend(("--compiler-cache-root", str(campaign.cache)))
    if not activate:
        argv.append("--observe-compiler-cache")
    return argv


def set_ep(argv, ep):
    if ep != 8:
        argv.extend(("--parallelism.expert_parallel_degree", str(ep)))
    return argv


def dry_arm(campaign, ep, arm):
    base = campaign.base
    attempt = campaign.path / "dry" / f"ep{ep}-{arm}"
    attempt.mkdir(parents=True, exist_ok=False)
    paths = {key: attempt / key for key in ("config", "native", "logs", "measurement")}
    for path in paths.values():
        path.mkdir()
    base._write_experiment_config(paths["config"] / "experiment.yaml")
    args = base._parse_args(forwarded_args(campaign, attempt, "dry-run"))
    bridge = load_module(FLASH_RUNNER, "_linear_flash_bridge")
    argv = set_ep(bridge._build_command_hook(base)(args, paths), ep)
    campaign.check_native_command(argv, ep, STEPS)
    native_argv = argv[argv.index("--") + 1:]
    env = os.environ | flags(arm) | {
        "PYTHONPATH": os.pathsep.join((
            str(base.GIFT_ROOT), str(base.GIFT_ROOT / "contrib/flash-linear-attention"),
        )),
        "WORLD_SIZE": str(WORLD),
        "LOCAL_WORLD_SIZE": str(WORLD),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (attempt / "resolve.log").open("x") as log:
        result = subprocess.run(
            [str(base.PYTHON), str(Path(__file__).resolve()), "resolve", *native_argv],
            cwd=base.GIFT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    require(result.returncode == 0, f"Native config preflight failed; see {log.name}")
    campaign.check_resolved(paths, ep, STEPS)
    write_json(campaign.path / "dry" / f"ep{ep}-{arm}.verified.json", campaign.receipt(ep, arm) | {
        "dry_argv": argv,
        "resolved_config_sha256": digest(paths["native"] / "config/resolved_config.yaml"),
    })


def run_arm(campaign, ep, arm, mode):
    base = campaign.base
    activate = mode == "activate"
    attempt = case_path(campaign.path, ep, arm, mode)
    dry = read_json(campaign.path / "dry" / f"ep{ep}-{arm}.verified.json")
    receipt = campaign.receipt(ep, arm)
    require(all(dry.get(key) == val for key, val in receipt.items()), "Dry preflight identity differs")
    flash = load_module(FLASH_RUNNER, "_linear_flash_bridge")
    original_command_hook = flash._build_command_hook
    original_environment_hook = flash._build_environment_hook
    original_snapshot = base.source_snapshot
    original_config = base._verify_resolved_config
    original_expect = base._resolved_config_expectations
    original_cache_runtime = base._assert_runtime_compatible

    def build_command(b):
        command = original_command_hook(b)

        def get_command(args, paths):
            argv = set_ep(command(args, paths), ep)
            if activate:
                index = argv.index(str(MEASURE))
                argv[index:index + 1] = [
                    str(ACTIVATE), "--expected-ep", str(ep),
                    "--expected-switches", ARMS[arm],
                ]
            campaign.check_native_command(argv, ep, args.steps)
            write_json(paths["attempt"] / "acceptance-launch.json", receipt | {
                "argv": argv, "arm": arm, "mode": mode,
                "runner_sha256": digest(Path(__file__)),
                "compiler_cache_root": str(paths["cache"]),
            })
            return argv

        return get_command

    def build_environment(b, **kwargs):
        environment = original_environment_hook(b, **kwargs)

        def get_environment(paths):
            return environment(paths) | flags(arm)

        return get_environment

    def verify_resolved(paths, args):
        resolved_hash = original_config(paths, args)
        campaign.check_resolved(paths, ep, args.steps)
        return resolved_hash

    campaign.verify_inputs()
    # Native workers resolve the default artifact root before accepting a sibling run root.
    base.RAW_ROOT.mkdir(parents=True, exist_ok=True)
    base.source_snapshot = lambda root: campaign.verify_snapshot(original_snapshot, root)
    base._resolved_config_expectations = lambda args: original_expect(args) | {
        "parallelism.expert_parallel_degree": ep,
    }
    base._verify_resolved_config = verify_resolved
    base._assert_runtime_compatible = lambda producer, current, field="runtime": (
        campaign.verify_cache_runtime(original_cache_runtime, producer, current)
        if field == "runtime" else original_cache_runtime(producer, current, field)
    )
    flash._load_base_runner = lambda: base
    flash._build_command_hook = build_command
    flash._build_environment_hook = build_environment
    result = flash.main([
        "--gdn-backend", "fla", "--routing-metadata", "original", "--cross-entropy", "original",
        *forwarded_args(campaign, attempt, mode),
    ])
    if result == 0 and activate:
        activation = load_module(ACTIVATE, "_linear_activation")
        evidence = activation.verify_receipts(
            attempt / "measurement", ep, tuple(bit == "1" for bit in ARMS[arm])
        )
        destination = campaign.path / "activation"
        destination.mkdir(exist_ok=True)
        write_json(destination / f"ep{ep}-{arm}.verified.json", receipt | {"evidence": evidence})
    return result


def resolve(argv):
    from torchtitan.config.cli import load_trainer_config
    from torchtitan.config.save import save_loaded_run_bundle
    from torchtitan.tools.logging import init_logger, logger

    init_logger()
    loaded = load_trainer_config(argv, logger=logger)
    save_loaded_run_bundle(loaded, logger=logger)


def rank_log(attempt, rank, ep):
    paths = list((attempt / "logs/torchrun").glob(f"*/attempt_*/{rank}/stdout.log"))
    require(len(paths) == 1 and paths[0].parents[1].name == "attempt_0",
            f"Rank {rank} stdout missing or ambiguous")
    log = ANSI.sub("", paths[0].read_text())
    require(MESH.findall(log) == [str(ep)], f"Rank {rank} dense/expert mesh mismatch")
    created = CREATED_MESH.findall(log)
    require(len(created) == 1 and "'fsdp'" in created[0] and "'ep'" in created[0]
            and ("'efsdp'" in created[0]) == (ep == 4),
            f"Rank {rank} lacks confirmation of active mesh axes: {created}")
    result = [
        [int(step), float(loss), float(norm), float(mem), int(tps.replace(",", ""))]
        for step, loss, norm, mem, tps in METRIC.findall(log)
    ]
    require([row[0] for row in result] == list(range(1, STEPS + 1)),
            f"Rank {rank} did not log updates 1–{STEPS} exactly once")
    require(all(math.isfinite(value) for row in result for value in row[1:]),
            f"Rank {rank} logged nonfinite metrics")
    return result


def summarize_arm(path, ep, arm):
    attempt = case_path(path, ep, arm)
    comp = read_json(attempt / "completion.json")
    summary = read_json(attempt / "summary.json")
    require(comp["status"] == "PASS" and comp["source_unchanged"] and comp["config_unchanged"],
            f"Native run did not complete with frozen source and config: {attempt}")
    require(digest(attempt / "native/config/resolved_config.yaml") == comp["resolved_config_sha256"],
            f"Resolved config changed: {attempt}")
    launch = read_json(attempt / "acceptance-launch.json")
    inputs = read_json(path / "inputs.json")
    require(
        launch["switches"] == flags(arm) and launch["source_sha256"] == inputs["source_sha256"]
        and launch["inputs_sha256"] == digest(path / "inputs.json"),
        f"Timed launch identity mismatch: {attempt}",
    )
    require(summary["status"] == "complete" and summary["world_size"] == WORLD
            and summary["training_steps"] == STEPS
            and summary["window"]["start_after_update"] == WARMUP
            and summary["window"]["end_after_update"] == STEPS
            and summary["tokens"]["physical_input_tokens_global"] == TOKENS,
            f"Missing complete all-rank 30+30 window: {attempt}")
    elapsed = summary["window"]["primary_elapsed_s"]
    require(math.isfinite(elapsed) and elapsed > 0, f"Invalid window elapsed time: {attempt}")
    tps = TOKENS / elapsed
    require(math.isclose(tps, summary["tokens"]["physical_input_tokens_per_second_global"]),
            f"Native TPS mismatch: {attempt}")
    bridge = read_json(attempt / "flash-qla-bridge.json")
    logs = {}
    routing_hash = digest(ROOT / "gift/torchtitan/distributed/deepep/routing.py")
    for rank in range(WORLD):
        rec = read_json(attempt / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json")
        require(
            rec["rank"] == rank and rec["backend_variant"] == "fla"
            and rec["routing_metadata"] == "original" and rec["cross_entropy"] == "original"
            and rec["capture_mode"] == "disabled" and rec["native_main_completed"]
            and rec["source_identity"] == bridge["source_identity"]
            and rec["routing_metadata_callable"]["sha256"] == routing_hash
            and rec["installer_identity"]["variant"] == "fla"
            and "flash_qla" not in rec["installer_identity"]
            and bool(rec["constructed_loss_callables"])
            and all(entry["callable"] == rec["cross_entropy_callable"]
                    for entry in rec["constructed_loss_callables"]),
            f"Rank {rank} lacks native Original A route evidence",
        )
        logs[rank] = rank_log(attempt, rank, ep)
    memory, update, rank = max(
        (row[3], row[0], rank)
        for rank, rows in logs.items() for row in rows[WARMUP:]
    )
    update_tps = [
        sum(logs[rank][index][4] for rank in logs) for index in range(WARMUP, STEPS)
    ]
    require(min(update_tps) > 0, f"Nonpositive native update TPS: {attempt}")
    first_tps, last_tps = median(update_tps[:10]), median(update_tps[-10:])
    observer = summary["compilation_observer"]
    result = launch | {
        "attempt": str(attempt),
        "window_elapsed_s": elapsed,
        "physical_input_tokens": TOKENS,
        "window_global_tps": tps,
        "median_step_global_tps": median(update_tps),
        "last_vs_first_ten_step_tps_percent": 100 * (last_tps / first_tps - 1),
        "compiler_observer_status": observer["status"],
        "allocator_counter_deltas_by_rank": summary["memory"]["allocator_counter_deltas_by_rank"],
        "total_gpu_memory_gib": memory,
        "memory_max_rank": rank,
        "memory_max_update": update,
        "console_fields": ["step", "loss", "grad_norm", "total_gpu_memory_gib", "local_tps"],
        "per_rank_console": logs,
    }
    write_json(attempt / "acceptance-summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_rank_console"}), flush=True)


def percentage(candidate, control):
    return 100 * (candidate["window_global_tps"] / control["window_global_tps"] - 1)


def matrix(campaign):
    rows = []
    unavailable = []
    unstable = []
    for ep, arm in ORDER:
        path = case_path(campaign, ep, arm) / "acceptance-summary.json"
        if path.is_file():
            row = read_json(path)
            if row["compiler_observer_status"] != "observed_stable":
                unstable.append(f"EP{ep} {arm} ({row['compiler_observer_status']})")
            rows.append(row)
        else:
            unavailable.append(f"EP{ep} {arm}")
    results = {(row["ep"], row["arm"]): row for row in rows}
    lines = ["# Linear-fixes acceptance", ""]
    for ep in (8, 4):
        lines.extend([
            f"## EP{ep} / eFSDP{WORLD // ep}", "",
            "| Arm | Global TPS | TPS vs. baseline | Total GPU memory, GiB | Memory vs. baseline, GiB | Peak rank / update |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ])
        baseline = results.get((ep, "baseline"))
        for arm in ARMS:
            item = results.get((ep, arm))
            if not item:
                lines.append(f"| {arm} | — | — | — | — | — |")
                continue
            compare = baseline is not None and arm != "baseline"
            tps_delta = percentage(item, baseline) if compare else None
            memory_delta = item["total_gpu_memory_gib"] - baseline["total_gpu_memory_gib"] if compare else None
            item["baseline_tps_change_percent"] = tps_delta
            item["baseline_memory_change_gib"] = memory_delta
            lines.append(
                f"| {arm} | {item['window_global_tps']:,.2f} | "
                + (f"{tps_delta:+.2f}%" if tps_delta is not None else "—")
                + f" | {item['total_gpu_memory_gib']:.2f} | "
                + (f"{memory_delta:+.2f}" if memory_delta is not None else "—")
                + f" | {item['memory_max_rank']} / {item['memory_max_update']} |"
            )
        lines.append("")
        fp8 = results.get((ep, "fp8"))
        backward = results.get((ep, "norm-backward"))
        if fp8 and backward:
            backward["fp8_tps_change_percent"] = percentage(backward, fp8)
            backward["fp8_memory_change_gib"] = backward["total_gpu_memory_gib"] - fp8["total_gpu_memory_gib"]
            lines.extend([
                f"Output-norm backward vs. FP8 forward only: "
                f"{backward['fp8_tps_change_percent']:+.2f}% TPS, "
                f"{backward['fp8_memory_change_gib']:+.2f} GiB.", "",
            ])
    lines.append(
        "Global physical TPS uses updates 31–60 and the all-rank elapsed window. "
        "Memory is the maximum native GIFT total over the same updates and all ranks. "
        "Small differences remain provisional with one run per arm."
    )
    if unavailable:
        lines.extend(("", "Pending or failed arms: " + ", ".join(unavailable) + "."))
    if unstable:
        lines.extend(("", "Compiler-window stability unverified: " + ", ".join(unstable) + "."))
    write_json(campaign / "matrix-results.json", {
        "arms": rows, "unavailable": unavailable, "compiler_stability_unverified": unstable,
    })
    with (campaign / "matrix-results.md").open("x") as report:
        report.write("\n".join(lines) + "\n")
    return int(bool(unavailable or unstable))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["resolve"]:
        resolve(argv[1:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("dry-run", "activate", "run", "summarize", "preflight", "matrix", "verify-ready"):
        command = sub.add_parser(mode)
        command.add_argument("campaign", type=Path)
        if mode in ("dry-run", "activate", "run", "summarize"):
            command.add_argument("ep", type=int, choices=(8, 4))
            command.add_argument("arm", choices=ARMS)
    sub.add_parser("source")
    sub.add_parser("order")
    sub.add_parser("activation-order")
    args = parser.parse_args(argv)
    if args.mode in ("order", "activation-order"):
        for ep, arm in ORDER:
            if args.mode == "order" or arm in ACTIVATION_ARMS:
                print(f"{ep}:{arm}")
        return 0
    if args.mode == "source":
        base = load_module(BASE_RUNNER, "_linear_native_runner")
        print(json.dumps(base.source_snapshot(ROOT), indent=2))
        return 0
    path = args.campaign.resolve(strict=True)
    require(path.parent == RAW.resolve(), "Campaign must be a direct child of the linear-fixes raw directory")
    if args.mode == "matrix":
        return matrix(path)
    if args.mode == "summarize":
        summarize_arm(path, args.ep, args.arm)
        return 0

    base = load_module(BASE_RUNNER, "_linear_native_runner")
    campaign = Campaign(path, base)
    if args.mode == "verify-ready":
        for ep, arm in ORDER:
            for folder in ("dry", "activation") if arm in ACTIVATION_ARMS else ("dry",):
                observed = read_json(path / folder / f"ep{ep}-{arm}.verified.json")
                expected = campaign.receipt(ep, arm)
                require(all(observed.get(key) == val for key, val in expected.items()),
                        f"Campaign receipt identity differs: {folder} EP{ep} {arm}")
        return 0
    if args.mode in ("dry-run", "preflight"):
        campaign.verify_inputs()
        campaign.verify_snapshot(base.source_snapshot, ROOT)
        cases = ORDER if args.mode == "preflight" else [(args.ep, args.arm)]
        for ep, arm in cases:
            dry_arm(campaign, ep, arm)
            print(f"Dry preflight verified EP{ep} {arm}", flush=True)
        campaign.verify_snapshot(base.source_snapshot, ROOT)
        return 0
    return run_arm(campaign, args.ep, args.arm, args.mode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"acceptance: {error}", file=sys.stderr)
        raise SystemExit(1) from error
