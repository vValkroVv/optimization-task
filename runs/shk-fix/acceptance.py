#!/usr/bin/env python3
"""Run matched Original A shared-Hopper-kernel acceptance arms serially."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "artifacts/raw"
FLASH = ROOT / "runs/flash-qla/run.py"
ORIGINAL_A = Path("/workspace-SR008.nfs2/vkropotin/a10-cute-001")
REVISION = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
REFERENCE_HASH = "6710fb476fb75c3cd0763f242506cf0ebeee194f53e82b70baf1f3b418b69d87"
DATA = Path("runs/production-data-abcd")
A_RAW = DATA / "artifacts/raw"
DATASET = A_RAW / "data-002"
REFERENCE = A_RAW / "original-a-vs-cute-140-001-a"
CACHE_PRODUCER = A_RAW / "original-a-vs-cute-001-a2"
CACHE = ROOT / CACHE_PRODUCER / "cache"
SWITCHES = (
    "GIFT_CUTE_NORM_FUSE_RESIDUAL",
    "GIFT_FP8_QUANTIZE_WEIGHT_TAIL",
    "GIFT_CUTE_NORM_SAC_STATS_ONLY",
)
EXPERT_PARALLEL_DEGREES = (8, 4)
WORLD = 8
PHYSICAL_TOKENS_PER_UPDATE = 512 * 4096
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"\bstep:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)\s+memory:\s*([\d.]+)\s*GiB")
MESH = re.compile(r"Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=(\d+)\b")
MESH_AXES = re.compile(r"Successfully created meshes with active dimensions: (\[[^\n]*\])")
SKIP_ACTIVATION = re.compile(r"CuTe norm skip-gradient fusion applied to (\d+) site\(s\)")
SKIP = "GIFT_CUTE_NORM_BWD_SKIP"
ROUTED_OVERWRITE = "GIFT_ROUTED_DW_BF16_OVERWRITE"
ROUTED_BUILD = f"{ROUTED_OVERWRITE}_BUILD"
ROUTED_ACTIVATION = "Routed BF16 dW overwrite activated"
NORM_PAIR = "GIFT_CUTE_NORM_PAIR"
NORM_PAIR_ACTIVATION = "CuTe norm pair candidate activated"
NORM_PAIR_WORKER = HERE / "followup/native_norm_pair_measure.py"
NORM_PAIR_RESULT = RAW / "norm-pair-tile-compiled-002/component.json"
NORM_PAIR_CANDIDATE = RAW / "norm-pair-tile-reuse-002/candidate.py"
NORM_PAIR_SOURCES = (
    NORM_PAIR_WORKER,
    NORM_PAIR_CANDIDATE,
    HERE / "followup/norm_pair_tile_compiled.py",
    HERE / "followup/norm_pair_tile_ops.py",
    HERE / "followup/norm_f1_component.py",
    HERE / "followup/norm_skip.py",
)
SWIGLU_ORDER = "GIFT_ROUTED_SWIGLU_FWD_FEATURE_FIRST"
SWIGLU_ORDER_WORKER = HERE / "followup/native_routed_swiglu_fwd_measure.py"
NORM_PAIR_SWIGLU_WORKER = HERE / "followup/native_norm_pair_swiglu_measure.py"
SWIGLU_ORDER_PROBE = HERE / "followup/routed_swiglu_fwd_order_component.py"
SWIGLU_ORDER_RESULT = RAW / "routed-swiglu-fwd-order-001/component.json"
SWIGLU_SOURCE = ROOT / "gift/torchtitan/ops/float8/triton_kernels/fused_swiglu_quantization.py"
SWIGLU_BWD_ORDER = "GIFT_ROUTED_SWIGLU_BWD_FEATURE_FIRST"
SWIGLU_BWD_ORDER_WORKER = HERE / "followup/native_routed_swiglu_bwd_measure.py"
SWIGLU_BWD_ORDER_PROBE = HERE / "followup/routed_swiglu_bwd_order_component.py"
SWIGLU_BWD_ORDER_RESULT = RAW / "routed-swiglu-bwd-order-001/result.json"
DENSE_DY_QKV = "GIFT_FP8_DENSE_DY_QKV_FUSION"
DENSE_DY_WORKER = HERE / "followup/native_dense_dy_qkv_measure.py"
DENSE_DY_PROBE = HERE / "followup/dense_dy_quant_transpose.py"
DENSE_DY_RESULTS = (
    RAW / "dense-dy-backward-qkv-001/component.json",
    RAW / "dense-dy-backward-qkv-early-002/component.json",
)
DENSE_DY_SHAPES = {"dy": [2, 4096, 8192], "x": [2, 4096, 4096], "weight": [8192, 4096]}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_flash():
    spec = importlib.util.spec_from_file_location("_shk_flash", FLASH)
    flash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flash)
    return flash


def configure_inputs(base, original):
    # Keep the original dataset and its receipts read-only at their verified paths.
    base.DATASET_ROOT = original / DATASET
    base.DATA_FRAGMENT = original / "runs/overnight-streaming-perf/mds-dataloader.yaml"
    base.DATA_RECEIPT = original / DATA / "data-002.json"
    base.PREPARATION_STATUS = base.DATA_RECEIPT.with_name("data-002-preparation-status.json")
    base.LOADER_SUMMARY = base.DATA_RECEIPT.with_name("data-002-loader-summary.json")
    base.ROW_MANIFEST = base.DATASET_ROOT / "data-002-rows.jsonl"
    resolve = base._configured_path
    inside = base._inside

    def configured(value, *, base: Path, label):
        path = resolve(value, base=base, label=label)
        if label == "canonical dataset stream local" and path == (ROOT / DATASET).resolve():
            return (original / DATASET).resolve()
        return path

    def inside_path(root, path, label):
        if label == "MDS data path" and path.resolve() == base.DATASET_ROOT.resolve():
            return path.resolve()
        return inside(root, path, label)

    base._configured_path = configured
    base._inside = inside_path


def native_args(argv):
    split = argv.index("--") + 1
    items = argv[split + 1:]
    overrides = dict(zip(items[::2], items[1::2], strict=True))
    require(len(overrides) * 2 == len(items), "Native CLI repeats an override")
    overrides.pop("--dump_folder")
    return Path(argv[split]), overrides


def assert_config(base, expected, path, ep, native, window):
    config = deepcopy(expected)
    config["dump_folder"] = str(native)
    config["training"]["steps"] = 2 * window
    config["parallelism"]["expert_parallel_degree"] = ep
    actual = base._load_yaml(path)
    if actual != config:
        changed = sorted(key for key in actual.keys() | config.keys() if actual.get(key) != config.get(key))
        raise RuntimeError(f"Resolved config differs from Original A at {changed}")


def command_options(ep):
    return ["--parallelism.expert_parallel_degree", str(ep)] if ep == 4 else []


def comparison_arms(followups, *, six_fix_control=False):
    enabled = dict.fromkeys(SWITCHES, "1")
    if six_fix_control:
        require(
            followups == {SWIGLU_ORDER: "1"},
            "The six-fix control requires the routed SwiGLU forward candidate alone",
        )
        enabled.update(dict.fromkeys((SKIP, NORM_PAIR, ROUTED_OVERWRITE), "1"))
    if not followups:
        return {"baseline": dict.fromkeys(SWITCHES, "0"), "all-three": enabled}
    return {
        "baseline": enabled | dict.fromkeys(followups, "0"),
        "candidate": enabled | followups,
    }


def final_arms():
    additions = (SKIP, NORM_PAIR, ROUTED_OVERWRITE)
    return {
        "baseline": dict.fromkeys((*SWITCHES, *additions), "0"),
        "all-three": dict.fromkeys(SWITCHES, "1") | dict.fromkeys(additions, "0"),
        "all-accepted": dict.fromkeys((*SWITCHES, *additions), "1"),
    }


def comparison_order(arms, candidate_first=False):
    arm_order = tuple(arms.items())
    if candidate_first:
        arm_order = arm_order[::-1]
    return ((ep, arm, env) for ep in EXPERT_PARALLEL_DEGREES for arm, env in arm_order)


def forwarded(run_id, window):
    return [
        "--run-id", run_id, "--output-root", str(RAW), "--compiler-cache-root", str(CACHE),
        "--num-prefetch-batches", "0", "--epoch-size", "102400",
        "--steps", str(2 * window), "--warmup", str(window), "--window", str(window),
        "--observe-compiler-cache", "--timeout-seconds", "7200",
    ]


def build_arm_command(builder, base, switches):
    bridges = {
        SWIGLU_ORDER: SWIGLU_ORDER_WORKER,
        SWIGLU_BWD_ORDER: SWIGLU_BWD_ORDER_WORKER,
        NORM_PAIR: NORM_PAIR_WORKER,
        DENSE_DY_QKV: DENSE_DY_WORKER,
    }
    selected = [path for switch, path in bridges.items() if switch in switches]
    if len(selected) == 2 and switches.get(NORM_PAIR) == "1" and SWIGLU_ORDER in switches:
        return builder(base, NORM_PAIR_SWIGLU_WORKER)
    require(len(selected) <= 1, "Measurement bridges require separate campaigns")
    return builder(base, *selected)


def arm_command(command, ep, switches, reference_argv, generated, base, window):
    argv = command + command_options(ep)
    _, expected_overrides = native_args(reference_argv)
    expected_overrides["--training.steps"] = str(2 * window)
    if ep == 4:
        expected_overrides["--parallelism.expert_parallel_degree"] = str(ep)
    config, current_overrides = native_args(argv)
    require(current_overrides == expected_overrides, "Native CLI differs from Original A")
    require(base._load_yaml(config) == generated, "Generated experiment differs from Original A")
    # Apply the exact arm values after the existing launcher's image profile is sourced.
    env = switches
    if switches.get(ROUTED_OVERWRITE) == "1":
        env = switches | {ROUTED_BUILD: str(Path(os.environ[ROUTED_BUILD]).resolve(strict=True))}
    return ["/usr/bin/env", *(f"{key}={value}" for key, value in env.items()), *argv]


def preflight(campaign, flash, base, reference_argv, generated, resolved, window, state):
    for ep, arm, switches in comparison_order(state["arms"], state["candidate_first"]):
        attempt = campaign / f"ep{ep}-{arm}"
        paths = {key: attempt / key for key in ("config", "native", "measurement", "logs")}
        for path in paths.values():
            path.mkdir(parents=True)
        base._write_experiment_config(paths["config"] / "experiment.yaml")
        args = base._parse_args(forwarded(attempt.name, window))
        command = build_arm_command(flash._build_command_hook, base, switches)(args, paths)
        argv = arm_command(command, ep, switches, reference_argv, generated, base, window)
        env = os.environ | switches | {
            "PYTHONPATH": os.pathsep.join((str(base.GIFT_ROOT), str(base.GIFT_ROOT / "contrib/flash-linear-attention"))),
            "WORLD_SIZE": str(WORLD), "LOCAL_WORLD_SIZE": str(WORLD), "RANK": "0", "LOCAL_RANK": "0",
        }
        log = attempt / "resolve.log"
        with log.open("x") as output:
            result = subprocess.run(
                [str(base.PYTHON), "-B", str(Path(__file__).resolve()), "resolve", *argv[argv.index("--") + 1:]],
                cwd=base.GIFT_ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
            )
        require(result.returncode == 0, f"Native config preflight failed; see {log}")
        config = paths["native"] / "config/resolved_config.yaml"
        assert_config(base, resolved, config, ep, paths["native"], window)
        write_json(attempt / "verified.json", {"argv": argv, "resolved_config_sha256": sha256(config)})
        print(f"CPU config preflight passed for EP{ep} {arm}", flush=True)


def run_arm(flash, base, attempt, ep, switches, original, source, reference_argv, generated, resolved, window):
    build_command = flash._build_command_hook
    snapshot = base.source_snapshot
    expectations = base._resolved_config_expectations
    verify_config = base._verify_resolved_config
    verify_runtime = base._assert_runtime_compatible
    preflight_runtime = base._runtime_preflight

    def wrap_command(runner):
        command = build_arm_command(build_command, runner, switches)

        def build(args, paths):
            argv = arm_command(command(args, paths), ep, switches, reference_argv, generated, base, window)
            write_json(attempt / "acceptance-launch.json", {"env": switches, "argv": argv, "cache": str(CACHE)})
            return argv

        return build

    def verify_snapshot(root):
        current = snapshot(root)
        require(current == source, "GIFT source differs from the preflight snapshot")
        return current

    def verify_resolved(paths, args):
        digest = verify_config(paths, args)
        assert_config(base, resolved, paths["native"] / "config/resolved_config.yaml", ep, paths["native"], window)
        return digest

    def verify_cache_runtime(producer, current, field="runtime"):
        if field == "runtime":
            producer = deepcopy(producer)
            package = producer["packages"]["torchtitan"]
            require(package["path"] == str(original / "gift/torchtitan/__init__.py"),
                    "Unexpected Original A cache producer imports")
            package["path"] = str(base.GIFT_ROOT / "torchtitan/__init__.py")
        verify_runtime(producer, current, field)

    def check_gpus(env, paths):
        report = base._wm_run([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader",
        ], env=env).stdout
        base._write_new_text(paths["preflight"] / "gpu-identity.csv", report)
        rows = list(csv.reader(report.splitlines(), skipinitialspace=True))
        require(len(rows) == WORLD and len({row[1] for row in rows if len(row) == 3}) == WORLD
                and all(row[2] == "0 MiB" for row in rows),
                "Expected eight distinct idle GPUs with 0 MiB used")
        return preflight_runtime(env, paths)

    base.source_snapshot = verify_snapshot
    base._verify_resolved_config = verify_resolved
    base._resolved_config_expectations = lambda args: expectations(args) | {"parallelism.expert_parallel_degree": ep}
    base._assert_runtime_compatible = verify_cache_runtime
    base._runtime_preflight = check_gpus
    flash._build_command_hook = wrap_command
    flash._load_base_runner = lambda: base
    # Native workers resolve the default raw directory before accepting a sibling run root.
    base.RAW_ROOT.mkdir(parents=True, exist_ok=True)
    return flash.main([
        "--gdn-backend", "fla", "--routing-metadata", "original", "--cross-entropy", "original",
        *forwarded(attempt.name, window),
    ])


def summarize(attempt, ep, arm, window, switches):
    steps = 2 * window
    completion = read_json(attempt / "completion.json")
    summary = read_json(attempt / "summary.json")
    require(completion["status"] == "PASS" and summary["status"] == "complete"
            and summary["world_size"] == WORLD and summary["training_steps"] == steps
            and summary["window"]["start_after_update"] == window
            and summary["window"]["end_after_update"] == steps
            and summary["tokens"]["physical_input_tokens_global"] == window * PHYSICAL_TOKENS_PER_UPDATE,
            f"Incomplete native {window}+{window} window in {attempt}")
    tps = summary["tokens"]["physical_input_tokens_per_second_global"]
    require(math.isfinite(tps) and tps > 0, f"Invalid measured TPS in {attempt}")
    memory = {}
    skip_sites = {}
    routed_sites = {}
    norm_pair_sites = {}
    swiglu_order_sites = {}
    swiglu_bwd_order_sites = {}
    dense_dy_sites = {}
    for rank in range(WORLD):
        receipt = read_json(attempt / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json")
        require(receipt["rank"] == rank and receipt["backend_variant"] == "fla"
                and receipt["routing_metadata"] == "original" and receipt["cross_entropy"] == "original"
                and receipt["capture_mode"] == "disabled" and receipt["native_main_completed"],
                f"Missing native FLA/routing/CE evidence for rank {rank}")
        (log,) = (attempt / "logs/torchrun").glob(f"*/attempt_0/{rank}/stdout.log")
        lines = ANSI.sub("", log.read_text())
        if SKIP in switches:
            skip_sites[rank] = [int(count) for count in SKIP_ACTIVATION.findall(lines)]
            require(any(skip_sites[rank]) == (switches[SKIP] == "1"),
                    f"Unexpected CuTe skip-gradient activation on rank {rank}")
        if ROUTED_OVERWRITE in switches:
            routed_sites[rank] = lines.count(ROUTED_ACTIVATION)
            require(routed_sites[rank] == int(switches[ROUTED_OVERWRITE]),
                    f"Unexpected routed dW activation count on rank {rank}")
        if NORM_PAIR in switches:
            norm_pair_sites[rank] = lines.count(NORM_PAIR_ACTIVATION)
            require(norm_pair_sites[rank] == int(switches[NORM_PAIR]),
                    f"Unexpected CuTe norm-pair activation count on rank {rank}")
        if SWIGLU_ORDER in switches:
            marker = f"Routed SwiGLU forward feature-first executed rank={rank} width=1536"
            swiglu_order_sites[rank] = lines.count(marker)
            require(swiglu_order_sites[rank] == int(switches[SWIGLU_ORDER]),
                    f"Unexpected routed SwiGLU forward activation count on rank {rank}")
            launch_marker = f"Routed SwiGLU launch rank={rank} grid="
            warmup_end = next((entry.start() for entry in STEP.finditer(lines)
                               if int(entry.group(1)) == window), -1)
            require(lines.count(launch_marker) == 1 and 0 <= lines.find(launch_marker) < warmup_end,
                    f"Routed SwiGLU launch audit did not finish during warmup on rank {rank}")
        if SWIGLU_BWD_ORDER in switches:
            enabled = switches[SWIGLU_BWD_ORDER] == "1"
            marker = f"Routed SwiGLU backward feature-first executed rank={rank} width=1536"
            swiglu_bwd_order_sites[rank] = lines.count(marker)
            launches = list(
                re.finditer(
                    rf"^Routed SwiGLU backward launch rank={rank} grid=\((\d+), (\d+), 1\)$",
                    lines, re.MULTILINE,
                )
            )
            warmup_end = next((entry.start() for entry in STEP.finditer(lines)
                               if int(entry.group(1)) == window), -1)
            require(
                swiglu_bwd_order_sites[rank] == int(enabled)
                and (not enabled or 0 <= lines.find(marker) < warmup_end)
                and len(launches) == 1 and launches[0].start() < warmup_end,
                f"Unexpected routed SwiGLU backward activation on rank {rank}",
            )
            x, y = (int(axis) for axis in launches[0].groups())
            require(
                (x == 12 and 0 < y <= 65535) if enabled else (x > 0 and y == 12),
                f"Unexpected routed SwiGLU backward launch grid on rank {rank}",
            )
        if DENSE_DY_QKV in switches:
            enabled = switches[DENSE_DY_QKV] == "1"
            marker = f"Dense QKV dY fusion executed rank={rank} grid=(64, 64, 1)"
            path = attempt / f"measurement/dense-dy-qkv-rank-{rank:02d}.json"
            audit = read_json(path)
            shapes = {key: audit[f"{key}_shape"] for key in DENSE_DY_SHAPES}
            count = sum(len(entry["sites"]) for entry in audit["wrappers"])
            require(audit["rank"] == rank and audit["switch"] == DENSE_DY_QKV
                    and audit["enabled"] is enabled and audit["candidate_sha256"] == sha256(DENSE_DY_PROBE)
                    and shapes == DENSE_DY_SHAPES and count == audit["compiled_sites"],
                    f"Dense QKV gradient-consumer audit differs on rank {rank}")
            if enabled:
                warmup_end = next((entry.start() for entry in STEP.finditer(lines)
                                   if int(entry.group(1)) == window), -1)
                require(lines.count(marker) == 1 and 0 <= lines.find(marker) < warmup_end
                        and audit["native_launch_grid"] == [64, 64, 1] and count > 0,
                        f"Dense QKV launch audit did not finish during warmup on rank {rank}")
            else:
                require(marker not in lines and audit["native_launch_grid"] is None and count == 0,
                        f"Unexpected dense QKV activation in control on rank {rank}")
            dense_dy_sites[rank] = count
        axes = MESH_AXES.findall(lines)
        require(MESH.findall(lines) == [str(ep)] and len(axes) == 1
                and ("'efsdp'" in axes[0]) == (ep == 4), f"Unexpected device mesh in {log}")
        samples = STEP.findall(lines)
        require(all(math.isfinite(float(value)) for _, loss, grad, _ in samples for value in (loss, grad)),
                f"Nonfinite native loss or gradient norm in {log}")
        entries = [(int(step), float(gib)) for step, _, _, gib in samples]
        require([step for step, _ in entries] == list(range(1, steps + 1)),
                f"Missing native memory updates in {log}")
        memory[rank] = entries[window:]
    gib, rank, update = max((gib, rank, step) for rank, entries in memory.items() for step, gib in entries)
    result = {
        "ep": ep, "efsdp": WORLD // ep, "arm": arm, "attempt": str(attempt),
        "window_global_tps": tps, "native_total_gpu_memory_gib": gib,
        "memory_max_rank": rank, "memory_max_update": update,
        "compiler_observer_status": summary["compilation_observer"]["status"],
    }
    require(result["compiler_observer_status"] == "observed_stable",
            f"Compiler activity remained in the measured window in {attempt}")
    if skip_sites:
        result["skip_rewrite_counts_by_rank"] = skip_sites
    if routed_sites:
        result["routed_overwrite_activation_counts_by_rank"] = routed_sites
    if norm_pair_sites:
        result["norm_pair_activation_counts_by_rank"] = norm_pair_sites
    if swiglu_order_sites:
        result["routed_swiglu_fwd_activation_counts_by_rank"] = swiglu_order_sites
    if swiglu_bwd_order_sites:
        result["routed_swiglu_bwd_activation_counts_by_rank"] = swiglu_bwd_order_sites
    if dense_dy_sites:
        result["dense_dy_qkv_compiled_sites_by_rank"] = dense_dy_sites
    write_json(attempt / "acceptance-summary.json", result | {"native_memory_by_rank": memory})
    return result


def add_deltas(row, reference, suffix=""):
    row[f"tps_change{suffix}_percent"] = 100 * (
        row["window_global_tps"] / reference["window_global_tps"] - 1
    )
    row[f"memory_change{suffix}_gib"] = (
        row["native_total_gpu_memory_gib"] - reference["native_total_gpu_memory_gib"]
    )


def run_campaign(campaign, progress, original, state, reference_argv, generated, resolved, window):
    for file in ("metadata.json", "completion.json", "process.json", "status.txt"):
        require(sha256(CACHE.parent / file) == sha256(original / CACHE_PRODUCER / file),
                f"Staged Original A cache producer differs at {file}")
    rows = {}
    for ep, arm, switches in comparison_order(state["arms"], state["candidate_first"]):
        attempt = RAW / f"{campaign.name}-ep{ep}-{arm}"
        progress.write(f"{ep}\t{arm}\tSTART\t{datetime.now(timezone.utc).isoformat()}\n")
        status = "FAIL"
        try:
            flash = load_flash()
            base = flash._load_base_runner()
            configure_inputs(base, original)
            code = run_arm(flash, base, attempt, ep, switches, original, state["source"],
                           reference_argv, generated, resolved, window)
            if code == 130:
                raise KeyboardInterrupt
            require(code == 0, f"Native runner exited {code}; see {attempt}")
            row = summarize(attempt, ep, arm, window, switches)
            rows[ep, arm] = row
            baseline = rows.get((ep, "baseline"))
            if baseline is not None:
                for name in state["arms"]:
                    if name != "baseline" and (ep, name) in rows:
                        add_deltas(rows[ep, name], baseline)
            if (ep, "all-three") in rows and (ep, "all-accepted") in rows:
                add_deltas(rows[ep, "all-accepted"], rows[ep, "all-three"], "_vs_original_three")
            print(json.dumps(row), flush=True)
            status = "PASS"
        finally:
            progress.write(f"{ep}\t{arm}\t{status}\t{datetime.now(timezone.utc).isoformat()}\n")
    write_json(
        campaign / "matrix-results.json",
        [rows[ep, arm] for ep, arm, _ in comparison_order(state["arms"])],
    )


def campaign_inputs(original, window, arms, candidate_first):
    reference = original / REFERENCE
    metadata = read_json(reference / "metadata.json")
    completion = read_json(reference / "completion.json")
    resolved_path = reference / "native/config/resolved_config.yaml"
    require(completion["status"] == "PASS" and metadata["source_head_before"] == REVISION
            and Path(metadata["task_root"]) == original
            and sha256(resolved_path) == completion["resolved_config_sha256"] == REFERENCE_HASH,
            "Original A reference differs from the verified run")
    flash = load_flash()
    base = flash._load_base_runner()
    configure_inputs(base, original)
    base._verify_ready_inputs(ROOT)
    hashes = base._config_hashes()
    require(all(metadata["config_hashes_before"][key] == value for key, value in hashes.items()),
            "A input, preset, or base runner hashes changed")
    frozen = base.source_snapshot(ROOT)
    require(frozen["head"] == REVISION and frozen["tracked_and_untracked_sha256"]
            != metadata["source_snapshot_before"]["tracked_and_untracked_sha256"],
            "Expected the SHK source patch on Original A")
    original_config, _ = native_args(metadata["argv"])
    require(sha256(original_config) == metadata["config_hashes_before"]["generated_experiment_sha256"],
            "Generated Original A experiment hash changed")
    generated = base._load_yaml(original_config)
    generated["imports"][0] = str(base.BASE_CONFIG)
    resolved = base._load_yaml(resolved_path)
    state = {"original_a_root": str(original), "source": frozen, "runner_sha256": sha256(Path(__file__)),
             "window_size": window, "arms": arms, "candidate_first": candidate_first}
    if any(ROUTED_OVERWRITE in switches for switches in arms.values()):
        build_root = os.environ.get(ROUTED_BUILD)
        require(build_root, "Routed dW overwrite needs a verified private extension build")
        build = Path(build_root).resolve(strict=True)
        manifest = build / "manifest.json"
        receipt = read_json(manifest)
        artifact = (build / receipt["artifact"]).resolve(strict=True)
        require(artifact.is_relative_to(build) and sha256(artifact) == receipt["artifact_sha256"],
                "Routed dW overwrite build artifact differs from its receipt")
        state["routed_overwrite_build"] = {
            "root": str(build), "manifest_sha256": sha256(manifest),
            "artifact_sha256": receipt["artifact_sha256"],
        }
    if any(NORM_PAIR in switches for switches in arms.values()):
        receipt = read_json(NORM_PAIR_RESULT)
        require(
            receipt["rewrites"] == {"live": 1, "stats_only": 1}
            and len(receipt["mismatched_bytes"]) == 9
            and not any(receipt["mismatched_bytes"].values())
            and receipt["candidate_sha256"] == sha256(NORM_PAIR_CANDIDATE)
            and receipt["runner_sha256"] == sha256(HERE / "followup/norm_pair_tile_compiled.py")
            and receipt["op_sha256"] == sha256(HERE / "followup/norm_pair_tile_ops.py"),
            "Norm-pair native run requires its matched compiled parity receipt",
        )
        state["norm_pair_sources"] = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (*NORM_PAIR_SOURCES, NORM_PAIR_RESULT)
        }
    if any(SWIGLU_ORDER in switches for switches in arms.values()):
        receipt = read_json(SWIGLU_ORDER_RESULT)
        require(
            receipt["bitwise"] and receipt["gate_up_shape"] == [63616, 3072]
            and receipt["grids"] == {"A": [994, 12], "B": [12, 994]}
            and all(receipt["profiles"][arm]["activation_verified"] for arm in ("A", "B"))
            and receipt["runner_sha256"] == sha256(SWIGLU_ORDER_PROBE)
            and receipt["production_file_sha256"] == sha256(SWIGLU_SOURCE),
            "Routed SwiGLU native run requires its matched producer parity receipt",
        )
        state["routed_swiglu_fwd_sources"] = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (SWIGLU_ORDER_WORKER, SWIGLU_ORDER_PROBE, SWIGLU_ORDER_RESULT)
        }
        if any(switches.get(NORM_PAIR) == "1" for switches in arms.values()):
            state["norm_pair_swiglu_worker_sha256"] = sha256(NORM_PAIR_SWIGLU_WORKER)
    if any(SWIGLU_BWD_ORDER in switches for switches in arms.values()):
        receipt = read_json(SWIGLU_BWD_ORDER_RESULT)
        require(
            receipt["bitwise"] and receipt["non_target_launches_matched"]
            and receipt["input_shape_stride_dtype"]["dy"][0] == [63616, 1536]
            and receipt["grids"] == {"A": [1988, 12], "B": [12, 1988]}
            and all(receipt["profiles"][arm]["activation_verified"] for arm in ("A", "B"))
            and receipt["probe_sha256"] == sha256(SWIGLU_BWD_ORDER_PROBE)
            and receipt["native_swiglu_source_sha256"] == sha256(SWIGLU_SOURCE),
            "Routed SwiGLU backward requires its matched producer and dgrad parity receipt",
        )
        state["routed_swiglu_bwd_sources"] = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                SWIGLU_BWD_ORDER_WORKER, SWIGLU_BWD_ORDER_PROBE, SWIGLU_BWD_ORDER_RESULT,
                SWIGLU_ORDER_WORKER, SWIGLU_ORDER_PROBE,
            )
        }
    if any(DENSE_DY_QKV in switches for switches in arms.values()):
        from importlib.metadata import version

        import torch
        import triton

        probe_digest = sha256(DENSE_DY_PROBE)
        for path in DENSE_DY_RESULTS:
            receipt = read_json(path)
            require(
                receipt["bitwise"]
                and receipt["bf16_input_shapes"] == DENSE_DY_SHAPES
                and receipt["candidate_sha256"] == probe_digest
                and receipt["gemm"] == {"out_dtype": "bfloat16", "use_split_accumulator": True}
                and receipt["torch_revision"] == torch.version.git_version
                and receipt["cuda"] == torch.version.cuda and receipt["triton"] == triton.__version__
                and set(receipt["cudacherry_extensions"]) == {"quantize", "gemm"}
                and all(sha256(Path(so["path"])) == so["sha256"]
                        for so in receipt["cudacherry_extensions"].values())
                and all(receipt["profiles"][arm]["activation_verified"] for arm in ("A", "B")),
                f"Dense QKV native run requires matching compiled parity in {path}",
            )
        state["dense_dy_qkv_sources"] = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (DENSE_DY_WORKER, DENSE_DY_PROBE, *DENSE_DY_RESULTS)
        }
        state["dense_dy_qkv_runtime"] = {
            "torch_revision": torch.version.git_version,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "triton_distribution": version("triton"),
        }
    return flash, base, state, (metadata["argv"], generated, resolved)


def main(argv):
    if argv[:1] == ["resolve"]:
        from torchtitan.config.cli import load_trainer_config
        from torchtitan.config.save import save_loaded_run_bundle
        from torchtitan.tools.logging import init_logger, logger

        init_logger()
        save_loaded_run_bundle(load_trainer_config(argv[1:], logger=logger), logger=logger)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("campaign_id")
    parser.add_argument("--original-a-root", type=Path, default=ORIGINAL_A)
    parser.add_argument("--window-size", type=int, choices=(10, 30), default=30,
                        help="optimizer updates in each of warmup and measurement (default: %(default)s)")
    parser.add_argument("--followup-switch", action="append", default=[], metavar="GIFT_SWITCH=VALUE",
                        help="enable a candidate switch (repeatable)")
    parser.add_argument("--six-fix-control", action="store_true",
                        help="compare routed SwiGLU forward against the six-fix control in a 10+10 run")
    parser.add_argument("--final-acceptance", action="store_true",
                        help="compare Original A, three fixes and all accepted fixes in a 10+10 run")
    parser.add_argument("--candidate-first", action="store_true",
                        help="run the candidate before the baseline on each topology")
    args = parser.parse_args(argv)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.campaign_id), "Invalid campaign ID")
    followups = {}
    for setting in args.followup_switch:
        switch, _, value = setting.partition("=")
        require(re.fullmatch(r"GIFT_[A-Z0-9_]+=[1-9][0-9]*", setting)
                and switch not in SWITCHES and switch not in followups,
                "Expected distinct GIFT candidate switches with positive integer values")
        if switch in (SKIP, ROUTED_OVERWRITE, NORM_PAIR, SWIGLU_ORDER, SWIGLU_BWD_ORDER, DENSE_DY_QKV):
            require(value == "1", f"{switch} uses a binary switch")
        followups[switch] = value
    if SWIGLU_BWD_ORDER in followups:
        require(
            followups == {SWIGLU_BWD_ORDER: "1"}
            and not args.six_fix_control and args.window_size == 10,
            "Routed SwiGLU backward requires its own 10+10 comparison against the three-fix control",
        )
    require(
        not args.six_fix_control or args.window_size == 10,
        "The six-fix control requires --window-size 10",
    )
    if args.final_acceptance:
        require(
            args.window_size == 10 and not args.six_fix_control and not followups,
            "Final acceptance requires 10+10 and uses fixed arms",
        )
        arms = final_arms()
    else:
        arms = comparison_arms(followups, six_fix_control=args.six_fix_control)
    original = args.original_a_root.resolve(strict=True)
    require(original != ROOT, "Original A artifacts must be outside the candidate worktree")
    campaign = RAW / args.campaign_id
    if args.mode == "preflight":
        flash, base, state, reference = campaign_inputs(
            original, args.window_size, arms, args.candidate_first
        )
        campaign.mkdir(parents=True, exist_ok=False)
        preflight(campaign, flash, base, *reference, args.window_size, state)
        require(base.source_snapshot(ROOT) == state["source"], "GIFT source changed during CPU config preflight")
        write_json(campaign / "preflight.json", state)
        return

    with (campaign / "matrix.tsv").open("x", buffering=1) as progress:
        progress.write("ep\tarm\tstatus\ttimestamp_utc\n")
        exit_code = 1
        try:
            _, _, state, reference = campaign_inputs(
                original, args.window_size, arms, args.candidate_first
            )
            require(read_json(campaign / "preflight.json") == state, "Preflight source, runner identity, or window differs")
            run_campaign(campaign, progress, original, state, *reference, args.window_size)
            exit_code = 0
        except KeyboardInterrupt:
            exit_code = 130
            raise
        finally:
            with (campaign / "exit-code.txt").open("x") as status:
                status.write(f"{exit_code}\n")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError) as error:
        print(f"acceptance: {error}", file=sys.stderr)
        raise SystemExit(1) from error
