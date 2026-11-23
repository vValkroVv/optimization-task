#!/usr/bin/env python3
"""Compare HybridEP routing and capacity on the clean original GIFT training recipe."""

import argparse
import hashlib
import importlib.util
import json
import re
import runpy
import subprocess
import sys
from itertools import batched
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER = ROOT / "runs/flash-qla/run.py"
MEASURE = RUNNER.with_name("measure.py")
RAW_RUNS = ROOT / "runs/production-data-abcd/artifacts/raw"
REFERENCE = RAW_RUNS / "original-a-vs-cute-140-001-a"
CACHE = RAW_RUNS / "original-a-vs-cute-001-a2/cache"
ARMS = {
    "baseline": (False, False),
    "routing": (True, False),
    "selector": (False, True),
    "both": (True, True),
}
SELECTORS = tuple(
    f"model.config.blocks.{block}.moe.experts.token_dispatcher.token_drop.branchless_capacity_drop"
    for block in ("gdn_moe", "mla_moe")
)
REVISION = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
ROUTING_SHA256 = "01c0ccc7ebb7bf3c4616b7a22cd5c9c1e20e05acdd0cb2821d08f002576ef402"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRIC = re.compile(
    r"\bstep:\s*(\d+)\s+loss:\s*([\d.]+)\s+grad_norm:\s*([\d.]+)\s+"
    r"memory:\s*([\d.]+)\s*GiB.*?tps:\s*([\d,]+)"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("_hybridep_original_a", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    with path.open("x") as output:
        json.dump(data, output, indent=2, sort_keys=True)
        output.write("\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_path(output, ep, arm):
    tag = f"{output.parent.name}-dry" if output.name == "dry" else output.name
    return RAW_RUNS / f"{tag}-ep{ep}-{arm}"


def verify_candidate(file):
    if sha256(file) != ROUTING_SHA256:
        raise RuntimeError(
            f"Routing candidate differs from the GPU-verified source: {file}"
        )


def rank_main(candidate, native_args):
    from torchtitan.experiments import four_fusion

    file = Path(candidate)
    verify_candidate(file)
    four_fusion.__path__.insert(0, str(file.parent))
    sys.argv = [str(MEASURE), *native_args]
    runpy.run_path(str(MEASURE), run_name="__main__")
    return 0


def native_config_and_overrides(command):
    start = command.index("--") + 1
    config_file = Path(command[start])
    pairs = list(batched(command[start + 1 :], 2))
    overrides = dict(pairs)
    if len(pairs) != len(overrides):
        raise RuntimeError("Native command repeats config flags")
    overrides.pop("--dump_folder")
    return config_file, overrides


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


def run_arm(output, ep, arm, *, dry_run=False):
    routing, selector = ARMS[arm]
    attempt = case_path(output, ep, arm)
    candidate = output / "candidate/routing_metadata.py"
    verify_candidate(candidate)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty_gift = subprocess.check_output(
        ["git", "status", "--porcelain", "-uall", "--", "gift/"], cwd=ROOT
    )
    if head != REVISION or dirty_gift:
        raise RuntimeError(
            f"Expected clean GIFT at the original A revision, found {head}"
        )

    reference = read_json(REFERENCE / "metadata.json")
    reference_done = read_json(REFERENCE / "completion.json")
    config_path = REFERENCE / "native/config/resolved_config.yaml"
    if (
        reference_done["status"] != "PASS"
        or reference["source_head_before"] != REVISION
        or sha256(config_path) != reference_done["resolved_config_sha256"]
    ):
        raise RuntimeError("Original A reference is not the verified native run")

    changes = {f"--{flag}": "true" for flag in SELECTORS} if selector else {}
    if ep != 8:
        changes["--parallelism.expert_parallel_degree"] = str(ep)
    _, expected_overrides = native_config_and_overrides(reference["argv"])
    expected_overrides["--training.steps"] = "60"
    expected_overrides.update(changes)
    config_hashes = reference["config_hashes_before"]
    flash = load_runner()
    build_native_command = flash._build_command_hook
    load_base_runner = flash._load_base_runner

    def build_command(base):
        native_command = build_native_command(base)

        def command(args, paths):
            argv = native_command(args, paths)
            if routing:
                index = argv.index(str(MEASURE))
                argv[index : index + 1] = [
                    str(Path(__file__).resolve()),
                    "rank",
                    str(candidate),
                ]
            for flag, value in changes.items():
                argv.extend((flag, value))
            config, overrides = native_config_and_overrides(argv)
            if overrides != expected_overrides:
                differing = sorted(
                    key
                    for key in overrides.keys() | expected_overrides.keys()
                    if overrides.get(key) != expected_overrides.get(key)
                )
                raise RuntimeError(f"Native CLI differs from original A at {differing}")
            if sha256(config) != config_hashes["generated_experiment_sha256"]:
                raise RuntimeError("Generated experiment differs from original A")
            write_json(
                paths["attempt"] / "acceptance-launch.json",
                {
                    "original_a_revision": REVISION,
                    "original_a_reference": str(REFERENCE),
                    "ep": ep,
                    "efsdp": 8 // ep,
                    "routing_sha256": ROUTING_SHA256 if routing else None,
                    "gpu_capacity_selector": selector,
                    "argv": argv,
                    "cache": str(CACHE),
                },
            )
            return argv

        return command

    def load_base():
        base = load_base_runner()
        source_snapshot = base.source_snapshot
        expected_source = reference["source_snapshot_before"][
            "tracked_and_untracked_sha256"
        ]
        verify_config = base._verify_resolved_config
        expect_config = base._resolved_config_expectations
        expected_config = base._load_yaml(config_path)
        expected_config.pop("dump_folder")
        expected_config["training"]["steps"] = 60
        expected_config["parallelism"]["expert_parallel_degree"] = ep
        for flag in SELECTORS:
            parent, attribute = flag.rsplit(".", 1)
            policy = expected_config
            for key in parent.split("."):
                policy = policy[key]
            policy[attribute] = selector

        def verify_snapshot(root):
            snapshot = source_snapshot(root)
            if (
                snapshot["head"] != REVISION
                or snapshot["tracked_and_untracked_sha256"] != expected_source
            ):
                raise RuntimeError("GIFT source manifest differs from original A")
            return snapshot

        def config_expectations(args):
            return expect_config(args) | {
                "parallelism.expert_parallel_degree": ep,
                **dict.fromkeys(SELECTORS, selector),
            }

        def verify_resolved(paths, args):
            digest = verify_config(paths, args)
            resolved = base._load_yaml(paths["native"] / "config/resolved_config.yaml")
            resolved.pop("dump_folder")
            differing = list(differing_config_keys(expected_config, resolved))
            if differing:
                raise RuntimeError(
                    f"Resolved config differs from original A at {differing}"
                )
            return digest

        base.source_snapshot = verify_snapshot
        base._resolved_config_expectations = config_expectations
        base._verify_resolved_config = verify_resolved
        if dry_run:

            def after_preflight(*args):
                raise RuntimeError("ACCEPTANCE_DRY_RUN_VERIFIED")

            base._launch = after_preflight
        return base

    flash._build_command_hook = build_command
    flash._load_base_runner = load_base
    result = flash.main(
        [
            "--gdn-backend",
            "fla",
            "--routing-metadata",
            "fused" if routing else "original",
            "--cross-entropy",
            "original",
            "--num-prefetch-batches",
            "0",
            "--epoch-size",
            "102400",
            "--run-id",
            attempt.name,
            "--output-root",
            str(RAW_RUNS),
            "--compiler-cache-root",
            str(CACHE),
            "--steps",
            "60",
            "--warmup",
            "30",
            "--window",
            "30",
            "--observe-compiler-cache",
            "--timeout-seconds",
            "7200",
        ]
    )
    launch_receipt = attempt / "acceptance-launch.json"
    if dry_run and launch_receipt.is_file():
        status = read_json(attempt / "failure.json")
        if status["error"] == "RuntimeError: ACCEPTANCE_DRY_RUN_VERIFIED":
            write_json(
                output / f"ep{ep}-{arm}.verified.json",
                {
                    "attempt": str(attempt),
                    "acceptance_launch_sha256": sha256(launch_receipt),
                },
            )
            return 0
    return result


def summarize_arm(output, ep, arm):
    routing, selector = ARMS[arm]
    path = case_path(output, ep, arm)
    completion = read_json(path / "completion.json")
    if completion["status"] != "PASS":
        raise RuntimeError(f"Incomplete arm: {path}")
    summary = read_json(path / "summary.json")
    if (
        summary["status"] != "complete"
        or summary["window"]["start_after_update"] != 30
        or summary["window"]["end_after_update"] != 60
    ):
        raise RuntimeError("Missing complete 30+30 window")
    logs = {}
    expected_routing = "fused" if routing else "original"
    expected_file = (
        output / "candidate/routing_metadata.py"
        if routing
        else ROOT / "gift/torchtitan/distributed/deepep/routing.py"
    )
    for rank in range(8):
        receipt = read_json(
            path / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json"
        )
        if (
            receipt["routing_metadata"] != expected_routing
            or receipt["routing_metadata_callable"]["sha256"] != sha256(expected_file)
            or not receipt["native_main_completed"]
        ):
            raise RuntimeError(f"Unverified native routing at rank {rank}")
        (stdout,) = (path / "logs/torchrun").glob(f"*/attempt_0/{rank}/stdout.log")
        entries = [
            m.groups() for m in METRIC.finditer(ANSI.sub("", stdout.read_text()))
        ]
        if [int(row[0]) for row in entries] != list(range(1, 61)):
            raise RuntimeError(f"Missing native update log at rank {rank}")
        logs[rank] = [
            [int(step), float(loss), float(grad), float(mem), int(tps.replace(",", ""))]
            for step, loss, grad, mem, tps in entries
        ]
    memory, step, rank = max(
        (row[3], row[0], rank) for rank, rows in logs.items() for row in rows[30:]
    )
    result = {
        "ep": ep,
        "efsdp": 8 // ep,
        "arm": arm,
        "routing": routing,
        "selector": selector,
        "window_global_tps": summary["tokens"][
            "physical_input_tokens_per_second_global"
        ],
        "median_step_global_tps": median(
            sum(logs[rank][step][4] for rank in logs) for step in range(30, 60)
        ),
        "total_gpu_memory_gib": memory,
        "memory_max_rank": rank,
        "memory_max_update": step,
        "console_fields": [
            "step",
            "loss",
            "grad_norm",
            "total_gpu_memory_gib",
            "local_tps",
        ],
        "per_rank_console": logs,
        "compiler_observer": summary.get("compilation_observer"),
    }
    write_json(path / "acceptance-summary.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"per_rank_console", "compiler_observer"}
            }
        ),
        flush=True,
    )


def main():
    if sys.argv[1:2] == ["rank"]:
        return rank_main(sys.argv[2], sys.argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "run", "summarize"))
    parser.add_argument("output", type=Path)
    parser.add_argument("ep", type=int, choices=(4, 8))
    parser.add_argument("arm", choices=ARMS)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    if not output.is_relative_to(HERE / "artifacts/raw"):
        parser.error("output must be inside the HybridEP task evidence directory")
    if args.mode == "summarize":
        summarize_arm(output, args.ep, args.arm)
        return 0
    return run_arm(output, args.ep, args.arm, dry_run=args.mode == "dry-run")


if __name__ == "__main__":
    raise SystemExit(main())
