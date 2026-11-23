#!/usr/bin/env python3
"""Run selected native EP arms for the dedicated row-owner communicator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import campaign


SCRIPT = Path(__file__).resolve()
ARMS = ("a", "shared", "narrow-4", "narrow-8", "combined")
NARROW_MARKER = "row_owner_embedding dedicated NCCL communicator active max_ctas="
COMPACT_MARKER = (
    "row_owner_embedding compact source coalescing active fqn=tok_embeddings.weight"
)
MARKER_TAG = "dense-gfsdp-narrow-activation "
SOURCE_TAG = "dense-gfsdp-narrow-source "
PARALLELIZE = "torchtitan.models.gift_moe._base.parallelize"
ROW_OWNER = "torchtitan.distributed.gfsdp.row_owner_embedding"
MEMORY = re.compile(r"\bmemory:\s*(\d+\.\d+)GiB\s+non_torch:\s*(-?\d+\.\d+)GiB\b")


@dataclass(frozen=True)
class Arm:
    root: Path
    source_sha256: str
    max_ctas: int | None = None
    compact: bool = False


def module_path(root, module):
    return root / "gift" / (module.replace(".", "/") + ".py")


def source_records(root, *, row_owner_required):
    records = {}
    for name in (PARALLELIZE, ROW_OWNER):
        loaded = sys.modules.get(name)
        if loaded is None and name == ROW_OWNER and not row_owner_required:
            records[name] = None
            continue
        expected = module_path(root, name).resolve(strict=True)
        actual = (
            Path(loaded.__file__).resolve(strict=True) if loaded is not None else None
        )
        campaign.require(
            actual == expected,
            f"Native worker imported {name} from {actual}, expected {expected}",
        )
        records[name] = {"path": str(actual), "sha256": campaign.digest(actual)}
    return records


def worker(argv):
    # Keep the exact native imports and compiler startup order. Read module
    # provenance after the measured window from the modules GIFT really used.
    bridge = Path(argv[0]).resolve(strict=True)
    campaign.require(
        bridge.relative_to(bridge.parents[2])
        == campaign.BRIDGE.with_name("measure.py"),
        f"Unexpected native bridge path: {bridge}",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(MARKER_TAG + "%(message)s"))
    handler.addFilter(
        lambda record: record.getMessage().startswith((NARROW_MARKER, COMPACT_MARKER))
    )
    logger = logging.getLogger("gfsdp")
    logger.addHandler(handler)
    try:
        campaign.worker(argv)
    finally:
        logger.removeHandler(handler)
    rank = int(os.environ["RANK"])
    row = {
        "rank": rank,
        "modules": source_records(bridge.parents[2], row_owner_required=False),
        "driver_sha256": campaign.digest(SCRIPT),
    }
    print(SOURCE_TAG + json.dumps(row, sort_keys=True), flush=True)


def read_extra_rank(attempt, rank, root, arm, *, row_owner_required, console_rows):
    paths = list((attempt / "logs/torchrun").glob(f"*/attempt_0/{rank}/stdout.log"))
    campaign.require(len(paths) == 1, f"Rank {rank} lacks its unique native stdout")
    provenance, narrow, compact, memory = [], [], [], []
    for line in campaign.ANSI.sub("", paths[0].read_text()).splitlines():
        if line.startswith(SOURCE_TAG):
            provenance.append(json.loads(line[len(SOURCE_TAG) :]))
        if line.startswith(MARKER_TAG + NARROW_MARKER):
            narrow.append((len(memory), line[len(MARKER_TAG) :]))
        if line.startswith(MARKER_TAG + COMPACT_MARKER):
            compact.append((len(memory), line[len(MARKER_TAG) :]))
        if step := campaign.STEP.search(line):
            match = MEMORY.search(line)
            campaign.require(
                match is not None,
                f"Rank {rank} update {step[1]} lacks the same-line non-Torch sample",
            )
            memory.append(
                {
                    "update": int(step[1]),
                    "total_gift_memory_gib": float(match[1]),
                    "non_torch_gib": float(match[2]),
                }
            )
    campaign.require(
        [(row["update"], row["total_gift_memory_gib"]) for row in memory]
        == [(row[0], row[3]) for row in console_rows],
        f"Rank {rank} memory samples did not match the native update lines",
    )
    campaign.require(
        len(narrow) == int(arm.max_ctas is not None),
        f"Rank {rank} narrow activation count differs",
    )
    if narrow:
        before_step, line = narrow[0]
        prefix = NARROW_MARKER + str(arm.max_ctas)
        campaign.require(
            before_step == 0 and (line == prefix or line.startswith(prefix + " ")),
            f"Rank {rank} has the wrong cap or late activation: {line}",
        )
    campaign.require(
        len(compact) == int(arm.compact) and all(i == 0 for i, _ in compact),
        f"Rank {rank} compact activation differs",
    )
    campaign.require(
        len(provenance) == 1, f"Rank {rank} lacks one imported-module receipt"
    )
    (source,) = provenance
    campaign.require(
        source["rank"] == rank and source["driver_sha256"] == campaign.digest(SCRIPT),
        f"Rank {rank} ran a different diagnostic worker",
    )
    for name in (PARALLELIZE, ROW_OWNER):
        record = source["modules"][name]
        if record is None and name == ROW_OWNER and not row_owner_required:
            continue
        expected = module_path(root, name).resolve(strict=True)
        campaign.check_equal(
            {"path": str(expected), "sha256": campaign.digest(expected)},
            record,
            f"Rank {rank} loaded {name}",
        )
    measured = memory[campaign.WARMUP : campaign.STEPS]
    return {
        "narrow_marker": narrow[0][1] if narrow else None,
        "compact_marker": compact[0][1] if compact else None,
        "imported_modules": source["modules"],
        "memory_samples": memory,
        "peak_total_sample": max(
            measured, key=lambda row: (row["total_gift_memory_gib"], row["update"])
        ),
        "peak_non_torch_sample": max(
            measured, key=lambda row: (row["non_torch_gib"], row["update"])
        ),
    }


def extra_evidence(case, arm, result):
    rows = result["per_rank_console"]
    attempt = Path(result["attempt"])
    ranks = {
        rank: read_extra_rank(
            attempt,
            rank,
            case.root,
            arm,
            row_owner_required=case.features[campaign.ROW_OWNER],
            console_rows=rows[rank],
        )
        for rank in range(campaign.WORLD)
    }
    # Campaign's total-peak rank tie-break is (total, rank, update).
    peak_total = ranks[result["memory_max_rank"]]["peak_total_sample"]
    campaign.require(
        peak_total["update"] == result["memory_max_update"]
        and peak_total["total_gift_memory_gib"] == result["total_gift_memory_gib"],
        "Native peak total and its non-Torch sample do not refer to the same line",
    )
    external, rank, update = max(
        (
            row["peak_non_torch_sample"]["non_torch_gib"],
            rank,
            row["peak_non_torch_sample"]["update"],
        )
        for rank, row in ranks.items()
    )
    return {
        "per_rank_native_receipts": ranks,
        "non_torch_gib_at_total_peak": peak_total["non_torch_gib"],
        "observed_non_torch_max_gib": external,
        "observed_non_torch_max_rank": rank,
        "observed_non_torch_max_update": update,
        "memory_scope": "Console updates 11–20: total is peak torch reserved plus non-Torch sampled on that line",
    }


def selection(args):
    campaign.require(
        len(args.arm) == len(set(args.arm)), "Select each native arm at most once"
    )
    choices = {
        "a": (
            args.original_root,
            "",
        ),  # Original A is pinned by the campaign reference.
        "shared": (args.candidate_root, args.candidate_source_sha256),
        "narrow-4": (args.narrow4_root, args.narrow4_source_sha256, 4),
        "narrow-8": (args.narrow8_root, args.narrow8_source_sha256, 8),
        "combined": (
            args.combined_root,
            args.combined_source_sha256,
            args.combined_max_ctas,
            True,
        ),
    }
    selected = {}
    for name in args.arm:
        root, source_sha256, *options = choices[name]
        campaign.require(
            root is not None and (name == "a" or source_sha256),
            f"Supply a frozen root and source hash for {name}",
        )
        if name == "combined":
            campaign.require(
                args.combined_max_ctas is not None
                and args.compact_exact_receipt is not None,
                "Combined requires its dedicated cap and the completed compact exact receipt",
            )
            exact = campaign.read_json(args.compact_exact_receipt)
            campaign.require(
                exact["exact"] is True
                and not exact["mismatches"]
                and [row["rank"] for row in exact["ranks"]]
                == list(range(campaign.WORLD))
                and all(
                    row["case_matches"] == {"real-high-u": True, "real-low-u": True}
                    for row in exact["ranks"]
                )
                and bool(exact["low_fixture_padded_ranks"]),
                "Combined requires a successful eight-rank compact exact replay",
            )
        selected[name] = Arm(root.resolve(), source_sha256, *options)
    roots = {args.original_root.resolve(), args.candidate_root.resolve()}
    roots.update(
        arm.root for name, arm in selected.items() if name not in ("a", "shared")
    )
    campaign.require(
        len(roots) == 2 + sum(name not in ("a", "shared") for name in selected),
        "Use distinct isolated training roots for Original A, shared, and each selected candidate",
    )
    return selected


def native_case(args, name, arm):
    return campaign.NativeCase(
        SimpleNamespace(
            original_root=args.original_root,
            candidate_root=arm.root,
            reference=args.reference,
            candidate_head=args.candidate_head,
            candidate_source_sha256=arm.source_sha256,
        ),
        args.ep,
        "a" if name == "a" else "all",
    )


def comparisons(results):
    pairs = [
        ("a", "shared"),
        ("shared", "narrow-4"),
        ("shared", "narrow-8"),
        ("narrow-4", "narrow-8"),
    ]
    if combined := results.get("combined"):
        pairs.extend(
            (("shared", "combined"), (f"narrow-{combined['max_ctas']}", "combined"))
        )
    deltas = []
    for control, candidate in pairs:
        left, right = results.get(control), results.get(candidate)
        if (
            left is None
            or right is None
            or left["status"] != "pass"
            or right["status"] != "pass"
        ):
            continue
        campaign.check_equal(
            left["input_hashes"],
            right["input_hashes"],
            "Native comparison input identity",
        )
        deltas.append(
            {
                "control": control,
                "candidate": candidate,
                "window_global_tps_percent": 100
                * (right["window_global_tps"] / left["window_global_tps"] - 1),
                "total_gift_memory_delta_gib": right["total_gift_memory_gib"]
                - left["total_gift_memory_gib"],
                "non_torch_at_total_peak_delta_gib": (
                    right["non_torch_gib_at_total_peak"]
                    - left["non_torch_gib_at_total_peak"]
                ),
                "observed_non_torch_max_delta_gib": (
                    right["observed_non_torch_max_gib"]
                    - left["observed_non_torch_max_gib"]
                ),
            }
        )
    return deltas


def validated_case(args, name, arm):
    case = native_case(args, name, arm)
    reference = case.base._load_yaml(case.reference_cfg)
    campaign.require(
        reference["compile"]["enable"] is True
        and reference["compile"]["backend"] == "inductor"
        and reference["compile"]["components"] == ["model", "loss"]
        and reference["debug"]["seed"] == 2048
        and reference["parallelism"]["gfsdp_match_fsdp2_init"] is True,
        "Reference is missing the pinned native compiler or initialization settings",
    )
    case.preflight(args.timeout_seconds)
    print(
        f"PASS EP{args.ep} {name} frozen source, native CLI and real inputs", flush=True
    )
    return case


def matrix(args):
    selected = selection(args)
    output = args.output.resolve()
    campaign.require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output.name), "Invalid output name"
    )
    if args.preflight_only:
        for name, arm in selected.items():
            validated_case(args, name, arm)
        return 0

    output.mkdir()
    with (output / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.write_json(
            output / "plan.json",
            {
                "ep": args.ep,
                "order": list(selected),
                "warmup_updates": campaign.WARMUP,
                "updates": campaign.STEPS,
                "initialization_scope": "Fresh native initialization under the same seed and resolved init settings",
                "driver_sha256": campaign.digest(SCRIPT),
                "campaign_sha256": campaign.digest(Path(campaign.__file__)),
                "reference": str(args.reference.resolve()),
                "reference_metadata_sha256": campaign.digest(
                    args.reference / "metadata.json"
                ),
                "compact_exact_receipt_sha256": (
                    campaign.digest(args.compact_exact_receipt)
                    if "combined" in selected
                    else None
                ),
                "arms": {
                    name: {
                        "root": str(arm.root),
                        "source_sha256": arm.source_sha256 or None,
                        "max_ctas": arm.max_ctas,
                        "compact": arm.compact,
                    }
                    for name, arm in selected.items()
                },
            },
        )
        results = {}
        for name, arm in selected.items():
            run_id = f"{output.name}-ep{args.ep}-{name}-{uuid4().hex[:8]}"
            result = {"native_arm": name, "run_id": run_id, "max_ctas": arm.max_ctas}
            try:
                case = validated_case(args, name, arm)
                attempt = case.raw / run_id
                result["attempt"] = str(attempt)
                campaign.check_gpus(output / f"{name}.gpus.csv")
                print(f"START {attempt}", flush=True)
                with patch.object(campaign, "SCRIPT", SCRIPT):
                    status = case.launch(run_id, args.timeout_seconds)
                result["training_exit_code"] = status
                campaign.require(status == 0, f"Native training exited {status}")
                result.update(case.summarize(attempt))
                result.update(extra_evidence(case, arm, result))
                result["compiler_cache_policy"] = campaign.read_json(
                    attempt / "metadata.json"
                )["compiler_cache_policy"]
                campaign.require(
                    result["compiler_cache_policy"] == "fresh",
                    "Native run did not start with a fresh compiler cache",
                )
                campaign.require(
                    result["compilation_status"] == "observed_stable",
                    "Native compiler evidence did not establish a stable measured window",
                )
                result["status"] = "pass"
            except (
                KeyError,
                OSError,
                RuntimeError,
                ValueError,
                subprocess.CalledProcessError,
            ) as error:
                result.update(status="fail", error=f"{type(error).__name__}: {error}")
            results[name] = result
            campaign.write_json(output / f"{name}.json", result)
            campaign.write_json(
                output / "results.json",
                {
                    "ep": args.ep,
                    "measured_updates": [campaign.WARMUP + 1, campaign.STEPS],
                    "arms": results,
                    "comparisons": comparisons(results),
                    "pending": [key for key in selected if key not in results],
                },
            )
            print(
                f"{result['status'].upper()} {name} {output / f'{name}.json'}",
                flush=True,
            )
    return int(any(result["status"] != "pass" for result in results.values()))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["worker"]:
        return worker(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new native evidence directory")
    parser.add_argument("--ep", type=int, choices=(4, 8), required=True)
    parser.add_argument(
        "--arm",
        choices=ARMS,
        action="append",
        required=True,
        help="ordered and repeatable",
    )
    parser.add_argument("--narrow4-root", type=Path)
    parser.add_argument("--narrow4-source-sha256")
    parser.add_argument("--narrow8-root", type=Path)
    parser.add_argument("--narrow8-source-sha256")
    parser.add_argument("--combined-root", type=Path)
    parser.add_argument("--combined-source-sha256")
    parser.add_argument("--combined-max-ctas", choices=(4, 8), type=int)
    parser.add_argument("--compact-exact-receipt", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    campaign.add_native_args(parser)
    return matrix(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1) from error
