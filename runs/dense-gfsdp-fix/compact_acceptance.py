#!/usr/bin/env python3
"""Run the compact source-coalescing native comparison after exact replay."""

import argparse
import fcntl
import logging
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import campaign


SCRIPT = Path(__file__).resolve()
COMPACT_MARKER = "row_owner_embedding compact source coalescing active fqn=tok_embeddings.weight"
LOG_TAG = "dense-gfsdp-compact "


def worker(argv):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_TAG + "%(message)s"))
    handler.addFilter(lambda row: row.getMessage().startswith(COMPACT_MARKER))
    logger = logging.getLogger("gfsdp")
    logger.addHandler(handler)
    try:
        campaign.worker(argv)
    finally:
        logger.removeHandler(handler)


def activation(attempt, expected):
    markers = {}
    for rank in range(campaign.WORLD):
        (log,) = (attempt / "logs/torchrun").glob(f"*/attempt_0/{rank}/stdout.log")
        text = campaign.ANSI.sub("", log.read_text())
        needle = LOG_TAG + COMPACT_MARKER
        hits = [line.strip() for line in text.splitlines() if needle in line]
        campaign.require(len(hits) == int(expected), f"Compact path activation differs on rank {rank}")
        if hits:
            campaign.require(text.index(needle) < campaign.STEP.search(text).start(),
                             f"Compact path first activated after update 1 on rank {rank}")
        markers[str(rank)] = hits[0] if hits else None
    return markers


def native_case(args, *, compact):
    return campaign.NativeCase(SimpleNamespace(
        original_root=args.original_root,
        candidate_root=args.compact_root if compact else args.candidate_root,
        reference=args.reference,
        candidate_head=args.candidate_head,
        candidate_source_sha256=(args.compact_source_sha256 if compact else args.candidate_source_sha256),
    ), args.ep, "all")


def matrix(args):
    output = args.output.resolve()
    campaign.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output.name), "Invalid output name")
    campaign.require(args.candidate_root.resolve() != args.compact_root.resolve(), "Use isolated checkouts")
    controls = [("all-four", native_case(args, compact=False)), ("compact", native_case(args, compact=True))]
    for _, case in controls:
        case.preflight(args.timeout_seconds)
    if args.preflight_only:
        return 0

    output.mkdir()
    with (output / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        for name, case in controls:
            run_id = f"{output.name}-ep{args.ep}-{name}-{uuid4().hex[:8]}"
            campaign.check_gpus(output / f"{name}.gpus.csv")
            print(f"START {case.raw / run_id}", flush=True)
            with patch.object(campaign, "SCRIPT", SCRIPT):
                status = case.launch(run_id, args.timeout_seconds)
            campaign.require(status == 0, f"{name} native training exit code {status}")
            attempt = case.raw / run_id
            result = case.summarize(attempt)
            result["compact_coalescing_activation_by_rank"] = activation(attempt, name == "compact")
            campaign.require(result["compilation_status"] == "observed_stable", f"{name} compiler is unstable")
            results[name] = result
            campaign.write_json(output / f"{name}.json", result)
        control, candidate = (results[name] for name, _ in controls)
        campaign.check_equal(control["input_hashes"], candidate["input_hashes"], "Input identity")
        campaign.write_json(output / "results.json", {
            "measured_updates": [campaign.WARMUP + 1, campaign.STEPS],
            "arms": results,
            "window_tps_percent": 100 * (candidate["window_global_tps"] / control["window_global_tps"] - 1),
            "total_gift_memory_delta_gib": candidate["total_gift_memory_gib"] - control["total_gift_memory_gib"],
        })
    print(f"PASS {output / 'results.json'}", flush=True)
    return 0


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["worker"]:
        return worker(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ep", type=int, choices=(4, 8), required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--compact-source-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    campaign.add_native_args(parser)
    return matrix(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
