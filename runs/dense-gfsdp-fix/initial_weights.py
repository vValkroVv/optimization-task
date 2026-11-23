#!/usr/bin/env python3
"""Compare initialized weights for native EP8 Original A and layer-0 FFN split."""

import argparse
import fcntl
import hashlib
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
ARMS = ("a", "layer0")
LAYER0_PREFIX = "layers.0."
COPY_BYTES = 16 * 1024 * 1024


def weight_sha256(tensor):
    import torch

    # Read the stored representation, including signed zeros and NaN payloads.
    flat = tensor.detach().reshape(-1).contiguous().view(torch.uint8)
    digest = hashlib.sha256()
    for block in flat.split(COPY_BYTES):
        digest.update(memoryview(block.cpu().numpy()))
    return digest.hexdigest()


def placement_record(placement):
    kind = type(placement).__name__
    # Ragged offsets are relative to the partition selected by the other axes.
    fields = {
        "Replicate": (),
        "Shard": ("dim",),
        "_StridedShard": ("dim", "split_factor"),
        "RaggedShard": ("global_offset", "local_numel", "shard_numels"),
    }
    campaign.require(kind in fields, f"Unsupported parameter placement: {placement}")
    return {"kind": kind} | {field: getattr(placement, field) for field in fields[kind]}


def parameter_record(parameter, *, logical):
    import torch

    campaign.require(not parameter.is_meta, "Parameter is still on the meta device")
    record = {"dtype": str(parameter.dtype), "shape": list(parameter.shape)}
    to_local = getattr(parameter, "to_local", None)
    if logical:
        campaign.require(parameter.dtype == torch.float32, "Layer 0 master is not FP32")
        value = parameter.full_tensor() if callable(to_local) else parameter
        campaign.require(value.shape == parameter.shape, "Layer 0 reconstruction shape changed")
        record["scope"] = "logical"
    elif callable(to_local):
        mesh = parameter.device_mesh
        coordinate = mesh.get_coordinate()
        campaign.require(coordinate is not None, "Rank is outside the parameter mesh")
        value = to_local()
        record.update(
            scope="local",
            local_shape=list(value.shape),
            mesh_ranks=mesh.mesh.tolist(),
            mesh_coordinate=list(coordinate),
            placements=[placement_record(placement) for placement in parameter.placements],
        )
    else:
        value = parameter
        record["scope"] = "logical"
    record["sha256"] = weight_sha256(value)
    return record


def parameter_fingerprints(model_parts):
    import torch

    (model,) = model_parts
    named = sorted(
        (name.replace("_checkpoint_wrapped_module.", ""), parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
    )
    names = [name for name, _ in named]
    campaign.require(len(names) == len(set(names)), "Duplicate canonical parameter FQN")
    campaign.require(
        all(f"layers.0.feed_forward.{name}.weight" in names for name in ("w1", "w2", "w3")),
        "Native layer 0 FFN parameters are missing",
    )
    with torch.no_grad():
        return {
            name: parameter_record(parameter, logical=name.startswith(LAYER0_PREFIX))
            for name, parameter in named
        }


def worker(argv):
    def run_bridge(path, *, run_name):
        campaign.require(run_name == "__main__", "Unexpected native worker entry point")
        bridge = campaign.load_module(Path(path), "_initial_weights_flash_bridge")
        base = bridge._load_base_measure()
        enter = base.NativeAdapter.__enter__

        def install(adapter):
            enter(adapter)
            train = adapter.trainer_class.train
            output = Path(adapter.options["output_dir"])
            rank = int(os.environ["RANK"])

            def capture_before_train(trainer, *args, **kwargs):
                parallelism = trainer.config.parallelism
                campaign.require(
                    parallelism.gfsdp_match_fsdp2_init and trainer.step == 0,
                    "Expected a fresh native model with FSDP2-matching initialization",
                )
                campaign.write_json(output / f"initial-weights-rank-{rank:02d}.json", {
                    "run_id": adapter.options["run_id"],
                    "rank": rank,
                    "layer0_split": parallelism.gfsdp_split_layer0_ffn,
                    "diagnostic_sha256": campaign.digest(SCRIPT),
                    "parameters": parameter_fingerprints(trainer.model_parts),
                })
                return train(trainer, *args, **kwargs)

            adapter._replace(adapter.trainer_class, "train", capture_before_train)
            return adapter

        with patch.object(bridge, "_load_base_measure", return_value=base), patch.object(
            base.NativeAdapter, "__enter__", install
        ):
            bridge.main(sys.argv[1:])

    # Preserve the existing worker's activation logging and FlashQLA dispatch.
    with patch.object(campaign, "runpy", SimpleNamespace(run_path=run_bridge)):
        campaign.worker(argv)


class InitialWeightsCase(campaign.NativeCase):
    def base_args(self, run_id, timeout):
        argv = super().base_args(run_id, timeout)
        argv[argv.index("--window") + 1] = "0"
        argv.remove("--observe-compiler-cache")
        return [*argv, "--capture-microbatches"]

    def summarize(self, attempt):
        completion = campaign.read_json(attempt / "completion.json")
        meta = campaign.read_json(attempt / "metadata.json")
        summary = campaign.read_json(attempt / "summary.json")
        config = attempt / "native/config/resolved_config.yaml"
        script_hash = campaign.digest(SCRIPT)
        campaign.require(
            completion["status"] == "PASS" and completion["source_unchanged"]
            and completion["config_unchanged"]
            and completion["resolved_config_sha256"] == campaign.digest(config)
            and meta["source_head_before"] == self.expected_head
            and meta["source_snapshot_before"]["tracked_and_untracked_sha256"] == self.expected_hash,
            f"Native source or configuration verification failed in {attempt}",
        )
        campaign.require(
            summary["status"] == "complete" and summary["mode"] == "cadence_only"
            and summary["training_steps"] == 1 and meta["capture_microbatches"],
            "Expected an unprofiled single native update with full microbatch coverage",
        )
        self.check_native_command(meta["argv"])
        self.check_resolved({"native": attempt / "native"})
        self.check_input_hashes(meta["config_hashes_before"])
        bridge = campaign.read_json(attempt / "flash-qla-bridge.json")
        route_hash = campaign.digest(self.root / "gift/torchtitan/distributed/deepep/routing.py")
        ranks = []
        for rank in range(campaign.WORLD):
            measurement = attempt / "measurement"
            bridge_rank = campaign.read_json(measurement / f"flash-qla-bootstrap-rank-{rank:02d}.json")
            campaign.verify_bridge(bridge_rank, bridge, rank, route_hash)
            campaign.read_rank(attempt, rank, self.ep, self.features)
            receipt = campaign.read_json(measurement / f"initial-weights-rank-{rank:02d}.json")
            campaign.require(
                receipt["rank"] == rank and receipt["run_id"] == attempt.name
                and receipt["layer0_split"] == (self.arm == "layer0")
                and receipt["diagnostic_sha256"] == script_hash,
                f"Unexpected initial-weight receipt for rank {rank}",
            )
            ranks.append(receipt["parameters"])
        campaign.require(all(ranks) and all(row.keys() == ranks[0].keys() for row in ranks),
                         "Rank parameter coverage differs")
        return {
            "arm": self.arm, "attempt": str(attempt), "reference": str(self.reference),
            "input_hashes": self.input_hashes, "resolved_config_sha256": campaign.digest(config),
            "source_sha256": self.expected_hash, "diagnostic_sha256": script_hash,
            "parameters_by_rank": ranks,
        }


def compare(original, split):
    campaign.require((original["arm"], split["arm"]) == ARMS, "Expected Original A and layer0 receipts")
    for key in ("reference", "input_hashes", "source_sha256", "diagnostic_sha256"):
        campaign.check_equal(original[key], split[key], key)
    a, b = original["parameters_by_rank"], split["parameters_by_rank"]
    campaign.require(len(a) == len(b) == campaign.WORLD, "Expected all eight rank receipts in each arm")
    a_names, b_names = set(a[0]), set(b[0])
    campaign.require(a_names and b_names and all(
        set(left) == a_names and set(right) == b_names for left, right in zip(a, b, strict=True)
    ), "Each arm must contain the same FQNs on all ranks")
    different, unresolved = set(), set()
    for rank, (left, right) in enumerate(zip(a, b, strict=True)):
        for name in a_names & b_names:
            if name.startswith(LAYER0_PREFIX):
                for row, first in ((left, a[0]), (right, b[0])):
                    campaign.require(row[name]["scope"] == "logical" and row[name] == first[name],
                                     f"Layer 0 logical fingerprint disagrees on rank {rank}: {name}")
            lhs, rhs = dict(left[name]), dict(right[name])
            lhs_hash, rhs_hash = lhs.pop("sha256"), rhs.pop("sha256")
            if lhs == rhs:
                if lhs_hash != rhs_hash:
                    different.add(name)
            else:
                unresolved.add(name)
    missing = a_names ^ b_names
    return {
        "initial_weights_equal": False if different else (None if unresolved or missing else True),
        "layer0_logical_parameter_count": sum(name.startswith(LAYER0_PREFIX) for name in a_names & b_names),
        "matched_parameter_count": len((a_names & b_names) - different - unresolved),
        "different_weight_fqns": sorted(different), "unresolved_layout_fqns": sorted(unresolved),
        "only_original_fqns": sorted(a_names - b_names), "only_split_fqns": sorted(b_names - a_names),
        "original_attempt": original["attempt"], "split_attempt": split["attempt"],
    }


def run(args):
    case = InitialWeightsCase(args, 8, args.arm)
    case.preflight(args.timeout_seconds)
    if args.preflight_only:
        print(f"PASS native source, inputs and command for EP8 {args.arm}")
        return 0
    root = args.campaign.resolve()
    campaign.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", root.name), "Invalid campaign name")
    output = root / "initial-weights"
    output.mkdir(parents=True, exist_ok=True)
    key = f"ep8-{args.arm}"
    run_id = f"{root.name}-initial-weights-{key}-{uuid4().hex[:8]}"
    verified = output / f"{key}.json"
    with (root / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.require(not verified.exists(), f"A verified receipt already exists at {verified}")
        campaign.check_gpus(output / f"{run_id}.gpus.csv")
        print(f"START {run_id} ({case.raw / run_id})", flush=True)
        status = case.launch(run_id, args.timeout_seconds)
        if status:
            return status
        campaign.write_json(verified, case.summarize(case.raw / run_id))
    print(f"PASS {verified}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["worker"]:
        return worker(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    launch = modes.add_parser("run", help="capture initialized weights and complete one native update")
    launch.add_argument("campaign", type=Path, help="same output directory as the native timing matrix")
    launch.add_argument("arm", choices=ARMS)
    launch.add_argument("--preflight-only", action="store_true")
    campaign.add_native_args(launch)
    comparison = modes.add_parser("compare", help="compare the two verified arms")
    comparison.add_argument("campaign", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "compare":
        output = args.campaign / "initial-weights"
        result = compare(*(campaign.read_json(output / f"ep8-{arm}.json") for arm in ARMS))
        path = output / "comparison.json"
        campaign.write_json(path, result)
        print(f"Initial weights equal: {result['initial_weights_equal']} ({path})")
        return 0
    with patch.multiple(campaign, SCRIPT=SCRIPT, STEPS=1, WARMUP=0):
        return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
