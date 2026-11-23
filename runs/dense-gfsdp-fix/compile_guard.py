#!/usr/bin/env python3
"""Capture checkpoint inventory and Dynamo guard logs in one native update."""

import argparse
import fcntl
from functools import partial
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
ARMS = ("a", "row-owner", "layer0-row-owner")
ERROR_TYPE = "FailOnRecompileLimitHit"
LIMIT_LOG = "torch._dynamo hit config.recompile_limit (8)"
FULLGRAPH_LOG = f"{ERROR_TYPE}: Hard failure due to fullgraph=True"


def checkpoint_inventory(model, *, layer0_split):
    rows = {}
    for segment in model.named_segments():
        if not segment.is_block:
            continue
        module = segment.module
        regions = getattr(module, "_ffn_overlap_regions", None)
        units = (module,) if regions is None else regions
        expected_units = 3 if layer0_split and segment.fqn == "layers.0" else 1
        campaign.require(len(units) == expected_units,
                         f"Unexpected checkpoint topology for {segment.fqn}")
        unit_rows = []
        for unit in units:
            kwargs = (unit.checkpoint_fn.keywords if regions is None else unit._ffn_checkpoint_kwargs)
            context = kwargs["context_fn"]
            campaign.require(isinstance(context, partial) and len(context.args) == 2,
                             f"Unrecognized selective checkpoint context for {segment.fqn}")
            unit_rows.append({
                "module_type": type(unit).__qualname__,
                "forward": unit.forward.__qualname__,
                "compiled": unit._compiled_call_impl is not None,
                "saved_categories": context.args[0],
                "force_recompute_mm_rhs_shapes": context.args[1],
            })
        rows[segment.fqn] = unit_rows
    expected = {f"layers.{i}" for i in range(8)} | {"mtp.layers.0"}
    campaign.require(rows.keys() == expected and all(row["compiled"] for units in rows.values() for row in units),
                     "Expected all nine crop blocks under native selective AC and compilation")
    return rows


def worker(argv):
    campaign.require("torch" not in sys.modules, "Torch imported before Dynamo logging was set")
    os.environ["TORCH_LOGS"] = "recompiles"

    def run_bridge(path, *, run_name):
        campaign.require(run_name == "__main__", "Unexpected native worker entry point")
        bridge = campaign.load_module(Path(path), "_compile_guard_flash_bridge")
        base = bridge._load_base_measure()
        enter = base.NativeAdapter.__enter__

        def install(adapter):
            enter(adapter)
            train = adapter.trainer_class.train
            rank = int(os.environ["RANK"])
            output = Path(adapter.options["output_dir"])

            def record_before_train(trainer, *args, **kwargs):
                import torch

                (model,) = trainer.model_parts
                config = trainer.config
                campaign.require(trainer.step == 0 and config.activation_checkpoint.mode == "selective"
                                 and config.compile.enable and "model" in config.compile.components,
                                 "Expected fresh native model with selective AC and compilation")
                campaign.write_json(output / f"compile-guard-rank-{rank:02d}.json", {
                    "rank": rank,
                    "run_id": adapter.options["run_id"],
                    "driver_sha256": campaign.digest(SCRIPT),
                    "torch_logs": os.environ["TORCH_LOGS"],
                    "recompile_limit": torch._dynamo.config.recompile_limit,
                    "row_owner_enabled": config.parallelism.gfsdp_row_owner_embeddings,
                    "layer0_split_enabled": config.parallelism.gfsdp_split_layer0_ffn,
                    "checkpoint_wrappers": checkpoint_inventory(
                        model, layer0_split=config.parallelism.gfsdp_split_layer0_ffn
                    ),
                })
                return train(trainer, *args, **kwargs)

            adapter._replace(adapter.trainer_class, "train", record_before_train)
            return adapter

        with patch.object(bridge, "_load_base_measure", return_value=base), patch.object(
            base.NativeAdapter, "__enter__", install
        ):
            bridge.main(sys.argv[1:])

    with patch.object(campaign, "runpy", SimpleNamespace(run_path=run_bridge)):
        campaign.worker(argv)


class GuardCase(campaign.NativeCase):
    def base_args(self, run_id, timeout):
        argv = super().base_args(run_id, timeout)
        argv[argv.index("--window") + 1] = "0"
        argv.remove("--observe-compiler-cache")
        return argv

    def read_evidence(self, attempt, result):
        result["process"] = optional_json(attempt / "process.json")
        result["completion"] = optional_json(attempt / "completion.json")
        result["failure"] = optional_json(attempt / "failure.json")
        meta = campaign.read_json(attempt / "metadata.json")
        self.check_native_command(meta["argv"])
        self.check_resolved({"native": attempt / "native"})
        self.check_input_hashes(meta["config_hashes_before"])
        final = campaign.read_json(attempt / "source-final.json")
        campaign.require(
            final["head"] == meta["source_head_before"] == self.expected_head
            and final["tracked_and_untracked_sha256"]
            == meta["source_snapshot_before"]["tracked_and_untracked_sha256"] == self.expected_hash
            and campaign.read_json(attempt / "config-hashes-final.json") == meta["config_hashes_before"],
            "Native source or config freeze differs",
        )
        result["resolved_config_sha256"] = campaign.digest(attempt / "native/config/resolved_config.yaml")
        bridge = campaign.read_json(attempt / "flash-qla-bridge.json")
        driver_hash = campaign.digest(SCRIPT)
        route_hash = campaign.digest(self.root / "gift/torchtitan/distributed/deepep/routing.py")
        rank_rows = result["ranks"]
        for rank in range(campaign.WORLD):
            path = attempt / "measurement"
            inventory = optional_json(path / f"compile-guard-rank-{rank:02d}.json")
            receipt_path = path / f"flash-qla-bootstrap-rank-{rank:02d}.json"
            receipt = optional_json(receipt_path)
            row = rank_rows.setdefault(str(rank), {})
            row["inventory"] = inventory
            row["bootstrapped"] = bool(receipt)
            row["native_main_completed"] = receipt.get("native_main_completed")
            row["native_error_type"] = receipt.get("native_main_error_type")
            log_paths = sorted((attempt / "logs/torchrun").glob(f"*/attempt_0/{rank}/*.log"))
            row["logs"] = [str(log) for log in log_paths]
            row["flash_qla_receipt"] = str(receipt_path) if receipt else None
            lines = [line for log in log_paths for line in campaign.ANSI.sub(
                "", log.read_text(errors="replace")
            ).splitlines()]
            row["guard_log_excerpt"] = [
                line.strip() for line in lines
                if any(word in line for word in (LIMIT_LOG, FULLGRAPH_LOG, "[__recompiles]",
                                                  "last reason:", "checkpoint_wrapper.py"))
            ]
            row["updated"] = any(campaign.STEP.search(line) for line in lines)
            row["target_error_logged"] = checkpoint_limit_logged(lines)
            row["verified_route"] = False
            if receipt.get("constructed_loss_callables"):
                # The common verifier checks the route after requiring a completed
                # native main. Here the independent error receipt determines completion.
                campaign.verify_bridge(receipt | {"native_main_completed": True}, bridge, rank, route_hash)
                row["verified_route"] = True
            if inventory:
                campaign.require(
                    inventory["rank"] == rank and inventory["run_id"] == attempt.name
                    and inventory["driver_sha256"] == driver_hash
                    and inventory["torch_logs"] == "recompiles" and inventory["recompile_limit"] == 8
                    and inventory["row_owner_enabled"] is self.features[campaign.ROW_OWNER]
                    and inventory["layer0_split_enabled"] is self.features[campaign.LAYER0],
                    f"Checkpoint inventory identity differs for rank {rank}",
                )
            if row["target_error_logged"] or row["native_main_completed"]:
                stdout = next((log for log in log_paths if log.name == "stdout.log"), None)
                campaign.require(stdout is not None, f"Missing rank {rank} stdout")
                out = stdout.read_text(errors="replace")
                campaign.require(
                    (campaign.GFSDP_LOG_TAG + campaign.MARKERS[campaign.ROW_OWNER] in out)
                    is self.features[campaign.ROW_OWNER],
                    f"Row-owner activation differs for rank {rank}",
                )

def optional_json(path):
    return campaign.read_json(path) if path.is_file() else {}


def checkpoint_limit_logged(lines):
    return any(FULLGRAPH_LOG in line for line in lines) and any(
        LIMIT_LOG in line
        and any("checkpoint_wrapper.py" in detail and "forward" in detail for detail in lines[i + 1:i + 6])
        for i, line in enumerate(lines)
    )


def classify(result):
    process, ranks, status = result["process"], list(result["ranks"].values()), result["native_exit_code"]
    if (len(ranks) != campaign.WORLD or not process.get("process_group_stopped")
            or process.get("timeout") is not False or process.get("orphaned_process_group") is not False
            or process.get("child_exit_code") != status):
        return "unexpected_failure"

    def verified(row):
        return row["verified_route"] and bool(row["inventory"])

    if (status == 0 and result["completion"].get("status") == "PASS"
            and result["completion"].get("resolved_config_sha256") == result["resolved_config_sha256"]
            and all(verified(row) and row["native_main_completed"] is True and row["updated"] for row in ranks)):
        return "completed"
    matching = [row for row in ranks if row["native_error_type"] == ERROR_TYPE]
    if (status == 1 and result["failure"].get("status") == "FAIL" and not any(row["updated"] for row in ranks)
            and matching and all(verified(row) and row["target_error_logged"] for row in matching)
            and not any(row["native_error_type"] not in (None, ERROR_TYPE) for row in ranks)):
        return "checkpoint_recompile_limit"
    return "unexpected_failure"


def run(args):
    case = GuardCase(args, 8, args.arm)
    case.preflight(args.timeout_seconds)
    if args.preflight_only:
        print(f"PASS native source, inputs and one-update command for EP8 {args.arm}")
        return 0

    root = args.campaign.resolve()
    campaign.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", root.name), "Invalid campaign name")
    output = root / "compile-guard"
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"{root.name}-compile-guard-ep8-{args.arm}-{uuid4().hex[:8]}"
    attempt = case.raw / run_id
    path = output / f"{run_id}.json"
    result = {
        "attempt": str(attempt), "arm": args.arm, "ep": 8, "classification": "unexpected_failure",
        "native_exit_code": None, "ranks": {}, "source_sha256": case.expected_hash,
        "input_hashes": case.input_hashes,
    }
    with (root / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.check_gpus(output / f"{run_id}.gpus.csv")
        print(f"START {attempt}", flush=True)
        try:
            result["native_exit_code"] = case.launch(run_id, args.timeout_seconds)
            case.read_evidence(attempt, result)
            result["classification"] = classify(result)
        except (KeyError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            result["evidence_error"] = str(error)
        finally:
            campaign.write_json(path, result)
    print(f"{result['classification']} {path}")
    reproduced = args.arm == "row-owner" and result["classification"] == "checkpoint_recompile_limit"
    return int(result["classification"] != "completed" and not reproduced)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["worker"]:
        return worker(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--preflight-only", action="store_true")
    campaign.add_native_args(parser)
    args = parser.parse_args(argv)
    with patch.multiple(campaign, SCRIPT=SCRIPT, STEPS=1, WARMUP=0), patch.dict(
        campaign.ARMS, {"layer0-row-owner": (campaign.LAYER0, campaign.ROW_OWNER)}
    ):
        return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
