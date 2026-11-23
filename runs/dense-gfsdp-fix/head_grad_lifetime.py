#!/usr/bin/env python3
"""Measure the native EP8 head FP32 release and downstream pack boundaries."""

from __future__ import annotations

import argparse
import fcntl
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import campaign


SCRIPT = Path(__file__).resolve()
UPDATE = campaign.WARMUP + 1
MICROBATCHES = 32
HEAD = "lm_head"
EARLY = "mtp.layers.0.block.feed_forward"
# The trunk norm keeps its view of the early FFN backing in the proposed layout.
LATE = ("mtp.layers.0", "layers.7[fsdp]")
CONSUMERS = (EARLY, *LATE, "norm")


class UnsupportedCapture(RuntimeError):
    """The current source or topology does not expose the pinned boundaries."""


def locate_groups(trainer, torch):
    if (
        trainer.parallel_dims.pp_enabled
        or len(trainer.model_parts) != 1
        or trainer.gradient_accumulation_steps != MICROBATCHES
        or getattr(trainer.loss_fn, "num_chunks", None) != 2
    ):
        raise UnsupportedCapture(
            "expected PP1 with 32 microbatches and two loss chunks"
        )
    root = trainer.model_parts[0]._get_fsdp_state()
    groups = {}
    for state in root._state_ctx.all_states:
        for group in state._fsdp_param_groups:
            name = group._module_fqn
            if name not in (HEAD, *CONSUMERS):
                continue
            if (
                name in groups
                or group.device.type != "cuda"
                or group.param_dtype != torch.bfloat16
                or group.reduce_dtype != torch.float32
                or group.mesh_info.shard_mesh_size != campaign.WORLD
                or group._all_reduce_process_group is not None
                or group.grad_buffer_ctx is None
                or any(
                    handle is not None
                    for handle in group.grad_buffer_ctx.yccl_handles.values()
                )
            ):
                raise UnsupportedCapture(
                    f"unsupported dense BF16/FP32 GFSDP group {name}"
                )
            groups[name] = group
    if set(groups) != {HEAD, *CONSUMERS}:
        raise UnsupportedCapture(
            f"missing GFSDP group(s): {sorted({HEAD, *CONSUMERS} - groups.keys())}"
        )
    head = groups[HEAD]
    if not trainer.model_parts[0].lm_head._get_fsdp_state()._own_grad_buffer:
        raise UnsupportedCapture("expected the native private head gradient buffer")
    if head._post_backward_prefetch_target is not None or not callable(
        getattr(head.comm_ctx.reduce_scatter_stream, "record_event", None)
    ):
        raise UnsupportedCapture(
            "head does not expose the terminal native reduce stream"
        )
    return groups


class LifetimeCapture:
    def __init__(self, trainer, torch):
        self.torch = torch
        groups = locate_groups(trainer, torch)
        self.head = groups[HEAD]
        self.consumer_of = {
            id(param): name for name in CONSUMERS for param in groups[name].fsdp_params
        }
        self.compute = torch.cuda.current_stream(self.head.device)
        self.reduce = self.head.comm_ctx.reduce_scatter_stream
        if self.compute.device != self.reduce.device or self.compute == self.reduce:
            raise UnsupportedCapture(
                "expected separate compute and reduce streams on one CUDA device"
            )
        self.anchor = None
        self.rows = []
        self.active = None

    def begin_microbatch(self):
        row = {"writes": {}, "head_releases": 0, "tail": None, "problems": []}
        self.rows.append(row)
        self.active = row
        if len(self.rows) == 1:
            self.anchor = self.mark(self.compute)

    def mark(self, stream):
        try:
            event = self.torch.cuda.Event(enable_timing=True)
            event.record(stream)
            return event
        except (RuntimeError, TypeError, ValueError) as error:
            self.active["problems"].append(f"CUDA event recording unsupported: {error}")
            return None

    def after_head_post_backward(self, previous_event):
        row = self.active
        release = self.head._post_reduce_event
        # The first detached chunk accumulates without reduction. Idle group
        # callbacks and unchanged pending events are also outside this boundary.
        if row is None or release is None or release is previous_event:
            return
        row["head_releases"] += 1
        grad = self.head.grad_buffer_ctx
        if (
            grad.release_event is not release
            or grad.owner is not None
            or self.torch.cuda.current_stream(self.head.device) != self.compute
        ):
            row["problems"].append(
                "head reduction did not publish its expected release"
            )
        elif row["head_releases"] == 1:
            # Native reduce_scatter records release after the last FP32
            # output copy/add. This marker follows it on that same stream.
            row["tail"] = self.mark(self.reduce)

    def before_pack(self, param):
        row = self.active
        if row is None:
            return
        name = self.consumer_of.get(id(param))
        if name is None or name in row["writes"]:
            return
        stream = self.torch.cuda.current_stream(self.head.device)
        if stream != self.compute:
            row["problems"].append(f"unexpected pack stream for {name}")
            row["writes"][name] = None
            return
        # Only the real gradient-pack path visits unsharded_grad_data.
        # Its native buffer wait has been enqueued and the pack follows.
        row["writes"][name] = self.mark(stream)

    def finish(self):
        if self.anchor is not None:
            self.torch.cuda.synchronize(self.head.device)
        rows = []
        for index, row in enumerate(self.rows):
            problems = row["problems"]
            if row["head_releases"] != 1:
                problems.append(
                    f"expected one head release, observed {row['head_releases']}"
                )
            if self.anchor is None or row["tail"] is None:
                problems.append("missing anchor or head timing event")
            if missing := [
                name for name in CONSUMERS if row["writes"].get(name) is None
            ]:
                problems.append(f"missing FP32 pack event(s): {missing}")
            record = {
                "microbatch": index,
                "head_releases": row["head_releases"],
                "problems": problems,
            }
            if not problems:
                try:
                    tail_ms = self.anchor.elapsed_time(row["tail"])
                    writes = {
                        name: self.anchor.elapsed_time(row["writes"][name])
                        for name in CONSUMERS
                    }
                    if not all(
                        math.isfinite(x) and x >= 0 for x in (tail_ms, *writes.values())
                    ):
                        raise ValueError("invalid same-device elapsed time")
                    record.update(
                        {
                            "head_release_marker_ms": tail_ms,
                            "fp32_pack_marker_ms": writes,
                            "head_release_to_pack_marker_ms": {
                                name: value - tail_ms for name, value in writes.items()
                            },
                        }
                    )
                except (RuntimeError, TypeError, ValueError) as error:
                    problems.append(f"CUDA elapsed time unsupported: {error}")
            rows.append(record)
        return {
            "status": "complete"
            if len(rows) == MICROBATCHES and not any(row["problems"] for row in rows)
            else "incomplete",
            "microbatches": rows,
            "update_end_device_drains": int(self.anchor is not None),
            "stream_ids": {
                "compute": self.compute.cuda_stream,
                "reduce": self.reduce.cuda_stream,
            },
        }


def install_probe(adapter):
    import torch
    from torchtitan.distributed.gfsdp._param import GFSDPParam
    from torchtitan.distributed.gfsdp._param_group import GFSDPParamGroup

    trainer_class = adapter.trainer_class
    original_step = trainer_class.train_step
    original_micro = trainer_class.forward_backward_step
    original_post = GFSDPParamGroup.post_backward
    grad_property = GFSDPParam.unsharded_grad_data
    output = Path(adapter.options["output_dir"])
    rank = int(os.environ["RANK"])
    capture = None

    def write_receipt(value):
        campaign.write_json(
            output / f"head-grad-lifetime-rank-{rank:02d}.json",
            {
                "rank": rank,
                "run_id": adapter.options["run_id"],
                "update": UPDATE,
                "diagnostic_sha256": campaign.digest(SCRIPT),
                **value,
            },
        )

    def track_micro(trainer, *args, **kwargs):
        if capture is None:
            return original_micro(trainer, *args, **kwargs)
        capture.begin_microbatch()
        try:
            return original_micro(trainer, *args, **kwargs)
        finally:
            capture.active = None

    def track_step(trainer, *args, **kwargs):
        nonlocal capture
        if trainer.step != UPDATE:
            return original_step(trainer, *args, **kwargs)
        try:
            capture = LifetimeCapture(trainer, torch)
        except UnsupportedCapture as error:
            result = original_step(trainer, *args, **kwargs)
            write_receipt({"status": "unsupported", "reason": str(error)})
            return result

        def track_post(group, *post_args, **post_kwargs):
            if group is not capture.head:
                return original_post(group, *post_args, **post_kwargs)
            previous = group._post_reduce_event
            result = original_post(group, *post_args, **post_kwargs)
            capture.after_head_post_backward(previous)
            return result

        def track_grad(param):
            grad = grad_property.fget(param)
            capture.before_pack(param)
            return grad

        try:
            # These GFSDP hooks are outside the compiled block regions.
            with (
                patch.object(GFSDPParamGroup, "post_backward", track_post),
                patch.object(
                    GFSDPParam, "unsharded_grad_data", grad_property.getter(track_grad)
                ),
            ):
                result = original_step(trainer, *args, **kwargs)
            write_receipt(capture.finish())
            return result
        finally:
            capture = None

    adapter._replace(trainer_class, "train_step", track_step)
    adapter._replace(trainer_class, "forward_backward_step", track_micro)


def worker(argv):
    def run_bridge(path, *, run_name):
        campaign.require(run_name == "__main__", "Unexpected native entry point")
        bridge = campaign.load_module(Path(path), "_head_lifetime_flash_bridge")
        base = bridge._load_base_measure()
        enter = base.NativeAdapter.__enter__

        def install(adapter):
            enter(adapter)
            install_probe(adapter)
            return adapter

        with (
            patch.object(bridge, "_load_base_measure", return_value=base),
            patch.object(base.NativeAdapter, "__enter__", install),
        ):
            bridge.main(sys.argv[1:])

    with patch.object(campaign, "runpy", SimpleNamespace(run_path=run_bridge)):
        campaign.worker(argv)


class LifetimeCase(campaign.NativeCase):
    def base_args(self, run_id, timeout):
        argv = super().base_args(run_id, timeout)
        argv[argv.index("--window") + 1] = "0"
        argv.remove("--observe-compiler-cache")
        return argv

    def collect(self, attempt):
        completion = campaign.read_json(attempt / "completion.json")
        summary = campaign.read_json(attempt / "summary.json")
        metadata = campaign.read_json(attempt / "metadata.json")
        config = attempt / "native/config/resolved_config.yaml"
        script_hash = campaign.digest(SCRIPT)
        campaign.require(
            completion["status"] == "PASS"
            and completion["source_unchanged"]
            and completion["config_unchanged"]
            and completion["resolved_config_sha256"] == campaign.digest(config)
            and metadata["source_head_before"] == self.expected_head
            and metadata["source_snapshot_before"]["tracked_and_untracked_sha256"]
            == self.expected_hash
            and summary["status"] == "complete"
            and summary["training_steps"] == UPDATE
            and summary["mode"] == "cadence_only"
            and summary["measurement_status"] == "disabled",
            "Expected the unchanged unprofiled EP8 crop through update 11",
        )
        self.check_native_command(metadata["argv"])
        self.check_resolved({"native": attempt / "native"})
        self.check_input_hashes(metadata["config_hashes_before"])
        bridge = campaign.read_json(attempt / "flash-qla-bridge.json")
        route_hash = campaign.digest(
            self.root / "gift/torchtitan/distributed/deepep/routing.py"
        )
        ranks = []
        for rank in range(campaign.WORLD):
            measurement = attempt / "measurement"
            bridge_rank = campaign.read_json(
                measurement / f"flash-qla-bootstrap-rank-{rank:02d}.json"
            )
            campaign.verify_bridge(bridge_rank, bridge, rank, route_hash)
            campaign.read_rank(attempt, rank, self.ep, self.features)
            path = measurement / f"head-grad-lifetime-rank-{rank:02d}.json"
            if not path.is_file():
                ranks.append({"rank": rank, "status": "missing_receipt"})
                continue
            receipt = campaign.read_json(path)
            campaign.require(
                receipt["rank"] == rank
                and receipt["run_id"] == attempt.name
                and receipt["update"] == UPDATE
                and receipt["diagnostic_sha256"] == script_hash,
                f"Head gradient lifetime receipt identity differs on rank {rank}",
            )
            ranks.append(receipt)
        return summarize_ranks(ranks) | {
            "attempt": str(attempt),
            "update": UPDATE,
            "warmup_updates": campaign.WARMUP,
            "expected_microbatches_per_rank": MICROBATCHES,
            "source_sha256": self.expected_hash,
            "source_head": self.expected_head,
            "diagnostic_sha256": script_hash,
            "input_hashes": self.input_hashes,
            "resolved_config_sha256": campaign.digest(config),
            "ranks": ranks,
        }


def summarize_ranks(ranks):
    expected = list(range(MICROBATCHES))
    invalid = [
        {"rank": rank["rank"], "status": rank["status"]}
        for rank in ranks
        if rank["status"] != "complete"
        or [row["microbatch"] for row in rank["microbatches"]] != expected
        or rank["update_end_device_drains"] != 1
    ]
    if len(ranks) != campaign.WORLD or invalid:
        return {"status": "incomplete", "invalid_ranks": invalid}
    margins = {
        name: [
            row["head_release_to_pack_marker_ms"][name]
            for rank in ranks
            for row in rank["microbatches"]
        ]
        for name in CONSUMERS
    }
    minimum_late_margin = min(min(margins[name]) for name in LATE)
    return {
        "status": "complete",
        "late_pack_marker_at_or_before_head_release": minimum_late_margin <= 0,
        "minimum_late_margin_ms": minimum_late_margin,
        "late_margin_consumers": LATE,
        "scope": "Same-device marker order for this update; a pack marker precedes its actual write",
        "by_consumer": {
            name: {
                "minimum_margin_ms": min(values),
                "maximum_margin_ms": max(values),
                "at_or_before_head_release": sum(value <= 0 for value in values),
                "paired_samples": len(values),
            }
            for name, values in margins.items()
        },
    }


def run(args):
    case = LifetimeCase(args, 8, "all")
    case.preflight(args.timeout_seconds)
    if args.preflight_only:
        print("PASS unchanged EP8 all-four source, inputs and native CLI")
        return 0
    root = args.output.resolve()
    campaign.require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", root.name),
        "Invalid diagnostic output name",
    )
    root.mkdir()
    run_id = f"{root.name}-head-lifetime-{uuid4().hex[:8]}"
    with (root / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.check_gpus(root / "gpus.csv")
        status = case.launch(run_id, args.timeout_seconds)
        if status:
            return status
        evidence = case.collect(case.raw / run_id)
        output = root / "result.json"
        campaign.write_json(output, evidence)
        print(f"{evidence['status']} {output}")
        return int(evidence["status"] != "complete")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["worker"]:
        return worker(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    campaign.add_native_args(parser)
    with patch.multiple(campaign, SCRIPT=SCRIPT, STEPS=UPDATE):
        return run(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1) from error
