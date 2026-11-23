#!/usr/bin/env python3
"""Compare one native EP8 update from the same live and EMA layer-0 weights."""

import argparse
from contextlib import ExitStack
import hashlib
import math
import os
from pathlib import Path
import pickle
import random
import sys
from unittest.mock import patch

import campaign
import initial_weights


SCRIPT = Path(__file__).resolve()
ENV_PREFIX = "DENSE_MATCHED_LAYER0_"
PARAMETERS = 243
CHANGED = 18
LAYER0_PARAMETERS = 23


def read_selection(path):
    receipt = campaign.read_json(path)
    names = receipt["different_weight_fqns"]
    campaign.require(
        receipt["initial_weights_equal"] is False
        and type(names) is list and len(names) == len(set(names)) == CHANGED
        and all(name.startswith(initial_weights.LAYER0_PREFIX) for name in names)
        and receipt["layer0_logical_parameter_count"] == LAYER0_PARAMETERS
        and receipt["matched_parameter_count"] == PARAMETERS - CHANGED
        and not any(receipt[key] for key in (
            "only_original_fqns", "only_split_fqns", "unresolved_layout_fqns"
        )),
        "Expected the verified native initial-weight comparison with 18 changed layer-0 FQNs",
    )
    return sorted(names)


def selected_views(state, names, ema_key, flat_type, fp32):
    campaign.require(ema_key in state, "The native EMA checkpoint namespace is absent")
    result = {name: state[name] for name in names}
    result[ema_key] = {name: state[ema_key][name] for name in names}
    for name in names:
        main, ema = result[name], result[ema_key][name]
        campaign.require(
            isinstance(main, flat_type) and isinstance(ema, flat_type)
            and main.dtype == ema.dtype == fp32 and main.shape == ema.shape,
            f"Expected compatible live and EMA flat FP32 views for {name}",
        )
    return result


def ema_hashes(views, names, ema_key):
    # Each view aliases the backing slice for its local layout. A local EMA
    # hash equal to the local main hash on every rank establishes logical
    # EMA equality once the live logical masters have matched.
    hashes = {}
    for name in names:
        live = initial_weights.weight_sha256(views[name]._local_tensor)
        ema = initial_weights.weight_sha256(views[ema_key][name]._local_tensor)
        campaign.require(live == ema, f"Fresh EMA differs from the live master for {name}")
        hashes[name] = ema
    return hashes


def rng_snapshot(torch):
    import numpy as np

    algorithm, key, position, has_gauss, cached_gauss = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "numpy": (algorithm, key.tolist(), position, has_gauss, cached_gauss),
        "python": random.getstate(),
    }


def rng_hashes(state):
    return {
        name: initial_weights.weight_sha256(value) if name.startswith("torch_")
        else hashlib.sha256(pickle.dumps(value)).hexdigest()
        for name, value in state.items()
    }


def restore_rng(torch, state):
    import numpy as np

    algorithm, key, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((algorithm, np.asarray(key, dtype=np.uint32), position, has_gauss, cached_gauss))
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def tensor_record(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": initial_weights.weight_sha256(tensor),
    }


def gradient_file(value, path, *, control):
    """Write or compare one logical FP32 gradient in bounded CPU blocks."""
    import numpy as np

    actual_digest = hashlib.sha256()
    reference_digest = hashlib.sha256()
    diff_square = reference_square = max_abs = 0.0
    numel = value.numel()
    mode = "xb" if control else "rb"
    with path.open(mode) as stream:
        for values in value.detach().reshape(-1).split(initial_weights.COPY_BYTES // 4):
            block = values.cpu().contiguous().numpy()
            campaign.require(np.isfinite(block).all(), f"Nonfinite gradient for {path.stem}")
            actual_digest.update(memoryview(block))
            if control:
                stream.write(memoryview(block))
                continue
            reference = np.fromfile(stream, dtype=np.float32, count=block.size)
            campaign.require(reference.shape == block.shape, f"Truncated gradient for {path.stem}")
            reference_digest.update(memoryview(reference))
            ref64 = reference.astype(np.float64)
            delta = block.astype(np.float64) - ref64
            diff_square += float(np.dot(delta, delta))
            reference_square += float(np.dot(ref64, ref64))
            max_abs = max(max_abs, float(np.abs(delta).max(initial=0)))
        if not control:
            campaign.require(not stream.read(1), f"Extra gradient values for {path.stem}")
    record = {"sha256": actual_digest.hexdigest()}
    if not control:
        record.update(
            reference_sha256=reference_digest.hexdigest(),
            max_abs=max_abs,
            rms=math.sqrt(diff_square / numel),
            relative_l2=math.sqrt(diff_square / reference_square) if reference_square else None,
        )
    return record


class WorkerCapture:
    def __init__(self):
        self.arm = os.environ[ENV_PREFIX + "ARM"]
        self.output = Path(os.environ[ENV_PREFIX + "OUTPUT"])
        self.selection = Path(os.environ[ENV_PREFIX + "SELECTION"])
        self.names = read_selection(self.selection)
        self.rank = int(os.environ["RANK"])
        self.control = self.arm == "a"
        self.output_rank = self.rank == 0
        self.initial = None
        self.microbatches = []
        self.finished = False

    def prepare(self, model_parts, fingerprint):
        import torch
        import torch.distributed as dist
        import torch.distributed.checkpoint as dcp
        from torchtitan.distributed.gfsdp._dcp import (
            EMA_MODEL, GFSDPCheckpointAdapter, GFSDPLoadPlanner, _FlatCheckpointTensor,
        )

        folder = self.output / "checkpoint"
        adapter = GFSDPCheckpointAdapter(model_parts)
        native_rng = rng_snapshot(torch)
        before = rng_hashes(native_rng)
        with torch.no_grad():
            state = adapter.build_state_dict(load_target=not self.control)
            views = selected_views(state, self.names, EMA_MODEL, _FlatCheckpointTensor, torch.float32)
            del state
            if self.control:
                dcp.save(views, checkpoint_id=str(folder))
            else:
                planner = GFSDPLoadPlanner()
                dcp.load(views, checkpoint_id=str(folder), planner=planner)
                campaign.require(not planner.ema_init_fqns, "The selected EMA payload was incomplete")
                adapter.apply_main_state_dict({name: views[name] for name in self.names})
            ema = ema_hashes(views, self.names, EMA_MODEL)
            parameters = fingerprint(model_parts)
        after_transfer = rng_hashes(rng_snapshot(torch))
        campaign.require(after_transfer == before, "The weight transfer changed a native RNG state")
        campaign.require(len(parameters) == PARAMETERS and all(
            parameters[name]["dtype"] == "torch.float32" and parameters[name]["scope"] == "logical"
            for name in self.names
        ), "Initial fingerprint coverage or selected master dtype differs")
        self.initial = {
            "ema_selected_local_sha256": ema,
            "rng_before_transfer": before,
            "rng_after_transfer": after_transfer,
            "selection_sha256": campaign.digest(self.selection),
            "checkpoint_metadata_sha256": campaign.digest(folder / ".metadata"),
        }
        rng_file = self.output / "rng-control" / f"rank-{self.rank:02d}.pt"
        if self.control:
            with rng_file.open("xb") as stream:
                torch.save(native_rng, stream)
            self.initial["rng_at_native_start"] = rng_hashes(rng_snapshot(torch))
            campaign.require(self.initial["rng_at_native_start"] == before, "RNG snapshot changed the control")
        else:
            baseline = campaign.read_json(self.output / "ep8-a.json")
            left = baseline["parameters_by_rank"][self.rank]
            differences = [
                name for name in sorted(left.keys() | parameters.keys())
                if left.get(name) != parameters.get(name)
            ]
            original = baseline["native_by_rank"][self.rank]["initial"]
            campaign.require(campaign.digest(rng_file) == original["rng_snapshot_sha256"],
                             "The trusted control RNG snapshot changed")
            expected_rng = torch.load(rng_file, map_location="cpu", weights_only=True)
            campaign.require(rng_hashes(expected_rng) == original["rng_at_native_start"],
                             "The restored RNG fingerprint differs from the control receipt")
            if before != original["rng_at_native_start"]:
                restore_rng(torch, expected_rng)
            self.initial["rng_at_native_start"] = rng_hashes(rng_snapshot(torch))
            for key in ("rng_at_native_start", "selection_sha256", "checkpoint_metadata_sha256"):
                if original[key] != self.initial[key]:
                    differences.append(key)
            all_differences = [None] * campaign.WORLD
            dist.all_gather_object(all_differences, differences)
            campaign.require(
                not any(all_differences),
                f"Native start gate failed before training (by rank): {all_differences}",
            )
        self.initial["rng_snapshot_sha256"] = campaign.digest(rng_file)
        return parameters

    def preclip_gradients(self, trainer):
        import torch
        import torch.distributed as dist

        (model,) = trainer.model_parts
        layer = {
            name: parameter.grad
            for wrapped, parameter in model.named_parameters(remove_duplicate=False)
            if (name := wrapped.replace("_checkpoint_wrapped_module.", "")).startswith(
                initial_weights.LAYER0_PREFIX
            )
        }
        manifest = {
            name: None if grad is None else {"dtype": str(grad.dtype), "shape": list(grad.shape)}
            for name, grad in sorted(layer.items())
        }
        manifests = [None] * campaign.WORLD
        dist.all_gather_object(manifests, manifest)
        campaign.require(
            len(manifest) == LAYER0_PARAMETERS and all(value == manifest for value in manifests)
            and all(manifest[name] and manifest[name]["dtype"] == "torch.float32" for name in self.names),
            "The pre-clipping layer-0 FP32 gradient inventory differs across ranks",
        )
        if self.output_rank and self.control:
            (self.output / "preclip-fp32-control").mkdir()
        result = {}
        with torch.no_grad():
            for name, grad in sorted(layer.items()):
                if grad is None:
                    result[name] = None
                    continue
                campaign.require(grad.dtype == torch.float32, f"Expected an FP32 gradient for {name}")
                full = grad.full_tensor() if hasattr(grad, "full_tensor") else grad
                if self.output_rank:
                    path = self.output / "preclip-fp32-control" / f"{name}.fp32"
                    result[name] = manifest[name] | gradient_file(full, path, control=self.control)
                del full
        return manifest, result

    def observe(self, event, trainer, fields):
        from torchtitan.callbacks import EMA, Event

        if event is Event.after_load:
            ema = trainer.callbacks.find(EMA)
            campaign.require(
                self.initial is not None and trainer.step == 0
                and not trainer.config.checkpoint.enable
                and trainer.config.parallelism.gfsdp_split_layer0_ffn == (not self.control)
                and ema is not None and ema.store.use_ema and ema.config.start_step == 10000,
                "Expected the fresh native checkpoint and live EMA settings",
            )
        elif event is Event.after_forward:
            campaign.require(trainer.step == 1 and fields["microbatch_idx"] == len(self.microbatches),
                             "Unexpected native microbatch order")
            loss = fields["loss"]
            campaign.require(loss is not None and loss.numel() == 1, "Missing native microbatch loss")
            self.microbatches.append({
                "inputs": {name: tensor_record(value) for name, value in sorted(fields["input_dict"].items())},
                "labels": tensor_record(fields["labels"]),
                "loss": float(loss.detach().item()),
                "loss_tensor": tensor_record(loss),
            })
        elif event is Event.before_optimizer_step:
            campaign.require(
                trainer.step == 1 and not self.finished
                and len(self.microbatches) == trainer.gradient_accumulation_steps,
                "Expected complete native first-update microbatch coverage before clipping",
            )
            inventory, gradients = self.preclip_gradients(trainer)
            campaign.write_json(self.output / f"{self.arm}-rank-{self.rank:02d}.json", {
                "arm": self.arm, "rank": self.rank, "initial": self.initial,
                "run_id": Path(trainer.config.dump_folder).parent.name,
                "global_valid_tokens": int(fields["global_valid_tokens"]),
                "microbatches": self.microbatches, "gradient_inventory": inventory,
                "preclip_fp32_gradients": gradients,
            })
            self.finished = True


def worker(argv):
    capture = WorkerCapture()
    fingerprint = initial_weights.parameter_fingerprints
    with ExitStack() as patches:
        if os.environ.get(ENV_PREFIX + "PROBE") == "1":
            install_initializer_probe(patches, capture.rank)

        def prepare(model_parts):
            from torchtitan.callbacks import CallbackManager

            emit = CallbackManager.emit

            def observe_after_emit(manager, event, trainer, **fields):
                result = emit(manager, event, trainer, **fields)
                capture.observe(event, trainer, fields)
                return result

            patches.enter_context(patch.object(CallbackManager, "emit", observe_after_emit))
            return capture.prepare(model_parts, fingerprint)

        patches.enter_context(patch.multiple(initial_weights, SCRIPT=SCRIPT, parameter_fingerprints=prepare))
        initial_weights.worker(argv)
        campaign.require(capture.finished, "Native pre-clipping gradient capture did not run")


def install_initializer_probe(patches, rank):
    load = campaign.load_module

    def load_bridge(path, name):
        bridge = load(path, name)
        if name != "_initial_weights_flash_bridge":
            return bridge
        main = bridge.main

        def launch(argv):
            # initial_weights.worker has already installed its native adapter.
            base = bridge._load_base_measure()
            enter = base.NativeAdapter.__enter__

            def enter_with_probe(adapter):
                result = enter(adapter)
                probe = load(SCRIPT.with_name("initial_rng_probe.py"), "_matched_layer0_rng_probe")
                output = Path(adapter.options["output_dir"])
                patches.enter_context(probe.capture(output / f"initial-rng-rank-{rank:02d}.json"))
                return result

            with patch.object(base.NativeAdapter, "__enter__", enter_with_probe):
                return main(argv)

        patches.enter_context(patch.object(bridge, "main", launch))
        return bridge

    patches.enter_context(patch.object(campaign, "load_module", load_bridge))


class MatchedWeightsCase(initial_weights.InitialWeightsCase):
    def __init__(self, args, ep, arm):
        super().__init__(args, ep, arm)
        self.selection = args.selection.resolve(strict=True)
        read_selection(self.selection)
        self.output = args.campaign.resolve() / "initial-weights"
        self.initial_rng_probe = args.initial_rng_probe
        if self.initial_rng_probe:
            SCRIPT.with_name("initial_rng_probe.py").resolve(strict=True)

    def launch(self, run_id, timeout):
        control = self.arm == "a"
        if control:
            campaign.require(not (self.output / "checkpoint").exists(),
                             "Use a fresh campaign for the control checkpoint")
            (self.output / "rng-control").mkdir()
        else:
            baseline = campaign.read_json(self.output / "ep8-a.json")
            initial = baseline["native_by_rank"][0]["initial"]
            campaign.require(
                initial["selection_sha256"] == campaign.digest(self.selection)
                and initial["checkpoint_metadata_sha256"] == campaign.digest(self.output / "checkpoint/.metadata"),
                "The control checkpoint or FQN selection changed before the split arm",
            )
        with patch.dict(os.environ, {
            ENV_PREFIX + "ARM": self.arm,
            ENV_PREFIX + "OUTPUT": str(self.output),
            ENV_PREFIX + "SELECTION": str(self.selection),
            ENV_PREFIX + "PROBE": str(int(self.initial_rng_probe)),
        }):
            return super().launch(run_id, timeout)

    def summarize(self, attempt):
        result = super().summarize(attempt)
        rows = [
            campaign.read_json(self.output / f"{self.arm}-rank-{rank:02d}.json")
            for rank in range(campaign.WORLD)
        ]
        campaign.require(
            all(row["rank"] == rank and row["arm"] == self.arm and row["run_id"] == attempt.name
                and row["microbatches"]
                and row["initial"]["selection_sha256"] == campaign.digest(self.selection)
                for rank, row in enumerate(rows)),
            "Missing or stale native first-update evidence",
        )
        if self.initial_rng_probe:
            for rank, row in enumerate(rows):
                path = attempt / "measurement" / f"initial-rng-rank-{rank:02d}.json"
                receipt = campaign.read_json(path)
                campaign.require(receipt["rank"] == str(rank), "Initializer probe rank mismatch")
                row["initial_rng_probe"] = {
                    "path": str(path), "sha256": campaign.digest(path), "status": receipt["status"],
                }
        return result | {"native_by_rank": rows}


def compare(original, split):
    initial = initial_weights.compare(original, split)
    campaign.require(
        initial["initial_weights_equal"] is True and initial["matched_parameter_count"] == PARAMETERS,
        "All 243 parameters must match before interpreting update 1",
    )
    loss_changes = []
    exact_losses = True
    aligned_rng_ranks = []
    for a, b in zip(original["native_by_rank"], split["native_by_rank"], strict=True):
        for key in ("rank", "global_valid_tokens", "gradient_inventory"):
            campaign.check_equal(a[key], b[key], key)
        campaign.check_equal(a["initial"]["rng_at_native_start"], b["initial"]["rng_at_native_start"],
                             f"rank {a['rank']} native starting RNG")
        if b["initial"]["rng_before_transfer"] != b["initial"]["rng_at_native_start"]:
            aligned_rng_ranks.append(b["rank"])
        campaign.require(a["microbatches"] and len(a["microbatches"]) == len(b["microbatches"]),
                         "Native microbatch counts differ")
        for left, right in zip(a["microbatches"], b["microbatches"], strict=True):
            for key in ("inputs", "labels"):
                campaign.check_equal(left[key], right[key], f"rank {a['rank']} native microbatch {key}")
            exact_losses &= left["loss_tensor"] == right["loss_tensor"]
            loss_changes.append(right["loss"] - left["loss"])
    a = original["native_by_rank"][0]["preclip_fp32_gradients"]
    b = split["native_by_rank"][0]["preclip_fp32_gradients"]
    campaign.require(a.keys() == b.keys() and len(a) == LAYER0_PARAMETERS,
                     "Layer-0 gradient FQNs differ")
    gradients = {}
    for name, expected in a.items():
        actual = b[name]
        if expected is None:
            campaign.require(actual is None, f"Gradient presence differs for {name}")
            gradients[name] = None
            continue
        campaign.require(actual["reference_sha256"] == expected["sha256"],
                         f"The original pre-clipping raw bytes changed for {name}")
        gradients[name] = actual | {"exact": actual["sha256"] == expected["sha256"]}
    return {
        "initial_weights": initial,
        "split_rng_aligned_ranks": aligned_rng_ranks,
        "native_microbatch_input_pairs_equal": len(loss_changes),
        "native_microbatch_losses_exact": exact_losses,
        "max_abs_native_microbatch_loss_delta": max(map(abs, loss_changes)),
        "sum_native_microbatch_loss_delta_all_ranks": math.fsum(loss_changes),
        "preclip_fp32_layer0_gradients_exact": all(row["exact"] for row in gradients.values() if row is not None),
        "preclip_fp32_layer0_gradients": gradients,
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["worker"]:
        return worker(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("campaign", type=Path)
    launch.add_argument("arm", choices=initial_weights.ARMS)
    launch.add_argument("--selection", type=Path, required=True, help="Initial-weight comparison.json")
    launch.add_argument("--initial-rng-probe", action="store_true", help="Record native initializer draw order")
    launch.add_argument("--preflight-only", action="store_true")
    campaign.add_native_args(launch)
    verify = commands.add_parser("compare")
    verify.add_argument("campaign", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "run":
        with patch.multiple(campaign, SCRIPT=SCRIPT, STEPS=1, WARMUP=0), patch.multiple(
            initial_weights, InitialWeightsCase=MatchedWeightsCase, SCRIPT=SCRIPT,
        ):
            return initial_weights.run(args)
    output = args.campaign / "initial-weights"
    original, split = (campaign.read_json(output / f"ep8-{arm}.json") for arm in initial_weights.ARMS)
    result = compare(original, split)
    path = output / "matched-comparison.json"
    campaign.write_json(path, result)
    print(f"PASS matched initial weights and native inputs ({path})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
