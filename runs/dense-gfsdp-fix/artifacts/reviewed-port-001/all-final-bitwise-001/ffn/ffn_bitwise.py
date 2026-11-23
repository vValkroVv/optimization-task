#!/usr/bin/env python3
"""Compare full-shape native FFN component VJPs by raw tensor bytes."""

import argparse
import gc
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch


HEAD = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
HARNESSES = {
    "ffn_component.py": "e3dc86401d19e2eb010c6db66679d20512a540a04b40eaa3d7d9278a1404e098",
    "ffn_components.py": "38ad5eabf08582f8a1f55701238970dfc6ce5c5a60b24bad3d64fd6367bdcb37",
}
SOURCE = {
    "torchtitan/distributed/ffn_overlap.py": "26dbad929489bc195203cc4878c9d93634c85103260aa25e5192bde99dea23c5",
    "torchtitan/distributed/compile.py": "b6e91e2768fcb7a8181b0d166fd46406e9665f41afbc81463f0e186174e7e6a2",
    "torchtitan/distributed/gfsdp/_param_group.py": "2c090a48484b4cbb2d31c5fe96e2b368710d9a91767a03d6c8e527de5b48203d",
    "torchtitan/distributed/gfsdp/_state.py": "0778902cd42be236b608eb4fe35b878d5aaee49c87b3c2377345685af8dac9cf",
    "torchtitan/models/gift_moe/_base/parallelize.py": "e4acbf6a3f098b2aa15cafca5fbda21573df082c1ea3843489cf598ca976ad21",
}
CASES = {"captured": None, "seeded-2048": 0, "seeded-4099": 2051}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_file(path, digest):
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Changed file: {path}"
    )


def digest(tensor):
    import torch

    data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def differences(reference, actual):
    import torch

    require(reference.keys() == actual.keys(), "Tensor coverage differs")
    result = {}
    for name, left in reference.items():
        right = actual[name]
        require(
            left.shape == right.shape and left.dtype == right.dtype,
            f"Layout differs: {name}",
        )
        fint = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[left.dtype]
        count = nonfinite = 0
        maximum = 0.0
        for expected, observed in zip(
            left.reshape(-1).split(1048576),
            right.reshape(-1).split(1048576),
            strict=True,
        ):
            count += int((expected.view(fint) != observed.view(fint)).sum())
            nonfinite += int(
                (~torch.isfinite(expected) | ~torch.isfinite(observed)).sum()
            )
            maximum = max(
                maximum, float((expected.double() - observed.double()).abs().max())
            )
        result[name] = {
            "dtype": str(left.dtype),
            "shape": list(left.shape),
            "bitwise_equal": count == 0,
            "unequal_elements": count,
            "nonfinite_elements": nonfinite,
            "max_abs": maximum if not nonfinite else None,
        }
    return result


def logical_weights(harness, model):
    import torch

    weights = {
        harness.canonical(name): harness.cpu_tensor(param)
        for name, param in model.named_parameters()
    }
    require(
        weights and all(value.dtype == torch.float32 for value in weights.values()),
        "Expected FP32 masters",
    )
    return weights


def restore_weights(harness, model, reference, changed):
    import torch
    from torch.distributed.tensor import Replicate
    from torchtitan.distributed.gfsdp._tensor import (
        RaggedShard,
        RaggedShardDTensor,
        get_ragged_shard_placement,
    )

    if not changed:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            name = harness.canonical(name)
            if name not in changed:
                continue
            require(
                isinstance(param, RaggedShardDTensor)
                and all(
                    isinstance(axis, (RaggedShard, Replicate))
                    for axis in param._spec.placements
                ),
                f"Unsupported non-dense checkpoint layout: {name}",
            )
            spec = get_ragged_shard_placement(param._spec)
            master = reference[name].reshape(-1)
            local = param.to_local()
            require(
                local.dtype == master.dtype
                and local.ndim == 1
                and local.numel() == spec.local_numel
                and sum(spec.shard_numels) == master.numel(),
                f"Invalid local FP32 master slice: {name}",
            )
            local.copy_(master.narrow(0, spec.global_offset, spec.local_numel))


def inventory(harness, model, component, split):
    unit = model.layers["0"] if component == "layer0" else model.mtp.layers["0"]
    block = getattr(unit, "block", unit)
    child = (
        "layers.0.feed_forward"
        if component == "layer0"
        else "mtp.layers.0.block.feed_forward"
    )
    buffers = harness.buffer_inventory(model)
    actual = {row["kind"] for row in buffers["views"] if row["fqn"] == child}
    require(
        bool(getattr(unit, "_ffn_overlap_regions", ())) == split
        and actual == ({"data", "grad"} if split else set()),
        f"Incorrect compiled FFN split or GFSDP allocation for {child}",
    )
    require(
        all(
            type(getattr(block.feed_forward, role)).__name__ == "FP8Linear"
            for role in ("w1", "w2", "w3")
        ),
        f"Inactive native FP8 FFN: {child}",
    )
    return {"child": child, "split": split, "buffers": buffers}


def vjp(harness, model, component, tensors):
    import torch

    model.zero_grad(set_to_none=True)
    output = harness.microbatch(model, component, tensors)
    activations = {"output": output, "hidden_grad": tensors[0].grad}
    if component == "mtp":
        activations["embedding_grad"] = tensors[1].grad
    require(
        all(
            value is not None and value.dtype == torch.bfloat16
            for value in activations.values()
        ),
        "A BF16 output or input gradient is missing",
    )
    result = {key: harness.cpu_tensor(value) for key, value in activations.items()}
    for name, param in model.named_parameters():
        require(
            param.grad is not None and param.grad.dtype == torch.float32,
            f"Missing logical pre-clip FP32 gradient: {name}",
        )
        result["grad/" + harness.canonical(name)] = harness.cpu_tensor(param.grad)
    return result


def run(harness, fixtures, source_root, args, dims, device, rank):
    import subprocess
    import torch
    import torchtitan

    require(
        Path(torchtitan.__file__)
        .resolve()
        .is_relative_to(source_root / "gift/torchtitan"),
        "Native torchtitan came from a different checkout",
    )
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip()
        == HEAD,
        "Wrong final source HEAD",
    )
    initial = initial_digests = rng = None
    baseline, answers, fixture = {}, {}, None
    for arm in ("A", "AA", "B"):
        torch._dynamo.reset()
        cfg = harness.resolved_config(args.component, arm == "B")
        cfg.dump_folder = str(args.output)
        model = harness.build(cfg, args.component, dims, device)
        active = inventory(harness, model, args.component, arm == "B")
        weights = logical_weights(harness, model)
        seen = {name: digest(value) for name, value in weights.items()}
        if arm == "A":
            initial, initial_digests = weights, seen
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device))
        else:
            require(
                seen.keys() == initial_digests.keys(), f"Initial FQNs differ for {arm}"
            )
            changed = [name for name in seen if seen[name] != initial_digests[name]]
            if changed:
                restore_weights(harness, model, initial, changed)
                weights = logical_weights(harness, model)
                seen = {name: digest(value) for name, value in weights.items()}
            require(
                seen == initial_digests, f"Logical FP32 initial bytes differ for {arm}"
            )
            active["restored_initial_fqns"] = changed
        del weights
        active["initial_tensors_sha256"] = seen
        answers[arm] = {"activation": active, "cases": {}}
        for case, offset in CASES.items():
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1], device)
            if offset is None:
                tensors, fixture = harness.captured_inputs(
                    fixtures, device, rank, args.component
                )
            else:
                tensors = harness.inputs(device, rank, offset, args.component)
            observed = vjp(harness, model, args.component, tensors)
            if arm == "A":
                baseline[case] = observed
                answers[arm]["cases"][case] = {
                    name: digest(value) for name, value in observed.items()
                }
            else:
                answers[arm]["cases"][case] = differences(baseline[case], observed)
            del tensors, observed
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "scope": "Whole selected FP8 block VJP; full logical pre-clip gradients; no optimizer update",
        "fixture": fixture,
        "synthetic_seed_rule": "2048 + rank * 1000 + case offset",
        "case_offsets": CASES,
        "source_root": str(source_root),
        "arms": answers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("layer0", "mtp"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    location = Path(__file__).resolve().parent
    suite = root / "runs/ffn-row-owner"
    check_file(location / "ffn_component.py", HARNESSES["ffn_component.py"])
    check_file(suite / "ffn_components.py", HARNESSES["ffn_components.py"])
    for relative, expected in SOURCE.items():
        check_file(root / "gift" / relative, expected)
    sys.path.insert(0, str(suite))
    import ffn_components
    import ffn_component

    def numeric_check(config, dims, device, rank):
        return run(ffn_components, args.fixture, root, config, dims, device, rank)

    with patch.object(ffn_components, "numeric_check", numeric_check):
        ffn_component.main(
            [
                "--component",
                args.component,
                "--mode",
                "check",
                "--output",
                str(args.output),
            ]
        )


if __name__ == "__main__":
    main()
