#!/usr/bin/env python3
"""Locate the first changed logical layer-zero gradient bytes around NCCL."""

import argparse
import hashlib
import importlib.util
from itertools import accumulate, product
import json
import os
from pathlib import Path
import struct
import sys
from unittest.mock import patch


DRIVER = "ca0b10618427c5ed261bb6b56957e65956ccd9161413116577a5ac58e76023d1"
VIEWS = {
    "torchtitan/distributed/gfsdp/_collectives.py": "4d1755ac5977fc1afe12a89750a6c5862741b1d34e079cea62ba1e2500849800",
    "torchtitan/distributed/gfsdp/_param.py": "fb46af2f3a87db09c1724f09878574fbdbe35dea0bd11cc75f734c33b7c0c8c2",
}
CASES = {"captured": None, "seeded-2048": 0}
ARMS = ("A", "AA", "B")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def layout(input_numel, output_numel, rank, sizes_and_shards):
    """Translate a logical parameter into the two flat NCCL tensor views."""
    require(output_numel > 0 and input_numel == 8 * output_numel and 0 <= rank < 8,
            "Expected a padded input and eight equal output shards")
    offsets = list(accumulate((item[0] for item in sizes_and_shards), initial=0))
    require(offsets[-1] <= input_numel, "Logical parameters exceed the packed input")
    begin, end = rank * output_numel, (rank + 1) * output_numel
    result = []
    for packed, (size, logical_offset, local_numel) in zip(offsets[:-1], sizes_and_shards, strict=True):
        lo, hi = max(begin, packed), min(end, packed + size)
        require(local_numel == max(0, hi - lo), "Ragged local size disagrees with the NCCL slice")
        if local_numel:
            require(logical_offset == lo - packed, "Ragged logical offset disagrees with the NCCL slice")
        result.append((packed, size, logical_offset, local_numel, lo - begin if local_numel else 0))
    return result


def differing_words(left, right, packed, start, limit=8):
    """Retain actual operands at a bounded set of changed logical indices."""
    left, right = (value.reshape(-1).view(packed.dtype) for value in (left, right))
    samples = []
    for begin in range(0, left.numel(), 1 << 20):
        end = min(left.numel(), begin + (1 << 20))
        indices = (left[begin:end] != right[begin:end]).nonzero().flatten()[:limit - len(samples)]
        for index in (indices + begin).tolist():
            words = (left[index], right[index], packed[start + index])
            samples.append([index, *(f"{int(word) & 0xffffffff:08x}" for word in words)])
        if len(samples) == limit:
            break
    return samples


def observed_vjp(original, driver, harness, model, component, tensors, reference):
    import torch
    from torch import distributed as dist
    from torchtitan.distributed.gfsdp import _param_group as pg
    from torchtitan.distributed.gfsdp._tensor import get_ragged_shard_placement

    reduce = pg.reduce_scatter
    records = []

    def observed_reduce(params, grads, *args, **kwargs):
        require(len(grads) == len(params), "The component must produce all parameter gradients")
        exchange = dist.reduce_scatter_tensor

        def capture(output, input, **options):
            group = options["group"]
            rank = group.rank()
            require(group.size() == 8 and rank == dist.get_rank() and dist.get_backend(group) == "nccl"
                    and input.dtype == output.dtype == torch.float32 and not options.get("async_op"),
                    "Expected ordered eight-rank NCCL FP32 reduce-scatter")
            specs = [get_ragged_shard_placement(p.sharded_param._spec) for p in params]
            ranges = layout(input.numel(), output.numel(), rank, [
                (p._unsharded_accumulated_grad.numel(), spec.global_offset, spec.local_numel)
                for p, spec in zip(params, specs, strict=True)
            ])
            for p, (start, size, *_rest) in zip(params, ranges, strict=True):
                view = p._unsharded_accumulated_grad
                require(view.dtype == torch.float32 and view.is_contiguous()
                        and view.data_ptr() == input.data_ptr() + 4 * start
                        and size == view.numel(), "Parameter does not alias the claimed packed gradient")
            # These clones run on the existing reduction stream, at the actual
            # NCCL boundary. Hashing runs only after the component VJP finishes.
            before = input.detach().clone()
            result = exchange(output, input, **options)
            after = output.detach().clone()
            op = options["op"]
            records.append((params, ranges, before, after, {
                "input_numel": input.numel(), "output_numel": output.numel(),
                "send_address_mod_4096": input.data_ptr() % 4096,
                "receive_address_mod_4096": output.data_ptr() % 4096,
                "send_storage_offset_elements": input.storage_offset(),
                "receive_storage_offset_elements": output.storage_offset(),
                "receive_aliases_input": output.untyped_storage().data_ptr() == input.untyped_storage().data_ptr(),
                "op": getattr(op, "op", op).name,
                "members": [harness.canonical(p._param_fqn) for p in params],
            }))
            return result

        with patch.object(dist, "reduce_scatter_tensor", capture):
            return reduce(params, grads, *args, **kwargs)

    with patch.object(pg, "reduce_scatter", observed_reduce):
        final = original(harness, model, component, tensors)
    torch.cuda.synchronize()
    parameters, calls = {}, []
    for params, ranges, before, after, metadata in records:
        call = len(calls)
        calls.append(metadata)
        before, after = before.cpu(), after.cpu()
        for param, (start, size, logical, local_size, out_start) in zip(params, ranges, strict=True):
            name = harness.canonical(param._param_fqn)
            require(name not in parameters and "grad/" + name in final, f"Unexpected reduction coverage for {name}")
            final_local = param.sharded_param.grad._local_tensor.detach().cpu()
            require(final_local.dtype == torch.float32 and final_local.numel() == local_size,
                    f"Unexpected final local FP32 shard for {name}")
            parameters[name] = {
                "call": call, "logical_numel": size, "input_offset": start,
                "logical_local_offset": logical, "local_numel": local_size, "output_offset": out_start,
                "pre_nccl_sha256": driver.digest(before.narrow(0, start, size)),
                "raw_nccl_shard_sha256": driver.digest(after.narrow(0, out_start, local_size)),
                "final_local_shard_sha256": driver.digest(final_local),
            }
            if reference is not None:
                key = "grad/" + name
                parameters[name]["sampled_words"] = differing_words(
                    reference[key], final[key], before.view(torch.int32), start
                )
    require(set(parameters) == {name[5:] for name in final if name.startswith("grad/")},
            "Native FP32 reduce-scatter observations must cover every logical parameter")
    return final, {"calls": calls, "parameters": parameters}


def load_driver(path):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == DRIVER, f"Unexpected FFN driver: {path}")
    spec = importlib.util.spec_from_file_location("reviewed_ffn_bitwise", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def ascending_ring_words(words, destination):
    """One candidate order for an ascending NCCL ring, rounded at each addition."""
    order = [words[(destination + offset) % 8] for offset in range(1, 9)]
    value, *remaining = (struct.unpack(">f", bytes.fromhex(word))[0] for word in order)
    for term in remaining:
        value = struct.unpack(">f", struct.pack(">f", value + term))[0]
    return struct.pack(">f", value).hex()


def sample_report(stages, name, pre_diff):
    samples = []
    observed = [stage["B"]["sampled_words"] for stage in stages]
    for donor_rows in zip(*observed, strict=True):
        index, control, candidate = donor_rows[0][:3]
        require(all(row[:3] == [index, control, candidate] for row in donor_rows),
                f"Sampled logical outputs differ across ranks for {name}")
        words = [row[3] for row in donor_rows]
        destinations = {}
        for arm in ("A", "B"):
            (rank,) = [rank for rank, stage in enumerate(stages)
                       if 0 <= index - stage[arm]["logical_local_offset"] < stage[arm]["local_numel"]]
            destinations[arm] = rank
        predictions = ({arm: ascending_ring_words(words, rank) for arm, rank in destinations.items()}
                       if not pre_diff else None)
        samples.append({"logical_index": index, "observed_fp32_words": {"A": control, "B": candidate},
                        "candidate_donor_fp32_words_rank_order": words, "destination_rank": destinations,
                        "ascending_ring_predicted_fp32_words": predictions})
    return samples


def summarize(output):
    rows = [json.loads((output / f"rank{rank}.json").read_text())["results"] for rank in range(8)]
    result = {}
    for case in CASES:
        params = set(rows[0]["reduction_boundary"]["A"][case]["parameters"])
        first_difference = {}
        for name in sorted(params):
            final_key = "grad/" + name
            controls = [row["arms"]["AA"]["cases"][case][final_key] for row in rows]
            candidates = [row["arms"]["B"]["cases"][case][final_key] for row in rows]
            require(all(r == controls[0] for r in controls) and all(r == candidates[0] for r in candidates),
                    f"Cross-rank logical output comparison differs for {name}")
            stages = [{arm: row["reduction_boundary"][arm][case]["parameters"][name] for arm in ARMS}
                      for row in rows]
            for arm in ARMS:
                parts = sorted((stage[arm]["logical_local_offset"], stage[arm]["local_numel"])
                               for stage in stages if stage[arm]["local_numel"])
                expected = list(accumulate((size for _, size in parts), initial=0))
                require([offset for offset, _ in parts] == expected[:-1]
                        and all(stage[arm]["logical_numel"] == expected[-1] for stage in stages),
                        f"Rank-local NCCL shards do not exactly partition {arm} {name}")
            unstable = [rank for rank, stage in enumerate(stages)
                        if stage["A"] != stage["AA"]]
            pre_diff = [rank for rank, stage in enumerate(stages)
                        if stage["A"]["pre_nccl_sha256"] != stage["B"]["pre_nccl_sha256"]]
            postprocess = [[rank, arm] for rank, stage in enumerate(stages) for arm in ARMS
                           if stage[arm]["raw_nccl_shard_sha256"] != stage[arm]["final_local_shard_sha256"]]
            if unstable or not controls[0]["bitwise_equal"]:
                origin = "unresolved_unstable_control"
            elif pre_diff:
                origin = "at_or_before_collective_input"
            elif candidates[0]["bitwise_equal"]:
                origin = "no_logical_difference"
            elif postprocess:
                origin = "unresolved_postprocessing"
            else:
                origin = "raw_nccl_reduce_scatter"
            first_difference[name] = {"first_difference": origin, "unstable_control_ranks": unstable,
                                      "pre_nccl_different_ranks": pre_diff, "raw_vs_final_shards": postprocess,
                                      "final_comparison": candidates[0], "samples": sample_report(stages, name, pre_diff)}
        result[case] = first_difference
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--component", choices=("layer0",), default="layer0")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffn-driver", type=Path, default=Path(__file__).resolve().parents[2]
                        / "all-final-bitwise-001/ffn/ffn_bitwise.py")
    parser.add_argument("--summarize", action="store_true", help="Classify completed rank receipts offline")
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.output), indent=2))
        return
    if args.source_root is None or args.fixture is None:
        parser.error("--source-root and --fixture are required for torchrun")
    rank_receipt = args.output / f"rank{int(os.environ['RANK'])}.json"
    require(not rank_receipt.exists(), f"Refusing to replace {rank_receipt}")
    driver = load_driver(args.ffn_driver.resolve(strict=True))
    sys.path.insert(0, str(args.ffn_driver.resolve().parent))
    driver.SOURCE = {**driver.SOURCE, **VIEWS}
    original_run, original_vjp = driver.run, driver.vjp

    def run(*args, **kwargs):
        labels = iter(product(ARMS, CASES))
        stages = {arm: {} for arm in ARMS}
        baseline = {}

        def vjp(harness, model, component, tensors):
            arm, case = next(labels)
            final, stage = observed_vjp(original_vjp, driver, harness, model, component, tensors,
                                        baseline[case] if arm == "B" else None)
            if arm == "A":
                baseline[case] = final
            stages[arm][case] = stage
            return final

        with patch.object(driver, "vjp", vjp):
            result = original_run(*args, **kwargs)
        require(all(set(cases) == set(CASES) for cases in stages.values()), "Incomplete component evidence")
        result["reduction_boundary"] = stages
        result["reduction_boundary_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    with patch.object(sys, "argv", [str(args.ffn_driver), "--component", args.component, "--source-root",
                                   str(args.source_root), "--fixture", str(args.fixture), "--output", str(args.output)]), \
            patch.multiple(driver, CASES=CASES, run=run):
        driver.main()


if __name__ == "__main__":
    main()
