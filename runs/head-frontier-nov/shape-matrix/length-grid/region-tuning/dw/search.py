"""Tune the eight isolated dW writes for the length-grid C8 fixtures."""

import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes as c
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from statistics import median
import subprocess

import torch

from ce_bench import evaluate
from matrix_fixture import load_inputs
from torchtitan.components.loss import MTPLoss
from torchtitan.ops.exact_folded_ce import ExactFoldedCrossEntropy
from tuning.linear_policy import LinearPolicy
from tuning.lt.lt_bench import (
    EXACT_CE_SHA256, bit_mismatches, chain, check_beta, measure_chains,
    measure_single_arm_memory, screen, write_json,
)
from tuning.lt.lt_writer import make_dw_writer, native, operand_key
from tuning.lt_extended.search_c8 import (
    CAP_ALIGNMENT_A, CAP_CUSTOM_MAX, CLUSTER, CLUSTERS, CUSTOM, TILE, Neighborhood,
)

CHUNKS = 8
SEQUENCES = (32768, 65536, 131072, 262144)
MAX_EXTRA_PER_LAYOUT = 24
MAX_EXTRA_ATTEMPTS_PER_LAYOUT = 160


def file_hash(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def load_pair(root, model, seq, device):
    spec = importlib.util.spec_from_file_location("grid_replay", root / "length-grid/replay.py")
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    fixture = root / "inputs" / f"{model}.pt"
    base = load_inputs(model, fixture)
    period = base["labels"].size(1)
    step = seq // CHUNKS
    assert step % period == 0
    # Use the actual resize/shift rules on the full sequence. Encode the
    # underlying hidden source positions in a narrow proxy to verify the copies.
    source_positions = torch.arange(period).unsqueeze(-1).expand(*base["hidden"].shape[:-1], 1)
    proxy = {**base, "hidden": source_positions}
    proxy.pop("z_hidden", None)
    resized, valid = grid.resize_inputs(proxy, 1, seq, "cpu")
    mtp = MTPLoss._shift_labels(resized["labels"], 1, resized["positions"])
    checks = {}
    for name, value, dim in (
        ("labels", resized["labels"], 1),
        ("positions", resized["positions"], 1),
        ("shifted_mtp_labels", mtp, 1),
        ("hidden_source_positions", resized["hidden"], 2),
    ):
        pieces = value.chunk(CHUNKS, dim)
        checks[name] = [torch.equal(pieces[0], item) for item in pieces]
    if not all(all(values) for values in checks.values()):
        raise RuntimeError(f"Chunk inputs differ: {checks}")
    weight = base["weight"].to(device)
    hidden = grid.resize_hidden(base["hidden"], 1, step, device)
    x = hidden.view(-1, weight.size(1))
    labels = torch.stack((resized["labels"][:, :step], mtp[:, :step])).flatten().to(device)
    policy = LinearPolicy(root / "tuning/linear-plans.json")
    if policy.supports_shape(x, weight.T):
        logits = policy.run(x, weight.T)
        projection = policy.calls
    else:
        logits = torch.nn.functional.linear(x, weight)
        projection = "aten-fallback"
    route = ExactFoldedCrossEntropy(
        str(root / "native-ce/centered-folded-v1.so"), library_sha256=EXACT_CE_SHA256,
    )
    denominator = float(resized["denominator"])
    main, aux, _, dz = evaluate(route, logits.requires_grad_(), labels, base["head_weights"], denominator)
    if dz.data_ptr() != logits.data_ptr() or not route.scales_checked:
        raise RuntimeError("Expected the consuming BF16 CUDA CE backward")
    return (dz, x), {
        "fixture": str(fixture), "fixture_bytes": fixture.stat().st_size,
        "provenance": base["provenance"], "period_tokens": period,
        "physical_tokens_per_chunk": step, "chunk_equality": checks,
        "physical_tokens": seq, "valid_tokens": valid, "valid_mtp_tokens": int(mtp.ne(-100).sum()),
        "denominator": denominator, "head_weights": base["head_weights"],
        "projection_route": projection, "ce_terms_first_chunk": [float(main), float(aux)],
        "ce_sha256": EXACT_CE_SHA256, "grid_replay_sha256": file_hash(spec.origin),
    }


def explore_neighborhood(search, seeds):
    result = list(seeds)
    used = {(item.order, bytes(item.algorithm)) for item in seeds}
    records = {}
    for layout in search.layouts:
        stats = Counter()
        records[layout] = stats
        anchors = [item for item in seeds if item.order == layout][:4]
        for anchor in anchors:
            config = anchor.config
            if config["split_k"] != 1 or config["reduction"] != 0:
                continue
            required = [search.capability(anchor.algorithm, CAP_ALIGNMENT_A + i, c.c_uint32())
                        for i in range(4)]
            a, b, cd = search.alignments
            if layout == "column":
                a, b = b, a
            if any(got < need for got, need in zip((a, b, cd, cd), required, strict=True)):
                continue
            custom_max = search.capability(anchor.algorithm, CAP_CUSTOM_MAX, c.c_int32())
            tile, custom, cluster = (config[name] for name in ("tile", "custom", "cluster"))
            proposal = [(tile, custom, option) for option in CLUSTERS]
            proposal.extend((tile, option, cluster) for option in range(custom_max + 1))
            proposal.extend((option, custom, cluster) for option in search.supported_tiles(anchor.algorithm))
            for tile_id, custom_id, cluster_id in proposal:
                if stats["added"] >= MAX_EXTRA_PER_LAYOUT or stats["checked"] >= MAX_EXTRA_ATTEMPTS_PER_LAYOUT:
                    break
                algorithm = native._Algo.from_buffer_copy(bytes(anchor.algorithm))
                try:
                    for attr, value in ((TILE, c.c_uint32(tile_id)), (CUSTOM, c.c_uint32(custom_id)),
                                        (CLUSTER, c.c_uint16(cluster_id))):
                        native._set_attribute(search.set_config, algorithm, attr, value)
                    key = (layout, bytes(algorithm))
                    if key in used:
                        continue
                    used.add(key)
                    stats["checked"] += 1
                    candidate = search._candidate(layout, algorithm, -1, 32 << 20)
                except RuntimeError as error:
                    stats[str(error)] += 1
                    continue
                stats["added"] += 1
                result.append(candidate)
    return result, records


def active(writer, expected_calls):
    plan = writer.receipt()
    if plan["tuned_calls_beta0_beta1"] != expected_calls or len(plan["plans"]) != 1:
        raise RuntimeError(f"Unexpected writer activation: {plan}")
    if expected_calls == [0, 0]:
        entry = plan["plans"][0]
        if entry["pair"] != "native-generic":
            raise RuntimeError(f"Expected native generic dW: {plan}")
        fallback = next(iter(writer.plan_cache.values()))[0]
        chosen = next(iter(fallback._plans.values()))
        plan["native_workspace_bytes"] = chosen.workspace_bytes
        plan["native_order"] = "row" if chosen.layout.row_major else "column"
        plan["native_serialized_algorithms"] = [bytes(algo).hex() for algo in chosen.algorithms]
    return plan


def paired_comparison(timings, candidate):
    result = {}
    for metric, values in (
        ("wall_ms", lambda arm: [item["wall_ms"] for item in arm["eager"]]),
        ("graph_ms", lambda arm: arm["graph_ms"]),
    ):
        a, b, c_values = (values(timings[name]) for name in ("control", "control_repeat", candidate))
        repeats, delta_a, delta_b = list(zip(*[(y-x, z-x, z-y) for x, y, z in zip(a, b, c_values, strict=True)]))
        noise = median(abs(value) for value in repeats)
        result[metric] = {
            "repeat_minus_control_median_ms": median(repeats),
            "median_absolute_control_delta_ms": noise,
            "candidate_minus_control_median_ms": median(delta_a),
            "candidate_minus_repeat_median_ms": median(delta_b),
            "beats_both_fraction": sum(x < 0 and y < 0 for x, y in zip(delta_a, delta_b, strict=True)) / len(a),
        }
    return result


def elapsed_win(comparison):
    wall = comparison["wall_ms"]
    return (
        max(wall["candidate_minus_control_median_ms"], wall["candidate_minus_repeat_median_ms"])
        < -wall["median_absolute_control_delta_ms"] and wall["beats_both_fraction"] >= 0.75
    )


def timings(factories, operands, slot, stream, count):
    with ExitStack() as resources:
        arms = {name: factory() for name, factory in factories.items()}
        for writer in arms.values():
            resources.callback(writer.close)
        return measure_chains(arms, operands, slot, stream, count)


def run(args, stream):
    root = args.root
    case = f"{args.model}-c8-b1-s{args.seq}"
    pair, fixture = load_pair(root, args.model, args.seq, stream.device)
    operands = (pair,) * CHUNKS
    slot = torch.empty((pair[0].size(1), pair[1].size(1)), dtype=torch.float32, device=stream.device)
    shape, alignment = operand_key(*pair, slot)
    if shape[0] != args.seq // 4:
        raise RuntimeError(f"Unexpected folded dW shape: {shape}")
    catalog_path = root / "tuning/lt/results/calibrated-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    calibrated = [entry for entry in catalog["plans"] if tuple(entry["shape"]) == shape]
    if bool(calibrated) != (args.model == "100b" and args.seq == 32768):
        raise RuntimeError(f"Unexpected calibrated list: {calibrated}")
    calls = [1, 7] if calibrated else [0, 0]

    def control():
        return make_dw_writer(receipt=catalog)

    print(f"{case}: shape={shape} control={'calibrated' if calibrated else 'generic'} fixture={fixture}",
          flush=True)
    with ExitStack() as resources:
        search = Neighborhood(shape, alignment, stream, resources)
        seeds = search.candidates(32)
        candidates, extras = explore_neighborhood(search, seeds)
        print(f"{case}: initial={Counter(item.order for item in seeds)}, extras={extras}", flush=True)
        screens = screen(search, candidates, operands, slot, 3)
        current = control()
        beta_counts = check_beta(search, candidates, screens, current, operands, slot)
        current.close()
        ranks = [[i for i in screens[str(beta)]["ranked"] if beta_counts[str(beta)][i] == 0]
                 for beta in (0, 1)]
        print(f"{case}: bitwise screened counts={[len(rank) for rank in ranks]}", flush=True)
        shortlist = {}
        for layout in (None, "column", "row"):
            best = [next((i for i in rank if layout is None or candidates[i].order == layout), None)
                    for rank in ranks]
            if None not in best:
                shortlist[tuple(best)] = layout or "overall"
        if all(ranks):
            for alpha in (0, 1):
                if len(ranks[alpha]) > 1:
                    indices = [rank[int(beta == alpha)] for beta, rank in enumerate(ranks)]
                    shortlist.setdefault(tuple(indices), f"alternate-beta{alpha}")

        def receipt(indices):
            return {
                "format": 1, "case": case, "shape": list(shape), "alignments": list(alignment),
                "capability": list(torch.cuda.get_device_capability(stream.device)),
                "device_name": torch.cuda.get_device_name(stream.device),
                "cublasLt_version": search.version,
                "pair": [candidates[i].receipt() for i in indices],
            }

        plans = {name: receipt(indices) for indices, name in shortlist.items()}
        factories = {"control": control, "control_repeat": control}
        factories.update({name: lambda plan=plan: make_dw_writer(receipt=plan) for name, plan in plans.items()})
        expected = torch.empty_like(slot)
        expected.fill_(float("nan"))
        writer = control()
        chain(writer, operands, expected, stream)
        acts = {"control": active(writer, calls)}
        writer.close()
        nonfinite = sum(torch.count_nonzero(~block.isfinite()) for block in expected.split(2048))
        if int(nonfinite):
            raise RuntimeError("Non-finite output from the control")
        chain_parity = {}
        for name, factory in factories.items():
            arm = factory()
            slot.fill_(float("nan"))
            chain(arm, operands, slot, stream)
            chain_parity[name] = int(bit_mismatches(slot, expected))
            acts[name] = active(arm, calls if name.startswith("control") else [1, 7])
            arm.close()
        del expected
        if any(chain_parity[name] for name in ("control", "control_repeat")):
            raise RuntimeError("Control repeat failed chain parity")

        selected = None
        discovery = timings({name: fn for name, fn in factories.items() if chain_parity[name] == 0},
                            operands, slot, stream, 8)
        compared = {name: paired_comparison(discovery, name) for name in plans if name in discovery}
        finalist = min(compared, key=lambda name: max(
            compared[name]["wall_ms"]["candidate_minus_control_median_ms"],
            compared[name]["wall_ms"]["candidate_minus_repeat_median_ms"],
        ), default=None)
        confirmation = {}
        confirmation_delta = {}
        stages = []
        if finalist:
            print(f"{case}: discovery finalist={finalist} delta={compared[finalist]}", flush=True)
            confirmation = timings({name: factories[name] for name in ("control_repeat", "control", finalist)},
                                   operands, slot, stream, 12)
            confirmation_delta = paired_comparison(confirmation, finalist)
            expected = torch.empty_like(slot)
            expected.fill_(float("nan"))
            slot.fill_(float("nan"))
            with ExitStack() as final_resources:
                ref, arm = control(), factories[finalist]()
                final_resources.callback(ref.close)
                final_resources.callback(arm.close)
                for index in range(CHUNKS):
                    ref(*pair, expected, index > 0, stream)
                    arm(*pair, slot, index > 0, stream)
                    stages.append(int(bit_mismatches(slot, expected)))
            del expected
            if not any(stages) and elapsed_win(compared[finalist]) and elapsed_win(confirmation_delta):
                selected = finalist
                plan = {
                    **plans[selected], "selected": selected,
                    "numerics": {"control_bit_mismatches": chain_parity[selected],
                                 "per_write_bit_mismatches": stages},
                }
                write_json(args.out / f"{case}-plan.json", plan)
        memory = measure_single_arm_memory(
            {name: fn for name, fn in factories.items() if name in discovery}, operands, slot, stream,
        )
        result = {
            "case": case, "shape": shape, "alignments": alignment, "data": fixture,
            "operand_bytes": [tensor.numel() * tensor.element_size() for tensor in pair],
            "output_bytes": slot.numel() * slot.element_size(), "allocated_unique_operand_pairs": 1,
            "cuda": torch.version.cuda, "torch": torch.__version__, "cublasLt_version": search.version,
            "device_name": torch.cuda.get_device_name(stream.device), "device_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "driver": subprocess.check_output(
                ["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"], "--query-gpu=driver_version",
                 "--format=csv,noheader"], text=True,
            ).strip(),
            "native_source_sha256": file_hash(native.__file__),
            "catalog_sha256": file_hash(catalog_path), "script_sha256": file_hash(__file__),
            "control": "grid-calibrated" if calibrated else "grid-native-generic",
            "math": "BF16 inputs, FP32 compute and output, allowed split reductions retain FP32",
            "heuristics": dict(Counter(item.order for item in seeds)), "neighborhood": extras,
            "candidates": [item.receipt() for item in candidates], "screen": screens,
            "per_beta_bit_mismatches": beta_counts, "shortlist": plans, "activation": acts,
            "chain_bit_mismatches": chain_parity, "finalist_per_write_bit_mismatches": stages,
            "discovery": discovery, "discovery_deltas": compared,
            "confirmation": confirmation, "confirmation_deltas": confirmation_delta,
            "isolated_memory": memory, "finalist": finalist, "selected": selected,
        }
        write_json(args.out / f"{case}.json", result)
        print(f"{case}: finalist={finalist} selected={selected} confirmation={confirmation_delta}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("100b", "700b"))
    parser.add_argument("seq", type=int, choices=SEQUENCES)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one H100")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    with torch.cuda.stream(torch.cuda.Stream()):
        run(args, torch.cuda.current_stream())


if __name__ == "__main__":
    main()
