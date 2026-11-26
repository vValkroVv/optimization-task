"""Calibrate the exact B1 folded dW chain on one H100."""
import argparse
from collections import Counter
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import torch

from ce_bench import evaluate
from matrix_fixture import load_inputs
from torchtitan.components.loss import MTPLoss
from torchtitan.ops.exact_folded_ce import ExactFoldedCrossEntropy
from tuning.linear_policy import LinearPolicy
from tuning.lt import lt_bench
from tuning.lt.lt_bench import EXACT_CE_SHA256, bit_mismatches, chain, check_beta, screen, write_json
from tuning.lt.lt_writer import make_dw_writer, native, operand_key
from tuning.lt_extended.search_c8 import Neighborhood


SEQUENCES = (4096, 8192, 16384, 32768, 65536, 131072, 262144)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_operands(root, model, seq, chunks, device):
    grid = load_module("grid_replay", root / "length-grid/replay.py")
    fixture = root / "inputs" / f"{model}.pt"
    base = load_inputs(model, fixture)
    period = base["labels"].size(1)
    step = seq // chunks
    source = torch.arange(period).unsqueeze(-1).expand(*base["hidden"].shape[:-1], 1)
    proxy = {**base, "hidden": source}
    proxy.pop("z_hidden", None)
    resized, valid = grid.resize_inputs(proxy, 1, seq, "cpu")
    mtp = MTPLoss._shift_labels(resized["labels"], 1, resized["positions"])
    weight = base["weight"].to(device)
    policy = LinearPolicy(root / "tuning/linear-plans.json")
    ce = ExactFoldedCrossEntropy(
        str(root / "native-ce/centered-folded-v1.so"), library_sha256=EXACT_CE_SHA256,
    )
    denominator = float(resized["denominator"])
    distinct = []
    operands = []
    details = []
    for chunk, (main, aux, positions, source_indices) in enumerate(zip(
        resized["labels"].chunk(chunks, 1), mtp.chunk(chunks, 1),
        resized["positions"].chunk(chunks, 1), resized["hidden"].chunk(chunks, 2), strict=True,
    )):
        inputs = (main, aux, positions, source_indices)
        same = next((i for i, prior in enumerate(distinct)
                     if all(torch.equal(x, y) for x, y in zip(inputs, prior[0], strict=True))), None)
        if same is not None:
            operands.append(distinct[same][1])
            details.append({"chunk": chunk, "reuse_chunk": distinct[same][2]})
            continue
        start = chunk * step % period
        if start == 0:
            hidden = grid.resize_hidden(base["hidden"], 1, step, device)
        else:
            if start + step > period:
                raise RuntimeError("Unexpected fixture chunk crossing a seed period")
            hidden = base["hidden"][:, :1, start:start + step].to(device).contiguous()
        x = hidden.view(-1, weight.size(1))
        labels = torch.stack((main, aux)).flatten().to(device)
        if policy.supports_shape(x, weight.T):
            logits = policy.run(x, weight.T)
            projection = dict(policy.calls)
        else:
            logits = torch.nn.functional.linear(x, weight)
            projection = "aten-fallback"
        loss, mtp_loss, _, dz = evaluate(ce, logits.requires_grad_(), labels, base["head_weights"], denominator)
        if dz.data_ptr() != logits.data_ptr() or not ce.scales_checked:
            raise RuntimeError("Expected the consuming BF16 CUDA CE route")
        pair = (dz, x)
        distinct.append((inputs, pair, chunk))
        operands.append(pair)
        details.append({
            "chunk": chunk, "seed_offset": start, "projection": projection,
            "ce_terms": [float(loss), float(mtp_loss)],
            "main_valid": int(main.ne(-100).sum()), "mtp_valid": int(aux.ne(-100).sum()),
        })
    return tuple(operands), {
        "fixture": str(fixture), "fixture_bytes": fixture.stat().st_size,
        "provenance": base["provenance"], "ce_sha256": EXACT_CE_SHA256,
        "period_tokens": period, "physical_tokens_per_chunk": step, "physical_tokens": seq,
        "valid_tokens": valid, "valid_mtp_tokens": int(mtp.ne(-100).sum()),
        "denominator": denominator, "head_weights": base["head_weights"],
        "allocated_unique_operand_pairs": len(distinct), "chunks": details,
    }


def memory_eligible(memory, name):
    baseline = memory["control"]
    candidate = memory[name]
    return (candidate["gift_total_gib"] <= baseline["gift_total_gib"]
            and candidate["allocator_retries"] == baseline["allocator_retries"] == 0)


def select_shortlist(candidates, ranks):
    shortlist = {}
    for layout in (None, "column", "row"):
        best = tuple(next((i for i in rank if layout is None or candidates[i].order == layout), None)
                     for rank in ranks)
        if None not in best:
            shortlist[best] = layout or "overall"
    if all(ranks):
        for beta in (0, 1):
            if len(ranks[beta]) > 1:
                alternate = tuple(rank[int(index == beta)] for index, rank in enumerate(ranks))
                shortlist.setdefault(alternate, f"alternate-beta{beta}")
    return shortlist


def verify_chain(old, factories, operands, slot, stream, calls):
    counts = [1, len(operands) - 1]
    expected = torch.empty_like(slot)
    expected.fill_(float("nan"))
    with ExitStack() as stack:
        baseline = factories["control"]()
        stack.callback(baseline.close)
        chain(baseline, operands, expected, stream)
        if int(sum(torch.count_nonzero(~block.isfinite()) for block in expected.split(2048))):
            raise RuntimeError("The control produced non-finite output")
        activation = {"control": old.active(baseline, calls)}
        parity = {"control": 0}
        for name, factory in factories.items():
            if name == "control":
                continue
            writer = factory()
            try:
                slot.fill_(float("nan"))
                chain(writer, operands, slot, stream)
                parity[name] = int(bit_mismatches(slot, expected))
                activation[name] = old.active(writer, calls if name == "control_repeat" else counts)
            finally:
                writer.close()
    if parity["control_repeat"]:
        raise RuntimeError("The identical control failed bitwise chain parity")
    return parity, activation


def verify_stages(control, finalist, operands, slot, stream):
    expected = torch.empty_like(slot)
    expected.fill_(float("nan"))
    slot.fill_(float("nan"))
    stages = []
    with ExitStack() as stack:
        ref, arm = control(), finalist()
        stack.callback(ref.close)
        stack.callback(arm.close)
        for index, pair in enumerate(operands):
            ref(*pair, expected, index > 0, stream)
            arm(*pair, slot, index > 0, stream)
            stages.append(int(bit_mismatches(slot, expected)))
    return stages


def run(args, case, stream, result):
    root = args.root
    prior = root / "length-grid/region-tuning/dw"
    old = load_module("region_dw", prior / "search.py")
    prior_catalog_path = Path(__file__).with_name("prior-c8.selected.json")
    catalog_path = root / "tuning/lt/results/calibrated-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    result["files_sha256"] = {
        str(path.relative_to(root)): old.file_hash(path)
        for path in (Path(__file__).resolve(), prior / "search.py", catalog_path,
                     prior_catalog_path, root / "length-grid/replay.py",
                     root / "tuning/linear-plans.json", Path(native.__file__), Path(lt_bench.__file__))
    }
    result["phase"] = "fixture"
    operands, fixture = load_operands(root, args.model, args.seq, args.chunks, stream.device)
    grad, hidden = operands[0]
    slot = torch.empty((grad.size(1), hidden.size(1)), dtype=torch.float32, device=stream.device)
    shape, alignment = operand_key(grad, hidden, slot)
    if shape[0] != 2 * args.seq // args.chunks:
        raise RuntimeError(f"Unexpected folded dW shape: {shape}")
    if any(operand_key(*pair, slot) != (shape, alignment) for pair in operands):
        raise RuntimeError("Inconsistent dW descriptors in the chain")
    calibrated = any(tuple(entry["shape"]) == shape for entry in catalog["plans"])
    calls = [1, args.chunks - 1] if calibrated else [0, 0]
    print(f"{case}: {shape}, {alignment}, {'calibrated' if calibrated else 'generic'}, {fixture}", flush=True)
    result.update(
        shape=shape, alignments=alignment, data=fixture,
        operand_bytes=[tensor.numel() * tensor.element_size() for tensor in operands[0]],
        output_bytes=slot.numel() * slot.element_size(),
        control="grid-calibrated" if calibrated else "grid-native-generic",
    )

    def control():
        return make_dw_writer(receipt=catalog)

    result["phase"] = "search"
    with ExitStack() as resources:
        search = Neighborhood(shape, alignment, stream, resources)
        result["cublasLt_version"] = search.version
        seeds = search.candidates(32)
        candidates, extras = old.explore_neighborhood(search, seeds)
        result["heuristics"] = dict(Counter(item.order for item in seeds))
        result["neighborhood"] = extras
        # The retained controls and admitted neighborhood plans have only four
        # workspace bytes. Larger variants must show no measured memory growth.
        if not candidates:
            raise RuntimeError("No candidate descriptor passed cuBLASLt checks")
        result["candidates"] = [item.receipt() for item in candidates]
        screened = screen(search, candidates, operands, slot, 3)
        result["screen"] = screened
        baseline = control()
        try:
            differences = check_beta(search, candidates, screened, baseline, operands, slot)
        finally:
            baseline.close()
        result["per_beta_bit_mismatches"] = differences
        ranks = [[i for i in screened[str(beta)]["ranked"] if not differences[str(beta)][i]]
                 for beta in (0, 1)]
        print(f"{case}: exact beta candidates {[len(rank) for rank in ranks]}", flush=True)
        plans = {}
        for indices, name in select_shortlist(candidates, ranks).items():
            plans[name] = {
                "format": 1, "case": case, "shape": list(shape), "alignments": list(alignment),
                "capability": list(torch.cuda.get_device_capability(stream.device)),
                "device_name": torch.cuda.get_device_name(stream.device),
                "cublasLt_version": search.version, "pair": [candidates[i].receipt() for i in indices],
            }
        prior_catalog = json.loads(prior_catalog_path.read_text())
        for entry in prior_catalog["plans"]:
            if tuple(entry["shape"]) == shape and not any(
                [p["algorithm"] for p in entry["pair"]] == [p["algorithm"] for p in plan["pair"]]
                and [p["order"] for p in entry["pair"]] == [p["order"] for p in plan["pair"]]
                for plan in plans.values()
            ):
                plans["prior-c8"] = entry

        result["phase"] = "chain-verification"
        factories = {"control": control, "control_repeat": control}
        factories.update({name: lambda plan=plan: make_dw_writer(receipt=plan) for name, plan in plans.items()})
        parity, activation = verify_chain(old, factories, operands, slot, stream, calls)
        result.update(shortlist=plans, chain_bit_mismatches=parity, activation=activation)
        eligible = {name: fn for name, fn in factories.items() if parity[name] == 0}
        memory = lt_bench.measure_single_arm_memory(eligible, operands, slot, stream)
        result["isolated_memory"] = memory
        eligible = {name: fn for name, fn in eligible.items()
                    if name.startswith("control") or memory_eligible(memory, name)}

        result["phase"] = "paired-timing"
        discovery = old.timings(eligible, operands, slot, stream, 8)
        compared = {name: old.paired_comparison(discovery, name) for name in plans if name in discovery}
        result.update(discovery=discovery, discovery_deltas=compared)
        finalists = sorted((name for name in compared if old.elapsed_win(compared[name])), key=lambda name: max(
            compared[name]["wall_ms"]["candidate_minus_control_median_ms"],
            compared[name]["wall_ms"]["candidate_minus_repeat_median_ms"],
        ))
        result.update(confirmation={}, confirmation_deltas={}, finalist_per_write_bit_mismatches={}, selected=None)
        if finalists:
            # Confirm the discovery winner with fresh writer/graph state.
            name = finalists[0]
            arms = {arm: eligible[arm] for arm in ("control_repeat", "control", name)}
            confirmation = old.timings(arms, operands, slot, stream, 12)
            differences = old.paired_comparison(confirmation, name)
            stages = verify_stages(control, factories[name], operands, slot, stream)
            result["confirmation"][name] = confirmation
            result["confirmation_deltas"][name] = differences
            result["finalist_per_write_bit_mismatches"][name] = stages
            if not any(stages) and old.elapsed_win(differences):
                result["selected"] = name
                plan = {
                    **plans[name], "case": case, "selected": name,
                    "numerics": {"control_bit_mismatches": parity[name], "per_write_bit_mismatches": stages},
                }
                write_json(args.out / f"{case}-plan.json", plan)
        print(f"{case}: selected={result['selected']}, confirmation={result['confirmation_deltas']}", flush=True)
    result.update(phase="finished", status="complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("100b", "700b"))
    parser.add_argument("chunks", type=int, choices=(2, 8))
    parser.add_argument("seq", type=int, choices=SEQUENCES)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    case = f"{args.model}-c{args.chunks}-b1-s{args.seq}"
    result = {"case": case, "phase": "setup", "argv": sys.argv, "torch": torch.__version__,
              "cuda": torch.version.cuda, "device_uuid": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one H100")
        torch.cuda.set_device(0)
        result["device_name"] = torch.cuda.get_device_name()
        result["device_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        with torch.cuda.stream(torch.cuda.Stream()):
            run(args, case, torch.cuda.current_stream(), result)
    except Exception as exc:
        oom = isinstance(exc, torch.OutOfMemoryError) and "cuda out of memory" in str(exc).lower()
        result.update(status="oom" if oom else "failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.out / f"{case}.json", result)


if __name__ == "__main__":
    main()
