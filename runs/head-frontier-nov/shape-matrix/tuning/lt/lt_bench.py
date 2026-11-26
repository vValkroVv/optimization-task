"""Search shape-general cuBLASLt algorithms and replay whole dW chains."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import logging
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import time

import torch

from ce_bench import evaluate
from matrix_fixture import load_inputs
from torchtitan.components.loss import MTPLoss
from torchtitan.distributed.gfsdp import _staged_head_dw_writer as native
from torchtitan.ops.exact_folded_ce import ExactFoldedCrossEntropy
from torchtitan.tools.utils import device_memory_usage
from tuning.lt.lt_writer import Search, make_dw_writer, operand_key, workspace_for


CASES = {"100b-c2": ("100b", 2), "100b-c8": ("100b", 8), "700b-c8": ("700b", 8)}
EXACT_CE_SHA256 = "17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8"
GIB = 1 << 30


def load_operands(case, fixture_path, exact_ce_path):
    model, chunks = CASES[case]
    record = load_inputs(model, fixture_path)
    hidden, weight, labels, positions = (
        record[key].cuda() for key in ("hidden", "weight", "labels", "positions")
    )
    targets = torch.stack((labels, MTPLoss._shift_labels(labels, 1, positions)))
    width = weight.size(1)
    route = ExactFoldedCrossEntropy(str(exact_ce_path), library_sha256=EXACT_CE_SHA256)
    operands = []
    step = hidden.size(2) // chunks
    for start in range(0, hidden.size(2), step):
        h = hidden[:, :, start : start + step].contiguous().view(-1, width)
        y = targets[:, :, start : start + step].contiguous().view(-1)
        logits = torch.nn.functional.linear(h, weight).requires_grad_()
        _, _, _, grad = evaluate(route, logits, y, record["head_weights"], float(record["denominator"]))
        if grad.data_ptr() != logits.data_ptr():
            raise RuntimeError("Benchmark did not obtain the native consumed BF16 logits gradient")
        operands.append((grad, h))
    if not route.scales_checked:
        raise RuntimeError("Benchmark did not reach the native CUDA CE backward")
    return operands, record["provenance"]


def record_graph(fn, stream):
    fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    graph.replay()
    graph.replay()
    stream.synchronize()
    return graph


def event_sample(fn, stream, events):
    start, end = events
    start.record(stream)
    fn()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end)


def timing_events():
    return [torch.cuda.Event(enable_timing=True) for _ in range(2)]


def screen(search, candidates, operands, slot, warmup):
    workspace = workspace_for(slot.device, *candidates)
    results = {}
    events = timing_events()
    for beta1 in (False, True):
        grad, hidden = operands[-1 if beta1 else 0]
        slot.zero_()
        values = {}
        eligible = {}
        for index, candidate in enumerate(candidates):
            def launch(candidate=candidate):
                search.launch(candidate, grad, hidden, slot, beta1, workspace)

            try:
                graph = record_graph(launch, search.stream)
            except RuntimeError as error:
                values[index] = {"error": str(error)}
                continue
            eligible[index] = graph
            values[index] = {"samples_ms": []}

        indices = list(eligible)
        rng = random.Random(int(beta1))
        for _ in range(warmup):
            rng.shuffle(indices)
            for index in indices:
                values[index]["samples_ms"].append(event_sample(eligible[index].replay, search.stream, events))
        for index in indices:
            values[index]["median_ms"] = median(values[index]["samples_ms"])
        ranked = sorted(eligible, key=lambda index: values[index]["median_ms"])
        if not ranked:
            raise RuntimeError(f"No cuBLASLt algorithm completed for beta={int(beta1)}")
        results[str(int(beta1))] = {"candidates": values, "ranked": ranked}
        del eligible
    return results


def bit_mismatches(actual, expected):
    return sum(torch.count_nonzero(a.view(torch.int32) != b.view(torch.int32))
               for a, b in zip(actual.split(2048), expected.split(2048), strict=True))


def check_beta(search, candidates, ranked, control, operands, slot):
    stream = search.stream
    workspace = workspace_for(slot.device, *candidates)
    expected = torch.empty_like(slot)
    seed = torch.empty_like(slot)
    control(*operands[0], seed, False, stream)
    differences = {}
    for add in (False, True):
        key = str(int(add))
        inputs = operands[-1 if add else 0]
        if add:
            expected.copy_(seed)
        else:
            expected.fill_(float("nan"))
        control(*inputs, expected, add, stream)
        counts = []
        for index in ranked[key]["ranked"]:
            if add:
                slot.copy_(seed)
            else:
                slot.fill_(float("nan"))
            search.launch(candidates[index], *inputs, slot, add, workspace)
            counts.append(bit_mismatches(slot, expected))
        differences[key] = dict(zip(ranked[key]["ranked"], torch.stack(counts).tolist(), strict=True))
    return differences


def chain(writer, operands, slot, stream):
    for index, (grad, hidden) in enumerate(operands):
        writer(grad, hidden, slot, index > 0, stream)


def numerical(arms, operands, slot, stream, control):
    """Compare every FP32 dW element to the strongest existing control."""
    expected = torch.empty_like(slot)
    expected.fill_(float("nan"))
    chain(arms[control], operands, expected, stream)
    if not torch.isfinite(expected.split(2048)[0]).all():
        raise RuntimeError("The control did not overwrite the NaN-initialized FP32 slot")

    oracle = torch.empty_like(expected, dtype=torch.float64)
    hidden_fp64 = [hidden.double() for _, hidden in operands]
    for offset in range(0, slot.size(0), 2048):
        out = oracle[offset : offset + 2048]
        for index, (grad, _) in enumerate(operands):
            contribution = grad[:, offset : offset + 2048].double().T
            torch.addmm(out, contribution, hidden_fp64[index], beta=float(index > 0), out=out)
    oracle_norm_sq = sum(block.square().sum() for block in oracle.split(2048))
    result = {}
    for name, writer in arms.items():
        slot.fill_(float("nan"))
        chain(writer, operands, slot, stream)
        sums = torch.zeros(3, dtype=torch.float64, device=slot.device)
        max_absolute = torch.zeros(2, dtype=torch.float64, device=slot.device)
        for actual, ref, baseline in zip(slot.split(2048), oracle.split(2048), expected.split(2048), strict=True):
            error = actual.double() - ref
            delta = actual.double() - baseline.double()
            sums += torch.stack((error.square().sum(), delta.square().sum(), (~actual.isfinite()).sum()))
            max_absolute = torch.maximum(max_absolute, torch.stack((error.abs().max(), delta.abs().max())))
        error_sq, delta_sq, nonfinite = sums.tolist()
        max_reference, max_control = max_absolute.tolist()
        result[name] = {
            "oracle_relative_l2": (error_sq / float(oracle_norm_sq)) ** 0.5,
            "oracle_max_abs": max_reference,
            "control_bit_mismatches": int(bit_mismatches(slot, expected)),
            "control_l2": delta_sq ** 0.5,
            "control_max_abs": max_control,
            "nonfinite": int(nonfinite),
        }
    return result


def memory():
    stats = torch.cuda.memory_stats()
    external, total = device_memory_usage(stats)
    return {
        "gift_total_gib": total,
        "gift_external_gib": external,
        "peak_allocated_gib": stats["allocated_bytes.all.peak"] / GIB,
        "peak_reserved_gib": stats["reserved_bytes.all.peak"] / GIB,
        "allocator_retries": stats["num_alloc_retries"],
    }


def measure_single_arm_memory(factories, operands, slot, stream):
    measurements = {}
    for name, factory in factories.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_stats()
        writer = factory()
        chain(writer, operands, slot, stream)
        stream.synchronize()
        measurements[name] = {
            **memory(),
            "starting_allocated_gib": start["allocated_bytes.all.current"] / GIB,
            "starting_reserved_gib": start["reserved_bytes.all.current"] / GIB,
        }
        if hasattr(writer, "close"):
            writer.close()
        del writer
    return measurements


def measure_chains(arms, operands, slot, stream, samples):
    torch.cuda.empty_cache()
    graphs = {
        name: record_graph(lambda writer=writer: chain(writer, operands, slot, stream), stream)
        for name, writer in arms.items()
    }
    events = timing_events()
    report = {name: {"graph_ms": [], "eager": [], "memory": []} for name in arms}
    names = list(arms)
    begin, finish = events
    for index in range(samples):
        for name in names if index % 2 == 0 else names[::-1]:
            report[name]["graph_ms"].append(event_sample(graphs[name].replay, stream, events))
    del graphs
    torch.cuda.empty_cache()
    for index in range(samples):
        for name in names if index % 2 == 0 else names[::-1]:
            torch.cuda.reset_peak_memory_stats()
            begin.record(stream)
            start = time.perf_counter()
            chain(arms[name], operands, slot, stream)
            submit = time.perf_counter()
            finish.record(stream)
            finish.synchronize()
            stop = time.perf_counter()
            report[name]["eager"].append({
                "cuda_event_ms": begin.elapsed_time(finish),
                "submit_ms": 1000 * (submit - start),
                "final_wait_ms": 1000 * (stop - submit),
                "wall_ms": 1000 * (stop - start),
            })
            report[name]["memory"].append(memory())
    for measurements in report.values():
        measurements["median_graph_ms"] = median(measurements["graph_ms"])
        for metric in measurements["eager"][0]:
            measurements[f"median_{metric}"] = median(item[metric] for item in measurements["eager"])
        measurements["max_gift_total_gib"] = max(item["gift_total_gib"] for item in measurements["memory"])
    return report


def write_json(path, receipt):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(receipt, indent=2) + "\n")
    partial.replace(path)


def run(args, stream):
    model, chunks = CASES[args.case]
    fixture = args.root / "inputs" / f"{model}.pt"
    operands, provenance = load_operands(args.case, fixture, args.root / "native-ce" / "centered-folded-v1.so")
    rows, vocab = operands[0][0].shape
    width = operands[0][1].size(1)
    slot = torch.empty((vocab, width), dtype=torch.float32, device=stream.device)
    shape, alignments = operand_key(*operands[0], slot)
    print(f"{args.case} shape={shape} alignments={alignments} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    with ExitStack() as resources:
        search = Search(shape, alignments, stream, resources)
        if args.reuse_screen:
            prior = json.loads(args.reuse_screen.read_text())
            if (
                prior["case"] != args.case
                or prior["shape"] != list(shape)
                or prior["alignments"] != list(alignments)
                or prior["cublasLt_version"] != search.version
                or prior["visible_device"] != os.environ.get("CUDA_VISIBLE_DEVICES")
                or prior["fixture"] != str(fixture)
                or prior["ce_sha256"] != EXACT_CE_SHA256
            ):
                raise ValueError("Prior screen does not match this input and runtime")
            candidates = [search.restore(item) for item in prior["candidates"]]
            screens = prior["screen"]
        else:
            candidates = search.candidates(args.heuristics)
            screens = screen(search, candidates, operands, slot, 4)
        print(f"{args.case} candidates={len(candidates)} cuBLASLt={search.version}", flush=True)
        rank0, rank1 = [screens[str(beta)]["ranked"] for beta in (0, 1)]
        factories = {"generic": lambda: native.make_head_dw_writer("generic", chunks, *shape, stream)}
        control = "generic"
        if model == "100b":
            factories["fixed"] = lambda: native.make_head_dw_writer("new", chunks, *shape, stream)
            control = "fixed"
        beta_differences = check_beta(search, candidates, screens, factories[control](), operands, slot)
        pairs = {"timed": (rank0[0], rank1[0])}
        exact = [next((index for index in ranks if beta_differences[str(beta)][index] == 0), None)
                 for beta, ranks in enumerate((rank0, rank1))]
        if None not in exact:
            pairs["timed-bitwise-beta"] = tuple(exact)
        first_column = next((i for i, item in enumerate(candidates) if item.order == "column"), None)
        first_row = next((i for i, item in enumerate(candidates) if item.order == "row"), None)
        if first_column in rank0 and first_column in rank1:
            pairs["heuristic-column"] = (first_column, first_column)
        if first_row in rank0 and first_row in rank1:
            pairs["heuristic-row"] = (first_row, first_row)
        zero0 = next((index for index in rank0 if candidates[index].workspace == 0), None)
        zero1 = next((index for index in rank1 if candidates[index].workspace == 0), None)
        if zero0 is not None and zero1 is not None:
            pairs["timed-zero-workspace"] = (zero0, zero1)
        if len(rank0) > 1 and len(rank1) > 1:
            pairs["timed-beta0-runner-up"] = (rank0[1], rank1[0])
            pairs["timed-beta1-runner-up"] = (rank0[0], rank1[1])
        def plan(arm):
            return {
                "format": 1,
                "case": args.case,
                "shape": shape,
                "alignments": alignments,
                "capability": torch.cuda.get_device_capability(stream.device),
                "cublasLt_version": search.version,
                "device_name": torch.cuda.get_device_name(stream.device),
                "pair": [candidates[index].receipt() for index in pairs[arm]],
            }

        for name in pairs:
            factories[name] = lambda name=name: make_dw_writer(receipt=json.loads(json.dumps(plan(name))))
        stream.synchronize()
        isolated_memory = measure_single_arm_memory(factories, operands, slot, stream)
        arms = {name: factory() for name, factory in factories.items()}
        arms["control_repeat"] = arms[control]
        for writer in arms.values():
            if hasattr(writer, "close"):
                resources.callback(writer.close)
        print(f"{args.case} fastest beta0={rank0[:5]} beta1={rank1[:5]}", flush=True)
        # Drop screen-only allocator reservations before measuring preallocated chains.
        stream.synchronize()
        timings = measure_chains(arms, operands, slot, stream, args.samples)
        print(f"{args.case} chain graph medians " + str({k: round(v["median_graph_ms"], 4) for k, v in timings.items()}), flush=True)
        numbers = numerical(arms, operands, slot, stream, control)
        accepted = [
            name for name in pairs
            if numbers[name]["nonfinite"] == 0
            and all(numbers[name][metric] <= numbers[control][metric]
                    for metric in ("oracle_relative_l2", "oracle_max_abs"))
        ]
        exact = [name for name in accepted if numbers[name]["control_bit_mismatches"] == 0]
        def graph_median(name):
            return timings[name]["median_graph_ms"]

        selected = min(accepted, key=graph_median, default=None)
        selected_bitwise = min(exact, key=graph_median, default=None)
        selected_raw = min(pairs, key=graph_median)
        info = {
            "case": args.case, "chunks": chunks, "shape": shape, "alignments": alignments,
            "provenance": provenance, "fixture": str(fixture), "fixture_bytes": fixture.stat().st_size,
            "torch": torch.__version__, "cuda": torch.version.cuda, "cublasLt_version": search.version,
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0],
            "device": torch.cuda.get_device_name(),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "job_id": next((os.environ[key] for key in ("MLSPACE_JOB_ID", "JOB_ID") if key in os.environ), None),
            "ce_sha256": EXACT_CE_SHA256,
            "source_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "beta_contract": {"beta0": 1, "beta1": chunks - 1},
            "candidates": [item.receipt() for item in candidates], "screen": screens,
            "reused_screen": str(args.reuse_screen) if args.reuse_screen else None,
            "arms": pairs, "control": control, "measurements": timings,
            "numerics": numbers, "per_beta_control_bit_mismatches": beta_differences,
            "isolated_memory": isolated_memory,
            "eligible": accepted, "selected": selected, "selected_raw": selected_raw, "selected_bitwise": selected_bitwise,
            "scope": "Preallocated same-stream dW chain, eager submission/wait/wall and warm CUDA Graph replay; operands ready, no timed allocations, projection or collectives.",
        }
        for suffix, chosen in (("plan", selected), ("plan-raw", selected_raw), ("plan-bitwise", selected_bitwise)):
            if chosen is not None:
                selected_plan = plan(chosen)
                selected_plan.update({"selected": chosen, "numerics": numbers[chosen]})
                write_json(args.out / f"{args.case}-{suffix}.json", selected_plan)
        write_json(args.out / f"{args.case}.json", info)
    print(f"{args.case} selected={selected} eligible={accepted}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heuristics", type=int, default=32)
    parser.add_argument("--reuse-screen", type=Path)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one authorized GPU")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(args, stream)


if __name__ == "__main__":
    main()
