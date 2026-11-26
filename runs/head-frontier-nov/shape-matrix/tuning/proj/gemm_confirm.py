"""Confirm the measured GEMMs on all chunks and quantify warmed allocator latency."""
import argparse
from collections import defaultdict
from functools import partial
import json
from pathlib import Path
import statistics

import torch

from gemm_bench import (Timer, bits_equal, capture, dlogits, file_sha256, groups_by_index, inputs)
from lt_ops import _library, lt_dhidden, lt_mm, prepare

ROOT = Path(__file__).resolve().parent


def project(h, w):
    return torch.nn.functional.linear(h, w).flatten(0, 1)


def candidate_order(cell):
    return sorted(range(len(cell["tested"])), key=lambda i: statistics.median(cell["graph"][f"candidate-{i}"]))


def choose(screen, path):
    chosen = {}
    manifest = {}
    for role, metrics in screen["roles"].items():
        if not metrics["tested"]:
            continue
        pick = metrics["tested"][candidate_order(metrics)[0]]
        key = json.dumps([screen["cublaslt"], screen["capability"], pick["problem"]])
        manifest[key] = pick["config"]
        chosen[role] = pick
    path.write_text(json.dumps(manifest, indent=2))
    return chosen


def paired(calls, timer, count):
    result = defaultdict(list)
    for call in calls.values():
        for _ in range(5):
            call()
    torch.cuda.synchronize()
    for i in range(count):
        for name in groups_by_index(list(calls), i):
            output, sample = timer.sample(calls[name])
            del output
            result[name].append(sample)
    return result


def memory(calls, timer):
    values = {}
    for name, call in calls.items():
        torch.cuda.empty_cache()
        output, values[name] = timer.sample(call)
        del output
    return values


def compare(role, lhs, rhs, output, c, args, timer, hidden=None):
    if role == "projection":
        control = partial(project, hidden, rhs.T)
    elif role == "dhidden-direct":
        def control():
            return output.copy_(torch.mm(lhs.flatten(0, 1), rhs).view_as(output))
    else:
        control = partial(torch.mm, lhs, rhs)
    if role == "dhidden-direct":
        candidate = partial(lt_dhidden, lhs, rhs, output, choices=c)
    else:
        candidate = partial(lt_mm, lhs, rhs, choices=c)
    eager = dict(control=control, control_repeat=control, candidate=candidate)
    result = dict(warmed_eager=paired(eager, timer, args.rounds))
    result["isolated_memory"] = memory(eager, timer)

    if role == "dhidden-direct":
        control_graph = control
        candidate_graph = candidate
    else:
        graph_out = torch.empty((lhs.size(0), rhs.size(1)), device=lhs.device, dtype=lhs.dtype)
        control_graph = partial(torch.mm, lhs, rhs, out=graph_out)
        candidate_graph = partial(lt_mm, lhs, rhs, out=graph_out, choices=c)
    g_base = capture(control_graph)
    g_candidate = capture(candidate_graph)

    def replay(graph):
        for _ in range(5):
            graph.replay()

    timing = paired(dict(control=partial(replay, g_base), control_repeat=partial(replay, g_base),
                         candidate=partial(replay, g_candidate)), timer, args.rounds)
    result["graph_ms"] = {name: [item["event_ms"] / 5 for item in values] for name, values in timing.items()}
    return result


def profile(calls, directory):
    results = {}
    for name, call in calls.items():
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            call()
            torch.cuda.synchronize()
        path = directory / f"{name}-profiler.json"
        profiler.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        gpu = sorted(
            (entry for entry in trace["traceEvents"]
             if entry.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
             and entry.get("ph") == "X"),
            key=lambda entry: entry["ts"],
        )
        results[name] = [
            dict(name=entry["name"], category=entry["cat"], duration_us=entry["dur"],
                 start_us=entry["ts"] - gpu[0]["ts"], stream=entry.get("args", {}).get("stream"))
            for entry in gpu
        ]
    return results


def verify_roles(record, shape, hidden, weight, labels, loss, choices, manifest):
    total = {role: 0 for role in manifest}
    direct = weight.new_full(shape, float("nan"))
    expected_dh = weight.new_empty(shape)
    for i, (x, y) in enumerate(zip(hidden, labels, strict=True)):
        h = x.flatten(0, 1)
        if "projection" in manifest:
            # Projection is compared before either CE path, because the full-head
            # FP64 CE oracle takes the ATen BF16 logits as its starting point.
            z_expected = project(x, weight)
            assert prepare(h, weight.T, z_expected, choices=choices)
            z = lt_mm(h, weight.T, choices=choices)
            assert bits_equal(z, z_expected), ("projection", i)
            total["projection"] += z.numel()
            del z, z_expected
        dz = dlogits(loss, x, weight, y, record)
        d = dz.flatten(0, 1)
        dh = torch.mm(d, weight)
        if "dhidden" in manifest:
            assert prepare(d, weight, dh, choices=choices)
            actual = lt_mm(d, weight, choices=choices)
            assert bits_equal(actual, dh), ("dhidden", i)
            total["dhidden"] += actual.numel()
        if "dhidden-direct" in manifest:
            t = x.size(1)
            target = direct[:, i*t:(i+1)*t]
            assert prepare(dz, weight, target, choices=choices)
            lt_dhidden(dz, weight, target, choices=choices)
            expected_dh[:, i*t:(i+1)*t].copy_(dh.view_as(target))
            assert bits_equal(target, dh.view_as(target)), ("dhidden-direct", i)
            total["dhidden-direct"] += target.numel()
    if "dhidden-direct" in manifest:
        assert bits_equal(direct, expected_dh), "direct write changed another chunk"
    return total


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    screen = json.loads(args.screen.read_text())
    assert screen["cublaslt"] == _library().head_lt_version()
    choice_file = args.output.with_suffix(".choices.json")
    choices = choose(screen, choice_file)
    rec, shape, hidden, weight, labels, loss = inputs(screen["model"], screen["chunks"])
    assert file_sha256(ROOT.parents[1] / "inputs" / f"{screen['model']}.pt") == screen["fixture_sha256"]
    if args.parity_receipt:
        receipt = json.loads(args.parity_receipt.read_text())
        assert receipt["screen_sha256"] == file_sha256(args.screen)
        assert receipt["choices"] == choices
        totals = receipt["parity_elements"]
        print("reusing all-chunk bitwise receipt", args.parity_receipt, flush=True)
    else:
        totals = verify_roles(rec, shape, hidden, weight, labels, loss, choice_file, choices)
        print("all chunks bitwise", totals, flush=True)
    result = dict(
        screen=args.screen.name, screen_sha256=file_sha256(args.screen), choices=choices,
        parity_elements=totals, measurements={},
    )
    timer = Timer()
    final = weight.new_empty(shape)
    idx = screen["part"]
    for role in choices:
        t = hidden[idx].size(1)
        target = final[:, idx*t:(idx+1)*t]
        if role == "projection":
            op = hidden[idx].flatten(0, 1), weight.T, None
        else:
            dz = dlogits(loss, hidden[idx], weight, labels[idx], rec)
            op = (dz, weight, target) if role == "dhidden-direct" else (dz.flatten(0, 1), weight, None)
        result["measurements"][role] = compare(role, *op, choice_file, args, timer, hidden=hidden[idx])
        args.output.write_text(json.dumps(result, indent=2))
        print("confirmed", role, flush=True)
    if args.profile:
        dz = dlogits(loss, hidden[idx], weight, labels[idx], rec)
        result["profiler"] = profile(
            dict(
                dh_baseline=partial(torch.mm, dz.flatten(0, 1), weight),
                dh_candidate=partial(lt_mm, dz.flatten(0, 1), weight, choices=choice_file),
                dh_direct=partial(lt_dhidden, dz, weight, target, choices=choice_file),
            ), args.output.parent
        )
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--parity-receipt", type=Path)
    p.add_argument("--profile", action="store_true")
    main(p.parse_args())
