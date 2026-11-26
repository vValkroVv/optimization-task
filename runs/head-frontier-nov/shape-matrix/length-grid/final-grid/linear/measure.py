"""Calibrate missing allocating linear descriptors for the B1 length grid."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
GRID = ROOT / "length-grid"
REGION = GRID / "region-tuning/linear"
TUNING = ROOT / "tuning"
sys.path[:0] = [str(REGION), str(TUNING / "proj"), str(GRID), str(ROOT)]

import torch

import importlib.util
spec = importlib.util.spec_from_file_location("linear_region_tuning", REGION / "measure.py")
region = importlib.util.module_from_spec(spec)
spec.loader.exec_module(region)

from matrix_fixture import load_inputs
from replay import resize_hidden
from torchtitan.components.loss import MTPLoss
from tuning.linear_policy import LinearPolicy
from tuning.proj.lt_ops import CONFIG_FIELDS, Plan, _library


def chunk_input(model, seq, chunks, fixture):
    record = load_inputs(model, fixture)
    labels, positions = record["labels"], record["positions"]
    tiles, tail = divmod(seq, labels.shape[1])
    if tiles < 1 or tail or seq % chunks:
        raise ValueError("Expected a whole number of recorded fixture tiles and equal chunks")
    base_valid = int(labels.ne(-100).sum())
    scale = record["denominator"] / base_valid
    assert scale > 0 and scale.is_integer()
    assert labels[:, -1].eq(-100).all() and positions[:, 0].eq(0).all()
    labels, positions = labels[:1].repeat(1, tiles), positions[:1].repeat(1, tiles)
    valid = int(labels.ne(-100).sum())
    shifted = MTPLoss._shift_labels(labels, 1, positions)
    tokens = seq // chunks
    targets = torch.cat((labels, shifted))[:, :tokens].contiguous().cuda()
    hidden = resize_hidden(record["hidden"], 1, tokens, "cuda").flatten(0, 1)
    weight = record["weight"].cuda()
    return hidden, weight, targets, {
        "fixture": str(fixture), "fixture_bytes": fixture.stat().st_size,
        "provenance": record["provenance"], "base_row": 0, "chunk_index": 0,
        "physical_tokens": seq, "tokens_per_head_in_chunk": tokens,
        "sequence_valid_by_head": [valid, int(shifted.ne(-100).sum())],
        "chunk_valid_by_head": targets.ne(-100).sum(dim=1).tolist(),
        "folded_hidden_shape": list(hidden.shape), "head_weights": record["head_weights"],
        "denominator": float(valid * scale), "vocab": weight.size(0),
    }


def bits_equal(actual, reference):
    assert actual.shape == reference.shape and actual.dtype == reference.dtype == torch.bfloat16
    rows_per_tile = 1024 if reference.is_cpu else 128
    for lhs, rhs in zip(actual.split(rows_per_tile), reference.split(rows_per_tile), strict=True):
        if not torch.equal(lhs.view(torch.int16), rhs.to(lhs.device).view(torch.int16)):
            return False
    return True


def screen(lhs, rhs, dest, reference, timer, budget):
    reports, scored = [], []
    for row in (False, True):
        plan = Plan(lhs, rhs, dest, row=row, exhaustive=True)
        candidates = []
        for config in region.choose_screen_configs(plan, budget):
            status, timing = timer.sample(lambda: plan.launch(config["index"], lhs, rhs, dest))
            entry = {**config, "status": status}
            if status == 0:
                entry["event_ms"] = timing["event_ms"]
                scored.append((entry, plan))
            candidates.append(entry)
        reports.append(dict(problem=plan.problem, candidates=candidates,
                            enumeration_counts=plan.counts, workspace_limit_bytes=plan.workspace_limit))

    # Retiming and exact parity use the same addresses as the first screen.
    # Size the shortlist before copying a large reference back in bounded tiles.
    shortlist = []
    for entry, plan in sorted(scored, key=lambda pair: pair[0]["event_ms"])[:12]:
        def call():
            return plan.run(entry["index"], lhs, rhs, dest)
        entry["rescreen_event_ms"] = [timer.sample(call)[1]["event_ms"] for _ in range(3)]
        shortlist.append((entry, plan))
    distinct = {}
    for entry, plan in sorted(shortlist, key=lambda p: region.median(p[0]["rescreen_event_ms"])):
        key = tuple(entry[field] for field in CONFIG_FIELDS[:-1])
        if key not in distinct:
            plan.run(entry["index"], lhs, rhs, dest)
            entry["bitwise"] = bits_equal(dest, reference)
            if entry["bitwise"]:
                distinct[key] = (entry, plan)
            if len(distinct) == 3:
                break
    return reports, list(distinct.values())


def measure_graphs(calls, timer, rounds):
    """Capture one unique arm at a time to bound live graph storage."""
    output, errors = {}, {}
    for name, call in calls.items():
        if name == "control_repeat":
            continue
        graph = None
        torch.cuda.empty_cache()
        try:
            graph = region.capture(call)
            graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(rounds):
                def replay():
                    for _ in range(3):
                        graph.replay()
                _, sample = timer.sample(replay)
                samples.append(sample["event_ms"] / 3)
            output[name] = samples
        except RuntimeError as error:
            errors[name] = str(error)
        finally:
            graph = None
            torch.cuda.empty_cache()
    return output, errors


def measure(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    timer = region.Timer()
    fixture = ROOT / "inputs" / f"{args.model}.pt"
    hidden, weight, targets, receipt = chunk_input(args.model, args.seq, args.chunks, fixture)
    owner_shape = (*hidden.shape[:-1], weight.shape[0])

    def project():
        return torch.nn.functional.linear(hidden, weight)

    if args.role == "projection":
        lhs, rhs = hidden.flatten(0, 1), weight.T
        control = project
        receipt["control"] = "ATen linear"
    else:
        lhs, rhs = region.exact_gradient(project, targets, receipt), weight
        def control():
            return torch.mm(lhs, rhs)
        owner_shape = None
        receipt["control"] = "ATen mm"

    cublaslt = _library().head_lt_version()
    capability = torch.cuda.get_device_capability()
    reference = control().view(lhs.size(0), rhs.size(1))
    output_bytes = reference.numel() * reference.element_size()
    # Leave room for the destination and a bounded parity tile. The reference
    # offload and comparison are outside all timing and memory windows.
    if torch.cuda.mem_get_info()[0] < output_bytes + (1 << 30):
        reference = reference.cpu()
        torch.cuda.empty_cache()
    dest = torch.empty(reference.shape, device=lhs.device, dtype=reference.dtype)

    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__).resolve(), REGION / "measure.py",
                    TUNING / "linear_policy.py", TUNING / "proj/lt_ops.py",
                    TUNING / "proj/lt_bridge.cpp", TUNING / "proj/lt_bridge.so",
                    GRID / "replay.py", ROOT / "matrix_fixture.py"]
    receipt.update(
        model=args.model, sequence=args.seq, batch=1, chunks=args.chunks, role=args.role,
        dtype="bfloat16", compute="fp32", process=os.getpid(), host=socket.gethostname(),
        gpu=torch.cuda.get_device_name(), capability=capability,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], torch=torch.__version__,
        torch_git=torch.version.git_version, cuda=torch.version.cuda, cublaslt=cublaslt,
        sources_sha256={str(path.relative_to(ROOT)): region.sha256(path) for path in source_paths},
        bf16_reduced_precision_reduction=str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        operands=[{"shape": list(t.shape), "stride": list(t.stride())} for t in (lhs, rhs)],
        output_shape=list(reference.shape), output_bytes=output_bytes,
        parity_reference_device=str(reference.device),
        timing_scope="Single stream allocating eager call, timed through final CUDA event synchronization. "
        "Graph diagnostic excludes Python submission and measures one unique arm at a time. "
        "CE input preparation and exact parity are excluded.",
        memory_scope="GIFT total at eager readout, including live component operands and peak Torch reserve. "
        "Untimed parity reference and graph pools are excluded.",
    )
    print(f"{args.model} C{args.chunks} S{args.seq} {args.role}: {list(reference.shape)}, "
          f"reference {reference.device}", flush=True)
    receipt["screen"], shortlist = screen(lhs, rhs, dest, reference, timer, args.budget)
    del dest
    calls = {"control": control, "control_repeat": control}
    finalists, policies = {}, {}
    for i, (entry, plan) in enumerate(shortlist):
        name = f"candidate-{i}"
        choices = args.out / f"{name}.choices.json"
        config = {field: entry[field] for field in CONFIG_FIELDS}
        key = json.dumps([cublaslt, capability, plan.problem])
        region.save(choices, {key: config})
        policy = LinearPolicy(choices)

        def run(policy=policy):
            return policy.run(lhs, rhs, owner_shape)

        actual = run().view_as(reference)
        matched = bits_equal(actual, reference)
        activated = policy.calls == {"candidate": 1, "fallback": 0}
        finalists[name] = dict(descriptor=plan.problem, config=config, workspace_bytes=entry["workspace_bytes"],
                               loaded_exact_descriptor=activated, bitwise=matched,
                               compared_elements=reference.numel(), choices=choices.name)
        del actual
        if matched and activated:
            calls[name], policies[name] = run, policy
    del reference
    torch.cuda.empty_cache()
    region.save(args.out / "screen.json", dict(receipt, finalists=finalists))
    print(f"{args.model} C{args.chunks} S{args.seq} {args.role}: {len(policies)} matching finalists", flush=True)
    warmed = region.benchmark(calls, timer, args.rounds)
    isolated = region.isolated_memory(calls, timer)
    # Persist the paired result before running the independent graph diagnostic.
    graph, graph_errors = {}, {}
    comparison = region.evaluate_candidates(warmed, isolated, graph, args.rounds)
    eligible = (name for name, sample in comparison.items() if sample["retained"])
    selected = min(eligible, key=lambda name: comparison[name]["paired_wall_delta_ms"], default=None)
    for name, policy in policies.items():
        if policy.calls["fallback"]:
            raise RuntimeError("Prepared candidate fell back during confirmation")
        finalists[name]["activation"] = dict(policy.calls)
    payload = dict(receipt, finalists=finalists, warmed_eager=warmed, isolated_memory=isolated,
                   graph_ms=graph, graph_errors=graph_errors, comparison=comparison, selected=selected)
    region.save(args.out / "confirm.json", payload)
    print(f"{args.model} C{args.chunks} S{args.seq} {args.role}: selected {selected}", flush=True)
    graph, graph_errors = measure_graphs(calls, timer, args.graph_rounds)
    comparison = region.evaluate_candidates(warmed, isolated, graph, args.rounds)
    region.save(args.out / "confirm.json", dict(payload, graph_ms=graph, graph_errors=graph_errors,
                                              comparison=comparison))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("100b", "700b"), required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--chunks", type=int, choices=(2, 8), required=True)
    parser.add_argument("--role", choices=("projection", "dhidden"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--graph-rounds", type=int, default=5)
    measure(parser.parse_args())
