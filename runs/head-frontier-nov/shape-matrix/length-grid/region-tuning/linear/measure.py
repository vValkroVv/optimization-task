"""Screen allocating cuBLASLt projection and dHidden against the length-grid routes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
GRID = ROOT / "length-grid"
TUNING = ROOT / "tuning"
sys.path[0:0] = [str(TUNING / "proj"), str(GRID), str(ROOT)]
from ce_bench import evaluate
from gemm_bench import Timer, capture, groups_by_index
from matrix_fixture import load_inputs
from replay import resize_hidden
from torchtitan.components.loss import MTPLoss
from torchtitan.ops.exact_folded_ce import ExactFoldedCrossEntropy
from tuning.linear_policy import LinearPolicy
from tuning.proj.lt_ops import CONFIG_FIELDS, Plan, _library

CALIBRATED = TUNING / "linear-plans.json"
CE = TUNING / "ce/centered-tuned.so"
CE_HASH = "d3343cd518a05a6d5bea58b0493fa33373e040f1ce3dd5864b14c03033528929"
MODELS = ("100b", "700b")
LENGTHS = (32768, 65536, 131072, 262144)
ROLES = ("projection", "dhidden")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def chunk_input(model, seq, fixture):
    record = load_inputs(model, fixture)
    labels, positions = record["labels"], record["positions"]
    replicas = seq // labels.size(1)
    tokens_per_chunk = seq // 8
    assert replicas * labels.size(1) == seq and tokens_per_chunk % labels.size(1) == 0
    valid_base = int(labels.ne(-100).sum())
    denominator_scale = record["denominator"] / valid_base
    assert denominator_scale > 0 and denominator_scale.is_integer()
    assert labels[:, -1].eq(-100).all() and positions[:, 0].eq(0).all()
    labels = labels[:1].repeat(1, replicas)
    positions = positions[:1].repeat(1, replicas)
    valid = int(labels.ne(-100).sum())
    shifted = MTPLoss._shift_labels(labels, 1, positions)
    targets = torch.cat((labels, shifted))[:, :tokens_per_chunk].contiguous().cuda()
    hidden = resize_hidden(record["hidden"], 1, tokens_per_chunk, "cuda").flatten(0, 1)
    weight = record["weight"].cuda()
    receipt = {
        "fixture": str(fixture),
        "fixture_bytes": fixture.stat().st_size,
        "provenance": record["provenance"],
        "base_sequence": record["labels"].size(1),
        "base_row": 0,
        "physical_tokens": seq,
        "valid_main_tokens": valid,
        "valid_mtp1_tokens": int(shifted.ne(-100).sum()),
        "chunk_index": 0,
        "chunk_valid_tokens_by_head": targets.ne(-100).sum(dim=1).tolist(),
        "folded_hidden_shape": list(hidden.shape),
        "vocab": weight.size(0),
        "head_weights": record["head_weights"],
        "denominator": float(valid * denominator_scale),
    }
    return hidden, weight, targets, receipt


def exact_gradient(project, targets, receipt):
    loss = ExactFoldedCrossEntropy(str(CE), library_sha256=CE_HASH)
    logits = project()
    assert logits._base is None and logits.is_contiguous()
    _, _, _, dz = evaluate(loss, logits.requires_grad_(True), targets,
                           receipt["head_weights"], receipt["denominator"])
    assert logits.data_ptr() == dz.data_ptr() and loss.scales_checked
    receipt["dlogits"] = {
        "route": "compiled consuming CUDA 13 CE",
        "sha256": CE_HASH,
        "input_output_alias": True,
        "shape": list(dz.shape),
        "dtype": str(dz.dtype),
    }
    return dz.flatten(0, 1)


def bits_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
    for lhs, rhs in zip(a.split(128), b.split(128), strict=True):
        if not torch.equal(lhs.view(torch.int16), rhs.view(torch.int16)):
            return False
    return True


def median(values):
    return statistics.median(values)


def choose_screen_configs(plan, budget):
    configs = plan.configs
    # Heuristics emit exact-layout picks and can include attributes beyond the
    # exhaustive bridge's default CTA/swizzle/option limits. Add distinct
    # algorithm/tile families using the checked occupancy proxy.
    heuristic = sorted((c for c in configs if c["heuristic_rank"] >= 0),
                       key=lambda c: c["heuristic_rank"])
    generic = sorted(configs, key=lambda c: (c["split_k"] > 1, c["waves"]))
    chosen = {c["index"]: c for c in heuristic[:budget // 2]}
    families = {(c["algorithm_id"], c["tile_id"]) for c in chosen.values()}
    for c in generic:
        if len(chosen) >= budget:
            break
        family = c["algorithm_id"], c["tile_id"]
        if family not in families:
            chosen[c["index"]] = c
            families.add(family)
    for c in heuristic + generic:
        if len(chosen) >= budget:
            break
        chosen[c["index"]] = c
    return list(chosen.values())


def screen(lhs, rhs, dest, reference, control, timer, budget):
    reports = []
    scored = []
    # The launch-only screen reuses one output. Allocating eager parity and timing
    # use the real cached policy after screening.
    for row in (False, True):
        plan = Plan(lhs, rhs, dest, row=row, exhaustive=True)
        report = {
            "problem": list(plan.problem),
            "enumeration_counts": plan.counts,
            "workspace_limit_bytes": plan.workspace_limit,
            "candidates": [],
        }
        reports.append(report)
        cfgs = choose_screen_configs(plan, budget)
        for offset, cfg in enumerate(cfgs):
            if offset % 8 == 0:
                value, base = timer.sample(control)
                del value
            index = cfg["index"]
            status, sample = timer.sample(lambda: plan.launch(index, lhs, rhs, dest))
            entry = {**cfg, "status": status}
            if status == 0:
                entry.update(event_ms=sample["event_ms"], control_event_ms=base["event_ms"])
                scored.append((entry, plan))
            report["candidates"].append(entry)
    scored.sort(key=lambda pair: pair[0]["event_ms"] / pair[0]["control_event_ms"])
    shortlist = []
    for entry, plan in scored[:12]:
        def run():
            return plan.run(entry["index"], lhs, rhs, dest)
        measurements = [timer.sample(run)[1]["event_ms"] for _ in range(3)]
        entry["rescreen_event_ms"] = measurements
        entry["bitwise"] = bits_equal(dest, reference)
        if entry["bitwise"]:
            shortlist.append((entry, plan))
    shortlist.sort(key=lambda pair: median(pair[0]["rescreen_event_ms"]))
    # The same configuration can be emitted by several layout descriptors.
    # Keep the fastest descriptor for each distinct kernel configuration.
    distinct = {}
    for entry, plan in shortlist:
        key = tuple(entry[field] for field in CONFIG_FIELDS[:-1])
        distinct.setdefault(key, (entry, plan))
    return reports, list(distinct.values())[:3]


def benchmark(calls, timer, rounds):
    found = {name: [] for name in calls}
    for name, call in calls.items():
        for _ in range(5):
            call()
    torch.cuda.synchronize()
    for i in range(rounds):
        for name in groups_by_index(list(calls), i):
            value, sample = timer.sample(calls[name])
            del value
            found[name].append(sample)
    return found


def isolated_memory(calls, timer):
    result = {name: [] for name in calls}
    for i in range(3):
        for name in groups_by_index(list(calls), i):
            torch.cuda.empty_cache()
            value, sample = timer.sample(calls[name])
            del value
            result[name].append(sample)
    return result


def graph_measure(calls, timer, rounds):
    graphs = {}
    failures = {}
    for name, call in calls.items():
        try:
            graphs[name] = capture(call)
        except RuntimeError as error:
            failures[name] = str(error)
    output = {name: [] for name in graphs}
    if not graphs:
        return output, failures
    for i in range(rounds):
        for name in groups_by_index(list(graphs), i):
            graph = graphs[name]
            def replay():
                for _ in range(3):
                    graph.replay()
            _, sample = timer.sample(replay)
            output[name].append(sample["event_ms"] / 3)
    return output, failures


def evaluate_candidates(timings, isolation, graph, repeats):
    result = {}
    a, b = timings["control"], timings["control_repeat"]

    def ms(samples, key):
        return [s[key] for s in samples]
    base = ms(a, "wall_ms")
    repeat = ms(b, "wall_ms")
    noise = median(abs(x - y) for x, y in zip(base, repeat, strict=True))
    gc_base = graph.get("control", []) + graph.get("control_repeat", [])
    for name, samples in timings.items():
        if name in ("control", "control_repeat"):
            continue
        compared = ms(samples, "wall_ms")
        deltas = [z - (x + y) / 2 for x, y, z in zip(base, repeat, compared, strict=True)]
        wins = sum(delta < 0 for delta in deltas)
        memory_check = {}
        for scope, source in (("warmed", timings), ("isolated", isolation)):
            key = "gift_total_peak_gib"
            left, right = ms(source["control"], key), ms(source["control_repeat"], key)
            control_delta = max(abs(x - y) for x, y in zip(left, right, strict=True))
            control_peak = max(left + right)
            candidate_peak = max(ms(source[name], key))
            memory_check[scope] = dict(control_peak_gib=control_peak,
                                       candidate_peak_gib=candidate_peak,
                                       max_control_difference_gib=control_delta,
                                       nonincrease=candidate_peak <= control_peak + control_delta)
        enough_wins = wins >= (repeats * 4 + 4) // 5
        first = median(deltas[:repeats // 2])
        second = median(deltas[repeats // 2:])
        speed_win = median(deltas) < -2 * noise and enough_wins and first < 0 and second < 0
        memory_win = all(entry["nonincrease"] for entry in memory_check.values())
        retry_free = all(s["allocator_retries"] == 0 for series in (timings, isolation)
                         for arm in ("control", "control_repeat", name) for s in series[arm])
        result[name] = {
            "control_wall_ms": median(base + repeat),
            "candidate_wall_ms": median(compared),
            "paired_wall_delta_ms": median(deltas),
            "control_wall_abs_delta_ms": noise,
            "candidate_pair_wins": wins,
            "first_half_wall_delta_ms": first,
            "second_half_wall_delta_ms": second,
            "memory": memory_check,
            "allocator_retry_free": retry_free,
            "control_graph_ms": median(gc_base) if gc_base else None,
            "candidate_graph_ms": median(graph[name]) if graph.get(name) else None,
            "retained": speed_win and memory_win and retry_free,
        }
    return result


def measure(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    timer = Timer()
    fixture = ROOT / "inputs" / f"{args.model}.pt"
    hidden, weight, targets, receipt = chunk_input(args.model, args.seq, fixture)
    is_calibrated = args.model == "100b" and args.seq == 32768
    calibrated_policy = LinearPolicy(CALIBRATED) if is_calibrated else None
    projection_shape = (*hidden.shape[:-1], weight.size(0))

    def project():
        if calibrated_policy is not None:
            return calibrated_policy.run(hidden.flatten(0, 1), weight.T, projection_shape)
        return torch.nn.functional.linear(hidden, weight)

    args.out.mkdir(parents=True, exist_ok=True)
    receipt.update(model=args.model, sequence=args.seq, batch=1, chunks=8, role=args.role,
                   dtype="bfloat16", compute="fp32", process=os.getpid(), host=socket.gethostname(),
                   gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                   cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                   torch=torch.__version__, torch_git=torch.version.git_version, cuda=torch.version.cuda,
                   cublaslt=_library().head_lt_version(), bridge_sha256=sha256(TUNING / "proj/lt_bridge.so"),
                   bf16_reduced_precision_reduction=str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
    if args.role == "projection":
        lhs, rhs = hidden.flatten(0, 1), weight.T
        control = project
        owner_shape = projection_shape
        receipt["control"] = "calibrated projection" if is_calibrated else "ATen linear"
    else:
        lhs, rhs = exact_gradient(project, targets, receipt), weight
        def control():
            return torch.mm(lhs, rhs)
        owner_shape = None
        receipt["control"] = "ATen mm"
    output_shape = (lhs.size(0), rhs.size(1))
    receipt["output_shape"] = list(output_shape)
    receipt["operands"] = [{"shape": list(t.shape), "stride": list(t.stride())} for t in (lhs, rhs)]
    reference = control().view(output_shape)
    dest = torch.empty_like(reference)
    receipt["screen"], shortlist = screen(lhs, rhs, dest, reference, control, timer, args.budget)
    del dest
    calls = {"control": control, "control_repeat": control}
    finalists = {}
    policies = {}
    for i, (entry, plan) in enumerate(shortlist):
        name = f"candidate-{i}"
        choices = args.out / f"{name}.choices.json"
        key = json.dumps([receipt["cublaslt"], receipt["capability"], plan.problem])
        config = {field: entry[field] for field in CONFIG_FIELDS}
        save(choices, {key: config})
        policy = LinearPolicy(choices)
        def run(policy=policy):
            return policy.run(lhs, rhs, owner_shape)

        actual = run().view(output_shape)
        matched = bits_equal(actual, reference)
        finalists[name] = {
            "descriptor": list(plan.problem),
            "config": config,
            "workspace_bytes": entry["workspace_bytes"],
            "loaded_exact_descriptor": policy.calls == {"candidate": 1, "fallback": 0},
            "bitwise": matched,
            "compared_elements": reference.numel() if matched else None,
            "choices": choices.name,
        }
        del actual
        if matched and finalists[name]["loaded_exact_descriptor"]:
            calls[name] = run
            policies[name] = policy
    if calibrated_policy is not None:
        if calibrated_policy.calls["candidate"] == 0 or calibrated_policy.calls["fallback"]:
            raise RuntimeError(f"Unactivated calibrated length-grid route: {calibrated_policy.calls}")
        receipt["calibrated_calls_before_confirmation"] = dict(calibrated_policy.calls)
    del reference
    torch.cuda.empty_cache()
    save(args.out / "screen.json", dict(receipt, finalists=finalists))
    print(f"{args.model} S={args.seq} {args.role}: {len(policies)} matching finalists", flush=True)
    warmed = benchmark(calls, timer, args.rounds)
    isolation = isolated_memory(calls, timer)
    graph, graph_errors = graph_measure(calls, timer, args.graph_rounds)
    comparison = evaluate_candidates(warmed, isolation, graph, args.rounds)
    eligible = (name for name, sample in comparison.items() if sample["retained"])
    selected = min(eligible, key=lambda name: comparison[name]["paired_wall_delta_ms"], default=None)
    if any(policy.calls["fallback"] for policy in policies.values()):
        raise RuntimeError("Prepared candidate fell back during confirmation")
    if calibrated_policy is not None and calibrated_policy.calls["fallback"]:
        raise RuntimeError("Calibrated control fell back during confirmation")
    for name, policy in policies.items():
        finalists[name]["activation"] = policy.calls
    save(args.out / "confirm.json", dict(
        receipt,
        finalists=finalists,
        calibrated_calls=None if calibrated_policy is None else calibrated_policy.calls,
        warmed_eager=warmed,
        isolated_memory=isolation,
        graph_ms=graph,
        graph_errors=graph_errors,
        comparison=comparison,
        selected=selected,
    ))
    print(f"{args.model} S={args.seq} {args.role}: selected {selected}; {comparison}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seq", choices=LENGTHS, type=int, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=40, help="Maximum launches per layout in the first screen.")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--graph-rounds", type=int, default=5)
    args = parser.parse_args()
    if args.budget < 1 or args.rounds < 2 or args.graph_rounds < 1:
        parser.error("Invalid screen, eager or graph sample count")
    measure(args)
