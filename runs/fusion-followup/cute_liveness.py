"""Join profiled CuTe replay sites to generated Inductor graph output uses."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


PREFIX = "cute_gated_norm_"
FORWARD = {PREFIX + "fwd", PREFIX + "residual_fwd"}
NORM = FORWARD | {PREFIX + "bwd", PREFIX + "fwd_stats"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def generated_calls(path, data):
    module = ast.parse(data, filename=str(path))
    # The executable wrapper is top-level or a method of Runner.
    runner = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "Runner"
        ),
        None,
    )
    entries = [*module.body, *(runner.body if runner is not None else [])]
    entry = next(
        (
            node
            for node in entries
            if isinstance(node, ast.FunctionDef) and node.name == "call"
        ),
        None,
    )
    if entry is None:
        return [], []
    nodes = sorted(
        (node for node in ast.walk(entry) if isinstance(node, ast.stmt)),
        key=lambda node: node.lineno,
    )
    tuple_shards = defaultdict(list)
    for node in ast.walk(entry):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            tuple_shards[node.value.id].append(node)
    names = []
    sites = []
    for statement in nodes:
        if not isinstance(statement, (ast.Assign, ast.Expr)) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        target = ast.unparse(statement.value.func)
        if target.startswith("triton_") and target.endswith(".run"):
            names.append(target.removesuffix(".run"))
        prefix = "torch.ops.torchtitan."
        if not target.startswith(prefix) or not target.endswith(".default"):
            continue
        name = target.removeprefix(prefix).removesuffix(".default")
        if name not in NORM:
            continue
        names.append(name)
        if name in FORWARD:
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], ast.Name
            ):
                raise ValueError(
                    f"Unknown Inductor assignment for {name} at {path}:{statement.lineno}"
                )
            output = statement.targets[0].id
            reads = tuple_shards[output]
            indexes = sorted(
                {
                    read.slice.value
                    for read in reads
                    if isinstance(read.slice, ast.Constant)
                }
            )
            # Any executable use of the whole tuple is conservative. A `del` is not a use.
            num_reads = sum(
                isinstance(node, ast.Name)
                and node.id == output
                and isinstance(node.ctx, ast.Load)
                for node in ast.walk(entry)
            )
            sites.append(
                {
                    "op": name,
                    "line": statement.lineno,
                    "tuple": output,
                    "read_indices": indexes,
                    "activation_dead": num_reads == len(reads)
                    and all(isinstance(read.slice, ast.Constant) for read in reads)
                    and 0 not in indexes,
                }
            )
    return names, sites


def cache_index(cache, required):
    wrappers = defaultdict(list)
    norm_orders = {
        tuple(name for name in signature if name in NORM) for signature in required
    }
    expression = re.compile(
        rb"torch\.ops\.torchtitan\.(cute_gated_norm_\w+)\.default\("
    )
    for path in sorted(cache.glob("[a-z0-9][a-z0-9]/*.py")):
        data = path.read_bytes()
        if PREFIX.encode() + b"bwd.default(" not in data:
            continue
        short_order = tuple(match[1].decode() for match in expression.finditer(data))
        if short_order not in norm_orders:
            continue
        names, sites = generated_calls(path, data)
        if names and tuple(names) in required:
            wrappers[tuple(names)].append(
                {
                    "file": str(path.relative_to(cache)),
                    "sha256": digest(data),
                    "sites": sites,
                }
            )
    return wrappers


def traced_signatures(trace):
    events = trace["traceEvents"]
    device_kernels = [event for event in events if event.get("cat") == "kernel"]
    triton_kernel = {
        event["args"]["correlation"]: event["name"]
        for event in device_kernels
        if event["name"].startswith("triton_")
    }
    output_stage = {
        event["args"]["External id"]: event
        for event in device_kernels
        if "GnFwdOut" in event["name"]
    }
    tokens = defaultdict(list)
    scopes = []
    for event in events:
        name = event.get("name", "")
        if event.get("ph") != "X":
            continue
        if name == "CompiledFunctionBackward":
            scopes.append(event)
        elif name.startswith("torchtitan::" + PREFIX) and event.get("cat") == "cpu_op":
            tokens[event["tid"]].append(
                (event["ts"], name.removeprefix("torchtitan::"), event)
            )
        elif event.get("cat") in {"cuda_runtime", "cuda_driver"} and "Launch" in name:
            kernel = triton_kernel.get(event["args"].get("correlation"))
            if kernel is not None:
                tokens[event["tid"]].append((event["ts"], kernel, event))
    for scope in scopes:
        start = scope["ts"]
        inside = sorted(
            (
                token
                for token in tokens[scope["tid"]]
                if start <= token[0] < start + scope["dur"]
            ),
            key=lambda token: token[0],
        )
        forwards = [(name, event) for _, name, event in inside if name in FORWARD]
        if not forwards:
            continue
        sites = []
        for name, event in forwards:
            external_id = event["args"]["External id"]
            kernel = output_stage[external_id]
            width = re.search(r"tensor00oi64(4096|512)1", kernel["name"])
            sites.append(
                {
                    "op": name,
                    "external_id": external_id,
                    "output_stage_us": kernel["dur"],
                    "hidden": int(width[1]) if width is not None else None,
                }
            )
        yield tuple(name for _, name, _ in inside), scope["args"]["External id"], sites


def run(cache, trace):
    data = trace.read_bytes()
    launches = list(traced_signatures(json.loads(data)))
    index = cache_index(cache, {signature for signature, _, _ in launches})
    results = []
    counts = Counter()
    for signature, region, traced in launches:
        wrappers = index.get(signature, [])
        result = {
            "trace_external_id": region,
            "signature": signature,
            "wrappers": wrappers,
            "calls": traced,
        }
        results.append(result)
        for position, call in enumerate(traced):
            answers = {
                wrapper["sites"][position]["activation_dead"] for wrapper in wrappers
            }
            call["activation_dead"] = answers.pop() if len(answers) == 1 else None
            counts[(call["op"], call["hidden"], call["activation_dead"])] += 1
    return {
        "trace": str(trace),
        "trace_sha256": digest(data),
        "cache": str(cache),
        "counts": [
            dict(op=op, hidden=hidden, activation_dead=dead, count=count)
            for (op, hidden, dead), count in counts.items()
        ],
        "region_matches": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cache, args.trace)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
