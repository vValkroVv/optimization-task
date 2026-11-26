"""Verify CE input parity at the one long C8 shape with a calibrated projection."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from bench_long import (
    CHUNKS, LIBRARIES, ROOT, MTPLoss, bit_differences, digest, eager, elapsed_series,
    evaluate, exact_folded_ce, graph, load_inputs, resize_inputs, tensor_sha256, verify, write_json,
)

sys.path.insert(0, str(ROOT / "tuning"))
from head_run import install_linear_ops
from torchtitan.distributed.gfsdp import _staged_head_dw
from tuning.proj import lt_ops


@torch.no_grad()
def projections(fixture, length, plan_file):
    record = load_inputs("100b", fixture)
    record.pop("z_hidden", None)
    record, _ = resize_inputs(record, 1, length, torch.device("cuda"))
    interval = slice(-(length // CHUNKS), None)
    inputs = record["hidden"].flatten(0, 1)[:, interval].contiguous()
    weight = record["weight"].cuda()
    plain = torch.nn.functional.linear(inputs, weight)
    counts, policy = install_linear_ops(["projection"], plan_file)
    lease = SimpleNamespace(param=SimpleNamespace(unsharded_param=weight))
    deployed = _staged_head_dw._StagedHeadLinear.apply(inputs, weight, lease)
    if counts != {"projection": 1, "dhidden": 0} or policy.calls != {"candidate": 1, "fallback": 0}:
        raise RuntimeError(f"Unexpected projection dispatch: {counts}, {policy.calls}")
    (plan, index), = policy.cache.values()
    choice = dict(problem=plan.problem, algorithm=plan.configs[index], counts=counts, calls=policy.calls)
    target, positions = record["labels"], record["positions"]
    shifted = MTPLoss._shift_labels(target, 1, positions)
    labels = torch.cat((target, shifted))[:, interval].contiguous()
    return plain, deployed, labels, tuple(record["head_weights"]), record["denominator"], choice


def ce_recheck(pristine, labels, weights, denominator):
    working = torch.empty_like(pristine, requires_grad=True)
    labels = labels.cuda()
    paths = {name: ROOT / LIBRARIES[name.removesuffix("-repeat")][0]
             for name in ("tuned", "tuned-repeat", "cuda13")}
    hashes = {name: digest(path) for name, path in paths.items()}
    for name, sha in hashes.items():
        if sha != LIBRARIES[name.removesuffix("-repeat")][1]:
            raise RuntimeError(f"Unexpected digest for {paths[name]}: {sha}")
    routes = {name: exact_folded_ce.ExactFoldedCrossEntropy(str(path), library_sha256=hashes[name])
              for name, path in paths.items()}

    def call(route):
        return evaluate(route, working, labels, weights, denominator)

    arms = eager(routes, pristine, working, call, 10, 24)
    spans = graph(routes, pristine, working, call, 24)
    if not all(route.scales_checked for route in routes.values()):
        raise RuntimeError("CE VJP scales were not verified")
    comparisons = verify(paths, routes, pristine, working, labels, weights, denominator, call)
    return {name: dict(sha256=hashes[name], eager=arms[name], graph=elapsed_series(spans[name]),
                       **comparisons[name]) for name in routes}


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._inductor.config.triton.cudagraphs = False
    source_receipt = Path(__file__).with_name("100b-32768.json")
    previous = json.loads(source_receipt.read_text())["fixture"]
    fixture = Path(previous["path"])
    plan_file = ROOT / "tuning/linear-plans.json"
    plain, deployed, labels, weights, denominator, choice = projections(fixture, previous["seq"], plan_file)
    if plain.shape != deployed.shape or plain.dtype != deployed.dtype:
        raise RuntimeError(f"Projection output layout mismatch: {plain.shape}, {deployed.shape}")
    sampled_sha = tensor_sha256(plain[..., ::4093])
    if (sampled_sha != previous["projected_vocab_stride_4093_sample_sha256"]
            or tensor_sha256(labels) != previous["label_sha256"]
            or list(plain.shape) != previous["logits_shape"]
            or weights != tuple(previous["head_weights"]) or denominator != previous["global_denominator"]):
        raise RuntimeError("Projection operands did not match the previous isolated CE replay")
    differences = bit_differences(plain.flatten(0, 1), deployed.flatten(0, 1))
    del plain
    sources = (Path(__file__).resolve(), Path(__file__).with_name("bench_long.py"), source_receipt,
               plan_file, ROOT / "tuning/head_run.py", ROOT / "tuning/linear_policy.py",
               Path(_staged_head_dw.__file__), Path(lt_ops.__file__), lt_ops.LIBRARY)
    output = dict(
        model="100b", batch=1, seq=previous["seq"], chunks=CHUNKS, chunk_index=CHUNKS-1,
        shape=list(deployed.shape), dtype=str(deployed.dtype), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        weights=weights, denominator=denominator, label_sha256=previous["label_sha256"],
        aten_sample_sha256=sampled_sha, deployed_sample_sha256=tensor_sha256(deployed[..., ::4093]),
        checked_logit_elements=deployed.numel(), different_logit_elements=differences,
        cuBLASLt_version=lt_ops._library().head_lt_version(), projection=choice,
        sources_sha256={str(path): digest(path) for path in sources},
        ce_recheck=ce_recheck(deployed, labels, weights, denominator) if differences else None,
    )
    write_json(source_receipt.with_name("100b-32768-projection-parity.json"), output)
    print(f"Projection differed in {differences} of {deployed.numel()} BF16 elements", flush=True)


if __name__ == "__main__":
    main()
