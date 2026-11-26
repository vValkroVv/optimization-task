"""Repeat a borderline short projection using its original measured plan."""

import argparse
import json
from pathlib import Path

import torch

from measure import ROOT, LinearPolicy, _library, bits_equal, chunk_input, region


def confirm(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    prior_path = args.out / "confirm.json"
    prior = json.loads(prior_path.read_text())
    assert prior["torch"] == torch.__version__ and prior["cublaslt"] == _library().head_lt_version()
    assert prior["capability"] == list(torch.cuda.get_device_capability())
    hidden, weight, _, fixture = chunk_input(args.model, args.seq, args.chunks,
                                              ROOT / "inputs" / f"{args.model}.pt")
    assert prior["folded_hidden_shape"] == fixture["folded_hidden_shape"]
    assert prior["denominator"] == fixture["denominator"]
    choices = args.out / prior["finalists"][prior["selected"]]["choices"]
    policy = LinearPolicy(choices)
    lhs, rhs = hidden.flatten(0, 1), weight.T
    owner = (*hidden.shape[:-1], weight.size(0))

    def control():
        return torch.nn.functional.linear(hidden, weight)

    def candidate():
        return policy.run(lhs, rhs, owner)

    actual, reference = candidate(), control()
    parity = bits_equal(actual.flatten(0, 1), reference.flatten(0, 1))
    assert parity and policy.calls == {"candidate": 1, "fallback": 0}
    count = reference.numel()
    del actual, reference
    torch.cuda.empty_cache()
    timer = region.Timer()
    calls = {"control": control, "control_repeat": control, "candidate": candidate}
    rounds = 24
    warmed = region.benchmark(calls, timer, rounds)
    isolated = region.isolated_memory(calls, timer)
    graph, graph_errors = region.graph_measure(calls, timer, 9)
    verdict = region.evaluate_candidates(warmed, isolated, graph, rounds)["candidate"]
    assert policy.calls["fallback"] == 0
    region.save(args.out / "recheck.json", dict(
        initial_confirm_sha256=region.sha256(prior_path), choice_sha256=region.sha256(choices),
        bitwise=parity, compared_elements=count, fixture=fixture,
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), activation=policy.calls,
        warmed_eager=warmed, isolated_memory=isolated, graph_ms=graph, graph_errors=graph_errors,
        comparison=verdict, source_sha256=region.sha256(Path(__file__).resolve()),
    ))
    print(verdict, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("700b",), required=True)
    parser.add_argument("--seq", type=int, choices=(8192, 16384), required=True)
    parser.add_argument("--chunks", type=int, choices=(8,), required=True)
    parser.add_argument("--role", choices=("projection",), required=True)
    parser.add_argument("--out", type=Path, required=True)
    confirm(parser.parse_args())
