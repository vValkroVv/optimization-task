"""Repeat the allocating eager A/A/B comparison for one previously selected plan."""

import argparse
import json
import os
from pathlib import Path

import torch

from measure import (LinearPolicy, Timer, _library, benchmark, bits_equal, chunk_input,
                     evaluate_candidates, exact_gradient, isolated_memory, save, sha256)


def confirm(path, out):
    torch.backends.cuda.matmul.allow_tf32 = False
    initial = json.loads(path.read_text())
    hidden, weight, targets, fixture = chunk_input(
        initial["model"], initial["sequence"], Path(initial["fixture"])
    )
    assert fixture["folded_hidden_shape"] == initial["folded_hidden_shape"]
    assert fixture["denominator"] == initial["denominator"]
    assert initial["torch"] == torch.__version__ and initial["cublaslt"] == _library().head_lt_version()
    assert initial["capability"] == list(torch.cuda.get_device_capability())
    timer = Timer()

    def projection():
        return torch.nn.functional.linear(hidden, weight)

    if initial["role"] == "projection":
        assert initial["control"] == "ATen linear"
        lhs, rhs = hidden.flatten(0, 1), weight.T
        owner = (*hidden.shape[:-1], weight.size(0))
        control = projection
    else:
        assert initial["control"] == "ATen mm"
        lhs, rhs = exact_gradient(projection, targets, fixture), weight
        owner = None

        def control():
            return torch.mm(lhs, rhs)

    chosen = initial["selected"]
    choices = path.parent / initial["finalists"][chosen]["choices"]
    policy = LinearPolicy(choices)

    def candidate():
        return policy.run(lhs, rhs, owner)

    output_shape = initial["output_shape"]
    actual, expected = candidate(), control()
    parity = bits_equal(actual.view(output_shape), expected.view(output_shape))
    elements = expected.numel()
    del actual, expected
    expected_desc = tuple(initial["finalists"][chosen]["descriptor"])
    cache_match = next(iter(policy.cache.values()))
    exact_descriptor = cache_match is not None and cache_match[0].problem == expected_desc
    assert parity and exact_descriptor and policy.calls == {"candidate": 1, "fallback": 0}
    torch.cuda.empty_cache()
    calls = {"control": control, "control_repeat": control, "candidate": candidate}
    rounds = 24
    warmed = benchmark(calls, timer, rounds)
    isolation = isolated_memory(calls, timer)
    verdict = evaluate_candidates(warmed, isolation, {}, rounds)["candidate"]
    assert policy.calls["fallback"] == 0
    save(out, dict(
        confirm_file=path.name,
        confirm_sha256=sha256(path),
        choices_file=choices.name,
        choices_sha256=sha256(choices),
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cublaslt=_library().head_lt_version(),
        same_shape_bitwise=parity,
        compared_elements=elements,
        loaded_exact_descriptor=exact_descriptor,
        activation=policy.calls,
        fixture=fixture,
        warmed_eager=warmed,
        isolated_memory=isolation,
        comparison=verdict,
    ))
    print(out, verdict, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    confirm(args.initial, args.out)
