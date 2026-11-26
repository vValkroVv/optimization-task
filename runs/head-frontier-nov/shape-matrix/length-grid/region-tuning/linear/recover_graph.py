"""Recheck projection graph timing using a single capture for both control slots."""

import argparse
import json
import os
from pathlib import Path

import torch

from measure import (CALIBRATED, LinearPolicy, Timer, _library, capture,
                     chunk_input, groups_by_index, save, sha256)


def run(path):
    torch.backends.cuda.matmul.allow_tf32 = False
    data = json.loads(path.read_text())
    assert data["role"] == "projection" and data["graph_errors"]
    assert data["cuda_visible_devices"] == os.environ["CUDA_VISIBLE_DEVICES"]
    assert data["torch"] == torch.__version__ and data["cublaslt"] == _library().head_lt_version()
    hidden, weight, _, receipt = chunk_input(data["model"], data["sequence"], Path(data["fixture"]))
    assert receipt["folded_hidden_shape"] == data["folded_hidden_shape"]
    shape = (*hidden.shape[:-1], weight.size(0))
    lhs, rhs = hidden.flatten(0, 1), weight.T
    timer = Timer()
    if data["control"] == "calibrated projection":
        calibrated = LinearPolicy(CALIBRATED)

        def control():
            return calibrated.run(lhs, rhs, shape)
    else:
        def control():
            return torch.nn.functional.linear(hidden, weight)

    control_graph = capture(control)
    graphs = {"control": control_graph, "control_repeat": control_graph}
    policies = {}
    for name in data["comparison"]:
        policy = LinearPolicy(path.parent / data["finalists"][name]["choices"])
        policies[name] = policy

        def call():
            return policy.run(lhs, rhs, shape)

        graphs[name] = capture(call)
        assert policy.calls["candidate"] == 4 and policy.calls["fallback"] == 0
    outputs = {name: [] for name in graphs}
    for i in range(5):
        for name in groups_by_index(list(graphs), i):
            graph = graphs[name]

            def replay():
                for _ in range(3):
                    graph.replay()

            _, sample = timer.sample(replay)
            outputs[name].append(sample["event_ms"] / 3)
    save(path.parent / "graph-recovery.json", dict(
        original_confirm_sha256=sha256(path),
        capture_layout="identical control slots replay the same captured graph",
        cuda_visible_devices=data["cuda_visible_devices"],
        graph_ms=outputs,
        graph_errors={},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirm", type=Path)
    run(parser.parse_args().confirm)
