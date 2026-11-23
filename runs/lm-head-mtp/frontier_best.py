"""Run the matched default-allocator control and the complete memory stack."""

import argparse
import json
import os
from pathlib import Path
import sys

import frontier_gather
import frontier_stack
import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("control", "memory"))
    parser.add_argument("chunks", type=int, choices=frontier_stack.PAIRS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    candidate = args.arm == "memory"
    assert os.environ["PYTORCH_ALLOC_CONF"] == f"expandable_segments:{candidate}"
    native_head = replay.make_head
    gather = None
    allocator = None

    def make_head(weight):
        nonlocal gather, allocator
        head = native_head(weight)
        gather = frontier_gather.install_head_probe(head, candidate)
        allocator = replay.torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]
        assert allocator == candidate
        return head

    replay.make_head = make_head
    sys.argv[1:] = ["stack" if candidate else "control", str(args.chunks), str(args.output)]
    frontier_stack.main()
    assert gather["gather_calls"] == gather["native_all_gather_into_tensor_calls"] == 24
    result = json.loads(args.output.read_text())
    result["full_arm"] = args.arm
    result["expandable_segments"] = allocator
    result["gather_probe"] = gather
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
