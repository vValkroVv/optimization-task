"""Validate and compare the current and memory + cuBLASLt replay paths."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import frontier_lifetime
import replay


PAIRS = {2: "9.14", 4: "13.13", 8: "5.5", 16: "5.5"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("control", "stack"))
    parser.add_argument("chunks", type=int, choices=PAIRS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    replay_head = replay.make_head
    writer = None
    tuned = args.arm == "stack"

    def make_head(weight):
        head = replay_head(weight)
        import direct_dw

        if tuned:
            from frontier_dw_lib import PlannedWriter

            name = f"tune-{args.chunks}.json"
            plan = Path(__file__).with_name("frontier-dw") / name
            plan_data = plan.read_bytes()
            tuning = json.loads(plan_data)
            assert tuning["chunks"] == args.chunks
            planned = PlannedWriter(tuning, PAIRS[args.chunks])
            algorithms = planned.configs
            build = planned.lt.version
            plan_hash = hashlib.sha256(plan_data).hexdigest()
        else:
            planned = direct_dw.CublasWriter()
            algorithms = build = plan_hash = None

        class CountedWriter:
            def __init__(self):
                nonlocal writer
                writer = self
                self.calls = 0

            def __call__(self, grad, hidden, output, add):
                planned(grad, hidden, output, add)
                self.calls += 1

            def receipt(self):
                if tuned:
                    planned.close()
                return {
                    "engine": "cublasLtMatmul" if tuned else "cublasGemmEx",
                    "algorithm_pair": algorithms,
                    "cublaslt_version": build,
                    "tuning_sha256": plan_hash,
                    "calls": self.calls,
                }

        direct_dw.CublasWriter = CountedWriter
        return head

    replay.make_head = make_head
    samples = 1 if args.profile else 10
    sys.argv[1:] = [
        "combined" if tuned else "native", "current", str(args.output),
        "--chunks", str(args.chunks), "--writer", "cublas", "--sum-reduction",
        "--warmup", "10", "--samples", str(samples),
    ]
    if args.profile:
        import frontier_profile

        frontier_profile.main()
    else:
        sys.argv.extend(("--dump", str(args.output.with_suffix(".digest.json"))))
        frontier_lifetime.main()
    assert writer.calls == (1 + 10 + 1 + samples + int(not args.profile)) * args.chunks
    result = json.loads(args.output.read_text())
    result["dw_writer"] = writer.receipt()
    result["stack_arm"] = args.arm
    result["profile"] = args.profile
    result["stack_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
