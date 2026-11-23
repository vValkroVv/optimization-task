"""Replay the native live CuTe residual component with a different TMA depth."""

import argparse
import json
import os
from pathlib import Path
import sys

import residual_replay_probe
from torchtitan.models.common.norms.cute_gated_norm import forward


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--late-residual", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args, replay_args = parser.parse_known_args()
    if args.stages != 0 and args.stages < 2:
        parser.error("--stages must be 0 for baseline or at least 2")

    pick_stages = forward.pick_f2_stages
    observed = {}

    def select_stages(hidden, chunks, budget=220 * 1024, *, add_residual=False):
        default = pick_stages(hidden, chunks, budget, add_residual=add_residual)
        if not add_residual:
            return default
        stages = args.stages or default
        if stages > default:
            raise ValueError(f"TMA depth {stages} exceeds baseline {default}")
        observed[(hidden, chunks)] = {"default": default, "actual": stages}
        return stages

    forward.pick_f2_stages = select_stages
    os.environ["GIFT_CUTE_NORM_LATE_RESIDUAL"] = str(int(args.late_residual))
    sys.argv = [
        sys.argv[0], *replay_args, "--scenario", "live", "--arm", "B",
        "--output", str(args.output),
    ]
    residual_replay_probe.main()
    if not observed:
        raise RuntimeError("No residual F2 plan selected")
    receipt = json.loads(args.output.read_text())
    receipt["late_residual"] = args.late_residual
    receipt["requested_stages"] = args.stages
    receipt["residual_plans"] = [
        {"hidden": hidden, "chunks": chunks, **values}
        for (hidden, chunks), values in observed.items()
    ]
    receipt["implementation_path"] = str(Path(forward.__file__).resolve())
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
