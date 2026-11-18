#!/usr/bin/env python3
"""Profile the selected S plus GPU selector recipe with and without Dynamo."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAPACITY_REVISION = "766fcc4a016a6e9a4af53042df2c3298596474cb"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_rank(args):
    enabled, *forwarded = args
    if enabled not in {"true", "false"} or (
        os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    ) == (enabled == "true"):
        raise RuntimeError("rank Dynamo environment differs from the planned arm")
    measure = load(ROOT / "runs/schedule-embedding/measure.py", "_compile_baseline_measure")

    def capacity_reference():
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{CAPACITY_REVISION}:{measure.CAPACITY_RELATIVE_PATH}"]
        )

    # The merged checkout contains the group-counter experiment. The measured
    # S bridge already loads the original capacity module on each rank. Verify
    # its bytes against the selected revision instead of the merged HEAD.
    measure._git_head_capacity_bytes = capacity_reference
    print(f"compile_baseline enabled={enabled} capacity_reference={CAPACITY_REVISION}", flush=True)
    measure.main(forwarded)


def run_profile(mode, run_id):
    enabled = mode == "compile"
    os.environ.pop("TORCH_COMPILE_DISABLE", None)
    os.environ["TORCHDYNAMO_DISABLE"] = "0" if enabled else "1"
    native = load(ROOT / "runs/capacity-selector/native.py", "_compile_baseline_native")
    native.RAW_ROOT = HERE / "artifacts/raw"
    load_native = native._load

    def load_schedule(path, name):
        schedule = load_native(path, name)
        append = schedule._append_variant_overrides
        expectations = schedule._variant_config_expectations

        def append_compile(command, variant):
            command = append(command, variant)
            pos = command.index(str(native.S_BRIDGE / "measure.py"))
            command[pos:pos + 1] = [str(Path(__file__).resolve()), "rank", str(enabled).lower()]
            return [*command, "--compile.enable", str(enabled).lower()]

        def compile_expectations(original, variant):
            expected = expectations(original, variant)

            def resolve(args):
                checks = expected(args)
                checks["compile.enable"] = enabled
                return checks

            return resolve

        schedule._append_variant_overrides = append_compile
        schedule._variant_config_expectations = compile_expectations
        return schedule

    native._load = load_schedule
    return native._run("trace", run_id)


def main():
    if sys.argv[1:2] == ["rank"]:
        run_rank(sys.argv[2:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "no_compile"))
    parser.add_argument("run_id")
    args = parser.parse_args()
    return run_profile(args.mode, args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
