#!/usr/bin/env python3
"""Measure kernel changes on the selected compiled S plus selector recipe."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "abbbcaf0311409a817b16d794af79e3f3c106864"
SWITCHES = {
    "mla-rope": "GIFT_MLA_FUSED_ROPE",
    "dense-pad": "GIFT_FP8_QUANTIZE_WEIGHT_TAIL",
    "gdn-short-conv": "GIFT_GDN_FUSE_SHORT_CONV_BWD",
    "gdn-output-fp8": "GIFT_GDN_FUSE_OUTPUT_NORM_FP8",
    "moe-dual-fp8": "GIFT_FUSE_MOE_SWIGLU_DUAL_FP8",
    "cute-residual": "GIFT_CUTE_NORM_FUSE_RESIDUAL",
    "mla-gate-fp8": "GIFT_MLA_FUSED_GATE_FP8",
}
COMBINED = "combined-five"
COMBINED_VARIANTS = (
    "gdn-output-fp8",
    "mla-rope",
    "cute-residual",
    "gdn-short-conv",
    "dense-pad",
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def variant_switches(variant: str) -> set[str]:
    if variant == "baseline":
        return set()
    if variant == COMBINED:
        return {SWITCHES[name] for name in COMBINED_VARIANTS}
    return {SWITCHES[variant]}


def activate(variant: str) -> None:
    for switch in SWITCHES.values():
        os.environ.pop(switch, None)
    for switch in variant_switches(variant):
        os.environ[switch] = "1"


def timing_140(command: list[str]) -> list[str]:
    for flag, value in (("--steps", "140"), ("--warmup", "40"), ("--window", "100")):
        command[command.index(flag) + 1] = value
    return command


def main() -> int:
    baseline = load(ROOT / "runs/compile-baseline/run.py", "_kernel_fusion_compile")
    if sys.argv[1:2] == ["rank"]:
        variant = sys.argv[2]
        expected = variant_switches(variant)
        actual = {flag for flag in SWITCHES.values() if os.environ.get(flag) == "1"}
        if actual != expected:
            raise RuntimeError(f"Expected {expected} but rank sees {actual}")
        print(f"kernel_fusion variant={variant} switches={sorted(actual)}", flush=True)
        baseline.run_rank(["true", *sys.argv[3:]])
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("baseline", *SWITCHES, COMBINED))
    parser.add_argument("mode", choices=("timing", "timing-140", "trace"))
    parser.add_argument("run_id")
    parser.add_argument("--candidate-source", type=Path)
    args = parser.parse_args()
    if (args.variant == "baseline") != (args.candidate_source is None):
        parser.error("Candidate runs require an isolated --candidate-source worktree")
    activate(args.variant)
    os.environ.pop("TORCH_COMPILE_DISABLE", None)
    os.environ["TORCHDYNAMO_DISABLE"] = "0"
    native = load(ROOT / "runs/capacity-selector/native.py", "_kernel_fusion_native")
    native.RAW_ROOT = HERE / "artifacts/raw"
    if args.mode == "timing-140":
        native_args = native._native_args
        native._native_args = lambda mode, run_id: timing_140(native_args(mode, run_id))
    load_native = native._load
    baseline_sha256 = native.S_SOURCE_SHA256
    source = (
        args.candidate_source.resolve(strict=True) if args.candidate_source else ROOT
    )
    if args.candidate_source:
        source.relative_to(ROOT / "artifacts/kernel-fusion")

    def load_schedule(path, name):
        schedule = load_native(path, name)
        append = schedule._append_variant_overrides
        expectations = schedule._variant_config_expectations
        load_flash = schedule._load_flash_runner

        def append_variant(command, variant):
            command = append(command, variant)
            pos = command.index(str(native.S_BRIDGE / "measure.py"))
            command[pos : pos + 1] = [
                str(Path(__file__).resolve()),
                "rank",
                args.variant,
            ]
            return [*command, "--compile.enable", "true"]

        def compiled_expectations(original, variant):
            expected = expectations(original, variant)

            def resolve(args):
                checks = expected(args)
                checks["compile.enable"] = True
                return checks

            return resolve

        def load_flash_runner():
            flash = load_flash()
            load_base = flash._load_base_runner

            def load_base_runner():
                base = load_base()
                base.verify_task_root(source)
                baseline_snapshot = base.source_snapshot(ROOT)
                if (
                    baseline_snapshot["head"] != PIN
                    or baseline_snapshot["tracked_and_untracked_sha256"]
                    != baseline_sha256
                ):
                    raise RuntimeError("The measured baseline source changed")
                if args.candidate_source:
                    snapshot = base.source_snapshot
                    candidate = snapshot(source)
                    if candidate["head"] != PIN:
                        raise RuntimeError(
                            "Candidate worktree has a different base revision"
                        )
                    canonical_gift = base.GIFT_ROOT
                    canonical_inputs = base._canonical_input_paths
                    validate_cache = base._validate_compiler_cache_reuse

                    def canonical_input_paths():
                        # The canonical YAML stays in the baseline tree. Its
                        # relative stream path is anchored at that GIFT root.
                        runtime_gift = base.GIFT_ROOT
                        base.GIFT_ROOT = canonical_gift
                        try:
                            return canonical_inputs()
                        finally:
                            base.GIFT_ROOT = runtime_gift

                    def validate_candidate_cache(paths, runtime, canonical_sha256):
                        # The runtime preflight independently verifies the
                        # candidate module origin. The historical cache receipt
                        # names the canonical checkout for that same package.
                        packages = dict(runtime["packages"])
                        packages["torchtitan"] = {
                            **packages["torchtitan"],
                            "path": str(
                                (canonical_gift / "torchtitan/__init__.py").resolve()
                            ),
                        }
                        return validate_cache(
                            paths, {**runtime, "packages": packages}, canonical_sha256
                        )

                    base.GIFT_ROOT = source / "gift"
                    base._canonical_input_paths = canonical_input_paths
                    base._validate_compiler_cache_reuse = validate_candidate_cache
                    base.source_snapshot = lambda _root: snapshot(source)
                    native.S_SOURCE_SHA256 = candidate["tracked_and_untracked_sha256"]
                return base

            flash._load_base_runner = load_base_runner
            return flash

        schedule._append_variant_overrides = append_variant
        schedule._variant_config_expectations = compiled_expectations
        schedule._load_flash_runner = load_flash_runner
        return schedule

    native._load = load_schedule
    return native._run("trace" if args.mode == "trace" else "candidate", args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
