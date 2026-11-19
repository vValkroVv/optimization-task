#!/usr/bin/env python3
"""Measure kernel changes on the selected compiled S plus selector recipe."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "abbbcaf0311409a817b16d794af79e3f3c106864"
PRODUCTION_REVISION = "1f02311a52ef979a84e164836583e73c1c5e6767"
PRODUCTION_SOURCE_SHA256 = (
    "44176fbe151220d6f1d511104e71ad43d6f520773076cedd2fb1ddf9fbf92850"
)
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
PRODUCTION = "production"
COMBINED_VARIANTS = (
    "gdn-output-fp8",
    "mla-rope",
    "cute-residual",
    "gdn-short-conv",
    "dense-pad",
)
GFSDP_CONTROLS = {
    "retain-tail-weights": "parallelism.gfsdp_retain_tail_weights",
    "fp8-expert-weights": "parallelism.gfsdp_fp8_expert_weights",
    "direct-expert-dw": "parallelism.gfsdp_direct_expert_dw",
}
DIRECT_DW_BUILD_ENV = "GIFT_DIRECT_EXPERT_DW_BUILD"
FIXED_TIMING_CONTROL = "efsdp-control-ep8-001-timing"
FIXED_TRACE_CONTROL = "efsdp-control-ep8-002-trace"


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
    if variant in {COMBINED, PRODUCTION}:
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


def verify_follow_up_revision(revision: str) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != head:
        raise ValueError(
            f"Expected the full follow-up revision {revision}, found {head}"
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            PRODUCTION_REVISION,
            head,
        ],
        check=True,
    )
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--",
            "gift/",
            str(HERE / "run.py"),
        ],
        text=True,
    )
    if status:
        raise ValueError(f"Follow-up source has uncommitted changes:\n{status}")


def training_overrides(command: list[str]) -> tuple[Path, dict[str, str]]:
    config, *tokens = command[command.index("--") + 1 :]
    overrides = dict(zip(tokens[::2], tokens[1::2], strict=True))
    if len(tokens) != 2 * len(overrides) or any(
        not flag.startswith("--") for flag in overrides
    ):
        raise RuntimeError("Expected unique training overrides with explicit values")
    overrides.pop("--dump_folder")
    return Path(config), overrides


def enforce_fixed_recipe(base, control: Path, controls: dict[str, bool], mode: str):
    resolved_path = control / "native/config/resolved_config.yaml"
    completion = base._load_json(control / "completion.json")
    if completion["status"] != "PASS" or completion[
        "resolved_config_sha256"
    ] != base.sha256_file(resolved_path):
        raise RuntimeError(
            f"Fixed production control receipt failed verification: {control}"
        )
    source = base._load_json(control / "metadata.json")
    if (
        source["source_snapshot_before"]["tracked_and_untracked_sha256"]
        != PRODUCTION_SOURCE_SHA256
    ):
        raise RuntimeError(
            f"Fixed control used a different production source: {control}"
        )
    control_hashes = source["config_hashes_before"]
    current_hashes = base._config_hashes()
    for key, digest in current_hashes.items():
        if control_hashes[key] != digest:
            raise RuntimeError(f"Fixed production input changed: {key}")

    _, expected_overrides = training_overrides(source["argv"])
    expected = base._load_yaml(resolved_path)
    expected.pop("dump_folder")
    if mode == "timing-140":
        expected_overrides["--training.steps"] = "140"
        expected["training"]["steps"] = 140
    for dotted, enabled in controls.items():
        section, field = dotted.split(".", 1)
        expected_overrides[f"--{dotted}"] = str(enabled).lower()
        expected[section][field] = enabled
    launch = base._launch
    verify = base._verify_resolved_config

    def launch_verified(command, *args):
        config, actual_overrides = training_overrides(command)
        if base.sha256_file(config) != control_hashes["generated_experiment_sha256"]:
            raise RuntimeError("Generated experiment differs from the fixed control")
        if actual_overrides != expected_overrides:
            raise RuntimeError("Native training overrides differ from the fixed recipe")
        return launch(command, *args)

    def verify_config(paths, args):
        digest = verify(paths, args)
        actual = base._load_yaml(paths["native"] / "config/resolved_config.yaml")
        actual.pop("dump_folder")
        if actual != expected:
            differences = sorted(
                key
                for key in actual.keys() | expected.keys()
                if key not in actual
                or key not in expected
                or actual[key] != expected[key]
            )
            raise RuntimeError(f"Fixed production config differs in {differences}")
        return digest

    base._launch = launch_verified
    base._verify_resolved_config = verify_config


def configure_direct_dw_rank(build: str) -> None:
    import torch

    from torchtitan.ops.float8.direct_expert_dw import configure_direct_expert_dw

    component = load(
        ROOT / "runs/efsdp-optimizations/direct-dw/direct_dw.py",
        "_direct_expert_dw_build",
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    native = component.load_direct_dw(build)
    configure_direct_expert_dw(native.k_grouped_fp8_gemm_nt_contiguous_rounded_f32)


def main() -> int:
    baseline = load(ROOT / "runs/compile-baseline/run.py", "_kernel_fusion_compile")
    if sys.argv[1:2] == ["rank"]:
        variant = sys.argv[2]
        expected = variant_switches(variant)
        actual = {flag for flag in SWITCHES.values() if os.environ.get(flag) == "1"}
        if actual != expected:
            raise RuntimeError(f"Expected {expected} but rank sees {actual}")
        print(f"kernel_fusion variant={variant} switches={sorted(actual)}", flush=True)
        if direct_dw_build := os.environ.get(DIRECT_DW_BUILD_ENV):
            configure_direct_dw_rank(direct_dw_build)
        baseline.run_rank(["true", *sys.argv[3:]])
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant", choices=("baseline", *SWITCHES, COMBINED, PRODUCTION)
    )
    parser.add_argument("mode", choices=("timing", "timing-140", "trace"))
    parser.add_argument("run_id")
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument(
        "--source-revision", help="Full committed HEAD for a production follow-up"
    )
    parser.add_argument(
        "--gfsdp-optimization", choices=GFSDP_CONTROLS, action="append", default=[]
    )
    parser.add_argument("--direct-dw-build", type=Path)
    args = parser.parse_args()
    if (args.variant in {"baseline", PRODUCTION}) != (args.candidate_source is None):
        parser.error("Only historical candidate runs require --candidate-source")
    if args.source_revision and args.variant != PRODUCTION:
        parser.error("Follow-up revisions require the production recipe")
    if args.gfsdp_optimization and not args.source_revision:
        parser.error("GFSDP optimizations require a committed --source-revision")
    direct_dw_requested = "direct-expert-dw" in args.gfsdp_optimization
    if direct_dw_requested != (args.direct_dw_build is not None):
        parser.error("--direct-dw-build is required for the direct-expert-dw arm")
    os.environ.pop(DIRECT_DW_BUILD_ENV, None)
    direct_dw_manifest = None
    if direct_dw_requested:
        build_root = args.direct_dw_build.resolve(strict=True)
        direct_dw_manifest = build_root / "manifest.json"
        if not build_root.is_relative_to(ROOT) or not direct_dw_manifest.is_file():
            parser.error(
                "--direct-dw-build needs a verified build inside the task checkout"
            )
        os.environ[DIRECT_DW_BUILD_ENV] = str(build_root)
    if args.source_revision:
        verify_follow_up_revision(args.source_revision)
    controls = (
        {
            field: name in args.gfsdp_optimization
            for name, field in GFSDP_CONTROLS.items()
        }
        if args.source_revision
        else {}
    )
    activate(args.variant)
    os.environ.pop("TORCH_COMPILE_DISABLE", None)
    os.environ["TORCHDYNAMO_DISABLE"] = "0"
    native = load(ROOT / "runs/capacity-selector/native.py", "_kernel_fusion_native")
    native.RAW_ROOT = HERE / "artifacts/raw"
    if args.variant == PRODUCTION:
        native.S_SOURCE_SHA256 = PRODUCTION_SOURCE_SHA256
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
            command.extend(("--compile.enable", "true"))
            for field, enabled in controls.items():
                command.extend((f"--{field}", str(enabled).lower()))
            return command

        def compiled_expectations(original, variant):
            expected = expectations(original, variant)

            def resolve(args):
                checks = expected(args)
                checks["compile.enable"] = True
                checks.update(controls)
                return checks

            return resolve

        def load_flash_runner():
            flash = load_flash()
            load_base = flash._load_base_runner

            def load_base_runner():
                base = load_base()
                if args.variant == PRODUCTION:
                    if args.source_revision:
                        snapshot = base.source_snapshot(ROOT)
                        if (
                            snapshot["head"] != args.source_revision
                            or snapshot["untracked_file_count"]
                        ):
                            raise RuntimeError(
                                "Follow-up GIFT source changed during preflight"
                            )
                        native.S_SOURCE_SHA256 = snapshot[
                            "tracked_and_untracked_sha256"
                        ]
                        control = native.RAW_ROOT / (
                            FIXED_TRACE_CONTROL
                            if args.mode == "trace"
                            else FIXED_TIMING_CONTROL
                        )
                        enforce_fixed_recipe(base, control, controls, args.mode)
                        config_hashes = base._config_hashes

                        def follow_up_hashes():
                            hashes = {
                                **config_hashes(),
                                "kernel_fusion_runner_sha256": native._sha256(
                                    Path(__file__)
                                ),
                            }
                            if direct_dw_manifest is not None:
                                hashes["direct_dw_build_manifest_sha256"] = (
                                    native._sha256(direct_dw_manifest)
                                )
                            return hashes

                        base._config_hashes = follow_up_hashes
                    return base
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
