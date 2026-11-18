#!/usr/bin/env python3
"""Launch the capacity-selector comparison through the measured S runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parents[1]
S_BRIDGE = TASK_ROOT / "runs/schedule-embedding"
RAW_ROOT = HERE / "artifacts/raw"
CACHE_ROOT = (
    TASK_ROOT / "runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache"
)
ORIGINAL_CAPACITY = TASK_ROOT / "runs/capacity-group-allreduce/capacity.original.py"
S_SOURCE_SHA256 = "6fc4f3c7e6deba5875d13d675704dbdee27b4e09ca3fd106f3a0f61ee6b2b2f5"
S_BRIDGE_HASHES = {
    S_BRIDGE / "run.py": "f887559fee173ce6a2fafe804d3d26bbedc5ad1e8b5737c283f2572a3937e72a",
    S_BRIDGE / "measure.py": "d68ca4f294af76b524503034629d2a5979b03bfee4ec33164f9021125a738d1a",
    ORIGINAL_CAPACITY: "2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120",
}
POLICY_PATHS = tuple(
    f"model.config.blocks.{block}.moe.experts.token_dispatcher.token_drop"
    for block in ("gdn_moe", "mla_moe")
)
SELECTOR_PATHS = tuple(f"{path}.branchless_capacity_drop" for path in POLICY_PATHS)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_args(mode: str, run_id: str) -> list[str]:
    common = [
        "--variant", "S",
        "--gdn-backend", "external_gate",
        "--routing-metadata", "fused",
        "--cross-entropy", "stable",
        "--num-prefetch-batches", "32",
        "--epoch-size", "102400",
        "--gfsdp-reuse-bf16-expert-weights",
        "--gfsdp-split-mtp-ffn",
        "--gfsdp-split-layer0-ffn",
        "--gfsdp-row-owner-embeddings",
        "--run-id", run_id,
        "--output-root", str(RAW_ROOT),
        "--compiler-cache-root", str(CACHE_ROOT),
    ]
    if mode == "capture":
        return [*common, "--steps", "1", "--warmup", "0", "--window", "0"]
    if mode == "trace":
        return [
            *common,
            "--steps", "21", "--warmup", "20", "--window", "0",
            "--profile-update", "21", "--profile-microbatches", "3",
        ]
    return [
        *common,
        "--steps", "20", "--warmup", "10", "--window", "10",
        "--observe-compiler-cache",
    ]


def _append_selector(command: list[str], enabled: bool) -> list[str]:
    for path in SELECTOR_PATHS:
        option = f"--{path}"
        if any(arg == option or arg.startswith(f"{option}=") for arg in command):
            raise RuntimeError(f"duplicate selector override: {option}")
        command.extend((option, str(enabled).lower()))
    return command


def _capture_rank(argv: list[str]) -> None:
    measure = _load(S_BRIDGE / "measure.py", "_capacity_selector_s_measure")
    install_bridge = measure._patch_flash_measure

    def patch_flash_measure(flash, *args):
        install_bridge(flash, *args)
        bootstrap = flash._write_bootstrap_receipt

        def record_bootstrap(**kwargs):
            # This runs after rank cache and CUDA device setup, before training.
            from component import install_route_capture

            output = Path(kwargs["measurement"]).parent / "routing-fixtures"
            install_route_capture(output)
            return bootstrap(**kwargs)

        flash._write_bootstrap_receipt = record_bootstrap

    measure._patch_flash_measure = patch_flash_measure
    measure.main(argv)


def _run(mode: str, run_id: str) -> int:
    for path, expected in S_BRIDGE_HASHES.items():
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"historical S input changed or is missing: {path}")
    schedule = _load(S_BRIDGE / "run.py", "_capacity_selector_s_runner")
    append_variant = schedule._append_variant_overrides
    config_expectations = schedule._variant_config_expectations
    load_flash = schedule._load_flash_runner
    selector_enabled = mode in {"candidate", "trace"}

    def append_overrides(command, variant):
        command = _append_selector(append_variant(command, variant), selector_enabled)
        if mode == "capture":
            pos = command.index(str(S_BRIDGE / "measure.py"))
            command[pos:pos + 1] = [str(Path(__file__).resolve()), "_capture-rank"]
        return command

    def expectations(original, variant):
        inherited = config_expectations(original, variant)

        def verify(args):
            expected = inherited(args)
            key = "parallelism.gfsdp_row_owner_embedding_optimization_stages"
            # S's default None is omitted from YAML. Its separate all-rank
            # startup check verifies that no embedding optimization stages run.
            if expected.pop(key) is not None:
                raise RuntimeError("S must use the default embedding optimization stages")
            for path in POLICY_PATHS:
                expected.update({
                    f"{path}.strategy": "fixed_capacity",
                    f"{path}.expert_capacity_tokens": 50000,
                    f"{path}.drop_by_group_capacity": True,
                    f"{path}.branchless_capacity_drop": selector_enabled,
                })
            return expected

        return verify

    def verified_flash_runner():
        flash = load_flash()
        load_base = flash._load_base_runner

        def verified_base_runner():
            base = load_base()
            snapshot = base.source_snapshot
            config_hashes = base._config_hashes

            def verify_snapshot(root):
                state = snapshot(root)
                actual = state["tracked_and_untracked_sha256"]
                if actual != S_SOURCE_SHA256:
                    raise RuntimeError(
                        f"GIFT source differs from S-140-004: {actual} != {S_SOURCE_SHA256}"
                    )
                return state

            def hashes():
                result = config_hashes()
                result["capacity_selector_native_sha256"] = _sha256(Path(__file__))
                result["capacity_selector_original_capacity_sha256"] = _sha256(ORIGINAL_CAPACITY)
                if mode == "capture":
                    result["capacity_selector_capture_sha256"] = _sha256(HERE / "component.py")
                return result

            base.source_snapshot = verify_snapshot
            base._config_hashes = hashes
            return base

        flash._load_base_runner = verified_base_runner
        return flash

    schedule._append_variant_overrides = append_overrides
    schedule._variant_config_expectations = expectations
    schedule._load_flash_runner = verified_flash_runner
    result = int(schedule.main(_native_args(mode, run_id)))
    if mode == "capture" and result == 0:
        output = RAW_ROOT / run_id / "routing-fixtures"
        if not all((output / f"rank{rank}.pt").is_file() for rank in range(8)):
            raise RuntimeError(f"capture did not produce all eight rank fixtures: {output}")
        print(f"routing_fixtures={output}", flush=True)
    return result


def main(argv: list[str]) -> int:
    if argv[:1] == ["_capture-rank"]:
        _capture_rank(argv[1:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("capture", "control", "candidate", "trace"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    return _run(args.mode, args.run_id)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
