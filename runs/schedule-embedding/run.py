#!/usr/bin/env python3
"""Launch a native schedule/embedding variant through the FlashQLA runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

TASK_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = Path(__file__).resolve().parent
FLASH_RUNNER_PATH = TASK_ROOT / "runs/flash-qla/run.py"
MEASURE_PATH = BRIDGE_DIR / "measure.py"
VARIANTS = ("A", "B", "C", "D0", "S", "SE", "SE-fused")

ROW_OWNER_FIELD = "gfsdp_row_owner_embeddings"
SPLIT_MTP_FIELD = "gfsdp_split_mtp_ffn"
SPLIT_LAYER0_FIELD = "gfsdp_split_layer0_ffn"
SCHEDULING_FIELD = "gfsdp_mtp_transition_prefetch"
EMBEDDING_FIELD = "gfsdp_row_owner_embedding_optimizations"
STAGES_FIELD = "gfsdp_row_owner_embedding_optimization_stages"
EMBEDDING_STAGES = {
    "SE": ("compact", "owner_add", "owner_lookup"),
    "SE-fused": ("compact", "owner_add", "owner_lookup", "routing"),
}
ENV_VARIANT = "TASK_SCHEDULE_EMBEDDING_VARIANT"
ENV_IDENTITY = "TASK_SCHEDULE_EMBEDDING_IDENTITY"


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_flash_runner() -> Any:
    return _load_module(FLASH_RUNNER_PATH, "_schedule_embedding_flash_runner")


def _parse_variant_args(argv: list[str] | None = None) -> tuple[str, list[str]]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    variant_flags = [
        token for token in tokens
        if token == "--variant" or token.startswith("--variant=")
    ]
    if len(variant_flags) != 1:
        raise SystemExit("--variant must appear exactly once")
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog="All remaining arguments are forwarded to runs/flash-qla/run.py.",
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args, forwarded = parser.parse_known_args(tokens)
    if not forwarded:
        parser.error("FlashQLA runner arguments are required")
    return args.variant, forwarded


def _controls(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported schedule/embedding variant {variant!r}")
    row_owner = variant not in {"A", "B", "C"}
    return {
        SPLIT_MTP_FIELD: row_owner,
        SPLIT_LAYER0_FIELD: row_owner,
        ROW_OWNER_FIELD: row_owner,
        SCHEDULING_FIELD: variant in {"S", "SE", "SE-fused"},
        EMBEDDING_FIELD: False,
        STAGES_FIELD: (
            None
            if variant not in EMBEDDING_STAGES
            else list(EMBEDDING_STAGES[variant])
        ),
    }


def _option_values(argv: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                raise RuntimeError(f"{option} lacks a value")
            values.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.partition("=")[2])
    return values


def _override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ", ".join(value) + "]"
    raise TypeError(f"unsupported schedule/embedding override value {value!r}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"schedule/embedding source is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = {
        "run.py": BRIDGE_DIR / "run.py",
        "measure.py": MEASURE_PATH,
        "campaign.py": BRIDGE_DIR / "campaign.py",
    }
    return {
        str(path.relative_to(TASK_ROOT)): _file_sha256(path)
        for path in paths.values()
    }


def _identity(variant: str) -> dict[str, Any]:
    controls = _controls(variant)
    return {
        "schema_version": 1,
        "variant": variant,
        "controls": controls,
        "source_sha256": _source_hashes(),
        "variant_sha256": _sha256_bytes(variant.encode("utf-8")),
        "controls_sha256": _sha256_bytes(
            json.dumps(controls, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def _append_variant_overrides(command: list[str], variant: str) -> list[str]:
    try:
        separator = command.index("--")
    except ValueError as exc:
        raise RuntimeError("generated native command has no `--` config separator") from exc
    train_args = command[separator + 1 :]
    controls = _controls(variant)
    overrides: list[str] = []
    for field, value in controls.items():
        option = f"--parallelism.{field}"
        existing = _option_values(train_args, option)
        expected = _override_value(value)
        if existing:
            if existing != [expected]:
                raise RuntimeError(
                    f"generated native command sets {option} to {existing}, expected {expected}"
                )
            continue
        overrides.extend((option, expected))
    return [*command, *overrides]


def _build_command_hook(builder: Callable[..., Any], base: Any, variant: str) -> Callable[..., list[str]]:
    replace_measure = builder(base, bridge_measure_path=MEASURE_PATH)

    def command(args: Any, paths: dict[str, Path]) -> list[str]:
        return _append_variant_overrides(replace_measure(args, paths), variant)

    return command


def _build_config_hashes_hook(
    builder: Callable[..., Any], base: Any, variant: str
) -> Callable[[], dict[str, str]]:
    original = builder(base)

    def config_hashes() -> dict[str, str]:
        hashes = original()
        identity = _identity(variant)
        hashes["schedule_embedding_variant_sha256"] = identity["variant_sha256"]
        hashes["schedule_embedding_controls_sha256"] = identity["controls_sha256"]
        for relative, digest in identity["source_sha256"].items():
            key = relative.removeprefix("runs/schedule-embedding/")
            hashes[f"schedule_embedding_{key.replace('.', '_')}_sha256"] = digest
        return hashes

    return config_hashes


def _write_bridge_receipt(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _build_environment_hook(
    builder: Callable[..., Any], base: Any, variant: str, **flash_options: Any
) -> Callable[[dict[str, Path]], dict[str, str]]:
    original = builder(base, **flash_options)

    def environment(paths: dict[str, Path]) -> dict[str, str]:
        env = original(paths)
        identity = _identity(variant)
        env[ENV_VARIANT] = variant
        env[ENV_IDENTITY] = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        attempt = Path(paths["attempt"]).resolve()
        _write_bridge_receipt(
            attempt / "schedule-embedding-bridge.json",
            {
                **identity,
                "attempt": str(attempt),
                "config_hash_guard": {
                    "schedule_embedding_variant_sha256": identity["variant_sha256"],
                    "schedule_embedding_controls_sha256": identity["controls_sha256"],
                    **{
                        f"schedule_embedding_{name.removeprefix('runs/schedule-embedding/').replace('.', '_')}_sha256": digest
                        for name, digest in identity["source_sha256"].items()
                    },
                },
            },
        )
        return env

    return environment


def _variant_config_expectations(original: Callable[[Any], dict[str, Any]], variant: str):
    controls = _controls(variant)

    def expectations(args: Any) -> dict[str, Any]:
        result = original(args)
        result.update(
            {
                f"parallelism.{field}": value
                for field, value in controls.items()
            }
        )
        return result

    return expectations


def main(argv: list[str] | None = None) -> int:
    variant, forwarded = _parse_variant_args(argv)
    flash = _load_flash_runner()

    original_command_builder = flash._build_command_hook
    original_hash_builder = flash._build_config_hashes_hook
    original_environment_builder = flash._build_environment_hook
    original_base_loader = flash._load_base_runner
    patched_bases: list[tuple[Any, Callable[[Any], dict[str, Any]]]] = []

    def build_command(base: Any) -> Callable[..., list[str]]:
        return _build_command_hook(original_command_builder, base, variant)

    def build_hashes(base: Any) -> Callable[[], dict[str, str]]:
        return _build_config_hashes_hook(original_hash_builder, base, variant)

    def build_environment(base: Any, **kwargs: Any) -> Callable[[dict[str, Path]], dict[str, str]]:
        return _build_environment_hook(original_environment_builder, base, variant, **kwargs)

    def load_base() -> Any:
        base = original_base_loader()
        if Path(base.TASK_ROOT).resolve() != TASK_ROOT.resolve():
            raise RuntimeError(f"base runner root differs from bridge root: {base.TASK_ROOT}")
        original_expectations = base._resolved_config_expectations
        base._resolved_config_expectations = _variant_config_expectations(
            original_expectations, variant
        )
        patched_bases.append((base, original_expectations))
        return base

    flash._build_command_hook = build_command
    flash._build_config_hashes_hook = build_hashes
    flash._build_environment_hook = build_environment
    flash._load_base_runner = load_base
    try:
        return int(flash.main(forwarded))
    finally:
        flash._build_command_hook = original_command_builder
        flash._build_config_hashes_hook = original_hash_builder
        flash._build_environment_hook = original_environment_builder
        flash._load_base_runner = original_base_loader
        for base, original_expectations in patched_bases:
            base._resolved_config_expectations = original_expectations


if __name__ == "__main__":
    raise SystemExit(main())
