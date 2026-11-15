#!/usr/bin/env python3
"""Run the native streaming benchmark with an explicitly selected FlashQLA backend."""

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
BASE_RUNNER_PATH = TASK_ROOT / "runs/overnight-streaming-perf/run.py"
BRIDGE_MEASURE_PATH = BRIDGE_DIR / "measure.py"
DEFAULT_OUTPUT_ROOT = BRIDGE_DIR / "artifacts/raw"
BACKENDS = ("fla", "external_gate", "fused_gate")
ROUTING_METADATA = ("original", "fused")
CROSS_ENTROPY = ("original", "stable")
DEFAULT_ROUTING_METADATA = "fused"
DEFAULT_CROSS_ENTROPY = "stable"

ENV_BACKEND = "TASK_FLASHQLA_BACKEND"
ENV_CAPTURE_MODE = "TASK_FLASHQLA_CAPTURE_MODE"
ENV_CAPTURE_DIR = "TASK_FLASHQLA_CAPTURE_DIR"
ENV_TILELANG_ROOT = "TASK_FLASHQLA_TILELANG_ROOT"
ENV_SOURCE_IDENTITY = "TASK_FLASHQLA_SOURCE_IDENTITY"
ENV_ROUTING_METADATA = "TASK_FLASHQLA_ROUTING_METADATA"
ENV_CROSS_ENTROPY = "TASK_FLASHQLA_CROSS_ENTROPY"


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load task runner module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_base_runner() -> Any:
    return _load_module(BASE_RUNNER_PATH, "_flash_qla_overnight_runner")


def _parse_bridge_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog=(
            "All remaining arguments are forwarded unchanged to "
            "runs/overnight-streaming-perf/run.py."
        ),
    )
    parser.add_argument(
        "--gdn-backend",
        choices=BACKENDS,
        required=True,
        help="explicit native FlashQLA backend variant for this run",
    )
    parser.add_argument(
        "--capture-fixtures",
        type=Path,
        metavar="PATH",
        help="capture native FLA VJP fixtures into this fresh run-owned directory",
    )
    parser.add_argument(
        "--routing-metadata",
        choices=ROUTING_METADATA,
        default=DEFAULT_ROUTING_METADATA,
        help="routing metadata implementation (default: fused)",
    )
    parser.add_argument(
        "--cross-entropy",
        choices=CROSS_ENTROPY,
        default=DEFAULT_CROSS_ENTROPY,
        help="cross-entropy implementation (default: stable)",
    )
    args, forwarded = parser.parse_known_args(argv)
    if args.capture_fixtures is not None and args.gdn_backend != "fla":
        parser.error("--capture-fixtures requires --gdn-backend fla")
    return args, forwarded


def _add_default_output_root(argv: list[str], output_root: Path) -> list[str]:
    if any(arg == "--output-root" or arg.startswith("--output-root=") for arg in argv):
        return list(argv)
    return [*argv, "--output-root", str(output_root)]


def _file_sha256(path: Path) -> str:
    if not path.is_symlink() and not path.is_file():
        raise RuntimeError(f"FlashQLA bridge source file is missing: {path}")
    if path.is_symlink():
        raise RuntimeError(f"FlashQLA bridge source file must not be a symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> tuple[str | None, int]:
    if not root.is_dir():
        return None, 0
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink():
            raise RuntimeError(f"FlashQLA source tree contains a symlink: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return (digest.hexdigest() if count else None), count


def _bridge_source_hashes() -> dict[str, str]:
    paths = (
        BRIDGE_DIR / "run.py",
        BRIDGE_MEASURE_PATH,
        BRIDGE_DIR / "capture.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"FlashQLA bridge source files are missing: {missing}")
    return {
        str(path.relative_to(TASK_ROOT)): _file_sha256(path)
        for path in paths
    }


def _source_identity(base: Any) -> dict[str, Any]:
    backend_path = Path(base.GIFT_ROOT) / "torchtitan/experiments/flash_qla/backend.py"
    staged_package = Path(base.GIFT_ROOT) / "contrib/flash-qla/flash_qla"
    staged_sha256, staged_file_count = _tree_sha256(staged_package)
    if not backend_path.is_file():
        raise RuntimeError(f"FlashQLA backend integration is missing: {backend_path}")
    if staged_sha256 is None:
        raise RuntimeError(f"staged FlashQLA Python source is missing: {staged_package}")
    bridge_sources = _bridge_source_hashes()
    head = base._git(TASK_ROOT, "rev-parse", "HEAD").stdout.decode().strip()
    return {
        "task_git_head": head,
        "backend_module": {
            "path": str(backend_path),
            "sha256": _file_sha256(backend_path),
        },
        "staged_python_source": {
            "path": str(staged_package),
            "sha256": staged_sha256,
            "file_count": staged_file_count,
        },
        "bridge_sources": bridge_sources,
    }


def _path_has_symlink_component(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    if current.is_symlink():
        return True
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            return True
    return False


def _capture_directory(requested: Path, attempt: Path) -> Path:
    root = TASK_ROOT.resolve()
    attempt = attempt.resolve()
    candidate = requested if requested.is_absolute() else root / requested
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("--capture-fixtures must be an absolute task-owned path without ..")
    try:
        relative = candidate.relative_to(attempt)
    except ValueError as exc:
        raise RuntimeError("--capture-fixtures must be inside this run attempt") from exc
    if not relative.parts:
        raise RuntimeError("--capture-fixtures must name a new directory inside the attempt")
    if _path_has_symlink_component(candidate, attempt):
        raise RuntimeError("--capture-fixtures must not contain symlink components")
    resolved = candidate.resolve()
    if resolved.exists():
        raise RuntimeError(f"--capture-fixtures must be a fresh directory: {resolved}")
    return resolved


def _build_config_hashes_hook(base: Any) -> Callable[[], dict[str, Any]]:
    original = base._config_hashes

    def config_hashes() -> dict[str, Any]:
        hashes = original()
        hashes["flash_qla_bridge_sources"] = _bridge_source_hashes()
        return hashes

    return config_hashes


def _build_environment_hook(
    base: Any,
    *,
    backend: str,
    capture_fixtures: Path | None,
    routing_metadata: str = DEFAULT_ROUTING_METADATA,
    cross_entropy: str = DEFAULT_CROSS_ENTROPY,
) -> Callable[[dict[str, Path]], dict[str, str]]:
    original = base._run_environment

    def run_environment(paths: dict[str, Path]) -> dict[str, str]:
        env = original(paths)
        for key in (
            ENV_BACKEND,
            ENV_CAPTURE_MODE,
            ENV_CAPTURE_DIR,
            ENV_TILELANG_ROOT,
            ENV_SOURCE_IDENTITY,
            ENV_ROUTING_METADATA,
            ENV_CROSS_ENTROPY,
            "TASK_COMPILER_CACHE_ROOT",
            "TILELANG_CACHE_DIR",
            "TILELANG_TMP_DIR",
        ):
            env.pop(key, None)

        existing_pythonpath = env.get("PYTHONPATH")
        existing_entries = existing_pythonpath.split(os.pathsep) if existing_pythonpath else []
        staged_sources = (
            str((Path(base.GIFT_ROOT).resolve() / "contrib/flash-linear-attention").resolve()),
            str((Path(base.GIFT_ROOT).resolve() / "contrib/flash-qla").resolve()),
        )
        required_paths = (
            str(Path(base.GIFT_ROOT).resolve()),
            str(Path(base.HELPER_DIR).resolve()),
            str(BRIDGE_DIR.resolve()),
        )
        env["PYTHONPATH"] = os.pathsep.join(
            dict.fromkeys((*staged_sources, *existing_entries, *required_paths))
        )
        env[ENV_BACKEND] = backend
        env[ENV_CAPTURE_MODE] = "fixtures" if capture_fixtures is not None else "disabled"
        env[ENV_ROUTING_METADATA] = routing_metadata
        env[ENV_CROSS_ENTROPY] = cross_entropy

        attempt = Path(paths["attempt"]).resolve()
        cache_path = Path(paths["cache"])
        if not cache_path.is_absolute() or ".." in cache_path.parts:
            raise RuntimeError("compiler cache root must be an absolute task path")
        if cache_path.is_symlink() or not cache_path.is_dir():
            raise RuntimeError(f"compiler cache root is missing or unsafe: {cache_path}")
        cache_root = cache_path.resolve()
        try:
            cache_root.relative_to(Path(base.TASK_ROOT).resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"compiler cache root must remain inside the task: {cache_root}"
            ) from exc
        fresh_cache_root = (attempt / "cache").resolve()
        cache_reuse = cache_root != fresh_cache_root
        tilelang_root = cache_root / "tilelang"
        env["TASK_COMPILER_CACHE_ROOT"] = str(cache_root)

        def ensure_cache_directory(path: Path) -> None:
            if path.is_symlink():
                raise RuntimeError(f"TileLang cache directory must not be a symlink: {path}")
            path.mkdir(exist_ok=cache_reuse)
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"TileLang cache directory is missing or unsafe: {path}")

        ensure_cache_directory(tilelang_root)
        ensure_cache_directory(tilelang_root / "tmp")
        for rank in range(8):
            rank_root = tilelang_root / f"rank_{rank}"
            ensure_cache_directory(rank_root)
            ensure_cache_directory(rank_root / "tmp")

        env[ENV_TILELANG_ROOT] = str(tilelang_root)
        env["TILELANG_CACHE_DIR"] = str(tilelang_root)
        env["TILELANG_TMP_DIR"] = str(tilelang_root / "tmp")
        source_identity = _source_identity(base)
        env[ENV_SOURCE_IDENTITY] = json.dumps(source_identity, sort_keys=True)

        capture_dir: Path | None = None
        if capture_fixtures is not None:
            if not (BRIDGE_DIR / "capture.py").is_file():
                raise RuntimeError("fixture capture was requested but capture.py is missing")
            capture_dir = _capture_directory(capture_fixtures, attempt)
            capture_dir.mkdir(parents=True, exist_ok=False)
            env[ENV_CAPTURE_DIR] = str(capture_dir)

        receipt = {
            "schema_version": 1,
            "run_id": attempt.name,
            "backend_variant": backend,
            "routing_metadata": routing_metadata,
            "cross_entropy": cross_entropy,
            "capture_mode": env[ENV_CAPTURE_MODE],
            "capture_fixture_dir": None if capture_dir is None else str(capture_dir),
            "capture_timing_disposition": (
                "discarded; fixture acquisition is not a performance comparison"
                if capture_dir is not None
                else None
            ),
            "source_identity": source_identity,
            "compiler_cache_root": str(cache_root),
            "tilelang_cache_policy": "reuse" if cache_reuse else "fresh",
            "tilelang_cache_root": str(tilelang_root),
            "tilelang_tmp_root": str(tilelang_root / "tmp"),
            "tilelang_rank_directories": {
                str(rank): {
                    "cache": str(tilelang_root / f"rank_{rank}"),
                    "tmp": str(tilelang_root / f"rank_{rank}" / "tmp"),
                }
                for rank in range(8)
            },
        }
        with (attempt / "flash-qla-bridge.json").open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, sort_keys=True, indent=2)
            handle.write("\n")
        return env

    return run_environment


def _replace_measure_script(
    command: list[str], base_measure_path: Path, bridge_measure_path: Path
) -> list[str]:
    expected = str(base_measure_path.resolve())
    matches = [index for index, value in enumerate(command) if value == expected]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one generated base measure.py path, found {len(matches)}"
        )
    replaced = list(command)
    replaced[matches[0]] = str(bridge_measure_path.resolve())
    return replaced


def _build_command_hook(
    base: Any, bridge_measure_path: Path = BRIDGE_MEASURE_PATH
) -> Callable[[Any, dict[str, Path]], list[str]]:
    original = base._command
    base_measure_path = Path(base.HELPER_DIR) / "measure.py"

    def command(args: Any, paths: dict[str, Path]) -> list[str]:
        generated = original(args, paths)
        return _replace_measure_script(generated, base_measure_path, bridge_measure_path)

    return command


def main(argv: list[str] | None = None) -> int:
    bridge_args, forwarded = _parse_bridge_args(argv)
    forwarded = _add_default_output_root(forwarded, DEFAULT_OUTPUT_ROOT)
    base = _load_base_runner()
    if Path(base.TASK_ROOT).resolve() != TASK_ROOT.resolve():
        raise RuntimeError(
            f"base runner task root differs from FlashQLA bridge root: {base.TASK_ROOT}"
        )

    original_environment = base._run_environment
    original_command = base._command
    original_config_hashes = base._config_hashes
    helper_dir = Path(base.HELPER_DIR).resolve()
    inserted_helper = str(helper_dir) not in sys.path
    if inserted_helper:
        sys.path.insert(0, str(helper_dir))
    base._run_environment = _build_environment_hook(
        base,
        backend=bridge_args.gdn_backend,
        capture_fixtures=bridge_args.capture_fixtures,
        routing_metadata=bridge_args.routing_metadata,
        cross_entropy=bridge_args.cross_entropy,
    )
    base._command = _build_command_hook(base)
    base._config_hashes = _build_config_hashes_hook(base)
    try:
        return int(base.main(forwarded))
    finally:
        base._run_environment = original_environment
        base._command = original_command
        base._config_hashes = original_config_hashes
        if inserted_helper:
            try:
                sys.path.remove(str(helper_dir))
            except ValueError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

