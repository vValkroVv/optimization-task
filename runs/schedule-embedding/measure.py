#!/usr/bin/env python3
"""Install the original capacity control before native imports, then delegate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = Path(__file__).resolve().parent
FLASH_MEASURE_PATH = TASK_ROOT / "runs/flash-qla/measure.py"
ORIGINAL_CAPACITY_PATH = TASK_ROOT / "runs/capacity-group-allreduce/capacity.original.py"
CURRENT_CAPACITY_PATH = TASK_ROOT / "gift/torchtitan/models/common/moe/dispatch/capacity.py"
CAPACITY_MODULE_NAME = "torchtitan.models.common.moe.dispatch.capacity"
CAPACITY_RELATIVE_PATH = "gift/torchtitan/models/common/moe/dispatch/capacity.py"
ORIGINAL_CAPACITY_SHA256 = "2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120"

VARIANTS = {"A", "B", "C", "D0", "S", "SE", "SE-fused"}
SPLIT_MTP_FIELD = "gfsdp_split_mtp_ffn"
SPLIT_LAYER0_FIELD = "gfsdp_split_layer0_ffn"
ROW_OWNER_FIELD = "gfsdp_row_owner_embeddings"
SCHEDULING_FIELD = "gfsdp_mtp_transition_prefetch"
EMBEDDING_FIELD = "gfsdp_row_owner_embedding_optimizations"
STAGES_FIELD = "gfsdp_row_owner_embedding_optimization_stages"
EMBEDDING_STAGES = {
    "SE": ["compact", "owner_add", "owner_lookup"],
    "SE-fused": ["compact", "owner_add", "owner_lookup", "routing"],
}
ENV_VARIANT = "TASK_SCHEDULE_EMBEDDING_VARIANT"
ENV_IDENTITY = "TASK_SCHEDULE_EMBEDDING_IDENTITY"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required schedule/embedding file is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head_capacity_bytes() -> bytes:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(TASK_ROOT),
            "show",
            f"HEAD:{CAPACITY_RELATIVE_PATH}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _load_original_capacity_module() -> tuple[Any, dict[str, Any]]:
    if ORIGINAL_CAPACITY_PATH.is_symlink() or not ORIGINAL_CAPACITY_PATH.is_file():
        raise RuntimeError(f"original capacity snapshot is missing or unsafe: {ORIGINAL_CAPACITY_PATH}")
    snapshot_sha256 = _file_sha256(ORIGINAL_CAPACITY_PATH)
    head_sha256 = _sha256_bytes(_git_head_capacity_bytes())
    if snapshot_sha256 != ORIGINAL_CAPACITY_SHA256 or head_sha256 != ORIGINAL_CAPACITY_SHA256:
        raise RuntimeError(
            "original capacity snapshot must match the pinned git HEAD bytes; "
            f"snapshot={snapshot_sha256} HEAD={head_sha256}"
        )

    existing = sys.modules.get(CAPACITY_MODULE_NAME)
    if existing is not None:
        origin = Path(getattr(existing, "__file__", "")).resolve()
        if origin != ORIGINAL_CAPACITY_PATH.resolve():
            raise RuntimeError(
                f"capacity module was imported before the control overlay: {origin}"
            )
        module = existing
    else:
        spec = importlib.util.spec_from_file_location(
            CAPACITY_MODULE_NAME, ORIGINAL_CAPACITY_PATH
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load original capacity module from {ORIGINAL_CAPACITY_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[CAPACITY_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(CAPACITY_MODULE_NAME) is module:
                sys.modules.pop(CAPACITY_MODULE_NAME, None)
            raise

    if sys.modules.get(CAPACITY_MODULE_NAME) is not module:
        raise RuntimeError("capacity control module registration changed during import")
    if Path(module.__file__).resolve() != ORIGINAL_CAPACITY_PATH.resolve():
        raise RuntimeError("capacity control module did not retain the original snapshot path")
    if _file_sha256(Path(module.__file__)) != ORIGINAL_CAPACITY_SHA256:
        raise RuntimeError("loaded capacity control module bytes changed")
    function = getattr(module, "_branchless_capacity_mask", None)
    if not callable(function) or function.__module__ != CAPACITY_MODULE_NAME:
        raise RuntimeError("original capacity control function is not available from the overlay")
    identity = {
        "module_name": CAPACITY_MODULE_NAME,
        "module_file": str(Path(module.__file__).resolve()),
        "module_sha256": ORIGINAL_CAPACITY_SHA256,
        "git_head_sha256": head_sha256,
        "function_name": "_branchless_capacity_mask",
        "function_module": function.__module__,
        "runtime_module_id": id(module),
        "current_capacity_file_sha256_at_import": _file_sha256(CURRENT_CAPACITY_PATH),
    }
    return module, identity


def _expected_controls(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise RuntimeError(f"unsupported schedule/embedding variant {variant!r}")
    row_owner = variant not in {"A", "B", "C"}
    return {
        SPLIT_MTP_FIELD: row_owner,
        SPLIT_LAYER0_FIELD: row_owner,
        ROW_OWNER_FIELD: row_owner,
        SCHEDULING_FIELD: variant in {"S", "SE", "SE-fused"},
        EMBEDDING_FIELD: False,
        STAGES_FIELD: EMBEDDING_STAGES.get(variant),
    }


def _expected_stages(variant: str) -> list[str]:
    return list(EMBEDDING_STAGES.get(variant, []))


def _record_stage_selection(
    trainer: Any,
    *,
    measurement: Path,
    run_id: str,
    rank: int,
    variant: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    from torchtitan.distributed.gfsdp.row_owner_embedding import RowOwnerEmbedding

    model_parts = getattr(trainer, "model_parts", None)
    if model_parts is None:
        raise RuntimeError("trainer.model_parts is unavailable for the stage-selection receipt")
    parts = model_parts if isinstance(model_parts, (list, tuple)) else [model_parts]
    modules = []
    for part_index, part in enumerate(parts):
        named_modules = getattr(part, "named_modules", None)
        if not callable(named_modules):
            raise RuntimeError(f"trainer.model_parts[{part_index}] is not a module")
        for name, module in named_modules():
            if isinstance(module, RowOwnerEmbedding):
                modules.append(
                    {
                        "name": f"model_parts[{part_index}].{name}".rstrip("."),
                        "optimization_stages": list(module.optimization_stages),
                    }
                )

    expected_stages = _expected_stages(variant)
    expected_count = 0 if variant in {"A", "B", "C"} else 1
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "rank": rank,
        "world_size": 8,
        "variant": variant,
        "source_sha256": identity["source_sha256"],
        "capture_stage": "after_trainer_init_before_first_optimizer_update",
        "device_reads": 0,
        "expected_module_count": expected_count,
        "expected_stages": expected_stages,
        "modules": modules,
    }
    path = measurement / f"embedding-stage-selection-rank-{rank:02d}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    actual_stages = [module["optimization_stages"] for module in modules]
    if len(modules) != expected_count or actual_stages != [expected_stages] * expected_count:
        raise RuntimeError(
            f"{variant} resolved row-owner stages differ from the expected selection: "
            f"modules={modules}, expected_count={expected_count}, "
            f"expected_stages={expected_stages}"
        )
    return record


def _source_hashes() -> dict[str, str]:
    paths = {
        "runs/schedule-embedding/run.py": BRIDGE_DIR / "run.py",
        "runs/schedule-embedding/measure.py": BRIDGE_DIR / "measure.py",
        "runs/schedule-embedding/campaign.py": BRIDGE_DIR / "campaign.py",
    }
    return {relative: _file_sha256(path) for relative, path in paths.items()}


def _validate_environment() -> tuple[str, dict[str, Any]]:
    variant = os.environ.get(ENV_VARIANT, "")
    controls = _expected_controls(variant)
    try:
        identity = json.loads(os.environ[ENV_IDENTITY])
    except KeyError as exc:
        raise RuntimeError(f"{ENV_IDENTITY} must be supplied by the native runner") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{ENV_IDENTITY} must contain JSON") from exc
    if not isinstance(identity, dict):
        raise RuntimeError(f"{ENV_IDENTITY} must contain an object")
    expected_controls_sha256 = _sha256_bytes(
        json.dumps(controls, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if (
        identity.get("schema_version") != 1
        or identity.get("variant") != variant
        or identity.get("controls") != controls
        or identity.get("variant_sha256") != _sha256_bytes(variant.encode("utf-8"))
        or identity.get("controls_sha256") != expected_controls_sha256
        or identity.get("source_sha256") != _source_hashes()
    ):
        raise RuntimeError("schedule/embedding variant or source identity changed before rank startup")
    return variant, identity


def _capacity_module_identity(module: Any) -> dict[str, Any]:
    current_path = Path(getattr(module, "__file__", "")).resolve()
    parent = sys.modules.get("torchtitan.models.common.moe.dispatch")
    parent_attribute = None if parent is None else getattr(parent, "capacity", None)
    return {
        "module_name": CAPACITY_MODULE_NAME,
        "module_file": str(current_path),
        "module_sha256": _file_sha256(current_path),
        "registered_same_object": sys.modules.get(CAPACITY_MODULE_NAME) is module,
        "runtime_module_id": id(module),
        "parent_capacity_attribute_same_object": (
            None if parent_attribute is None else parent_attribute is module
        ),
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _patch_flash_measure(
    flash: Any,
    module: Any,
    capacity_identity: dict[str, Any],
    identity: dict[str, Any],
    variant: str,
) -> None:
    original_write = flash._write_bootstrap_receipt
    original_update = flash._update_bootstrap_receipt
    original_load_base_measure = flash._load_base_measure

    def load_base_measure() -> Any:
        base = original_load_base_measure()
        original_init = base.WindowMeasurement.__init__
        if getattr(original_init, "_schedule_embedding_stage_hook", False):
            raise RuntimeError("stage-selection receipt hook is already installed")

        def record_stage_selection(self: Any, *args: Any, **kwargs: Any) -> None:
            try:
                original_init(self, *args, **kwargs)
                trainer = kwargs.get("trainer", args[0] if args else None)
                if trainer is None:
                    raise RuntimeError("WindowMeasurement was created without a trainer")
                _record_stage_selection(
                    trainer,
                    measurement=Path(self.output_dir),
                    run_id=self.run_id,
                    rank=self.rank,
                    variant=variant,
                    identity=identity,
                )
            finally:
                if base.WindowMeasurement.__init__ is record_stage_selection:
                    base.WindowMeasurement.__init__ = original_init

        record_stage_selection._schedule_embedding_stage_hook = True
        base.WindowMeasurement.__init__ = record_stage_selection
        return base

    def write_bootstrap_receipt(**kwargs: Any) -> dict[str, Any]:
        receipt = original_write(**kwargs)
        receipt["schedule_embedding"] = {
            "variant": variant,
            "controls": identity["controls"],
            "source_sha256": identity["source_sha256"],
            "config_guard": {
                "variant_sha256": identity["variant_sha256"],
                "controls_sha256": identity["controls_sha256"],
                "source_sha256": identity["source_sha256"],
            },
            "capacity_control": {
                **capacity_identity,
                "current_capacity_file_sha256_at_rank_start": _file_sha256(
                    CURRENT_CAPACITY_PATH
                ),
                "loaded_before_native_imports": True,
            },
        }
        path = Path(kwargs["measurement"]) / f"flash-qla-bootstrap-rank-{kwargs['rank']:02d}.json"
        _atomic_write_json(path, receipt)
        print(
            "schedule_embedding_bootstrap="
            + json.dumps(receipt["schedule_embedding"], sort_keys=True),
            flush=True,
        )
        return receipt

    def update_bootstrap_receipt(measurement: Path, rank: int, receipt: dict[str, Any]) -> None:
        control = _capacity_module_identity(module)
        control["current_capacity_file_sha256_final"] = _file_sha256(CURRENT_CAPACITY_PATH)
        control["current_capacity_file_unchanged"] = (
            control["current_capacity_file_sha256_final"]
            == receipt["schedule_embedding"]["capacity_control"][
                "current_capacity_file_sha256_at_rank_start"
            ]
        )
        control["module_path_is_original_snapshot"] = (
            Path(control["module_file"]).resolve() == ORIGINAL_CAPACITY_PATH.resolve()
            and control["module_sha256"] == ORIGINAL_CAPACITY_SHA256
        )
        receipt["schedule_embedding"]["capacity_control_final_check"] = control
        original_update(measurement, rank, receipt)

    flash._write_bootstrap_receipt = write_bootstrap_receipt
    flash._update_bootstrap_receipt = update_bootstrap_receipt
    flash._load_base_measure = load_base_measure


def main(argv: list[str] | None = None) -> Any:
    # Install the canonical control before importing FlashQLA's adapter or any
    # native TorchTitan modules. Its package parents cannot replace this entry.
    module, capacity_identity = _load_original_capacity_module()
    variant, identity = _validate_environment()
    flash_spec = importlib.util.spec_from_file_location(
        "_schedule_embedding_flash_measure", FLASH_MEASURE_PATH
    )
    if flash_spec is None or flash_spec.loader is None:
        raise RuntimeError(f"cannot load FlashQLA measurement adapter from {FLASH_MEASURE_PATH}")
    flash = importlib.util.module_from_spec(flash_spec)
    sys.modules[flash_spec.name] = flash
    try:
        flash_spec.loader.exec_module(flash)
    except BaseException:
        sys.modules.pop(flash_spec.name, None)
        raise
    _patch_flash_measure(flash, module, capacity_identity, identity, variant)
    return flash.main(argv)


if __name__ == "__main__":
    main()
