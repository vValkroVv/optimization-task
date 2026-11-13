"""Boundary-only snapshots of existing compiler and kernel-cache state."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEEP_GEMM_GUARD_ENV = "DG_CACHE_FORCE_SKIP"
DEEP_GEMM_GUARD_VALUE = "0"
DEEP_GEMM_GUARD_BUILD = "2.3.0+f093a4c"
DEEP_GEMM_GUARD_EXTENSION_SHA256 = "99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26"
SOURCE_PINS = {
    "torch._dynamo.utils": "21217311dbe8022f3793e9dfd61a2be66c50ecb97078c83b4c414e538fc5e832",
    "triton.runtime.jit": "38b8151fcceec1a0210a696775b7c64d7fac5e9b69eee2dda7e93c1e2a377509",
    "cute_gated_norm.common": "ab2912468af5661b0b9f23ddaaf36a829d0924a64e25b7c2455cabe70e12a205",
}
COMPILE_ONLY_EVENTS = (
    "_compile.compile_inner",
    "compile_attempt_",
    ".fw_compiler",
    ".bw_compiler",
    "fx_codegen_and_compile",
    "CachingAutotuner.",
)
OBSERVED_SOURCES = [
    "Dynamo compilation_time_metrics key/list identities and lengths",
    "Inductor async_compile_cache_miss/hit counters",
    "active CuTe COMPILED keys/object identities",
    "Triton JIT registry/function identities and existing device kernel-cache keys/object identities",
    "DeepGEMM DG_CACHE_FORCE_SKIP=0 guard for the verified binary and tested grouped APIs",
]
COVERAGE_LIMITS = [
    "Asynchronous compiler workers may outlive the parent timing scope, so Inductor counters and Triton cache state provide partial evidence only.",
    "Triton cache changes may represent a disk-cache load or specialization; any change conservatively opens the compilation gate.",
    "DeepGEMM cache misses are rejected for the verified binary; runtime behavior was probed through m_grouped and k_grouped, while other native JIT paths remain unverified.",
]
CUTE_MODULE = "torchtitan.models.common.norms.cute_gated_norm.common"


def deep_gemm_guard_manifest(*, enabled: bool, activate_after_update: int | None) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "activate_after_update": activate_after_update,
        "flag": DEEP_GEMM_GUARD_ENV,
        "guarded_value": DEEP_GEMM_GUARD_VALUE,
        "scope": "cache misses rejected after settled warmup; startup unchanged",
    }


def _key_token(value: Any) -> str:
    """Encode map keys without inspecting values such as tensors or cache entries."""
    if value is None:
        encoded: Any = ["NoneType"]
    elif type(value) is bool:
        encoded = ["bool", value]
    elif type(value) is int:
        encoded = ["int", value]
    elif type(value) is str:
        encoded = ["str", value]
    elif type(value) is float:
        encoded = ["float", value.hex()]
    elif type(value) is tuple:
        encoded = ["tuple", [_key_token(item) for item in value]]
    else:
        value_type = type(value)
        encoded = [f"{value_type.__module__}.{value_type.__qualname__}", id(value)]
    return json.dumps(encoded, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_deep_gemm_guard_identity(identity: Any) -> list[str]:
    if not isinstance(identity, dict):
        return ["DeepGEMM guard binary identity is missing or malformed"]
    errors: list[str] = []
    if identity.get("build_version") != DEEP_GEMM_GUARD_BUILD:
        errors.append("DeepGEMM guard build version does not match the verified binary")
    if identity.get("sha256") != DEEP_GEMM_GUARD_EXTENSION_SHA256:
        errors.append("DeepGEMM guard extension hash does not match the verified binary")
    path = identity.get("extension_path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        errors.append("DeepGEMM guard extension path is missing or not absolute")
    return errors


def verify_deep_gemm_guard_identity(deep_gemm_module: Any | None = None) -> dict[str, str]:
    """Hash the imported extension before training; this path performs no CUDA work."""
    module = deep_gemm_module or importlib.import_module("deep_gemm")
    extension = getattr(module, "_C", None)
    path_value = getattr(extension, "__file__", None)
    build = getattr(module, "__build_version__", None)
    if not isinstance(path_value, str):
        raise RuntimeError("DeepGEMM guard extension path is unavailable")
    path = Path(path_value).resolve()
    try:
        digest = _sha256(path)
    except OSError as exc:
        raise RuntimeError(f"could not hash DeepGEMM guard extension {path}: {exc}") from exc
    identity = {
        "build_version": str(build),
        "extension_path": str(path),
        "sha256": digest,
    }
    errors = validate_deep_gemm_guard_identity(identity)
    if errors:
        raise RuntimeError("; ".join(errors))
    return identity


def validate_deep_gemm_guard_receipt(
    receipt: Any,
    *,
    activation_after_update: int,
    end_after_update: int,
    activation_sync_calls: int,
) -> list[str]:
    if not isinstance(receipt, dict):
        return ["rank receipt lacks a DeepGEMM guard record"]
    errors: list[str] = []
    if receipt.get("environment_variable") != DEEP_GEMM_GUARD_ENV:
        errors.append("guard environment variable is missing or incorrect")
    if receipt.get("guarded_value") != DEEP_GEMM_GUARD_VALUE:
        errors.append("guarded environment value is missing or incorrect")
    for field, expected in (
        ("activation_after_update", activation_after_update),
        ("end_after_update", end_after_update),
        ("activation_sync_calls", activation_sync_calls),
    ):
        if type(receipt.get(field)) is not int or receipt[field] != expected:
            errors.append(f"guard {field} is missing or differs from {expected}")
    if "previous_value" not in receipt:
        errors.append("guard previous environment value is missing")
    elif receipt["previous_value"] is not None and not isinstance(
        receipt["previous_value"], str
    ):
        errors.append("guard previous environment value is malformed")
    for field in ("flag_after_activation", "flag_at_end_boundary"):
        if receipt.get(field) != DEEP_GEMM_GUARD_VALUE:
            errors.append(f"guard {field} is missing or is not '0'")
    errors.extend(validate_deep_gemm_guard_identity(receipt.get("binary_identity")))
    return errors


def _source_record(name: str, path: Path | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": None if path is None else str(path),
        "expected_sha256": SOURCE_PINS.get(name),
        "sha256": None,
    }
    if path is None:
        record["error"] = "source path is unavailable"
    else:
        try:
            record["sha256"] = _sha256(path)
        except OSError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _source_errors(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict) or not isinstance(metadata.get("source_files"), dict):
        return ["compilation observer source metadata or file hashes are missing"]
    errors: list[str] = []
    files = metadata["source_files"]
    for name, expected in SOURCE_PINS.items():
        record = files.get(name)
        if not isinstance(record, dict):
            errors.append(f"source metadata for {name} is missing")
        elif record.get("expected_sha256") != expected or record.get("sha256") != expected:
            errors.append(f"source hash for {name} does not match the reviewed pin")
    async_source = files.get("torch._inductor.async_compile")
    if not isinstance(async_source, dict) or not isinstance(async_source.get("sha256"), str):
        errors.append("source hash for torch._inductor.async_compile is missing")
    return errors


def _unavailable(error: str) -> dict[str, Any]:
    return {"available": False, "error": error}


class CompilationWindowObserver:
    """Read existing CPU-side compiler state without binding devices or changing caches."""

    def __init__(self) -> None:
        self._imports: dict[str, Any] = {}
        self._import_errors: dict[str, str] = {}
        for name in ("torch._dynamo.utils", "triton.runtime.jit"):
            try:
                self._imports[name] = importlib.import_module(name)
            except Exception as exc:  # the parser reports missing observer coverage
                self._import_errors[name] = f"{type(exc).__name__}: {exc}"
        self.source_metadata = self._build_source_metadata()

    @staticmethod
    def _module_path(module: Any | None) -> Path | None:
        path = getattr(module, "__file__", None)
        return None if not isinstance(path, str) else Path(path).resolve()

    def _build_source_metadata(self) -> dict[str, Any]:
        dynamo = self._imports.get("torch._dynamo.utils")
        triton = self._imports.get("triton.runtime.jit")
        dynamo_path = self._module_path(dynamo)
        triton_path = self._module_path(triton)
        async_path = (
            None
            if dynamo_path is None
            else dynamo_path.parents[1] / "_inductor/async_compile.py"
        )
        cute_path = (
            Path(__file__).resolve().parents[2]
            / "gift/torchtitan/models/common/norms/cute_gated_norm/common.py"
        )
        try:
            import torch

            torch_version = str(torch.__version__)
        except Exception as exc:
            torch_version = None
            self._import_errors["torch"] = f"{type(exc).__name__}: {exc}"
        try:
            triton_version = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            triton_version = None
        source_files = {
            "torch._dynamo.utils": _source_record("torch._dynamo.utils", dynamo_path),
            "torch._inductor.async_compile": _source_record(
                "torch._inductor.async_compile", async_path
            ),
            "triton.runtime.jit": _source_record("triton.runtime.jit", triton_path),
            "cute_gated_norm.common": _source_record("cute_gated_norm.common", cute_path),
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "torch_version": torch_version,
            "triton_version": triton_version,
            "source_files": source_files,
            "observed_sources": list(OBSERVED_SOURCES),
            "coverage_limits": list(COVERAGE_LIMITS),
            "observer_import_errors": dict(self._import_errors),
        }

    def _snapshot_dynamo(self) -> dict[str, Any]:
        module = self._imports.get("torch._dynamo.utils")
        if module is None:
            return _unavailable(self._import_errors.get("torch._dynamo.utils", "module is unavailable"))
        metrics = getattr(module, "compilation_time_metrics", None)
        if not isinstance(metrics, Mapping):
            return _unavailable("compilation_time_metrics is not a mapping")
        timing_lists: dict[str, Any] = {}
        for name, values in list(metrics.items()):
            if not isinstance(name, str) or not isinstance(values, list):
                return _unavailable("compilation_time_metrics contains an unsupported key or list")
            timing_lists[name] = {"identity": id(values), "length": len(values)}
        return {"available": True, "mapping_id": id(metrics), "timing_lists": timing_lists}

    def _snapshot_inductor(self) -> dict[str, Any]:
        module = self._imports.get("torch._dynamo.utils")
        counters = None if module is None else getattr(module, "counters", None)
        if not isinstance(counters, Mapping):
            return _unavailable("Dynamo counters is not a mapping")
        inductor = counters.get("inductor")  # .get avoids defaultdict insertion
        if inductor is not None and not isinstance(inductor, Mapping):
            return _unavailable("Dynamo counters['inductor'] is not a mapping")
        values: dict[str, Any] = {}
        for name in ("async_compile_cache_miss", "async_compile_cache_hit"):
            present = inductor is not None and name in inductor
            value = inductor.get(name) if present else None
            if present and type(value) is not int:
                return _unavailable(f"Inductor counter {name} is not an integer")
            values[name] = {"present": present, "value": value}
        return {
            "available": True,
            "counters_id": id(counters),
            "inductor_mapping_id": None if inductor is None else id(inductor),
            "counters": values,
        }

    def _snapshot_cute(self) -> dict[str, Any]:
        module = sys.modules.get(CUTE_MODULE)  # do not import a previously inactive path
        if module is None:
            return _unavailable(f"active module {CUTE_MODULE} is not loaded")
        compiled = getattr(module, "COMPILED", None)
        if not isinstance(compiled, Mapping):
            return _unavailable("active CuTe COMPILED cache is not a mapping")
        entries = {_key_token(key): id(value) for key, value in list(compiled.items())}
        return {
            "available": True,
            "module_id": id(module),
            "mapping_id": id(compiled),
            "entries": entries,
        }

    def _snapshot_triton(self) -> dict[str, Any]:
        module = self._imports.get("triton.runtime.jit")
        registry = None if module is None else getattr(module, "_triton_jit_function_registry", None)
        if not isinstance(registry, Mapping):
            return _unavailable("Triton JIT function registry is not a mapping")
        functions: dict[str, int] = {}
        device_cache_maps: dict[str, int] = {}
        kernel_cache_maps: dict[str, int] = {}
        kernel_entries: dict[str, int] = {}
        for name, function in list(registry.items()):
            if not isinstance(name, str):
                return _unavailable("Triton JIT registry contains a non-string key")
            functions[name] = id(function)
            device_caches = getattr(function, "device_caches", None)
            if not isinstance(device_caches, Mapping):
                return _unavailable(f"Triton JIT function {name} has no device_caches mapping")
            device_cache_maps[name] = id(device_caches)
            for device, cache_state in list(device_caches.items()):
                if not isinstance(cache_state, tuple) or len(cache_state) != 5:
                    return _unavailable(f"Triton JIT function {name} has an unsupported cache tuple")
                kernel_cache = cache_state[0]
                if not isinstance(kernel_cache, Mapping):
                    return _unavailable(f"Triton JIT function {name} kernel_cache is not a mapping")
                device_token = _key_token(device)
                pair_token = _key_token((name, device_token))
                kernel_cache_maps[pair_token] = id(kernel_cache)
                for key, value in list(kernel_cache.items()):
                    entry_token = _key_token((name, device_token, _key_token(key)))
                    kernel_entries[entry_token] = id(value)
        return {
            "available": True,
            "registry_id": id(registry),
            "function_ids": functions,
            "device_caches_ids": device_cache_maps,
            "kernel_cache_ids": kernel_cache_maps,
            "kernel_entry_ids": kernel_entries,
        }

    def snapshot(self) -> dict[str, Any]:
        """Read only existing maps; never index Triton's device defaultdict."""
        sources: dict[str, Any] = {}
        errors: list[str] = []
        for name, method in (
            ("dynamo", self._snapshot_dynamo),
            ("inductor", self._snapshot_inductor),
            ("cute", self._snapshot_cute),
            ("triton", self._snapshot_triton),
        ):
            try:
                state = method()
            except Exception as exc:
                state = _unavailable(f"{type(exc).__name__}: {exc}")
            sources[name] = state
            if state.get("available") is not True:
                errors.append(f"{name}: {state.get('error', 'unavailable')}")
        return {"schema_version": SCHEMA_VERSION, "sources": sources, "snapshot_errors": errors}


def _positive_id(value: Any) -> bool:
    return type(value) is int and value > 0


def _snapshot_sources(snapshot: Any, label: str, errors: list[str]) -> dict[str, Any] | None:
    if (
        not isinstance(snapshot, dict)
        or type(snapshot.get("schema_version")) is not int
        or snapshot["schema_version"] != SCHEMA_VERSION
    ):
        errors.append(f"compilation observer {label} snapshot is missing or malformed")
        return None
    sources = snapshot.get("sources")
    if not isinstance(sources, dict):
        errors.append(f"compilation observer {label} sources are missing or malformed")
        return None
    snapshot_errors = snapshot.get("snapshot_errors")
    if not isinstance(snapshot_errors, list) or any(not isinstance(item, str) for item in snapshot_errors):
        errors.append(f"compilation observer {label} snapshot errors are missing or malformed")
        return None
    if snapshot_errors:
        errors.extend(f"{label} snapshot: {item}" for item in snapshot_errors)
        return None
    return sources


def _validate_source_state(name: str, state: dict[str, Any], label: str, errors: list[str]) -> bool:
    valid = True

    def require_id(value: Any, field: str) -> None:
        nonlocal valid
        if not _positive_id(value):
            errors.append(f"{name} {label} {field} is missing or malformed")
            valid = False

    if name == "dynamo":
        require_id(state.get("mapping_id"), "timing-metrics mapping identity")
        lists = state.get("timing_lists")
        if not isinstance(lists, dict):
            errors.append(f"Dynamo {label} timing_lists is missing or malformed")
            return False
        for event, entry in lists.items():
            if not isinstance(event, str) or not isinstance(entry, dict):
                errors.append(f"Dynamo {label} timing-list entry is malformed")
                valid = False
                continue
            require_id(entry.get("identity"), f"timing list {event} identity")
            if type(entry.get("length")) is not int or entry["length"] < 0:
                errors.append(f"Dynamo {label} timing list {event} length is missing or malformed")
                valid = False
    elif name == "inductor":
        require_id(state.get("counters_id"), "counters mapping identity")
        if "inductor_mapping_id" not in state:
            errors.append(f"Inductor {label} counter-map identity field is missing")
            valid = False
        inductor_id = state.get("inductor_mapping_id")
        if inductor_id is not None and not _positive_id(inductor_id):
            errors.append(f"Inductor {label} counter-map identity is malformed")
            valid = False
        counters = state.get("counters")
        required = {"async_compile_cache_miss", "async_compile_cache_hit"}
        if not isinstance(counters, dict) or not required.issubset(counters):
            errors.append(f"Inductor {label} miss/hit counter records are missing or malformed")
            return False
        for counter_name in required:
            counter = counters[counter_name]
            if (
                not isinstance(counter, dict)
                or "present" not in counter
                or "value" not in counter
                or type(counter.get("present")) is not bool
            ):
                errors.append(f"Inductor {label} counter {counter_name} present flag is missing or malformed")
                valid = False
                continue
            present, value = counter["present"], counter.get("value")
            if present:
                if not _positive_id(inductor_id):
                    errors.append(f"Inductor {label} counter {counter_name} exists without a counter map")
                    valid = False
                if type(value) is not int or value < 0:
                    errors.append(f"Inductor {label} counter {counter_name} value is missing or malformed")
                    valid = False
            elif value is not None:
                errors.append(f"Inductor {label} absent counter {counter_name} has a value")
                valid = False
    elif name == "cute":
        require_id(state.get("module_id"), "module identity")
        require_id(state.get("mapping_id"), "COMPILED mapping identity")
        entries = state.get("entries")
        if not isinstance(entries, dict):
            errors.append(f"CuTe {label} entries are missing or malformed")
            return False
        for key, identity in entries.items():
            if not isinstance(key, str) or not _positive_id(identity):
                errors.append(f"CuTe {label} COMPILED entry is malformed")
                valid = False
    elif name == "triton":
        require_id(state.get("registry_id"), "JIT registry identity")
        maps = (
            "function_ids",
            "device_caches_ids",
            "kernel_cache_ids",
            "kernel_entry_ids",
        )
        for field in maps:
            entries = state.get(field)
            if not isinstance(entries, dict):
                errors.append(f"Triton {label} {field} is missing or malformed")
                valid = False
                continue
            if any(not isinstance(key, str) or not _positive_id(identity) for key, identity in entries.items()):
                errors.append(f"Triton {label} {field} contains malformed identities")
                valid = False
        functions = state.get("function_ids")
        device_caches = state.get("device_caches_ids")
        if isinstance(functions, dict) and isinstance(device_caches, dict) and set(functions) != set(device_caches):
            errors.append(f"Triton {label} function and device-cache keys differ")
            valid = False
    else:
        errors.append(f"unknown compilation observer source {name}")
        valid = False
    return valid


def _pair(
    before_sources: dict[str, Any] | None,
    after_sources: dict[str, Any] | None,
    name: str,
    errors: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    states: list[dict[str, Any] | None] = []
    for label, sources in (("before", before_sources), ("after", after_sources)):
        if sources is None:
            states.append(None)
            continue
        state = sources.get(name) if isinstance(sources, dict) else None
        if not isinstance(state, dict) or state.get("available") is not True:
            detail = state.get("error") if isinstance(state, dict) else "source snapshot is missing"
            errors.append(f"{name} {label} snapshot unavailable: {detail}")
            states.append(None)
        elif not _validate_source_state(name, state, label, errors):
            states.append(None)
        else:
            states.append(state)
    return states[0], states[1]


def _identity_deltas(
    before: Any,
    after: Any,
    *,
    label: str,
    errors: list[str],
    replacement_is_error: bool = False,
) -> tuple[dict[str, Any], list[str], int]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        errors.append(f"{label} state is malformed")
        return {}, [], 0
    deltas: dict[str, Any] = {}
    changes: list[str] = []
    unchanged = 0
    for key in sorted(set(before).union(after)):
        old, new = before.get(key), after.get(key)
        if old is None:
            changes.append(key)
            deltas[key] = {"before": None, "after": new, "change": "added"}
        elif new is None:
            errors.append(f"{label} entry {key} disappeared")
            deltas[key] = {"before": old, "after": None, "change": "removed"}
        elif old != new:
            if replacement_is_error:
                errors.append(f"{label} entry {key} identity changed")
            else:
                changes.append(key)
            deltas[key] = {"before": old, "after": new, "change": "identity_changed"}
        else:
            unchanged += 1
    return deltas, changes, unchanged


def _compare_dynamo(before: dict[str, Any], after: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    if before.get("mapping_id") != after.get("mapping_id"):
        errors.append("Dynamo compilation_time_metrics mapping was replaced")
    old, new = before.get("timing_lists"), after.get("timing_lists")
    deltas: dict[str, Any] = {}
    compile_growth: list[str] = []
    runtime_growth: list[str] = []
    if not isinstance(old, dict) or not isinstance(new, dict):
        errors.append("Dynamo timing-list state is malformed")
        return {"status": "incomplete", "timing_list_deltas": deltas}
    for name in sorted(set(old).union(new)):
        first, last = old.get(name), new.get(name)
        compile_only = any(marker in name for marker in COMPILE_ONLY_EVENTS)
        if first is None:
            if not isinstance(last, dict) or type(last.get("identity")) is not int or type(last.get("length")) is not int:
                errors.append(f"new Dynamo timing list {name} is malformed")
                continue
            delta = last["length"]
            (compile_growth if compile_only else runtime_growth).append(name)
        elif last is None:
            errors.append(f"Dynamo timing list {name} disappeared")
            delta = None
        elif not isinstance(first, dict) or not isinstance(last, dict):
            errors.append(f"Dynamo timing list {name} state is malformed")
            delta = None
        elif type(first.get("identity")) is not int or type(last.get("identity")) is not int:
            errors.append(f"Dynamo timing list {name} identity is missing")
            delta = None
        elif first["identity"] != last["identity"]:
            errors.append(f"Dynamo timing list {name} was replaced")
            delta = None
        elif type(first.get("length")) is not int or type(last.get("length")) is not int:
            errors.append(f"Dynamo timing list {name} length is missing")
            delta = None
        elif last["length"] < first["length"]:
            errors.append(f"Dynamo timing list {name} length decreased")
            delta = last["length"] - first["length"]
        else:
            delta = last["length"] - first["length"]
            if delta:
                (compile_growth if compile_only else runtime_growth).append(name)
        deltas[name] = {
            "before": first,
            "after": last,
            "delta": delta,
            "compile_only": compile_only,
        }
    return {
        "status": "changes_observed" if compile_growth else "observed_stable",
        "timing_list_deltas": deltas,
        "compile_only_growth": compile_growth,
        "runtime_only_growth": runtime_growth,
    }


def _compare_inductor(before: dict[str, Any], after: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    if before.get("counters_id") != after.get("counters_id"):
        errors.append("Dynamo counters mapping was replaced")
    old_map, new_map = before.get("inductor_mapping_id"), after.get("inductor_mapping_id")
    if old_map is not None and new_map is None:
        errors.append("Inductor counter mapping disappeared")
    elif old_map is not None and new_map is not None and old_map != new_map:
        errors.append("Inductor counter mapping was replaced")
    old, new = before.get("counters"), after.get("counters")
    deltas: dict[str, Any] = {}
    changes: list[str] = []
    if not isinstance(old, dict) or not isinstance(new, dict):
        errors.append("Inductor async-compile counter state is malformed")
        return {"status": "incomplete", "counter_deltas": deltas}
    for name in ("async_compile_cache_miss", "async_compile_cache_hit"):
        first, last = old.get(name), new.get(name)
        if not isinstance(first, dict) or not isinstance(last, dict):
            errors.append(f"Inductor counter {name} state is missing")
            continue
        delta = None
        if first.get("present") is True and last.get("present") is False:
            errors.append(f"Inductor counter {name} disappeared")
        elif first.get("present") is False and last.get("present") is True:
            if type(last.get("value")) is not int:
                errors.append(f"Inductor counter {name} value is malformed")
            else:
                delta = last["value"]
                changes.append(name)
        elif first.get("present") is True and last.get("present") is True:
            old_value, new_value = first.get("value"), last.get("value")
            if type(old_value) is not int or type(new_value) is not int:
                errors.append(f"Inductor counter {name} value is malformed")
            elif new_value < old_value:
                errors.append(f"Inductor counter {name} decreased")
                delta = new_value - old_value
            else:
                delta = new_value - old_value
                if delta:
                    changes.append(name)
        deltas[name] = {"before": first, "after": last, "delta": delta}
    return {
        "status": "changes_observed" if changes else "observed_stable",
        "counter_deltas": deltas,
        "compile_path_growth": changes,
    }


def compare_compilation_snapshots(before: Any, after: Any, source_metadata: Any) -> dict[str, Any]:
    """Compare exact boundary snapshots; this gate never claims complete JIT coverage."""
    errors = _source_errors(source_metadata)
    before_sources = _snapshot_sources(before, "before", errors)
    after_sources = _snapshot_sources(after, "after", errors)
    results: dict[str, Any] = {}
    changed_sources: list[str] = []
    for name, compare in (
        ("dynamo", _compare_dynamo),
        ("inductor", _compare_inductor),
    ):
        first, last = _pair(before_sources, after_sources, name, errors)
        if first is None or last is None:
            results[name] = {"status": "incomplete"}
            continue
        result = compare(first, last, errors)
        results[name] = result
        if result.get("status") == "changes_observed":
            changed_sources.append(name)

    first, last = _pair(before_sources, after_sources, "cute", errors)
    if first is None or last is None:
        results["cute"] = {"status": "incomplete"}
    else:
        if first.get("module_id") != last.get("module_id"):
            errors.append("active CuTe common module identity changed")
        if first.get("mapping_id") != last.get("mapping_id"):
            errors.append("CuTe COMPILED mapping was replaced")
        deltas, changes, unchanged = _identity_deltas(
            first.get("entries"), last.get("entries"), label="CuTe COMPILED cache", errors=errors
        )
        results["cute"] = {
            "status": "changes_observed" if changes else "observed_stable",
            "entry_deltas": deltas,
            "changed_keys": changes,
            "unchanged_entry_count": unchanged,
        }
        if changes:
            changed_sources.append("cute")

    first, last = _pair(before_sources, after_sources, "triton", errors)
    if first is None or last is None:
        results["triton"] = {"status": "incomplete"}
    else:
        if first.get("registry_id") != last.get("registry_id"):
            errors.append("Triton JIT registry was replaced")
        triton_changes: list[str] = []
        triton_deltas: dict[str, Any] = {}
        for label, field, replacement_is_error in (
            ("Triton JIT function", "function_ids", True),
            ("Triton device_caches mapping", "device_caches_ids", True),
            ("Triton kernel_cache mapping", "kernel_cache_ids", True),
            ("Triton kernel-cache specialization", "kernel_entry_ids", False),
        ):
            deltas, changes, unchanged = _identity_deltas(
                first.get(field),
                last.get(field),
                label=label,
                errors=errors,
                replacement_is_error=replacement_is_error,
            )
            triton_deltas[field] = deltas
            triton_deltas[field + "_unchanged_count"] = unchanged
            triton_changes.extend(changes)
        results["triton"] = {
            "status": "changes_observed" if triton_changes else "observed_stable",
            "identity_deltas": triton_deltas,
            "changed_keys": sorted(set(triton_changes)),
        }
        if triton_changes:
            changed_sources.append("triton")

    status = "incomplete" if errors else "changes_observed" if changed_sources else "observed_stable"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "observed_stable": status == "observed_stable",
        "changed_sources": sorted(set(changed_sources)),
        "sources": results,
        "source_coverage": {
            "status": "partial",
            "observed": list(OBSERVED_SOURCES),
            "unverified": [
                "asynchronous compiler worker completion",
                "DeepGEMM native JIT paths outside the tested grouped APIs",
            ],
            "limits": list(COVERAGE_LIMITS),
        },
        "errors": errors,
    }
