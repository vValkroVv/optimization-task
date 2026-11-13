"""Rank-local tensor reference export and streamed A/A comparison.

This module is imported only by the explicit native numerical calibration
mode. Payloads preserve each local tensor's dtype and logical element order;
they are not distributed checkpoints and never gather or pad shards.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "native-numerical-reference-v1"
COMPARISON_SCHEMA = "native-numerical-comparison-v1"
BLOCK_BYTES = 1024 * 1024
_IDENTITY_FIELDS = ("rank", "update", "phase", "name", "microbatch")
_METADATA_FIELDS = (
    "update",
    "phase",
    "name",
    "microbatch",
    "rank",
    "dtype",
    "global_shape",
    "local_shape",
    "local_stride",
    "local_numel",
    "source_device_type",
    "mesh",
    "placements",
    "owner",
    "aliases",
    "present",
)


def _same_json(left: Any, right: Any) -> bool:
    """Compare JSON values without treating bools as integers."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right)
        )
    return left == right


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _key(record: dict[str, Any]) -> dict[str, Any]:
    try:
        key = {field: record[field] for field in _IDENTITY_FIELDS}
    except KeyError as exc:
        raise ValueError(f"observation lacks canonical identity field {exc.args[0]}") from exc
    if type(key["rank"]) is not int or type(key["update"]) is not int:
        raise ValueError("observation rank and update must be integers")
    if key["microbatch"] is not None and type(key["microbatch"]) is not int:
        raise ValueError("observation microbatch must be an integer or None")
    if not isinstance(key["phase"], str) or not key["phase"]:
        raise ValueError("observation phase must be a nonempty string")
    if not isinstance(key["name"], str) or not key["name"]:
        raise ValueError("observation name must be a nonempty string")
    return key


def _key_id(key: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(key)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, chunks) -> str:
    """Durably publish a new file without replacing evidence at that path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        # A failed publication is retained as a partial artifact for diagnosis.
        raise
    return digest.hexdigest()


def _json_chunks(value: Any):
    encoded = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    yield encoded


def _observation_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record)
    metadata.setdefault("owner", None)
    metadata.setdefault("aliases", [])
    return metadata


def _metadata_mismatches(reference: dict[str, Any], current: dict[str, Any]) -> list[str]:
    fields = list(_IDENTITY_FIELDS) + ["present", "owner", "aliases"]
    optional = (
        "dtype",
        "shape",
        "global_shape",
        "local_shape",
        "local_stride",
        "local_numel",
        "source_device_type",
        "mesh",
        "placements",
    )
    fields.extend(field for field in optional if field in reference or field in current)
    return [
        field
        for field in fields
        if field not in reference
        or field not in current
        or not _same_json(reference[field], current[field])
    ]


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_observation_metadata(metadata: dict[str, Any], rank: int) -> None:
    if type(metadata.get("rank")) is not int or metadata["rank"] != rank:
        raise ValueError("reference observation has a wrong rank owner")
    if type(metadata.get("present")) is not bool:
        raise ValueError("reference observation lacks present metadata")
    if not isinstance(metadata.get("owner"), (dict, type(None))):
        raise ValueError("reference observation owner metadata is malformed")
    if not isinstance(metadata.get("aliases"), list):
        raise ValueError("reference observation aliases are malformed")
    if (
        metadata["present"] is not True
        or metadata.get("phase") == "monitor_record"
        or not str(metadata.get("dtype", "")).startswith("torch.")
    ):
        return
    for field in ("global_shape", "local_shape", "local_stride"):
        shape = metadata.get(field)
        if not isinstance(shape, list) or any(type(size) is not int for size in shape):
            raise ValueError(f"reference tensor {field} metadata is malformed")
    if any(size < 0 for size in metadata["local_shape"] + metadata["global_shape"]):
        raise ValueError("reference tensor shape dimensions must be nonnegative")
    if type(metadata.get("local_numel")) is not int or metadata["local_numel"] != math.prod(metadata["local_shape"]):
        raise ValueError("reference local element count differs from its shape")
    if not isinstance(metadata.get("source_device_type"), str):
        raise ValueError("reference tensor source device type is missing")
    if "mesh" not in metadata or not isinstance(metadata.get("placements"), list):
        raise ValueError("reference tensor mesh or placement metadata is missing")
    if not _valid_digest(metadata.get("sha256")):
        raise ValueError("reference tensor SHA-256 is invalid")
    dtype = str(metadata["dtype"])
    floating = dtype.startswith(("torch.float", "torch.bfloat", "torch.complex"))
    finite = metadata.get("is_finite")
    if floating and type(finite) is not bool:
        raise ValueError("reference floating tensor finite flag is missing")
    if not floating and finite is not None:
        raise ValueError("reference nonfloating tensor finite flag must be null")
    _dtype_size(dtype)


def _numeric_tree(reference: Any, current: Any, path: str = "$") -> dict[str, Any]:
    """Compare JSON structures, reporting floats numerically and other values exactly."""
    if type(reference) is not type(current):
        return {"path": path, "kind": "type_mismatch"}
    if isinstance(reference, dict):
        if reference.keys() != current.keys():
            return {"path": path, "kind": "key_mismatch"}
        children = [
            result
            for key in sorted(reference)
            if (result := _numeric_tree(reference[key], current[key], f"{path}.{key}"))
        ]
        return {"children": children} if children else {}
    if isinstance(reference, list):
        if len(reference) != len(current):
            return {"path": path, "kind": "length_mismatch"}
        children = [
            result
            for index, (left, right) in enumerate(zip(reference, current))
            if (result := _numeric_tree(left, right, f"{path}[{index}]"))
        ]
        return {"children": children} if children else {}
    if type(reference) is float:
        if not math.isfinite(reference) or not math.isfinite(current):
            return {
                "path": path,
                "kind": "nonfinite_float",
                "reference": repr(reference),
                "current": repr(current),
                "same": repr(reference) == repr(current),
            }
        difference = current - reference
        return {
            "path": path,
            "kind": "float",
            "reference": reference,
            "current": current,
            "absolute_error": abs(difference),
            "relative_error": (abs(difference) / abs(reference)) if reference else None,
            "zero_reference": reference == 0.0,
            "zero_reference_mismatch": reference == 0.0 and current != 0.0,
        }
    if reference != current:
        return {"path": path, "kind": "exact_mismatch", "reference": reference, "current": current}
    return {}


def _record_id(record: dict[str, Any]) -> tuple[dict[str, Any], str]:
    key = _key(record)
    return key, _key_id(key)


class NativeNumericalPayload:
    """Own one rank's export or consuming comparison namespace."""

    def __init__(
        self,
        *,
        mode: str,
        root: str | Path,
        rank: int,
        capture_update: int,
        run_id: str,
        comparison_root: str | Path | None = None,
    ) -> None:
        if mode not in {"export", "compare"}:
            raise ValueError("reference payload mode must be 'export' or 'compare'")
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        if capture_update not in {1, 2, 50}:
            raise ValueError("capture update must be 1, 2, or 50")
        self.mode = mode
        self.root = Path(root).resolve()
        self.rank = rank
        self.capture_update = capture_update
        self.run_id = run_id
        self.rank_dir = self.root / f"rank-{rank:02d}"
        self.items: dict[str, dict[str, Any]] = {}
        self._payload_paths: dict[str, str] = {}
        self._processed: set[str] = set()
        self._initial_gates_complete = False
        self._prefix_inputs_complete = False
        self._comparison_records: list[dict[str, Any]] = []
        self._reference_manifest_sha256: str | None = None
        self._reference_manifest: dict[str, Any] | None = None

        if mode == "export":
            if not self.root.is_dir():
                raise FileNotFoundError(f"reference output root must exist: {self.root}")
            self.rank_dir.mkdir(mode=0o755, exist_ok=False)
            (self.rank_dir / "tensors").mkdir()
            self.manifest_path = self.rank_dir / "manifest.json"
            self.failure_path = self.rank_dir / "failure.json"
            self.comparison_dir = None
        else:
            self.manifest_path = self.rank_dir / "manifest.json"
            if comparison_root is None:
                raise ValueError("comparison mode requires an owned comparison output directory")
            self.comparison_dir = Path(comparison_root).resolve() / f"rank-{rank:02d}"
            self.comparison_dir.mkdir(parents=True, exist_ok=False)
            (self.comparison_dir / "observations").mkdir()
            self.failure_path = self.comparison_dir / "failure.json"
            try:
                self._load_reference_manifest()
            except BaseException as exc:
                self.record_failure(f"{type(exc).__name__}: {exc}")
                raise

    def record_failure(self, message: str) -> None:
        """Leave a compact failure receipt without completing or replacing a namespace."""
        directory = self.comparison_dir if self.mode == "compare" else self.rank_dir
        if directory is None or self.failure_path.exists():
            return
        value = {
            "schema": COMPARISON_SCHEMA if self.mode == "compare" else SCHEMA,
            "complete": False,
            "rank": self.rank,
            "capture_update": self.capture_update,
            "error": message,
        }
        _write_new_file(self.failure_path, _json_chunks(value))

    def observe_tensor(self, record: dict[str, Any], cpu_tensor: Any) -> None:
        """Export or compare only selected-update numerical tensors."""
        if record.get("update") != self.capture_update or record.get("phase") == "dataloader_input":
            return
        if self.mode == "export":
            self.export_tensor(record, cpu_tensor)
        else:
            self.compare_tensor(record, cpu_tensor)

    def _load_reference_manifest(self) -> None:
        try:
            raw = self.manifest_path.read_bytes()
            manifest = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read complete reference manifest {self.manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise ValueError("reference manifest schema is missing or unsupported")
        if manifest.get("complete") is not True:
            raise ValueError("reference manifest is incomplete")
        if type(manifest.get("rank")) is not int or manifest.get("rank") != self.rank:
            raise ValueError("reference manifest belongs to a different rank")
        if type(manifest.get("capture_update")) is not int or manifest.get("capture_update") != self.capture_update:
            raise ValueError("reference manifest capture update differs")
        expected_prefix = list(range(1, self.capture_update + 1))
        if manifest.get("prefix_updates") != expected_prefix:
            raise ValueError("reference manifest prefix updates differ")
        if (
            type(manifest.get("expected_microbatches")) is not int
            or manifest["expected_microbatches"] != 32
            or type(manifest.get("block_bytes")) is not int
            or manifest["block_bytes"] != BLOCK_BYTES
            or manifest.get("payload_order")
            != "logical contiguous elements in original dtype; local shard only"
            or not isinstance(manifest.get("run_id"), str)
            or not isinstance(manifest.get("coverage"), dict)
            or not isinstance(manifest.get("training_metadata"), dict)
            or manifest.get("monitor_attestation") is not None
            and not isinstance(manifest.get("monitor_attestation"), dict)
        ):
            raise ValueError("reference manifest is missing required calibration metadata")
        expected_coverage = {
            str(self.capture_update): {
                "input_batches": 32,
                "microbatches": 32,
                "summary_calls": 1,
                "clip_calls": 1,
                "optimizer_calls": 1,
                "batch_end_calls": 1,
            }
        }
        if manifest["coverage"] != expected_coverage:
            raise ValueError("reference manifest selected-update coverage differs")
        entries = manifest.get("observations")
        if not isinstance(entries, list):
            raise ValueError("reference manifest observations must be a list")
        self.items = {}
        payload_paths: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("metadata"), dict):
                raise ValueError("reference manifest contains a malformed observation")
            metadata = entry["metadata"]
            _validate_observation_metadata(metadata, self.rank)
            key = _key(metadata)
            key_id = _key_id(key)
            if key_id in self.items:
                raise ValueError(f"reference manifest contains duplicate identity {key}")
            if not _same_json(entry.get("key"), key):
                raise ValueError("reference observation key differs from its metadata")
            payload_file = entry.get("payload_file")
            if payload_file is not None:
                if not isinstance(payload_file, str):
                    raise ValueError("reference payload path must be a string")
                relative = PurePosixPath(payload_file)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("reference payload path escapes its rank namespace")
                path = (self.rank_dir / Path(*relative.parts)).resolve()
                if self.rank_dir not in path.parents:
                    raise ValueError("reference payload resolves outside its rank namespace")
                if metadata.get("update") != self.capture_update or metadata.get("phase") == "dataloader_input":
                    raise ValueError("payload is attached to a nonselected or prefix-input observation")
                if metadata.get("present") is not True or not str(metadata.get("dtype", "")).startswith("torch."):
                    raise ValueError("payload is attached to a non-present tensor")
                if not path.is_file():
                    raise FileNotFoundError(f"reference payload is missing: {path}")
                expected_payload = f"tensors/{key_id}.bin"
                if payload_file != expected_payload or payload_file in payload_paths:
                    raise ValueError("reference payload path is duplicate or not canonical for its key")
                payload_paths.add(payload_file)
                nbytes = entry.get("nbytes")
                expected_bytes = metadata["local_numel"] * _dtype_size(metadata["dtype"])
                if type(nbytes) is not int or nbytes != expected_bytes or path.stat().st_size != nbytes:
                    raise ValueError(f"reference payload size differs for {key}")
                self._payload_paths[key_id] = payload_file
            elif (
                metadata.get("update") == self.capture_update
                and metadata.get("phase") not in {"dataloader_input", "monitor_record"}
                and metadata.get("present") is True
                and str(metadata.get("dtype", "")).startswith("torch.")
            ):
                raise ValueError(f"selected tensor is missing payload for {key}")
            self.items[key_id] = entry
        expected_files = {"manifest.json", *payload_paths}
        actual_files = {
            path.relative_to(self.rank_dir).as_posix()
            for path in self.rank_dir.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ValueError("reference rank namespace contains partial or unexpected files")
        self._reference_manifest = manifest
        self._reference_manifest_sha256 = hashlib.sha256(raw).hexdigest()

    def _failure(self, message: str, *, key: dict[str, Any] | None = None, details: Any = None) -> None:
        if self.failure_path is not None:
            value = {
                "schema": COMPARISON_SCHEMA if self.mode == "compare" else SCHEMA,
                "complete": False,
                "rank": self.rank,
                "capture_update": self.capture_update,
                "error": message,
            }
            if key is not None:
                value["key"] = key
            if details is not None:
                value["details"] = details
            if not self.failure_path.exists():
                _write_new_file(self.failure_path, _json_chunks(value))
        raise ValueError(message)

    def export_tensor(self, record: dict[str, Any], cpu_tensor: Any) -> None:
        if self.mode != "export":
            raise RuntimeError("export_tensor is only valid in export mode")
        key, key_id = _record_id(record)
        if key_id in self._payload_paths:
            self._failure("duplicate payload identity", key=key)
        if record.get("update") != self.capture_update or record.get("phase") == "dataloader_input":
            self._failure("only selected-update numerical tensors may be exported", key=key)
        torch = _torch()
        if cpu_tensor.device.type != "cpu" or not cpu_tensor.is_contiguous():
            self._failure("export tensor must be a contiguous CPU tensor", key=key)
        raw = memoryview(cpu_tensor.reshape(-1).view(torch.uint8).numpy())
        digest = hashlib.sha256()
        path = self.rank_dir / "tensors" / f"{key_id}.bin"

        def chunks():
            for start in range(0, len(raw), BLOCK_BYTES):
                block = raw[start : start + BLOCK_BYTES]
                digest.update(block)
                yield block

        try:
            _write_new_file(path, chunks())
        except BaseException:
            raise
        if digest.hexdigest() != record.get("sha256"):
            self._failure("export payload hash differs from observed tensor hash", key=key)
        self._payload_paths[key_id] = path.relative_to(self.rank_dir).as_posix()

    def compare_tensor(self, record: dict[str, Any], cpu_tensor: Any) -> None:
        if self.mode != "compare":
            raise RuntimeError("compare_tensor is only valid in compare mode")
        if not self._initial_gates_complete or not self._prefix_inputs_complete:
            self._failure("numerical comparison began before initial and prefix-input gates", key=_key(record))
        key, key_id = _record_id(record)
        entry = self.items.get(key_id)
        if entry is None:
            self._failure("current observation is absent from the reference manifest", key=key)
        reference = entry["metadata"]
        mismatches = _metadata_mismatches(reference, _observation_metadata(record))
        if mismatches:
            self._failure("tensor metadata differs from reference", key=key, details={"fields": mismatches})
        payload_file = entry.get("payload_file")
        if payload_file is None:
            self._failure("current selected tensor has no reference payload", key=key)
        if key_id in self._processed:
            self._failure("duplicate current tensor comparison", key=key)
        reference_path = (self.rank_dir / Path(*PurePosixPath(payload_file).parts)).resolve()
        try:
            metrics = _compare_tensor_file(
                cpu_tensor,
                reference_path,
                reference,
                expected_current_sha256=record.get("sha256"),
                expected_current_finite=record.get("is_finite"),
            )
        except BaseException as exc:
            self._failure("reference tensor comparison failed", key=key, details={"reason": f"{type(exc).__name__}: {exc}"})
        metrics_path = self.comparison_dir / "observations" / f"{key_id}.json"
        result = {
            "schema": COMPARISON_SCHEMA,
            "key": key,
            "metadata_equal": True,
            "comparison_eligible": True,
            "metrics": metrics,
        }
        try:
            result_digest = _write_new_file(metrics_path, _json_chunks(result))
        except BaseException as exc:
            self._failure("durable comparison evidence could not be written; reference payload retained", key=key, details={"reason": f"{type(exc).__name__}: {exc}"})
        reference_path.unlink()
        _fsync_directory(reference_path.parent)
        self._processed.add(key_id)
        self._comparison_records.append(
            {
                "key": key,
                "result": metrics_path.relative_to(self.comparison_dir).as_posix(),
                "result_sha256": result_digest,
            }
        )

    def validate_tensor_metadata(self, record: dict[str, Any]) -> None:
        """Reject a wrong owner, shard, or dtype before copying tensor values."""
        if self.mode != "compare":
            return
        key, key_id = _record_id(record)
        entry = self.items.get(key_id)
        if entry is None:
            self._failure("current tensor identity is absent from the reference manifest", key=key)
        mismatches = _metadata_mismatches(
            entry["metadata"], _observation_metadata(record)
        )
        if mismatches:
            self._failure(
                "tensor metadata differs from reference before value transfer",
                key=key,
                details={"fields": mismatches},
            )

    def verify_initial_gates(
        self, observations: list[dict[str, Any]], training_metadata: dict[str, Any]
    ) -> None:
        if self.mode == "export":
            self._initial_gates_complete = True
            return
        current: dict[str, dict[str, Any]] = {}
        for record in observations:
            if record.get("update") == 0:
                key, key_id = _record_id(record)
                if key_id in current:
                    self._failure("duplicate initial-state identity", key=key)
                current[key_id] = record
        expected = {
            key_id: entry
            for key_id, entry in self.items.items()
            if entry["metadata"].get("update") == 0
        }
        if set(current) != set(expected):
            self._failure(
                "initial-state observation inventory differs",
                details={"missing": len(set(expected) - set(current)), "extra": len(set(current) - set(expected))},
            )
        mismatches = []
        for key_id, record in current.items():
            reference = expected[key_id]["metadata"]
            if not _same_json(reference, _observation_metadata(record)):
                mismatches.append(_key(record))
        reference_metadata = self._reference_manifest.get("training_metadata", {})
        if not _same_json(reference_metadata, training_metadata):
            self._failure("initial configuration, ownership, or training metadata differs")
        if mismatches:
            self._failure("initial parameter, optimizer, scheduler, buffer, RNG, or counter gate differs", details={"identities": mismatches[:20], "count": len(mismatches)})
        self._initial_gates_complete = True
        self._processed.update(current)
        if self.mode == "compare":
            inventory = [
                (key, _observation_metadata(record))
                for key, record in current.items()
            ]
            inventory.sort(key=lambda item: _canonical(item[0]))
            report = {
                "schema": COMPARISON_SCHEMA,
                "complete": True,
                "gate": "initial_state",
                "rank": self.rank,
                "observation_count": len(current),
                "identity_sha256": hashlib.sha256(_canonical(inventory)).hexdigest(),
                "training_metadata_equal": True,
            }
            _write_new_file(self.comparison_dir / "initial-gates.json", _json_chunks(report))

    def verify_prefix_inputs(self, observations: list[dict[str, Any]]) -> None:
        current: dict[str, dict[str, Any]] = {}
        for record in observations:
            if record.get("phase") == "dataloader_input" and record.get("update", 0) <= self.capture_update:
                key, key_id = _record_id(record)
                if key_id in current:
                    self._failure("duplicate prefix-input identity", key=key)
                current[key_id] = record
        expected = {
            key_id: entry
            for key_id, entry in self.items.items()
            if entry["metadata"].get("phase") == "dataloader_input"
            and entry["metadata"].get("update", 0) <= self.capture_update
        }
        if self.mode == "export":
            for update in range(1, self.capture_update + 1):
                update_records = [
                    record
                    for record in current.values()
                    if record.get("update") == update
                ]
                labels = [record for record in update_records if record.get("name") == "labels"]
                inputs = [record for record in update_records if str(record.get("name", "")).startswith("input.")]
                if (
                    len(labels) != 32
                    or {record.get("microbatch") for record in labels} != set(range(32))
                    or len(inputs) < 32
                    or {record.get("microbatch") for record in inputs} != set(range(32))
                ):
                    self._failure("export lacks all-prefix CPU input hashes", details={"update": update})
            self._prefix_inputs_complete = True
            return
        if set(current) != set(expected):
            self._failure(
                "all-prefix CPU input inventory differs",
                details={"missing": len(set(expected) - set(current)), "extra": len(set(current) - set(expected))},
            )
        rows = []
        for key_id, record in current.items():
            reference = expected[key_id]["metadata"]
            mismatches = _metadata_mismatches(reference, _observation_metadata(record))
            if mismatches or reference.get("sha256") != record.get("sha256"):
                self._failure(
                    "all-prefix CPU input hash or metadata differs",
                    key=_key(record),
                    details={"fields": mismatches, "reference_sha256": reference.get("sha256"), "current_sha256": record.get("sha256")},
                )
            rows.append({"key": _key(record), "sha256": record["sha256"], "equal": True})
        report = {
            "schema": COMPARISON_SCHEMA,
            "complete": True,
            "gate": "all_prefix_cpu_inputs",
            "rank": self.rank,
            "updates": list(range(1, self.capture_update + 1)),
            "observation_count": len(rows),
            "observations": sorted(rows, key=lambda row: _canonical(row["key"])),
        }
        if self.mode == "compare":
            _write_new_file(self.comparison_dir / "prefix-inputs.json", _json_chunks(report))
        self._processed.update(current)
        self._prefix_inputs_complete = True

    def finish(
        self,
        observations: list[dict[str, Any]],
        *,
        coverage: dict[str, Any],
        training_metadata: dict[str, Any],
        monitor_attestation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        current: dict[str, dict[str, Any]] = {}
        for record in observations:
            key, key_id = _record_id(record)
            if key_id in current:
                self._failure("duplicate observation identity", key=key)
            current[key_id] = _observation_metadata(record)

        if self.mode == "export":
            if not self._initial_gates_complete or not self._prefix_inputs_complete:
                self._failure("export did not complete its initial and prefix-input gates")
            entries = []
            for key_id, metadata in current.items():
                entry: dict[str, Any] = {"key": _key(metadata), "metadata": metadata}
                payload_file = self._payload_paths.get(key_id)
                if payload_file is not None:
                    entry["payload_file"] = payload_file
                    entry["nbytes"] = metadata["local_numel"] * _dtype_size(metadata["dtype"])
                elif (
                    metadata.get("update") == self.capture_update
                    and metadata.get("phase") not in {"dataloader_input", "monitor_record"}
                    and metadata.get("present") is True
                    and str(metadata.get("dtype", "")).startswith("torch.")
                ):
                    self._failure("selected numerical tensor was not exported", key=_key(metadata))
                entries.append(entry)
            manifest = {
                "schema": SCHEMA,
                "complete": True,
                "run_id": self.run_id,
                "rank": self.rank,
                "capture_update": self.capture_update,
                "prefix_updates": list(range(1, self.capture_update + 1)),
                "expected_microbatches": 32,
                "block_bytes": BLOCK_BYTES,
                "payload_order": "logical contiguous elements in original dtype; local shard only",
                "payload_semantics": "post_optimizer_parameter is the parameter value after the optimizer step, not a parameter delta",
                "ema_state": "EMA shadow payloads and continuation state are not captured; no EMA value is used in updates 1..50 of this crop",
                "coverage": coverage,
                "training_metadata": training_metadata,
                "monitor_attestation": monitor_attestation,
                "observations": sorted(entries, key=lambda item: _canonical(item["key"])),
            }
            _write_new_file(self.manifest_path, _json_chunks(manifest))
            return {
                "mode": "export",
                "manifest": str(self.manifest_path),
                "capture_update": self.capture_update,
                "payload_count": len(self._payload_paths),
                "payload_bytes": sum(
                    entry["nbytes"] for entry in entries if "nbytes" in entry
                ),
            }

        if not self._initial_gates_complete or not self._prefix_inputs_complete:
            self._failure("comparison did not complete its initial and prefix-input gates")
        normalized_coverage = {str(key): value for key, value in coverage.items()}
        if not _same_json(self._reference_manifest.get("coverage"), normalized_coverage):
            self._failure("native selected-update call counts differ from reference")
        if not _same_json(self._reference_manifest.get("monitor_attestation"), monitor_attestation):
            self._failure("monitor50 record or publication inventory differs from reference")
        expected_ids = set(self.items)
        if set(current) != expected_ids:
            self._failure(
                "current observation inventory differs from complete reference",
                details={"missing": len(expected_ids - set(current)), "extra": len(set(current) - expected_ids)},
            )
        scalar_results = []
        for key_id, record in current.items():
            if key_id in self._processed:
                continue
            entry = self.items[key_id]
            reference = entry["metadata"]
            key = _key(record)
            mismatches = _metadata_mismatches(reference, record)
            if mismatches:
                self._failure("observation metadata differs from reference", key=key, details={"fields": mismatches})
            if entry.get("payload_file") is not None:
                self._failure("reference payload was not compared", key=key)
            if record.get("phase") == "dataloader_input":
                self._failure("prefix input escaped its required precomparison gate", key=key)
            exact_fields = (
                ("record_name", "index", "step", "reduce_mode", "reduce", "pending_stages", "post_stats", "post_axes", "dtype", "shape", "scalar", "multiplicity")
                if record.get("phase") == "monitor_record"
                else ("publish_call_index", "publish_metric_index", "publish_step", "metric_name", "resource_observation")
                if record.get("phase") == "monitor_publish"
                else ()
            )
            structural_mismatches = [
                field
                for field in exact_fields
                if field not in reference
                or field not in record
                or not _same_json(reference[field], record[field])
            ]
            if structural_mismatches:
                self._failure(
                    "monitor record structure differs from reference",
                    key=key,
                    details={"fields": structural_mismatches},
                )
            is_resource = record.get("resource_observation") is True
            if is_resource:
                comparison = {
                    "comparison_eligible": False,
                    "reference_value": reference.get("value"),
                    "current_value": record.get("value"),
                    "reason": "timing, throughput, remaining-time, or memory observation",
                }
            else:
                comparison = {
                    "comparison_eligible": True,
                    "numeric_comparison": _numeric_tree(reference, record),
                }
            result = {
                "schema": COMPARISON_SCHEMA,
                "key": key,
                "metadata_equal": True,
                **comparison,
            }
            path = self.comparison_dir / "observations" / f"{key_id}.json"
            result_digest = _write_new_file(path, _json_chunks(result))
            self._processed.add(key_id)
            scalar_results.append(
                {
                    "key": key,
                    "result": path.relative_to(self.comparison_dir).as_posix(),
                    "result_sha256": result_digest,
                }
            )

        payload_ids = set(self._payload_paths)
        if payload_ids - self._processed:
            self._failure(
                "some reference payloads were not consumed",
                details={"remaining_payloads": len(payload_ids - self._processed)},
            )
        result_manifest = {
            "schema": COMPARISON_SCHEMA,
            "complete": True,
            "run_id": self.run_id,
            "rank": self.rank,
            "capture_update": self.capture_update,
            "reference_manifest_sha256": self._reference_manifest_sha256,
            "reference_run_id": self._reference_manifest.get("run_id"),
            "observation_count": len(current),
            "payload_comparison_count": len(payload_ids),
            "scalar_comparison_count": len(scalar_results),
            "coverage": normalized_coverage,
            "training_metadata_equal": True,
            "initial_gate": "exact_match",
            "prefix_input_gate": "all_hashes_equal",
            "monitor_attestation": monitor_attestation,
            "payload_deletion": "each reference tensor was unlinked only after its atomic per-observation metrics file was fsynced and published",
            "observations": sorted(
                self._comparison_records + scalar_results,
                key=lambda item: _canonical(item["key"]),
            ),
        }
        result_path = self.comparison_dir / "comparison.json"
        _write_new_file(result_path, _json_chunks(result_manifest))
        return {
            "mode": "compare",
            "manifest": str(result_path),
            "capture_update": self.capture_update,
            "reference_manifest_sha256": self._reference_manifest_sha256,
            "comparison_count": len(current),
        }


def _dtype_size(dtype: str) -> int:
    import torch

    name = dtype.removeprefix("torch.")
    value = getattr(torch, name, None)
    if value is None:
        raise ValueError(f"unsupported reference tensor dtype {dtype}")
    return torch.empty((), dtype=value).element_size()


def _torch():
    import torch

    return torch


def _compare_tensor_file(
    cpu_tensor: Any,
    reference_path: Path,
    metadata: dict[str, Any],
    *,
    expected_current_sha256: str | None,
    expected_current_finite: bool | None,
) -> dict[str, Any]:
    """Compare a CPU tensor to raw reference bytes using bounded read blocks."""
    torch = _torch()
    if cpu_tensor.device.type != "cpu" or not cpu_tensor.is_contiguous():
        raise ValueError("current tensor must be a contiguous CPU tensor")
    if cpu_tensor.dtype.__str__() != metadata.get("dtype"):
        raise ValueError("current tensor dtype differs from reference metadata")
    element_size = cpu_tensor.element_size()
    count = cpu_tensor.numel()
    expected_bytes = count * element_size
    if reference_path.stat().st_size != expected_bytes:
        raise ValueError("reference payload byte count differs from tensor shape and dtype")

    flat = cpu_tensor.reshape(-1)
    exact_elements = 0
    nonfinite_reference = 0
    nonfinite_current = 0
    nonfinite_mismatch = 0
    finite_pairs = 0
    zero_reference_mismatch = 0
    zero_reference_max = 0.0
    error_square_sum = 0.0
    reference_square_sum = 0.0
    current_square_sum = 0.0
    max_abs_error: float | None = None
    max_abs_index: int | None = None
    reference_hash = hashlib.sha256()
    current_hash = hashlib.sha256()
    block_elements = max(1, BLOCK_BYTES // element_size)
    complex_values = cpu_tensor.is_complex()

    with reference_path.open("rb") as handle:
        start = 0
        while start < count:
            end = min(count, start + block_elements)
            block_count = end - start
            reference_bytes = bytearray(handle.read(block_count * element_size))
            if len(reference_bytes) != block_count * element_size:
                raise ValueError("reference payload is truncated")
            reference_hash.update(reference_bytes)
            current_block = flat.narrow(0, start, block_count)
            current_bytes = memoryview(current_block.view(torch.uint8).numpy())
            current_hash.update(current_bytes)
            reference_byte_tensor = torch.frombuffer(
                reference_bytes, dtype=torch.uint8
            ).reshape(block_count, element_size)
            current_byte_tensor = current_block.view(torch.uint8).reshape(
                block_count, element_size
            )
            exact_elements += int(
                current_byte_tensor.eq(reference_byte_tensor).all(dim=1).sum().item()
            )

            if cpu_tensor.is_floating_point() or complex_values:
                reference_block = torch.frombuffer(
                    reference_bytes, dtype=cpu_tensor.dtype, count=block_count
                )
                reference_work_dtype = torch.complex128 if complex_values else torch.float64
                reference_work = reference_block.to(reference_work_dtype)
                current_work = current_block.to(reference_work_dtype)
                reference_finite = torch.isfinite(reference_work)
                current_finite = torch.isfinite(current_work)
                nonfinite_reference += int((~reference_finite).sum().item())
                nonfinite_current += int((~current_finite).sum().item())
                nonfinite_mismatch += int((reference_finite != current_finite).sum().item())
                finite = reference_finite & current_finite
                finite_count = int(finite.sum().item())
                finite_pairs += finite_count
                if finite_count:
                    difference = (current_work - reference_work).abs()
                    valid_difference = difference[finite]
                    block_max, max_in_block = valid_difference.max(dim=0)
                    block_max_value = float(block_max.item())
                    if max_abs_error is None or block_max_value > max_abs_error:
                        max_abs_error = block_max_value
                        valid_indices = torch.nonzero(finite, as_tuple=False).reshape(-1)
                        max_abs_index = start + int(valid_indices[int(max_in_block.item())].item())
                    error_square_sum += float(valid_difference.square().sum(dtype=torch.float64).item())
                    reference_magnitude = reference_work.abs()[finite]
                    current_magnitude = current_work.abs()[finite]
                    reference_square_sum += float(reference_magnitude.square().sum(dtype=torch.float64).item())
                    current_square_sum += float(current_magnitude.square().sum(dtype=torch.float64).item())
                    zero_reference = reference_finite & (reference_work == 0)
                    zero_mismatch = zero_reference & (current_work != 0)
                    zero_reference_mismatch += int(zero_mismatch.sum().item())
                    finite_zero_mismatch = zero_mismatch & current_finite
                    if bool(finite_zero_mismatch.any().item()):
                        zero_reference_max = max(
                            zero_reference_max,
                            float(difference[finite_zero_mismatch].max().item()),
                        )
            start = end
        if handle.read(1):
            raise ValueError("reference payload has trailing bytes")

    reference_digest = reference_hash.hexdigest()
    current_digest = current_hash.hexdigest()
    if reference_digest != metadata.get("sha256"):
        raise ValueError("reference payload hash does not match its complete manifest")
    if current_digest != expected_current_sha256:
        raise ValueError("current CPU tensor hash differs from its captured observation")
    floating = cpu_tensor.is_floating_point() or complex_values
    if floating and bool(metadata.get("is_finite")) != (nonfinite_reference == 0):
        raise ValueError("reference finite metadata conflicts with payload values")
    if floating and bool(expected_current_finite) != (nonfinite_current == 0):
        raise ValueError("current finite metadata conflicts with CPU tensor values")

    rms_error = math.sqrt(error_square_sum / finite_pairs) if finite_pairs else None
    reference_rms = math.sqrt(reference_square_sum / finite_pairs) if finite_pairs else None
    current_rms = math.sqrt(current_square_sum / finite_pairs) if finite_pairs else None
    if count == 0:
        relative_l2 = None
        relative_l2_status = "empty_tensor"
    elif not finite_pairs:
        relative_l2 = None
        relative_l2_status = "no_finite_pairs"
    elif reference_square_sum == 0.0:
        relative_l2 = 0.0 if error_square_sum == 0.0 else None
        relative_l2_status = "zero_reference_zero_error" if error_square_sum == 0.0 else "zero_reference_nonzero_error"
    else:
        relative_l2 = math.sqrt(error_square_sum / reference_square_sum)
        relative_l2_status = "defined"
    return {
        "count": count,
        "exact_element_match_count": exact_elements,
        "exact_element_match": exact_elements == count,
        "reference_sha256": reference_digest,
        "current_sha256": current_digest,
        "exact_hash_match": reference_digest == current_digest,
        "nonfinite_reference_count": nonfinite_reference,
        "nonfinite_current_count": nonfinite_current,
        "nonfinite_mismatch_count": nonfinite_mismatch,
        "finite_pair_count": finite_pairs if floating else None,
        "max_abs_error": max_abs_error if floating else None,
        "max_abs_error_logical_flat_index": max_abs_index if floating else None,
        "rms_error": rms_error if floating else None,
        "reference_rms": reference_rms if floating else None,
        "current_rms": current_rms if floating else None,
        "relative_l2": relative_l2 if floating else None,
        "relative_l2_status": relative_l2_status if floating else "not_floating_point",
        "zero_reference_mismatch_count": zero_reference_mismatch if floating else None,
        "zero_reference_max_abs_error": zero_reference_max if floating else None,
        "floating_metrics_use_fp64_diagnostics": True,
    }
