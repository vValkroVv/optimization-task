#!/usr/bin/env python3
"""Capture one CPU-only receipt for loaded CudaCherry quantize/GEMM binaries.

The caller supplies the exact rank-to-PID map. This script reads only those
/proc entries and mapped family files; it never imports Torch or uses CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

RANKS = tuple(range(8))
FAMILIES = ("quantize", "gemm")
RANK_PID = re.compile(r"([0-9]+):([0-9]+)\Z")
DEVICE = re.compile(r"([0-9a-fA-F]+):([0-9a-fA-F]+)\Z")
DELETED = " (deleted)"
CHUNK_BYTES = 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_bindings(specs: list[str]) -> tuple[dict[int, int], list[str]]:
    pairs: list[tuple[int, int]] = []
    errors: list[str] = []
    for raw in specs:
        match = RANK_PID.fullmatch(raw)
        if match is None:
            errors.append(f"invalid_rank_pid:{raw!r}")
            continue
        rank, pid = map(int, match.groups())
        if rank not in RANKS:
            errors.append(f"rank_out_of_range:{rank}")
        elif pid <= 0:
            errors.append(f"pid_not_positive:{rank}:{pid}")
        else:
            pairs.append((rank, pid))

    rank_counts = {rank: sum(r == rank for r, _ in pairs) for rank in RANKS}
    for rank, count in rank_counts.items():
        if count == 0:
            errors.append(f"missing_rank:{rank}")
        elif count > 1:
            errors.append(f"duplicate_rank:{rank}")
    pid_ranks: dict[int, list[int]] = {}
    for rank, pid in pairs:
        pid_ranks.setdefault(pid, []).append(rank)
    for pid, ranks in pid_ranks.items():
        if len(ranks) > 1:
            errors.append(f"duplicate_pid:{pid}:ranks={','.join(map(str, ranks))}")
    bindings = {
        rank: pid
        for rank, pid in pairs
        if rank_counts[rank] == 1 and len(pid_ranks[pid]) == 1
    }
    return bindings, errors


def parse_maps_line(line: str) -> dict | None:
    """Split at most five times so spaces and the deleted suffix stay in path."""
    fields = line.rstrip("\n").split(maxsplit=5)
    if len(fields) < 5 or DEVICE.fullmatch(fields[3]) is None or not fields[4].isdecimal():
        return None
    try:
        start, end = fields[0].split("-", 1)
        int(start, 16)
        int(end, 16)
        int(fields[2], 16)
        inode = int(fields[4], 10)
    except ValueError:
        return None
    path = fields[5] if len(fields) == 6 else None
    return {
        "address": fields[0],
        "permissions": fields[1],
        "offset": fields[2],
        "device": fields[3].lower(),
        "inode": inode,
        "path": path,
        "deleted": bool(path and path.endswith(DELETED)),
    }


def families_for(path: str | None) -> tuple[str, ...]:
    if not path:
        return ()
    name = path.removesuffix(DELETED).lower()
    if ".so" not in name or "cudacherry" not in name:
        return ()
    families = []
    if re.search(r"(?:^|[^a-z0-9])gemm(?:[^a-z0-9]|$)", name):
        families.append("gemm")
    if "quantiz" in name:
        families.append("quantize")
    return tuple(families)


def _dev_parts(device: str) -> tuple[int, int] | None:
    match = DEVICE.fullmatch(device)
    return tuple(int(part, 16) for part in match.groups()) if match else None


def _stat_dev(st_dev: int) -> tuple[int, int]:
    return os.major(st_dev), os.minor(st_dev)


def _snapshot(info: os.stat_result) -> dict:
    return {
        "device": f"{os.major(info.st_dev):x}:{os.minor(info.st_dev):x}",
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _map_identity_matches(info: os.stat_result, device: str, inode: int) -> bool:
    expected = _dev_parts(device)
    return expected is not None and _stat_dev(info.st_dev) == expected and info.st_ino == inode


class ProcAccess:
    """Read only caller-selected process entries; never enumerate /proc."""

    def __init__(self, root: Path = Path("/proc")):
        self.root = root

    def _read(self, pid: int, name: str) -> bytes:
        return (self.root / str(pid) / name).read_bytes()

    def cmdline(self, pid: int) -> bytes:
        return self._read(pid, "cmdline")

    def maps(self, pid: int) -> bytes:
        return self._read(pid, "maps")

    def start_ticks(self, pid: int) -> int:
        raw = self._read(pid, "stat").decode("ascii", errors="replace")
        close = raw.rfind(")")
        if close < 0:
            raise ValueError("malformed_proc_stat")
        fields = raw[close + 2 :].split()  # starts at proc stat field 3
        if len(fields) <= 19:
            raise ValueError("short_proc_stat")
        return int(fields[19], 10)  # proc stat field 22


class FileAccess:
    def stat(self, path: str) -> os.stat_result:
        return os.stat(path)

    def open(self, path: str):
        return open(path, "rb")

    def fstat(self, handle) -> os.stat_result:
        return os.fstat(handle.fileno())


def _scope(cmdline: bytes, attempt_dir: str) -> dict:
    target = os.fsencode(attempt_dir)
    matched = [i for i, arg in enumerate(cmdline.split(b"\0")) if target in arg]
    return {
        "source": "/proc/<pid>/cmdline",
        "delimiter": "NUL",
        "full_command_line_saved": False,
        "exact_attempt_path_found": bool(matched),
        "matching_argument_indices": matched,
    }


def _binding_check(proc, pid: int, attempt_dir: str, expected_ticks: int | None = None) -> dict:
    issues: list[str] = []
    result = {
        "ok": False,
        "start_ticks_before": None,
        "start_ticks_after": None,
        "scope": None,
        "issues": issues,
    }
    try:
        result["start_ticks_before"] = proc.start_ticks(pid)
        result["scope"] = _scope(proc.cmdline(pid), attempt_dir)
        result["start_ticks_after"] = proc.start_ticks(pid)
    except FileNotFoundError:
        issues.append("changed_pid" if result["start_ticks_before"] is not None else "missing_pid")
        return result
    except OSError:
        issues.append("proc_read_error")
        return result
    except (ValueError, IndexError):
        issues.append("proc_stat_error")
        return result
    if result["start_ticks_before"] != result["start_ticks_after"]:
        issues.append("changed_pid")
    if expected_ticks is not None and result["start_ticks_before"] != expected_ticks:
        issues.append("changed_pid")
    if not result["scope"]["exact_attempt_path_found"]:
        issues.append("wrong_attempt_scope")
    result["ok"] = not issues
    return result


def _issue(record: dict, code: str) -> None:
    if code not in record["issues"]:
        record["issues"].append(code)


def _record_check(record: dict, stage: str, check: dict) -> None:
    record["binding_checks"].append(
        {
            "stage": stage,
            "start_ticks_before": check["start_ticks_before"],
            "start_ticks_after": check["start_ticks_after"],
            "scope": check["scope"],
            "issues": check["issues"],
        }
    )
    if record["cmdline_scope"] is None or stage == "preflight":
        record["cmdline_scope"] = check["scope"]
    if record["start_time_ticks"] is None:
        record["start_time_ticks"] = check["start_ticks_before"]
    for code in check["issues"]:
        _issue(record, code)


def _matching_rows(record: dict, maps_data: bytes) -> None:
    rows = []
    for raw in maps_data.decode("utf-8", errors="surrogateescape").splitlines():
        parsed = parse_maps_line(raw)
        if parsed is None:
            record["malformed_map_lines"] += 1
            continue
        matched = families_for(parsed["path"])
        if not matched:
            continue
        rows.append({**parsed, "families": list(matched), "raw_line": raw})
        for family in matched:
            record["family_coverage"][family] = True
        if len(matched) > 1:
            _issue(record, "ambiguous_family_path")
    record["matching_map_rows"] = rows
    if record["malformed_map_lines"]:
        _issue(record, "malformed_maps_lines")


def _stat_error(exc: OSError) -> str:
    return "missing_file" if isinstance(exc, FileNotFoundError) else "file_read_error"


def _hash_path(files, path: str, identity: tuple[str, int] | None = None) -> dict:
    result = {
        "path": path,
        "stat_before": None,
        "stat_after": None,
        "sha256": None,
        "size_bytes": None,
        "status": "not_hashed",
        "issues": [],
    }
    try:
        path_before = files.stat(path)
        result["stat_before"] = _snapshot(path_before)
    except OSError as exc:
        result["status"] = _stat_error(exc)
        result["issues"].append(result["status"])
        return result
    if identity and not _map_identity_matches(path_before, *identity):
        result.update(status="changed_file", issues=["changed_file", "mapping_stat_mismatch"])
        return result

    try:
        handle = files.open(path)
    except OSError as exc:
        result["status"] = _stat_error(exc)
        result["issues"].append(result["status"])
        return result
    with handle:
        try:
            before = files.fstat(handle)
            if identity and not _map_identity_matches(before, *identity):
                result.update(status="changed_file", issues=["changed_file", "mapping_stat_mismatch"])
                return result
            if not _same_stat(path_before, before):
                result.update(status="changed_file", issues=["changed_file"])
                return result
            digest = hashlib.sha256()
            size = 0
            while chunk := handle.read(CHUNK_BYTES):
                digest.update(chunk)
                size += len(chunk)
            after = files.fstat(handle)
        except OSError:
            result.update(status="file_read_error", issues=["file_read_error"])
            return result
        try:
            path_after = files.stat(path)
        except OSError:
            path_after = None

    result["sha256"] = digest.hexdigest()
    result["size_bytes"] = size
    result["stat_after"] = _snapshot(after)
    changed = not _same_stat(before, after) or size != before.st_size
    changed |= path_after is None or not _same_stat(after, path_after)
    if identity:
        changed |= not _map_identity_matches(after, *identity)
    if changed:
        result.update(status="changed_file", issues=["changed_file"])
    else:
        result["status"] = "hashed"
    return result


def _mapped_groups(records: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, int], dict] = {}
    invalid = []
    for rank_record in records:
        for row in rank_record["matching_map_rows"]:
            ref = {
                "rank": rank_record["rank"],
                "pid": rank_record["pid"],
                "address": row["address"],
                "path": row["path"],
            }
            if row["inode"] <= 0 or _dev_parts(row["device"]) is None:
                _issue(rank_record, "invalid_mapping_identity")
                invalid.append(
                    {
                        "identity": {"device": row["device"], "inode": row["inode"]},
                        "families": row["families"],
                        "mapping_refs": [ref],
                        "status": "invalid_mapping_identity",
                        "issues": ["invalid_mapping_identity"],
                    }
                )
                continue
            group = groups.setdefault(
                (row["device"], row["inode"]),
                {
                    "identity": {"device": row["device"], "inode": row["inode"]},
                    "families": set(),
                    "mapping_refs": [],
                    "paths": [],
                    "deleted_paths": [],
                },
            )
            group["families"].update(row["families"])
            group["mapping_refs"].append(ref)
            if row["path"] not in group["paths"]:
                group["paths"].append(row["path"])
            if row["deleted"] and row["path"] not in group["deleted_paths"]:
                group["deleted_paths"].append(row["path"])
    return list(groups.values()), invalid


def _hash_group(group: dict, files) -> dict:
    record = {
        "identity": group["identity"],
        "families": sorted(group["families"]),
        "mapping_refs": group["mapping_refs"],
        "mapping_paths": group["paths"],
        "deleted_mapping_paths": group["deleted_paths"],
        "selected_path": None,
        "sha256": None,
        "size_bytes": None,
        "stat_before": None,
        "stat_after": None,
        "status": "not_hashed",
        "issues": [],
        "path_checks": [],
    }
    candidates = [p for p in group["paths"] if p and not p.endswith(DELETED)]
    if not candidates:
        record.update(status="deleted_file", issues=["deleted_mapping_file"])
        return record
    errors = []
    for path in candidates:
        check = _hash_path(files, path, (group["identity"]["device"], group["identity"]["inode"]))
        record["path_checks"].append(check)
        errors.extend(code for code in check["issues"] if code not in errors)
        if check["status"] in ("hashed", "changed_file"):
            record.update(
                selected_path=path,
                sha256=check["sha256"],
                size_bytes=check["size_bytes"],
                stat_before=check["stat_before"],
                stat_after=check["stat_after"],
            )
        if check["status"] == "hashed":
            record["status"] = "hashed"
            if group["deleted_paths"]:
                errors.append("deleted_mapping_path_present")
            record["issues"] = errors
            return record
    if "changed_file" in errors:
        record["status"] = "changed_file"
    elif "missing_file" in errors:
        record["status"] = "missing_file"
    elif "file_read_error" in errors:
        record["status"] = "read_error"
    else:
        record["status"] = "changed_file"
    record["issues"] = errors or [record["status"]]
    return record


def _package_roots(records: list[dict]) -> list[Path]:
    roots = set()
    for record in records:
        for row in record["matching_map_rows"]:
            path = row["path"]
            if not path:
                continue
            parts = Path(path.removesuffix(DELETED)).parts
            indices = [i for i, part in enumerate(parts) if part.lower() == "cudacherry"]
            if indices:
                roots.add(Path(*parts[: indices[-1] + 1]))
    return sorted(roots, key=str)


def _static_sources(records: list[dict], files) -> list[dict]:
    result = []
    for root in _package_roots(records):
        for relative in ("_loader.py", "quantization/blockwise.py"):
            path = str(root / relative)
            item = _hash_path(files, path)
            result.append(
                {
                    "package_root_from_mapped_path": str(root),
                    "relative_path": relative,
                    **item,
                    "discovery": "cudacherry ancestor of a matching map path; no import",
                }
            )
    return result


def _valid_attempt(attempt_dir: str) -> bool:
    return os.path.isabs(attempt_dir) and os.path.normpath(attempt_dir) == attempt_dir


def collect_receipt(
    attempt_dir: str,
    rank_pid_specs: list[str],
    *,
    proc=None,
    files=None,
    clock=now,
) -> dict:
    started = clock()
    proc = proc or ProcAccess()
    files = files or FileAccess()
    bindings, input_errors = parse_bindings(rank_pid_specs)
    if not _valid_attempt(attempt_dir):
        input_errors.append("attempt_dir_must_be_absolute_and_normalized")
    input_valid = not input_errors
    ranks = [
        {
            "rank": rank,
            "pid": bindings.get(rank),
            "timestamps_utc": {"capture_started": clock()},
            "cmdline_scope": None,
            "start_time_ticks": None,
            "maps_status": "not_read",
            "matching_map_rows": [],
            "malformed_map_lines": 0,
            "family_coverage": {family: False for family in FAMILIES},
            "issues": [],
            "binding_checks": [],
        }
        for rank in RANKS
    ]

    preflight = {}
    start_ticks = {}
    if not input_valid:
        for record in ranks:
            if record["pid"] is None:
                _issue(record, "missing_rank_pid")
    else:
        for record in ranks:
            check = _binding_check(proc, record["pid"], attempt_dir)
            _record_check(record, "preflight", check)
            record["timestamps_utc"]["preflight_finished"] = clock()
            preflight[record["rank"]] = check["ok"]
            if check["start_ticks_before"] is not None:
                start_ticks[record["rank"]] = check["start_ticks_before"]

        for record in ranks:
            rank, pid = record["rank"], record["pid"]
            if not preflight.get(rank):
                record["maps_status"] = "not_read_binding_invalid"
                continue
            check = _binding_check(proc, pid, attempt_dir, start_ticks[rank])
            _record_check(record, "immediately_before_maps", check)
            if not check["ok"]:
                record["maps_status"] = "not_read_binding_invalid"
                continue
            record["timestamps_utc"]["maps_read_started"] = clock()
            try:
                data = proc.maps(pid)
            except FileNotFoundError:
                record["maps_status"] = "read_error"
                _issue(record, "changed_pid")
                _issue(record, "maps_read_error")
                continue
            except OSError as exc:
                record["maps_status"] = "read_error"
                _issue(record, "maps_read_error")
                _issue(record, type(exc).__name__)
                continue
            record["maps_status"] = "read"
            _matching_rows(record, data)
            record["timestamps_utc"]["maps_read_finished"] = clock()
            after = _binding_check(proc, pid, attempt_dir, start_ticks[rank])
            _record_check(record, "after_maps_read", after)

        groups, invalid = _mapped_groups(ranks)
        unique_files = [_hash_group(group, files) for group in groups] + invalid
        static = _static_sources(ranks, files)
        for record in ranks:
            rank = record["rank"]
            if preflight.get(rank):
                check = _binding_check(proc, record["pid"], attempt_dir, start_ticks[rank])
                _record_check(record, "after_hashing", check)
    if not input_valid:
        unique_files, static = [], []

    for record in ranks:
        record["timestamps_utc"].setdefault("capture_finished", clock())
    maps_complete = all(
        record["maps_status"] == "read" and record["malformed_map_lines"] == 0
        for record in ranks
    )
    family_coverage = {
        family: all(record["family_coverage"][family] for record in ranks)
        for family in FAMILIES
    }
    files_complete = all(
        item["status"] == "hashed" and not item["issues"] for item in unique_files
    )
    bindings_stable = all(
        not check["issues"]
        for item in ranks
        for check in item["binding_checks"]
    ) and all(item["binding_checks"] for item in ranks)
    complete = bool(
        input_valid
        and maps_complete
        and family_coverage["quantize"]
        and family_coverage["gemm"]
        and files_complete
        and bindings_stable
    )
    return {
        "schema_version": 1,
        "capture_status": "complete" if complete else "partial",
        "timestamps_utc": {"capture_started": started, "capture_finished": clock()},
        "attempt_scope": {
            "exact_attempt_dir": attempt_dir,
            "absolute_and_normalized": _valid_attempt(attempt_dir),
        },
        "collector": {
            "method": "caller-supplied /proc cmdline/stat/maps plus mapped-file stat and SHA-256",
            "proc_root": "/proc",
            "python_version": sys.version.split()[0],
            "torch_or_extension_imported": False,
            "process_enumeration_used": False,
            "environment_read": False,
            "cuda_or_barrier_used": False,
            "jit_or_reload_used": False,
            "sys_modules_snapshot": "not captured",
        },
        "input_validation": {
            "valid": input_valid,
            "requested_rank_pid_specs": rank_pid_specs,
            "required_ranks": list(RANKS),
            "errors": input_errors,
        },
        "coverage": {
            "all_ranks_have_quantize_mapping": family_coverage["quantize"],
            "all_ranks_have_gemm_mapping": family_coverage["gemm"],
            "all_rank_maps_read": maps_complete,
            "all_unique_mapped_family_files_hashed_stably": files_complete,
            "rank_pid_bindings_stable": bindings_stable,
            "loaded_binary_file_provenance_complete": complete,
        },
        "rank_receipts": ranks,
        "unique_mapped_family_binaries": unique_files,
        "static_loader_wrapper_provenance": static,
        "limitations": [
            "Map paths and device/inode-bound hashes do not attest an extension's registered operator body, build origin, executed branch, or fault causality.",
            "No live sys.modules snapshot was captured.",
            "Conservative family matching requires a CudaCherry shared-object path naming quantize or GEMM; opaque JIT names remain uncovered.",
            "Adjacent loader/wrapper hashes are static source provenance, not evidence that those code paths executed.",
        ],
    }


def write_exclusive(output: str | Path, receipt: dict) -> None:
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--rank-pid", action="append", default=[], metavar="RANK:PID")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = collect_receipt(args.attempt_dir, args.rank_pid)
    try:
        write_exclusive(args.output, result)
    except FileExistsError:
        print(f"refusing to overwrite receipt: {args.output}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"receipt write failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({"capture_status": result["capture_status"], "output": args.output, "coverage": result["coverage"]}, sort_keys=True))
    return 0 if result["capture_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

