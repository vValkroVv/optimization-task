#!/usr/bin/env python3
"""Focused CPU-only checks for capture_loaded_binaries.py."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import capture_loaded_binaries as receipt


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeProc:
    def __init__(
        self,
        attempt_dir: str,
        maps_data: bytes,
        *,
        missing: set[int] | None = None,
        wrong_scope: set[int] | None = None,
        change_on_maps: set[int] | None = None,
        read_error_after_maps: set[int] | None = None,
        on_first_maps=None,
    ) -> None:
        self.attempt_dir = attempt_dir.encode()
        self.maps_data = maps_data
        self.missing = missing or set()
        self.wrong_scope = wrong_scope or set()
        self.change_on_maps = change_on_maps or set()
        self.read_error_after_maps = read_error_after_maps or set()
        self.on_first_maps = on_first_maps
        self.read_calls: list[tuple[str, int]] = []
        self.map_calls: list[int] = []
        self.maps_seen: set[int] = set()
        self.changed: set[int] = set()

    def start_ticks(self, pid: int) -> int:
        self.read_calls.append(("stat", pid))
        if pid in self.missing:
            raise FileNotFoundError(pid)
        return 100_000 + pid + (1 if pid in self.changed else 0)

    def cmdline(self, pid: int) -> bytes:
        self.read_calls.append(("cmdline", pid))
        if pid in self.missing:
            raise FileNotFoundError(pid)
        if pid in self.read_error_after_maps and pid in self.maps_seen:
            raise PermissionError(pid)
        scope = b"/other/attempt" if pid in self.wrong_scope else self.attempt_dir
        return b"python\0--attempt\0" + scope + b"\0"

    def maps(self, pid: int) -> bytes:
        self.read_calls.append(("maps", pid))
        self.map_calls.append(pid)
        if self.on_first_maps is not None and len(self.map_calls) == 1:
            self.on_first_maps()
        if pid in self.change_on_maps:
            self.changed.add(pid)
        self.maps_seen.add(pid)
        return self.maps_data


class TrackingFiles(receipt.FileAccess):
    def __init__(self) -> None:
        self.opened_paths: list[str] = []

    def open(self, path: str):
        self.opened_paths.append(path)
        return super().open(path)


def _rank_specs() -> list[str]:
    return [f"{rank}:{20_000 + rank}" for rank in range(8)]


def _map_row(path: Path, address: str = "7f0000000000-7f0000010000") -> str:
    result = path.stat()
    device = f"{os.major(result.st_dev):x}:{os.minor(result.st_dev):x}"
    return f"{address} r-xp 00000000 {device} {result.st_ino} {path}"


def _family_files(root: Path) -> tuple[Path, Path]:
    package = root / "cache path with spaces" / "cudacherry"
    quantize = package / "quantization" / "_C_quantize.cpython-312-x86_64-linux-gnu.so"
    gemm = package / "gemm" / "_C_gemm.cpython-312-x86_64-linux-gnu.so"
    quantize.parent.mkdir(parents=True)
    gemm.parent.mkdir(parents=True)
    quantize.write_bytes(b"quantize-binary-fixture\x00" * 3)
    gemm.write_bytes(b"gemm-binary-fixture\x00" * 5)
    return quantize, gemm


def _collect(attempt: Path, maps_data: bytes, *, proc_kwargs=None, files=None):
    fake_proc = FakeProc(str(attempt), maps_data, **(proc_kwargs or {}))
    result = receipt.collect_receipt(
        str(attempt),
        _rank_specs(),
        proc=fake_proc,
        files=files or TrackingFiles(),
        clock=lambda: "2026-11-12T00:00:00.000Z",
    )
    return result, fake_proc


def check_maps_parser_spaces_and_deleted_suffix() -> None:
    line = (
        "7f0000000000-7f0000010000 r-xp 00000000 fd:00 74482301 "
        "/opt/Torch Cherry/cudacherry/gemm/_C_gemm.cpython-312-x86_64-linux-gnu.so (deleted)"
    )
    parsed = receipt.parse_maps_line(line)
    _check(parsed is not None, "maps row with a path containing spaces did not parse")
    _check(parsed["path"].startswith("/opt/Torch Cherry/"), "path spaces were lost")
    _check(parsed["path"].endswith(" (deleted)"), "deleted suffix was lost")
    _check(parsed["deleted"], "deleted suffix was not recorded")
    _check(receipt.families_for(parsed["path"]) == ("gemm",), "GEMM family was not recognized")


def check_stable_unique_hashes_across_eight_ranks() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        quantize, gemm = _family_files(root)
        maps_data = (_map_row(quantize) + "\n" + _map_row(gemm, "7f1000000000-7f1000010000") + "\n").encode()
        files = TrackingFiles()
        result, _proc = _collect(attempt, maps_data, files=files)
        _check(result["capture_status"] == "complete", "stable fixture did not complete")
        _check(result["coverage"]["all_ranks_have_quantize_mapping"], "quantize coverage missing")
        _check(result["coverage"]["all_ranks_have_gemm_mapping"], "GEMM coverage missing")
        binaries = result["unique_mapped_family_binaries"]
        _check(len(binaries) == 2, "file identities were not deduplicated across ranks")
        _check(all(item["status"] == "hashed" for item in binaries), "stable binaries were not hashed")
        _check(len(files.opened_paths) == 2, "a mapped identity was opened more than once")
        by_hash = {item["sha256"]: item for item in binaries}
        _check(hashlib.sha256(quantize.read_bytes()).hexdigest() in by_hash, "quantize digest mismatch")
        _check(hashlib.sha256(gemm.read_bytes()).hexdigest() in by_hash, "GEMM digest mismatch")
        _check(all(len(rank["matching_map_rows"]) == 2 for rank in result["rank_receipts"]), "rank rows missing")


def check_wrong_scope_and_missing_pid_never_read_maps() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        quantize, gemm = _family_files(root)
        maps_data = (_map_row(quantize) + "\n" + _map_row(gemm, "7f1000000000-7f1000010000") + "\n").encode()
        specs = _rank_specs()
        pids = [int(spec.split(":")[1]) for spec in specs]
        result, fake = _collect(
            attempt,
            maps_data,
            proc_kwargs={"missing": {pids[0]}, "wrong_scope": {pids[1]}},
        )
        by_rank = {item["rank"]: item for item in result["rank_receipts"]}
        _check("missing_pid" in by_rank[0]["issues"], "missing PID not flagged")
        _check(
            "wrong_attempt_scope" in by_rank[1]["issues"],
            "wrong attempt scope not flagged",
        )
        _check(pids[0] not in fake.map_calls and pids[1] not in fake.map_calls, "maps read before scope validation")
        _check(result["capture_status"] == "partial", "scope failure incorrectly completed")


def check_static_source_hashes_are_optional_and_nonimporting() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        quantize, gemm = _family_files(root)
        package = quantize.parent.parent
        loader = package / "_loader.py"
        wrapper = package / "quantization" / "blockwise.py"
        loader.write_text("loader source fixture\n")
        wrapper.write_text("wrapper source fixture\n")
        maps_data = (_map_row(quantize) + "\n" + _map_row(gemm, "7f1000000000-7f1000010000") + "\n").encode()
        result, _proc = _collect(attempt, maps_data)
        sources = {item["relative_path"]: item for item in result["static_loader_wrapper_provenance"]}
        _check(set(sources) == {"_loader.py", "quantization/blockwise.py"}, "static sources missing")
        _check(all(item["status"] == "hashed" for item in sources.values()), "static source hashes failed")
        _check(
            sources["_loader.py"]["sha256"] == hashlib.sha256(loader.read_bytes()).hexdigest(),
            "loader source digest mismatch",
        )
        _check(
            sources["quantization/blockwise.py"]["sha256"] == hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            "wrapper source digest mismatch",
        )
        _check(result["capture_status"] == "complete", "optional static source status changed binary coverage")


def check_duplicate_pid_is_rejected_before_proc_reads() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt = Path(temp) / "attempt"
        attempt.mkdir()
        specs = _rank_specs()
        specs[1] = f"1:{specs[0].split(':')[1]}"
        fake = FakeProc(str(attempt), b"")
        result = receipt.collect_receipt(
            str(attempt),
            specs,
            proc=fake,
            clock=lambda: "2026-11-12T00:00:00.000Z",
        )
        _check(
            any(error.startswith("duplicate_pid:") for error in result["input_validation"]["errors"]),
            "duplicate PID not rejected",
        )
        _check(not fake.read_calls, "process files were read with a duplicate PID map")
        _check(result["capture_status"] == "partial", "duplicate PID incorrectly completed")


def check_missing_file_and_inode_replacement_are_partial() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        quantize, gemm = _family_files(root)
        missing_quantize = quantize.parent / "missing_quantize.so"
        missing_row = _map_row(quantize).replace(str(quantize), str(missing_quantize))
        maps_data = (missing_row + "\n" + _map_row(gemm, "7f1000000000-7f1000010000") + "\n").encode()
        result, _proc = _collect(attempt, maps_data)
        quantize_record = next(
            item for item in result["unique_mapped_family_binaries"] if "quantize" in item["families"]
        )
        _check(quantize_record["status"] == "missing_file", "missing mapped file not flagged")
        _check(result["capture_status"] == "partial", "missing file incorrectly completed")

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        quantize, gemm = _family_files(root)
        old_quantize_row = _map_row(quantize)
        replacement = quantize.with_suffix(".replacement")
        replacement.write_bytes(b"replacement-inode\x00")

        def replace_after_maps_snapshot() -> None:
            os.replace(replacement, quantize)

        maps_data = (old_quantize_row + "\n" + _map_row(gemm, "7f1000000000-7f1000010000") + "\n").encode()
        result, _proc = _collect(
            attempt,
            maps_data,
            proc_kwargs={"on_first_maps": replace_after_maps_snapshot},
        )
        quantize_record = next(
            item for item in result["unique_mapped_family_binaries"] if "quantize" in item["families"]
        )
        _check(quantize_record["status"] == "changed_file", "in-place inode replacement not flagged")
        _check("mapping_stat_mismatch" in quantize_record["issues"], "mapping/stat inode mismatch missing")
        _check(result["capture_status"] == "partial", "inode change incorrectly completed")


def check_missing_family_and_changed_pid_are_partial() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        _quantize, gemm = _family_files(root)
        maps_data = (_map_row(gemm) + "\n").encode()
        result, _proc = _collect(attempt, maps_data)
        _check(not result["coverage"]["all_ranks_have_quantize_mapping"], "absent quantize family was covered")
        _check(not result["coverage"]["loaded_binary_file_provenance_complete"], "absent family incorrectly completed")

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        quantize, gemm = _family_files(root)
        maps_data = (_map_row(quantize) + "\n" + _map_row(gemm, "7f1000000000-7f1000010000") + "\n").encode()
        specs = _rank_specs()
        changed_pid = int(specs[0].split(":")[1])
        result, _proc = _collect(
            attempt,
            maps_data,
            proc_kwargs={"change_on_maps": {changed_pid}},
        )
        first_rank = result["rank_receipts"][0]
        _check("changed_pid" in first_rank["issues"], "changed PID not flagged")
        _check(result["capture_status"] == "partial", "changed PID incorrectly completed")

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        quantize, gemm = _family_files(root)
        maps_data = (_map_row(quantize) + "\n" + _map_row(gemm, "7f1000000000-7f1000010000") + "\n").encode()
        specs = _rank_specs()
        read_error_pid = int(specs[0].split(":")[1])
        result, _proc = _collect(
            attempt,
            maps_data,
            proc_kwargs={"read_error_after_maps": {read_error_pid}},
        )
        first_rank = result["rank_receipts"][0]
        _check("proc_read_error" in first_rank["issues"], "late process read error not flagged")
        _check(not result["coverage"]["rank_pid_bindings_stable"], "late binding read error was ignored")
        _check(result["capture_status"] == "partial", "late binding read error incorrectly completed")


def check_deleted_and_unreadable_files_are_explicit() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        quantize, gemm = _family_files(root)
        deleted_row = _map_row(quantize) + " (deleted)"
        maps_data = (deleted_row + "\n" + _map_row(gemm, "7f1000000000-7f1000010000") + "\n").encode()
        result, _proc = _collect(attempt, maps_data)
        deleted = next(
            item for item in result["unique_mapped_family_binaries"] if "quantize" in item["families"]
        )
        _check(deleted["status"] == "deleted_file", "deleted-only mapping was not explicit")
        _check("deleted_mapping_file" in deleted["issues"], "deleted mapping issue missing")
        _check(result["capture_status"] == "partial", "deleted-only mapping incorrectly completed")

    class DeniedFiles(TrackingFiles):
        def stat(self, path: str):
            if "_C_quantize" in path:
                raise PermissionError(path)
            return super().stat(path)

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        quantize, gemm = _family_files(root)
        maps_data = (_map_row(quantize) + "\n" + _map_row(gemm, "7f1000000000-7f1000010000") + "\n").encode()
        result, _proc = _collect(attempt, maps_data, files=DeniedFiles())
        unreadable = next(
            item for item in result["unique_mapped_family_binaries"] if "quantize" in item["families"]
        )
        _check(unreadable["status"] == "read_error", "unreadable mapped file was not explicit")
        _check("file_read_error" in unreadable["issues"], "file read error issue missing")
        _check(result["capture_status"] == "partial", "unreadable file incorrectly completed")


def check_output_is_exclusive() -> None:
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "receipt.json"
        receipt.write_exclusive(output, {"capture_status": "partial"})
        try:
            receipt.write_exclusive(output, {"capture_status": "complete"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output was overwritten")
        _check('"partial"' in output.read_text(), "exclusive write changed the first receipt")


def check_cli_saves_partial_validation_receipt() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        attempt = root / "attempt"
        attempt.mkdir()
        output = root / "receipt.json"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = receipt.main(
                [
                    "--attempt-dir",
                    str(attempt),
                    "--rank-pid",
                    "0:12345",
                    "--output",
                    str(output),
                ]
            )
        saved = json.loads(output.read_text())
        _check(status == 1, "incomplete CLI capture did not return partial status")
        _check(saved["capture_status"] == "partial", "CLI did not save a partial receipt")
        _check(not saved["input_validation"]["valid"], "incomplete CLI mapping was accepted")
        _check("missing_rank:7" in saved["input_validation"]["errors"], "input error not recorded")


def main() -> None:
    checks = (
        check_maps_parser_spaces_and_deleted_suffix,
        check_stable_unique_hashes_across_eight_ranks,
        check_wrong_scope_and_missing_pid_never_read_maps,
        check_static_source_hashes_are_optional_and_nonimporting,
        check_duplicate_pid_is_rejected_before_proc_reads,
        check_missing_file_and_inode_replacement_are_partial,
        check_missing_family_and_changed_pid_are_partial,
        check_deleted_and_unreadable_files_are_explicit,
        check_output_is_exclusive,
        check_cli_saves_partial_validation_receipt,
    )
    for check in checks:
        check()
    print(f"PASS: {len(checks)} focused CPU checks")


if __name__ == "__main__":
    main()
