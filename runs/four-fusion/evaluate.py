#!/usr/bin/env python3
"""Run one independent four-fusion variant through the existing native runner."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


TASK_ROOT = Path(__file__).resolve().parents[2]
FUSION_DIR = TASK_ROOT / "runs/four-fusion"
RUNNER = TASK_ROOT / "runs/overnight-streaming-perf/run.py"
RUNNER_PYTHON = Path("/opt/torchtitan-venv/bin/python3.12")
RAW_ROOT = TASK_ROOT / "runs/overnight-streaming-perf/artifacts/raw"
RECORDS_DIR = FUSION_DIR / "driver-records"
LOCK_PATH = TASK_ROOT / ".tmp/four-fusion-evaluate.lock"

PATCH_FILES = {
    "input": FUSION_DIR / "input-prepare.patch",
    "routing": FUSION_DIR / "routing-metadata.patch",
    "linear-ce": FUSION_DIR / "linear-ce.patch",
    "ce": FUSION_DIR / "ce.patch",
}
CORE_TARGETS = (
    "gift/torchtitan/trainer.py",
    "gift/torchtitan/components/loss.py",
    "gift/torchtitan/distributed/deepep/routing.py",
    "gift/torchtitan/distributed/deepep/hybridep.py",
    "gift/torchtitan/models/common/moe/dispatch/hybridep.py",
)
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
PATCH_HEADER_RE = re.compile(rb"(?m)^diff --git a/([^\s]+) b/([^\s]+)\r?$")
UNIFIED_HEADER_RE = re.compile(
    rb"(?m)^--- a/([^\s\r\n]+)\r?\n\+\+\+ b/([^\s\r\n]+)\r?$"
)


class DriverError(RuntimeError):
    """A fail-closed validation, launch, or cleanup error."""


class RunnerLaunchError(DriverError):
    def __init__(self, message: str, *, started: bool):
        super().__init__(message)
        self.started = started


def _git(
    *args: str, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(TASK_ROOT), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _git_checked(
    *args: str, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    result = _git(*args, input_bytes=input_bytes)
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = result.stdout.decode("utf-8", errors="replace").strip()
        raise DriverError(
            f"{shlex.join(('git', *args))} failed with {result.returncode}: {detail}"
        )
    return result


def _patch_paths(patch: bytes) -> tuple[str, ...]:
    headers = PATCH_HEADER_RE.findall(patch) or UNIFIED_HEADER_RE.findall(patch)
    if not headers:
        raise DriverError(
            "candidate patch has no supported Git or unified file headers"
        )
    paths = []
    for old_raw, new_raw in headers:
        old = old_raw.decode("utf-8", errors="strict")
        new = new_raw.decode("utf-8", errors="strict")
        parts = Path(old).parts
        if (
            old != new
            or Path(old).is_absolute()
            or ".." in parts
            or not parts
            or parts[0] != "gift"
        ):
            raise DriverError(
                f"candidate patch path is not an in-place gift/ path: {old}"
            )
        paths.append(old)
    return tuple(dict.fromkeys(paths))


def _load_patch(variant: str) -> tuple[bytes, str, tuple[str, ...]]:
    path = PATCH_FILES[variant]
    try:
        patch = path.read_bytes()
    except FileNotFoundError as exc:
        raise DriverError(f"candidate patch is not ready: {path}") from exc
    if not patch:
        raise DriverError(f"candidate patch is empty: {path}")
    return patch, hashlib.sha256(patch).hexdigest(), _patch_paths(patch)


def _build_command(
    variant: str, mode: str, run_id: str, driver_argv: list[str]
) -> dict[str, Any]:
    if variant not in {"control", *PATCH_FILES}:
        raise DriverError(f"unknown variant: {variant}")
    if mode not in {"tps", "trace"}:
        raise DriverError(f"unknown mode: {mode}")
    if not RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise DriverError(
            "run-id must contain only letters, digits, dot, underscore, or hyphen"
        )

    runner_run_id = f"four-fusion-{variant}-{run_id}"
    workload_args = (
        ("--steps", "20", "--warmup", "10", "--window", "10")
        if mode == "tps"
        else (
            "--steps",
            "11",
            "--warmup",
            "10",
            "--window",
            "0",
            "--profile-update",
            "11",
            "--profile-microbatches",
            "3",
        )
    )
    command = (
        str(RUNNER_PYTHON),
        str(RUNNER),
        "--run-id",
        runner_run_id,
        *workload_args,
        "--epoch-size",
        "51200",
        "--num-prefetch-batches",
        "32",
    )
    patch_info = None
    if variant != "control":
        patch, patch_sha, patch_targets = _load_patch(variant)
        if not set(patch_targets).intersection(CORE_TARGETS):
            raise DriverError(
                "candidate patch does not integrate a core execution target"
            )
        patch_info = {
            "path": str(PATCH_FILES[variant]),
            "bytes": patch,
            "sha256": patch_sha,
            "targets": patch_targets,
        }
    return {
        "variant": variant,
        "mode": mode,
        "run_id": run_id,
        "runner_run_id": runner_run_id,
        "command": command,
        "patch": patch_info,
        "driver_command": [sys.executable, str(Path(__file__).resolve()), *driver_argv],
    }


def _check_preflight(plan: dict[str, Any]) -> None:
    if not RUNNER.is_file() or not RUNNER_PYTHON.is_file():
        raise DriverError(
            f"existing runner or interpreter is missing: {RUNNER}, {RUNNER_PYTHON}"
        )
    root = Path(
        _git_checked("rev-parse", "--show-toplevel").stdout.decode().strip()
    ).resolve()
    if root != TASK_ROOT.resolve():
        raise DriverError(f"expected task Git root {TASK_ROOT.resolve()}, found {root}")

    patch_info = plan["patch"]
    targets = list(CORE_TARGETS)
    if patch_info is not None:
        targets.extend(patch_info["targets"])
    tracked = _git_checked("ls-files", "-z", "--", *dict.fromkeys(targets))
    tracked_paths = set(filter(None, tracked.stdout.decode().split("\0")))
    missing = sorted(set(CORE_TARGETS) - tracked_paths)
    if missing:
        raise DriverError(
            f"expected core target files are not tracked: {', '.join(missing)}"
        )
    clean = _git("diff", "--quiet", "HEAD", "--", *sorted(tracked_paths))
    if clean.returncode == 1:
        raise DriverError(
            "candidate integration or patch stack is active in baseline target files; "
            "restore the pristine V2 target files before evaluation"
        )
    if clean.returncode:
        raise DriverError("could not verify clean baseline target files")

    if patch_info is not None:
        patch = patch_info["bytes"]
        reverse = _git("apply", "--reverse", "--check", input_bytes=patch)
        if reverse.returncode == 0:
            raise DriverError(
                f"selected patch is already integrated: {patch_info['path']}"
            )
        check = _git("apply", "--check", input_bytes=patch)
        if check.returncode:
            detail = check.stderr.decode("utf-8", errors="replace").strip()
            raise DriverError(f"selected patch does not apply cleanly: {detail}")

    record_path = RECORDS_DIR / f"{plan['run_id']}.json"
    attempt_path = RAW_ROOT / plan["runner_run_id"]
    if record_path.exists() or attempt_path.exists():
        raise DriverError("run-id already exists; choose a new ID for each attempt")
    if any(path.is_symlink() for path in (FUSION_DIR, RECORDS_DIR, RAW_ROOT)):
        raise DriverError("task output paths must not be symlinks")


@contextmanager
def _exclusive_lock() -> Iterator[None]:
    if LOCK_PATH.parent.is_symlink() or LOCK_PATH.is_symlink():
        raise DriverError(f"task-local lock path must not be a symlink: {LOCK_PATH}")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(LOCK_PATH, flags, 0o600)
    except OSError as exc:
        raise DriverError(f"could not open task-local lock: {exc}") from exc
    with os.fdopen(fd, "r+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DriverError(
                "another four-fusion driver holds the task-local lock"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write_record(path: Path, record: dict[str, Any], *, create: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, indent=2)
            handle.write("\n")
        return
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _run_runner(command: list[str]) -> int:
    child: subprocess.Popen[bytes] | None = None
    old_handlers: dict[int, Any] = {}

    def forward_to_runner(_signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signal.SIGINT)
            except OSError:
                pass

    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    try:
        for signum in signals:
            old_handlers[signum] = signal.signal(signum, forward_to_runner)
        try:
            child = subprocess.Popen(command, cwd=TASK_ROOT, start_new_session=True)
        except OSError as exc:
            raise RunnerLaunchError(
                f"could not start native runner: {exc}", started=False
            ) from exc
        try:
            return child.wait()
        except BaseException as exc:
            raise RunnerLaunchError(
                f"native runner wait failed for PID {child.pid}: {type(exc).__name__}: {exc}",
                started=True,
            ) from exc
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _runner_receipts(plan: dict[str, Any]) -> dict[str, Any]:
    attempt = RAW_ROOT / plan["runner_run_id"]
    process = _read_json(attempt / "process.json")
    completion = _read_json(attempt / "completion.json")
    failure = _read_json(attempt / "failure.json")
    return {
        "attempt_path": str(attempt),
        "receipt_paths": {
            name: str(attempt / f"{name}.json")
            for name in ("process", "completion", "failure")
            if (attempt / f"{name}.json").is_file()
        },
        "process": {
            key: process[key]
            for key in (
                "pid",
                "process_group_id",
                "process_group_stopped",
                "orphaned_process_group",
            )
            if process is not None and key in process
        }
        if process is not None
        else None,
        "completion_status": completion.get("status") if completion else None,
        "failure_status": failure.get("status") if failure else None,
        "failure_error": failure.get("error") if failure else None,
        "failure_pid": failure.get("pid") if failure else None,
        "launcher_pid_path": str(attempt / "launcher.pid"),
    }


def _group_stopped(receipts: dict[str, Any]) -> bool:
    process = receipts.get("process")
    if isinstance(process, dict):
        return (
            isinstance(process.get("pid"), int)
            and process.get("pid") == process.get("process_group_id")
            and process.get("process_group_stopped") is True
        )
    if receipts.get("failure_status") != "INTERRUPTED":
        return False
    try:
        launcher_pid = int(Path(receipts["launcher_pid_path"]).read_text().strip())
    except (OSError, ValueError):
        return False
    if receipts.get("failure_pid") not in (None, launcher_pid):
        return False
    try:
        os.killpg(launcher_pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _reverse_patch(patch: bytes) -> None:
    check = _git("apply", "--reverse", "--check", input_bytes=patch)
    if check.returncode:
        detail = check.stderr.decode("utf-8", errors="replace").strip()
        raise DriverError(f"selected patch cannot be safely reversed: {detail}")
    _git_checked("apply", "--reverse", input_bytes=patch)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_evaluation(
    variant: str,
    mode: str,
    run_id: str,
    *,
    driver_argv: list[str],
    dry_run: bool = False,
) -> int:
    plan = _build_command(variant, mode, run_id, driver_argv)
    with _exclusive_lock():
        _check_preflight(plan)
        if dry_run:
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "mode": mode,
                        "run_id": run_id,
                        "runner_run_id": plan["runner_run_id"],
                        "decision": "control: no patch"
                        if plan["patch"] is None
                        else "apply exactly one selected patch",
                        "patch": (
                            None
                            if plan["patch"] is None
                            else {
                                "path": plan["patch"]["path"],
                                "sha256": plan["patch"]["sha256"],
                                "targets": plan["patch"]["targets"],
                            }
                        ),
                        "driver_command_argv": plan["driver_command"],
                        "runner_command_argv": plan["command"],
                        "runner_attempt_path": str(RAW_ROOT / plan["runner_run_id"]),
                        "dry_run": True,
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0

        record_path = RECORDS_DIR / f"{run_id}.json"
        patch = plan["patch"]
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "PREPARING",
            "run_id": run_id,
            "variant": variant,
            "mode": mode,
            "runner_run_id": plan["runner_run_id"],
            "patch": (
                None
                if patch is None
                else {key: patch[key] for key in ("path", "sha256", "targets")}
            ),
            "driver_command_argv": plan["driver_command"],
            "runner_command_argv": plan["command"],
            "runner_attempt_path": str(RAW_ROOT / plan["runner_run_id"]),
            "started_at_utc": _now(),
            "selected_patch_restored": None,
        }
        _write_record(record_path, record, create=True)

        applied = False
        child_started = False
        receipts = None
        try:
            if patch is not None:
                _git_checked("apply", "--check", input_bytes=patch["bytes"])
                _git_checked("apply", input_bytes=patch["bytes"])
                applied = True
                reverse_check = _git(
                    "apply", "--reverse", "--check", input_bytes=patch["bytes"]
                )
                if reverse_check.returncode:
                    raise DriverError(
                        "applied patch cannot be exactly reversed; source remains active"
                    )
                record["status"] = "PATCH_APPLIED"
                _write_record(record_path, record)

            try:
                child_code = _run_runner(plan["command"])
            except RunnerLaunchError as exc:
                child_started = exc.started
                raise
            child_started = True
            receipts = _runner_receipts(plan)
            group_stopped = _group_stopped(receipts)
            record["runner_return_code"] = child_code
            record["runner_receipts"] = receipts
            record["group_exit_confirmed"] = group_stopped
            record["completion_status"] = receipts["completion_status"]

            if not group_stopped:
                record["status"] = "RUNNER_EXIT_UNCERTAIN"
                record["selected_patch_restored"] = not applied
                raise DriverError(
                    "native runner group exit is not confirmed; selected patch remains active"
                )

            if applied:
                try:
                    _reverse_patch(patch["bytes"])
                except DriverError as exc:
                    record["selected_patch_restored"] = False
                    record["restore_error"] = str(exc)
                    raise
                applied = False
                record["selected_patch_restored"] = True

            success = child_code == 0 and receipts["completion_status"] == "PASS"
            record["status"] = "PASS" if success else "FAIL"
            record["finished_at_utc"] = _now()
            _write_record(record_path, record)
            if success:
                return 0
            return (
                child_code
                if child_code > 0
                else (128 - child_code if child_code < 0 else 1)
            )
        except BaseException as exc:
            if isinstance(exc, RunnerLaunchError):
                child_started = exc.started
            if child_started:
                receipts = receipts or _runner_receipts(plan)
                group_stopped = _group_stopped(receipts)
                record["runner_receipts"] = receipts
                record["group_exit_confirmed"] = group_stopped
                if not group_stopped:
                    record["status"] = "RUNNER_EXIT_UNCERTAIN"
                    record["selected_patch_restored"] = not applied
                elif applied and record.get("selected_patch_restored") is not False:
                    try:
                        _reverse_patch(patch["bytes"])
                        applied = False
                        record["selected_patch_restored"] = True
                    except DriverError as restore_error:
                        record["selected_patch_restored"] = False
                        record["restore_error"] = str(restore_error)
            elif applied:
                try:
                    _reverse_patch(patch["bytes"])
                    applied = False
                    record["selected_patch_restored"] = True
                except DriverError as restore_error:
                    record["selected_patch_restored"] = False
                    record["restore_error"] = str(restore_error)

            if record.get("status") != "RUNNER_EXIT_UNCERTAIN":
                record["status"] = "FAIL"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["finished_at_utc"] = _now()
            _write_record(record_path, record)
            raise


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("control", *PATCH_FILES))
    parser.add_argument("mode", choices=("tps", "trace"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(raw_argv)
    try:
        return run_evaluation(
            args.variant,
            args.mode,
            args.run_id,
            driver_argv=raw_argv,
            dry_run=args.dry_run,
        )
    except (DriverError, OSError, subprocess.SubprocessError) as exc:
        print(f"four-fusion driver: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
