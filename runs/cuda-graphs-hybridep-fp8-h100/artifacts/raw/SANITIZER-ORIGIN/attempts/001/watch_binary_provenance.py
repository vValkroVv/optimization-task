#!/usr/bin/env python3
"""One-attempt READY gate and rank-log trigger for the loaded-binary receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

TASK_ROOT = Path("/workspace-SR008.nfs2/vkropotin/optimization-task")
DEFAULT_ATTEMPT = TASK_ROOT / "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/SANITIZER-ORIGIN/attempts/001"
HELPER_SHA256 = "9b2b304147532be2168867f7075a7599f0cf30c775034604deecdfe5a75f6073"
SOURCE_MANIFEST_SHA256 = "2585e504350276cb4689ae9ea45a33706e9f898bb61969abe5d08f2650c55620"
SOURCE_ARCHIVE_SHA256 = "0fc5ddad83c5f99b4036a47dd7f485ba2f42d1284c42f086d5f051cebad66082"
SANITIZER_SHA256 = "5cba659e1cd603aa2370e103425f184368e6ed405da90ce85743c4399c324dbc"
STORAGE_FLOOR = 4 * 1024**3
OUTPUT_BUDGET = 64 * 1024**2
RANKS = tuple(range(8))
RUN_DEADLINE_SECONDS = 20 * 60
HELPER_DEADLINE_SECONDS = 120
PROCESS_SCAN_SECONDS = 0.5


class WatchError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, record: dict) -> None:
    """Publish one complete JSON receipt atomically without replacing a prior one."""
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        # A same-directory hard link makes the complete file visible in one
        # operation and fails if the attempt already has this receipt.
        os.link(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        raise
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def publish_file_exclusive(source: Path, target: Path) -> None:
    """Expose a completed same-filesystem file without replacing a prior path."""
    os.link(source, target)
    directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WatchError(f"invalid or unavailable JSON record: {path.name}") from exc
    if not isinstance(value, dict):
        raise WatchError(f"expected JSON object: {path.name}")
    return value


def proc_identity(pid: int, proc_root: Path = Path("/proc")) -> dict:
    directory = proc_root / str(pid)
    raw = (directory / "stat").read_text()
    close = raw.rfind(")")
    if close < 0:
        raise WatchError(f"malformed proc stat for PID {pid}")
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        raise WatchError(f"short proc stat for PID {pid}")
    try:
        start_ticks = int(fields[19])
        ppid = int(fields[1])
        process_group = int(fields[2])
        session = int(fields[3])
    except ValueError as exc:
        raise WatchError(f"invalid proc start identity for PID {pid}") from exc
    return {
        "start_ticks": start_ticks,
        "state": fields[0],
        "ppid": ppid,
        "process_group": process_group,
        "session": session,
        "cmdline": (directory / "cmdline").read_bytes(),
    }


def validate_live_pid(
    pid: int,
    start_ticks: int,
    attempt_dir: Path,
    *,
    marker: Path | None = None,
    proc_root: Path = Path("/proc"),
) -> None:
    try:
        before = proc_identity(pid, proc_root)
        after = proc_identity(pid, proc_root)
    except (OSError, WatchError) as exc:
        raise WatchError(f"PID {pid} is not live") from exc
    if before["start_ticks"] != start_ticks or after["start_ticks"] != start_ticks:
        raise WatchError(f"PID {pid} identity changed")
    if before["cmdline"] != after["cmdline"]:
        raise WatchError(f"PID {pid} command changed during validation")
    if os.fsencode(str(attempt_dir.resolve())) not in after["cmdline"]:
        raise WatchError(f"PID {pid} command does not name this attempt")
    if marker is not None and os.fsencode(str(marker.resolve())) not in after["cmdline"]:
        raise WatchError(f"PID {pid} command does not name {marker.name}")


def write_ready(
    attempt_dir: Path,
    *,
    watcher_pid: int | None = None,
    proc_root: Path = Path("/proc"),
    watcher_path: Path | None = None,
    created_epoch_ns: int | None = None,
) -> dict:
    attempt_dir = attempt_dir.resolve()
    watcher_path = (watcher_path or Path(__file__)).resolve()
    helper_path = attempt_dir / "capture_loaded_binaries.py"
    if sha256(helper_path) != HELPER_SHA256:
        raise WatchError("binary receipt helper SHA-256 mismatch")
    watcher_pid = os.getpid() if watcher_pid is None else watcher_pid
    identity = proc_identity(watcher_pid, proc_root)
    validate_live_pid(
        watcher_pid,
        identity["start_ticks"],
        attempt_dir,
        marker=watcher_path,
        proc_root=proc_root,
    )
    record = {
        "schema_version": 1,
        "status": "READY",
        "attempt_dir": str(attempt_dir),
        "watcher_pid": watcher_pid,
        "watcher_start_ticks": identity["start_ticks"],
        "watcher_path": str(watcher_path),
        "watcher_sha256": sha256(watcher_path),
        "helper_path": str(helper_path),
        "helper_sha256": HELPER_SHA256,
        "created_epoch_ns": created_epoch_ns or time.time_ns(),
        "process_discovery": "track /proc descendants from the receipt-bound runner; validate rank/PID from structured logs",
        "required_ranks": list(RANKS),
        "trigger": {"log_type_name": "optim_end", "step": 1},
    }
    write_json_exclusive(attempt_dir / "watcher-ready.json", record)
    return record


def verify_ready(
    attempt_dir: Path,
    *,
    proc_root: Path = Path("/proc"),
    watcher_path: Path | None = None,
    allow_launch_receipt: bool = False,
) -> dict:
    attempt_dir = attempt_dir.resolve()
    watcher_path = (watcher_path or Path(__file__)).resolve()
    ready = read_json(attempt_dir / "watcher-ready.json")
    helper_path = attempt_dir / "capture_loaded_binaries.py"
    required = {
        "status": "READY",
        "attempt_dir": str(attempt_dir),
        "watcher_path": str(watcher_path),
        "watcher_sha256": sha256(watcher_path),
        "helper_path": str(helper_path),
        "helper_sha256": HELPER_SHA256,
        "required_ranks": list(RANKS),
        "process_discovery": "track /proc descendants from the receipt-bound runner; validate rank/PID from structured logs",
        "trigger": {"log_type_name": "optim_end", "step": 1},
    }
    if any(ready.get(key) != expected for key, expected in required.items()):
        raise WatchError("READY record does not match this attempt or pinned watcher/helper")
    if sha256(helper_path) != HELPER_SHA256:
        raise WatchError("binary receipt helper SHA-256 mismatch")
    if not allow_launch_receipt and (attempt_dir / "launch-receipt.json").exists():
        raise WatchError("launch receipt already exists; attempt cannot be reused")
    validate_live_pid(
        int(ready["watcher_pid"]),
        int(ready["watcher_start_ticks"]),
        attempt_dir,
        marker=watcher_path,
        proc_root=proc_root,
    )
    return ready


def validate_launch_gate(attempt_dir: Path, gate_path: Path) -> dict:
    gate = read_json(gate_path)
    pins_path = attempt_dir / "pins.json"
    checks = {
        "status": "COORDINATOR_CONFIRMED_FOR_LAUNCH",
        "attempt_dir": str(attempt_dir.resolve()),
        "artifact_pins_sha256": sha256(pins_path),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "sanitizer_sha256": SANITIZER_SHA256,
        "compute_release": True,
        "gpu_preflight_status": "PASS",
        "gpu_count": 8,
        "output_budget_bytes": OUTPUT_BUDGET,
    }
    if any(gate.get(key) != expected for key, expected in checks.items()):
        raise WatchError("launch gate is absent, stale, or does not match the reviewed pins")
    try:
        free_bytes = int(gate["initial_storage_free_bytes"])
        checked = datetime.fromisoformat(gate["storage_checked_at_utc"].replace("Z", "+00:00"))
        gpu_checked = datetime.fromisoformat(gate["gpu_checked_at_utc"].replace("Z", "+00:00"))
        gpu_free = [int(value) for value in gate["gpu_free_bytes"]]
        gpu_uuids = [str(value) for value in gate["gpu_uuids"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise WatchError("launch gate lacks timestamped storage/GPU free records") from exc
    if any(value.tzinfo is None or value.utcoffset() is None for value in (checked, gpu_checked)):
        raise WatchError("launch gate timestamps must include UTC offset")
    if free_bytes < STORAGE_FLOOR + OUTPUT_BUDGET:
        raise WatchError("launch gate storage is below the 64 MiB budget plus 4 GiB floor")
    now = datetime.now(timezone.utc)
    if any(value.astimezone(timezone.utc) > now for value in (checked, gpu_checked)):
        raise WatchError("launch gate timestamp is in the future")
    if len(gpu_free) != 8 or min(gpu_free) < STORAGE_FLOOR:
        raise WatchError("launch gate requires at least 4 GiB free on each of eight GPUs")
    if len(gpu_uuids) != 8 or len(set(gpu_uuids)) != 8 or any(not value for value in gpu_uuids):
        raise WatchError("launch gate must identify eight unique GPUs")
    if gate.get("topology") != "single_node_nvlink":
        raise WatchError("launch gate does not confirm the selected single-node NVLink topology")
    return gate


def write_launch_receipt(
    attempt_dir: Path,
    *,
    runner_pid: int,
    launcher_path: Path,
    gate_path: Path,
    proc_root: Path = Path("/proc"),
) -> dict:
    attempt_dir = attempt_dir.resolve()
    launcher_path = launcher_path.resolve()
    ready = verify_ready(attempt_dir, proc_root=proc_root)
    gate = validate_launch_gate(attempt_dir, gate_path)
    if not launcher_path.is_file():
        raise WatchError("pinned launch script is missing")
    pins = read_json(attempt_dir / "pins.json")
    launcher_sha256 = pins.get("attempt_files", {}).get("launch.sh")
    if not isinstance(launcher_sha256, str) or sha256(launcher_path) != launcher_sha256:
        raise WatchError("launch script SHA-256 mismatch")
    identity = proc_identity(runner_pid, proc_root)
    validate_live_pid(
        runner_pid,
        identity["start_ticks"],
        attempt_dir,
        marker=launcher_path,
        proc_root=proc_root,
    )
    record = {
        "schema_version": 1,
        "attempt_dir": str(attempt_dir),
        "created_epoch_ns": time.time_ns(),
        "watcher_pid": ready["watcher_pid"],
        "watcher_start_ticks": ready["watcher_start_ticks"],
        "runner_pid": runner_pid,
        "runner_start_ticks": identity["start_ticks"],
        "launcher_path": str(launcher_path),
        "launcher_sha256": launcher_sha256,
        "launch_gate_sha256": sha256(gate_path),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
    }
    if record["created_epoch_ns"] <= int(ready["created_epoch_ns"]):
        raise WatchError("launch receipt is not newer than READY")
    write_json_exclusive(attempt_dir / "launch-receipt.json", record)
    return record


def write_launch_ack(attempt_dir: Path, ready: dict, receipt: dict) -> dict:
    receipt_path = attempt_dir / "launch-receipt.json"
    ack = {
        "schema_version": 1,
        "status": "WATCHER_ACCEPTED_LAUNCH",
        "attempt_dir": str(attempt_dir.resolve()),
        "launch_receipt_sha256": sha256(receipt_path),
        "runner_pid": receipt["runner_pid"],
        "runner_start_ticks": receipt["runner_start_ticks"],
        "watcher_pid": ready["watcher_pid"],
        "watcher_start_ticks": ready["watcher_start_ticks"],
        "accepted_at_utc": utc_now(),
    }
    write_json_exclusive(attempt_dir / "launch-ack.json", ack)
    return ack


def verify_launch_ack(attempt_dir: Path, *, proc_root: Path = Path("/proc")) -> dict:
    attempt_dir = attempt_dir.resolve()
    ready = verify_ready(attempt_dir, proc_root=proc_root, allow_launch_receipt=True)
    receipt = _verify_launch_receipt(attempt_dir, ready, proc_root)
    ack = read_json(attempt_dir / "launch-ack.json")
    if (
        ack.get("status") != "WATCHER_ACCEPTED_LAUNCH"
        or ack.get("attempt_dir") != str(attempt_dir)
        or ack.get("launch_receipt_sha256") != sha256(attempt_dir / "launch-receipt.json")
        or ack.get("runner_pid") != receipt["runner_pid"]
        or ack.get("runner_start_ticks") != receipt["runner_start_ticks"]
        or ack.get("watcher_pid") != ready["watcher_pid"]
        or ack.get("watcher_start_ticks") != ready["watcher_start_ticks"]
    ):
        raise WatchError("launch acknowledgement does not bind the live watcher and receipt")
    return ack


def wait_for_launch_ack(
    attempt_dir: Path,
    *,
    timeout_seconds: float = 15.0,
    proc_root: Path = Path("/proc"),
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> dict:
    deadline = monotonic() + timeout_seconds
    ack_path = attempt_dir / "launch-ack.json"
    while True:
        if ack_path.exists():
            return verify_launch_ack(attempt_dir, proc_root=proc_root)
        verify_ready(attempt_dir, proc_root=proc_root, allow_launch_receipt=True)
        if monotonic() >= deadline:
            raise WatchError("timed out waiting for the live launch acknowledgement")
        sleep(0.1)


def _new_rank_state() -> dict:
    return {
        "path": None,
        "inode": None,
        "offset": 0,
        "buffer": b"",
        "next_seq": 0,
        "pid": None,
        "start_ticks": None,
        "started": False,
        "optim_start": False,
        "optim_end": False,
    }


def _consume_record(
    rank: int,
    record: dict,
    state: dict,
    attempt_dir: Path,
    proc_root: Path,
) -> None:
    if record.get("rank") != rank or record.get("global_rank") != rank:
        raise WatchError(f"rank {rank} structured log identity mismatch")
    if record.get("seq_id") != state["next_seq"]:
        raise WatchError(f"rank {rank} structured log sequence gap")
    state["next_seq"] += 1
    name = record.get("log_type_name")
    level = str(record.get("log_level", "")).upper()
    if level in {"ERROR", "CRITICAL"} or (isinstance(name, str) and name.endswith("_error")):
        raise WatchError(f"rank {rank} logged {name or level} before receipt trigger")
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise WatchError(f"rank {rank} structured log has no valid PID")
    if not state["started"]:
        if state["next_seq"] != 1 or name != "structured_logger_started":
            raise WatchError(f"rank {rank} first record is not structured_logger_started")
        identity = proc_identity(pid, proc_root)
        validate_live_pid(pid, identity["start_ticks"], attempt_dir, proc_root=proc_root)
        state["pid"] = pid
        state["start_ticks"] = identity["start_ticks"]
        state["started"] = True
    else:
        if pid != state["pid"]:
            raise WatchError(f"rank {rank} PID changed within its structured log")
        validate_live_pid(pid, state["start_ticks"], attempt_dir, proc_root=proc_root)
    if name == "optim_start" and record.get("step") == 1:
        if state["optim_start"]:
            raise WatchError(f"rank {rank} duplicate optim_start for update 1")
        state["optim_start"] = True
    if name == "optim_end" and record.get("step") == 1:
        if not state["optim_start"] or state["optim_end"]:
            raise WatchError(f"rank {rank} invalid optim_end for update 1")
        state["optim_end"] = True


def scan_structured_logs(
    attempt_dir: Path,
    states: dict[int, dict],
    *,
    proc_root: Path = Path("/proc"),
) -> dict[int, dict]:
    log_dir = attempt_dir / "native/structured_logs"
    for rank in RANKS:
        state = states.setdefault(rank, _new_rank_state())
        matches = sorted(log_dir.glob(f"training.global_rank_{rank}.*.jsonl")) if log_dir.exists() else []
        if len(matches) > 1:
            raise WatchError(f"rank {rank} has more than one structured log")
        if not matches or state["optim_end"]:
            continue
        path = matches[0]
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if state["inode"] is None:
            state["path"] = str(path)
            state["inode"] = inode
        elif state["inode"] != inode or state["path"] != str(path):
            raise WatchError(f"rank {rank} structured log was replaced")
        with path.open("rb") as source:
            source.seek(state["offset"])
            chunk = source.read()
        state["offset"] += len(chunk)
        lines = (state["buffer"] + chunk).split(b"\n")
        state["buffer"] = lines.pop()
        for line in lines:
            if not line:
                raise WatchError(f"rank {rank} structured log contains a blank record")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WatchError(f"rank {rank} structured log JSON is invalid") from exc
            if not isinstance(record, dict):
                raise WatchError(f"rank {rank} structured log record is not an object")
            _consume_record(rank, record, state, attempt_dir, proc_root)
            if state["optim_end"]:
                break
        if state["pid"] is not None and not state["optim_end"]:
            validate_live_pid(state["pid"], state["start_ticks"], attempt_dir, proc_root=proc_root)
    return states


def _helper_bindings(
    attempt_dir: Path,
    states: dict[int, dict],
    *,
    proc_root: Path = Path("/proc"),
) -> list[str]:
    attempt_dir = attempt_dir.resolve()
    helper_path = attempt_dir / "capture_loaded_binaries.py"
    if sha256(helper_path) != HELPER_SHA256:
        raise WatchError("binary receipt helper SHA-256 mismatch")
    if (attempt_dir / "helper-started.json").exists():
        raise WatchError("one-shot helper was already started")
    if any(not states[rank]["optim_end"] for rank in RANKS):
        raise WatchError("helper trigger requires optim_end step 1 on all eight ranks")
    bindings = []
    pids = set()
    for rank in RANKS:
        pid = int(states[rank]["pid"])
        ticks = int(states[rank]["start_ticks"])
        if pid in pids:
            raise WatchError("duplicate training PID across ranks")
        pids.add(pid)
        validate_live_pid(pid, ticks, attempt_dir, proc_root=proc_root)
        bindings.append(f"{rank}:{pid}")
    return bindings


def start_helper_once(
    attempt_dir: Path,
    states: dict[int, dict],
    *,
    proc_root: Path = Path("/proc"),
    popen=subprocess.Popen,
    monotonic=time.monotonic,
) -> dict:
    attempt_dir = attempt_dir.resolve()
    helper_path = attempt_dir / "capture_loaded_binaries.py"
    bindings = _helper_bindings(attempt_dir, states, proc_root=proc_root)
    started = {
        "status": "STARTED_ONCE",
        "attempt_dir": str(attempt_dir),
        "helper_sha256": HELPER_SHA256,
        "rank_pid": bindings,
        "started_at_utc": utc_now(),
    }
    write_json_exclusive(attempt_dir / "helper-started.json", started)
    output_path = attempt_dir / "binary-provenance-receipt.pending"
    command = [sys.executable, str(helper_path), "--attempt-dir", str(attempt_dir)]
    for binding in bindings:
        command.extend(("--rank-pid", binding))
    command.extend(("--output", str(output_path)))
    stdout_path = attempt_dir / "helper.stdout.log"
    stderr_path = attempt_dir / "helper.stderr.log"
    stdout = stdout_path.open("xb")
    try:
        stderr = stderr_path.open("xb")
    except BaseException:
        stdout.close()
        raise
    try:
        process = popen(
            command,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
    except BaseException as exc:
        stdout.close()
        stderr.close()
        try:
            write_json_exclusive(
                attempt_dir / "helper-completion.json",
                {
                    "status": "UNAVAILABLE_START_FAILED",
                    "helper_sha256": HELPER_SHA256,
                    "reason": str(exc),
                    "finished_at_utc": utc_now(),
                },
            )
        except OSError:
            pass
        raise
    identity = None
    try:
        identity = proc_identity(int(process.pid), proc_root)
        validate_live_pid(
            int(process.pid), identity["start_ticks"], attempt_dir,
            marker=helper_path, proc_root=proc_root,
        )
        write_json_exclusive(
            attempt_dir / "helper-process.json",
            {
                "status": "RUNNING_ONCE",
                "attempt_dir": str(attempt_dir),
                "helper_pid": int(process.pid),
                "helper_start_ticks": identity["start_ticks"],
                "helper_path": str(helper_path),
                "helper_sha256": HELPER_SHA256,
                "rank_pid": bindings,
            },
        )
    except BaseException as exc:
        try:
            if identity is not None:
                stop_exact_process(
                    int(process.pid),
                    int(identity["start_ticks"]),
                    attempt_dir,
                    helper_path,
                    proc_root=proc_root,
                )
        except BaseException:
            pass
        try:
            # Popen is the direct owner of this child and guards against PID reuse.
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
        except (OSError, AttributeError, subprocess.TimeoutExpired):
            pass
        stdout.close()
        stderr.close()
        try:
            write_json_exclusive(
                attempt_dir / "helper-completion.json",
                {
                    "status": "UNAVAILABLE_PROCESS_IDENTITY",
                    "helper_sha256": HELPER_SHA256,
                    "helper_pid": int(process.pid),
                    "reason": str(exc),
                    "finished_at_utc": utc_now(),
                },
            )
        except OSError:
            pass
        raise
    return {
        "process": process,
        "pid": int(process.pid),
        "start_ticks": identity["start_ticks"],
        "helper_path": helper_path,
        "bindings": bindings,
        "deadline": monotonic() + HELPER_DEADLINE_SECONDS,
        "stdout": stdout,
        "stderr": stderr,
        "terminal": False,
    }


def _process_is_live(pid: int, start_ticks: int, proc_root: Path) -> bool:
    try:
        current = proc_identity(pid, proc_root)
    except (OSError, WatchError):
        return False
    return current["start_ticks"] == start_ticks and current["state"] != "Z"


def stop_exact_process(
    pid: int,
    start_ticks: int,
    attempt_dir: Path,
    marker: Path,
    *,
    proc_root: Path = Path("/proc"),
    send_signal=os.kill,
    monotonic=time.monotonic,
    sleep=time.sleep,
    term_grace: float = 2.0,
    kill_grace: float = 2.0,
) -> dict:
    """Boundedly stop one previously validated task process by PID/start ticks."""
    record = {"pid": int(pid), "start_ticks": int(start_ticks), "term_sent": False, "kill_sent": False}

    def validate_current(*, require_attempt_marker: bool) -> bool:
        try:
            current = proc_identity(pid, proc_root)
        except (OSError, WatchError):
            return False
        if current["start_ticks"] != start_ticks or current["state"] == "Z":
            return False
        if require_attempt_marker:
            if os.fsencode(str(attempt_dir.resolve())) not in current["cmdline"]:
                raise WatchError(f"PID {pid} does not name this attempt")
            if os.fsencode(str(marker.resolve())) not in current["cmdline"]:
                raise WatchError(f"PID {pid} does not name {marker.name}")
        return True

    if validate_current(require_attempt_marker=True):
        try:
            send_signal(pid, signal.SIGTERM)
            record["term_sent"] = True
        except ProcessLookupError:
            pass
    deadline = monotonic() + term_grace
    while validate_current(require_attempt_marker=False) and monotonic() < deadline:
        sleep(0.05)
    if validate_current(require_attempt_marker=False):
        try:
            send_signal(pid, signal.SIGKILL)
            record["kill_sent"] = True
        except ProcessLookupError:
            pass
    deadline = monotonic() + kill_grace
    while validate_current(require_attempt_marker=False) and monotonic() < deadline:
        sleep(0.05)
    if validate_current(require_attempt_marker=False):
        raise WatchError(f"PID {pid} survived bounded TERM/KILL cleanup")
    record["exit_verified"] = True
    return record


def poll_helper_once(
    job: dict,
    attempt_dir: Path,
    *,
    proc_root: Path = Path("/proc"),
    now: float | None = None,
    send_signal=os.kill,
    monotonic=time.monotonic,
    sleep=time.sleep,
    term_grace: float = 2.0,
    kill_grace: float = 2.0,
) -> bool:
    """Poll the one helper process; return when it has a final receipt."""
    if job["terminal"]:
        return True
    returncode = job["process"].poll()
    stop_record = None
    if returncode is not None:
        pending = attempt_dir / "binary-provenance-receipt.pending"
        status = "PARTIAL_OR_UNAVAILABLE"
        if returncode == 0 and pending.is_file():
            try:
                read_json(pending)
                publish_file_exclusive(pending, attempt_dir / "binary-provenance-receipt.json")
                status = "AVAILABLE"
            except (WatchError, OSError):
                status = "PARTIAL_OR_UNAVAILABLE"
    else:
        current_time = monotonic() if now is None else now
        if current_time < job["deadline"]:
            return False
        stop_record = stop_exact_process(
            job["pid"], job["start_ticks"], attempt_dir, job["helper_path"],
            proc_root=proc_root, send_signal=send_signal, monotonic=monotonic,
            sleep=sleep, term_grace=term_grace, kill_grace=kill_grace,
        )
        returncode = job["process"].poll()
        status = "UNAVAILABLE_TIMEOUT" if returncode is not None else "UNAVAILABLE_HELPER_STOP_UNVERIFIED"
    completion = {
        "status": status,
        "returncode": returncode,
        "rank_pid": job["bindings"],
        "helper_sha256": HELPER_SHA256,
        "helper_pid": job["pid"],
        "helper_start_ticks": job["start_ticks"],
        "finished_at_utc": utc_now(),
    }
    if stop_record is not None:
        completion["stop"] = stop_record
    write_json_exclusive(attempt_dir / "helper-completion.json", completion)
    for stream in (job["stdout"], job["stderr"]):
        stream.close()
    job["terminal"] = True
    if status == "UNAVAILABLE_HELPER_STOP_UNVERIFIED":
        raise WatchError("provenance helper survived bounded TERM/KILL cleanup")
    return True


class TaskProcessTracker:
    """Track runner descendants and exact compiler-entrypoint processes for this attempt."""

    def __init__(
        self,
        attempt_dir: Path,
        runner: dict,
        *,
        proc_root: Path = Path("/proc"),
        on_new=None,
        identity_reader=proc_identity,
    ):
        self.attempt_dir = attempt_dir.resolve()
        self.runner = runner
        self.proc_root = proc_root
        self.on_new = on_new
        self.identity_reader = identity_reader
        self.receipt_sha256 = sha256(self.attempt_dir / "launch-receipt.json")
        self.runner_start_ticks = int(runner["runner_start_ticks"])
        self.entrypoint_path = (self.attempt_dir / "compiler_entry.py").resolve()
        pins = read_json(self.attempt_dir / "pins.json")
        expected_entrypoint_sha = pins.get("attempt_files", {}).get("compiler_entry.py")
        if not isinstance(expected_entrypoint_sha, str) or sha256(self.entrypoint_path) != expected_entrypoint_sha:
            raise WatchError("pinned compiler entrypoint is missing or changed")
        self.entrypoint_argument = os.fsencode(str(self.entrypoint_path))
        self.records: dict[tuple[int, int], dict] = {}
        root_pid = int(runner["runner_pid"])
        root_ticks = self.runner_start_ticks
        root_record = {
            "pid": root_pid,
            "start_ticks": root_ticks,
            "ppid": None,
            "session": None,
            "process_group": None,
            "cmdline_sha256": None,
            "cmdline_has_attempt": True,
            "attempt_dir": str(self.attempt_dir),
            "launch_receipt_sha256": self.receipt_sha256,
            "ancestry": [{"pid": root_pid, "start_ticks": root_ticks}],
            "depth": 0,
            "depth_basis": "receipt_bound_runner",
            "proofs": [{"method": "receipt_bound_runner", "launch_receipt_sha256": self.receipt_sha256}],
            "runner": True,
        }
        self.records[(root_pid, root_ticks)] = root_record

    def _has_entrypoint_argument(self, cmdline: bytes) -> bool:
        return self.entrypoint_argument in cmdline.split(b"\0")

    def _stable_entrypoint_identity(self, pid: int, expected_ticks: int) -> dict | None:
        try:
            first = self.identity_reader(pid, self.proc_root)
            second = self.identity_reader(pid, self.proc_root)
        except (OSError, WatchError):
            return None
        if (
            first["start_ticks"] != expected_ticks
            or second["start_ticks"] != expected_ticks
            or expected_ticks < self.runner_start_ticks
            or first["state"] == "Z"
            or second["state"] == "Z"
            or not self._has_entrypoint_argument(first["cmdline"])
            or not self._has_entrypoint_argument(second["cmdline"])
        ):
            return None
        return second

    def _add_proof(self, record: dict, proof: dict) -> bool:
        proofs = record.setdefault("proofs", [])
        if proof in proofs:
            return False
        proofs.append(proof)
        if self.on_new is not None:
            self.on_new({"event": "identity_proof_update", **record})
        return True

    def _register_entrypoint(
        self,
        pid: int,
        start_ticks: int,
        *,
        proof: dict,
        observed: dict | None = None,
    ) -> dict | None:
        identity = self._stable_entrypoint_identity(pid, start_ticks)
        if identity is None:
            return None
        key = (int(pid), int(start_ticks))
        existing = self.records.get(key)
        if existing is not None:
            self._add_proof(existing, proof)
            return existing
        record = {
            "pid": int(pid),
            "start_ticks": int(start_ticks),
            "ppid": identity["ppid"],
            "parent_start_ticks": None,
            "session": identity["session"],
            "process_group": identity["process_group"],
            "cmdline_sha256": hashlib.sha256(identity["cmdline"]).hexdigest(),
            "cmdline_has_attempt": os.fsencode(str(self.attempt_dir)) in identity["cmdline"],
            "attempt_dir": str(self.attempt_dir),
            "launch_receipt_sha256": self.receipt_sha256,
            "ancestry": None,
            "depth": 0,
            "depth_basis": proof["method"],
            "proofs": [proof],
            "runner": False,
        }
        if observed is not None:
            record["observed_from"] = observed
        self.records[key] = record
        if self.on_new is not None:
            self.on_new(record)
        return record

    def merge_rank_states(self, states: dict[int, dict]) -> None:
        for rank in RANKS:
            state = states.get(rank, {})
            if not state.get("started") or state.get("pid") is None or state.get("start_ticks") is None:
                continue
            self._register_entrypoint(
                int(state["pid"]),
                int(state["start_ticks"]),
                proof={
                    "method": "validated_rank_log_and_exact_pinned_compiler_entrypoint_argument",
                    "rank": rank,
                },
                observed={"source": "structured_rank_log", "rank": rank},
            )

    def _recover_pinned_entrypoints(self, snapshot: dict[int, dict]) -> list[dict]:
        added = []
        for pid, stat in snapshot.items():
            if stat["state"] == "Z" or stat["start_ticks"] < self.runner_start_ticks:
                continue
            key = (int(pid), int(stat["start_ticks"]))
            was_known = key in self.records
            record = self._register_entrypoint(
                pid,
                stat["start_ticks"],
                proof={"method": "exact_pinned_compiler_entrypoint_argument", "path": str(self.entrypoint_path)},
                observed={"source": "nul_delimited_proc_cmdline_scan"},
            )
            if record is not None and not was_known:
                added.append(record)
        return added

    def can_signal_record(self, record: dict) -> bool:
        """Revalidate PID/start ticks and the original proof immediately before a signal."""
        pid = int(record["pid"])
        ticks = int(record["start_ticks"])
        try:
            first = self.identity_reader(pid, self.proc_root)
            second = self.identity_reader(pid, self.proc_root)
        except (OSError, WatchError):
            return False
        if first["start_ticks"] != ticks or second["start_ticks"] != ticks:
            return False
        if first["state"] == "Z" or second["state"] == "Z":
            return False
        if (
            record.get("attempt_dir") != str(self.attempt_dir)
            or record.get("launch_receipt_sha256") != self.receipt_sha256
        ):
            raise WatchError(f"PID {pid} lost its receipt-bound attempt identity")
        methods = {proof["method"] for proof in record.get("proofs", [])}
        exact_entrypoint_methods = {
            "exact_pinned_compiler_entrypoint_argument",
            "validated_rank_log_and_exact_pinned_compiler_entrypoint_argument",
        }
        if methods & exact_entrypoint_methods:
            if ticks < self.runner_start_ticks or not (
                self._has_entrypoint_argument(first["cmdline"])
                and self._has_entrypoint_argument(second["cmdline"])
            ):
                raise WatchError(f"PID {pid} no longer matches the pinned compiler entrypoint")
            exact_proofs = [
                proof for proof in record.get("proofs", [])
                if proof["method"] == "exact_pinned_compiler_entrypoint_argument"
            ]
            if any(proof.get("path") != str(self.entrypoint_path) for proof in exact_proofs):
                raise WatchError(f"PID {pid} has a mismatched pinned compiler entrypoint proof")
        if "receipt_bound_process_ancestry" in methods:
            ancestry = record.get("ancestry")
            if not ancestry or ancestry[0] != {
                "pid": int(self.runner["runner_pid"]),
                "start_ticks": self.runner_start_ticks,
            }:
                raise WatchError(f"PID {pid} lost its receipt-bound ancestry proof")
        if "receipt_bound_runner" in methods and pid != int(self.runner["runner_pid"]):
            raise WatchError(f"PID {pid} has an invalid runner proof")
        if "observed_child_of_tracked_identity" in methods:
            parent = record.get("observed_from", {})
            parent_key = (int(parent.get("pid", -1)), int(parent.get("start_ticks", -1)))
            parent_record = self.records.get(parent_key)
            if (
                parent_record is None
                or parent_record.get("launch_receipt_sha256") != self.receipt_sha256
                or record.get("ppid") != parent_key[0]
                or record.get("parent_start_ticks") != parent_key[1]
            ):
                raise WatchError(f"PID {pid} lost its tracked-parent proof")
        if not methods.intersection(
            exact_entrypoint_methods
            | {"receipt_bound_runner", "receipt_bound_process_ancestry", "observed_child_of_tracked_identity"}
        ):
            raise WatchError(f"PID {pid} has no supported task identity proof")
        return True

    def _snapshot(self) -> dict[int, dict]:
        result = {}
        try:
            entries = list(self.proc_root.iterdir())
        except OSError as exc:
            raise WatchError("unable to enumerate /proc while supervising task") from exc
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                raw = (entry / "stat").read_text()
            except (OSError, UnicodeDecodeError):
                continue
            close = raw.rfind(")")
            if close < 0:
                continue
            fields = raw[close + 2 :].split()
            if len(fields) <= 19:
                continue
            try:
                result[pid] = {
                    "pid": pid,
                    "state": fields[0],
                    "ppid": int(fields[1]),
                    "process_group": int(fields[2]),
                    "session": int(fields[3]),
                    "start_ticks": int(fields[19]),
                }
            except ValueError:
                continue
        return result

    def refresh(self) -> list[dict]:
        snapshot = self._snapshot()
        by_parent: dict[int, list[dict]] = {}
        for record in snapshot.values():
            by_parent.setdefault(record["ppid"], []).append(record)
        live_known: dict[int, dict] = {}
        for (pid, ticks), stored in self.records.items():
            current = snapshot.get(pid)
            if current is not None and current["start_ticks"] == ticks and current["state"] != "Z":
                live_known[pid] = stored
        root_pid = int(self.runner["runner_pid"])
        if root_pid in live_known:
            validate_live_pid(
                root_pid,
                int(self.runner["runner_start_ticks"]),
                self.attempt_dir,
                marker=Path(self.runner["launcher_path"]),
                proc_root=self.proc_root,
            )
        recovered = self._recover_pinned_entrypoints(snapshot)
        added = list(recovered)
        queue = list(live_known.values()) + recovered
        visited = set(live_known)
        visited.update(int(record["pid"]) for record in recovered)
        while queue:
            parent = queue.pop()
            parent_pid = int(parent["pid"])
            parent_ticks = int(parent["start_ticks"])
            current_parent = snapshot.get(parent_pid)
            if current_parent is None or current_parent["start_ticks"] != parent_ticks:
                continue
            for stat in by_parent.get(parent_pid, []):
                child_pid = int(stat["pid"])
                child_ticks = int(stat["start_ticks"])
                if child_pid in visited or stat["state"] == "Z":
                    continue
                try:
                    child = proc_identity(child_pid, self.proc_root)
                    confirm = proc_identity(child_pid, self.proc_root)
                except (OSError, WatchError):
                    continue
                if (
                    child["start_ticks"] != child_ticks
                    or confirm["start_ticks"] != child_ticks
                    or child["ppid"] != parent_pid
                    or confirm["ppid"] != parent_pid
                ):
                    continue
                key = (child_pid, child_ticks)
                ancestry = None
                observed_from = None
                if parent["ancestry"] is not None:
                    ancestry = parent["ancestry"] + [{"pid": child_pid, "start_ticks": child_ticks}]
                    proof = {
                        "method": "receipt_bound_process_ancestry",
                        "parent_pid": parent_pid,
                        "parent_start_ticks": parent_ticks,
                        "launch_receipt_sha256": self.receipt_sha256,
                    }
                    depth = len(parent["ancestry"])
                    depth_basis = "receipt_bound_process_ancestry"
                else:
                    observed_from = {
                        "pid": parent_pid,
                        "start_ticks": parent_ticks,
                        "launch_receipt_sha256": self.receipt_sha256,
                    }
                    proof = {
                        "method": "observed_child_of_tracked_identity",
                        "parent_pid": parent_pid,
                        "parent_start_ticks": parent_ticks,
                        "launch_receipt_sha256": self.receipt_sha256,
                    }
                    depth = 0
                    depth_basis = "observed_child_of_tracked_identity"
                record = {
                    "pid": child_pid,
                    "start_ticks": child_ticks,
                    "ppid": parent_pid,
                    "parent_start_ticks": parent_ticks,
                    "session": child["session"],
                    "process_group": child["process_group"],
                    "cmdline_sha256": hashlib.sha256(child["cmdline"]).hexdigest(),
                    "cmdline_has_attempt": os.fsencode(str(self.attempt_dir)) in child["cmdline"],
                    "attempt_dir": str(self.attempt_dir),
                    "launch_receipt_sha256": self.receipt_sha256,
                    "ancestry": ancestry,
                    "depth": depth,
                    "depth_basis": depth_basis,
                    "proofs": [proof],
                    "runner": False,
                }
                if observed_from is not None:
                    record["observed_from"] = observed_from
                self.records[key] = record
                added.append(record)
                visited.add(child_pid)
                queue.append(record)
                if self.on_new is not None:
                    self.on_new(record)
        return added

    def live_records(self) -> list[dict]:
        return [
            record for record in self.records.values()
            if _process_is_live(int(record["pid"]), int(record["start_ticks"]), self.proc_root)
        ]


def stop_task_processes(
    tracker: TaskProcessTracker,
    *,
    send_signal=os.kill,
    monotonic=time.monotonic,
    sleep=time.sleep,
    term_grace: float = 5.0,
    kill_grace: float = 3.0,
) -> dict:
    """Stop tracked exact identities, deepest first, with bounded verified exit."""
    tracker.refresh()
    actions: dict[tuple[int, int], dict] = {}

    def term_new_identities() -> None:
        live_now = tracker.live_records()
        live_now.sort(key=lambda record: int(record.get("depth", 0)), reverse=True)
        for record in live_now:
            pid = int(record["pid"])
            ticks = int(record["start_ticks"])
            key = (pid, ticks)
            if key in actions and actions[key]["term_sent"]:
                continue
            action = actions.setdefault(key, {**record, "term_sent": False, "kill_sent": False})
            if not tracker.can_signal_record(record):
                continue
            try:
                send_signal(pid, signal.SIGTERM)
                action["term_sent"] = True
            except ProcessLookupError:
                pass

    def kill_live_identities() -> None:
        live_now = tracker.live_records()
        live_now.sort(key=lambda record: int(record.get("depth", 0)), reverse=True)
        for record in live_now:
            pid = int(record["pid"])
            ticks = int(record["start_ticks"])
            key = (pid, ticks)
            action = actions.setdefault(key, {**record, "term_sent": False, "kill_sent": False})
            if action["kill_sent"] or not tracker.can_signal_record(record):
                continue
            try:
                send_signal(pid, signal.SIGKILL)
                action["kill_sent"] = True
            except ProcessLookupError:
                pass

    term_new_identities()
    deadline = monotonic() + term_grace
    while monotonic() < deadline:
        tracker.refresh()
        term_new_identities()
        if not tracker.live_records():
            break
        sleep(0.05)
    kill_live_identities()
    deadline = monotonic() + kill_grace
    while monotonic() < deadline:
        tracker.refresh()
        term_new_identities()
        kill_live_identities()
        if not tracker.live_records():
            break
        sleep(0.05)
    # This final full tracker refresh includes the narrow exact-argv recovery
    # scan. Newly discovered identities receive TERM then KILL before any exit
    # claim; failure to observe their exit remains a failed cleanup.
    tracker.refresh()
    term_new_identities()
    kill_live_identities()
    final_deadline = monotonic() + kill_grace
    while tracker.live_records() and monotonic() < final_deadline:
        tracker.refresh()
        term_new_identities()
        kill_live_identities()
        if tracker.live_records():
            sleep(0.05)
    tracker.refresh()
    still_live = [
        f"{record['pid']}:{record['start_ticks']}" for record in tracker.live_records()
    ]
    if still_live:
        raise WatchError("task processes survived bounded TERM/KILL cleanup: " + ",".join(still_live))
    for action in actions.values():
        action["exit_verified"] = True
    return {
        "attempt_dir": str(tracker.attempt_dir),
        "launch_receipt_sha256": tracker.receipt_sha256,
        "actions": list(actions.values()),
        "verified_exit": True,
    }


def append_jsonl(path: Path, record: dict) -> None:
    payload = (json.dumps(record, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        with os.fdopen(fd, "ab") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def storage_guard_tick(
    free_bytes: int,
    *,
    attempt_dir: Path,
    runner: dict,
    states: dict[int, dict],
    write_watch,
    write_stop,
    stop,
    clock=None,
) -> str | None:
    sample = {
        "time_utc": clock() if clock is not None else utc_now(),
        "free_bytes": int(free_bytes),
        "floor_bytes": STORAGE_FLOOR,
    }
    try:
        write_watch(sample)
    except OSError:
        stop_record = stop()
        try:
            write_stop({"reason": "storage_watch_write_failure", **sample, "stop": stop_record})
        except OSError:
            pass
        return "storage_watch_write_failure"
    if free_bytes < STORAGE_FLOOR:
        stop_record = stop()
        write_stop({"reason": "free_storage_below_floor", **sample, "stop": stop_record})
        return "free_storage_below_floor"
    return None


def _verify_launch_receipt(attempt_dir: Path, ready: dict, proc_root: Path) -> dict:
    receipt = read_json(attempt_dir / "launch-receipt.json")
    if receipt.get("attempt_dir") != str(attempt_dir.resolve()):
        raise WatchError("launch receipt attempt mismatch")
    if receipt.get("watcher_pid") != ready["watcher_pid"] or receipt.get("watcher_start_ticks") != ready["watcher_start_ticks"]:
        raise WatchError("launch receipt does not bind this live READY watcher")
    if int(receipt.get("created_epoch_ns", 0)) <= int(ready["created_epoch_ns"]):
        raise WatchError("launch receipt is not fresh")
    launcher_path = Path(receipt["launcher_path"]).resolve()
    pins = read_json(attempt_dir / "pins.json")
    expected_launcher_sha = pins.get("attempt_files", {}).get("launch.sh")
    if (
        launcher_path != (attempt_dir / "launch.sh").resolve()
        or not isinstance(expected_launcher_sha, str)
        or receipt.get("launcher_sha256") != expected_launcher_sha
        or sha256(launcher_path) != expected_launcher_sha
    ):
        raise WatchError("launch receipt no longer matches the canonical launcher pin")
    validate_live_pid(
        int(receipt["runner_pid"]),
        int(receipt["runner_start_ticks"]),
        attempt_dir,
        marker=launcher_path,
        proc_root=proc_root,
    )
    return receipt


def _stop_helper_job(
    job: dict | None,
    attempt_dir: Path,
    *,
    status: str,
    proc_root: Path = Path("/proc"),
    send_signal=os.kill,
    monotonic=time.monotonic,
    sleep=time.sleep,
    term_grace: float = 2.0,
    kill_grace: float = 2.0,
) -> dict | None:
    if job is None or job["terminal"]:
        return None
    stop_record = stop_exact_process(
        job["pid"], job["start_ticks"], attempt_dir, job["helper_path"],
        proc_root=proc_root, send_signal=send_signal, monotonic=monotonic,
        sleep=sleep, term_grace=term_grace, kill_grace=kill_grace,
    )
    completion = {
        "status": status,
        "returncode": job["process"].poll(),
        "rank_pid": job["bindings"],
        "helper_sha256": HELPER_SHA256,
        "helper_pid": job["pid"],
        "helper_start_ticks": job["start_ticks"],
        "stop": stop_record,
        "finished_at_utc": utc_now(),
    }
    write_json_exclusive(attempt_dir / "helper-completion.json", completion)
    for stream in (job["stdout"], job["stderr"]):
        stream.close()
    job["terminal"] = True
    return stop_record


def cleanup_after_active_failure(
    attempt_dir: Path,
    error: BaseException,
    tracker: TaskProcessTracker | None,
    helper_job: dict | None = None,
    *,
    term_grace: float = 5.0,
    kill_grace: float = 3.0,
    send_signal=os.kill,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> dict:
    cleanup = None
    if tracker is not None:
        try:
            cleanup = stop_task_processes(
                tracker, term_grace=term_grace, kill_grace=kill_grace,
                send_signal=send_signal, monotonic=monotonic, sleep=sleep,
            )
        except BaseException as stop_error:
            cleanup = {"verified_exit": False, "reason": str(stop_error)}
    helper_cleanup = None
    try:
        helper_cleanup = _stop_helper_job(
            helper_job, attempt_dir, status="UNAVAILABLE_WATCHER_FAILURE",
            send_signal=send_signal, monotonic=monotonic, sleep=sleep,
            term_grace=term_grace, kill_grace=kill_grace,
        )
    except BaseException as helper_error:
        helper_cleanup = {"verified_exit": False, "reason": str(helper_error)}
    record = {
        "reason": str(error),
        "time_utc": utc_now(),
        "task_process_cleanup": cleanup,
        "helper_cleanup": helper_cleanup,
    }
    try:
        write_json_exclusive(attempt_dir / "task-stop.json", record)
    except (OSError, FileExistsError):
        pass
    try:
        write_json_exclusive(
            attempt_dir / "watcher-failure.json",
            {
                "status": "FAILED_CLOSED",
                "time_utc": utc_now(),
                "reason": str(error),
                "cleanup": cleanup,
            },
        )
    except (OSError, FileExistsError):
        pass
    return record


def run_watcher(attempt_dir: Path) -> int:
    attempt_dir = attempt_dir.resolve()
    runner = None
    tracker = None
    helper_job = None
    states: dict[int, dict] = {}
    try:
        ready = write_ready(attempt_dir)
        print(json.dumps({"status": "READY", "pid": ready["watcher_pid"], "attempt_dir": str(attempt_dir)}), flush=True)
        while not (attempt_dir / "launch-receipt.json").exists():
            time.sleep(0.2)
        runner = _verify_launch_receipt(attempt_dir, ready, Path("/proc"))
        storage_log = attempt_dir / "storage-watch.jsonl"
        for path in (storage_log, attempt_dir / "task-processes.jsonl"):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        tracker = TaskProcessTracker(
            attempt_dir,
            runner,
            on_new=lambda row: append_jsonl(attempt_dir / "task-processes.jsonl", row),
        )
        tracker.refresh()
        write_launch_ack(attempt_dir, ready, runner)
        deadline = time.monotonic() + RUN_DEADLINE_SECONDS
        helper_done = False
        last_storage = 0.0
        last_process_scan = 0.0
        last_log_scan = 0.0
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise WatchError("20-minute attempt deadline expired")
            if now - last_storage >= 3.0:
                stop = lambda: stop_task_processes(tracker)
                outcome = storage_guard_tick(
                    shutil.disk_usage(attempt_dir).free,
                    attempt_dir=attempt_dir,
                    runner=runner,
                    states=states,
                    write_watch=lambda row: append_jsonl(storage_log, row),
                    write_stop=lambda row: write_json_exclusive(attempt_dir / "storage-stop.json", row),
                    stop=stop,
                )
                last_storage = now
                if outcome:
                    _stop_helper_job(helper_job, attempt_dir, status="UNAVAILABLE_TASK_STOPPED")
                    return 2
            if now - last_process_scan >= PROCESS_SCAN_SECONDS:
                tracker.refresh()
                last_process_scan = now
            if not helper_done:
                if now - last_log_scan >= PROCESS_SCAN_SECONDS:
                    scan_structured_logs(attempt_dir, states)
                    tracker.merge_rank_states(states)
                    last_log_scan = now
                complete = all(states.get(rank, {}).get("optim_end") for rank in RANKS)
                if complete and helper_job is None:
                    process_receipt = {
                        "status": "ALL_RANKS_OPTIM_END_1",
                        "rank_pid": [f"{rank}:{states[rank]['pid']}" for rank in RANKS],
                        "rank_start_ticks": [states[rank]["start_ticks"] for rank in RANKS],
                        "logs_contiguous_from_seq0": True,
                        "validated_at_utc": utc_now(),
                    }
                    write_json_exclusive(attempt_dir / "process-discovery.json", process_receipt)
                    helper_job = start_helper_once(attempt_dir, states)
                if helper_job is not None:
                    helper_done = poll_helper_once(helper_job, attempt_dir)
            runner_live = _process_is_live(int(runner["runner_pid"]), int(runner["runner_start_ticks"]), Path("/proc"))
            if not runner_live:
                tracker.refresh()
                remaining = tracker.live_records()
                if helper_job is not None and not helper_done:
                    _stop_helper_job(helper_job, attempt_dir, status="UNAVAILABLE_RUNNER_EXITED")
                    helper_done = True
                if not all(states.get(rank, {}).get("optim_end") for rank in RANKS):
                    raise WatchError("runner exited before all eight optim_end step 1 records")
                if remaining:
                    cleanup = stop_task_processes(tracker)
                    write_json_exclusive(
                        attempt_dir / "watcher-failure.json",
                        {
                            "status": "FAILED_CLOSED",
                            "time_utc": utc_now(),
                            "reason": "runner exited while tracked descendants remained",
                            "cleanup": cleanup,
                        },
                    )
                    return 1
                if helper_done:
                    write_json_exclusive(
                        attempt_dir / "watcher-complete.json",
                        {
                            "status": "RUNNER_EXITED_AFTER_ONE_SHOT_RECEIPT",
                            "time_utc": utc_now(),
                            "launch_ack_sha256": sha256(attempt_dir / "launch-ack.json"),
                            "launch_receipt_sha256": tracker.receipt_sha256,
                        },
                    )
                    return 0
            time.sleep(0.2)
    except BaseException as exc:
        cleanup_after_active_failure(attempt_dir, exc, tracker, helper_job)
        print(f"watcher failed closed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "verify-ready"):
        child = subparsers.add_parser(command)
        child.add_argument("--attempt-dir", type=Path, default=DEFAULT_ATTEMPT)
    wait_ack = subparsers.add_parser("wait-ack")
    wait_ack.add_argument("--attempt-dir", type=Path, default=DEFAULT_ATTEMPT)
    wait_ack.add_argument("--timeout-seconds", type=float, default=15.0)
    launch = subparsers.add_parser("launch-receipt")
    launch.add_argument("--attempt-dir", type=Path, default=DEFAULT_ATTEMPT)
    launch.add_argument("--runner-pid", type=int, required=True)
    launch.add_argument("--launcher-path", type=Path, required=True)
    launch.add_argument("--gate-path", type=Path, required=True)
    args = parser.parse_args()
    attempt_dir = args.attempt_dir.resolve()
    if attempt_dir != DEFAULT_ATTEMPT.resolve():
        parser.error("CLI attempt path must be the pinned SANITIZER-ORIGIN/attempts/001 directory")
    if args.command == "run":
        return run_watcher(attempt_dir)
    if args.command == "verify-ready":
        try:
            ready = verify_ready(attempt_dir)
            print(json.dumps({"status": "READY_LIVE", "pid": ready["watcher_pid"], "attempt_dir": str(attempt_dir)}))
            return 0
        except WatchError as exc:
            print(f"READY gate failed: {exc}", file=sys.stderr)
            return 1
    if args.command == "wait-ack":
        try:
            ack = wait_for_launch_ack(attempt_dir, timeout_seconds=args.timeout_seconds)
            print(json.dumps({"status": ack["status"], "watcher_pid": ack["watcher_pid"]}))
            return 0
        except (WatchError, OSError, KeyError, TypeError, ValueError) as exc:
            print(f"launch acknowledgement failed closed: {exc}", file=sys.stderr)
            return 1
    try:
        receipt = write_launch_receipt(
            attempt_dir,
            runner_pid=args.runner_pid,
            launcher_path=args.launcher_path,
            gate_path=args.gate_path,
        )
        print(json.dumps({"status": "LAUNCH_RECEIPT_WRITTEN", "runner_pid": receipt["runner_pid"]}))
        return 0
    except (WatchError, OSError) as exc:
        print(f"launch gate failed closed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
