#!/usr/bin/env python3
"""Own and release the descendants of one native numerical-probe phase."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


PROC = Path("/proc")
PR_SET_CHILD_SUBREAPER = 36
POLL_SECONDS = 0.25


class OwnershipError(RuntimeError):
    pass


def proc_identity(pid: int) -> dict[str, Any] | None:
    try:
        raw = (PROC / str(pid) / "stat").read_text(encoding="ascii")
        fields = raw[raw.rfind(")") + 2 :].split()
        if len(fields) < 20:
            return None
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except (OSError, UnicodeError, ValueError):
        return None


def proc_command(pid: int) -> list[bytes]:
    try:
        raw = (PROC / str(pid) / "cmdline").read_bytes()
        return [part for part in raw.split(b"\0") if part]
    except OSError:
        return []


def process_rss_bytes(pid: int, start_ticks: int) -> int | None:
    identity = proc_identity(pid)
    if identity is None or identity["start_ticks"] != start_ticks:
        return None
    try:
        for line in (PROC / str(pid) / "status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return 0


def read_records(path: Path) -> list[dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                records[(int(item["pid"]), int(item["start_ticks"]))] = item
    return list(records.values())


def verify_ledger(path: Path) -> tuple[bool, list[str]]:
    remaining = []
    for record in read_records(path):
        current = proc_identity(int(record["pid"]))
        if current is not None and current["start_ticks"] == int(record["start_ticks"]):
            remaining.append(f"{record['pid']}:{record['start_ticks']}:{current['state']}")
    return not remaining, remaining


def rss_from_ledger(path: Path) -> tuple[int, int]:
    total = 0
    counted = 0
    for record in read_records(path):
        value = process_rss_bytes(int(record["pid"]), int(record["start_ticks"]))
        if value is not None:
            total += value
            counted += 1
    return total, counted


class PhaseOwner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pid = os.getpid()
        self.parent_pid = args.parent_pid
        self.parent_ticks = args.parent_start_ticks
        self.entrypoint = Path(args.entrypoint).resolve(strict=True)
        self.cwd = Path(args.cwd).resolve(strict=True)
        self.ledger = Path(args.ledger)
        self.status = Path(args.status)
        self.owner_record_path = Path(args.owner_record)
        self.log_path = Path(args.log)
        self.started_ticks = 0
        self.records: dict[tuple[int, int], dict[str, Any]] = {}
        self.pidfds: dict[tuple[int, int], int] = {}
        self.child: subprocess.Popen[bytes] | None = None
        self.stop_reason: str | None = None
        self.phase_started = time.monotonic()
        self.deadline = self.phase_started + args.deadline_seconds
        self.term_sent: set[tuple[int, int]] = set()
        self.kill_sent: set[tuple[int, int]] = set()

    def _sync_json(self, path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{self.pid}.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _append_record(self, record: dict[str, Any]) -> None:
        key = (int(record["pid"]), int(record["start_ticks"]))
        if key in self.records:
            return
        self.records[key] = record
        try:
            fd = os.pidfd_open(key[0], 0)
        except ProcessLookupError:
            self.records.pop(key, None)
            return
        try:
            current = proc_identity(key[0])
            if current is None or current["start_ticks"] != key[1]:
                os.close(fd)
                self.records.pop(key, None)
                return
            self.pidfds[key] = fd
            with self.ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            os.close(fd)
            self.pidfds.pop(key, None)
            self.records.pop(key, None)
            raise

    def _register(
        self, pid: int, expected_parent: int, expected_parent_ticks: int, proof: str
    ) -> bool:
        first = proc_identity(pid)
        if first is None or first["state"] == "Z" or first["ppid"] != expected_parent:
            return False
        second = proc_identity(pid)
        if (
            second is None
            or second["state"] == "Z"
            or second["ppid"] != expected_parent
            or second["start_ticks"] != first["start_ticks"]
        ):
            return False
        key = (pid, second["start_ticks"])
        if key in self.records:
            return False
        parent_record = (
            {"ancestry": [{"pid": self.pid, "start_ticks": self.started_ticks}]}
            if expected_parent == self.pid
            else self.records.get((expected_parent, expected_parent_ticks))
        )
        if parent_record is None:
            raise OwnershipError(f"unrecorded parent PID {expected_parent} cannot prove ancestry")
        ancestry = list(parent_record["ancestry"]) + [
            {"pid": pid, "start_ticks": second["start_ticks"]}
        ]
        record = {
            "phase": self.args.phase,
            "pid": pid,
            "start_ticks": second["start_ticks"],
            "ppid_at_discovery": expected_parent,
            "pgid": second["pgid"],
            "sid": second["sid"],
            "ancestry": ancestry,
            "proof": proof,
            "cmdline_sha256_at_discovery": hashlib.sha256(
                b"\0".join(proc_command(pid))
            ).hexdigest(),
            "discovered_at_unix_ns": time.time_ns(),
        }
        self._append_record(record)
        return True

    def _register_recovered_entrypoints(self) -> None:
        expected = os.fsencode(str(self.entrypoint))
        owner_script = os.fsencode(str(Path(__file__).resolve()))
        for entry in PROC.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == self.pid:
                continue
            identity = proc_identity(pid)
            if identity is None or identity["state"] == "Z" or identity["start_ticks"] < self.started_ticks:
                continue
            key = (pid, identity["start_ticks"])
            if key in self.records:
                continue
            command = proc_command(pid)
            if expected not in command or owner_script in command:
                continue
            try:
                process_cwd = Path(os.readlink(PROC / str(pid) / "cwd")).resolve()
            except OSError:
                continue
            if process_cwd != self.cwd:
                continue
            confirm = proc_identity(pid)
            if (
                confirm is None
                or confirm["state"] == "Z"
                or confirm["start_ticks"] != identity["start_ticks"]
            ):
                continue
            record = {
                "phase": self.args.phase,
                "pid": pid,
                "start_ticks": identity["start_ticks"],
                "ppid_at_discovery": identity["ppid"],
                "pgid": identity["pgid"],
                "sid": identity["sid"],
                "ancestry": [{"pid": self.pid, "start_ticks": self.started_ticks},
                             {"pid": pid, "start_ticks": identity["start_ticks"]}],
                "proof": "exact_phase_entrypoint_argv_cwd_and_bounded_start_tick_recovery",
                "entrypoint": str(self.entrypoint),
                "cwd": str(self.cwd),
                "cmdline_sha256_at_discovery": hashlib.sha256(b"\0".join(command)).hexdigest(),
                "discovered_at_unix_ns": time.time_ns(),
            }
            self._append_record(record)

    def scan(self) -> None:
        snapshot: dict[int, dict[str, Any]] = {}
        try:
            entries = list(PROC.iterdir())
        except OSError as error:
            raise OwnershipError(f"cannot enumerate /proc: {error}") from error
        for entry in entries:
            if entry.name.isdigit():
                identity = proc_identity(int(entry.name))
                if identity is not None:
                    snapshot[identity["pid"]] = identity
        parents = {pid: [] for pid in snapshot}
        for identity in snapshot.values():
            parents.setdefault(identity["ppid"], []).append(identity["pid"])
        queue = [(self.pid, self.started_ticks), *self.records]
        visited: set[tuple[int, int]] = set()
        while queue:
            parent, parent_ticks = queue.pop()
            parent_key = (parent, parent_ticks)
            if parent_key in visited:
                continue
            current_parent = snapshot.get(parent)
            if (
                current_parent is None
                or current_parent["start_ticks"] != parent_ticks
                or current_parent["state"] == "Z"
            ):
                continue
            visited.add(parent_key)
            for child_pid in parents.get(parent, []):
                identity = snapshot[child_pid]
                key = (child_pid, identity["start_ticks"])
                if key in visited:
                    continue
                if key not in self.records:
                    if parent == self.pid:
                        proof = (
                            "phase_owner_direct_child"
                            if self.child is not None and child_pid == self.child.pid
                            else "subreaper_adopted_descendant"
                        )
                    else:
                        proof = "recorded_descendant_ancestry"
                    if self._register(
                        child_pid,
                        parent,
                        parent_ticks,
                        proof,
                    ):
                        queue.append(key)
                else:
                    queue.append(key)
        self._register_recovered_entrypoints()
        self._reap_orphan_zombies(snapshot)

    def _reap_orphan_zombies(self, snapshot: dict[int, dict[str, Any]]) -> None:
        root_pid = self.child.pid if self.child is not None else -1
        for (pid, ticks), _record in list(self.records.items()):
            if pid == root_pid:
                continue
            identity = snapshot.get(pid)
            if (
                identity is not None
                and identity["start_ticks"] == ticks
                and identity["state"] == "Z"
                and identity["ppid"] == self.pid
            ):
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass

    def _live_records(self) -> list[dict[str, Any]]:
        live = []
        for key, record in self.records.items():
            current = proc_identity(key[0])
            if current is not None and current["start_ticks"] == key[1]:
                live.append(record)
        return live

    def _signal_live(self, sig: int) -> None:
        live = sorted(self._live_records(), key=lambda item: len(item["ancestry"]), reverse=True)
        sent = self.term_sent if sig == signal.SIGTERM else self.kill_sent
        for record in live:
            key = (int(record["pid"]), int(record["start_ticks"]))
            if key in sent:
                continue
            fd = self.pidfds.get(key)
            if fd is None:
                raise OwnershipError(f"owned PID {key[0]} has no retained pidfd")
            current = proc_identity(key[0])
            if current is None or current["start_ticks"] != key[1] or current["state"] == "Z":
                continue
            try:
                signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                continue
            sent.add(key)

    def _wait_for_release(self, duration: float, sig: int) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.scan()
            self._signal_live(sig)
            self._reap_child()
            if not self._live_records():
                return
            time.sleep(POLL_SECONDS)

    def _reap_child(self) -> int | None:
        if self.child is None:
            return None
        return self.child.poll()

    def release(self, reason: str) -> tuple[bool, list[str]]:
        release_error = None
        try:
            self.scan()
            self._signal_live(signal.SIGTERM)
            self._wait_for_release(self.args.term_grace, signal.SIGTERM)
            self._signal_live(signal.SIGKILL)
            self._wait_for_release(self.args.kill_grace, signal.SIGKILL)
            self.scan()
        except BaseException as error:
            release_error = f"{type(error).__name__}: {error}"
        absent, remaining = verify_ledger(self.ledger)
        if release_error:
            remaining.append(f"release_error={release_error}")
            absent = False
        return absent, remaining

    def _parent_is_live(self) -> bool:
        current = proc_identity(self.parent_pid)
        return (
            current is not None
            and current["state"] != "Z"
            and current["start_ticks"] == self.parent_ticks
        )

    def _write_status(self, exit_code: int, reason: str, absent: bool, remaining: list[str]) -> None:
        result = {
            "phase": self.args.phase,
            "owner_pid": self.pid,
            "owner_start_ticks": self.started_ticks,
            "parent_pid": self.parent_pid,
            "parent_start_ticks": self.parent_ticks,
            "deadline_seconds": self.args.deadline_seconds,
            "reason": reason,
            "exit_code": exit_code,
            "all_recorded_identities_absent": absent,
            "remaining_identities": remaining,
            "record_count": len(self.records),
            "finished_at_unix_ns": time.time_ns(),
        }
        self._sync_json(self.status, result)

    def run(self) -> int:
        if os.getppid() != self.parent_pid or not self._parent_is_live():
            raise OwnershipError("launch controller identity changed before phase owner startup")
        os.setsid()
        result = ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
        if result != 0:
            errno = ctypes.get_errno()
            raise OwnershipError(f"PR_SET_CHILD_SUBREAPER failed with errno {errno}")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise OwnershipError("Python pidfd support is required for exact task signalling")
        owner_identity = proc_identity(self.pid)
        if owner_identity is None or owner_identity["state"] == "Z":
            raise OwnershipError("phase owner identity disappeared during startup")
        self.started_ticks = owner_identity["start_ticks"]
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.touch(mode=0o600, exist_ok=False)
        self._sync_json(self.owner_record_path, {
            "phase": self.args.phase,
            "pid": self.pid,
            "start_ticks": self.started_ticks,
            "parent_pid": self.parent_pid,
            "parent_start_ticks": self.parent_ticks,
            "session": os.getsid(self.pid),
            "started_at_unix_ns": time.time_ns(),
        })

        if self.stop_reason or not self._parent_is_live():
            reason = "controller_signal" if self.stop_reason else "controller_lost"
            code = 143 if self.stop_reason else 75
            self._write_status(code, reason, True, [])
            return code
        with self.log_path.open("ab", buffering=0) as log_stream:
            self.child = subprocess.Popen(
                self.args.command,
                cwd=self.cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
        launch_status = 0
        reason = "natural_exit"
        try:
            registered = self._register(
                self.child.pid, self.pid, self.started_ticks, "phase_owner_direct_child"
            )
        except BaseException as error:
            registered = False
            self.stop_reason = f"initial_registration_error:{type(error).__name__}"
        if not registered:
            reason = "initial_registration_failure"
            launch_status = 70
            absent, remaining = self.release(reason)
            if not absent:
                launch_status = 70
            self._reap_child()
            self._write_status(launch_status, reason, absent, remaining)
            for fd in self.pidfds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            return launch_status
        try:
            while True:
                if self.stop_reason:
                    reason = self.stop_reason
                    launch_status = 143 if reason == "controller_signal" else 124 if reason == "deadline" else 75
                    break
                if not self._parent_is_live():
                    reason = "controller_lost"
                    launch_status = 75
                    break
                if time.monotonic() >= self.deadline:
                    reason = "deadline"
                    launch_status = 124
                    break
                self.scan()
                child_status = self._reap_child()
                if child_status is not None:
                    self.scan()
                    if self._live_records():
                        reason = "orphaned_descendants_after_torchrun_exit"
                        launch_status = 70
                    else:
                        launch_status = child_status
                    break
                time.sleep(POLL_SECONDS)
        except BaseException as error:
            reason = f"owner_error:{type(error).__name__}"
            launch_status = 70
            self.stop_reason = str(error)

        if launch_status != 0 or self._live_records():
            absent, remaining = self.release(reason)
            if not absent:
                launch_status = 70
            elif launch_status == 0:
                launch_status = 70
        else:
            absent, remaining = verify_ledger(self.ledger)
            if not absent:
                absent, remaining = self.release("identity_verification_failed")
                launch_status = 70

        child_status = self._reap_child()
        if child_status is not None and launch_status == 0:
            launch_status = child_status
        self._write_status(launch_status, reason, absent, remaining)
        for fd in self.pidfds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        return launch_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-start-ticks", type=int, required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--owner-record", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--deadline-seconds", type=float, default=1800.0)
    parser.add_argument("--term-grace", type=float, default=8.0)
    parser.add_argument("--kill-grace", type=float, default=8.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a native phase command is required after --")
    if args.deadline_seconds <= 0 or args.term_grace < 0 or args.kill_grace < 0:
        parser.error("deadline must be positive and cleanup grace must be nonnegative")
    return args


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] in {"verify", "rss"}:
        path = Path(sys.argv[2])
        if sys.argv[1] == "verify":
            absent, remaining = verify_ledger(path)
            print(json.dumps({"all_recorded_identities_absent": absent, "remaining": remaining}))
            return 0 if absent else 1
        total, counted = rss_from_ledger(path)
        print(f"{total}\t{counted}")
        return 0
    args = parse_args()
    owner = PhaseOwner(args)
    signal.signal(signal.SIGTERM, lambda *_: setattr(owner, "stop_reason", "controller_signal"))
    signal.signal(signal.SIGINT, lambda *_: setattr(owner, "stop_reason", "controller_signal"))
    signal.signal(signal.SIGHUP, lambda *_: setattr(owner, "stop_reason", "controller_signal"))
    try:
        return owner.run()
    except BaseException as error:
        payload = {
            "phase": args.phase,
            "owner_pid": os.getpid(),
            "reason": f"startup_error:{type(error).__name__}",
            "error": str(error),
            "exit_code": 70,
            "all_recorded_identities_absent": False,
            "remaining_identities": ["startup_ownership_unverified"],
            "finished_at_unix_ns": time.time_ns(),
        }
        temporary = Path(args.status).with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.status)
        print(f"phase owner refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
