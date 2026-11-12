#!/usr/bin/env python3
"""Mock-only checks for the SANITIZER-ORIGIN READY and one-shot watcher."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import shutil
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import watch_binary_provenance as watch


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_atomic_receipt_publication_never_exposes_partial_json() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "receipt.json"
        entered = threading.Event()
        release = threading.Event()
        original_link = watch.os.link
        errors = []

        def delayed_link(source, target):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release atomic publication")
            return original_link(source, target)

        def writer():
            try:
                watch.write_json_exclusive(path, {"status": "COMPLETE", "payload": "x" * 1000})
            except BaseException as exc:
                errors.append(exc)

        watch.os.link = delayed_link
        thread = threading.Thread(target=writer)
        thread.start()
        try:
            check(entered.wait(2), "receipt writer did not reach the atomic publish point")
            check(not path.exists(), "reader could observe a receipt before atomic publication")
        finally:
            release.set()
            thread.join(timeout=3)
            watch.os.link = original_link
        check(not thread.is_alive(), "atomic writer did not finish")
        check(not errors, f"atomic receipt writer failed: {errors!r}")
        check(watch.read_json(path)["status"] == "COMPLETE", "published JSON receipt was incomplete")
        try:
            watch.write_json_exclusive(path, {"status": "OVERWRITE"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic receipt publication overwrote an existing path")


class FakeProc:
    def __init__(self, root: Path):
        self.root = root
        self.signals: list[tuple[int, int]] = []

    def add(
        self, pid: int, ticks: int, attempt: Path, marker: Path, *,
        ppid: int = 1, process_group: int = 1, session: int = 1, state: str = "S",
    ) -> None:
        directory = self.root / str(pid)
        directory.mkdir(parents=True, exist_ok=True)
        rest = [state, str(ppid), str(process_group), str(session)] + ["0"] * 15 + [str(ticks)]
        (directory / "stat").write_text(f"{pid} (fixture process) " + " ".join(rest))
        args = [b"python", os.fsencode(marker), os.fsencode(attempt)]
        (directory / "cmdline").write_bytes(b"\0".join(args) + b"\0")

    def remove(self, pid: int) -> None:
        shutil.rmtree(self.root / str(pid), ignore_errors=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root: Path) -> tuple[Path, FakeProc, dict]:
    attempt = root / "attempt"
    attempt.mkdir()
    watcher_path = Path(watch.__file__).resolve()
    helper_source = Path(watch.__file__).resolve().with_name("capture_loaded_binaries.py")
    attempt_source = Path(watch.__file__).resolve().parent
    shutil.copyfile(helper_source, attempt / "capture_loaded_binaries.py")
    shutil.copyfile(attempt_source / "compiler_entry.py", attempt / "compiler_entry.py")
    shutil.copyfile(attempt_source / "pins.json", attempt / "pins.json")
    shutil.copyfile(watcher_path, attempt / watcher_path.name)
    shutil.copyfile(watcher_path.with_name("launch.sh"), attempt / "launch.sh")
    proc = FakeProc(root / "proc")
    proc.root.mkdir()
    proc.add(501, 1001, attempt, watcher_path)
    ready = watch.write_ready(
        attempt,
        watcher_pid=501,
        proc_root=proc.root,
        watcher_path=watcher_path,
        created_epoch_ns=100,
    )
    return attempt, proc, ready


def add_rank_logs(attempt: Path, proc: FakeProc, ranks=range(8), *, gap_rank=None, error_rank=None) -> dict:
    log_dir = attempt / "native/structured_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    states = {}
    for rank in ranks:
        pid = 700 + rank
        proc.add(pid, 2000 + rank, attempt, attempt / "compiler_entry.py")
        records = [
            {"rank": rank, "global_rank": rank, "pid": pid, "tid": pid, "seq_id": 0, "log_type_name": "structured_logger_started", "log_level": "INFO"},
            {"rank": rank, "global_rank": rank, "pid": pid, "tid": pid, "seq_id": 1, "step": 1, "log_type_name": "optim_start", "log_level": "INFO"},
            {"rank": rank, "global_rank": rank, "pid": pid, "tid": pid, "seq_id": 2 if rank != gap_rank else 3, "step": 1, "log_type_name": "optim_end", "log_level": "INFO"},
        ]
        if rank == error_rank:
            records.insert(1, {"rank": rank, "global_rank": rank, "pid": pid, "tid": pid, "seq_id": 1, "log_type_name": "step_error", "log_level": "ERROR"})
        path = log_dir / f"training.global_rank_{rank}.fixture.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return states


def add_fake_runner(attempt: Path, proc: FakeProc) -> tuple[dict, watch.TaskProcessTracker]:
    launcher = attempt / "launch.sh"
    proc.add(601, 1201, attempt, launcher, ppid=501)
    runner = {
        "runner_pid": 601,
        "runner_start_ticks": 1201,
        "launcher_path": str(launcher),
    }
    watch.write_json_exclusive(attempt / "launch-receipt.json", runner)
    tracker = watch.TaskProcessTracker(attempt, runner, proc_root=proc.root)
    tracker.refresh()
    return runner, tracker


def write_fixture_launch_gate(attempt: Path) -> tuple[Path, dict]:
    pins_path = attempt / "pins.json"
    now = datetime.now(timezone.utc).isoformat()
    gate = {
        "status": "COORDINATOR_CONFIRMED_FOR_LAUNCH",
        "attempt_dir": str(attempt.resolve()),
        "artifact_pins_sha256": digest(pins_path),
        "source_manifest_sha256": watch.SOURCE_MANIFEST_SHA256,
        "source_archive_sha256": watch.SOURCE_ARCHIVE_SHA256,
        "sanitizer_sha256": watch.SANITIZER_SHA256,
        "compute_release": True,
        "gpu_preflight_status": "PASS",
        "gpu_count": 8,
        "output_budget_bytes": watch.OUTPUT_BUDGET,
        "initial_storage_free_bytes": watch.STORAGE_FLOOR + watch.OUTPUT_BUDGET,
        "storage_checked_at_utc": now,
        "gpu_checked_at_utc": now,
        "gpu_free_bytes": [watch.STORAGE_FLOOR] * 8,
        "gpu_uuids": [f"GPU-{rank}" for rank in range(8)],
        "topology": "single_node_nvlink",
    }
    gate_path = attempt / "launch-gate.json"
    gate_path.write_text(json.dumps(gate))
    return gate_path, gate


def check_ready_gate_fails_closed_and_is_exclusive() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt = Path(temp) / "attempt"
        attempt.mkdir()
        try:
            watch.verify_ready(attempt, proc_root=Path(temp) / "proc")
        except watch.WatchError:
            pass
        else:
            raise AssertionError("missing READY record passed")

    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        verified = watch.verify_ready(attempt, proc_root=proc.root)
        check(verified["status"] == "READY", "live READY not accepted")
        try:
            watch.write_ready(attempt, watcher_pid=501, proc_root=proc.root)
        except FileExistsError:
            pass
        else:
            raise AssertionError("READY record was overwritten")
        proc.remove(501)
        try:
            watch.verify_ready(attempt, proc_root=proc.root)
        except watch.WatchError:
            pass
        else:
            raise AssertionError("dead watcher PID passed READY gate")


def check_launch_gate_and_receipt_are_pinned_and_exclusive() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, ready = fixture(Path(temp))
        launcher = attempt / "launch.sh"
        pins_path = attempt / "pins.json"
        pins = json.loads(pins_path.read_text())
        check("launcher_sha256" not in pins, "fixture manufactured a noncanonical top-level pin")
        check(
            pins["attempt_files"]["launch.sh"] == digest(launcher),
            "fixture does not exercise the actual attempt_files.launch.sh pin",
        )
        gate_path, gate = write_fixture_launch_gate(attempt)
        watch.validate_launch_gate(attempt, gate_path)
        proc.add(601, 1401, attempt, launcher)
        receipt = watch.write_launch_receipt(
            attempt,
            runner_pid=601,
            launcher_path=launcher,
            gate_path=gate_path,
            proc_root=proc.root,
        )
        check(receipt["watcher_pid"] == ready["watcher_pid"], "receipt lost READY watcher binding")
        try:
            watch.verify_launch_ack(attempt, proc_root=proc.root)
        except watch.WatchError:
            pass
        else:
            raise AssertionError("launch was accepted before watcher acknowledgement")
        try:
            watch.wait_for_launch_ack(attempt, timeout_seconds=0, proc_root=proc.root)
        except watch.WatchError:
            pass
        else:
            raise AssertionError("runner wait gate accepted a missing watcher acknowledgement")
        watch.write_launch_ack(attempt, ready, receipt)
        verified_ack = watch.wait_for_launch_ack(
            attempt, timeout_seconds=0, proc_root=proc.root,
        )
        check(
            verified_ack["launch_receipt_sha256"] == digest(attempt / "launch-receipt.json")
            and verified_ack["watcher_start_ticks"] == ready["watcher_start_ticks"],
            "launch acknowledgement lost its live receipt binding",
        )
        try:
            watch.write_launch_receipt(
                attempt,
                runner_pid=601,
                launcher_path=launcher,
                gate_path=gate_path,
                proc_root=proc.root,
            )
        except watch.WatchError:
            pass
        else:
            raise AssertionError("second launch receipt was accepted")

        proc.remove(501)
        try:
            watch.wait_for_launch_ack(attempt, timeout_seconds=0, proc_root=proc.root)
        except watch.WatchError:
            pass
        else:
            raise AssertionError("dead watcher acknowledgement passed")

        gate["gpu_free_bytes"][7] = watch.STORAGE_FLOOR - 1
        invalid_gate = attempt / "invalid-gate.json"
        invalid_gate.write_text(json.dumps(gate))
        try:
            watch.validate_launch_gate(attempt, invalid_gate)
        except watch.WatchError as exc:
            check("4 GiB free" in str(exc), "low NVML gate misclassified")
        else:
            raise AssertionError("low GPU free memory passed launch gate")


def check_watcher_death_before_acknowledgement_blocks_launch() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        gate_path, _gate = write_fixture_launch_gate(attempt)
        launcher = attempt / "launch.sh"
        proc.add(601, 1401, attempt, launcher, ppid=501)
        watch.write_launch_receipt(
            attempt,
            runner_pid=601,
            launcher_path=launcher,
            gate_path=gate_path,
            proc_root=proc.root,
        )
        proc.remove(501)
        try:
            watch.wait_for_launch_ack(attempt, timeout_seconds=0, proc_root=proc.root)
        except watch.WatchError:
            pass
        else:
            raise AssertionError("runner proceeded after watcher died before acknowledgement")
        check(not (attempt / "launch-ack.json").exists(), "dead watcher published an acknowledgement")


def check_all_rank_update1_starts_pinned_helper_once() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        states = add_rank_logs(attempt, proc)
        watch.scan_structured_logs(attempt, states, proc_root=proc.root)
        calls = []

        class MockProcess:
            pid = 900

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

        process = MockProcess()

        def helper(command, **_kwargs):
            calls.append(command)
            proc.add(900, 3000, attempt, attempt / "capture_loaded_binaries.py", ppid=501)
            return process

        job = watch.start_helper_once(attempt, states, proc_root=proc.root, popen=helper, monotonic=lambda: 10.0)
        check(len(calls) == 1, "helper was not called exactly once")
        command = calls[0]
        bindings = [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--rank-pid"]
        check(bindings == [f"{rank}:{700 + rank}" for rank in range(8)], "rank/PID map differs")
        check(not watch.poll_helper_once(job, attempt, proc_root=proc.root, now=11.0), "pending helper blocked or completed early")
        try:
            watch.start_helper_once(attempt, states, proc_root=proc.root, popen=helper)
        except watch.WatchError:
            pass
        else:
            raise AssertionError("second helper call was accepted")
        check(len(calls) == 1, "helper call repeated after one-shot marker")
        process.returncode = 0
        check(watch.poll_helper_once(job, attempt, proc_root=proc.root, now=12.0), "completed helper was not finalized")
        check(json.loads((attempt / "helper-completion.json").read_text())["status"] == "PARTIAL_OR_UNAVAILABLE",
              "missing provenance output was misreported as available")


def check_async_helper_keeps_storage_guard_live_and_kills_only_stalled_helper() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        states = add_rank_logs(attempt, proc)
        watch.scan_structured_logs(attempt, states, proc_root=proc.root)

        class MockProcess:
            pid = 901

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode if not (proc.root / str(self.pid)).exists() else None

        process = MockProcess()
        proc.add(901, 3101, attempt, attempt / "capture_loaded_binaries.py", ppid=501, session=901)
        job = watch.start_helper_once(
            attempt, states, proc_root=proc.root,
            popen=lambda _command, **_kwargs: process,
            monotonic=lambda: 0.0,
        )
        check(not watch.poll_helper_once(job, attempt, proc_root=proc.root, now=1.0),
              "nonreturning helper was treated as complete")
        samples = []
        signalled = []
        outcome = watch.storage_guard_tick(
            watch.STORAGE_FLOOR - 1,
            attempt_dir=attempt,
            runner={"runner_pid": 601, "runner_start_ticks": 1201},
            states=states,
            write_watch=samples.append,
            write_stop=lambda _row: None,
            stop=lambda: {"verified_exit": True, "actions": []},
            clock=lambda: "2026-11-12T00:00:00.000Z",
        )
        check(outcome == "free_storage_below_floor" and len(samples) == 1,
              "storage floor was not checked while the helper was pending")

        def stop_helper(pid, sig):
            signalled.append((pid, sig))
            if sig == watch.signal.SIGTERM:
                proc.remove(pid)
                process.returncode = -int(sig)

        stopped = watch._stop_helper_job(
            job,
            attempt,
            status="UNAVAILABLE_TASK_STOPPED",
            proc_root=proc.root,
            send_signal=stop_helper,
            term_grace=0,
            kill_grace=0,
        )
        check(stopped["exit_verified"], "stalled helper exit was not verified")
        check([pid for pid, _sig in signalled] == [901], "helper stop targeted outside its exact PID")
        check(not (attempt / "binary-provenance-receipt.json").exists(),
              "unavailable provenance was published as a completed receipt")

    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        states = add_rank_logs(attempt, proc)
        watch.scan_structured_logs(attempt, states, proc_root=proc.root)

        class TermResistantProcess:
            pid = 902
            returncode = None

            def poll(self):
                return self.returncode if not (proc.root / str(self.pid)).exists() else None

        process = TermResistantProcess()
        proc.add(902, 3102, attempt, attempt / "capture_loaded_binaries.py", ppid=501, session=902)
        now = [0.0]

        def sleep_tick(seconds):
            now[0] += seconds

        job = watch.start_helper_once(
            attempt, states, proc_root=proc.root,
            popen=lambda _command, **_kwargs: process,
            monotonic=lambda: now[0],
        )
        signalled = []

        def escalate_only_helper(pid, sig):
            signalled.append((pid, sig))
            if sig == watch.signal.SIGKILL:
                proc.remove(pid)
                process.returncode = -int(sig)

        done = watch.poll_helper_once(
            job,
            attempt,
            proc_root=proc.root,
            now=job["deadline"],
            send_signal=escalate_only_helper,
            monotonic=lambda: now[0],
            sleep=sleep_tick,
            term_grace=0.1,
            kill_grace=0.1,
        )
        check(done, "timed-out helper did not reach a bounded terminal state")
        check(signalled == [(902, watch.signal.SIGTERM), (902, watch.signal.SIGKILL)],
              "helper timeout did not escalate only its exact PID")
        completion = json.loads((attempt / "helper-completion.json").read_text())
        check(completion["status"] == "UNAVAILABLE_TIMEOUT" and completion["stop"]["exit_verified"],
              "timed-out helper result was not preserved as unavailable")


def check_sequence_gap_error_missing_rank_and_dead_pid_do_not_trigger() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        states = add_rank_logs(attempt, proc, gap_rank=3)
        try:
            watch.scan_structured_logs(attempt, states, proc_root=proc.root)
        except watch.WatchError as exc:
            check("sequence gap" in str(exc), "sequence gap misclassified")
        else:
            raise AssertionError("sequence gap passed")

    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        states = add_rank_logs(attempt, proc, range(7))
        watch.scan_structured_logs(attempt, states, proc_root=proc.root)
        try:
            watch._helper_bindings(attempt, states, proc_root=proc.root)
        except watch.WatchError as exc:
            check("all eight" in str(exc), "missing rank misclassified")
        else:
            raise AssertionError("missing rank triggered helper")

    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        states = add_rank_logs(attempt, proc)
        watch.scan_structured_logs(attempt, states, proc_root=proc.root)
        proc.remove(704)
        try:
            watch._helper_bindings(attempt, states, proc_root=proc.root)
        except watch.WatchError as exc:
            check("not live" in str(exc), "dead rank PID misclassified")
        else:
            raise AssertionError("dead rank PID triggered helper")


def check_error_before_update1_does_not_trigger() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        states = add_rank_logs(attempt, proc, error_rank=2)
        try:
            watch.scan_structured_logs(attempt, states, proc_root=proc.root)
        except watch.WatchError as exc:
            check("step_error" in str(exc), "pre-update error misclassified")
        else:
            raise AssertionError("pre-update error passed")


def check_real_native_prefixes_use_the_same_parser() -> None:
    source_dir = Path(
        "/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/COMBINE-BWD-PROBE/attempts/001/native/structured_logs"
    )
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        log_dir = attempt / "native/structured_logs"
        log_dir.mkdir(parents=True)
        source_pids = {}
        for rank in range(8):
            matches = sorted(source_dir.glob(f"training.global_rank_{rank}.*.jsonl"))
            check(len(matches) == 1, f"expected one source structured log for rank {rank}")
            prefix = []
            found_end = False
            for line in matches[0].open("rb"):
                record = json.loads(line)
                prefix.append(line)
                if record.get("log_type_name") == "optim_end" and record.get("step") == 1:
                    found_end = True
                    break
            check(found_end, f"source rank {rank} lacks optim_end step 1")
            first = json.loads(prefix[0])
            pid = int(first["pid"])
            source_pids[rank] = pid
            proc.add(pid, 4000 + rank, attempt, attempt / "compiler_entry.py")
            (log_dir / f"training.global_rank_{rank}.fixture.jsonl").write_bytes(b"".join(prefix))

        states = {}
        watch.scan_structured_logs(attempt, states, proc_root=proc.root)
        check(all(states[rank]["optim_end"] for rank in range(8)), "native log prefixes did not close update 1")
        runner, tracker = add_fake_runner(attempt, proc)
        tracker.merge_rank_states(states)
        for rank in range(8):
            identity = tracker.records.get((source_pids[rank], 4000 + rank))
            check(identity is not None, f"validated native rank {rank} was not merged into task tracking")
            methods = {proof["method"] for proof in identity["proofs"]}
            check(
                "validated_rank_log_and_exact_pinned_compiler_entrypoint_argument" in methods,
                f"native rank {rank} lost its structured-log proof",
            )
        calls = []

        class MockProcess:
            pid = 900
            returncode = 0

            def poll(self):
                return self.returncode

        mock_process = MockProcess()

        def mocked_popen(command, **_kwargs):
            calls.append(command)
            proc.add(900, 5000, attempt, attempt / "capture_loaded_binaries.py")
            return mock_process

        job = watch.start_helper_once(attempt, states, proc_root=proc.root, popen=mocked_popen)
        (attempt / "binary-provenance-receipt.pending").write_text('{"capture_status":"captured"}\n')
        check(watch.poll_helper_once(job, attempt, proc_root=proc.root), "mocked helper did not finish")
        mapped = [
            int(calls[0][index + 1].split(":", 1)[1])
            for index, arg in enumerate(calls[0])
            if arg == "--rank-pid"
        ]
        check(mapped == [source_pids[rank] for rank in range(8)], "native rank/PID map changed")
        check((attempt / "binary-provenance-receipt.json").is_file(), "completed helper output was not published")
        storage_samples = []
        result = watch.storage_guard_tick(
            watch.STORAGE_FLOOR + 1,
            attempt_dir=attempt,
            runner={"runner_pid": 800, "runner_start_ticks": 5000},
            states=states,
            write_watch=storage_samples.append,
            write_stop=lambda _row: (_ for _ in ()).throw(AssertionError("unexpected storage stop")),
            stop=lambda: (_ for _ in ()).throw(AssertionError("unexpected task signal")),
            clock=lambda: "2026-11-12T00:00:00.000Z",
        )
        check(result is None and len(storage_samples) == 1, "mock storage gate changed native parse result")


def check_orphaned_pinned_entrypoint_recovery_and_bounded_stop() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        runner, tracker = add_fake_runner(attempt, proc)
        entrypoint = attempt / "compiler_entry.py"
        orphan_pid = 760
        proc.add(orphan_pid, 1701, attempt, entrypoint, ppid=1, process_group=900, session=901)

        # These look similar in command-line text but are outside the exact
        # NUL-delimited absolute compiler-entry argument contract.
        proc.add(761, 1702, attempt, Path("/other/compiler_entry.py"), ppid=1)
        proc.add(762, 1703, attempt, Path(str(entrypoint) + ".bak"), ppid=1)
        proc.add(763, 1200, attempt, entrypoint, ppid=1)

        tracker.refresh()
        recovered = tracker.records.get((orphan_pid, 1701))
        check(recovered is not None, "already-reparented pinned compiler entrypoint was not recovered")
        check(recovered["ancestry"] is None, "recovery invented a runner ancestry chain")
        check(
            {proof["method"] for proof in recovered["proofs"]}
            == {"exact_pinned_compiler_entrypoint_argument"},
            "recovered process proof did not identify the exact argv evidence",
        )
        check(
            all((pid, ticks) not in tracker.records for pid, ticks in ((761, 1702), (762, 1703), (763, 1200))),
            "unrelated path, path substring, or pre-runner process was recovered",
        )

        signals = []

        def stop_exact(pid, sig):
            signals.append((pid, sig))
            if pid != orphan_pid or sig == watch.signal.SIGKILL:
                proc.remove(pid)

        result = watch.stop_task_processes(
            tracker,
            send_signal=stop_exact,
            term_grace=0,
            kill_grace=0,
        )
        orphan_actions = [
            action for action in result["actions"]
            if action["pid"] == orphan_pid and action["start_ticks"] == 1701
        ]
        check(result["verified_exit"], "bounded cleanup did not verify the recovered orphan exit")
        check(len(orphan_actions) == 1, "recovered orphan did not receive exactly one cleanup action")
        check(
            orphan_actions[0]["term_sent"] and orphan_actions[0]["kill_sent"]
            and orphan_actions[0]["exit_verified"],
            "TERM-resistant recovered orphan was not boundedly escalated and verified",
        )
        check(
            [pid for pid, _sig in signals].count(orphan_pid) == 2
            and all(pid not in {761, 762, 763} for pid, _sig in signals),
            "cleanup signalled a lookalike or pre-runner process",
        )


def check_recovered_entrypoint_pid_reuse_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        runner, tracker = add_fake_runner(attempt, proc)
        entrypoint = attempt / "compiler_entry.py"
        reused_pid = 770
        proc.add(reused_pid, 1800, attempt, entrypoint, ppid=1)
        tracker.refresh()
        check((reused_pid, 1800) in tracker.records, "fixture did not capture original exact identity")

        # Reuse that PID with a new start tick and an unrelated executable.
        proc.remove(reused_pid)
        proc.add(reused_pid, 2800, attempt, Path("/other/worker.py"), ppid=1)
        tracker.refresh()
        check((reused_pid, 2800) not in tracker.records, "PID reuse with unrelated argv was registered")

        signalled = []

        def record_signal(pid, sig):
            signalled.append((pid, sig))
            proc.remove(pid)

        result = watch.stop_task_processes(
            tracker,
            send_signal=record_signal,
            term_grace=0,
            kill_grace=0,
        )
        check(result["verified_exit"], "cleanup failed to finish after exact PID reuse was rejected")
        check(all(pid != reused_pid for pid, _sig in signalled), "cleanup signalled the reused PID")

    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        runner, _tracker = add_fake_runner(attempt, proc)
        entrypoint = attempt / "compiler_entry.py"
        reused_pid = 771
        proc.add(reused_pid, 1900, attempt, entrypoint, ppid=1)
        reads = 0

        def identity_changes_between_reads(pid, proc_root):
            nonlocal reads
            identity = watch.proc_identity(pid, proc_root)
            if pid == reused_pid:
                reads += 1
                if reads == 1:
                    proc.remove(reused_pid)
                    proc.add(reused_pid, 2900, attempt, Path("/other/worker.py"), ppid=1)
            return identity

        tracker = watch.TaskProcessTracker(
            attempt,
            runner,
            proc_root=proc.root,
            identity_reader=identity_changes_between_reads,
        )
        tracker.refresh()
        check((reused_pid, 1900) not in tracker.records, "start-tick change during registration was accepted")


def check_exact_entrypoint_argument_is_revalidated_before_signal() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        _runner, tracker = add_fake_runner(attempt, proc)
        entrypoint = attempt / "compiler_entry.py"
        pid, ticks = 780, 2000
        proc.add(pid, ticks, attempt, entrypoint, ppid=1)
        tracker.refresh()
        check((pid, ticks) in tracker.records, "exact entrypoint fixture was not recovered")

        # Keep PID/start ticks stable while changing argv after discovery. The
        # signal-time proof must reject this changed command.
        proc.add(pid, ticks, attempt, Path(str(entrypoint) + ".renamed"), ppid=1)
        signalled = []
        try:
            watch.stop_task_processes(
                tracker,
                send_signal=lambda target, sig: signalled.append((target, sig)),
                term_grace=0,
                kill_grace=0,
            )
        except watch.WatchError as exc:
            check("no longer matches" in str(exc), "changed argv was rejected for an unrelated reason")
        else:
            raise AssertionError("changed exact argv still received a verified cleanup result")
        check(all(target != pid for target, _sig in signalled), "changed-argv PID received a signal")


def check_final_recovery_scan_failure_never_reports_verified_exit() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        _runner, tracker = add_fake_runner(attempt, proc)
        proc.add(790, 2100, attempt, attempt / "compiler_entry.py", ppid=1)
        original_refresh = tracker.refresh
        calls = 0

        def fail_final_scan():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise watch.WatchError("mocked final exact-argv recovery scan failure")
            return original_refresh()

        tracker.refresh = fail_final_scan
        try:
            watch.stop_task_processes(
                tracker,
                send_signal=lambda _pid, _sig: None,
                term_grace=0,
                kill_grace=0,
            )
        except watch.WatchError as exc:
            check("final exact-argv" in str(exc), "cleanup failure was misclassified")
        else:
            raise AssertionError("cleanup reported verified exit without the final recovery scan")
        check(calls == 2, "stop path did not make the required final recovery scan")


def check_storage_stop_signals_only_live_exact_attempt_pids() -> None:
    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        runner, tracker = add_fake_runner(attempt, proc)
        states = {rank: {"pid": 700 + rank, "start_ticks": 2000 + rank} for rank in range(3)}
        proc.add(700, 2000, attempt, attempt / "compiler_entry.py", ppid=601, session=701)
        proc.add(701, 2001, attempt, attempt / "compiler_entry.py", ppid=601, session=702)
        proc.add(702, 2002, attempt, attempt / "compiler_entry.py", ppid=601, session=703)
        proc.add(1, 1, Path(temp), Path("/bin/init"))
        signalled = []

        def stop():
            def signal_and_remove(pid, sig):
                signalled.append((pid, sig))
                proc.remove(pid)

            return watch.stop_task_processes(
                tracker,
                send_signal=signal_and_remove,
                term_grace=0,
                kill_grace=0,
            )

        samples = []
        low = watch.storage_guard_tick(
            watch.STORAGE_FLOOR - 1,
            attempt_dir=attempt,
            runner=runner,
            states=states,
            write_watch=samples.append,
            write_stop=lambda _row: None,
            stop=stop,
            clock=lambda: "2026-11-12T00:00:00.000Z",
        )
        check(low == "free_storage_below_floor", "low storage did not stop task")
        check(samples[0]["time_utc"] == "2026-11-12T00:00:00.000Z", "mock clock was not used")
        check({pid for pid, _sig in signalled} == {601, 700, 701, 702}, "stop targeted an unexpected PID set")
        check(all(sig == watch.signal.SIGTERM for _pid, sig in signalled), "stop used unexpected signal")
        check(all(pid != 1 for pid, _sig in signalled), "unrelated system process was targeted")

    with tempfile.TemporaryDirectory() as temp:
        attempt, proc, _ready = fixture(Path(temp))
        runner, tracker = add_fake_runner(attempt, proc)
        signalled = []
        outcome = watch.storage_guard_tick(
            watch.STORAGE_FLOOR + 1,
            attempt_dir=attempt,
            runner=runner,
            states={},
            write_watch=lambda _row: (_ for _ in ()).throw(OSError("mock write failure")),
            write_stop=lambda _row: None,
            stop=lambda: watch.stop_task_processes(
                tracker,
                send_signal=lambda pid, sig: (signalled.append((pid, sig)), proc.remove(pid)),
                term_grace=0,
                kill_grace=0,
            ),
            clock=lambda: "2026-11-12T00:00:00.000Z",
        )
        check(outcome == "storage_watch_write_failure", "write failure did not stop task")
        check([pid for pid, _sig in signalled] == [601], "write-failure stop targeted outside runner")


def check_process_tree_cleans_separate_session_and_preserves_sibling() -> None:
    attempt_root = Path(watch.__file__).resolve().parent
    evidence = {}
    with tempfile.TemporaryDirectory(prefix=".cpu-process-tree-", dir=attempt_root) as temp:
        attempt = Path(temp)
        for name in ("pins.json", "compiler_entry.py"):
            shutil.copyfile(attempt_root / name, attempt / name)
        parent_script = attempt / "parent.py"
        child_script = attempt / "term_resistant.py"
        sentinel_script = attempt / "sibling_sentinel.py"
        pid_file = attempt / "child.pid"
        release_file = attempt / "release-parent"
        child_script.write_text(
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True: time.sleep(0.1)\n"
        )
        sentinel_script.write_text(
            "import sys, time\n"
            "while True: time.sleep(0.1)\n"
        )
        parent_script.write_text(
            "import pathlib, subprocess, sys, time\n"
            "attempt, child_script, pid_file, release = map(pathlib.Path, sys.argv[1:])\n"
            "child = subprocess.Popen([sys.executable, str(child_script), str(attempt)], "
            "start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
            "stderr=subprocess.DEVNULL, close_fds=True)\n"
            "pid_file.write_text(str(child.pid))\n"
            "while not release.exists(): time.sleep(0.02)\n"
        )
        sentinel = subprocess.Popen(
            [sys.executable, str(sentinel_script), str(attempt)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        parent = None
        child_pid = None
        child_ticks = None
        tracker = None
        try:
            sentinel_identity = watch.proc_identity(sentinel.pid)
            parent = subprocess.Popen(
                [
                    sys.executable,
                    str(parent_script),
                    str(attempt),
                    str(child_script),
                    str(pid_file),
                    str(release_file),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            deadline = time.monotonic() + 3
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            check(pid_file.exists(), "dummy task parent did not create its child")
            child_pid = int(pid_file.read_text())
            root_identity = watch.proc_identity(parent.pid)
            child_identity = watch.proc_identity(child_pid)
            child_ticks = child_identity["start_ticks"]
            check(child_identity["session"] != root_identity["session"],
                  "dummy worker did not enter its own session")
            receipt = {
                "attempt_dir": str(attempt),
                "runner_pid": parent.pid,
                "runner_start_ticks": root_identity["start_ticks"],
                "launcher_path": str(parent_script),
            }
            watch.write_json_exclusive(attempt / "launch-receipt.json", receipt)
            tracker = watch.TaskProcessTracker(attempt, receipt)
            tracker.refresh()
            tracked_child = next(
                (
                    record for record in tracker.records.values()
                    if record["pid"] == child_pid and record["start_ticks"] == child_identity["start_ticks"]
                ),
                None,
            )
            check(tracked_child is not None, "separate-session child was not captured before parent exit")
            check(tracked_child["ancestry"][0]["pid"] == parent.pid,
                  "captured child lost its receipt-bound ancestry")
            release_file.touch()
            check(parent.wait(timeout=3) == 0, "dummy task parent failed to exit")

            log_dir = attempt / "native/structured_logs"
            log_dir.mkdir(parents=True)
            (log_dir / "training.global_rank_0.cpu-fixture.jsonl").write_text("{invalid-json}\n")
            try:
                watch.scan_structured_logs(attempt, {})
            except watch.WatchError as parser_error:
                cleanup = watch.cleanup_after_active_failure(
                    attempt,
                    parser_error,
                    tracker,
                    term_grace=0.2,
                    kill_grace=1.5,
                )
            else:
                raise AssertionError("invalid structured log did not trigger the active failure path")
            process_cleanup = cleanup["task_process_cleanup"]
            check(process_cleanup["verified_exit"], "active parser failure did not verify task exit")
            child_action = next(
                action for action in process_cleanup["actions"]
                if action["pid"] == child_pid and action["start_ticks"] == child_identity["start_ticks"]
            )
            check(child_action["term_sent"] and child_action["kill_sent"] and child_action["exit_verified"],
                  "TERM-resistant separate-session child was not boundedly escalated")
            sentinel_live = watch._process_is_live(sentinel.pid, sentinel_identity["start_ticks"], Path("/proc"))
            check(sentinel_live and sentinel.poll() is None,
                  "cleanup targeted a sibling sentinel outside the runner ancestry")
            sentinel_cleanup = watch.stop_exact_process(
                sentinel.pid,
                sentinel_identity["start_ticks"],
                attempt,
                sentinel_script,
                term_grace=0.2,
                kill_grace=0.5,
            )
            sentinel.wait(timeout=2)
            evidence = {
                "status": "PASS",
                "task_root": {"pid": parent.pid, "start_ticks": root_identity["start_ticks"], "exit_code": 0},
                "separate_session_child": {
                    "pid": child_pid,
                    "start_ticks": child_identity["start_ticks"],
                    "session": child_identity["session"],
                    "captured_before_parent_exit": True,
                    "term_sent": child_action["term_sent"],
                    "kill_sent": child_action["kill_sent"],
                    "exit_verified": child_action["exit_verified"],
                },
                "sibling_sentinel": {
                    "pid": sentinel.pid,
                    "start_ticks": sentinel_identity["start_ticks"],
                    "survived_task_cleanup": sentinel_live,
                    "own_exact_cleanup": sentinel_cleanup,
                },
                "failure_path": "invalid actual structured-log JSON -> cleanup_after_active_failure",
                "cleanup_actions": process_cleanup["actions"],
            }
        finally:
            if parent is not None:
                try:
                    if tracker is None and parent.poll() is None:
                        identity = watch.proc_identity(parent.pid)
                        receipt = {
                            "attempt_dir": str(attempt),
                            "runner_pid": parent.pid,
                            "runner_start_ticks": identity["start_ticks"],
                            "launcher_path": str(parent_script),
                        }
                        watch.write_json_exclusive(attempt / "launch-receipt.json", receipt)
                        tracker = watch.TaskProcessTracker(attempt, receipt)
                        tracker.refresh()
                    if tracker is not None:
                        watch.stop_task_processes(tracker, term_grace=0.2, kill_grace=1.0)
                    elif parent.poll() is None:
                        identity = watch.proc_identity(parent.pid)
                        watch.stop_exact_process(
                            parent.pid, identity["start_ticks"], attempt, parent_script,
                            term_grace=0.2, kill_grace=1.0,
                        )
                    if parent.poll() is None:
                        parent.wait(timeout=2)
                except (OSError, watch.WatchError, subprocess.TimeoutExpired):
                    pass
            if child_pid is not None and child_ticks is not None:
                if watch._process_is_live(child_pid, child_ticks, Path("/proc")):
                    watch.stop_exact_process(
                        child_pid, child_ticks, attempt, child_script,
                        term_grace=0.2, kill_grace=1.0,
                    )
            if sentinel.poll() is None:
                current = watch.proc_identity(sentinel.pid)
                watch.stop_exact_process(
                    sentinel.pid, current["start_ticks"], attempt, sentinel_script,
                    term_grace=0.2, kill_grace=1.0,
                )
                sentinel.wait(timeout=2)
        if not evidence:
            raise AssertionError("CPU process-tree cleanup evidence was not recorded")
        evidence_dir = attempt_root.parents[1] / "review/correction-003"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        watch.write_json_exclusive(evidence_dir / f"cpu-process-tree-{stamp}.json", evidence)
        print(
            "CPU_PROCESS_TREE_EVIDENCE "
            + json.dumps(
                {
                    "path": str(evidence_dir / f"cpu-process-tree-{stamp}.json"),
                    "runner_pid": parent.pid,
                    "child_pid": child_pid,
                    "sentinel_pid": sentinel.pid,
                    "child_kill_verified": evidence["separate_session_child"]["exit_verified"],
                    "sentinel_survived": evidence["sibling_sentinel"]["survived_task_cleanup"],
                },
                sort_keys=True,
            )
        )


def main() -> None:
    checks = [
        check_atomic_receipt_publication_never_exposes_partial_json,
        check_ready_gate_fails_closed_and_is_exclusive,
        check_launch_gate_and_receipt_are_pinned_and_exclusive,
        check_watcher_death_before_acknowledgement_blocks_launch,
        check_all_rank_update1_starts_pinned_helper_once,
        check_async_helper_keeps_storage_guard_live_and_kills_only_stalled_helper,
        check_sequence_gap_error_missing_rank_and_dead_pid_do_not_trigger,
        check_error_before_update1_does_not_trigger,
        check_real_native_prefixes_use_the_same_parser,
        check_orphaned_pinned_entrypoint_recovery_and_bounded_stop,
        check_recovered_entrypoint_pid_reuse_is_rejected,
        check_exact_entrypoint_argument_is_revalidated_before_signal,
        check_final_recovery_scan_failure_never_reports_verified_exit,
        check_storage_stop_signals_only_live_exact_attempt_pids,
        check_process_tree_cleans_separate_session_and_preserves_sibling,
    ]
    for test in checks:
        test()
    print(f"PASS: {len(checks)} watcher CPU fixture groups")


if __name__ == "__main__":
    main()
