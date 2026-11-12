"""Targeted correction checks; mock-only process access and signals."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
REVIEW = Path(__file__).resolve().parent
ATTEMPT = REVIEW.parent.parent / "attempts/001"
sys.path.insert(0, str(ATTEMPT))
import verify_watcher_cpu as checks

watch = checks.watch
tempfile.tempdir = str(REVIEW)
for test in (
    checks.check_launch_gate_and_receipt_are_pinned_and_exclusive,
    checks.check_atomic_receipt_publication_never_exposes_partial_json,
    checks.check_watcher_death_before_acknowledgement_blocks_launch,
    checks.check_async_helper_keeps_storage_guard_live_and_kills_only_stalled_helper,
    checks.check_storage_stop_signals_only_live_exact_attempt_pids,
):
    test()
    print("PASS " + test.__name__)

# A task parent starts and exits between /proc samples. Its long-lived worker
# is now reparented, so the second snapshot contains no edge to the runner.
with tempfile.TemporaryDirectory(dir=REVIEW) as temp:
    attempt, proc, _ = checks.fixture(Path(temp))
    runner, tracker = checks.add_fake_runner(attempt, proc)
    proc.add(701, 1701, attempt, attempt / "worker.py", ppid=1, session=701)
    tracker.refresh()
    signals = []

    def signal_mock(pid, sig):
        signals.append((pid, sig))
        proc.remove(pid)

    result = watch.stop_task_processes(
        tracker, send_signal=signal_mock, term_grace=0, kill_grace=0,
    )
    assert result["verified_exit"] is True
    assert watch._process_is_live(701, 1701, proc.root)
    assert [pid for pid, _ in signals] == [runner["runner_pid"]]
    print("CONFIRMED F3 remains: previously unseen reparented worker survives verified_exit=true")
