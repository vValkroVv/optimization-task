"""Final F3 verification against the actual current modules; all signals mocked."""

import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
REVIEW = Path(__file__).resolve().parent
ATTEMPT = REVIEW.parent.parent / "attempts/001"
sys.path.insert(0, str(ATTEMPT))
import verify_watcher_cpu as checks

tempfile.tempdir = str(REVIEW)
for test in (
    checks.check_orphaned_pinned_entrypoint_recovery_and_bounded_stop,
    checks.check_recovered_entrypoint_pid_reuse_is_rejected,
    checks.check_exact_entrypoint_argument_is_revalidated_before_signal,
    checks.check_final_recovery_scan_failure_never_reports_verified_exit,
):
    test()
    print("PASS " + test.__name__)

with tempfile.TemporaryDirectory(dir=REVIEW) as temp:
    attempt, proc, _ready = checks.fixture(Path(temp))
    _runner, tracker = checks.add_fake_runner(attempt, proc)
    proc.add(799, 2200, attempt, attempt / "compiler_entry.py", ppid=1)
    tracker.merge_rank_states({3: {"started": True, "pid": 799, "start_ticks": 2200}})
    record = tracker.records[(799, 2200)]
    assert any(
        p["method"] == "validated_rank_log_and_exact_pinned_compiler_entrypoint_argument"
        and p["rank"] == 3 for p in record["proofs"]
    )
    assert tracker.can_signal_record(record)
    print("PASS parsed rank PID identity merges with exact pinned entrypoint proof")
