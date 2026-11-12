"""CPU-only reproductions of review findings; no real process access or signals."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
REVIEW = Path(__file__).resolve().parent
ATTEMPT = REVIEW.parent.parent / "attempts/001"
spec = importlib.util.spec_from_file_location("watch_review", ATTEMPT / "watch_binary_provenance.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)

# Actual preparation pins must be consumable by the actual receipt writer.
with patch.object(watch, "verify_ready", return_value={}), patch.object(
    watch, "validate_launch_gate", return_value={}
):
    try:
        watch.write_launch_receipt(
            ATTEMPT, runner_pid=999,
            launcher_path=ATTEMPT / "launch.sh", gate_path=ATTEMPT / "launch-gate.json",
        )
    except watch.WatchError as exc:
        assert str(exc) == "launch script SHA-256 mismatch", str(exc)
        print("CONFIRMED F1: actual pins reject actual launcher before process inspection")
    else:
        raise AssertionError("Expected actual pin-schema failure")

# Model the visible interval between O_EXCL creation and writing JSON bytes.
with tempfile.TemporaryDirectory(dir=REVIEW) as temp:
    directory = Path(temp)
    (directory / "launch-receipt.json").touch()
    signals = []
    with patch.object(watch, "write_ready", return_value={"watcher_pid": 501}), patch.object(
        watch, "stop_exact_attempt_pids", side_effect=lambda *a: signals.append(a)
    ):
        result = watch.run_watcher(directory)
    assert result == 1 and not signals
    failure = json.loads((directory / "watcher-failure.json").read_text())
    assert failure["status"] == "FAILED_CLOSED"
    assert failure["reason"] == "invalid or unavailable JSON record: launch-receipt.json"
    print("CONFIRMED F2: partially published launch receipt exits watcher without stopping runner")

# Even when other exact task processes exist, the stop routine only receives
# log-discovered rank identities; absent logs leave only the runner target.
signals = []
with patch.object(watch, "validate_live_pid", return_value=None):
    targets = watch.stop_exact_attempt_pids(
        ATTEMPT, {"runner_pid": 601, "runner_start_ticks": 1201}, {},
        send_signal=lambda pid, sig: signals.append((pid, sig)),
    )
assert targets == [601] and [pid for pid, _ in signals] == [601]
print("CONFIRMED F3: early storage stop directly targets only runner when rank logs are absent")
