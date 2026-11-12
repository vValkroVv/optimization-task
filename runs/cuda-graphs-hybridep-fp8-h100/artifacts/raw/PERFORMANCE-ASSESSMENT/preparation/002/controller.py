"""Supervise one explicitly released performance attempt with the accepted owner."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PREP = Path(__file__).resolve().parent
CAMPAIGN = PREP.parent.parent
TASK = CAMPAIGN.parents[4]
OWNER_SHA256 = "04d47459b9f24e7568f4b14d224673c862eb9a0b7f624650ad35328796d874ba"


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"006", "007", "008", "009", "010"}:
        raise SystemExit("usage: controller.py {006|007|008|009|010}")
    attempt = CAMPAIGN / "attempts" / sys.argv[1]
    owner = PREP / "phase_process_owner.py"
    if hashlib.sha256(owner.read_bytes()).hexdigest() != OWNER_SHA256:
        raise SystemExit("accepted process owner SHA-256 mismatch")
    if not (attempt / "release.json").is_file():
        raise SystemExit("root must release this attempt before launch")
    for name in ("controller.json", "controller-complete.json", "process-ledger.jsonl",
                 "owner-status.json", "owner.json", "launcher.log", "runner.pid"):
        if (attempt / name).exists():
            raise SystemExit(f"refusing existing attempt output: {name}")
    ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    record = {"pid": os.getpid(), "start_ticks": ticks,
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    with (attempt / "controller.json").open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    command = [sys.executable, str(owner), "--phase", f"performance-{sys.argv[1]}",
               "--parent-pid", str(os.getpid()), "--parent-start-ticks", str(ticks),
               "--entrypoint", str(attempt / "phase_entry.py"), "--cwd", str(TASK / "gift"),
               "--ledger", str(attempt / "process-ledger.jsonl"),
               "--status", str(attempt / "owner-status.json"),
               "--owner-record", str(attempt / "owner.json"),
               "--log", str(attempt / "launcher.log"), "--deadline-seconds", "1800",
               "--", "/bin/bash", str(attempt / "launch.sh")]
    result = subprocess.run(command, check=False)
    with (attempt / "controller-complete.json").open("x") as output:
        json.dump({"exit_code": result.returncode,
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}, output, indent=2)
        output.write("\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
