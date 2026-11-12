"""Supervise one explicitly released native trace attempt."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PREP = Path('/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT/preparation/002')
CAMPAIGN = Path('/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE')
TASK = Path('/workspace-SR008.nfs2/vkropotin/optimization-task')
OWNER_SHA256 = "04d47459b9f24e7568f4b14d224673c862eb9a0b7f624650ad35328796d874ba"
RELEASE_STATUS = "ROOT_RELEASED_REPAIRED_NATIVE_TRACE"
ATTEMPTS = {"003"}
OUTPUTS = (
    "controller.json", "controller-complete.json", "process-ledger.jsonl",
    "owner-status.json", "owner.json", "launcher.log", "runner.pid",
    "native", "torchrun", "scan-runtime", "deepgemm-cache",
    "deepgemm-shared-cache", "cores",
)


def main() -> int:
    if len(sys.argv) == 1:
        attempt_id = "003"
    elif len(sys.argv) == 2 and sys.argv[1] in ATTEMPTS:
        attempt_id = sys.argv[1]
    else:
        raise SystemExit("usage: controller.py [003]")

    attempt = CAMPAIGN / "attempts" / attempt_id
    owner = PREP / "phase_process_owner.py"
    if hashlib.sha256(owner.read_bytes()).hexdigest() != OWNER_SHA256:
        raise SystemExit("accepted process owner SHA-256 mismatch")
    release_path = attempt / "release.json"
    if not release_path.is_file():
        raise SystemExit("root must release this attempt before launch")
    release = json.loads(release_path.read_text())
    if release.get("status") != RELEASE_STATUS:
        raise SystemExit(f"root release status must be {RELEASE_STATUS}")
    for name in OUTPUTS:
        output = attempt / name
        if output.exists() or output.is_symlink():
            raise SystemExit(f"refusing existing attempt output: {name}")

    ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    record = {
        "pid": os.getpid(),
        "start_ticks": ticks,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (attempt / "controller.json").open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")

    command = [
        sys.executable,
        str(owner),
        "--phase",
        f"baseline-trace-{attempt_id}",
        "--parent-pid",
        str(os.getpid()),
        "--parent-start-ticks",
        str(ticks),
        "--entrypoint",
        str(attempt / "profile_entry.py"),
        "--cwd",
        str(TASK / "gift"),
        "--ledger",
        str(attempt / "process-ledger.jsonl"),
        "--status",
        str(attempt / "owner-status.json"),
        "--owner-record",
        str(attempt / "owner.json"),
        "--log",
        str(attempt / "launcher.log"),
        "--deadline-seconds",
        "1800",
        "--",
        "/bin/bash",
        str(attempt / "launch.sh"),
    ]
    result = subprocess.run(command, check=False)
    with (attempt / "controller-complete.json").open("x") as output:
        json.dump(
            {
                "exit_code": result.returncode,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            output,
            indent=2,
        )
        output.write("\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
