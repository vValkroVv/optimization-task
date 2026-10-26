from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
import time
from pathlib import Path

from metric_sampler.contract.raw_layout import RawRoot


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submits the collector's batch script when the raw root holds a complete window that no collector has "
            "moved to `_collected` and no job of the collector's name is queued or running."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--job-name", required=True, help="the Slurm job name of the collector's batch script")
    parser.add_argument("--sbatch", type=Path, required=True, help="the collector's batch script")
    parser.add_argument("--dumps", action="store_true", help="the collector uploads activation dumps too")
    parser.add_argument("--interval-seconds", type=int, default=300)
    return parser.parse_args(argv)


def _collector_queued(job_name: str) -> bool:
    jobs = subprocess.run(
        ["squeue", "--noheader", "--user", getpass.getuser(), "--name", job_name, "--format", "%i"],
        capture_output=True, text=True, check=True,
    ).stdout
    return bool(jobs.strip())


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    raw_root = RawRoot(args.raw_root)
    while True:
        if raw_root.has_uncollected(dumps=args.dumps) and not _collector_queued(args.job_name):
            submitted = subprocess.run(["sbatch", str(args.sbatch)], capture_output=True, text=True, check=True).stdout
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} windows to collect: {submitted.strip()}",
                  file=sys.stderr)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":  # pragma: no cover
    main()
