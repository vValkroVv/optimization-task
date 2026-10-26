from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from contrib.job_submitter.job_submitter.jobs.pam.test import TestJobPAM

SUCCESS_STATUSES = {"Completed", "Succeeded"}

UNKNOWN_STATUS = "Cant get status for job"
ACTIVE_STATUSES = {"Pending", "Running", UNKNOWN_STATUS}
REPO_ROOT = Path(__file__).parent.parent
JUNIT_XML_NAME = "pytest-e2e.xml"

SUITES = {
    "all": ("e2e", "sber_tests/e2e", 1),
    "text": ("e2e", "sber_tests/e2e/test_e2e.py", 1),
    "vision": ("e2e", "sber_tests/e2e/test_e2e_vision.py", 1),
    "430b": ("e2e_430b", "sber_tests/e2e/test_e2e_430b.py", 2),
}


def _extract_status_name(status: str) -> str:
    prefix = "Job status="
    if status.startswith(prefix):
        return status[len(prefix) :]
    return status


def _print_junit_failures(junit_path: Path) -> None:
    if not junit_path.exists():
        print(f"No junit report found at {junit_path}")
        return
    try:
        root = ET.parse(junit_path).getroot()
    except ET.ParseError as exc:
        print(f"Failed to parse junit report {junit_path}: {exc}")
        return

    failures_found = False
    for testcase in root.iter("testcase"):
        for node in testcase:
            if node.tag not in ("failure", "error"):
                continue
            failures_found = True
            test_name = f"{testcase.get('classname', '')}::{testcase.get('name', '')}"
            print(f"\n===== {node.tag.upper()}: {test_name} =====")
            message = node.get("message")
            if message:
                print(message)
            if node.text:
                print(node.text)
    if not failures_found:
        print(
            f"Junit report {junit_path} contains no failure details "
            "(job may have died before pytest finished)"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=tuple(SUITES),
        default="all",
        help="E2E suite to submit",
    )
    args = parser.parse_args()
    marker, pytest_target, n_workers = SUITES[args.suite]
    pytest_comand = f"python3 -m pytest -m {marker} --strict-markers --maxfail=1 -q --tb=short -rf --junitxml={JUNIT_XML_NAME} {pytest_target}"
    script = f"--wdir {REPO_ROOT} {pytest_comand}"

    junit_path = REPO_ROOT / JUNIT_XML_NAME
    junit_path.unlink(missing_ok=True)

    test_job = TestJobPAM(
        user_login="autotest",
        exp_name=f"PAM e2e tests ({args.suite})",
        script=script,
        queue_name="pretrain-ml-test",
        n_workers=n_workers,
        base_image=os.environ.get("IMAGE_PATH"),
        extra_env={k: v for k, v in os.environ.items() if k.startswith("PAM_E2E_")},
    )

    print("Submitting e2e test job...")
    test_job.submit()

    monitor_result = test_job.monitor(active_statuses=ACTIVE_STATUSES)
    final_status = monitor_result.final_status
    final_status_name = _extract_status_name(final_status)
    print(f"Final status: {final_status}")

    if final_status_name not in SUCCESS_STATUSES:
        _print_junit_failures(junit_path)
        raise AssertionError(f"Job finished unsuccessfully: {final_status}")

    print("Job finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
