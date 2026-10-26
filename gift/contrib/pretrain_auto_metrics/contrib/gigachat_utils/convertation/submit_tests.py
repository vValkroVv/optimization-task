#!/usr/bin/env python3
"""
Submit tests to cluster.

Supports both unit and integration tests.

Usage:
    # Unit tests (default)
    python submit_tests.py
    python submit_tests.py --type unit

    # Integration tests
    python submit_tests.py --type integration
    python submit_tests.py --type integration --marker gigar
    python submit_tests.py --type integration --marker gigamix

    # Dry run
    python submit_tests.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contrib.job_submitter.job_submitter.jobs.gigachat_utils.test_convertation import (
    TestConvertationJobGigachatUtils,
)

GITLAB_TOKEN = "REPLACE_WITH_GITLAB_TOKEN.01.0z13yhw6a"

# Test paths
TEST_PATHS = {
    "unit": "tests/unit/",
    "integration": "tests/integration/",
    "all": "tests/",
}


# ======================== ARGUMENT PARSING ========================


def parse_args():
    parser = argparse.ArgumentParser(description="Submit tests to cluster")
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="unit",
        choices=["unit", "integration", "all"],
        help="Type of tests to run: 'unit' (default), 'integration', or 'all'",
    )
    parser.add_argument(
        "--marker",
        "-m",
        type=str,
        default=None,
        help="Pytest marker to filter tests (e.g., 'gigar', 'gigamix')",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Custom path to tests (overrides --type)",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="/workspace-SR008.fs2/msenikolaev/models_for_tests",
        help="Path to test data (checkpoints) for integration tests",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Verbose pytest output (default: True)",
    )
    parser.add_argument(
        "--no-cov", action="store_true", help="Disable coverage reporting"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print command but don't submit job"
    )
    return parser.parse_args()


# ======================== SCRIPT GENERATION ========================


def generate_test_script(
    test_path: str,
    marker: str = None,
    verbose: bool = True,
    with_coverage: bool = True,
    fork_each_test: bool = False,
) -> str:
    """Generate the test script to run on cluster."""

    pytest_args = [test_path, "--tb=short"]

    if verbose:
        pytest_args.append("-v")

    # pytest-cov does not see what a forked child executed, so the report would read as
    # near-zero rather than as partial. No number beats a wrong one.
    if with_coverage and not fork_each_test:
        pytest_args.extend(["--cov=.", "--cov-report=term-missing"])

    if marker:
        pytest_args.extend(["-m", marker])

    # One conversion of the 10B fixture peaks around 86 GB of RSS, and freed arenas are
    # not returned to the OS, so a second conversion in the same process runs into the
    # job's memory ceiling. Forking per test gives each one a fresh address space.
    if fork_each_test:
        pytest_args.append("--forked")

    pytest_cmd = "pytest " + " ".join(pytest_args)
    pip_packages = "pytest pytest-cov safetensors"
    if fork_each_test:
        pip_packages += " pytest-forked"

    script = f"""git config --global credential.helper store && \
echo "https://oauth2:{GITLAB_TOKEN}@gitlab.ai.cloud.ru" > ~/.git-credentials && \
git clone --recurse-submodules https://oauth2:{GITLAB_TOKEN}@gitlab.ai.cloud.ru/gigallms/llm-foundry.git /tmp/llm-foundry && \
cd /tmp/llm-foundry/contrib/gigachat-utils && \
git checkout feature/convertation-refactoring && \
cd /tmp/llm-foundry && \
export PYTHONPATH=/tmp/llm-foundry:/tmp/llm-foundry/contrib/composer:/tmp/llm-foundry/contrib/streaming:/tmp/llm-foundry/contrib/gigafsdp:/tmp/llm-foundry/contrib/gigachat-utils/convertation && \
cd /tmp/llm-foundry/contrib/gigachat-utils/convertation && \
pip install {pip_packages} && \
{pytest_cmd}"""

    return script


# ======================== MAIN ========================


def _parse_test_results_from_logs(logs_text: str) -> tuple[int, dict]:
    """
    Парсит результаты pytest из текста логов.
    Возвращает (exit_code, stats_dict)
    """
    import re

    stats = {
        "failed": 0,
        "errors": 0,
        "interrupted": 0,
        "passed": 0,
        "skipped": 0,
        "xfailed": 0,
        "warnings": 0,
    }

    exit_code = 0

    if not logs_text:
        return exit_code, stats

    summary_pattern = r"=+\s*((?:\d+\s+\w+[,\s]*)+)\s+in\s+[\d.]+s\s*=+"
    summary_match = re.search(summary_pattern, logs_text)

    if summary_match:
        summary_text = summary_match.group(1)
        for count_raw, label_raw in re.findall(r"(\d+)\s+([A-Za-z]+)", summary_text):
            count = int(count_raw)
            label = label_raw.lower()
            if label == "error":
                label = "errors"
            if label == "warning":
                label = "warnings"
            if label in stats:
                stats[label] = count

        if stats["failed"] > 0 or stats["errors"] > 0 or stats["interrupted"] > 0:
            exit_code = exit_code or 1

    exit_patterns = [
        r"exit\s+(?:status|code)[:\s]+(\d+)",
        r"returned\s+exit\s+code\s+(\d+)",
        r"Exit\s+code:\s+(\d+)",
    ]

    for pattern in exit_patterns:
        match = re.search(pattern, logs_text, re.IGNORECASE)
        if match:
            reported_exit = int(match.group(1))
            if reported_exit != 0:
                exit_code = reported_exit
            break

    return exit_code, stats


def main():
    args = parse_args()

    # Determine test path
    if args.test_path:
        test_path = args.test_path
    else:
        test_path = TEST_PATHS.get(args.type, TEST_PATHS["unit"])

    # Generate script
    script = generate_test_script(
        test_path=test_path,
        marker=args.marker,
        verbose=args.verbose,
        with_coverage=not args.no_cov,
        fork_each_test=args.type in ("integration", "all"),
    )

    env_variables = {
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "CONVERTATION_TEST_DATA_PATH": args.test_data_path,
        # The unshard allocates from many threads; glibc then keeps one arena per thread
        # and the freed memory never goes back to the OS. Two arenas cost a little
        # contention and save tens of GB of resident memory on the conversion tests.
        "MALLOC_ARENA_MAX": "2",
    }

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - Would submit job with:")
        print("=" * 60)
        print(f"Test type: {args.type}")
        print(f"Test path: {test_path}")
        print(f"Test data path: {args.test_data_path}")
        print(f"Marker: {args.marker or 'none'}")
        print("\nEnvironment variables:")
        for k, v in env_variables.items():
            print(f"  {k}={v}")
        print(f"\nScript:\n{script}")
        return

    test_job = TestConvertationJobGigachatUtils(
        user_login="autotest",
        exp_name=f"g-utils-test | {args.type.upper()}",
        script=script,
        n_gpus=1,
    )
    job = test_job.submit()
    result = job.monitor()

    final_status = result.final_status
    logs_text = result.logs or ""
    print(f"\nFinal job status: {final_status}")

    if final_status in ["Job status=Completed", "Job status=Succeeded"]:
        if logs_text:
            print("\n" + "=" * 80)
            print("ANALYZING TEST RESULTS FROM LOGS")
            print("=" * 80)

            parsed_exit, stats = _parse_test_results_from_logs(logs_text)

            print("Test statistics:")
            for key, value in stats.items():
                if value > 0:
                    print(f"  {key}: {value}")

            if parsed_exit != 0:
                print(f"\nDetected test failures! Setting exit_code={parsed_exit}")
            else:
                print("\nAll tests passed successfully")

            print("=" * 80 + "\n")
            if parsed_exit != 0:
                raise Exception(f"Test failures detected: {parsed_exit}")
        else:
            print("Warning: Could not retrieve logs for analysis")
    else:
        print(f"Job failed with status: {final_status}")
        raise Exception(f"Job failed with status: {final_status}")


if __name__ == "__main__":
    sys.exit(main())
