import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from contrib.job_submitter.job_submitter.jobs.llmf.test import TestJobLLMF

from argparse import Namespace


def run_unit_tests_locally(script: str, env_variables: dict) -> int:
    log_dir = Path("run") / "unit_tests" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"

    print("\nRunning unit tests locally...")
    print(f"Logs will be saved to: {log_dir}")

    run_env = os.environ.copy()
    run_env.update(env_variables)

    with open(stdout_log, "w") as stdout_f, open(stderr_log, "w") as stderr_f:
        result = subprocess.run(
            script,
            shell=True,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout_f.write(result.stdout)
        stderr_f.write(result.stderr)

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    print(f"\nUnit tests completed with return code: {result.returncode}")
    print(f"Logs saved to: {log_dir}")

    return result.returncode


def _parse_test_results_from_logs(logs_text: str) -> tuple[int, dict]:
    """
    Парсит результаты pytest из текста логов.
    Возвращает (exit_code, stats_dict)
    """
    import re

    stats = {"failed": 0, "passed": 0, "skipped": 0, "xfailed": 0, "warnings": 0}

    exit_code = 0

    if not logs_text:
        return exit_code, stats

    summary_pattern = r"=+\s*((?:\d+\s+\w+[,\s]*)+)\s+in\s+[\d.]+s\s*=+"
    summary_match = re.search(summary_pattern, logs_text)

    if summary_match:
        summary_text = summary_match.group(1)

        for stat_name in stats.keys():
            pattern = rf"(\d+)\s+{stat_name}"
            match = re.search(pattern, summary_text)
            if match:
                stats[stat_name] = int(match.group(1))

        if stats["failed"] > 0:
            exit_code = 1

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


def run_unit_tests_on_cluster(
    job_desc: str,
    script: str,
    env_variables: dict,
    print_logs_callback=None,
    code_path: str = None,
    args: Namespace = None,
) -> tuple[str, int]:
    if code_path:
        wrapper_script = f"{code_path}/sber_scripts/run_unit_tests_wrapper.py"
        wrapped_script = f"python {wrapper_script} {script}"
    else:
        wrapped_script = f"python sber_scripts/run_unit_tests_wrapper.py {script}"

    test_job = TestJobLLMF(
        user_login="autotest",
        exp_name=job_desc,
        script=wrapped_script,
        extra_env=env_variables,
        args=args,
        queue_name="pretrain-ml-test",
    )
    job = test_job.submit()

    try:
        result = job.monitor()
    finally:
        if print_logs_callback is not None:
            print_logs_callback()

    final_status = result.final_status
    logs_text = result.logs or ""
    print(f"\nFinal job status: {final_status}")

    exit_code = 0

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
                exit_code = parsed_exit
            else:
                print("\nAll tests passed successfully")

            print("=" * 80 + "\n")
        else:
            print("Warning: Could not retrieve logs for analysis")
    else:
        exit_code = 1
        print(f"Job failed with status: {final_status}")

    return final_status, exit_code


def run_integration_tests_on_cluster(
    job_desc: str,
    script: str,
    env_variables: dict,
    print_logs_callback=None,
    args: Namespace = None,
) -> tuple[str, int]:
    test_job = TestJobLLMF(
        user_login="autotest",
        exp_name=job_desc,
        script=script,
        extra_env=env_variables,
        args=args,
        queue_name="pretrain-ml-test",
    )
    job = test_job.submit()

    try:
        result = job.monitor()
    finally:
        if print_logs_callback is not None:
            print_logs_callback()

    final_status = result.final_status
    logs_text = result.logs or ""
    print(f"\nFinal job status: {final_status}")

    exit_code = 0

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
                exit_code = parsed_exit
            else:
                print("\nAll tests passed successfully")

            print("=" * 80 + "\n")
        else:
            print("Warning: Could not retrieve logs for analysis")
    else:
        exit_code = 1
        print(f"Job failed with status: {final_status}")

    return final_status, exit_code
