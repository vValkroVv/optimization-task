import os
import glob
import re
from datetime import datetime
from pathlib import Path
from .extract_traceback import extract_stderr_traceback


def extract_start_time(log_content):
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[PYTEST-TEST\] INFO: Starting pytest test:'
    match = re.search(pattern, log_content)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass

    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[PYTEST-UNIT\] INFO: Starting unit test:'
    match = re.search(pattern, log_content)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass

    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[PYTEST-GLOBAL\] INFO: .+: PYTEST SESSION STARTED'
    match = re.search(pattern, log_content)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass

    return None


def check_for_test_failures_and_stderr(log_content):
    failures = []
    failure_patterns = [
        r'\[PYTEST-TEST\] ERROR: TEST FAILED',
        r'\[PYTEST-TEST\] ERROR: FAILED',
        r'\[PYTEST-TEST\] ERROR: EXCEPTION OCCURRED',
        r'\[PYTEST-UNIT\] ERROR: TEST FAILED',
        r'\[PYTEST-UNIT\] ERROR: FAILED',
        r'\[PYTEST-UNIT\] ERROR: EXCEPTION OCCURRED',
        r'\[PYTEST-GLOBAL\] ERROR: .+: FAILED:',
        r'\[PYTEST-GLOBAL\] ERROR: .+: Tests failed with exit status:'
    ]

    for pattern in failure_patterns:
        if re.search(pattern, log_content):
            failures.append("Test failure detected")
            break
    return failures


def search_and_print_stderr_tracebacks(log_file_path):
    log_directory = os.path.dirname(log_file_path)
    stderr_file_path = os.path.join(log_directory, "stderr.txt")

    print(f"\n{'=' * 60}")
    print(f"SEARCHING FOR STDERR TRACEBACK in same folder: {log_directory}")
    print(f"Looking for: {stderr_file_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(stderr_file_path):
        print(f"No stderr.txt file found in {log_directory}")
        return False

    print(f"Found stderr.txt file: {stderr_file_path}")

    traceback_content = extract_stderr_traceback(stderr_file_path)

    if traceback_content:
        print(f"\n{'=' * 80}")
        print(f"STDERR TRACEBACK FOUND in {stderr_file_path}:")
        print(f"{'=' * 80}")
        print(traceback_content)
        print(f"{'=' * 80}")
        return True
    else:
        print(f"No traceback found in {stderr_file_path}")
        return False


def print_pytest_logs(modality=None):
    print("\n" + "=" * 80)

    search_patterns = []
    if modality == 'UNIT':
        search_patterns.append("run/unit_tests/**/pytest_log.txt")
        print(f"COLLECTING UNIT TEST LOGS FROM:")
        for pattern in search_patterns:
            print(f"  - {pattern}")
    else:
        search_patterns.append("run/**/pytest_log.txt")
        print(f"COLLECTING PYTEST LOGS FROM:")
        for pattern in search_patterns:
            print(f"  - {pattern}")

    print("=" * 80)

    try:
        pytest_log_files = []
        for pattern in search_patterns:
            pytest_log_files.extend(glob.glob(pattern, recursive=True))

        if not pytest_log_files:
            print(f"No pytest log files found")
            return

        log_data = []
        for log_file in pytest_log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    start_time = extract_start_time(content)
                    failures = check_for_test_failures_and_stderr(content)
                    log_data.append({
                        'file': log_file,
                        'content': content,
                        'start_time': start_time,
                        'has_failures': bool(failures)
                    })
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
                log_data.append({
                    'file': log_file,
                    'content': f"Error reading file: {e}",
                    'start_time': None,
                    'has_failures': False
                })

        log_data.sort(key=lambda x: x['start_time'] if x['start_time'] is not None else datetime.max)

        for log_entry in log_data:
            print(f"\n{'=' * 60}")
            print(f"PYTEST LOG: {log_entry['file']}")
            print(f"{'=' * 60}")
            print(log_entry['content'])

            if log_entry['has_failures']:
                print(f"Test failures detected in {log_entry['file']}")

                if modality != 'UNIT':
                    print("Searching for stderr tracebacks...")
                    search_and_print_stderr_tracebacks(log_entry['file'])

    except Exception as e:
        print(f"Error searching for pytest log files: {e}")