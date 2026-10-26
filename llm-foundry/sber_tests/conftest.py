import pytest
import logging
import os
import sys
import shutil
import glob
from pathlib import Path
from datetime import datetime
from sber_tests.test_utils.extract_traceback import extract_stderr_traceback
from sber_tests.test_utils.misc import run_timestamp
from sber_tests.test_local_core.constants import ARTIFACT_DIR


failed_tests = []
test_results = {
    'failed': 0,
    'passed': 0,
    'skipped': 0,
    'xfailed': 0,
    'warnings': 0,
    'total_collected': 0,
    'session_duration': 0
}
session_start_time = None


def log_stderr_traceback_for_failed_test(test_logger, test_run_path, test_nodeid):
    if not test_run_path:
        return

    stderr_file_path = os.path.join(test_run_path, "stderr.txt")

    test_logger.info(f"Searching for stderr.txt in test directory: {test_run_path}")
    test_logger.info(f"Looking for: {stderr_file_path}")

    if not os.path.exists(stderr_file_path):
        test_logger.info("No stderr.txt file found in test directory")
        return

    test_logger.info(f"Found stderr.txt file: {stderr_file_path}")

    traceback_content = extract_stderr_traceback(stderr_file_path)

    if traceback_content:
        test_logger.error("=" * 60)
        test_logger.error(f"STDERR TRACEBACK FOUND in {stderr_file_path}:")
        test_logger.error("=" * 60)
        test_logger.error(traceback_content)
        test_logger.error("=" * 60)
    else:
        test_logger.info(f"No traceback found in {stderr_file_path}")


def pytest_configure(config):

    config.addinivalue_line(
        "markers", "unit: Юнит-тесты (тестируют один класс/функцию в изоляции)"
    )
    config.addinivalue_line(
        "markers", "integration: Интеграционные тесты (полный workflow с GPU)"
    )
    config.addinivalue_line(
        "markers", "fast: Быстрые тесты (< 5 секунд)"
    )
    config.addinivalue_line(
        "markers", "slow: Медленные тесты (> 1 минута)"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: Тесты требующие GPU"
    )

    pytest_logs_dir = Path("run") / "pytest_logs"
    pytest_logs_dir.mkdir(parents=True, exist_ok=True)

    pytest_log_file = pytest_logs_dir / f"pytest_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(pytest_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt='%(asctime)s [PYTEST-GLOBAL] %(levelname)s: %(name)s: %(message)s (%(filename)s:%(lineno)d)',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)

    config._pytest_file_handler = file_handler


def pytest_unconfigure(config):
    if hasattr(config, '_pytest_file_handler'):
        root_logger = logging.getLogger()
        root_logger.removeHandler(config._pytest_file_handler)
        config._pytest_file_handler.close()


def pytest_sessionstart(session):
    global session_start_time
    session_start_time = datetime.now()

    logging.info("=" * 80)
    logging.info("PYTEST SESSION STARTED")
    logging.info(f"Session ID: {id(session)}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Python executable: {sys.executable}")
    logging.info(f"Command line: {' '.join(sys.argv)}")

    logging.info("=" * 80)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    global failed_tests, test_results, session_start_time

    if session_start_time:
        session_end_time = datetime.now()
        test_results['session_duration'] = (session_end_time - session_start_time).total_seconds()

    stats = terminalreporter.stats

    test_results['failed'] = len(stats.get('failed', []))
    test_results['passed'] = len(stats.get('passed', []))
    test_results['skipped'] = len(stats.get('skipped', []))
    test_results['xfailed'] = len(stats.get('xfailed', []))
    test_results['warnings'] = getattr(terminalreporter, 'warning_count', 0)
    test_results['total_collected'] = getattr(terminalreporter, '_numcollected', 0)

    failed_tests = []
    for report in stats.get('failed', []):
        if hasattr(report, 'longrepr') and report.longrepr:
            error_lines = str(report.longrepr).strip().split('\n')
            error_msg = "Unknown error"
            for line in reversed(error_lines):
                if 'AssertionError:' in line or 'Error:' in line:
                    error_msg = line.strip()
                    break

            if len(error_msg) > 100:
                error_msg = error_msg[:97] + "..."
            failed_tests.append({
                'nodeid': report.nodeid,
                'error': error_msg
            })


def append_final_summary_to_logs():
    global failed_tests, test_results

    try:
        pytest_log_files = glob.glob("run/**/pytest_log.txt", recursive=True)

        if not pytest_log_files:
            return

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        summary_lines = []

        summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: ")
        summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: FINAL SESSION SUMMARY")
        summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: " + "=" * 80)

        if failed_tests:
            summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: FAILED TESTS:")
            for test in failed_tests:
                summary_lines.append(f"{current_time} [PYTEST-TEST] ERROR: FAILED {test['nodeid']} - {test['error']}")

        summary_parts = []
        if test_results['failed'] > 0:
            summary_parts.append(f"{test_results['failed']} failed")
        if test_results['passed'] > 0:
            summary_parts.append(f"{test_results['passed']} passed")
        if test_results['skipped'] > 0:
            summary_parts.append(f"{test_results['skipped']} skipped")
        if test_results['xfailed'] > 0:
            summary_parts.append(f"{test_results['xfailed']} xfailed")
        if test_results['warnings'] > 0:
            summary_parts.append(f"{test_results['warnings']} warnings")

        duration_str = ""
        if test_results['session_duration'] > 0:
            minutes = int(test_results['session_duration'] // 60)
            seconds = int(test_results['session_duration'] % 60)
            duration_str = f" in {test_results['session_duration']:.2f}s ({minutes}:{seconds:02d})"

        if summary_parts:
            summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: = {', '.join(summary_parts)}{duration_str} =")

        summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: ")

        exit_status = 1 if test_results['failed'] > 0 else 0

        summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: " + "=" * 80)
        summary_lines.append(
            f"{current_time} [PYTEST-TEST] INFO: PYTEST SESSION FINISHED with exit status: {exit_status}")
        summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: Failed tests: {test_results['failed']}")
        summary_lines.append(
            f"{current_time} [PYTEST-TEST] INFO: Total tests collected: {test_results['total_collected']}")

        if exit_status != 0:
            summary_lines.append(f"{current_time} [PYTEST-TEST] ERROR: Tests failed with exit status: {exit_status}")
        else:
            summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: All tests passed successfully!")

        summary_lines.append(f"{current_time} [PYTEST-TEST] INFO: " + "=" * 80)

        if pytest_log_files:
            latest_log_file = max(pytest_log_files, key=lambda f: os.path.getmtime(f))

            with open(latest_log_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines) + '\n')

    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    global failed_tests, test_results

    # Removes checkpoints after pytest session ends
    clear_run_dir = os.getenv("CLEAR_RUN_DIR", "0") == "1"
    if not clear_run_dir:
        return

    current_run_dir = Path(ARTIFACT_DIR) / "run" / run_timestamp()
    if not current_run_dir.exists():
        return

    for test_case_dir in current_run_dir.iterdir():
        ckpt_dir = test_case_dir / "ckpt"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)


def pytest_collection_modifyitems(config, items):
    global test_results
    test_results['total_collected'] = len(items)
    logging.info(f"Collected {len(items)} test items:")
    for item in items:
        logging.info(f"  - {item.nodeid}")


def pytest_runtest_setup(item):
    logging.info(f"SETUP: {item.nodeid}")


def pytest_runtest_call(item):
    logging.info(f"CALL: {item.nodeid}")


def pytest_runtest_teardown(item):
    logging.info(f"TEARDOWN: {item.nodeid}")


def pytest_runtest_logreport(report):
    if report.when == "call":
        duration = getattr(report, 'duration', 0)

        if report.passed:
            logging.info(f"PASSED: {report.nodeid} (duration: {duration:.2f}s)")
        elif report.failed:
            logging.error(f"FAILED: {report.nodeid} (duration: {duration:.2f}s)")
            logging.error(f"Failure reason: {report.longrepr}")
        elif report.skipped:
            logging.warning(f"SKIPPED: {report.nodeid}")
            if hasattr(report, 'longrepr') and report.longrepr:
                logging.warning(f"Skip reason: {report.longrepr}")

        test_logger = _get_test_logger(report.nodeid)
        if test_logger:
            if report.passed:
                test_logger.info(f"TEST PASSED (duration: {duration:.2f}s)")
                test_logger.info("=" * 50)
            elif report.failed:
                test_logger.error(f"TEST FAILED (duration: {duration:.2f}s)")
                test_logger.error("FAILURE DETAILS:")
                test_logger.error(str(report.longrepr))
                test_logger.error("=" * 50)

                test_run_path = getattr(report, '_test_run_path', None)
                if test_run_path:
                    log_stderr_traceback_for_failed_test(test_logger, test_run_path, report.nodeid)

            elif report.skipped:
                test_logger.warning("TEST SKIPPED")
                if hasattr(report, 'longrepr') and report.longrepr:
                    test_logger.warning(f"Skip reason: {report.longrepr}")
                test_logger.warning("=" * 50)


def pytest_exception_interact(node, call, report):
    if report.failed:
        logging.error(f"EXCEPTION in {node.nodeid}:")
        logging.error(f"Exception type: {call.excinfo.type.__name__}")
        logging.error(f"Exception value: {call.excinfo.value}")
        logging.error(f"Full traceback: {report.longrepr}")

        test_logger = _get_test_logger(node.nodeid)
        if test_logger:
            test_logger.error("EXCEPTION OCCURRED:")
            test_logger.error(f"Exception type: {call.excinfo.type.__name__}")
            test_logger.error(f"Exception value: {call.excinfo.value}")
            test_logger.error(f"Full traceback: {report.longrepr}")

            test_run_path = getattr(report, '_test_run_path', None)
            if test_run_path:
                log_stderr_traceback_for_failed_test(test_logger, test_run_path, node.nodeid)


def _get_test_logger(nodeid):
    test_name = nodeid.replace("::", "_").replace("/", "_").replace("[", "_").replace("]", "")
    logger_name = f"test_{test_name}"

    test_logger = logging.getLogger(logger_name)
    if test_logger.handlers:
        return test_logger
    return None


@pytest.fixture(autouse=True)
def log_test_execution(request):
    test_name = request.node.nodeid

    logging.info(f"Starting test: {test_name}")

    entry_info = {}
    if hasattr(request, 'getfixturevalue'):
        try:
            entry = request.getfixturevalue('entry')
            if hasattr(entry, 'id'):
                entry_info['id'] = entry.id
                entry_info['config_path'] = getattr(entry, 'config_path', 'N/A')
                entry_info['run_path'] = getattr(entry, 'run_path', 'N/A')
                entry_info['n_gpus'] = getattr(entry, 'n_gpus', 'N/A')

                logging.info(f"Test entry details: {entry_info}")
        except Exception as e:
            logging.debug(f"Could not get entry fixture: {e}")

    start_time = datetime.now()

    yield

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info(f"Finished test: {test_name} (total duration: {duration:.2f}s)")


@pytest.fixture(autouse=True, scope="function")
def test_specific_logger(request):
    test_name = request.node.nodeid.replace("::", "_").replace("/", "_").replace("[", "_").replace("]", "")

    is_unit_test = any(mark.name == 'unit' for mark in request.node.iter_markers())

    if is_unit_test:
        unit_tests_dir = Path("run") / "unit_tests" / run_timestamp()
        log_dir = unit_tests_dir / test_name
        log_dir.mkdir(parents=True, exist_ok=True)

        pytest_log_file = log_dir / "pytest_log.txt"

        test_logger = logging.getLogger(f"unit_test_{test_name}")
        test_logger.setLevel(logging.INFO)
        test_logger.handlers.clear()

        test_file_handler = logging.FileHandler(pytest_log_file, mode='w', encoding='utf-8')
        test_file_handler.setLevel(logging.INFO)

        test_formatter = logging.Formatter(
            fmt='%(asctime)s [PYTEST-UNIT] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        test_file_handler.setFormatter(test_formatter)
        test_logger.addHandler(test_file_handler)
        test_logger.propagate = False

        test_logger.info(f"Starting unit test: {request.node.nodeid}")
        test_logger.info(f"Test location: {request.node.location}")
        test_logger.info(f"Log directory: {log_dir}")

        request.node.test_logger = test_logger
        request.node._test_run_path = str(log_dir)
    else:
        try:
            entry = request.getfixturevalue('entry')
            if hasattr(entry, 'run_path') and entry.run_path:
                log_dir = Path(entry.run_path)
                log_dir.mkdir(parents=True, exist_ok=True)

                pytest_log_file = log_dir / "pytest_log.txt"

                test_logger = logging.getLogger(f"test_{test_name}")
                test_logger.setLevel(logging.INFO)
                test_logger.handlers.clear()

                test_file_handler = logging.FileHandler(pytest_log_file, mode='w', encoding='utf-8')
                test_file_handler.setLevel(logging.INFO)

                test_formatter = logging.Formatter(
                    fmt='%(asctime)s [PYTEST-TEST] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                test_file_handler.setFormatter(test_formatter)
                test_logger.addHandler(test_file_handler)
                test_logger.propagate = False

                test_logger.info(f"Starting pytest test: {request.node.nodeid}")
                test_logger.info(f"Test entry ID: {getattr(entry, 'id', 'N/A')}")
                test_logger.info(f"Config path: {getattr(entry, 'config_path', 'N/A')}")
                test_logger.info(f"Run path: {getattr(entry, 'run_path', 'N/A')}")

                request.node.test_logger = test_logger

                if hasattr(entry, 'run_path'):
                    request.node._test_run_path = entry.run_path

        except Exception as e:
            logging.debug(f"Could not setup test-specific logger: {e}")

    yield

    if hasattr(request.node, 'test_logger'):
        request.node.test_logger.info(f"Finished pytest test: {request.node.nodeid}")
        for handler in request.node.test_logger.handlers:
            handler.close()
        request.node.test_logger.handlers.clear()
