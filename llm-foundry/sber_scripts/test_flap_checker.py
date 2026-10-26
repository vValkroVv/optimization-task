#!/usr/bin/env python3
"""
python test_flap_checker.py TestHybridGroup -n 2 --show-errors
python test_flap_checker.py Test1 Test2 Test3 Test4
"""

import os
import subprocess
import time
import socket
import argparse
import signal
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TestResult:
    run_id: int
    passed: bool
    duration: float
    output: str = ""


class TestRunner:
    def __init__(self, test_patterns: List[str], show_errors: bool = False):
        self.test_patterns = test_patterns
        self.show_errors = show_errors
        self.results: Dict[str, List[TestResult]] = {}
        self.interrupted = False
        self.code_path = Path(__file__).parent.parent

        for pattern in test_patterns:
            self.results[pattern] = []

        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _cleanup_jobs(self):
        try:
            print("  Cleaning up processes:", end="", flush=True)

            subprocess.run(
                ["pkill", "-9", "-f", "torchrun"], capture_output=True, timeout=5
            )
            print(".", end="", flush=True)

            subprocess.run(
                ["pkill", "-9", "-f", "train.py"], capture_output=True, timeout=5
            )
            print(".", end="", flush=True)

            subprocess.run(
                ["bash", "-c", "lsof -ti:29500-32000 | xargs -r kill -9"],
                capture_output=True,
                timeout=5,
            )
            print(".", end="", flush=True)

            subprocess.run(
                ["pkill", "-9", "-f", "torch.distributed"],
                capture_output=True,
                timeout=5,
            )
            print(".", end="", flush=True)

            print(" done")
            time.sleep(3)
        except Exception as e:
            print(f"\n  Warning: cleanup failed: {e}")

    def _quick_cleanup(self):
        try:
            subprocess.run(
                ["pkill", "-9", "-f", "torchrun"], capture_output=True, timeout=5
            )

            subprocess.run(
                ["pkill", "-9", "-f", "train.py"], capture_output=True, timeout=5
            )

            subprocess.run(
                ["bash", "-c", "lsof -ti:29500-32000 | xargs -r kill -9"],
                capture_output=True,
                timeout=5,
            )

            time.sleep(2)
        except Exception as e:
            print(f"  Quick cleanup warning: {e}")

    def _handle_interrupt(self, signum, frame):
        total_runs = sum(len(results) for results in self.results.values())
        print(f"\nInterrupted! Showing results for {total_runs} completed runs...")
        self.interrupted = True
        self._show_results()
        sys.exit(0)

    def _build_pytest_cmd(self, test_pattern: str) -> List[str]:
        code_path = (
            Path(__file__).parent.parent
            / "sber_tests"
            / "integration"
            / "test_local_jobs.py"
        )

        cmd = ["python", "-m", "pytest", str(code_path), "-v", "-x", "--tb=short"]

        if "::" in test_pattern:
            cmd.append(f"-k{test_pattern}")
        elif test_pattern.startswith("Test"):
            cmd.extend(["-k", test_pattern])
        else:
            cmd.extend(["-k", test_pattern])

        return cmd

    def _find_free_port(self, start_port=29500, end_port=32000, max_attempts=100):
        import random

        for _ in range(max_attempts):
            port = random.randint(start_port, end_port)

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("", port))
                    sock.close()
                    return port
                except OSError:
                    continue

        raise RuntimeError(f"Could not find free port after {max_attempts} attempts")

    def run_once(self, test_pattern: str, run_id: int) -> TestResult:
        print(f"[{test_pattern}] Run #{run_id}... ", end="", flush=True)

        try:
            master_port = self._find_free_port()
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return TestResult(run_id=run_id, passed=False, duration=0.0, output=str(e))

        env = os.environ.copy()
        env["MASTER_PORT"] = str(master_port)

        start_time = time.time()
        try:
            result = subprocess.run(
                self._build_pytest_cmd(test_pattern),
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )
            duration = time.time() - start_time
            passed = result.returncode == 0

            emoji = "✅" if passed else "❌"
            print(f"{emoji} ({duration:.1f}s)")

            time.sleep(1)
            self._quick_cleanup()

            return TestResult(
                run_id=run_id,
                passed=passed,
                duration=duration,
                output=result.stdout + result.stderr,
            )

        except subprocess.TimeoutExpired:
            print("TIMEOUT - cleaning up...")
            self._quick_cleanup()
            return TestResult(
                run_id=run_id, passed=False, duration=1800.0, output="TIMEOUT"
            )

    def run_multiple(self, n_runs: int, delay: int = 0):
        if len(self.test_patterns) == 1:
            print(f"Testing '{self.test_patterns[0]}' {n_runs} times\n")
        else:
            print(
                f"Testing {len(self.test_patterns)} test patterns {n_runs} times each\n"
            )
            print(f"Test patterns: {', '.join(self.test_patterns)}\n")

        for i in range(1, n_runs + 1):
            if self.interrupted:
                break

            if i > 1:
                print(f"\nFull cleanup before run #{i}...")
                self._cleanup_jobs()
                time.sleep(3)

            for test_pattern in self.test_patterns:
                if self.interrupted:
                    break

                result = self.run_once(test_pattern, i)
                self.results[test_pattern].append(result)

                if (
                    len(self.test_patterns) > 1
                    and test_pattern != self.test_patterns[-1]
                ):
                    time.sleep(0.5)

            if delay > 0 and i < n_runs and not self.interrupted:
                time.sleep(delay)

        if not self.interrupted:
            self._show_results()

    def _show_results(self):
        if not any(self.results.values()):
            return

        print(f"\n{'=' * 70}")
        print("SUMMARY RESULTS")
        print(f"{'=' * 70}")

        overall_total = 0
        overall_passed = 0
        overall_time = 0

        for test_pattern in self.test_patterns:
            results = self.results[test_pattern]
            if not results:
                continue

            total = len(results)
            passed = sum(r.passed for r in results)
            success_rate = passed / total * 100 if total > 0 else 0
            avg_time = sum(r.duration for r in results) / total if total > 0 else 0

            overall_total += total
            overall_passed += passed
            overall_time += sum(r.duration for r in results)

            print(f"\nTest pattern: '{test_pattern}'")
            print(f"   Runs:        {total}")
            print(f"   Passed:      {passed}")
            print(f"   Failed:      {total - passed}")
            print(f"   Success:     {success_rate:.1f}%")
            print(f"   Avg time:    {avg_time:.1f}s")

            if success_rate == 100:
                verdict = "OK"
            else:
                verdict = "FLAP"
            print(f"   Verdict:     {verdict}")

            failures = [r for r in results if not r.passed]
            if failures:
                print(f"   Failed runs: {[r.run_id for r in failures]}")

                if self.show_errors:
                    print(f"\n   Error details for '{test_pattern}':")
                    for failure in failures:
                        print(f"\n   Run #{failure.run_id} output:")
                        print("   " + "-" * 50)
                        for line in failure.output.split("\n"):
                            if line.strip():
                                print(f"   {line}")
                        print("   " + "-" * 50)

        if len(self.test_patterns) > 1:
            overall_success_rate = (
                overall_passed / overall_total * 100 if overall_total > 0 else 0
            )
            overall_avg_time = overall_time / overall_total if overall_total > 0 else 0

            print("\nOVERALL SUMMARY:")
            print(f"   Total runs:   {overall_total}")
            print(f"   Total passed: {overall_passed}")
            print(f"   Total failed: {overall_total - overall_passed}")
            print(f"   Success rate: {overall_success_rate:.1f}%")
            print(f"   Avg time:     {overall_avg_time:.1f}s")

            if overall_success_rate == 100:
                overall_verdict = "OK"
            else:
                overall_verdict = "FLAP"
            print(f"   Verdict:      {overall_verdict}")


def main():
    parser = argparse.ArgumentParser(
        description="Test flappiness checker - supports single or multiple test patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_flap_checker.py TestHybridGroup -n 2 --show-errors # Show detailed error output
        """,
    )

    parser.add_argument(
        "tests",
        nargs="+",
        help="Test pattern(s) to check. Can be single test or multiple tests separated by spaces.",
    )
    parser.add_argument(
        "-n", type=int, default=5, help="Number of runs per test (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=1,
        help="Delay between runs in seconds (default: 1)",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show detailed error output for failed runs",
    )

    args = parser.parse_args()

    unique_tests = list(dict.fromkeys(args.tests))

    if len(unique_tests) != len(args.tests):
        print(
            f"Removed {len(args.tests) - len(unique_tests)} duplicate test pattern(s)"
        )

    runner = TestRunner(unique_tests, show_errors=args.show_errors)
    runner.run_multiple(args.n, args.delay)


if __name__ == "__main__":
    main()
