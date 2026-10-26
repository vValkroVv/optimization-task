import os
import sys
import subprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_unit_tests_wrapper.py <pytest_command>")
        sys.exit(1)

    pytest_cmd = " ".join(sys.argv[1:])

    local_run_dir = os.getenv("LOCAL_RUN_DIR")
    if local_run_dir:
        os.chdir(local_run_dir)
        print(f"Changed directory to: {local_run_dir}")

    print(f"\n{'=' * 80}")
    print(f"UNIT TEST WRAPPER - Running pytest on cluster")
    print(f"Current directory: {os.getcwd()}")
    print(f"Pytest command: {pytest_cmd}")
    print(f"conftest.py will create log files in run/unit_tests/...")
    print(f"{'=' * 80}\n")

    # Run pytest directly - conftest handles all logging and we bypass some restrictions with subprocess run
    result = subprocess.run(
        pytest_cmd,
        shell=True
    )

    print(f"\n{'=' * 80}")
    print(f"Unit tests completed with return code: {result.returncode}")
    print(f"{'=' * 80}\n")

    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
