#!/usr/bin/env python3
"""Submit sglang_pretrain tests to cluster.

The cluster pod sees the working tree directly via NFS (same view as the
developer host or CI runner), so we don't clone anything — `TestSglangPretrainJob`
puts the host paths on PYTHONPATH via `_external_code_libraries`, and pytest
runs against the live checkout. This mirrors the PAM e2e pattern in
`sber_tests/submit_e2e_tests.py`.

Usage:
    # Unit tests (default)
    python submit_tests.py
    python submit_tests.py --type unit

    # Integration tests
    python submit_tests.py --type integration
    python submit_tests.py --type integration --marker gigachat35

    # Entire local test tree
    python submit_tests.py --type all

    # Dry run
    python submit_tests.py --dry-run
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

# Use the in-tree vendored job_submitter (shipped with the sglang_pretrain
# submodule) so this CLI is self-contained.
SGLANG_PRETRAIN_ROOT = Path(__file__).resolve().parent
_JOB_SUBMITTER_ROOT = SGLANG_PRETRAIN_ROOT / "contrib" / "job_submitter"
sys.path.insert(0, str(_JOB_SUBMITTER_ROOT))
# job_submitter depends on sqlite_s3_query, vendored as a nested submodule. Its
# own sys.path bootstrap (job_submitter/__init__.py) only fires when imported
# under a "contrib."-prefixed name; we import it top-level, so wire up the
# nested package here too (otherwise tracking_storage's import fails when the
# package isn't editable-installed, e.g. on a fresh CI checkout).
_SQLITE_S3_QUERY_ROOT = _JOB_SUBMITTER_ROOT / "contrib" / "sqlite_s3_query"
if (_SQLITE_S3_QUERY_ROOT / "sqlite_s3_query").is_dir():
    sys.path.insert(0, str(_SQLITE_S3_QUERY_ROOT))
from job_submitter.jobs.sglang_pretrain.test import TestSglangPretrainJob

TEST_PATHS = {
    "unit": "test/unit/",
    "integration": "test/integration/",
    "all": "test/",
}
DEFAULT_GIGACHAT35_CHECKPOINT_DIR = (
    "/workspace-SR008.nfs2/models/models_for_tests/sglang_pretrain/"
    "3b_3_5_arch_5000steps_4ba_ep8_tp1_sp1"
)
DEFAULT_GIGAVISION_CHECKPOINT_DIR = (
    "/home/jovyan/models/10b_vision_test_no_delete"
)
MODEL_TEST_TP_SIZES = {
    "gigachat35": 1,
    "gigaomni": 1,
    "gigavision": 4,
}
DEFAULT_GIGAOMNI_CHECKPOINT_DIR = (
    "/workspace-SR008.nfs2/models/models_for_tests/sglang_pretrain/"
    "10b_v30.1_chatml_v4__3__sft__va_ve_llm__video_sft_timestamps_tr__gigaomni_v2"
)
DEFAULT_N_GPUS = 1
SUCCESS_STATUSES = {"Completed", "Succeeded"}


def _extract_status_name(status: str) -> str:
    prefix = "Job status="
    if status.startswith(prefix):
        return status[len(prefix):]
    return status


def parse_args():
    parser = argparse.ArgumentParser(description="Submit sglang_pretrain tests to cluster")
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
        help="Pytest marker to filter tests",
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
        help="Reserved for tests that need local model data on cluster",
    )
    parser.add_argument(
        "--gigachat35-checkpoint-dir",
        type=str,
        default=DEFAULT_GIGACHAT35_CHECKPOINT_DIR,
        help="Directory holding the gigachat35 integration-test checkpoints "
        "(subdirs: original, multi_mtp)",
    )
    parser.add_argument(
        "--gigavision-checkpoint-dir",
        type=str,
        default=DEFAULT_GIGAVISION_CHECKPOINT_DIR,
        help="Directory holding the GigaVision HF checkpoint",
    )
    parser.add_argument(
        "--gigaomni-checkpoint-dir",
        type=str,
        default=DEFAULT_GIGAOMNI_CHECKPOINT_DIR,
        help="GigaOmni HF checkpoint used by gigaomni integration tests",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Verbose pytest output (default: True)",
    )
    parser.add_argument(
        "--cov",
        action="store_true",
        help="Enable coverage reporting (off by default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command but do not submit job",
    )
    return parser.parse_args()


def generate_test_script(
    *,
    test_path: str,
    marker: str | None = None,
    verbose: bool = True,
    with_coverage: bool = False,
) -> str:
    """Build the bash command the cluster container runs.
    """
    pytest_args = [test_path, "--tb=short"]
    if verbose:
        pytest_args.append("-v")
    if with_coverage:
        pytest_args.extend(["--cov=.", "--cov-report=term-missing"])
    if marker:
        pytest_args.extend(["-m", marker])
    quoted_pytest = " ".join(shlex.quote(arg) for arg in pytest_args)
    python = "/home/user/sglang/bin/python"
    inner = (
        f"cd {shlex.quote(str(SGLANG_PRETRAIN_ROOT))} && "
        f"{python} -m pip install --quiet pytest pytest-cov && "
        f"{python} -m pytest {quoted_pytest}"
    )
    return f"bash -c {shlex.quote(inner)}"


def main():
    args = parse_args()

    test_path = args.test_path or TEST_PATHS.get(args.type, TEST_PATHS["unit"])

    script = generate_test_script(
        test_path=test_path,
        marker=args.marker,
        verbose=args.verbose,
        with_coverage=args.cov,
    )

    env_variables = {
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "SGLANG_PRETRAIN_TEST_DATA_PATH": args.test_data_path,
        "GIGACHAT35_TEST_CHECKPOINT_DIR": args.gigachat35_checkpoint_dir,
        "GIGAVISION_TEST_CHECKPOINT_DIR": args.gigavision_checkpoint_dir,
        "GIGAOMNI_TEST_CHECKPOINT_DIR": args.gigaomni_checkpoint_dir,
    }
    for test_name, tp_size in MODEL_TEST_TP_SIZES.items():
        env_variables[f"{test_name.upper()}_TP_SIZE"] = str(tp_size)

    if args.marker in MODEL_TEST_TP_SIZES:
        n_gpus = MODEL_TEST_TP_SIZES[args.marker]
    elif args.type in {"integration", "all"} or args.test_path:
        n_gpus = max(MODEL_TEST_TP_SIZES.values(), default=DEFAULT_N_GPUS)
    else:
        n_gpus = DEFAULT_N_GPUS
    if os.getenv("GIGAVISION_UPDATE_GOLDENS"):
        env_variables["GIGAVISION_UPDATE_GOLDENS"] = os.environ[
            "GIGAVISION_UPDATE_GOLDENS"
        ]

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - Would submit job with:")
        print("=" * 60)
        print(f"Test type: {args.type}")
        print(f"Test path: {test_path}")
        print(f"Test data path: {args.test_data_path}")
        print(f"Gigachat35 checkpoint dir: {args.gigachat35_checkpoint_dir}")
        print(f"GigaVision checkpoint dir: {args.gigavision_checkpoint_dir}")
        print(f"GigaOmni checkpoint dir: {args.gigaomni_checkpoint_dir}")
        print(f"Marker: {args.marker or 'none'}")
        print(f"GPUs: {n_gpus}")
        print("\nEnvironment variables:")
        for key, value in env_variables.items():
            print(f"  {key}={value}")
        print(f"\nScript:\n{script}")
        return 0

    # BaseJob requires n_gpus in {1, 2, 4, 8}. Unit tests don't actually use them.
    test_job = TestSglangPretrainJob(
        user_login="autotest",
        exp_name=f"sglang-pretrain-test | {args.type.upper()}",
        script=script,
        n_gpus=n_gpus,
        extra_env=env_variables,
        queue_name="pretrain-ml-test",
    )

    print("Submitting sglang_pretrain test job...")
    test_job.submit()
    monitor_result = test_job.monitor()

    final_status = monitor_result.final_status
    final_status_name = _extract_status_name(final_status)
    print(f"Final status: {final_status}")

    if final_status_name not in SUCCESS_STATUSES:
        raise AssertionError(f"Job finished unsuccessfully: {final_status}")

    print("Job finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
