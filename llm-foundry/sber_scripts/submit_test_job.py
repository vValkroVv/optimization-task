#!/usr/bin/env python3

import os
import sys
from pathlib import Path

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

# ruff: noqa: E402
from sber_tests.test_utils.argfilter import get_test_filter_args
from sber_tests.test_utils.argument_parser import create_parser
from sber_tests.test_utils.git_utils import make_run_name_suffix
from sber_tests.test_utils.log_analysis import print_pytest_logs
from sber_tests.test_utils.test_runner import (
    run_integration_tests_on_cluster,
    run_unit_tests_on_cluster,
)

N_GPUS = 8

CODE_PATH = str(Path(__file__).absolute().parent.parent)


def main():
    parser = create_parser(N_GPUS)

    args = parser.parse_args()

    test_filter = get_test_filter_args(CODE_PATH, args.test_classes, args.modality)

    if args.modality == 'UNIT':
        base_pytest_cmd = f"pytest {CODE_PATH}/sber_tests/unit/ -v --tb=short"
    else:
        base_pytest_cmd = f"pytest {CODE_PATH}/sber_tests/integration/test_local_jobs.py -s -vvv"

    if test_filter:
        if test_filter.startswith('-k'):
            script = f"{base_pytest_cmd} {test_filter}"
        elif args.modality == 'UNIT':
            script = f"pytest {test_filter} -v --tb=short"
        else:
            script = f"pytest {test_filter} -s -vvv"
    else:
        script = base_pytest_cmd

    run_suffix = make_run_name_suffix(CODE_PATH)
    modality_suffix = f" | {args.modality}" if args.modality != 'ALL' else ""
    test_classes_suffix = f" | {','.join(args.test_classes)}" if args.test_classes else ""
    job_desc = f"autotest{modality_suffix}{test_classes_suffix} | {run_suffix}"

    env_variables = {
        "LOCAL_RUN_DIR": CODE_PATH,
        "XFAIL_VISION": os.getenv("XFAIL_VISION", "0"),
        "TESTS_PARALLEL": os.getenv("TESTS_PARALLEL", "0"),
        "USE_FA3": os.getenv("USE_FA3", "0"),
        "XFAIL_SFT": os.getenv("XFAIL_SFT", "0"),
        "XFAIL_TTS": os.getenv("XFAIL_TTS", "0"),
        "TEST_MODALITY": args.modality,
        # The image ships a warmed DeepGEMM JIT cache, but it does not cover every
        # kernel a test may reach (the DSA fp8 selection path, for one). Without this
        # DeepGEMM aborts on a cache miss instead of compiling the kernel once.
        "DG_CACHE_FORCE_SKIP": os.getenv("DG_CACHE_FORCE_SKIP", "1"),
        # torch symmetric memory (async TP) binds a unix socket at <tmpdir>/symm_mem-*,
        # taking tmpdir from TMPDIR, then TEMP, then TEMPDIR. Whatever the runner leaves
        # in those points at a directory this job does not have, and the bind fails with
        # "Failed to bind socket: No such file or directory".
        "TMPDIR": "/tmp",
        "TEMP": "/tmp",
        "TEMPDIR": "/tmp",
    }

    if args.modality == 'SFT':
        env_variables["XFAIL_SFT"] = os.getenv("XFAIL_SFT", "1")
    elif args.modality == 'VISION':
        env_variables["XFAIL_VISION"] = os.getenv("XFAIL_VISION", "0")
    elif args.modality == 'TTS':
        env_variables["XFAIL_TTS"] = os.getenv("XFAIL_TTS", "0")

    print(f"Running modality: {args.modality}")
    print(f"Pytest command: {script}")
    print(f"Job description: {job_desc}")
    print(f"Environment variables: {env_variables}")

    runner_func = (run_unit_tests_on_cluster if args.modality == 'UNIT'
                   else run_integration_tests_on_cluster)

    kwargs = {
        'job_desc': job_desc,
        'script': script,
        'env_variables': env_variables,
        'print_logs_callback': lambda: print_pytest_logs(modality=args.modality),
        'args': args,
    }

    if args.modality == 'UNIT':
        kwargs['code_path'] = CODE_PATH

    final_status, exit_code = runner_func(**kwargs)

    if exit_code != 0:
        print(f"[FAIL] Job failed with status: {final_status}")
        exit(1)
    else:
        print("Job completed successfully")


if __name__ == '__main__':
    main()