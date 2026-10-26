#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Single source of truth for the pinned torchtitan images, shared with the
# job_submitter GIFT jobs and gift_scripts.
GIFT_IMAGES_PATH = REPO_ROOT / "gift_docker" / "images.json"

# mpirun resolves bare `python` to /usr/bin/python (no torchtitan). Use `python3`
# from the NGC image PATH and PYTHONPATH so `tests` is importable.
REMOTE_PYTHON = "python3"
FEATURES_EXCLUDED_TESTS = (
    "cpu_offload+opt_in_bwd+TP+DP+CP",
    # torchcomms sees the outer Cloud MPI job as a single-rank world here.
    "torchcomms_3d_dp+cp+pp+compile",
    "torchcomms_3d_dp+tp+pp+compile",
)
# (test name, description, experiment YAML, dot-path overrides). One job each:
# torchcomms needs the Cloud pytorch2 launcher rather than torchrun.
TORCHCOMMS_TESTS = (
    (
        "torchcomms_3d_dp+cp+pp+compile",
        "FSDP+CP+PP+compile with torchcomms",
        "configs/experiments/llama3_debug_flex_attn.yaml",
        "--comm.mode torchcomms "
        "--parallelism.context_parallel_degree 2 "
        "--parallelism.pipeline_parallel_degree 2 "
        "--compile.enable",
    ),
    (
        "torchcomms_3d_dp+tp+pp+compile",
        "FSDP+TP+PP+compile with torchcomms",
        "configs/experiments/llama3_debug_ce_loss.yaml",
        "--parallelism.full_dtensor "
        "--comm.mode torchcomms "
        "--parallelism.tensor_parallel_degree 2 "
        "--parallelism.pipeline_parallel_degree 2 "
        "--compile.enable",
    ),
)


def _load_cloud_env() -> None:
    if os.environ.get("CLUSTER_KEY"):
        return
    try:
        with open("/proc/1/environ", encoding="utf-8") as f:
            for entry in f.read().split("\0"):
                if "=" in entry:
                    key, value = entry.split("=", 1)
                    os.environ.setdefault(key, value)
    except OSError:
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit TorchTitan integration tests to cloud"
    )
    parser.add_argument(
        "--suite",
        required=True,
        choices=["smoke", "features", "features_torchcomms", "models", "h100"],
    )
    parser.add_argument(
        "--image",
        default=(os.getenv("IMAGE_PATH") or os.getenv("TORCHTITAN_INTEGRATION_IMAGE")),
        help="Override the image pinned in gift_docker/images.json",
    )
    parser.add_argument("--ngpu", type=int, default=int(os.getenv("NGPU", "8")))
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "artifacts-to-be-uploaded/integration"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=os.getenv(
            "TORCHTITAN_CLOUD_ARTIFACT_DIR",
            "artifacts-to-be-uploaded/cloud-job",
        ),
        help="Local CI artifact directory for cloud logs and metadata",
    )
    return parser


def _resolve_image(image: str | None, region: str) -> str:
    if image:
        return image
    images = json.loads(GIFT_IMAGES_PATH.read_text(encoding="utf-8"))
    if region not in images:
        raise SystemExit(
            f"No torchtitan image pinned for region {region!r} in "
            f"{GIFT_IMAGES_PATH}. Set TORCHTITAN_INTEGRATION_IMAGE, IMAGE_PATH, "
            "or pass --image."
        )
    return images[region]


def _validate_artifact_dir(*, output_dir: str, artifact_dir: Path) -> None:
    output_path = Path(output_dir).resolve()
    artifact_path = artifact_dir.resolve()
    if artifact_path == output_path or artifact_path.is_relative_to(output_path):
        raise ValueError(
            "artifact-dir must be outside output-dir so integration tests start "
            f"with an empty output directory: {artifact_path}"
        )


def _resolve_expected_commit(repo_root: Path) -> str:
    expected_commit = os.getenv("CI_COMMIT_SHA")
    if expected_commit is None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Failed to resolve the current commit for {repo_root}"
            ) from exc
        expected_commit = result.stdout.strip()

    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError(f"Invalid expected commit SHA: {expected_commit!r}")
    return expected_commit


def _verified_command(
    *,
    repo_root: str,
    expected_commit: str,
    target: str,
) -> str:
    wrapper = f"{repo_root}/scripts/ci/verify_checkout.py"
    return shlex.join(
        [
            wrapper,
            "--repo-root",
            repo_root,
            "--expected-commit",
            expected_commit,
            target,
        ]
    )


def _build_remote_script(
    *,
    repo_root: str,
    expected_commit: str,
    suite: str,
    output_dir: str,
    ngpu: int,
) -> str:
    # Cloud binary jobs: --wdir sets cwd (mpirun ignores container WORKDIR).
    # llm-foundry uses the same pattern for cloud autometrics jobs.
    if " " in output_dir:
        raise ValueError(f"output_dir must not contain spaces: {output_dir!r}")

    runner = _verified_command(
        repo_root=repo_root,
        expected_commit=expected_commit,
        target=f"{repo_root}/tests/integration_tests/run_tests.py",
    )
    if suite == "smoke":
        test_cmd = (
            f"{REMOTE_PYTHON} {runner} {output_dir} "
            f"--test_suite features --test_name seed_checkpoint --ngpu {ngpu} "
            "--no-parallel"
        )
    elif suite == "features":
        excluded_tests = ",".join(FEATURES_EXCLUDED_TESTS)
        test_cmd = (
            f"{REMOTE_PYTHON} {runner} "
            f"--test_suite features "
            f"--exclude {excluded_tests} "
            f"--ngpu {ngpu} {output_dir}"
        )
    elif suite == "models":
        test_cmd = (
            f"{REMOTE_PYTHON} {runner} --test_suite models --ngpu {ngpu} {output_dir}"
        )
    elif suite == "h100":
        test_cmd = (
            f"{REMOTE_PYTHON} {runner} {output_dir} --test_suite h100 --ngpu {ngpu}"
        )
    else:
        raise ValueError(f"Unknown integration suite: {suite}")

    return f"--wdir {shlex.quote(repo_root)} {test_cmd}"


def _remote_output_dir(repo_root: str, output_dir: str, test_name: str) -> str:
    if os.path.isabs(output_dir):
        return f"{output_dir}/{test_name}"
    return f"{repo_root}/{output_dir}/{test_name}"


def _check_experiment_configs(
    config_files: list[str],
    *,
    raw_overrides: list[tuple[str, str]] | None = None,
    world_size: int | None = None,
) -> None:
    """Validate the experiment YAMLs (torch-free) before the job is scheduled."""
    sys.path.insert(0, str(REPO_ROOT))
    from torchtitan.config import ConfigError, run_pre_submit

    for config_file in config_files:
        try:
            result = run_pre_submit(
                str(REPO_ROOT / config_file), raw_overrides, world_size=world_size
            )
        except ConfigError as exc:
            raise SystemExit(f"{config_file}: {exc}") from exc
        for issue in result.report.warnings:
            print(f"{config_file}: warning: {issue}", file=sys.stderr)
        if not result.ok:
            errors = "\n".join(f"  - {issue}" for issue in result.report.errors)
            raise SystemExit(f"{config_file}: invalid experiment config:\n{errors}")


def _build_torchcomms_scripts(
    *,
    repo_root: str,
    expected_commit: str,
    output_dir: str,
) -> tuple[tuple[str, str, str], ...]:
    if " " in output_dir:
        raise ValueError(f"output_dir must not contain spaces: {output_dir!r}")

    scripts = []
    for test_name, test_descr, config_file, override_args in TORCHCOMMS_TESTS:
        dump_folder = _remote_output_dir(repo_root, output_dir, test_name)
        # verify_checkout.py chdirs to repo_root before exec, so the relative
        # config path resolves the same way run_train.sh resolves it.
        runner = _verified_command(
            repo_root=repo_root,
            expected_commit=expected_commit,
            target=f"{repo_root}/torchtitan/train.py",
        )
        scripts.append(
            (
                test_name,
                test_descr,
                f"{runner} {config_file} --dump_folder {dump_folder} {override_args}",
            )
        )
    return tuple(scripts)


def _extra_env(
    *,
    repo_root: str,
    expected_commit: str,
    suite: str,
    output_dir: str,
    ngpu: int,
) -> dict[str, str]:
    keys = (
        "CI_COMMIT_SHA",
        "CI_COMMIT_SHORT_SHA",
        "CI_COMMIT_REF_NAME",
        "CI_PIPELINE_ID",
        "CI_JOB_ID",
        "CI_PROJECT_PATH",
    )
    env = {key: value for key in keys if (value := os.getenv(key))}
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHON": REMOTE_PYTHON,
            "OUTPUT_DIR": output_dir,
            "NGPU": str(ngpu),
            "TORCHTITAN_EXPECTED_COMMIT": expected_commit,
            "TORCHTITAN_INTEGRATION_SUITE": suite,
            "TORCHTITAN_INTEGRATION_OUTPUT_DIR": output_dir,
            "TORCHTITAN_REPO_ROOT": repo_root,
        }
    )
    return env


def _write_metadata(
    *,
    path: Path,
    suite: str,
    image: str,
    region: str,
    queue: str | None,
    priority_class: str | None,
    ngpu: int,
    script: str,
    final_status: str | None = None,
) -> None:
    lines = [
        f"suite={suite}",
        f"image={image}",
        f"region={region}",
        f"queue={queue or ''}",
        f"priority_class={priority_class or ''}",
        f"ngpu={ngpu}",
        f"script={script}",
    ]
    if final_status is not None:
        lines.append(f"final_status={final_status}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _submit_and_monitor(
    *,
    base_job_cls,
    test_job_cls,
    repo_root: str,
    artifact_dir: Path,
    metadata_name: str,
    log_name: str,
    user_login: str,
    exp_name: str,
    script: str,
    image: str,
    region: str,
    queue: str | None,
    priority_class: str | None,
    ngpu: int,
    suite: str,
    extra_env: dict[str, str],
    script_type: str = "binary",
    processes_per_worker: int | None = None,
) -> str:
    job_kwargs = {
        "user_login": user_login,
        "exp_name": exp_name,
        "script": script,
        "repo_root": repo_root,
        "n_gpus": ngpu,
        "base_image": image,
        "region": region,
        "queue_name": queue,
        "priority_class": priority_class,
        "extra_env": extra_env,
    }
    if script_type != "binary":
        job_kwargs["script_type"] = script_type
    if processes_per_worker is not None:
        job_kwargs["processes_per_worker"] = processes_per_worker

    job = test_job_cls(**job_kwargs)
    try:
        job.submit()
    except TypeError as exc:
        print(f"Cloud job submission failed: {exc}", file=sys.stderr)
        raise

    if job._last_submitted_job is None:
        raise RuntimeError("Cloud job submission failed: no job handle returned.")

    metadata_path = artifact_dir / metadata_name
    _write_metadata(
        path=metadata_path,
        suite=suite,
        image=image,
        region=region,
        queue=queue,
        priority_class=priority_class,
        ngpu=ngpu,
        script=script,
    )

    result = job.monitor(
        # Public API registration can lag behind successful submission.
        active_statuses={"Pending", "Running", "job not found"},
        poll_interval_seconds=60,
        logs_tail=2000,
        logs_verbose=True,
    )
    final_status = base_job_cls._extract_status_name(result.final_status)

    if result.logs:
        (artifact_dir / log_name).write_text(result.logs, encoding="utf-8")
    _write_metadata(
        path=metadata_path,
        suite=suite,
        image=image,
        region=region,
        queue=queue,
        priority_class=priority_class,
        ngpu=ngpu,
        script=script,
        final_status=final_status,
    )
    return final_status


def main() -> int:
    args = _parser().parse_args()
    _load_cloud_env()

    sys.path.insert(0, str(REPO_ROOT / "contrib" / "job_submitter"))
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))

    try:
        from job_submitter.jobs.base import BaseJob
        from test_job import TestJobTorchTitan
    except ImportError:
        print(
            "job_submitter is not installed. Run pip install ./contrib/job_submitter.",
            file=sys.stderr,
        )
        return 2

    repo_root = os.environ.get("SR008_REMOTE_DIR") or str(REPO_ROOT)
    expected_commit = _resolve_expected_commit(REPO_ROOT)
    artifact_dir = Path(args.artifact_dir)
    _validate_artifact_dir(
        output_dir=args.output_dir,
        artifact_dir=artifact_dir,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_suffix = os.getenv("CI_COMMIT_SHORT_SHA", "local")
    region = os.getenv("TORCHTITAN_CLOUD_REGION", "SR008")
    image = _resolve_image(args.image, region)
    queue = os.getenv("TORCHTITAN_CLOUD_QUEUE")
    priority_class = os.getenv("TORCHTITAN_CLOUD_PRIORITY_CLASS", "medium")
    user_login = os.getenv("TORCHTITAN_CLOUD_USER", "autotest")
    os.environ.setdefault("CLUSTER_KEY", region)

    if args.suite == "features_torchcomms":
        sys.path.insert(0, str(REPO_ROOT))
        from torchtitan.config.overrides import parse_argv

        for _, _, config_file, override_args in TORCHCOMMS_TESTS:
            _, overrides = parse_argv([config_file, *shlex.split(override_args)])
            _check_experiment_configs(
                [config_file], raw_overrides=overrides, world_size=args.ngpu
            )
        failed = False
        for test_name, test_descr, script in _build_torchcomms_scripts(
            repo_root=repo_root,
            expected_commit=expected_commit,
            output_dir=args.output_dir,
        ):
            exp_name = f"torchtitan integration {test_name} | {run_suffix}"

            print(f"Submitting TorchTitan integration suite: {args.suite}")
            print(f"Test: {test_name} ({test_descr})")
            print(f"Image: {image}")
            print(f"Region: {region}")
            print(f"Queue: {queue or '<default>'}")
            print(f"Priority: {priority_class or '<default>'}")
            print(f"NGPU: {args.ngpu}")
            print("Script type: pytorch2")
            print(f"Processes per worker: {args.ngpu}")
            print(f"Remote script: {script}")

            env = _extra_env(
                repo_root=repo_root,
                expected_commit=expected_commit,
                suite=args.suite,
                output_dir=args.output_dir,
                ngpu=args.ngpu,
            )
            env["TORCHTITAN_INTEGRATION_TEST_NAME"] = test_name
            env["TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD"] = "torchrun"
            env["TORCHCOMM_NCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD"] = "tcpstore"
            env["TORCH_TRACE"] = (
                _remote_output_dir(repo_root, args.output_dir, test_name)
                + "/compile_trace"
            )

            try:
                final_status = _submit_and_monitor(
                    base_job_cls=BaseJob,
                    test_job_cls=TestJobTorchTitan,
                    repo_root=repo_root,
                    artifact_dir=artifact_dir,
                    metadata_name=f"cloud-job-{test_name}.txt",
                    log_name=f"cloud-job-{test_name}.log",
                    user_login=user_login,
                    exp_name=exp_name,
                    script=script,
                    image=image,
                    region=region,
                    queue=queue,
                    priority_class=priority_class,
                    ngpu=args.ngpu,
                    suite=args.suite,
                    extra_env=env,
                    script_type="pytorch2",
                    processes_per_worker=args.ngpu,
                )
            except (RuntimeError, TypeError) as exc:
                print(str(exc), file=sys.stderr)
                return 1

            print(f"\nFinal status for {test_name}: {final_status}")
            if final_status != "Completed":
                failed = True

        return 1 if failed else 0

    exp_name = f"torchtitan integration {args.suite} | {run_suffix}"
    script = _build_remote_script(
        repo_root=repo_root,
        expected_commit=expected_commit,
        suite=args.suite,
        output_dir=args.output_dir,
        ngpu=args.ngpu,
    )

    print(f"Submitting TorchTitan integration suite: {args.suite}")
    print(f"Image: {image}")
    print(f"Region: {region}")
    print(f"Queue: {queue or '<default>'}")
    print(f"Priority: {priority_class or '<default>'}")
    print(f"NGPU: {args.ngpu}")
    print(f"Remote script: {script}")

    try:
        final_status = _submit_and_monitor(
            base_job_cls=BaseJob,
            test_job_cls=TestJobTorchTitan,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            metadata_name="cloud-job.txt",
            log_name="cloud-job.log",
            user_login=user_login,
            exp_name=exp_name,
            script=script,
            image=image,
            region=region,
            queue=queue,
            priority_class=priority_class,
            ngpu=args.ngpu,
            suite=args.suite,
            extra_env=_extra_env(
                repo_root=repo_root,
                expected_commit=expected_commit,
                suite=args.suite,
                output_dir=args.output_dir,
                ngpu=args.ngpu,
            ),
        )
    except (RuntimeError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"\nFinal status: {final_status}")
    return 0 if final_status == "Completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
