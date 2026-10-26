from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from enum import Enum

from job_submitter.core.job_kill import kill_job
from job_submitter.jobs.any_script import AnyScriptJob
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.gift.autometrics import AutometricsJobGIFT
from job_submitter.jobs.gift.autometrics_external_api import (
    AutometricsJobGIFTExternalAPI,
)
from job_submitter.jobs.gift.train import TrainJobGIFT
from job_submitter.jobs.infinite import InfiniteJob
from job_submitter.jobs.llmf.autometrics import AutometricsJobLLMF
from job_submitter.jobs.llmf.train import TrainJobLLMF
from job_submitter.jobs.llmf.warmup import WarmupJobLLMF
from job_submitter.jobs.pam.autometrics import AutometricsJobPAM
from job_submitter.jobs.pam.autometrics_external_api import (
    AutometricsJobPAMExternalAPI,
)
from job_submitter.jobs.slurm_cpu import SlurmCpuJob
from job_submitter.jobs.verl.autometrics import AutometricsJobVERL
from job_submitter.jobs.verl.cluster import RayClusterJobVERL
from job_submitter.jobs.verl.train import RayTrainJobVERL


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        required_prefix = "the following arguments are required:"
        if message.startswith(required_prefix):
            missing = message.removeprefix(required_prefix).strip()
            message = f"Missing required arguments: {missing}"
        super().error(message)


class JobType(str, Enum):
    gift_autometrics = "gift.autometrics"
    gift_autometrics_external_api = "gift.autometrics_external_api"
    gift_train = "gift.train"
    llmf_train = "llmf.train"
    llmf_autometrics = "llmf.autometrics"
    llmf_warmup = "llmf.warmup"
    verl_train = "verl.train"
    verl_autometrics = "verl.autometrics"
    verl_cluster = "verl.cluster"
    pam_autometrics = "pam.autometrics"
    pam_autometrics_external_api = "pam.autometrics_external_api"
    any_script = "any_script"
    slurm_cpu = "slurm_cpu"
    infinite = "infinite"

    @property
    def job_class(self) -> type[BaseJob]:
        return _JOB_CLASSES[self]


_JOB_CLASSES: dict[JobType, type[BaseJob]] = {
    JobType.gift_autometrics: AutometricsJobGIFT,
    JobType.gift_autometrics_external_api: AutometricsJobGIFTExternalAPI,
    JobType.gift_train: TrainJobGIFT,
    JobType.llmf_train: TrainJobLLMF,
    JobType.llmf_autometrics: AutometricsJobLLMF,
    JobType.llmf_warmup: WarmupJobLLMF,
    JobType.verl_train: RayTrainJobVERL,
    JobType.verl_autometrics: AutometricsJobVERL,
    JobType.verl_cluster: RayClusterJobVERL,
    JobType.pam_autometrics: AutometricsJobPAM,
    JobType.pam_autometrics_external_api: AutometricsJobPAMExternalAPI,
    JobType.any_script: AnyScriptJob,
    JobType.slurm_cpu: SlurmCpuJob,
    JobType.infinite: InfiniteJob,
}


def kill_command(job_id: str) -> int:
    try:
        result = kill_job(job_id)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result)
    return 0


def submit_command(spec: str, job_type: JobType) -> int:
    try:
        job = job_type.job_class.from_spec(spec)
        job.submit()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    handle = job._last_submitted_job
    job_id = handle.job_name if handle is not None else None
    if job_id:
        print(f"PIPELINE_JOB_ID={job_id}")
    return 0


def _cmd_kill(args: argparse.Namespace) -> int:
    return kill_command(args.job_id)


def _cmd_submit(args: argparse.Namespace) -> int:
    return submit_command(args.spec, args.job_type)


def build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        prog="job_submitter",
        description="Job submitter CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    kill_parser = subparsers.add_parser("kill", help="Kill a job by id/name")
    kill_parser.add_argument("job_id", help="Job id/name to kill")
    kill_parser.set_defaults(func=_cmd_kill)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit a job from a spec YAML",
    )
    submit_parser.add_argument(
        "spec",
        help="Path to a job spec YAML (job __init__ fields)",
    )
    submit_parser.add_argument(
        "--job-type",
        required=True,
        type=JobType,
        choices=[job_type.value for job_type in JobType],
        help="Job type",
    )
    submit_parser.set_defaults(func=_cmd_submit)

    return parser


def app(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return int(app(argv) or 0)
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        # Keep compatibility with callers that surface Click-like exceptions,
        # without depending on Click or Typer in the argparse implementation.
        if type(exc).__name__ == "Abort":
            print("Aborted!", file=sys.stderr)
            return 1

        exit_code = getattr(exc, "exit_code", None)
        if exit_code is not None:
            show = getattr(exc, "show", None)
            if callable(show):
                show()
            return int(exit_code)
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
