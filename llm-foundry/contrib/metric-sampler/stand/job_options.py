from __future__ import annotations

import argparse
from pathlib import Path

QUEUE_NAME = "pretrain"
PRIORITY_CLASS = "high"
GPUS_PER_NODE = 8


def resolved_path(text: str) -> Path:
    return Path(text).expanduser().resolve()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llmf-root", type=resolved_path, required=True,
                        help="llm-foundry checkout that provides the job submitter")
    parser.add_argument("--stand-root", type=resolved_path, required=True,
                        help="directory that receives the job and Slurm logs")
    parser.add_argument("--user-login", required=True)
    parser.add_argument("--region", required=True)


def submission_options(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        "user_login": arguments.user_login,
        "region": arguments.region,
        "job_logdir": str(arguments.stand_root / "jobs_logs"),
        "slurm_logdir": str(arguments.stand_root / "slurm_logs"),
        "enable_auto_track": False,
        "max_retry": None,
        "extra_env": {},
        "queue_name": QUEUE_NAME,
        "priority_class": PRIORITY_CLASS,
    }
