import argparse

import pytest

from stand import job_options

pytestmark = pytest.mark.unit


def _parse(*command_line: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    job_options.add_arguments(parser)
    return parser.parse_args(command_line)


def test_roots_are_expanded_to_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    arguments = _parse("--llmf-root", "~/foundry", "--stand-root", "stand", "--user-login", "u", "--region", "r")

    assert arguments.llmf_root == (tmp_path / "foundry").resolve()
    assert arguments.stand_root.is_absolute() and arguments.stand_root.name == "stand"


def test_job_options_name_the_logs_of_the_stand_root_and_leave_the_job_unattended(tmp_path):
    arguments = _parse(
        "--llmf-root", str(tmp_path), "--stand-root", str(tmp_path / "stand"), "--user-login", "researcher",
        "--region", "region-a",
    )

    assert job_options.submission_options(arguments) == {
        "user_login": "researcher",
        "region": "region-a",
        "job_logdir": str(tmp_path.resolve() / "stand" / "jobs_logs"),
        "slurm_logdir": str(tmp_path.resolve() / "stand" / "slurm_logs"),
        "enable_auto_track": False,
        "max_retry": None,
        "extra_env": {},
        "queue_name": "pretrain",
        "priority_class": "high",
    }
