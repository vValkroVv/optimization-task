import json
import runpy
import sys
from pathlib import Path

import pytest

from stand import submit_stand

pytestmark = pytest.mark.unit


def _command_line(
    foundry: Path, stand_root: Path, *positionals: str, region: str = "region-a", without: tuple[str, ...] = ()
) -> list[str]:
    # Positional arguments first, as documented: argparse before Python 3.13 rejects optional positionals
    # given after an option.
    options = {
        "--llmf-root": str(foundry), "--stand-root": str(stand_root), "--user-login": "researcher", "--region": region,
    }
    return [
        "submit_stand.py", *positionals,
        *(argument for name, value in options.items() if name not in without for argument in (name, value)),
    ]


@pytest.mark.parametrize("region", ["region-a", "region-b"])
@pytest.mark.parametrize("explicit_job", [False, True])
def test_submits_from_selected_checkout(foundry, tmp_path, monkeypatch, explicit_job, region):
    stand_root = tmp_path / "own-stand"
    positionals = ("training.yaml", "8", "attention-monitor") if explicit_job else ("training.yaml",)
    monkeypatch.setattr(sys, "argv", _command_line(foundry, stand_root, *positionals, region=region))

    submit_stand.main()

    assert json.loads((foundry / "submitted.json").read_text()) == {
        "user_login": "researcher",
        "exp_name": "attention-monitor" if explicit_job else "metric-sampler-stand",
        "path_to_config": str(stand_root.resolve() / "configs/training.yaml"),
        "n_workers": 8 if explicit_job else 1,
        "region": region,
        "job_logdir": str(stand_root.resolve() / "jobs_logs"),
        "slurm_logdir": str(stand_root.resolve() / "slurm_logs"),
        "enable_auto_track": False,
        "max_retry": None,
        "extra_env": {},
        "queue_name": "pretrain",
        "priority_class": "high",
    }


@pytest.mark.parametrize(
    "missing", ["scripts/train/train.py", "contrib/job_submitter/job_submitter/jobs/llmf/train.py"]
)
def test_rejects_incomplete_checkout(foundry, tmp_path, monkeypatch, capsys, missing):
    (foundry / missing).unlink()
    monkeypatch.setattr(sys, "argv", _command_line(foundry, tmp_path, "training.yaml"))

    with pytest.raises(SystemExit, match="2"):
        submit_stand.main()

    assert f"selected checkout is missing {foundry.resolve() / missing}" in capsys.readouterr().err
    assert not (foundry / "submitted.json").exists()


@pytest.mark.parametrize("omitted", ["--llmf-root", "--stand-root", "--user-login", "--region"])
def test_site_values_have_no_defaults(foundry, tmp_path, monkeypatch, capsys, omitted):
    monkeypatch.setattr(sys, "argv", _command_line(foundry, tmp_path, "training.yaml", without=(omitted,)))

    with pytest.raises(SystemExit, match="2"):
        submit_stand.main()

    assert f"the following arguments are required: {omitted}" in capsys.readouterr().err
    assert not (foundry / "submitted.json").exists()


@pytest.mark.parametrize("n_workers", ["0", "-1"])
def test_rejects_nonpositive_workers(foundry, tmp_path, monkeypatch, capsys, n_workers):
    monkeypatch.setattr(sys, "argv", _command_line(foundry, tmp_path, "training.yaml", n_workers))

    with pytest.raises(SystemExit, match="2"):
        submit_stand.main()

    assert "n_workers must be positive" in capsys.readouterr().err
    assert not (foundry / "submitted.json").exists()


def test_script_entrypoint(foundry, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", _command_line(foundry, tmp_path, "training.yaml"))

    runpy.run_path(str(Path(submit_stand.__file__)), run_name="__main__")

    assert json.loads((foundry / "submitted.json").read_text())["n_workers"] == 1
