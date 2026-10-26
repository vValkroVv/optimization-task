import json
import runpy
import sys
from pathlib import Path

import pytest

from stand import submit_kernel_check

pytestmark = pytest.mark.unit


def _command_line(site_arguments: list[str], repo: Path, *check: str) -> list[str]:
    return [
        "submit_kernel_check.py", *site_arguments, "--base-image", "/images/training.sqsh", "--repo-root", str(repo),
        *check,
    ]


@pytest.mark.parametrize(
    ("check", "expected_check"),
    [
        ([], "check_activation_kernel.py "),
        (["check_token_loss_kernel.py"], "check_token_loss_kernel.py "),
        (["check_attention_kernel.py", "--correctness-only"], "check_attention_kernel.py --correctness-only"),
        (["check_activation_kernel.py", "--baseline-ref", "abc123", "--timing-only"],
         "check_activation_kernel.py --baseline-ref abc123 --timing-only"),
    ],
)
def test_submits_the_check_run_from_the_repository_checkout_as_a_one_node_job(
    foundry, tmp_path, monkeypatch, site_arguments, common_job_options, check, expected_check
):
    repo = tmp_path / "repository"
    monkeypatch.setattr(sys, "argv", _command_line(site_arguments, repo, *check))

    submit_kernel_check.main()

    resolved = repo.resolve()
    assert json.loads((foundry / "submitted.json").read_text()) == {
        "exp_name": "metric-sampler-kernel-check",
        "script": f"cd {resolved} && PYTHONPATH={resolved} python {resolved}/stand/{expected_check}",
        "base_image": "/images/training.sqsh",
        "n_workers": 1,
        "n_gpus": 8,
        **common_job_options,
    }


@pytest.mark.parametrize("omitted", ["--repo-root", "--base-image"])
def test_the_repository_and_the_image_have_no_defaults(foundry, tmp_path, monkeypatch, capsys, site_arguments, omitted):
    command_line = _command_line(site_arguments, tmp_path)
    index = command_line.index(omitted)
    del command_line[index:index + 2]
    monkeypatch.setattr(sys, "argv", command_line)

    with pytest.raises(SystemExit, match="2"):
        submit_kernel_check.main()

    assert f"the following arguments are required: {omitted}" in capsys.readouterr().err
    assert not (foundry / "submitted.json").exists()


def test_script_entrypoint(foundry, tmp_path, monkeypatch, site_arguments):
    monkeypatch.setattr(sys, "argv", _command_line(site_arguments, tmp_path))

    runpy.run_path(str(Path(submit_kernel_check.__file__)), run_name="__main__")

    assert json.loads((foundry / "submitted.json").read_text())["exp_name"] == "metric-sampler-kernel-check"
