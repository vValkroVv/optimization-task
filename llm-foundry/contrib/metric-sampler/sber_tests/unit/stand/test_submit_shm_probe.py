import json
import runpy
import sys
from pathlib import Path

import pytest

from stand import submit_shm_probe

pytestmark = pytest.mark.unit


def test_submits_the_probe_of_the_stand_root_as_a_one_node_job(
    foundry, tmp_path, monkeypatch, site_arguments, common_job_options
):
    monkeypatch.setattr(
        sys, "argv", ["submit_shm_probe.py", *site_arguments, "--base-image", "/images/training.sqsh"]
    )

    submit_shm_probe.main()

    probe = tmp_path.resolve() / "stand" / "shm_node_probe.py"
    assert json.loads((foundry / "submitted.json").read_text()) == {
        "exp_name": "metric-sampler-shm-probe",
        "script": f"python -m torch.distributed.run --standalone --nproc-per-node 8 {probe}",
        "base_image": "/images/training.sqsh",
        "n_workers": 1,
        "n_gpus": 8,
        **common_job_options,
    }


def test_the_image_has_no_default(foundry, monkeypatch, capsys, site_arguments):
    monkeypatch.setattr(sys, "argv", ["submit_shm_probe.py", *site_arguments])

    with pytest.raises(SystemExit, match="2"):
        submit_shm_probe.main()

    assert "the following arguments are required: --base-image" in capsys.readouterr().err
    assert not (foundry / "submitted.json").exists()


def test_script_entrypoint(foundry, monkeypatch, site_arguments):
    monkeypatch.setattr(sys, "argv", ["submit_shm_probe.py", *site_arguments, "--base-image", "/images/training.sqsh"])

    runpy.run_path(str(Path(submit_shm_probe.__file__)), run_name="__main__")

    assert json.loads((foundry / "submitted.json").read_text())["n_gpus"] == 8
