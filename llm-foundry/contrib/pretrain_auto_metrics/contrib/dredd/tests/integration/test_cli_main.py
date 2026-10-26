import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from omegaconf import OmegaConf

import dredd

pytestmark = [pytest.mark.integration]


def _write_yaml(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=path)
    return path


def test_main_applies_cli_overrides_and_phase(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        {
            "experiment": {"overwrite_generations": False},
            "datasets": [{"name": "arena-hard-v2.0"}],
            "judge": {"temperature": 0.0},
            "model": {"name": "old-name", "path": "tmp/model"},
            "baseline": {"name": "base-model"},
        },
    )

    captured = {}

    def fake_run(config, phase="full"):
        captured["config"] = config
        captured["phase"] = phase

    monkeypatch.setattr(dredd, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dredd.py",
            "--config",
            str(config_path),
            "--phase",
            "generation",
            "model.name=new-name",
        ],
    )

    dredd.main()

    assert captured["phase"] == "generation"
    assert captured["config"]["model"]["name"] == "new-name"


def test_main_rejects_unknown_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        {
            "experiment": {"overwrite_generations": False},
            "datasets": [{"name": "arena-hard-v2.0"}],
            "judge": {"temperature": 0.0},
            "model": {"name": "old-name", "path": "tmp/model"},
            "baseline": {"name": "base-model"},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dredd.py",
            "--config",
            str(config_path),
            "model.unknown_key=1",
        ],
    )

    with pytest.raises(ValueError, match="Invalid CLI parameters"):
        dredd.main()
