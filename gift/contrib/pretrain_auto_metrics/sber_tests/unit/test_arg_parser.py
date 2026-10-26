import argparse
from pathlib import Path

import pytest

from node_runner.src.arg_parser import parse_arguments, validate_paths

pytestmark = pytest.mark.unit


class TestValidatePaths:
    def test_ok(self, tmp_path: Path):
        validate_paths(str(tmp_path), None)

    def test_raises_for_missing(self, tmp_path: Path):
        missing = tmp_path / "missing"
        with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
            validate_paths(str(missing), None)

    def test_ok_with_existing_config_file(self, tmp_path: Path):
        config_path = tmp_path / "pam_config.yaml"
        config_path.write_text("run-name: test\n", encoding="utf-8")
        validate_paths(None, str(config_path))

    def test_raises_for_missing_config_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            validate_paths(None, str(tmp_path / "missing.yaml"))

    def test_raises_when_checkpoint_and_config_are_absent(self):
        with pytest.raises(ValueError, match="either --checkpoint or --config"):
            validate_paths(None, None)


class TestParseArguments:
    def test_returns_values(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(
            argparse.ArgumentParser,
            "parse_args",
            lambda self: argparse.Namespace(
                checkpoint=str(tmp_path), mode="autometrics", config=None
            ),
        )
        checkpoint, mode, config = parse_arguments()
        assert checkpoint == str(tmp_path)
        assert mode == "autometrics"
        assert config is None

    def test_validates_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        missing = tmp_path / "not_exists"
        monkeypatch.setattr(
            argparse.ArgumentParser,
            "parse_args",
            lambda self: argparse.Namespace(
                checkpoint=str(missing), mode="inference", config=None
            ),
        )
        with pytest.raises(FileNotFoundError):
            parse_arguments()
