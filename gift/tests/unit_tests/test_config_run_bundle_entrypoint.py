"""Launch provenance is shared and validated before rank-zero filesystem writes."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest

from torchtitan.config import save
from torchtitan.config.cli import LoadedConfig
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.validation import RuntimeInfo


def loaded_config(rank=0) -> LoadedConfig:
    from torchtitan.config import load_document

    document, _ = load_document("configs/experiments/llama3_debug.yaml")
    return cast(
        LoadedConfig,
        SimpleNamespace(
            trainer_config=SimpleNamespace(dump_folder="outputs"),
            source_document=document.data,
            runtime=SimpleNamespace(rank=rank),
            applied_overrides={"training.steps": 10},
            import_graph=SimpleNamespace(to_dict=lambda: {"path": "experiment.yaml"}),
            preflight_report=SimpleNamespace(to_dict=lambda: {"issues": []}),
        ),
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_every_rank_checks_round_trip_before_bundle_write(rank):
    loaded = loaded_config(rank)
    with (
        patch.object(save, "get_typed_document", return_value={}),
        patch.object(
            save, "check_round_trip", side_effect=ConfigSchemaError("bad config")
        ),
        patch.object(save, "_write_run_bundle") as write,
    ):
        logger = Mock()
        with pytest.raises(SystemExit) as exc:
            save.save_loaded_run_bundle(loaded, logger=logger)
    assert exc.value.code == 1
    write.assert_not_called()
    logger.error.assert_called_once()


def test_nonzero_rank_validates_without_filesystem_write():
    with (
        patch.object(save, "get_typed_document", return_value={}),
        patch.object(save, "check_round_trip") as validate,
        patch.object(save, "_write_run_bundle") as write,
    ):
        result = save.save_loaded_run_bundle(loaded_config(1), logger=Mock())
    assert result is None
    validate.assert_called_once()
    write.assert_not_called()


def test_bundle_uses_actual_process_invocation():
    command = [
        "python3",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=8",
        "gift_scripts/train.py",
    ]
    with (
        patch.object(save.sys, "orig_argv", command),
        patch.object(save, "get_typed_document", return_value={}),
        patch.object(save, "check_round_trip"),
        patch.object(
            save,
            "_write_run_bundle",
            return_value="outputs/config/resolved_config.yaml",
        ) as write,
    ):
        result = save.save_loaded_run_bundle(loaded_config(), logger=Mock())
        assert result is not None
        assert result.endswith("resolved_config.yaml")
    assert write.call_args.kwargs["command"] == command


def test_filesystem_error_becomes_clean_launch_failure():
    with (
        patch.object(save, "get_typed_document", return_value={}),
        patch.object(save, "check_round_trip"),
        patch.object(save, "_write_run_bundle", side_effect=OSError("disk full")),
    ):
        with pytest.raises(SystemExit) as exc:
            save.save_loaded_run_bundle(loaded_config(), logger=Mock())
    assert exc.value.code == 1


def test_metadata_records_argv_and_only_allowlisted_environment(monkeypatch):
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("PRIVATE_ACCESS_TOKEN", "must-not-be-recorded")
    command = ["python3", "train.py", "config with spaces.yaml"]
    with (
        patch.object(save, "_get_git_info", return_value={}),
        patch.object(save, "_get_package_versions", return_value={}),
    ):
        metadata = save.collect_metadata(command=command, runtime=RuntimeInfo())
    assert metadata["argv"] == command
    assert metadata["command"].endswith("'config with spaces.yaml'")
    assert metadata["environment"]["NGPU"] == "8"
    assert "PRIVATE_ACCESS_TOKEN" not in metadata["environment"]
