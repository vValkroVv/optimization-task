import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

import node_runner.src.config_manager as config_manager_module
from node_runner.src.config_manager import ConfigManager

pytestmark = pytest.mark.unit


def _make_manager_config(tmp_path: Path, *, with_worker_num: bool) -> SimpleNamespace:
    dist_init_path = tmp_path / "mpimaster.host"
    worker_num_file_path = tmp_path / "mpiworker.num"

    dist_init_path.write_text("127.0.0.1\n127.1.1.1\n", encoding="utf-8")
    if with_worker_num:
        worker_num_file_path.write_text("2\n", encoding="utf-8")

    return SimpleNamespace(
        autometrics=SimpleNamespace(
            globals=SimpleNamespace(
                dist_init_path=str(dist_init_path),
                worker_num_file_path=str(worker_num_file_path),
            )
        )
    )


class TestConfigManagerEnvironment:
    def test_get_environment_info_cloud_with_worker_count_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        manager = ConfigManager()
        manager.config = _make_manager_config(tmp_path, with_worker_num=True)

        monkeypatch.setenv("MLS_JOB_N_GPU", "8")
        monkeypatch.setattr(socket, "gethostname", lambda: "job_id-mpimaster-0")

        manager._get_environment_info()

        assert manager.is_cloud is True
        assert manager.is_slurm is False
        assert manager.job_id == "job_id"
        assert manager.node_name == "mpimaster-0"
        assert manager.global_dist_init_host == "127.0.0.1"
        assert manager.global_node_rank == 0
        assert manager.global_nnodes == 3
        assert manager.gpus_per_node == 8

    def test_get_environment_info_cloud_without_worker_count_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        manager = ConfigManager()
        manager.config = _make_manager_config(tmp_path, with_worker_num=False)

        monkeypatch.setenv("MLS_JOB_N_GPU", "4")
        monkeypatch.setattr(socket, "gethostname", lambda: "job_id-mpimaster-0")

        manager._get_environment_info()

        assert manager.is_cloud is True
        assert manager.global_nnodes == 1
        assert manager.global_node_rank == 0
        assert manager.gpus_per_node == 4

    def test_config_manager_broken_config(self, tmp_path: Path):
        broken_path = tmp_path / "mock_broken_config.yml"
        broken_path.write_text("broken: [", encoding="utf-8")
        manager = ConfigManager()

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            manager.setup(checkpoint_path=broken_path)

    def test_get_environment_info_local_for_external_api(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        manager = ConfigManager()
        manager.config = SimpleNamespace(runs_without_checkpoint=True)
        monkeypatch.setattr(manager, "_is_cloud_env", lambda: False)
        monkeypatch.setattr(manager, "_is_slurm_env", lambda: False)

        manager._get_environment_info()

        assert manager.job_id == "local"
        assert manager.node_name == "mpimaster-0"
        assert manager.global_nnodes == 1
        assert manager.global_dist_init_host == "127.0.0.1"
        assert manager.global_node_rank == 0
        assert manager.gpus_per_node == 1

    @pytest.mark.parametrize(
        ("rank", "expected"),
        [
            (0, True),
            (1, False),
            (None, False),
        ],
    )
    def test_is_master_follows_global_node_rank(self, rank, expected):
        manager = ConfigManager()
        manager.global_node_rank = rank

        assert manager.is_master is expected


class TestConfigManagerPaths:
    def test_load_config_uses_checkpoint_artifacts_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        checkpoint_path = tmp_path / "checkpoint"
        config_path = checkpoint_path / "artifacts" / "pam_config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("run-name: test\n", encoding="utf-8")
        loaded = object()
        monkeypatch.setattr(
            config_manager_module.NodeRunnerConfigModel,
            "load",
            lambda path: loaded,
        )
        manager = ConfigManager()

        manager._load_config(config_path=None, checkpoint_path=checkpoint_path)

        assert manager.config is loaded
        assert manager.config_path == config_path

    def test_load_config_uses_explicit_config_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        config_path = tmp_path / "explicit.yaml"
        config_path.write_text("run-name: test\n", encoding="utf-8")
        loaded = object()
        monkeypatch.setattr(
            config_manager_module.NodeRunnerConfigModel,
            "load",
            lambda path: loaded,
        )
        manager = ConfigManager()

        manager._load_config(config_path=config_path, checkpoint_path=None)

        assert manager.config is loaded
        assert manager.config_path == config_path

    def test_set_paths_external_api_writes_config_snapshot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        saved = {}

        class _Config:
            is_external_api_mode = True
            autometrics = SimpleNamespace(inference=SimpleNamespace(model="api-model"))

            def model_dump(self, **kwargs):
                saved["model_dump_kwargs"] = kwargs
                return {"run-name": "api"}

        module_file = tmp_path / "repo" / "node_runner" / "src" / "config_manager.py"
        monkeypatch.setattr(config_manager_module, "__file__", module_file.as_posix())
        monkeypatch.setattr(
            config_manager_module.OmegaConf,
            "save",
            lambda config, f: saved.update({"config": config, "path": f}),
        )
        manager = ConfigManager()
        manager.config = _Config()

        manager._set_paths(checkpoint_path=None)

        assert manager.checkpoint_path == tmp_path / "repo" / "api-model"
        assert manager.artifacts_path == manager.checkpoint_path
        assert saved["config"] == {"run-name": "api"}
        assert saved["path"] == manager.artifacts_path / "pam_config.yaml"
        assert saved["model_dump_kwargs"]["by_alias"] is True

    def test_set_paths_requires_checkpoint_for_non_external_api(self):
        manager = ConfigManager()
        manager.config = SimpleNamespace(is_external_api_mode=False)

        with pytest.raises(ValueError, match="checkpoint_path is required"):
            manager._set_paths(checkpoint_path=None)

    def test_set_paths_rejects_missing_checkpoint(self, tmp_path: Path):
        manager = ConfigManager()
        manager.config = SimpleNamespace(is_external_api_mode=False)

        with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
            manager._set_paths(checkpoint_path=tmp_path / "missing")
