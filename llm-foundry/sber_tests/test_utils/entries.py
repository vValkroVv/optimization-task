import os
from pathlib import Path
from typing import Optional, Dict
import shutil

from omegaconf import OmegaConf, DictConfig

from sber_tests.test_utils.job import JobHandle
from sber_tests.test_utils.misc import run_timestamp


class TestEntry:
    __test__ = False

    def __init__(
            self,
            test_name: str,
            config_dir: str | os.PathLike,
            artifact_dir: str | os.PathLike,
            golden_version: str,
            configs: DictConfig,
            components: Optional[Dict[str, str]],
            checkpoint_pattern: str,
            n_gpus: int,
            start_from_checkpoint: bool = False,
            local_run_dir: None | str | os.PathLike = None,
            allow_to_update_golden: bool = False,
            modality: str = "",
            modality_version: str = ""
    ) -> None:
        self.config_dir = config_dir
        self.artifact_dir = artifact_dir
        self.golden_version = golden_version
        self.components = components
        self.checkpoint_pattern = checkpoint_pattern
        self.n_gpus = n_gpus
        self.start_from_checkpoint = start_from_checkpoint
        self.local_run_dir = local_run_dir
        self.allow_to_update_golden = allow_to_update_golden
        self.test_name = test_name
        self.job: Optional[JobHandle] = None
        self.modality = modality
        self.modality_version = modality_version

        run_overrides = dict(
            run_name=self.id,
            global_train_batch_size=n_gpus,
            save_folder=os.path.join(self.run_path, "ckpt"),
            tensorboard_path=os.path.join(self.run_path, "tb"),
            autoresume=start_from_checkpoint,
        )
        self.config = OmegaConf.merge(*configs, run_overrides)

    def prepare_run_env(self):
        if self.start_from_checkpoint:
            shutil.copytree(self.autoresume_path, self.run_path, symlinks=True)

        OmegaConf.save(self.config, self.config_path)
        
        if self.config.get("gigafsdp_config", None):
            self.config.fsdp_config = self.config.gigafsdp_config
            self.config.fsdp_config.gigafsdp_enabled = True

    def finish(self):
        if self.allow_to_update_golden and not os.path.exists(self.golden_path):
            shutil.copytree(self.run_path, self.golden_path, symlinks=True)

    @property
    def config_path(self):
        return os.path.join(self.config_dir, f"temp-{self.id}.yaml")

    @property
    def id(self):
        parts = [f"{k}={Path(v).stem}" for k, v in self.components.items()]
        return ",".join(parts)

    @property
    def run_path(self):
        artifact_root_dir = self.local_run_dir or self.artifact_dir
        return os.path.join(artifact_root_dir, "run", run_timestamp(), self.id)

    @property
    def checkpoint_id(self):
        checkpoint_params = {
            k: f"{k}={Path(v).stem}" for k, v in self.components.items()
        }
        return self.checkpoint_pattern.format(**checkpoint_params)

    @property
    def autoresume_path(self):
        return os.path.join(self.artifact_dir, "autoresume", self.golden_version, self.checkpoint_id)

    @property
    def golden_path(self):
        if self.modality:
            return os.path.join(self.artifact_dir, "golden", self.modality, self.modality_version, self.checkpoint_id)
        else:
            return os.path.join(self.artifact_dir, "golden", self.golden_version, self.test_name, self.checkpoint_id)
