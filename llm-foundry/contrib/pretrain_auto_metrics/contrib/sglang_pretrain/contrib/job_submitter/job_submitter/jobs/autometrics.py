import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from job_submitter.asserts.autometrics import AutometricsJobAssertor
from job_submitter.core.job_config import JobTemplate
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import PAM_BASE_IMAGES


@dataclass(slots=True, kw_only=True)
class AutometricsJob(BaseJob, AutometricsJobAssertor):
    checkpoint_dir: str
    path_to_config: Optional[str] = None
    mode: str = "autometrics"
    sglang_tag: str = "custom"

    BASE_IMAGES: ClassVar[dict[str, str]] = PAM_BASE_IMAGES

    def _move_config_to_artifacts(self) -> None:
        artifacts_path = Path(self.checkpoint_dir) / "artifacts"
        pam_config_path = artifacts_path / "pam_config.yaml"

        if not self.path_to_config:
            assert pam_config_path.exists() and pam_config_path.is_file(), (
                self._pretty_params_print(
                    {"path_to_config": ["<path to valid config file>"]}
                )
            )
            return

        config_to_replace = Path(self.path_to_config)
        assert config_to_replace.exists() and config_to_replace.is_file(), (
            self._pretty_params_print(
                {"path_to_config": ["<path to valid config file>"]}
            )
        )

        artifacts_path.mkdir(parents=True, exist_ok=True)

        backup_path = pam_config_path.with_suffix(".yaml.backup")

        if pam_config_path.exists() or pam_config_path.is_symlink():
            if backup_path.exists() or backup_path.is_symlink():
                backup_path.unlink()

            if pam_config_path.is_symlink():
                pam_config_path.unlink()
            else:
                pam_config_path.replace(backup_path)

        shutil.copy2(config_to_replace, pam_config_path)

    @property
    def base_images(self) -> dict[str, dict[str, str]]:
        return self.BASE_IMAGES

    def _resolve_base_image(self) -> Optional[str]:
        assert self.sglang_tag in self.base_images, self._pretty_params_print(
            {"sglang_tag": ["opensource", "custom"]}
        )
        return self.base_image or self.base_images[self.sglang_tag][self.region]

    def _tracking_config_path(self, cfg: JobTemplate) -> Optional[str]:
        # ``path_to_config`` is resolved by the generic base hook; when it is absent
        # the config has been copied into the checkpoint artifacts by
        # ``_move_config_to_artifacts``, so fall back to that canonical location.
        # NB: bare ``super()`` is unusable in ``@dataclass(slots=True)`` classes.
        return BaseJob._tracking_config_path(self, cfg) or str(
            Path(self.checkpoint_dir) / "artifacts" / "pam_config.yaml"
        )

    def _before_submit(self) -> None:
        self.exp_name = self.exp_name + f" | {self.mode} mode"
        self._move_config_to_artifacts()

    def submit(self) -> BaseJob:
        cfg = self.to_config()
        self._anonimize_config(self.path_to_config)
        self.assert_job(self, cfg)
        self._before_submit()
        return self._submit_config(cfg)
