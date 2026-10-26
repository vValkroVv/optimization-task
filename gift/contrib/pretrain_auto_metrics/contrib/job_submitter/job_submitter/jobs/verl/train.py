from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from job_submitter.asserts.verl_train import VerlTrainJobAssertor
from job_submitter.core.job_config import JobTemplate
from job_submitter.core.verl_tracking import (
    build_verl_tracking_config,
    resolve_verl_logs_path,
)
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.verl.base import RayBaseJobVERL
from omegaconf import OmegaConf


@dataclass(slots=True, kw_only=True)
class RayTrainJobVERL(RayBaseJobVERL, VerlTrainJobAssertor):
    script_path: str
    runtime_env_config_path: str
    cpus_per_task: Optional[int] = 192
    pam_config_path: Optional[str] = None
    pam_n_workers: int = 2
    pam_sglang_tag: str = "custom"
    pam_queue_name: Optional[str] = None
    pam_base_image: Optional[str] = None
    extra_mounts: Optional[str] = "/dev/shm:/dev/shm"

    def _script(self) -> str:
        if self.submission_type == "cloud":
            return f"bash {self.verl_root}/sber_src/submit/setup_ray_cluster/cloud.sh"
        else:
            return f"bash {self.verl_root}/sber_src/submit/setup_ray_cluster/slurm.sh"

    def _tracking_config_path(self, _cfg: JobTemplate) -> str | None:
        return self.pam_config_path or self.script_path

    def _tracking_script_path(self, _cfg: JobTemplate) -> str | None:
        return self.script_path

    def _tracking_merged_config(self, cfg: JobTemplate) -> Optional[dict[str, Any]]:
        return build_verl_tracking_config(
            self.script_path,
            pam_config_path=self.pam_config_path,
            runtime_env_path=self.runtime_env_config_path,
            exp_name_override=self.exp_name,
            job_logdir=cfg.job_logdir or self._resolve_job_logdir(),
        )

    def _resolve_tb_log_dir(self) -> str:
        logs_path = resolve_verl_logs_path(
            self.script_path,
            runtime_env_path=self.runtime_env_config_path,
            exp_name_override=self.exp_name,
            job_logdir=self._resolve_job_logdir(),
        )
        return str(PurePosixPath(logs_path).parent)

    def _resolve_turn_off_first_srun(self) -> bool:
        if self.turn_off_first_srun is not None:
            return self.turn_off_first_srun
        if self.region == "o1":
            return True
        if self.region == "o2":
            return True
        return False

    def _replace_runtime_env_config(
        self, runtime_env_config_path: Path, pam_runtime_envs: dict[str, str]
    ) -> None:
        assert runtime_env_config_path.exists(), self._pretty_params_print(
            {
                "runtime_env_config_path": [
                    "/some/valid/existing/yaml/pam/config.yaml"
                ],
            }
        )
        loaded_config = OmegaConf.load(runtime_env_config_path)
        loaded_config = OmegaConf.to_container(loaded_config)
        if "excludes" not in loaded_config:
            loaded_config["excludes"] = []
        if self.pam_config_path:
            if "env_vars" not in loaded_config:
                loaded_config["env_vars"] = {}
            loaded_config["env_vars"].update(pam_runtime_envs)
        else:
            if (
                "/sber_src/contribs/pretrain_auto_metrics/"
                not in loaded_config["excludes"]
            ):
                loaded_config["excludes"].append(
                    "/sber_src/contribs/pretrain_auto_metrics/"
                )
        for key in ("WANDB_BASE_URL", "WANDB_INSECURE_DISABLE_SSL", "WANDB_DIR"):
            if key in pam_runtime_envs:
                loaded_config.setdefault("env_vars", {})[key] = pam_runtime_envs[key]
        backup_path = runtime_env_config_path.with_suffix(".backup.yaml")
        runtime_env_config_path.rename(backup_path)
        OmegaConf.save(loaded_config, runtime_env_config_path)
        print(
            f"Runtime env config file {runtime_env_config_path} updated and backed up to {backup_path}"
        )

    def _build_env(self) -> dict[str, str]:
        env = BaseJob._build_env(self)
        env.update(
            {
                "VERL_ROOT_PATH": str(self.verl_root),
                "PAM_NNODES": str(self.pam_n_workers),
                "PAM_SGLANG_TAG": self.pam_sglang_tag,
            }
        )
        if self.pam_config_path and self.pam_config_path != "None":
            env["PAM_CONFIG_PATH"] = self.pam_config_path

        env["JS_ENGINE"] = self.engine
        if self.pam_queue_name:
            env["PAM_QUEUE_NAME"] = self.pam_queue_name
        if self.pam_base_image:
            env["PAM_BASE_IMAGE"] = self.pam_base_image
        self._replace_runtime_env_config(Path(self.runtime_env_config_path), env)
        env.update(self.ray_envs)
        if self.use_sharp:
            env.update(
                {
                    "NCCL_COLLNET_ENABLE": "1",
                }
            )
        env.update(
            {
                "RUNTIME_ENV": self.runtime_env_config_path,
                "TRAINING_SCRIPT": self.script_path,
            }
        )
        return env

    def submit(self) -> BaseJob:
        cfg = self.to_config()
        self._anonimize_config(self.pam_config_path)
        self.assert_job(self, cfg)
        self._before_submit()
        return self._submit_config(cfg)
