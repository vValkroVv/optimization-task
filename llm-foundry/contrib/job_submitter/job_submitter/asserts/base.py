from __future__ import annotations

import logging
import warnings
from base64 import b64decode
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from omegaconf import OmegaConf

from job_submitter.core.config_loader import (
    apply_overrides,
    register_relative_config_loader,
)
from job_submitter.core.job_config import JobTemplate
from job_submitter.core.job_templates.models import User
from job_submitter.core.job_templates.region_configs import (
    SLURM_TEAM_QOS_REGIONS,
    SLURM_TEAMS,
)
from job_submitter.core.mattermost import MattermostClient

if TYPE_CHECKING:
    from job_submitter.jobs.base import BaseJob

logger = logging.getLogger(__name__)

_SERVICE_LOGINS = frozenset({"autotest"})


class BaseJobAssertor:
    def _pretty_params_print(
        self, params: dict[str, list[Any]], class_name: Optional[str] = None
    ) -> str:
        parts = []
        for k, vs in params.items():
            values = [f'"{v}"' if isinstance(v, str) else str(v) for v in vs]
            parts.append(f"{k}=" + "/".join(values))
        body = ",\n    ".join(parts)
        if class_name:
            return f"Please specify \n{class_name}(\n    {body}\n)"
        else:
            return f"Please specify \n{self.__class__.__name__}(\n    {body}\n)"

    @staticmethod
    def _pretty_config_param_print(
        dict_for_print: dict[str, list[Any]],
        text_for_print: str = "Please, use one of the following options",
    ) -> str:
        parts = []

        for k, vs in dict_for_print.items():
            values = [v.lower() if isinstance(v, str) else str(v) for v in vs]
            lines = []

            for indent, part in enumerate(k.split(".")):
                prefix = " " * (indent * 2)
                if indent == len(k.split(".")) - 1:
                    lines.append(
                        f"{prefix}{part}: " + "/".join(values) + f" <- {text_for_print}"
                    )
                else:
                    lines.append(f"{prefix}{part}:")

            parts.append("\n".join(lines))

        return "\n\n" + "\n".join(parts)

    @staticmethod
    def _pretty_print_missing_key(key: str) -> str:
        lines = []
        for indent, part in enumerate(key.split(".")):
            prefix = " " * (indent * 2)
            if indent == len(key.split(".")) - 1:
                lines.append(f"{prefix}{part}:")
            else:
                lines.append(f"{prefix}{part}:")
        return "\n\n" + "\n".join(lines) + " <- You forgot this key"

    @staticmethod
    def _is_encoded_in_base64(text: str | bytes) -> bool:
        try:
            if isinstance(text, bytes):
                normalized = text.strip()
            else:
                normalized = text.strip().encode("ascii")
            decoded = b64decode(normalized, validate=True)
            decoded.decode("utf-8")
            return True
        except Exception:
            return False

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        cfg = OmegaConf.load(config_path)
        register_relative_config_loader(config_path.parent)
        if not OmegaConf.has_resolver("sum"):
            OmegaConf.register_new_resolver("sum", lambda x, y: int(x) + int(y))
        if not OmegaConf.has_resolver("mult"):
            OmegaConf.register_new_resolver("mult", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(cfg)
        cfg = apply_overrides(cfg)
        assert config_path.exists() and config_path.is_file()
        loaded_config = OmegaConf.to_container(cfg)
        return loaded_config

    def _assert_slurm_region(self, job_template: JobTemplate) -> None:
        assert job_template.region in ("o1", "o2", "o5"), self._pretty_params_print(
            {"region": ["o1", "o2", "o5"]}
        )
        if job_template.cpu_only:
            assert job_template.partition == "transfer", self._pretty_params_print(
                {"partition": ["transfer"]}
            )
            assert job_template.qos is None, self._pretty_params_print(
                {"queue_name": ["None", "Or delete this param"]}
            )
            assert job_template.cpus_per_task and job_template.cpus_per_task > 0, (
                self._pretty_params_print({"cpus_per_task": ["positive integer"]})
            )
            assert job_template.memory, self._pretty_params_print({"memory": ["32G"]})
            return
        assert job_template.base_image and job_template.base_image.endswith(".sqsh"), (
            self._pretty_params_print({"base_image": ["/mnt/....sqsh"]})
        )
        assert job_template.nice is None or job_template.nice >= 0, (
            self._pretty_params_print({"nice": ["None", "non-negative integer"]})
        )
        if job_template.region in SLURM_TEAM_QOS_REGIONS:
            self._assert_slurm_team_qos(job_template)
            return
        assert job_template.partition in ("middle", "main", "leads"), (
            self._pretty_params_print(
                {"priority_class": ["high", "medium", "low", "leads"]}
            )
        )
        self._assert_slurm_qos(job_template)

    def _assert_cloud_region(self, job_template: JobTemplate) -> None:
        assert job_template.region in (
            "SR004",
            "SR006",
            "SR008",
        ), self._pretty_params_print({"region": ["SR004", "SR006", "SR008"]})
        assert job_template.base_image and job_template.base_image.startswith(
            "cr.ai.cloud.ru/"
        ), self._pretty_params_print({"base_image": ["cr.ai.cloud.ru/..."]})
        self._assert_cloud_queue_name(job_template.queue_name)

    def _assert_base_params(self, job: BaseJob, job_template: JobTemplate) -> None:
        assert job_template.submission_type in (
            "slurm",
            "cloud",
        ), self._pretty_params_print(
            {
                "submission_type": ["slurm", "cloud"],
            }
        )
        assert job.user_login != "default", self._pretty_params_print(
            {"user_login": ["<not default>"]}
        )
        assert job.exp_name != "default", self._pretty_params_print(
            {"exp_name": ["<not default>"]}
        )
        if job_template.cpu_only:
            assert job_template.submission_type == "slurm", self._pretty_params_print(
                {"submission_type": ["slurm"]}
            )
            assert job.n_gpus == 0, self._pretty_params_print({"n_gpus": [0]})
        else:
            assert job.n_gpus in (1, 2, 4, 8), self._pretty_params_print(
                {"n_gpus": [1, 2, 4, 8]}
            )

    def _assert_cloud_queue_name(self, queue_name: Optional[str]) -> None:
        assert queue_name in (
            "default",
            "giga-prod",
            "harness-agentic",
            "instruction-tuning",
            "ml-evaluation",
            "jupyter",
            "online-rl",
            "pretrain-data",
            "gigachat-data",
            "pretrain-ml",
            "pretrain-ml-test",
            "shared",
            "giga-audio",
            "prod-ml-data",
            "lora",
            None,
        ), self._pretty_params_print(
            {
                "queue_name": [
                    "default",
                    "giga-prod",
                    "harness-agentic",
                    "instruction-tuning",
                    "ml-evaluation",
                    "jupyter",
                    "online-rl",
                    "pretrain-data",
                    "gigachat-data",
                    "pretrain-ml",
                    "pretrain-ml-test",
                    "shared",
                    "giga-audio",
                    "prod-ml-data",
                    "lora",
                ]
            }
        )

    def _assert_slurm_qos(self, job_template: JobTemplate) -> None:
        if (
            job_template.region not in ("o1", "o2", "o5")
            or job_template.partition != "main"
        ):
            assert job_template.qos is None, self._pretty_params_print(
                {"queue_name": ["None", "Or delete this param"]}
            )
            return
        assert job_template.qos in ("pretrain", "data", "alignment"), (
            self._pretty_params_print({"queue_name": ["pretrain", "data", "alignment"]})
        )

    def _assert_slurm_team_qos(self, job_template: JobTemplate) -> None:
        if job_template.priority_class == "leads":
            return
        assert job_template.queue_name in SLURM_TEAMS, self._pretty_params_print(
            {"queue_name": list(SLURM_TEAMS)}
        )
        assert job_template.priority_class in ("urgent", "high", "medium", "low"), (
            self._pretty_params_print(
                {"priority_class": ["urgent", "high", "medium", "low", "leads"]}
            )
        )

    def _assert_user_login(self, job: BaseJob, job_template: JobTemplate) -> None:
        """Verify user_login via Mattermost."""
        if job_template.submission_type == "slurm":
            return
        if job.user_login in _SERVICE_LOGINS:
            return
        user = User.verify(job.user_login, client=MattermostClient.from_env())

        if user.skipped:
            warnings.warn(
                "MM_TOKEN not set — skipping Mattermost user validation.",
                UserWarning,
                stacklevel=2,
            )
            return

        if user.error:
            if "not found" in user.error.lower() or not user.is_valid_format:
                assert False, f"\n\n{user.error}\n" + self._pretty_params_print(
                    {"user_login": ["<valid MM username>"]}
                )
            warnings.warn(
                f"Could not verify user_login: {user.error}. Proceeding anyway.",
                UserWarning,
                stacklevel=2,
            )
            return

        logger.info("MM user verified: %s (%s)", user.username, user.display_name)

    def assert_job(self, job: BaseJob, job_template: JobTemplate) -> None:
        self._assert_base_params(job, job_template)
        self._assert_user_login(job, job_template)
        if job_template.submission_type == "slurm":
            self._assert_slurm_region(job_template)
        else:
            self._assert_cloud_region(job_template)
