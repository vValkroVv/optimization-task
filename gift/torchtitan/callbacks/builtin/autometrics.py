import importlib
import json
import shutil
from pathlib import Path
from typing import Any, TypeAlias

import torch.distributed as dist
import yaml

from torchtitan.config.schema.callbacks.builtin.autometrics import AutometricsConfig
from torchtitan.config.schema.callbacks.builtin.autometrics import EvalTaskConfig
from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.callbacks.builtin.ema import EMA
from torchtitan.config.schema.protocols.types import StrEnum
from torchtitan.protocols.sharding_backend import ShardingBackend
from torchtitan.components.checkpoint import resolve_hf_export
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger

__all__ = [
    "Autometrics",
    "EvalTaskConfig",
]

# PAM reads plain strings; without this PyYAML would tag our StrEnum members.
yaml.SafeDumper.add_multi_representer(
    StrEnum, lambda dumper, data: dumper.represent_str(str.__str__(data))
)


def _ema_config(trainer_config: Any) -> EMA.Config | None:
    """The run's ``EMA`` callback config, or None when it does not average."""
    for callback in trainer_config.callbacks.values():
        if isinstance(callback, EMA.Config):
            return callback
    return None


class Autometrics(Configurable, Observer):
    """Submits a Pretrain Auto Metrics job for stable checkpoints.

    Only lm_eval's pretrain suite is expressible here. PAM's instruct-style
    evaluators (mera, dredd, obm) additionally require a reasoning parser or a
    chat template, neither of which a pretraining run has.

    ``interval`` is this callback's own cadence, not the checkpointer's: PAM
    evaluates only stable checkpoints, a subset of what the saver writes.

    What is submitted depends on whether the backend exports HF:

    * it does -- PAM is given the export the checkpointer wrote
      (``checkpoint.save_in_hf``) and runs ``fsdp2_mappers`` on it;
    * it does not -- PAM is given the DCP checkpoint itself and converts it
      with ``gfsdp2_mapper``.

    Example:
        ``callbacks["autometrics"] = Autometrics.Config(user="username",
        exp_name="exp_name", interval=1000)`` in the preset; the
        fields are then addressable as ``--callbacks.autometrics.interval``.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = AutometricsConfig

    def __init__(
        self,
        config: Config,
        *,
        sharding_backend: ShardingBackend,
    ) -> None:
        self.sharding_backend = sharding_backend
        self._sglang_tag = ""
        self._hf_config: dict[str, Any] | None = None
        self._job_class = self._resolve(config.job_class)
        self.config = config

    def fit_start(self, state: ObserverState) -> None:
        """Reject a configuration that cannot produce a single submission."""
        config = self.config
        trainer_config = state.config
        checkpoint = trainer_config.checkpoint
        problems: list[str] = []
        if not checkpoint.enable:
            problems.append(
                "checkpoint.enable is False, so nothing is ever saved and this "
                "callback never runs"
            )
        if self.sharding_backend.exports_hf_checkpoint:
            if not checkpoint.save_in_hf:
                problems.append(
                    "checkpoint.save_in_hf is False, so no artifacts/hf_checkpoint "
                    "is exported for the evaluation service to load; "
                    f"{self.sharding_backend.name} submits that export"
                )
        else:
            problems.extend(self._raw_dcp_problems(trainer_config))
        if checkpoint.enable and checkpoint.keep_latest_k > 0:
            problems.append(
                f"checkpoint.keep_latest_k is {checkpoint.keep_latest_k}, so the "
                "checkpointer deletes older checkpoints while their PAM jobs may "
                "still be queued or running -- and PAM reads the checkpoint "
                "directory and writes its results back into it. Measured on the "
                "9b pp=1 pair: the last four evaluations died on "
                "FileNotFoundError. Set checkpoint.keep_latest_k=0"
            )
        if checkpoint.enable and config.interval % checkpoint.interval:
            problems.append(
                f"autometrics.interval ({config.interval}) is not a multiple of "
                f"checkpoint.interval ({checkpoint.interval}), so no saved step "
                "will ever match this callback's cadence"
            )
        assets = Path(config.tokenizer_path or trainer_config.hf_assets_path)
        if not assets.is_dir():
            problems.append(
                f"neither autometrics.tokenizer_path nor hf_assets_path is a "
                f"directory ({assets}), so no tokenizer travels with the weights"
            )
        adapter = trainer_config.model_spec.state_dict_adapter
        if adapter is None:
            problems.append(
                f"model '{trainer_config.model_spec.name}' has no state dict "
                "adapter, so its config.json cannot be described"
            )
        else:
            try:
                hf_config, code_dir = resolve_hf_export(
                    adapter,
                    trainer_config.model_spec.model,
                    checkpoint.hf_remote_code_dir,
                )
            except Exception as exc:
                problems.append(
                    f"the HF export of '{trainer_config.model_spec.flavor}' "
                    f"would not be loadable: {exc}"
                )
            else:
                self._sglang_tag = "custom" if code_dir else "opensource"
                self._hf_config = hf_config
        if problems:
            raise ValueError(
                "Autometrics is enabled but cannot evaluate anything:\n  - "
                + "\n  - ".join(problems)
            )
        self._warn_if_ema_is_not_evaluated(trainer_config)

    def _warn_if_ema_is_not_evaluated(self, trainer_config: Any) -> None:
        """Say so loudly when the run averages weights but PAM measures live ones."""
        ema = _ema_config(trainer_config)
        if ema is None or ema.use_ema:
            return
        logger.warning(
            "EMA callback is on, but autometrics evaluates the LIVE weights: "
            "the EMA has use_ema=False."
        )

    def _raw_dcp_problems(self, trainer_config: Any) -> list[str]:
        """What submitting the DCP checkpoint itself needs, beyond the common checks."""
        problems: list[str] = []
        async_mode = str(trainer_config.checkpoint.async_mode).lower()
        if async_mode != "disabled":
            problems.append(
                f"checkpoint.async_mode is {async_mode!r}: CheckpointManager."
                "save() returns once the state is staged, and the trainer emits "
                "after_checkpoint_save right then, so PAM would be pointed at a "
                "directory whose upload has not finished. A backend that exports "
                "HF never hits this: save_in_hf already forces a synchronous save"
            )
        return problems

    @staticmethod
    def _resolve(dotted_path: str):
        module_name, _, class_name = dotted_path.rpartition(".")
        try:
            return getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"Autometrics cannot import its job class {dotted_path!r}: "
                f"{exc}. Put the PAM job submitter on PYTHONPATH, point "
                "callbacks.autometrics.job_class at the right class, or "
                "disable the callback."
            ) from exc

    def after_checkpoint_save(self, state: ObserverState) -> None:
        if state.step % self.config.interval != 0:
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        saved_path = state.last_saved_checkpoint
        if saved_path is None:
            return
        checkpoint_dir = Path(saved_path).resolve()

        try:
            submit_dir, pam_config = self._write_artifacts(checkpoint_dir, state)
            exp_name = (
                f"{self.config.exp_name}: autometrics {checkpoint_dir.name}"
                if self.config.exp_name
                else f"Autometrics run for {checkpoint_dir}"
            )
            canonical = submit_dir / "artifacts" / "pam_config.yaml"
            path_to_config = None if pam_config == canonical else str(pam_config)
            job = self._job_class(
                checkpoint_dir=str(submit_dir),
                path_to_config=path_to_config,
                user_login=self.config.user,
                exp_name=exp_name,
                mode=self.config.mode,
                n_workers=self.config.n_workers,
                n_gpus=self.config.n_gpus,
                processes_per_worker=self.config.processes_per_worker,
                sglang_tag=self._sglang_tag,
                max_retry=self.config.max_retry,
                slurm_submit_type=self.config.slurm_submit_type,
                **self._optional_submit_kwargs(),
            )
            job.submit()
            logger.info(f"Autometrics job submitted for {checkpoint_dir}")
        except Exception as exc:
            logger.error(f"Autometrics failed for {checkpoint_dir}: {exc}")

    def _write_artifacts(
        self, checkpoint_dir: Path, state: ObserverState
    ) -> tuple[Path, Path]:
        """Stage the checkpoint for PAM: ``(submit_dir, pam_config)``."""
        if self.sharding_backend.exports_hf_checkpoint:
            return self._stage_hf_export(checkpoint_dir, state)
        return self._stage_raw_dcp_checkpoint(checkpoint_dir, state)

    def _stage_hf_export(
        self, checkpoint_dir: Path, state: ObserverState
    ) -> tuple[Path, Path]:
        """Stage the HF export the checkpointer wrote beside the checkpoint."""
        hf_dir = checkpoint_dir / "artifacts" / "hf_checkpoint"
        if not hf_dir.is_dir():
            raise FileNotFoundError(
                f"{hf_dir} does not exist. Autometrics evaluates an HF "
                "checkpoint; enable checkpoint.save_in_hf so every saved step "
                "exports one."
            )

        missing = [
            name
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
            if not (hf_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{hf_dir} lacks {missing}; the HF export did not write its "
                "metadata (is hf_assets_path a tokenizer directory?)."
            )
        if self.config.tokenizer_path:
            # An explicit tokenizer replaces the exported one.
            assets = Path(self.config.tokenizer_path)
            if not assets.is_dir():
                raise FileNotFoundError(
                    f"Tokenizer directory {assets} does not exist; fix "
                    "callbacks.autometrics.tokenizer_path."
                )
            for item in assets.iterdir():
                if item.is_file():
                    shutil.copyfile(item, hf_dir / item.name)

        pam_config = hf_dir / "artifacts" / "pam_config.yaml"
        pam_config.parent.mkdir(parents=True, exist_ok=True)
        pam_config.write_text(
            yaml.safe_dump(
                self._pam_config(state, convertation=self._hf_export_convertation()),
                sort_keys=False,
                allow_unicode=True,
            )
        )
        return hf_dir, pam_config

    def _stage_raw_dcp_checkpoint(
        self, checkpoint_dir: Path, state: ObserverState
    ) -> tuple[Path, Path]:
        """Stage a gfsdp2 DCP checkpoint for conversion inside the PAM job.

        Everything goes directly under ``artifacts/``, because that layout is
        PAM's and not ours to choose: ``ConvertationManager`` sets
        ``tokenizer_path = Path(artifacts_path, "tokenizer")`` outright and never
        reads the one this config names. Staging a run_name subdirectory to keep
        two callbacks on one checkpoint from overwriting each other therefore
        cannot work -- it leaves the converted checkpoint with no tokenizer, and
        sglang fails on it minutes into the evaluation job. Two gfsdp2
        autometrics callbacks on one run would still collide here; the place to
        prevent that is the config, not the layout.
        """
        artifacts = checkpoint_dir / "artifacts"
        tokenizer_dir = artifacts / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        if self._hf_config is None:
            raise RuntimeError(
                "no HF config was resolved for this model; fit_start should "
                "have refused the run before reaching a checkpoint."
            )
        (artifacts / "hf_config.json").write_text(
            json.dumps(self._hf_config, indent=2) + "\n"
        )

        assets = Path(self.config.tokenizer_path or state.config.hf_assets_path)
        if not assets.is_dir():
            raise FileNotFoundError(
                f"Tokenizer directory {assets} does not exist; fix "
                "callbacks.autometrics.tokenizer_path."
            )
        for item in assets.iterdir():
            if item.is_file():
                shutil.copyfile(item, tokenizer_dir / item.name)

        pam_config = artifacts / "pam_config.yaml"
        pam_config.write_text(
            yaml.safe_dump(
                self._pam_config(
                    state,
                    convertation=self._raw_dcp_convertation(state, artifacts),
                ),
                sort_keys=False,
                allow_unicode=True,
            )
        )
        return checkpoint_dir, pam_config

    def _hf_export_convertation(self) -> dict[str, Any]:
        return {
            "mappers": list(self.config.fsdp2_mappers),
            "input_tp_size": self.config.input_tp_size,
        }

    def _raw_dcp_convertation(
        self, state: ObserverState, artifacts: Path
    ) -> dict[str, Any]:
        # The mapper reads the DCP directly, so it needs the two things the
        # format drops: the model description and the tokenizer.
        config = self.config
        return {
            "mappers": [config.gfsdp2_mapper],
            "input_tp_size": config.input_tp_size,
            "hf_config_path": str(artifacts / "hf_config.json"),
            "tokenizer_path": str(artifacts / "tokenizer"),
            "use_ema": self._use_ema(state),
        }

    def _use_ema(self, state: ObserverState) -> bool:
        """Whether this checkpoint is evaluated on the EMA weights"""
        ema = _ema_config(state.config)
        if ema is None or not ema.use_ema:
            return False
        if state.step < ema.start_step:
            logger.warning(
                f"Autometrics at step {state.step} evaluates the LIVE weights: "
                f"EMA starts averaging at step {ema.start_step}."
            )
            return False
        return True

    def _pam_config(
        self, state: ObserverState, *, convertation: dict[str, Any]
    ) -> dict[str, Any]:
        """The parameters PAM reads, in the shape its assertor requires."""
        config = self.config
        trainer_config = state.config
        return {
            "run_name": config.run_name or Path(trainer_config.dump_folder).name,
            "exp_name": config.exp_name,
            "user": config.user,
            "tensorboard_path": config.tensorboard_path or None,
            "max_seq_len": trainer_config.training.seq_len,
            "autometrics": {
                "globals": {"delete_converted_hf": config.delete_converted_hf},
                "router": {"host": config.router_host},
                "convertation": convertation,
                "preparation": {"skip": True},
                "inference": config.inference_parameters(),
                "model_info": {
                    "model_class": config.model_class,
                    "size": config.model_size,
                    "username": config.user,
                },
                "evaluators": {
                    "lm_eval": {
                        config.lm_eval_suite: {
                            task.name: {
                                "num_fewshot": task.num_fewshot,
                                "log_samples": task.log_samples,
                                "model_args": {"num_concurrent": task.num_concurrent},
                            }
                            for task in config.tasks
                        }
                    }
                },
            },
        }

    def _optional_submit_kwargs(self) -> dict[str, str]:
        kwargs = {}
        if self.config.queue_name:
            kwargs["queue_name"] = self.config.queue_name
        if self.config.priority_class:
            kwargs["priority_class"] = self.config.priority_class
        if self.config.submission_type:
            kwargs["submission_type"] = self.config.submission_type
        if self.config.region:
            kwargs["region"] = self.config.region
        if self.config.workspace_name:
            kwargs["workspace_name"] = self.config.workspace_name
        if self.config.base_image:
            kwargs["base_image"] = self.config.base_image
        return kwargs
