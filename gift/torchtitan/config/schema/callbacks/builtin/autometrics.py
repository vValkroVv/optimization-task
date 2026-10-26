from dataclasses import dataclass, field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.value_types import ConfigValue
from torchtitan.config.schema.callbacks.base import CallbackConfig
from torchtitan.config.schema.protocols.types import StrEnum


class PamMode(StrEnum):
    autometrics = "autometrics"
    convertation = "convertation"
    inference = "inference"


class ModelClass(StrEnum):
    sft = "sft"
    dpo = "dpo"
    vision = "vision"
    audio = "audio"
    grpo = "grpo"


class ModelSize(StrEnum):
    lite = "lite"
    pro = "pro"
    max = "max"
    ultra = "ultra"
    custom = "custom"


class LmEvalSuite(StrEnum):
    pretrain = "pretrain"
    instruct = "instruct"


class SlurmSubmitType(StrEnum):
    sbatch = "sbatch"
    api = "api"


@dataclass(frozen=True, kw_only=True, slots=True)
class EvalTaskConfig:
    """One lm_eval benchmark and the options PAM forwards to the harness."""

    name: str
    num_fewshot: int | list[int] = 0
    log_samples: bool = True
    num_concurrent: int = 16

    def __post_init__(self) -> None:
        shots = (
            self.num_fewshot
            if isinstance(self.num_fewshot, list)
            else [self.num_fewshot]
        )
        if not shots or any(type(shot) is not int or shot < 0 for shot in shots):
            raise ConfigFieldError(
                "num_fewshot", "must be a nonnegative int or nonempty list of them"
            )


@config(type="autometrics")
class AutometricsConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = "torchtitan.callbacks.builtin.autometrics:Autometrics"
    requires: ClassVar[tuple[str, ...]] = ("sharding_backend",)

    # Evaluate checkpoints on a whole multiple of this many steps.
    interval: int = 1000

    def __post_init__(self) -> None:
        for name in ("interval", "inference_tp", "inference_dp", "n_workers", "n_gpus"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigFieldError(name, "must be a positive integer")
        if not self.user:
            raise ConfigFieldError("user", "a job submitter login is required")
        if self.mode == PamMode.autometrics and not self.tasks:
            raise ConfigFieldError("tasks", "must not be empty in autometrics mode")
        names = [task.name for task in self.tasks]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ConfigFieldError("tasks", f"task names repeat: {', '.join(repeated)}")
        if type(self.inference_ep) is not int or self.inference_ep < 0:
            raise ConfigFieldError("inference_ep", "must be a non-negative integer")
        if self.inference_ep and self.inference_ep != self.inference_tp:
            raise ConfigFieldError("inference_ep", "must be 0 or equal inference_tp")
        if self.inference_tp * self.inference_dp != self.n_workers * self.n_gpus:
            raise ConfigFieldError(
                "inference_dp",
                "inference_tp * inference_dp must equal n_workers * n_gpus",
            )
        if self.submission_type not in ("cloud", "slurm"):
            raise ConfigFieldError("submission_type", "must be 'cloud' or 'slurm'")
        if type(self.max_retry) is not int or self.max_retry < 2:
            raise ConfigFieldError("max_retry", "must be an integer >= 2")
        self.inference_parameters()

    user: str = ""
    run_name: str = ""
    exp_name: str = ""

    tensorboard_path: str = ""

    mode: PamMode = PamMode.autometrics
    model_class: ModelClass = ModelClass.sft
    model_size: ModelSize = ModelSize.lite
    # fsdp2: mappers PAM runs on the HF export.
    fsdp2_mappers: list[str] = field(
        default_factory=lambda: ["map_gigafsdp_to_hf_safetensor"]
    )
    # gfsdp2: the mapper PAM converts the gfsdp2 DCP checkpoint with.
    gfsdp2_mapper: str = "map_gfsdp2_dcp_to_hf"
    input_tp_size: int = 1
    delete_converted_hf: bool = False
    router_host: str = "0.0.0.0"

    inference_tp: int = 1
    inference_dp: int = 8
    # Expert parallelism of the serving replicas (sglang --ep-size). PAM's
    # assertor accepts it only equal to inference_tp; 0 leaves it unset.
    inference_ep: int = 0

    max_running_requests: int = 256
    max_queued_requests: int = 1024
    inference_options: dict[str, ConfigValue] = field(default_factory=dict)

    def inference_parameters(self) -> dict[str, ConfigValue]:
        """PAM serving options; additional flags cannot override checked settings."""
        values: dict[str, ConfigValue] = {
            "tp": self.inference_tp,
            "dp": self.inference_dp,
            "ep": self.inference_ep,
            "max-running-requests": self.max_running_requests,
            "max-queued-requests": self.max_queued_requests,
        }
        aliases = {
            "tensor-parallel-size": "tp",
            "data-parallel-size": "dp",
            "expert-parallel-size": "ep",
        }
        for key in self.inference_options:
            name = key.replace("_", "-")
            if aliases.get(name, name).removesuffix("-size") in values:
                raise ConfigFieldError(
                    "inference_options", f"{key!r} has a dedicated config field"
                )
        if not self.inference_ep:
            values.pop("ep")
        return values | self.inference_options

    tasks: list[EvalTaskConfig] = field(default_factory=list)
    lm_eval_suite: LmEvalSuite = LmEvalSuite.pretrain
    n_workers: int = 1
    n_gpus: int = 8
    processes_per_worker: int = 1
    queue_name: str = ""
    priority_class: str = ""
    submission_type: str = ""  # "cloud" | "slurm"
    region: str = ""  # cloud: SR004 | SR008; slurm: o1 | o2 | o5
    workspace_name: str = ""  # cloud only
    base_image: str = ""

    max_retry: int = 3
    slurm_submit_type: SlurmSubmitType = SlurmSubmitType.api
    tokenizer_path: str = ""
    job_class: str = (
        "contrib.job_submitter.job_submitter.jobs.gift.autometrics.AutometricsJobGIFT"
    )
