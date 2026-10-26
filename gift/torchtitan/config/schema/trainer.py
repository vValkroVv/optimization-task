import json
from dataclasses import replace
from dataclasses import field
from typing import Any, ClassVar, TYPE_CHECKING

from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    FP8Config,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.configurable import config, Configurable, to_plain
from torchtitan.config.schema.callbacks.base import CallbackConfig
from torchtitan.config.schema.callbacks.builtin.ema import EMAConfig
from torchtitan.config.schema.components.checkpoint import CheckpointManagerConfig
from torchtitan.config.schema.components.dataloader import BaseDataLoaderConfig
from torchtitan.config.schema.components.loss import BaseLossConfig
from torchtitan.config.schema.components.lr_scheduler import LRSchedulersContainerConfig
from torchtitan.config.schema.components.metrics import MetricsProcessorConfig
from torchtitan.config.schema.components.optimizer import (
    OptimizersContainerConfig,
    OptimizersInBackwardContainerConfig,
)
from torchtitan.config.schema.components.tokenizer import (
    BaseTokenizerConfig,
    HuggingFaceTokenizerConfig,
)
from torchtitan.config.schema.components.validate import ValidatorConfig
from torchtitan.config.schema.distributed.parallelism import (
    resolve_data_parallel_shard_degree,
)
from torchtitan.config.schema.distributed.sharding_backend import (
    model_sharding_backend_issues,
    sharding_backend_issues,
)
from torchtitan.config.schema.models.common.cross_section import (
    model_fits_parallelism,
    model_fits_fp8,
    mtp_matches_loss,
    seq_len_fits_model,
    vision_dataloader_agrees,
)
from torchtitan.config.schema.tools.profiler import ProfilerConfig
from torchtitan.config.validation import Issue
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@config
class TrainerConfig(Configurable.Config):
    """
    Default container for training configuration.
    """

    _owner_spec: ClassVar[str] = "torchtitan.trainer:Trainer"

    model_spec: "ModelSpec | None" = field(init=False, default=None)
    """Built from the YAML ``model:`` section by ``torchtitan.config.build``;
    ``init=False`` keeps it out of the decoder. ``dataclasses.replace`` drops
    it; ``Config.build`` copies non-init fields itself."""

    hf_assets_path: str = "./tests/assets/tokenizer"
    """
    Path to HF assets folder. This folder contains local copies of Hugging Face assets,
    including model weights in .safetensors format, the model.safetensor.index.json file
    (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
    """

    dump_folder: str = "./outputs"

    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    callbacks: dict[str, CallbackConfig] = field(default_factory=dict)
    metrics: MetricsProcessorConfig = field(default_factory=MetricsProcessorConfig)
    tokenizer: BaseTokenizerConfig = field(default_factory=HuggingFaceTokenizerConfig)
    dataloader: BaseDataLoaderConfig
    """Required: the family has no default ``type``."""
    optimizer: OptimizersContainerConfig = field(
        default_factory=OptimizersContainerConfig
    )
    lr_scheduler: LRSchedulersContainerConfig = field(
        default_factory=LRSchedulersContainerConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    checkpoint: CheckpointManagerConfig = field(default_factory=CheckpointManagerConfig)
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    compile: CompileConfig = field(default_factory=CompileConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    fp8: FP8Config = field(default_factory=FP8Config)
    loss: BaseLossConfig
    """Required: the family has no default ``type``."""

    def __post_init__(self):
        if self.debug.batch_invariant:
            raise ValueError("Batch-invariant mode is not supported in pre-training.")
        if isinstance(self.optimizer, OptimizersInBackwardContainerConfig):
            if self.parallelism.expert_parallel_degree > 1:
                raise ValueError(
                    "Optimizers in backward is not supported with Expert Parallel."
                )
            if self.parallelism.pipeline_parallel_degree > 1:
                raise ValueError(
                    "Optimizers in backward is not supported with Pipeline Parallel."
                )
        if self.comm.mode == "torchcomms" and (
            self.dataloader.use_data_world
            or (self.validator.enable and self.validator.dataloader.use_data_world)
        ):
            raise ValueError(
                "Dataloaders with use_data_world are not supported with "
                "comm.mode='torchcomms': torchcomms cannot create the data mesh process group."
            )

    def validate(self, ctx):
        ema_count = sum(isinstance(c, EMAConfig) for c in self.callbacks.values())
        issues = sharding_backend_issues(
            self.parallelism,
            checkpoint=self.checkpoint,
            cpu_offload=self.training.enable_cpu_offload,
            has_ema=bool(ema_count),
        )
        if (
            self.parallelism.pipeline_parallel_degree > 1
            and self.training.local_batch_size
            % self.parallelism.pipeline_parallel_microbatch_size
        ):
            issues.append(
                Issue(
                    path="parallelism.pipeline_parallel_microbatch_size",
                    message="must divide training.local_batch_size",
                )
            )
        if (
            self.checkpoint.enable
            and not self.checkpoint.load_only
            and self.checkpoint.last_save_ema_model_also
            and ema_count != 1
        ):
            issues.append(
                Issue(
                    path="checkpoint.last_save_ema_model_also",
                    message=f"requires exactly one EMA callback in callbacks; found {ema_count}",
                )
            )
        previous: list[CallbackConfig] = []
        for name, callback in self.callbacks.items():
            for issue in callback.validate_dependencies(previous):
                path = f"callbacks.{name}"
                issues.append(
                    replace(issue, path=f"{path}.{issue.path}" if issue.path else path)
                )
            previous.append(callback)
        issues += _world_size_issues(
            self.parallelism,
            ctx.runtime.world_size if ctx.runtime else None,
            self.training,
        )
        return issues

    def validate_model(self, model_config) -> list[Issue]:
        """Check the expanded model tree against the sibling sections."""
        return [
            *model_sharding_backend_issues(self.parallelism.fsdp_backend, model_config),
            *model_fits_parallelism(model_config, self.parallelism),
            *model_fits_fp8(model_config, self.fp8, self.parallelism),
            *seq_len_fits_model(model_config, self.training.seq_len, self.parallelism),
            *mtp_matches_loss(model_config, self.loss),
            *vision_dataloader_agrees(model_config, self.dataloader),
        ]

    def to_dict(self) -> dict[str, Any]:
        """The decodable sections plus ``ModelSpec.to_dict`` (no callables)."""
        d = to_plain(self)
        if self.model_spec is not None:
            d["model_spec"] = self.model_spec.to_dict()
        return d

    def maybe_log(self) -> None:
        if self.debug.print_config:
            logger.info(
                f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
            )


def _world_size_issues(
    parallelism: ParallelismConfig,
    world_size: int | None,
    training: TrainingConfig | None = None,
) -> list[Issue]:
    """Check world geometry and global batch against the actual DP degree."""
    try:
        dp_shard = parallelism.data_parallel_shard_degree
        if world_size is not None:
            dp_shard = resolve_data_parallel_shard_degree(
                dp_replicate=parallelism.data_parallel_replicate_degree,
                dp_shard=parallelism.data_parallel_shard_degree,
                cp=parallelism.context_parallel_degree,
                tp=parallelism.tensor_parallel_degree,
                pp=parallelism.pipeline_parallel_degree,
                ep=parallelism.expert_parallel_degree,
                world_size=world_size,
            )
    except ValueError as error:
        return [Issue(path="parallelism", message=str(error))]
    if training is not None and dp_shard > 0 and training.global_batch_size > 0:
        batch_multiple = (
            training.local_batch_size
            * parallelism.data_parallel_replicate_degree
            * dp_shard
        )
        if training.global_batch_size % batch_multiple:
            return [
                Issue(
                    path="training.global_batch_size",
                    message=f"must be divisible by local_batch_size * DP degree ({batch_multiple})",
                )
            ]
    return []
