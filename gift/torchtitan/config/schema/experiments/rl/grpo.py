from dataclasses import field, replace
from typing import ClassVar, TYPE_CHECKING

from torchtitan.config.configs import CompileConfig
from torchtitan.config.configurable import config, Configurable
from torchtitan.config.schema.experiments.rl.actors.generator import VLLMGeneratorConfig
from torchtitan.config.schema.experiments.rl.actors.trainer import PolicyTrainerConfig
from torchtitan.config.schema.experiments.rl.batcher import BatcherConfig
from torchtitan.config.schema.experiments.rl.env import EnvConfig
from torchtitan.config.schema.experiments.rl.loss import RLLossConfig
from torchtitan.config.schema.experiments.rl.observability.metrics.processor import (
    MetricsProcessorConfig,
)
from torchtitan.config.schema.distributed.sharding_backend import (
    model_sharding_backend_issues,
    sharding_backend_issues,
)
from torchtitan.config.schema.models.common.cross_section import (
    model_fits_parallelism,
    seq_len_fits_model,
)
from torchtitan.config.validation import Issue

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@config(type="grpo")
class GRPOLossConfig(RLLossConfig):
    _owner_spec: ClassVar[str] = "torchtitan.experiments.rl.grpo:GRPOLoss"

    clip_eps: float = 0.2
    """PPO clipping epsilon for the probability ratio."""


@config
class RLTrainerConfig(Configurable.Config):
    """Top-level config for RL training."""

    _owner_spec: ClassVar[str] = "torchtitan.experiments.rl.grpo:RLTrainer"

    model_spec: "ModelSpec | None" = field(init=False, default=None)
    """Shared by trainer and generator; built from the YAML ``model:`` section
    by ``torchtitan.config.build`` (``init=False``, not a YAML key)."""

    hf_assets_path: str = "./tests/assets/tokenizer"
    """Path to HF assets folder (model weights, tokenizer, config files)."""

    num_steps: int = 10

    dump_folder: str = "outputs/rl"
    """Root output folder for RL artifacts (temp weights, logs, etc.)."""

    num_prompts_per_step: int = 5
    """Number of distinct prompts (= GRPO groups) drawn per training step.

    The total episodes per step is `num_prompts_per_step` * `group_size`,
    where `group_size` is `generator.sampling.n` (completions per prompt).
    """

    num_validation_samples: int = 20
    """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

    env: EnvConfig

    validation_env: EnvConfig

    log_samples: bool = False
    """Log first completion per episode during training and validation."""

    compile: CompileConfig = field(default_factory=CompileConfig)
    """torch.compile config shared by trainer and generator."""

    batcher: BatcherConfig = field(default_factory=BatcherConfig)

    policy_trainer: PolicyTrainerConfig = field(
        default_factory=lambda: PolicyTrainerConfig(loss=GRPOLossConfig())
    )
    """Named for the actor because the root ``trainer`` key selects the
    trainer type in YAML."""

    generator: VLLMGeneratorConfig = field(default_factory=VLLMGeneratorConfig)

    metrics: MetricsProcessorConfig = field(default_factory=MetricsProcessorConfig)

    def __post_init__(self):
        if self.generator.checkpoint.enable:
            raise ValueError(
                "Generator checkpoint must be disabled in the RL loop "
                "(weights are synced from the trainer via TorchStore). "
                "Set generator.checkpoint.enable=False."
            )

        if self.policy_trainer.debug.batch_invariant:
            if not self.policy_trainer.debug.deterministic:
                raise ValueError("batch_invariant requires deterministic=True")
            # TODO: Replace trainer dtype constraint to use mixed
            #  training enabled by FSDP.
            if self.policy_trainer.training.dtype != "bfloat16":
                raise ValueError(
                    f"batch_invariant requires bfloat16 training dtype, "
                    f"got {self.policy_trainer.training.dtype!r}"
                )
            if self.generator.model_dtype != "bfloat16":
                raise ValueError(
                    f"batch_invariant requires bfloat16 generator dtype, "
                    f"got {self.generator.model_dtype!r}"
                )
            if self.policy_trainer.parallelism.enable_sequence_parallel:
                raise ValueError(
                    "batch_invariant mode doesn't support SP now. "
                    "SP uses reduce-scatter which only supports Ring in NCCL "
                    "and has not been validated for determinism."
                )

    def validate(self, ctx):
        # The generator is vLLM: it never shards through fsdp_backend.
        return [
            replace(issue, path=f"policy_trainer.{issue.path}")
            for issue in sharding_backend_issues(
                self.policy_trainer.parallelism,
                checkpoint=self.policy_trainer.checkpoint,
                cpu_offload=self.policy_trainer.training.enable_cpu_offload,
            )
        ]

    def validate_model(self, model_config) -> list[Issue]:
        """Check the expanded model tree against the sibling sections."""
        return [
            *(
                replace(issue, path=f"policy_trainer.{issue.path}")
                for issue in model_sharding_backend_issues(
                    self.policy_trainer.parallelism.fsdp_backend, model_config
                )
            ),
            *model_fits_parallelism(model_config, self.policy_trainer.parallelism),
            *model_fits_parallelism(model_config, self.generator.parallelism),
            *seq_len_fits_model(
                model_config,
                self.batcher.batch.seq_len,
                self.policy_trainer.parallelism,
                path="batcher.batch.seq_len",
            ),
        ]
