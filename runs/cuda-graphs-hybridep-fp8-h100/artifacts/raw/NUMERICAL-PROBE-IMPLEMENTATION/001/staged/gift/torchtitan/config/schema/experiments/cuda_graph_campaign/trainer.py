from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.datasets.hf.text_datasets import (
    HuggingFaceTextDataLoaderConfig,
)
from torchtitan.config.schema.experiments.cuda_graph_campaign.continuation import (
    CampaignRNGContinuationConfig,
)
from torchtitan.config.schema.experiments.cuda_graph_campaign.numerical_probe import (
    CampaignNumericalProbeConfig,
)
from torchtitan.config.schema.trainer import TrainerConfig
from torchtitan.config.validation import Issue


@config
class CampaignTrainerConfig(TrainerConfig):
    """Native trainer config with the target loader's deferred RNG restore."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.cuda_graph_campaign.trainer:CampaignTrainer"
    )

    def validate(self, ctx):
        issues = list(TrainerConfig.validate(self, ctx))

        continuation_callbacks = [
            callback
            for callback in self.callbacks.values()
            if isinstance(callback, CampaignRNGContinuationConfig)
        ]
        if len(continuation_callbacks) != 1:
            issues.append(
                Issue(
                    path="callbacks",
                    message=(
                        "cuda_graph_campaign requires exactly one "
                        "campaign_rng_continuation callback"
                    ),
                )
            )

        if not isinstance(self.dataloader, HuggingFaceTextDataLoaderConfig):
            issues.append(
                Issue(
                    path="dataloader.type",
                    message=(
                        "campaign RNG restore supports the deterministic hf_text "
                        "target loader only"
                    ),
                )
            )
        else:
            for name, expected in (
                ("dataset", "c4_test"),
                ("infinite", True),
                ("num_workers", 0),
                ("pin_memory", False),
            ):
                if getattr(self.dataloader, name) != expected:
                    issues.append(
                        Issue(
                            path=f"dataloader.{name}",
                            message=(
                                f"campaign RNG restore requires {expected!r} "
                                "for the exact target loader"
                            ),
                        )
                    )

        probes = [
            callback
            for callback in self.callbacks.values()
            if isinstance(callback, CampaignNumericalProbeConfig)
        ]
        if len(probes) > 1:
            issues.append(
                Issue(
                    path="callbacks",
                    message="at most one campaign_numerical_probe callback is supported",
                )
            )
        if probes:
            probe = probes[0]
            if ctx.runtime and ctx.runtime.world_size != 8:
                issues.append(
                    Issue(
                        path="parallelism",
                        message="campaign_numerical_probe requires exactly eight ranks",
                    )
                )
            expected_stop = 1 if probe.mode == "create_s0" else 1 + probe.updates
            if expected_stop is not None and self.training.steps != expected_stop:
                issues.append(
                    Issue(
                        path="training.steps",
                        message=(
                            f"numerical probe mode {probe.mode!r} requires absolute "
                            f"stop step {expected_stop}"
                        ),
                    )
                )
            if not self.checkpoint.enable or self.checkpoint.async_mode != "disabled":
                issues.append(
                    Issue(
                        path="checkpoint",
                        message="numerical probe requires synchronous checkpointing",
                    )
                )
            if probe.mode == "create_s0":
                if self.checkpoint.load_only or self.checkpoint.initial_load_path:
                    issues.append(
                        Issue(
                            path="checkpoint",
                            message="create_s0 requires a fresh run with checkpoint.load_only=false",
                        )
                    )
                if self.checkpoint.last_save_model_only:
                    issues.append(
                        Issue(
                            path="checkpoint.last_save_model_only",
                            message="create_s0 requires a full native checkpoint",
                        )
                    )
            elif probe.mode == "observe":
                if (
                    not self.checkpoint.load_only
                    or not self.checkpoint.initial_load_path
                    or self.checkpoint.initial_load_model_only
                    or self.checkpoint.load_step != -1
                ):
                    issues.append(
                        Issue(
                            path="checkpoint",
                            message=(
                                "observe requires a full initial load with "
                                "load_only=true, load_step=-1, and initial_load_model_only=false"
                            ),
                        )
                    )

        return issues
