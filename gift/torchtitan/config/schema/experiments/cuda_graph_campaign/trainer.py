from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.datasets.hf.text_datasets import (
    HuggingFaceTextDataLoaderConfig,
)
from torchtitan.config.schema.experiments.cuda_graph_campaign.continuation import (
    CampaignRNGContinuationConfig,
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

        return issues
