from pathlib import Path
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="campaign_combine_bwd_completion")
class CampaignCombineBwdCompletionConfig(CallbackConfig):
    """Opt in to rank-local HybridEP combine backward completion records."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.cuda_graph_campaign.combine_bwd_completion:"
        "CampaignCombineBwdCompletion"
    )

    output_dir: str = "combine_bwd_completion"
    """Rank-local JSONL directory, relative to ``dump_folder`` by default."""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_dir, str)
            or not self.output_dir
            or self.output_dir != self.output_dir.strip()
            or "\x00" in self.output_dir
        ):
            raise ConfigFieldError("output_dir", "must be a nonempty valid path")
        path = Path(self.output_dir)
        if path == Path(".") or not path.name or ".." in path.parts:
            raise ConfigFieldError(
                "output_dir", "must name a valid path without parent traversal"
            )
