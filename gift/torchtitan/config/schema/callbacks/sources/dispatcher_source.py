from dataclasses import dataclass, field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.metrics.sample import MetricConfig
from torchtitan.config.schema.callbacks.sources.base import FqnFilterConfig
from torchtitan.config.schema.protocols.types import StrEnum


class MaxVioWindow(StrEnum):
    global_ = "global"
    no_pad = "no_pad"
    no_pad_no_prefix = "no_pad_no_prefix"


class MaxVioLevel(StrEnum):
    microbatch = "microbatch"
    batch = "batch"
    ep_group = "ep_group"


@dataclass(frozen=True, kw_only=True, slots=True)
class MaxVioConfig:
    """One max-violation series of the token dispatcher.

    ``window`` picks which per-microbatch histogram the model collected:
    the full one, the one excluding padding, or the one excluding padding and
    the prefix. ``level`` picks the aggregation:

    - ``microbatch``: violation of each microbatch, then averaged;
    - ``batch``: the window's histograms summed into optimizer batches first,
      then summed over the data ranks, then violation, then averaged;
    - ``ep_group``: as microbatch, but the expert axis is folded into ep_size
      device groups first — device imbalance instead of expert imbalance, and
      only meaningful at ep>1.
    """

    name: str
    window: MaxVioWindow
    level: MaxVioLevel


@config(type="dispatcher")
class DispatcherSourceConfig(FqnFilterConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.sources.dispatcher_source:DispatcherSource"
    )

    max_vio: list[MaxVioConfig] = field(default_factory=list)
    # Over this rank's own histogram — rank-local, no reduce stage.
    local_tokens_metrics: list[MetricConfig] = field(default_factory=list)
    # Over the ep-wide histogram. The ep gather is a concatenation, not a
    # reduction, so it happens before the metrics run rather than as a
    # reduce stage.
    global_tokens_metrics: list[MetricConfig] = field(default_factory=list)
    namespace: str = "moe/token_dispatcher"
    model_level_leaf: str = "dispatcher_sum"
