from dataclasses import field

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.schema.callbacks.metrics.sample import MetricConfig


@config(family="monitor source")
class MonitorSourceConfig(Configurable.Config):
    """Base monitor source config; no ``enable`` field, omit the source instead."""

    @property
    def uses_module_hooks(self) -> bool:
        """Whether the source reads the model through ``nn.Module`` hooks."""
        return False


@config
class FqnFilterConfig(MonitorSourceConfig):
    """A source that narrows what it monitors by canonical-FQN patterns.

    A pattern is a regex matched against whole dot-segment runs of the FQN
    (:meth:`MonitorSource._segment_regex`): ``layers.1`` selects ``layers.1.*``
    (and ``mtp.layers.1.*``) but not ``layers.10``. Empty ``include_patterns``
    means everything; ``exclude_patterns`` applies after and wins. FQNs have
    AC/FSDP/compile wrapper segments stripped, so patterns do not depend on
    parallelism.
    """

    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)


@config
class FilteredSourceConfig(FqnFilterConfig):
    """FQN-filtered sources that also run per-tensor metric pipelines."""

    metrics: list[MetricConfig] = field(default_factory=list)
