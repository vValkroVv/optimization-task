from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.sources.base import MonitorSourceConfig


@config(type="data")
class DataSourceConfig(MonitorSourceConfig):
    _owner_spec: ClassVar[str] = "torchtitan.callbacks.sources.data_source:DataSource"

    emit_batch_stats: bool = False
