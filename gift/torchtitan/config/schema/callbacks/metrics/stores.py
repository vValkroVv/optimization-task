from dataclasses import dataclass

from torchtitan.config.schema.protocols.types import StrEnum


class StoreMode(StrEnum):
    none = "none"
    cpu = "cpu"
    disk = "disk"


@dataclass(frozen=True, kw_only=True, slots=True)
class MetricStoreConfig:
    mode: StoreMode = StoreMode.none
    directory: str = "monitoring_metrics"
