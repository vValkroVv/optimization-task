from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable
