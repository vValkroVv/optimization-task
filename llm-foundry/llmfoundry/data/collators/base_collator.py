import inspect

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping

from llmfoundry.data.processors.base_processor import DataModality


class BaseCollator(ABC):
    data_modality: ClassVar[DataModality]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        data_modality = getattr(cls, "data_modality", None)
        if not isinstance(data_modality, DataModality):
            raise TypeError(
                f"{cls.__name__} must define data_modality as a DataModality enum value"
            )

    @abstractmethod
    def __call__(self, examples: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def split_batch(batch: Mapping, microbatch_size: int):
        pass
