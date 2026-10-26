import inspect
import torch

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin


class DataModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    VIDEO_AUDIO = "video_audio"

    def __str__(self) -> str:
        return self.value


class BaseProcessor(ProcessorMixin, ABC):
    data_modality: ClassVar[DataModality]

    @classmethod
    def get_attributes(cls) -> List[str]:
        return []

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
    def __init__(
        self,
        *args: Any,
        tokenizer: Optional[Union[str, Path, PreTrainedTokenizerBase]] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = self._load_tokenizer(tokenizer)
        self._processor_fallback_count = 0

    @staticmethod
    def _load_tokenizer(
        tokenizer: Optional[Union[str, PreTrainedTokenizerBase]] = None,
    ) -> Optional[PreTrainedTokenizerBase]:
        if tokenizer is None:
            return None
        if isinstance(tokenizer, str) or isinstance(tokenizer, Path):
            try:
                return AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                raise ValueError(
                    f"GigaOmni processors must contain tokenizer, but it was not found in {tokenizer}"
                )
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            return tokenizer
        raise TypeError(
            f"tokenizer must be a string path/name or PreTrainedTokenizerBase, got {type(tokenizer).__name__}"
        )

    def save_pretrained(self, save_directory: str, **kwargs: Any):  # type: ignore
        if self.tokenizer is not None and not isinstance(self.tokenizer, str):
            self.tokenizer.save_pretrained(save_directory, **kwargs)
        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ):  # type: ignore
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor.tokenizer = cls._load_tokenizer(
            Path(pretrained_model_name_or_path).absolute()
        )
        return processor

    @abstractmethod
    def __call__(
        self,
        files: Optional[List[Dict[str, Any]]] = None,
        dataset_sample: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def get_num_tokens(self, files: Optional[List[Dict[str, Any]]] = None) -> int:
        pass

    def _prepare_dataset_sample(
        self,
        dataset_sample: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, Any]]:
        assert dataset_sample is not None or input_ids is not None, (
            "Either dataset_sample or input_ids and labels must be provided! Labels can be None during evaluation."
        )

        if dataset_sample is None:
            assert input_ids is not None, "input_ids must be provided!"

            dataset_sample = {"text": {"input_ids": input_ids}}
            if labels is not None:
                dataset_sample["text"]["labels"] = labels
            if attention_mask is not None:
                dataset_sample["text"]["attention_mask"] = attention_mask

        return dataset_sample
