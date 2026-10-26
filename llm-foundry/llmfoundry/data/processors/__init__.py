from typing import Any, Dict, Iterable, List

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from .gigaomni_video_processor import GigaOmniVideoProcessor
from .gigaomni_audio_processor import GigaOmniAudioProcessor
from .gigaomni_vision_processor import GigaOmniVisionProcessor


__all__ = [
    "GigaOmniVideoProcessor",
    "GigaOmniAudioProcessor",
    "GigaOmniVisionProcessor",
    "build_processors",
]


def build_processors(
    processor_configs: Iterable[Dict[str, Any]], num_dataloader_workers: int = 1
) -> List[Any]:
    """
    Instantiate a `processors` config list (`name` + kwargs per entry).

    A processor whose `valid_kwargs` list `num_dataloader_workers` is told how
    many dataloader workers share the node's CPU so it can size its thread
    pools; a value set in the config wins.
    """
    processors_list = []
    for processor_config in processor_configs:
        if isinstance(processor_config, DictConfig):
            processor_params = om.to_container(processor_config, resolve=True)
        else:
            processor_params = dict(processor_config)

        processor_cls = globals()[processor_params.pop("name")]
        if "num_dataloader_workers" in getattr(processor_cls, "valid_kwargs", []):
            processor_params.setdefault(
                "num_dataloader_workers", num_dataloader_workers
            )

        processors_list.append(processor_cls(**processor_params))

    return processors_list
