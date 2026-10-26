import shutil
import importlib

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Union, List, Optional
from transformers import AutoConfig, PretrainedConfig


class GigaOmniConfig(PretrainedConfig):
    model_type = "gigaomni"
    is_composition = True

    def __init__(
        self,
        language_model: Optional[Union[DictConfig, Dict[str, Any]]] = None,
        encoders: Optional[List[Union[DictConfig, Dict[str, Any]]]] = None,
        adapters: Optional[List[Union[DictConfig, Dict[str, Any]]]] = None,
        heads: Optional[List[Union[DictConfig, Dict[str, Any]]]] = None,
        modalities: Optional[List[Union[DictConfig, Dict[str, Any]]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.language_model = self._try_convert_to_container(language_model)
        self.encoders = [
            self._try_convert_to_container(encoder) for encoder in encoders or []
        ]
        self.adapters = [
            self._try_convert_to_container(adapter) for adapter in adapters or []
        ]
        self.heads = [self._try_convert_to_container(head) for head in heads or []]
        self.modalities = [
            self._try_convert_to_container(modality) for modality in modalities or []
        ]

    def _try_convert_to_container(
        self, config: Optional[Union[DictConfig, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        if not config:
            return {}
        if isinstance(config, DictConfig):
            return OmegaConf.to_container(config, resolve=True)  # type: ignore
        return config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        from transformers.utils import (
            HF_MODULES_CACHE,
            TRANSFORMERS_DYNAMIC_MODULE_NAME,
        )

        # NOTE (Sbr): Automatic processing of modeling files from transformers can't handle folders, so we need to copy
        # the modalities folder to the cache directory by ourselves. This must be done in GigaOmniConfig.from_pretrained
        # (not GigaOmniForCausalLM.from_pretrained), because config is loaded first when creating HF model; by the time
        # GigaOmniForCausalLM.from_pretrained runs, all code is already in place and imports resolve correctly.

        assert cache_dir is None, (
            "Custom cache directory is not supported for GigaOmni!"
        )
        assert pretrained_model_name_or_path, (
            "pretrained_model_name_or_path must be provided!"
        )

        cached_submodule_path = (
            Path(HF_MODULES_CACHE)
            / TRANSFORMERS_DYNAMIC_MODULE_NAME
            / Path(pretrained_model_name_or_path).name
        )
        shutil.copytree(
            str(Path(pretrained_model_name_or_path) / "modalities"),
            str(Path(cached_submodule_path) / "modalities"),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        importlib.invalidate_caches()

        return super().from_pretrained(
            pretrained_model_name_or_path, cache_dir=cache_dir, **kwargs
        )


GigaOmniConfig.register_for_auto_class(AutoConfig)
