from pathlib import Path
from typing import Optional, Union, Tuple
import os
import warnings
try:
    from composer.utils.dist import get_tp_group_size
except ImportError:
    warnings.warn("Can not import modules for training. Probably running in inference mode.")

from omegaconf import DictConfig, OmegaConf
from transformers import PretrainedConfig

from .mm_audio_adapters import AudioAdapterConfig
from .mm_projectors import ProjectorConfig
from .registry import CONFIG_CLS_REGISTRY
from llmfoundry.utils.config_utils import to_dict_container


#####
##### TODO: refactor


class BaseBackBoneLLMConfig(PretrainedConfig):
    model_type: str = "backbone_llm"
    llm_config_subfolder: str = "llm_config"
    def __init__(
        self,
        type: str = "llama",
        config_path: Optional[str] = None,
        pretrain_path: Optional[str] = None,
        freeze: bool = False,
        params: Optional[DictConfig] = None,
        **kwargs,
    ):
        """

        kwargs - kwargs for from_pretrained method for Config

        """
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        super().__init__(**kwargs)
        self.type = type
        self.config_path = config_path or pretrain_path
        self.pretrain_path = pretrain_path
        self.freeze = freeze
        self.config = self.get_config_and_override(params, trust_remote_code=trust_remote_code, **kwargs)

    def set_config(self):
        if self.config_path is None:
            self.config_path = str(Path(self.name_or_path) / self.llm_config_subfolder)
            self.config = self.get_config_and_override(None)

    def get_config_and_override(self, params, **kwargs):
        if self.type not in CONFIG_CLS_REGISTRY:
            raise ValueError(f"Can not init config with type {self.type}")

        dict_params = convert_to_dict(params) if params else kwargs.pop("config", {})
        dict_params["tp_size"] = dict_params.get("tp_size", 1)
        dict_params.update(kwargs)

        if "trust_remote_code" not in dict_params:
            dict_params["trust_remote_code"] = True

        if self.config_path:
            if "pretrained_model_name_or_path" in dict_params:
                del dict_params["pretrained_model_name_or_path"]
            config = CONFIG_CLS_REGISTRY[self.type].from_pretrained(self.config_path, **dict_params)
        else:
            config = CONFIG_CLS_REGISTRY[self.type](**dict_params)

        if not hasattr(config, "tp_size"):
            # If config does not have tp_size parameter, you are loading non-sber model 
            # and don't need following checks
            return config

        assert config.tp_size >= 1
        if config.tp_size > 1:
            tp_group_size = get_tp_group_size()
            assert config.tp_size == tp_group_size, f"Wrong tensor parallel group size. {tp_group_size} instead of {config.tp_size}"
        return config


def convert_to_dict(value : Optional[DictConfig]) -> dict:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value or {}


#####

class GigaTTSConfig(PretrainedConfig):
    model_type = "giga_tts"
    audio_adapter_config_cls = AudioAdapterConfig
    projector_config_cls = ProjectorConfig
    llm_config_cls = BaseBackBoneLLMConfig

    def __init__(
        self,
        audio_adapter_config: Optional[DictConfig] = None,
        llm_projector_config: Optional[DictConfig] = None,
        audio_llm_projector_config: Optional[DictConfig] = None,
        llm_config: Optional[DictConfig] = None,
        speech_embeddings_config: Optional[DictConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_adapter_config = self.audio_adapter_config_cls(**to_dict_container(audio_adapter_config or {}))
        self.llm_projector_config = self.projector_config_cls(**to_dict_container(llm_projector_config or {}))
        self.audio_llm_projector_config = self.projector_config_cls(**to_dict_container(audio_llm_projector_config or {}))
        self.llm_config = self.llm_config_cls(**to_dict_container(llm_config or {}))
        self.speech_embeddings_config = to_dict_container(speech_embeddings_config or {})
        self.return_dict = self.llm_config.config.return_dict

        # TODO: fix sp_split_type in train.py
        self.sp_split_type = 'equal'

    @classmethod
    def from_dict(cls, config_dict, **kwargs) -> Union["GigaTTSConfig", Tuple["GigaTTSConfig", dict]]:
        """
        Override method from PretrainedConfig
        Method is called in AutoConfig.from_pretrained and AutoModelForCasualLM.from_pretrained
        """
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        """
        Override from PretrainedConfig to pass _name_or_path in final dict. Need for working `GigaTTSConfig.from_pretrained`
        In vanilla method _name_or_path is not set in config_dict to use it `from_dict`
        """
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        config_dict["_name_or_path"] = str(pretrained_model_name_or_path)
        return config_dict, kwargs
