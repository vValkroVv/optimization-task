from transformers import PretrainedConfig, AutoConfig


class GigaSGLangConfig(PretrainedConfig):
    model_type = "gigasglang"


GigaSGLangConfig.register_for_auto_class(AutoConfig)
