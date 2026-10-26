import os
from dataclasses import dataclass

from .version_manager import validate_and_update_version
from .golden_versions import (
    DEFAULT_VERSION,
    SFT_GOLDEN_VERSION,
    VISION_GOLDEN_VERSION,
    VIDEO_GOLDEN_VERSION,
    SPEECH_GOLDEN_VERSION,
    TTS_GOLDEN_VERSION,
)


@dataclass
class Modality:
    name: str
    version: str
    allow_update: bool = False
    description: str = ""
    config_files: tuple[str, ...] = ()

    def __post_init__(self):
        self.version = validate_and_update_version(
            self.version, f"{self.name}_VERSION", self.allow_update
        )

    def __str__(self) -> str:
        return self.name


ARE_WE_UPDATING_GOLDENS = False


MODALITIES = {
    "GOLDEN": Modality(
        name="GOLDEN",
        version=DEFAULT_VERSION,
        allow_update=os.getenv(
            "ALLOW_TO_UPDATE_GOLDEN", str(ARE_WE_UPDATING_GOLDENS)
        ).lower()
        == "true",
        description="Golden reference checkpoints",
    ),
    "SFT": Modality(
        name="SFT",
        version=SFT_GOLDEN_VERSION,
        allow_update=os.getenv(
            "ALLOW_TO_UPDATE_SFT", str(ARE_WE_UPDATING_GOLDENS)
        ).lower()
        == "true",
    ),
    "VISION": Modality(
        name="VISION",
        version=VISION_GOLDEN_VERSION,
        allow_update=os.getenv(
            "ALLOW_TO_UPDATE_VISION", str(ARE_WE_UPDATING_GOLDENS)
        ).lower()
        == "true",
        config_files=(
            # base configurations
            "model_gigar.yaml",
            "model_gigar_qwen.yaml",
            "model_gigar_qwen3.yaml",
            "model_gigamix.yaml",
            "base_train.yaml",
            "base_train_qwen.yaml",
            "base_train_qwen3.yaml",
            # freezing strategies
            "freeze_llm.yaml",
            "freeze_encoders.yaml",
            "freeze_adapters.yaml",
            "freeze_no_module.yaml",
            # sequence parallelism
            "sp-dynamic-1-2.yaml",
            "sp-dynamic-2-4.yaml",
            # weight loading
            "load_full_weights.yaml",
            "load_full_sf_weights.yaml",
            "load_llm_weights.yaml",
            "load_llm_sf_weights.yaml",
            "load_moe_full_weights.yaml",
            "load_moe_full_sf_weights.yaml",
            "load_moe_llm_weights.yaml",
            "load_moe_llm_sf_weights.yaml",
            # schedulers
            "scheduler_cosine_warmup.yaml",
            "scheduler_zero_lr.yaml",
            # init types
            "init_cpu.yaml",
            "init_meta.yaml",
            # other
            "icl_task.yaml",
            "icl_task_qwen.yaml",
            "icl_task_qwen3.yaml",
            "lora.yaml",
            "duration_1ba.yaml",
        ),
    ),
    "VIDEO": Modality(
        name="VIDEO",
        version=VIDEO_GOLDEN_VERSION,
        allow_update=os.getenv(
            "ALLOW_TO_UPDATE_VIDEO", str(ARE_WE_UPDATING_GOLDENS)
        ).lower()
        == "true",
        config_files=(
            "model_gigar.yaml",
            "model_gigar_token_reduction.yaml",
            "model_gigamix_token_reduction.yaml",
            "base_train.yaml",
            "freeze_llm.yaml",
            "freeze_no_module.yaml",
            "icl_task.yaml",
            "sp-2-misaligned-reduction.yaml",
        ),
    ),
    "SPEECH": Modality(
        name="SPEECH",
        version=SPEECH_GOLDEN_VERSION,
        allow_update=os.getenv(
            "ALLOW_TO_UPDATE_SPEECH", str(ARE_WE_UPDATING_GOLDENS)
        ).lower()
        == "true",
    ),
    "TTS": Modality(
        name="TTS",
        version=TTS_GOLDEN_VERSION,
        allow_update=os.getenv(
            "ALLOW_TO_UPDATE_TTS", str(ARE_WE_UPDATING_GOLDENS)
        ).lower()
        == "true",
    ),
}


def get_modality_version(modality_name: str) -> str:
    modality = MODALITIES.get(modality_name)
    if modality:
        return modality.version
    return MODALITIES["GOLDEN"].version


def get_modality_config_files(modality_name: str) -> tuple[str, ...]:
    modality = MODALITIES.get(modality_name)
    if modality:
        return modality.config_files
    return ()
