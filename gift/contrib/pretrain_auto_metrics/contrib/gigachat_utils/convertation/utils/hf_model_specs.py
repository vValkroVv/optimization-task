"""HF model families the converter can emit: template dir, runtime files, class names."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _models_root() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class HFModelSpec:
    model_type: str
    template_dir: Path
    configuration_filename: str
    modeling_filename: str
    config_class: str
    architecture: str
    modular_filename: Optional[str] = None
    transformers_model_dir: Optional[str] = None
    extra_output_files: tuple[tuple[Path, str], ...] = ()
    extra_stage_files: tuple[tuple[Path, str], ...] = ()

    @property
    def auto_map(self) -> dict[str, str]:
        configuration_module = Path(self.configuration_filename).stem
        modeling_module = Path(self.modeling_filename).stem
        base_model = self.architecture.removesuffix("ForCausalLM") + "Model"
        return {
            "AutoConfig": f"{configuration_module}.{self.config_class}",
            "AutoModel": f"{modeling_module}.{base_model}",
            "AutoModelForCausalLM": f"{modeling_module}.{self.architecture}",
        }

    @property
    def runtime_filenames(self) -> tuple[str, str]:
        return (self.configuration_filename, self.modeling_filename)


DEEPSEEK_V3_SPEC = HFModelSpec(
    model_type="deepseek_v3",
    template_dir=_models_root() / "DeepSeekV3",
    configuration_filename="configuration_deepseek.py",
    modeling_filename="modeling_deepseek.py",
    config_class="DeepseekV3Config",
    architecture="DeepseekV3ForCausalLM",
)

GIGACHAT3_5_SPEC = HFModelSpec(
    model_type="gigachat3_5",
    template_dir=_models_root() / "gigachat3_5",
    configuration_filename="configuration_gigachat3_5.py",
    modeling_filename="modeling_gigachat3_5.py",
    config_class="Gigachat35Config",
    architecture="Gigachat35ForCausalLM",
    modular_filename="modular_gigachat3_5.py",
    transformers_model_dir="gigachat3_5",
)

GIGACHAT3_5_DENSE_SPEC = HFModelSpec(
    model_type="gigachat3_5_dense",
    template_dir=_models_root() / "gigachat3_5_dense",
    configuration_filename="configuration_gigachat3_5_dense.py",
    modeling_filename="modeling_gigachat3_5_dense.py",
    config_class="Gigachat35DenseConfig",
    architecture="Gigachat35DenseForCausalLM",
    modular_filename="modular_gigachat3_5_dense.py",
    transformers_model_dir="gigachat3_5_dense",
)

GIGACHAT_SLM_SPEC = HFModelSpec(
    model_type="gigachat_slm",
    template_dir=_models_root() / "gigachat_slm",
    configuration_filename="configuration_gigachat_slm.py",
    modeling_filename="modeling_gigachat_slm.py",
    config_class="GigachatSLMConfig",
    architecture="GigachatSLMForCausalLM",
    modular_filename="modular_gigachat_slm.py",
    transformers_model_dir="gigachat_slm",
)

MODEL_SPECS: dict[str, HFModelSpec] = {
    spec.model_type: spec
    for spec in (GIGACHAT3_5_SPEC, DEEPSEEK_V3_SPEC, GIGACHAT3_5_DENSE_SPEC, GIGACHAT_SLM_SPEC)
}

__all__ = [
    "DEEPSEEK_V3_SPEC",
    "GIGACHAT3_5_DENSE_SPEC",
    "GIGACHAT3_5_SPEC",
    "GIGACHAT_SLM_SPEC",
    "HFModelSpec",
    "MODEL_SPECS",
]
