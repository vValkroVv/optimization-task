from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.internvl import InternVisionModel


class InternVLMultiModalProjector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        in_channels = config.params["mm_hidden_size"]
        hidden_channels = config.params["hidden_size"]
        downsample_ratio = config.params["downsample_ratio"]
        self.downsample_ratio = downsample_ratio

        self.module_f = nn.Sequential(
            nn.LayerNorm(in_channels * int(1 / downsample_ratio) ** 2),
            nn.Linear(in_channels * int(1 / downsample_ratio) ** 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        ).to(config.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module_f(x)


class GigaVisionForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()

        self.config = config
        self.quant_config = quant_config

        self.vision_tower = InternVisionModel(config=config.vision_config.ve_config)
        self.vision_tower = self.vision_tower.to(
            config.vision_config.ve_config.torch_dtype
        )
        self.select_layer = config.vision_config.params["select_layer"]

        self.vision_projector = InternVLMultiModalProjector(
            config=config.projector_config
        )
        self.downsample_ratio = self.vision_projector.downsample_ratio

        self.language_model = self._build_language_model(
            config=config.llm_config.config, quant_config=quant_config, prefix=prefix
        )

    @staticmethod
    def _get_language_model_config(config):
        llm_config = GigaVisionForCausalLM._config_get(config, "llm_config")
        return GigaVisionForCausalLM._config_get(llm_config, "config", llm_config)

    @staticmethod
    def _config_get(config: Any, key: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    @staticmethod
    def _config_set(config: Any, key: str, value: Any) -> None:
        if isinstance(config, dict):
            config[key] = value
        else:
            setattr(config, key, value)

    @staticmethod
    def _as_arch_list(archs) -> Optional[List[str]]:
        if not archs:
            return None
        if isinstance(archs, str):
            return [archs]
        return list(archs)

    @staticmethod
    def _infer_language_model_architectures(config) -> Optional[List[str]]:
        model_type = GigaVisionForCausalLM._config_get(config, "model_type")
        if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            return [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]]

        return None

    @staticmethod
    def _get_language_model_architectures(config) -> Optional[List[str]]:
        archs = GigaVisionForCausalLM._config_get(config, "architectures")

        if not archs:
            archs = GigaVisionForCausalLM._infer_language_model_architectures(config)
            if archs:
                GigaVisionForCausalLM._config_set(config, "architectures", archs)

        return GigaVisionForCausalLM._as_arch_list(archs)

    @staticmethod
    def _ensure_language_model_config(config):
        if isinstance(config, dict):
            model_type = config.get("model_type")
            built = None
            if model_type:
                config_kwargs = dict(config)
                config_kwargs.pop("model_type", None)
                try:
                    built = AutoConfig.for_model(model_type, **config_kwargs)
                except (KeyError, TypeError, ValueError):
                    pass
            config = built if built is not None else PretrainedConfig(**config)

        if hasattr(config, "standardize_rope_params") and (
            getattr(config, "rope_parameters", None) is not None
            or getattr(config, "rope_theta", None) is not None
        ):
            config.standardize_rope_params()

        return config

    def _build_language_model(
        self, config, quant_config=None, prefix: str = ""
    ) -> nn.Module:
        archs = self._get_language_model_architectures(config)
        if archs is None:
            raise ValueError(
                "GigaVision llm_config.config must specify a supported architecture."
            )

        config = self._ensure_language_model_config(config)
        from sglang.srt.models.registry import ModelRegistry

        lm_cls, _ = ModelRegistry.resolve_model_cls(archs)
        return lm_cls(config=config, quant_config=quant_config, prefix=prefix)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    # taken from InternVLChatModel.pixel_shuffle
    # removed pixel_shuffle version 1
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    # taken from InternVLChatModel.extract_feature
    # renamed vision_model -> vision_tower and mpl1 -> vision_projector
    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_tower(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_tower(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.vision_projector(vit_embeds)
        return vit_embeds

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        pixel_values = pixel_values.to(
            device=self.vision_tower.device, dtype=self.vision_tower.dtype
        )

        image_embeds = self.extract_feature(pixel_values)
        return image_embeds.reshape(-1, image_embeds.shape[-1])

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
        )

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        expected_params: Set[str] = set()
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            expected_params.add(original_name)

            # LLM part
            if name.startswith("_language_model."):
                sub_name = name[len("_language_model.") :]
                self.language_model.load_weights([(sub_name, loaded_weight)])
                loaded_params.add(original_name)
                continue

            # Vision part
            if name.startswith("_mm_vision_tower.vision_tower.vision_tower."):
                name = name.replace(
                    "_mm_vision_tower.vision_tower.vision_tower.", "vision_tower."
                )
                # strange weight names
                if ".attn.proj." in name:
                    name = name.replace(".attn.proj.", ".attn.attn.proj.")
                elif ".attn.qkv." in name:
                    name = name.replace(".attn.qkv.", ".attn.attn.qkv_proj.")
            elif name.startswith("_mm_projector.projector."):
                name = name.replace("_mm_projector.projector.", "vision_projector.")

            assert name in params_dict, (
                f"GigaVision: unknown weight tensor: {original_name} "
                f"(mapped to: {name})"
            )

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(original_name)

        assert loaded_params == expected_params, (
            "GigaVision: failed to load weights: "
            f"{sorted(expected_params - loaded_params)}"
        )
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(
        cls, config
    ) -> Optional[ModelConfigForExpertLocation]:
        llm_config = cls._get_language_model_config(config)
        if llm_config is None:
            return None

        archs = cls._get_language_model_architectures(llm_config)
        if archs is None:
            return None

        llm_config = cls._ensure_language_model_config(llm_config)
        from sglang.srt.models.registry import ModelRegistry

        lm_cls, _ = ModelRegistry.resolve_model_cls(archs)
        get_expert_location_config = getattr(
            lm_cls, "get_model_config_for_expert_location", None
        )
        if get_expert_location_config is None:
            return None

        return get_expert_location_config(llm_config)


EntryClass = GigaVisionForCausalLM
