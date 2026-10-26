from __future__ import annotations

import copy

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
from transformers import PretrainedConfig

from .data_flow import DataFlow

if TYPE_CHECKING:
    from ..modelling_gigaomni import GigaOmniDataState


class BaseModalityConfig(PretrainedConfig):
    model_type = "giga_omni_base_modality"
    allowed_data_flow_presets: tuple[str, ...] = ()

    @staticmethod
    def merge_from_preset(
        preset_name: Optional[str],
        override: Optional[Dict[str, Any]],
        presets: Dict[str, Dict[str, Any]],
        mode: Literal["data_flow", "modality_output"],
    ) -> Dict[str, Any]:
        """Deep-copy a preset dict, then apply optional shallow key overrides.

        Args:
            preset_name: Key into ``presets`` (e.g. ``data_flow_preset`` value).
            override: If not ``None``, merged with ``{**base, **override}`` at top level only.
            presets: Mapping from preset name to dict payload.
            mode: ``data_flow`` or ``modality_output``; used only in assertion messages.

        Returns:
            Resolved dictionary after copy and optional merge.
        """
        if preset_name is None:
            return override or {}

        assert preset_name in presets, (
            f"Unknown {mode} preset {preset_name!r}; expected one of {sorted(presets.keys())}"
        )
        base = copy.deepcopy(presets[preset_name])
        if override is None:
            return base
        assert isinstance(override, dict), (
            f"{mode} override must be a dictionary or None, got {type(override)}"
        )
        return {**base, **override}

    @classmethod
    def validate_data_flow_preset_compatibility(
        cls, preset_name: Optional[str]
    ) -> None:
        if preset_name is None or not cls.allowed_data_flow_presets:
            return

        assert preset_name in cls.allowed_data_flow_presets, (
            f"Data flow preset {preset_name} is incompatible with `{cls.__name__.replace('Config', '')}`. "
            f"Expected one of {sorted(cls.allowed_data_flow_presets)}."
        )

    def __init__(
        self,
        data_flow_preset: Optional[str] = None,
        data_flow: Optional[DataFlow] = None,
        modality_output: Optional[Dict[str, Union[str, int]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_flow_preset = data_flow_preset
        self.data_flow = data_flow
        self.modality_output = modality_output

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Config", "")


class BaseModality(ABC):
    config_class = BaseModalityConfig

    def __init__(self, config: BaseModalityConfig):
        self.config = config

    def process_inputs_for_component(
        self, data_state: GigaOmniDataState, component_type: str
    ) -> None:
        if component_type == "encoder":
            self.process_inputs_for_encoders(data_state)
        elif component_type == "adapter":
            self.process_inputs_for_adapters(data_state)
        elif component_type == "head":
            self.process_inputs_for_heads(data_state)
        else:
            raise ValueError(f"Invalid component type: {component_type}")

    @abstractmethod
    def process_inputs_for_encoders(self, data_state: GigaOmniDataState) -> None:
        pass

    @abstractmethod
    def process_inputs_for_adapters(self, data_state: GigaOmniDataState) -> None:
        pass

    @abstractmethod
    def process_inputs_for_language_model(self, data_state: GigaOmniDataState) -> None:
        pass

    @abstractmethod
    def process_inputs_for_heads(self, data_state: GigaOmniDataState) -> None:
        pass

    @abstractmethod
    def update_modality_inputs_with_dummy_values(
        self, modality_inputs: Dict[str, Dict[str, Any]]
    ) -> None:
        pass

    def get_custom_input_contract(self) -> Dict[str, list[str]]:
        return {}

    @property
    def name(self) -> str:
        return self.__class__.__name__
