import shutil
import torch
import torch.nn as nn

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from transformers import AutoModelForCausalLM, PreTrainedModel, GenerationConfig

from .configuration_gigaomni import GigaOmniConfig
from .utils_gigaomni import (
    build_language_model,
    build_modality_config,
    build_modality_component,
    get_tp_sp_group_size,
    import_module_if_available_from_local_path,
)

# NOTE (Sbr): This file will be used as modeling for the gigaomni in huggingface format. Thus, we need to hide
# all imports from lmmfoundry and other parts of the project as well as imports of the modules to avoid bugs in
# automatic processing of modeling files by transformers. The modalities package is handled separately in the
# from_pretrained method. Since this code may be executed as a modeling file, we also need to try doing similar
# imports from local files inside huggingface checkpoint.
modalities = import_module_if_available_from_local_path(
    "llmfoundry.models.gigaomni.modalities"
)
encoders = import_module_if_available_from_local_path(
    "llmfoundry.models.gigaomni.modalities.encoders"
)
adapters = import_module_if_available_from_local_path(
    "llmfoundry.models.gigaomni.modalities.adapters"
)
heads = import_module_if_available_from_local_path(
    "llmfoundry.models.gigaomni.modalities.heads"
)
BaseModality = import_module_if_available_from_local_path(
    "llmfoundry.models.gigaomni.modalities", "BaseModality"
)


@dataclass
class GigaOmniDataState:
    modality_inputs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    custom_inputs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    encoder_outputs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    adapter_outputs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    language_model_outputs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    head_outputs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    modality_outputs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # Bundle -> keys the collator emitted, before dummies were injected.
    real_modality_input_keys: Dict[str, Set[str]] = field(default_factory=dict)

    # A modality that changes the text length (video token reduction) pads it
    # back to a multiple of this, the divisor the language model shards by,
    # with this token id.
    pad_to_multiple_of: int = 1
    pad_token_id: int = 0

    def get(
        self,
        type: str,
        name: Optional[str] = None,
        key: Optional[str] = None,
        keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        data = getattr(self, type)

        if name is not None:
            data = data[name]

        assert key is None or keys is None, (
            "Only one of `key` or `keys` parameters can be provided!"
        )

        if key is not None:
            data = data[key]

        if keys is not None:
            data = {k: v for k, v in data.items() if k in keys}

        return data

    def update(self, tp: str, name: str, value: Dict[str, Any]) -> None:
        getattr(self, tp)[name].update(value)


class GigaOmniForCausalLM(PreTrainedModel):
    model_type = "gigaomni"
    config_class = GigaOmniConfig
    _tp_plan = {}

    def __init__(self, config: GigaOmniConfig):
        super().__init__(config)

        self.encoders = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.modalities: List[BaseModality] = []

        self.build_modality_modules(config.modalities)
        self.language_model = build_language_model(config.language_model)

    def forward(
        self,
        input_ids: torch.LongTensor,
        modality_inputs: Dict[str, Dict[str, torch.Tensor]],
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        real_modality_input_keys = self._real_modality_input_keys(modality_inputs)

        for modality in self.modalities:
            modality.update_modality_inputs_with_dummy_values(modality_inputs)

        data_state = GigaOmniDataState(
            modality_inputs=defaultdict(dict, modality_inputs),
            real_modality_input_keys=real_modality_input_keys,
            pad_to_multiple_of=2 * (get_tp_sp_group_size() or 1),
            pad_token_id=self.language_model.config.pad_token_id or 0,
        )

        # NOTE (Sbr): Some parts of the model (e.g. video token reduction module) can change total number of
        # multimodal tokens in the input. In this case, values are only changed in modality_inputs["text"],
        # so we need to use values from data_state instead of forward method arguments.
        data_state.update(
            "modality_inputs",
            "text",
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
        )

        self._forward_components(data_state, self.encoders, "encoder")

        data_state.update(
            "encoder_outputs",
            "text",
            {
                "inputs_embeds": self.language_model.get_input_embeddings()(
                    data_state.get("modality_inputs", "text", key="input_ids")
                )
            },
        )

        self._forward_components(data_state, self.adapters, "adapter")

        for modality in self.modalities:
            modality.process_inputs_for_language_model(data_state)

        inputs_embeds = self.merge_modality_embeddings(data_state)

        lm_kwargs = {}
        # Opt-in: the language models an hf_omni export vendors either reject this
        # kwarg or read it as their own attention mask.
        if getattr(self.language_model, "accepts_padding_mask", False):
            # From the final input_ids (after token reduction and the re-pad), so the
            # MoE router statistics skip every pad; attention never reads this mask.
            lm_kwargs["padding_mask"] = (
                data_state.get("modality_inputs", "text", key="input_ids")
                != data_state.pad_token_id
            )
            attention_mask = data_state.get(
                "modality_inputs", "text", key="attention_mask"
            )

            if attention_mask is not None:
                # Positions attention skips stay out of the router as well.
                lm_kwargs["padding_mask"] &= attention_mask.bool()

        llm_outputs = self.language_model.forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            labels=data_state.get("modality_inputs", "text", key="labels"),
            attention_mask=data_state.get(
                "modality_inputs", "text", key="attention_mask"
            ),
            position_ids=data_state.get("modality_inputs", "text", key="position_ids"),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=self.language_model.config.return_dict,
            **lm_kwargs,
            **kwargs,
        )

        data_state.update("language_model_outputs", "text", {"logits": llm_outputs[1]})

        # TODO: Add logic to forward heads

        return llm_outputs

    def merge_modality_embeddings(self, data_state: Dict[str, Any]) -> torch.Tensor:
        llm_inputs_embeds = data_state.get(
            "encoder_outputs", "text", key="inputs_embeds"
        )
        flat_llm_inputs_embeds = llm_inputs_embeds.reshape(
            -1, llm_inputs_embeds.shape[-1]
        )
        flat_input_ids = data_state.get(
            "modality_inputs", "text", key="input_ids"
        ).reshape(-1)

        for modality in self.modalities:
            modality_output_cfg = modality.config.modality_output
            placeholder_token_id = modality_output_cfg["placeholder_token_id"]

            modality_embeds = data_state.get(
                **{
                    k: v
                    for k, v in modality_output_cfg.items()
                    if k in ["type", "name", "key"]
                }
            )
            flat_modality_embeds = modality_embeds.reshape(
                -1, modality_embeds.shape[-1]
            )

            flat_mask = flat_input_ids == placeholder_token_id
            num_modality_tokens = flat_mask.sum()
            modality_token_count_matches = (
                num_modality_tokens == flat_modality_embeds.shape[0]
            )
            token_count_error = (
                f"Number of multimodal tokens {placeholder_token_id} in input_ids "
                f"must equal the number of output tokens from modality "
                f"{modality.name}: expected {flat_modality_embeds.shape[0]}."
            )

            # Only a dummy input may produce rows for no placeholder; a real input
            # with none means the processor inserted a different placeholder id.
            if self._has_real_input(modality, data_state):
                token_count_error += (
                    " The modality got a real input, so its placeholder id must "
                    "match the id its processor inserts."
                )
            else:
                modality_token_count_matches = torch.logical_or(
                    num_modality_tokens == 0, modality_token_count_matches
                )
            if modality_token_count_matches.device.type == "cpu":
                assert modality_token_count_matches, token_count_error
            else:
                torch._assert_async(
                    modality_token_count_matches,
                    token_count_error,
                )

            assert llm_inputs_embeds.shape[-1] == flat_modality_embeds.shape[-1], (
                f"Hidden size of inputs_embeds is not equal to the hidden size of output tokens from modality {modality.name}: "
                f"{llm_inputs_embeds.shape[-1]} vs {flat_modality_embeds.shape[-1]}!"
            )

            flat_llm_inputs_embeds = torch.masked_scatter(
                flat_llm_inputs_embeds, flat_mask[:, None], flat_modality_embeds
            )

        return flat_llm_inputs_embeds.reshape(llm_inputs_embeds.shape).contiguous()

    @staticmethod
    def _real_modality_input_keys(
        modality_inputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Set[str]]:
        """Keys the collator emitted per bundle.

        A bundle a dummy created is marked `dummy_value` and stays out: the
        dummies are injected in place, so a second forward on the same batch
        (GigaFSDP's order-recording first call) or a serving batch built with
        its own dummies must not take them for real inputs.
        """
        return {
            bundle_name: set(bundle)
            for bundle_name, bundle in modality_inputs.items()
            if not bundle.get("dummy_value", False)
        }

    @staticmethod
    def _has_real_input(modality, data_state: GigaOmniDataState) -> bool:
        """Whether the collator emitted the modality's first encoder input."""
        first_input = modality.config.data_flow.encoders[0].inputs[0]
        real_keys = data_state.real_modality_input_keys.get(first_input.name, ())

        return (
            first_input.type == "modality_inputs" and first_input.keys[0] in real_keys
        )

    def _forward_components(
        self,
        data_state: GigaOmniDataState,
        components: nn.ModuleDict,
        component_type: str,
    ) -> None:
        for modality in self.modalities:
            modality.process_inputs_for_component(data_state, component_type)

        for name, component in components.items():
            inputs = []
            all_required_outputs = []

            for modality in self.modalities:
                required_inputs, required_outputs = (
                    modality.config.data_flow.get_required_inputs_and_outputs_for_component(
                        name
                    )
                )
                for input_item in required_inputs:
                    inputs.append(data_state.get(**input_item))

                all_required_outputs.extend(required_outputs)

            # Reaches the wrapped module through the gigafsdp wrappers' __getattr__.
            component.set_top_level_training(self.training)
            outputs = component(inputs)

            assert hasattr(component, "unwrap_tuple_output"), (
                "Each component must have `unwrap_tuple_output` method that will convert tuple output to dict output!"
            )

            dict_outputs = component.unwrap_tuple_output(inputs, outputs)
            for modality in self.modalities:
                modality.config.data_flow.validate_runtime_outputs(
                    dict_outputs=dict_outputs,
                    required_outputs=all_required_outputs,
                    component_name=name,
                )

            for dict_output, required_output in zip(dict_outputs, all_required_outputs):
                data_state.update(
                    f"{component_type}_outputs",
                    required_output["name"],
                    {
                        k: v
                        for k, v in dict_output.items()
                        if k in required_output["keys"]
                    },
                )

    def build_modality_modules(self, modality_configs: List[Dict[str, Any]]) -> None:
        for modality_config in modality_configs:
            modality_config = build_modality_config(modality_config)
            self.modalities.append(
                getattr(modalities, modality_config.name)(modality_config)
            )

        multimodal_token_ids = {
            modality.config.modality_output["placeholder_token_id"]
            for modality in self.modalities
        }
        assert len(multimodal_token_ids) == len(self.modalities), (
            "All used modalities must have different placeholder token ids!"
        )

        for encoder_config in self.config.encoders:
            self.encoders.update(build_modality_component(encoder_config, encoders))

        for adapter_config in self.config.adapters:
            self.adapters.update(build_modality_component(adapter_config, adapters))

        for head_config in self.config.heads:
            self.heads.update(build_modality_component(head_config, heads))

        for modality in self.modalities:
            modality.config.data_flow.validate_graph(
                available_components={
                    "encoders": list(self.encoders.keys()),
                    "adapters": list(self.adapters.keys()),
                    "heads": list(self.heads.keys()),
                },
                modality_output=modality.config.modality_output,
                modality_name=modality.name,
                custom_input_contract=modality.get_custom_input_contract(),
            )

    def param_init_fn(self, module: torch.nn.Module):
        if (
            hasattr(self.language_model, "param_init_fn")
            and module in self.language_model.modules()
        ):
            self.language_model.param_init_fn(module)
            return

        if getattr(module, "reset_parameters", False):
            module.reset_parameters()
        else:
            raise ValueError(
                f"Module {module.__class__.__name__} is on meta device but can't be initialized!"
            )

    def save_pretrained(self, save_directory: str, **kwargs):
        super().save_pretrained(save_directory, **kwargs)

        # NOTE (Sbr): Automatic processing of modeling files from transformers can't handle folders, so we need to copy
        # the modalities folder to the checkpoint destination by ourselves.
        if (Path(save_directory) / "modalities").exists():
            shutil.rmtree(Path(save_directory) / "modalities")
        shutil.copytree(
            Path(__file__).parent / "modalities", Path(save_directory) / "modalities"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        config: Optional[GigaOmniConfig] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        # NOTE (Sbr): The value of pretrained_model_name_or_path for llm must be absolute, but this path may be invalidated after
        # moving checkpoint to another location. Thus, we will update it every time we load the model.
        assert config is not None, (
            "Parameter `config` is automatically passed by transformers and can't be None!"
        )
        llm_pretrained_model_name_or_path = str(
            Path(pretrained_model_name_or_path).absolute() / "language_model"
        )
        config.language_model["pretrained_model_name_or_path"] = (
            llm_pretrained_model_name_or_path
        )

        return super().from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            force_download=True,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        modality_inputs: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> torch.LongTensor:
        modality_inputs = modality_inputs or {}
        real_modality_input_keys = self._real_modality_input_keys(modality_inputs)

        for modality in self.modalities:
            modality.update_modality_inputs_with_dummy_values(modality_inputs)

        data_state = GigaOmniDataState(
            modality_inputs=defaultdict(dict, modality_inputs),
            real_modality_input_keys=real_modality_input_keys,
        )
        data_state.update(
            "modality_inputs",
            "text",
            {
                "input_ids": inputs,
                "labels": None,
                "attention_mask": attention_mask,
                "position_ids": kwargs.get("position_ids", None),
            },
        )

        self._forward_components(data_state, self.encoders, "encoder")

        data_state.update(
            "encoder_outputs",
            "text",
            {
                "inputs_embeds": self.language_model.get_input_embeddings()(
                    data_state.get("modality_inputs", "text", key="input_ids")
                )
            },
        )

        self._forward_components(data_state, self.adapters, "adapter")

        for modality in self.modalities:
            modality.process_inputs_for_language_model(data_state)

        inputs_embeds = self.merge_modality_embeddings(data_state)

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=data_state.get(
                "modality_inputs", "text", key="attention_mask"
            ),
            generation_config=generation_config,
            **kwargs,
        )


GigaOmniForCausalLM.register_for_auto_class(AutoModelForCausalLM)
