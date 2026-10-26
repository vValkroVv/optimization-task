import sys
import importlib
import torch
import torch.nn as nn

from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Tuple, Dict, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict
from transformers import PretrainedConfig, AutoConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.managers.schedule_batch import MultimodalInputs, MultimodalDataItem
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens


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


class GigaOmniForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        model_path: str,
        quant_config: Optional[QuantizationConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = quant_config

        (
            gigaomni_utils_module,
            modalities_module,
            encoders_module,
            adapters_module,
        ) = self._load_modules_from_checkpoint(Path(model_path))

        self.modalities = self._build_modalities(gigaomni_utils_module, modalities_module)

        self.encoders = self._build_checkpoint_component(
            gigaomni_utils_module, encoders_module, self.config.encoders
        )
        self.adapters = self._build_checkpoint_component(
            gigaomni_utils_module, adapters_module, self.config.adapters
        )

        self.language_model = self._build_language_model()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        if (
            not forward_batch.forward_mode.is_decode() and
            not forward_batch.forward_mode.is_target_verify() and
            not torch.cuda.is_current_stream_capturing() and 
            forward_batch.contains_mm_inputs()
        ):
            modality_inputs = self._get_modality_inputs_from_mm_inputs(forward_batch.mm_inputs)            

            for modality in self.modalities:
                modality.update_modality_inputs_with_dummy_values(modality_inputs)

            data_state = GigaOmniDataState(
                modality_inputs=defaultdict(dict, modality_inputs)
            )
            data_state.update("modality_inputs", "text", {
                "input_ids": input_ids,
                "extend_prefix_lens_cpu": forward_batch.extend_prefix_lens_cpu,
                "extend_seq_lens_cpu": forward_batch.extend_seq_lens_cpu,
            })

            self._forward_components(data_state, self.encoders, "encoder")

            embeddings = self.language_model.get_input_embeddings()
            text_input_ids = data_state.get("modality_inputs", "text", key="input_ids")

            # mm placeholder ids are fake (>= vocab), embed 0 instead; merge overwrites them
            mm_placeholder_mask = text_input_ids >= embeddings.num_embeddings
            data_state.update("modality_inputs", "text", {"mm_placeholder_mask": mm_placeholder_mask})
            data_state.update(
                "encoder_outputs",
                "text",
                {"inputs_embeds": embeddings(text_input_ids.masked_fill(mm_placeholder_mask, 0))},
            )

            self._forward_components(data_state, self.adapters, "adapter")

            for modality in self.modalities:
                modality.process_inputs_for_language_model(data_state)

            inputs_embeds = self._merge_modality_embeddings(data_state)
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        return self.language_model(
            input_ids=None,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        lm_weights = []

        for name, loaded_weight in weights:
            loaded_params.add(name)

            if name.startswith("language_model."):
                lm_weights.append((name[len("language_model.") :], loaded_weight))
                continue

            default_weight_loader(params_dict[name], loaded_weight)

        self.language_model.load_weights(lm_weights)
        return loaded_params

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

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

            inputs = self._convert_component_inputs_to_device_dtype(inputs, component)
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

    def _merge_modality_embeddings(self, data_state: GigaOmniDataState) -> torch.Tensor:
        llm_inputs_embeds = data_state.get(
            "encoder_outputs", "text", key="inputs_embeds"
        )
        flat_llm_inputs_embeds = llm_inputs_embeds.reshape(
            -1, llm_inputs_embeds.shape[-1]
        )
        mm_placeholder_mask = data_state.get(
            "modality_inputs", "text", key="mm_placeholder_mask"
        ).reshape(-1)
        covered = torch.zeros_like(mm_placeholder_mask)

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

            required_data_modalities = []
            for encoder in modality.config.data_flow.encoders:
                required_data_modalities.extend([input.name for input in encoder.inputs])

            offsets = []
            batch_indices = []
            for required_data_modality in required_data_modalities:
                if not data_state.get("modality_inputs", required_data_modality).get("dummy_value", False):
                    offsets.extend(data_state.get("modality_inputs", required_data_modality, key="offsets"))
                    batch_indices.extend(data_state.get("modality_inputs", required_data_modality, key="batch_indices"))

            extend_seq_lens_cpu = data_state.get("modality_inputs", "text", key="extend_seq_lens_cpu")
            extend_seq_lens_cummulative_cpu = torch.cumsum(torch.tensor([0] + extend_seq_lens_cpu), dim=0).tolist()
            extend_prefix_lens_cpu = data_state.get("modality_inputs", "text", key="extend_prefix_lens_cpu")

            num_modality_tokens = sum(offset[1] - offset[0] + 1 for offset in offsets)
            assert (
                num_modality_tokens == 0
                or num_modality_tokens == flat_modality_embeds.shape[0]
            ), (
                f"Number of multimodal tokens `{placeholder_token_id}` in input_ids is not equal to the number of output tokens from modality {modality.name}: "
                f"{num_modality_tokens} vs {flat_modality_embeds.shape[0]}!"
            )

            assert llm_inputs_embeds.shape[-1] == flat_modality_embeds.shape[-1], (
                f"Hidden size of inputs_embeds is not equal to the hidden size of output tokens from modality {modality.name}: "
                f"{llm_inputs_embeds.shape[-1]} vs {flat_modality_embeds.shape[-1]}!"
            )

            modality_embeds_start_idx = 0
            for batch_idx, offset in zip(batch_indices, offsets):
                mm_start, mm_end = offset
                mm_len = mm_end - mm_start + 1
                modality_embeds_end_idx = modality_embeds_start_idx + mm_len

                extend_start = extend_prefix_lens_cpu[batch_idx]
                extend_end = extend_start + extend_seq_lens_cpu[batch_idx] - 1

                overlap_start = max(mm_start, extend_start)
                overlap_end = min(mm_end, extend_end)

                if overlap_start <= overlap_end:
                    chunk_start_idx = (
                        extend_seq_lens_cummulative_cpu[batch_idx]
                        + overlap_start
                        - extend_start
                    )
                    chunk_end_idx = chunk_start_idx + overlap_end - overlap_start + 1

                    embeddings_start_idx = modality_embeds_start_idx + overlap_start - mm_start
                    embeddings_end_idx = embeddings_start_idx + overlap_end - overlap_start + 1

                    flat_llm_inputs_embeds[chunk_start_idx:chunk_end_idx] = (
                        flat_modality_embeds[embeddings_start_idx:embeddings_end_idx]
                    )
                    covered[chunk_start_idx:chunk_end_idx] = True

                modality_embeds_start_idx = modality_embeds_end_idx

        assert bool(covered[mm_placeholder_mask].all()), (
            "Some mm placeholder positions were not overwritten by modality embeddings!"
        )
        return flat_llm_inputs_embeds.reshape(llm_inputs_embeds.shape).contiguous()

    def _get_modality_inputs_from_mm_inputs(self, mm_inputs: List[MultimodalInputs]) -> Dict[str, Dict[str, Any]]:
        modality_inputs = defaultdict(list, [])
        for batch_idx, mm_input in enumerate(mm_inputs):
            if mm_input is None:
                continue

            for mm_item in mm_input.mm_items:
                mm_item.batch_idx = batch_idx
                modality_inputs[mm_item.modality.name.lower()].append(mm_item)

        for modality_str, modality_input in modality_inputs.items():
            modality_inputs[modality_str] = self._collate_modality_input(modality_input, modality_str)
        return modality_inputs

    @classmethod
    def _collate_modality_input(
        cls,
        modality_input: List[MultimodalDataItem],
        modality_str: str,
    ) -> Dict[str, Any]:
        if len(modality_input) == 1:
            feature = {**modality_input[0].feature}
            feature["offsets"] = modality_input[0].offsets
            feature["batch_indices"] = [modality_input[0].batch_idx for _ in range(len(modality_input[0].offsets))]
            return feature

        assert modality_str in ["image", "audio"], \
            f"Inferencing with batch_size > 1 supported only for image and audio modalities, but got {modality_str}!"

        collated: Dict[str, Any] = {}
        possible_keys = sorted(list(modality_input[0].feature.keys()))

        assert all(sorted(mm_item.feature.keys()) == possible_keys for mm_item in modality_input), \
            f"All modality inputs must have the same keys!"

        for key in possible_keys:
            values = [mm_item.feature[key] for mm_item in modality_input]
            collated[key] = torch.cat(values, dim=0)

        collated["offsets"] = sum([mm_item.offsets for mm_item in modality_input], [])
        collated["batch_indices"] = sum([[mm_item.batch_idx] * len(mm_item.offsets) for mm_item in modality_input], [])
        return collated

    @staticmethod
    def _torch_dtype(config: PretrainedConfig) -> torch.dtype:
        dtype = getattr(config, "dtype", getattr(config, "torch_dtype", None))
        if isinstance(dtype, str):
            return getattr(torch, dtype)
        if isinstance(dtype, torch.dtype):
            return dtype
        return torch.bfloat16

    @staticmethod
    def _load_modules_from_checkpoint(
        checkpoint_root: Path,
    ) -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
        if not (checkpoint_root / "modalities").is_dir():
            raise ValueError(
                f"GigaOmni checkpoint root `{checkpoint_root}` has no `modalities` "
                "directory."
            )

        with _prepend_sys_path(checkpoint_root):
            gigaomni_utils_module = importlib.import_module("utils_gigaomni")
            modalities_module = importlib.import_module("modalities")
            encoders_module = importlib.import_module("modalities.encoders")
            adapters_module = importlib.import_module("modalities.adapters")
        return gigaomni_utils_module, modalities_module, encoders_module, adapters_module

    def _build_checkpoint_component(
        self,
        checkpoint_utils: ModuleType,
        component_module: ModuleType,
        component_configs: dict[str, Any],
    ) -> nn.ModuleDict:
        components = nn.ModuleDict()
        for config in component_configs:
            components.update(checkpoint_utils.build_modality_component(
                dict[str, Any](config), component_module
            ))
        return components.to(self._torch_dtype(self.config))

    def _build_language_model_config(self) -> PretrainedConfig:
        try:
            if "model_type" in self.config.language_model:
                lm_dict_config = dict(self.config.language_model)
                model_type = lm_dict_config.pop("model_type", None)
                return AutoConfig.for_model(model_type, **lm_dict_config)
        except ValueError:
            pass

        return PretrainedConfig(**self.config.language_model)

    def _build_language_model(self) -> nn.Module:
        from sglang.srt.models.registry import ModelRegistry

        archs = self.config.language_model["architectures"]
        assert archs is not None, "GigaOmni language_model config must specify a supported architecture."
        lm_cfg = self._build_language_model_config()
        lm_cls, _ = ModelRegistry.resolve_model_cls(archs)
        return lm_cls(config=lm_cfg, quant_config=self.quant_config, prefix="")

    def _build_modalities(
        self,
        checkpoint_utils: ModuleType,
        modalities_module: ModuleType,
    ) -> List[Any]:
        modalities = []

        for modality_config in self.config.modalities:
            modality_config = checkpoint_utils.build_modality_config(modality_config)
            modalities.append(
                getattr(modalities_module, modality_config.name)(modality_config)
            )

        return modalities

    def _convert_component_inputs_to_device_dtype(
        self,
        inputs: List[Dict[str, Any]],
        component: nn.Module,
    ) -> List[Dict[str, Any]]:
        component_device, component_dtype = self._get_component_device_dtype(component)
        return [
            self._convert_value_to_device_dtype(
                input_item,
                device=component_device,
                dtype=component_dtype,
            )
            for input_item in inputs
        ]

    def _get_component_device_dtype(
        self, component: nn.Module
    ) -> Tuple[torch.device, torch.dtype]:
        try:
            ref_tensor = next(component.parameters())
        except StopIteration:
            try:
                ref_tensor = next(component.buffers())
            except StopIteration:
                return torch.device("cuda"), self._torch_dtype(self.config)

        dtype = (
            ref_tensor.dtype
            if torch.is_floating_point(ref_tensor)
            else self._torch_dtype(self.config)
        )
        return ref_tensor.device, dtype

    def _convert_value_to_device_dtype(
        self,
        value: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)

        if isinstance(value, dict):
            return {
                key: self._convert_value_to_device_dtype(val, device, dtype)
                for key, val in value.items()
            }

        if isinstance(value, list):
            return [
                self._convert_value_to_device_dtype(item, device, dtype)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._convert_value_to_device_dtype(item, device, dtype)
                for item in value
            )

        return value


@contextmanager
def _prepend_sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


EntryClass = GigaOmniForCausalLM