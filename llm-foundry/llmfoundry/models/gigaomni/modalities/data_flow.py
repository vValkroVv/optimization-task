from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..modelling_gigaomni import GigaOmniDataState


VALID_INPUT_TYPES = {
    "modality_inputs",
    "custom_inputs",
    "encoder_outputs",
    "adapter_outputs",
    "language_model_outputs",
}

VALID_MODALITY_OUTPUT_TYPES = {"encoder_outputs", "adapter_outputs", "head_outputs"}

OUTPUT_TYPE_BY_STAGE = {
    "encoders": "encoder_outputs",
    "adapters": "adapter_outputs",
    "heads": "head_outputs",
}

ALLOWED_INPUT_TYPES_BY_STAGE = {
    "encoders": {"modality_inputs", "custom_inputs"},
    "adapters": {"modality_inputs", "custom_inputs", "encoder_outputs"},
    "heads": {
        "modality_inputs",
        "custom_inputs",
        "encoder_outputs",
        "adapter_outputs",
        "language_model_outputs",
    },
}


class DataFlowValidationError(ValueError):
    def __init__(self, message: str, errors: Optional[List[str]] = None) -> None:
        self.errors = errors or []
        if self.errors:
            message = f"{message}\n" + "\n".join(f"- {error}" for error in self.errors)
        super().__init__(message)


def _raise_validation_error(message: str, errors: List[str]) -> None:
    if errors:
        raise DataFlowValidationError(message, errors)


def _ensure_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise DataFlowValidationError(
            f"{field_name} must be a string, got {type(value).__name__}."
        )
    if not value:
        raise DataFlowValidationError(f"{field_name} must not be empty.")


def _ensure_list(value: Any, field_name: str, non_empty: bool = True) -> None:
    if not isinstance(value, list):
        raise DataFlowValidationError(
            f"{field_name} must be a list, got {type(value).__name__}."
        )
    if non_empty and not value:
        raise DataFlowValidationError(f"{field_name} must not be empty.")


def _ensure_string_list(value: Any, field_name: str) -> None:
    _ensure_list(value, field_name)
    if not all(isinstance(item, str) for item in value):
        raise DataFlowValidationError(f"{field_name} must contain only strings.")


def _ensure_list_has_no_duplicates(value: List[str], name: str) -> None:
    duplicates = _find_duplicates(value)
    if duplicates:
        raise DataFlowValidationError(
            f"Input item {name} contains duplicate keys: {sorted(duplicates)}."
        )


def _ensure_list_has_values_of_one_type_or_empty(
    value: List[Any], expected_type: type, field_name: str
) -> None:
    unique_types = set(type(item) for item in value)
    if len(unique_types) > 1:
        raise DataFlowValidationError(
            f"{field_name} must contain only values of one type, got {unique_types}."
        )
    if unique_types and expected_type not in unique_types:
        raise DataFlowValidationError(
            f"{field_name} must contain only values of type {type}, got {unique_types}."
        )


def _find_duplicates(items: Iterable[str]) -> List[str]:
    counts = defaultdict(int)
    duplicates = []
    for item in items:
        counts[item] += 1
        if counts[item] == 2:
            duplicates.append(item)
    return duplicates


@dataclass(frozen=True)
class DataFlowInputItem:
    type: str
    name: str
    keys: List[str]

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.type, "Input item `type`")
        _ensure_non_empty_string(self.name, "Input item `name`")
        _ensure_string_list(self.keys, "Input item `keys`")
        _ensure_list_has_no_duplicates(self.keys, self.name)
        if self.type not in VALID_INPUT_TYPES:
            raise DataFlowValidationError(
                f"Invalid input item type `{self.type}`. Expected one of {sorted(VALID_INPUT_TYPES)}.",
            )


@dataclass(frozen=True)
class DataFlowOutputItem:
    name: str
    keys: List[str]

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.name, "Output item `name`")
        _ensure_string_list(self.keys, "Output item `keys`")
        _ensure_list_has_no_duplicates(self.keys, self.name)


@dataclass(frozen=True)
class DataFlowComponent:
    name: str
    inputs: List[DataFlowInputItem]
    outputs: List[DataFlowOutputItem]

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.name, "Component `name`")
        _ensure_list(self.inputs, f"Component `{self.name}` outputs")
        _ensure_list(self.outputs, f"Component `{self.name}` outputs")
        _ensure_list_has_values_of_one_type_or_empty(
            self.inputs, DataFlowInputItem, f"Component `{self.name}` inputs"
        )
        _ensure_list_has_values_of_one_type_or_empty(
            self.outputs, DataFlowOutputItem, f"Component `{self.name}` outputs"
        )


@dataclass(frozen=True)
class DataFlow:
    encoders: List[DataFlowComponent] = field(default_factory=list)
    adapters: List[DataFlowComponent] = field(default_factory=list)
    heads: List[DataFlowComponent] = field(default_factory=list)

    def __post_init__(self) -> None:
        for stage_name, components in self._iter_stages():
            _ensure_list(components, f"DataFlow field `{stage_name}`", non_empty=False)
            _ensure_list_has_values_of_one_type_or_empty(
                components, DataFlowComponent, f"DataFlow field `{stage_name}`"
            )

    def get_required_inputs_and_outputs_for_component(
        self, component_name: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        required_inputs = []
        required_outputs = []
        for component in self.encoders + self.adapters + self.heads:
            if component.name != component_name:
                continue

            required_inputs += [asdict(input_item) for input_item in component.inputs]
            required_outputs += [
                asdict(output_item) for output_item in component.outputs
            ]

        return required_inputs, required_outputs

    def validate(
        self, encoder_names: List[str], adapter_names: List[str], head_names: List[str]
    ) -> None:
        self.validate_graph(
            available_components={
                "encoders": encoder_names,
                "adapters": adapter_names,
                "heads": head_names,
            },
        )

    def validate_schema(self) -> None:
        errors = []
        produced_bundles = defaultdict(lambda: defaultdict(list))

        for stage_name, components in self._iter_stages():
            for component in components:
                if len(component.inputs) != len(component.outputs):
                    errors.append(
                        f"Component `{component.name}` in stage `{stage_name}` has "
                        f"{len(component.inputs)} inputs and {len(component.outputs)} outputs; they must be equal.",
                    )

                for output_item in component.outputs:
                    produced_bundles[stage_name][output_item.name].append(
                        component.name
                    )

        for stage_name, bundles in produced_bundles.items():
            for bundle_name, producers in bundles.items():
                if len(producers) > 1:
                    errors.append(
                        f"Output bundle `{bundle_name}` is produced by multiple components in stage `{stage_name}`: "
                        f"{sorted(producers)}.",
                    )

        _raise_validation_error("DataFlow schema validation failed.", errors)

    def validate_graph(
        self,
        available_components: Dict[str, List[str]],
        modality_output: Optional[Dict[str, Any]] = None,
        modality_name: Optional[str] = None,
        custom_input_contract: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        errors = []

        if custom_input_contract is not None:
            errors.extend(
                self._validate_custom_input_contract(
                    custom_input_contract, modality_name
                )
            )

        produced_outputs = self.get_produced_outputs()
        custom_input_contract = custom_input_contract or {}

        for stage_name, components in self._iter_stages():
            available_stage_components = set(available_components.get(stage_name, []))
            missing_components = sorted(
                component.name
                for component in components
                if component.name not in available_stage_components
            )
            if missing_components:
                errors.append(
                    f"Stage `{stage_name}` requires missing components: {missing_components}."
                )

            allowed_input_types = ALLOWED_INPUT_TYPES_BY_STAGE[stage_name]
            for component in components:
                for input_item in component.inputs:
                    if input_item.type not in allowed_input_types:
                        errors.append(
                            f"Component `{component.name}` in stage `{stage_name}` cannot consume `{input_item.type}`."
                        )
                        continue

                    if input_item.type in {"encoder_outputs", "adapter_outputs"}:
                        produced_bundle = produced_outputs[input_item.type].get(
                            input_item.name
                        )
                        if produced_bundle is None:
                            errors.append(
                                f"Component `{component.name}` requires missing bundle `{input_item.name}` "
                                f"from `{input_item.type}`.",
                            )
                            continue

                        missing_keys = sorted(set(input_item.keys) - produced_bundle)
                        if missing_keys:
                            errors.append(
                                f"Component `{component.name}` requires missing keys {missing_keys} "
                                f"from `{input_item.type}.{input_item.name}`.",
                            )

                    if input_item.type == "custom_inputs":
                        declared_keys = custom_input_contract.get(input_item.name)
                        if declared_keys is None:
                            errors.append(
                                f"Component `{component.name}` requires undeclared custom input bundle `{input_item.name}`.",
                            )
                            continue

                        missing_keys = sorted(set(input_item.keys) - set(declared_keys))
                        if missing_keys:
                            errors.append(
                                f"Component `{component.name}` requires missing custom input keys {missing_keys} "
                                f"from `{input_item.name}`.",
                            )

        if modality_output is not None:
            errors.extend(
                self._validate_modality_output(
                    modality_output, modality_name, produced_outputs
                )
            )

        _raise_validation_error("DataFlow graph validation failed.", errors)

    def validate_runtime_inputs(
        self, data_state: GigaOmniDataState, components: List[DataFlowComponent]
    ) -> None:
        errors = []
        for component in components:
            for input_item in component.inputs:
                current_type_state = getattr(data_state, input_item.type, None)
                if current_type_state is None:
                    errors.append(
                        f"Component `{component.name}` uses unknown runtime input type `{input_item.type}`.",
                    )
                    continue

                if input_item.name not in current_type_state:
                    errors.append(
                        f"Component `{component.name}` requires `{input_item.name}` in `{input_item.type}`, "
                        f"but only {sorted(current_type_state.keys())} are available.",
                    )
                    continue

                available_keys = current_type_state[input_item.name]
                missing_keys = sorted(
                    key for key in input_item.keys if key not in available_keys
                )
                if missing_keys:
                    errors.append(
                        f"Component `{component.name}` requires keys {missing_keys} from "
                        f"`{input_item.type}.{input_item.name}`, but only {sorted(available_keys.keys())} are available.",
                    )

        _raise_validation_error("DataFlow runtime input validation failed.", errors)

    def validate_runtime_outputs(
        self,
        dict_outputs: List[Dict[str, Any]],
        required_outputs: List[Dict[str, Any]],
        component_name: str,
    ) -> None:
        errors = []

        if not isinstance(dict_outputs, list):
            raise DataFlowValidationError(
                f"Component `{component_name}` returned `{type(dict_outputs).__name__}` from `unwrap_tuple_output`, expected `list`.",
            )

        if len(dict_outputs) != len(required_outputs):
            errors.append(
                f"Component `{component_name}` returned {len(dict_outputs)} outputs, "
                f"but DataFlow expects {len(required_outputs)}.",
            )

        for index, required_output in enumerate(required_outputs):
            if index >= len(dict_outputs):
                break

            dict_output = dict_outputs[index]
            if not isinstance(dict_output, dict):
                errors.append(
                    f"Component `{component_name}` output at index {index} must be a dict, "
                    f"got {type(dict_output).__name__}.",
                )
                continue

            missing_keys = sorted(
                set(required_output["keys"]) - set(dict_output.keys())
            )
            if missing_keys:
                errors.append(
                    f"Component `{component_name}` output `{required_output['name']}` is missing keys {missing_keys}.",
                )

        _raise_validation_error("DataFlow runtime output validation failed.", errors)

    def get_produced_outputs(self) -> Dict[str, Dict[str, set[str]]]:
        produced_outputs: Dict[str, Dict[str, set[str]]] = {
            output_type: {} for output_type in OUTPUT_TYPE_BY_STAGE.values()
        }
        for stage_name, components in self._iter_stages():
            output_type = OUTPUT_TYPE_BY_STAGE[stage_name]
            for component in components:
                for output_item in component.outputs:
                    produced_outputs[output_type][output_item.name] = set(
                        output_item.keys
                    )
        return produced_outputs

    def _validate_modality_output(
        self,
        modality_output: Dict[str, Any],
        modality_name: Optional[str],
        produced_outputs: Dict[str, Dict[str, set[str]]],
    ) -> List[str]:
        errors = []
        context = f" for modality `{modality_name}`" if modality_name else ""

        if not isinstance(modality_output, dict):
            return [f"`modality_output`{context} must be a dictionary."]

        output_type = modality_output.get("type")
        bundle_name = modality_output.get("name")
        output_key = modality_output.get("key")

        if output_type not in VALID_MODALITY_OUTPUT_TYPES:
            errors.append(
                f"`modality_output.type`{context} must be one of {sorted(VALID_MODALITY_OUTPUT_TYPES)}, got {output_type!r}.",
            )
            return errors

        if not isinstance(bundle_name, str) or not bundle_name:
            errors.append(
                f"`modality_output.name`{context} must be a non-empty string."
            )
            return errors

        if not isinstance(output_key, str) or not output_key:
            errors.append(f"`modality_output.key`{context} must be a non-empty string.")
            return errors

        produced_bundle = produced_outputs[output_type].get(bundle_name)
        if produced_bundle is None:
            errors.append(
                f"`modality_output`{context} references missing bundle `{bundle_name}` in `{output_type}`.",
            )
            return errors

        if output_key not in produced_bundle:
            errors.append(
                f"`modality_output`{context} references missing key `{output_key}` in `{output_type}.{bundle_name}`.",
            )

        return errors

    def _validate_custom_input_contract(
        self,
        custom_input_contract: Dict[str, List[str]],
        modality_name: Optional[str],
    ) -> List[str]:
        errors = []
        context = f" for modality `{modality_name}`" if modality_name else ""

        if not isinstance(custom_input_contract, dict):
            return [f"`custom_input_contract`{context} must be a dictionary."]

        for bundle_name, bundle_keys in custom_input_contract.items():
            if not isinstance(bundle_name, str) or not bundle_name:
                errors.append(
                    f"`custom_input_contract`{context} contains an invalid bundle name: {bundle_name!r}."
                )
                continue

            if not isinstance(bundle_keys, list) or not bundle_keys:
                errors.append(
                    f"`custom_input_contract.{bundle_name}`{context} must be a non-empty list of keys.",
                )
                continue

            non_string_keys = sorted(
                repr(key) for key in bundle_keys if not isinstance(key, str) or not key
            )
            if non_string_keys:
                errors.append(
                    f"`custom_input_contract.{bundle_name}`{context} must contain only non-empty strings, "
                    f"got {non_string_keys}.",
                )
                continue

            duplicate_keys = _find_duplicates(bundle_keys)
            if duplicate_keys:
                errors.append(
                    f"`custom_input_contract.{bundle_name}`{context} contains duplicate keys {sorted(duplicate_keys)}.",
                )

        return errors

    def _iter_stages(self) -> List[Tuple[str, List[DataFlowComponent]]]:
        return [
            ("encoders", self.encoders),
            ("adapters", self.adapters),
            ("heads", self.heads),
        ]


def build_data_flow_from_dict(data_flow_dict: Dict[str, Any]) -> DataFlow:
    if not isinstance(data_flow_dict, dict):
        raise DataFlowValidationError(
            f"DataFlow config must be a dictionary, got {type(data_flow_dict).__name__}.",
        )

    data_flow = DataFlow(
        encoders=[
            build_data_flow_component_from_dict(encoder_data_flow)
            for encoder_data_flow in data_flow_dict.get("encoders", [])
        ],
        adapters=[
            build_data_flow_component_from_dict(adapter_data_flow)
            for adapter_data_flow in data_flow_dict.get("adapters", [])
        ],
        heads=[
            build_data_flow_component_from_dict(head_data_flow)
            for head_data_flow in data_flow_dict.get("heads", [])
        ],
    )
    data_flow.validate_schema()
    return data_flow


def build_data_flow_component_from_dict(
    component_data_flow_dict: Dict[str, Any],
) -> DataFlowComponent:
    if not isinstance(component_data_flow_dict, dict):
        raise DataFlowValidationError(
            f"DataFlow component config must be a dictionary, got {type(component_data_flow_dict).__name__}.",
        )
    if "name" not in component_data_flow_dict:
        raise DataFlowValidationError("DataFlow component config must define `name`.")
    if "inputs" not in component_data_flow_dict:
        raise DataFlowValidationError(
            f"DataFlow component `{component_data_flow_dict['name']}` must define `inputs`."
        )
    if "outputs" not in component_data_flow_dict:
        raise DataFlowValidationError(
            f"DataFlow component `{component_data_flow_dict['name']}` must define `outputs`."
        )

    return DataFlowComponent(
        name=component_data_flow_dict["name"],
        inputs=[
            build_data_flow_input_item_from_dict(
                input_item, component_data_flow_dict["name"]
            )
            for input_item in component_data_flow_dict["inputs"]
        ],
        outputs=[
            build_data_flow_output_item_from_dict(
                output_item, component_data_flow_dict["name"]
            )
            for output_item in component_data_flow_dict["outputs"]
        ],
    )


def build_data_flow_input_item_from_dict(
    input_item: Dict[str, Any], component_name: str
) -> DataFlowInputItem:
    if not isinstance(input_item, dict):
        raise DataFlowValidationError(
            f"Input item for component `{component_name}` must be a dictionary, got {type(input_item).__name__}.",
        )

    missing_keys = [key for key in ["type", "name", "keys"] if key not in input_item]
    if missing_keys:
        raise DataFlowValidationError(
            f"Input item for component `{component_name}` is missing required fields: {missing_keys}.",
        )

    return DataFlowInputItem(
        type=input_item["type"], name=input_item["name"], keys=input_item["keys"]
    )


def build_data_flow_output_item_from_dict(
    output_item: Dict[str, Any], component_name: str
) -> DataFlowOutputItem:
    if not isinstance(output_item, dict):
        raise DataFlowValidationError(
            f"Output item for component `{component_name}` must be a dictionary, got {type(output_item).__name__}.",
        )

    missing_keys = [key for key in ["name", "keys"] if key not in output_item]
    if missing_keys:
        raise DataFlowValidationError(
            f"Output item for component `{component_name}` is missing required fields: {missing_keys}.",
        )

    return DataFlowOutputItem(name=output_item["name"], keys=output_item["keys"])
