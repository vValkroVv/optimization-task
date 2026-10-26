"""Freezing selects actual modules and preserves their runtime namespace."""

import pytest
import torch
from torch import nn

from torchtitan.components.freeze import FreezeConverter
from torchtitan.components.lora import build_frozen
from torchtitan.components.lora import LoRAConverter
from torchtitan.config.schema.components.lora import LoRAConverterConfig
from torchtitan.config.configurable import Configurable, config
from torchtitan.config.build import build_model_spec, materialize_model_config
from torchtitan.config.cli import load_document
from torchtitan.config.decode import decode
from torchtitan.config.sections import ModelSection
from torchtitan.config.schema.components.freeze import FreezeConverterConfig
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.module import ModuleConfig, iter_module_configs


@config
class StackConfig(Configurable.Config):
    layers: list[LinearConfig | FrozenConfig]


@config
class BlockConfig(ModuleConfig):
    projection: LinearConfig | FrozenConfig


@config
class NestedStackConfig(Configurable.Config):
    layers: list[BlockConfig | FrozenConfig]


def stack():
    return StackConfig(
        layers=[LinearConfig(in_features=4, out_features=4) for _ in range(3)]
    )


def test_freeze_selectors_wrap_only_selected_layers_and_preserve_paths():
    tree = stack()
    FreezeConverter(FreezeConverterConfig(selectors=["layers.0", "layers.1"])).convert(
        tree
    )
    assert isinstance(tree.layers[0], FrozenConfig)
    assert isinstance(tree.layers[1], FrozenConfig)
    assert isinstance(tree.layers[2], LinearConfig)
    assert [path for path, _ in iter_module_configs(tree)] == [
        "layers.0",
        "layers.1",
        "layers.2",
    ]
    assert [path for path, *_ in tree.traverse(ModuleConfig)] == [
        "layers.0",
        "layers.1",
        "layers.2",
    ]


def test_freeze_missing_selector_is_atomic():
    tree = stack()
    with pytest.raises(ValueError, match="matches no module FQN"):
        FreezeConverter(
            FreezeConverterConfig(selectors=["layers.0", "missing.*"])
        ).convert(tree)
    assert all(isinstance(layer, LinearConfig) for layer in tree.layers)


def test_freeze_globs_are_idempotent():
    tree = stack()
    converter = FreezeConverter(FreezeConverterConfig(selectors=["layers.*"]))
    converter.convert(tree)
    converter.convert(tree)
    assert all(
        isinstance(layer, FrozenConfig) and isinstance(layer.inner, LinearConfig)
        for layer in tree.layers
    )


def test_overlapping_selectors_wrap_parent_once():
    tree = NestedStackConfig(
        layers=[BlockConfig(projection=LinearConfig(in_features=4, out_features=4))]
    )
    FreezeConverter(
        FreezeConverterConfig(selectors=["layers.0.projection", "layers.0", "layers.*"])
    ).convert(tree)
    assert isinstance(tree.layers[0], FrozenConfig)
    assert isinstance(tree.layers[0].inner, BlockConfig)
    assert isinstance(tree.layers[0].inner.projection, LinearConfig)
    assert [path for path, _ in iter_module_configs(tree)] == [
        "layers.0",
        "layers.0.projection",
    ]


@pytest.mark.parametrize("freeze_first", [True, False])
def test_freeze_and_lora_preserve_selected_module_freezing(freeze_first):
    tree = stack()
    freezing = FreezeConverter(FreezeConverterConfig(selectors=["layers.0"]))
    lora = LoRAConverter(LoRAConverterConfig(rank=2))
    for converter in [freezing, lora] if freeze_first else [lora, freezing]:
        converter.convert(tree)
    with torch.device("meta"):
        frozen = tree.layers[0].build()
        trainable = tree.layers[1].build()
    assert all(not param.requires_grad for param in frozen.parameters())
    assert any(
        param.requires_grad
        for name, param in trainable.named_parameters()
        if "lora_" in name
    )


@pytest.mark.parametrize("selectors", [[], [""], [" "]])
def test_freeze_requires_explicit_nonempty_selectors(selectors):
    with pytest.raises(ValueError, match="nonempty"):
        FreezeConverterConfig(selectors=selectors)


def test_frozen_factory_preserves_parameters_and_freezes_gradients():
    class ExistingConfig(Configurable.Config):
        def build(self, **kwargs):
            return nn.Linear(4, 4, device="meta")

    module = build_frozen(config=FrozenConfig(inner=ExistingConfig()))
    assert set(dict(module.named_parameters())) == {"weight", "bias"}
    assert all(not parameter.requires_grad for parameter in module.parameters())
    assert all(
        parameter.device == torch.device("meta") for parameter in module.parameters()
    )


def test_frozen_debug_recipe_preserves_runtime_decoder_helpers():
    document, _ = load_document("configs/experiments/gift_moe_debug_frozen.yaml")
    compact = decode(ModelSection, document.data["model"]).config
    tree = materialize_model_config(compact)
    original_paths = [path for path, _ in iter_module_configs(tree)]
    FreezeConverter(compact.converters[0]).convert(tree)
    assert isinstance(tree.layers[0], FrozenConfig)
    assert isinstance(tree.layers[1], FrozenConfig)
    assert [path for path, _ in iter_module_configs(tree)] == original_paths
    tree._fill_rope_contract()
    assert tree.attention is not None


def test_real_decoder_freezing_preserves_checkpoint_names_and_other_layers():
    document, _ = load_document(
        "configs/experiments/llama3_debug.yaml",
        [("model.config.layers", "[{block: dense, repeat: 3}]")],
    )
    section = decode(ModelSection, document.data["model"])
    baseline = build_model_spec(section)
    section.config.converters = [
        FreezeConverterConfig(selectors=["layers.0", "layers.1"])
    ]
    frozen = build_model_spec(section)
    with torch.device("meta"):
        baseline_model = baseline.model.build()
        frozen_model = frozen.model.build()
    assert set(baseline_model.state_dict()) == set(frozen_model.state_dict())
    selected = [
        parameter
        for name, parameter in frozen_model.named_parameters()
        if name.startswith(("layers.0.", "layers.1."))
    ]
    remaining = [
        parameter
        for name, parameter in frozen_model.named_parameters()
        if name.startswith("layers.2.")
    ]
    assert selected and all(not parameter.requires_grad for parameter in selected)
    assert remaining and all(parameter.requires_grad for parameter in remaining)
