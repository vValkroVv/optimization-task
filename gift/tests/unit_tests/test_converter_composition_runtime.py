"""Converter composition fails before destructive tree replacement."""

import pytest

from torchtitan.components.lora import LoRAConverter
from torchtitan.components.quantization.utils import apply_float8
from torchtitan.config.build import build_model_spec
from torchtitan.config.cli import load_document
from torchtitan.config.configs import FP8Config
from torchtitan.config.decode import decode
from torchtitan.config.schema.components.lora import LoRAConverterConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.sections import ModelSection
from torchtitan.ops.float8.cuda_gemm import FP8Linear


def _tree(converters):
    document, _ = load_document(
        "configs/experiments/llama3_debug.yaml",
        [("model.config.converters", converters)],
    )
    return build_model_spec(decode(ModelSection, document.data["model"])).model


def _positions(tree):
    return [(fqn, cfg) for fqn, cfg, _, _ in tree.traverse(LinearConfig)]


def test_dense_fp8_rejects_overlap_before_replacing_any_projection():
    tree = _tree("[{type: lora, target_modules: [wo]}]")
    before = _positions(tree)
    with pytest.raises(ValueError, match="dense FP8 cannot replace specialized linear"):
        apply_float8(
            tree, FP8Config(dense_include_linear_layers=".*"), tensor_parallel_degree=1
        )
    after = _positions(tree)
    assert [name for name, _ in after] == [name for name, _ in before]
    assert all(left is right for (_, left), (_, right) in zip(before, after))


def test_disjoint_fp8_preserves_lora_and_converts_frozen_base_projection():
    tree = _tree("[{type: lora, target_modules: [wo]}]")
    adapters = {name: cfg for name, cfg in _positions(tree) if name.endswith(".wo")}
    apply_float8(
        tree, FP8Config(dense_include_linear_layers=".*w1$"), tensor_parallel_degree=1
    )
    after = dict(_positions(tree))
    assert all(after[name] is cfg for name, cfg in adapters.items())
    assert all(
        isinstance(cfg, FP8Linear.Config)
        for name, cfg in after.items()
        if name.endswith(".w1")
    )
    assert adapters


def test_direct_repeated_lora_conversion_fails_atomically():
    tree = _tree("[{type: lora}]")
    before = _positions(tree)
    with pytest.raises(ValueError, match="LoRA was already applied"):
        LoRAConverter(LoRAConverterConfig()).convert(tree)
    assert all(left is right for (_, left), (_, right) in zip(before, _positions(tree)))


@pytest.mark.parametrize(
    "converters",
    [
        "[{type: freeze, selectors: [layers.0]}, {type: lora}]",
        "[{type: lora}, {type: freeze, selectors: [layers.0]}]",
    ],
)
def test_freeze_lora_order_preserves_adapter_configurations(converters):
    from torchtitan.config.schema.components.lora import FrozenConfig

    tree = _tree(converters)
    assert isinstance(tree.layers[0], FrozenConfig)
    assert any(
        getattr(type(cfg), "_lora_applied", False)
        for _, cfg in _positions(tree.layers[0])
    )
