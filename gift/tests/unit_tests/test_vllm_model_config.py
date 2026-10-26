"""Paged-kernel adaptation preserves a completed model's typed ownership."""

import pytest
import subprocess
import sys

from torchtitan.config import load_raw_document
from torchtitan.config.configurable import config, Configurable, family_of
from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.components.lora import FrozenConfig, unwrap_frozen
from torchtitan.config.schema.experiments.rl.models.attention import (
    VLLMAttentionWrapperConfig,
)
from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)
from torchtitan.config.sections import ModelSection
from torchtitan.config.schema.experiments.rl.models.model_config import (
    with_vllm_attention,
)


@config
class PagedKernelConfig(VLLMAttentionWrapperConfig):
    _owner_spec = "tests.unit_tests.test_vllm_model_config:PagedKernelOwner"


class PagedKernelOwner(Configurable):
    Config = PagedKernelConfig


def _tree(recipe="llama3_debug"):
    raw = load_raw_document(f"configs/experiments/{recipe}.yaml")
    return decode(ModelSection, raw.data["model"]).config.materialize()


def test_vllm_kernel_has_a_typed_runtime_contract_without_a_yaml_type():
    schema = VLLMAttentionWrapperConfig
    assert issubclass(schema, InnerAttentionConfig)
    assert schema not in family_of(InnerAttentionConfig).types.values()
    with pytest.raises(ConfigSchemaError, match="unknown inner_attention type"):
        decode(InnerAttentionConfig, {"type": "vllm"})


def test_pure_vllm_parameters_remain_unbound_without_importing_runtime():
    script = """
import sys
from torchtitan.config.schema.experiments.rl.models.attention import VLLMAttentionWrapperConfig
from torchtitan.config.schema.models.common.attention.softmax.gqa import GQAttentionConfig
schema = VLLMAttentionWrapperConfig
assert '_owner_spec' not in schema.__dict__ and '_owner' not in schema.__dict__
kernel = schema(hidden_size=64, num_heads=4, num_kv_heads=4, head_dim=16)
try:
    GQAttentionConfig(dim=64, n_heads=4, inner_attention=kernel)
except ValueError as error:
    assert 'not bound to a kernel class' in str(error)
else:
    raise AssertionError('unbound kernel accepted')
assert not any(name == 'vllm' or name.startswith('vllm.') for name in sys.modules)
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_direct_runtime_binding_is_a_concrete_kernel_without_an_import_spec():
    from torchtitan.config.schema.models.common.attention.softmax.gqa import (
        GQAttentionConfig,
    )

    @config
    class DirectKernelConfig(VLLMAttentionWrapperConfig):
        pass

    class DirectKernelOwner(Configurable):
        Config = DirectKernelConfig

    kernel = DirectKernelConfig(
        hidden_size=64, num_heads=4, num_kv_heads=4, head_dim=16
    )
    assert kernel.owner() is DirectKernelOwner
    attention = GQAttentionConfig(dim=64, n_heads=4, inner_attention=kernel)
    assert attention.inner_attention is kernel


@pytest.mark.parametrize("wrapper_depth", [0, 1, 2])
def test_adaptation_preserves_source_metadata_and_wrappers(wrapper_depth):
    original = _tree()
    attention = original.layers[0].attention
    attention._set_metadata(
        "source_fields", {"n_heads": "model.config.blocks.dense.attention.n_heads"}
    )
    for _ in range(wrapper_depth):
        original.layers[0].attention = FrozenConfig(inner=original.layers[0].attention)
    before = original.layers[0].attention
    adapted = with_vllm_attention(original, PagedKernelConfig)
    selected = adapted.layers[0].attention
    remaining = selected
    for _ in range(wrapper_depth):
        assert isinstance(remaining, FrozenConfig)
        remaining = remaining.inner
    assert not isinstance(remaining, FrozenConfig)
    inner = unwrap_frozen(selected)
    assert isinstance(inner.inner_attention, VLLMAttentionWrapperConfig)
    assert inner._source_fields == attention._source_fields
    assert selected is not before
    assert not isinstance(attention.inner_attention, VLLMAttentionWrapperConfig)
    assert inner.attention_mask_kind is AttentionMaskKind.NONE
    assert inner.mask_type is MaskType.CAUSAL
    assert inner.inner_attention.hidden_size == original.dim
    assert len(
        {id(layer.attention.inner_attention) for layer in adapted.layers}
    ) == len(adapted.layers)
    adapted.validate_materialized()


def test_training_varlen_mask_does_not_leak_into_paged_backend():
    from torchtitan.config.schema.models.common.attention.softmax.kernels.varlen import (
        VarlenAttentionConfig,
    )

    tree = _tree()
    for layer in tree.layers:
        layer.attention = layer.attention._copy(
            inner_attention=VarlenAttentionConfig(), mask_type=MaskType.BLOCK_CAUSAL
        )
    adapted = with_vllm_attention(tree, PagedKernelConfig)
    assert all(layer.attention.mask_type is MaskType.CAUSAL for layer in adapted.layers)
    assert all(
        layer.attention.mask_type is MaskType.BLOCK_CAUSAL for layer in tree.layers
    )


def test_backend_geometry_is_derived_for_each_layer():
    tree = _tree()
    second = tree.layers[1].attention
    tree.layers[1].attention = second._copy(n_kv_heads=8, qkv_linear=None)
    adapted = with_vllm_attention(tree, PagedKernelConfig)
    assert adapted.layers[0].attention.inner_attention.num_kv_heads == (
        tree.layers[0].attention.n_kv_heads or tree.layers[0].attention.n_heads
    )
    assert adapted.layers[1].attention.inner_attention.num_kv_heads == 8


def test_non_gqa_attention_is_rejected_at_the_inference_boundary():
    with pytest.raises(ValueError, match="vLLM requires GQA"):
        with_vllm_attention(_tree("gift_moe_debug"), PagedKernelConfig)
