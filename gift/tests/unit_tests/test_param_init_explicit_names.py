"""Explicit initializer typos fail without rejecting shared role defaults."""

from dataclasses import replace

import pytest

from tests.unit_tests.config_helpers import init_policy
from torchtitan.components.lora import LoRAConverter
from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.init_policy import apply_init
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.norms.gated_norm import GatedRMSNormConfig
from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig
from torchtitan.config.schema.protocols.module import ModuleConfig, Zeros

DEPTH_SCALED_INIT = init_policy("depth_scaled")
GIFT_HYBRID_INIT = init_policy("gift_hybrid")


@config
class ProjectionConfig(ModuleConfig):
    projection: LinearConfig


@config
class NormContainerConfig(ModuleConfig):
    norm: RMSNormConfig


@config
class GatedNormContainerConfig(ModuleConfig):
    norm: GatedRMSNormConfig


def test_direct_projection_initializer_typo_is_rejected_at_build():
    config = LinearConfig(
        in_features=4, out_features=4, param_init={"weigh_t": Zeros()}
    )
    with pytest.raises(ValueError, match="unknown parameter.*weigh_t"):
        config.build()


def test_partial_module_override_preserves_role_defaults_and_valid_optional_bias():
    projection = LinearConfig(
        in_features=4, out_features=4, bias=False, param_init={"weight": Zeros()}
    )
    parent = ProjectionConfig(projection=projection)
    apply_init(parent, replace(DEPTH_SCALED_INIT), layer_id=0, dim=4)
    assert parent.projection._explicit_param_init_names == frozenset({"weight"})
    assert parent.projection.param_init is not None
    assert "bias" in parent.projection.param_init
    parent.projection.build()


def test_repeated_policy_application_does_not_promote_defaults_to_explicit_names():
    parent = NormContainerConfig(norm=RMSNormConfig(normalized_shape=4))
    policy = replace(DEPTH_SCALED_INIT)
    apply_init(parent, policy, layer_id=0, dim=4)
    apply_init(parent, policy, layer_id=0, dim=4)
    assert parent.norm._explicit_param_init_names == frozenset()
    parent.norm.build()


def test_overlay_does_not_hide_an_explicit_typo():
    parent = NormContainerConfig(
        norm=RMSNormConfig(normalized_shape=4, param_init={"weigh_t": Zeros()})
    )
    apply_init(parent, replace(DEPTH_SCALED_INIT), layer_id=0, dim=4)
    with pytest.raises(ValueError, match="unknown parameter.*weigh_t"):
        parent.norm.build()


def test_lora_conversion_preserves_explicit_names_for_validation():
    parent = ProjectionConfig(
        projection=LinearConfig(
            in_features=4, out_features=4, bias=False, param_init={"weight": Zeros()}
        )
    )
    apply_init(parent, replace(DEPTH_SCALED_INIT), layer_id=0, dim=4)
    parent.projection._source_path = "model.config.ffn.w1"
    converter = LoRAConverter(LoRAConverter.Config(rank=2, alpha=2.0))
    converted = converter._make_lora_config(parent.projection)
    assert converted._explicit_param_init_names == frozenset({"weight"})
    assert converted._source_path == "model.config.ffn.w1"
    converted.build()


def test_dense_fp8_conversion_preserves_initializer_and_source_metadata():
    from torchtitan.components.quantization.utils import apply_float8
    from torchtitan.config.configs import FP8Config
    from torchtitan.config.provenance import SourceMap
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    parent = ProjectionConfig(
        projection=LinearConfig(
            in_features=128, out_features=128, param_init={"weight": Zeros()}
        )
    )
    apply_init(parent, replace(DEPTH_SCALED_INIT), layer_id=0, dim=128)
    original = parent.projection
    original._source_path = "model.config.ffn.w1"
    original._source_fields = SourceMap({"in_features": "model.config.dim"})
    original._source_origins = SourceMap({"model.config.dim": "recipe.yaml"})
    apply_float8(
        parent,
        FP8Config(dense_include_linear_layers="projection"),
        tensor_parallel_degree=1,
    )
    converted = parent.projection
    assert isinstance(converted, FP8Linear.Config)
    assert converted.param_init == original.param_init
    assert converted._explicit_param_init_names == frozenset({"weight"})
    assert converted._source_path == original._source_path
    assert converted._source_fields is original._source_fields
    assert converted._source_origins is original._source_origins
    converted._source_path = "converted"
    assert original._source_path == "model.config.ffn.w1"


def test_partial_gated_norm_override_initializes_all_parameters():
    parent = GatedNormContainerConfig(
        norm=GatedRMSNormConfig(
            normalized_shape=4, gate_rank=2, param_init={"weight": Zeros()}
        )
    )
    apply_init(parent, replace(GIFT_HYBRID_INIT), layer_id=0, dim=4)
    module = parent.norm.build()
    module.init_states()
    assert module.weight.count_nonzero().item() == 0
    assert {name for name, _ in module.named_parameters()} == {
        "weight",
        "gate_up_weight",
        "gate_down_weight",
    }
