"""Model YAML materializes internal trees without a second YAML format."""

import dataclasses
import functools
import typing

import pytest

from tests.unit_tests.config_helpers import PRESET_ROOT
from torchtitan.config import deep_merge, load_raw_document
from torchtitan.config.configurable import DERIVED, fields_with_role, OWNER_FILLED
from torchtitan.config.decode import decode
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.attention.linear.kda import (
    KimiDeltaAttentionConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.module import iter_module_configs, ModuleConfig
from torchtitan.config.sections import ModelSection

GIFT_3_5 = "gift_moe/gift_3_5/debugmodel.yaml"

PRESETS = sorted(
    str(path.relative_to(PRESET_ROOT)) for path in PRESET_ROOT.rglob("*.yaml")
)


@pytest.mark.parametrize("preset", PRESETS)
def test_yaml_materializes_an_internal_independent_tree(preset):
    section = decode(
        ModelSection, load_raw_document(str(PRESET_ROOT / preset)).data["model"]
    )
    actual = section.config.materialize()
    assert actual is not section.config
    assert actual.materialized and not section.config.materialized
    again = actual.materialize()
    assert again.materialized and again is not actual
    with pytest.raises(ValueError, match="runtime model trees are internal"):
        actual.to_dict()
    if hasattr(actual, "layers") and len(actual.layers) > 1:
        assert actual.layers[0] is not actual.layers[1]


@pytest.mark.parametrize(
    "type_name",
    [
        "llama3_tree",
        "llama4_tree",
        "qwen3_tree",
        "qwen3_vl_tree",
        "deepseek_v3_tree",
        "gift_moe_tree",
        "gift_3_5_tree",
        "gift_4_tree",
        "gpt_oss_tree",
        "flux_tree",
        "ft/llama3_tree",
        "graph_trainer/llama3_tree",
        "graph_trainer/qwen3_tree",
        "graph_trainer/deepseek_v3_tree",
    ],
)
def test_runtime_model_types_are_not_public_yaml(type_name):
    from torchtitan.config import ConfigSchemaError

    with pytest.raises(ConfigSchemaError, match="unknown model type"):
        decode(ModelSection, {"config": {"type": type_name}})


def test_materialization_unshares_sibling_norms_before_per_fqn_changes():
    section = decode(
        ModelSection,
        load_raw_document(str(PRESET_ROOT / "llama3/debugmodel.yaml")).data["model"],
    )
    section.config.hidden_z_loss_patterns = {"layers.0.attention_norm": 0.2}
    actual = section.config.materialize()
    block = actual.layers[0]
    assert block.attention_norm is not block.ffn_norm
    from torchtitan.config.schema.models.common.regularization import HiddenZLoss

    assert block.attention_norm.hidden_z_loss == HiddenZLoss(coef=0.2)
    assert block.ffn_norm.hidden_z_loss is None
    from torchtitan.config.schema.protocols.module import Zeros

    assert isinstance(block.attention_norm.param_init, dict)
    assert isinstance(block.ffn_norm.param_init, dict)
    block.attention_norm.param_init["weight"] = Zeros()
    assert block.ffn_norm.param_init["weight"] != Zeros()


def test_independent_positions_preserve_frozen_transparency():
    from torchtitan.config.configurable import independent_config_positions
    from torchtitan.config.schema.components.lora import FrozenConfig

    section = decode(
        ModelSection,
        load_raw_document(str(PRESET_ROOT / "llama3/debugmodel.yaml")).data["model"],
    )
    actual = section.config.materialize()
    shared = FrozenConfig(inner=actual.layers[0])
    actual.layers[0] = actual.layers[1] = shared
    copied = independent_config_positions(actual)
    assert copied.layers[0] is not copied.layers[1]
    assert copied.layers[0].inner is not copied.layers[1].inner
    assert copied.layers[0].inner.attention_norm is not copied.layers[0].inner.ffn_norm


def test_independent_positions_separate_shared_list_containers():
    from torchtitan.config.configurable import (
        Configurable,
        config,
        independent_config_positions,
    )
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig

    @config
    class TwoLists(Configurable.Config):
        left: list[LinearConfig]
        right: list[LinearConfig]

    shared = [LinearConfig(in_features=2, out_features=2)]
    original = TwoLists(left=shared, right=shared)
    copied = independent_config_positions(original)
    assert copied.left is not copied.right
    assert copied.left[0] is not copied.right[0]
    copied.left[0].in_features = 3
    assert copied.right[0].in_features == original.left[0].in_features == 2


def test_unregistered_converted_linear_schema_does_not_emit_broken_yaml():
    from torchtitan.config.configurable import config
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig

    @config
    class ConvertedLinearConfig(LinearConfig):
        rank: int = 4

    converted = ConvertedLinearConfig(in_features=2, out_features=2)
    with pytest.raises(
        ValueError, match="no registered YAML type.*compact.*converters"
    ):
        converted.to_dict()


@pytest.mark.parametrize(
    "experiment",
    ["llama3_debug.yaml", "gift_moe_debug_frozen.yaml", "gift_4_debug.yaml"],
)
def test_trainer_diagnostic_dump_keeps_internal_runtime_model(experiment):
    import json
    from tests.unit_tests.config_helpers import load_experiment_config
    from torchtitan.config.configurable import type_name_of

    trainer = load_experiment_config(experiment)
    document = trainer.to_dict()
    model = document["model_spec"]["model"]
    # One schema per model: the materialized tree dumps under its registered type.
    assert model["type"] == type_name_of(type(trainer.model_spec.model))
    assert model["layers"]
    assert "_source_path" not in json.dumps(document)
    assert json.loads(json.dumps(document))["model_spec"]["model"]["layers"]


def test_diagnostic_dump_handles_unregistered_converted_projection_recursively():
    import json
    from torchtitan.config.configurable import config
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig
    from tests.unit_tests.config_helpers import load_experiment_config

    @config
    class ConvertedLinearConfig(LinearConfig):
        recipe: str = "rowwise"

    trainer = load_experiment_config("llama3_debug.yaml")
    projection = trainer.model_spec.model.layers[0].feed_forward.w1
    trainer.model_spec.model.layers[0].feed_forward.w1 = ConvertedLinearConfig(
        in_features=projection.in_features,
        out_features=projection.out_features,
        param_init=projection.param_init,
    )
    dumped = trainer.to_dict()["model_spec"]["model"]["layers"][0]["feed_forward"]["w1"]
    assert dumped["recipe"] == "rowwise"
    assert dumped["__class__"].endswith("ConvertedLinearConfig")
    json.dumps(dumped)
    with pytest.raises(ValueError, match="no registered YAML type"):
        trainer.model_spec.model.layers[0].feed_forward.w1.to_dict()


@functools.cache
def _module_fields(cls: type) -> tuple[str, ...]:
    hints = typing.get_type_hints(cls)
    return tuple(
        f.name
        for f in dataclasses.fields(cls)
        if any(
            isinstance(arg, type) and issubclass(arg, ModuleConfig)
            for arg in typing.get_args(hints[f.name]) or (hints[f.name],)
        )
    )


def _unset_module_fields(tree) -> list[str]:
    """Module-typed fields left None that are required, owner-filled or derived."""
    unset = []
    for fqn, module in [("", tree), *iter_module_configs(tree)]:
        cls = type(module)
        filled = fields_with_role(cls, OWNER_FILLED) | fields_with_role(cls, DERIVED)
        defaults = {f.name: f.default for f in dataclasses.fields(cls)}
        for name in _module_fields(cls):
            if getattr(module, name) is None and (
                name in filled or defaults[name] is not None
            ):
                unset.append(f"{fqn}.{name}" if fqn else name)
    return unset


@pytest.mark.parametrize("preset", PRESETS)
def test_materialized_module_fields_are_set_unless_optional(preset):
    section = decode(
        ModelSection, load_raw_document(str(PRESET_ROOT / preset)).data["model"]
    )
    assert _unset_module_fields(section.config.materialize()) == []


def test_kda_projections_are_derived_and_filled():
    kda = {"attention": {"type": "kda", "conv_size": 4}}
    model = deep_merge(
        load_raw_document(str(PRESET_ROOT / GIFT_3_5)).data["model"],
        {"config": {"blocks": {"gdn_dense": kda, "gdn_moe": kda}}},
    )
    tree = decode(ModelSection, model).config.materialize()
    assert _unset_module_fields(tree) == []
    projections = {"b_proj", "f_proj_in", "f_proj_out", "g_proj_in", "g_proj_out"}
    assert projections <= fields_with_role(KimiDeltaAttentionConfig, DERIVED)
    kda_layers = [
        layer.attention
        for layer in tree.layers
        if isinstance(layer.attention, KimiDeltaAttentionConfig)
    ]
    assert kda_layers
    for name in projections:
        setattr(kda_layers[0], name, None)
        with pytest.raises(ValueError, match=f"{name}: still unset"):
            tree.validate_materialized()
        setattr(kda_layers[0], name, getattr(kda_layers[1], name))


@dataclasses.dataclass
class _Holder:
    named: dict


def test_frozen_module_configs_in_dicts_are_walked():
    inner = LinearConfig(in_features=2, out_features=3)
    holder = _Holder(named={"proj": FrozenConfig(inner=inner), "coef": 1.0})
    assert list(iter_module_configs(holder)) == [("named.proj", inner)]
