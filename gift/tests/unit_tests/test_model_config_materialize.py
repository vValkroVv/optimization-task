"""The model Config's two forms: the compact one ``model.config`` decodes into
(what YAML and ``resolved_config.yaml`` hold) and the tree ``materialize()``
derives from it (what ``ModelSpec.model`` holds and the runtime builds)."""

import copy
import dataclasses

import pytest

from tests.unit_tests.config_helpers import (
    EXPERIMENT_ROOT,
    load_experiment_config,
    PRESET_ROOT,
)
from torchtitan.config import (
    ConfigSchemaError,
    deep_merge,
    load_raw_document,
    run_pre_submit,
)
from torchtitan.config.build import materialize_model_config
from torchtitan.config.configurable import config
from torchtitan.config.decode import decode
from torchtitan.config.schema.models.common.attention.const import MaskType
from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
    FlexAttentionConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.decoder import (
    DecoderConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
    LayerGroupConfig,
)
from torchtitan.config.schema.models.common.moe.moe import MoEConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.model import ModelConfigConverterConfig
from torchtitan.config.schema.protocols.module import iter_module_configs
from torchtitan.config.sections import ModelSection
from torchtitan.config.validation import (
    Issue,
    validate_config_tree,
    ValidationContext,
    validate_typed_config,
)

GIFT_3_5 = "gift_moe/gift_3_5/debugmodel.yaml"


def _compact(preset: str, overrides: dict | None = None):
    """The preset's decoded ``model.config``, *overrides* layered on it."""
    model = deep_merge(
        load_raw_document(str(PRESET_ROOT / preset)).data["model"],
        {"config": overrides or {}},
    )
    return decode(ModelSection, model, path="model").config


def _nested(path: str, value) -> dict:
    """``{"a": {"b": value}}`` for *path* ``a.b``."""
    for key in reversed(path.split(".")):
        value = {key: value}
    return value


def test_materialize_leaves_the_compact_form_untouched():
    compact = _compact(GIFT_3_5)
    before = copy.deepcopy(compact)
    tree = compact.materialize()
    assert compact == before
    assert not compact.materialized
    assert all(isinstance(entry, LayerGroupConfig) for entry in compact.layers)
    assert tree.materialized and len(tree.layers) == compact.n_layers
    # A materialized tree is its own materialization.
    assert tree.materialize() == tree


def test_templates_are_not_part_of_the_tree():
    """The named ``blocks`` are not modules of the built model."""
    compact = _compact(GIFT_3_5)
    tree = compact.materialize()
    assert {fqn.split(".")[0] for fqn, _ in iter_module_configs(tree)} == {
        "layers",
        "lm_head",
        "norm",
        "rope",
        "tok_embeddings",
    }
    assert all(fqn.startswith("layers.") for fqn, *_ in tree.traverse(MoEConfig))
    # Before materialize the named blocks are the modules walked.
    fqns = {fqn for fqn, _ in iter_module_configs(compact)}
    assert any(fqn.startswith("blocks.") for fqn in fqns)
    assert not any(fqn.startswith("layers") for fqn in fqns)


def test_hybrid_layout_places_the_named_blocks_and_mtp_heads():
    """gift_3_5 debugmodel: GDN/MLA blocks placed by ``layers``, one dense
    layer; its MTP head is placed from the dense ``mtp`` block."""
    tree = _compact(GIFT_3_5, {"mtp": {"layers": [{"block": "mtp"}]}}).materialize()
    assert [type(layer.attention).__name__ for layer in tree.layers] == [
        "GatedDeltaNetConfig",
        "GatedDeltaNetConfig",
        "MLAConfig",
        "GatedDeltaNetConfig",
        "GatedDeltaNetConfig",
        "MLAConfig",
    ]
    assert [layer.moe is None for layer in tree.layers] == [True] + [False] * 5
    head = tree.mtp.layers[0].block
    assert head.feed_forward is not None and head.moe is None


def test_varlen_sets_the_linear_layers_mask_type():
    config = load_experiment_config("gift_3_5_debug_varlen.yaml")
    assert config.model_spec is not None
    tree = config.model_spec.model
    assert isinstance(tree, DecoderConfig)
    linear = [
        layer.attention
        for layer in tree.layers
        if type(layer.attention).__name__ == "GatedDeltaNetConfig"
    ]
    assert linear and {a.mask_type for a in linear} == {MaskType.BLOCK_CAUSAL}


def test_linear_attention_is_picked_by_type():
    """``attention`` is a component family: a hybrid's linear layers switch to
    KDA from YAML, the full-attention layers keep their kernel."""
    kda = {"attention": {"type": "kda", "conv_size": 4}}
    tree = _compact(
        GIFT_3_5, {"blocks": {"gdn_dense": kda, "gdn_moe": kda}}
    ).materialize()
    assert [type(layer.attention).__name__ for layer in tree.layers] == [
        "KimiDeltaAttentionConfig",
        "KimiDeltaAttentionConfig",
        "MLAConfig",
        "KimiDeltaAttentionConfig",
        "KimiDeltaAttentionConfig",
        "MLAConfig",
    ]


def test_hybrid_full_attention_kernel_must_match_the_linear_layers_masking():
    """Linear attention masks only causally and the stack shares one mask
    object: a hybrid's full-attention kernel must be NONE-kind or VARLEN-kind;
    flex (BLOCK_MASK) is rejected at decode. Without linear
    layers (``gift_moe``) flex is fine."""
    flex = {"attention": {"inner_attention": {"type": "flex"}}}
    with pytest.raises(ValueError, match="mixes attention masking"):
        _compact(GIFT_3_5, {"blocks": {"mla_moe": flex, "mtp": flex}}).materialize()
    _compact(
        "gift_moe/debugmodel.yaml", {"blocks": {"mla_dense": flex, "mla_moe": flex}}
    ).materialize()


def test_vocab_size_must_be_positive():
    for preset in ("llama3/debugmodel.yaml", GIFT_3_5):
        with pytest.raises(ConfigSchemaError, match="vocab_size must be positive"):
            _compact(preset, {"vocab_size": 0})


def test_layers_must_not_be_empty():
    with pytest.raises(ConfigSchemaError, match="at least one entry in `layers`"):
        _compact("llama3/debugmodel.yaml", {"layers": []})
    with pytest.raises(ConfigSchemaError, match="`repeat` must be at least 1"):
        _compact(
            "llama3/debugmodel.yaml", {"layers": [{"block": "dense", "repeat": 0}]}
        )


def test_mla_takes_the_yarn_factor_from_the_rope():
    """``MLAConfig.accept_rope`` fills an unset ``yarn_factor`` from the
    decoder's YaRN scaling and rejects an explicit one that disagrees."""
    compact = _compact("gift_moe/3b.yaml")
    factor = compact.rope.scaling.rope_factor
    tree = compact.materialize()
    assert {layer.attention.yarn_factor for layer in tree.layers} == {factor}
    stale_attention = {"attention": {"yarn_factor": 2.0}}
    stale = _compact(
        "gift_moe/3b.yaml",
        {"blocks": {"mla_dense": stale_attention, "mla_moe": stale_attention}},
    )
    with pytest.raises(ValueError, match=f"set yarn_factor={factor}"):
        stale.materialize()


def test_a_restated_projection_keeps_its_owners_init_role():
    """A restated FFN ``w2`` stays ``linear_out`` (depth-scaled init) unless it
    names a role."""
    llama3 = "llama3/debugmodel.yaml"
    built = _compact(llama3).materialize()
    w2 = built.layers[5].feed_forward.w2
    shape = {"in_features": w2.in_features, "out_features": w2.out_features}
    restated = _compact(
        llama3, _nested("blocks.dense.feed_forward.w2", shape)
    ).materialize()
    got = restated.layers[5].feed_forward.w2
    assert (got.init_role, got.param_init) == ("linear_out", w2.param_init)
    named = _compact(
        llama3,
        _nested("blocks.dense.feed_forward.w2", {**shape, "init_role": "linear"}),
    )
    got = named.materialize().layers[5].feed_forward.w2
    assert got.init_role == "linear"
    assert got.param_init == built.layers[5].feed_forward.w1.param_init


@pytest.mark.parametrize(
    "path,built",
    [
        ("blocks.sliding_gpt_oss_sink_moe.attention.wo", "attention.wo"),
        (
            "blocks.sliding_gpt_oss_sink_moe.attention.qkv_linear.wq",
            "attention.qkv_linear.wq",
        ),
        ("blocks.sliding_gpt_oss_sink_moe.moe.router.gate", "moe.router.gate"),
    ],
)
def test_gpt_oss_projections_keep_their_bias(path, built):
    """gpt-oss checkpoints carry these biases, so a restated projection's ``bias``
    is checked like its shape."""
    gpt_oss = "gpt_oss/debugmodel.yaml"
    child = _compact(gpt_oss).materialize().layers[0]
    for name in built.split("."):
        child = getattr(child, name)
    assert child.bias
    shape = {"in_features": child.in_features, "out_features": child.out_features}
    with pytest.raises(ValueError, match=r"requires bias: true \(got false\)"):
        _compact(gpt_oss, _nested(path, shape)).materialize()
    restated = _compact(gpt_oss, _nested(path, {**shape, "bias": True}))
    restated.materialize()


def test_a_projection_owns_its_bias_where_the_family_does_not_fix_it():
    qwen3 = "qwen3/debugmodel.yaml"
    layer = _compact(qwen3).materialize().layers[0]
    wq = layer.attention.qkv_linear.wq
    biased = {"in_features": wq.in_features, "out_features": wq.out_features}
    compact = _compact(
        qwen3, _nested("blocks.dense.attention.qkv_linear.wq", {**biased, "bias": True})
    )
    tree = compact.materialize()
    assert tree.layers[0].attention.qkv_linear.wq.bias


@pytest.mark.parametrize(
    "preset,block", [("deepseek_v3/debugmodel.yaml", "mla_moe"), (GIFT_3_5, "gdn_moe")]
)
def test_shared_experts_take_the_routed_experts_width(preset, block):
    """The shared experts add to the routed ones' output, so a ``dim`` set on
    them must match; unset, the MoE fills it."""
    tree = _compact(preset).materialize()
    moe = next(layer.moe for layer in tree.layers if layer.moe is not None)
    assert moe.shared_experts.dim == moe.experts.dim == tree.dim
    wrong = _compact(preset, _nested(f"blocks.{block}.moe.shared_experts.dim", 7))
    with pytest.raises(
        ValueError, match=r"shared_experts\.dim must match experts\.dim"
    ):
        wrong.materialize()


@pytest.mark.parametrize(
    "preset", ["llama3/debugmodel.yaml", "deepseek_v3/debugmodel.yaml"]
)
def test_mtp_heads_name_defined_blocks(preset):
    """``mtp.layers`` places named ``blocks``; an undefined name is a config
    error at decode, not a failure in ``materialize()``."""
    with pytest.raises(ConfigSchemaError, match="mtp.layers: unknown block"):
        _compact(preset, {"mtp": {"layers": [{"block": "mtp"}]}})
    tree = _compact(GIFT_3_5, {"mtp": {"layers": [{"block": "mtp"}]}}).materialize()
    assert len(tree.mtp.layers) == 1


def test_materialize_checks_the_stack_it_derives(monkeypatch):
    """``materialize()`` checks the embedding, head and final norm after the
    converters that may swap them."""
    compact = _compact("llama3/debugmodel.yaml")
    head = type(compact)._head
    monkeypatch.setattr(
        type(compact),
        "_head",
        lambda self: dataclasses.replace(head(self), out_features=self.vocab_size + 1),
    )
    with pytest.raises(ValueError, match="lm_head"):
        compact.materialize()


def test_layout_errors_are_config_errors_at_model_config():
    """Errors found in ``materialize()`` are reported at ``model.config``, not as
    a bare ``ValueError``."""
    w2 = {"in_features": 1, "out_features": 2}
    compact = _compact(
        "llama3/debugmodel.yaml", _nested("blocks.dense.feed_forward.w2", w2)
    )
    with pytest.raises(ConfigSchemaError, match=r"^model\.config: .*w2 mismatches"):
        materialize_model_config(compact)
    result = run_pre_submit(
        str(EXPERIMENT_ROOT / "llama3_debug.yaml"),
        [
            (
                "model.config.blocks.dense.feed_forward.w2",
                "{in_features: 1, out_features: 2}",
            )
        ],
    )
    [issue] = result.report.errors
    assert issue.path == "model.config" and "w2 mismatches" in issue.message


def test_converters_run_only_when_asked():
    from torchtitan.config.build import build_model_spec

    compact = _compact(GIFT_3_5, {"converters": [{"type": "lora", "rank": 4}]})

    def lora_linears(tree) -> int:
        return sum(
            type(cfg).__qualname__.endswith("LoRALinear.Config")
            for _, cfg in iter_module_configs(tree)
        )

    section = ModelSection(label="converter-check", config=compact)
    assert lora_linears(build_model_spec(section).model) > 0
    assert lora_linears(compact.materialize()) == 0
    assert lora_linears(build_model_spec(section, apply_converters=False).model) == 0


@pytest.mark.parametrize(
    "experiment,name,flavor",
    [
        ("llama3_debug.yaml", "llama3", "debugmodel"),
        ("qwen3_moe_debug.yaml", "qwen3", "qwen3_moe/debugmodel_moe"),
        ("gift_moe_debug.yaml", "gift_moe", "debugmodel"),
        ("gift_3_5_debug.yaml", "gift_moe", "gift_3_5/debugmodel"),
        ("gift_4_700b.yaml", "gift_moe", "gift_4/700b"),
    ],
)
def test_model_spec_holds_the_tree_named_by_family(experiment, name, flavor):
    """``ModelSpec.name`` is the Config's family; the flavor is the label."""
    spec = load_experiment_config(experiment).model_spec
    assert spec is not None
    assert isinstance(spec.model, DecoderConfig)
    assert spec.model.materialized
    assert (spec.name, spec.flavor) == (name, flavor)


def test_the_compact_form_is_rejected_where_the_tree_is_expected():
    """Edits to the compact form would be dropped when it is materialized, so
    the consumers of the tree refuse it instead of working on templates."""
    config = load_experiment_config("gift_3_5_debug.yaml")
    compact = _compact(GIFT_3_5)
    with pytest.raises(ValueError, match="materialized"):
        compact.owner().prepare_config(compact, config=config)
    with pytest.raises(ValueError, match="materialized"):
        validate_typed_config(config, compact, ValidationContext(stage="pre_submit"))


def test_validation_walks_the_tree_not_the_templates(monkeypatch):
    """A materialized model's issues come from its layers, once per layer:
    the template a layer was copied from is not a module of the model."""
    monkeypatch.setattr(
        FlexAttentionConfig,
        "validate",
        lambda self, ctx: [Issue(path="", message="checked")],
    )
    tree = _compact("gpt_oss/debugmodel.yaml").materialize()
    report = validate_config_tree(
        tree, ValidationContext(stage="in_container"), path="model"
    )
    assert [issue.path for issue in report.issues] == [
        f"model.layers[{i}].attention.inner_attention" for i in range(tree.n_layers)
    ]


@config
class _TemplateConverterConfig(ModelConfigConverterConfig):
    template: LinearConfig = dataclasses.field(default_factory=LinearConfig)


def test_converter_templates_are_not_modules_of_the_tree():
    """A converter's replacement template keeps its owner-filled widths unset."""
    tree = _compact(GIFT_3_5).materialize()
    tree.converters.append(_TemplateConverterConfig())
    tree.validate_materialized()
    assert not any(fqn.startswith("converters") for fqn, _ in iter_module_configs(tree))
    assert not any(
        fqn.startswith("converters") for fqn, *_ in tree.traverse(LinearConfig)
    )


def test_owner_filled_fields_are_labeled_in_valid_keys():
    with pytest.raises(ConfigSchemaError) as raised:
        _compact(GIFT_3_5, _nested("blocks.gdn_moe.feed_forwad", {}))
    valid = str(raised.value).split("Valid keys", 1)[1]
    assert "dim (filled by owner)" in valid
    assert "layer_id (filled by owner)" in valid


def test_owner_filled_field_reset_with_null_over_an_import():
    result = run_pre_submit(
        "configs/experiments/gift_3_5_debug.yaml",
        [("model.config.blocks.gdn_moe.moe.shared_experts.dim", "null")],
    )
    assert result.ok, result.report


def test_explicit_mtp_head_norm_takes_the_model_width():
    mtp_block = load_raw_document(str(PRESET_ROOT / GIFT_3_5)).data["model"]["config"][
        "blocks"
    ]["mtp"]
    head = {"block": mtp_block, "token_norm": {"type": "gated_rmsnorm", "eps": 1.0e-5}}
    tree = _compact(GIFT_3_5, {"mtp": {"layers": [head]}}).materialize()
    assert tree.mtp.layers[0].token_norm.normalized_shape == tree.dim
    head["token_norm"]["normalized_shape"] = tree.dim + 1
    with pytest.raises(ValueError, match="token_norm mismatches: normalized_shape"):
        _compact(GIFT_3_5, {"mtp": {"layers": [head]}}).materialize()


@pytest.mark.parametrize(
    ("edit", "path"),
    [
        (
            lambda t: setattr(t.layers[1].attention, "mask_type", None),
            "layers.1.attention.mask_type",
        ),
        (
            lambda t: setattr(t.layers[0].feed_forward, "dim", None),
            "layers.0.feed_forward.dim",
        ),
        (lambda t: setattr(t, "lm_head", None), "lm_head"),
    ],
)
def test_materialized_tree_has_every_owner_filled_and_derived_field(edit, path):
    tree = _compact(GIFT_3_5).materialize()
    edit(tree)
    with pytest.raises(ValueError, match=f"{path}: still unset after materialize"):
        tree.validate_materialized()
