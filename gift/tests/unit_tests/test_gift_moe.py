import pytest
import torch

from tests.unit_tests.config_helpers import load_experiment_config, load_model_spec
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.models.common.attention.linear.gated_delta import (
    GatedDeltaAttentionConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from torchtitan.config.schema.models.common.moe.experts.gmm import GroupedExpertsConfig
from torchtitan.config.schema.models.common.norms.gated_norm import GatedRMSNormConfig
from torchtitan.config.schema.protocols.module import Ones, Zeros
from torchtitan.models.common.regularization import HiddenZLoss, set_hidden_z_loss
from torchtitan.config.schema.protocols.module import iter_module_configs

BASE = "gift_moe/debugmodel.yaml"
HYBRID = "gift_moe/gift_3_5/debugmodel.yaml"
GIFT_4 = "gift_moe/gift_4/debugmodel.yaml"

# Block names the presets place (configs/presets/models/gift_moe/**).
BASE_BLOCKS = ("mla_dense", "mla_moe")
HYBRID_GDN_BLOCKS = ("gdn_dense", "gdn_moe")
HYBRID_MOE_BLOCKS = ("gdn_moe", "mla_moe")
ONE_MTP = {"mtp": {"layers": [{"block": "mtp"}]}}


def _blocks(names, block):
    """The same *block* override on every named block."""
    return {"blocks": {name: block for name in names}}


VARLEN = {"attention": {"inner_attention": {"type": "varlen"}}}


def test_base_type_resolves():
    spec = load_model_spec(BASE)
    assert spec.name == "gift_moe"
    assert spec.flavor == "debugmodel"


def test_base_varlen_resolves():
    # Base (pure MLA) accepts varlen with a VARLEN-kind backend.
    spec = load_model_spec(BASE, {"config": _blocks(BASE_BLOCKS, VARLEN)})
    assert spec.flavor == "debugmodel"
    assert spec.model.document_packing


def test_base_varlen_backend_mismatch_raises():
    # A varlen kernel on some layers and the default sdpa on others is rejected
    # by the model config itself: one masks object serves every layer.
    with pytest.raises(ConfigSchemaError, match=r"model\.config: .*mixes attention"):
        load_model_spec(BASE, {"config": _blocks(("mla_moe",), VARLEN)})


def test_hybridep_rejects_varlen():
    # HybridEP dispatch has never been run against varlen attention; the
    # decoder config rejects the pair rather than training something unverified.
    with pytest.raises(ConfigSchemaError, match="model.config: .*HybridEP"):
        load_model_spec(
            BASE,
            {
                "config": {
                    "blocks": {
                        "mla_dense": VARLEN,
                        "mla_moe": {
                            **VARLEN,
                            "moe": {
                                "experts": {"token_dispatcher": {"type": "hybridep"}}
                            },
                        },
                    }
                }
            },
        )


def test_hybridep_accepts_non_varlen():
    # The guard is about varlen, not about HybridEP: the ordinary pairing stands.
    spec = load_model_spec(
        BASE,
        {
            "config": _blocks(
                ("mla_moe",),
                {"moe": {"experts": {"token_dispatcher": {"type": "hybridep"}}}},
            )
        },
    )
    assert spec.flavor == "debugmodel"


def test_gift_3_5_resolves():
    # gift_3_5 is the GDN + MLA hybrid; its type prefixes the flavor label.
    spec = load_model_spec(HYBRID)
    assert spec.name == "gift_moe"
    assert spec.flavor == "gift_3_5/debugmodel"


@pytest.mark.parametrize("size", ["debugmodel", "3b", "700b"])
def test_gift_4_resolves(size):
    # gift_4 mirrors every gift_3_5 preset (soft-capped LM head, plain gated norms).
    spec = load_model_spec(f"gift_moe/gift_4/{size}.yaml")
    assert spec.name == "gift_moe"
    assert spec.flavor == f"gift_4/{size}"


def test_gift_4_shares_the_hybrid_impl():
    # gift_4 presets are gift_3_5 presets, so they must not fall back to the
    # pure-MLA state-dict adapter. They share the base parallelize: the
    # sharding configs branch per attention family on their own.
    from torchtitan.models.gift_moe._base.parallelize import parallelize_gift_moe
    from torchtitan.models.gift_moe.gift_3_5 import GiftHybridStateDictAdapter

    spec = load_model_spec(GIFT_4)
    assert spec.parallelize_fn is parallelize_gift_moe
    assert spec.state_dict_adapter is GiftHybridStateDictAdapter


def test_gift_4_softcaps_lm_head():
    from torchtitan.models.common import Linear, Unembedding

    head = load_model_spec(GIFT_4).model.lm_head
    assert isinstance(head, Unembedding.Config)
    # 30.0 follows the Gemma 2 report; the preset carries the value.
    assert head.logit_softcap == 30.0

    # gift_3_5 keeps a plain, uncapped head.
    head_3_5 = load_model_spec(HYBRID).model.lm_head
    assert type(head_3_5) is Linear.Config
    assert not hasattr(head_3_5, "logit_softcap")
    assert (head.in_features, head.out_features) == (
        head_3_5.in_features,
        head_3_5.out_features,
    )


def test_gdn_o_norm_is_the_blocks_own_setting():
    """``blocks.<gdn>.attention.o_norm`` is the component's config: a preset sets
    its ``eps`` without a model field for it, the stack fills only the width
    and the init."""
    cfg = load_model_spec(
        HYBRID,
        {
            "config": _blocks(
                HYBRID_GDN_BLOCKS, {"attention": {"o_norm": {"eps": 1e-5}}}
            )
        },
    ).model
    gdns = [
        m
        for _, m in iter_module_configs(cfg)
        if isinstance(m, GatedDeltaAttentionConfig)
    ]
    assert gdns
    for g in gdns:
        assert g.o_norm.eps == 1e-5
        # Layered on the gift default, not replacing it.
        assert g.o_norm.zero_centered and g.o_norm.activation == "sigmoid"
        assert g.o_norm.hidden_size == g.head_v_dim
        # Zero-centered: the unit-scale init is a zero weight.
        assert g.o_norm.param_init == {"weight": Zeros()}
        assert g.o_norm_output_multiplier == 2.0


def test_gift_4_differs_from_gift_3_5_only_in_its_recipe():
    from tests.unit_tests.config_golden import strip_class
    from tests.unit_tests.config_parity import structural_dump

    cfg_4 = load_model_spec(GIFT_4, {"config": ONE_MTP}).model
    cfg_3_5 = load_model_spec(HYBRID, {"config": ONE_MTP}).model

    modules = list(iter_module_configs(cfg_4))
    norms = [(fqn, m) for fqn, m in modules if isinstance(m, GatedRMSNormConfig)]
    gdns = [m for _, m in modules if isinstance(m, GatedDeltaAttentionConfig)]
    assert norms and gdns
    assert any(fqn == "mtp.layers.0.block.post_attn_norm" for fqn, _ in norms)
    for fqn, n in norms:
        zero_init = fqn.endswith("post_attn_norm")
        assert not n.zero_centered, fqn
        assert n.param_init["weight"] == (Zeros() if zero_init else Ones()), fqn
    assert all(
        not g.o_norm.zero_centered and g.o_norm.param_init["weight"] == Ones()
        for g in gdns
    )
    # Flip every norm to gift_3_5's parametrization and its unit-scale init.
    for _, n in norms:
        n.zero_centered = True
        n.param_init = {**n.param_init, "weight": Zeros()}
    for g in gdns:
        g.o_norm.zero_centered = True
        g.o_norm.param_init = {**g.o_norm.param_init, "weight": Zeros()}
    assert isinstance(cfg_4, type(cfg_3_5))
    # Up to the norms' parametrization and the soft-capped head, the two
    # materialized trees are the same model.
    dump_4, dump_3_5 = (strip_class(structural_dump(c)) for c in (cfg_4, cfg_3_5))
    for key in (f.name for f in cfg_4.tree_fields()):
        if key == "lm_head":
            continue
        assert dump_4[key] == dump_3_5[key], key


def test_gift_4_post_attn_norms_start_at_zero():
    model = load_model_spec(GIFT_4, {"config": ONE_MTP}).model.build()
    model.init_weights()
    blocks = [
        *model.layers.values(),
        *(head.block for head in model.mtp.layers.values()),
    ]
    x = torch.randn(2, 4, blocks[0].post_attn_norm.weight.numel())
    for block in blocks:
        assert torch.equal(block.post_attn_norm(x), torch.zeros_like(x))
        assert torch.any(block.post_ffn_norm(x) != 0)


def _gift_3_5_moe_and_dense_layers(cfg):
    moe = [layer for layer in cfg.layers if getattr(layer, "moe", None) is not None]
    dense = [
        layer
        for layer in cfg.layers
        if getattr(layer, "feed_forward", None) is not None
    ]
    return moe[0], dense[0]


def test_gift_3_5_z_loss_defaults_off():
    cfg = load_model_spec(HYBRID).model
    moe, dense = _gift_3_5_moe_and_dense_layers(cfg)
    assert moe.moe.router_z_loss_coef == 0.0
    for layer in (moe, dense):
        assert layer.attention.hidden_z_loss is None
        assert (layer.moe or layer.feed_forward).hidden_z_loss is None


def test_gift_3_5_hidden_z_loss_patterns():
    # Patterns pick modules by FQN, so MoE and dense layers can take different
    # coefficients without the builder knowing which layer is which.
    cfg = load_model_spec(
        HYBRID,
        {
            "config": {
                **ONE_MTP,
                "hidden_z_loss_patterns": {
                    "layers.*.attention": 2e-3,
                    "layers.*.moe": 3e-3,
                    "layers.*.feed_forward": 4e-3,
                    "mtp.layers.*.block.attention": 6e-3,
                    "*.attention.kv_latent": 7e-3,
                    "*.attention.q_latent": 8e-3,
                },
            }
        },
    ).model
    # A full HiddenZLoss (not just a coefficient) is set_hidden_z_loss's own API.
    set_hidden_z_loss(cfg, {"norm": HiddenZLoss(coef=5e-3, on="input")})
    moe, dense = _gift_3_5_moe_and_dense_layers(cfg)
    assert moe.attention.hidden_z_loss.coef == 2e-3
    assert moe.moe.hidden_z_loss.coef == 3e-3
    assert dense.attention.hidden_z_loss.coef == 2e-3
    assert dense.feed_forward.hidden_z_loss.coef == 4e-3
    assert cfg.norm.hidden_z_loss == HiddenZLoss(coef=5e-3, on="input")
    # MTP heads are attached in materialize(), which applies the patterns after.
    assert cfg.mtp.layers[0].block.attention.hidden_z_loss.coef == 6e-3
    mla_configs = {
        fqn: child
        for fqn, child in iter_module_configs(cfg)
        if fqn.endswith(".attention") and hasattr(child, "kv_latent_hidden_z_loss_coef")
    }
    assert any(fqn.startswith("layers.") for fqn in mla_configs)
    assert any(fqn.startswith("mtp.layers.") for fqn in mla_configs)
    for child in mla_configs.values():
        assert child.kv_latent_hidden_z_loss_coef == 7e-3
        assert child.q_latent_hidden_z_loss_coef == 8e-3
        assert child.wkv_a.hidden_z_loss is None
        assert child.wkv_b.hidden_z_loss is None


@pytest.mark.parametrize("size", ["debugmodel", "3b", "700b"])
def test_gift_4_projection_z_loss_resolves(size):
    """Presets target MLA latents rather than down/up-projection outputs.

    All flavors accept the targets; full-rank Q skips its latent loss at runtime.
    """
    from torchtitan.config.loader import load_raw_document

    attachment_points = load_raw_document(
        "configs/presets/regularization/gift_4_projection_z_loss.yaml"
    ).data["model"]["config"]["hidden_z_loss_patterns"]
    cfg = load_model_spec(
        f"gift_moe/gift_4/{size}.yaml",
        {"config": {"hidden_z_loss_patterns": dict.fromkeys(attachment_points, 1e-3)}},
    ).model
    marked = {
        fqn.split(".", 2)[2]
        for fqn, c in iter_module_configs(cfg)
        if c.hidden_z_loss is not None
    }
    assert marked == {
        "attention.qkv.qkv_proj",
        "feed_forward.w1",
        "feed_forward.w3",
        "moe.shared_experts.w1",
        "moe.shared_experts.w3",
    }
    attentions = [c for _fqn, c in iter_module_configs(cfg) if isinstance(c, MLAConfig)]
    assert attentions
    for attention in attentions:
        assert attention.q_latent_hidden_z_loss_coef == 1e-3
        assert attention.kv_latent_hidden_z_loss_coef == 1e-3
        for name in ("wq", "wq_a", "wq_b", "wkv_a", "wkv_b"):
            projection = getattr(attention, name)
            assert projection is None or projection.hidden_z_loss is None
    # Routed experts carry theirs as coefs, not as a module hook.
    experts = [
        c for _fqn, c in iter_module_configs(cfg) if isinstance(c, GroupedExpertsConfig)
    ]
    assert experts
    assert all(
        (e.w1_hidden_z_loss_coef, e.w3_hidden_z_loss_coef) == (1e-3, 1e-3)
        for e in experts
    )


def test_gift_4_projection_z_loss_is_inert_at_zero():
    """The presets wire the points but leave the coefficient at 0.0, so nothing
    is hooked until someone picks a value."""
    model = load_experiment_config("gift_4_debug.yaml").model_spec.model.build()
    assert not [
        n for n, m in model.named_modules() if m._forward_hooks or m._forward_pre_hooks
    ]


def test_gift_3_5_router_z_loss_threading():
    cfg = load_model_spec(
        HYBRID,
        {"config": _blocks(HYBRID_MOE_BLOCKS, {"moe": {"router_z_loss_coef": 1e-3}})},
    ).model
    moe, _ = _gift_3_5_moe_and_dense_layers(cfg)
    assert moe.moe.router_z_loss_coef == 1e-3
