"""ProRes: config surface, the depth schedule, and who is inside it.

`tests/test_init_fn.py` covers the `zero_init_post_attention_norm` marker on a
bare norm module. What it cannot cover is which modules the *model* puts the
marker on, and which layers ProRes claims -- and that is where the two ways to
get this wrong live:

* the MTP head is built with ``layer_idx=-1``, which has no place in a
  ``[0, n_layers)`` depth schedule: ``prores_layer_warmup_steps`` rejects it, so
  a ProRes run with ``mtp_block_type="moe"`` used to die inside the callback's
  ``fit_start`` rather than at config time;
* ``output_scale`` threads through five norm classes, and a norm that silently
  ignores it would leave the residual unscaled with no error anywhere.

Real `GigaMixConfig` and a real (tiny) model -- no stubbing -- and no GPU.
"""

import pytest
import torch

from llmfoundry.callbacks.prores_callback import ProResWarmup
from llmfoundry.experimental.prores import (
    prores_alpha,
    prores_alphas,
    prores_layer_warmup_steps,
)
from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM

pytestmark = [pytest.mark.unit, pytest.mark.fast]

N_LAYERS = 4


def _model(**over):
    """Smallest GigaMix that still has post-norms, MoE layers and an MTP head."""
    kw = dict(
        hidden_size=64,
        num_hidden_layers=N_LAYERS,
        vocab_size=128,
        max_position_embeddings=64,
        intermediate_size=32,
        dense_intermediate_size=32,
        num_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        attention_type="LlamaAttention",
        norm_type="GatedNorm",
        layernorm_type="pre_post",
        init_type="deepseek",
        deepseek_init_std=6e-3,
        initializer_range=None,
        first_k_dense_replace=1,
        use_mtp=True,
        mtp_predictor_num=1,
        mtp_loss_weight=0.1,
        mtp_block_type="dense",
        tp_size=1,
        init_device="cpu",
    )
    kw.update(over)
    config = GigaMixConfig(**kw)
    config.sp_split_type = None
    torch.manual_seed(0)
    model = GigaMixForCausalLM(config)
    model.apply(model.param_init_fn)
    return config, model


def _norms(model, suffix):
    return {n: m for n, m in model.named_modules() if n.endswith(suffix)}


class TestZeroInitPostAttentionNorm:

    def test_only_post_attention_norms_are_zeroed(self):
        _, model = _model(zero_init_post_attention_norm=True)
        post_attn = _norms(model, "post_self_attn_layernorm")
        post_ffn = _norms(model, "post_feedforward_layernorm")
        assert post_attn and post_ffn
        for name, mod in post_attn.items():
            assert mod.weight.abs().max().item() == 0.0, name
        for name, mod in post_ffn.items():
            assert mod.weight.abs().max().item() != 0.0, name

    def test_default_leaves_norms_at_unit_scale(self):
        _, model = _model()
        for name, mod in _norms(model, "post_self_attn_layernorm").items():
            assert mod.weight.abs().max().item() != 0.0, name

    def test_rejected_for_pure_pre_norm(self):
        with pytest.raises(ValueError, match="zero_init_post_attention_norm"):
            GigaMixConfig(layernorm_type="pre", zero_init_post_attention_norm=True)


class TestProResSchedule:

    def test_deeper_layers_warm_up_longer(self):
        steps = [
            prores_layer_warmup_steps(warmup_steps=100, layer_idx=i, num_layers=N_LAYERS)
            for i in range(N_LAYERS)
        ]
        assert steps == sorted(steps) and steps[-1] == 100.0

    def test_starts_at_zero_and_saturates_at_one(self):
        assert prores_alphas(step=0, num_layers=N_LAYERS, warmup_steps=8) == [0.0] * N_LAYERS
        assert prores_alphas(step=8, num_layers=N_LAYERS, warmup_steps=8) == [1.0] * N_LAYERS

    def test_shallow_layers_saturate_first(self):
        mid = prores_alphas(step=4, num_layers=N_LAYERS, warmup_steps=8)
        assert mid == sorted(mid, reverse=True)
        assert mid[0] == 1.0 and mid[-1] < 1.0

    def test_mtp_layer_idx_is_rejected_outright(self):
        """The guard that `blocks.py` relies on: -1 is not a schedulable depth."""
        with pytest.raises(ValueError, match=r"layer_idx must be in"):
            prores_alpha(step=0, layer_idx=-1, num_layers=N_LAYERS, warmup_steps=8)


class TestProResModuleWiring:

    def test_disabled_by_default(self):
        _, model = _model()
        roots = [m for m in model.modules() if getattr(m, "_prores_root_module", False)]
        assert roots, "no ProRes root modules at all -- wiring changed"
        assert all(not m._prores_enabled for m in roots)
        assert all(m._prores_output_scale() is None for m in roots)

    def test_trunk_layers_are_claimed_but_the_mtp_head_is_not(self):
        """`mtp_block_type='moe'` builds a `LlamaMixDecoderLayer` with layer_idx=-1."""
        _, model = _model(
            prores_enabled=True, prores_warmup_steps=8, mtp_block_type="moe"
        )
        claimed = {
            n: m
            for n, m in model.named_modules()
            if getattr(m, "_prores_enabled", False) and hasattr(m, "_prores_layer_idx")
        }
        assert len(claimed) == N_LAYERS
        assert all(m._prores_layer_idx >= 0 for m in claimed.values())
        assert not any("mtp" in n for n in claimed)

    def test_callback_survives_a_moe_mtp_head(self):
        """Regression: the callback used to raise on the MTP head's layer_idx=-1."""
        _, model = _model(
            prores_enabled=True, prores_warmup_steps=8, mtp_block_type="moe"
        )
        callback = ProResWarmup()
        found = callback._find_modules(_FakeState(model))
        assert [m._prores_layer_idx for m in found] == list(range(N_LAYERS))
        callback._modules = found
        callback._update_modules(step=4)
        assert callback._last_alphas == prores_alphas(
            step=4, num_layers=N_LAYERS, warmup_steps=8
        )
        callback._update_modules(step=8)
        assert callback._complete is True

    def test_alpha_reaches_both_buffers_and_scales_the_norm(self):
        config, model = _model(prores_enabled=True, prores_warmup_steps=8)
        layer = [
            m for m in model.modules()
            if getattr(m, "_prores_root_module", False) and m._prores_enabled
        ][0]
        layer.set_prores_alpha(0.25)
        assert layer.get_prores_alpha() == 0.25
        assert layer.self_attn._prores_alpha.item() == 0.25
        assert layer._prores_output_scale() is not None

        layer.set_prores_alpha(1.0)
        assert layer._prores_output_scale() is None, "alpha=1 must cost nothing"

    # `output_scale` on real norm forwards is a GPU test: `GatedNorm.forward` calls
    # `mark_dynamic`, which blows up on a fixed-shape CPU tensor, and `LlamaRMSNorm`
    # dispatches to the CUDA-only `dropout_add_rms_norm`. See
    # `sber_tests/unit/gpu/test_prores_output_scale.py`.
    @pytest.mark.parametrize("norm_type", ["rmsnorm", "ZeroCenteredRMSNorm"])
    def test_cpu_norms_honour_output_scale(self, norm_type):
        from llmfoundry.models.layers.norm import resolve_norm_class

        cls = resolve_norm_class(norm_type)
        config = GigaMixConfig(hidden_size=64, tp_size=1, init_device="cpu")
        norm = cls.from_config(config)
        with torch.no_grad():
            torch.nn.init.normal_(norm.weight, std=0.5)
            x = torch.randn(2, 3, 64)
            ref = norm(x)
            assert torch.equal(norm(x, output_scale=None), ref), "None must be a no-op"
            half = norm(x, output_scale=0.5)
        assert ref.abs().max().item() > 0
        torch.testing.assert_close(half, ref * 0.5, rtol=2e-3, atol=1e-5)


class _FakeState:
    """Just enough of `composer.core.State` for `ProResWarmup._find_modules`."""

    def __init__(self, model):
        self.model = model
