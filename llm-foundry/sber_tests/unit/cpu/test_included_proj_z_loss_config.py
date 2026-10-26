# Copyright 2024 LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Config-surface tests for the hidden z-loss coefficient dicts.

`sber_tests/unit/gpu/test_hidden_z_loss.py` covers the hooks firing on real
modules. What it cannot cover is what the *config* accepts and rejects, and that
is where a typo silently costs a run: a misspelled projection that is quietly
ignored, or YAML string layer keys that never match an int `layer_idx`.

Everything here builds a real `GigaMixConfig` and calls the real resolvers -- no
import stubbing, so this file is safe to run alongside any other test -- and it
needs no GPU.
"""

import pytest

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.ops.included_proj_z_loss import (
    INCLUDED_PROJECTIONS,
    MTP_Z_LOSS_COMPONENTS,
    included_proj_active,
    resolve_included_proj_coef,
    resolve_mtp_z_loss,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _cfg(**kw):
    return GigaMixConfig(**kw)


# --------------------------------------------------------------------------- #
# hidden_z_loss_included_proj_coef (trunk, keyed by layer_idx)
# --------------------------------------------------------------------------- #

class TestIncludedProjConfig:

    def test_inactive_by_default(self):
        cfg = _cfg()
        for proj in INCLUDED_PROJECTIONS:
            assert included_proj_active(cfg, proj) is False

    def test_qkv_active_with_supported_attention(self):
        cfg = _cfg(
            hidden_z_loss_included_proj_coef={"qkv": {"base": 1e-3}},
            attention_type="LlamaPackedAttention",
        )
        assert included_proj_active(cfg, "qkv") is True

    def test_projections_are_independent(self):
        cfg = _cfg(
            hidden_z_loss_included_proj_coef={"up_mlp": 1e-3, "up_moe": {"base": 2e-3}},
        )
        assert included_proj_active(cfg, "up_mlp") is True
        assert included_proj_active(cfg, "up_moe") is True
        assert included_proj_active(cfg, "gate_mlp") is False

    def test_layerwise_keys_normalized_to_int(self):
        """YAML gives string keys; an unnormalized key would never match layer_idx."""
        cfg = _cfg(
            hidden_z_loss_included_proj_coef={
                "qkv": {"base": 0.0, "layerwise": {"3": 2e-3}},
            },
        )
        spec = cfg.hidden_z_loss_included_proj_coef["qkv"]
        assert spec["layerwise"] == {3: 2e-3}
        assert included_proj_active(cfg, "qkv") is True
        assert resolve_included_proj_coef(cfg, "qkv", 3) == pytest.approx(2e-3)
        assert resolve_included_proj_coef(cfg, "qkv", 2) == 0.0

    def test_bare_scalar_applies_to_every_layer(self):
        cfg = _cfg(hidden_z_loss_included_proj_coef={"up_mlp": 1e-3})
        assert resolve_included_proj_coef(cfg, "up_mlp", 0) == pytest.approx(1e-3)
        assert resolve_included_proj_coef(cfg, "up_mlp", 99) == pytest.approx(1e-3)

    def test_unknown_projection_raises(self):
        with pytest.raises(ValueError, match="attn"):
            _cfg(hidden_z_loss_included_proj_coef={"attn": 1e-3})

    def test_no_raise_for_qkv_on_unsupported_attention(self):
        # A qkv coef with an unsupported attention class must NOT raise: mixed
        # networks are allowed, so the config builds and stays active and the
        # model emits a per-layer warning at build time instead.
        cfg = _cfg(
            hidden_z_loss_included_proj_coef={"qkv": 1e-3},
            attention_type="LlamaLatentAttention",
        )
        assert included_proj_active(cfg, "qkv") is True

    def test_default_is_none_so_config_overrides_can_replace_it(self):
        """HF `config_overrides` only *updates* an existing dict but *replaces* a
        None, so the default must stay None for YAML to set the whole dict."""
        assert _cfg().hidden_z_loss_included_proj_coef is None


# --------------------------------------------------------------------------- #
# hidden_z_loss_mtp_coef (MTP heads, keyed by predictor index)
# --------------------------------------------------------------------------- #

class TestMtpCoefConfig:

    def test_absent_component_returns_none(self):
        """None means "no override" -- distinct from an explicit 0.0."""
        cfg = _cfg(hidden_z_loss_mtp_coef={"attn": 1e-7})
        assert resolve_mtp_z_loss(cfg, "up_mlp", 0) is None
        assert resolve_mtp_z_loss(cfg, "attn", 0) == pytest.approx(1e-7)

    def test_unset_config_returns_none_for_everything(self):
        cfg = _cfg()
        assert cfg.hidden_z_loss_mtp_coef is None
        for component in MTP_Z_LOSS_COMPONENTS:
            assert resolve_mtp_z_loss(cfg, component, 0) is None

    def test_bare_scalar_applies_to_every_head(self):
        cfg = _cfg(hidden_z_loss_mtp_coef={"dense_mlp": 1e-7})
        assert resolve_mtp_z_loss(cfg, "dense_mlp", 0) == pytest.approx(1e-7)
        assert resolve_mtp_z_loss(cfg, "dense_mlp", 3) == pytest.approx(1e-7)

    def test_layerwise_is_keyed_by_predictor_index(self):
        cfg = _cfg(
            hidden_z_loss_mtp_coef={
                "up_mlp": {"base": 1e-8, "layerwise": {1: 5e-8}},
            },
        )
        assert resolve_mtp_z_loss(cfg, "up_mlp", 0) == pytest.approx(1e-8)
        assert resolve_mtp_z_loss(cfg, "up_mlp", 1) == pytest.approx(5e-8)

    def test_layerwise_accepts_yaml_string_keys(self):
        cfg = _cfg(
            hidden_z_loss_mtp_coef={"attn": {"base": 0.0, "layerwise": {"2": 3e-7}}},
        )
        assert resolve_mtp_z_loss(cfg, "attn", 2) == pytest.approx(3e-7)
        assert resolve_mtp_z_loss(cfg, "attn", 1) == 0.0

    def test_components_cover_block_level_and_per_projection(self):
        assert set(INCLUDED_PROJECTIONS) <= set(MTP_Z_LOSS_COMPONENTS)
        assert {"attn", "moe", "dense_mlp"} <= set(MTP_Z_LOSS_COMPONENTS)

    def test_stored_raw_so_config_overrides_survive(self):
        """Unlike the trunk dict this one is NOT normalized in __init__ (HF
        `config_overrides` bypass it anyway and the resolver tolerates the raw
        form), so what goes in is what comes out."""
        raw = {"up_mlp": {"base": 1e-8, "layerwise": {"1": 5e-8}}}
        assert _cfg(hidden_z_loss_mtp_coef=raw).hidden_z_loss_mtp_coef == raw
