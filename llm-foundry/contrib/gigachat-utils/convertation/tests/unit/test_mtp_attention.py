from types import SimpleNamespace

import pytest

from utils.mtp_attention import (
    resolve_grafted_mtp_fields,
    resolve_mtp_uses_linear_attention,
    resolve_mtp_uses_sparse_ffn,
)


def _cfg(**overrides):
    cfg = {
        "num_hidden_layers": 4,
        "linear_attention_type": "Qwen3NextGatedDeltaNet",
        "full_attention_layers": [1, 3],
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.unit
class TestResolveMtpUsesLinearAttention:
    @pytest.mark.parametrize("flag", [True, False])
    def test_hf_field_wins_over_anchor(self, flag):
        cfg = _cfg(nextn_is_linear=flag, full_attention_layers=[1])
        assert resolve_mtp_uses_linear_attention(cfg) is flag

    @pytest.mark.parametrize("flag", [True, False])
    def test_llmfoundry_field_wins_over_anchor(self, flag):
        cfg = _cfg(use_linear_attention_in_mtp=flag, full_attention_layers=[1])
        assert resolve_mtp_uses_linear_attention(cfg) is flag

    def test_hf_field_takes_precedence_over_llmfoundry_field(self):
        cfg = _cfg(nextn_is_linear=False, use_linear_attention_in_mtp=True)
        assert resolve_mtp_uses_linear_attention(cfg) is False

    def test_true_without_linear_attention_type_is_an_error(self):
        with pytest.raises(ValueError, match="linear_attention_type"):
            resolve_mtp_uses_linear_attention(
                _cfg(nextn_is_linear=True, linear_attention_type=None, full_attention_layers=[])
            )

    def test_absent_flag_infers_true_from_linear_anchor(self):
        assert resolve_mtp_uses_linear_attention(_cfg(full_attention_layers=[1])) is True

    def test_absent_flag_infers_false_from_full_anchor(self):
        assert resolve_mtp_uses_linear_attention(_cfg(full_attention_layers=[1, 3])) is False

    def test_absent_flag_non_hybrid_is_false(self):
        cfg = _cfg(linear_attention_type=None, full_attention_layers=[])
        assert resolve_mtp_uses_linear_attention(cfg) is False

    def test_accepts_config_objects(self):
        cfg = SimpleNamespace(**_cfg(nextn_is_linear=True))
        assert resolve_mtp_uses_linear_attention(cfg) is True


def _hf_cfg(num_nextn_predict_layers=0, **overrides):
    cfg = {
        "num_hidden_layers": 4,
        "linear_attention_type": "Qwen3NextGatedDeltaNet",
        "full_attention_layers": [1],
        "num_nextn_predict_layers": num_nextn_predict_layers,
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.unit
class TestResolveMtpUsesSparseFfn:
    def test_absent_field_is_a_moe_head(self):
        assert resolve_mtp_uses_sparse_ffn(_hf_cfg()) is True

    @pytest.mark.parametrize("flag", [True, False])
    def test_explicit_field_wins(self, flag):
        assert resolve_mtp_uses_sparse_ffn(_hf_cfg(nextn_is_sparse=flag)) is flag


@pytest.mark.unit
class TestResolveGraftedMtpFields:
    def test_base_without_heads_takes_the_donor_mode_not_its_linear_anchor(self):
        base = _hf_cfg()
        donor = _hf_cfg(1, nextn_is_linear=False, nextn_is_sparse=False)
        assert resolve_grafted_mtp_fields(base, donor, keep_base_heads=False) == {
            "nextn_is_linear": False,
            "nextn_is_sparse": False,
        }

    def test_base_without_heads_takes_a_linear_donor(self):
        base = _hf_cfg(full_attention_layers=[1, 3])
        donor = _hf_cfg(1, nextn_is_linear=True)
        assert resolve_grafted_mtp_fields(base, donor, keep_base_heads=False) == {
            "nextn_is_linear": True,
            "nextn_is_sparse": True,
        }

    def test_kept_base_heads_in_the_same_mode_are_fine(self):
        base = _hf_cfg(1, nextn_is_linear=False, nextn_is_sparse=False)
        donor = _hf_cfg(2, nextn_is_linear=False, nextn_is_sparse=False)
        assert resolve_grafted_mtp_fields(base, donor, keep_base_heads=True) == {
            "nextn_is_linear": False,
            "nextn_is_sparse": False,
        }

    def test_kept_base_heads_in_another_attention_mode_are_rejected(self):
        base = _hf_cfg(1)  # no flag, linear last layer -> linear heads
        donor = _hf_cfg(2, nextn_is_linear=False)
        with pytest.raises(ValueError, match="nextn_is_linear: base=True, donor=False"):
            resolve_grafted_mtp_fields(base, donor, keep_base_heads=True)

    def test_kept_base_heads_with_another_ffn_are_rejected(self):
        base = _hf_cfg(1, nextn_is_linear=False, nextn_is_sparse=False)
        donor = _hf_cfg(2, nextn_is_linear=False)
        with pytest.raises(ValueError, match="nextn_is_sparse: base=False, donor=True"):
            resolve_grafted_mtp_fields(base, donor, keep_base_heads=True)

    def test_replacing_every_base_head_drops_the_base_mode(self):
        base = _hf_cfg(1)
        donor = _hf_cfg(1, nextn_is_linear=False, nextn_is_sparse=False)
        assert resolve_grafted_mtp_fields(base, donor, keep_base_heads=False) == {
            "nextn_is_linear": False,
            "nextn_is_sparse": False,
        }

    def test_linear_donor_needs_the_same_linear_attention_type(self):
        base = _hf_cfg(linear_attention_type="KimiDeltaAttention")
        donor = _hf_cfg(1, nextn_is_linear=True)
        with pytest.raises(ValueError, match="linear_attention_type differs"):
            resolve_grafted_mtp_fields(base, donor, keep_base_heads=False)
