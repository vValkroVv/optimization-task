"""DSA config-validation contract (the `validate_dsa_config` rules).

Every rule here maps to a real kernel constraint or a production incident:
  * indexer_topk is a power of two >= 128 -- fused radix merge and FlashMLA;
  * indexer_topk <= 4096 -- Hopper shared memory for the fused radix scratch;
  * stage2 + dsa_block_checkpoint REJECTED -- a nested per-block checkpoint under the
    module-level activation checkpoint retains full-sequence args/outputs (~6GB per DSA
    layer at 1M context) and OOMed a 512-GPU run; the config must refuse it.
"""
import json

import pytest

import torch  # noqa: F401
from llmfoundry.models.giga_mix import configuration_giga_mix as giga_mix
from llmfoundry.models.utils.dsa_config import validate_dsa_config

GigaMixConfig = giga_mix.GigaMixConfig

BASE = dict(
    attention_type="LlamaLatentSparseAttention",
    indexer_n_heads=64,
    indexer_head_dim=128,
    aux_kl_loss_koeff=0.01,
)


def make(**over):
    kw = dict(BASE, indexer_topk=2048, dsa_stage=2)
    kw.update(over)
    return GigaMixConfig(**kw)


class TestIndexerTopkRules:
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize("topk", [128, 2048, 4096])
    def test_valid_topk_accepted(self, topk):
        cfg = make(indexer_topk=topk)
        assert cfg.indexer_topk == topk

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize("topk", [0, -128, True])
    def test_non_positive_topk_rejected(self, topk):
        with pytest.raises(ValueError, match="positive"):
            make(indexer_topk=topk)

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize("topk", [100, 64, 384, 768, 2000])
    def test_non_multiple_of_128_rejected(self, topk):
        with pytest.raises(ValueError, match="power of two"):
            make(indexer_topk=topk)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_topk_above_smem_cap_rejected(self):
        with pytest.raises(ValueError, match="4096"):
            make(indexer_topk=4224)


class TestStageRules:
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize("stage", [0, 3, None])
    def test_invalid_stage_rejected(self, stage):
        with pytest.raises(ValueError, match="dsa_stage"):
            make(dsa_stage=stage)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_stage2_block_checkpoint_rejected(self):
        """Regression for the 1M-context 512-GPU OOM: stage 2 must refuse per-block ckpt."""
        with pytest.raises(ValueError, match="stage-1-only"):
            make(dsa_stage=2, dsa_block_checkpoint=True)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_stage1_block_checkpoint_accepted(self):
        cfg = make(dsa_stage=1, dsa_block_checkpoint=True)
        assert cfg.dsa_block_checkpoint is True

    @pytest.mark.unit
    @pytest.mark.fast
    def test_stage2_without_block_checkpoint_accepted(self):
        cfg = make(dsa_stage=2, dsa_block_checkpoint=False)
        assert cfg.dsa_block_checkpoint is False


class TestIndexerShapeRules:
    @pytest.mark.unit
    @pytest.mark.fast
    def test_indexer_heads_must_be_64(self):
        with pytest.raises(ValueError, match="indexer_n_heads"):
            make(indexer_n_heads=32)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_indexer_head_dim_must_be_128(self):
        with pytest.raises(ValueError, match="indexer_head_dim"):
            make(indexer_head_dim=64)

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize("value", [True, "true", None])
    def test_removed_fp8_scoring_rejected(self, value):
        with pytest.raises(ValueError, match="indexer_fp8_scoring"):
            make(indexer_fp8_scoring=value)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_non_dsa_attention_skips_dsa_validation(self):
        # the _dsa_require block is gated on attention_type; plain MLA must not demand DSA knobs
        cfg = GigaMixConfig(attention_type="LlamaLatentAttention")
        assert cfg.attention_type == "LlamaLatentAttention"

class TestOverridesCannotBypassValidation:
    """`config_overrides` reach the config through setattr, after __init__ has validated.

    Nothing looked at them until the layer was built, and the layer only re-checked two
    of the rules, so a value like indexer_topk=3000 first surfaced inside the selector
    kernel. The layer now runs the same validator the config does.
    """

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("indexer_topk", 3000, "power of two"),
            ("indexer_topk", 8192, "4096"),
            ("indexer_topk", 0, "positive"),
            ("dsa_stage", 3, "dsa_stage"),
            ("indexer_n_heads", 32, "indexer_n_heads"),
            ("indexer_head_dim", 64, "indexer_head_dim"),
            ("indexer_fp8_scoring", "yes", "indexer_fp8_scoring"),
            ("aux_kl_loss_koeff", None, "aux_kl_loss_koeff"),
            ("sp_split_type", "zigzag", "sp_split_type"),
        ],
    )
    def test_override_applied_after_init_is_still_rejected(self, field, value, match):
        cfg = make()                      # valid at construction time
        setattr(cfg, field, value)        # exactly what config_overrides does
        with pytest.raises(ValueError, match=match):
            validate_dsa_config(cfg)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_stage2_block_checkpoint_override_is_rejected(self):
        cfg = make(dsa_stage=2)
        cfg.dsa_block_checkpoint = True
        with pytest.raises(ValueError, match="stage-1-only"):
            validate_dsa_config(cfg)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_a_valid_config_passes_the_shared_validator(self):
        validate_dsa_config(make())       # must not raise

    @pytest.mark.unit
    @pytest.mark.fast
    def test_plain_mla_is_not_asked_for_dsa_knobs(self):
        cfg = GigaMixConfig(attention_type="LlamaLatentAttention")
        validate_dsa_config(cfg)          # must not raise

    @pytest.mark.unit
    @pytest.mark.fast
    def test_indexer_knobs_without_dsa_are_rejected(self):
        cfg = GigaMixConfig(attention_type="LlamaLatentAttention")
        cfg.indexer_n_heads = 64
        with pytest.raises(ValueError, match="attention_type"):
            validate_dsa_config(cfg)

class TestOptionalFieldsFollowTheLayerDefaults:
    """The layer reads some knobs through getattr with a default, so a config that never
    mentions them is valid -- hand-built PretrainedConfigs in the GPU tests do exactly
    that. A value that IS set is still checked, so an explicit None cannot reach a kernel.
    """

    @staticmethod
    def _bare(**over):
        from types import SimpleNamespace
        kw = dict(
            attention_type="LlamaLatentSparseAttention",
            indexer_n_heads=64,
            indexer_head_dim=128,
            indexer_topk=2048,
            aux_kl_loss_koeff=0.01,
            dsa_stage=2,
        )
        kw.update(over)
        return SimpleNamespace(**kw)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_a_config_without_the_optional_knobs_is_accepted(self):
        validate_dsa_config(self._bare())

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("indexer_fp8_scoring", None, "indexer_fp8_scoring"),
            ("sp_split_type", "zigzag", "sp_split_type"),
        ],
    )
    def test_a_knob_that_is_set_is_still_checked(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            validate_dsa_config(self._bare(**{field: value}))


class TestLegacyConfigRoundTrip:
    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize("stage", [1, 2])
    def test_saved_config_drops_removed_fields(self, tmp_path, stage):
        legacy = make(dsa_stage=stage, dsa_sel_cache_offload=True).to_dict()
        legacy.update(indexer_tile=256, indexer_fp8_scoring=False)
        (tmp_path / "config.json").write_text(json.dumps(legacy))

        loaded = GigaMixConfig.from_pretrained(tmp_path)
        assert loaded.indexer_topk == 2048 and loaded.dsa_stage == stage
        assert loaded.dsa_sel_cache_offload is True
        assert not hasattr(loaded, "indexer_tile")
        assert not hasattr(loaded, "indexer_fp8_scoring")

        loaded.save_pretrained(tmp_path / "converted")
        saved = json.loads((tmp_path / "converted/config.json").read_text())
        assert "indexer_tile" not in saved and "indexer_fp8_scoring" not in saved
        reloaded = GigaMixConfig.from_pretrained(tmp_path / "converted")
        assert reloaded.indexer_topk == loaded.indexer_topk
        assert reloaded.dsa_stage == stage and reloaded.dsa_sel_cache_offload is True

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "field,value,match",
        [("indexer_topk", 384, "power of two"),
         ("indexer_topk", 768, "power of two"),
         ("indexer_fp8_scoring", True, "indexer_fp8_scoring")],
    )
    def test_unsupported_saved_config_fails_at_load(self, tmp_path, field, value, match):
        legacy = make().to_dict()
        legacy[field] = value
        (tmp_path / "config.json").write_text(json.dumps(legacy))
        with pytest.raises(ValueError, match=match):
            GigaMixConfig.from_pretrained(tmp_path)
