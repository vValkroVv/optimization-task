import pytest
from pydantic import ValidationError

from node_runner.src.config_model import (
    SGLANG_DEFAULT_DISAGGREGATION_BOOTSTRAP_PORT,
    AutometricsConfig,
    SglangInferenceConfig,
    SglangPdInferenceConfig,
)

pytestmark = pytest.mark.unit


def _autometrics(inference: dict) -> AutometricsConfig:
    return AutometricsConfig.model_validate(
        {
            "globals": {},
            "router": {},
            "evaluators": {},
            "preparation": {},
            "convertation": {},
            "model_info": {"model_class": "giga", "size": "10b", "username": "u"},
            "inference": inference,
        }
    )


class TestPdDetection:
    def test_prefill_and_decode_blocks_enable_pd(self):
        cfg = _autometrics({"prefill": {"tp": 16}, "decode": {"tp": 16}})
        assert isinstance(cfg.inference, SglangPdInferenceConfig)

    def test_explicit_engine_type_is_accepted(self):
        cfg = _autometrics(
            {
                "engine_type": "sglang_pd",
                "prefill": {"tp": 8},
                "decode": {"tp": 8},
            }
        )
        assert isinstance(cfg.inference, SglangPdInferenceConfig)

    def test_flat_config_stays_plain_sglang(self):
        cfg = _autometrics({"tp": 16, "ep": 16})
        assert isinstance(cfg.inference, SglangInferenceConfig)
        assert "--disaggregation-mode" not in cfg.inference.to_bash_list()

    def test_pd_requires_both_sides(self):
        with pytest.raises(ValidationError):
            SglangPdInferenceConfig.model_validate({"prefill": {"tp": 8}})

    @pytest.mark.parametrize(
        "inference",
        [
            {"tp": 16, "prefill": {"tp": 16}},
            {"tp": 16, "decode": {"tp": 16}},
            {"tp": 16, "prefill": {"tp": 16}, "decdoe": {"tp": 16}},
        ],
    )
    def test_one_sided_pd_block_is_rejected(self, inference: dict):
        with pytest.raises(ValidationError, match="both prefill and decode"):
            _autometrics(inference)

    def test_one_sided_pd_block_is_rejected_in_secondary_preset(self):
        with pytest.raises(ValueError, match="both prefill and decode"):
            SglangInferenceConfig.for_secondary_preset(
                "pollux-judge-32b", {"pollux-judge-32b": {"decode": {"tp": 8}}}
            )


class TestDisaggregationRoles:
    def test_sides_get_modes_and_prefill_gets_bootstrap_port(self):
        pd = _autometrics({"prefill": {"tp": 16}, "decode": {"tp": 16}}).inference

        assert pd.prefill.disaggregation_mode == "prefill"
        assert pd.decode.disaggregation_mode == "decode"
        assert (
            pd.prefill.disaggregation_bootstrap_port
            == SGLANG_DEFAULT_DISAGGREGATION_BOOTSTRAP_PORT
        )
        assert pd.decode.disaggregation_bootstrap_port is None

    def test_explicit_bootstrap_port_is_kept(self):
        pd = _autometrics(
            {
                "prefill": {"tp": 16, "disaggregation-bootstrap-port": 8997},
                "decode": {"tp": 16},
            }
        ).inference
        assert pd.prefill.disaggregation_bootstrap_port == 8997

    def test_side_cli_contains_disaggregation_flags(self):
        pd = _autometrics({"prefill": {"tp": 16}, "decode": {"tp": 16}}).inference

        prefill_bash = pd.prefill.to_bash_list()
        i = prefill_bash.index("--disaggregation-mode")
        assert prefill_bash[i + 1] == "prefill"
        assert "--disaggregation-bootstrap-port" in prefill_bash

        decode_bash = pd.decode.to_bash_list()
        i = decode_bash.index("--disaggregation-mode")
        assert decode_bash[i + 1] == "decode"
        assert "--disaggregation-bootstrap-port" not in decode_bash

    def test_prefill_fraction_bounds(self):
        for bad_fraction in (0.0, 1.0):
            with pytest.raises(ValidationError):
                _autometrics(
                    {
                        "prefill": {"tp": 8},
                        "decode": {"tp": 8},
                        "prefill_fraction": bad_fraction,
                    }
                )


class TestSharedParams:
    def test_side_value_wins_over_shared(self):
        pd = _autometrics(
            {
                "shared_params": {"mem-fraction-static": 0.7},
                "prefill": {"tp": 16, "mem-fraction-static": 0.9},
                "decode": {"tp": 16, "mem-fraction-static": 0.85},
            }
        ).inference
        assert pd.prefill.mem_fraction_static == 0.9
        assert pd.decode.mem_fraction_static == 0.85

    def test_shared_value_reaches_both_sides(self):
        pd = _autometrics(
            {
                "shared-params": {"mem-fraction-static": 0.75},
                "prefill": {"tp": 8},
                "decode": {"tp": 8},
            }
        ).inference
        assert pd.prefill.mem_fraction_static == 0.75
        assert pd.decode.mem_fraction_static == 0.75

    def test_empty_shared_params_is_allowed(self):
        pd = _autometrics(
            {"shared_params": {}, "prefill": {"tp": 8}, "decode": {"tp": 8}}
        ).inference
        assert isinstance(pd, SglangPdInferenceConfig)

    def test_stray_top_level_key_is_rejected(self):
        with pytest.raises(ValidationError, match="mem-fraction-static"):
            _autometrics(
                {
                    "mem-fraction-static": 0.7,
                    "prefill": {"tp": 8},
                    "decode": {"tp": 8},
                }
            )


class TestTokenizerPath:
    def test_shared_tokenizer_reaches_sides_and_obm_proxy(self):
        cfg = _autometrics(
            {
                "shared_params": {"tokenizer-path": "/tok"},
                "prefill": {"tp": 8},
                "decode": {"tp": 8},
            }
        )
        assert cfg.inference.prefill.tokenizer_path == "/tok"
        assert cfg.inference.decode.tokenizer_path == "/tok"
        assert cfg.obm_proxy.tokenizer_path == "/tok"

    def test_obm_proxy_tokenizer_propagates_into_both_sides(self):
        cfg = AutometricsConfig.model_validate(
            {
                "globals": {},
                "router": {},
                "evaluators": {},
                "preparation": {},
                "convertation": {},
                "model_info": {"model_class": "giga", "size": "10b", "username": "u"},
                "obm_proxy": {"tokenizer-path": "/from-obm"},
                "inference": {"prefill": {"tp": 8}, "decode": {"tp": 8}},
            }
        )
        assert cfg.inference.prefill.tokenizer_path == "/from-obm"
        assert cfg.inference.decode.tokenizer_path == "/from-obm"

    def test_diverged_side_tokenizers_are_rejected(self):
        with pytest.raises(ValidationError, match="share tokenizer_path"):
            _autometrics(
                {
                    "prefill": {"tp": 8, "tokenizer-path": "/only-prefill"},
                    "decode": {"tp": 8},
                }
            )


class TestSecondaryPreset:
    def test_flat_preset_stays_plain_sglang(self):
        judge = SglangInferenceConfig.for_secondary_preset("pollux-judge-32b", {})
        assert isinstance(judge, SglangInferenceConfig)
        assert judge.tp == 2 and judge.dp == 4

    def test_pd_override_uses_preset_as_shared_base(self):
        judge = SglangInferenceConfig.for_secondary_preset(
            "pollux-judge-32b",
            {"pollux-judge-32b": {"prefill": {}, "decode": {"dp": 2}}},
        )
        assert isinstance(judge, SglangPdInferenceConfig)
        assert judge.prefill.tp == 2 and judge.prefill.dp == 4
        assert judge.decode.tp == 2 and judge.decode.dp == 2
        assert judge.prefill.disaggregation_mode == "prefill"

    def test_user_shared_params_override_preset(self):
        judge = SglangInferenceConfig.for_secondary_preset(
            "pollux-judge-32b",
            {
                "pollux-judge-32b": {
                    "shared_params": {"dp": 8},
                    "prefill": {},
                    "decode": {"dp": 2},
                }
            },
        )
        assert judge.prefill.dp == 8
        assert judge.decode.dp == 2

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown SGLang preset"):
            SglangInferenceConfig.for_secondary_preset("no-such-model", {})
