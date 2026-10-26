from pathlib import Path
from typing import Any, Dict

import pytest
from node_runner.src.config_model import (
    BaseConfigModel,
    InferenceConfig,
    LmEvalEvaluatorConfig,
    LmEvalPretrainTaskConfig,
    SglangInferenceConfig,
)
from node_runner.src.metrics.evaluators.lm_eval import LmEvalEvaluator

pytestmark = pytest.mark.unit

# ============================================================================
# Constants for test configuration
# ============================================================================

# Model configuration values
TP_VALUE = 8
DP_VALUE = 1
MEM_FRACTION_STATIC = 0.7

# Data types and flags
DTYPE_AUTO = "auto"
ENABLE_TORCH_COMPILE = "enable-torch-compile"
ENABLE_DP_ATTENTION = "enable-dp-attention"

# Paths and URLs
BASE_URL = "http://127.0.0.1:30000/generate"
LM_EVAL_PYTHON_PATH = "/home/user/lm_eval/bin/python"

# Task names
TASK_AGIEVAL = "agieval"
TASK_BBH = "bbh"

# Run names (task name + requested shots)
RUN_AGIEVAL_0SHOT = "agieval_0shot"
RUN_BBH_3SHOT = "bbh_3shot"

# Model arguments
MAX_LENGTH = 8192
NUM_CONCURRENT = 8
BATCH_SIZE = 16
NUM_FEWSHOT_0 = 0
NUM_FEWSHOT_3 = 3

# Test values
TEST_BOOL_TRUE = True
TEST_BOOL_FALSE = False
TEST_NUMERIC_VALUE = 42
TEST_FLOAT_VALUE = 3.14

# ============================================================================
# Helper functions
# ============================================================================


def assert_bash_list(config: Dict[str, Any], expected: list[str]) -> None:
    """Helper to assert bash list conversion result."""
    result = BaseConfigModel.model_validate(config).to_bash_list()
    assert result == expected, f"Expected {expected}, got {result}"


class TestConfigConverter:
    """Tests for BaseConfigModel.to_bash_list()."""

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (
                {
                    "tp": TP_VALUE,
                    "dp": DP_VALUE,
                    "dtype": DTYPE_AUTO,
                    ENABLE_TORCH_COMPILE: TEST_BOOL_FALSE,
                    ENABLE_DP_ATTENTION: TEST_BOOL_TRUE,
                    "mem-fraction-static": MEM_FRACTION_STATIC,
                },
                [
                    "--tp",
                    str(TP_VALUE),
                    "--dp",
                    str(DP_VALUE),
                    "--dtype",
                    DTYPE_AUTO,
                    f"--{ENABLE_DP_ATTENTION}",
                    "--mem-fraction-static",
                    str(MEM_FRACTION_STATIC),
                ],
            ),
            (
                {},
                [],
            ),
            (
                {
                    ENABLE_TORCH_COMPILE: TEST_BOOL_FALSE,
                    ENABLE_DP_ATTENTION: TEST_BOOL_TRUE,
                },
                [f"--{ENABLE_DP_ATTENTION}"],
            ),
        ],
    )
    def test_dict_to_bash_list_basic_cases(
        self, config: Dict[str, Any], expected: list[str]
    ) -> None:
        """Test basic cases of dictionary to bash list conversion."""
        assert_bash_list(config, expected)

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (
                {"int_value": TEST_NUMERIC_VALUE},
                ["--int_value", str(TEST_NUMERIC_VALUE)],
            ),
            (
                {"float_value": TEST_FLOAT_VALUE},
                ["--float_value", str(TEST_FLOAT_VALUE)],
            ),
            (
                {"path": "/home/user/test/path with spaces/file.txt"},
                ["--path", "/home/user/test/path with spaces/file.txt"],
            ),
            (
                {"nested": {"list": [1, 2, 3], "bool": True}},
                ["--nested", '{"list": [1, 2, 3], "bool": true}'],
            ),
            ({"empty_dict": {}}, ["--empty_dict", "{}"]),
        ],
    )
    def test_dict_to_bash_list_edge_cases(
        self, config: Dict[str, Any], expected: list[str]
    ) -> None:
        """Test edge cases of dictionary to bash list conversion."""
        assert_bash_list(config, expected)

    def test_dict_to_bash_list_with_complex_nested_structures(self) -> None:
        """Test conversion with complex nested data structures."""
        config = {
            "simple": "value",
            "nested_dict": {
                "level1": {"level2": ["a", "b", "c"], "level2_bool": False}
            },
            "list_value": [1, 2, 3],
            "boolean_true": True,
            "boolean_false": False,
        }

        result = BaseConfigModel.model_validate(config).to_bash_list()

        assert "--simple" in result
        assert "value" in result

        assert "--nested_dict" in result
        nested_json = result[result.index("--nested_dict") + 1]
        assert "level1" in nested_json
        assert "level2" in nested_json
        assert '"level2_bool": false' in nested_json

        assert "--boolean_true" in result
        assert "--boolean_false" not in result

        assert "--list_value" in result
        list_json = result[result.index("--list_value") + 1]
        assert "[1, 2, 3]" in list_json

    def test_to_dict_uses_aliases_and_skips_none_values(self) -> None:
        cfg = SglangInferenceConfig.model_validate({"tp": 2, "chat_template": None})
        dumped = cfg.to_dict()
        assert dumped["tp-size"] == 2
        assert "chat-template" not in dumped

    def test_base_inference_config_accepts_runtime_engine_type(self) -> None:
        cfg = InferenceConfig.model_validate({"engine-type": "vllm"})
        assert cfg.engine_type == "vllm"

    def test_eval_task_to_bash_list_serializes_extra_dict_values(self) -> None:
        cfg = LmEvalPretrainTaskConfig.model_validate(
            {
                "model_args": {"base_url": BASE_URL},
                "trust_remote_code": False,
                "extra_payload": {"nested": "value"},
            }
        )
        out = cfg.to_bash_list()

        assert "--trust_remote_code" not in out
        assert "--extra_payload" in out
        assert out[out.index("--extra_payload") + 1] == '{"nested": "value"}'
        assert out[out.index("--model_args") + 1] == f"base_url={BASE_URL}"


class TestLmEvalConversion:
    """Tests for LM Eval configuration conversion using Pydantic models."""

    def test_lm_eval_to_bash_list(self) -> None:
        """Test conversion of LM Eval runs to bash commands."""
        lm_eval_config = LmEvalEvaluatorConfig(
            pretrain={
                TASK_AGIEVAL: LmEvalPretrainTaskConfig(
                    batch_size=BATCH_SIZE,
                    model_args={
                        "base_url": BASE_URL,
                        "max_length": MAX_LENGTH,
                        "num_concurrent": NUM_CONCURRENT,
                        "tokenized_requests": True,
                    },
                    num_fewshot=NUM_FEWSHOT_0,
                ),
                TASK_BBH: LmEvalPretrainTaskConfig(
                    batch_size=BATCH_SIZE,
                    model_args={
                        "base_url": BASE_URL,
                        "max_length": MAX_LENGTH,
                        "num_concurrent": NUM_CONCURRENT,
                        "tokenized_requests": True,
                    },
                    num_fewshot=NUM_FEWSHOT_3,
                ),
            }
        )
        out = LmEvalEvaluator()._lm_eval_runs_to_bash_list(
            lm_eval_config, Path("metrics")
        )

        assert out[RUN_AGIEVAL_0SHOT][:3] == [LM_EVAL_PYTHON_PATH, "-m", "lm_eval"]
        assert out[RUN_AGIEVAL_0SHOT][3:5] == ["--tasks", TASK_AGIEVAL]

        assert out[RUN_BBH_3SHOT][:3] == [LM_EVAL_PYTHON_PATH, "-m", "lm_eval"]
        assert out[RUN_BBH_3SHOT][3:5] == ["--tasks", TASK_BBH]
