"""Training scalar invariants fail before runtime imports in every public loader."""

import pytest

from torchtitan.config import load_config, run_pre_submit
from torchtitan.config.configs import TrainingConfig
from torchtitan.config.errors import ConfigFieldError, ConfigSchemaError

EXPERIMENT = "configs/experiments/llama3_debug.yaml"


@pytest.mark.parametrize("field", ["steps", "seq_len"])
@pytest.mark.parametrize("value", ["0", "-5", "true", "0.5"])
def test_invalid_training_scalar_is_a_precise_pre_submit_and_loader_error(field, value):
    path = f"training.{field}"
    result = run_pre_submit(EXPERIMENT, [(path, value)])
    assert not result.ok
    assert [issue.path for issue in result.report.errors] == [path]
    with pytest.raises(ConfigSchemaError, match=path):
        load_config(EXPERIMENT, overrides=[f"--{path}", value])


@pytest.mark.parametrize("field", ["steps", "seq_len"])
@pytest.mark.parametrize("value", [0, -5, True, 0.5])
def test_typed_training_construction_enforces_the_same_positive_integer_contract(
    field, value
):
    with pytest.raises(ConfigFieldError) as caught:
        TrainingConfig(**{field: value})
    assert caught.value.field == field
    assert "positive integer" in str(caught.value)


def test_training_resets_and_unlimited_validation_keep_their_defaults():
    result = run_pre_submit(
        EXPERIMENT,
        [
            ("training.steps", "null"),
            ("training.seq_len", "null"),
            ("validator.steps", "-1"),
        ],
    )
    assert result.ok, result.report.errors


def test_one_step_one_token_training_remains_valid():
    result = run_pre_submit(
        EXPERIMENT, [("training.steps", "1"), ("training.seq_len", "1")]
    )
    assert result.ok, result.report.errors
