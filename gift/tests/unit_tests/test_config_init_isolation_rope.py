"""Config edits cannot mutate global init defaults or switch RoPE implicitly."""

from dataclasses import replace

import pytest

from torchtitan.config.build import build_trainer_config
from torchtitan.config.cli import load_document
from torchtitan.config.errors import ConfigSchemaError
from tests.unit_tests.config_helpers import init_policy
from torchtitan.config.schema.protocols.module import Ones, Zeros

DEPTH_SCALED_INIT = init_policy("depth_scaled")


def test_replaced_init_policy_has_independent_tables_and_roles():
    first = replace(DEPTH_SCALED_INIT)
    second = replace(DEPTH_SCALED_INIT)
    assert isinstance(first.linear, dict)
    assert isinstance(first.linear_out, dict)
    first.linear["weight"] = Zeros()
    first.linear_out["weight"] = Ones()
    assert second.linear == DEPTH_SCALED_INIT.linear
    assert second.linear_out == DEPTH_SCALED_INIT.linear_out
    assert first.router_gate == DEPTH_SCALED_INIT.router_gate
    assert first.linear is not second.linear


def test_model_default_init_edit_does_not_contaminate_next_model():
    document, _ = load_document("configs/experiments/llama3_debug.yaml")
    first = build_trainer_config(document, with_model_spec=False)
    second = build_trainer_config(document, with_model_spec=False)
    # Model specs are runtime-only; inspect the actual compact schema instead.
    from torchtitan.config.decode import decode
    from torchtitan.config.sections import ModelSection

    left = decode(ModelSection, document.data["model"]).config
    right = decode(ModelSection, document.data["model"]).config
    left.init.linear["weight"] = Zeros()
    assert right.init.linear == DEPTH_SCALED_INIT.linear
    assert first.model_spec is None and second.model_spec is None


@pytest.mark.parametrize("experiment", ["qwen3_debug.yaml", "llama3_debug.yaml"])
def test_rope_type_reset_requires_explicit_backend(experiment):
    document, _ = load_document(
        "configs/experiments/" + experiment, [("model.config.rope.type", "null")]
    )
    with pytest.raises(ConfigSchemaError, match="type"):
        build_trainer_config(document, with_model_spec=False)


def test_explicit_qwen_rope_backend_survives_reset_and_reselection():
    document, _ = load_document(
        "configs/experiments/qwen3_debug.yaml",
        [("model.config.rope.type", "null"), ("model.config.rope.type", "cos_sin")],
    )
    config = build_trainer_config(document, with_model_spec=False)
    assert config.model_spec is None
