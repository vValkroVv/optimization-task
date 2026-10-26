"""Completed-tree checks run at the boundaries a tree crosses, not on copies:
in-place edits bypass ``_copy``, so the boundaries must check anyway."""

import pytest

from torchtitan.config.configurable import config
from torchtitan.config.schema.protocols.model import BaseModelConfig


@config
class CompletionProbeConfig(BaseModelConfig):
    width: int | None = None

    def validate_materialized(self):
        if self.width is None or self.width <= 0:
            raise ValueError("completed probe needs a positive width")


def test_written_copy_remains_partial():
    written = CompletionProbeConfig()
    copied = written._copy()
    assert copied.width is None
    assert not copied.materialized


def test_completed_copy_is_checked_at_the_next_boundary():
    from torchtitan.protocols.model import materialize_runtime_model_config

    tree = CompletionProbeConfig(width=4).materialize()
    assert tree.materialized
    broken = tree._copy(width=0)
    assert broken.materialized and tree.width == 4
    with pytest.raises(ValueError, match="positive width"):
        broken.materialize()
    with pytest.raises(ValueError, match="positive width"):
        materialize_runtime_model_config(broken)
    copied = tree._copy(width=8)
    assert copied.materialized and copied.width == 8
    assert materialize_runtime_model_config(copied) is copied


def test_base_materialization_checks_its_completed_result():
    with pytest.raises(ValueError, match="positive width"):
        CompletionProbeConfig().materialize()


def test_completed_copy_preserves_independent_metadata():
    tree = CompletionProbeConfig(width=4).materialize()
    tree._set_metadata("source_fields", {"width": "model.config.width"})
    copied = tree._copy(width=8)
    assert copied._get_metadata("source_fields", None) == {
        "width": "model.config.width"
    }
    copied._set_metadata("source_fields", {"width": "other.width"})
    assert tree._get_metadata("source_fields", None) == {"width": "model.config.width"}


def _decoder_tree():
    from torchtitan.config import load_raw_document
    from torchtitan.config.decode import decode
    from torchtitan.config.sections import ModelSection

    document = load_raw_document("configs/experiments/llama3_debug.yaml")
    return decode(ModelSection, document.data["model"]).config.materialize()


def test_completed_decoder_copy_with_incompatible_head_fails_at_the_boundary():
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig

    tree = _decoder_tree()
    broken = tree._copy(lm_head=LinearConfig(in_features=1, out_features=2))
    with pytest.raises(ValueError, match="lm_head"):
        broken.materialize()
    assert tree.lm_head.in_features == tree.dim


@pytest.mark.parametrize("entrypoint", ["direct", "builder"])
def test_inplace_decoder_edit_is_checked_before_reuse(entrypoint):
    from torchtitan.config.errors import ConfigSchemaError
    from torchtitan.config.build import materialize_model_config
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig

    tree = _decoder_tree()
    tree.lm_head = LinearConfig(in_features=1, out_features=2)
    error = ValueError if entrypoint == "direct" else ConfigSchemaError
    with pytest.raises(error, match="lm_head"):
        if entrypoint == "direct":
            tree.materialize()
        else:
            materialize_model_config(tree)
