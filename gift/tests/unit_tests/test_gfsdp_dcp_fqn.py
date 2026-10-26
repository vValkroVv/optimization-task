"""``ptd_checkpoint_wrapper`` inserts a ``_checkpoint_wrapped_module`` segment into
parameter FQNs, and DCP's ``get_model_state_dict`` strips it. GFSDP's checkpoint
bindings are keyed by raw ``named_parameters`` FQNs, so they must strip it too --
otherwise every activation-checkpointed block goes unmapped and interval saving
fails."""

import pytest

pytest.importorskip("torch")

from torchtitan.distributed.gfsdp._dcp import _strip_ac_wrapper


def test_strips_wrapper_segment():
    assert (
        _strip_ac_wrapper("layers.0._checkpoint_wrapped_module.attention.A_log")
        == "layers.0.attention.A_log"
    )


def test_strips_every_occurrence():
    assert (
        _strip_ac_wrapper(
            "layers.0._checkpoint_wrapped_module.moe._checkpoint_wrapped_module.w1"
        )
        == "layers.0.moe.w1"
    )


def test_leaves_unwrapped_fqns_untouched():
    for fqn in ("tok_embeddings.weight", "norm.weight", "lm_head.weight"):
        assert _strip_ac_wrapper(fqn) == fqn


def test_matches_whole_segments_only():
    # A substring strip would corrupt these; only an exact segment is a wrapper.
    assert (
        _strip_ac_wrapper("layers.0.my_checkpoint_wrapped_module.w")
        == "layers.0.my_checkpoint_wrapped_module.w"
    )
    assert (
        _strip_ac_wrapper("_checkpoint_wrapped_module_extra.w")
        == "_checkpoint_wrapped_module_extra.w"
    )
