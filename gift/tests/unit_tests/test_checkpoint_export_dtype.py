"""HF export dtype conversion preserves the semantics of non-weight buffers."""

import torch

from torchtitan.components.checkpoint_conversion import cast_export_state_dict


def test_export_casts_float_weights_and_preserves_integer_and_boolean_buffers():
    state = {
        "weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "indices": torch.tensor([1, 2], dtype=torch.int64),
        "mask": torch.tensor([True, False]),
    }
    exported = cast_export_state_dict(state, torch.bfloat16)
    assert exported["weight"].dtype == torch.bfloat16
    assert exported["indices"] is state["indices"]
    assert exported["mask"] is state["mask"]
    assert state["weight"].dtype == torch.float32


def test_float32_export_upcasts_bfloat16_weights():
    state = {"weight": torch.tensor([1.0], dtype=torch.bfloat16)}
    exported = cast_export_state_dict(state, torch.float32)
    assert exported["weight"].dtype == torch.float32
    assert torch.equal(exported["weight"], state["weight"].float())
