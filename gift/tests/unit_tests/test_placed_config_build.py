"""Checked collection views must preserve the wrapper's build policy."""

import torch

from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
    placed_configs,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig


def test_checked_view_builds_frozen_module_instead_of_unwrapped_trainable_module():
    wrapped = FrozenConfig(inner=LinearConfig(in_features=2, out_features=3))
    view = placed_configs([wrapped], LinearConfig, path="layers")
    with torch.device("meta"):
        frozen = view[0].build()
        trainable = wrapped.inner.build()
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    assert all(parameter.requires_grad for parameter in trainable.parameters())
