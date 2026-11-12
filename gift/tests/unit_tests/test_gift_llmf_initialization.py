"""Packed initialization matches the separate logical projection recipe."""

from math import sqrt

import pytest
import torch
from torch import nn

from torchtitan.config.schema.models.gift_moe.const import (
    MLAKVAInit,
    MLAKVBInit,
    RenormKeepStd,
)
from torchtitan.config.schema.protocols.module import XavierNormal
from torchtitan.protocols.module import resolve_param_init


@torch.no_grad()
def test_xavier_normal_and_router_renorm_match_recipe():
    expected = torch.empty(32, 128)
    actual = torch.empty_like(expected)
    torch.manual_seed(42)
    nn.init.xavier_normal_(expected)
    std = expected.std()
    expected.div_(expected.norm(dim=1, keepdim=True))
    expected.mul_(std / expected.std())
    torch.manual_seed(42)
    resolve_param_init(RenormKeepStd(XavierNormal()))(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_mla_kv_a_matches_two_separate_xavier_matrices():
    expected = torch.empty(18, 64)
    actual = torch.empty_like(expected)
    torch.manual_seed(42)
    nn.init.xavier_normal_(expected[:12])
    nn.init.xavier_normal_(expected[12:])
    torch.manual_seed(42)
    resolve_param_init(MLAKVAInit(kv_rank=12))(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_mla_kv_b_uses_logical_fans_and_packed_head_order():
    expected = torch.empty(4 * (8 + 16), 12)
    actual = torch.empty_like(expected)
    packed = expected.view(4, 24, 12)
    torch.manual_seed(42)
    nn.init.normal_(packed[:, :8], std=sqrt(2 / (12 + 4 * 8)))
    nn.init.normal_(packed[:, 8:], std=sqrt(2 / (12 + 4 * 16)))
    torch.manual_seed(42)
    resolve_param_init(MLAKVBInit(num_heads=4, key_dim=8, value_dim=16))(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "spec, shape",
    [
        (MLAKVAInit(kv_rank=12), (12, 64)),
        (MLAKVBInit(num_heads=4, key_dim=8, value_dim=16), (95, 12)),
    ],
)
def test_packed_initializers_reject_invalid_shapes(spec, shape):
    with pytest.raises(ValueError):
        resolve_param_init(spec)(torch.empty(shape))
