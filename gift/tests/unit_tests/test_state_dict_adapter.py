# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchtitan.protocols.state_dict_adapter import StateDictAdapter


@pytest.mark.parametrize("transposed", [False, True])
def test_w13_round_trip(transposed):
    # Unequal dims expose stacking along the wrong axis; transposed inputs are
    # the noncontiguous views HF adapters hand over after [E, D, F] -> [E, F, D].
    shape = (2, 5, 3) if transposed else (2, 3, 5)
    w1 = torch.arange(30, dtype=torch.float32).reshape(shape)
    w3 = -w1 - 1
    if transposed:
        w1, w3 = w1.transpose(1, 2), w3.transpose(1, 2)

    fused = StateDictAdapter.separate_to_fused_w13(w1, w3)
    assert fused.shape == (2, 2, 3, 5)
    assert torch.equal(fused[:, 0], w1) and torch.equal(fused[:, 1], w3)
    restored_w1, restored_w3 = StateDictAdapter.fused_to_separate_w13(fused)
    assert torch.equal(restored_w1, w1) and torch.equal(restored_w3, w3)


def test_w13_slot_order_matches_the_kernel():
    """Slot 0 is gate and slot 1 is up, as the folded (E, 2F, D) weight is read."""
    w1 = torch.arange(30, dtype=torch.float64).reshape(2, 3, 5) / 30
    w3 = w1.flip(-1) - 0.5
    x = torch.arange(20, dtype=torch.float64).reshape(2, 2, 5) / 20
    folded = StateDictAdapter.separate_to_fused_w13(w1, w3).view(2, 6, 5)
    gate, up = (x @ folded.transpose(-2, -1)).chunk(2, dim=-1)
    expected = F.silu(x @ w1.transpose(-2, -1)) * (x @ w3.transpose(-2, -1))
    torch.testing.assert_close(F.silu(gate) * up, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("w1", "w3", "match"),
    [
        (torch.empty(2, 3, 5), torch.empty(2, 4, 5), "W1 and W3"),
        (torch.empty(3, 5), torch.empty(3, 5), "W1 and W3"),
        (torch.empty(2, 3, 5, dtype=torch.bfloat16), torch.empty(2, 3, 5), "dtype"),
        (torch.empty(2, 3, 5), torch.empty(2, 3, 5, device="meta"), "device"),
    ],
)
def test_separate_to_fused_w13_rejects(w1, w3, match):
    with pytest.raises(ValueError, match=match):
        StateDictAdapter.separate_to_fused_w13(w1, w3)


@pytest.mark.parametrize("shape", [(2, 6, 5), (2, 3, 3, 5)])
def test_fused_to_separate_w13_rejects(shape):
    # (2, 6, 5) is the old (E, 2F, D) layout; (2, 3, 3, 5) has no gate/up slot dim.
    with pytest.raises(ValueError, match="W13"):
        StateDictAdapter.fused_to_separate_w13(torch.empty(shape))
