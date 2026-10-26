# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The EP and fp8 ops selective activation checkpointing must not recompute.

Run:
    pytest tests/unit_tests/test_sac_ep_fp8_ops.py

It needs no GPU, only DeepEP installed, which the CPU unit-test job does not
have.

``_get_save_ops`` names the optional ops as strings and resolves them with
``getattr``, swallowing an ``AttributeError`` so a build without DeepEP still
works. That makes a rename or a namespace change silent: the op simply stops
being found, SAC recomputes it, and an all-to-all runs twice per step -- slower,
and for a dispatch whose handle is consumed by its combine, wrong.

Importing the backends is what registers the ops, so this is also the test that
fails if they stop registering at import time, which is what
``DeepEPTokenDispatcher.__init__`` relies on to beat ``apply_ac``.
"""

import pytest
import torch

import torchtitan.ops.float8.quantize_expert_weights  # noqa: F401  registers the op
from torchtitan.distributed.activation_checkpoint import _get_save_ops

pytest.importorskip("deep_ep")

from torchtitan.distributed.deepep import deepep, hybridep  # noqa: E402,F401


def test_the_ep_communication_is_saved_rather_than_repeated():
    save_ops = _get_save_ops()
    for namespace in ("deepep", "hybridep"):
        for name in ("dispatch", "combine"):
            op = getattr(getattr(torch.ops, namespace), name).default
            assert op in save_ops, f"{namespace}.{name} is not in the SAC save list"


def test_the_expert_weight_quantization_is_saved():
    """Reached on the DeepEP path, where the experts quantize their own
    weights; HybridEP quantizes them inside its dispatch op instead."""
    assert torch.ops.torchtitan.fp8_blockwise_quantize.default in _get_save_ops()
