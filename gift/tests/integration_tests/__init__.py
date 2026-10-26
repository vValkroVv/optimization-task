# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "OverrideDefinitions",
]


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    env: Mapping[str, str] = field(default_factory=dict)
    """Environment for this row's training run, on top of CONFIG_FILE /
    NGPU. For a knob that is not part of the trainer config -- a kernel cache
    policy, say -- and that only some rows need."""
    disabled: bool = False
    skip_rocm_test: bool = False
    timeout: int | None = None
    test_module: str | None = None
    """Optional torchrun module for distributed assertions instead of training."""

    def __repr__(self):
        return self.test_descr
