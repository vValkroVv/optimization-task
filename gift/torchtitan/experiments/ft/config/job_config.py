# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field

from torchtitan.config.schema.experiments.ft.config.job_config import FaultTolerance
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "FaultTolerance",
    "FaultTolerantModelSpec",
    "JobConfig",
]


@dataclass
class FaultTolerantModelSpec(ModelSpec):
    fragment_fn: Callable | None = None


@dataclass
class JobConfig:
    fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)
