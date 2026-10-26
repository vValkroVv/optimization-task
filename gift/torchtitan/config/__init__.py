# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration: the shared config dataclasses and the YAML experiment loader.

See docs/config.md. Importing this package must stay torch-free (pre-submit
runs in a minimal environment), so :mod:`.build` is never imported here.
"""

from .configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    FP8Config,
    ParallelismConfig,
    TrainingConfig,
)
from .configurable import Configurable
from .errors import ConfigError, ConfigFileError, ConfigSchemaError
from .function import Function
from .loader import deep_merge, ImportNode, load_raw_document
from .overrides import parse_argv
from .cli import load_config, load_document
from .pre_submit import PreSubmitResult, run_pre_submit
from .sections import CONFIG_VERSION, ModelSection
from .validation import (
    RuntimeInfo,
    ValidationContext,
    ValidationReport,
)

__all__ = [
    # shared config dataclasses
    "Configurable",
    "load_config",
    "Function",
    "ActivationCheckpointConfig",
    "CompileConfig",
    "ParallelismConfig",
    "CommConfig",
    "TrainingConfig",
    "DebugConfig",
    "FP8Config",
    # YAML experiment configs
    "CONFIG_VERSION",
    "ConfigError",
    "ConfigFileError",
    "ConfigSchemaError",
    "ImportNode",
    "ModelSection",
    "PreSubmitResult",
    "RuntimeInfo",
    "ValidationContext",
    "ValidationReport",
    "deep_merge",
    "load_document",
    "load_raw_document",
    "parse_argv",
    "run_pre_submit",
]
