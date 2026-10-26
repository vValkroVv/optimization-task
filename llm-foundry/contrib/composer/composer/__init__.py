# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Some key classes are available directly in the ``composer`` namespace."""

from composer._version import __version__
from composer.core import Algorithm, Callback, DataSpec, Engine, Evaluator, Event, State, Time, Timestamp, TimeUnit
from composer.loggers import Logger
from composer.models import ComposerModel
from composer.trainer import Trainer
from composer.utils.misc import get_torch_version
from packaging import version

supported_torch_version = ["2.5.1", "2.5", "2.4", "2.6", "2.8"]
if get_torch_version() not in [version.parse(v) for v in supported_torch_version]:
    raise ImportError(f"Torch version {get_torch_version()} not supported. Please run one of {supported_torch_version}")

__all__ = [
    'Algorithm',
    'Callback',
    'DataSpec',
    'Engine',
    'Evaluator',
    'Event',
    'State',
    'Time',
    'Timestamp',
    'TimeUnit',
    'Logger',
    'ComposerModel',
    'Trainer',
]
