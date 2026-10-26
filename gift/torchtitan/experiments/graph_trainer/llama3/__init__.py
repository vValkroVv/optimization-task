# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .parallelize import parallelize_llama


def _parallelize_fn(model, *, compile_config, **kwargs):
    if compile_config.enable_autoparallel:
        from .parallelize_autoparallel import parallelize_autoparallel_llama

        return parallelize_autoparallel_llama(
            model, compile_config=compile_config, **kwargs
        )
    return parallelize_llama(model, compile_config=compile_config, **kwargs)
