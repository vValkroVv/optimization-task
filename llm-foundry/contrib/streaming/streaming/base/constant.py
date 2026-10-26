# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os

"""Constants."""

# Shared Memory names
SHM_PREFIX = os.environ.get("STREAMING_LOCALS", "")
def format_with_shm_prefix(s:str):
    return f"{SHM_PREFIX}{s}"

LOCALS = format_with_shm_prefix("locals")
BARRIER = format_with_shm_prefix('barrier')
NEXT_EPOCH = format_with_shm_prefix('next_epoch')
CACHE_USAGE = format_with_shm_prefix('cache_usage')
SHARD_STATES = format_with_shm_prefix('shard_states')
SHARD_ACCESS_TIMES = format_with_shm_prefix('shard_access_times')
RESUME = format_with_shm_prefix('resume')
EPOCH_SHAPE = format_with_shm_prefix('epoch_shape')
EPOCH_DATA = format_with_shm_prefix('epoch_data')
SHM_TO_CLEAN = [
    LOCALS,
    BARRIER,
    NEXT_EPOCH,
    CACHE_USAGE,
    SHARD_STATES,
    SHARD_ACCESS_TIMES,
    RESUME,
    EPOCH_SHAPE,
    EPOCH_DATA,
]

# filelock names
BARRIER_FILELOCK = 'barrier_filelock'
CACHE_FILELOCK = 'cache_filelock'

# Time to wait, in seconds.
TICK = 0.007
