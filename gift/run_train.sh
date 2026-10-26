#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Train from an experiment YAML (CONFIG_FILE or the first positional argument)
# plus dot-path overrides:
#
#   CONFIG_FILE=configs/experiments/llama3_debug.yaml NGPU=1 ./run_train.sh
#   NGPU=8 ./run_train.sh configs/experiments/gift_3_5_debug.yaml --training.steps 100
#
# Validate before submitting (works without torch):
#
#   python scripts/config_check.py configs/experiments/llama3_debug.yaml
#
# COMM_MODE runs a single process without torchrun:
#   "fake_backend" - build the trainer for NGPU simulated ranks, skip training
#   "local_tensor" - simulate NGPU ranks on one GPU, skip training
#   Example: NGPU=32 COMM_MODE="fake_backend" ./run_train.sh configs/experiments/gift_3_5_700b.yaml

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
COMM_MODE=${COMM_MODE:-""}

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
    CONFIG_FILE="$1"
    shift
fi
CONFIG_FILE=${CONFIG_FILE:-"configs/experiments/llama3_debug.yaml"}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train ${CONFIG_FILE} "$@" --comm.mode=${COMM_MODE} --training.steps 1
else
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train ${CONFIG_FILE} "$@"
fi
