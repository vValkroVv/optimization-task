#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dedicated training script for graph_trainer with CooR precompile support.
# Passes --virtual-local-rank to torchrun so that every worker sees
# LOCAL_RANK=0 and uses cuda:0. torchrun isolates each worker's GPU via
# CUDA_VISIBLE_DEVICES, so cuda:0 maps to a different physical GPU per
# worker. This is required when loading a CooR-precompiled artifact,
# because the artifact was compiled on a single process targeting cuda:0.

set -ex

#
# Takes the experiment YAML (CONFIG_FILE or the first positional argument) plus
# dot-path overrides, like run_train.sh:
#   NGPU=8 CONFIG_FILE=configs/experiments/graph_llama3_debug.yaml \
#       ./torchtitan/experiments/graph_trainer/run_train_precompile.sh \
#       --compile.precompile_artifact_dir /tmp/precompile_artifacts

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
    CONFIG_FILE="$1"
    shift
fi
CONFIG_FILE=${CONFIG_FILE:-"configs/experiments/graph_llama3_debug.yaml"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--virtual-local-rank \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.train ${CONFIG_FILE} "$@"
