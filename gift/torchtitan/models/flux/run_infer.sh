#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Sample images from a trained FLUX checkpoint. Takes the experiment YAML the
# run was trained with, as a positional argument or CONFIG_FILE, plus dot-path
# overrides:
#
#   CONFIG_FILE=configs/experiments/flux_debug.yaml NGPU=4 ./torchtitan/models/flux/run_infer.sh
#   NGPU=8 ./torchtitan/models/flux/run_infer.sh configs/experiments/flux_debug.yaml --checkpoint.load_step 10
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
    CONFIG_FILE="$1"
    shift
fi
CONFIG_FILE=${CONFIG_FILE:-"configs/experiments/flux_debug.yaml"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.models.flux.inference.infer ${CONFIG_FILE} \
--checkpoint.enable \
--checkpoint.exclude_from_loading=lr_scheduler,dataloader,optimizer "$@"
