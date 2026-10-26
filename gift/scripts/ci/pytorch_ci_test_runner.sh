#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stable entrypoint for PyTorch CI to run torchtitan integration tests.
# PyTorch CI calls this script so that torchtitan maintainers can adjust
# test configuration without modifying the PyTorch repo.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-artifacts-to-be-uploaded}"
NGPU="${NGPU:-8}"
PYTHON="${PYTHON:-python3}"

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  smoke           Run seed_checkpoint smoke test"
    echo "  feature_tests   Run feature integration tests"
    echo "  model_tests     Run model integration tests"
    echo "  h100            Run H100 integration tests"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    smoke)
        "$PYTHON" -m tests.integration_tests.run_tests \
            "$OUTPUT_DIR" \
            --test_suite features \
            --test_name seed_checkpoint \
            --ngpu "$NGPU" \
            --no-parallel
        ;;
    feature_tests)
        "$PYTHON" -m tests.integration_tests.run_tests \
            --test_suite features \
            --exclude "cpu_offload+opt_in_bwd+TP+DP+CP" \
            --ngpu "$NGPU" \
            "$OUTPUT_DIR"
        ;;
    model_tests)
        "$PYTHON" -m tests.integration_tests.run_tests \
            --test_suite models \
            --ngpu "$NGPU" \
            "$OUTPUT_DIR"
        ;;
    h100)
        "$PYTHON" -m tests.integration_tests.run_tests \
            "$OUTPUT_DIR" \
            --test_suite h100 \
            --ngpu "$NGPU"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        ;;
esac
