#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Single-node gift_moe launcher: an experiment YAML (CONFIG_FILE or the first
# positional argument) plus dot-path overrides.
#   NGPU=2 CONFIG_FILE=configs/experiments/gift_moe_debug_deepep.yaml torchtitan/models/gift_moe/train.sh
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
    CONFIG_FILE="$1"
    shift
fi
CONFIG_FILE="${CONFIG_FILE:-configs/experiments/gift_moe_debug.yaml}"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then B=$'\e[1m'; C=$'\e[36m'; R=$'\e[0m'; else B=; C=; R=; fi
printf '%s==>%s %sgift_moe%s  config=%s  ngpu=%s\n' "$C$B" "$R$B" "$B" "$R" "$CONFIG_FILE" "$NGPU"

CONFIG_FILE="$CONFIG_FILE" NGPU="$NGPU" ./run_train.sh "$@"
