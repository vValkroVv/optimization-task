#!/usr/bin/env bash
set -eo pipefail

checkout=$1
campaign=$2
candidate=$3
launcher=$4
shift 4
trap 'printf "%s\n" "$?" > "$launcher/exit"' EXIT
source /etc/profile.d/wm.sh
set -u

cd "$checkout"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
unset GIFT_CUTE_NORM_BWD_SKIP GIFT_FP8_WGRAD_LOGICAL_WIDTH GIFT_MOE_SWIGLU_BWD_MN_MAJOR_SCALE
unset GIFT_ROUTED_DW_BF16_OVERWRITE GIFT_CUTE_NORM_PAIR
unset GIFT_ROUTED_SWIGLU_FWD_FEATURE_FIRST GIFT_FP8_DENSE_DY_QKV_FUSION
sha256sum runs/shk-fix/acceptance.py > "$launcher/runner.sha256"
python=/opt/torchtitan-venv/bin/python3.12
options=("$campaign" --window-size 10)
if [[ "$candidate" == --final-acceptance ]]; then
    options+=(--final-acceptance)
else
    options+=(--followup-switch "$candidate")
fi
options+=("$@")
"$python" -B -u runs/shk-fix/acceptance.py preflight "${options[@]}"
"$python" -B -u runs/shk-fix/acceptance.py run "${options[@]}"
