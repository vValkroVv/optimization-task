#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./run_env.sh "${1:?physical GPU}"
chunks=${2:?chunks}
method=${3:?method}
series=${4:?series}
shift 4
mkdir "$series"
trap 'echo "$?" > "$series/exit-code"' EXIT
extra=()
if [[ ${FUSED_CE_ALIAS_GATHER:-0} == 1 ]]; then
  extra+=(--alias-gather)
fi
i=0
for arm in "$@"; do
  echo "GPU $gpu C$chunks $method $arm"
  timeout 250 "$python" -Bu integrated.py "$arm" "$chunks" "$series/$i-$arm.json" --method "$method" "${extra[@]}" > "$series/$i-$arm.log" 2>&1
  ((i+=1))
done
