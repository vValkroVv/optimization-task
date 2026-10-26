#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 TASK_ID [DATASET_ROOT]" >&2
  exit 2
fi

TASK_ID="$1"
DATASET_ROOT="${2:-${SWE_BENCH_DATASET_ROOT:-/home/user/swe-bench-verified/datasets/swebench-verified}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_ROOT="$POC_ROOT/build/1.0.2"
REPORT_DIR="$POC_ROOT/outputs/validation-1.0.2/$TASK_ID"
IMAGE=""
STAGE="prepare"

mkdir -p "$REPORT_DIR"
rm -f "$REPORT_DIR/failure-report.json"

write_failure_report() {
  python3 - "$REPORT_DIR/failure-report.json" \
    "$TASK_ID" "$IMAGE" "$STAGE" "$1" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "task_id": sys.argv[2],
            "image": sys.argv[3] or None,
            "stage": sys.argv[4],
            "exit_code": int(sys.argv[5]),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
}

on_exit() {
  status=$?
  trap - EXIT
  if [ "$status" -ne 0 ]; then
    write_failure_report "$status" || true
  fi
  exit "$status"
}
trap on_exit EXIT

python3 "$SCRIPT_DIR/image_builder_1_0_2.py" \
  "$TASK_ID" \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$BUILD_ROOT"

BUILD_DIR="$BUILD_ROOT/$TASK_ID"
IMAGE="$(
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["image"])' \
    "$BUILD_DIR/build-info.json"
)"

echo "TASK_ID=$TASK_ID"
echo "IMAGE=$IMAGE"
echo "BUILD_DIR=$BUILD_DIR"

STAGE="build"
docker build --pull --platform linux/amd64 -t "$IMAGE" "$BUILD_DIR" \
  >"$REPORT_DIR/build-stdout.log" 2>"$REPORT_DIR/build-stderr.log"

STAGE="offline_validation"
SWE_IMAGE_BUILD_DIR="$BUILD_DIR" \
SWE_IMAGE_REPORT_DIR="$REPORT_DIR" \
  "$SCRIPT_DIR/validate_offline_task.sh" "$TASK_ID" "$IMAGE" "$DATASET_ROOT"

STAGE="harness_validation"
docker run --rm --network none "$IMAGE" bash -lc '
  set -e
  tmux -V
  . /root/.local/bin/env
  mini-swe-agent --help >/dev/null
  /opt/openhands-sdk-venv/bin/python -c "import fastapi, openhands.sdk"
  /opt/sweagent-venv/bin/pip show sweagent
  sweagent --help >/dev/null
' >"$REPORT_DIR/harness-stdout.log" 2>"$REPORT_DIR/harness-stderr.log"

STAGE="push"
docker push "$IMAGE" \
  >"$REPORT_DIR/push-stdout.log" 2>"$REPORT_DIR/push-stderr.log"

STAGE="inspect"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}' \
  | tee "$REPORT_DIR/image-digests.json"
echo "pushed $IMAGE"
trap - EXIT
exit 0
