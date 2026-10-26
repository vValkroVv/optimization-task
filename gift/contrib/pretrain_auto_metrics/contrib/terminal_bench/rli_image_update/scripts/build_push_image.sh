#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 TASK_ID [DATASET_ROOT]" >&2
  exit 2
fi

TASK_ID="$1"
DATASET_ROOT="${2:-${TERMINAL_BENCH_DATASET_ROOT:-/home/user/terminal-bench-2}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UPDATE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$UPDATE_ROOT/build/$TASK_ID"
REPORT_DIR="$UPDATE_ROOT/outputs/validation/$TASK_ID"
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
    + "\n",
    encoding="utf-8",
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

python3 "$SCRIPT_DIR/image_builder.py" \
  "$TASK_ID" \
  --dataset-root "$DATASET_ROOT"

IMAGE="$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["image"])' \
    "$BUILD_DIR/build-info.json"
)"

echo "TASK_ID=$TASK_ID"
echo "IMAGE=$IMAGE"
echo "BUILD_DIR=$BUILD_DIR"

STAGE="build"
for attempt in {1..5}; do
  if docker build --pull --platform linux/amd64 -t "$IMAGE" "$BUILD_DIR" \
    >"$REPORT_DIR/build-stdout.log" \
    2>"$REPORT_DIR/build-stderr.log"; then
    break
  fi
  if [ "$attempt" -eq 5 ] || \
    ! grep -Eqi \
      '429 Too Many Requests|TOOMANYREQUESTS|Client.Timeout|context deadline exceeded|i/o timeout|TLS handshake timeout|connection reset|unexpected EOF|503 Service Unavailable' \
      "$REPORT_DIR/build-stderr.log"; then
    exit 1
  fi
  cat "$REPORT_DIR/build-stderr.log" >&2
  sleep $((attempt * 30))
done

STAGE="offline_validation"
"$SCRIPT_DIR/validate_offline_task.sh" "$TASK_ID" "$IMAGE" "$DATASET_ROOT"

STAGE="push"
for attempt in {1..5}; do
  if docker push "$IMAGE" \
    >"$REPORT_DIR/push-stdout.log" \
    2>"$REPORT_DIR/push-stderr.log"; then
    break
  fi
  cat "$REPORT_DIR/push-stderr.log" >&2
  [ "$attempt" -lt 5 ] || exit 1
  sleep $((attempt * 30))
done

STAGE="inspect"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}' \
  | tee "$REPORT_DIR/image-digests.json"

echo "pushed $IMAGE"
trap - EXIT
