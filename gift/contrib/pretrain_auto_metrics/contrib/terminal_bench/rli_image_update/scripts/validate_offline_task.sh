#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 TASK_ID IMAGE [DATASET_ROOT]" >&2
  exit 2
fi

TASK_ID="$1"
IMAGE="$2"
DATASET_ROOT="${3:-${TERMINAL_BENCH_DATASET_ROOT:-/home/user/terminal-bench-2}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UPDATE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$UPDATE_ROOT/build/$TASK_ID"
REPORT_DIR="$UPDATE_ROOT/outputs/validation/$TASK_ID"
CONTAINER_ID=""
REMOTE_TESTS_DIR="/tmp/terminal-bench/tests"
REMOTE_LOGS_DIR="/tmp/terminal-bench/logs"

mkdir -p "$REPORT_DIR"
rm -f \
  "$REPORT_DIR/offline-report.json" \
  "$REPORT_DIR/solution-stdout.log" \
  "$REPORT_DIR/solution-stderr.log" \
  "$REPORT_DIR/test-stdout.log" \
  "$REPORT_DIR/test-stderr.log"

if [ ! -d "$BUILD_DIR/task/tests" ] || [ ! -d "$BUILD_DIR/task/solution" ]; then
  echo "missing generated task context: $BUILD_DIR/task" >&2
  exit 1
fi

cleanup() {
  if [ -n "$CONTAINER_ID" ]; then
    docker rm -f "$CONTAINER_ID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

CONTAINER_ID="$(
  docker create --entrypoint /bin/sh "$IMAGE" -c \
    'while :; do sleep 3600; done'
)"
docker start "$CONTAINER_ID" >/dev/null

WORKDIR="$(docker inspect --format '{{.Config.WorkingDir}}' "$CONTAINER_ID")"
WORKDIR="${WORKDIR:-/}"

docker exec "$CONTAINER_ID" tmux -V >/dev/null
docker exec "$CONTAINER_ID" test -r \
  /usr/local/share/.dev_logs/test_runtime.tar.zst
docker exec "$CONTAINER_ID" sh -c 'test -z "$(dpkg --audit)"'

docker exec "$CONTAINER_ID" mkdir -p /solution
docker cp "$BUILD_DIR/task/solution/." \
  "$CONTAINER_ID:/solution"

set +e
docker exec --workdir "$WORKDIR" "$CONTAINER_ID" \
  bash /solution/solve.sh \
  >"$REPORT_DIR/solution-stdout.log" \
  2>"$REPORT_DIR/solution-stderr.log"
SOLUTION_RC=$?
set -e
if [ "$SOLUTION_RC" -ne 0 ]; then
  echo "oracle solution failed for $TASK_ID; see $REPORT_DIR" >&2
  exit "$SOLUTION_RC"
fi

docker exec "$CONTAINER_ID" rm -rf /solution

mapfile -t NETWORKS < <(
  docker inspect \
    --format '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}}{{"\n"}}{{end}}' \
    "$CONTAINER_ID"
)
for network in "${NETWORKS[@]}"; do
  [ -n "$network" ] || continue
  docker network disconnect "$network" "$CONTAINER_ID"
done

NETWORK_COUNT="$(
  docker inspect --format '{{len .NetworkSettings.Networks}}' "$CONTAINER_ID"
)"
if [ "$NETWORK_COUNT" != 0 ]; then
  echo "failed to isolate verifier network for $TASK_ID" >&2
  exit 1
fi

docker exec "$CONTAINER_ID" \
  rm -rf /var/lib/terminal-bench/test_runtime \
    "$REMOTE_TESTS_DIR" "$REMOTE_LOGS_DIR/verifier"
docker exec "$CONTAINER_ID" \
  mkdir -p "$REMOTE_TESTS_DIR" "$REMOTE_LOGS_DIR/verifier"
docker cp "$BUILD_DIR/task/tests/." "$CONTAINER_ID:$REMOTE_TESTS_DIR"

set +e
docker exec --workdir "$WORKDIR" "$CONTAINER_ID" \
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
    UV_OFFLINE=1 PIP_NO_INDEX=1 \
    bash "$REMOTE_TESTS_DIR/test.sh" \
  >"$REPORT_DIR/test-stdout.log" \
  2>"$REPORT_DIR/test-stderr.log"
TEST_RC=$?
set -e

if docker exec "$CONTAINER_ID" test -d "$REMOTE_LOGS_DIR/verifier"; then
  docker cp "$CONTAINER_ID:$REMOTE_LOGS_DIR/verifier/." \
    "$REPORT_DIR/verifier" || true
fi

REWARD="$(
  docker exec "$CONTAINER_ID" sh -c \
    "test -r $REMOTE_LOGS_DIR/verifier/reward.txt && \
      tr -d '[:space:]' <$REMOTE_LOGS_DIR/verifier/reward.txt" \
    2>/dev/null || true
)"

python3 - "$REPORT_DIR/offline-report.json" \
  "$TASK_ID" "$IMAGE" "$SOLUTION_RC" "$TEST_RC" "$REWARD" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "task_id": sys.argv[2],
            "image": sys.argv[3],
            "solution_network": "default",
            "verifier_network": "none",
            "uv_offline": True,
            "pip_no_index": True,
            "solution_returncode": int(sys.argv[4]),
            "test_returncode": int(sys.argv[5]),
            "reward": float(sys.argv[6]) if sys.argv[6] else None,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

if [ "$TEST_RC" -ne 0 ] || [ "$REWARD" != 1 ]; then
  echo "offline verifier failed for $TASK_ID; see $REPORT_DIR" >&2
  exit 1
fi

echo "offline verifier passed for $TASK_ID: reward=$REWARD"
