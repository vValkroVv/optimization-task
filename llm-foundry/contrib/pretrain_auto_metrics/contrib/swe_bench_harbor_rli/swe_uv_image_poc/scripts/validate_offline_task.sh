#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 TASK_ID IMAGE [DATASET_ROOT]" >&2
  exit 2
fi

TASK_ID="$1"
IMAGE="$2"
DATASET_ROOT="${3:-${SWE_BENCH_DATASET_ROOT:-/home/user/swe-bench-verified/datasets/swebench-verified}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${SWE_IMAGE_BUILD_DIR:-$POC_ROOT/build/$TASK_ID}"
REPORT_DIR="${SWE_IMAGE_REPORT_DIR:-$POC_ROOT/outputs/validation/$TASK_ID}"

if [ ! -d "$BUILD_DIR/task/tests" ] || [ ! -d "$BUILD_DIR/task/solution" ]; then
  echo "missing generated task context: $BUILD_DIR/task" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"
rm -f \
  "$REPORT_DIR"/offline-report.json \
  "$REPORT_DIR"/install-stdout.log \
  "$REPORT_DIR"/install-stderr.log \
  "$REPORT_DIR"/install-docker-stderr.log \
  "$REPORT_DIR"/test-stdout.log \
  "$REPORT_DIR"/docker-stderr.log

install_container_id="$(
  docker create --network none \
  --env PIP_NO_INDEX=1 --env UV_OFFLINE=1 \
  --env SWEBENCH_INSTALL_ONLY=1 \
  --env NO_COLOR=1 --env PY_COLORS=0 --env TERM=dumb \
  --entrypoint /bin/bash "$IMAGE" -lc '
    set -eu
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
    unset http_proxy https_proxy all_proxy no_proxy
    mkdir -p /tests /logs/verifier /validation-artifacts
    cp /opt/swebench-task/tests/config.json /tests/config.json
    bash /opt/swebench-task/tests/test.sh \
      > /validation-artifacts/install-stdout.log \
      2> /validation-artifacts/install-stderr.log
  '
)"

test_container_id=""
test_network_name=""

cleanup_containers() {
  if [ -n "$install_container_id" ]; then
    docker rm -f "$install_container_id" >/dev/null 2>&1 || true
  fi
  if [ -n "$test_container_id" ]; then
    docker rm -f "$test_container_id" >/dev/null 2>&1 || true
  fi
}
trap cleanup_containers EXIT

docker cp "$BUILD_DIR/task" "$install_container_id:/opt/swebench-task"

set +e
docker start --attach "$install_container_id" 2>"$REPORT_DIR/install-docker-stderr.log"
install_rc=$?
set -e

docker cp "$install_container_id:/validation-artifacts/." "$REPORT_DIR"
docker rm -f "$install_container_id" >/dev/null 2>&1 || true
install_container_id=""

if [ "$install_rc" -ne 0 ]; then
  echo "offline installation validation failed for $TASK_ID; see $REPORT_DIR" >&2
  exit "$install_rc"
fi

test_network_name="swe-offline-validation"
if ! docker network inspect "$test_network_name" >/dev/null 2>&1; then
  docker network create \
    --driver bridge \
    --internal \
    --subnet 10.255.255.0/24 \
    --gateway 10.255.255.254 \
    --aux-address timeout-target=10.255.255.1 \
    "$test_network_name" >/dev/null 2>&1 \
    || docker network inspect "$test_network_name" >/dev/null
fi

test_container_id="$(
  docker create --network "$test_network_name" \
  --env PIP_NO_INDEX=1 --env UV_OFFLINE=1 \
  --env NO_COLOR=1 --env PY_COLORS=0 --env TERM=dumb \
  --env VALIDATION_TASK_ID="$TASK_ID" --env VALIDATION_IMAGE="$IMAGE" \
  --entrypoint /bin/bash "$IMAGE" -lc '
    set -eu
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
    unset http_proxy https_proxy all_proxy no_proxy
    mkdir -p /tests /logs/verifier /validation-artifacts
    cp /opt/swebench-task/tests/config.json /tests/config.json
    bash /opt/swebench-task/solution/solve.sh
    set +e
    bash /opt/swebench-task/tests/test.sh > /validation-artifacts/test-stdout.log 2>&1
    test_rc=$?
    set -e
    reward=""
    resolved=false
    if [ -r /logs/verifier/reward.txt ]; then
      reward="$(tr -d "[:space:]" < /logs/verifier/reward.txt)"
    fi
    if [ -r /logs/verifier/report.json ]; then
      resolved="$(python3 -c "import json; print(str(json.load(open(\"/logs/verifier/report.json\"))[\"$VALIDATION_TASK_ID\"][\"resolved\"]).lower())")"
      cp /logs/verifier/report.json /validation-artifacts/report.json
    fi
    python3 - "$test_rc" "$reward" "$resolved" <<"PY"
import json
import os
import sys
from pathlib import Path

proxy_names = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)

Path("/validation-artifacts/offline-report.json").write_text(
    json.dumps(
        {
            "task_id": os.environ["VALIDATION_TASK_ID"],
            "image": os.environ["VALIDATION_IMAGE"],
            "install_network": "none",
            "test_network": "internal",
            "proxy_variables_absent": all(
                name not in os.environ for name in proxy_names
            ),
            "pip_no_index": True,
            "uv_offline": True,
            "install_returncode": 0,
            "test_returncode": int(sys.argv[1]),
            "reward": float(sys.argv[2]) if sys.argv[2] else None,
            "resolved": sys.argv[3] == "true",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
    if [ "$test_rc" -ne 0 ] || [ "$reward" != 1 ] || [ "$resolved" != true ]; then
      exit 1
    fi
  '
)"

docker cp "$BUILD_DIR/task" "$test_container_id:/opt/swebench-task"

set +e
docker start --attach "$test_container_id" 2>"$REPORT_DIR/docker-stderr.log"
docker_rc=$?
set -e

docker cp "$test_container_id:/validation-artifacts/." "$REPORT_DIR"
cleanup_containers
trap - EXIT

if [ "$docker_rc" -ne 0 ]; then
  echo "offline validation failed for $TASK_ID; see $REPORT_DIR" >&2
  exit "$docker_rc"
fi

python3 - "$REPORT_DIR/offline-report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("install_returncode") != 0
    or report.get("install_network") != "none"
    or report.get("test_network") != "internal"
    or report.get("reward") != 1
    or report.get("resolved") is not True
    or report.get("proxy_variables_absent") is not True
):
    raise SystemExit(f"incomplete offline report: {report!r}")
print(json.dumps(report, ensure_ascii=False, sort_keys=True))
PY
