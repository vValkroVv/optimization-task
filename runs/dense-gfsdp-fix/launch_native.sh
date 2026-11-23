#!/usr/bin/env bash
set -eEo pipefail
home_dir=/workspace-SR008.nfs2/vkropotin
candidate="$home_dir/dg-fix"
original="$home_dir/dg-a"
source /etc/profile.d/wm.sh
set -u
py=/opt/torchtitan-venv/bin/python3.12
owner="$candidate/runs/dense-gfsdp-fix"
reviewed="$owner/artifacts/reviewed-port-001"
campaign="$owner/artifacts/raw/native-001"
mode="${1:?Provide a launch mode}"
shift
receipt="$campaign/$mode.exit"
cd "$candidate"
trap 'status=$?; printf "%s\n" "$status" > "$receipt"; exit "$status"' EXIT
source_hash="$("$py" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["tracked_and_untracked_sha256"])' "$reviewed/source.json")"
pins=(
    --original-root "$original"
    --candidate-root "$candidate"
    --reference "$home_dir/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2"
    --candidate-head e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
    --candidate-source-sha256 "$source_hash"
)
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
case "$mode" in
  pilot)
    "$py" -Bu "$owner/campaign.py" preflight 8 a "${pins[@]}"
    "$py" -Bu "$owner/campaign.py" preflight 4 all "${pins[@]}"
    "$py" -Bu "$owner/campaign.py" matrix "$campaign" --suite per-fix --case ep8-a "${pins[@]}"
    ;;
  per-fix|final)
    "$py" -Bu "$owner/campaign.py" matrix "$campaign" --suite "$mode" "${pins[@]}" "$@"
    ;;
  trace)
    "$py" -Bu "$owner/diagnostic.py" "$campaign" "$@" "${pins[@]}"
    ;;
  *)
    exit 2
    ;;
esac
