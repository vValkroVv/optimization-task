#!/usr/bin/env bash
set -e
source /etc/profile.d/wm.sh
cd /workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix/runs/lm-head-mtp
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
out=artifacts/frontier-projection-001
for pair in native:01 lt:02 lt:03 native:04; do
    arm=${pair%:*}
    index=${pair#*:}
    flags=()
    if [[ "$index" == 01 || "$index" == 02 ]]; then
        flags+=(--numerics)
    fi
    /opt/torchtitan-venv/bin/python3.12 frontier_projection_replay.py "$arm" "$out/replay-$index-$arm.json" --screen "$out/screen.json" "${flags[@]}" > "$out/replay-$index-$arm.log" 2>&1
done
