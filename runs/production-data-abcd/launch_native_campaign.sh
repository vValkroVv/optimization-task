#!/usr/bin/env bash
set -Eeuo pipefail
source /etc/profile.d/wm.sh
exec /opt/torchtitan-venv/bin/python3.12 \
  /workspace-SR008.nfs2/vkropotin/optimization-task/runs/production-data-abcd/execute_native_campaign.py \
  "$@"
