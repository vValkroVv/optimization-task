#!/usr/bin/env bash
set -eo pipefail
source /etc/profile.d/wm.sh
set -u
cd /workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/gather-repeat
exec /opt/torchtitan-venv/bin/python3.12 -Bu campaign8.py "$@"
