#!/bin/bash
set -eu
# The parent may interrupt this bounded, read-only sampler after the last capture.
dst=$(cd "$(dirname "$0")" && pwd)
name="gpu-$(date -u +%Y%m%dT%H%M%SZ).csv"
exec env TZ=UTC timeout 40m nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown --format=csv,nounits -lms 500 > "$dst/$name"
