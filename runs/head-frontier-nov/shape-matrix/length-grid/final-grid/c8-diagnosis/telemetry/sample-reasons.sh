#!/bin/bash
set -eu
dst=$(cd "$(dirname "$0")" && pwd)
name="reasons-$(date -u +%Y%m%dT%H%M%SZ).csv"
exec env TZ=UTC timeout 30m nvidia-smi -i 0,6 --query-gpu=timestamp,index,clocks.sm,temperature.gpu,temperature.memory,power.draw,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.sw_power_cap --format=csv,nounits -lms 150 > "$dst/$name"
