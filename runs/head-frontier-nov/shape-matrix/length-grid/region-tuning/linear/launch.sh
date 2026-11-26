#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
nohup bash run.sh 3 700b:262144:projection 100b:262144:dhidden 700b:32768:projection \
    100b:32768:projection 100b:65536:dhidden </dev/null > gpu-3.log 2>&1 &
nohup bash run.sh 4 700b:262144:dhidden 100b:262144:projection 700b:65536:projection \
    100b:65536:projection </dev/null > gpu-4.log 2>&1 &
nohup bash run.sh 5 700b:131072:projection 700b:131072:dhidden 700b:65536:dhidden \
    100b:131072:projection 100b:131072:dhidden 700b:32768:dhidden </dev/null > gpu-5.log 2>&1 &
