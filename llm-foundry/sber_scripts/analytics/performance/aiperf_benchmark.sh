#!/usr/bin/env bash
set -euo pipefail

# Defaults keep the previous ad-hoc benchmark values, while allowing the
# end-to-end wrapper to control every aiperf parameter through env vars.
IP=${AIPERF_IP:-10.225.134.128}
PORT=${AIPERF_PORT:-30000}
URL=${AIPERF_URL:-http://${IP}:${PORT}}
MODEL=${AIPERF_MODEL:-default}
TOKENIZER=${AIPERF_TOKENIZER:-/workspace-SR008.nfs2/nvokhotnikov/model_save_folder/120B_3.5/random_hf_checkpoint_uniform_fp8/}
ENDPOINT_TYPE=${AIPERF_ENDPOINT_TYPE:-completions}
INPUT_LEN=${AIPERF_INPUT_LEN:-1024}
OUTPUT_LEN=${AIPERF_OUTPUT_LEN:-8192}
INPUT_STDDEV=${AIPERF_INPUT_STDDEV:-0}
CONCURRENCY=${AIPERF_CONCURRENCY:-64}
REQUESTS=${AIPERF_REQUESTS:-128}
REQUEST_TIMEOUT_SECONDS=${AIPERF_REQUEST_TIMEOUT_SECONDS:-}
STREAMING=${AIPERF_STREAMING:-True}
OUTPUT_DIR=${AIPERF_OUTPUT_DIR:-/workspace-SR008.nfs2/nvokhotnikov/llm-foundry/sber_benchmarks/sglang_benchmark_results/GIGA_ULTRA_COMPARISON/430B_3.5_ZC_norm_impl_compile/aiperf_${INPUT_LEN}_${OUTPUT_LEN}_concurrency_${CONCURRENCY}}
EXTRA_INPUTS=${AIPERF_EXTRA_INPUTS:-max_tokens:${OUTPUT_LEN};min_tokens:${OUTPUT_LEN};ignore_eos:true}
AIPERF_BIN=${AIPERF_BIN:-}
AIPERF_PYTHON=${AIPERF_PYTHON:-}
AIPERF_VENV=${AIPERF_VENV:-}

if [ -n "${AIPERF_VENV}" ] && [ -f "${AIPERF_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${AIPERF_VENV}/bin/activate"
fi

if [ -z "${AIPERF_BIN}" ] && [ -z "${AIPERF_PYTHON}" ]; then
  if command -v aiperf >/dev/null 2>&1; then
    AIPERF_BIN=$(command -v aiperf)
  elif [ -x "/workspace-SR008.nfs2/nvokhotnikov/venv/bin/aiperf" ]; then
    AIPERF_BIN="/workspace-SR008.nfs2/nvokhotnikov/venv/bin/aiperf"
  fi
fi

if [ -z "${AIPERF_BIN}" ] && [ -z "${AIPERF_PYTHON}" ]; then
  echo "ERROR: aiperf is not available; set AIPERF_BIN, AIPERF_PYTHON, or AIPERF_VENV" >&2
  exit 127
fi

mkdir -p "${OUTPUT_DIR}"

echo "Benchmarking raw completions against ${URL}..."

if [ -n "${AIPERF_PYTHON}" ]; then
  cmd=("${AIPERF_PYTHON}" -m aiperf profile)
else
  cmd=("${AIPERF_BIN}" profile)
fi

cmd+=(
  --model "${MODEL}"
  --tokenizer "${TOKENIZER}"
  --url "${URL}"
  --endpoint-type "${ENDPOINT_TYPE}"
  --synthetic-input-tokens-mean "${INPUT_LEN}"
  --synthetic-input-tokens-stddev "${INPUT_STDDEV}"
  --output-tokens-mean "${OUTPUT_LEN}"
  --concurrency "${CONCURRENCY}"
  --request-count "${REQUESTS}"
  --artifact-dir "${OUTPUT_DIR}"
)

if [ "${STREAMING}" = "True" ] || [ "${STREAMING}" = "true" ] || [ "${STREAMING}" = "1" ]; then
  cmd+=(--streaming)
fi
if [ -n "${REQUEST_TIMEOUT_SECONDS}" ] && [ "${REQUEST_TIMEOUT_SECONDS}" != "0" ]; then
  cmd+=(--request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}")
fi

IFS=';' read -r -a extra_inputs_array <<< "${EXTRA_INPUTS}"
for extra_input in "${extra_inputs_array[@]}"; do
  if [ -n "${extra_input}" ]; then
    cmd+=(--extra-inputs "${extra_input}")
  fi
done

printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"
