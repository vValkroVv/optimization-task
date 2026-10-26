#!/usr/bin/env bash
set -euo pipefail

# llm-foundry config used to create/reuse the random HF checkpoint.
CONFIG_PATH=/home/jovyan/nvokhotnikov/model_save_folder/ultra_v5_crop_ba32k_PERF_BENCH_TEST/llm_foundry_config.yaml

# Run id under <config_dir>/artifacts/model_perf/<run_id>.
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Cloud job name prefix shown in job_submitter/GitLab logs.
EXP_NAME=perf_bench_test_ultra

# User name passed to job_submitter jobs.
USER_LOGIN=nvokhotnikov

# Optional artifacts root; empty means <config_dir>/artifacts.
ARTIFACTS_DIR=

# Python executable used locally to submit the inf_bench job.
PYTHON_BIN=python

# Python executable inside the inf_bench job image.
JOB_PYTHON=/home/user/sglang/bin/python

# Optional venv sourced inside the inf_bench job before running Python.
OUTER_JOB_VENV=

# inf_bench job queue/priority/image for init, quantization, aiperf and traces.
QUEUE_NAME=pretrain-ml
PRIORITY_CLASS=medium
BASE_IMAGE=cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_opensource:2026_06_12_commit8f7a8687

# Resources for the inf_bench preparation/benchmark job.
PREP_N_GPUS=1
PREP_N_WORKERS=1

# Recreate random HF checkpoint even if artifacts already contain one.
FORCE_INIT=1

# Run FP8 quantization; set false to benchmark the bf16 HF checkpoint directly.
QUANTIZE_FP8=true # true or false

# Recreate FP8 checkpoint even if artifacts already contain one.
FORCE_QUANT=1

# Validate existing checkpoints strictly; false means warn and recreate invalid checkpoints.
CHECKPOINT_STRICT=true

# Unsafe: allow force/removal of checkpoint dirs outside ARTIFACTS_DIR.
FORCE_REMOVE_OUTSIDE_ARTIFACTS=0

# Optional custom HF checkpoint dir; empty means <artifacts>/random_hf_checkpoint_uniform.
HF_CHECKPOINT_DIR=

# Random HF model init options.
INCLUDE_MTP=true
USE_CUSTOM_SGLANG=true
MODEL_TYPE=gigachat35

# only for opensource sglang
MODEL_TAG=

INIT_DEVICE=cuda
INIT_CUDA_VISIBLE_DEVICES=0
UNIFORM_MOE=false
EXPAND_CONTEXT_LEN_TO=262144

# Optional FP8 checkpoint dir; empty means <artifacts>/random_hf_checkpoint_uniform_fp8.
FP8_CHECKPOINT_DIR=

# FP8 quantization options.
QUANT_DEVICE=cuda:0
WEIGHT_BLOCK_SIZE="128 128"
ACTIVATION_SCHEME=dynamic # dynamic or static
# Quantizer default skip list without self_attn.gate_proj and shared_expert_sigmoid.
MODULES_TO_NOT_CONVERT="lm_head embed_tokens shared_head.head mlp.gate.weight eh_proj self_attn.A_log self_attn.conv1d self_attn.dt_bias self_attn.in_proj_ba self_attn.norm.weight gate_up_projection gate_down_projection"

# Resources for the nested custom-SGLang inference job.
INFERENCE_N_GPUS=8
INFERENCE_N_WORKERS=2
INFERENCE_QUEUE_NAME=
INFERENCE_PRIORITY_CLASS=
INFERENCE_BASE_IMAGE=

# Parallelism and serving ports for SGLang inference.
TP=16
EP=16
DP=
PP=
ROUTER_PORT=30000
SGLANG_COMMUNICATION_PORT=50000

# SGLang serving parameters
MAX_SEQ_LEN=262144
GATED_NORM_IMPL=torch_compile
MOE_RUNNER_BACKEND=triton_llmf
MEM_FRACTION_STATIC=0.8

# Tokenizer used by OBM proxy, aiperf and trace bench_serving.
TOKENIZER_PATH=/workspace-SR008.nfs2/nvokhotnikov/model_save_folder/0.6B_MoE/MoE_0.6B_64_exp_6exp_per_token_512exp_hidden_2shared_experts_qwen3_sonic_moe/artifacts/tokenizer

# Copy tokenizer into <artifacts>/tokenizer for stable remote-job access.
COPY_TOKENIZER_TO_ARTIFACTS=true

# How long the inf_bench job waits for the nested inference server to become ready.
SERVER_TIMEOUT_MINUTES=300
SERVER_POLL_SECONDS=15

# Keep nested inference job alive after benchmark completion/failure.
KEEP_INFERENCE_JOB=0

# Kill the nested inference job if the parent inf_bench job stops updating this
# heartbeat after the inference job has left the queue and started running.
INFERENCE_HEARTBEAT_INTERVAL_SECONDS=30
INFERENCE_HEARTBEAT_STALE_SECONDS=1800
INFERENCE_HEARTBEAT_POLL_SECONDS=30

# Monitor the inf_bench job until completion from this launcher process.
MONITOR=1
MONITOR_POLL_SECONDS=30

# Skip aiperf phase if set to 1/true.
SKIP_AIPERF=0

# aiperf executable environment; needed when aiperf is not in the job image PATH.
AIPERF_VENV=/workspace-SR008.nfs2/nvokhotnikov/venv
AIPERF_BIN=/workspace-SR008.nfs2/nvokhotnikov/venv/bin/aiperf
AIPERF_PYTHON=

# aiperf request shape and output location.
AIPERF_MODEL=default
AIPERF_ENDPOINT_TYPE=completions
AIPERF_STREAMING=true
AIPERF_INPUT_LEN=1024
AIPERF_OUTPUT_LEN=8192
AIPERF_INPUT_STDDEV=0
AIPERF_CONCURRENCY=64
AIPERF_REQUESTS=128
AIPERF_REQUEST_TIMEOUT_SECONDS=3600
AIPERF_TIMEOUT_SECONDS=4500
AIPERF_OUTPUT_DIR=
AIPERF_EXTRA_INPUTS=

# Collect SGLang profiler traces after aiperf if set to 1/true.
COLLECT_TRACES=1

# Trace bench_serving request shape and output location.
TRACE_INPUT_LEN=10240
TRACE_OUTPUT_LEN=1024
TRACE_NUM_PROMPTS=32
TRACE_MAX_CONCURRENCY=0
TRACE_RPS=8
TRACE_SAVE_DETAILS=true
TRACE_WARMUP_STEPS=5
TRACE_TIMEOUT_SECONDS=3600
TRACE_OUTPUT_DIR=

bool_enabled() {
  case "${1,,}" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

add_optional_arg() {
  local flag=$1
  local value=$2
  if [ -n "${value}" ]; then
    cmd+=("${flag}" "${value}")
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
read -r -a WEIGHT_BLOCK_SIZE_ARGS <<< "${WEIGHT_BLOCK_SIZE}"
read -r -a MODULES_TO_NOT_CONVERT_ARGS <<< "${MODULES_TO_NOT_CONVERT}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/model_perf_benchmark.py"
  --config "${CONFIG_PATH}"
  --run-id "${RUN_ID}"
  --user-login "${USER_LOGIN}"
  --exp-name "${EXP_NAME}"
  --job-python "${JOB_PYTHON}"
  --queue-name "${QUEUE_NAME}"
  --priority-class "${PRIORITY_CLASS}"
  --prep-n-gpus "${PREP_N_GPUS}"
  --prep-n-workers "${PREP_N_WORKERS}"
  --base-image "${BASE_IMAGE}"
  --include-mtp "${INCLUDE_MTP}"
  --use-custom-sglang "${USE_CUSTOM_SGLANG}"
  --model-type "${MODEL_TYPE}"
  --init-device "${INIT_DEVICE}"
  --init-cuda-visible-devices "${INIT_CUDA_VISIBLE_DEVICES}"
  --uniform-moe "${UNIFORM_MOE}"
  --expand-context-len-to "${EXPAND_CONTEXT_LEN_TO}"
  --checkpoint-strict "${CHECKPOINT_STRICT}"
  --quantize-fp8 "${QUANTIZE_FP8}"
  --quant-device "${QUANT_DEVICE}"
  --weight-block-size "${WEIGHT_BLOCK_SIZE_ARGS[@]}"
  --activation-scheme "${ACTIVATION_SCHEME}"
  --inference-n-gpus "${INFERENCE_N_GPUS}"
  --inference-n-workers "${INFERENCE_N_WORKERS}"
  --tp "${TP}"
  --ep "${EP}"
  --router-port "${ROUTER_PORT}"
  --sglang-communication-port "${SGLANG_COMMUNICATION_PORT}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --gated-norm-impl "${GATED_NORM_IMPL}"
  --moe-runner-backend "${MOE_RUNNER_BACKEND}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --copy-tokenizer-to-artifacts "${COPY_TOKENIZER_TO_ARTIFACTS}"
  --server-timeout-minutes "${SERVER_TIMEOUT_MINUTES}"
  --server-poll-seconds "${SERVER_POLL_SECONDS}"
  --inference-heartbeat-interval-seconds "${INFERENCE_HEARTBEAT_INTERVAL_SECONDS}"
  --inference-heartbeat-stale-seconds "${INFERENCE_HEARTBEAT_STALE_SECONDS}"
  --inference-heartbeat-poll-seconds "${INFERENCE_HEARTBEAT_POLL_SECONDS}"
  --aiperf-model "${AIPERF_MODEL}"
  --aiperf-endpoint-type "${AIPERF_ENDPOINT_TYPE}"
  --aiperf-streaming "${AIPERF_STREAMING}"
  --aiperf-input-len "${AIPERF_INPUT_LEN}"
  --aiperf-output-len "${AIPERF_OUTPUT_LEN}"
  --aiperf-input-stddev "${AIPERF_INPUT_STDDEV}"
  --aiperf-concurrency "${AIPERF_CONCURRENCY}"
  --aiperf-requests "${AIPERF_REQUESTS}"
  --aiperf-request-timeout-seconds "${AIPERF_REQUEST_TIMEOUT_SECONDS}"
  --aiperf-timeout-seconds "${AIPERF_TIMEOUT_SECONDS}"
  --trace-input-len "${TRACE_INPUT_LEN}"
  --trace-output-len "${TRACE_OUTPUT_LEN}"
  --trace-num-prompts "${TRACE_NUM_PROMPTS}"
  --trace-max-concurrency "${TRACE_MAX_CONCURRENCY}"
  --trace-rps "${TRACE_RPS}"
  --trace-save-details "${TRACE_SAVE_DETAILS}"
  --trace-warmup-steps "${TRACE_WARMUP_STEPS}"
  --trace-timeout-seconds "${TRACE_TIMEOUT_SECONDS}"
)

add_optional_arg --artifacts-dir "${ARTIFACTS_DIR}"
add_optional_arg --venv "${OUTER_JOB_VENV}"
add_optional_arg --hf-checkpoint-dir "${HF_CHECKPOINT_DIR}"
add_optional_arg --model-tag "${MODEL_TAG}"
add_optional_arg --fp8-checkpoint-dir "${FP8_CHECKPOINT_DIR}"
add_optional_arg --inference-queue-name "${INFERENCE_QUEUE_NAME}"
add_optional_arg --inference-priority-class "${INFERENCE_PRIORITY_CLASS}"
add_optional_arg --inference-base-image "${INFERENCE_BASE_IMAGE}"
add_optional_arg --dp "${DP}"
add_optional_arg --pp "${PP}"
add_optional_arg --aiperf-output-dir "${AIPERF_OUTPUT_DIR}"
add_optional_arg --aiperf-extra-inputs "${AIPERF_EXTRA_INPUTS}"
add_optional_arg --aiperf-bin "${AIPERF_BIN}"
add_optional_arg --aiperf-python "${AIPERF_PYTHON}"
add_optional_arg --aiperf-venv "${AIPERF_VENV}"
add_optional_arg --trace-output-dir "${TRACE_OUTPUT_DIR}"

if [ -n "${MODULES_TO_NOT_CONVERT}" ]; then
  cmd+=(--modules-to-not-convert "${MODULES_TO_NOT_CONVERT_ARGS[@]}")
fi

if bool_enabled "${FORCE_INIT}"; then
  cmd+=(--force-init)
fi
if bool_enabled "${FORCE_REMOVE_OUTSIDE_ARTIFACTS}"; then
  cmd+=(--force-remove-outside-artifacts)
fi
if bool_enabled "${FORCE_QUANT}"; then
  cmd+=(--force-quant)
fi
if bool_enabled "${KEEP_INFERENCE_JOB}"; then
  cmd+=(--keep-inference-job)
fi
if bool_enabled "${MONITOR}"; then
  cmd+=(--monitor --monitor-poll-seconds "${MONITOR_POLL_SECONDS}")
fi
if bool_enabled "${SKIP_AIPERF}"; then
  cmd+=(--skip-aiperf)
fi
if bool_enabled "${COLLECT_TRACES}"; then
  cmd+=(--collect-traces)
fi

cd "${REPO_ROOT}"
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
printf 'Running:'
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
