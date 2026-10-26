#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# path_prefix=/home/jovyan/nvokhotnikov/model_save_folder/700B_random_linear_full_every4_no_sigmoid/
# path_prefix=/home/jovyan/nvokhotnikov/model_save_folder/700B_random_linear_full_every4_no_sigmoid_vhead128/

path_prefix=${PATH_PREFIX:-/home/jovyan/nvokhotnikov/model_save_folder/700B_old_random/}
config_path=${CONFIG_PATH:-${path_prefix}/llm_foundry_config.yaml}
output_dir=${OUTPUT_DIR:-${path_prefix}/random_hf_checkpoint_uniform}

use_custom_sglang=${USE_CUSTOM_SGLANG:-True}
include_mtp=${INCLUDE_MTP:-True}
device=${INIT_DEVICE:-cuda}
uniform_moe=${UNIFORM_MOE:-False}
expand_context_len_to=${EXPAND_CONTEXT_LEN_TO:-262144}
python_bin=${PYTHON:-python}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0}
tokenizer_path=${TOKENIZER_PATH:-}

cmd=(
    "${python_bin}" "${SCRIPT_DIR}/init_random_hf_model.py"
    --config "${config_path}"
    --output_dir "${output_dir}"
    --include_mtp "${include_mtp}"
    --use_custom_sglang "${use_custom_sglang}"
    --device "${device}"
    --uniform_moe "${uniform_moe}"
)

if [ -n "${expand_context_len_to}" ]; then
    cmd+=(--expand_context_len_to "${expand_context_len_to}")
fi
if [ -n "${tokenizer_path}" ]; then
    cmd+=(--tokenizer_path "${tokenizer_path}")
fi

if [ "${use_custom_sglang}" = "True" ]; then
    model_type=${MODEL_TYPE:-gigachat35}
    cmd+=(--model_type "${model_type}")
elif [ "${use_custom_sglang}" = "False" ]; then
    model_tag=${MODEL_TAG:-}
    cmd+=(--model_tag "${model_tag}")
else
    echo "USE_CUSTOM_SGLANG must be True or False"
    exit 1
fi

echo "Running random HF checkpoint init:"
printf ' %q' "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}" "${cmd[@]}"
echo
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${cmd[@]}"
