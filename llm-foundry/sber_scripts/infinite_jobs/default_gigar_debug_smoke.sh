#!/usr/bin/env bash
set -euo pipefail

CODE_PATH="$(dirname "$(realpath "$0")")/../../"
cd "$CODE_PATH"

export CODE_PATH
export NNODES="${NNODES:-2}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_PATH="${MODEL_PATH:-/workspace-SR008.nfs2/models/pam_speedtest/27b_full_run/artifacts/hf_checkpoint}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
export USER_NAME="${USER_NAME:-${USER:-debug}}"
export RUN_NAME="${RUN_NAME:-default-gigar-debug-smoke}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
export EFFECTIVE_RUN_NAME="${RUN_NAME}-${RUN_STAMP}"
export PPORT="${PPORT:-6001}"
export OPAL_PREFIX=/opt/hpcx/ompi
export PATH=/opt/hpcx/ompi/bin:$PATH
export PYTHONPATH="$CODE_PATH:$CODE_PATH/contrib/composer:$CODE_PATH/contrib/streaming:$CODE_PATH/contrib/gigafsdp:$CODE_PATH/contrib/flash-linear-attention:$CODE_PATH/contrib/metric-sampler"
export LOG_ROOT="${LOG_ROOT:-$CODE_PATH/tmp/inf-job-logs}"
export SAVE_ROOT="${SAVE_ROOT:-$CODE_PATH/tmp/checkpoints}"
export TB_ROOT="${TB_ROOT:-$CODE_PATH/tmp/tensorboard}"

bash "$(dirname "$(realpath "$0")")/clean.sh"

HOSTLIST=/tmp/hostlist_default_gigar_debug_smoke
cat /etc/volcano/mpimaster.host > "$HOSTLIST"
echo "" >> "$HOSTLIST"
cat /etc/volcano/mpiworker.host >> "$HOSTLIST"
echo "" >> "$HOSTLIST"

mkdir -p "$CODE_PATH/tmp" "$LOG_ROOT" "$SAVE_ROOT" "$TB_ROOT"
JOB_LOGS_DIR="$LOG_ROOT/$EFFECTIVE_RUN_NAME"
mkdir -p "$JOB_LOGS_DIR"
printf '%s\n' "$JOB_LOGS_DIR" > "$CODE_PATH/tmp/default_gigar_debug_smoke_last_logs_dir.txt"
printf 'JOB_LOGS_DIR=%s\n' "$JOB_LOGS_DIR"

mpirun -N 1 -np "$NNODES" \
  --hostfile "$HOSTLIST" \
  -bind-to none \
  -x NNODES \
  -x MASTER_ADDR="$(grep mpimaster-0 "$HOSTLIST" | tr -d '\n')" \
  -x PPORT \
  -x NGPUS_PER_NODE \
  -x CUDA_VISIBLE_DEVICES \
  -x NCCL_DEBUG=INFO \
  -x TORCH_CPP_LOG_LEVEL=ERROR \
  -x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  -x PYTHONPATH \
  -x NCCL_IB_HCA=^mlx5_bond \
  -x PYTHONUNBUFFERED=1 \
  -x LD_LIBRARY_PATH \
  -x OPAL_PREFIX \
  -x OMP_NUM_THREADS=1 \
  -x PATH \
  -mca pml ucx \
  -mca plm_rsh_args "-p 2222" \
  -output-filename "$JOB_LOGS_DIR/nojobid" \
  -merge-stderr-to-stdout \
  -tag-output \
  /bin/bash -c "export NODE_RANK=\$(hostname | python3 -c \"hostname = input(); rank = 0 if hostname.endswith('mpimaster-0') else int(hostname.split('-')[-1]) + 1; print(rank, end='')\") && python -m torch.distributed.launch --nproc_per_node \$NGPUS_PER_NODE --nnodes \$NNODES --node_rank \$NODE_RANK --master_addr \$MASTER_ADDR --master_port \$PPORT --use_env $CODE_PATH/scripts/train/train.py $CODE_PATH/sber_configs/default_gigar_config.yaml use_clearml=false user=$USER_NAME exp_name=$RUN_NAME run_name=$EFFECTIVE_RUN_NAME tensorboard_path=$TB_ROOT/$EFFECTIVE_RUN_NAME save_folder=$SAVE_ROOT/$EFFECTIVE_RUN_NAME model_path=$MODEL_PATH model.pretrained_model_name_or_path=$MODEL_PATH tokenizer.name=$TOKENIZER_PATH max_seq_len=128 train_loader.enable_data_profiler=false debug.use_debug=true model.config_overrides.gated_attention=false model.config_overrides.layernorm_type=pre callbacks.training_metrics_monitor.batch_log_interval=1 callbacks.optimizer_monitor.batch_log_interval=1"
