#!/usr/bin/env bash
set -euo pipefail
artifact=/workspace-SR008.nfs2/vkropotin/optimization-task/artifacts/linear-fixes
task=/workspace-SR008.nfs2/vkropotin/optimization-task
stage="$artifact/source"
work="$artifact/results/mla-4k-crop-followup-r2"
fixture="$task/runs/production-data-abcd/artifacts/raw/proddata-component-capture-001/native-inputs"
donor="$task/artifacts/kernel-fusion/native-mla-rope-001/gift/torchtitan/models/common/attention/softmax/mla_rope_pack.py"
resolved=/workspace-SR008.nfs2/vkropotin/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2/native/config/resolved_config.yaml
trap 'rc=$?; printf "%s\n" "$rc" > "$work/exit-code"; exit "$rc"' EXIT
cd "$stage"
export CUDA_VISIBLE_DEVICES=0 FLA_DISABLE_BACKEND_DISPATCH=1
export GIFT_GDN_DIRECT_FLA_BACKWARD=0 GIFT_GDN_FUSE_SHORT_CONV_BWD=0 GIFT_GDN_FUSE_OUTPUT_NORM_FP8=0 GIFT_MLA_FUSED_ROPE=0
export PYTHONPATH="$stage/gift/contrib/flash-linear-attention:$stage/gift"
export TORCHINDUCTOR_CACHE_DIR="$artifact/cache/mla/inductor" TRITON_CACHE_DIR="$artifact/cache/mla/triton"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
{
  date -Is
  hostname
  nvidia-smi --id=0 --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader
  sha256sum "$resolved" runs/linear-fixes/mla_component.py "$donor" gift/torchtitan/models/common/attention/softmax/mla_rope_pack.py
  grep -nE '^[[:space:]]+(local_batch_size|seq_len|n_heads|q_lora_rank|kv_lora_rank|qk_nope_head_dim|qk_rope_head_dim|v_head_dim):|save:experts_up' "$resolved"
} > "$work/provenance.txt"
common=(--seqlen 4096 --input-fixtures "$fixture" --crop-kernel "$donor")
timeout --signal=TERM --kill-after=60s 900s /opt/torchtitan-venv/bin/python3.12 -B runs/linear-fixes/mla_component.py "${common[@]}" --mode check --output "$work/check.json"
timeout --signal=TERM --kill-after=60s 900s /opt/torchtitan-venv/bin/python3.12 -B runs/linear-fixes/mla_component.py "${common[@]}" --mode timing --warmup 10 --samples 24 --output "$work/timing.json"
