#!/usr/bin/env bash
set -euo pipefail
artifact=/workspace-SR008.nfs2/vkropotin/optimization-task/artifacts/linear-fixes
task=/workspace-SR008.nfs2/vkropotin/optimization-task
stage="$artifact/source"
work="$artifact/results/mla-rope-numerics-confirmed"
first_inductor="$artifact/results/mla-rope-numerics/inductor"
fixture="$task/runs/production-data-abcd/artifacts/raw/proddata-component-capture-001/native-inputs"
resolved=/workspace-SR008.nfs2/vkropotin/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2/native/config/resolved_config.yaml
trap 'rc=$?; printf "%s\n" "$rc" > "$work/exit-code"; exit "$rc"' EXIT
source /etc/profile.d/wm.sh
cd "$stage"
export CUDA_VISIBLE_DEVICES=0 FLA_DISABLE_BACKEND_DISPATCH=1
export GIFT_GDN_DIRECT_FLA_BACKWARD=0 GIFT_GDN_FUSE_SHORT_CONV_BWD=0 GIFT_GDN_FUSE_OUTPUT_NORM_FP8=0 GIFT_MLA_FUSED_ROPE=0
export PYTHONPATH="$stage/gift/contrib/flash-linear-attention:$stage/gift"
export TORCHINDUCTOR_CACHE_DIR="$work/inductor" TRITON_CACHE_DIR="$artifact/cache/mla/triton"
export TORCH_LOGS=output_code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
{
  date -Is
  hostname
  nvidia-smi --id=0 --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader
  sha256sum "$resolved" runs/linear-fixes/mla_rope_numerics.py gift/torchtitan/models/common/attention/softmax/mla_rope_pack.py
  grep -nE '^[[:space:]]+(local_batch_size|seq_len|n_heads|q_lora_rank|kv_lora_rank|qk_nope_head_dim|qk_rope_head_dim|v_head_dim):|save:experts_up' "$resolved"
} > "$work/provenance.txt"
for seqlen in 4096 8192; do
  case "$seqlen" in
    4096) backward="$first_inductor/ig/cigkqemqsrznkkwsh2ctfhwyz35zsi2aq52ladw4rbxmpjwaahmb.py" ;;
    8192) backward="$first_inductor/ac/cacqolgt2xntxwna5bprayudra3ec3mn4kycgzpgshoozgxd53ah.py" ;;
  esac
  timeout --signal=TERM --kill-after=30s 480s /opt/torchtitan-venv/bin/python3.12 -B runs/linear-fixes/mla_rope_numerics.py \
    --input-fixtures "$fixture" --seqlen "$seqlen" --output "$work/$seqlen.json" --baseline-backward-module "$backward" \
    > "$work/$seqlen.log" 2>&1
done
