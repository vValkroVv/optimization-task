#!/bin/bash

ppath=/home/jovyan/dtitov/llm-foundry
export PYTHONPATH=$ppath/:$ppath/contrib/composer:$ppath/contrib/streaming:$ppath/contrib/gigafsdp

OUTPUT_DIR="/home/jovyan/dtitov/dump/benchmarks_qwen3next"
mkdir -p "$OUTPUT_DIR"

SCRIPT="mla_and_qwen3_next_benchmark.py"

run_benchmark() {
    local config_name=$1
    shift  
    
    echo "=========================================="
    echo "Running benchmark: $config_name"
    echo "Parameters: $@"
    echo "=========================================="
    
    local trace_path="$OUTPUT_DIR/${config_name}_trace.json.gz"
    local memory_path="$OUTPUT_DIR/${config_name}_memory.pickle"
    
    python "$SCRIPT" \
        --trace_output_path "$trace_path" \
        --memory_profiler_output_path "$memory_path" \
        "$@"
    
    echo "Completed: $config_name"
    echo ""
}

# Configuration 1: 10B-1.8A
run_benchmark "10b_h1536_s4096" \
    --hidden_size 1536 \
    --seq_len 4096 \
    --num_attention_heads 32 \
    --num_key_value_heads 32 \
    --qk_nope_head_dim 128 \
    --qk_rope_head_dim 64 \
    --v_head_dim 192 \
    --kv_lora_rank 512 \
    --linear_key_head_dim 128 \
    --linear_value_head_dim 128 \
    --linear_num_key_heads 8 \
    --linear_num_value_heads 8 \

# # Configuration 2: Ultra 
run_benchmark "700b_h7168_s4096" \
    --hidden_size 7168 \
    --seq_len 4096 \
    --num_attention_heads 64 \
    --num_key_value_heads 64 \
    --qk_nope_head_dim 128 \
    --qk_rope_head_dim 64 \
    --v_head_dim 192 \
    --q_lora_rank 1536 \
    --kv_lora_rank 512 \
    --linear_key_head_dim 128 \
    --linear_value_head_dim 128 \
    --linear_num_key_heads 32 \
    --linear_num_value_heads 64 \

echo "=========================================="
echo "All benchmarks completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="

