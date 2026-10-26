#!/usr/bin/env python3
"""Benchmark SGLang 0.5.13 sparse and linear attention kernels.

Examples:
  python sber_benchmarks/sglang_attention_kernel_benchmark.py \
    --kernel linear-gdn-e2e --model-preset qwen3_5_397b_a17b \
    --input-len 262144 --output-len 128 --batch-size 1 \
    --tp-size 8 --dtype fp8 --state-dtype bf16

  python sber_benchmarks/sglang_attention_kernel_benchmark.py \
    --kernel flashmla-kvcache-e2e --model-preset deepseek_v4_pro \
    --input-len 262144 --output-len 128 --batch-size 1 \
    --tp-size 8 --dtype fp8

The script imports sglang/sgl_kernel from the active Python environment only. It
never rewrites sys.path and never loads model weights. See the adjacent Markdown
file for the methodology, dimensional constraints, and JSON output schema.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Callable


DEFAULT_SEQ_LEN = 262_144
LINEAR_BACKENDS = ("triton", "cutedsl", "flashinfer")
MLA_ATTENTION_BACKENDS = (
    "auto",
    "triton",
    "fa3",
    "flashinfer",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
    "cutedsl_mla",
)
DSA_BACKENDS = (
    "auto",
    "flashmla_sparse",
    "flashmla_kv",
    "flashmla_auto",
    "fa3",
    "tilelang",
    "aiter",
    "trtllm",
)
DSA_TOPK_BACKENDS = ("sgl-kernel", "torch", "flashinfer")
MLA_KV_CACHE_DTYPES = ("auto", "bfloat16", "bf16", "fp8_e4m3")
DSV4_DEFAULT_CHUNKED_PREFILL_SIZE = 8192
DSV4_SPARSE_PREFILL_QUERY_THRESHOLD = 11673
DSV4_FLASHMLA_SPARSE_PREFILL_KNOWN_Q_HEADS = (64, 128)
DSV4_FLASHMLA_SPARSE_DECODE_KNOWN_Q_HEADS = (64, 128)
DSV4_FLASHMLA_MIN_PADDED_Q_HEADS = 64
DSV4_COMPRESS_RATIOS = (0, 4, 128)
DSV4_COMPRESSION_RATIOS = {"none": 0, "csa": 4, "hca": 128}
DSV4_ATTENTION_KERNELS = ("triton", "flashmla", "tilelang")
DSV4_ATTENTION_KERNEL_ENTRYPOINTS = {
    "flashmla": "kernel",
    "triton": "triton",
    "tilelang": "tilelang",
}
DSV4_NSA_QUANT_BLOCK_SIZE = 128
DSV4_SWA_WINDOW = 128
DSV4_SPARSE_PREFILL_TOPK_ALIGNMENT = 128
DSV4_PAGE_INDEX_ALIGNMENT = 64


@dataclass(frozen=True)
class Preset:
    family: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    index_head_dim: int | None = None
    index_n_heads: int | None = None
    index_topk: int | None = None
    o_groups: int | None = None
    o_lora_rank: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_conv_kernel_dim: int | None = None
    sliding_window: int | None = None
    max_position_embeddings: int | None = None
    rope_theta: float | None = None
    rms_norm_eps: float | None = None
    index_topk_freq: int | None = None
    index_skip_topk_offset: int | None = None
    index_topk_pattern: str | None = None
    rope_interleave: bool | None = None
    indexer_rope_interleave: bool | None = None


PRESETS: dict[str, Preset] = {
    # Largest Qwen3.5 Gated Delta Net preset used for linear-attention shapes.
    "qwen3_5_397b_a17b": Preset(
        family="qwen3_5_gdn",
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=256,
        hidden_size=8192,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        linear_conv_kernel_dim=4,
    ),
    # Kept as a smaller comparison point from the same SGLang Qwen hybrid path.
    "qwen3_next_80b_a3b": Preset(
        family="qwen3_next_gdn",
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        hidden_size=4096,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
    ),
    # DeepSeek V4 Pro sparse-attention shape preset.
    "deepseek_v4_pro": Preset(
        family="deepseek_v4",
        num_attention_heads=128,
        num_key_value_heads=1,
        head_dim=512,
        hidden_size=8192,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        v_head_dim=512,
        kv_lora_rank=512,
        q_lora_rank=1024,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=1024,
        o_groups=16,
        o_lora_rank=1024,
        sliding_window=128,
    ),
    # Smaller DeepSeek V4 Flash sparse-attention shape preset for comparison.
    "deepseek_v4_flash": Preset(
        family="deepseek_v4",
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        hidden_size=4096,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        v_head_dim=512,
        kv_lora_rank=512,
        q_lora_rank=1024,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        o_groups=8,
        o_lora_rank=1024,
        sliding_window=128,
    ),
    # DeepSeek V3/R1 MLA shape used by the largest DeepSeek V3 family models.
    "deepseek_v3_mla": Preset(
        family="deepseek_v3_mla",
        num_attention_heads=128,
        num_key_value_heads=1,
        head_dim=192,
        hidden_size=7168,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        q_lora_rank=1536,
        max_position_embeddings=163840,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    ),
    # GLM-5.2 FP8 DSA config fields from the largest GLM-5.2 family checkpoint.
    "glm5_2_dsa": Preset(
        family="glm5_2_dsa",
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=192,
        hidden_size=6144,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        kv_lora_rank=512,
        q_lora_rank=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        rope_interleave=True,
        indexer_rope_interleave=True,
        max_position_embeddings=1048576,
        rope_theta=8000000.0,
        rms_norm_eps=1e-5,
    ),
}


# These are benchmark entry points, not all production model branches.
# `linear-gdn-e2e` and `flashmla-kvcache-e2e` are production-like sequences;
# standalone names call one concrete helper/backend directly for isolation.
KERNEL_GROUPS: dict[str, list[str]] = {
    "linear-all": [
        "linear-triton-prefill",
        "linear-triton-decode",
        "linear-triton-packed-decode",
        "linear-flashinfer-prefill",
        "linear-flashinfer-decode",
        "linear-cutedsl-prefill",
        "linear-cutedsl-decode",
        "linear-fused-qkv-split",
        "linear-gdn-e2e",
    ],
    "sparse-all": [
        "sparse-flash",
        "sparse-flash-varlen",
        "sparse-index-convert",
        "flashmla-sparse-prefill",
        "flashmla-kvcache-e2e",
    ],
    "dsv4-all": [
        "flashmla-sparse-prefill",
        "flashmla-kvcache-e2e",
        "dsv4-c4-compress-prefill",
        "dsv4-c4-compress-decode",
        "dsv4-c128-compress-prefill",
        "dsv4-c128-compress-decode",
        "dsv4-topk-v1",
        "dsv4-topk-v2",
        "dsv4-q-norm-rope",
        "dsv4-k-norm-rope-flashmla",
        "dsv4-q-indexer-rope-hadamard-fp8",
    ],
    "mla-all": [
        "deepseek-v3-mla-e2e",
    ],
    "glm-dsa-all": [
        "glm-dsa-e2e",
    ],
}
KERNEL_GROUPS["all"] = (
    KERNEL_GROUPS["sparse-all"]
    + [
        name
        for name in KERNEL_GROUPS["dsv4-all"]
        if name not in KERNEL_GROUPS["sparse-all"]
    ]
    + KERNEL_GROUPS["mla-all"]
    + KERNEL_GROUPS["glm-dsa-all"]
    + KERNEL_GROUPS["linear-all"]
)
ALL_KERNELS = sorted({k for group in KERNEL_GROUPS.values() for k in group})


DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
    "fp8": 1,
    "float8_e4m3fn": 1,
}


KERNEL_COMMENTS = {
    "sparse-flash": "Vertical+slash sparse FlashAttention wrapper over sgl_kernel.fwd_sparse.",
    "sparse-flash-varlen": "Varlen layout for the same sparse FlashAttention kernel.",
    "sparse-index-convert": "Converts vertical/slash token lists into block metadata consumed by sparse-flash.",
    "flashmla-sparse-prefill": "DeepSeek V4 sparse prefill path over flash_mla_sparse_fwd.",
    "flashmla-kvcache-e2e": "Synthetic DSV4 prefill/decode request with production-like KV writes, growing cache_seqlens, paged FP8 cache, and selectable SGLang DSV4 attention kernel.",
    "linear-gdn-e2e": "Synthetic Qwen Gated Delta Net chunked prefill plus sequential decode with production GDN gating and selected SGLang backends.",
    "deepseek-v3-mla-e2e": "DeepSeek V3 MLA attention module E2E path using SGLang DeepseekV2AttentionMLA, RadixAttention, KV cache writes, output projection, and selected MLA backend.",
    "glm-dsa-e2e": "GLM-5.2 DSA attention module E2E path using SGLang DeepseekV2AttentionMLA, production Indexer, DSATokenToKVPool, RadixAttention, and selected DSA backends.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="SGLang sparse attention and Qwen GDN kernel benchmark.",
    )
    parser.add_argument(
        "--kernel",
        action="append",
        choices=ALL_KERNELS + sorted(KERNEL_GROUPS),
        help="Kernel or group to benchmark. Repeatable.",
    )
    parser.add_argument(
        "--list-kernels", action="store_true", help="Print kernels and exit."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on first kernel error instead of recording it.",
    )

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--dtype",
        default="fp8",
        choices=sorted(DTYPE_BYTES),
        help="Requested precision. FP8 is the target mode; kernels that only accept BF16 activations allocate BF16 inputs and record that explicitly.",
    )
    parser.add_argument(
        "--state-dtype",
        default="fp32",
        choices=sorted(DTYPE_BYTES),
        help="GDN recurrent state dtype. For FlashInfer decode on SM100 use bf16.",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--output-json",
        dest="output_json",
        default=None,
        help="Optional pretty JSON output path. The file is a list of records.",
    )
    parser.add_argument(
        "--output-jsonl",
        dest="output_json",
        default=None,
        help="Deprecated alias for --output-json; writes a JSON list, not JSONL.",
    )
    parser.add_argument(
        "--max-input-gb",
        type=float,
        default=0.0,
        help="Skip cases whose estimated live tensors exceed this many GiB. 0 disables the guard.",
    )

    parser.add_argument(
        "--model-preset",
        default="qwen3_5_397b_a17b",
        choices=sorted(PRESETS) + ["custom"],
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional HF-style config.json. text_config is merged automatically when present.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Sequence length for one-stage prefill-like kernels.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Prompt length for *-e2e cases. Defaults to --seq-len.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of sequential decode steps for *-e2e cases.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=None,
        help="Batch size for decode kernels. Defaults to --batch-size.",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--enable-dp-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the SGLang DP-attention serving layout. When enabled, the benchmark sets "
            "attention_dp_size=tp_size and effective attention TP to 1 unless --attn-tp-size explicitly overrides it."
        ),
    )
    parser.add_argument(
        "--attn-tp-size",
        type=int,
        default=None,
        help="Explicit effective attention TP override for isolated kernel-specialization probes.",
    )

    parser.add_argument(
        "--linear-attn-backend",
        default="triton",
        choices=LINEAR_BACKENDS,
        help="SGLang-style default GDN backend for --kernel linear-gdn-e2e.",
    )
    parser.add_argument(
        "--linear-attn-prefill-backend",
        default=None,
        choices=LINEAR_BACKENDS,
        help="Override GDN prefill backend for --kernel linear-gdn-e2e.",
    )
    parser.add_argument(
        "--linear-attn-decode-backend",
        default=None,
        choices=LINEAR_BACKENDS,
        help="Override GDN decode backend for --kernel linear-gdn-e2e.",
    )
    parser.add_argument(
        "--mla-attention-backend",
        default="auto",
        choices=MLA_ATTENTION_BACKENDS,
        help="SGLang attention backend for DeepSeek V3 MLA E2E. auto follows server defaults: fa3 on Hopper, trtllm_mla on Blackwell.",
    )
    parser.add_argument(
        "--mla-kv-cache-dtype",
        default="auto",
        choices=MLA_KV_CACHE_DTYPES,
        help="KV cache dtype for DeepSeek V3 MLA E2E; auto uses BF16 storage in this benchmark.",
    )
    parser.add_argument(
        "--dsa-prefill-backend",
        default="auto",
        choices=DSA_BACKENDS,
        help="Server-like DSA prefill backend for GLM DSA E2E. auto follows SGLang H100 defaults from kv_cache_dtype.",
    )
    parser.add_argument(
        "--dsa-decode-backend",
        default="auto",
        choices=DSA_BACKENDS,
        help="Server-like DSA decode backend for GLM DSA E2E. auto follows SGLang H100 defaults from kv_cache_dtype.",
    )
    parser.add_argument(
        "--dsa-topk-backend",
        default="sgl-kernel",
        choices=DSA_TOPK_BACKENDS,
        help="SGLang DSA indexer top-k backend.",
    )
    parser.add_argument(
        "--dsa-kv-cache-dtype",
        default="auto",
        choices=MLA_KV_CACHE_DTYPES,
        help="KV cache dtype for GLM DSA E2E. auto maps to bfloat16 on H100 and fp8_e4m3 on Blackwell, matching server_args.",
    )
    parser.add_argument(
        "--mla-page-size",
        type=int,
        default=64,
        help="Page size for DeepSeek V3 MLA KV pool.",
    )
    parser.add_argument(
        "--dsa-page-size",
        type=int,
        default=64,
        help="Page size for GLM DSA KV/indexer pools. SGLang DSA CUDA path expects 64.",
    )
    parser.add_argument(
        "--glm-dsa-layer-id",
        type=int,
        default=2,
        help="Representative GLM DSA layer id. Default layer 2 computes topk and returns it for the next shared-topk layer under GLM5.2 index_topk_freq/index_skip_topk_offset semantics.",
    )

    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--qk-nope-head-dim", type=int, default=None)
    parser.add_argument("--qk-rope-head-dim", type=int, default=None)
    parser.add_argument("--v-head-dim", type=int, default=None)
    parser.add_argument("--kv-lora-rank", type=int, default=None)
    parser.add_argument("--q-lora-rank", type=int, default=None)
    parser.add_argument("--index-head-dim", type=int, default=None)
    parser.add_argument("--index-n-heads", type=int, default=None)
    parser.add_argument("--index-topk", type=int, default=None)
    parser.add_argument("--index-topk-freq", type=int, default=None)
    parser.add_argument("--index-skip-topk-offset", type=int, default=None)
    parser.add_argument("--max-position-embeddings", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=None)
    parser.add_argument("--o-groups", type=int, default=None)
    parser.add_argument("--o-lora-rank", type=int, default=None)
    parser.add_argument("--linear-key-head-dim", type=int, default=None)
    parser.add_argument("--linear-value-head-dim", type=int, default=None)
    parser.add_argument("--linear-num-key-heads", type=int, default=None)
    parser.add_argument("--linear-num-value-heads", type=int, default=None)

    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--vertical-nnz", type=int, default=512)
    parser.add_argument("--slash-nnz", type=int, default=8)
    parser.add_argument(
        "--sparse-topk",
        type=int,
        default=None,
        help="FlashMLA sparse topk. Defaults to model index_topk when present.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Build causal sparse metadata for sparse-flash kernels.",
    )
    parser.add_argument(
        "--softcap",
        type=float,
        default=0.0,
        help="Forwarded to sparse-flash if supported by installed wrapper.",
    )
    parser.add_argument(
        "--attn-sink",
        action="store_true",
        help="Pass random attn_sink to FlashMLA sparse prefill and DSV4 E2E when the installed FlashMLA wrapper supports it.",
    )
    parser.add_argument(
        "--topk-length",
        action="store_true",
        help="Pass per-query topk_length to FlashMLA sparse prefill.",
    )
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1e-6,
        help="Epsilon for DSV4 fused RMSNorm+RoPE elementwise kernels.",
    )

    parser.add_argument("--dsv4-context", default="paged", choices=["paged", "legacy"])
    parser.add_argument(
        "--dsv4-compression",
        default="csa",
        choices=["none", "csa", "hca", "custom"],
        help="DeepSeek V4 sparse attention compression mode for flashmla-kvcache-e2e. none=SWA-only sparse, csa=4x, hca=128x; custom requires --dsv4-compress-ratio.",
    )
    parser.add_argument(
        "--dsv4-compress-ratio",
        type=int,
        default=None,
        choices=list(DSV4_COMPRESS_RATIOS),
        help="Custom DeepSeek V4 compression ratio. Used only with --dsv4-compression=custom: 0=SWA-only, 4=CSA/C4, 128=HCA/C128.",
    )
    parser.add_argument(
        "--dsv4-attention-kernel",
        default="triton",
        choices=DSV4_ATTENTION_KERNELS,
        help="SGLang DSV4 kvcache attention implementation for flashmla-kvcache-e2e. flashmla maps to the default sgl_kernel FlashMLA path; triton/tilelang use SGLang alternative kernels with the same DSV4 kwargs.",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=DSV4_DEFAULT_CHUNKED_PREFILL_SIZE,
        help="E2E prefill chunk size for DSV4 and GDN. Values >0 model production chunked serving; <=0 requests one monolithic prefill where the kernel supports it.",
    )
    parser.add_argument("--swa-page-size", type=int, default=256)
    parser.add_argument("--ring-size", type=int, default=None)
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Override GDN/linear state pool slots.",
    )
    args = parser.parse_args()
    return normalize_dsv4_compression_args(parser, args)


def normalize_dsv4_compression_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    requested_ratio = args.dsv4_compress_ratio
    args.dsv4_compress_ratio_requested = requested_ratio
    args.dsv4_compress_ratio_overridden = False
    if args.dsv4_compression == "custom":
        if requested_ratio is None:
            parser.error(
                "--dsv4-compress-ratio is required when --dsv4-compression=custom"
            )
        args.dsv4_compress_ratio = int(requested_ratio)
    else:
        effective_ratio = DSV4_COMPRESSION_RATIOS[args.dsv4_compression]
        args.dsv4_compress_ratio_overridden = (
            requested_ratio is not None and int(requested_ratio) != effective_ratio
        )
        args.dsv4_compress_ratio = effective_ratio
    return args


def stdout_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def should_print_stdout() -> bool:
    return stdout_rank() == 0


def emit(record: dict[str, Any]) -> None:
    if not should_print_stdout():
        return
    print("[sglang-attn-bench][record]", flush=True)
    keys = (
        "kernel",
        "status",
        "model_preset",
        "tp_size",
        "enable_dp_attention",
        "attention_dp_size",
        "attn_tp_size",
        "distributed_world_size",
        "distributed_rank",
        "local_rank",
        "tp_communication_mode",
        "input_len",
        "output_len",
        "batch_size",
    )
    for key in keys:
        if key in record:
            print(f"  {key}: {record[key]}", flush=True)
    timing = record.get("timing")
    if isinstance(timing, dict):
        for key in (
            "latency_ms_p50",
            "latency_ms_p90",
            "latency_ms_p95",
            "latency_ms_p99",
            "tokens_per_second_p50",
        ):
            if key in timing:
                print(f"  {key}: {timing[key]}", flush=True)
        for phase in ("prefill", "decode"):
            phase_timing = timing.get(phase)
            if isinstance(phase_timing, dict):
                latency = phase_timing.get("latency_ms", {})
                p50 = latency.get("p50_ms") if isinstance(latency, dict) else None
                tps = phase_timing.get("tokens_per_second_p50")
                print(f"  {phase}_latency_ms_p50: {p50}", flush=True)
                print(f"  {phase}_tokens_per_second_p50: {tps}", flush=True)
    for key in (
        "estimated_live_gib",
        "max_memory_allocated_gib",
        "max_memory_reserved_gib",
    ):
        if key in record:
            print(f"  {key}: {record[key]}", flush=True)
    message = record.get("error_message") or record.get("error")
    if message:
        print("  error:", flush=True)
        for line in str(message).splitlines() or [""]:
            print(f"    {line}", flush=True)
    if record.get("output_json"):
        print(f"  output_json: {record['output_json']}", flush=True)


def warn_stdout(message: str) -> None:
    if not should_print_stdout():
        return
    print("[sglang-attn-bench][warning]", flush=True)
    for line in str(message).splitlines() or [""]:
        print(f"  {line}", flush=True)


def write_records(records: list[dict[str, Any]], output_json: str | None) -> None:
    if not output_json:
        return
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def selected_kernels(args: argparse.Namespace) -> list[str]:
    names = args.kernel or ["all"]
    expanded: list[str] = []
    for name in names:
        expanded.extend(KERNEL_GROUPS.get(name, [name]))
    deduped: list[str] = []
    seen: set[str] = set()
    for name in expanded:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def normalize_dp_attention_args(
    args: argparse.Namespace, model: dict[str, Any], kernels: list[str]
) -> tuple[list[str], str | None]:
    _ = (model, kernels)
    warnings = []
    args.enable_dp_attention_requested = bool(args.enable_dp_attention)
    args.enable_dp_attention = bool(args.enable_dp_attention)
    if getattr(args, "dsv4_compress_ratio_overridden", False):
        warnings.append(
            f"Ignoring --dsv4-compress-ratio={args.dsv4_compress_ratio_requested} because "
            f"--dsv4-compression={args.dsv4_compression} maps to ratio {args.dsv4_compress_ratio}. "
            "Use --dsv4-compression custom to force an explicit ratio."
        )
    return warnings, None


def error_stdout(message: str) -> None:
    if not should_print_stdout():
        return
    print("[sglang-attn-bench][error]", flush=True)
    for line in str(message).splitlines() or [""]:
        print(f"  {line}", flush=True)


def list_kernels() -> None:
    print("Kernel groups:")
    for name in sorted(KERNEL_GROUPS):
        print(f"  {name}: {', '.join(KERNEL_GROUPS[name])}")
    print("Kernels:")
    for name in ALL_KERNELS:
        comment = KERNEL_COMMENTS.get(name, "")
        suffix = f"  # {comment}" if comment else ""
        print(f"  {name}{suffix}")


def preset_to_dict(preset: Preset) -> dict[str, Any]:
    return {k: v for k, v in preset.__dict__.items() if v is not None}


def load_config_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    text_config = loaded.get("text_config")
    if isinstance(text_config, dict):
        loaded = {**loaded, **text_config}
    return loaded


def resolve_model(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_preset == "custom":
        base = Preset(
            family="custom",
            num_attention_heads=args.num_attention_heads or 32,
            num_key_value_heads=args.num_key_value_heads or 1,
            head_dim=args.head_dim or 128,
        )
    else:
        base = PRESETS[args.model_preset]
    model = preset_to_dict(base)

    cfg = load_config_json(args.config_json)
    config_map = {
        "family": cfg.get("family")
        or ("deepseek_v4" if cfg.get("model_type") == "deepseek_v4" else None),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "head_dim": cfg.get("head_dim"),
        "hidden_size": cfg.get("hidden_size"),
        "qk_nope_head_dim": cfg.get("qk_nope_head_dim"),
        "qk_rope_head_dim": cfg.get("qk_rope_head_dim"),
        "v_head_dim": cfg.get("v_head_dim") or cfg.get("head_dim"),
        "kv_lora_rank": cfg.get("kv_lora_rank"),
        "q_lora_rank": cfg.get("q_lora_rank"),
        "index_head_dim": cfg.get("index_head_dim"),
        "index_n_heads": cfg.get("index_n_heads"),
        "index_topk": cfg.get("index_topk"),
        "o_groups": cfg.get("o_groups"),
        "o_lora_rank": cfg.get("o_lora_rank"),
        "linear_key_head_dim": cfg.get("linear_key_head_dim"),
        "linear_value_head_dim": cfg.get("linear_value_head_dim"),
        "linear_num_key_heads": cfg.get("linear_num_key_heads"),
        "linear_num_value_heads": cfg.get("linear_num_value_heads"),
        "linear_conv_kernel_dim": cfg.get("linear_conv_kernel_dim"),
        "sliding_window": cfg.get("sliding_window"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "rope_theta": cfg.get("rope_theta")
        or (cfg.get("rope_parameters") or {}).get("rope_theta"),
        "rms_norm_eps": cfg.get("rms_norm_eps"),
        "index_topk_freq": cfg.get("index_topk_freq"),
        "index_skip_topk_offset": cfg.get("index_skip_topk_offset"),
        "index_topk_pattern": cfg.get("index_topk_pattern"),
        "rope_interleave": cfg.get("rope_interleave"),
        "indexer_rope_interleave": cfg.get("indexer_rope_interleave"),
    }
    for key, value in config_map.items():
        if value is not None:
            model[key] = value

    override_map = {
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "head_dim": args.head_dim,
        "hidden_size": args.hidden_size,
        "qk_nope_head_dim": args.qk_nope_head_dim,
        "qk_rope_head_dim": args.qk_rope_head_dim,
        "v_head_dim": args.v_head_dim,
        "kv_lora_rank": args.kv_lora_rank,
        "q_lora_rank": args.q_lora_rank,
        "index_head_dim": args.index_head_dim,
        "index_n_heads": args.index_n_heads,
        "index_topk": args.index_topk,
        "index_topk_freq": args.index_topk_freq,
        "index_skip_topk_offset": args.index_skip_topk_offset,
        "max_position_embeddings": args.max_position_embeddings,
        "rope_theta": args.rope_theta,
        "o_groups": args.o_groups,
        "o_lora_rank": args.o_lora_rank,
        "linear_key_head_dim": args.linear_key_head_dim,
        "linear_value_head_dim": args.linear_value_head_dim,
        "linear_num_key_heads": args.linear_num_key_heads,
        "linear_num_value_heads": args.linear_num_value_heads,
    }
    for key, value in override_map.items():
        if value is not None:
            model[key] = value

    # Make custom non-DeepSeek configs usable for quick GDN shape sweeps. For
    # model-accurate runs pass the real linear_* fields in --config-json or
    # explicit overrides. DeepSeek presets intentionally do not grow synthetic
    # linear_* fields, because those are not part of the DSV4 attention path.
    model.setdefault("v_head_dim", model.get("head_dim", 128))
    if "hidden_size" not in model:
        model["hidden_size"] = int(model.get("num_attention_heads", 1)) * int(
            model.get("head_dim", 128)
        )
    if model.get("family") != "deepseek_v4":
        model.setdefault("linear_key_head_dim", model.get("head_dim", 128))
        model.setdefault("linear_value_head_dim", model.get("head_dim", 128))
        model.setdefault(
            "linear_num_key_heads",
            model.get("num_key_value_heads", model.get("num_attention_heads", 1)),
        )
        model.setdefault("linear_num_value_heads", model.get("num_attention_heads", 1))
        model.setdefault("linear_conv_kernel_dim", 4)
    if model.get("family") == "deepseek_v4":
        model.setdefault(
            "kv_lora_rank", model.get("v_head_dim", model.get("head_dim", 512))
        )
        model.setdefault("q_lora_rank", 1024)
    if (
        model.get("family") in {"deepseek_v4", "glm5_2_dsa"}
        or args.index_topk is not None
    ):
        model.setdefault("index_head_dim", 128)
        model.setdefault("index_n_heads", 64)
        model.setdefault("index_topk", 512)
    model.setdefault("qk_rope_head_dim", 64)
    model.setdefault("max_position_embeddings", input_len(args) + args.output_len + 1)
    model["max_position_embeddings"] = max(
        int(model["max_position_embeddings"]), input_len(args) + args.output_len + 1
    )
    model.setdefault("rope_theta", 10000.0)
    model.setdefault("rms_norm_eps", 1e-6)
    if model.get("family") == "glm5_2_dsa":
        model.setdefault("index_topk_freq", 4)
        model.setdefault("index_skip_topk_offset", 3)
        model.setdefault("rope_interleave", True)
        model.setdefault("indexer_rope_interleave", True)
    if model.get("family") == "deepseek_v4":
        model.setdefault(
            "o_groups", max(1, int(model.get("num_attention_heads", 1)) // 8)
        )
        model.setdefault("o_lora_rank", 1024)
    return model


def attn_tp_size(args: argparse.Namespace) -> int:
    if args.attn_tp_size is not None:
        if args.attn_tp_size <= 0:
            raise ValueError("--attn-tp-size must be positive")
        return args.attn_tp_size
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive")
    local = 1 if bool(args.enable_dp_attention) else args.tp_size
    if local <= 0:
        raise ValueError("effective attention TP is zero")
    return local


def attention_dp_size(args: argparse.Namespace) -> int:
    return int(args.tp_size) if bool(args.enable_dp_attention) else 1


def attention_tp_rank(args: argparse.Namespace) -> int:
    atp = attn_tp_size(args)
    if atp <= 1:
        return 0
    return int(getattr(args, "rank", 0)) % atp


def require_divisible(name: str, value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"{name}: divisor must be positive")
    if value % divisor != 0:
        raise ValueError(f"{name}={value} is not divisible by attn_tp_size={divisor}")
    local = value // divisor
    if local <= 0:
        raise ValueError(f"{name}//attn_tp_size is zero")
    return local


def local_kv_heads_for_attention(model: dict[str, Any], atp: int) -> int:
    kv_heads = int(model["num_key_value_heads"])
    if kv_heads == 1:
        return 1
    if kv_heads % atp == 0:
        return kv_heads // atp
    if kv_heads < atp:
        return kv_heads
    raise ValueError(
        f"num_key_value_heads={kv_heads} is not divisible by attn_tp_size={atp}"
    )


def dsv4_local_q_heads(model: dict[str, Any], atp: int) -> int:
    """Model-accurate TP-local DSV4 query heads before the production FlashMLA pad."""
    return require_divisible(
        "num_attention_heads", int(model["num_attention_heads"]), atp
    )


def dsv4_flashmla_q_heads(model: dict[str, Any], atp: int) -> int:
    local_q_heads = dsv4_local_q_heads(model, atp)
    if atp > 1 and local_q_heads <= DSV4_FLASHMLA_MIN_PADDED_Q_HEADS:
        return DSV4_FLASHMLA_MIN_PADDED_Q_HEADS
    if atp > 1:
        return int(model["num_attention_heads"])
    return local_q_heads


def dsv4_q_padded_for_flashmla(model: dict[str, Any], atp: int) -> bool:
    return dsv4_flashmla_q_heads(model, atp) != dsv4_local_q_heads(model, atp)


def dsv4_q_padded_to_full_heads(model: dict[str, Any], atp: int) -> bool:
    return dsv4_q_padded_for_flashmla(model, atp) and dsv4_flashmla_q_heads(
        model, atp
    ) == int(model["num_attention_heads"])


def dsv4_supported_flashmla_attn_tp_sizes(
    model: dict[str, Any], supported_q_heads: tuple[int, ...]
) -> list[int]:
    num_heads = int(model["num_attention_heads"])
    return sorted(
        atp
        for atp in range(1, num_heads + 1)
        if num_heads % atp == 0
        and dsv4_flashmla_q_heads(model, atp) in supported_q_heads
    )


def dsv4_dp_off_padding_warning(
    args: argparse.Namespace, model: dict[str, Any], atp: int
) -> str | None:
    if bool(getattr(args, "enable_dp_attention", False)) or atp <= 1:
        return None
    if not dsv4_q_padded_for_flashmla(model, atp):
        return None
    local_q_heads = dsv4_local_q_heads(model, atp)
    flashmla_q_heads = dsv4_flashmla_q_heads(model, atp)
    return (
        "WARNING: DeepSeek V4 DP attention is disabled, so the SGLang production TP path pads Q "
        f"from local_q_heads={local_q_heads} to flashmla_q_heads={flashmla_q_heads} before FlashMLA and slices "
        f"the output back to {local_q_heads} local heads. This is production behavior from DeepseekV4AttentionMLA.forward, "
        "but it measures the DP-off head-padding path, which is not the preferred TP=8 DSV4 sparse-attention layout. "
        f"For the DP-attention path use --enable-dp-attention so attention_dp_size=tp_size={args.tp_size} and effective attn_tp_size=1."
    )


def dsv4_topk(args: argparse.Namespace, model: dict[str, Any]) -> int:
    topk = int(args.sparse_topk or model.get("index_topk", 512))
    if topk <= 0:
        raise ValueError(f"DSV4 sparse topk must be positive, got {topk}")
    return topk


def ceil_align(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def dsv4_sparse_prefill_combined_topk(
    args: argparse.Namespace, model: dict[str, Any]
) -> int:
    extra_topk = 0 if int(args.dsv4_compress_ratio) == 0 else dsv4_topk(args, model)
    return ceil_align(extra_topk + DSV4_SWA_WINDOW, DSV4_SPARSE_PREFILL_TOPK_ALIGNMENT)


def dsv4_kv_lora_rank(model: dict[str, Any]) -> int:
    return int(
        model.get("kv_lora_rank", model.get("v_head_dim", model.get("head_dim", 512)))
    )


def dsv4_q_lora_rank(model: dict[str, Any]) -> int:
    return int(model.get("q_lora_rank", 1024))


def dsv4_index_head_dim(model: dict[str, Any]) -> int:
    return int(model.get("index_head_dim", 128))


def dsv4_index_n_heads(model: dict[str, Any]) -> int:
    return int(model.get("index_n_heads", 64))


def dsv4_c4_indexer_enabled(args: argparse.Namespace) -> bool:
    return int(args.dsv4_compress_ratio) == 4


def dsv4_c128_indices_width(pages_per_seq: int, page_size: int) -> int:
    return ceil_align(
        int(pages_per_seq) * (int(page_size) // 128), DSV4_PAGE_INDEX_ALIGNMENT
    )


def dsv4_c4_indexer_state_pool_size(max_tokens_per_seq: int) -> int:
    ring_size = 8
    state_slots = (
        math.ceil(int(max_tokens_per_seq) / DSV4_SWA_WINDOW) * ring_size + ring_size + 1
    )
    return ceil_align(state_slots, 4)


def dsv4_c4_indexer_state_dim(model: dict[str, Any]) -> int:
    return 2 * 2 * dsv4_index_head_dim(model)


def dsv4_rope_head_dim(model: dict[str, Any]) -> int:
    return int(model.get("qk_rope_head_dim", 64))


def dsv4_mla_head_dim(model: dict[str, Any]) -> int:
    # DSV4 attention Q uses qk_nope + qk_rope. The value/cache rank is kv_lora_rank and may differ.
    return int(
        model.get("qk_nope_head_dim", model.get("head_dim", 512))
    ) + dsv4_rope_head_dim(model)


def dsv4_qk_nope_head_dim(model: dict[str, Any]) -> int:
    return int(
        model.get(
            "qk_nope_head_dim", dsv4_mla_head_dim(model) - dsv4_rope_head_dim(model)
        )
    )


def dsv4_kv_cache_storage_dim(args: argparse.Namespace, model: dict[str, Any]) -> int:
    nope_dim = dsv4_qk_nope_head_dim(model)
    rope_dim = dsv4_rope_head_dim(model)
    if is_requested_fp8(args):
        # DeepSeekV4SingleKVPool stores 448 FP8 nope bytes, 64 BF16 rope values,
        # one FP8 scale byte per 64 nope values, and one padding scale byte.
        scale_bytes = nope_dim // 64 + 1
        return nope_dim + rope_dim * DTYPE_BYTES["bf16"] + scale_bytes
    return nope_dim + rope_dim


def dsv4_kv_cache_page_bytes(
    args: argparse.Namespace, model: dict[str, Any], page_size: int
) -> int:
    row_bytes = dsv4_kv_cache_storage_dim(args, model) * dsv4_kv_cache_element_bytes(
        args
    )
    if is_requested_fp8(args):
        return ceil_align(row_bytes * int(page_size), 576)
    return row_bytes * int(page_size)


def dsv4_kv_cache_storage_dtype(args: argparse.Namespace) -> str:
    return (
        "uint8_dsv4_fp8_nope_bf16_rope_packed"
        if is_requested_fp8(args)
        else activation_dtype_name(args, "flashmla-kvcache-e2e")
    )


def dsv4_kv_cache_element_bytes(args: argparse.Namespace) -> int:
    return (
        1
        if is_requested_fp8(args)
        else DTYPE_BYTES[activation_dtype_name(args, "flashmla-kvcache-e2e")]
    )


def dsv4_packed_cache_bytes_per_page(
    args: argparse.Namespace, model: dict[str, Any], page_size: int
) -> int:
    if page_size <= 0:
        raise ValueError(f"DSV4 page_size must be positive, got {page_size}")
    return (
        page_size
        * dsv4_kv_cache_storage_dim(args, model)
        * dsv4_kv_cache_element_bytes(args)
    )


def dsv4_validate_e2e_geometry(
    args: argparse.Namespace, model: dict[str, Any], atp: int
) -> None:
    if int(model.get("head_dim", 512)) != 512:
        raise ValueError(
            f"DeepSeek V4 FlashMLA path expects head_dim=512, got {model.get('head_dim')}"
        )
    if int(model.get("num_key_value_heads", 1)) != 1:
        raise ValueError(
            f"DeepSeek V4 FlashMLA path expects num_key_value_heads=1, got {model.get('num_key_value_heads')}"
        )
    if args.swa_page_size != 256:
        raise ValueError(
            f"DeepseekV4AttnBackend in SGLang 0.5.13 asserts page_size=256, got --swa-page-size={args.swa_page_size}"
        )
    if (
        args.chunked_prefill_size
        and args.chunked_prefill_size > 0
        and args.chunked_prefill_size % args.swa_page_size != 0
    ):
        raise ValueError(
            f"--chunked-prefill-size={args.chunked_prefill_size} must be divisible by --swa-page-size={args.swa_page_size} "
            "for the DSV4 production-like chunked prefill path"
        )
    dsv4_local_q_heads(model, atp)


def dsv4_supported_attention_dp_sizes(
    args: argparse.Namespace, supported_attn_tp_sizes: list[int]
) -> list[int]:
    values = {
        args.tp_size // atp
        for atp in supported_attn_tp_sizes
        if atp > 0 and args.tp_size > 0 and args.tp_size % atp == 0
    }
    return sorted(v for v in values if v > 0)


def dsv4_sparse_decode_plan(
    args: argparse.Namespace, model: dict[str, Any], atp: int | None = None
) -> dict[str, Any]:
    atp = attn_tp_size(args) if atp is None else atp
    local_q_heads = dsv4_local_q_heads(model, atp)
    flashmla_q_heads = dsv4_flashmla_q_heads(model, atp)
    requested_kernel = args.dsv4_attention_kernel
    entrypoint = DSV4_ATTENTION_KERNEL_ENTRYPOINTS[requested_kernel]
    supported = DSV4_FLASHMLA_SPARSE_DECODE_KNOWN_Q_HEADS
    supported_atp = dsv4_supported_flashmla_attn_tp_sizes(model, supported)
    supported_attention_dp = dsv4_supported_attention_dp_sizes(args, supported_atp)
    unsupported_reason = None
    if requested_kernel == "flashmla" and flashmla_q_heads not in supported:
        if supported_atp:
            suggested = (
                f"\nSupported effective attn_tp_size values for this model shape: {supported_atp}."
                f"\nEquivalent attention_dp_size values with current tp_size={args.tp_size}: {supported_attention_dp}."
            )
        else:
            suggested = "\nNo supported --attn-tp-size can be inferred for this model shape with production FlashMLA padded Q input."
        unsupported_reason = (
            "DeepSeek V4 FlashMLA sparse decode preflight rejected the production FlashMLA Q-head layout.\n"
            f"  local_q_heads_before_output_slice: {local_q_heads}\n"
            f"  flashmla_q_heads: {flashmla_q_heads}\n"
            f"  q_padded_to_full_heads: {dsv4_q_padded_to_full_heads(model, atp)}\n"
            f"  num_attention_heads: {model.get('num_attention_heads')}\n"
            f"  attn_tp_size: {atp}\n"
            f"  enable_dp_attention: {bool(args.enable_dp_attention)}\n"
            f"  attention_dp_size: {attention_dp_size(args)}\n"
            f"  supported h_q values in the installed SGLang sparse_decode_fwd path: {list(supported)}\n"
            "The benchmark will not switch to generic sparse-flash because that would measure a different attention mechanism."
            + suggested
        )
    impl_name = f"dsv4_{requested_kernel}_sparse_decode"
    return {
        "dsv4_attention_kernel": requested_kernel,
        "dsv4_attention_kernel_entrypoint": entrypoint,
        "decode_impl_requested": impl_name,
        "decode_impl_effective": impl_name if unsupported_reason is None else None,
        "decode_supported": unsupported_reason is None,
        "decode_unsupported_reason": unsupported_reason,
        "decode_fallback_reason": None
        if unsupported_reason is None
        else "No equivalent DeepSeek V4 sparse decode fallback exists without changing model geometry or attention semantics.",
        "known_sparse_decode_q_heads": list(supported),
        "supported_decode_attn_tp_size": supported_atp,
        "supported_decode_attention_dp_size": supported_attention_dp,
        "local_q_heads": local_q_heads,
        "flashmla_q_heads": flashmla_q_heads,
        "q_padded_to_full_heads": dsv4_q_padded_to_full_heads(model, atp),
        "q_padded_for_flashmla": dsv4_q_padded_for_flashmla(model, atp),
        "production_output_slice_heads": local_q_heads,
    }


def dsv4_sparse_prefill_support_plan(
    args: argparse.Namespace, model: dict[str, Any], atp: int
) -> dict[str, Any]:
    local_q_heads = dsv4_local_q_heads(model, atp)
    flashmla_q_heads = dsv4_flashmla_q_heads(model, atp)
    supported = DSV4_FLASHMLA_SPARSE_PREFILL_KNOWN_Q_HEADS
    supported_atp = dsv4_supported_flashmla_attn_tp_sizes(model, supported)
    supported_attention_dp = dsv4_supported_attention_dp_sizes(args, supported_atp)
    unsupported_reason = None
    if flashmla_q_heads not in supported:
        if supported_atp:
            suggested = (
                f"\nSupported effective attn_tp_size values for this model shape: {supported_atp}."
                f"\nEquivalent attention_dp_size values with current tp_size={args.tp_size}: {supported_attention_dp}."
            )
        else:
            suggested = "\nNo supported --attn-tp-size can be inferred for this model shape with production FlashMLA padded Q input."
        unsupported_reason = (
            "DeepSeek V4 FlashMLA sparse prefill preflight rejected the production FlashMLA Q-head layout.\n"
            f"  local_q_heads_before_output_slice: {local_q_heads}\n"
            f"  flashmla_q_heads: {flashmla_q_heads}\n"
            f"  q_padded_to_full_heads: {dsv4_q_padded_to_full_heads(model, atp)}\n"
            f"  num_attention_heads: {model.get('num_attention_heads')}\n"
            f"  attn_tp_size: {atp}\n"
            f"  enable_dp_attention: {bool(args.enable_dp_attention)}\n"
            f"  attention_dp_size: {attention_dp_size(args)}\n"
            f"  supported h_q values in the standalone flash_mla_sparse_fwd path: {list(supported)}\n"
            "The benchmark will not switch to generic sparse-flash because that would measure a different attention mechanism."
            + suggested
        )
    return {
        "sparse_prefill_supported": unsupported_reason is None,
        "sparse_prefill_unsupported_reason": unsupported_reason,
        "known_sparse_prefill_q_heads": list(supported),
        "supported_prefill_attn_tp_size": supported_atp,
        "supported_prefill_attention_dp_size": supported_attention_dp,
        "local_q_heads": local_q_heads,
        "flashmla_q_heads": flashmla_q_heads,
        "q_padded_to_full_heads": dsv4_q_padded_to_full_heads(model, atp),
        "q_padded_for_flashmla": dsv4_q_padded_for_flashmla(model, atp),
        "production_output_slice_heads": local_q_heads,
    }


def dsv4_prefill_plan(
    args: argparse.Namespace, model: dict[str, Any], atp: int | None = None
) -> dict[str, Any]:
    """Choose the DSV4 E2E prefill path without changing model dimensions.

    SGLang's DeepseekV4AttentionMLA pads per-rank Q to 64 heads when
    effective attention TP is greater than one and local heads fit in 64;
    otherwise it pads to full num_attention_heads. The backend output is then
    sliced back to the local heads. Support checks below use that FlashMLA
    input geometry, not the local post-slice head count.
    """
    atp = attn_tp_size(args) if atp is None else atp
    dsv4_validate_e2e_geometry(args, model, atp)
    local_q_heads = dsv4_local_q_heads(model, atp)
    flashmla_q_heads = dsv4_flashmla_q_heads(model, atp)
    in_len = input_len(args)
    requested_chunk = int(args.chunked_prefill_size or 0)
    fallback_reason = None
    if requested_chunk <= 0:
        prefill_chunk_tokens = in_len
        requested_impl = "flashmla_sparse_prefill"
    else:
        prefill_chunk_tokens = min(in_len, requested_chunk)
        requested_impl = (
            "flashmla_sparse_prefill"
            if prefill_chunk_tokens > DSV4_SPARSE_PREFILL_QUERY_THRESHOLD
            else "flashmla_kvcache_chunked"
        )

    effective_impl = requested_impl
    if (
        requested_impl == "flashmla_sparse_prefill"
        and flashmla_q_heads not in DSV4_FLASHMLA_SPARSE_PREFILL_KNOWN_Q_HEADS
    ):
        fallback_chunk = min(in_len, DSV4_DEFAULT_CHUNKED_PREFILL_SIZE)
        fallback_chunk = max(
            args.swa_page_size,
            (fallback_chunk // args.swa_page_size) * args.swa_page_size,
        )
        fallback_chunk = min(in_len, fallback_chunk)
        if fallback_chunk > DSV4_SPARSE_PREFILL_QUERY_THRESHOLD:
            fallback_chunk = (
                DSV4_SPARSE_PREFILL_QUERY_THRESHOLD // args.swa_page_size
            ) * args.swa_page_size
        if fallback_chunk <= 0:
            fallback_chunk = min(in_len, args.swa_page_size)
        fallback_reason = (
            f"DeepSeek V4 requested a monolithic FlashMLA sparse-prefill chunk with flashmla_q_heads={flashmla_q_heads} "
            f"(local_q_heads_before_output_slice={local_q_heads}, attn_tp_size={atp}), but the installed "
            f"sparse_prefill_fwd build is known/specialized for h_q={list(DSV4_FLASHMLA_SPARSE_PREFILL_KNOWN_Q_HEADS)}. "
            f"Keeping the original model shape and falling back to FlashMLA kvcache chunked prefill with chunked_prefill_size={fallback_chunk}."
        )
        effective_impl = "flashmla_kvcache_chunked"
        prefill_chunk_tokens = fallback_chunk

    return {
        "dsv4_compression": args.dsv4_compression,
        "dsv4_compress_ratio": int(args.dsv4_compress_ratio),
        "dsv4_compress_ratio_requested": args.dsv4_compress_ratio_requested,
        "dsv4_compress_ratio_overridden": args.dsv4_compress_ratio_overridden,
        "prefill_impl_requested": requested_impl,
        "prefill_impl_effective": effective_impl,
        "prefill_fallback_reason": fallback_reason,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "prefill_num_chunks": len(
            prefill_schedule_from_chunk(in_len, prefill_chunk_tokens)
        ),
        "local_q_heads": local_q_heads,
        "flashmla_q_heads": flashmla_q_heads,
        "q_padded_to_full_heads": dsv4_q_padded_to_full_heads(model, atp),
        "q_padded_for_flashmla": dsv4_q_padded_for_flashmla(model, atp),
        "production_output_slice_heads": local_q_heads,
        "known_sparse_prefill_q_heads": list(
            DSV4_FLASHMLA_SPARSE_PREFILL_KNOWN_Q_HEADS
        ),
        "sparse_prefill_query_threshold": DSV4_SPARSE_PREFILL_QUERY_THRESHOLD,
    }


def is_requested_fp8(args: argparse.Namespace) -> bool:
    return args.dtype in {"fp8", "float8_e4m3fn"}


def activation_dtype_name(args: argparse.Namespace, kernel: str) -> str:
    # SGLang DSV4/GDN serving can be FP8 at weight/KV-cache level while most
    # attention kernels still consume BF16 activations. Keep the requested dtype
    # and the actually allocated activation dtype separate in JSON output.
    if is_requested_fp8(args):
        return "bf16"
    return args.dtype


def activation_dtype_bytes(args: argparse.Namespace, kernel: str) -> int:
    return DTYPE_BYTES[activation_dtype_name(args, kernel)]


def input_len(args: argparse.Namespace) -> int:
    return args.input_len or args.seq_len


def e2e_prefill_chunk_tokens(args: argparse.Namespace) -> int:
    in_len = input_len(args)
    requested = int(args.chunked_prefill_size or 0)
    if requested <= 0:
        return in_len
    return min(in_len, requested)


def prefill_schedule_from_chunk(in_len: int, chunk: int) -> list[int]:
    if chunk <= 0:
        raise ValueError(f"prefill chunk must be positive, got {chunk}")
    full, tail = divmod(in_len, chunk)
    return [chunk] * full + ([tail] if tail else [])


def e2e_prefill_schedule(args: argparse.Namespace) -> list[int]:
    return prefill_schedule_from_chunk(input_len(args), e2e_prefill_chunk_tokens(args))


def require_multi_chunk_e2e_schedule(
    kernel: str, schedule: list[int], args: argparse.Namespace
) -> list[int]:
    if len(schedule) < 2:
        raise RuntimeError(
            f"{kernel} E2E requires more than one prefill chunk for a meaningful chunked-prefill comparison: "
            f"input_len={input_len(args)}, chunked_prefill_size={args.chunked_prefill_size}, schedule={schedule}."
        )
    return schedule


def numel(shape: tuple[int, ...] | list[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def bytes_for(shape: tuple[int, ...] | list[int], bytes_per_element: int) -> int:
    return numel(shape) * bytes_per_element


def model_hidden_size(model: dict[str, Any]) -> int:
    return int(
        model.get("hidden_size")
        or (int(model.get("num_attention_heads", 1)) * int(model.get("head_dim", 128)))
    )


def dsv4_o_lora_rank(model: dict[str, Any]) -> int:
    return int(model.get("o_lora_rank", 1024))


def dsv4_local_o_groups(model: dict[str, Any], atp: int) -> int:
    return require_divisible("o_groups", int(model.get("o_groups", 1)), atp)


def dsv4_output_projection_shapes(
    args: argparse.Namespace, model: dict[str, Any], info: dict[str, Any]
) -> dict[str, list[int]]:
    atp = attn_tp_size(args)
    local_heads = int(
        info.get("production_output_slice_heads") or info.get("local_q_heads")
    )
    local_groups = dsv4_local_o_groups(model, atp)
    vdim = dsv4_kv_lora_rank(model)
    if local_heads % local_groups != 0:
        raise ValueError(
            f"local_q_heads={local_heads} must be divisible by local_o_groups={local_groups}"
        )
    per_group_dim = (local_heads // local_groups) * vdim
    o_lora_rank = dsv4_o_lora_rank(model)
    hidden = model_hidden_size(model)
    return {
        "dsv4_wo_a_weight": [local_groups, o_lora_rank, per_group_dim],
        "dsv4_wo_b_weight_shard": [local_groups * o_lora_rank, hidden],
        "dsv4_projected_prefill_hidden": [
            args.batch_size * int(info.get("prefill_chunk_tokens", input_len(args))),
            hidden,
        ],
        "dsv4_projected_decode_hidden_steps": [
            args.output_len,
            args.batch_size,
            hidden,
        ],
    }


def linear_output_projection_shapes(
    args: argparse.Namespace, model: dict[str, Any]
) -> dict[str, list[int]]:
    _, hv, _, v, _ = linear_shapes(args, model)
    hidden = model_hidden_size(model)
    decode_b = args.decode_batch_size or args.batch_size
    return {
        "linear_out_proj_weight_shard": [hv * v, hidden],
        "linear_projected_prefill_hidden": [
            args.batch_size * e2e_prefill_chunk_tokens(args),
            hidden,
        ],
        "linear_projected_decode_hidden_steps": [args.output_len, decode_b, hidden],
    }


def gib(value: int | float) -> float:
    return float(value) / (1024.0**3)


def guard_estimate(args: argparse.Namespace, case: str, estimated_bytes: int) -> None:
    if args.max_input_gb > 0 and estimated_bytes > args.max_input_gb * 1024**3:
        raise RuntimeError(
            f"{case} estimated live tensor bytes {gib(estimated_bytes):.2f} GiB "
            f"exceeds --max-input-gb={args.max_input_gb}"
        )


def linear_backends(args: argparse.Namespace) -> tuple[str, str]:
    prefill = args.linear_attn_prefill_backend or args.linear_attn_backend
    decode = args.linear_attn_decode_backend or args.linear_attn_backend
    return prefill, decode


def ceil_align_int(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def normalize_kv_cache_dtype(value: str, *, auto: str) -> str:
    dtype = (value or "auto").lower()
    if dtype == "auto":
        dtype = auto
    if dtype == "bf16":
        dtype = "bfloat16"
    return dtype


def resolve_mla_kv_cache_dtype(
    args: argparse.Namespace, torch_major: int | None = None
) -> str:
    _ = torch_major
    return normalize_kv_cache_dtype(args.mla_kv_cache_dtype, auto="bfloat16")


def resolve_mla_attention_backend(
    args: argparse.Namespace, torch_major: int | None = None
) -> str:
    backend = str(args.mla_attention_backend)
    if backend != "auto":
        return backend
    # ServerArgs auto-selects trtllm_mla on Blackwell for DeepSeek V3/R1;
    # Hopper falls through to the FA3 MLA path. For shape summaries without
    # CUDA access, assume the benchmark target H100/Hopper platform.
    return "trtllm_mla" if torch_major is not None and torch_major >= 10 else "fa3"


def resolve_dsa_backend_defaults(
    args: argparse.Namespace, torch_major: int | None = None
) -> tuple[str, str, str]:
    major = 9 if torch_major is None else int(torch_major)
    kv_cache_dtype = normalize_kv_cache_dtype(
        args.dsa_kv_cache_dtype,
        auto=("fp8_e4m3" if major >= 10 else "bfloat16"),
    )
    prefill = None if args.dsa_prefill_backend == "auto" else args.dsa_prefill_backend
    decode = None if args.dsa_decode_backend == "auto" else args.dsa_decode_backend
    if kv_cache_dtype == "fp8_e4m3":
        default_prefill = "trtllm" if major >= 10 else "flashmla_kv"
        default_decode = "trtllm" if major >= 10 else "flashmla_kv"
    else:
        default_prefill = "flashmla_sparse"
        default_decode = "trtllm" if major >= 10 else "fa3"
    return prefill or default_prefill, decode or default_decode, kv_cache_dtype


def dsa_layer_skips_topk_from_model(model: dict[str, Any], layer_id: int) -> bool:
    pattern = model.get("index_topk_pattern")
    if pattern is not None:
        return layer_id < len(pattern) and pattern[layer_id] == "S"
    freq = model.get("index_topk_freq", 1) or 1
    offset = model.get("index_skip_topk_offset")
    if offset is not None:
        return max(int(layer_id) - int(offset) + 1, 0) % int(freq) != 0
    return max(int(layer_id) - 1, 0) % int(freq) != 0


def shape_summary(
    kernel: str, args: argparse.Namespace, model: dict[str, Any]
) -> dict[str, Any]:
    atp = attn_tp_size(args)
    seq = (
        input_len(args)
        if (
            kernel.endswith("e2e")
            or "prefill" in kernel
            or kernel.startswith("sparse")
            or kernel == "linear-fused-qkv-split"
            or kernel.startswith("dsv4-q")
            or kernel == "dsv4-k-norm-rope-flashmla"
            or kernel.startswith("dsv4-topk")
        )
        else args.seq_len
    )
    total_tokens = args.batch_size * seq
    dbytes = activation_dtype_bytes(args, kernel)

    if kernel == "linear-gdn-e2e":
        h = require_divisible(
            "linear_num_key_heads", int(model["linear_num_key_heads"]), atp
        )
        hv = require_divisible(
            "linear_num_value_heads", int(model["linear_num_value_heads"]), atp
        )
        k = int(model["linear_key_head_dim"])
        v = int(model["linear_value_head_dim"])
        decode_b = args.decode_batch_size or args.batch_size
        pool = args.pool_size or max(args.batch_size, decode_b, 1)
        state_bytes = DTYPE_BYTES[args.state_dtype]
        chunk_tokens = e2e_prefill_chunk_tokens(args)
        chunk_total = args.batch_size * chunk_tokens
        hidden = model_hidden_size(model)
        mixed_dim = 2 * h * k + hv * v
        qkvz_dim = mixed_dim + hv * v
        ba_dim = 2 * hv
        conv_kernel = int(model.get("linear_conv_kernel_dim", 4))
        conv_state_width = max(conv_kernel - 1, 0)
        shapes = {
            "prefill_hidden_full": [args.batch_size, input_len(args), hidden],
            "prefill_hidden_chunk": [chunk_total, hidden],
            "in_proj_qkvz_weight": [hidden, qkvz_dim],
            "in_proj_ba_weight": [hidden, ba_dim],
            "prefill_projected_qkvz_chunk": [chunk_total, qkvz_dim],
            "prefill_projected_ba_chunk": [chunk_total, ba_dim],
            "prefill_mixed_qkv_chunk": [chunk_total, mixed_dim],
            "prefill_q_chunk": [1, chunk_total, h, k],
            "prefill_k_chunk": [1, chunk_total, h, k],
            "prefill_v_chunk": [1, chunk_total, hv, v],
            "prefill_z_chunk": [chunk_total, hv, v],
            "prefill_g_chunk": [chunk_total, hv],
            "prefill_beta_chunk": [chunk_total, hv],
            "decode_hidden_steps": [args.output_len, decode_b, hidden],
            "decode_projected_qkvz_steps": [args.output_len, decode_b, qkvz_dim],
            "decode_projected_ba_steps": [args.output_len, decode_b, ba_dim],
            "decode_mixed_qkv_steps": [args.output_len, decode_b, mixed_dim],
            "decode_z_steps": [args.output_len, decode_b, hv, v],
            "decode_a_steps": [args.output_len, decode_b, hv],
            "decode_b_steps": [args.output_len, decode_b, hv],
            "decode_q_step": [1, decode_b, h, k],
            "decode_k_step": [1, decode_b, h, k],
            "decode_v_step": [1, decode_b, hv, v],
            "A_log": [hv],
            "dt_bias": [hv],
            "state": [pool, hv, v, k],
            "conv_weight": [mixed_dim, conv_kernel],
            "conv_state": [pool, mixed_dim, conv_state_width],
            **linear_output_projection_shapes(args, model),
        }
        persistent = (
            bytes_for(shapes["state"], state_bytes)
            + bytes_for(shapes["conv_state"], dbytes)
            + bytes_for(shapes["conv_weight"], dbytes)
            + bytes_for(shapes["in_proj_qkvz_weight"], dbytes)
            + bytes_for(shapes["in_proj_ba_weight"], dbytes)
            + bytes_for(shapes["linear_out_proj_weight_shard"], dbytes)
        )
        transient = (
            bytes_for(shapes["prefill_hidden_full"], dbytes)
            + bytes_for(shapes["prefill_projected_qkvz_chunk"], dbytes)
            + bytes_for(shapes["prefill_projected_ba_chunk"], dbytes)
            + bytes_for(shapes["prefill_mixed_qkv_chunk"], dbytes)
            + bytes_for(shapes["prefill_z_chunk"], dbytes)
            + bytes_for(shapes["prefill_q_chunk"], dbytes)
            + bytes_for(shapes["prefill_k_chunk"], dbytes)
            + bytes_for(shapes["prefill_v_chunk"], dbytes)
            + bytes_for(shapes["prefill_g_chunk"], 4)
            + bytes_for(shapes["prefill_beta_chunk"], 4)
            + bytes_for(shapes["decode_hidden_steps"], dbytes)
            + bytes_for(shapes["decode_projected_qkvz_steps"], dbytes)
            + bytes_for(shapes["decode_projected_ba_steps"], dbytes)
            + bytes_for(shapes["decode_mixed_qkv_steps"], dbytes)
            + bytes_for(shapes["decode_z_steps"], dbytes)
            + bytes_for(shapes["decode_a_steps"], dbytes)
            + bytes_for(shapes["decode_b_steps"], dbytes)
            + bytes_for(shapes["decode_q_step"], dbytes)
            + bytes_for(shapes["decode_k_step"], dbytes)
            + bytes_for(shapes["decode_v_step"], dbytes)
            + bytes_for(shapes["A_log"], 4)
            + bytes_for(shapes["dt_bias"], 4)
            + bytes_for(shapes["linear_projected_prefill_hidden"], dbytes)
            + bytes_for(shapes["linear_projected_decode_hidden_steps"], dbytes)
        )
        estimated = persistent + transient
        prefill_backend, decode_backend = linear_backends(args)
        return {
            "activation_dtype": activation_dtype_name(args, kernel),
            "state_dtype": args.state_dtype,
            "linear_conv_kernel_dim": conv_kernel,
            "linear_conv_dim": mixed_dim,
            "linear_prefill_backend_requested": prefill_backend,
            "linear_decode_backend_requested": decode_backend,
            "shapes": shapes,
            "estimated_persistent_bytes": persistent,
            "estimated_persistent_gib": gib(persistent),
            "estimated_transient_bytes": transient,
            "estimated_transient_gib": gib(transient),
            "estimated_live_bytes": estimated,
            "estimated_live_gib": gib(estimated),
        }

    if kernel in ("deepseek-v3-mla-e2e", "glm-dsa-e2e"):
        is_dsa = kernel == "glm-dsa-e2e"
        local_heads = require_divisible(
            "num_attention_heads", int(model["num_attention_heads"]), atp
        )
        hidden = model_hidden_size(model)
        qk_nope = int(model.get("qk_nope_head_dim", model.get("head_dim", 128)))
        rope_dim = int(model.get("qk_rope_head_dim", 0))
        qk_head_dim = qk_nope + rope_dim
        v_head_dim = int(model.get("v_head_dim", model.get("head_dim", qk_head_dim)))
        kv_lora_rank = int(model.get("kv_lora_rank", v_head_dim))
        q_lora_rank = int(model.get("q_lora_rank", 0))
        page_size = int(args.dsa_page_size if is_dsa else args.mla_page_size)
        total_context = input_len(args) + args.output_len
        pool_tokens = args.batch_size * (total_context + page_size) + page_size
        chunk_tokens = e2e_prefill_chunk_tokens(args)
        prefill_backend, decode_backend, kv_cache_dtype = (
            resolve_dsa_backend_defaults(args, torch_major=None)
            if is_dsa
            else (None, None, resolve_mla_kv_cache_dtype(args))
        )
        mla_backend = (
            None if is_dsa else resolve_mla_attention_backend(args, torch_major=None)
        )
        if is_dsa and kv_cache_dtype == "fp8_e4m3":
            num_tiles = kv_lora_rank // DSV4_NSA_QUANT_BLOCK_SIZE
            kv_cache_dim = kv_lora_rank + num_tiles * 4 + rope_dim * 2
            kv_cache_elem_bytes = 1
        else:
            kv_cache_dim = kv_lora_rank + rope_dim
            kv_cache_elem_bytes = DTYPE_BYTES["bf16"]
        shapes = {
            "hidden_prefill_full": [args.batch_size, input_len(args), hidden],
            "hidden_prefill_chunk": [args.batch_size * chunk_tokens, hidden],
            "hidden_decode_steps": [args.output_len, args.batch_size, hidden],
            "fused_qkv_a_proj_weight": [q_lora_rank + kv_lora_rank + rope_dim, hidden],
            "q_b_proj_weight_shard": [local_heads * qk_head_dim, q_lora_rank],
            "kv_b_proj_weight_shard": [
                local_heads * (qk_nope + v_head_dim),
                kv_lora_rank,
            ],
            "o_proj_weight_shard": [hidden, local_heads * v_head_dim],
            "mla_token_to_kv_pool": [pool_tokens, kv_cache_dim],
        }
        if is_dsa:
            index_head_dim = int(model.get("index_head_dim", 128))
            index_n_heads = int(model.get("index_n_heads", 1))
            index_topk = int(model.get("index_topk", 0))
            shapes.update(
                {
                    "dsa_index_wq_b_weight": [
                        index_n_heads * index_head_dim,
                        q_lora_rank,
                    ],
                    "dsa_index_wk_weight": [index_head_dim, hidden],
                    "dsa_index_weights_proj": [index_n_heads, hidden],
                    "dsa_index_k_with_scale_pool": [pool_tokens, index_head_dim + 4],
                    "dsa_topk_indices_chunk": [
                        args.batch_size * chunk_tokens,
                        ceil_align_int(index_topk, 2048),
                    ],
                    "dsa_topk_indices_decode_step": [
                        args.batch_size,
                        ceil_align_int(index_topk, 2048),
                    ],
                }
            )
        persistent = (
            bytes_for(shapes["fused_qkv_a_proj_weight"], dbytes)
            + bytes_for(shapes["q_b_proj_weight_shard"], dbytes)
            + bytes_for(shapes["kv_b_proj_weight_shard"], dbytes)
            + bytes_for(shapes["o_proj_weight_shard"], dbytes)
            + bytes_for(shapes["mla_token_to_kv_pool"], kv_cache_elem_bytes)
        )
        if is_dsa:
            persistent += (
                bytes_for(shapes["dsa_index_wq_b_weight"], dbytes)
                + bytes_for(shapes["dsa_index_wk_weight"], dbytes)
                + bytes_for(shapes["dsa_index_weights_proj"], dbytes)
                + bytes_for(shapes["dsa_index_k_with_scale_pool"], 1)
            )
        transient = (
            bytes_for(shapes["hidden_prefill_full"], dbytes)
            + bytes_for(shapes["hidden_prefill_chunk"], dbytes)
            + bytes_for(shapes["hidden_decode_steps"], dbytes)
        )
        if is_dsa:
            transient += bytes_for(shapes["dsa_topk_indices_chunk"], 4) + bytes_for(
                shapes["dsa_topk_indices_decode_step"], 4
            )
        estimated = persistent + transient
        return {
            "attn_tp_size": atp,
            "local_q_heads": local_heads,
            "activation_dtype": activation_dtype_name(args, kernel),
            "qk_nope_head_dim": qk_nope,
            "qk_rope_head_dim": rope_dim,
            "qk_head_dim": qk_head_dim,
            "v_head_dim": v_head_dim,
            "kv_lora_rank": kv_lora_rank,
            "q_lora_rank": q_lora_rank,
            "mla_attention_backend_effective": mla_backend,
            "mla_kv_cache_dtype_effective": None if is_dsa else kv_cache_dtype,
            "dsa_prefill_backend_effective": prefill_backend,
            "dsa_decode_backend_effective": decode_backend,
            "dsa_topk_backend": args.dsa_topk_backend if is_dsa else None,
            "dsa_kv_cache_dtype_effective": kv_cache_dtype if is_dsa else None,
            "dsa_indexer_enabled": is_dsa,
            "dsa_layer_id": int(args.glm_dsa_layer_id) if is_dsa else None,
            "dsa_layer_skips_topk": bool(
                dsa_layer_skips_topk_from_model(model, int(args.glm_dsa_layer_id))
            )
            if is_dsa
            else None,
            "dsa_next_layer_skips_topk": bool(
                dsa_layer_skips_topk_from_model(model, int(args.glm_dsa_layer_id) + 1)
            )
            if is_dsa
            else None,
            "index_topk": int(model.get("index_topk", 0)) if is_dsa else None,
            "index_topk_freq": model.get("index_topk_freq") if is_dsa else None,
            "index_skip_topk_offset": model.get("index_skip_topk_offset")
            if is_dsa
            else None,
            "page_size": page_size,
            "kv_cache_dim": kv_cache_dim,
            "kv_cache_elem_bytes": kv_cache_elem_bytes,
            "shapes": shapes,
            "estimated_persistent_bytes": persistent,
            "estimated_persistent_gib": gib(persistent),
            "estimated_transient_bytes": transient,
            "estimated_transient_gib": gib(transient),
            "estimated_live_bytes": estimated,
            "estimated_live_gib": gib(estimated),
        }

    if kernel.startswith("dsv4-c"):
        ratio = 4 if "-c4-" in kernel else 128
        head_dim = int(model.get("head_dim", 512))
        last_dim = head_dim * (4 if ratio == 4 else 2)
        batch = (
            (args.decode_batch_size or args.batch_size)
            if "decode" in kernel
            else args.batch_size
        )
        actual_seq = args.seq_len if "decode" in kernel else input_len(args)
        capacity_seq = args.seq_len
        actual_tokens = batch if "decode" in kernel else args.batch_size * actual_seq
        if args.dsv4_context == "legacy":
            num_pages = batch * (2 if ratio == 4 else 1)
            metadata_shapes = {}
        else:
            ring_size = args.ring_size or (8 if ratio == 4 else 128)
            pages_per_req = math.ceil(capacity_seq / args.swa_page_size)
            stride = pages_per_req * args.swa_page_size
            capacity = max(batch, 16)
            num_pages = (
                batch * pages_per_req * ring_size + args.swa_page_size
            ) // ratio + 1
            metadata_shapes = {
                "req_to_token": [capacity, stride],
                "full_to_swa": [capacity * stride],
            }
        shapes = {
            "ape": [8 if ratio == 4 else 128, head_dim],
            "pool": [num_pages, ratio, last_dim],
            **metadata_shapes,
        }
        shapes["kv_score_input"] = [actual_tokens, last_dim]
        estimated = (
            bytes_for(shapes["ape"], 4)
            + bytes_for(shapes["pool"], 4)
            + bytes_for(shapes["kv_score_input"], 4)
            + (bytes_for(shapes["req_to_token"], 4) if "req_to_token" in shapes else 0)
            + (bytes_for(shapes["full_to_swa"], 8) if "full_to_swa" in shapes else 0)
        )
        return {
            "attn_tp_size": atp,
            "compress_ratio": ratio,
            "shapes": shapes,
            "estimated_live_bytes": estimated,
            "estimated_live_gib": gib(estimated),
        }

    if kernel == "flashmla-sparse-prefill":
        support_plan = dsv4_sparse_prefill_support_plan(args, model, atp)
        warning_messages = []
        padding_warning = dsv4_dp_off_padding_warning(args, model, atp)
        if padding_warning:
            warning_messages.append(padding_warning)
        if support_plan.get("sparse_prefill_unsupported_reason"):
            warning_messages.append(
                "DSV4 FlashMLA standalone sparse-prefill unsupported: "
                + str(support_plan["sparse_prefill_unsupported_reason"])
            )
        local_hq = dsv4_local_q_heads(model, atp)
        hq = dsv4_flashmla_q_heads(model, atp)
        hk = 1
        d = dsv4_mla_head_dim(model)
        kv_lora_rank = dsv4_kv_lora_rank(model)
        sparse_topk = dsv4_topk(args, model)
        topk = dsv4_sparse_prefill_combined_topk(args, model)
        shapes = {
            "q": [total_tokens, hq, d],
            "kv": [total_tokens, hk, d],
            "indices": [total_tokens, hk, topk],
            "out": [total_tokens, hq, kv_lora_rank],
        }
        estimated = (
            bytes_for(shapes["q"], dbytes)
            + bytes_for(shapes["kv"], dbytes)
            + bytes_for(shapes["out"], dbytes)
            + bytes_for(shapes["indices"], 4)
        )
        return {
            "attn_tp_size": atp,
            "local_q_heads": local_hq,
            "flashmla_q_heads": hq,
            "q_padded_to_full_heads": dsv4_q_padded_to_full_heads(model, atp),
            "q_padded_for_flashmla": dsv4_q_padded_for_flashmla(model, atp),
            "production_output_slice_heads": local_hq,
            **support_plan,
            "local_kv_heads": hk,
            "topk": topk,
            "dsv4_sparse_topk": sparse_topk,
            "swa_topk": DSV4_SWA_WINDOW,
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": dsv4_rope_head_dim(model),
            "mla_head_dim": d,
            "activation_dtype": activation_dtype_name(args, kernel),
            "warning_messages": warning_messages,
            "shapes": shapes,
            "estimated_live_bytes": estimated,
            "estimated_live_gib": gib(estimated),
        }

    if kernel == "flashmla-kvcache-e2e":
        plan = dsv4_prefill_plan(args, model, atp)
        decode_plan = dsv4_sparse_decode_plan(args, model, atp)
        warning_messages = []
        padding_warning = dsv4_dp_off_padding_warning(args, model, atp)
        if padding_warning:
            warning_messages.append(padding_warning)
        if plan.get("prefill_fallback_reason"):
            warning_messages.append(
                "DSV4 FlashMLA prefill fallback: "
                + str(plan["prefill_fallback_reason"])
            )
        if decode_plan.get("decode_unsupported_reason"):
            warning_messages.append(
                "DSV4 FlashMLA sparse decode unsupported: "
                + str(decode_plan["decode_unsupported_reason"])
            )
        hq = int(plan["flashmla_q_heads"])
        d = dsv4_mla_head_dim(model)
        kv_lora_rank = dsv4_kv_lora_rank(model)
        q_lora_rank = dsv4_q_lora_rank(model)
        rope_dim = dsv4_rope_head_dim(model)
        index_heads = dsv4_index_n_heads(model)
        index_head_dim = dsv4_index_head_dim(model)
        hidden = model_hidden_size(model)
        topk = dsv4_topk(args, model)
        swa_topk = DSV4_SWA_WINDOW
        sparse_prefill_topk = dsv4_sparse_prefill_combined_topk(args, model)
        ratio = int(args.dsv4_compress_ratio)
        page_size = args.swa_page_size
        per_seq_tokens = input_len(args) + args.output_len
        pages_per_seq = math.ceil(per_seq_tokens / page_size)
        pages = args.batch_size * pages_per_seq
        cache_tokens = pages * page_size
        chunk_tokens = int(plan["prefill_chunk_tokens"])
        chunk_total_tokens = args.batch_size * chunk_tokens
        local_q_heads = int(plan["local_q_heads"])
        qk_nope_dim = dsv4_qk_nope_head_dim(model)
        cache_dim = dsv4_kv_cache_storage_dim(args, model)
        cache_elem_bytes = dsv4_kv_cache_element_bytes(args)
        cache_page_bytes = dsv4_kv_cache_page_bytes(args, model, page_size)
        cache_bytes = pages * cache_page_bytes
        c4_indexer_enabled = dsv4_c4_indexer_enabled(args)
        c4_page_size = page_size // 4
        c4_max_seq_len = pages_per_seq * c4_page_size
        c128_indices_width = dsv4_c128_indices_width(pages_per_seq, page_size)

        prefill_shapes = {
            "prefill_hidden_full": [args.batch_size, input_len(args), hidden],
            "prefill_hidden_chunk": [chunk_total_tokens, hidden],
            "prefill_q_lora_chunk": [chunk_total_tokens, q_lora_rank],
            "prefill_q_chunk": [chunk_total_tokens, 1, hq, d],
            "prefill_kv_bf16_chunk": [chunk_total_tokens, d],
            "prefill_indices_chunk": [chunk_total_tokens, 1, swa_topk],
            "dsv4_wq_a_weight": [hidden, q_lora_rank],
            "dsv4_wq_b_weight_shard": [q_lora_rank, local_q_heads * d],
            "dsv4_wkv_weight": [hidden, d],
            "dsv4_q_norm_weight": [q_lora_rank],
            "dsv4_kv_norm_weight": [d],
        }
        if c4_indexer_enabled:
            prefill_shapes.update(
                {
                    "prefill_indexer_q_lora_chunk": [chunk_total_tokens, q_lora_rank],
                    "prefill_indexer_hidden_chunk": [chunk_total_tokens, hidden],
                    "prefill_indexer_q_input_chunk": [
                        chunk_total_tokens,
                        index_heads,
                        index_head_dim,
                    ],
                    "prefill_indexer_weight_chunk": [chunk_total_tokens, index_heads],
                    "prefill_indexer_q_fp8_chunk": [
                        chunk_total_tokens,
                        index_heads,
                        index_head_dim,
                    ],
                    "prefill_indexer_scores_chunk": [
                        chunk_total_tokens,
                        c4_max_seq_len,
                    ],
                    "prefill_indexer_page_table_chunk": [
                        chunk_total_tokens,
                        pages_per_seq,
                    ],
                }
            )
        if plan["prefill_impl_effective"] == "flashmla_sparse_prefill":
            full_prefill_tokens = args.batch_size * input_len(args)
            prefill_shapes["sparse_prefill_q_full"] = [full_prefill_tokens, hq, d]
            prefill_shapes["sparse_prefill_kv_full"] = [full_prefill_tokens, 1, d]
            prefill_shapes["sparse_prefill_indices_full"] = [
                full_prefill_tokens,
                1,
                sparse_prefill_topk,
            ]

        shapes = {
            **prefill_shapes,
            "decode_hidden_steps": [args.output_len, args.batch_size, hidden],
            "decode_q_lora_steps": [args.output_len, args.batch_size, q_lora_rank],
            "decode_q_steps": [args.output_len, args.batch_size, 1, hq, d],
            "decode_kv_bf16_steps": [args.output_len, args.batch_size, d],
            "decode_indices_per_step": [args.batch_size, 1, swa_topk],
            "k_cache_flat": [pages, cache_page_bytes],
            "k_cache_paged_view": [pages, page_size, 1, cache_dim],
            **dsv4_output_projection_shapes(args, model, plan),
        }
        if c4_indexer_enabled:
            shapes.update(
                {
                    "decode_indexer_q_lora_steps": [
                        args.output_len,
                        args.batch_size,
                        1,
                        q_lora_rank,
                    ],
                    "decode_indexer_hidden_steps": [
                        args.output_len,
                        args.batch_size,
                        1,
                        hidden,
                    ],
                    "decode_indexer_scores_per_step": [args.batch_size, c4_max_seq_len],
                    "decode_indexer_page_table_per_step": [
                        args.batch_size,
                        pages_per_seq,
                    ],
                    "dsv4_indexer_wq_b_weight": [
                        q_lora_rank,
                        index_heads * index_head_dim,
                    ],
                    "dsv4_indexer_weight_proj": [hidden, index_heads],
                    "dsv4_indexer_compressor_wkv_gate_weight": [
                        dsv4_c4_indexer_state_dim(model),
                        hidden,
                    ],
                    "dsv4_indexer_compressor_ape": [8, index_head_dim],
                    "dsv4_indexer_compressor_norm_weight": [index_head_dim],
                    "indexer_compressor_state": [
                        dsv4_c4_indexer_state_pool_size(per_seq_tokens),
                        dsv4_c4_indexer_state_dim(model),
                    ],
                    "prefill_indexer_kv_score_chunk": [
                        chunk_total_tokens,
                        dsv4_c4_indexer_state_dim(model),
                    ],
                    "prefill_indexer_compressed_chunk": [
                        chunk_total_tokens,
                        index_head_dim,
                    ],
                    "decode_indexer_kv_score_steps": [
                        args.output_len,
                        args.batch_size,
                        dsv4_c4_indexer_state_dim(model),
                    ],
                    "decode_indexer_compressed_steps": [
                        args.output_len,
                        args.batch_size,
                        index_head_dim,
                    ],
                    "indexer_k_cache_flat": [
                        pages,
                        c4_page_size * (index_head_dim + 4),
                    ],
                    "indexer_k_cache_paged_view": [
                        pages,
                        c4_page_size,
                        1,
                        index_head_dim + 4,
                    ],
                }
            )
        extra_cache_bytes = 0
        if ratio in (4, 128):
            extra_page_size = page_size // ratio
            extra_cache_page_bytes = dsv4_kv_cache_page_bytes(
                args, model, extra_page_size
            )
            extra_cache_bytes = pages * extra_cache_page_bytes
            coff = 2 if ratio == 4 else 1
            compressor_ring_size = 8 if ratio == 4 else 128
            compressor_state_slots = ceil_align(
                pages * compressor_ring_size + compressor_ring_size + ratio + 1, ratio
            )
            shapes["extra_k_cache_flat"] = [pages, extra_cache_page_bytes]
            shapes["extra_k_cache_paged_view"] = [pages, extra_page_size, 1, cache_dim]
            shapes["dsv4_compressor_wkv_gate_weight"] = [2 * coff * d, hidden]
            shapes["dsv4_compressor_ape"] = [ratio, coff * d]
            shapes["dsv4_compressor_norm_weight"] = [d]
            shapes["dsv4_compressor_state"] = [compressor_state_slots, 2 * coff * d]
            shapes["prefill_compressor_kv_score_chunk"] = [
                chunk_total_tokens,
                2 * coff * d,
            ]
            shapes["prefill_compressed_kv_chunk"] = [
                math.ceil(chunk_total_tokens / ratio),
                d,
            ]
            shapes["decode_compressor_kv_score_steps"] = [
                args.output_len,
                args.batch_size,
                2 * coff * d,
            ]
            shapes["decode_compressed_kv_steps"] = [args.output_len, args.batch_size, d]
            if ratio == 4:
                shapes["prefill_extra_indices_chunk"] = [chunk_total_tokens, 1, topk]
                shapes["decode_extra_indices_per_step"] = [args.batch_size, 1, topk]
            else:
                shapes["prefill_extra_indices_chunk"] = [
                    chunk_total_tokens,
                    1,
                    c128_indices_width,
                ]
                shapes["decode_extra_indices_per_step"] = [
                    args.batch_size,
                    1,
                    c128_indices_width,
                ]

        def e2e_shape_bytes(name: str, shape: list[int]) -> int:
            if name.endswith("paged_view"):
                return 0
            if name.endswith("cache_flat"):
                return bytes_for(shape, cache_elem_bytes)
            if name.startswith(
                (
                    "prefill_hidden",
                    "prefill_q",
                    "prefill_kv",
                    "sparse_prefill_q",
                    "sparse_prefill_kv",
                    "decode_hidden",
                    "decode_q",
                    "decode_kv",
                    "dsv4_wq",
                    "dsv4_wkv",
                    "dsv4_q_norm",
                    "dsv4_wo",
                )
            ):
                return bytes_for(shape, dbytes)
            if "indexer_q_fp8" in name or name.startswith("indexer_k_cache"):
                return bytes_for(shape, 1)
            if (
                name.startswith(("indexer_compressor_state", "dsv4_compressor_state"))
                or "kv_score" in name
                or "compressed" in name
            ):
                return bytes_for(shape, 4)
            if name.startswith(
                (
                    "dsv4_indexer_compressor_ape",
                    "dsv4_indexer_compressor_norm_weight",
                    "dsv4_compressor_ape",
                    "dsv4_compressor_norm_weight",
                    "dsv4_kv_norm_weight",
                )
            ):
                return bytes_for(shape, 4)
            if name.startswith(
                (
                    "dsv4_indexer_compressor_wkv_gate_weight",
                    "dsv4_compressor_wkv_gate_weight",
                )
            ):
                return bytes_for(shape, dbytes)
            if name.startswith(
                (
                    "prefill_indexer_q_lora",
                    "prefill_indexer_hidden",
                    "prefill_indexer_q_input",
                    "prefill_indexer_weight",
                    "decode_indexer_q_lora",
                    "decode_indexer_hidden",
                    "dsv4_indexer",
                )
            ):
                return bytes_for(shape, dbytes)
            if name.startswith("dsv4_projected"):
                return bytes_for(shape, dbytes)
            return bytes_for(shape, 4)

        persistent = (
            cache_bytes
            + extra_cache_bytes
            + bytes_for(shapes["dsv4_wo_a_weight"], dbytes)
            + bytes_for(shapes["dsv4_wo_b_weight_shard"], dbytes)
        )
        transient = sum(
            e2e_shape_bytes(name, shape)
            for name, shape in shapes.items()
            if not name.endswith("cache_flat") and not name.endswith("paged_view")
        )
        estimated = persistent + transient
        return {
            "attn_tp_size": atp,
            **plan,
            **decode_plan,
            "local_kv_heads": 1,
            "topk": topk,
            "swa_topk": swa_topk,
            "extra_topk": topk
            if ratio == 4
            else (c128_indices_width if ratio == 128 else None),
            "sparse_prefill_topk": sparse_prefill_topk,
            "dsv4_c4_indexer_enabled": c4_indexer_enabled,
            "q_lora_rank": q_lora_rank,
            "index_n_heads": index_heads,
            "index_head_dim": index_head_dim,
            "c4_page_size": c4_page_size,
            "c4_max_seq_len": c4_max_seq_len,
            "c128_indices_width": c128_indices_width,
            "page_size": page_size,
            "pages_per_seq": pages_per_seq,
            "kv_cache_capacity_tokens": cache_tokens,
            "input_len": input_len(args),
            "output_len": args.output_len,
            "activation_dtype": activation_dtype_name(args, kernel),
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": rope_dim,
            "mla_head_dim": d,
            "kv_cache_dtype": dsv4_kv_cache_storage_dtype(args),
            "kv_cache_storage_dim": cache_dim,
            "qk_nope_head_dim": qk_nope_dim,
            "kv_cache_bytes_per_token": cache_dim * cache_elem_bytes,
            "kv_cache_page_bytes": cache_page_bytes,
            "main_kv_cache_bytes": cache_bytes,
            "extra_kv_cache_bytes": extra_cache_bytes,
            "kvcache_topk_length_enabled": True,
            "kvcache_topk_length_reason": "Matches DeepseekV4AttnBackend: pass swa_topk_lengths and extra_topk_lengths into FlashMLA sparse kvcache attention.",
            "kvcache_attn_sink_enabled": True,
            "warning_messages": warning_messages,
            "shapes": shapes,
            "estimated_persistent_bytes": persistent,
            "estimated_persistent_gib": gib(persistent),
            "estimated_transient_bytes": transient,
            "estimated_transient_gib": gib(transient),
            "estimated_live_bytes": estimated,
            "estimated_live_gib": gib(estimated),
        }

    if kernel in ("dsv4-topk-v1", "dsv4-topk-v2"):
        topk = args.sparse_topk or int(model.get("index_topk", 512))
        page_size = 64
        pages = math.ceil(seq / page_size)
        shapes = {
            "scores": [args.batch_size, seq],
            "seq_lens": [args.batch_size],
            "page_tables": [args.batch_size, pages],
            "out_page_indices": [args.batch_size, topk],
        }
        if kernel == "dsv4-topk-v1":
            shapes["out_raw_indices"] = [args.batch_size, topk]
        estimated = (
            bytes_for(shapes["scores"], 4)
            + bytes_for(shapes["seq_lens"], 4)
            + bytes_for(shapes["page_tables"], 4)
            + bytes_for(shapes["out_page_indices"], 4)
            + (
                bytes_for(shapes["out_raw_indices"], 4)
                if "out_raw_indices" in shapes
                else 0
            )
        )
        return {
            "attn_tp_size": atp,
            "topk": topk,
            "page_size": page_size,
            "shapes": shapes,
            "estimated_live_bytes": estimated,
            "estimated_live_gib": gib(estimated),
        }

    if kernel in ("dsv4-q-norm-rope", "dsv4-q-indexer-rope-hadamard-fp8"):
        if kernel == "dsv4-q-norm-rope":
            hq = require_divisible(
                "num_attention_heads", int(model["num_attention_heads"]), atp
            )
            d = int(model.get("index_head_dim", 128)) + int(
                model.get("qk_rope_head_dim", 64)
            )
        else:
            hq = int(model.get("index_n_heads", 64))
            d = int(model.get("index_head_dim", 128))
        rope_dim = int(model.get("qk_rope_head_dim", 64))
        shapes = {
            "q_input": [total_tokens, hq, d],
            "freqs_cis": [seq, rope_dim // 2],
            "positions": [total_tokens],
        }
        if kernel == "dsv4-q-norm-rope":
            shapes["q_output"] = [total_tokens, hq, d]
        else:
            shapes["weight"] = [total_tokens, hq]
            shapes["weights_out"] = [total_tokens, hq, 1]
            shapes["q_fp8"] = [total_tokens, hq, d]
        estimated = (
            bytes_for(shapes["q_input"], dbytes)
            + bytes_for(shapes["freqs_cis"], 8)
            + bytes_for(shapes["positions"], 4)
        )
        if "q_output" in shapes:
            estimated += bytes_for(shapes["q_output"], dbytes)
        if "weight" in shapes:
            estimated += bytes_for(shapes["weight"], dbytes) + bytes_for(
                shapes["weights_out"], 4
            )
        if "q_fp8" in shapes:
            estimated += bytes_for(shapes["q_fp8"], 1)
        return {
            "attn_tp_size": atp,
            "local_q_heads": hq,
            "rope_dim": rope_dim,
            "activation_dtype": activation_dtype_name(args, kernel),
            "shapes": shapes,
            "estimated_live_bytes": estimated,
            "estimated_live_gib": gib(estimated),
        }

    if kernel == "dsv4-k-norm-rope-flashmla":
        d = dsv4_mla_head_dim(model)
        rope_dim = dsv4_rope_head_dim(model)
        page_size = args.swa_page_size
        bytes_per_page = dsv4_packed_cache_bytes_per_page(args, model, page_size)
        pages = math.ceil(args.batch_size * seq / page_size) + 1
        shapes = {
            "kv": [total_tokens, d],
            "kv_weight": [d],
            "freqs_cis": [seq, rope_dim // 2],
            "positions": [total_tokens],
            "out_loc": [total_tokens],
            "kvcache_raw": [pages, bytes_per_page],
        }
        estimated = (
            bytes_for(shapes["kv"], dbytes)
            + bytes_for(shapes["kv_weight"], dbytes)
            + bytes_for(shapes["freqs_cis"], 8)
            + bytes_for(shapes["positions"], 4)
            + bytes_for(shapes["out_loc"], 4)
            + bytes_for(shapes["kvcache_raw"], 1)
        )
        return {
            "attn_tp_size": atp,
            "rope_dim": rope_dim,
            "kv_lora_rank": dsv4_kv_lora_rank(model),
            "mla_head_dim": d,
            "page_size": page_size,
            "kv_cache_dtype": dsv4_kv_cache_storage_dtype(args),
            "activation_dtype": activation_dtype_name(args, kernel),
            "shapes": shapes,
            "estimated_live_bytes": estimated,
            "estimated_live_gib": gib(estimated),
        }

    hq = require_divisible(
        "num_attention_heads", int(model["num_attention_heads"]), atp
    )
    hk = local_kv_heads_for_attention(model, atp)
    d = int(model["head_dim"])
    rows = math.ceil(seq / args.block_m)
    shapes = {
        "q": [args.batch_size, seq, hq, d],
        "k": [args.batch_size, seq, hk, d],
        "v": [args.batch_size, seq, hk, d],
        "block_count": [args.batch_size, hq, rows],
        "block_offset": [args.batch_size, hq, rows, args.slash_nnz],
        "column_count": [args.batch_size, hq, rows],
        "column_index": [args.batch_size, hq, rows, args.vertical_nnz],
    }
    estimated = (
        bytes_for(shapes["q"], dbytes)
        + bytes_for(shapes["k"], dbytes)
        + bytes_for(shapes["v"], dbytes)
        + bytes_for(shapes["block_count"], 4)
        + bytes_for(shapes["block_offset"], 4)
        + bytes_for(shapes["column_count"], 4)
        + bytes_for(shapes["column_index"], 4)
    )
    return {
        "attn_tp_size": atp,
        "local_q_heads": hq,
        "local_kv_heads": hk,
        "num_rows": rows,
        "activation_dtype": activation_dtype_name(args, kernel),
        "shapes": shapes,
        "estimated_live_bytes": estimated,
        "estimated_live_gib": gib(estimated),
    }


def resolve_torch_dtype(torch: Any, name: str) -> Any:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("fp8", "float8_e4m3fn"):
        return torch.float8_e4m3fn
    raise ValueError(name)


def randn_tensor(
    torch: Any, shape: tuple[int, ...] | list[int], dtype: Any, device: int
) -> Any:
    # torch.randn does not generate float8 directly on all supported PyTorch
    # builds. Generate in BF16 and cast for FP8 storage tensors.
    if dtype is torch.float8_e4m3fn:
        return torch.randn(*shape, dtype=torch.bfloat16, device=device).to(dtype)
    return torch.randn(*shape, dtype=dtype, device=device)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def initialize_sglang_dp_attention_state(
    torch: Any,
    args: argparse.Namespace,
    model: dict[str, Any],
    *,
    server_args: Any | None = None,
    model_config: Any | None = None,
) -> None:
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    if server_args is None:
        server_args = SimpleNamespace(
            enable_dp_attention=bool(args.enable_dp_attention),
            dp_size=attention_dp_size(args),
            moe_dense_tp_size=None,
            attn_cp_size=1,
            moe_dp_size=1,
            device="cuda",
        )
    if model_config is None:
        model_config = SimpleNamespace(
            hidden_size=model_hidden_size(model),
            dtype=resolve_torch_dtype(
                torch, activation_dtype_name(args, "dp-attention-init")
            ),
            hf_config=SimpleNamespace(
                hybrid_override_pattern=model.get("hybrid_override_pattern")
            ),
        )
    initialize_dp_attention(server_args=server_args, model_config=model_config)
    args.sglang_dp_attention_initialized = True
    args.sglang_dp_attention_moe_dense_tp_size = getattr(
        server_args, "moe_dense_tp_size", None
    )


def setup_torch(args: argparse.Namespace, model: dict[str, Any]) -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    args.rank = env_int("RANK", 0)
    args.local_rank = env_int("LOCAL_RANK", args.device)
    args.distributed_world_size = env_int("WORLD_SIZE", 1)
    args.distributed_backend = None
    args.sglang_model_parallel_initialized = False
    args.sglang_dp_attention_initialized = False
    args.sglang_dp_attention_moe_dense_tp_size = None
    if args.distributed_world_size > 1:
        if args.distributed_world_size != args.tp_size:
            raise RuntimeError(
                f"torch.distributed WORLD_SIZE={args.distributed_world_size} must match --tp-size={args.tp_size}; "
                "launch with torch.distributed.run --nproc_per_node equal to TP_SIZE."
            )
        torch.cuda.set_device(args.local_rank)
        args.device = args.local_rank
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=args.distributed_world_size,
            rank=args.rank,
            local_rank=args.local_rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=args.tp_size,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            attention_data_parallel_size=attention_dp_size(args),
            attention_context_model_parallel_size=1,
            moe_data_model_parallel_size=1,
            backend="nccl",
        )
        args.distributed_backend = "sglang_nccl"
        args.sglang_model_parallel_initialized = True
        initialize_sglang_dp_attention_state(torch, args, model)
    else:
        torch.cuda.set_device(args.device)
        args.rank = 0
        args.local_rank = args.device

    seed = args.seed + int(getattr(args, "rank", 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch


def is_report_rank(args: argparse.Namespace) -> bool:
    return int(getattr(args, "rank", 0)) == 0


def distributed_is_initialized(torch: Any) -> bool:
    if not hasattr(torch, "distributed"):
        return False
    dist = torch.distributed
    return bool(dist.is_available() and dist.is_initialized())


def tp_communication_mode(args: argparse.Namespace) -> str:
    if int(getattr(args, "distributed_world_size", 1)) <= 1:
        return "single_process"
    if bool(args.enable_dp_attention):
        return "dp_attention_all_gather_into_tensor"
    if attn_tp_size(args) > 1:
        return "attention_tp_all_reduce"
    return "none"


def first_tensor(value: Any) -> Any | None:
    if hasattr(value, "is_cuda"):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def replace_first_tensor(value: Any, old: Any, new: Any) -> Any:
    if value is old:
        return new
    if isinstance(value, tuple):
        items = list(value)
        for idx, item in enumerate(items):
            if item is old:
                items[idx] = new
                return tuple(items)
    if isinstance(value, list):
        items = list(value)
        for idx, item in enumerate(items):
            if item is old:
                items[idx] = new
                return items
    return new


def sglang_tensor_model_parallel_all_reduce(value: Any) -> Any:
    from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(value.contiguous())


def apply_attention_boundary_communication(
    torch: Any,
    args: argparse.Namespace,
    value: Any,
    *,
    pure_tp: bool = True,
    dp_attention: bool = True,
) -> Any:
    mode = tp_communication_mode(args)
    if mode in {"single_process", "none"}:
        return value
    if mode == "attention_tp_all_reduce" and not pure_tp:
        return value
    if mode == "dp_attention_all_gather_into_tensor" and not dp_attention:
        return value
    tensor = first_tensor(value)
    if tensor is None or not getattr(tensor, "is_cuda", False):
        return value
    if not bool(getattr(args, "sglang_model_parallel_initialized", False)):
        raise RuntimeError(
            "SGLang model-parallel groups are not initialized for distributed communication"
        )
    if mode == "attention_tp_all_reduce":
        from sglang.srt.distributed.communication_op import (
            attention_tensor_model_parallel_all_reduce,
        )

        reduced = attention_tensor_model_parallel_all_reduce(tensor.contiguous())
        return replace_first_tensor(value, tensor, reduced)
    if mode == "dp_attention_all_gather_into_tensor":
        from sglang.srt.distributed.parallel_state import get_tp_group

        local = tensor.contiguous().view(-1)
        gathered = torch.empty(
            local.numel() * int(getattr(args, "distributed_world_size", 1)),
            dtype=local.dtype,
            device=local.device,
        )
        get_tp_group().all_gather_into_tensor(gathered, local)
        start = int(getattr(args, "rank", 0)) * local.numel()
        restored = gathered[start : start + local.numel()].view_as(tensor).contiguous()
        return replace_first_tensor(value, tensor, restored)
    raise ValueError(f"unknown tp communication mode: {mode}")


def dsv4_slice_flashmla_output(
    value: Any, args: argparse.Namespace, info: dict[str, Any]
) -> Any:
    tensor = first_tensor(value)
    if tensor is None or not hasattr(tensor, "narrow"):
        return value
    local_heads = int(
        info.get("production_output_slice_heads") or info.get("local_q_heads") or 0
    )
    flash_heads = int(info.get("flashmla_q_heads") or local_heads)
    if local_heads <= 0 or flash_heads <= local_heads:
        return value
    start = attention_tp_rank(args) * local_heads
    if start + local_heads > int(tensor.shape[-2]):
        raise RuntimeError(
            f"Cannot slice FlashMLA output heads: rank slice [{start}, {start + local_heads}) exceeds output head dim {tensor.shape[-2]}"
        )
    sliced = tensor.narrow(-2, start, local_heads).contiguous()
    if value is tensor:
        return sliced
    if isinstance(value, tuple):
        items = list(value)
        for idx, item in enumerate(items):
            if item is tensor:
                items[idx] = sliced
                return tuple(items)
    if isinstance(value, list):
        items = list(value)
        for idx, item in enumerate(items):
            if item is tensor:
                items[idx] = sliced
                return items
    return sliced


def merge_distributed_rank_statuses(
    torch: Any, args: argparse.Namespace, record: dict[str, Any]
) -> dict[str, Any]:
    if int(
        getattr(args, "distributed_world_size", 1)
    ) <= 1 or not distributed_is_initialized(torch):
        return record
    local = {
        "rank": int(getattr(args, "rank", 0)),
        "local_rank": int(getattr(args, "local_rank", 0)),
        "status": record.get("status"),
        "error_message": record.get("error_message") or record.get("error"),
    }
    gathered: list[Any] = [
        None for _ in range(int(getattr(args, "distributed_world_size", 1)))
    ]
    torch.distributed.all_gather_object(gathered, local)
    if is_report_rank(args):
        record["distributed_rank_statuses"] = gathered
        bad = [
            item
            for item in gathered
            if isinstance(item, dict) and item.get("status") != "ok"
        ]
        if bad and record.get("status") == "ok":
            record["status"] = "error"
            record["error_message"] = (
                "One or more distributed ranks failed; see distributed_rank_statuses."
            )
    return record


def cleanup_distributed(torch: Any, args: argparse.Namespace) -> None:
    if int(getattr(args, "distributed_world_size", 1)) <= 1:
        return
    if bool(getattr(args, "sglang_model_parallel_initialized", False)):
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    elif distributed_is_initialized(torch):
        torch.distributed.destroy_process_group()


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def memory_snapshot(torch: Any, device: int) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def time_cuda_with_setup(
    torch: Any,
    setup_fn: Callable[[], Any] | None,
    fn: Callable[[], Any],
    warmup: int,
    iters: int,
    device: int,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    before_warmup = memory_snapshot(torch, device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        if setup_fn is not None:
            setup_fn()
            torch.cuda.synchronize(device)
        out = fn()
    torch.cuda.synchronize(device)
    after_warmup = memory_snapshot(torch, device)
    torch.cuda.reset_peak_memory_stats(device)
    times: list[float] = []
    for _ in range(iters):
        if setup_fn is not None:
            setup_fn()
            torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    torch.cuda.synchronize(device)
    if "out" in locals():
        del out
    after_timed = memory_snapshot(torch, device)
    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    latency = {
        "mean_ms": sum(times) / len(times),
        "stdev_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "p05_ms": pct(times, 0.05),
        "p10_ms": pct(times, 0.10),
        "p20_ms": pct(times, 0.20),
        "p50_ms": pct(times, 0.50),
        "p80_ms": pct(times, 0.80),
        "p90_ms": pct(times, 0.90),
        "p95_ms": pct(times, 0.95),
        "p99_ms": pct(times, 0.99),
    }
    memory = {
        "allocated_before_warmup_bytes": before_warmup["allocated_bytes"],
        "reserved_before_warmup_bytes": before_warmup["reserved_bytes"],
        "allocated_after_warmup_bytes": after_warmup["allocated_bytes"],
        "reserved_after_warmup_bytes": after_warmup["reserved_bytes"],
        "allocated_after_timed_bytes": after_timed["allocated_bytes"],
        "reserved_after_timed_bytes": after_timed["reserved_bytes"],
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "delta_peak_allocated_from_after_warmup_bytes": max(
            0, peak_alloc - after_warmup["allocated_bytes"]
        ),
        "delta_peak_reserved_from_after_warmup_bytes": max(
            0, peak_reserved - after_warmup["reserved_bytes"]
        ),
        "peak_allocated_gib": gib(peak_alloc),
        "peak_reserved_gib": gib(peak_reserved),
        "delta_peak_allocated_from_after_warmup_gib": gib(
            max(0, peak_alloc - after_warmup["allocated_bytes"])
        ),
        "delta_peak_reserved_from_after_warmup_gib": gib(
            max(0, peak_reserved - after_warmup["reserved_bytes"])
        ),
    }
    return {
        "iters": iters,
        "warmup_iters": warmup,
        "samples_ms": times,
        "latency_ms": latency,
        "memory": memory,
    }


def time_cuda(
    torch: Any, fn: Callable[[], Any], warmup: int, iters: int, device: int
) -> dict[str, Any]:
    return time_cuda_with_setup(torch, None, fn, warmup, iters, device)


def summarize_per_second(samples: list[float], work_items: int) -> dict[str, float]:
    values = [float(work_items) / (float(ms) / 1000.0) for ms in samples if ms > 0]
    if not values:
        return {}
    return {
        "mean_per_second": sum(values) / len(values),
        "stdev_per_second": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_per_second": min(values),
        "max_per_second": max(values),
        "p05_per_second": pct(values, 0.05),
        "p10_per_second": pct(values, 0.10),
        "p20_per_second": pct(values, 0.20),
        "p50_per_second": pct(values, 0.50),
        "p80_per_second": pct(values, 0.80),
        "p90_per_second": pct(values, 0.90),
        "p95_per_second": pct(values, 0.95),
        "p99_per_second": pct(values, 0.99),
    }


def add_work_metrics(
    timing: dict[str, Any], work_items: int, unit: str
) -> dict[str, Any]:
    timing["work_items"] = int(work_items)
    timing["work_unit"] = unit
    samples = [float(ms) for ms in timing.get("samples_ms", [])]
    throughput = summarize_per_second(samples, int(work_items))
    timing[f"{unit}_per_second"] = throughput
    # Backward-compatible convenience aliases for quick jq/grep usage.
    timing[f"{unit}_per_second_p50"] = throughput.get("p50_per_second")
    timing[f"{unit}_per_second_mean"] = throughput.get("mean_per_second")
    return timing


def combine_phase_memory(
    prefill: dict[str, Any], decode: dict[str, Any]
) -> dict[str, Any]:
    prefill_memory = prefill.get("memory", {}) if isinstance(prefill, dict) else {}
    decode_memory = decode.get("memory", {}) if isinstance(decode, dict) else {}
    peak_alloc = max(
        int(prefill_memory.get("peak_allocated_bytes", 0)),
        int(decode_memory.get("peak_allocated_bytes", 0)),
    )
    peak_reserved = max(
        int(prefill_memory.get("peak_reserved_bytes", 0)),
        int(decode_memory.get("peak_reserved_bytes", 0)),
    )
    return {
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": gib(peak_alloc),
        "peak_reserved_gib": gib(peak_reserved),
        "prefill": prefill_memory,
        "decode": decode_memory,
    }


def phase_timing(
    torch: Any,
    args: argparse.Namespace,
    device: int,
    *,
    reset_state: Callable[[], Any],
    run_prefill: Callable[[], Any],
    run_decode: Callable[[], Any],
    prefill_work_items: int,
    decode_work_items: int,
    unit: str = "tokens",
) -> dict[str, Any]:
    prefill = add_work_metrics(
        time_cuda_with_setup(
            torch, reset_state, run_prefill, args.warmup_iters, args.iters, device
        ),
        prefill_work_items,
        unit,
    )

    def setup_decode() -> None:
        reset_state()
        run_prefill()

    decode = add_work_metrics(
        time_cuda_with_setup(
            torch, setup_decode, run_decode, args.warmup_iters, args.iters, device
        ),
        decode_work_items,
        unit,
    )
    timing = {
        "mode": "prefill_decode_phases",
        "prefill": prefill,
        "decode": decode,
        "memory": combine_phase_memory(prefill, decode),
        "work_unit": unit,
        "prefill_work_items": int(prefill_work_items),
        "decode_work_items": int(decode_work_items),
        "prefill_latency_ms_p50": prefill["latency_ms"].get("p50_ms"),
        "decode_latency_ms_p50": decode["latency_ms"].get("p50_ms"),
        f"prefill_{unit}_per_second_p50": prefill.get(f"{unit}_per_second_p50"),
        f"decode_{unit}_per_second_p50": decode.get(f"{unit}_per_second_p50"),
    }
    if unit == "tokens":
        timing["prefill_tokens_per_second_p50"] = timing.get(
            "prefill_tokens_per_second_p50"
        )
        timing["decode_tokens_per_second_p50"] = timing.get(
            "decode_tokens_per_second_p50"
        )
    return timing


def make_sparse_metadata(
    torch: Any, args: argparse.Namespace, hq: int, device: int, seq: int | None = None
) -> tuple[Any, Any, Any, Any]:
    seq = seq or args.seq_len
    rows = math.ceil(seq / args.block_m)
    b = args.batch_size
    nnz_s = args.slash_nnz
    nnz_v = args.vertical_nnz
    block_count = torch.full((b, hq, rows), nnz_s, dtype=torch.int32, device=device)
    column_count = torch.full((b, hq, rows), nnz_v, dtype=torch.int32, device=device)
    if args.causal:
        row_end = (
            (torch.arange(rows, dtype=torch.int32, device=device) + 1) * args.block_m
            - 1
        ).clamp(max=seq - 1)
        cols = (
            row_end[:, None]
            - torch.arange(nnz_v, dtype=torch.int32, device=device)[None, :]
        ).clamp(min=0)
        block_starts = (
            (
                row_end[:, None]
                - torch.arange(nnz_s, dtype=torch.int32, device=device)[None, :]
                * args.block_n
            ).clamp(min=0)
            // args.block_n
            * args.block_n
        )
    else:
        cols = torch.arange(nnz_v, dtype=torch.int32, device=device)[None, :] % seq
        cols = cols.expand(rows, nnz_v)
        block_starts = (
            torch.arange(nnz_s, dtype=torch.int32, device=device)[None, :]
            * args.block_n
        ) % seq
        block_starts = block_starts.expand(rows, nnz_s)
    column_index = cols[None, None, :, :].expand(b, hq, rows, nnz_v).contiguous()
    block_offset = (
        block_starts[None, None, :, :].expand(b, hq, rows, nnz_s).contiguous()
    )
    return block_count, block_offset, column_count, column_index


def run_sparse_flash(
    kernel: str, args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    from sgl_kernel.sparse_flash_attn import sparse_attn_func, sparse_attn_varlen_func

    info = shape_summary(kernel, args, model)
    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    dtype = resolve_torch_dtype(torch, activation_dtype_name(args, kernel))
    device = args.device
    hq = int(info.get("flashmla_q_heads", info["local_q_heads"]))
    hk = int(info["local_kv_heads"])
    d = int(model["head_dim"])
    seq = input_len(args)
    q = randn_tensor(torch, (args.batch_size, seq, hq, d), dtype, device)
    k = randn_tensor(torch, (args.batch_size, seq, hk, d), dtype, device)
    v = randn_tensor(torch, (args.batch_size, seq, hk, d), dtype, device)
    block_count, block_offset, column_count, column_index = make_sparse_metadata(
        torch, args, hq, device, seq=seq
    )
    kwargs = {"return_softmax_lse": True, "causal": args.causal}
    if args.softcap > 0:
        kwargs["softcap"] = args.softcap

    if kernel == "sparse-flash-varlen":
        cu = torch.arange(
            0, (args.batch_size + 1) * seq, seq, dtype=torch.int32, device=device
        )
        qv = q.reshape(args.batch_size * seq, hq, d)
        kv = k.reshape(args.batch_size * seq, hk, d)
        vv = v.reshape(args.batch_size * seq, hk, d)

        def fn() -> Any:
            out = sparse_attn_varlen_func(
                qv,
                kv,
                vv,
                block_count,
                block_offset,
                column_count,
                column_index,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=seq,
                max_seqlen_k=seq,
                **kwargs,
            )
            return apply_attention_boundary_communication(torch, args, out)
    else:

        def fn() -> Any:
            out = sparse_attn_func(
                q, k, v, block_count, block_offset, column_count, column_index, **kwargs
            )
            return apply_attention_boundary_communication(torch, args, out)

    timing = add_work_metrics(
        time_cuda(torch, fn, args.warmup_iters, args.iters, device),
        args.batch_size * seq,
        "tokens",
    )
    return {
        **info,
        "tp_communication_mode": tp_communication_mode(args),
        "timing": timing,
    }


def run_sparse_index_convert(
    args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    from sgl_kernel.sparse_flash_attn import convert_vertical_slash_indexes

    kernel = "sparse-index-convert"
    info = shape_summary("sparse-flash", args, model)
    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    device = args.device
    hq = int(info.get("flashmla_q_heads", info["local_q_heads"]))
    seq = input_len(args)
    q_seqlens = torch.full((args.batch_size,), seq, dtype=torch.int32, device=device)
    kv_seqlens = q_seqlens.clone()
    vertical = (
        torch.arange(args.vertical_nnz, dtype=torch.int32, device=device)[None, None, :]
        % seq
    )
    slash = (
        torch.arange(args.slash_nnz, dtype=torch.int32, device=device)[None, None, :]
        * args.block_n
    )
    vertical = vertical.expand(args.batch_size, hq, args.vertical_nnz).contiguous()
    slash = (slash % seq).expand(args.batch_size, hq, args.slash_nnz).contiguous()

    def fn() -> Any:
        return convert_vertical_slash_indexes(
            q_seqlens,
            kv_seqlens,
            vertical,
            slash,
            seq,
            args.block_m,
            args.block_n,
            causal=args.causal,
        )

    timing = add_work_metrics(
        time_cuda(torch, fn, args.warmup_iters, args.iters, device),
        args.batch_size * hq,
        "heads",
    )
    return {
        **info,
        "vertical_indexes_shape": list(vertical.shape),
        "slash_indexes_shape": list(slash.shape),
        "timing": timing,
    }


def make_freqs_cis(torch: Any, seq_len: int, rope_dim: int, device: int) -> Any:
    return torch.ones(seq_len, rope_dim // 2, dtype=torch.complex64, device=device)


def run_dsv4_topk(
    kernel: str, args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    from sglang.jit_kernel.dsv4 import (
        plan_topk_v2,
        topk_transform_512,
        topk_transform_512_v2,
    )

    info = shape_summary(kernel, args, model)
    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    topk = int(info["topk"])
    if topk not in (512, 1024):
        raise ValueError(
            f"SGLang DSV4 topk kernels support topk=512 or 1024, got {topk}"
        )
    page_size = int(info["page_size"])
    seq = input_len(args)
    pages = math.ceil(seq / page_size)
    scores = torch.randn(args.batch_size, seq, dtype=torch.float32, device=args.device)
    seq_lens = torch.full(
        (args.batch_size,), seq, dtype=torch.int32, device=args.device
    )
    page_tables = (
        torch.arange(pages, dtype=torch.int32, device=args.device)[None, :]
        .expand(args.batch_size, pages)
        .contiguous()
    )
    out_page_indices = torch.empty(
        args.batch_size, topk, dtype=torch.int32, device=args.device
    )
    if kernel == "dsv4-topk-v1":
        out_raw_indices = torch.empty_like(out_page_indices)

        def fn() -> Any:
            return topk_transform_512(
                scores,
                seq_lens,
                page_tables,
                out_page_indices,
                page_size,
                out_raw_indices,
            )
    else:
        metadata = plan_topk_v2(seq_lens)

        def fn() -> Any:
            return topk_transform_512_v2(
                scores, seq_lens, page_tables, out_page_indices, page_size, metadata
            )

    timing = add_work_metrics(
        time_cuda(torch, fn, args.warmup_iters, args.iters, args.device),
        args.batch_size * seq,
        "tokens",
    )
    return {**info, "timing": timing}


def run_dsv4_fused_elementwise(
    kernel: str, args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    from sglang.jit_kernel.dsv4 import (
        fused_k_norm_rope_flashmla,
        fused_q_indexer_rope_hadamard_quant,
        fused_q_norm_rope,
    )

    info = shape_summary(kernel, args, model)
    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    dtype = resolve_torch_dtype(torch, activation_dtype_name(args, kernel))
    seq = input_len(args)
    total = args.batch_size * seq
    d = int(model.get("head_dim", 512))
    rope_dim = int(info["rope_dim"])
    positions = (
        torch.arange(total, dtype=torch.int32, device=args.device) % seq
    ).contiguous()
    freqs_cis = make_freqs_cis(torch, seq, rope_dim, args.device)
    if kernel == "dsv4-q-norm-rope":
        hq = int(info["local_q_heads"])
        d = int(info["shapes"]["q_input"][-1])
        q_input = randn_tensor(torch, (total, hq, d), dtype, args.device)
        q_output = torch.empty_like(q_input)

        def fn() -> Any:
            return fused_q_norm_rope(
                q_input, q_output, args.norm_eps, freqs_cis, positions
            )
    elif kernel == "dsv4-k-norm-rope-flashmla":
        kv = randn_tensor(torch, (total, d), dtype, args.device)
        kv_weight = torch.ones(d, dtype=dtype, device=args.device)
        page_size = args.swa_page_size
        bytes_per_page = dsv4_packed_cache_bytes_per_page(args, model, page_size)
        pages = math.ceil(total / page_size) + 1
        kvcache = torch.zeros(
            pages, bytes_per_page, dtype=torch.uint8, device=args.device
        )
        out_loc = torch.arange(total, dtype=torch.int32, device=args.device)

        def fn() -> Any:
            return fused_k_norm_rope_flashmla(
                kv,
                kv_weight,
                args.norm_eps,
                freqs_cis,
                positions,
                out_loc,
                kvcache,
                page_size,
            )
    else:
        hq = int(info["local_q_heads"])
        d = int(model.get("index_head_dim", 128))
        q_input = randn_tensor(torch, (total, hq, d), dtype, args.device)
        weight = randn_tensor(torch, (total, hq), dtype, args.device)
        weight_scale = d**-0.5 * hq**-0.5

        def fn() -> Any:
            return fused_q_indexer_rope_hadamard_quant(
                q_input, weight, weight_scale, freqs_cis, positions
            )

    timing = add_work_metrics(
        time_cuda(torch, fn, args.warmup_iters, args.iters, args.device),
        total,
        "tokens",
    )
    return {**info, "timing": timing}


def make_flashmla_indices(
    torch: Any, total_tokens: int, batch_size: int, seq_len: int, topk: int, device: int
) -> tuple[Any, Any]:
    token = torch.arange(total_tokens, dtype=torch.int32, device=device)
    seq_base = (token // seq_len) * seq_len
    local_pos = token % seq_len
    rel = torch.arange(topk, dtype=torch.int32, device=device)
    indices = (
        seq_base[:, None] + (local_pos[:, None] - rel[None, :]).clamp(min=0)
    ).contiguous()
    topk_len = torch.minimum(local_pos + 1, torch.full_like(local_pos, topk)).to(
        torch.int32
    )
    return indices[:, None, :], topk_len


def make_flashmla_swa_indices(
    torch: Any,
    batch_size: int,
    query_len: int,
    pages_per_seq: int,
    page_size: int,
    topk: int,
    device: int,
    query_start_pos: int,
) -> tuple[Any, Any]:
    if query_len <= 0:
        raise ValueError(f"query_len must be positive, got {query_len}")
    token = torch.arange(batch_size * query_len, dtype=torch.int32, device=device)
    req = token // query_len
    local_q_pos = token % query_len
    local_kv_pos = int(query_start_pos) + local_q_pos
    rel = torch.arange(topk, dtype=torch.int32, device=device)
    valid = rel[None, :] <= local_kv_pos[:, None]
    local_token = (local_kv_pos[:, None] - rel[None, :]).clamp(min=0)
    indices = req[:, None] * (pages_per_seq * page_size) + local_token
    indices = indices.masked_fill(~valid, -1).contiguous()
    topk_length = torch.minimum(
        local_kv_pos + 1, torch.full_like(local_kv_pos, topk)
    ).to(torch.int32)
    return indices[:, None, :], topk_length


def make_dsv4_query_request_ids(
    torch: Any, batch_size: int, query_len: int, device: int
) -> Any:
    token = torch.arange(batch_size * query_len, dtype=torch.int64, device=device)
    return token // query_len


def make_dsv4_page_table_rows(
    torch: Any,
    batch_size: int,
    query_len: int,
    pages_per_seq: int,
    device: int,
) -> Any:
    req = make_dsv4_query_request_ids(torch, batch_size, query_len, device)
    per_req = (
        torch.arange(pages_per_seq, dtype=torch.int32, device=device)[None, :]
        + torch.arange(batch_size, dtype=torch.int32, device=device)[:, None]
        * pages_per_seq
    )
    return per_req[req].contiguous()


def make_dsv4_seq_lens_casual(
    torch: Any, batch_size: int, query_len: int, query_start_pos: int, device: int
) -> Any:
    token = torch.arange(batch_size * query_len, dtype=torch.int32, device=device)
    return (int(query_start_pos) + (token % query_len) + 1).contiguous()


def make_dsv4_raw_out_locs(
    torch: Any,
    batch_size: int,
    query_len: int,
    query_start_pos: int,
    pages_per_seq: int,
    page_size: int,
    device: int,
) -> Any:
    req = make_dsv4_query_request_ids(torch, batch_size, query_len, device).to(
        torch.int32
    )
    local_pos = make_dsv4_positions(
        torch, batch_size, query_len, query_start_pos, device
    )
    return (req * (pages_per_seq * page_size) + local_pos).to(torch.int32).contiguous()


def make_dsv4_compression_metadata(
    torch: Any,
    batch_size: int,
    query_len: int,
    query_start_pos: int,
    pages_per_seq: int,
    page_size: int,
    device: int,
) -> dict[str, Any]:
    from sglang.srt.layers.attention.dsv4.metadata_kernel import (
        init_compression_metadata,
    )

    seq_lens = make_dsv4_seq_lens_casual(
        torch, batch_size, query_len, query_start_pos, device
    )
    positions = seq_lens - 1
    raw_out_locs = make_dsv4_raw_out_locs(
        torch, batch_size, query_len, query_start_pos, pages_per_seq, page_size, device
    )
    page_table = make_dsv4_page_table_rows(
        torch, batch_size, query_len, pages_per_seq, device
    )
    (
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_topk_lengths_clamp1,
        c128_page_indices,
    ) = init_compression_metadata(
        seq_lens,
        positions,
        raw_out_locs,
        page_table,
        page_size,
        compute_page_indices=True,
    )
    return {
        "seq_lens": seq_lens,
        "positions": positions,
        "page_table": page_table,
        "c4_out_loc": c4_out_loc,
        "c4_positions": c4_positions,
        "c4_seq_lens_raw": c4_seq_lens_raw,
        "c4_seq_lens_clamp1": c4_seq_lens_clamp1,
        "c128_out_loc": c128_out_loc,
        "c128_positions": c128_positions,
        "c128_topk_lengths_clamp1": c128_topk_lengths_clamp1,
        "c128_page_indices": c128_page_indices,
    }


def pad_page_indices_last_dim(
    torch: Any, x: Any, multiple: int = DSV4_PAGE_INDEX_ALIGNMENT
) -> Any:
    target = ceil_align(int(x.shape[-1]), multiple)
    if target == int(x.shape[-1]):
        return x.contiguous()
    return torch.nn.functional.pad(
        x, (0, target - int(x.shape[-1])), value=-1
    ).contiguous()


def dsv4_c4_indexer_page_indices(
    torch: Any,
    q_lora: Any,
    hidden: Any,
    positions: Any,
    c4_seq_lens_raw: Any,
    c4_seq_lens_clamp1: Any,
    page_table: Any,
    indexer_k_cache: Any,
    wq_b_weight: Any,
    weights_proj: Any,
    freqs_cis: Any,
    weight_scale: float,
    topk: int,
    c4_page_size: int,
    max_c4_seq_len: int,
) -> tuple[Any, Any]:
    from deep_gemm import (
        fp8_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
    )
    from sglang.jit_kernel.dsv4 import (
        plan_topk_v2,
        topk_transform_512,
        topk_transform_512_v2,
    )
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsv4.indexer import (
        topk_transform_512_pytorch_vectorized,
    )
    from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_hadamard_quant

    index_heads = weights_proj.shape[1]
    index_head_dim = wq_b_weight.shape[1] // index_heads
    q_input = (
        q_lora.matmul(wq_b_weight)
        .view(q_lora.shape[0], index_heads, index_head_dim)
        .contiguous()
    )
    weights = hidden.matmul(weights_proj).contiguous()
    q_fp8, weights = fused_q_indexer_rope_hadamard_quant(
        q_input, weights, weight_scale, freqs_cis, positions
    )
    c4_seq_lens_2d = c4_seq_lens_raw.to(torch.int32).unsqueeze(-1)
    metadata = get_paged_mqa_logits_metadata(
        c4_seq_lens_2d, c4_page_size, get_num_sms()
    )
    logits = fp8_paged_mqa_logits(
        q_fp8.unsqueeze(1),
        indexer_k_cache,
        weights.squeeze(2),
        c4_seq_lens_2d,
        page_table,
        metadata,
        max_c4_seq_len,
        False,
    )
    out_page_indices = torch.empty(
        q_lora.shape[0], topk, dtype=torch.int32, device=q_lora.device
    )
    if envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
        topk_transform_512_pytorch_vectorized(
            logits, c4_seq_lens_raw, page_table, out_page_indices, c4_page_size, None
        )
    elif envs.SGLANG_OPT_USE_TOPK_V2.get():
        topk_metadata = plan_topk_v2(c4_seq_lens_raw)
        topk_transform_512_v2(
            logits,
            c4_seq_lens_raw,
            page_table,
            out_page_indices,
            c4_page_size,
            topk_metadata,
        )
    else:
        topk_transform_512(
            logits, c4_seq_lens_raw, page_table, out_page_indices, c4_page_size, None
        )
    topk_lengths = torch.clamp(c4_seq_lens_clamp1, max=topk).to(torch.int32)
    return out_page_indices.contiguous(), topk_lengths.contiguous()


def make_dsv4_c4_compressor_metadata(
    torch: Any,
    batch_size: int,
    query_len: int,
    query_start_pos: int,
    req_to_token: Any,
    full_to_swa: Any,
    ring_size: int,
    device: int,
    *,
    is_decode: bool,
) -> dict[str, Any]:
    from sglang.jit_kernel.dsv4 import triton_create_paged_compress_data
    from sglang.jit_kernel.dsv4.compress_old import (
        CompressorDecodePlan,
        CompressorPrefillPlan,
    )

    seq_lens = torch.full(
        (batch_size,),
        int(query_start_pos) + int(query_len),
        dtype=torch.int32,
        device=device,
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    if is_decode:
        write_positions = ((seq_lens - 1) // 4) * 4

        def get_raw_loc(positions: Any) -> Any:
            positions = torch.clamp(positions, min=0).to(torch.int64)
            loc = req_to_token[req_pool_indices.to(torch.int64), positions]
            swa_loc = full_to_swa[loc.to(torch.int64)].to(torch.int32)
            swa_page = swa_loc // DSV4_SWA_WINDOW
            state_loc = swa_page * ring_size + (swa_loc % ring_size)
            return (state_loc // 4).to(torch.int32)

        write_loc = get_raw_loc(write_positions)
        extra_data = get_raw_loc(write_positions - 4).view(-1, 1)
        plan = CompressorDecodePlan(4, seq_lens.to(torch.int32))
    else:
        extend_lens = torch.full(
            (batch_size,), int(query_len), dtype=torch.int32, device=device
        )
        write_loc, extra_data = triton_create_paged_compress_data(
            compress_ratio=4,
            is_overlap=True,
            swa_page_size=DSV4_SWA_WINDOW,
            ring_size=ring_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=full_to_swa,
        )
        plan = CompressorPrefillPlan.generate(
            compress_ratio=4,
            num_q_tokens=batch_size * int(query_len),
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            device=torch.device("cuda", device),
        )
    return {
        "write_loc": write_loc.contiguous(),
        "extra_data": extra_data.contiguous(),
        "plan": plan,
    }


def update_dsv4_c4_indexer_cache(
    torch: Any,
    hidden: Any,
    c4_out_loc: Any,
    query_len: int,
    query_start_pos: int,
    req_to_token: Any,
    full_to_swa: Any,
    compressor_state: Any,
    wkv_gate_weight: Any,
    ape: Any,
    norm_weight: Any,
    freqs_cis: Any,
    indexer_k_cache_storage: Any,
    ring_size: int,
    c4_page_size: int,
    index_head_dim: int,
    norm_eps: float,
    *,
    is_decode: bool,
) -> None:
    from sglang.jit_kernel.dsv4 import linear_bf16_fp32
    from sglang.jit_kernel.dsv4.compress_old import (
        compress_forward,
        compress_fused_norm_rope_inplace,
    )
    from sglang.srt.layers.attention.dsa import index_buf_accessor
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
    from sglang.srt.layers.attention.dsa.triton_kernel import act_quant

    metadata = make_dsv4_c4_compressor_metadata(
        torch,
        batch_size=hidden.shape[0] // int(query_len),
        query_len=query_len,
        query_start_pos=query_start_pos,
        req_to_token=req_to_token,
        full_to_swa=full_to_swa,
        ring_size=ring_size,
        device=hidden.device.index,
        is_decode=is_decode,
    )
    kv_score_input = linear_bf16_fp32(hidden.contiguous(), wkv_gate_weight)
    kv_score_buffer = compressor_state.view(-1, 4, 2 * 2 * int(index_head_dim))
    compressed = compress_forward(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=metadata["write_loc"],
        plan=metadata["plan"],
        extra_data=metadata["extra_data"],
        head_dim=int(index_head_dim),
        compress_ratio=4,
    )
    compress_fused_norm_rope_inplace(
        compressed, norm_weight, norm_eps, freqs_cis, metadata["plan"]
    )
    compressed = rotate_activation(compressed).contiguous()
    index_k, index_k_scale = act_quant(compressed)
    loc = c4_out_loc[: index_k.shape[0]].to(torch.int64).contiguous()
    index_buf_accessor.SetKAndS.execute(
        pool=SimpleNamespace(page_size=int(c4_page_size)),
        buf=indexer_k_cache_storage,
        loc=loc,
        index_k=index_k.contiguous(),
        index_k_scale=index_k_scale.contiguous(),
    )


def make_dsv4_compressor_metadata(
    torch: Any,
    *,
    ratio: int,
    batch_size: int,
    query_len: int,
    query_start_pos: int,
    req_to_token: Any,
    full_to_swa: Any,
    swa_page_size: int,
    ring_size: int,
    device: int,
    is_decode: bool,
) -> dict[str, Any]:
    from sglang.jit_kernel.dsv4 import triton_create_paged_compress_data
    from sglang.jit_kernel.dsv4.compress_old import (
        CompressorDecodePlan,
        CompressorPrefillPlan,
    )

    seq_lens = torch.full(
        (batch_size,),
        int(query_start_pos) + int(query_len),
        dtype=torch.int32,
        device=device,
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

    def get_raw_loc(positions: Any) -> Any:
        positions = torch.clamp(positions, min=0).to(torch.int64)
        loc = req_to_token[req_pool_indices.to(torch.int64), positions]
        swa_loc = full_to_swa[loc.to(torch.int64)].to(torch.int32)
        swa_page = swa_loc // int(swa_page_size)
        state_loc = swa_page * int(ring_size) + (swa_loc % int(ring_size))
        return (state_loc // int(ratio)).to(torch.int32)

    if is_decode:
        write_positions = ((seq_lens - 1) // int(ratio)) * int(ratio)
        write_loc = get_raw_loc(write_positions)
        if ratio == 4:
            extra_data = get_raw_loc(write_positions - int(ratio)).view(-1, 1)
        else:
            extra_data = None
        plan = CompressorDecodePlan(int(ratio), seq_lens.to(torch.int32))
    else:
        extend_lens = torch.full(
            (batch_size,), int(query_len), dtype=torch.int32, device=device
        )
        write_loc, extra_data = triton_create_paged_compress_data(
            compress_ratio=int(ratio),
            is_overlap=(int(ratio) == 4),
            swa_page_size=int(swa_page_size),
            ring_size=int(ring_size),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=full_to_swa,
        )
        plan = CompressorPrefillPlan.generate(
            compress_ratio=int(ratio),
            num_q_tokens=batch_size * int(query_len),
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            device=torch.device("cuda", device),
        )
    return {
        "write_loc": write_loc.contiguous(),
        "extra_data": None if extra_data is None else extra_data.contiguous(),
        "plan": plan,
    }


def dsv4_cache_view_for_attention(
    cache_2d: Any, page_tokens: int, row_bytes: int
) -> Any:
    return cache_2d[:, : int(page_tokens) * int(row_bytes)].view(
        cache_2d.shape[0], int(page_tokens), 1, int(row_bytes)
    )


def write_dsv4_fused_cache(
    torch: Any,
    args: argparse.Namespace,
    model: dict[str, Any],
    cache_2d: Any,
    loc: Any,
    cache_k: Any,
    page_tokens: int,
) -> None:
    if loc.numel() == 0:
        return
    if cache_k.shape[0] != loc.numel():
        raise ValueError(
            f"DSV4 fused cache write loc/tensor mismatch: loc={loc.numel()}, cache_k={cache_k.shape}"
        )
    if not is_requested_fp8(args):
        k_nope = cache_k[..., : dsv4_qk_nope_head_dim(model)].contiguous()
        k_rope = cache_k[
            ..., dsv4_qk_nope_head_dim(model) : dsv4_mla_head_dim(model)
        ].contiguous()
        write_dsv4_kv_cache(torch, args, model, cache_2d, loc, k_nope, k_rope)
        return
    from sglang.jit_kernel.dsv4 import fused_store_cache

    fused_store_cache(
        input=cache_k.contiguous(),
        cache=cache_2d,
        indices=loc.to(torch.int32).contiguous(),
        page_size=int(page_tokens),
        type="flashmla",
    )


def dsv4_compress_and_write_extra_cache(
    torch: Any,
    *,
    hidden: Any,
    ratio: int,
    metadata: dict[str, Any],
    compressor_state: Any,
    wkv_gate_weight: Any,
    ape: Any,
    norm_weight: Any,
    freqs_cis: Any,
    norm_eps: float,
    extra_cache_2d: Any,
    extra_cache_out_loc: Any,
    extra_page_tokens: int,
    head_dim: int,
) -> Any:
    from sglang.jit_kernel.dsv4 import fused_store_cache, linear_bf16_fp32
    from sglang.jit_kernel.dsv4.compress_old import (
        compress_forward,
        compress_fused_norm_rope_inplace,
    )

    coff = 2 if int(ratio) == 4 else 1
    kv_score_input = linear_bf16_fp32(hidden.contiguous(), wkv_gate_weight)
    kv_score_buffer = compressor_state.view(-1, int(ratio), 2 * coff * int(head_dim))
    compressed = compress_forward(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape.view(-1, int(head_dim)),
        indices=metadata["write_loc"],
        plan=metadata["plan"],
        extra_data=metadata["extra_data"],
        head_dim=int(head_dim),
        compress_ratio=int(ratio),
    )
    compress_fused_norm_rope_inplace(
        compressed, norm_weight, float(norm_eps), freqs_cis, metadata["plan"]
    )
    loc = extra_cache_out_loc[: compressed.shape[0]].to(torch.int32).contiguous()
    fused_store_cache(
        input=compressed.contiguous(),
        cache=extra_cache_2d,
        indices=loc,
        page_size=int(extra_page_tokens),
        type="flashmla",
    )
    return compressed


def make_cache_locs(
    torch: Any,
    batch_size: int,
    start_pos: int,
    query_len: int,
    pages_per_seq: int,
    page_size: int,
    device: int,
    token_compress_ratio: int = 1,
) -> Any:
    if token_compress_ratio <= 0:
        raise ValueError(
            f"token_compress_ratio must be positive, got {token_compress_ratio}"
        )
    if query_len <= 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    if token_compress_ratio == 1:
        local_pos = torch.arange(
            start_pos, start_pos + query_len, dtype=torch.int64, device=device
        )
    else:
        first = start_pos // token_compress_ratio
        last = math.ceil((start_pos + query_len) / token_compress_ratio)
        local_pos = torch.arange(first, last, dtype=torch.int64, device=device)
    req = torch.arange(batch_size, dtype=torch.int64, device=device)[:, None]
    locs = req * pages_per_seq * page_size + local_pos[None, :]
    return locs.reshape(-1).contiguous()


def quantize_flashmla_q_if_needed(
    torch: Any, args: argparse.Namespace, q: Any
) -> tuple[Any, dict[str, Any]]:
    # DSV4 sparse_decode_fwd in the installed SGLang/sgl-kernel build expects
    # BF16 Q even when KV cache storage is FP8-packed. Production DSV4 keeps Q
    # as an activation tensor and only quantizes the KV cache write path here.
    return q, {}


def write_dsv4_kv_cache(
    torch: Any,
    args: argparse.Namespace,
    model: dict[str, Any],
    k_cache_flat: Any,
    loc: Any,
    k_nope: Any,
    k_rope: Any,
) -> None:
    if loc.numel() == 0:
        return
    if k_nope.shape[0] != loc.numel() or k_rope.shape[0] != loc.numel():
        raise ValueError(
            f"KV write loc/tensor mismatch: loc={loc.numel()}, k_nope={k_nope.shape}, k_rope={k_rope.shape}"
        )
    from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton

    if is_requested_fp8(args):
        from sglang.srt.layers.attention.nsa.quant_k_cache import (
            quantize_k_cache_separate,
        )

        nope_part, rope_part = quantize_k_cache_separate(k_nope, k_rope)
        set_mla_kv_buffer_triton(k_cache_flat, loc, nope_part, rope_part)
    else:
        set_mla_kv_buffer_triton(k_cache_flat, loc, k_nope, k_rope)


def make_dsv4_positions(
    torch: Any, batch_size: int, query_len: int, query_start_pos: int, device: int
) -> Any:
    token = torch.arange(batch_size * query_len, dtype=torch.int64, device=device)
    return (int(query_start_pos) + (token % query_len)).contiguous()


def make_dsv4_output_projector(
    torch: Any,
    args: argparse.Namespace,
    model: dict[str, Any],
    info: dict[str, Any],
    dtype: Any,
    device: int,
) -> Callable[[Any, Any], Any]:
    from sglang.jit_kernel.dsv4 import fused_rope_inplace

    atp = attn_tp_size(args)
    local_heads = int(
        info.get("production_output_slice_heads") or info.get("local_q_heads")
    )
    local_groups = dsv4_local_o_groups(model, atp)
    vdim = dsv4_kv_lora_rank(model)
    rope_dim = dsv4_rope_head_dim(model)
    if local_heads % local_groups != 0:
        raise ValueError(
            f"local_q_heads={local_heads} must be divisible by local_o_groups={local_groups}"
        )
    per_group_dim = (local_heads // local_groups) * vdim
    o_lora_rank = dsv4_o_lora_rank(model)
    hidden = model_hidden_size(model)
    wo_a = randn_tensor(
        torch, (local_groups, o_lora_rank, per_group_dim), dtype, device
    )
    wo_b = randn_tensor(torch, (local_groups * o_lora_rank, hidden), dtype, device)
    freqs_cis = make_freqs_cis(
        torch, input_len(args) + args.output_len + 1, rope_dim, device
    )

    def project(value: Any, positions: Any) -> Any:
        tensor = first_tensor(value)
        if tensor is None:
            return value
        x = tensor.contiguous()
        if int(x.shape[-2]) != local_heads or int(x.shape[-1]) != vdim:
            raise RuntimeError(
                f"DSV4 output projection expected [..., {local_heads}, {vdim}], got {tuple(x.shape)}"
            )
        flat = x.reshape(-1, local_heads, vdim).contiguous()
        pos = positions.contiguous().view(-1)
        if pos.numel() != flat.shape[0]:
            raise RuntimeError(
                f"DSV4 output projection positions mismatch: positions={pos.numel()} rows={flat.shape[0]}"
            )
        fused_rope_inplace(
            flat[..., -rope_dim:], None, freqs_cis, positions=pos, inverse=True
        )
        grouped = flat.view(flat.shape[0], local_groups, per_group_dim)
        low_rank = torch.einsum("tgd,grd->tgr", grouped, wo_a)
        projected = low_rank.reshape(flat.shape[0], local_groups * o_lora_rank).matmul(
            wo_b
        )
        if atp > 1:
            projected = sglang_tensor_model_parallel_all_reduce(projected)
        return projected.view(*x.shape[:-2], hidden).contiguous()

    return project


def run_flashmla_sparse_prefill(
    args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    kernel = "flashmla-sparse-prefill"
    info = shape_summary(kernel, args, model)
    warning_messages = list(info.get("warning_messages", []))
    use_dsv4_sparse_metadata = model.get("family") == "deepseek_v4"
    use_attn_sink = bool(args.attn_sink or use_dsv4_sparse_metadata)
    use_topk_length = bool(args.topk_length or use_dsv4_sparse_metadata)
    if info.get("sparse_prefill_unsupported_reason"):
        for message in warning_messages:
            warn_stdout(message)
        if args.strict:
            raise RuntimeError(warning_messages[-1])
        return {
            **info,
            "status": "unsupported",
            "warning_messages": warning_messages,
            "attn_sink": use_attn_sink,
            "topk_length_enabled": use_topk_length,
        }

    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    dtype = resolve_torch_dtype(torch, activation_dtype_name(args, kernel))
    device = args.device
    seq = input_len(args)
    total = args.batch_size * seq
    hq = int(info["flashmla_q_heads"])
    d = dsv4_mla_head_dim(model)
    kv_lora_rank = dsv4_kv_lora_rank(model)
    topk = int(info["topk"])
    q = randn_tensor(torch, (total, hq, d), dtype, device)
    kv = randn_tensor(torch, (total, 1, d), dtype, device)
    indices, topk_length = make_flashmla_indices(
        torch, total, args.batch_size, seq, topk, device
    )
    attn_sink = (
        torch.randn(hq, dtype=torch.float32, device=device) if use_attn_sink else None
    )
    topk_len_arg = topk_length if use_topk_length else None
    scale = d**-0.5

    def fn() -> Any:
        out = flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            scale,
            d_v=kv_lora_rank,
            attn_sink=attn_sink,
            topk_length=topk_len_arg,
        )[0]
        out = dsv4_slice_flashmla_output(out, args, info)
        return apply_attention_boundary_communication(torch, args, out)

    timing = add_work_metrics(
        time_cuda(torch, fn, args.warmup_iters, args.iters, device), total, "tokens"
    )
    return {
        **info,
        "attn_sink": use_attn_sink,
        "topk_length_enabled": use_topk_length,
        "warning_messages": warning_messages,
        "tp_communication_mode": tp_communication_mode(args),
        "timing": timing,
    }


def run_flashmla_kvcache_e2e(
    args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    # Synthetic DSV4/NSA attention-layer request. This still avoids model weights
    # and the scheduler object, but it models the production cache lifecycle:
    # each prefill chunk writes KV before attention, and each decode step appends
    # one generated token, grows cache_seqlens, and recomputes page indices.
    kernel = "flashmla-kvcache-e2e"
    info = shape_summary(kernel, args, model)
    warning_messages = list(info.get("warning_messages", []))
    for message in warning_messages:
        warn_stdout(message)

    if info.get("prefill_fallback_reason") and args.strict:
        raise RuntimeError(warning_messages[0])
    if not info.get("decode_supported", True):
        if args.strict:
            raise RuntimeError(warning_messages[-1])
        return {**info, "status": "unsupported", "warning_messages": warning_messages}

    from sgl_kernel.flash_mla import get_mla_metadata
    from sglang.srt.layers.attention.hip_flash_mla import (
        flash_mla_with_kvcache_entrypoint,
    )

    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    dtype = resolve_torch_dtype(torch, activation_dtype_name(args, kernel))
    device = args.device
    hq = int(info["flashmla_q_heads"])
    d = int(info["mla_head_dim"])
    kv_lora_rank = int(info["kv_lora_rank"])
    q_lora_rank = int(info["q_lora_rank"])
    rope_dim = int(info["qk_rope_head_dim"])
    index_heads = int(info["index_n_heads"])
    index_head_dim = int(info["index_head_dim"])
    hidden = model_hidden_size(model)
    c4_indexer_enabled = bool(info["dsv4_c4_indexer_enabled"])
    c4_page_size = int(info["c4_page_size"])
    c4_max_seq_len = int(info["c4_max_seq_len"])
    topk = int(info["topk"])
    swa_topk = int(info["swa_topk"])
    ratio = int(info["dsv4_compress_ratio"])
    page_size = int(info["page_size"])
    pages_per_seq = int(info["pages_per_seq"])
    pages = args.batch_size * pages_per_seq
    cache_dim = int(info["kv_cache_storage_dim"])
    cache_dtype = torch.uint8 if is_requested_fp8(args) else dtype
    is_fp8_cache = bool(is_requested_fp8(args))
    scale = d**-0.5
    attention_entrypoint = str(info["dsv4_attention_kernel_entrypoint"])

    cache_page_bytes = dsv4_kv_cache_page_bytes(args, model, page_size)
    k_cache_flat = torch.zeros(
        pages, cache_page_bytes, dtype=cache_dtype, device=device
    )
    k_cache = dsv4_cache_view_for_attention(k_cache_flat, page_size, cache_dim)
    use_attn_sink = True
    attn_sink = (
        torch.randn(hq, dtype=torch.float32, device=device) if use_attn_sink else None
    )

    extra_k_cache = None
    extra_k_cache_flat = None
    extra_page_size = None
    extra_cache_page_bytes = None
    if ratio in (4, 128):
        extra_page_size = page_size // ratio
        extra_cache_page_bytes = dsv4_kv_cache_page_bytes(args, model, extra_page_size)
        extra_k_cache_flat = torch.zeros(
            pages, extra_cache_page_bytes, dtype=cache_dtype, device=device
        )
        extra_k_cache = dsv4_cache_view_for_attention(
            extra_k_cache_flat, extra_page_size, cache_dim
        )

    chunk_tokens = int(info["prefill_chunk_tokens"])
    chunk_schedule = require_multi_chunk_e2e_schedule(
        kernel, prefill_schedule_from_chunk(input_len(args), chunk_tokens), args
    )
    local_q_heads = int(info["local_q_heads"])

    prefill_hidden = randn_tensor(
        torch, (args.batch_size, input_len(args), hidden), dtype, device
    )
    decode_hidden = randn_tensor(
        torch, (args.output_len, args.batch_size, hidden), dtype, device
    )
    wq_a_weight = randn_tensor(torch, (hidden, q_lora_rank), dtype, device)
    wq_b_weight = randn_tensor(torch, (q_lora_rank, local_q_heads * d), dtype, device)
    wkv_weight = randn_tensor(torch, (hidden, d), dtype, device)
    q_norm_weight = torch.ones(q_lora_rank, dtype=dtype, device=device)
    kv_norm_weight = torch.ones(d, dtype=dtype, device=device)
    freqs_cis = make_freqs_cis(
        torch, input_len(args) + args.output_len + 1, rope_dim, device
    )
    output_project = make_dsv4_output_projector(torch, args, model, info, dtype, device)

    max_request_tokens = pages_per_seq * page_size
    req_to_token = torch.arange(
        args.batch_size * max_request_tokens, dtype=torch.int32, device=device
    ).view(args.batch_size, max_request_tokens)
    full_to_swa = torch.arange(
        args.batch_size * max_request_tokens, dtype=torch.int32, device=device
    )

    compressor_state = None
    compressor_wkv_gate_weight = None
    compressor_ape = None
    compressor_norm_weight = None
    compressor_ring_size = 8 if ratio == 4 else 128
    if ratio in (4, 128):
        coff = 2 if ratio == 4 else 1
        compressor_state_slots = ceil_align(
            pages * compressor_ring_size + compressor_ring_size + ratio + 1, ratio
        )
        compressor_state = torch.zeros(
            compressor_state_slots, 2 * coff * d, dtype=torch.float32, device=device
        )
        compressor_wkv_gate_weight = randn_tensor(
            torch, (2 * coff * d, hidden), dtype, device
        )
        compressor_ape = torch.randn(
            ratio, coff * d, dtype=torch.float32, device=device
        )
        compressor_norm_weight = torch.ones(d, dtype=torch.float32, device=device)

    indexer_k_cache_storage = None
    indexer_k_cache = None
    indexer_wq_b_weight = None
    indexer_weight_proj = None
    indexer_freqs_cis = freqs_cis
    indexer_weight_scale = None
    indexer_compressor_state = None
    indexer_compressor_wkv_gate_weight = None
    indexer_compressor_ape = None
    indexer_compressor_norm_weight = None
    indexer_req_to_token = req_to_token
    indexer_full_to_swa = full_to_swa
    indexer_ring_size = 8
    if c4_indexer_enabled:
        indexer_k_cache_storage = torch.zeros(
            pages, c4_page_size * (index_head_dim + 4), dtype=torch.uint8, device=device
        )
        indexer_k_cache = indexer_k_cache_storage.view(
            pages, c4_page_size, 1, index_head_dim + 4
        )
        indexer_wq_b_weight = randn_tensor(
            torch, (q_lora_rank, index_heads * index_head_dim), dtype, device
        )
        indexer_weight_proj = randn_tensor(torch, (hidden, index_heads), dtype, device)
        indexer_weight_scale = index_head_dim**-0.5 * index_heads**-0.5
        indexer_compressor_state = torch.zeros(
            dsv4_c4_indexer_state_pool_size(pages_per_seq * page_size),
            dsv4_c4_indexer_state_dim(model),
            dtype=torch.float32,
            device=device,
        )
        indexer_compressor_wkv_gate_weight = randn_tensor(
            torch,
            (dsv4_c4_indexer_state_dim(model), hidden),
            dtype,
            device,
        )
        indexer_compressor_ape = torch.randn(
            8, index_head_dim, dtype=torch.float32, device=device
        )
        indexer_compressor_norm_weight = torch.ones(
            index_head_dim, dtype=torch.float32, device=device
        )

    def mla_metadata() -> tuple[Any, Any]:
        # DeepseekV4AttnBackend uses the newer FlashMLASchedMeta API for sparse
        # kvcache decode/prefill. The tensor metadata path asserts that
        # attn_sink and extra_k_cache are absent.
        return get_mla_metadata()

    def rms_norm(x: Any, weight: Any) -> Any:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (
            x * torch.rsqrt(variance + float(args.norm_eps)).to(x.dtype) * weight
        ).to(dtype)

    def compute_q_k_from_hidden(
        hidden_flat: Any, positions: Any
    ) -> tuple[Any, Any, Any]:
        from sglang.jit_kernel.dsv4 import fused_q_norm_rope
        from sglang.jit_kernel.dsv4.compress_old import fused_norm_rope_inplace

        q_lora = rms_norm(hidden_flat.matmul(wq_a_weight), q_norm_weight)
        q_local = q_lora.matmul(wq_b_weight).view(-1, local_q_heads, d).contiguous()
        q_normed = torch.empty_like(q_local)
        fused_q_norm_rope(q_local, q_normed, float(args.norm_eps), freqs_cis, positions)
        if hq != local_q_heads:
            q_padded = torch.zeros(
                q_normed.shape[0], hq, d, dtype=q_normed.dtype, device=device
            )
            q_padded[:, :local_q_heads, :].copy_(q_normed)
            q_normed = q_padded
        kv = hidden_flat.matmul(wkv_weight).contiguous()
        fused_norm_rope_inplace(
            kv, kv_norm_weight, float(args.norm_eps), freqs_cis, positions
        )
        return q_normed.unsqueeze(1).contiguous(), kv, q_lora

    def kvcache_attention(
        q: Any,
        kv_len: int,
        query_len: int,
        query_start_pos: int,
        *,
        hidden_flat: Any | None = None,
        q_lora: Any | None = None,
        indexer_q_lora: Any | None = None,
        indexer_hidden: Any | None = None,
        is_decode: bool = False,
    ) -> Any:
        indices, topk_length = make_flashmla_swa_indices(
            torch,
            batch_size=args.batch_size,
            query_len=query_len,
            pages_per_seq=pages_per_seq,
            page_size=page_size,
            topk=swa_topk,
            device=device,
            query_start_pos=query_start_pos,
        )
        extra_indices = None
        extra_topk_length = None
        if extra_k_cache is not None and extra_page_size is not None:
            assert hidden_flat is not None
            assert (
                compressor_state is not None and compressor_wkv_gate_weight is not None
            )
            assert compressor_ape is not None and compressor_norm_weight is not None
            assert extra_k_cache_flat is not None
            compression_metadata = make_dsv4_compression_metadata(
                torch,
                batch_size=args.batch_size,
                query_len=query_len,
                query_start_pos=query_start_pos,
                pages_per_seq=pages_per_seq,
                page_size=page_size,
                device=device,
            )
            compressor_metadata = make_dsv4_compressor_metadata(
                torch,
                ratio=ratio,
                batch_size=args.batch_size,
                query_len=query_len,
                query_start_pos=query_start_pos,
                req_to_token=req_to_token,
                full_to_swa=full_to_swa,
                swa_page_size=page_size,
                ring_size=compressor_ring_size,
                device=device,
                is_decode=is_decode,
            )
            dsv4_compress_and_write_extra_cache(
                torch,
                hidden=hidden_flat.view(-1, hidden).contiguous(),
                ratio=ratio,
                metadata=compressor_metadata,
                compressor_state=compressor_state,
                wkv_gate_weight=compressor_wkv_gate_weight,
                ape=compressor_ape,
                norm_weight=compressor_norm_weight,
                freqs_cis=freqs_cis,
                norm_eps=float(args.norm_eps),
                extra_cache_2d=extra_k_cache_flat,
                extra_cache_out_loc=compression_metadata["c4_out_loc"]
                if ratio == 4
                else compression_metadata["c128_out_loc"],
                extra_page_tokens=extra_page_size,
                head_dim=d,
            )
            if ratio == 4:
                if not c4_indexer_enabled:
                    raise RuntimeError("DSV4 CSA requires the C4 indexer path")
                if indexer_q_lora is None:
                    indexer_q_lora = q_lora
                if indexer_hidden is None:
                    indexer_hidden = hidden_flat
                assert indexer_q_lora is not None and indexer_hidden is not None
                assert (
                    indexer_k_cache is not None and indexer_k_cache_storage is not None
                )
                assert (
                    indexer_wq_b_weight is not None
                    and indexer_weight_proj is not None
                    and indexer_freqs_cis is not None
                )
                assert indexer_weight_scale is not None
                assert (
                    indexer_compressor_state is not None
                    and indexer_compressor_wkv_gate_weight is not None
                )
                assert (
                    indexer_compressor_ape is not None
                    and indexer_compressor_norm_weight is not None
                )
                assert (
                    indexer_req_to_token is not None and indexer_full_to_swa is not None
                )
                indexer_q_lora_flat = indexer_q_lora.view(-1, q_lora_rank).contiguous()
                indexer_hidden_flat = indexer_hidden.view(-1, hidden).contiguous()
                update_dsv4_c4_indexer_cache(
                    torch,
                    hidden=indexer_hidden_flat,
                    c4_out_loc=compression_metadata["c4_out_loc"],
                    query_len=query_len,
                    query_start_pos=query_start_pos,
                    req_to_token=indexer_req_to_token,
                    full_to_swa=indexer_full_to_swa,
                    compressor_state=indexer_compressor_state,
                    wkv_gate_weight=indexer_compressor_wkv_gate_weight,
                    ape=indexer_compressor_ape,
                    norm_weight=indexer_compressor_norm_weight,
                    freqs_cis=indexer_freqs_cis,
                    indexer_k_cache_storage=indexer_k_cache_storage,
                    ring_size=indexer_ring_size,
                    c4_page_size=c4_page_size,
                    index_head_dim=index_head_dim,
                    norm_eps=float(args.norm_eps),
                    is_decode=is_decode,
                )
                extra_indices, extra_topk_length = dsv4_c4_indexer_page_indices(
                    torch,
                    indexer_q_lora_flat,
                    indexer_hidden_flat,
                    compression_metadata["positions"],
                    compression_metadata["c4_seq_lens_raw"],
                    compression_metadata["c4_seq_lens_clamp1"],
                    compression_metadata["page_table"],
                    indexer_k_cache,
                    indexer_wq_b_weight,
                    indexer_weight_proj,
                    indexer_freqs_cis,
                    indexer_weight_scale,
                    topk,
                    c4_page_size,
                    c4_max_seq_len,
                )
            elif ratio == 128:
                extra_indices = pad_page_indices_last_dim(
                    torch, compression_metadata["c128_page_indices"]
                )
                extra_topk_length = compression_metadata["c128_topk_lengths_clamp1"]
            else:
                raise ValueError(f"unsupported compressed DSV4 ratio for E2E: {ratio}")
        if extra_indices is not None and extra_indices.ndim == 2:
            extra_indices = extra_indices[:, None, :]
        q_arg, q_quant_kwargs = quantize_flashmla_q_if_needed(torch, args, q)
        metadata, num_splits = mla_metadata()
        out = flash_mla_with_kvcache_entrypoint(
            backend=attention_entrypoint,
            q=q_arg,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=kv_lora_rank,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            softmax_scale=scale,
            causal=False,
            is_fp8_kvcache=is_fp8_cache,
            indices=indices,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_length,
            **q_quant_kwargs,
        )[0]
        return dsv4_slice_flashmla_output(out, args, info)

    def write_main_cache(start_pos: int, q_len: int, kv: Any) -> None:
        loc = make_cache_locs(
            torch, args.batch_size, start_pos, q_len, pages_per_seq, page_size, device
        )
        write_dsv4_fused_cache(
            torch, args, model, k_cache_flat, loc, kv[: loc.numel()], page_size
        )

    def reset_state() -> None:
        k_cache_flat.zero_()
        if extra_k_cache_flat is not None:
            extra_k_cache_flat.zero_()
        if compressor_state is not None:
            compressor_state.zero_()
        if indexer_k_cache_storage is not None:
            indexer_k_cache_storage.zero_()
        if indexer_compressor_state is not None:
            indexer_compressor_state.zero_()

    def run_prefill() -> Any:
        out = None
        start_pos = 0
        for q_len in chunk_schedule:
            token_count = args.batch_size * q_len
            chunk_slice = slice(start_pos, start_pos + q_len)
            hidden_chunk = (
                prefill_hidden[:, chunk_slice, :]
                .reshape(token_count, hidden)
                .contiguous()
            )
            positions = make_dsv4_positions(
                torch, args.batch_size, q_len, start_pos, device
            )
            q_chunk, kv_chunk, q_lora_chunk = compute_q_k_from_hidden(
                hidden_chunk, positions
            )
            write_main_cache(start_pos, q_len, kv_chunk)
            out = kvcache_attention(
                q_chunk,
                kv_len=start_pos + q_len,
                query_len=q_len,
                query_start_pos=start_pos,
                hidden_flat=hidden_chunk,
                q_lora=q_lora_chunk,
                is_decode=False,
            )
            out = output_project(out, positions)
            out = apply_attention_boundary_communication(
                torch, args, out, pure_tp=False, dp_attention=True
            )
            start_pos += q_len
        return out

    def run_decode() -> Any:
        out = None
        for step in range(args.output_len):
            pos = input_len(args) + step
            hidden_step = (
                decode_hidden[step].reshape(args.batch_size, hidden).contiguous()
            )
            positions = make_dsv4_positions(torch, args.batch_size, 1, pos, device)
            q_step, kv_step, q_lora_step = compute_q_k_from_hidden(
                hidden_step, positions
            )
            write_main_cache(pos, 1, kv_step)
            out = kvcache_attention(
                q_step,
                kv_len=pos + 1,
                query_len=1,
                query_start_pos=pos,
                hidden_flat=hidden_step,
                q_lora=q_lora_step,
                is_decode=True,
            )
            out = output_project(out, positions)
            out = apply_attention_boundary_communication(
                torch, args, out, pure_tp=False, dp_attention=True
            )
        return out

    timing = phase_timing(
        torch,
        args,
        device,
        reset_state=reset_state,
        run_prefill=run_prefill,
        run_decode=run_decode,
        prefill_work_items=args.batch_size * input_len(args),
        decode_work_items=args.batch_size * args.output_len,
        unit="tokens",
    )
    return {
        **info,
        "attn_sink": use_attn_sink,
        "production_forward_includes_input_projections": True,
        "production_forward_includes_rope": True,
        "production_forward_includes_kv_cache_writes": True,
        "production_forward_includes_output_projection": True,
        "production_forward_includes_compressor": bool(ratio in (4, 128)),
        "production_forward_uses_indexer": bool(c4_indexer_enabled),
        "dsv4_kv_cache_layout": "DeepSeekV4SingleKVPool packed FP8 nope/BF16 rope page layout",
        "kvcache_topk_length_enabled": True,
        "dsv4_c4_indexer_enabled": c4_indexer_enabled,
        "dsv4_attention_kernel": args.dsv4_attention_kernel,
        "dsv4_attention_kernel_entrypoint": attention_entrypoint,
        "tp_communication_mode": tp_communication_mode(args),
        "warning_messages": warning_messages,
        "dsv4_prefill_schedule": chunk_schedule,
        "timing": timing,
    }


@dataclass
class LegacyContext:
    bs: int
    head_dim: int
    compress_ratio: int
    req_pool_indices: Any
    pages_per_req: int
    torch: Any

    @property
    def num_pages(self) -> int:
        return int(self.req_pool_indices.max().item() + 1) * self.pages_per_req

    def make_prefill_plan(
        self, seq_lens_cpu: Any, extend_lens_cpu: Any, num_q_tokens: int
    ) -> Any:
        from sglang.jit_kernel.dsv4 import CompressorPrefillPlan

        return CompressorPrefillPlan.generate_legacy(
            self.compress_ratio,
            self.req_pool_indices,
            seq_lens_cpu,
            extend_lens_cpu,
            num_q_tokens,
            device=self.torch.device("cuda"),
        )

    def make_decode_plan(self, seq_lens_gpu: Any) -> Any:
        from sglang.jit_kernel.dsv4 import CompressorDecodePlan

        return CompressorDecodePlan.generate_legacy(
            self.compress_ratio, self.req_pool_indices, seq_lens_gpu
        )


@dataclass
class PagedContext:
    bs: int
    head_dim: int
    compress_ratio: int
    swa_page_size: int
    ring_size: int
    num_swa_pages_per_req: int
    req_pool_indices: Any
    req_to_token: Any
    full_to_swa: Any

    @property
    def num_pages(self) -> int:
        max_state_loc = (
            self.bs * self.num_swa_pages_per_req * self.ring_size + self.swa_page_size
        )
        return max_state_loc // self.compress_ratio + 1

    def make_prefill_plan(
        self, seq_lens_cpu: Any, extend_lens_cpu: Any, num_q_tokens: int
    ) -> Any:
        from sglang.jit_kernel.dsv4 import CompressorPrefillPlan

        return CompressorPrefillPlan.generate(
            self.compress_ratio,
            self.req_pool_indices,
            seq_lens_cpu,
            extend_lens_cpu,
            self.req_to_token,
            self.full_to_swa,
            self.swa_page_size,
            self.ring_size,
            num_q_tokens,
        )

    def make_decode_plan(self, seq_lens_gpu: Any) -> Any:
        from sglang.jit_kernel.dsv4 import CompressorDecodePlan

        return CompressorDecodePlan.generate(
            self.compress_ratio,
            self.req_pool_indices,
            self.req_to_token,
            self.full_to_swa,
            seq_lens_gpu,
            self.swa_page_size,
            self.ring_size,
        )


def make_dsv4_context(
    torch: Any, args: argparse.Namespace, ratio: int, batch: int
) -> Any:
    if args.dsv4_context == "legacy":
        req_pool_indices = torch.arange(batch, dtype=torch.int64, device=args.device)
        return LegacyContext(
            batch, 512, ratio, req_pool_indices, 2 if ratio == 4 else 1, torch
        )
    ring_size = args.ring_size or (8 if ratio == 4 else 128)
    if args.swa_page_size % ring_size != 0:
        raise ValueError("--swa-page-size must be divisible by ring_size")
    if ring_size % ratio != 0:
        raise ValueError("ring_size must be divisible by compress_ratio")
    pages_per_req = math.ceil(args.seq_len / args.swa_page_size)
    max_tokens = pages_per_req * args.swa_page_size
    capacity = max(batch, 16)
    stride = pages_per_req * args.swa_page_size
    req_to_token = torch.zeros(
        (capacity, max_tokens), dtype=torch.int32, device=args.device
    )
    for r in range(batch):
        req_to_token[r, :stride] = torch.arange(
            r * stride, (r + 1) * stride, dtype=torch.int32, device=args.device
        )
    full_to_swa = torch.arange(capacity * stride, dtype=torch.int64, device=args.device)
    req_pool_indices = torch.arange(batch, dtype=torch.int64, device=args.device)
    return PagedContext(
        batch,
        512,
        ratio,
        args.swa_page_size,
        ring_size,
        pages_per_req,
        req_pool_indices,
        req_to_token,
        full_to_swa,
    )


def run_dsv4_compress(
    kernel: str, args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    from sglang.jit_kernel.dsv4 import compress_forward

    ratio = 4 if "-c4-" in kernel else 128
    mode = "decode" if "decode" in kernel else "prefill"
    info = shape_summary(kernel, args, model)
    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    if int(model.get("head_dim", 512)) != 512:
        raise ValueError(
            "DeepSeek V4 compression kernels are specialized for head_dim=512"
        )
    batch = (
        (args.decode_batch_size or args.batch_size)
        if mode == "decode"
        else args.batch_size
    )
    ctx = make_dsv4_context(torch, args, ratio, batch)
    last_dim = 512 * (4 if ratio == 4 else 2)
    window = 8 if ratio == 4 else 128
    pool = torch.zeros(
        (ctx.num_pages, ratio, last_dim), dtype=torch.float32, device=args.device
    )
    ape = torch.randn(window, 512, dtype=torch.float32, device=args.device)
    if mode == "decode":
        kv_score_input = torch.randn(
            batch, last_dim, dtype=torch.float32, device=args.device
        )
        seq_lens_gpu = torch.full(
            (batch,), args.seq_len, dtype=torch.int64, device=args.device
        )

        def fn() -> Any:
            plan = ctx.make_decode_plan(seq_lens_gpu)
            return compress_forward(
                pool, kv_score_input, ape, plan, head_dim=512, compress_ratio=ratio
            )

        work_tokens = batch
    else:
        seq = input_len(args)
        total = args.batch_size * seq
        kv_score_input = torch.randn(
            total, last_dim, dtype=torch.float32, device=args.device
        )
        seq_lens_cpu = torch.full((args.batch_size,), seq, dtype=torch.int64)
        extend_lens_cpu = torch.full((args.batch_size,), seq, dtype=torch.int64)

        def fn() -> Any:
            plan = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, total)
            return compress_forward(
                pool, kv_score_input, ape, plan, head_dim=512, compress_ratio=ratio
            )

        work_tokens = total

    timing = add_work_metrics(
        time_cuda(torch, fn, args.warmup_iters, args.iters, args.device),
        work_tokens,
        "tokens",
    )
    return {**info, "dsv4_context": args.dsv4_context, "timing": timing}


def linear_shapes(
    args: argparse.Namespace, model: dict[str, Any]
) -> tuple[int, int, int, int, int]:
    atp = attn_tp_size(args)
    h = require_divisible(
        "linear_num_key_heads", int(model["linear_num_key_heads"]), atp
    )
    hv = require_divisible(
        "linear_num_value_heads", int(model["linear_num_value_heads"]), atp
    )
    k = int(model["linear_key_head_dim"])
    v = int(model["linear_value_head_dim"])
    pool = args.pool_size or max(
        args.batch_size, args.decode_batch_size or args.batch_size, 1
    )
    return h, hv, k, v, pool


def make_linear_output_projector(
    torch: Any,
    args: argparse.Namespace,
    model: dict[str, Any],
    dtype: Any,
) -> Callable[[Any, Any], Any]:
    from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated

    _, hv, _, v, _ = linear_shapes(args, model)
    hidden = model_hidden_size(model)
    norm = RMSNormGated(
        v,
        eps=float(model.get("rms_norm_eps", 1e-6)),
        group_size=None,
        norm_before_gate=True,
        device=args.device,
        dtype=dtype,
    )
    norm.eval()
    weight = randn_tensor(torch, (hv * v, hidden), dtype, args.device)

    def project(value: Any, z: Any) -> Any:
        tensor = first_tensor(value)
        if tensor is None:
            return value
        x = tensor.contiguous()
        if int(x.shape[-1]) == hv * v:
            x_heads = x.view(*x.shape[:-1], hv, v).contiguous()
        elif len(x.shape) >= 2 and int(x.shape[-2]) == hv and int(x.shape[-1]) == v:
            x_heads = x
        else:
            raise RuntimeError(
                f"GDN output projection expected [..., {hv}, {v}] or [..., {hv * v}], got {tuple(x.shape)}"
            )
        z_heads = z.contiguous()
        if z_heads.shape != x_heads.shape:
            if z_heads.numel() == x_heads.numel():
                z_heads = z_heads.view_as(x_heads)
            else:
                raise RuntimeError(
                    f"GDN output gate shape mismatch: z={tuple(z_heads.shape)} core={tuple(x_heads.shape)}"
                )
        normed = norm(x_heads.reshape(-1, v), z_heads.reshape(-1, v)).view_as(x_heads)
        flat = normed.reshape(*normed.shape[:-2], hv * v)
        projected = flat.reshape(-1, hv * v).matmul(weight)
        return projected.view(*flat.shape[:-1], hidden).contiguous()

    return project


def make_gdn_prefill_inputs(
    torch: Any,
    args: argparse.Namespace,
    model: dict[str, Any],
    seq_len: int | None = None,
) -> dict[str, Any]:
    kernel_name = "linear-gdn-e2e" if seq_len is not None else "linear-triton-prefill"
    dtype = resolve_torch_dtype(torch, activation_dtype_name(args, kernel_name))
    state_dtype = resolve_torch_dtype(torch, args.state_dtype)
    h, hv, k, v, pool = linear_shapes(args, model)
    seq = seq_len or input_len(args)
    total = args.batch_size * seq
    q = randn_tensor(torch, (1, total, h, k), dtype, args.device)
    key = randn_tensor(torch, (1, total, h, k), dtype, args.device)
    value = randn_tensor(torch, (1, total, hv, v), dtype, args.device)
    z = randn_tensor(torch, (1, total, hv, v), dtype, args.device)
    a = torch.randn(total, hv, dtype=torch.float32, device=args.device)
    b = torch.randn(total, hv, dtype=torch.float32, device=args.device)
    a_log = torch.randn(hv, dtype=torch.float32, device=args.device)
    dt_bias = torch.randn(hv, dtype=torch.float32, device=args.device)
    gate = torch.nn.functional.logsigmoid(
        torch.randn(1, total, hv, dtype=torch.float32, device=args.device)
    )
    beta = torch.rand(1, total, hv, dtype=torch.float32, device=args.device)
    state = torch.zeros(pool, hv, v, k, dtype=state_dtype, device=args.device)
    cache_indices = torch.arange(args.batch_size, dtype=torch.int32, device=args.device)
    query_start_loc = torch.arange(
        0, (args.batch_size + 1) * seq, seq, dtype=torch.int32, device=args.device
    )
    return {
        "q": q,
        "k": key,
        "v": value,
        "z": z,
        "g": gate,
        "beta": beta,
        "a": a,
        "b": b,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "state": state,
        "cache_indices": cache_indices,
        "query_start_loc": query_start_loc,
        "h": h,
        "hv": hv,
        "head_k": k,
        "head_v": v,
    }


def make_gdn_decode_inputs(
    torch: Any,
    args: argparse.Namespace,
    model: dict[str, Any],
    state: Any | None = None,
    steps: int | None = None,
) -> dict[str, Any]:
    dtype = resolve_torch_dtype(
        torch, activation_dtype_name(args, "linear-triton-decode")
    )
    state_dtype = resolve_torch_dtype(torch, args.state_dtype)
    h, hv, k, v, pool = linear_shapes(args, model)
    b = args.decode_batch_size or args.batch_size
    pool = max(pool, b)
    prefix = (int(steps),) if steps is not None else ()
    q = randn_tensor(torch, prefix + (1, b, h, k), dtype, args.device)
    key = randn_tensor(torch, prefix + (1, b, h, k), dtype, args.device)
    value = randn_tensor(torch, prefix + (1, b, hv, v), dtype, args.device)
    z = randn_tensor(torch, prefix + (1, b, hv, v), dtype, args.device)
    a = torch.randn(*(prefix + (b, hv)), dtype=torch.float32, device=args.device)
    beta = torch.randn(*(prefix + (b, hv)), dtype=torch.float32, device=args.device)
    a_log = torch.randn(hv, dtype=torch.float32, device=args.device)
    dt_bias = torch.randn(hv, dtype=torch.float32, device=args.device)
    if state is None:
        state = torch.zeros(pool, hv, v, k, dtype=state_dtype, device=args.device)
    cache_indices = torch.arange(b, dtype=torch.int32, device=args.device)
    query_start_loc = torch.arange(b + 1, dtype=torch.int32, device=args.device)
    mixed_dim = 2 * h * k + hv * v
    mixed_qkv = randn_tensor(torch, prefix + (b, mixed_dim), dtype, args.device)
    return {
        "q": q,
        "k": key,
        "v": value,
        "z": z,
        "a": a,
        "b": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "state": state,
        "cache_indices": cache_indices,
        "query_start_loc": query_start_loc,
        "mixed_qkv": mixed_qkv,
        "h": h,
        "hv": hv,
        "head_k": k,
        "head_v": v,
        "batch": b,
    }


class BenchmarkServerArgs(SimpleNamespace):
    def get_attention_backends(self) -> tuple[str, str]:
        prefill = self.prefill_attention_backend or self.attention_backend
        decode = self.decode_attention_backend or self.attention_backend
        return prefill, decode

    def use_mla_backend(self) -> bool:
        return True


class BenchmarkMLAModelConfig:
    def __init__(
        self, model: dict[str, Any], *, architecture: str, context_len: int, dtype: Any
    ):
        from sglang.srt.configs.model_config import AttentionArch

        self.attention_arch = AttentionArch.MLA
        self.context_len = context_len
        self.dtype = dtype
        self.hidden_size = int(model["hidden_size"])
        self.num_attention_heads = int(model["num_attention_heads"])
        self.num_key_value_heads = int(model.get("num_key_value_heads", 1))
        self.qk_nope_head_dim = int(
            model.get("qk_nope_head_dim", model.get("head_dim", 128))
        )
        self.qk_rope_head_dim = int(model.get("qk_rope_head_dim", 0))
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # SGLang ModelConfig normalizes all DeepSeek/GLM MLA-family models to
        # head_dim=256 for backend metadata; qk_nope/qk_rope/v dims stay separate.
        self.head_dim = int(model.get("sglang_model_config_head_dim", 256))
        self.v_head_dim = int(model.get("v_head_dim", self.head_dim))
        self.swa_v_head_dim = self.v_head_dim
        self.kv_lora_rank = int(model.get("kv_lora_rank", self.v_head_dim))
        self.q_lora_rank = int(model.get("q_lora_rank", 0))
        self.scaling = self.qk_head_dim**-0.5
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.is_hybrid_swa = False
        self.is_local_attention_model = False
        self.attention_chunk_size = None
        self.sliding_window_size = model.get("sliding_window")
        self.hf_config = SimpleNamespace(
            architectures=[architecture],
            model_type="glm_moe_dsa"
            if architecture == "GlmMoeDsaForCausalLM"
            else "deepseek_v3",
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.qk_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            index_head_dim=model.get("index_head_dim"),
            index_n_heads=model.get("index_n_heads"),
            index_topk=model.get("index_topk"),
            index_topk_freq=model.get("index_topk_freq"),
            index_skip_topk_offset=model.get("index_skip_topk_offset"),
            index_topk_pattern=model.get("index_topk_pattern"),
            indexer_rope_interleave=model.get("indexer_rope_interleave"),
            rope_interleave=model.get("rope_interleave", True),
            max_position_embeddings=int(
                model.get("max_position_embeddings", context_len)
            ),
            rms_norm_eps=float(model.get("rms_norm_eps", 1e-6)),
            rope_theta=float(model.get("rope_theta", 10000.0)),
            rope_scaling=None,
            num_hidden_layers=int(model.get("num_hidden_layers", 1)),
        )
        self.hf_text_config = self.hf_config

    def get_num_attention_heads(self, tp_size: int) -> int:
        assert self.num_attention_heads % tp_size == 0
        return self.num_attention_heads // tp_size

    def get_num_kv_heads(self, tp_size: int) -> int:
        _ = tp_size
        return 1


def torch_cuda_major(torch: Any, device: int) -> int:
    return int(torch.cuda.get_device_capability(device)[0])


def kv_cache_torch_dtype(torch: Any, dtype_name: str, fallback_dtype: Any) -> Any:
    if dtype_name == "fp8_e4m3":
        return torch.float8_e4m3fn
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return fallback_dtype


def make_cuda_graph_config_disabled() -> Any:
    try:
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            CudaGraphConfig,
            PhaseConfig,
        )

        return CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.DISABLED),
        )
    except Exception:
        disabled = SimpleNamespace(backend="disabled")
        return SimpleNamespace(decode=disabled, prefill=disabled)


def make_benchmark_server_args(
    args: argparse.Namespace,
    *,
    attention_backend: str,
    kv_cache_dtype: str,
    dsa_prefill_backend: str | None = None,
    dsa_decode_backend: str | None = None,
) -> BenchmarkServerArgs:
    return BenchmarkServerArgs(
        attention_backend=attention_backend,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        linear_attn_model_spec=None,
        dsa_prefill_backend=dsa_prefill_backend,
        dsa_decode_backend=dsa_decode_backend,
        dsa_prefill_cp_mode="round-robin-split",
        dsa_topk_backend=args.dsa_topk_backend,
        kv_cache_dtype=kv_cache_dtype,
        device="cuda",
        cuda_graph_config=make_cuda_graph_config_disabled(),
        disable_cuda_graph=True,
        chunked_prefill_size=args.chunked_prefill_size,
        disable_chunked_prefix_cache=True,
        disaggregation_mode=None,
        dllm_algorithm=None,
        dllm_algorithm_config=None,
        dp_size=attention_dp_size(args),
        attn_cp_size=1,
        moe_dp_size=1,
        cp_strategy=None,
        enable_attn_tp_input_scattered=False,
        enable_deterministic_inference=False,
        enable_dp_attention=bool(args.enable_dp_attention),
        enable_dsa_prefill_context_parallel=False,
        enable_hierarchical_cache=False,
        enable_mis=False,
        enable_mla_prefill_context_parallel=False,
        enable_moe_dense_fully_dp=False,
        enable_prefill_cp=False,
        enable_prefill_context_parallel=False,
        enable_symm_mem=False,
        flashinfer_mla_disable_ragged=True,
        is_embedding=False,
        max_running_requests=None,
        mem_fraction_static=0.8,
        model_path=None,
        moe_a2a_backend="none",
        moe_dense_tp_size=None,
        page_size=args.dsa_page_size
        if attention_backend == "dsa"
        else args.mla_page_size,
        pp_size=1,
        revision=None,
        rl_on_policy_target=None,
        speculative_algorithm=None,
        speculative_attention_mode="prefill",
        speculative_eagle_topk=0,
        speculative_num_draft_tokens=0,
        speculative_num_steps=0,
        tp_size=args.tp_size,
        triton_attention_num_kv_splits=8,
        triton_attention_split_tile_size=None,
    )


def prepopulate_req_to_token_pool(
    torch: Any, runner: Any, *, batch_size: int, max_context_len: int, device: int
) -> None:
    for req_idx in range(batch_size):
        start = req_idx * max_context_len
        locs = torch.arange(
            start, start + max_context_len, dtype=torch.int32, device=device
        )
        runner.req_to_token_pool.req_to_token[req_idx, :max_context_len] = locs


def initialize_module_forward_batch_dp_metadata(
    torch: Any,
    args: argparse.Namespace,
    batch: Any,
    *,
    token_count: int,
    is_extend: bool,
) -> None:
    if not bool(getattr(args, "enable_dp_attention", False)):
        return
    from sglang.srt.layers.dp_attention import (
        DpPaddingMode,
        get_attention_dp_rank,
        get_attention_dp_size,
        set_dp_buffer_len,
        set_is_extend_in_batch,
    )

    dp_size = int(get_attention_dp_size())
    dp_rank = int(get_attention_dp_rank())
    local_tokens = int(token_count)
    global_num_tokens = [local_tokens] * dp_size
    dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
        bool(is_extend), list(global_num_tokens)
    )
    if dp_padding_mode.is_max_len():
        padded_tokens = max(global_num_tokens)
        padded_global_num_tokens = [padded_tokens] * dp_size
        buffer_len = padded_tokens * dp_size
    else:
        padded_global_num_tokens = list(global_num_tokens)
        buffer_len = sum(padded_global_num_tokens)
    rank_tokens = (
        padded_global_num_tokens[dp_rank]
        if len(padded_global_num_tokens) > 1
        else padded_global_num_tokens[0]
    )

    batch.is_extend_in_batch = bool(is_extend)
    batch.all_extend_in_batch = bool(is_extend)
    batch.original_global_num_tokens_cpu = list(global_num_tokens)
    batch.global_num_tokens_cpu = list(padded_global_num_tokens)
    batch.global_num_tokens_for_logprob_cpu = list(padded_global_num_tokens)
    batch.global_num_tokens_gpu = torch.tensor(
        padded_global_num_tokens, dtype=torch.int64, device=batch.seq_lens.device
    )
    batch.global_num_tokens_for_logprob_gpu = torch.tensor(
        padded_global_num_tokens, dtype=torch.int64, device=batch.seq_lens.device
    )
    batch.dp_padding_mode = dp_padding_mode
    batch.global_dp_buffer_len = int(buffer_len)
    batch.tbo_padded_len = int(rank_tokens)
    set_dp_buffer_len(
        int(buffer_len),
        int(rank_tokens),
        dp_padding_mode.is_max_len(),
        list(padded_global_num_tokens),
    )
    set_is_extend_in_batch(bool(is_extend))


def make_module_forward_batch(
    torch: Any,
    args: argparse.Namespace,
    runner: Any,
    forward_mode: Any,
    *,
    prefix_len: int,
    query_len: int,
    max_context_len: int,
    mha_one_shot: bool = False,
) -> Any:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    batch_size = args.batch_size
    device = args.device
    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    seq_lens = [prefix_len + query_len] * batch_size
    out_cache_locs = []
    positions = []
    for req_idx in range(batch_size):
        base = req_idx * max_context_len
        for offset in range(query_len):
            pos = prefix_len + offset
            positions.append(pos)
            out_cache_locs.append(base + pos)
    batch = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=batch_size,
        input_ids=torch.arange(
            batch_size * query_len, dtype=torch.int64, device=device
        ),
        req_pool_indices=req_pool_indices,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.tensor(out_cache_locs, dtype=torch.int64, device=device),
        seq_lens_sum=sum(seq_lens),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
    )
    if forward_mode.is_extend(include_draft_extend_v2=True):
        extend_seq_lens = torch.full(
            (batch_size,), query_len, dtype=torch.int32, device=device
        )
        batch.extend_prefix_lens = torch.full(
            (batch_size,), prefix_len, dtype=torch.int32, device=device
        )
        batch.extend_prefix_lens_cpu = [prefix_len] * batch_size
        batch.extend_seq_lens = extend_seq_lens
        batch.extend_seq_lens_cpu = [query_len] * batch_size
        batch.extend_start_loc = torch.arange(
            0, (batch_size + 1) * query_len, query_len, dtype=torch.int32, device=device
        )[:-1]
        batch.extend_num_tokens = batch_size * query_len
        if mha_one_shot:
            # Production scheduler marks first MLA-family prefill chunks this way
            # so FA3 uses raw MHA q/k/v instead of reading the MLA KV pool.
            batch.attn_attend_prefix_cache = False
            batch.mha_one_shot = True
            batch.mha_return_lse = False
    initialize_module_forward_batch_dp_metadata(
        torch,
        args,
        batch,
        token_count=batch_size * query_len,
        is_extend=forward_mode.is_extend(include_draft_extend_v2=True),
    )
    return batch


def initialize_synthetic_attention_weights(torch: Any, module: Any, dtype: Any) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters(recurse=True):
            if not hasattr(param, "is_floating_point") or not param.is_floating_point():
                continue
            if param.ndim == 1 and ("norm" in name or name.endswith("weight")):
                param.fill_(1.0)
            else:
                param.normal_(mean=0.0, std=0.02)
            if param.dtype != dtype:
                param.data = param.data.to(dtype)


def restore_sglang_rope_cache_dtype(torch: Any, module: Any, device: str | int) -> None:
    # SGLang CUDA RoPE kernels keep cos/sin cache in fp32. A standalone
    # module.to(dtype=bf16) would otherwise cast non-parameter buffers too.
    cuda_device = torch.device("cuda", device)
    for owner in (
        getattr(module, "rotary_emb", None),
        getattr(getattr(module, "indexer", None), "rotary_emb", None),
    ):
        if owner is None or not hasattr(owner, "cos_sin_cache"):
            continue
        owner.cos_sin_cache = owner.cos_sin_cache.to(
            device=cuda_device, dtype=torch.float32
        )


def prepare_absorbed_mla_weights(module: Any) -> None:
    w = module.kv_b_proj.weight
    w_kc, w_vc = w.unflatten(
        0, (-1, module.qk_nope_head_dim + module.v_head_dim)
    ).split([module.qk_nope_head_dim, module.v_head_dim], dim=1)
    module.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
    module.w_vc = w_vc.contiguous().transpose(1, 2)
    module.w_scale = 1.0


def install_benchmark_attention_dispatch(
    attn: Any, backend: Any, *, prefill_backend: str, decode_backend: str
) -> None:
    # Some SGLang images used by the jobs predate the per-backend-name stamping
    # in DeepseekV2AttentionMLA.dispatch_attn_forward_method. Mirror the server
    # dispatch here so standalone module benchmarks cannot silently fall back to
    # the generic MHA path while running under the DSA backend object.
    try:
        from sglang.srt.models.deepseek_common.attention_backend_handler import (
            AttentionBackendRegistry,
        )
    except Exception:
        AttentionBackendRegistry = None
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
        AttnForwardMethod,
    )

    original_dispatch = attn.dispatch_attn_forward_method

    def _fallback_dispatch(
        self: Any, forward_batch: Any, attention_backend: str
    ) -> Any:
        if (
            attention_backend in ("fa3", "flashinfer", "flashmla")
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None) or []
            if sum(prefix_lens) == 0:
                return AttnForwardMethod.MHA_ONE_SHOT
        if (
            attention_backend in ("triton",)
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None) or []
            if sum(prefix_lens) == 0:
                return AttnForwardMethod.MHA
        return AttnForwardMethod.MLA

    def _benchmark_dispatch(self: Any, forward_batch: Any) -> Any:
        if forward_batch.forward_mode.is_decode_or_idle():
            attention_backend = decode_backend
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            attention_backend = prefill_backend
        else:
            attention_backend = prefill_backend
        self.current_attention_backend = attention_backend
        if attention_backend in ("dsa", "nsa"):
            if getattr(backend, "use_mha", False):
                return AttnForwardMethod.MHA_ONE_SHOT
            return AttnForwardMethod.MLA
        if AttentionBackendRegistry is not None:
            return AttentionBackendRegistry.get_handler(attention_backend)(
                self, forward_batch
            )
        try:
            return _fallback_dispatch(self, forward_batch, attention_backend)
        except Exception:
            return original_dispatch(forward_batch)

    attn.dispatch_attn_forward_method = MethodType(_benchmark_dispatch, attn)


def make_sglang_mla_runtime(
    torch: Any,
    args: argparse.Namespace,
    model: dict[str, Any],
    *,
    is_dsa: bool,
    dtype: Any,
) -> dict[str, Any]:
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.mem_cache.memory_pool import (
        DSATokenToKVPool,
        MLATokenToKVPool,
        ReqToTokenPool,
    )
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.server_args import set_global_server_args_for_scheduler

    major = torch_cuda_major(torch, args.device)
    if is_dsa:
        prefill_backend, decode_backend, kv_cache_dtype_name = (
            resolve_dsa_backend_defaults(args, major)
        )
        attention_backend = "dsa"
        architecture = "GlmMoeDsaForCausalLM"
        page_size = int(args.dsa_page_size)
        layer_id = int(args.glm_dsa_layer_id)
    else:
        prefill_backend = decode_backend = None
        kv_cache_dtype_name = resolve_mla_kv_cache_dtype(args, major)
        attention_backend = resolve_mla_attention_backend(args, major)
        architecture = "DeepseekV3ForCausalLM"
        page_size = int(args.mla_page_size)
        layer_id = 0

    server_args = make_benchmark_server_args(
        args,
        attention_backend=attention_backend,
        kv_cache_dtype=kv_cache_dtype_name,
        dsa_prefill_backend=prefill_backend,
        dsa_decode_backend=decode_backend,
    )
    set_global_server_args_for_scheduler(server_args)
    if is_dsa:
        from sglang.srt.environ import envs

        # Match ServerArgs post-processing for DeepSeek/GLM DSA: prefill may use
        # dense MHA only while the full KV length is no larger than model top-k.
        envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.set(
            int(model["index_topk"])
        )

    total_context = input_len(args) + args.output_len
    max_context_len = total_context + page_size
    pool_size = args.batch_size * max_context_len + page_size
    model_config = BenchmarkMLAModelConfig(
        model, architecture=architecture, context_len=max_context_len, dtype=dtype
    )
    initialize_sglang_dp_attention_state(
        torch, args, model, server_args=server_args, model_config=model_config
    )
    runner = SimpleNamespace(
        device="cuda",
        dtype=dtype,
        kv_cache_dtype=kv_cache_torch_dtype(torch, kv_cache_dtype_name, dtype),
        gpu_id=args.device,
        canary_manager=None,
        page_size=page_size,
        model_config=model_config,
        server_args=server_args,
        tp_size=args.tp_size,
        dp_size=attention_dp_size(args),
        pp_size=1,
        attn_cp_size=1,
        attention_chunk_size=None,
        hisparse_coordinator=None,
        init_new_workspace=False,
        is_hybrid_swa=False,
        is_draft_worker=False,
        sliding_window_size=None,
        use_mla_backend=True,
        _kernel_warmed_up=True,
        hybrid_gdn_config=None,
        mambaish_config=None,
        mamba2_config=None,
        kimi_linear_config=None,
        hybrid_lightning_config=None,
        plan_stream_for_flashinfer=None,
        linear_attn_model_spec=None,
    )
    runner.req_to_token_pool = ReqToTokenPool(
        size=args.batch_size,
        max_context_len=max_context_len,
        device="cuda",
        enable_memory_saver=False,
    )
    if is_dsa:
        if kv_cache_dtype_name == "fp8_e4m3":
            dim_nope = int(model["kv_lora_rank"])
            dim_rope = int(model.get("qk_rope_head_dim", 0))
            num_tiles = dim_nope // DSATokenToKVPool.quant_block_size
            pool_dtype = torch.float8_e4m3fn
            pool_kv_cache_dim = dim_nope + num_tiles * 4 + dim_rope * 2
        else:
            pool_dtype = dtype
            pool_kv_cache_dim = int(model["kv_lora_rank"]) + int(
                model.get("qk_rope_head_dim", 0)
            )
        runner.token_to_kv_pool = DSATokenToKVPool(
            size=pool_size,
            page_size=page_size,
            kv_lora_rank=int(model["kv_lora_rank"]),
            dtype=pool_dtype,
            qk_rope_head_dim=int(model.get("qk_rope_head_dim", 0)),
            layer_num=1,
            device="cuda",
            index_head_dim=int(model.get("index_head_dim", 128)),
            enable_memory_saver=False,
            kv_cache_dim=pool_kv_cache_dim,
            start_layer=layer_id,
        )
    else:
        runner.token_to_kv_pool = MLATokenToKVPool(
            size=pool_size,
            page_size=page_size,
            dtype=runner.kv_cache_dtype,
            kv_lora_rank=int(model["kv_lora_rank"]),
            qk_rope_head_dim=int(model.get("qk_rope_head_dim", 0)),
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            start_layer=layer_id,
        )
    runner.token_to_kv_pool_allocator = SimpleNamespace(page_size=page_size)
    prepopulate_req_to_token_pool(
        torch,
        runner,
        batch_size=args.batch_size,
        max_context_len=max_context_len,
        device=args.device,
    )

    backend = ATTENTION_BACKENDS[attention_backend](runner)
    backend.prefill_attention_backend_str = attention_backend
    backend.decode_attention_backend_str = attention_backend
    attn = DeepseekV2AttentionMLA(
        config=model_config.hf_config,
        hidden_size=int(model["hidden_size"]),
        num_heads=int(model["num_attention_heads"]),
        qk_nope_head_dim=int(model["qk_nope_head_dim"]),
        qk_rope_head_dim=int(model.get("qk_rope_head_dim", 0)),
        v_head_dim=int(model["v_head_dim"]),
        q_lora_rank=int(model["q_lora_rank"]),
        kv_lora_rank=int(model["kv_lora_rank"]),
        rope_theta=float(model.get("rope_theta", 10000.0)),
        rope_scaling=None,
        max_position_embeddings=int(
            model.get("max_position_embeddings", max_context_len)
        ),
        quant_config=None,
        reduce_results=attn_tp_size(args) > 1,
        layer_id=layer_id,
        prefix="benchmark_attn",
    ).to(device=torch.device("cuda", args.device), dtype=dtype)
    restore_sglang_rope_cache_dtype(torch, attn, args.device)
    initialize_synthetic_attention_weights(torch, attn, dtype)
    prepare_absorbed_mla_weights(attn)
    install_benchmark_attention_dispatch(
        attn,
        backend,
        prefill_backend=attention_backend,
        decode_backend=attention_backend,
    )
    return {
        "runner": runner,
        "backend": backend,
        "attention": attn,
        "attention_backend": attention_backend,
        "prefill_backend": prefill_backend,
        "decode_backend": decode_backend,
        "kv_cache_dtype": kv_cache_dtype_name,
        "page_size": page_size,
        "max_context_len": max_context_len,
        "layer_id": layer_id,
    }


def run_sglang_mla_module_e2e(
    kernel: str, args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    is_dsa = kernel == "glm-dsa-e2e"
    if is_dsa and dsa_layer_skips_topk_from_model(model, int(args.glm_dsa_layer_id)):
        raise RuntimeError(
            f"GLM DSA layer_id={args.glm_dsa_layer_id} is a shared-topk layer and would skip the indexer; "
            "choose a layer that computes topk, e.g. --glm-dsa-layer-id 0."
        )
    info = shape_summary(kernel, args, model)
    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    dtype = resolve_torch_dtype(torch, activation_dtype_name(args, kernel))
    device = args.device
    runtime = make_sglang_mla_runtime(torch, args, model, is_dsa=is_dsa, dtype=dtype)
    backend = runtime["backend"]
    attn = runtime["attention"]
    if is_dsa and (
        not getattr(attn, "use_dsa", False) or getattr(attn, "indexer", None) is None
    ):
        raise RuntimeError(
            "GLM DSA E2E did not instantiate SGLang DSA indexer; refusing to run non-sparse path."
        )
    max_context_len = int(runtime["max_context_len"])
    hidden = int(model["hidden_size"])
    chunk_schedule = require_multi_chunk_e2e_schedule(
        kernel, e2e_prefill_schedule(args), args
    )

    from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )
    from sglang.srt.utils.common import BumpAllocator

    get_attn_tp_context().init_context(attn.q_lora_rank, attn.use_dsa)
    prefill_hidden = randn_tensor(
        torch, (args.batch_size, input_len(args), hidden), dtype, device
    )
    decode_hidden = randn_tensor(
        torch, (args.output_len, args.batch_size, hidden), dtype, device
    )
    zero_allocator = BumpAllocator(buffer_size=1024, dtype=torch.float32, device=device)
    dsa_observed = {
        "sparse_dispatches": 0,
        "mha_one_shot_dispatches": 0,
        "topk_returns": 0,
    }

    def _shape_debug(value: Any) -> Any:
        if hasattr(value, "shape"):
            return list(value.shape)
        if isinstance(value, (tuple, list)):
            return [_shape_debug(item) for item in value]
        if isinstance(value, dict):
            return {key: _shape_debug(item) for key, item in value.items()}
        return type(value).__name__

    def _small_value_debug(value: Any) -> Any:
        if hasattr(value, "numel") and hasattr(value, "detach"):
            if int(value.numel()) <= 16:
                return value.detach().cpu().tolist()
            return f"{int(value.numel())} values"
        if isinstance(value, (tuple, list)):
            return [_small_value_debug(item) for item in value]
        return value

    def call_attention(
        hidden_states: Any, forward_batch: Any, prev_topk_indices: Any | None
    ) -> tuple[Any, Any | None]:
        zero_allocator._pointer = 0
        backend.init_forward_metadata(forward_batch)
        intermediate = None
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            get_attn_tp_context().set_attn_inputs(
                AttentionInputs(hidden_states, forward_batch, attn.prepare_qkv_latent)
            )
            intermediate = attn.forward_prepare(
                positions=forward_batch.positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                prev_topk_indices=prev_topk_indices,
            )
            if is_dsa:
                attn_forward_method = (
                    intermediate[1]
                    if isinstance(intermediate, tuple) and len(intermediate) > 1
                    else None
                )
                method_name = getattr(
                    attn_forward_method, "name", str(attn_forward_method)
                )
                backend_use_mha = bool(getattr(backend, "use_mha", False))
                if backend_use_mha:
                    dsa_observed["mha_one_shot_dispatches"] += 1
                else:
                    dsa_observed["sparse_dispatches"] += 1
                    if getattr(attn, "current_attention_backend", None) not in (
                        "dsa",
                        "nsa",
                    ):
                        raise RuntimeError(
                            "GLM DSA sparse step did not dispatch through attention_backend=dsa; "
                            f"current_attention_backend={getattr(attn, 'current_attention_backend', None)!r}."
                        )
                    if method_name != "MLA":
                        raise RuntimeError(
                            f"GLM DSA sparse step selected {method_name}, expected MLA absorbed DSA path."
                        )
                    inner_state = (
                        intermediate[3]
                        if isinstance(intermediate, tuple) and len(intermediate) > 3
                        else None
                    )
                    topk_indices = (
                        inner_state[7]
                        if isinstance(inner_state, (tuple, list))
                        and len(inner_state) > 7
                        else None
                    )
                    if topk_indices is None:
                        raise RuntimeError(
                            "GLM DSA sparse step did not produce topk_indices from the SGLang indexer."
                        )
            try:
                result = attn.forward_core(intermediate)
            except Exception as exc:
                import traceback

                try:
                    from sglang.srt.runtime_context import get_parallel

                    parallel = get_parallel()
                except Exception:
                    parallel = None
                inner_state = (
                    intermediate[3]
                    if isinstance(intermediate, tuple) and len(intermediate) > 3
                    else None
                )
                try:
                    from sglang.srt.environ import envs as _debug_envs

                    dsa_dense_threshold = (
                        _debug_envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()
                    )
                except Exception:
                    dsa_dense_threshold = None
                attn_forward_method = (
                    intermediate[1]
                    if isinstance(intermediate, tuple) and len(intermediate) > 1
                    else None
                )
                debug = {
                    "exception_type": type(exc).__name__,
                    "exception_repr": repr(exc),
                    "exception_traceback": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                    "hidden_states": _shape_debug(hidden_states),
                    "intermediate_forward_method": getattr(
                        attn_forward_method, "name", str(attn_forward_method)
                    ),
                    "backend_prefill_attention_backend_str": getattr(
                        backend, "prefill_attention_backend_str", None
                    ),
                    "backend_decode_attention_backend_str": getattr(
                        backend, "decode_attention_backend_str", None
                    ),
                    "server_attention_backend": getattr(
                        runtime["runner"].server_args, "attention_backend", None
                    ),
                    "server_dsa_prefill_backend": getattr(
                        runtime["runner"].server_args, "dsa_prefill_backend", None
                    ),
                    "server_dsa_decode_backend": getattr(
                        runtime["runner"].server_args, "dsa_decode_backend", None
                    ),
                    "positions": _shape_debug(
                        getattr(forward_batch, "positions", None)
                    ),
                    "input_ids": _shape_debug(
                        getattr(forward_batch, "input_ids", None)
                    ),
                    "out_cache_loc": _shape_debug(
                        getattr(forward_batch, "out_cache_loc", None)
                    ),
                    "seq_lens": _shape_debug(getattr(forward_batch, "seq_lens", None)),
                    "seq_lens_values": _small_value_debug(
                        getattr(forward_batch, "seq_lens", None)
                    ),
                    "seq_lens_cpu_values": _small_value_debug(
                        getattr(forward_batch, "seq_lens_cpu", None)
                    ),
                    "seq_lens_sum": getattr(forward_batch, "seq_lens_sum", None),
                    "extend_seq_lens": _shape_debug(
                        getattr(forward_batch, "extend_seq_lens", None)
                    ),
                    "extend_seq_lens_values": _small_value_debug(
                        getattr(forward_batch, "extend_seq_lens", None)
                    ),
                    "extend_seq_lens_cpu_values": _small_value_debug(
                        getattr(forward_batch, "extend_seq_lens_cpu", None)
                    ),
                    "extend_prefix_lens": _shape_debug(
                        getattr(forward_batch, "extend_prefix_lens", None)
                    ),
                    "extend_prefix_lens_values": _small_value_debug(
                        getattr(forward_batch, "extend_prefix_lens", None)
                    ),
                    "extend_prefix_lens_cpu_values": _small_value_debug(
                        getattr(forward_batch, "extend_prefix_lens_cpu", None)
                    ),
                    "backend_use_mha": getattr(backend, "use_mha", None),
                    "dsa_dense_threshold": dsa_dense_threshold,
                    "inner_state": _shape_debug(inner_state),
                    "current_attention_backend": getattr(
                        attn, "current_attention_backend", None
                    ),
                    "parallel_tp_size": getattr(parallel, "tp_size", None),
                    "parallel_tp_rank": getattr(parallel, "tp_rank", None),
                    "parallel_attn_tp_size": getattr(parallel, "attn_tp_size", None),
                    "parallel_attn_tp_rank": getattr(parallel, "attn_tp_rank", None),
                    "num_local_heads": getattr(attn, "num_local_heads", None),
                    "q_b_proj_tp_size": getattr(
                        getattr(attn, "q_b_proj", None), "tp_size", None
                    ),
                    "q_b_proj_tp_rank": getattr(
                        getattr(attn, "q_b_proj", None), "tp_rank", None
                    ),
                    "q_b_proj_weight": _shape_debug(
                        getattr(getattr(attn, "q_b_proj", None), "weight", None)
                    ),
                    "q_b_proj_output_partition": getattr(
                        getattr(attn, "q_b_proj", None),
                        "output_size_per_partition",
                        None,
                    ),
                    "fused_qkv_a_weight": _shape_debug(
                        getattr(
                            getattr(attn, "fused_qkv_a_proj_with_mqa", None),
                            "weight",
                            None,
                        )
                    ),
                    "attn_mqa_heads": getattr(
                        getattr(attn, "attn_mqa", None), "tp_q_head_num", None
                    ),
                    "attn_mqa_kv_heads": getattr(
                        getattr(attn, "attn_mqa", None), "tp_k_head_num", None
                    ),
                    "attn_mqa_head_dim": getattr(
                        getattr(attn, "attn_mqa", None), "head_dim", None
                    ),
                    "attn_mqa_v_head_dim": getattr(
                        getattr(attn, "attn_mqa", None), "v_head_dim", None
                    ),
                }
                raise RuntimeError(
                    f"SGLang MLA/DSA forward_core failed with debug={debug!r}"
                ) from exc
        if isinstance(result, tuple):
            if is_dsa and result[1] is not None:
                dsa_observed["topk_returns"] += 1
            return result[0], result[1]
        return result, None

    phase_state: dict[str, Any] = {
        "out": None,
        "prev_topk": None,
        "saw_returned_topk": False,
    }

    def reset_state() -> None:
        phase_state["out"] = None
        phase_state["prev_topk"] = None
        phase_state["saw_returned_topk"] = False

    def run_prefill() -> Any:
        out = None
        prev_topk = phase_state.get("prev_topk")
        saw_returned_topk = bool(phase_state.get("saw_returned_topk"))
        token_offset = 0
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            for q_len in chunk_schedule:
                token_count = args.batch_size * q_len
                hidden_chunk = (
                    prefill_hidden[:, token_offset : token_offset + q_len, :]
                    .reshape(token_count, hidden)
                    .contiguous()
                )
                forward_batch = make_module_forward_batch(
                    torch,
                    args,
                    runtime["runner"],
                    ForwardMode.EXTEND,
                    prefix_len=token_offset,
                    query_len=q_len,
                    max_context_len=max_context_len,
                    mha_one_shot=(not is_dsa and token_offset == 0),
                )
                out, prev_topk = call_attention(hidden_chunk, forward_batch, prev_topk)
                saw_returned_topk = saw_returned_topk or prev_topk is not None
                out = apply_attention_boundary_communication(
                    torch, args, out, pure_tp=False, dp_attention=True
                )
                token_offset += q_len
        phase_state["out"] = out
        phase_state["prev_topk"] = prev_topk
        phase_state["saw_returned_topk"] = saw_returned_topk
        if (
            is_dsa
            and dsa_layer_skips_topk_from_model(model, int(args.glm_dsa_layer_id) + 1)
            and not saw_returned_topk
        ):
            raise RuntimeError(
                "GLM DSA indexer did not return topk indices for the next shared-topk layer; refusing ambiguous sparse-path measurement."
            )
        return out

    def run_decode() -> Any:
        out = phase_state.get("out")
        prev_topk = phase_state.get("prev_topk")
        saw_returned_topk = bool(phase_state.get("saw_returned_topk"))
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            for step in range(args.output_len):
                pos = input_len(args) + step
                hidden_step = (
                    decode_hidden[step].reshape(args.batch_size, hidden).contiguous()
                )
                forward_batch = make_module_forward_batch(
                    torch,
                    args,
                    runtime["runner"],
                    ForwardMode.DECODE,
                    prefix_len=pos,
                    query_len=1,
                    max_context_len=max_context_len,
                )
                out, prev_topk = call_attention(hidden_step, forward_batch, prev_topk)
                saw_returned_topk = saw_returned_topk or prev_topk is not None
                out = apply_attention_boundary_communication(
                    torch, args, out, pure_tp=False, dp_attention=True
                )
        phase_state["out"] = out
        phase_state["prev_topk"] = prev_topk
        phase_state["saw_returned_topk"] = saw_returned_topk
        if (
            is_dsa
            and dsa_layer_skips_topk_from_model(model, int(args.glm_dsa_layer_id) + 1)
            and not saw_returned_topk
        ):
            raise RuntimeError(
                "GLM DSA indexer did not return topk indices for the next shared-topk layer; refusing ambiguous sparse-path measurement."
            )
        return out

    timing = phase_timing(
        torch,
        args,
        device,
        reset_state=reset_state,
        run_prefill=run_prefill,
        run_decode=run_decode,
        prefill_work_items=args.batch_size * input_len(args),
        decode_work_items=args.batch_size * args.output_len,
        unit="tokens",
    )
    return {
        **info,
        "production_module": "sglang.srt.models.deepseek_v2.DeepseekV2AttentionMLA",
        "production_forward_includes_input_projections": True,
        "production_forward_includes_rope": True,
        "production_forward_includes_kv_cache_writes": True,
        "production_forward_includes_output_projection": True,
        "production_forward_uses_indexer": bool(is_dsa),
        "dsa_sparse_indices_required": bool(is_dsa),
        "dsa_sparse_topk": int(model.get("index_topk", 0)) if is_dsa else None,
        "dsa_observed_sparse_dispatches": dsa_observed["sparse_dispatches"]
        if is_dsa
        else None,
        "dsa_observed_mha_one_shot_dispatches": dsa_observed["mha_one_shot_dispatches"]
        if is_dsa
        else None,
        "dsa_observed_topk_returns": dsa_observed["topk_returns"] if is_dsa else None,
        "dsa_topk_return_required": bool(
            is_dsa and dsa_layer_skips_topk_from_model(model, runtime["layer_id"] + 1)
        ),
        "mla_attention_backend_effective": runtime["attention_backend"]
        if not is_dsa
        else None,
        "mla_kv_cache_dtype_effective": runtime["kv_cache_dtype"]
        if not is_dsa
        else None,
        "dsa_prefill_backend_effective": runtime["prefill_backend"],
        "dsa_decode_backend_effective": runtime["decode_backend"],
        "dsa_kv_cache_dtype_effective": runtime["kv_cache_dtype"] if is_dsa else None,
        "dsa_topk_backend": args.dsa_topk_backend if is_dsa else None,
        "dsa_layer_id": runtime["layer_id"] if is_dsa else None,
        "dsa_layer_skips_topk": dsa_layer_skips_topk_from_model(
            model, runtime["layer_id"]
        )
        if is_dsa
        else None,
        "dsa_next_layer_skips_topk": dsa_layer_skips_topk_from_model(
            model, runtime["layer_id"] + 1
        )
        if is_dsa
        else None,
        "prefill_schedule": chunk_schedule,
        "tp_communication_mode": tp_communication_mode(args),
        "timing": timing,
    }


def make_gdn_e2e_inputs(
    torch: Any, args: argparse.Namespace, model: dict[str, Any]
) -> dict[str, Any]:
    dtype = resolve_torch_dtype(torch, activation_dtype_name(args, "linear-gdn-e2e"))
    state_dtype = resolve_torch_dtype(torch, args.state_dtype)
    h, hv, k, v, pool = linear_shapes(args, model)
    decode_b = args.decode_batch_size or args.batch_size
    pool = max(pool, decode_b)
    seq = input_len(args)
    hidden = model_hidden_size(model)
    mixed_dim = 2 * h * k + hv * v
    qkvz_dim = mixed_dim + hv * v
    ba_dim = 2 * hv
    conv_kernel = int(model.get("linear_conv_kernel_dim", 4))
    activation = str(model.get("hidden_act", "silu"))
    prefill_hidden = randn_tensor(
        torch, (args.batch_size, seq, hidden), dtype, args.device
    )
    decode_hidden = randn_tensor(
        torch, (args.output_len, decode_b, hidden), dtype, args.device
    )
    in_proj_qkvz_weight = randn_tensor(torch, (hidden, qkvz_dim), dtype, args.device)
    in_proj_ba_weight = randn_tensor(torch, (hidden, ba_dim), dtype, args.device)
    a_log = torch.randn(hv, dtype=torch.float32, device=args.device)
    dt_bias = torch.randn(hv, dtype=torch.float32, device=args.device)
    state = torch.zeros(pool, hv, v, k, dtype=state_dtype, device=args.device)
    conv_state = torch.zeros(
        pool, mixed_dim, max(conv_kernel - 1, 0), dtype=dtype, device=args.device
    )
    conv_weight = randn_tensor(torch, (mixed_dim, conv_kernel), dtype, args.device)
    prefill_cache_indices = torch.arange(
        args.batch_size, dtype=torch.int32, device=args.device
    )
    decode_cache_indices = torch.arange(decode_b, dtype=torch.int32, device=args.device)
    decode_query_start_loc = torch.arange(
        decode_b + 1, dtype=torch.int32, device=args.device
    )
    prefill_has_initial = {
        False: torch.zeros(args.batch_size, dtype=torch.bool, device=args.device),
        True: torch.ones(args.batch_size, dtype=torch.bool, device=args.device),
    }
    return {
        "prefill_hidden": prefill_hidden,
        "decode_hidden": decode_hidden,
        "in_proj_qkvz_weight": in_proj_qkvz_weight,
        "in_proj_ba_weight": in_proj_ba_weight,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "state": state,
        "conv_state": conv_state,
        "conv_weight": conv_weight,
        "prefill_cache_indices": prefill_cache_indices,
        "decode_cache_indices": decode_cache_indices,
        "decode_query_start_loc": decode_query_start_loc,
        "prefill_has_initial": prefill_has_initial,
        "activation": activation,
        "h": h,
        "hv": hv,
        "head_k": k,
        "head_v": v,
        "hidden": hidden,
        "mixed_dim": mixed_dim,
        "qkvz_dim": qkvz_dim,
        "ba_dim": ba_dim,
        "conv_kernel": conv_kernel,
        "decode_batch": decode_b,
    }


def gdn_kernel_class_from_backend(backend: str) -> Any:
    if backend == "triton":
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        return TritonGDNKernel()
    if backend == "flashinfer":
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            FlashInferGDNKernel,
        )

        return FlashInferGDNKernel()
    if backend == "cutedsl":
        from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
            CuteDSLGDNKernel,
        )

        return CuteDSLGDNKernel()
    raise ValueError(backend)


def gdn_kernel_class(kernel: str) -> Any:
    if "triton" in kernel:
        return gdn_kernel_class_from_backend("triton")
    if "flashinfer" in kernel:
        return gdn_kernel_class_from_backend("flashinfer")
    if "cutedsl" in kernel:
        return gdn_kernel_class_from_backend("cutedsl")
    raise ValueError(kernel)


@dataclass
class LightweightGDNDispatcher:
    decode_kernel: Any
    extend_kernel: Any
    supports_packed_decode: bool

    def packed_decode(self, *args: Any, **kwargs: Any) -> Any:
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.decode_kernel.decode(*args, **kwargs)

    def extend(self, *args: Any, **kwargs: Any) -> Any:
        return self.extend_kernel.extend(*args, **kwargs)


def gdn_dispatcher(prefill_backend: str, decode_backend: str) -> Any:
    # Production Qwen3.5 serving resolves server_args into prefill/decode
    # backend names, then GDNKernelDispatcher maps them to these same low-level
    # kernel classes. Importing the production dispatcher also imports the full
    # hybrid Mamba backend and optional DeepGEMM libraries. The benchmark only
    # needs the GDN kernel methods below, so keep dispatch local and dependency
    # narrow while preserving SGLang CuTeDSL prefill fallback on H100.
    triton_kernel = gdn_kernel_class_from_backend("triton")
    kernels = {"triton": triton_kernel}

    def get_kernel(name: str) -> Any:
        if name not in {"triton", "flashinfer", "cutedsl"}:
            raise ValueError(f"Unsupported GDN backend: {name}")
        if name not in kernels:
            kernels[name] = gdn_kernel_class_from_backend(name)
        return kernels[name]

    decode_kernel = get_kernel(decode_backend)
    extend_kernel = get_kernel(prefill_backend)
    if prefill_backend == "cutedsl" and not getattr(
        extend_kernel, "supports_prefill", True
    ):
        extend_kernel = triton_kernel

    return LightweightGDNDispatcher(
        decode_kernel=decode_kernel,
        extend_kernel=extend_kernel,
        supports_packed_decode=bool(
            getattr(decode_kernel, "supports_packed_decode", False)
        ),
    )


def run_linear_gdn_e2e(
    args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    # E2E here means one synthetic attention-layer path: chunked prefill/extend
    # over input_len, then output_len sequential decode calls against the same
    # conv + recurrent state pools. It includes synthetic input projections and
    # keeps production causal-conv preprocessing, GDN gating, recurrent update,
    # output norm/projection, and boundary communication. It still skips MLP.
    kernel = "linear-gdn-e2e"
    info = shape_summary(kernel, args, model)
    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    prefill_backend, decode_backend = linear_backends(args)
    dispatcher = gdn_dispatcher(prefill_backend, decode_backend)
    inputs = make_gdn_e2e_inputs(torch, args, model)
    dtype = resolve_torch_dtype(torch, activation_dtype_name(args, kernel))
    output_project = make_linear_output_projector(torch, args, model, dtype)
    chunk_schedule = require_multi_chunk_e2e_schedule(
        kernel, e2e_prefill_schedule(args), args
    )

    from sglang.jit_kernel.triton.gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

    gdn_gating_impl = (
        "sglang.srt.layers.attention.fla.fused_gdn_gating.fused_gdn_gating"
    )
    conv_prefill_impl = (
        "sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_fn"
    )
    conv_decode_impl = (
        "sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_update"
    )

    query_start_by_len: dict[int, Any] = {}
    seq_lens_cpu_by_len: dict[int, list[int]] = {}

    def query_start_loc(q_len: int) -> Any:
        value = query_start_by_len.get(q_len)
        if value is None:
            value = torch.arange(
                0,
                (args.batch_size + 1) * q_len,
                q_len,
                dtype=torch.int32,
                device=args.device,
            )
            query_start_by_len[q_len] = value
        return value

    def seq_lens_cpu(q_len: int) -> list[int]:
        value = seq_lens_cpu_by_len.get(q_len)
        if value is None:
            value = [q_len] * args.batch_size
            seq_lens_cpu_by_len[q_len] = value
        return value

    def project_gdn_hidden(hidden_flat: Any) -> tuple[Any, Any, Any, Any]:
        projected_qkvz = hidden_flat.matmul(inputs["in_proj_qkvz_weight"])
        projected_ba = hidden_flat.matmul(inputs["in_proj_ba_weight"])
        if int(inputs["hv"]) // int(inputs["h"]) in (1, 2, 4):
            return fused_qkvzba_split_reshape_cat_contiguous(
                projected_qkvz.contiguous(),
                projected_ba.contiguous(),
                int(inputs["h"]),
                int(inputs["hv"]),
                int(inputs["head_k"]),
                int(inputs["head_v"]),
            )
        q_dim = int(inputs["h"]) * int(inputs["head_k"])
        v_dim = int(inputs["hv"]) * int(inputs["head_v"])
        query, key, value, z = projected_qkvz.split(
            [q_dim, q_dim, v_dim, v_dim], dim=-1
        )
        b, a = projected_ba.split([int(inputs["hv"]), int(inputs["hv"])], dim=-1)
        mixed_qkv = torch.cat((query, key, value), dim=-1).contiguous()
        z = z.reshape(
            hidden_flat.shape[0], int(inputs["hv"]), int(inputs["head_v"])
        ).contiguous()
        return mixed_qkv, z, b.contiguous(), a.contiguous()

    def split_prefill_mixed(mixed_qkv: Any, token_count: int) -> tuple[Any, Any, Any]:
        if int(inputs["mixed_dim"]) <= 8192:
            return fused_qkv_split_gdn_prefill(
                mixed_qkv,
                int(inputs["h"]),
                int(inputs["h"]),
                int(inputs["hv"]),
                int(inputs["head_k"]),
                int(inputs["head_k"]),
                int(inputs["head_v"]),
            )
        query, key, value = torch.split(
            mixed_qkv,
            [
                int(inputs["h"]) * int(inputs["head_k"]),
                int(inputs["h"]) * int(inputs["head_k"]),
                int(inputs["hv"]) * int(inputs["head_v"]),
            ],
            dim=-1,
        )
        query = query.view(1, token_count, int(inputs["h"]), int(inputs["head_k"]))
        key = key.view(1, token_count, int(inputs["h"]), int(inputs["head_k"]))
        value = value.view(1, token_count, int(inputs["hv"]), int(inputs["head_v"]))
        return query, key, value

    def split_decode_mixed(mixed_qkv: Any, batch: int) -> tuple[Any, Any, Any]:
        query, key, value = torch.split(
            mixed_qkv,
            [
                int(inputs["h"]) * int(inputs["head_k"]),
                int(inputs["h"]) * int(inputs["head_k"]),
                int(inputs["hv"]) * int(inputs["head_v"]),
            ],
            dim=-1,
        )
        query = query.view(1, batch, int(inputs["h"]), int(inputs["head_k"]))
        key = key.view(1, batch, int(inputs["h"]), int(inputs["head_k"]))
        value = value.view(1, batch, int(inputs["hv"]), int(inputs["head_v"]))
        return query, key, value

    def reset_state() -> None:
        inputs["state"].zero_()
        inputs["conv_state"].zero_()

    def run_prefill() -> Any:
        out = None
        token_offset = 0
        for q_len in chunk_schedule:
            token_count = args.batch_size * q_len
            chunk_slice = slice(token_offset, token_offset + q_len)
            hidden_chunk = (
                inputs["prefill_hidden"][:, chunk_slice, :]
                .reshape(token_count, int(inputs["hidden"]))
                .contiguous()
            )
            mixed_chunk, z_chunk, b_chunk, a_chunk = project_gdn_hidden(hidden_chunk)
            conv_chunk = (
                causal_conv1d_fn(
                    mixed_chunk.transpose(0, 1),
                    inputs["conv_weight"],
                    None,
                    activation=inputs["activation"],
                    conv_states=inputs["conv_state"],
                    has_initial_state=inputs["prefill_has_initial"][token_offset > 0],
                    cache_indices=inputs["prefill_cache_indices"],
                    query_start_loc=query_start_loc(q_len),
                    seq_lens_cpu=seq_lens_cpu(q_len),
                )
                .transpose(0, 1)[:token_count]
                .contiguous()
            )
            query, key, value = split_prefill_mixed(conv_chunk, token_count)
            g, beta = fused_gdn_gating(
                inputs["A_log"],
                a_chunk,
                b_chunk,
                inputs["dt_bias"],
            )
            out = dispatcher.extend(
                query,
                key,
                value,
                g,
                beta,
                ssm_states=inputs["state"],
                cache_indices=inputs["prefill_cache_indices"],
                query_start_loc=query_start_loc(q_len),
            )
            out = output_project(out, z_chunk)
            out = apply_attention_boundary_communication(torch, args, out)
            token_offset += q_len
        return out

    def run_decode() -> Any:
        out = None
        for step in range(args.output_len):
            hidden_step = (
                inputs["decode_hidden"][step]
                .reshape(int(inputs["decode_batch"]), int(inputs["hidden"]))
                .contiguous()
            )
            mixed_raw, z_step, b_step, a_step = project_gdn_hidden(hidden_step)
            mixed_step = causal_conv1d_update(
                mixed_raw,
                inputs["conv_state"],
                inputs["conv_weight"],
                bias=None,
                activation=inputs["activation"],
                conv_state_indices=inputs["decode_cache_indices"],
            ).contiguous()
            if dispatcher.supports_packed_decode:
                out = dispatcher.packed_decode(
                    mixed_qkv=mixed_step,
                    a=a_step,
                    b=b_step,
                    A_log=inputs["A_log"],
                    dt_bias=inputs["dt_bias"],
                    scale=inputs["head_k"] ** -0.5,
                    ssm_states=inputs["state"],
                    cache_indices=inputs["decode_cache_indices"],
                    num_v_heads=inputs["hv"],
                    head_v_dim=inputs["head_v"],
                )
            else:
                query, key, value = split_decode_mixed(
                    mixed_step, int(inputs["decode_batch"])
                )
                out = dispatcher.decode(
                    query,
                    key,
                    value,
                    a_step,
                    b_step,
                    A_log=inputs["A_log"],
                    dt_bias=inputs["dt_bias"],
                    ssm_states=inputs["state"],
                    cache_indices=inputs["decode_cache_indices"],
                    query_start_loc=inputs["decode_query_start_loc"],
                )
            out = output_project(out, z_step)
            out = apply_attention_boundary_communication(torch, args, out)
        return out

    prefill_tokens = args.batch_size * input_len(args)
    decode_tokens = (args.decode_batch_size or args.batch_size) * args.output_len
    timing = phase_timing(
        torch,
        args,
        args.device,
        reset_state=reset_state,
        run_prefill=run_prefill,
        run_decode=run_decode,
        prefill_work_items=prefill_tokens,
        decode_work_items=decode_tokens,
        unit="tokens",
    )
    return {
        **info,
        "production_forward_includes_input_projections": True,
        "production_forward_includes_output_projection": True,
        "production_forward_includes_state_updates": True,
        "linear_effective_prefill_kernel": dispatcher.extend_kernel.__class__.__name__,
        "linear_effective_decode_kernel": dispatcher.decode_kernel.__class__.__name__,
        "linear_decode_call": "packed_decode"
        if dispatcher.supports_packed_decode
        else "decode",
        "linear_supports_packed_decode": bool(dispatcher.supports_packed_decode),
        "linear_gdn_gating_impl": gdn_gating_impl,
        "linear_conv_prefill_impl": conv_prefill_impl,
        "linear_conv_decode_impl": conv_decode_impl,
        "linear_conv_kernel_dim": inputs["conv_kernel"],
        "linear_conv_dim": inputs["mixed_dim"],
        "linear_prefill_schedule": chunk_schedule,
        "gdn_prefill_uses_unique_chunk_tensors": True,
        "gdn_conv_state_enabled": True,
        "gdn_ssm_state_enabled": True,
        "tp_communication_mode": tp_communication_mode(args),
        "timing": timing,
    }


def run_linear_gdn(
    kernel: str, args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    if kernel == "linear-gdn-e2e":
        return run_linear_gdn_e2e(args, model, torch)
    info = shape_summary(kernel, args, model)
    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    kern = gdn_kernel_class(kernel)
    if "prefill" in kernel:
        inp = make_gdn_prefill_inputs(torch, args, model)

        def fn() -> Any:
            out = kern.extend(
                inp["q"],
                inp["k"],
                inp["v"],
                inp["g"],
                inp["beta"],
                ssm_states=inp["state"],
                cache_indices=inp["cache_indices"],
                query_start_loc=inp["query_start_loc"],
            )
            return apply_attention_boundary_communication(torch, args, out)

        work_tokens = args.batch_size * input_len(args)
    elif "packed" in kernel:
        inp = make_gdn_decode_inputs(torch, args, model)

        def fn() -> Any:
            out = kern.packed_decode(
                inp["mixed_qkv"],
                inp["a"],
                inp["b"],
                A_log=inp["A_log"],
                dt_bias=inp["dt_bias"],
                scale=inp["head_k"] ** -0.5,
                ssm_states=inp["state"],
                cache_indices=inp["cache_indices"],
                num_v_heads=inp["hv"],
                head_v_dim=inp["head_v"],
            )
            return apply_attention_boundary_communication(torch, args, out)

        work_tokens = inp["batch"]
    else:
        inp = make_gdn_decode_inputs(torch, args, model)

        def fn() -> Any:
            out = kern.decode(
                inp["q"],
                inp["k"],
                inp["v"],
                inp["a"],
                inp["b"],
                A_log=inp["A_log"],
                dt_bias=inp["dt_bias"],
                ssm_states=inp["state"],
                cache_indices=inp["cache_indices"],
                query_start_loc=inp["query_start_loc"],
            )
            return apply_attention_boundary_communication(torch, args, out)

        work_tokens = inp["batch"]
    timing = add_work_metrics(
        time_cuda(torch, fn, args.warmup_iters, args.iters, args.device),
        work_tokens,
        "tokens",
    )
    return {
        **info,
        "tp_communication_mode": tp_communication_mode(args),
        "timing": timing,
    }


def run_linear_fused_qkv_split(
    args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkv_split_gdn_prefill

    kernel = "linear-fused-qkv-split"
    info = shape_summary(kernel, args, model)
    guard_estimate(args, kernel, int(info["estimated_live_bytes"]))
    dtype = resolve_torch_dtype(torch, activation_dtype_name(args, kernel))
    h, hv, k, v, _ = linear_shapes(args, model)
    total = args.batch_size * input_len(args)
    mixed_dim = 2 * h * k + hv * v
    mixed_qkv = randn_tensor(torch, (total, mixed_dim), dtype, args.device)

    def fn() -> Any:
        return fused_qkv_split_gdn_prefill(
            mixed_qkv,
            num_q_heads=h,
            num_k_heads=h,
            num_v_heads=hv,
            head_q=k,
            head_k=k,
            head_v=v,
        )

    timing = add_work_metrics(
        time_cuda(torch, fn, args.warmup_iters, args.iters, args.device),
        total,
        "tokens",
    )
    return {**info, "timing": timing}


def model_parallel_notes(
    args: argparse.Namespace, model: dict[str, Any], kernel: str
) -> list[str]:
    notes: list[str] = []
    atp = attn_tp_size(args)
    if kernel.startswith("linear"):
        for key in ("linear_num_key_heads", "linear_num_value_heads"):
            if key in model and int(model[key]) % atp != 0:
                notes.append(
                    f"{key}={model[key]} is not divisible by attn_tp_size={atp}; Qwen GDN kernels require clean sharding."
                )
    else:
        if int(model.get("num_attention_heads", 1)) % atp != 0:
            notes.append(
                f"num_attention_heads={model.get('num_attention_heads')} is not divisible by attn_tp_size={atp}; sharded Q-head kernels will fail or need a smaller attention TP."
            )
        kv = int(model.get("num_key_value_heads", 1))
        if kernel.startswith("sparse") and kv != 1 and kv < atp:
            notes.append(
                f"num_key_value_heads={kv} < attn_tp_size={atp}; generic sparse-flash probes replicate KV heads, while production support is backend-specific."
            )
        if "o_groups" in model:
            o_groups = int(model["o_groups"])
            if o_groups % atp != 0:
                notes.append(
                    f"o_groups={o_groups} is not divisible by attn_tp_size={atp}; DeepSeek V4 output/grouped paths may need padding or a different TP."
                )
        if kernel.startswith("flashmla"):
            heads = int(model.get("num_attention_heads", 1))
            if int(model.get("head_dim", 512)) != 512:
                notes.append(
                    f"DeepSeek V4 FlashMLA path expects head_dim=512, got {model.get('head_dim')}."
                )
            if int(model.get("num_key_value_heads", 1)) != 1:
                notes.append(
                    f"DeepSeek V4 FlashMLA path expects num_key_value_heads=1, got {model.get('num_key_value_heads')}."
                )
            if args.swa_page_size != 256:
                notes.append(
                    f"DeepseekV4AttnBackend asserts page_size=256, got --swa-page-size={args.swa_page_size}."
                )
            if (
                args.chunked_prefill_size
                and args.chunked_prefill_size > 0
                and args.chunked_prefill_size % args.swa_page_size != 0
            ):
                notes.append(
                    f"--chunked-prefill-size={args.chunked_prefill_size} must be divisible by --swa-page-size={args.swa_page_size} "
                    "for DSV4 chunked prefill."
                )
            if heads % atp == 0:
                local_hq = dsv4_local_q_heads(model, atp)
                flashmla_hq = dsv4_flashmla_q_heads(model, atp)
                padding_warning = dsv4_dp_off_padding_warning(args, model, atp)
                if padding_warning:
                    notes.append(padding_warning)
                elif dsv4_q_padded_for_flashmla(model, atp):
                    notes.append(
                        f"DeepSeek V4 production pads Q from local_q_heads={local_hq} to flashmla_q_heads={flashmla_hq} before FlashMLA, "
                        f"then slices output back to {local_hq} heads on this attention-TP rank."
                    )
                if (
                    kernel == "flashmla-sparse-prefill"
                    and flashmla_hq not in DSV4_FLASHMLA_SPARSE_PREFILL_KNOWN_Q_HEADS
                ):
                    notes.append(
                        f"flash_mla_sparse_fwd would use production flashmla_q_heads={flashmla_hq}; known supported h_q values are "
                        f"{list(DSV4_FLASHMLA_SPARSE_PREFILL_KNOWN_Q_HEADS)}. The benchmark will mark this standalone kernel unsupported before CUDA call."
                    )
                if kernel == "flashmla-kvcache-e2e":
                    requested_chunk = int(args.chunked_prefill_size or 0)
                    prefill_chunk = (
                        input_len(args)
                        if requested_chunk <= 0
                        else min(input_len(args), requested_chunk)
                    )
                    requests_monolithic_prefill = (
                        requested_chunk <= 0
                        or prefill_chunk > DSV4_SPARSE_PREFILL_QUERY_THRESHOLD
                    )
                    if (
                        requests_monolithic_prefill
                        and flashmla_hq
                        not in DSV4_FLASHMLA_SPARSE_PREFILL_KNOWN_Q_HEADS
                    ):
                        notes.append(
                            f"DSV4 E2E requested monolithic sparse prefill with flashmla_q_heads={flashmla_hq}; known supported prefill h_q values are "
                            f"{list(DSV4_FLASHMLA_SPARSE_PREFILL_KNOWN_Q_HEADS)}. Non-strict mode falls back to chunked FlashMLA kvcache prefill."
                        )
                    if flashmla_hq not in DSV4_FLASHMLA_SPARSE_DECODE_KNOWN_Q_HEADS:
                        supported_atp = dsv4_supported_flashmla_attn_tp_sizes(
                            model, DSV4_FLASHMLA_SPARSE_DECODE_KNOWN_Q_HEADS
                        )
                        notes.append(
                            f"DSV4 E2E sparse decode would use flashmla_q_heads={flashmla_hq}; known supported decode h_q values are "
                            f"{list(DSV4_FLASHMLA_SPARSE_DECODE_KNOWN_Q_HEADS)}. No equivalent DSV4 sparse decode fallback exists for this geometry; "
                            f"supported --attn-tp-size values for this model shape are {supported_atp}; equivalent attention_dp_size values for current tp_size are {dsv4_supported_attention_dp_sizes(args, supported_atp)}."
                        )
    if args.enable_dp_attention:
        notes.append(
            f"enable_dp_attention=True: benchmark sets attention_dp_size=tp_size={args.tp_size}, "
            f"effective attn_tp_size={atp}, matching SGLang's attention layout when the serving DP size equals tp_size."
        )
    if kernel in ("deepseek-v3-mla-e2e", "glm-dsa-e2e"):
        notes.append(
            "MLA/DSA E2E instantiates SGLang DeepseekV2AttentionMLA with real RadixAttention metadata, "
            "KV-pool writes, output projection, and measured layer-boundary communication."
        )
        if atp > 1:
            notes.append(
                "The module uses reduce_results=True so SGLang RowParallelLinear performs the pure-TP output all-reduce."
            )
        if kernel == "glm-dsa-e2e":
            if dsa_layer_skips_topk_from_model(model, int(args.glm_dsa_layer_id)):
                notes.append(
                    f"GLM DSA layer_id={args.glm_dsa_layer_id} would skip the indexer/topk and is rejected by the runtime benchmark."
                )
            else:
                notes.append(
                    f"GLM DSA layer_id={args.glm_dsa_layer_id} computes indexer/topk in this benchmark."
                )
    if args.enable_dp_attention and args.attn_tp_size is not None:
        notes.append(
            "Both --enable-dp-attention and --attn-tp-size were set. The explicit --attn-tp-size wins and no longer represents the default production DP-attention layout."
        )
    notes.append(
        f"Measured distributed boundary communication: {tp_communication_mode(args)}. "
        "Pure TP uses NCCL all_reduce after attention output; DP-attention uses NCCL all_gather_into_tensor plus local copy to model the production DP gather/scatter boundary."
    )
    return notes


def run_one(
    kernel: str, args: argparse.Namespace, model: dict[str, Any], torch: Any
) -> dict[str, Any]:
    if kernel in ("sparse-flash", "sparse-flash-varlen"):
        return run_sparse_flash(kernel, args, model, torch)
    if kernel == "sparse-index-convert":
        return run_sparse_index_convert(args, model, torch)
    if kernel == "flashmla-sparse-prefill":
        return run_flashmla_sparse_prefill(args, model, torch)
    if kernel == "flashmla-kvcache-e2e":
        return run_flashmla_kvcache_e2e(args, model, torch)
    if kernel.startswith("dsv4-c"):
        return run_dsv4_compress(kernel, args, model, torch)
    if kernel in ("dsv4-topk-v1", "dsv4-topk-v2"):
        return run_dsv4_topk(kernel, args, model, torch)
    if kernel in (
        "dsv4-q-norm-rope",
        "dsv4-k-norm-rope-flashmla",
        "dsv4-q-indexer-rope-hadamard-fp8",
    ):
        return run_dsv4_fused_elementwise(kernel, args, model, torch)
    if kernel == "linear-fused-qkv-split":
        return run_linear_fused_qkv_split(args, model, torch)
    if kernel.startswith("linear-"):
        return run_linear_gdn(kernel, args, model, torch)
    if kernel in ("deepseek-v3-mla-e2e", "glm-dsa-e2e"):
        return run_sglang_mla_module_e2e(kernel, args, model, torch)
    raise ValueError(f"Unknown kernel: {kernel}")


def args_as_json(args: argparse.Namespace) -> dict[str, Any]:
    return {k: v for k, v in vars(args).items()}


def non_ok_reason(record: dict[str, Any]) -> str:
    for key in (
        "error_message",
        "decode_unsupported_reason",
        "sparse_prefill_unsupported_reason",
        "prefill_fallback_reason",
        "error",
    ):
        value = record.get(key)
        if value:
            return str(value).splitlines()[0]
    return "see output JSON for details"


def emit_non_ok_summary(records: list[dict[str, Any]]) -> None:
    non_ok = [record for record in records if record.get("status") != "ok"]
    if not non_ok:
        return
    print("[sglang-attn-bench][summary] non-ok benchmark records", flush=True)
    for record in non_ok:
        print(
            f"  {record.get('kernel')}: status={record.get('status')} reason={non_ok_reason(record)}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if args.list_kernels:
        if should_print_stdout():
            list_kernels()
        return 0
    model = resolve_model(args)
    kernels = selected_kernels(args)
    startup_warnings, startup_error = normalize_dp_attention_args(args, model, kernels)
    if startup_error is not None:
        error_stdout(startup_error)
        return 2
    for message in startup_warnings:
        warn_stdout(message)
    torch = setup_torch(args, model)
    launch_args = args_as_json(args)
    base = {
        "schema_version": 2,
        "argv": sys.argv,
        "python_executable": sys.executable,
        "launch_args": launch_args,
        "model_preset": args.model_preset,
        "config_json": args.config_json,
        "output_json": args.output_json,
        "model": model,
        "seq_len": args.seq_len,
        "input_len": input_len(args),
        "output_len": args.output_len,
        "batch_size": args.batch_size,
        "decode_batch_size": args.decode_batch_size or args.batch_size,
        "batch_size_scope": "per_attention_dp_rank"
        if bool(args.enable_dp_attention)
        else "global_request_batch",
        "effective_global_batch_size": args.batch_size * attention_dp_size(args)
        if bool(args.enable_dp_attention)
        else args.batch_size,
        "effective_global_decode_batch_size": (
            args.decode_batch_size or args.batch_size
        )
        * attention_dp_size(args)
        if bool(args.enable_dp_attention)
        else (args.decode_batch_size or args.batch_size),
        "tp_size": args.tp_size,
        "enable_dp_attention_requested": args.enable_dp_attention_requested,
        "enable_dp_attention": bool(args.enable_dp_attention),
        "attention_dp_size": attention_dp_size(args),
        "attn_tp_size": attn_tp_size(args),
        "distributed_world_size": int(getattr(args, "distributed_world_size", 1)),
        "distributed_rank": int(getattr(args, "rank", 0)),
        "local_rank": int(getattr(args, "local_rank", args.device)),
        "distributed_backend": getattr(args, "distributed_backend", None),
        "sglang_model_parallel_initialized": bool(
            getattr(args, "sglang_model_parallel_initialized", False)
        ),
        "sglang_dp_attention_initialized": bool(
            getattr(args, "sglang_dp_attention_initialized", False)
        ),
        "sglang_dp_attention_moe_dense_tp_size": getattr(
            args, "sglang_dp_attention_moe_dense_tp_size", None
        ),
        "tp_communication_mode": tp_communication_mode(args),
        "startup_warnings": startup_warnings,
        "requested_dtype": args.dtype,
        "state_dtype": args.state_dtype,
        "kv_cache_owner_scope": "attention_dp_rank_local"
        if bool(args.enable_dp_attention)
        else "attention_tp_rank_local",
        "kv_cache_sharded_across_attention_tp": attn_tp_size(args) > 1,
    }
    records: list[dict[str, Any]] = []
    for kernel in kernels:
        record = {
            **base,
            "kernel": kernel,
            "kernel_comment": KERNEL_COMMENTS.get(kernel),
            "model_parallel_notes": model_parallel_notes(args, model, kernel),
            "status": "ok",
            "timestamp_unix": time.time(),
        }
        try:
            payload = run_one(kernel, args, model, torch)
            record.update(payload)
            timing = record.get("timing")
            if isinstance(timing, dict) and isinstance(timing.get("memory"), dict):
                memory = timing["memory"]
                record["max_memory_allocated_gib"] = memory.get("peak_allocated_gib")
                record["max_memory_reserved_gib"] = memory.get("peak_reserved_gib")
        except Exception as exc:
            if args.strict:
                raise
            try:
                record.update(
                    {
                        k: v
                        for k, v in shape_summary(kernel, args, model).items()
                        if k not in record
                    }
                )
            except Exception as shape_exc:
                record["shape_summary_error"] = repr(shape_exc)
            record["status"] = "error"
            record["error"] = repr(exc)
            record["error_message"] = str(exc)
            record["traceback"] = traceback.format_exc(limit=8)
        record = merge_distributed_rank_statuses(torch, args, record)
        if is_report_rank(args):
            records.append(record)
            emit(record)
            write_records(records, args.output_json)
    if is_report_rank(args):
        emit_non_ok_summary(records)
    cleanup_distributed(torch, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
