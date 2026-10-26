import argparse
import time

import torch

from llmfoundry.models.layers import dsa_monitor


def _measure(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _stage1_ms(args, num_queries: int) -> tuple[float, float]:
    device = torch.device("cuda")
    attn_score = torch.rand(
        1,
        num_queries,
        args.context,
        device=device,
        dtype=torch.float32,
    )
    attn_l1norm = attn_score.sum(-1)
    index_score = torch.randn_like(attn_score)
    rows = torch.arange(num_queries, device=device)
    offset = args.context - num_queries

    attention_mass_ms = _measure(
        lambda: dsa_monitor.dense_attention_mass_stats(
            attn_score,
            attn_l1norm,
            index_score,
            tuple(args.attention_mass_topk_values),
            offset,
            rows,
        ),
        args.warmup,
        args.iterations,
    )
    indexer_topk_ms = _measure(
        lambda: dsa_monitor.dense_indexer_topk_stats(
            index_score,
            tuple(args.indexer_topk_values),
            offset,
            rows,
        ),
        args.warmup,
        args.iterations,
    )
    return attention_mass_ms, indexer_topk_ms


def _stage2_ms(args, num_queries: int) -> float:
    device = torch.device("cuda")
    topk_ids = torch.randint(
        0,
        args.context,
        (1, num_queries, args.topk),
        device=device,
        dtype=torch.int32,
    )
    topk_length = torch.full(
        (num_queries,),
        args.topk,
        device=device,
        dtype=torch.int32,
    )
    positions_q = torch.full(
        (num_queries,),
        args.context - 1,
        device=device,
        dtype=torch.int64,
    )
    rows = torch.arange(num_queries, device=device)

    return _measure(
        lambda: dsa_monitor.sparse_indexer_topk_stats(
            topk_ids,
            topk_length,
            positions_q,
            rows,
            tuple(args.indexer_topk_values),
        ),
        args.warmup,
        args.iterations,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument(
        "--attention-mass-topk-values",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048],
    )
    parser.add_argument("--num-queries", type=int, nargs="+", default=[128, 1024])
    parser.add_argument(
        "--indexer-topk-values",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
    )
    parser.add_argument("--layers", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    if any(
        topk <= 0 or topk > args.context
        for topk in args.attention_mass_topk_values
    ):
        parser.error("--attention-mass-topk-values must be positive and <= --context")
    if any(topk <= 0 or topk > args.topk for topk in args.indexer_topk_values):
        parser.error("--indexer-topk-values must be positive and <= --topk")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(2048)
    torch.cuda.manual_seed_all(2048)

    print(
        f"context={args.context} topk={args.topk} layers={args.layers} "
        f"attention_mass_topk_values={args.attention_mass_topk_values} "
        f"indexer_topk_values={args.indexer_topk_values} "
        f"log_interval={args.log_interval}"
    )
    for num_queries in args.num_queries:
        started = time.perf_counter()
        stage1_attention_ms, stage1_indexer_ms = _stage1_ms(args, num_queries)
        torch.cuda.empty_cache()
        stage2_indexer_ms = _stage2_ms(args, num_queries)
        torch.cuda.empty_cache()
        print(
            f"queries={num_queries} "
            f"stage1_attention_mass_ms={stage1_attention_ms:.3f} "
            f"stage1_indexer_topk_ms={stage1_indexer_ms:.3f} "
            f"stage2_indexer_topk_ms={stage2_indexer_ms:.3f} "
            f"stage1_attention_amortized_ms_per_step="
            f"{stage1_attention_ms * args.layers / args.log_interval:.3f} "
            f"stage1_indexer_amortized_ms_per_step="
            f"{stage1_indexer_ms * args.layers / args.log_interval:.3f} "
            f"stage2_indexer_amortized_ms_per_step="
            f"{stage2_indexer_ms * args.layers / args.log_interval:.3f} "
            f"wall_s={time.perf_counter() - started:.2f}"
        )


if __name__ == "__main__":
    main()
