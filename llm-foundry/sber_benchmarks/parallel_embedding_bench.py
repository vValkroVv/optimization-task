import argparse
from typing import Type

import torch
import torch.distributed
from torch.distributed import ProcessGroup
from llmfoundry.models.gigar import GigarConfig

from composer.utils.dist import (
    initialize_dist,
    get_tp_group,
    get_tp_group_size,
    get_tp_group_rank,
)
from composer.utils.device import get_device

from llmfoundry.models.layers import EmbeddingParallelEmbedding, VocabParallelEmbedding

from benchmark import benchmark_module


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_model(
    args: argparse.Namespace,
    embedding_class: Type | None = None,
    gather_full_output: bool = True,
) -> EmbeddingParallelEmbedding | VocabParallelEmbedding:
    config = GigarConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        dtype=torch.bfloat16,
        fused_mlp=False,
        init_device="cpu",
        tp_size=get_tp_group_size(),
    )

    if embedding_class is None:
        embedding_class = (
            VocabParallelEmbedding if args.baseline else EmbeddingParallelEmbedding
        )

    vocab_shape = (config.vocab_size, config.hidden_size)

    kwargs = (
        dict()
        if embedding_class is VocabParallelEmbedding
        else dict(gather_full_output=gather_full_output)
    )
    return embedding_class(*vocab_shape, **kwargs, config=config).cuda()


def make_input(args: argparse.Namespace) -> torch.Tensor:
    return torch.randint(
        0,
        args.vocab_size,
        size=(args.batch_size, args.seq_len),
        dtype=torch.int32,
        device=torch.cuda.current_device(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--seq_len", type=int, default=8192, required=False)
    parser.add_argument("--vocab_size", type=int, default=128256, required=False)
    parser.add_argument("--hidden_size", type=int, default=8192, required=False)
    parser.add_argument("--warmup_iters", type=int, default=5, required=False)
    parser.add_argument("--iters", type=int, default=10, required=False)
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument("--profile_memory", action="store_true", default=True)

    args = parser.parse_args()

    tp_size = get_tp_group_size()
    assert args.seq_len % tp_size == 0
    assert args.hidden_size % tp_size == 0
    assert args.vocab_size % tp_size == 0

    return args


def test_embeddings_allclose():
    initialize_dist(
        get_device(None),
        tp_size=2,
        sp_size=1,
        timeout=999999,
    )

    torch.cuda.set_device(get_tp_group_rank())
    torch.cuda.synchronize()

    args = parse_args()

    seed_everything(0)
    vocab_parallel_emb = make_model(args, embedding_class=VocabParallelEmbedding)
    seed_everything(0)
    embedding_parallel_emb = make_model(
        args, embedding_class=EmbeddingParallelEmbedding
    )

    input_ = make_input(args)

    vocab_parallel_out = vocab_parallel_emb(input_)
    embedding_parallel_out = embedding_parallel_emb(input_)

    assert isinstance(vocab_parallel_emb, VocabParallelEmbedding)
    assert isinstance(embedding_parallel_emb, EmbeddingParallelEmbedding)
    assert torch.allclose(vocab_parallel_out, embedding_parallel_out)


def unshard_tp_tensor_like(
    reference: torch.Tensor,
    shard: torch.Tensor,
) -> torch.Tensor:
    full = torch.empty_like(reference)
    torch.distributed.all_gather_into_tensor(full, shard, get_tp_group())
    return full


def test_embeddings_allclose():
    initialize_dist(
        get_device(None),
        tp_size=2,
        sp_size=1,
        timeout=999999,
    )

    torch.cuda.set_device(get_tp_group_rank())
    torch.cuda.synchronize()

    args = parse_args()

    seed_everything(0)
    vocab_parallel_emb = make_model(args, embedding_class=VocabParallelEmbedding)
    seed_everything(0)
    embedding_parallel_emb = make_model(
        args,
        embedding_class=EmbeddingParallelEmbedding,
        gather_full_output=False,
    )

    input_ = make_input(args)

    vocab_parallel_out = vocab_parallel_emb(input_)
    embedding_parallel_out = embedding_parallel_emb(input_)

    full_emb_parallel_out = unshard_tp_tensor_like(
        vocab_parallel_out,
        embedding_parallel_out,
    )
    assert torch.allclose(vocab_parallel_out, full_emb_parallel_out)


def main():
    initialize_dist(
        get_device(None),
        tp_size=2,
        sp_size=1,
        timeout=999999,
    )

    torch.cuda.set_device(get_tp_group_rank())
    torch.cuda.synchronize()

    args = parse_args()

    module = make_model(args)
    input_ = make_input(args)

    torch.distributed.barrier()

    perf_result = benchmark_module(
        input_,
        module,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        trace=args.trace,
        profile_memory=args.profile_memory,
    )

    torch.distributed.barrier()

    perf_result.print_table()


if __name__ == "__main__":
    test_embeddings_allclose()
