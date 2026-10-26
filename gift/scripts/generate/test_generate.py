# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torchtitan.callbacks.builtin.system_monitor import SystemMetricsMonitor
from torchtitan.config import DebugConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.config.cli import load_trainer_config
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import (
    device_capacity_report,
    device_module,
    device_type,
    reset_peak_stats,
)

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# pyrefly: ignore[missing-import]
from generate._generation import generate  # noqa: E402


def apply_tp_minus_sp(model: nn.Module, tp_mesh: DeviceMesh):
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    for _, transformer_block in model.layers.items():
        layer_plan = {
            "attention.qkv_linear.wq": ColwiseParallel(),
            "attention.qkv_linear.wk": ColwiseParallel(),
            "attention.qkv_linear.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


@record
def test_generate(
    config_file: str,
    checkpoint_path: str,
    prompt: str,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    batch_size: int = 1,
    top_k: int | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    overrides: list[str] | None = None,
):
    init_logger()
    color = utils.Color

    # The experiment YAML the checkpoint was trained with, plus any dot-path
    # overrides (same prologue as torchtitan.train).
    config = load_trainer_config(
        [config_file, *(overrides or [])], logger=logger
    ).trainer_config

    if len(args.prompt) == 0:
        logger.warning(
            "The input prompt is empty, model will respond from a empty sequence."
        )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    reset_peak_stats()

    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    # Tokenizer setup
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tokenizer = HuggingFaceTokenizer.Config().build(
        tokenizer_path=config.hf_assets_path
    )

    model_config = config.model_spec.model
    model_config.owner().prepare_config(model_config, config=config)

    init_device = "meta" if world_size > 1 else device
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_config.build()

    parallel_dims = None
    # Init distributed env
    if world_size > 1:
        dist_utils.init_distributed(config.comm)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=world_size,
            pp=1,
            ep=1,
            world_size=world_size,
        )

        # apply_tp (with Sequence Parallel) on unevenly sharded
        # sequences would require https://github.com/pytorch/torchtitan/pull/686
        apply_tp_minus_sp(model, parallel_dims.get_mesh("tp"))
    else:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )

    debug_config = DebugConfig(seed=seed, deterministic=deterministic)
    dist_utils.set_determinism(
        parallel_dims=parallel_dims,
        device=device,
        debug_config=debug_config,
        distinct_seed_mesh_dims=["pp"],
    )

    # materalize model
    model.to_empty(device=device_type)
    with torch.no_grad():
        model.init_weights()
    model.eval()

    state_dict = model.state_dict()

    # Checkpoint Loading
    begin = time.monotonic()
    logger.info(f"Loading chkpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")

    memory_metrics, max_reserved_gib = device_capacity_report()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{max_reserved_gib:.2f}GiB"
        f"({memory_metrics['memory/max_reserved_pct']:.2f}%)"
    )

    # Tokenize prompt and repeat batch_size times
    input_ids = (
        torch.tensor(
            tokenizer.encode(prompt, add_bos=True, add_eos=False), dtype=torch.long
        )
        .view(1, -1)
        .repeat(batch_size, 1)
    ).to(device_type)

    reset_peak_stats()

    # Run generation
    t0 = time.monotonic()
    responses = generate(
        model,
        input_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
    )
    t1 = time.monotonic()
    elapsed_sec = t1 - t0

    # Post process
    B, T = responses.size()  # B: batch_size, T: total seq length
    input_n_tokens = input_ids.size(1)
    generated_n_tokens = T - input_n_tokens  # == max_new_tokens

    if local_rank == 0:
        logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")

        r, b = color.red, color.blue

        output_data = {
            "metadata": {},
            "responses": [],
        }

        for i, tokens in enumerate(responses):
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()

            input_text = tokenizer.decode(inp_tok)
            output_text = tokenizer.decode(out_tok)

            _data = {
                "response_idx": i,
                "input_text": input_text,
                "output_text": output_text,
            }
            output_data["responses"].append(_data)

            logger.info(f"{r}\n{input_text}{b}{output_text}\n{color.reset}")

        capacity_metrics, _ = device_capacity_report()
        output_data["metadata"] = {
            "generated_n_tokens": generated_n_tokens,
            "input_n_tokens": input_n_tokens,
            "generation_time_sec": elapsed_sec,
            "tokens_per_sec": (B * T) / elapsed_sec,
            "batch_size": B,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            # Same names and units as the training-side memory series: the peak
            # counters are decimal GB, the capacity ones percentages.
            **SystemMetricsMonitor.get_memory_report(),
            **capacity_metrics,
            "world_size": world_size,
            "torch_version": torch.__version__,
        }

        if args.out:
            print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Experiment YAML the checkpoint was trained with "
        "(e.g., configs/experiments/llama3_debug.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to load (required)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default is 1.0",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max number of tokens to generate. Default is 32",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of samples to run in batch"
    )
    parser.add_argument(
        "--top_k", type=int, help="Prune to select from top_k probabilities. Optional"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms wherever possible, may be slower",
    )

    parser.add_argument("--prompt", type=str, default="", help="Input prompt")

    parser.add_argument(
        "--out",
        action="store_true",
        default=False,
        help="If specified, prints the report to stdout. Defaults to no output.",
    )

    # Anything the parser does not know is a dot-path override on the YAML.
    args, overrides = parser.parse_known_args()

    test_generate(
        config_file=args.config_file,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
        deterministic=args.deterministic,
        overrides=overrides,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
