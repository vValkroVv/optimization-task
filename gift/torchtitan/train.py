# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training entrypoint: an experiment YAML plus dot-path overrides.

::

    torchrun --nproc_per_node=8 -m torchtitan.train \\
        configs/experiments/llama3_debug.yaml \\
        --training.steps 100
"""

import os

import torch

from torchtitan.config.cli import get_argv, load_trainer_config
from torchtitan.config.save import save_loaded_run_bundle
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger, logger


def main(argv: list[str] | None = None) -> None:
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    argv = get_argv(argv)
    loaded = load_trainer_config(argv, logger=logger)
    config = loaded.trainer_config

    sl.init_structured_logger(
        source="training",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    save_loaded_run_bundle(loaded, logger=logger)

    # TODO(local_tensor): Remove this special case once LocalTensor supports
    # init_states() and foreach_allgather. In local tensor mode, skip
    # training/checkpointing as the model is not fully initialized.
    if config.comm.mode == "local_tensor":
        logger.info("Local tensor mode enabled - skipping training execution")
        return

    trainer = None
    try:
        trainer = config.build()

        if config.comm.mode == "fake_backend":
            logger.info(
                "Fake backend mode enabled - trainer construction completed; "
                "skipping training execution"
            )
        elif config.checkpoint.create_seed_checkpoint:
            assert int(os.environ["WORLD_SIZE"]) == 1, (
                "Must create seed checkpoint using a single device, to disable sharding."
            )
            assert config.checkpoint.enable, (
                "Must enable checkpointing when creating a seed checkpoint."
            )
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            with sl.log_trace_span("torch_distributed_teardown"):
                torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
