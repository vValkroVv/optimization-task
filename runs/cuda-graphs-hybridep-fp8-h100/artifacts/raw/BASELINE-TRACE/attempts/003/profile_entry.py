"""Run the native campaign trainer with rank-limited profiler construction."""
from __future__ import annotations

from unittest.mock import patch

import torch

from torchtitan.experiments.cuda_graph_campaign.train_scan_fix import (
    main as train_scan_fix_main,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler


SELECTED_GLOBAL_RANKS = (0, 1)
NATIVE_BUILD_TORCH_PROFILER = Profiler.build_torch_profiler
REAL_TORCH_PROFILE = torch.profiler.profile


def _build_torch_profiler_with_rank_filter(
    self,
    *,
    global_step: int,
    base_folder: str,
    leaf_folder: str,
):
    rank = torch.distributed.get_rank()
    if rank not in SELECTED_GLOBAL_RANKS:
        logger.info(
            f"NATIVE_TRACE_ADAPTER rank={rank} enabled=false selected_global_ranks=0,1"
        )
        return None

    logger.info(
        f"NATIVE_TRACE_ADAPTER rank={rank} enabled=true selected_global_ranks=0,1"
    )

    def configured_constructor(*args, **kwargs):
        kwargs.update(
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        )
        logger.info(
            f"NATIVE_TRACE_ADAPTER rank={rank} constructor_options "
            "record_shapes=false with_stack=false profile_memory=false"
        )
        return REAL_TORCH_PROFILE(*args, **kwargs)

    with patch("torch.profiler.profile", side_effect=configured_constructor):
        return NATIVE_BUILD_TORCH_PROFILER(
            self,
            global_step=global_step,
            base_folder=base_folder,
            leaf_folder=leaf_folder,
        )


def main() -> None:
    Profiler.build_torch_profiler = _build_torch_profiler_with_rank_filter
    train_scan_fix_main()


if __name__ == "__main__":
    main()
