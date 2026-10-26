# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys


logger = logging.getLogger()


# Where a launcher writes this process's rank. torchrun sets RANK; the cloud
# backend runs one process per GPU under mpirun, where only the MPI variable
# exists. First one present wins.
_RANK_ENV_VARS = ("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID")


def _rank() -> int:
    for name in _RANK_ENV_VARS:
        value = os.environ.get(name)
        if value is not None and value.strip():
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def _console_ranks() -> set[int] | None:
    """Ranks whose INFO records reach the console; ``None`` means every rank.

    ``GIFT_LOG_RANK`` is ours (the GIFT_ prefix marks it as such); bare
    ``LOG_RANK`` is read as a fallback because ``run_train.sh`` already
    exports it for torchrun's ``--local-ranks-filter``, and the two filters
    should agree when only that one is set.
    """
    value = os.environ.get("GIFT_LOG_RANK", os.environ.get("LOG_RANK", "")).strip()
    if not value:
        return {0}
    if value.lower() == "all":
        return None
    ranks = set()
    for part in value.replace(" ", "").split(","):
        if part:
            ranks.add(int(part))
    return ranks or {0}


def init_logger() -> None:
    """Console logging, with one step line per step whatever the launcher.

    ``run_train.sh`` hides ranks > 0 with torchrun's ``--local-ranks-filter``,
    but the cloud backend has no torchrun in the tree, so nothing filtered and
    every INFO record appeared ``world_size`` times. The filtering happens here
    instead: only the ranks ``GIFT_LOG_RANK`` names (rank 0 by default,
    ``LOG_RANK`` honored as a fallback) print INFO.
    Warnings and errors keep coming from every rank -- a NaN on rank 5 is the
    one message that must not be swallowed -- and carry the rank in the prefix
    so it is clear who is speaking.
    """
    rank = _rank()
    ranks = _console_ranks()
    speaks = ranks is None or rank in ranks

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if speaks else logging.WARNING)
    tag = "[titan]" if rank == 0 else f"[titan][rank{rank}]"
    formatter = logging.Formatter(
        f"{tag} %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


_logged: set[str] = set()


def warn_once(logger: logging.Logger, msg: str) -> None:
    """Log a warning message only once per unique message.

    Uses a global set to track messages that have already been logged
    to prevent duplicate warning messages from cluttering the output.

    Args:
        logger (logging.Logger): The logger instance to use for warning.
        msg (str): The warning message to log.
    """
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)
