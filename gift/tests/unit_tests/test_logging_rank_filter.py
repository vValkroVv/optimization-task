# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One step line per step, whatever the launcher.

``run_train.sh`` hides ranks > 0 with torchrun's ``--local-ranks-filter``; the
cloud backend runs one process per GPU under ``mpirun`` with no torchrun in the
tree, so nothing filtered and every INFO record appeared ``world_size`` times.
``init_logger`` does the filtering itself now: only the ranks named by
``GIFT_LOG_RANK`` (rank 0 by default; bare ``LOG_RANK`` honored as a fallback)
print INFO, every rank still prints warnings.
"""

import io

import pytest

from torchtitan.tools.logging import init_logger, logger

_RANK_ENV = (
    "RANK",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "SLURM_PROCID",
    "LOG_RANK",
    "GIFT_LOG_RANK",
)


@pytest.fixture(autouse=True)
def restore_root_logger():
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


def _console(monkeypatch, **env: str) -> io.StringIO:
    """Init the logger in an environment and return what it writes."""
    for name in _RANK_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    init_logger()
    (handler,) = logger.handlers
    handler.stream = stream = io.StringIO()
    logger.info("step:  1  loss: 12.77173")
    logger.warning("NaN in the gradient")
    return stream


def test_rank_zero_prints_everything_under_the_unchanged_prefix(monkeypatch):
    lines = _console(monkeypatch, RANK="0").getvalue().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("[titan] ") for line in lines)
    assert "step:  1" in lines[0]


def test_other_ranks_drop_the_step_line_but_keep_warnings_and_name_themselves(
    monkeypatch,
):
    lines = _console(monkeypatch, RANK="3").getvalue().splitlines()
    assert len(lines) == 1  # the step line every rank repeats is gone
    assert lines[0].startswith("[titan][rank3] ") and "NaN" in lines[0]


def test_a_bare_mpirun_rank_is_recognised_without_torchrun(monkeypatch):
    # No RANK: only the MPI launcher's own variable is set.
    def lines(rank):
        return _console(monkeypatch, OMPI_COMM_WORLD_RANK=rank).getvalue().splitlines()

    assert len(lines("2")) == 1
    assert len(lines("0")) == 2


def test_log_rank_opens_the_console_for_the_ranks_it_names(monkeypatch):
    def infos(**env):
        return _console(monkeypatch, **env).getvalue().count("step:")

    assert infos(RANK="3", GIFT_LOG_RANK="0,3") == 1
    assert infos(RANK="3", GIFT_LOG_RANK="all") == 1
    assert infos(RANK="3", GIFT_LOG_RANK="0") == 0
    # Bare LOG_RANK (what run_train.sh exports for torchrun) still counts...
    assert infos(RANK="3", LOG_RANK="0,3") == 1
    # ...but the GIFT_ variable is the one with the final word.
    assert infos(RANK="3", GIFT_LOG_RANK="0", LOG_RANK="0,3") == 0


def test_no_rank_env_at_all_is_a_single_process_run(monkeypatch):
    assert _console(monkeypatch).getvalue().count("step:") == 1
