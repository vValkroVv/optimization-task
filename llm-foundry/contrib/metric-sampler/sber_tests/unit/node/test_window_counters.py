from __future__ import annotations

import pytest

from metric_sampler.node.window_counters import WindowCounters

pytestmark = pytest.mark.unit


def test_every_rank_starts_with_nothing_lost_and_reset_starts_over() -> None:
    counters = WindowCounters([3, 1])
    assert (counters.dropped_steps_per_rank, counters.torn_slots_per_rank, counters.late_slots_per_rank) == (
        {3: 0, 1: 0},
        {3: 0, 1: 0},
        {3: 0, 1: 0},
    )
    assert counters.identical_violations == 0

    counters.dropped_steps_per_rank[3] += 2
    counters.torn_slots_per_rank[1] += 1
    counters.late_slots_per_rank[3] += 4
    counters.identical_violations += 5
    counters.reset()

    assert (counters.dropped_steps_per_rank, counters.torn_slots_per_rank, counters.late_slots_per_rank) == (
        {3: 0, 1: 0},
        {3: 0, 1: 0},
        {3: 0, 1: 0},
    )
    assert counters.identical_violations == 0
