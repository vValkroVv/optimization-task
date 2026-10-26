from __future__ import annotations

from collections.abc import Iterable


class WindowCounters:
    """What the node lost or found inconsistent while assembling the steps of the window being written. The window's
    manifest reports them, and writing the window starts them over."""

    dropped_steps_per_rank: dict[int, int]
    torn_slots_per_rank: dict[int, int]
    late_slots_per_rank: dict[int, int]
    identical_violations: int

    def __init__(self, ranks: Iterable[int]) -> None:
        self._ranks = tuple(ranks)
        self.reset()

    def reset(self) -> None:
        self.dropped_steps_per_rank = dict.fromkeys(self._ranks, 0)
        self.torn_slots_per_rank = dict.fromkeys(self._ranks, 0)
        self.late_slots_per_rank = dict.fromkeys(self._ranks, 0)
        self.identical_violations = 0
