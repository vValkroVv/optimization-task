from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.ring import ShmRing, mask_word_count
from metric_sampler.contract.metrics import ValueType
from metric_sampler.node.rank_slots import RankSlots, SlotKey, recorded_slot_indices, recorded_words, slot_key
from metric_sampler.node.spool import Spool
from metric_sampler.node.window_counters import WindowCounters

_LIVENESS_PROBE_SIGNAL = 0


@dataclass
class PendingStep:
    first_seen: float
    ranks: dict[int, RankSlots] = field(default_factory=dict)
    verbose: bool = False


class StepCollector:
    """Reads the slots the ranks have ready in their rings into per-step, per-rank data, spooling each one first,
    and tells which steps no live rank still owes a slot to."""

    def __init__(
        self,
        rings: dict[int, ShmRing],
        rank_layouts: dict[int, Layout],
        spool: Spool,
        counters: WindowCounters,
        clock: Callable[[], float],
        rank_timeout_s: float,
    ) -> None:
        self._rings = rings
        self._rank_layouts = rank_layouts
        self._all_ranks = set(rank_layouts)
        self._spool = spool
        self._counters = counters
        self._clock = clock
        self._rank_timeout_s = rank_timeout_s
        self._settled_steps: set[int] = set()
        self._pending: dict[int, PendingStep] = {}
        # Per layout, each slot's key, words and value dtype, which _decode reads for every recorded slot.
        self._slot_spans: dict[Layout, list[tuple[SlotKey, int, int, type[np.generic]]]] = {}

    def restore_spooled_slots(self) -> None:
        """Takes back, at start-up, the slots a stopped node process spooled for steps no window has published."""
        self._settled_steps.update(self._spool.committed_steps)
        for slot in self._spool.load_slots():
            pending = self._pending.setdefault(slot.step, PendingStep(first_seen=self._clock()))
            pending.ranks[slot.rank] = self._decode(self._rank_layouts[slot.rank], slot.words, slot.mask)
            if slot.verbose:
                pending.verbose = True
            self._counters.dropped_steps_per_rank[slot.rank] += slot.dropped_before

    def collect_ready_slots(self) -> None:
        for rank, ring in self._rings.items():
            for slot_index in ring.ready_slots():
                header = ring.header(slot_index)

                if header.step in self._settled_steps:
                    ring.release(slot_index)
                    self._counters.late_slots_per_rank[rank] += 1
                    continue

                raw = ring.payload(slot_index)
                if int(raw[-1]) != header.step:
                    ring.release(slot_index)
                    self._counters.torn_slots_per_rank[rank] += 1
                    continue

                pending = self._pending.setdefault(header.step, PendingStep(first_seen=self._clock()))
                if rank in pending.ranks:
                    ring.release(slot_index)
                    continue
                if ring.verbose_until_step >= header.step:
                    pending.verbose = True
                self._spool.persist_slot(rank, header.step, header.dropped_before, pending.verbose, ring.layout, raw)
                decoded = self._decode(ring.layout, raw)
                ring.release(slot_index)
                self._counters.dropped_steps_per_rank[rank] += header.dropped_before
                pending.ranks[rank] = decoded

    def settleable_steps(self, force: bool = False) -> Iterator[tuple[int, PendingStep]]:
        """The pending steps, oldest first, that every live rank has delivered, or with `force` all of them. A step
        still waiting for live ranks after the rank timeout raises TimeoutError. The caller settles each step it
        takes."""
        gone_ranks = self._gone_ranks() if self._rings else self._all_ranks
        for step in sorted(self._pending):
            pending = self._pending[step]
            still_expected = self._all_ranks - (gone_ranks - pending.ranks.keys())
            complete = still_expected <= pending.ranks.keys()
            timed_out = (self._clock() - pending.first_seen) > self._rank_timeout_s
            if not complete and not force:
                if timed_out:
                    raise TimeoutError(
                        f"metric sampler step {step} is waiting for live ranks "
                        f"{sorted(still_expected - pending.ranks.keys())}"
                    )
                continue
            yield step, pending

    def settle(self, step: int) -> None:
        self._settled_steps.add(step)
        del self._pending[step]

    def missing_ranks(self, pending: PendingStep) -> list[int]:
        return sorted(self._all_ranks - pending.ranks.keys())

    def all_ranks_stopped(self) -> bool:
        return all(ring.stop_requested for ring in self._rings.values())

    def any_rank_alive(self, rank_pids: list[int]) -> bool:
        return any(self._rank_alive(pid) for pid in rank_pids)

    def _decode(self, layout: Layout, raw: np.ndarray, mask: np.ndarray | None = None) -> RankSlots:
        if mask is None:
            mask_words = mask_word_count(len(layout.slots))
            mask = raw[layout.words : layout.words + mask_words].view(np.uint32)
        recorded_indices = recorded_slot_indices(layout, mask)
        if not len(recorded_indices):
            return {}
        # Copy only the observed prefix before the ring slot is released.
        raw = raw[: recorded_words(layout, recorded_indices)].copy()
        spans = self._slot_spans.get(layout)
        if spans is None:
            spans = self._slot_spans[layout] = [
                (
                    slot_key(slot),
                    slot.offset,
                    slot.offset + slot.width,
                    np.float32 if slot.metric.value_type is ValueType.FLOAT32 else np.int32,
                )
                for slot in layout.slots
            ]
        decoded = {}
        for index in recorded_indices.tolist():
            key, start, stop, dtype = spans[index]
            words = raw[start:stop]
            decoded[key] = (words, words.view(dtype))
        return decoded

    def _gone_ranks(self) -> set[int]:
        # A rank that stopped with nothing left to collect, or whose process died, is excluded
        # from the timeout wait so one dropped rank doesn't stall every window for the run.
        return {
            rank
            for rank, ring in self._rings.items()
            if (ring.stop_requested and not ring.ready_slots()) or not self._rank_alive(ring.owner_pid)
        }

    @staticmethod
    def _rank_alive(pid: int) -> bool:
        try:
            os.kill(pid, _LIVENESS_PROBE_SIGNAL)
        except OSError:
            return False
        return True
