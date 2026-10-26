from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from metric_sampler.capture.layout import Layout
from metric_sampler.contract.catalog import METRICS, metric_by_id, metric_by_name
from metric_sampler.contract.metrics import NO_LAYER, NO_MICROBATCH, Family, MetricSpec, NodeMerge
from metric_sampler.node.rank_slots import RankSlots, SlotKey, slot_key
from metric_sampler.node.window_counters import WindowCounters

_DEAD_EXPERTS = metric_by_name("summary/dead_experts")
_MAX_EXPERT_SHARE = metric_by_name("summary/max_expert_share")
_MISSING_SLOTS = metric_by_name("sampler/missing_slots")
_ACTIVATION_COUNTS_BY_MOMENTS = {
    metric.metric_id: metric_by_name(metric.name.removesuffix("/moments") + "/counts").metric_id
    for metric in METRICS
    if metric.name.startswith("act/") and metric.name.endswith("/moments")
}
# act/<point>/moments ends with [nonfinite_count, finite_count], which act/<point>/counts holds as its first two words.
_MOMENTS_COUNTERS = slice(2, 4)
_COUNTS_COUNTERS = slice(0, 2)

# The source_metric_id of a row that is not a summary and the alias_metric_id of one that keeps its own values.
_NO_SOURCE_METRIC_ID = 0
_NO_ALIAS_METRIC_ID = 0
# The rank of a row stored once for every rank of the node.
_NODE_ROW_RANK = -1
# One rank's row has no other rank to be collapsed with.
_FEWEST_RANKS_TO_COLLAPSE = 2


# A node makes one row per rank, metric, layer and microbatch of every step, so rows are neither frozen nor built with
# keyword arguments, which both make creating one much slower. They are built positionally, in the order of the fields.
@dataclass(slots=True)
class Row:
    step: int
    rank: int
    microbatch: int
    layer: int
    metric_id: int
    source_metric_id: int
    contributors: int
    expected_contributors: int
    identical: bool
    verbose: bool
    values: np.ndarray
    alias_metric_id: int = _NO_ALIAS_METRIC_ID


def _same_words(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two slots hold the same words bit for bit. For integer words this is np.array_equal's answer
    at a fraction of its cost."""
    return first.tobytes() == second.tobytes()


class StepMerger:
    """Merges what every rank recorded in one step into the rows of the node."""

    def __init__(self, rank_layouts: dict[int, Layout], primary_node: bool, counters: WindowCounters) -> None:
        self._primary_node = primary_node
        self._counters = counters
        self._expected_contributors_by_key = Counter(
            key for layout in rank_layouts.values() for key in {slot_key(slot) for slot in layout.slots}
        )
        self._required_keys_by_rank = {
            rank: {slot_key(slot) for slot in layout.slots if slot.metric.required}
            for rank, layout in rank_layouts.items()
            if any(slot.metric.metric_id == _MISSING_SLOTS.metric_id for slot in layout.slots)
        }

    def merge(self, step: int, ranks_data: dict[int, RankSlots], verbose: bool) -> list[Row]:
        rows = self._missing_slots_rows(step, ranks_data, verbose)

        for key in sorted({key for data in ranks_data.values() for key in data}):
            metric_id = key[0]
            metric = metric_by_id(metric_id)
            contributing = sorted(rank for rank, data in ranks_data.items() if key in data)
            expected = self._expected_contributors(key)
            counts_metric_id = _ACTIVATION_COUNTS_BY_MOMENTS.get(metric_id)
            if counts_metric_id is not None:
                self._take_exact_activation_counters(ranks_data, key, contributing, counts_metric_id)
            skip_row = metric.job_identical and not self._primary_node

            if verbose:
                rows.extend(self._verbose_rows(step, key, metric, ranks_data, contributing, expected, skip_row))
            elif metric.node_merge is NodeMerge.PER_RANK:
                if metric.presence_index is not None:
                    contributing = [
                        rank for rank in contributing if ranks_data[rank][key][1][metric.presence_index] != 0
                    ]
                    expected = len(contributing)
                rows.extend(self._per_rank_rows(step, key, metric, ranks_data, contributing, expected))
            elif skip_row:
                # Only counts a violation: the primary node stores the row.
                self._identical_across(ranks_data, key, contributing)
            else:
                rows.append(self._identical_row(step, key, metric, ranks_data, contributing, expected))

        return rows

    def _expected_contributors(self, key: SlotKey) -> int:
        return self._expected_contributors_by_key.get(key, 0)

    def _missing_slots_rows(self, step: int, ranks_data: dict[int, RankSlots], verbose: bool) -> list[Row]:
        rows = []
        for rank, required_keys in self._required_keys_by_rank.items():
            if rank in ranks_data:
                missing_slots = len(required_keys - ranks_data[rank].keys())
                rows.append(
                    Row(
                        step, rank, NO_MICROBATCH, NO_LAYER, _MISSING_SLOTS.metric_id, _NO_SOURCE_METRIC_ID, 1,
                        len(self._required_keys_by_rank), False, verbose, np.array([missing_slots], dtype=np.float32),
                    )
                )
        return rows

    @staticmethod
    def _take_exact_activation_counters(
        ranks_data: dict[int, RankSlots], key: SlotKey, contributing: list[int], counts_metric_id: int
    ) -> None:
        """Replaces the float32 [nonfinite_count, finite_count] words of an activation's moments with the exact
        integer counts the rank recorded beside them, where it did."""
        _, layer, microbatch = key
        for rank in contributing:
            counts = ranks_data[rank].get((counts_metric_id, layer, microbatch))
            if counts is not None:
                raw, values = ranks_data[rank][key]
                values = values.copy()
                values[_MOMENTS_COUNTERS] = counts[1][_COUNTS_COUNTERS]
                ranks_data[rank][key] = (raw, values)

    def _verbose_rows(
        self,
        step: int,
        key: SlotKey,
        metric: MetricSpec,
        ranks_data: dict[int, RankSlots],
        contributing: list[int],
        expected: int,
        skip_row: bool,
    ) -> list[Row]:
        metric_id, layer, microbatch = key
        identical = False
        if metric.node_merge is NodeMerge.IDENTICAL:
            identical = self._identical_across(ranks_data, key, contributing)
        if skip_row:
            return []
        # Splitting an IDENTICAL metric into per-rank rows here must not multiply its
        # `expected`, or downstream expected_contributors_total inflates in a verbose window.
        row_expected = expected if metric.node_merge is NodeMerge.PER_RANK else 1
        rows = []
        for rank in contributing:
            values, alias_metric_id = self._rank_values(metric, ranks_data[rank], key)
            rows.append(
                Row(
                    step, rank, microbatch, layer, metric_id, _NO_SOURCE_METRIC_ID, 1, row_expected, identical, True,
                    values, alias_metric_id,
                )
            )
        if metric.summarize_experts:
            rows.extend(self._summary_rows(step, key, ranks_data, contributing))
        return rows

    def _per_rank_rows(
        self,
        step: int,
        key: SlotKey,
        metric: MetricSpec,
        ranks_data: dict[int, RankSlots],
        contributing: list[int],
        expected: int,
    ) -> list[Row]:
        metric_id, layer, microbatch = key
        rows = []
        if metric.collapse_equal_ranks and self._equal_across_all(ranks_data, key, contributing, expected):
            values = self._stored_values(metric, ranks_data[contributing[0]][key][1])
            rows.append(
                Row(
                    step, _NODE_ROW_RANK, microbatch, layer, metric_id, _NO_SOURCE_METRIC_ID, expected, expected,
                    True, False, values,
                )
            )
        else:
            for rank in contributing:
                values, alias_metric_id = self._rank_values(metric, ranks_data[rank], key)
                rows.append(
                    Row(
                        step, rank, microbatch, layer, metric_id, _NO_SOURCE_METRIC_ID, 1, expected, False, False,
                        values, alias_metric_id,
                    )
                )
        if metric.summarize_experts:
            rows.extend(self._summary_rows(step, key, ranks_data, contributing))
        return rows

    def _identical_across(self, ranks_data: dict[int, RankSlots], key: SlotKey, contributing: list[int]) -> bool:
        """Whether every contributing rank holds the same words for `key`; a difference counts as an identical
        violation of the window."""
        raw_values = [ranks_data[rank][key][0] for rank in contributing]
        identical = all(_same_words(raw_values[0], raw) for raw in raw_values[1:])
        if not identical:
            self._counters.identical_violations += 1
        return identical

    @staticmethod
    def _equal_across_all(
        ranks_data: dict[int, RankSlots], key: SlotKey, contributing: list[int], expected: int
    ) -> bool:
        """Every expected rank reported the key, more than one of them, and all with the same words."""
        if len(contributing) < _FEWEST_RANKS_TO_COLLAPSE or len(contributing) != expected:
            return False
        first = ranks_data[contributing[0]][key][0]
        return all(_same_words(first, ranks_data[rank][key][0]) for rank in contributing[1:])

    def _identical_row(
        self,
        step: int,
        key: SlotKey,
        metric: MetricSpec,
        ranks_data: dict[int, RankSlots],
        contributing: list[int],
        expected: int,
    ) -> Row:
        _, layer, microbatch = key
        identical = self._identical_across(ranks_data, key, contributing)
        first_typed = ranks_data[contributing[0]][key][1]
        values = self._stored_values(metric, first_typed)
        return Row(
            step, _NODE_ROW_RANK, microbatch, layer, metric.metric_id, _NO_SOURCE_METRIC_ID, len(contributing),
            expected, identical, False, values,
        )

    def _summary_rows(
        self, step: int, key: SlotKey, ranks_data: dict[int, RankSlots], contributing: list[int]
    ) -> list[Row]:
        source_metric_id, layer, microbatch = key
        # Every contributing rank at once: one numpy call per statistic instead of one per rank.
        counts = np.stack([ranks_data[rank][key][1] for rank in contributing]).astype(np.int64)
        totals = counts.sum(axis=1)
        dead = np.count_nonzero(counts == 0, axis=1).astype(np.float32)
        # The largest count over the total in float64, then float32; 0 for a rank without tokens.
        shares = np.divide(
            counts.max(axis=1).astype(np.float64), totals, out=np.zeros(len(contributing)), where=totals > 0
        ).astype(np.float32)
        rows = []
        for index, rank in enumerate(contributing):
            rows.append(
                Row(
                    step, rank, microbatch, layer, _DEAD_EXPERTS.metric_id, source_metric_id, 1, 1, False, False,
                    dead[index : index + 1],
                )
            )
            rows.append(
                Row(
                    step, rank, microbatch, layer, _MAX_EXPERT_SHARE.metric_id, source_metric_id, 1, 1, False, False,
                    shares[index : index + 1],
                )
            )
        return rows

    def _rank_values(self, metric: MetricSpec, rank_data: RankSlots, key: SlotKey) -> tuple[np.ndarray, int]:
        """A rank's stored values and 0, or no values and the aliased metric's id when they repeat
        the rank's row of `metric.alias_of` word for word."""
        values = self._stored_values(metric, rank_data[key][1])
        if metric.alias_of is None:
            return values, _NO_ALIAS_METRIC_ID
        aliased = metric_by_name(metric.alias_of)
        aliased_data = rank_data.get((aliased.metric_id, key[1], key[2]))
        if aliased_data is not None and _same_words(self._stored_values(aliased, aliased_data[1]), values):
            return values[:0], aliased.metric_id
        return values, _NO_ALIAS_METRIC_ID

    @staticmethod
    def _stored_values(metric: MetricSpec, typed: np.ndarray) -> np.ndarray:
        # Expert counts hold INT32 words, which are the uint32 values bit for bit; float32 values are kept as
        # they are. Either way no copy: the rank's decoded words are its own.
        if metric.family is Family.EXPERT_COUNTS:
            return typed.view(np.uint32)
        return typed.astype(np.float32, copy=False)
