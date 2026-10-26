from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from metric_sampler.capture.layout import Layout
from metric_sampler.contract.catalog import metric_by_id, metric_by_name
from metric_sampler.contract.metrics import ACTIVATION_COUNTS_WIDTH, INT32_MAX, Family, MetricSpec, ValueType
from metric_sampler.node.rank_slots import RankSlots
from metric_sampler.node.step_merger import Row, StepMerger
from metric_sampler.node.window_counters import WindowCounters

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")
_SCORES = metric_by_name("router/scores_mean")
_BIAS = metric_by_name("router/expert_bias_before_update")
_DEAD_EXPERTS = metric_by_name("summary/dead_experts")
_MAX_EXPERT_SHARE = metric_by_name("summary/max_expert_share")
_MISSING_SLOTS = metric_by_name("sampler/missing_slots")
_ROUTER_BIAS_LOCAL = metric_by_name("monitor/router_bias_local")
_OBSERVATIONS_LOCAL = metric_by_name("monitor/activation_local_observations")
_WEIGHTS_LOCAL = metric_by_name("monitor/materialized_weight_local")
# Above this a float32 can no longer hold every integer.
_FLOAT32_EXACT_INTEGER_LIMIT = 2**24
_NAN = float("nan")
_STEP = 7
_Recorded = Mapping[tuple[str, int, int], Sequence[float]]


@pytest.fixture
def layout() -> Layout:
    return Layout(
        [
            (_LOSS, -1, -1, 1),
            (_COUNTS, 0, -1, 4),
            (_SCORES, 0, -1, 4),
            (_BIAS, 0, -1, 4),
        ]
    )


def _full_step(
    loss: float = 1.0,
    counts: Sequence[int] = (1, 2, 3, 4),
    scores: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    bias: Sequence[float] = (0.5, 0.5, 0.5, 0.5),
) -> _Recorded:
    return {
        ("loss/total", -1, -1): [loss],
        ("router/tokens_per_expert", 0, -1): counts,
        ("router/scores_mean", 0, -1): scores,
        ("router/expert_bias_before_update", 0, -1): bias,
    }


def _rank_slots(layout: Layout, recorded: _Recorded) -> RankSlots:
    """The slots of one rank as the collector decodes them: the int32 words and the same words typed by the metric."""
    slots = {}
    for (metric_name, layer, microbatch), values in recorded.items():
        slot = layout.slot(metric_name, layer, microbatch)
        dtype = np.float32 if slot.metric.value_type is ValueType.FLOAT32 else np.int32
        words = np.array(values, dtype=dtype).view(np.int32)
        slots[(slot.metric.metric_id, layer, microbatch)] = (words, words.view(dtype))
    return slots


def _merge(
    layouts: dict[int, Layout],
    recorded_by_rank: dict[int, _Recorded],
    verbose: bool = False,
    primary_node: bool = True,
) -> tuple[list[Row], WindowCounters]:
    counters = WindowCounters(layouts)
    merger = StepMerger(layouts, primary_node, counters)
    ranks_data = {rank: _rank_slots(layouts[rank], recorded) for rank, recorded in recorded_by_rank.items()}
    return merger.merge(_STEP, ranks_data, verbose), counters


def _rows_of(rows: list[Row], metric: MetricSpec) -> list[Row]:
    """The rows of `metric` by layer, microbatch and rank."""
    return sorted(
        (row for row in rows if row.metric_id == metric.metric_id),
        key=lambda row: (row.source_metric_id, row.layer, row.microbatch, row.rank),
    )


def _per_rank(rows: list[Row], metric: MetricSpec) -> dict[int, list[float]]:
    return {row.rank: row.values.tolist() for row in _rows_of(rows, metric)}


@pytest.mark.parametrize(
    "point",
    [
        "residual", "attn_out", "mlp_out", "final_norm", "shared_expert_out",
        "attn_residual_contribution", "mlp_residual_contribution", "swiglu_in", "down_in", "embed",
    ],
)
@pytest.mark.parametrize("verbose", [False, True])
def test_exact_activation_counts_replace_the_moments_counters_and_stay_exact_in_their_own_rows(
    point: str, verbose: bool,
) -> None:
    moments = metric_by_name(f"act/{point}/moments")
    counts = metric_by_name(f"act/{point}/counts")
    layout = Layout([
        (metric, layer, microbatch, width)
        for layer in (0, 2) for microbatch in (0, 1)
        for metric, width in ((counts, ACTIVATION_COUNTS_WIDTH), (moments, 4))
    ])
    recorded = {}
    expected_counts = {}
    expected_moments = {}
    for rank in (0, 1):
        recorded[rank] = {}
        for layer in (0, 2):
            for microbatch in (0, 1):
                nonfinite = _FLOAT32_EXACT_INTEGER_LIMIT + 1 + rank * 100 + layer * 10 + microbatch
                finite = _FLOAT32_EXACT_INTEGER_LIMIT + 3 + rank * 100 + layer * 10 + microbatch
                vector = [nonfinite, finite] + [finite - 1, 1] + [0] * 62 + [0] * 63 + [7]
                recorded[rank][(counts.name, layer, microbatch)] = vector
                recorded[rank][(moments.name, layer, microbatch)] = [1.25, 4.5, 0.0, 0.0]
                expected_counts[(rank, layer, microbatch)] = vector
                expected_moments[(rank, layer, microbatch)] = [
                    1.25, 4.5, float(np.float32(nonfinite)), float(np.float32(finite)),
                ]

    rows, _ = _merge({0: layout, 1: layout}, recorded, verbose=verbose)

    for metric, expected in ((counts, expected_counts), (moments, expected_moments)):
        metric_rows = _rows_of(rows, metric)
        assert len(metric_rows) == 8
        assert {(row.rank, row.layer, row.microbatch): row.values.tolist() for row in metric_rows} == expected
        assert all(
            (row.step, row.source_metric_id, row.contributors, row.expected_contributors, row.identical, row.verbose)
            == (_STEP, 0, 1, 2, False, verbose)
            for row in metric_rows
        )
    assert _rows_of(rows, counts)[0].values.dtype == np.uint32
    assert _rows_of(rows, moments)[0].values.dtype == np.float32


@pytest.mark.parametrize("verbose", [False, True])
def test_a_row_repeating_its_alias_target_word_for_word_is_stored_without_values(verbose) -> None:
    names = ("router/tokens_per_expert", "dispatcher/assignments_before_capacity",
             "dispatcher/assignments_after_capacity",
             "dispatcher/valid_assignments_before_capacity", "dispatcher/valid_assignments_after_capacity")
    layout = Layout([(metric_by_name(name), 3, 1, 4) for name in names])
    # Rank 0 drops tokens at capacity, rank 1 does not; rank 0's valid counts differ before and after.
    counts = {
        0: ([5, 0, 7, 1], [5, 0, 7, 1], [5, 0, 6, 1], [4, 0, 7, 1], [4, 0, 6, 1]),
        1: ([2, 3, 0, 9], [2, 3, 0, 9], [2, 3, 0, 9], [2, 3, 0, 8], [2, 3, 0, 8]),
    }
    recorded = {rank: {(name, 3, 1): values for name, values in zip(names, counts[rank])} for rank in (0, 1)}

    rows, _ = _merge({0: layout, 1: layout}, recorded, verbose=verbose)

    ids = {name: metric_by_name(name).metric_id for name in names}
    stored = {(row.rank, row.metric_id): (row.alias_metric_id, row.values.tolist())
              for row in rows if row.source_metric_id == 0}
    tokens, valid = ids["router/tokens_per_expert"], ids["dispatcher/valid_assignments_before_capacity"]
    assert stored == {
        (0, tokens): (0, [5, 0, 7, 1]),
        (0, ids["dispatcher/assignments_before_capacity"]): (tokens, []),
        (0, ids["dispatcher/assignments_after_capacity"]): (0, [5, 0, 6, 1]),
        (0, valid): (0, [4, 0, 7, 1]),
        (0, ids["dispatcher/valid_assignments_after_capacity"]): (0, [4, 0, 6, 1]),
        (1, tokens): (0, [2, 3, 0, 9]),
        (1, ids["dispatcher/assignments_before_capacity"]): (tokens, []),
        (1, ids["dispatcher/assignments_after_capacity"]): (tokens, []),
        (1, valid): (0, [2, 3, 0, 8]),
        (1, ids["dispatcher/valid_assignments_after_capacity"]): (valid, []),
    }
    # Expert summaries still see every row's own counts.
    dead = {(row.rank, row.source_metric_id): row.values.tolist()
            for row in rows if row.metric_id == _DEAD_EXPERTS.metric_id}
    assert dead == {(rank, ids[name]): [1.0] for rank in (0, 1) for name in names}


def test_a_row_is_not_aliased_when_its_target_was_not_recorded() -> None:
    before = metric_by_name("dispatcher/assignments_before_capacity")
    layout = Layout([(before, 0, 0, 2)])

    rows, _ = _merge({0: layout}, {0: {(before.name, 0, 0): [3, 4]}})

    assert [(row.alias_metric_id, row.values.tolist()) for row in _rows_of(rows, before)] == [(0, [3, 4])]


def test_expert_summaries_count_dead_experts_and_the_largest_share_of_each_rank_exactly() -> None:
    layout = Layout([(_COUNTS, 2, 0, 4)])
    counts = {0: [7, 0, 6, 0], 1: [0, 0, 0, 0], 2: [1, 1, 1, INT32_MAX]}

    recorded = {rank: {(_COUNTS.name, 2, 0): values} for rank, values in counts.items()}

    rows, _ = _merge(dict.fromkeys(counts, layout), recorded)

    summaries = {
        (row.rank, row.metric_id): row.values.tolist() for row in rows if row.source_metric_id == _COUNTS.metric_id
    }
    largest_share = float(np.float32(INT32_MAX / (INT32_MAX + 3)))
    assert summaries == {
        (0, _DEAD_EXPERTS.metric_id): [2.0], (0, _MAX_EXPERT_SHARE.metric_id): [float(np.float32(7 / 13))],
        (1, _DEAD_EXPERTS.metric_id): [4.0], (1, _MAX_EXPERT_SHARE.metric_id): [0.0],
        (2, _DEAD_EXPERTS.metric_id): [0.0], (2, _MAX_EXPERT_SHARE.metric_id): [largest_share],
    }
    assert all(
        (row.contributors, row.expected_contributors, row.identical, row.verbose, row.alias_metric_id)
        == (1, 1, False, False, 0)
        for row in rows if row.source_metric_id == _COUNTS.metric_id
    )


def test_summary_rows_are_distinguished_by_source_metric_id() -> None:
    dispatcher_counts = metric_by_name("dispatcher/tokens_per_local_expert")
    layout = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (dispatcher_counts, 0, -1, 4)])
    recorded = {
        ("loss/total", -1, -1): [1.0],
        ("router/tokens_per_expert", 0, -1): [1, 2, 3, 4],
        ("dispatcher/tokens_per_local_expert", 0, -1): [5, 6, 7, 8],
    }

    rows, _ = _merge({0: layout}, {0: recorded})

    dead_rows = _rows_of(rows, _DEAD_EXPERTS)
    # Only source_metric_id tells the two summaries apart, preventing a dedup collision.
    assert len(dead_rows) == 2
    assert {(row.step, row.rank, row.layer, row.microbatch, row.metric_id) for row in dead_rows} == {
        (_STEP, 0, 0, -1, _DEAD_EXPERTS.metric_id)
    }
    assert {row.source_metric_id for row in dead_rows} == {_COUNTS.metric_id, dispatcher_counts.metric_id}
    assert [row.source_metric_id for row in _rows_of(rows, _LOSS) + _rows_of(rows, _COUNTS)] == [0, 0]


def test_summarize_experts_flag_gates_the_summary_rows_not_the_family() -> None:
    non_summarized_counts = metric_by_name("expert_loss/count")
    layout = Layout([(_LOSS, -1, -1, 1), (non_summarized_counts, 0, -1, 4)])
    recorded = {("loss/total", -1, -1): [1.0], ("expert_loss/count", 0, -1): [1, 2, 3, 4]}

    rows, _ = _merge({0: layout}, {0: recorded})

    assert sorted(row.metric_id for row in rows) == sorted([_LOSS.metric_id, non_summarized_counts.metric_id])


@pytest.mark.parametrize("declares_counts", [False, True])
def test_missing_exact_counts_keep_the_recorded_moments_and_do_not_hide_missing_required_slots(
    declares_counts: bool,
) -> None:
    moments = metric_by_name("act/residual/moments")
    requests = [(moments, 0, 0, 4), (_MISSING_SLOTS, -1, -1, 1)]
    if declares_counts:
        requests.append((metric_by_name("act/residual/counts"), 0, 0, ACTIVATION_COUNTS_WIDTH))
    layout = Layout(requests)

    rows, _ = _merge({0: layout}, {0: {(moments.name, 0, 0): [1.25, 4.5, 7.0, 9.0]}})

    assert [(row.metric_id, row.values.tolist()) for row in rows] == [
        (_MISSING_SLOTS.metric_id, [float(declares_counts)]),
        (moments.metric_id, [1.25, 4.5, 7.0, 9.0]),
    ]
    assert all(metric_by_id(row.metric_id).family is not Family.EXPERT_COUNTS for row in rows)


@pytest.mark.parametrize("third_rank_delivered", [False, True])
def test_health_counts_absent_required_microbatch_slots_without_inventing_measurements(
    third_rank_delivered: bool,
) -> None:
    activation = metric_by_name("act/residual/moments")
    optional_histogram = metric_by_name("expert_loss/hist")
    layout = Layout([
        (_COUNTS, 0, 0, 4), (_COUNTS, 0, 1, 4),
        (activation, 0, 0, 4), (activation, 0, 1, 4),
        (_MISSING_SLOTS, -1, -1, 1), (optional_histogram, 0, 0, 4),
    ])
    recorded = {
        0: {
            ("router/tokens_per_expert", 0, 0): [0, 0, 0, 0],
            ("router/tokens_per_expert", 0, 1): [1, 2, 3, 4],
            ("act/residual/moments", 0, 0): [0, 0, 0, 0],
            ("act/residual/moments", 0, 1): [1, 2, 0, 4],
        },
        1: {("router/tokens_per_expert", 0, 0): [4, 3, 2, 1]},
    }
    if third_rank_delivered:
        recorded[2] = {}

    rows, _ = _merge(dict.fromkeys(range(3), layout), recorded)

    expected_health = [(0, -1, [0.0], 1, 3), (1, -1, [3.0], 1, 3)]
    if third_rank_delivered:
        expected_health.append((2, -1, [4.0], 1, 3))
    assert [
        (row.rank, row.microbatch, row.values.tolist(), row.contributors, row.expected_contributors)
        for row in _rows_of(rows, _MISSING_SLOTS)
    ] == expected_health
    assert [(row.rank, row.microbatch, row.values.tolist()) for row in _rows_of(rows, activation)] == [
        (0, 0, [0.0, 0.0, 0.0, 0.0]), (0, 1, [1.0, 2.0, 0.0, 4.0]),
    ]
    assert [(row.rank, row.microbatch, row.values.tolist()) for row in _rows_of(rows, _COUNTS)] == [
        (0, 0, [0, 0, 0, 0]), (1, 0, [4, 3, 2, 1]), (0, 1, [1, 2, 3, 4]),
    ]


def test_a_key_nobody_recorded_produces_no_row() -> None:
    layout = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4)])

    rows, _ = _merge({0: layout, 1: layout}, {0: {("loss/total", -1, -1): [1.0]}, 1: {("loss/total", -1, -1): [2.0]}})

    assert [(row.rank, row.metric_id) for row in rows] == [(0, _LOSS.metric_id), (1, _LOSS.metric_id)]


def test_a_key_recorded_by_fewer_ranks_than_expected_shows_partial_contributors() -> None:
    layout = Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4)])
    recorded = {}
    for rank in range(8):
        recorded[rank] = {("loss/total", -1, -1): [float(rank)]}
        if rank < 3:
            recorded[rank][("router/tokens_per_expert", 0, -1)] = [rank + 1] * 4

    rows, _ = _merge(dict.fromkeys(range(8), layout), recorded)

    assert [
        (row.rank, row.contributors, row.expected_contributors, row.values.tolist()) for row in _rows_of(rows, _COUNTS)
    ] == [(0, 1, 8, [1, 1, 1, 1]), (1, 1, 8, [2, 2, 2, 2]), (2, 1, 8, [3, 3, 3, 3])]


def _heterogeneous_layouts() -> dict[int, Layout]:
    return {
        0: Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (_COUNTS, 1, -1, 4)]),
        1: Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4)]),
    }


def test_heterogeneous_layouts_expect_a_key_only_from_the_ranks_that_declare_it() -> None:
    recorded = {
        0: {
            ("loss/total", -1, -1): [1.0],
            ("router/tokens_per_expert", 0, -1): [1, 2, 3, 4],
            ("router/tokens_per_expert", 1, -1): [5, 6, 7, 8],
        },
        1: {("loss/total", -1, -1): [2.0], ("router/tokens_per_expert", 0, -1): [10, 20, 30, 40]},
    }

    rows, _ = _merge(_heterogeneous_layouts(), recorded)

    assert [
        (row.layer, row.rank, row.contributors, row.expected_contributors, row.values.tolist())
        for row in _rows_of(rows, _COUNTS)
    ] == [
        (0, 0, 1, 2, [1, 2, 3, 4]),
        (0, 1, 1, 2, [10, 20, 30, 40]),
        (1, 0, 1, 1, [5, 6, 7, 8]),
    ]


def test_the_expected_contributors_follow_the_layouts_also_when_a_rank_did_not_deliver() -> None:
    recorded = {
        0: {
            ("loss/total", -1, -1): [1.0],
            ("router/tokens_per_expert", 0, -1): [1, 2, 3, 4],
            ("router/tokens_per_expert", 1, -1): [5, 6, 7, 8],
        },
    }

    rows, _ = _merge(_heterogeneous_layouts(), recorded)

    assert {row.layer: (row.contributors, row.expected_contributors) for row in _rows_of(rows, _COUNTS)} == {
        0: (1, 2), 1: (1, 1),
    }


def test_large_counts_keep_each_rank_and_microbatch_without_summing() -> None:
    layout = Layout([(_COUNTS, 0, microbatch, 4) for microbatch in range(2)])
    recorded = {
        rank: {
            ("router/tokens_per_expert", 0, 0): [INT32_MAX, rank, 0, 0],
            ("router/tokens_per_expert", 0, 1): [0, 0, rank, INT32_MAX],
        }
        for rank in range(3)
    }

    rows, _ = _merge(dict.fromkeys(range(3), layout), recorded)

    counts_rows = _rows_of(rows, _COUNTS)
    assert [(row.rank, row.microbatch, row.values.tolist()) for row in counts_rows] == [
        (0, 0, [INT32_MAX, 0, 0, 0]),
        (1, 0, [INT32_MAX, 1, 0, 0]),
        (2, 0, [INT32_MAX, 2, 0, 0]),
        (0, 1, [0, 0, 0, INT32_MAX]),
        (1, 1, [0, 0, 1, INT32_MAX]),
        (2, 1, [0, 0, 2, INT32_MAX]),
    ]
    assert [(row.contributors, row.expected_contributors) for row in counts_rows] == [(1, 3)] * 6


def test_an_identical_metric_with_different_words_is_flagged_and_counted(layout: Layout) -> None:
    recorded = {
        0: _full_step(bias=[0.5, 0.5, 0.5, 0.5]),
        1: _full_step(bias=[0.75, 0.5, 0.5, 0.5]),
    }

    rows, counters = _merge({0: layout, 1: layout}, recorded)

    (bias_row,) = _rows_of(rows, _BIAS)
    assert (bias_row.rank, bias_row.contributors, bias_row.expected_contributors, bias_row.identical) == (
        -1, 2, 2, False,
    )
    assert counters.identical_violations == 1


def test_an_identical_metric_with_the_same_words_is_one_node_row(layout: Layout) -> None:
    rows, counters = _merge({0: layout, 1: layout}, {0: _full_step(), 1: _full_step(loss=2.0)})

    (bias_row,) = _rows_of(rows, _BIAS)
    assert (bias_row.rank, bias_row.contributors, bias_row.identical, bias_row.values.tolist()) == (
        -1, 2, True, [0.5, 0.5, 0.5, 0.5],
    )
    assert counters.identical_violations == 0


@pytest.mark.parametrize("verbose", [False, True])
def test_a_non_primary_node_skips_job_identical_rows_but_still_counts_violations(layout: Layout, verbose) -> None:
    recorded = {
        0: _full_step(bias=[0.5, 0.5, 0.5, 0.5]),
        1: _full_step(loss=2.0, bias=[0.75, 0.5, 0.5, 0.5]),
    }

    rows, counters = _merge({0: layout, 1: layout}, recorded, verbose=verbose, primary_node=False)

    assert _rows_of(rows, _BIAS) == []
    assert [row.rank for row in _rows_of(rows, _SCORES)] == [0, 1]
    assert [row.rank for row in _rows_of(rows, _COUNTS)] == [0, 1]
    assert counters.identical_violations == 1


@pytest.mark.parametrize(
    ("metric", "values_by_rank", "expected"),
    [
        (_ROUTER_BIAS_LOCAL, [[0.5, -1.0], [0.5, -1.0], [0.5, -1.0]], [(-1, 3, 3, True, [0.5, -1.0])]),
        (
            _ROUTER_BIAS_LOCAL, [[0.5, -1.0], [0.5, -2.0], [0.5, -1.0]],
            [(0, 1, 3, False, [0.5, -1.0]), (1, 1, 3, False, [0.5, -2.0]), (2, 1, 3, False, [0.5, -1.0])],
        ),
        (_OBSERVATIONS_LOCAL, [[3, 4096], [3, 4096], [3, 4096]], [(-1, 3, 3, True, [3, 4096])]),
        (
            _WEIGHTS_LOCAL, [[1.5, 20.0, 1.0], [_NAN, _NAN, 0.0], [1.5, 20.0, 1.0]],
            [(-1, 2, 2, True, [1.5, 20.0, 1.0])],
        ),
        (_WEIGHTS_LOCAL, [[_NAN, _NAN, 0.0], [1.5, 20.0, 1.0], [_NAN, _NAN, 0.0]],
         [(1, 1, 1, False, [1.5, 20.0, 1.0])]),
        (
            _WEIGHTS_LOCAL, [[1.5, 20.0, 1.0], [1.5, 21.0, 1.0], [_NAN, _NAN, 0.0]],
            [(0, 1, 2, False, [1.5, 20.0, 1.0]), (1, 1, 2, False, [1.5, 21.0, 1.0])],
        ),
        (_WEIGHTS_LOCAL, [[_NAN, _NAN, 0.0]] * 3, []),
    ],
)
def test_per_rank_rows_collapse_to_one_node_row_only_when_every_rank_reports_the_same_words(
    metric, values_by_rank, expected,
) -> None:
    layout = Layout([(metric, 4, 1, len(values_by_rank[0]))])
    recorded = {rank: {(metric.name, 4, 1): values} for rank, values in enumerate(values_by_rank)}

    rows, counters = _merge(dict.fromkeys(recorded, layout), recorded)

    metric_rows = _rows_of(rows, metric)
    assert [
        (row.rank, row.contributors, row.expected_contributors, row.identical,
         [pytest.approx(value) for value in row.values.tolist()])
        for row in metric_rows
    ] == expected
    assert all((row.step, row.layer, row.microbatch, row.verbose) == (_STEP, 4, 1, False) for row in metric_rows)
    assert counters.identical_violations == 0


def test_equal_rows_stay_per_rank_while_an_expected_rank_is_missing() -> None:
    layout = Layout([(_ROUTER_BIAS_LOCAL, 4, 1, 2)])
    recorded = {rank: {(_ROUTER_BIAS_LOCAL.name, 4, 1): [0.5, -1.0]} for rank in range(2)}

    rows, _ = _merge(dict.fromkeys(range(3), layout), recorded)

    assert [
        (row.rank, row.contributors, row.expected_contributors, row.identical)
        for row in _rows_of(rows, _ROUTER_BIAS_LOCAL)
    ] == [(0, 1, 3, False), (1, 1, 3, False)]


def test_a_verbose_step_keeps_a_row_per_rank_and_flags_it(layout: Layout) -> None:
    recorded = {
        0: _full_step(),
        1: _full_step(loss=2.0, counts=[10, 20, 30, 40], scores=[0.5] * 4),
    }

    rows, counters = _merge({0: layout, 1: layout}, recorded, verbose=True)

    counts_rows = _rows_of(rows, _COUNTS)
    assert len(counts_rows) == 2
    assert all(row.verbose for row in counts_rows)
    assert {tuple(row.values.tolist()) for row in counts_rows} == {(1, 2, 3, 4), (10, 20, 30, 40)}
    bias_rows = _rows_of(rows, _BIAS)
    assert len(bias_rows) == 2
    assert all(row.identical for row in bias_rows)
    assert all(not row.identical for row in _rows_of(rows, _SCORES))
    assert counters.identical_violations == 0
    assert _per_rank(rows, _DEAD_EXPERTS) == {0: [0.0], 1: [0.0]}
    assert _per_rank(rows, _MAX_EXPERT_SHARE) == {0: [pytest.approx(4 / 10)], 1: [pytest.approx(40 / 100)]}


def test_a_verbose_step_splits_an_identical_metric_without_multiplying_its_expected_contributors(
    layout: Layout,
) -> None:
    # health_per_step sums expected_contributors across every row; an IDENTICAL metric's
    # verbose per-rank rows must not repeat the full expected count, or the total inflates.
    recorded = {0: _full_step(), 1: _full_step(loss=2.0, counts=[10, 20, 30, 40], scores=[0.5] * 4)}

    rows, _ = _merge({0: layout, 1: layout}, recorded, verbose=True)

    assert [row.expected_contributors for row in _rows_of(rows, _COUNTS)] == [2, 2]
    assert [row.expected_contributors for row in _rows_of(rows, _BIAS)] == [1, 1]
    assert [row.expected_contributors for row in _rows_of(rows, _SCORES)] == [2, 2]
    assert [row.expected_contributors for row in _rows_of(rows, _LOSS)] == [2, 2]


def test_a_verbose_step_with_a_mismatched_identical_metric_is_flagged_and_counted(layout: Layout) -> None:
    recorded = {0: _full_step(), 1: _full_step(bias=[0.75, 0.5, 0.5, 0.5])}

    rows, counters = _merge({0: layout, 1: layout}, recorded, verbose=True)

    bias_rows = _rows_of(rows, _BIAS)
    assert len(bias_rows) == 2
    assert all(not row.identical for row in bias_rows)
    assert counters.identical_violations == 1
