from __future__ import annotations

import re

import pytest

from metric_sampler.cloud.summaries.catalog import DEFAULT_CHUNK_STEPS, SUMMARIES, SUMMARIES_BY_NAME, Summary
from sber_tests.unit.cloud.sql_sources import SCHEMA_SQL, SERVING_SQL

pytestmark = pytest.mark.unit

_PLACEHOLDER = re.compile(r"__[A-Z]+__")
_DEFINITIONS = SCHEMA_SQL.read_text(encoding="utf-8") + SERVING_SQL


def test_every_summary_has_one_loop_and_the_catalog_finds_it_by_name() -> None:
    assert len(SUMMARIES) == 43
    assert list(SUMMARIES_BY_NAME) == [summary.name for summary in SUMMARIES]


@pytest.mark.parametrize("summary", SUMMARIES, ids=lambda summary: summary.name)
def test_a_source_selects_through_the_filter_of_the_chunk_and_names_only_known_placeholders(summary) -> None:
    assert summary.source.startswith("SELECT ")
    assert "__WHERE__" in summary.source
    assert set(_PLACEHOLDER.findall(summary.source)) <= {"__EXP__", "__RUN__", "__FROM__", "__TO__", "__WHERE__"}
    assert "$" not in summary.source


@pytest.mark.parametrize("summary", SUMMARIES, ids=lambda summary: summary.name)
def test_a_source_reads_a_view_or_table_of_the_serving_layer(summary) -> None:
    source = re.search(r"FROM metric_sampler\.(\w+)", summary.source)[1]

    assert re.search(rf"CREATE (OR REPLACE VIEW|TABLE IF NOT EXISTS) metric_sampler\.{source}\s", _DEFINITIONS)


@pytest.mark.parametrize("summary", SUMMARIES, ids=lambda summary: summary.name)
def test_a_loop_keeps_the_progress_of_summary_tables_that_exist(summary) -> None:
    for name in summary.progress_names:
        assert f"CREATE TABLE IF NOT EXISTS metric_sampler.summary_{name}\n" in SERVING_SQL


@pytest.mark.parametrize("summary", [summary for summary in SUMMARIES if summary.after], ids=lambda s: s.name)
def test_the_summary_a_loop_is_derived_from_is_a_loop_of_the_catalog(summary) -> None:
    assert summary.after in SUMMARIES_BY_NAME


@pytest.mark.parametrize(
    ("name", "chunk_steps"),
    [
        ("layer_load", 5), ("expert_totals", 10), ("activation", 10), ("activation_rank", 5), ("activation_mb", 10),
        ("activation_rank_mb", 3), ("activation_massive_levels", 3), ("activation_exp_quantiles_levels", 3),
        ("expert_stream", 100), ("layer_load_mb", 5), ("attention_softmax", 5), ("attention_rank", 5),
        ("grad_per_rank", 10),
    ],
)
def test_the_loops_whose_chunks_hold_more_per_step_take_fewer_steps(name, chunk_steps) -> None:
    assert SUMMARIES_BY_NAME[name].chunk_steps == chunk_steps
    assert DEFAULT_CHUNK_STEPS == 10


def test_a_summary_keeps_its_own_progress_unless_it_lists_the_summaries_whose_progress_it_keeps() -> None:
    assert SUMMARIES_BY_NAME["layer_load"].progress_names == ("layer_load",)
    assert SUMMARIES_BY_NAME["activation_massive_levels"].progress_names == (
        "activation_massive", "activation_massive_rank", "activation_massive_mb", "activation_massive_rank_mb",
    )
    assert SUMMARIES_BY_NAME["activation_exp_quantiles_levels"].progress_names == (
        "activation_exp_quantiles", "activation_exp_quantiles_rank", "activation_exp_quantiles_mb",
        "activation_exp_quantiles_rank_mb",
    )


def test_the_catalog_is_in_the_order_the_loops_were_started_in() -> None:
    names = [summary.name for summary in SUMMARIES]

    assert names[:12] == [
        "layer_load", "expert_totals", "activation", "loss_quantiles", "stream_loss", "stream_share", "expert_loss",
        "loss_position", "sample_loss_quantiles", "grad_per_rank", "timing_per_rank", "router_rank",
    ]
    assert names[-4:] == ["attention_softmax", "attention_linear", "attention_rank", "attention_entropy_hist"]


def test_a_chunk_select_names_the_run_and_the_steps_of_the_chunk_in_every_placeholder() -> None:
    summary = Summary(
        "x",
        "SELECT * FROM v(exp_filter = '__EXP__', run_filter = '__RUN__', step_from = __FROM__, step_to = __TO__) "
        "WHERE __WHERE__",
    )

    assert summary.chunk_select("e1", "r2", 100, 120) == (
        "SELECT * FROM v(exp_filter = 'e1', run_filter = 'r2', step_from = 100, step_to = 120) "
        "WHERE exp = 'e1' AND run = 'r2' AND step BETWEEN 100 AND 120"
    )


def test_a_chunk_insert_selects_the_chunk_into_the_summary_table_with_the_limits_it_was_given() -> None:
    summary = Summary("x", "SELECT * FROM v(step_to = __TO__) WHERE __WHERE__")

    assert summary.chunk_insert("e", "r", 1, 5, max_threads=3, log_comment="check") == (
        "INSERT INTO metric_sampler.summary_x SELECT now64(3), * FROM "
        "(SELECT * FROM v(step_to = 5) WHERE exp = 'e' AND run = 'r' AND step BETWEEN 1 AND 5) "
        "SETTINGS max_threads = 3, priority = 10, max_execution_time = 1800, max_memory_usage = 30000000000, "
        "max_bytes_before_external_group_by = 10000000000, optimize_move_to_prewhere_if_final = 1, "
        "log_comment = 'check-x'"
    )


@pytest.mark.parametrize(
    ("source", "bounded"),
    [
        ("SELECT * FROM v(step_to = __TO__) WHERE __WHERE__", True),
        ("SELECT * FROM v(step_from = __FROM__) WHERE __WHERE__", False),
    ],
)
def test_a_source_is_bounded_to_the_chunk_when_it_takes_the_last_step_of_the_chunk(source, bounded) -> None:
    assert Summary("x", source).bounded_to_chunk is bounded


def test_the_sources_that_do_not_take_the_last_step_of_their_chunk_change_with_any_row_of_the_run() -> None:
    assert [summary.name for summary in SUMMARIES if not summary.bounded_to_chunk] == [
        "grad_per_rank", "timing_per_rank", "router_rank",
    ]
