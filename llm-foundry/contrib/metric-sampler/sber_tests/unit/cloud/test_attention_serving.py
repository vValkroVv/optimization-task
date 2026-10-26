from __future__ import annotations

import re

import pytest

from metric_sampler.cloud.summaries.catalog import DEFAULT_CHUNK_STEPS, SUMMARIES_BY_NAME
from sber_tests.unit.cloud.sql_sources import (
    ATTENTION_SUMMARIES,
    SCHEMA_SQL,
    SERVING_SQL,
    summary_columns,
    top_level_split,
    view_body,
)

pytestmark = pytest.mark.unit

# The loops that scan the per-head row statistics take chunks of 5 steps, the others the default of 10.
_CHUNK_STEPS = {"attention_softmax": 5, "attention_rank": 5}


def _range_columns(summary: str) -> list[str]:
    body = view_body(SERVING_SQL, f"{summary}_range")
    columns = body[body.index("\nSELECT\n") + len("\nSELECT\n"):].split("\nFROM", 1)[0]
    return [re.search(r"(\w+)$", column)[1] for column in top_level_split(columns)]


@pytest.mark.parametrize("summary", ATTENTION_SUMMARIES)
def test_summary_table_holds_the_columns_of_its_range_view_in_order(summary):
    assert summary_columns(summary) == _range_columns(summary)


@pytest.mark.parametrize(("summary", "key"), ATTENTION_SUMMARIES.items())
def test_summary_table_keeps_the_newest_refresh_of_each_key(summary, key):
    assert re.search(
        rf"CREATE TABLE IF NOT EXISTS metric_sampler\.summary_{summary}\n\(.*?\)\n"
        r"ENGINE = ReplacingMergeTree\(refreshed_at\)\n"
        rf"ORDER BY \({key}\)\nSETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD\(3\)';",
        SERVING_SQL, re.S,
    )


@pytest.mark.parametrize("summary", ATTENTION_SUMMARIES)
def test_route_views_read_the_summary_or_the_range_view_with_the_panels_selection(summary):
    all_view = view_body(SERVING_SQL, f"{summary}_all")
    live_view = view_body(SERVING_SQL, f"{summary}_live")

    assert " ".join(all_view.split()) == (
        f"SELECT * EXCEPT refreshed_at FROM metric_sampler.summary_{summary} FINAL "
        "WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})"
    )
    assert " ".join(live_view.split()) == (
        f"SELECT * FROM `metric_sampler.{summary}_range`(exp_filter = {{exp_filter:String}}, "
        f"run_filter = {{run_filter:String}}, "
        "rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, "
        "layer_filter = {layer_filter:Int32}, "
        "step_from = {step_from:UInt32}, step_to = {step_to:UInt32})"
    )


@pytest.mark.parametrize("summary", ATTENTION_SUMMARIES)
def test_summary_loop_fills_the_table_from_the_range_view_over_every_layer(summary):
    loop = SUMMARIES_BY_NAME[summary]

    assert loop.source == (
        f"SELECT * FROM metric_sampler.{summary}_range(exp_filter = '__EXP__', run_filter = '__RUN__', "
        "rank_filter = -2, microbatch_filter = -2, step_from = __FROM__, step_to = __TO__, layer_filter = -1) "
        "WHERE __WHERE__"
    )
    assert loop.chunk_steps == _CHUNK_STEPS.get(summary, DEFAULT_CHUNK_STEPS)


@pytest.mark.parametrize(
    ("body", "quantiles", "counts"),
    [
        (
            view_body(SCHEMA_SQL.read_text(encoding="utf-8"), "exact_quantiles_per_rank_selected"),
            "(rv.metric_id BETWEEN 114 AND 137 OR rv.metric_id = 177)",
            "(ec.metric_id BETWEEN 115 AND 139 OR ec.metric_id = 178)",
        ),
        (
            view_body(SERVING_SQL, "exact_quantiles_per_rank_step"),
            "(((rv.metric_id >= 114) AND (rv.metric_id <= 137)) OR (rv.metric_id = 177))",
            "(((ec.metric_id >= 115) AND (ec.metric_id <= 139)) OR (ec.metric_id = 178))",
        ),
    ],
)
def test_exact_quantile_views_include_the_attention_entropy_quantiles_and_their_counts(body, quantiles, counts):
    assert quantiles in body
    assert counts in body
