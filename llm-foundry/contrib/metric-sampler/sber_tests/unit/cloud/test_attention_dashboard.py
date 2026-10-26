from __future__ import annotations

import json
import re

import pytest

from metric_sampler.cloud.summaries.catalog import SUMMARIES_BY_NAME
from sber_tests.unit.cloud.sql_sources import (
    ATTENTION_SUMMARIES,
    DASHBOARDS_DIR,
    MICROBATCH,
    RANK,
    SERVING_SQL,
    STEP_RANGE,
    load_dashboard,
    summary_columns,
)

pytestmark = pytest.mark.unit

_DASHBOARD = load_dashboard("attention.json")
_PANELS = [panel for panel in _DASHBOARD["panels"] if panel["type"] != "row"]
_VARIABLES = {variable["name"]: variable for variable in _DASHBOARD["templating"]["list"]}


def test_dashboard_variables_select_the_run_the_kind_the_layer_the_head_and_the_ranks():
    assert list(_VARIABLES) == [
        "exp", "run", "capture_interval", "kind", "layer", "head", "rank", "microbatch", "asrc", "rsrc", "latest_step",
        "step_from", "step_to", "remember",
    ]
    assert _VARIABLES["kind"]["query"] == "softmax : softmax, linear : linear"
    for name in ("rank", "microbatch"):
        assert _VARIABLES[name]["current"] == {"text": "All (aggregate)", "value": "-2"}
        assert f"SELECT DISTINCT {name} AS v FROM metric_sampler.scalars" in _VARIABLES[name]["query"]
    assert [name for name, variable in _VARIABLES.items() if variable.get("hide") == 2] == [
        "capture_interval", "asrc", "rsrc", "latest_step", "step_from", "step_to", "remember",
    ]
    for name in ("layer", "head"):
        assert (
            "if('${kind}' = 'softmax', 'attn/softmax/lse_stats', 'attn/linear/norm_stats')" in _VARIABLES[name]["query"]
        )


@pytest.mark.parametrize(
    ("variable", "summaries", "selection"),
    [
        ("asrc", ["attention_softmax", "attention_linear", "attention_entropy_hist"],
         f"{RANK} = -2 AND {MICROBATCH} = -2"),
        ("rsrc", ["attention_rank"], f"{MICROBATCH} = -2"),
    ],
)
def test_summary_routes_are_taken_only_for_the_whole_run_and_a_covered_selection(variable, summaries, selection):
    query = _VARIABLES[variable]["query"]

    assert query.startswith(
        f"SELECT if({selection} AND (SELECT count() = {len(summaries)} AND min(done) + 50 >= (SELECT max(step)"
    )
    assert f"summary IN ({', '.join(repr(summary) for summary in summaries)}) GROUP BY summary" in query
    assert query.endswith(f", 'all', 'live') AS {variable}")
    assert all(summary in SUMMARIES_BY_NAME for summary in summaries)


def test_latest_step_is_the_newest_complete_step_of_the_chosen_route():
    query = _VARIABLES["latest_step"]["query"]

    limit = (
        "step <= (SELECT step_limit FROM metric_sampler.latest_step_limit("
        "exp_filter = '${exp}', run_filter = '${run}'))"
    )
    assert query == (
        "SELECT if('${asrc}' = 'all', greatest("
        "(SELECT max(step) FROM metric_sampler.summary_attention_softmax "
        f"WHERE exp = '${{exp}}' AND run = '${{run}}' AND head = -1 AND {limit}), "
        "(SELECT max(step) FROM metric_sampler.summary_attention_linear "
        f"WHERE exp = '${{exp}}' AND run = '${{run}}' AND head = -1 AND {limit})), "
        f"(SELECT max(step) FROM metric_sampler.step_index WHERE exp = '${{exp}}' AND run = '${{run}}' AND {limit}))"
    )


def _panel_sql(panel: dict) -> str:
    return panel["targets"][0]["rawSql"]


def _route_views(sql: str) -> list[tuple[str, str]]:
    return re.findall(r"metric_sampler\.attention_(\w+?)_\$\{(\w+)\}\(", sql)


_ROUTED = [panel for panel in _PANELS if _route_views(_panel_sql(panel))]
_ROUTED_IDS = [panel["title"][:70] for panel in _ROUTED]


@pytest.mark.parametrize("panel", _ROUTED, ids=_ROUTED_IDS)
def test_panels_call_route_views_that_serving_sql_defines_for_both_routes(panel):
    for summary, route in _route_views(_panel_sql(panel)):
        assert route == ("rsrc" if summary == "rank" else "asrc")
        for suffix in ("all", "live"):
            assert f"CREATE OR REPLACE VIEW metric_sampler.attention_{summary}_{suffix}\n" in SERVING_SQL


@pytest.mark.parametrize("panel", _ROUTED, ids=_ROUTED_IDS)
def test_panels_select_only_columns_of_the_views_they_read(panel):
    sql = re.sub(r"\$\{[^}]*\}", "", _panel_sql(panel))
    read = {f"attention_{summary}" for summary, _ in _route_views(_panel_sql(panel))}
    allowed = {column for summary in read for column in summary_columns(summary)}
    foreign = {column for summary in ATTENTION_SUMMARIES for column in summary_columns(summary)} - allowed

    assert foreign.isdisjoint(re.findall(r"\b\w+\b", sql))


@pytest.mark.parametrize("panel", _ROUTED, ids=_ROUTED_IDS)
def test_panels_pass_the_chosen_layer_or_every_layer_to_the_route_views(panel):
    layers = set(re.findall(r"layer_filter = (-1|toInt32OrZero\(\$\{layer:sqlstring\}\))", _panel_sql(panel)))

    assert layers == ({"toInt32OrZero(${layer:sqlstring})"} if "${layer}" in panel["title"] else {"-1"})


def test_the_quantile_panel_reads_the_exact_quantiles_of_the_selected_populations():
    sql = next(
        _panel_sql(panel) for panel in _PANELS if panel["title"].startswith("Normalized attention entropy quantiles")
    )

    assert "mn.name = 'attn/softmax/entropy_quantiles'" in sql
    assert f"({RANK} = -2 OR rv.rank = {RANK}) AND ({MICROBATCH} = -2 OR rv.microbatch = {MICROBATCH})" in sql
    assert "AND length(rv.`values`) = 9 AND isFinite(rv.`values`[5])" in sql
    for index, name in [(3, "p05"), (4, "p25"), (5, "p50"), (6, "p75"), (7, "p95")]:
        assert f"quantileExact(0.5)(rv.`values`[{index}]) AS {name}" in sql


def test_the_per_rank_panel_unpacks_the_head_of_the_latest_step_from_the_raw_tables():
    sql = next(_panel_sql(panel) for panel in _PANELS if panel["title"].startswith("Per-rank statistics"))

    assert "FROM metric_sampler.attention_heads_per_rank_selected(" in sql
    assert f"rank_filter = if({RANK} = -2, 0, {RANK}), microbatch_filter = {MICROBATCH}" in sql
    assert "ARRAY JOIN stat_names AS stat, stats AS value WHERE step = toUInt32OrZero(${latest_step:sqlstring})" in sql


@pytest.mark.parametrize(
    "panel", [panel for panel in _PANELS if panel["type"] == "heatmap"], ids=lambda panel: panel["title"][:70]
)
def test_heatmaps_use_numeric_steps_and_label_each_cell_row(panel):
    sql, column = _panel_sql(panel), panel["transformations"][0]["options"]["columnField"]

    assert panel["targets"][0]["format"] == 1
    assert sql.startswith("SELECT step,")
    assert sql.endswith(
        f"ORDER BY step, {column} SETTINGS join_algorithm = 'hash', optimize_move_to_prewhere_if_final = 1"
    )
    assert STEP_RANGE in sql
    assert panel["transformations"] == [
        {"id": "groupingToMatrix",
         "options": {"rowField": "step", "columnField": column, "valueField": "value", "emptyValue": "null"}},
        {"id": "convertFieldType",
         "options": {"fields": {}, "conversions": [{"targetField": f"step\\{column}", "destinationType": "number"}]}},
    ]


def test_panel_layout_has_unique_ids_inside_the_grid_and_sections_without_nested_panels():
    ids = [panel["id"] for panel in _DASHBOARD["panels"]]
    assert len(set(ids)) == len(ids)
    assert all(panel["gridPos"]["x"] + panel["gridPos"]["w"] <= 24 for panel in _DASHBOARD["panels"])
    assert [panel["title"] for panel in _DASHBOARD["panels"] if panel["type"] == "row"] == [
        "Overview: softmax and linear layers", "Softmax attention: per layer", "Softmax attention: selected layer",
        "Softmax attention: ranks", "Linear attention",
    ]
    assert all(panel["collapsed"] is False and panel["panels"] == []
               for panel in _DASHBOARD["panels"] if panel["type"] == "row")


def test_dashboard_is_listed_in_the_navigation_of_every_dashboard():
    # The link bar of each dashboard is a dropdown of the dashboards with this tag.
    dashboards = [json.loads(path.read_text(encoding="utf-8")) for path in DASHBOARDS_DIR.glob("*.json")]

    assert _DASHBOARD in dashboards
    assert all(dashboard["tags"] == ["metric-sampler"] for dashboard in dashboards)
    assert all(dashboard["links"][0]["tags"] == ["metric-sampler"] for dashboard in dashboards)
    assert _DASHBOARD["uid"] == "ms-attention"
