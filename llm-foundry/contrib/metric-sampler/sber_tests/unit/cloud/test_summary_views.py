from __future__ import annotations

import re

import pytest

from metric_sampler.cloud.summaries.catalog import SUMMARIES, SUMMARIES_BY_NAME
from sber_tests.unit.cloud.sql_sources import SERVING_SQL, load_dashboard, serving_view, view_body

pytestmark = pytest.mark.unit

_HISTOGRAM_VIEWS = [
    f"activation_{family}_{split}_range"
    for family, splits in (
        ("exp_quantiles", ("selected", "by_mb", "by_rank", "by_rank_mb", "levels")),
        ("massive", ("tokens_selected", "by_mb", "by_rank", "by_rank_mb", "levels")),
    )
    for split in splits
]
# Derived from the per rank and microbatch rows: view name and the summary loop that fills its table.
_DERIVED_VIEWS = {
    "activation_by_rank_range": "activation_rank",
    "activation_by_mb_range": "activation_mb",
    "activation_selected_range": "activation",
}


@pytest.mark.parametrize("view", _HISTOGRAM_VIEWS)
def test_histogram_views_add_the_exact_and_legacy_bins_pairwise_instead_of_indexing_arrays_in_a_lambda(view) -> None:
    sql = serving_view(view)

    assert (
        "arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), "
        "arrayResize(legacy_hist, bins_count)) AS hist"
    ) in sql
    assert "exact_hist[i]" not in sql


def test_histogram_source_looks_for_exact_counts_of_legacy_rows_only_in_steps_that_have_legacy_rows() -> None:
    sql = serving_view("activation_histograms_selected_range")

    legacy_exclusion = sql[sql.index("NOT IN ("):]
    assert re.search(
        r"FROM exact_counts\s+WHERE step IN \(\s+SELECT step\s+FROM metric_sampler\.router_vectors", legacy_exclusion
    )
    assert sql.count("FROM metric_sampler.expert_counts") == 1


def test_rank_microbatch_activation_view_reads_each_raw_table_once_and_aggregates_the_observations_once() -> None:
    sql = serving_view("activation_by_rank_mb_range")

    for table in ("router_vectors", "expert_counts", "scalars"):
        assert sql.count(f"FROM metric_sampler.{table} AS") == 1
    assert sql.count("GROUP BY") == 1
    assert "LEFT JOIN" not in sql


@pytest.mark.parametrize("view", _DERIVED_VIEWS)
def test_derived_activation_views_sum_the_finished_rank_microbatch_rows_and_read_only_moments_of_the_raw_rows(
    view,
) -> None:
    sql = serving_view(view)

    assert re.search(r"FROM metric_sampler\.summary_activation_rank_mb\s+FINAL", sql)
    assert "metric_sampler.expert_counts" not in sql
    assert "metric_sampler.scalars" not in sql
    assert re.findall(r"match\(name, '([^']*)'\)", sql) == ["^act/[^/]+/(moments|power_sums)$"] * 2


@pytest.mark.parametrize("view", _DERIVED_VIEWS)
def test_derived_activation_views_are_created_after_the_summary_table_they_read(view) -> None:

    table = SERVING_SQL.index("CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_rank_mb\n")
    assert table < SERVING_SQL.index(f"CREATE OR REPLACE VIEW metric_sampler.{view}\n")


def test_derived_activation_summaries_wait_for_the_rank_microbatch_summary_and_it_waits_for_nothing() -> None:
    for loop in _DERIVED_VIEWS.values():
        assert SUMMARIES_BY_NAME[loop].after == "activation_rank_mb"
    assert SUMMARIES_BY_NAME["activation_rank_mb"].after is None
    assert SUMMARIES_BY_NAME["activation_rank_mb"].chunk_steps == 3


def test_no_summary_loop_waits_for_a_summary_that_waits_for_another() -> None:
    waiting = {summary.name: summary.after for summary in SUMMARIES if summary.after}

    assert waiting
    assert set(waiting.values()).isdisjoint(waiting)


@pytest.mark.parametrize("view", ["activation_per_rank_peak", "activation_per_rank_mbpeak"])
def test_card_peak_views_scale_each_cards_largest_deviation_by_the_median_of_its_layer_and_point(view) -> None:
    sql = serving_view(view)

    assert "1.4826 * medianExact(abs(log_ratio)) AS scale" in sql
    assert "max(abs(log_ratio)) AS peak" in sql
    assert "max(peaks.peak / nullIf(run_scale.scale, 0)) AS max_abs_z" in sql
    assert sql.count("FINAL") == 2


def test_the_microbatch_card_peak_view_reads_one_microbatch_of_the_amax_summary() -> None:
    sql = serving_view("activation_per_rank_mbpeak")

    assert sql.count("FROM metric_sampler.summary_activation_amax_mb") == 2
    assert sql.count("(microbatch = {microbatch_filter:Int32})") == 2


@pytest.mark.parametrize("family", ["grad_per_rank", "timing_per_rank"])
def test_per_rank_run_views_pack_the_runs_rows_next_to_the_median_that_scales_them(family) -> None:
    sql = serving_view(f"{family}_run")

    assert "(exp = {exp_filter:String}) AND (run = {run_filter:String})" in sql
    assert sql.count("groupArrayIf((step, node, rank,") == 1
    assert sql.count("step >= {step_from:UInt32}) AS observations") == 1
    assert sql.count("1.4826 * medianExact(abs(log_ratio)) AS scale") == 1
    assert "ARRAY JOIN" in sql
    # A median window over a whole run is quadratic.
    assert not re.search(r"medianExact\(abs\(log_ratio\)\) OVER", sql)


@pytest.mark.parametrize("family", ["grad_per_rank", "timing_per_rank"])
def test_per_rank_run_views_read_the_raw_scalars_from_the_chunk_on_and_the_finished_steps_from_their_summary(
    family,
) -> None:
    sql = serving_view(f"{family}_run")

    assert sql.count("FROM metric_sampler.scalars AS s") == sql.count("(s.step >= {step_from:UInt32})") > 0
    assert re.search(
        rf"FROM metric_sampler\.summary_{family}\s+FINAL\s+"
        r"WHERE \(exp = \{exp_filter:String\}\) AND \(run = \{run_filter:String\}\) "
        r"AND \(step < \{step_from:UInt32\}\)",
        sql,
    )


@pytest.mark.parametrize("family", ["grad_per_rank", "timing_per_rank"])
def test_per_rank_summary_loops_read_the_run_views_with_the_run_and_first_step_of_the_chunk(family) -> None:
    assert SUMMARIES_BY_NAME[family].source == (
        f"SELECT * FROM metric_sampler.{family}_run(exp_filter = '__EXP__', run_filter = '__RUN__', "
        "step_from = __FROM__)"
        " WHERE __WHERE__"
    )


def _create_table_columns(table: str) -> list[str]:
    body = re.search(rf"CREATE TABLE IF NOT EXISTS metric_sampler\.{table}\n\((.*?)\n\)\n", SERVING_SQL, re.S)[1]
    return [re.match(r"\s*`(\w+)`", line)[1] for line in body.strip().split("\n")]


def test_layer_load_rank_summary_holds_the_columns_of_its_range_view_and_keeps_the_newest_refresh_of_a_layer_of_a_step(
) -> None:
    view = serving_view("layer_load_rank_range")
    selected = view[view.index("AS SELECT") + len("AS SELECT"):view.index("\nFROM\n")]

    assert (
        re.findall(r"^    (?:.* AS )?(\w+),?$", selected, re.M) == _create_table_columns("summary_layer_load_rank")[1:]
    )
    assert (
        "ENGINE = ReplacingMergeTree(refreshed_at)\nORDER BY (exp, run, rank, step, layer)\n"
        "SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';"
    ) in SERVING_SQL[SERVING_SQL.index("CREATE TABLE IF NOT EXISTS metric_sampler.summary_layer_load_rank\n"):]


def test_layer_load_rank_range_view_computes_the_loads_of_the_per_rank_view_for_every_rank_of_its_steps() -> None:
    per_rank = serving_view("layer_load_rank")
    summarized = serving_view("layer_load_rank_range")

    for expression in (
        "arrayMax(router_tokens) / (arraySum(router_tokens) / length(router_tokens)) AS max_load",
        "toUInt32(arrayCount(x -> (x = 0), router_tokens)) AS dead_experts",
        "min(r.metric_id) OVER (PARTITION BY r.rank, r.step, r.layer) AS preferred",
    ):
        assert expression in summarized
    assert "min(r.metric_id) OVER (PARTITION BY r.step, r.layer) AS preferred" in per_rank
    assert "(r.rank = {rank_filter:Int32})" in per_rank
    assert "rank_filter" not in summarized


def test_layer_load_rank_range_view_reads_the_raw_counts_that_summary_router_rank_copies() -> None:
    summarized = serving_view("layer_load_rank_range")

    assert re.search(r"FROM metric_sampler\.expert_counts AS r\s+FINAL", summarized)
    assert "(r.layer <= 255)" in summarized
    assert "(r.metric_id IN (1, 2))" in summarized
    assert "summary_router_rank" not in summarized[summarized.index("CREATE OR REPLACE VIEW"):]


def test_layer_load_ranksum_reads_one_rank_of_the_summary() -> None:
    assert " ".join(serving_view("layer_load_ranksum").split()) == (
        "CREATE OR REPLACE VIEW metric_sampler.layer_load_ranksum AS SELECT * EXCEPT (refreshed_at, rank) "
        "FROM metric_sampler.summary_layer_load_rank FINAL "
        "WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32})"
    )


def test_layer_load_rank_summary_loop_fills_the_table_from_the_range_view_of_the_chunk_s_run() -> None:
    assert SUMMARIES_BY_NAME["layer_load_rank"].source == (
        "SELECT * FROM metric_sampler.layer_load_rank_range(exp_filter = '__EXP__', run_filter = '__RUN__', "
        "step_from = __FROM__, step_to = __TO__) WHERE __WHERE__"
    )


def test_experts_layer_load_panels_take_the_per_rank_summary_for_one_rank_over_all_microbatches() -> None:
    dashboard = load_dashboard("experts.json")
    variables = {variable["name"]: variable for variable in dashboard["templating"]["list"]}
    panels = [*dashboard["panels"], *(inner for panel in dashboard["panels"] for inner in panel.get("panels", []))]
    sources = {panel["id"]: [target["rawSql"] for target in panel.get("targets", [])] for panel in panels}

    query = variables["llsrc"]["query"]
    assert query.startswith(
        "SELECT if(ifNull(toInt32OrNull(${rank:sqlstring}), -3) != -2 "
        "AND ifNull(toInt32OrNull(${microbatch:sqlstring}), -3) = -2 "
        "AND (SELECT count() = 1 AND min(done) + 50 >= (SELECT max(step)"
    )
    assert "summary IN ('layer_load_rank') GROUP BY summary" in query
    assert query.endswith(", 'ranksum', '${src}') AS llsrc")
    assert variables["llsrc"]["hide"] == 2
    for panel_id in (1, 2, 8):
        assert all(
            "metric_sampler.layer_load_${llsrc}(" in sql and "layer_load_${src}" not in sql for sql in sources[panel_id]
        )
    assert all("metric_sampler.expert_totals_${src}(" in sql for panel_id in (3, 7) for sql in sources[panel_id])



@pytest.mark.parametrize("view", ["expert_totals_selected_range", "expert_totals_by_mb_range"])
def test_expert_totals_range_views_sum_the_router_vectors_before_they_expand_them_to_experts(view) -> None:
    sql = serving_view(view)

    assert sql.rindex("GROUP BY") < sql.index("ARRAY JOIN")
    assert "sumForEach(ec.`values`) AS token_counts" in sql
    assert "sum(token_count)" not in sql


def test_module_activation_joins_counts_of_the_latest_attempt_on_the_step_layer_rank_microbatch_and_node() -> None:
    sql = serving_view("module_activation_range")

    assert (
        "INNER JOIN counts ON (counts.step = stats.step) AND (counts.layer = stats.layer) "
        "AND (counts.rank = stats.rank) "
        "AND (counts.microbatch = stats.microbatch) AND (counts.node = stats.node)"
    ) in sql
    # Without the attempt in the join keys, the counts of an attempt that a restart replaced must not reach the join.
    assert "((ec.step, ec.attempt) IN (" in sql
    assert "FROM metric_sampler.latest_attempt_step" in sql


_LEVEL_KEYS = (
    "            exp,\n            run,\n            step,\n            layer,\n            point,\n"
    "            rank,\n            microbatch\n"
)
_GROUPING_SETS = (
    "            GROUPING SETS (\n"
    "                (exp, run, step, layer, point, rank, microbatch),\n"
    "                (exp, run, step, layer, point, rank),\n"
    "                (exp, run, step, layer, point, microbatch),\n"
    "                (exp, run, step, layer, point))\n"
)
# grouping(rank, microbatch) of each summary's level and the columns its rows leave out.
_LEVELS = [
    ("_rank_mb", 0, "level"), ("_rank", 1, "level, microbatch"),
    ("_mb", 2, "level, rank"), ("", 3, "level, rank, microbatch"),
]


@pytest.mark.parametrize("family", ["exp_quantiles", "massive"])
def test_levels_view_is_the_rank_and_microbatch_view_grouped_by_the_four_levels(family) -> None:
    levels = view_body(SERVING_SQL, f"activation_{family}_levels_range")

    assert levels.count(_GROUPING_SETS) == 1
    restored = levels.replace(_GROUPING_SETS, _LEVEL_KEYS)
    for line in ("            grouping(rank, microbatch) AS level,\n", "            level,\n", "    level,\n"):
        restored = restored.replace(line, "")
    assert restored == view_body(SERVING_SQL, f"activation_{family}_by_rank_mb_range")


@pytest.mark.parametrize("family", ["exp_quantiles", "massive"])
@pytest.mark.parametrize(("suffix", "level", "left_out"), _LEVELS)
def test_levels_table_views_route_each_level_to_its_summary_table(family, suffix, level, left_out) -> None:
    target = f"summary_activation_{family}{suffix}"

    view = SERVING_SQL.split(
        f"CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.{target}_mv TO metric_sampler.{target}\n", 1
    )[1]
    view = view.split(";\n", 1)[0]
    assert view.endswith(
        f"AS SELECT * EXCEPT ({left_out})\nFROM metric_sampler.summary_activation_{family}_levels\n"
        f"WHERE level = {level}"
    )
    levels_columns = _create_table_columns(f"summary_activation_{family}_levels")
    assert [column for column in levels_columns if column not in left_out.split(", ")] == _create_table_columns(target)


@pytest.mark.parametrize("family", ["exp_quantiles", "massive"])
def test_levels_table_takes_the_levels_view_columns_and_keeps_no_rows(family) -> None:
    table = SERVING_SQL.split(f"CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_{family}_levels\n", 1)[1]
    levels = view_body(SERVING_SQL, f"activation_{family}_levels_range")
    final = levels.rsplit("\nSELECT\n", 1)[1].split("\nFROM ", 1)[0]

    assert table.split(";\n", 1)[0].endswith("\n)\nENGINE = Null")
    assert _create_table_columns(f"summary_activation_{family}_levels")[1:] == re.findall(
        r"^    (?:.* AS )?(\w+),?$", final, re.M
    )
