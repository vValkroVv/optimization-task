from __future__ import annotations

import json
import re
import sqlite3
import struct
from contextlib import closing

import pytest

from sber_tests.unit.cloud.sql_sources import INGEST_SQL, MICROBATCH, RANK, fixture_view_sql, load_dashboard

pytestmark = pytest.mark.unit

# An exact quantile vector holds the quantiles of these levels, each as a float64.
QUANTILE_LEVELS = 9
PARAMETERS = {
    "exp_filter": "e",
    "run_filter": "r",
    "rank_filter": -2,
    "microbatch_filter": -2,
}


def insert_observation(
    connection,
    family,
    metric,
    values,
    *,
    rank=0,
    microbatch=0,
    attempt="a",
    layer=2,
    node="n",
):
    connection.execute(
        f"INSERT INTO {family} (exp,run,attempt,step,node,rank,microbatch,layer,metric_id,vals) "
        "VALUES ('e','r',?,7,?,?,?,?,?,?)",
        (attempt, node, rank, microbatch, layer, metric, json.dumps(values)),
    )


def insert_name(connection, metric, name, attempt="a"):
    connection.execute(
        "INSERT INTO metric_names (exp,run,attempt,metric_id,name) VALUES ('e','r',?,?,?)",
        (attempt, metric, name),
    )


@pytest.mark.parametrize(
    "rank,microbatch", [(-2, -2), (0, -2), (-2, 1), (1, 0), (3, 0)]
)
def test_raw_scalars_keep_attempt_rank_microbatch_and_array_values(
    sql_observations, rank, microbatch
):
    connection = sql_observations
    expected = set()
    for attempt in ("a", "b"):
        insert_name(connection, 112, "loss/z_unscaled", attempt)
        for observed_rank in (0, 1):
            for observed_microbatch in (0, 1):
                values = [float(observed_rank), float(observed_microbatch), float(attempt == "b")]
                insert_observation(
                    connection,
                    "scalars",
                    112,
                    values,
                    rank=observed_rank,
                    microbatch=observed_microbatch,
                    attempt=attempt,
                )
                if rank in (-2, observed_rank) and microbatch in (-2, observed_microbatch):
                    expected.add((attempt, observed_rank, observed_microbatch, tuple(values)))
    insert_observation(connection, "scalars", 112, [999], rank=-1)
    sql = fixture_view_sql("scalar_observations_per_rank_selected").replace(
        "AS values", 'AS "values"'
    )
    rows = connection.execute(
        sql, {**PARAMETERS, "rank_filter": rank, "microbatch_filter": microbatch}
    ).fetchall()
    assert {
        (row[2], row[5], row[6], tuple(json.loads(row[-1]))) for row in rows
    } == expected


def _expert_load_sql() -> str:
    # The view in SQLite: JSON arrays stand in for ClickHouse arrays and json_each for ARRAY JOIN.
    sql = fixture_view_sql("expert_load_per_rank_selected")
    replacements = {
        "groupArray(ec.metric_id)": "json_group_array(ec.metric_id)",
        "groupArray(ec.alias_metric_id)": "json_group_array(ec.alias_metric_id)",
        "groupArray(ec.vals)": "json_group_array(json(ec.vals))",
        "microbatch, layer, metric_id,": "microbatch, layer, slot.value AS metric_id,",
        "if(alias_id = 0, row_values, rows_values[indexOf(metric_ids, alias_id)])":
            "CASE WHEN json_extract(alias_ids, '$[' || slot.key || ']') = 0 "
            "THEN json_extract(rows_values, '$[' || slot.key || ']') "
            "ELSE json_extract(rows_values, '$[' || (SELECT target.key FROM json_each(metric_ids) AS target "
            "WHERE target.value = json_extract(alias_ids, '$[' || slot.key || ']')) || ']') END",
        "ARRAY JOIN metric_ids AS metric_id, alias_ids AS alias_id, rows_values AS row_values":
            "JOIN json_each(rank_rows.metric_ids) AS slot",
        "expert_index - 1 AS expert, tokens,": "expert_slot.key AS expert, expert_slot.value AS tokens,",
        "tokens / nullIf": "expert_slot.value / nullIf",
        "length(r.assignments)": "json_array_length(r.assignments)",
        "ARRAY JOIN r.assignments AS tokens, arrayEnumerate(r.assignments) AS expert_index":
            "JOIN json_each(r.assignments) AS expert_slot",
    }
    for old, new in replacements.items():
        assert old in sql, old
        sql = sql.replace(old, new)
    return sql


@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("rank", [-2, 0, 1])
def test_expert_observations_keep_zero_experts_and_distinct_capacity_populations(
    sql_observations, rank, aliased
):
    connection = sql_observations
    names = {
        1: "router/tokens_per_expert",
        95: "dispatcher/assignments_before_capacity",
        96: "dispatcher/valid_assignments_before_capacity",
        97: "dispatcher/assignments_after_capacity",
        98: "dispatcher/valid_assignments_after_capacity",
    }
    # Stored as aliases, rows repeating their target keep no values of their own.
    aliases = {95: 1, 97: 1} if aliased else {}
    expected = set()
    for metric, name in names.items():
        insert_name(connection, metric, name)
        values = [0, 7, 1] if metric != 98 else [0, 0, 0]
        stored = [] if metric in aliases else values
        connection.execute(
            "INSERT INTO expert_counts (exp,run,attempt,step,node,rank,microbatch,layer,metric_id,vals,"
            "alias_metric_id) "
            "VALUES ('e','r','a',7,'n',0,0,2,?,?,?)",
            (metric, json.dumps(stored), aliases.get(metric, 0)),
        )
        if rank != 1:
            expected.update(
                (name, i, value, sum(values)) for i, value in enumerate(values)
            )
    rows = connection.execute(_expert_load_sql(), {**PARAMETERS, "rank_filter": rank}).fetchall()
    assert {(row[8], row[9], row[10], row[11]) for row in rows} == expected
    for row in rows:
        assert row[12] == 3
        assert row[13] == (row[10] / (row[11] / 3) if row[11] else None)


def test_exact_quantile_pairs_keep_missing_halves_empty_populations_and_float64(
    sql_observations,
):
    connection = sql_observations
    connection.create_function("if", 3, lambda condition, yes, no: yes if condition else no)
    connection.create_function(
        "replaceRegexpOne",
        3,
        lambda value, pattern, replacement: re.sub(
            pattern, replacement, value, count=1
        ),
    )
    connection.create_function(
        "decode_float64",
        1,
        lambda value: json.dumps(
            struct.unpack(f"<{QUANTILE_LEVELS}d", struct.pack(f"<{2 * QUANTILE_LEVELS}I", *json.loads(value)))
        ),
    )
    for metric, name in (
        (114, "act/residual/quantiles"),
        (115, "act/residual/quantile_counts"),
        (138, "router/entropy_quantiles"),
        (139, "router/entropy_quantile_counts"),
    ):
        insert_name(connection, metric, name)
    quantiles = [-7, -3, -2, 0, 1, 3, 4, 5, 9]
    precise = [0.1 + i * 2**-40 for i in range(QUANTILE_LEVELS)]
    for rank in (0, 1, 2, 4):
        values = [None] * QUANTILE_LEVELS if rank == 2 else quantiles[:-1] if rank == 4 else quantiles
        insert_observation(connection, "router_vectors", 114, values, rank=rank)
    for rank, counts in (
        (0, [21, 2, 3, 4, 5]),
        (2, [0, 1, 2, 3, 0]),
        (3, [9, 0, 0, 0, 0]),
        (4, [9, 0, 0, 0, 0]),
    ):
        insert_observation(connection, "expert_counts", 115, counts, rank=rank)
    bitwords = list(struct.unpack(f"<{2 * QUANTILE_LEVELS}I", struct.pack(f"<{QUANTILE_LEVELS}d", *precise)))
    insert_observation(connection, "expert_counts", 138, bitwords, rank=5)
    insert_observation(connection, "expert_counts", 139, [9, 0, 0, 0, 0], rank=5)
    sql = fixture_view_sql("exact_quantiles_per_rank_selected")
    sql = sql.replace("arrayMap(x -> toFloat64(x), rv.vals)", "rv.vals")
    sql = re.sub(
        r"arrayMap\(i -> reinterpretAsFloat64\(.*?range\(9\)\)",
        "decode_float64(ec.vals)",
        sql,
        flags=re.S,
    )
    sql = re.sub(r"length\((rv.vals|ec.vals|c.counts)\)", r"json_array_length(\1)", sql)
    sql = sql.replace(
        "[0., 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.]",
        "'[0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1]'",
    )
    connection.row_factory = sqlite3.Row
    rows = {row["rank"]: dict(row) for row in connection.execute(sql, PARAMETERS)}
    assert set(rows) == set(range(6))
    assert [rows[i]["complete_pair"] for i in range(6)] == [1, 0, 1, 0, 0, 1]
    assert json.loads(rows[0]["quantiles"]) == quantiles
    assert json.loads(rows[0]["population_counts"]) == [21, 2, 3, 4, 5]
    assert rows[1]["counts_present"] == 0 and rows[3]["quantiles_present"] == 0
    assert json.loads(rows[2]["quantiles"]) == [None] * QUANTILE_LEVELS
    assert json.loads(rows[5]["quantiles"]) == precise


def test_monitor_catalog_ingest_uses_object_identity_and_retains_names_and_rank():
    source = INGEST_SQL.read_text()
    sql = (
        source.split("INSERT INTO metric_sampler.monitor_parameters", 1)[1]
        .split(";", 1)[0]
        .split("SELECT", 1)[1]
    )
    sql = "SELECT" + sql.split("FROM s3(", 1)[0] + "FROM objects"
    sql = sql.replace("now64(3)", "'timestamp'").replace("AS index", 'AS "index"')
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.create_function(
            "extract", 2, lambda value, pattern: re.search(pattern, value).group(1)
        )
        connection.create_function(
            "JSONExtractString", 2, lambda value, key: json.loads(value).get(key, "")
        )
        connection.create_function(
            "JSONExtractUInt", 2, lambda value, key: json.loads(value).get(key, 0)
        )
        connection.create_function(
            "JSONExtractInt", 2, lambda value, key: json.loads(value).get(key, 0)
        )
        connection.create_function(
            "JSONHas", 2, lambda value, key: key in json.loads(value)
        )
        connection.create_function("if", 3, lambda condition, yes, no: yes if condition else no)
        connection.execute("CREATE TABLE objects (_path TEXT,json TEXT)")
        rows = [
            {"kind": "parameter", "index": 3, "name": "model.layers.1.weight"},
            {"kind": "activation", "rank": 2, "index": 4, "name": "model.layers.2"},
            {"kind": "optimizer_group", "rank": -1, "index": 0, "name": "embeddings"},
        ]
        for row in rows:
            connection.execute(
                "INSERT INTO objects VALUES (?,?)",
                (
                    "metrics/exp=e/run=r/attempt=a/monitor_parameters.json",
                    json.dumps({**row, "exp": "untrusted"}),
                ),
            )
        assert connection.execute(sql).fetchall() == [
            (
                "e",
                "r",
                "a",
                row["kind"],
                row.get("rank", -1),
                row["index"],
                row["name"],
                "metrics/exp=e/run=r/attempt=a/monitor_parameters.json",
                "timestamp",
            )
            for row in rows
        ]


def test_rank_panels_do_not_aggregate_limit_or_hide_incomplete_observations():
    dashboard = load_dashboard("node-rank.json")
    titles = (
        "Exact quantiles per rank and microbatch",
        "Ordinary monitor inputs per rank",
        "Expert assignments per rank and microbatch",
        "Raw scalar observations per rank",
    )
    panels = [panel for panel in dashboard["panels"] if panel["title"].startswith(titles)]
    assert len(panels) == 4
    for panel in panels:
        sql = panel["targets"][0]["rawSql"]
        assert panel["type"] == "table"
        assert "attempt, step, node, rank, microbatch, layer" in sql
        assert "WHERE step = toUInt32OrZero(${step:sqlstring})" in sql
        # With Rank = All a table shows one rank at a time (table_rank); every rank stays selectable.
        assert (
            f"rank_filter = if({RANK} = -2, toInt32OrZero(${{table_rank:sqlstring}}), {RANK}), "
            f"microbatch_filter = {MICROBATCH}"
        ) in sql
        assert not re.search(r"\b(GROUP BY|LIMIT|HAVING|avg|sum)\b", sql, re.I)
    quantile = next(panel for panel in panels if panel["title"].startswith("Exact"))
    assert "complete_pair" in quantile["targets"][0]["rawSql"].split(" FROM ")[0]
