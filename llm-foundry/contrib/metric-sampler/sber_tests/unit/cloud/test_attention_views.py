from __future__ import annotations

import json
import math
import re
import sqlite3

import pytest

from sber_tests.unit.cloud.sql_sources import SCHEMA_SQL, SERVING_SQL, top_level_split, view_body

pytestmark = pytest.mark.unit

_LSE, _QKV, _GATE, _ROW = (f"attn/softmax/{name}" for name in ("lse_stats", "qkv_stats", "gate_stats", "row_stats"))
_BETA, _NORM, _STATE, _DECAY = (
    f"attn/linear/{name}" for name in ("gate_stats", "norm_stats", "state_stats", "decay_params")
)
_HIST, _HEAD_HIST = "attn/softmax/entropy_hist", "attn/softmax/head_entropy_hist"

# The vector contract: after the row count the fields follow stat-major, one block of heads each. Every field is
# merged over ranks and microbatches by its own reduction: the sums add up and are divided by the rows, the rest take
# the minimum or the maximum. Each field is (name of the field, reduction, column of the range view or None).
_FIELDS = {
    _LSE: [("lse_sum", "sum", "lse_mean"), ("lse_min", "min", "lse_min"), ("lse_max", "max", "lse_max")],
    _QKV: [
        ("q_nope_norm_sum", "sum", "q_nope_norm_mean"), ("q_rope_norm_sum", "sum", "q_rope_norm_mean"),
        ("k_nope_norm_sum", "sum", "k_nope_norm_mean"), ("k_rope_norm_sum", "sum", "k_rope_norm_mean"),
        ("v_norm_sum", "sum", "v_norm_mean"), ("q_amax", "max", "q_amax"),
        ("k_amax", "max", "k_amax"), ("v_amax", "max", "v_amax"),
    ],
    _GATE: [("gate_sum", "sum", "gate_mean"), ("gate_min", "min", "gate_min"), ("gate_max", "max", "gate_max")],
    _ROW: [
        ("ent_sum", "sum", "ent_mean"), ("ent_sumsq", "sum", None), ("ent_min", "min", "ent_min"),
        ("ent_max", "max", "ent_max"),
        ("entn_sum", "sum", "entn_mean"), ("entn_min", "min", "entn_min"), ("entn_max", "max", "entn_max"),
        ("pmax_sum", "sum", "pmax_mean"), ("pmax_min", "min", "pmax_min"), ("pmax_max", "max", "pmax_max"),
        ("pmin_sum", "sum", "pmin_mean"), ("lpmin_sum", "sum", "lpmin_mean"), ("lpmin_min", "min", "lpmin_min"),
        ("lpmin_max", "max", "lpmin_max"), ("mlogit_sum", "sum", "mlogit_mean"), ("mlogit_max", "max", "mlogit_max"),
        ("coll_sum", "sum", "coll_mean"), ("sink_sum", "sum", "sink_mean"), ("sink_max", "max", "sink_max"),
        ("dist_sum", "sum", "dist_mean"), ("local_sum", "sum", "local_mean"),
    ],
    _BETA: [
        ("beta_sum", "sum", "beta_mean"), ("beta_min", "min", "beta_min"), ("beta_max", "max", "beta_max"),
        ("alpha_sum", "sum", "alpha_mean"), ("alpha_min", "min", "alpha_min"), ("alpha_max", "max", "alpha_max"),
        ("g_sum", "sum", "g_mean"),
    ],
    _NORM: [("v_norm_sum", "sum", "v_norm_mean"), ("o_norm_sum", "sum", "o_norm_mean"), ("o_amax", "max", "o_amax")],
    _STATE: [("state_norm_sum", "sum", "state_norm_mean"), ("state_norm_max", "max", "state_norm_max")],
}
_SOFTMAX_METRICS = (_LSE, _QKV, _GATE, _ROW)
_LINEAR_METRICS = (_BETA, _NORM, _STATE)
_IDS = {name: 171 + index for index, name in enumerate(
    (_LSE, _QKV, _GATE, _ROW, _HIST, _HEAD_HIST,
     "attn/softmax/entropy_quantiles", "attn/softmax/entropy_quantile_counts",
     _BETA, _NORM, _STATE, _DECAY))}

_SOFTMAX_HEADS, _LINEAR_HEADS = 2, 3
# (rank, microbatch, counted rows, documents): a row stat is counted per query row, a state statistic per document.
_ORIGINS = [(0, 0, 10, 2), (1, 0, 30, 3), (0, 1, 20, 1)]


# The views are ClickHouse SQL. They run here on sqlite after a rewrite of the few constructs it lacks: arrays are JSON
# texts, ARRAY JOIN is a join with the numbers, a select alias used by a later item of the same select is inlined,
# GROUPING SETS is two selects, and the ClickHouse functions are small python functions.
def _matching_paren(sql: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(sql)):
        depth += (sql[index] in "([") - (sql[index] in ")]")
        if depth == 0:
            return index
    raise ValueError("unbalanced parentheses")


def _rewrite_calls(sql: str, name: str, build) -> str:
    while match := re.search(rf"\b{name}\(", sql):
        closing = _matching_paren(sql, match.end() - 1)
        sql = sql[:match.start()] + build(top_level_split(sql[match.end():closing])) + sql[closing + 1:]
    return sql


def _substitute_aliases(expression: str, aliases: dict[str, str]) -> str:
    for name, value in aliases.items():
        expression = re.sub(rf"(?<![\w.]){name}(?![\w(])", f"({value})", expression)
    return expression


def _inline_aliases(items: list[str]) -> tuple[list[str], dict[str, str]]:
    aliases: dict[str, str] = {}
    inlined = []
    for item in items:
        alias = re.search(r"\s+AS\s+(\w+)$", item)
        expression = _substitute_aliases(item[:alias.start()] if alias else item, aliases)
        inlined.append(f"{expression} AS {alias[1]}" if alias else expression)
        if alias and not re.fullmatch(r"\w+\.\w+", expression):
            aliases[alias[1]] = expression
    return inlined, aliases


def _split_grouping_sets(sql: str) -> str:
    sets = re.search(r"GROUP BY\s+GROUPING SETS \(\s*\(([^)]*)\),\s*\(([^)]*)\)\)", sql)
    final = sql[sql.rindex("\nSELECT\n") + 1:sets.start()]
    marker = "if(grouping(h) = 1, toInt16(-1), toInt16(h)) AS head"
    return (sql[:sql.rindex("\nSELECT\n") + 1] + final.replace(marker, "h AS head") + f"GROUP BY {sets[1]}\nUNION ALL\n"
            + final.replace(marker, "-1 AS head") + f"GROUP BY {sets[2]}")


def _inline_select_aliases_and_array_joins(sql: str) -> str:
    tokens = re.compile(r"\bSELECT\b|ARRAY JOIN range\((.*)\) AS (\w+)")
    position, result, aliases = 0, "", {}
    while match := tokens.search(sql, position):
        if match[0] == "SELECT":
            end = match.end()
            depth = 0
            while depth > 0 or not re.match(r"\s+FROM\b", sql[end:]):
                depth += (sql[end] in "([") - (sql[end] in ")]")
                end += 1
            items, aliases = _inline_aliases(top_level_split(sql[match.end():end]))
            result += sql[position:match.end()] + " " + ", ".join(items)
            position = end
        else:
            result += (sql[position:match.start()] + f"JOIN (SELECT n AS {match[2]} FROM numbers) AS arr "
                       f"ON arr.{match[2]} < ({_substitute_aliases(match[1], aliases)})")
            position = match.end()
    return result + sql[position:]


def _array_map(args: list[str]) -> str:
    parameter, body = (part.strip() for part in args[0].split("->"))
    count = re.fullmatch(r"range\((.*)\)", args[1], re.S)[1]
    return (f"(SELECT json_group_array(v) FROM (SELECT {body} AS v FROM "
            f"(SELECT n AS {parameter} FROM numbers WHERE n < {count} ORDER BY n)))")


def _rewrite_indexes(sql: str) -> str:
    while match := re.search(r"\b(\w+(?:\.\w+)?)\[", sql):
        closing = _matching_paren(sql, match.end() - 1)
        index = sql[match.end():closing]
        # An index beyond the end of a ClickHouse array reads the default of the element type.
        element = f"COALESCE(json_extract({match[1]}, '$[' || (({index}) - 1) || ']'), 0)"
        sql = sql[:match.start()] + element + sql[closing + 1:]
    return sql


def _emulated(sql: str) -> str:
    sql = re.sub(r"\s+FINAL\b", "", sql.replace("metric_sampler.", ""))
    sql = re.sub(r"\.`values`|\.values\b", ".vals", sql).replace("`values`", "vals")
    sql = re.sub(r"\{(\w+):\w+\}", r":\1", sql)
    if "GROUPING SETS" in sql:
        sql = _split_grouping_sets(sql)
    sql = _inline_select_aliases_and_array_joins(sql)
    sql = _rewrite_calls(sql, "arrayMap", _array_map)
    sql = _rewrite_calls(sql, "length", lambda args: f"json_array_length({args[0]})")
    sql = _rewrite_calls(sql, "intDiv", lambda args: f"(({args[0]}) / ({args[1]}))")
    sql = _rewrite_calls(sql, "greatest", lambda args: f"max({', '.join(args)})")
    sql = _rewrite_calls(
        sql, "multiIf",
        lambda args: "CASE " + "".join(f"WHEN {args[i]} THEN {args[i + 1]} " for i in range(0, len(args) - 1, 2))
        + f"ELSE {args[-1]} END",
    )
    sql = _rewrite_calls(sql, "if", lambda args: f"CASE WHEN {args[0]} THEN {args[1]} ELSE {args[2]} END")
    for function, sql_type in (("toUInt64", "INTEGER"), ("toUInt32", "INTEGER"),
                               ("toInt64", "INTEGER"), ("toInt16", "INTEGER"),
                               ("toFloat64", "REAL")):
        sql = _rewrite_calls(sql, function, lambda args, sql_type=sql_type: f"CAST(({args[0]}) AS {sql_type})")
    sql = re.sub(r"\[('[^'\]]*'(?:\s*,\s*'[^'\]]*')*)\]", lambda match: "'[" + match[1].replace("'", '"') + "]'", sql)
    return _rewrite_indexes(sql)


class _ForEach:
    """sumForEachIf and its siblings: the element-wise reduction of the arrays of the rows that meet a condition."""

    reduce = None

    def __init__(self) -> None:
        self.total: list[float] | None = None

    def step(self, values: str, condition: int = 1) -> None:
        if condition:
            array = json.loads(values)
            self.total = (
                array if self.total is None else [self.reduce(a, b) for a, b in zip(self.total, array, strict=True)]
            )

    def finalize(self) -> str:
        return json.dumps(self.total or [])


def _for_each(reduce):
    return type("ForEach", (_ForEach,), {"reduce": staticmethod(reduce)})


class _AnyIf:
    def __init__(self) -> None:
        self.value = None

    def step(self, value: str, condition: int = 1) -> None:
        if condition and self.value is None:
            self.value = value

    def finalize(self) -> str:
        return self.value or "[]"


class _Any:
    def __init__(self) -> None:
        self.value = None

    def step(self, value) -> None:
        if self.value is None:
            self.value = value

    def finalize(self):
        return self.value


def _element_or_null(array: str, index: int):
    values = json.loads(array)
    return values[index - 1] if 1 <= index <= len(values) else None


@pytest.fixture
def clickhouse_tables():
    """The raw tables and registry of the views, an empty sqlite stand-in for each."""
    connection = sqlite3.connect(":memory:")
    for name, reduce in (("sumForEachIf", lambda a, b: a + b), ("minForEachIf", min), ("maxForEachIf", max)):
        connection.create_aggregate(name, 2, _for_each(reduce))
    connection.create_aggregate("sumForEach", 1, _for_each(lambda a, b: a + b))
    connection.create_aggregate("anyIf", 2, _AnyIf)
    connection.create_aggregate("any", 1, _Any)
    connection.create_function("arrayElementOrNull", 2, _element_or_null)
    connection.create_function(
        "arraySlice", 3, lambda array, offset, length: json.dumps(json.loads(array)[offset - 1:offset - 1 + length])
    )
    connection.create_function("arraySum", 1, lambda array: sum(json.loads(array)))
    connection.create_function("sqrt", 1, lambda value: None if value is None else math.sqrt(value))
    for table in ("router_vectors", "expert_counts"):
        connection.execute(f"CREATE TABLE {table} (exp TEXT, run TEXT, attempt TEXT, step INTEGER, layer INTEGER, "
                           "node TEXT, rank INTEGER, microbatch INTEGER, metric_id INTEGER, source_metric_id INTEGER, "
                           "vals TEXT)")
    connection.execute("CREATE TABLE latest_attempt_step (exp TEXT, run TEXT, step INTEGER, attempt TEXT)")
    connection.execute("CREATE TABLE metric_names (exp TEXT, run TEXT, attempt TEXT, metric_id INTEGER, name TEXT)")
    connection.execute("CREATE TABLE metrics (metric_id INTEGER, name TEXT)")
    connection.executemany(
        "INSERT INTO latest_attempt_step VALUES ('e', 'r', ?, 'new')", [(step,) for step in (1, 2, 3)]
    )
    connection.execute("CREATE TABLE numbers (n INTEGER)")
    connection.executemany("INSERT INTO numbers VALUES (?)", [(n,) for n in range(64)])
    for name, metric_id in _IDS.items():
        connection.execute("INSERT INTO metrics VALUES (?, ?)", (metric_id, name))
        for attempt in ("old", "new"):
            connection.execute("INSERT INTO metric_names VALUES ('e', 'r', ?, ?, ?)", (attempt, metric_id, name))
    yield connection
    connection.close()


def _insert(connection, metric, values, *, step=1, layer=0, rank=0, microbatch=0, attempt="new"):
    table = "expert_counts" if metric in (_HIST, _HEAD_HIST) else "router_vectors"
    connection.execute(
        f"INSERT INTO {table} VALUES ('e', 'r', ?, ?, ?, 'node', ?, ?, ?, 0, ?)",
        (attempt, step, layer, rank, microbatch, _IDS[metric], json.dumps(values)),
    )


def _query(connection, sql, **parameters):
    cursor = connection.execute(sql, {
        "exp_filter": "e", "run_filter": "r", "rank_filter": -2, "microbatch_filter": -2, "layer_filter": -1,
        "step_from": 0, "step_to": 100, **parameters,
    })
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _counted(metric: str, origin: tuple[int, int, int, int]) -> int:
    return origin[3] if metric == _STATE else origin[2]


def _vector(metric: str, heads: int, origin: tuple[int, int, int, int]) -> list[float]:
    """Field f of head h holds 100 f + 10 h + 1 + 2 rank + 4 microbatch, small multiples of 1/2 that sum exactly."""
    rank, microbatch = origin[:2]
    return [float(_counted(metric, origin))] + [
        100.0 * field + 10 * head + 1 + 2 * rank + 4 * microbatch
        for field in range(len(_FIELDS[metric])) for head in range(heads)
    ]


def _expected_field(metric: str, field: int, head: int, origins: list[tuple[int, int, int, int]], heads: int) -> float:
    kind = _FIELDS[metric][field][1]
    column = [_vector(metric, heads, origin)[1 + field * heads + head] for origin in origins]
    return {
        "sum": sum(column) / sum(_counted(metric, origin) for origin in origins),
        "min": min(column), "max": max(column),
    }[kind]


def _expected_head_columns(metrics, heads: int, origins) -> list[dict[str, float]]:
    return [
        {column: _expected_field(metric, field, head, origins, heads)
         for metric in metrics for field, (_, _, column) in enumerate(_FIELDS[metric]) if column}
        for head in range(heads)
    ]


def _with_layer(head_columns: list[dict[str, float]]) -> dict[str, float]:
    """The layer as a whole: the mean of the head means, the minimum of the minima and the maximum of the maxima."""
    reduction = {"min": min, "max": max, "amax": max}
    return {
        column: reduction.get(column.rsplit("_", 1)[1], lambda values: sum(values) / len(values))(
            [head[column] for head in head_columns]
        )
        for column in head_columns[0]
    }


def _entropy_std(heads: list[int], origins) -> float:
    means = [_expected_field(_ROW, 0, head, origins, _SOFTMAX_HEADS) for head in heads]
    squares = [_expected_field(_ROW, 1, head, origins, _SOFTMAX_HEADS) for head in heads]
    return math.sqrt(max(sum(squares) / len(heads) - (sum(means) / len(heads)) ** 2, 0))


def _selected(rank: int, microbatch: int):
    return [origin for origin in _ORIGINS if rank in (-2, origin[0]) and microbatch in (-2, origin[1])]


def _insert_metrics(connection, metrics, heads, **kwargs):
    for origin in _ORIGINS:
        for metric in metrics:
            _insert(connection, metric, _vector(metric, heads, origin), rank=origin[0], microbatch=origin[1], **kwargs)


def _range_view(name: str) -> str:
    return _emulated(view_body(SERVING_SQL, name))


def _by_head(rows):
    return {row["head"]: row for row in rows}


def _assert_columns(row: dict, expected: dict[str, float]) -> None:
    assert {column: row[column] for column in expected} == pytest.approx(expected)


def _assert_softmax_rows(rows, origins) -> None:
    per_head = _expected_head_columns(_SOFTMAX_METRICS, _SOFTMAX_HEADS, origins)
    by_head = _by_head(rows)
    assert sorted(by_head) == [-1, 0, 1]
    for head, expected in enumerate(per_head):
        _assert_columns(by_head[head], {**expected, "ent_std": _entropy_std([head], origins)})
    _assert_columns(by_head[-1], {**_with_layer(per_head), "ent_std": _entropy_std(range(_SOFTMAX_HEADS), origins)})
    counted = sum(origin[2] for origin in origins)
    assert [(row["rows"], row["stat_rows"]) for row in rows] == [(counted, counted)] * 3


@pytest.mark.parametrize(("rank", "microbatch"), [(-2, -2), (1, -2), (-2, 1), (0, 0), (1, 1)])
def test_softmax_range_merges_the_selected_ranks_and_microbatches_per_head_and_layer(
    clickhouse_tables, rank, microbatch,
):
    _insert_metrics(clickhouse_tables, _SOFTMAX_METRICS, _SOFTMAX_HEADS)

    rows = _query(
        clickhouse_tables, _range_view("attention_softmax_range"), rank_filter=rank, microbatch_filter=microbatch
    )

    origins = _selected(rank, microbatch)
    if origins:
        _assert_softmax_rows(rows, origins)
    else:
        assert rows == []


def test_softmax_range_leaves_the_metrics_a_layer_lacks_empty_instead_of_zero(clickhouse_tables):
    _insert_metrics(clickhouse_tables, (_LSE,), _SOFTMAX_HEADS)

    rows = _query(clickhouse_tables, _range_view("attention_softmax_range"))

    assert len(rows) == 3
    for row in rows:
        assert (row["rows"], row["stat_rows"]) == (60, 0)
        assert {column for column, value in row.items() if value is None} == {
            column for metric in (_QKV, _GATE, _ROW) for _, _, column in _FIELDS[metric] if column
        } | {"ent_std"}


def test_softmax_range_clamps_the_entropy_deviation_at_zero(clickhouse_tables):
    # The mean entropy is 10 but the mean of its squares only 5: rounding can push the variance below zero.
    values = [10.0] + [100.0] * _SOFTMAX_HEADS + [50.0] * _SOFTMAX_HEADS + [0.0] * (19 * _SOFTMAX_HEADS)
    _insert(clickhouse_tables, _ROW, values)

    rows = _query(clickhouse_tables, _range_view("attention_softmax_range"))

    assert [row["ent_std"] for row in rows] == [0.0] * 3


def test_softmax_range_ignores_uncounted_rows_malformed_widths_and_superseded_attempts(clickhouse_tables):
    _insert_metrics(clickhouse_tables, _SOFTMAX_METRICS, _SOFTMAX_HEADS)
    for metric in _SOFTMAX_METRICS:
        width = len(_FIELDS[metric]) * _SOFTMAX_HEADS
        _insert(clickhouse_tables, metric, [0.0] + [-999.0] * width, rank=2)
        _insert(clickhouse_tables, metric, [5.0] + [-999.0] * (width + 1), rank=3)
        _insert(clickhouse_tables, metric, [7.0] + [-999.0] * width, rank=4, attempt="old")

    rows = _query(clickhouse_tables, _range_view("attention_softmax_range"))

    _assert_softmax_rows(rows, _ORIGINS)


def test_softmax_range_reads_only_the_requested_steps_and_layer(clickhouse_tables):
    for step in (1, 2, 3):
        for layer in (0, 1):
            _insert_metrics(clickhouse_tables, (_LSE,), _SOFTMAX_HEADS, step=step, layer=layer)

    rows = _query(clickhouse_tables, _range_view("attention_softmax_range"), step_from=2, step_to=3, layer_filter=1)

    assert sorted((row["step"], row["layer"], row["head"]) for row in rows) == [
        (step, 1, head) for step in (2, 3) for head in (-1, 0, 1)
    ]


def _insert_decay(connection):
    # Identical on every rank, so the capture stores it once per step without rank and microbatch.
    _insert(connection, _DECAY, [0.5, 0.75, 1.0, 0.25, 0.5, 0.125], rank=-1, microbatch=-1)


@pytest.mark.parametrize(("rank", "microbatch"), [(-2, -2), (1, -2), (0, 1)])
def test_linear_range_merges_the_selection_and_reads_the_decay_parameters_whatever_the_selection(
    clickhouse_tables, rank, microbatch,
):
    _insert_metrics(clickhouse_tables, _LINEAR_METRICS, _LINEAR_HEADS)
    _insert_decay(clickhouse_tables)

    rows = _query(
        clickhouse_tables, _range_view("attention_linear_range"), rank_filter=rank, microbatch_filter=microbatch
    )

    origins = _selected(rank, microbatch)
    per_head = [
        {**columns, "decay_a": decay_a, "dt_bias": dt_bias}
        for columns, decay_a, dt_bias in zip(
            _expected_head_columns(_LINEAR_METRICS, _LINEAR_HEADS, origins),
            [0.5, 0.75, 1.0], [0.25, 0.5, 0.125], strict=True)
    ]
    by_head = _by_head(rows)
    assert sorted(by_head) == [-1, 0, 1, 2]
    for head, expected in enumerate(per_head):
        _assert_columns(by_head[head], expected)
    _assert_columns(by_head[-1], _with_layer(per_head))
    assert {(row["rows"], row["documents"]) for row in rows} == {
        (sum(origin[2] for origin in origins), sum(origin[3] for origin in origins))
    }


def test_linear_range_leaves_the_decay_parameters_empty_before_they_are_recorded(clickhouse_tables):
    _insert_metrics(clickhouse_tables, _LINEAR_METRICS, _LINEAR_HEADS)

    rows = _query(clickhouse_tables, _range_view("attention_linear_range"))

    assert len(rows) == 4
    assert {(row["decay_a"], row["dt_bias"]) for row in rows} == {(None, None)}


@pytest.mark.parametrize("microbatch", [-2, 0])
def test_rank_range_means_the_entropy_over_the_heads_and_microbatches_of_each_rank(clickhouse_tables, microbatch):
    _insert_metrics(clickhouse_tables, (_ROW,), _SOFTMAX_HEADS)

    rows = _query(clickhouse_tables, _range_view("attention_rank_range"), microbatch_filter=microbatch)

    def mean(origins, field):
        summed = sum(
            _vector(_ROW, _SOFTMAX_HEADS, origin)[1 + field * _SOFTMAX_HEADS + head]
            for origin in origins for head in range(_SOFTMAX_HEADS)
        )
        return summed / (sum(origin[2] for origin in origins) * _SOFTMAX_HEADS)

    expected = {
        rank: (sum(origin[2] for origin in origins), mean(origins, 0), mean(origins, 4))
        for rank in (0, 1) if (origins := _selected(rank, microbatch))
    }
    assert {
        row["rank"]: (row["stat_rows"], row["ent_mean"], row["entn_mean"]) for row in rows
    } == pytest.approx(expected)
    assert {(row["step"], row["layer"]) for row in rows} == {(1, 0)}


def test_entropy_histograms_sum_the_pooled_and_the_per_head_counts_of_the_selection(clickhouse_tables):
    for rank, microbatch, _, _ in _ORIGINS:
        pooled = [rank + bin_index for bin_index in range(32)]
        per_head = [microbatch + head + bin_index for head in range(_SOFTMAX_HEADS) for bin_index in range(32)]
        _insert(clickhouse_tables, _HIST, pooled, rank=rank, microbatch=microbatch)
        _insert(clickhouse_tables, _HEAD_HIST, per_head, rank=rank, microbatch=microbatch)

    rows = _query(clickhouse_tables, _range_view("attention_entropy_hist_range"), rank_filter=0)

    counts = {row["head"]: json.loads(row["counts"]) for row in rows}
    assert counts == {
        -1: [2 * bin_index for bin_index in range(32)],
        0: [1 + 2 * bin_index for bin_index in range(32)],
        1: [3 + 2 * bin_index for bin_index in range(32)],
    }


@pytest.mark.parametrize("microbatch", [-2, 1])
def test_per_rank_view_unpacks_every_field_of_every_head(clickhouse_tables, microbatch):
    _insert_metrics(clickhouse_tables, _SOFTMAX_METRICS, _SOFTMAX_HEADS)
    _insert_metrics(clickhouse_tables, _LINEAR_METRICS, _LINEAR_HEADS, layer=1)
    _insert_decay(clickhouse_tables)
    sql = _emulated(
        view_body(SCHEMA_SQL.read_text(encoding="utf-8"), "attention_heads_per_rank_selected")
    )

    rows = _query(clickhouse_tables, sql, microbatch_filter=microbatch)

    expected = {}
    for origin in _selected(-2, microbatch):
        for layer, metrics, heads in ((0, _SOFTMAX_METRICS, _SOFTMAX_HEADS), (1, _LINEAR_METRICS, _LINEAR_HEADS)):
            for metric in metrics:
                vector = _vector(metric, heads, origin)
                for head in range(heads):
                    expected[(origin[0], origin[1], layer, metric, head)] = (
                        vector[0], [name for name, _, _ in _FIELDS[metric]],
                        [vector[1 + field * heads + head] for field in range(len(_FIELDS[metric]))], 1)
    assert len(rows) == len(expected)
    assert {
        (row["rank"], row["microbatch"], row["layer"], row["metric"], row["head"]):
            (row["rows"], json.loads(row["stat_names"]), json.loads(row["stats"]), row["width_valid"])
        for row in rows
    } == expected


def test_per_rank_view_flags_a_width_that_is_not_a_whole_number_of_fields(clickhouse_tables):
    _insert(clickhouse_tables, _LSE, [10.0, 1.0, 2.0, 3.0, 4.0])
    sql = _emulated(
        view_body(SCHEMA_SQL.read_text(encoding="utf-8"), "attention_heads_per_rank_selected")
    )

    rows = _query(clickhouse_tables, sql)

    assert [(row["head"], row["width_valid"]) for row in rows] == [(0, 0)]
