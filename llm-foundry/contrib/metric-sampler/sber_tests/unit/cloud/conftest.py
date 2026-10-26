import json
import math
import re
import sqlite3

import pytest


class _SumIf:
    def __init__(self):
        self.total = 0.0

    def step(self, value, condition):
        if condition:
            self.total += value

    def finalize(self):
        return self.total


class _CountIf:
    def __init__(self):
        self.count = 0

    def step(self, condition):
        self.count += bool(condition)

    def finalize(self):
        return self.count


class _UniqueIf:
    def __init__(self):
        self.values = set()

    def step(self, value, condition):
        if condition:
            self.values.add(value)

    def finalize(self):
        return len(self.values)


class _Median:
    def __init__(self):
        self.values = []

    def step(self, value):
        self.values.append(value)

    def finalize(self):
        # ClickHouse quantileExact uses the upper middle element for even populations.
        return sorted(self.values)[len(self.values) // 2]


def _array_slice(values, start, count):
    return json.dumps(json.loads(values)[start - 1:start - 1 + count])


def _sqrt(value):
    return None if value is None else math.sqrt(value)


def _pow(value, exponent):
    return None if value is None else value ** exponent


# ClickHouse aggregate and function name -> (argument count, in-memory equivalent).
_AGGREGATES = {
    "sumIf": (2, _SumIf), "countIf": (1, _CountIf), "uniqExactIf": (2, _UniqueIf), "medianExact": (1, _Median),
}
_FUNCTIONS = {
    "arraySum": (1, lambda values: float(sum(json.loads(values)))),
    "arraySlice": (3, _array_slice),
    "emptyArrayUInt64": (0, lambda: "[]"),
    "emptyArrayFloat64": (0, lambda: "[]"),
    "toUInt64": (1, int),
    "toUInt32": (1, int),
    "toFloat64": (1, float),
    "match": (2, lambda value, pattern: bool(re.search(pattern, value))),
    "extract": (2, lambda value, pattern: re.search(pattern, value).group(1)),
    "greatest": (2, max),
    "sqrt": (1, _sqrt),
    "pow": (2, _pow),
    "log": (1, math.log),
    "log2": (1, math.log2),
}


def _create_tables(connection):
    for table in ("expert_counts", "router_vectors", "scalars"):
        connection.execute(f"CREATE TABLE {table} (exp TEXT, run TEXT, attempt TEXT, step INTEGER, "
                           "layer INTEGER, node TEXT, rank INTEGER, microbatch INTEGER, metric_id INTEGER, vals TEXT, "
                           "source_metric_id INTEGER DEFAULT 0, "
                           "alias_metric_id INTEGER DEFAULT 0)")
    connection.execute("CREATE TABLE latest_attempt_step (exp TEXT, run TEXT, step INTEGER, attempt TEXT)")
    connection.execute("INSERT INTO latest_attempt_step VALUES ('e', 'r', 1, 'new')")
    connection.execute("CREATE TABLE metric_names (exp TEXT, run TEXT, attempt TEXT, metric_id INTEGER, name TEXT, "
                       "bins_start REAL DEFAULT -40, bins_width REAL DEFAULT 1, bins_count INTEGER DEFAULT 64)")
    connection.execute("CREATE TABLE metrics (metric_id INTEGER, name TEXT)")


@pytest.fixture
def sql_observations():
    """Execute the view's relational SQL with in-memory equivalents of its ClickHouse aggregates."""
    connection = sqlite3.connect(":memory:")
    for name, (arguments, aggregate) in _AGGREGATES.items():
        connection.create_aggregate(name, arguments, aggregate)
    for name, (arguments, function) in _FUNCTIONS.items():
        connection.create_function(name, arguments, function)
    _create_tables(connection)
    yield connection
    connection.close()
