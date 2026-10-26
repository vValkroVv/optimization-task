from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from metric_sampler.cloud import queries
from metric_sampler.cloud.summaries import health as health_loop
from metric_sampler.contract.catalog import METRICS
from sber_tests.unit.cloud.sql_sources import (
    DASHBOARDS_DIR,
    INGEST_SQL,
    RANK,
    SCHEMA_SQL,
    SERVING_FILES,
    SERVING_SQL,
    STEP_RANGE,
    fixture_view_sql,
    load_dashboard,
    serving_view,
    view_body,
)

pytestmark = pytest.mark.unit

# A credential (S3 access/secret key) would show up as a quoted string literal made of a long,
# unbroken run of base64-alphabet characters. Real SQL identifiers and glob/regex literals in
# these files are always broken up by punctuation well before 20 characters.
_LINE_COMMENT = re.compile(r"--[^\n]*")
_STRING_LITERAL = re.compile(r"'([^']*)'")
_BASE64ISH_RUN = re.compile(r"[A-Za-z0-9+/=]{20,}")

_TABLE_DEFINITION = re.compile(r"CREATE TABLE IF NOT EXISTS metric_sampler\.(\w+)")
_TABLE_REFERENCE = re.compile(r"metric_sampler\.(\w+)")

_FAMILY_TABLES = ["scalars", "expert_counts", "router_vectors"]
_FAMILY_TABLE_BODY = re.compile(
    r"CREATE TABLE IF NOT EXISTS metric_sampler\.(\w+)\s*\((.*?)\)\s*ENGINE", re.DOTALL
)
_FAMILY_INSERT_SELECT = re.compile(
    r"INSERT INTO metric_sampler\.(scalars|expert_counts|router_vectors)\s*(?:\([^)]*\)\s*)?SELECT(.*?)"
    r"FROM s3\(obs_ms, filename = 'family=\1/",
    re.DOTALL,
)
_INGEST_INSERT = re.compile(r"INSERT INTO metric_sampler\.(\w+)\s*(\()?")


# Names of ClickHouse's own metrics that the backend views read.
_SERVER_METRIC_NAMES = {
    "FilesystemMainPathAvailableBytes", "OSIOWaitTimeNormalized", "OSUserTimeNormalized", "OSSystemTimeNormalized"
}


@pytest.mark.parametrize("sql_path", [SCHEMA_SQL, INGEST_SQL, *SERVING_FILES], ids=lambda path: path.name)
def test_sql_file_has_no_credential_looking_string_literals(sql_path: Path) -> None:
    text = _LINE_COMMENT.sub("", sql_path.read_text())
    registered_names = {metric.name for metric in METRICS}
    for literal in _STRING_LITERAL.findall(text):
        if literal in registered_names | _SERVER_METRIC_NAMES:
            continue
        assert not _BASE64ISH_RUN.search(literal), f"suspicious literal in {sql_path.name}: {literal!r}"


def test_every_table_ingest_sql_writes_to_is_defined_in_schema_sql() -> None:
    schema_text = SCHEMA_SQL.read_text()
    ingest_text = INGEST_SQL.read_text()

    defined_tables = set(_TABLE_DEFINITION.findall(schema_text))
    assert defined_tables == {
        "scalars",
        "expert_counts",
        "router_vectors",
        "logged",
        "metrics",
        "streams",
        "monitor_parameters",
        "monitor_expert_ownership",
        "node_manifests",
        "batches",
        "dumps",
        "processed_objects",
        "ingested_objects",
        "step_index",
        "attempt_start_index",
        "step_attempts",
        "health_snapshots",
        "health_refreshes",
    }

    referenced = set(_TABLE_REFERENCE.findall(ingest_text))
    defined_views = set(re.findall(r"CREATE OR REPLACE VIEW metric_sampler\.(\w+)", schema_text))
    assert referenced <= defined_tables | defined_views
    assert {name for name, _ in _INGEST_INSERT.findall(ingest_text)} == defined_tables - {
        "step_index", "attempt_start_index", "step_attempts", "ingested_objects",
        "health_snapshots", "health_refreshes",
    }


@pytest.mark.parametrize("table", _FAMILY_TABLES)
def test_family_inserts_leave_rows_with_equal_sort_keys_to_merges_and_final(table: str) -> None:
    insert = INGEST_SQL.read_text().split(f"FROM s3(obs_ms, filename = 'family={table}/", 1)[1].split(";", 1)[0]

    assert "optimize_on_insert = 0" in insert


@pytest.mark.parametrize("table", _FAMILY_TABLES)
def test_family_table_has_source_metric_id_right_after_metric_id(table: str) -> None:
    schema_text = SCHEMA_SQL.read_text()
    bodies = dict(_FAMILY_TABLE_BODY.findall(schema_text))
    columns = [line.strip().split()[0] for line in bodies[table].strip().splitlines()]

    assert columns.index("source_metric_id") == columns.index("metric_id") + 1


@pytest.mark.parametrize("table", _FAMILY_TABLES)
def test_family_table_order_by_has_source_metric_id_right_after_metric_id(table: str) -> None:
    schema_text = SCHEMA_SQL.read_text()
    order_by = re.search(
        rf"CREATE TABLE IF NOT EXISTS metric_sampler\.{table}\b.*?ORDER BY \(([^)]*)\);", schema_text, re.DOTALL
    )
    columns = [column.strip() for column in order_by.group(1).split(",")]

    assert columns.index("source_metric_id") == columns.index("metric_id") + 1


def test_columns_named_like_column_list_keywords_are_backquoted() -> None:
    # Inside a column list ClickHouse parses a bare `index` as an INDEX declaration.
    bodies = dict(_FAMILY_TABLE_BODY.findall(SCHEMA_SQL.read_text()))
    first_words = {line.strip().split()[0] for body in bodies.values() for line in body.strip().splitlines()}

    assert first_words.isdisjoint({"index", "projection"})
    assert "`index`" in first_words


def test_metrics_table_has_no_source_metric_id() -> None:
    schema_text = SCHEMA_SQL.read_text()
    bodies = dict(_FAMILY_TABLE_BODY.findall(schema_text))

    assert "source_metric_id" not in bodies["metrics"]


@pytest.mark.parametrize("table", _FAMILY_TABLES)
def test_family_insert_selects_source_metric_id_right_after_metric_id(table: str) -> None:
    ingest_text = INGEST_SQL.read_text()
    selects = dict(_FAMILY_INSERT_SELECT.findall(ingest_text))

    assert "metric_id, source_metric_id," in selects[table]


def test_metrics_ingest_does_not_array_join_an_already_split_json_object() -> None:
    ingest_text = INGEST_SQL.read_text()
    metrics_block = _LINE_COMMENT.sub("", ingest_text.split("INSERT INTO metric_sampler.streams")[0])

    assert "ARRAY JOIN" not in metrics_block
    assert "JSONExtractArrayRaw" not in metrics_block
    assert "JSONExtractUInt(item," not in metrics_block
    assert "JSONExtractUInt(json, 'metric_id')" in metrics_block


def test_metrics_table_gets_bins_and_flag_columns_via_add_column_if_not_exists() -> None:
    schema_text = SCHEMA_SQL.read_text()

    for column, column_type in [
        ("bins_start", "Float64"),
        ("bins_width", "Float64"),
        ("bins_count", "UInt32"),
        ("summarize_experts", "Bool"),
        ("detailed", "Bool"),
        ("job_identical", "Bool"),
    ]:
        assert (
            f"ALTER TABLE metric_sampler.metrics ADD COLUMN IF NOT EXISTS {column} {column_type}" in schema_text
        )


@pytest.mark.parametrize(
    "kind",
    ["metrics", "streams", "monitor_parameters", "monitor_expert_ownership", "node_manifests", "batches", "dumps"],
)
def test_json_objects_are_marked_processed_from_their_ingested_rows(kind: str) -> None:
    statements = INGEST_SQL.read_text().split(";")
    marking = [
        statement for statement in statements
        if "INSERT INTO metric_sampler.processed_objects" in statement and f"SELECT '{kind}' AS kind" in statement
    ]

    assert len(marking) == 1
    assert re.search(rf"FROM metric_sampler\.{kind}\s", marking[0])
    assert "s3(" not in marking[0]


def _processed_marker(kind: str) -> str:
    statements = _LINE_COMMENT.sub("", INGEST_SQL.read_text()).split(";")
    marking = [
        statement for statement in statements
        if "INSERT INTO metric_sampler.processed_objects" in statement and f"SELECT '{kind}' AS kind" in statement
    ]
    assert len(marking) == 1
    return marking[0]


@pytest.mark.parametrize("kind", ["scalars", "expert_counts", "router_vectors", "logged"])
def test_family_objects_are_marked_processed_from_the_rows_their_insert_wrote(kind: str) -> None:
    ingest_text = _LINE_COMMENT.sub("", INGEST_SQL.read_text())
    inserting = [
        statement for statement in ingest_text.split(";") if f"INSERT INTO metric_sampler.{kind}\n" in statement
    ]
    marking = _processed_marker(kind)

    assert len(inserting) == 1
    assert "s3(" not in marking
    assert "FROM metric_sampler.ingested_objects" in marking
    assert f"WHERE kind = '{kind}'" in marking
    assert f"object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = '{kind}')" in marking
    assert "optimize_count_from_files" not in ingest_text


@pytest.mark.parametrize("kind", ["scalars", "expert_counts", "router_vectors", "logged"])
def test_family_table_has_an_insert_trigger_counting_rows_per_object_and_statement(kind: str) -> None:
    schema_text = SCHEMA_SQL.read_text()
    trigger = schema_text.split(
        f"CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.{kind}_ingested_objects"
    )[1].split(";")[0]

    assert " ".join(trigger.split()) == (
        f"TO metric_sampler.ingested_objects AS SELECT '{kind}' AS kind, source_object AS object, ingested_at, "
        f"count() AS rows FROM metric_sampler.{kind} GROUP BY source_object, ingested_at"
    )


def test_ingested_objects_table_sums_the_blocks_of_one_statement() -> None:
    table_def = SCHEMA_SQL.read_text().split(
        "CREATE TABLE IF NOT EXISTS metric_sampler.ingested_objects"
    )[1].split(";")[0]

    assert "ENGINE = SummingMergeTree" in table_def
    assert "ORDER BY (kind, object, ingested_at)" in table_def


class _MarkerArgMax:
    def __init__(self):
        self.value = None
        self.version = None

    def step(self, value, version):
        if self.version is None or version > self.version:
            self.value, self.version = value, version

    def finalize(self):
        return self.value


def test_node_manifests_marker_records_one_row_per_object_holding_several_manifests() -> None:
    marker = _processed_marker("node_manifests").replace("metric_sampler.", "")
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.create_function("now", 0, lambda: 1000)
        connection.execute("CREATE TABLE node_manifests (source_object)")
        connection.execute("CREATE TABLE processed_objects (kind, object, rows, processed_at)")
        connection.executemany(
            "INSERT INTO node_manifests VALUES (?)", [("batch",), ("batch",), ("batch",), ("window",), ("known",)]
        )
        connection.execute("INSERT INTO processed_objects VALUES ('node_manifests', 'known', 1, 1)")

        connection.execute(marker)
        connection.execute(marker)

        assert connection.execute("SELECT * FROM processed_objects ORDER BY object").fetchall() == [
            ("node_manifests", "batch", 1, 1000), ("node_manifests", "known", 1, 1),
            ("node_manifests", "window", 1, 1000),
        ]


@pytest.mark.parametrize("kind", ["scalars", "expert_counts", "router_vectors", "logged"])
def test_family_marker_records_each_object_once_with_the_rows_of_its_latest_complete_insert(kind: str) -> None:
    marker = _processed_marker(kind).replace("metric_sampler.", "")
    other_kind = "logged" if kind == "scalars" else "scalars"
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.create_aggregate("argMax", 2, _MarkerArgMax)
        connection.create_function("now", 0, lambda: 1000)
        connection.execute("CREATE TABLE ingested_objects (kind, object, ingested_at, rows)")
        connection.execute("CREATE TABLE processed_objects (kind, object, rows, processed_at)")
        connection.executemany("INSERT INTO ingested_objects VALUES (?,?,?,?)", [
            (kind, "retried", 1, 6),  # a failed statement wrote part of the object
            (kind, "retried", 2, 7),  # its retry wrote the whole object in two insert blocks
            (kind, "retried", 2, 5),
            (kind, "single", 2, 9),
            (kind, "known", 1, 4),
            (other_kind, "other", 2, 3),
        ])
        connection.execute("INSERT INTO processed_objects VALUES (?,?,?,?)", (kind, "known", 4, 1))
        connection.execute("INSERT INTO processed_objects VALUES (?,?,?,?)", (other_kind, "single", 1, 1))

        connection.execute(marker)
        connection.execute(marker)

        assert connection.execute("SELECT * FROM processed_objects ORDER BY kind, object").fetchall() == sorted([
            (kind, "known", 4, 1), (kind, "retried", 12, 1000), (kind, "single", 9, 1000), (other_kind, "single", 1, 1),
        ])


def test_metrics_insert_reads_job_identical_from_the_metric_json() -> None:
    ingest_text = INGEST_SQL.read_text()
    metrics_block = ingest_text.split("INSERT INTO metric_sampler.metrics")[1].split(
        "INSERT INTO metric_sampler.processed_objects"
    )[0]

    assert "job_identical" in metrics_block.split("SELECT", 1)[0]
    assert "JSONExtractBool(json, 'job_identical') AS job_identical" in metrics_block


def test_logged_and_streams_tables_have_no_source_metric_id() -> None:
    schema_text = SCHEMA_SQL.read_text()
    bodies = dict(_FAMILY_TABLE_BODY.findall(schema_text))

    assert "source_metric_id" not in bodies["logged"]
    assert "source_metric_id" not in bodies["streams"]


@pytest.mark.parametrize(
    "view",
    [
        "metric_names",
        "stream_names",
        "attempt_starts",
        "latest_attempt_step",
        "loss_per_step",
        "layer_load_per_step",
        "expert_totals_per_step",
        "health_per_step",
        "pipeline_freshness",
        "loss_quantiles_per_step",
        "stream_loss_per_step",
        "expert_loss_per_step",
        "expert_stream_map",
        "grad_per_rank",
        "logged_per_step",
        "activation_per_step",
        "activation_exp_quantiles",
        "activation_massive_tokens",
        "activation_per_rank",
        "timing_per_rank",
        "capacity_per_step",
        "loss_position_per_step",
        "stream_share_per_step",
        "sample_loss_quantiles_per_step",
        "attention_heads_per_rank_selected",
    ],
)
def test_view_is_defined_in_schema_sql(view: str) -> None:
    schema_text = SCHEMA_SQL.read_text()

    assert f"CREATE OR REPLACE VIEW metric_sampler.{view} AS" in schema_text


def test_health_per_step_counts_identical_violations_regardless_of_verbose() -> None:
    schema_text = SCHEMA_SQL.read_text()
    health_view = schema_text.split("CREATE OR REPLACE VIEW metric_sampler.health_per_step AS")[1].split(
        "CREATE OR REPLACE VIEW metric_sampler.pipeline_freshness"
    )[0]

    assert "verbose" not in health_view
    assert "node_merge = 'identical'" in health_view


@pytest.mark.parametrize(
    "view",
    ["stream_loss_per_step", "expert_loss_per_step", "activation_selected"],
)
def test_std_formulas_clamp_the_variance_to_zero_before_the_square_root(view: str) -> None:
    schema_text = SCHEMA_SQL.read_text()
    view_text = schema_text.split(f"CREATE OR REPLACE VIEW metric_sampler.{view} AS")[1].split(
        "CREATE OR REPLACE VIEW metric_sampler."
    )[0]

    assert "sqrt(greatest(" in view_text


def test_activation_per_step_takes_elements_from_moments_finite_count_not_exp_hist() -> None:
    schema_text = SCHEMA_SQL.read_text()
    after_create = schema_text.split("CREATE OR REPLACE VIEW metric_sampler.activation_selected AS")[1]
    next_view = after_create.split("CREATE OR REPLACE VIEW metric_sampler.activation_per_step")[0]
    view_text = _LINE_COMMENT.sub("", next_view)  # drop the next view's leading comment block

    assert "ELSE rv.values[4] END) AS elements" in view_text
    assert "toUInt64(ec.values[2]) AS elements" in view_text
    assert "moments.all_counts_exact AS all_counts_exact" in view_text
    assert "exp_hist" not in view_text


@pytest.mark.parametrize(
    ("view", "group_column"),
    [("activation_per_rank_selected", "point"), ("timing_per_rank", "metric")],
)
def test_per_rank_view_pools_its_robust_z_scale_by_run_and_group(view: str, group_column: str) -> None:
    schema_text = SCHEMA_SQL.read_text()
    view_text = schema_text.split(f"CREATE OR REPLACE VIEW metric_sampler.{view} AS")[1].split(
        "CREATE OR REPLACE VIEW metric_sampler."
    )[0]

    assert "robust_z" in view_text
    assert f"GROUP BY exp, run, {group_column}" in view_text or f"GROUP BY exp, run, layer, {group_column}" in view_text


def test_every_ingest_insert_other_than_processed_objects_has_an_explicit_column_list() -> None:
    # A positional INSERT ... SELECT silently shifts columns when ALTER TABLE ADD COLUMN appends
    # a new one at the end.
    ingest_text = INGEST_SQL.read_text()

    for table, paren in _INGEST_INSERT.findall(ingest_text):
        if table == "processed_objects":
            continue
        assert paren == "(", f"INSERT INTO metric_sampler.{table} has no explicit column list"


def test_node_manifests_insert_reads_paths_without_hive_partition_columns() -> None:
    manifests_block = INGEST_SQL.read_text().split("INSERT INTO metric_sampler.node_manifests")[1].split(
        "INSERT INTO metric_sampler.processed_objects"
    )[0]

    assert "SETTINGS s3_throw_on_zero_files_match = 0, use_hive_partitioning = 0;" in manifests_block


def test_node_manifests_insert_reads_nodes_in_job_from_the_manifest_json() -> None:
    ingest_text = INGEST_SQL.read_text()
    manifests_block = ingest_text.split("INSERT INTO metric_sampler.node_manifests")[1].split(
        "INSERT INTO metric_sampler.processed_objects"
    )[0]

    assert "nodes_in_job" in manifests_block.split("SELECT", 1)[0]
    assert "JSONExtractUInt(json, 'nodes_in_job') AS nodes_in_job" in manifests_block


def test_node_manifests_table_has_nodes_in_job_added_via_add_column_if_not_exists() -> None:
    schema_text = SCHEMA_SQL.read_text()

    assert (
        "ALTER TABLE metric_sampler.node_manifests ADD COLUMN IF NOT EXISTS "
        "nodes_in_job UInt32 DEFAULT 0"
    ) in schema_text


def test_node_manifests_table_has_absent_ranks_added_via_add_column_if_not_exists() -> None:
    schema_text = SCHEMA_SQL.read_text()

    assert (
        "ALTER TABLE metric_sampler.node_manifests ADD COLUMN IF NOT EXISTS absent_ranks Array(UInt16) DEFAULT []"
        in schema_text
    )


def test_node_manifests_insert_reads_absent_ranks_from_the_manifest_json() -> None:
    ingest_text = INGEST_SQL.read_text()
    manifests_block = ingest_text.split("INSERT INTO metric_sampler.node_manifests")[1].split(
        "INSERT INTO metric_sampler.processed_objects"
    )[0]

    assert "absent_ranks" in manifests_block.split("SELECT", 1)[0]
    assert "JSONExtract(json, 'absent_ranks', 'Array(UInt16)') AS absent_ranks" in manifests_block


def test_health_per_step_nodes_expected_is_the_greatest_of_batches_and_manifest_nodes_in_job() -> None:
    schema_text = SCHEMA_SQL.read_text()
    health_view = schema_text.split("CREATE OR REPLACE VIEW metric_sampler.health_per_step AS")[1].split(
        "CREATE OR REPLACE VIEW metric_sampler.pipeline_freshness"
    )[0]

    assert "metric_sampler.node_manifests" in health_view
    assert (
        "greatest(coalesce(expected_nodes.nodes_expected, 0), coalesce(manifest_nodes_in_job.nodes_in_job, 0))"
        in health_view
    )


def test_health_per_step_aggregates_batches_and_manifests_with_duplicate_safe_reducers() -> None:
    # node_manifests/batches are plain MergeTree (idempotency relies on processed_objects), so a
    # concurrent manual ingest.sql run can duplicate a row; sum() would inflate nodes_expected.
    schema_text = SCHEMA_SQL.read_text()
    health_view = schema_text.split("CREATE OR REPLACE VIEW metric_sampler.health_per_step AS")[1].split(
        "CREATE OR REPLACE VIEW metric_sampler.pipeline_freshness"
    )[0]

    assert "uniqExact(node) AS nodes_expected" in health_view
    assert "max(nodes_in_job) AS nodes_in_job" in health_view
    assert not re.search(r"sum\([^)]*\)\s+AS\s+nodes_expected", health_view)


@pytest.mark.parametrize("family", _FAMILY_TABLES)
def test_health_family_aggregation_preserves_missing_ranks_summaries_and_identical_rows(family: str) -> None:
    health_view = SCHEMA_SQL.read_text().split("CREATE OR REPLACE VIEW metric_sampler.health_per_step AS")[1]
    sql = health_view.split(",\nexpected_nodes AS", 1)[0] + "\nSELECT * FROM health ORDER BY attempt, step"
    sql = sql.replace("metric_sampler.", "").replace(" FINAL", "")
    sql = sql.replace("uniqExact(node)", "count(DISTINCT node)").replace("countIf(", "sum(")
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.create_function("if", 3, lambda condition, yes, no: yes if condition else no)
        connection.execute(
            "CREATE TABLE metrics (exp TEXT, run TEXT, attempt TEXT, metric_id INTEGER, node_merge TEXT)"
        )
        for table in _FAMILY_TABLES:
            connection.execute(f"CREATE TABLE {table} (exp TEXT, run TEXT, attempt TEXT, step INTEGER, node TEXT, "
                               "rank INTEGER, microbatch INTEGER, layer INTEGER, "
                               "metric_id INTEGER, source_metric_id INTEGER, "
                               "contributors INTEGER, expected_contributors INTEGER, identical INTEGER)")
        connection.executemany("INSERT INTO metrics VALUES ('e','r',?,?,?)", [
            ("new", 10, "per_rank"), ("new", 11, "identical"), ("new", 12, "per_rank"),
            ("new", 20, "per_rank"), ("new", 30, "per_rank"), ("old", 10, "per_rank"),
        ])
        connection.executemany(f"INSERT INTO {family} VALUES ('e','r',?,?,?,?,?,?,?,?,?,?,?)", [
            ("new", 1, "a", 0, 0, 0, 10, 0, 1, 4, 0),
            ("new", 1, "a", 1, 0, 0, 10, 0, 1, 4, 0),
            ("new", 1, "a", 0, 1, 0, 10, 0, 1, 4, 0),
            ("new", 1, "a", 0, 0, 1, 10, 0, 1, 2, 0),
            ("new", 1, "a", 1, 0, 1, 10, 0, 1, 2, 0),
            ("new", 1, "b", 0, 0, 0, 10, 0, 1, 2, 0),
            ("new", 1, "b", 1, 0, 0, 10, 0, 1, 2, 0),
            ("new", 1, "a", 0, 0, 0, 12, 10, 1, 1, 0),
            ("new", 1, "a", 1, 0, 0, 12, 10, 1, 1, 0),
            ("new", 1, "a", -1, 0, 0, 11, 0, 2, 4, 0),
            ("new", 1, "a", 0, 2, 0, 11, 0, 1, 1, 0),
            ("new", 1, "a", 1, 2, 0, 11, 0, 1, 1, 1),
            ("new", 2, "a", 0, -1, 0, 10, 0, 1, 4, 0),
            ("old", 1, "a", 0, -1, 0, 10, 0, 1, 8, 0),
        ])
        for table, metric_id in zip((table for table in _FAMILY_TABLES if table != family), (20, 30)):
            connection.execute(f"INSERT INTO {table} VALUES ('e','r','new',1,'a',0,0,0,?,0,1,2,0)", (metric_id,))

        assert connection.execute(sql).fetchall() == [
            ("e", "r", "new", 1, 2, 15, 24, 2),
            ("e", "r", "new", 2, 1, 1, 4, 0),
            ("e", "r", "old", 1, 1, 1, 8, 0),
        ]


def test_pipeline_freshness_aggregates_batches_with_a_duplicate_safe_reducer() -> None:
    schema_text = SCHEMA_SQL.read_text()
    view_text = schema_text.split("CREATE OR REPLACE VIEW metric_sampler.pipeline_freshness AS")[1].split(
        "CREATE OR REPLACE VIEW metric_sampler."
    )[0]

    assert "max(collected_at) AS last_collected_at" in view_text


def test_pipeline_freshness_distinguishes_missing_summary_from_recorded_step_zero() -> None:
    sql = view_body(SCHEMA_SQL.read_text(), "pipeline_freshness").split(";", 1)[0]
    assert "toNullable(max(step)) AS health_max_step" in sql
    assert "toNullable(max(refreshed_at)) AS health_refreshed_at" in sql
    sql = sql.replace("metric_sampler.", "")
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.create_function("toNullable", 1, lambda value: value)
        connection.execute("CREATE TABLE step_index (exp,run,step,max_ingested_at)")
        connection.execute("CREATE TABLE batches (exp,run,collected_at)")
        connection.execute("CREATE TABLE health_serving_per_step (exp,run,step,refreshed_at)")
        connection.executemany("INSERT INTO step_index VALUES ('e',?,?,?)", [
            ("missing", 7, "2026-10-02 01:00:00"), ("zero", 0, "2026-10-02 01:00:00"),
        ])
        connection.executemany("INSERT INTO batches VALUES ('e',?,'2026-10-02 00:59:00')", [
            ("missing",), ("zero",),
        ])
        connection.execute("INSERT INTO health_serving_per_step VALUES ('e','zero',0,'2026-10-02 01:01:00')")

        assert connection.execute(sql + " ORDER BY run").fetchall() == [
            ("e", "missing", 7, "2026-10-02 01:00:00", "2026-10-02 00:59:00", None, None),
            ("e", "zero", 0, "2026-10-02 01:00:00", "2026-10-02 00:59:00", 0, "2026-10-02 01:01:00"),
        ]


def test_ingest_writes_logged_and_streams_gated_like_other_objects() -> None:
    ingest_text = INGEST_SQL.read_text()

    assert "INSERT INTO metric_sampler.logged" in ingest_text
    assert "INSERT INTO metric_sampler.streams" in ingest_text
    assert "kind = 'logged'" in ingest_text
    assert "kind = 'streams'" in ingest_text
    assert "family=logged/**/*.parquet" in ingest_text
    assert "metrics/**/streams.json" in ingest_text


def test_dumps_table_is_a_replacing_merge_tree_ordered_by_exp_run_attempt_step_rank() -> None:
    schema_text = SCHEMA_SQL.read_text()
    table_def = schema_text.split("CREATE TABLE IF NOT EXISTS metric_sampler.dumps")[1].split(
        "CREATE TABLE IF NOT EXISTS metric_sampler."
    )[0]

    assert "ENGINE = ReplacingMergeTree(ingested_at)" in table_def
    assert "ORDER BY (exp, run, attempt, step, rank)" in table_def


def test_dumps_ingest_reads_the_index_json_fields() -> None:
    ingest_text = INGEST_SQL.read_text()
    dumps_block = ingest_text.split("INSERT INTO metric_sampler.dumps")[1].split(
        "INSERT INTO metric_sampler.processed_objects"
    )[0]

    for column in ["rows", "bytes", "points", "layers", "tokens", "s3_key"]:
        assert column in dumps_block.split("SELECT", 1)[0]
    assert "JSONExtract(json, 'points', 'Array(String)') AS points" in dumps_block
    assert "JSONExtract(json, 'layers', 'Array(Int32)') AS layers" in dumps_block
    assert "kind = 'dumps'" in ingest_text


_GLOB_LITERAL = re.compile(r"filename = '([^']*)'")


def test_dump_glob_does_not_overlap_other_ingest_globs() -> None:
    ingest_text = INGEST_SQL.read_text()
    globs = _GLOB_LITERAL.findall(ingest_text)

    assert "dumps/**/*.json" in globs
    assert [glob for glob in globs if glob.startswith("dumps/")] == ["dumps/**/*.json"] * globs.count(
        "dumps/**/*.json"
    )
    assert all("dumps/" not in glob for glob in globs if glob != "dumps/**/*.json")


def test_stream_share_per_step_reads_stream_metrics_directly_and_keeps_tokenless_streams() -> None:
    # stream_loss_per_step's `tokens > 0` filter drops a stream that has samples but no valid
    # tokens this batch; this view must not inherit that filter.
    view_text = view_body(SCHEMA_SQL.read_text(), "stream_share_per_step")

    assert "metric_sampler.stream_loss_per_step" not in view_text
    for metric in ["loss/stream_samples", "loss/stream_count", "loss/stream_sum"]:
        assert metric in view_text
    assert "stream_samples.values[stream_index] > 0 OR stream_count.values[stream_index] > 0" in view_text


def test_capacity_view_uses_equal_mask_observations_and_preserves_legacy_unknowns() -> None:
    view_text = view_body(SCHEMA_SQL.read_text(), "capacity_selected")

    assert "ec.node, ec.rank, ec.microbatch" in view_text
    assert "ec.attempt = las.attempt" in view_text
    assert "valid_after_capacity / nullIf(valid_before_capacity, 0)" in view_text
    assert "legacy_valid_input / nullIf(legacy_input, 0)" in view_text
    assert "min(capacity_observed) AS capacity_observed" in view_text
    assert "AND ec.metric_id IN" in view_text


def test_loss_position_per_step_reads_position_metrics_and_guards_the_empty_bucket() -> None:
    view_text = view_body(SCHEMA_SQL.read_text(), "loss_position_per_step")

    assert "loss/position_sum" in view_text
    assert "loss/position_count" in view_text
    assert "nullIf(position_count.values[bucket_index], 0)" in view_text


def test_sample_loss_quantiles_per_step_keeps_per_stream_histograms_separate() -> None:
    # Unlike loss_quantiles_per_step's main_hist, the per-stream slices are not reduced into one
    # histogram: each stream gets its own row.
    view_text = view_body(SCHEMA_SQL.read_text(), "sample_loss_quantiles_per_step")

    assert "loss/sample_hist_by_stream" in view_text
    assert "arrayReduce" not in view_text
    assert "p50" in view_text
    assert "p90" in view_text
    assert "p99" in view_text
    assert "p999" not in view_text


# ClickHouse only accepts `FROM t AS a FINAL`; `FROM t FINAL AS a` is a syntax error.
_FINAL_BEFORE_ALIAS = re.compile(r"\bFINAL\s+AS\s", re.IGNORECASE)


def test_health_refresh_disables_ordered_aggregation_over_the_family_union() -> None:
    health = view_body(SCHEMA_SQL.read_text(), "health_per_step")
    aggregation = health.split("node_metric_health AS", 1)[1].split("health AS", 1)[0]
    branches = aggregation.split("UNION ALL")

    assert len(branches) == len(_FAMILY_TABLES)
    for table, branch in zip(_FAMILY_TABLES, branches):
        assert f"FROM metric_sampler.{table} FINAL" in branch
        assert "GROUP BY exp, run, attempt, step, node, metric_id, source_metric_id, layer, microbatch" in branch
    refresh = health_loop.SNAPSHOT_STATEMENT
    assert "FROM metric_sampler.health_per_step" in refresh
    # Aggregation in storage order is much slower over the narrow window of a refresh.
    assert "SETTINGS optimize_aggregation_in_order = 0" in refresh


def test_health_refresh_drops_rows_around_the_window_before_final_merges_them() -> None:
    assert "optimize_move_to_prewhere_if_final = 1" in health_loop.SNAPSHOT_STATEMENT


def test_the_ingest_worker_leaves_the_health_refresh_to_its_own_worker() -> None:
    assert "health_" not in INGEST_SQL.read_text()


@pytest.mark.parametrize(
    ("view", "metric"),
    [("expert_stream_latest_live", "expert_loss/stream_count"), ("expert_loss_latest_live", "expert_loss/count")],
)
def test_latest_live_views_read_the_raw_counts_of_their_own_metric_only(view, metric) -> None:
    body = SERVING_SQL.split(f"CREATE OR REPLACE VIEW metric_sampler.{view}\n", 1)[1].split(";\n", 1)[0]

    assert re.search(
        r"ec\.metric_id IN \(\s*SELECT metric_id\s+FROM metric_sampler\.metrics\s+"
        rf"WHERE name = '{re.escape(metric)}'\s*\)\)",
        body,
    )


def test_health_dashboard_reads_all_published_steps_and_exposes_refresh_lag() -> None:
    dashboard = load_dashboard("health.json")
    panel = next(panel for panel in dashboard["panels"] if panel["id"] == 3)
    sql = panel["targets"][0]["rawSql"]
    assert "FROM metric_sampler.health_serving_per_step" in sql
    assert "AND step >=" not in sql
    assert "optimize_aggregation_in_order" not in sql
    assert "NULL gap markers" in panel["description"]
    freshness = next(panel for panel in dashboard["panels"] if panel["id"] == 1)["targets"][0]["rawSql"]
    assert "health_max_step, health_refreshed_at" in freshness


def _sql_sources() -> list[tuple[str, str]]:
    sources = [(path.name, path.read_text(encoding="utf-8")) for path in (SCHEMA_SQL, INGEST_SQL, *SERVING_FILES)]
    sources += [(name, value) for name, value in vars(queries).items() if name.isupper() and isinstance(value, str)]
    for path in sorted(DASHBOARDS_DIR.glob("*.json")):
        dashboard = json.loads(path.read_text(encoding="utf-8"))
        panels = [nested for panel in dashboard["panels"] for nested in panel.get("panels", [panel])]
        for panel in panels:
            for target in panel["targets"]:
                sources.append((f"{path.name}: {panel['title']}", target["rawSql"]))
    return sources


_SQL_SOURCES = _sql_sources()


@pytest.mark.parametrize("sql", [sql for _, sql in _SQL_SOURCES], ids=[source for source, _ in _SQL_SOURCES])
def test_no_sql_puts_final_before_the_table_alias(sql: str) -> None:
    assert _FINAL_BEFORE_ALIAS.search(sql) is None


def test_activations_exponent_histogram_panel_reports_the_real_log2_exponent() -> None:
    dashboard = load_dashboard("activations.json")
    # Selected by its source, not its title.
    panel = next(p for p in dashboard["panels"] if "activation_histograms_${hsrc}" in json.dumps(p.get("targets", [])))
    sql = panel["targets"][0]["rawSql"]

    assert "bins_start + bins_width * (bin_index - 1) AS log2_exponent" in sql
    assert "sum(exact_count) AS exact_count" in sql
    assert "sum(legacy_count) AS legacy_count" in sql
    assert "all_counts_exact" in sql
    assert "bin_index - 1 AS bin," not in sql
    for layer_panel in dashboard["panels"][:3]:
        assert layer_panel["fieldConfig"]["defaults"]["displayName"] == "Layer ${__field.labels.layer}"
    massive_share = dashboard["panels"][2]
    assert {"id": "unit", "value": "percentunit"} in massive_share["fieldConfig"]["overrides"][0]["properties"]
    quantiles = dashboard["panels"][3]
    assert "approximate, log2" in quantiles["title"]
    assert "factor of 2" in quantiles["description"]
    assert "edge bins" in quantiles["description"]
    assert {
        override["matcher"]["options"]: override["properties"][0]["value"]
        for override in quantiles["fieldConfig"]["overrides"]
    } == {
        "p50_log2": "p50 (approx., log2)",
        "p99_log2": "p99 (approx., log2)",
        "p999_log2": "p99.9 (approx., log2)",
        "max_log2": "Max bin lower edge (log2)",
    }
    variables = {variable["name"]: variable for variable in dashboard["templating"]["list"]}
    assert variables["layer"]["label"] == "Layer (quantiles)"


@pytest.mark.parametrize(
    ("filename", "variable", "routes", "views"),
    [
        (
            "activations.json",
            "src",
            ["all", "ranksum", "rankmbsum", "mbsum", "live"],
            ["activation_{}", "activation_massive_{}", "activation_exp_quantiles_{}"],
        ),
        (
            "activations.json",
            "hsrc",
            ["all", "ranklatest", "rankmblatest", "mblatest", "step_live"],
            ["activation_histograms_{}"],
        ),
        ("experts.json", "copy", ["all", "ranksum", "mbsum", "rankmbsum", "live"], ["capacity_selected_{}"]),
        ("experts.json", "rbsrc", ["all", "ranksum", "mbsum", "rankmbsum", "live"], ["router_balance_{}"]),
        (
            "attention.json",
            "asrc",
            ["all", "live"],
            ["attention_softmax_{}", "attention_linear_{}", "attention_entropy_hist_{}"],
        ),
        ("attention.json", "rsrc", ["all", "live"], ["attention_rank_{}"]),
    ],
)
def test_source_routes_read_views_defined_in_serving_sql(filename, variable, routes, views) -> None:
    dashboard = load_dashboard(filename)
    query = next(item for item in dashboard["templating"]["list"] if item["name"] == variable)["query"]

    assert [a or b for a, b in re.findall(r"(?:\)\)|> 0), '(\w+)'|, '(\w+)'\) AS", query)] == routes
    for route in routes:
        for view in views:
            assert f"CREATE OR REPLACE VIEW metric_sampler.{view.format(route)}\n" in SERVING_SQL


_PANEL_SETTINGS = "join_algorithm = 'hash', optimize_move_to_prewhere_if_final = 1"
# Up to this many rows a panel shows every point; past it, the steps are bucketed. A trend keeps the two extremes of
# each bucket, so it has half as many buckets as rows.
_ROW_LIMIT = 900_000
_EXPERIMENT_DASHBOARD_COUNT = 9
_PER_RANK_OUTLIER_CALL_COUNT = 7


def test_panels_read_system_tables_only_through_the_definer_views_of_serving_sql() -> None:
    # The dashboard user may not read the system tables; the backend_* views read them as their definer.
    dashboard = load_dashboard("backend.json")
    views = set(re.findall(r"metric_sampler\.(backend_\w+)", json.dumps(dashboard)))

    assert [
        sql for _, sql in _SQL_SOURCES if _SQL_SOURCES and "system." in sql and sql.lstrip().startswith("SELECT")
    ] == []
    assert views == {
        "backend_disks", "backend_host_metrics", "backend_part_writes", "backend_table_sizes", "backend_errors"
    }
    for view in views:
        definition = f"CREATE OR REPLACE VIEW metric_sampler.{view} DEFINER = CURRENT_USER SQL SECURITY DEFINER AS"
        assert definition in SERVING_SQL


def test_every_panel_query_hashes_its_joins_and_filters_sort_key_columns_before_final() -> None:
    queries = []
    for path in DASHBOARDS_DIR.glob("*.json"):
        panels = json.loads(path.read_text(encoding="utf-8"))["panels"]
        queries += [
            target["rawSql"] for panel in [*panels, *(inner for panel in panels for inner in panel.get("panels", []))]
            for target in panel.get("targets", []) if "rawSql" in target
        ]

    assert queries
    assert [query[-120:] for query in queries if not query.endswith(_PANEL_SETTINGS)] == []


def test_no_query_substitutes_a_variable_that_may_still_be_empty_without_a_fallback() -> None:
    unguarded = re.compile(r"\$\{(?:rank|microbatch|step_from|step_to|latest_step|hist_step|layer|moe_layer)\}")
    queries = []
    for path in DASHBOARDS_DIR.glob("*.json"):
        dashboard = json.loads(path.read_text(encoding="utf-8"))
        panels = [*dashboard["panels"], *(inner for panel in dashboard["panels"] for inner in panel.get("panels", []))]
        queries += [target["rawSql"] for panel in panels for target in panel.get("targets", []) if "rawSql" in target]
        queries += [str(variable["query"]) for variable in dashboard["templating"]["list"]]

    assert queries
    assert [query[:100] for query in queries if unguarded.search(query)] == []


def test_serving_sql_defines_no_rank_sorted_copy_table_or_view_feeding_one() -> None:

    assert re.findall(r"by_rank_ready|metric_sampler\.\w+_by_rank(?:_mv)?\b", SERVING_SQL) == []


def test_no_dashboard_or_view_reads_the_rank_sorted_copies() -> None:
    copy_reference = re.compile(r"by_rank_ready|metric_sampler\.\w+_by_rank\b|byrank|mbcopy")
    views = [statement for statement in SERVING_SQL.split(";\n\n") if "CREATE OR REPLACE VIEW" in statement]

    assert [view for view in views if copy_reference.search(view)] == []
    for path in DASHBOARDS_DIR.glob("*.json"):
        assert copy_reference.search(path.read_text(encoding="utf-8")) is None, path.name


def test_dashboards_open_on_the_experiment_and_run_their_viewer_chose_last() -> None:
    remembered = []
    for path in sorted(DASHBOARDS_DIR.glob("*.json")):
        variables = json.loads(path.read_text(encoding="utf-8"))["templating"]["list"]
        names = [variable["name"] for variable in variables]
        if "exp" not in names:
            continue
        by_name = dict(zip(names, variables))
        remembered.append(path.name)

        assert by_name["remember"]["query"] == (
            "SELECT '' AS remembered SETTINGS log_comment = 'ms-selection ${__user.login} ${exp} ${run}', "
            "use_query_cache = 0"
        )
        assert by_name["remember"]["hide"] == 2
        assert names.index("remember") > names.index("run")
        assert (
            "(SELECT exp FROM metric_sampler.viewer_selection WHERE viewer = '${__user.login}' "
            "ORDER BY event_time DESC LIMIT 1) DESC"
        ) in by_name["exp"]["query"]
        assert (
            "WHERE viewer = '${__user.login}' AND exp = '${exp}' ORDER BY event_time DESC LIMIT 1) DESC"
            in by_name["run"]["query"]
        )
        assert by_name["exp"]["query"].endswith("SETTINGS use_query_cache = 0")
        assert by_name["run"]["query"].endswith("SETTINGS use_query_cache = 0")

    assert len(remembered) == _EXPERIMENT_DASHBOARD_COUNT
    view = SERVING_SQL[SERVING_SQL.index("CREATE OR REPLACE VIEW metric_sampler.viewer_selection"):]
    assert "SQL SECURITY DEFINER" in view.splitlines()[0]
    assert "startsWith(log_comment, 'ms-selection ')" in view


@pytest.mark.parametrize("family", ["grad_per_rank", "timing_per_rank"])
@pytest.mark.parametrize("route", ["fast", "live", "cards"])
def test_per_rank_outlier_views_take_the_run_as_parameters(family, route) -> None:
    assert "(exp = {exp_filter:String}) AND (run = {run_filter:String})" in serving_view(f"{family}_{route}")


@pytest.mark.parametrize("family", ["grad_per_rank", "timing_per_rank"])
def test_fast_outlier_views_read_the_summary_once_and_take_the_step_median_in_a_window(family) -> None:
    view = serving_view(f"{family}_fast")

    assert view.count("FROM metric_sampler.summary_") == 1
    assert "OVER (PARTITION BY step" in view


def test_expert_totals_rank_sums_the_vectors_of_a_layer_before_it_expands_them_to_experts() -> None:
    view = serving_view("expert_totals_rank")

    assert view.rindex("GROUP BY") < view.index("ARRAY JOIN")
    assert "argMin(metric_tokens, metric_id)" in view
    assert "min(r.metric_id) OVER" not in view


def test_router_stream_load_preserves_rank_microbatch_and_exposes_both_share_denominators() -> None:
    view = serving_view("router_stream_load_range")

    assert "mn.name = 'router/stream_tokens_per_expert'" in view
    assert "sumForEach(ec.`values`) AS `values`" in view
    assert view.index("GROUP BY") < view.index("ARRAY JOIN")
    assert "ec.attempt = las.attempt" in view
    assert "mn.attempt = ec.attempt" in view
    assert "stream_totals.attempt = counts.attempt" in view
    assert "totals.attempt AS attempt" in view
    assert "FROM metric_sampler.streams AS s FINAL" in view
    assert "stream_names.attempt = totals.attempt" in view
    assert "ec.source_metric_id = 0" in view
    assert "ec.rank >= 0" in view and "ec.microbatch >= 0" in view
    for dimension in ["ec.step", "ec.layer", "ec.rank", "ec.microbatch"]:
        assert dimension in view
    assert "intDiv(idx - 1, stream_totals.num_streams) + 1 AS expert" in view
    assert "(idx - 1) % stream_totals.num_streams AS stream_id" in view
    assert "totals.tokens / nullIf(totals.stream_tokens, 0) AS expert_share_within_stream" in view
    assert "totals.tokens / nullIf(totals.expert_tokens, 0) AS stream_mix_within_expert" in view
    assert "totals.stream_tokens / nullIf(totals.layer_tokens, 0) AS stream_share" in view
    assert "(totals.stream_max_tokens - totals.stream_mean_tokens) / nullIf(totals.stream_mean_tokens, 0)" in view


def _sqlite_router_stream_load_select(connection: sqlite3.Connection) -> str:
    class SumForEach:
        def __init__(self):
            self.values = None

        def step(self, encoded):
            values = json.loads(encoded)
            if self.values is None:
                self.values = [0] * len(values)
            assert len(self.values) == len(values)
            self.values = [left + right for left, right in zip(self.values, values)]

        def finalize(self):
            return json.dumps(self.values or [])

    connection.create_aggregate("sumForEach", 1, SumForEach)
    # Translate only ClickHouse table-function/array syntax; keep the view CTEs,
    # joins, filters, grouping and projections sourced from the SQL file.
    view_select = serving_view("router_stream_load_range").split("AS WITH", 1)[1]
    translated = "WITH" + view_select
    replacements = [
        ("metric_sampler.", ""), (" FINAL", ""),
        ("{exp_filter:String}", "'e'"), ("{run_filter:String}", "'r'"),
        ("{step_from:UInt32}", "0"), ("{step_to:UInt32}", "2"),
        ("{rank_filter:Int32}", ":rank_filter"), ("{microbatch_filter:Int32}", ":microbatch_filter"),
        ("ec.`values`", "ec.vals"), ("counts.`values`", "counts.vals"),
        ("AS `values`", "AS vals"),
        ("intDiv(idx - 1, stream_totals.num_streams) + 1",
         "CAST(CAST(j.key AS INTEGER) / stream_totals.num_streams AS INTEGER) + 1"),
        ("(idx - 1) % stream_totals.num_streams", "CAST(j.key AS INTEGER) % stream_totals.num_streams"),
        ("token_count AS tokens", "CAST(j.value AS INTEGER) AS tokens"),
        ("arrayEnumerate(counts.vals) AS idx,\n            counts.vals AS token_count", "json_each(counts.vals) AS j"),
        ("coalesce(stream_names.name, concat('stream', toString(totals.stream_id)))",
         "coalesce(stream_names.name, 'stream' || totals.stream_id)"),
        ("SELECT metric_id\n            FROM metrics\n            WHERE name = 'router/stream_tokens_per_expert'",
         "SELECT metric_id FROM metrics WHERE name = 'router/stream_tokens_per_expert'"),
    ]
    for source, target in replacements:
        translated = translated.replace(source, target)
    translated = translated.replace("ARRAY JOIN\n            ", "CROSS JOIN ")
    return translated + "\nORDER BY step, rank, microbatch, expert, stream_id"


def _assert_router_stream_load_panels(connection: sqlite3.Connection) -> None:
    connection.create_function("toInt32OrNull", 1, lambda value: None if value is None else int(value))
    connection.create_function("toString", 1, str)
    connection.create_function("concat", -1, lambda *values: "".join(str(value) for value in values))
    dashboard = load_dashboard("experts.json")
    expectations = [
        (14, {"0: short": pytest.approx(1 / 11), "1: long": pytest.approx(1 / 3), "2: new": pytest.approx(1)}),
        (15, {"0: short": 11, "1: long": 6, "2: new": 2}),
    ]
    for panel_id, expected_step_one in expectations:
        panels = dashboard["panels"]
        panel = next(panel for panel in panels if panel["id"] == panel_id)
        panel_query = panel["targets"][0]["rawSql"]
        start = panel_query.index("WITH base AS (")
        end = panel_query.index(", marked AS (")
        observed_sql = panel_query[start:end]
        observed_sql = observed_sql.replace("'${exp}'", "'e'").replace("'${run}'", "'r'")
        observed_sql = observed_sql.replace("${step_from:sqlstring}", "0").replace("${step_to:sqlstring}", "2")
        observed_sql = observed_sql.replace("${rank:sqlstring}", "-2").replace("${microbatch:sqlstring}", "-2")
        observed_sql = observed_sql.replace("${layer:sqlstring}", "'0'")
        start = observed_sql.index("metric_sampler.router_stream_load_range(")
        open_paren = observed_sql.index("(", start)
        depth = 0
        for end in range(open_paren, len(observed_sql)):
            depth += (observed_sql[end] == "(") - (observed_sql[end] == ")")
            if depth == 0:
                break
        observed_sql = observed_sql[:start] + "stream_load_rows" + observed_sql[end + 1:]
        observed_sql = observed_sql.replace(" SETTINGS join_use_nulls = 0", "")
        rows = connection.execute(observed_sql + " SELECT step, series, value FROM observed").fetchall()
        keys = [(step, series) for step, series, _value in rows]
        assert len(keys) == len(set(keys))
        assert {series: value for step, series, value in rows if step == 1} == expected_step_one


def test_router_stream_load_counts_conserve_and_recompute_combined_shares() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript("""
            CREATE TABLE expert_counts (
                exp TEXT, run TEXT, attempt TEXT, step INTEGER, layer INTEGER, node TEXT, rank INTEGER,
                microbatch INTEGER, metric_id INTEGER, source_metric_id INTEGER, vals TEXT
            );
            CREATE TABLE metric_names (exp TEXT, run TEXT, attempt TEXT, metric_id INTEGER, name TEXT);
            CREATE TABLE metrics (metric_id INTEGER, name TEXT);
            CREATE TABLE latest_attempt_step (exp TEXT, run TEXT, step INTEGER, attempt TEXT);
            CREATE TABLE streams (exp TEXT, run TEXT, attempt TEXT, stream_id INTEGER, name TEXT);
            CREATE TABLE router_counts (
                exp TEXT, run TEXT, attempt TEXT, step INTEGER, layer INTEGER, rank INTEGER,
                microbatch INTEGER, vals TEXT
            );
            CREATE TABLE stream_load_rows (
                exp TEXT, run TEXT, attempt TEXT, step INTEGER, layer INTEGER, rank INTEGER, microbatch INTEGER,
                expert INTEGER, stream_id INTEGER, stream_name TEXT, tokens INTEGER, stream_tokens INTEGER,
                expert_tokens INTEGER, layer_tokens INTEGER, expert_share REAL, stream_mix REAL,
                stream_share REAL, max_vio REAL
            );
            INSERT INTO latest_attempt_step VALUES
                ('e', 'r', 0, 'old'), ('e', 'r', 1, 'retry'), ('e', 'r', 2, 'retry');
            INSERT INTO metrics VALUES (187, 'router/stream_tokens_per_expert');
            INSERT INTO streams VALUES
                ('e', 'r', 'old', 0, 'long'), ('e', 'r', 'old', 1, 'short'),
                ('e', 'r', 'retry', 0, 'short'), ('e', 'r', 'retry', 1, 'long'),
                ('e', 'r', 'retry', 2, 'new');
        """)
        for attempt in ("old", "retry"):
            connection.execute(
                "INSERT INTO metric_names VALUES ('e','r',?,187,'router/stream_tokens_per_expert')", (attempt,)
            )
        source_rows = [
            ("old", 0, "n0", 0, 0, [2, 5, 1, 3]),
            ("retry", 1, "n0", 0, 0, [3, 1, 0, 0, 0, 0]),
            ("retry", 1, "n1", 0, 0, [1, 0, 0, 0, 0, 0]),
            ("retry", 1, "n0", 0, 1, [1, 3, 2, 2, 2, 0]),
            ("retry", 1, "n0", 1, 0, [0, 0, 0, 4, 0, 0]),
            ("retry", 2, "n0", 0, 0, [0, 0, 0, 0, 0, 0]),
            # A stale retry and an alias row must not inflate the latest original assignment counts.
            ("old", 1, "n0", 0, 0, [99, 0, 0, 0]),
            ("retry", 1, "n0", 0, 0, [99, 0, 0, 0]),
        ]
        for attempt, step, node, rank, microbatch, values in source_rows:
            source_metric_id = 1 if attempt == "retry" and values[0] == 99 else 0
            connection.execute(
                "INSERT INTO expert_counts VALUES ('e','r',?,?,?,?,?,?,187,?,?)",
                (attempt, step, 0, node, rank, microbatch, source_metric_id, json.dumps(values)),
            )
        for step, rank, microbatch, values in [
            (0, 0, 0, [7, 4]), (1, 0, 0, [5, 0]), (1, 0, 1, [6, 4]),
            (1, 1, 0, [0, 4]), (2, 0, 0, [0, 0]),
        ]:
            connection.execute(
                "INSERT INTO router_counts VALUES ('e','r','retry',?,0,?,?,?)",
                (step, rank, microbatch, json.dumps(values)),
            )

        expanded = _sqlite_router_stream_load_select(connection)
        rows = connection.execute(expanded, {"rank_filter": -2, "microbatch_filter": -2}).fetchall()
        assert len(rows) == 28
        connection.executemany("INSERT INTO stream_load_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        old_step = [row for row in rows if row[3] == 0]
        assert len(old_step) == 4
        assert {row[2] for row in old_step} == {"old"}
        assert {(row[8], row[9]) for row in old_step} == {(0, "long"), (1, "short")}
        retry_step = [row for row in rows if row[3] == 1]
        assert {(row[8], row[9]) for row in retry_step} == {(0, "short"), (1, "long"), (2, "new")}

        # Each rank/microbatch/expert margin equals the independent router count vector.
        actual = connection.execute(
            "SELECT rank, microbatch, expert, sum(tokens) FROM stream_load_rows WHERE step = 1 "
            "GROUP BY rank, microbatch, expert ORDER BY rank, microbatch, expert"
        ).fetchall()
        expected = []
        for _step, rank, microbatch, values in connection.execute(
            "SELECT step, rank, microbatch, vals FROM router_counts WHERE step = 1"
        ):
            expected.extend((rank, microbatch, expert, count) for expert, count in enumerate(json.loads(values), 1))
        assert actual == expected

        # Dashboard all-rank/all-microbatch aggregation recomputes ratios from summed counts.
        combined = connection.execute("""
            WITH grouped AS (
                SELECT step, layer, expert, stream_id, stream_name, sum(tokens) AS tokens
                FROM stream_load_rows WHERE step = 1
                GROUP BY step, layer, expert, stream_id, stream_name
            ), totals AS (
                SELECT *, sum(tokens) OVER (PARTITION BY step, layer, stream_id) AS stream_tokens,
                       sum(tokens) OVER (PARTITION BY step, layer, expert) AS expert_tokens,
                       avg(tokens) OVER (PARTITION BY step, layer, stream_id) AS stream_mean,
                       max(tokens) OVER (PARTITION BY step, layer, stream_id) AS stream_max
                FROM grouped
            )
            SELECT expert, stream_id, stream_name, tokens, stream_tokens, expert_tokens,
                   tokens * 1.0 / nullif(stream_tokens, 0), tokens * 1.0 / nullif(expert_tokens, 0),
                   (stream_max - stream_mean) / nullif(stream_mean, 0)
            FROM totals ORDER BY expert, stream_id
        """).fetchall()
        assert [row[:6] for row in combined] == [
            (1, 0, "short", 5, 11, 11), (1, 1, "long", 4, 6, 11), (1, 2, "new", 2, 2, 11),
            (2, 0, "short", 6, 11, 8), (2, 1, "long", 2, 6, 8), (2, 2, "new", 0, 2, 8),
        ]
        assert [row[6] for row in combined] == pytest.approx([5 / 11, 2 / 3, 1, 6 / 11, 1 / 3, 0])
        assert [row[7] for row in combined] == pytest.approx([5 / 11, 4 / 11, 2 / 11, 3 / 4, 1 / 4, 0])
        assert [row[8] for row in combined] == pytest.approx([1 / 11, 1 / 3, 1, 1 / 11, 1 / 3, 1])

        zero_rows = connection.execute(
            expanded, {"rank_filter": 0, "microbatch_filter": 0}
        ).fetchall()
        step_two = [row for row in zero_rows if row[3] == 2]
        assert len(step_two) == 6
        assert len(step_two) == 6
        assert all(row[11:17] == (0, 0, 0, None, None, None) for row in step_two)
        assert all(row[17] is None for row in step_two)
        low_volume = connection.execute(
            expanded, {"rank_filter": 0, "microbatch_filter": 0}
        ).fetchall()
        stream_one = next(row for row in low_volume if row[3] == 1 and row[8] == 1)
        assert stream_one[17] == pytest.approx(1.0)

        _assert_router_stream_load_panels(connection)

def test_experts_dashboard_exposes_router_stream_load_measures_and_selectors() -> None:
    dashboard = load_dashboard("experts.json")
    variables = {variable["name"] for variable in dashboard["templating"]["list"]}
    panels = {panel["id"]: panel for panel in dashboard["panels"]}

    assert {"stream_id", "expert_id"} <= variables
    assert panels[13]["title"] == "Router stream mix within ${expert_id:text}"
    for panel_id, numerator, denominator in [
        (12, "tokens", "nullIf(stream_tokens, 0)"),
        (13, "tokens", "nullIf(expert_tokens, 0)"),
        (14, "stream_max_tokens - stream_mean_tokens", "nullIf(stream_mean_tokens, 0)"),
        (15, "stream_tokens", "AS value"),
    ]:
        panel = panels[panel_id]
        query = panel["targets"][0]["rawSql"]
        assert "metric_sampler.router_stream_load_range(" in query
        assert numerator in query
        assert denominator in query
        assert "rank_filter = ifNull(toInt32OrNull(${rank:sqlstring}), -3)" in query
        assert "microbatch_filter = ifNull(toInt32OrNull(${microbatch:sqlstring}), -3)" in query
        assert query.endswith(_PANEL_SETTINGS)


def test_dashboards_call_the_per_rank_outlier_views_with_the_selected_run() -> None:
    call = re.compile(
        r"metric_sampler\.(?:grad|timing)_per_rank\$\{\w+\}(\(exp_filter = '\$\{exp\}', run_filter = '\$\{run\}'\))?"
    )
    calls = [
        match for path in DASHBOARDS_DIR.glob("*.json") for match in call.finditer(path.read_text(encoding="utf-8"))
    ]

    assert len(calls) == _PER_RANK_OUTLIER_CALL_COUNT
    assert all(match[1] for match in calls)


@pytest.mark.parametrize(
    ("filename", "panel_id", "column", "value"),
    [("experts.json", 3, "expert", "load"), ("loss-distributions.json", 14, "bucket", "mean")],
)
def test_heatmaps_use_numeric_steps_and_preserve_empty_cells(filename, panel_id, column, value) -> None:
    dashboard = load_dashboard(filename)
    panel = next(panel for panel in dashboard["panels"] if panel["id"] == panel_id)
    target = panel["targets"][0]

    assert target["format"] == 1
    assert target["rawSql"].startswith("WITH cells AS (SELECT step, ")
    assert " AS width SELECT step, " in target["rawSql"]
    assert target["rawSql"].endswith(f"ORDER BY step, {column} SETTINGS {_PANEL_SETTINGS}")
    assert "toDateTime" not in target["rawSql"]
    assert STEP_RANGE in target["rawSql"]
    assert panel["transformations"] == [
        {
            "id": "groupingToMatrix",
            "options": {"rowField": "step", "columnField": column, "valueField": value, "emptyValue": "null"},
        },
        {
            "id": "convertFieldType",
            "options": {"fields": {}, "conversions": [{"targetField": f"step\\{column}", "destinationType": "number"}]},
        },
    ]


@pytest.mark.parametrize(
    ("rank", "expected"),
    [("0", [(-1, -1, -1), (0, 0, 0), (0, 1, 3)]),
     ("1", [(-1, -1, -1), (1, 0, 0), (1, 1, 3)]),
     ("", [(-1, -1, -1)])],
)
def test_step_details_select_a_rank_and_shared_rows_without_truncation(rank, expected) -> None:
    dashboard = load_dashboard("run-overview.json")
    variables = {variable["name"]: variable for variable in dashboard["templating"]["list"]}
    assert variables["detail_rank"]["current"]["value"] == "0"
    panel = next(panel for panel in dashboard["panels"] if panel["id"] == 5)
    sql = panel["targets"][0]["rawSql"]
    assert "LIMIT" not in sql
    assert "AND s.microbatch" not in sql and "AND s.layer" not in sql
    predicate = re.search(r"\(s\.rank = -1 OR s\.rank = toInt16OrNull\(\$\{detail_rank:sqlstring\}\)\)", sql).group()
    predicate = predicate.replace("toInt16OrNull(${detail_rank:sqlstring})", "CAST(NULLIF(?, '') AS INTEGER)")
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("CREATE TABLE observations (rank INTEGER, microbatch INTEGER, layer INTEGER)")
        connection.executemany("INSERT INTO observations VALUES (?,?,?)", [
            (-1, -1, -1), (0, 0, 0), (0, 1, 3), (1, 0, 0), (1, 1, 3),
        ])
        assert connection.execute(
            f"SELECT * FROM observations AS s WHERE {predicate} ORDER BY rank, microbatch, layer", (rank,),
        ).fetchall() == expected


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ({95: 100, 96: 80, 97: 75, 98: 60}, (100.0, 60.0, 75.0, 0.25, 0.2, 1)),
        ({95: 100, 96: 80, 97: 100, 98: 80}, (100.0, 80.0, 100.0, 0.0, 0.2, 1)),
        ({3: 100, 4: 80}, (None, None, None, None, 0.2, 0)),
        ({95: 100, 96: 80, 97: 75}, (None, None, None, None, None, 0)),
        ({95: 100, 96: 0, 97: 50, 98: 0}, (100.0, 0.0, 50.0, None, 1.0, 1)),
        ({95: 0, 96: 0, 97: 0, 98: 0}, (0.0, 0.0, 0.0, None, None, 1)),
    ],
)
def test_capacity_results_distinguish_drops_input_padding_and_unavailable(sql_observations, rows, expected):
    names = {
        3: "dispatcher/tokens_per_local_expert", 4: "dispatcher/tokens_per_local_expert_no_pad",
        95: "dispatcher/assignments_before_capacity", 96: "dispatcher/valid_assignments_before_capacity",
        97: "dispatcher/assignments_after_capacity", 98: "dispatcher/valid_assignments_after_capacity",
    }
    for metric_id, value in rows.items():
        sql_observations.execute("INSERT INTO metrics VALUES (?,?)", (metric_id, names[metric_id]))
        for attempt, multiplier in (("new", 1), ("old", 100)):
            sql_observations.execute(
                "INSERT INTO expert_counts (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
                "VALUES ('e','r',?,1,0,'node',0,0,?,?)",
                (attempt, metric_id, json.dumps([value * multiplier])),
            )
            sql_observations.execute("INSERT INTO metric_names (exp,run,attempt,metric_id,name) VALUES ('e','r',?,?,?)",
                                     (attempt, metric_id, names[metric_id]))
    actual = sql_observations.execute(fixture_view_sql("capacity_selected"), {
        "exp_filter": "e", "run_filter": "r", "rank_filter": -2, "microbatch_filter": -2,
    }).fetchall()

    assert len(actual) == 1
    assert actual[0][:4] == ("e", "r", 1, 0)
    assert actual[0][4:] == pytest.approx(expected, nan_ok=True)
    assert sql_observations.execute(fixture_view_sql("capacity_selected"), {
        "exp_filter": "e", "run_filter": "r", "rank_filter": 1, "microbatch_filter": -2,
    }).fetchall() == []


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        # Before/after repeat router/tokens_per_expert, and valid after repeats valid before.
        ({
            1: ("router/tokens_per_expert", [60, 40], 0),
            95: ("dispatcher/assignments_before_capacity", [], 1),
            96: ("dispatcher/valid_assignments_before_capacity", [50, 30], 0),
            97: ("dispatcher/assignments_after_capacity", [], 1),
            98: ("dispatcher/valid_assignments_after_capacity", [], 96),
        }, (100.0, 80.0, 100.0, 0.0, 0.2, 1)),
        # The legacy count without padding repeats the one with padding.
        ({
            3: ("dispatcher/tokens_per_local_expert", [60, 40], 0),
            4: ("dispatcher/tokens_per_local_expert_no_pad", [], 3),
        }, (None, None, None, None, 0.0, 0)),
    ],
)
def test_capacity_results_resolve_aliased_rows_to_the_rows_they_repeat(sql_observations, rows, expected):
    # Stored rows are (values, alias_metric_id); an aliased row keeps no values.
    for metric_id, (name, values, alias) in rows.items():
        sql_observations.execute("INSERT INTO metrics VALUES (?,?)", (metric_id, name))
        sql_observations.execute(
            "INSERT INTO metric_names (exp,run,attempt,metric_id,name) VALUES ('e','r','new',?,?)", (metric_id, name)
        )
        sql_observations.execute(
            "INSERT INTO expert_counts (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals,"
            "alias_metric_id) "
            "VALUES ('e','r','new',1,0,'node',0,0,?,?,?)", (metric_id, json.dumps(values), alias),
        )
    actual = sql_observations.execute(fixture_view_sql("capacity_selected"), {
        "exp_filter": "e", "run_filter": "r", "rank_filter": -2, "microbatch_filter": -2,
    }).fetchall()

    assert len(actual) == 1
    assert actual[0][:4] == ("e", "r", 1, 0)
    assert actual[0][4:] == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize(
    ("rank", "microbatch", "mean", "root_mean_square", "std", "amax", "median_amax"),
    [(-2, -2, 5, math.sqrt(30), math.sqrt(5), 8, 8),
     (-2, 0, 3, math.sqrt(10), 1, 4, 4),
     (0, -2, 5, math.sqrt(34), 3, 8, 8),
     (1, 1, 6, 6, 0, 6, 6)],
)
def test_activation_selection_precedes_reduction_and_rank_max_precedes_median(
    sql_observations, rank, microbatch, mean, root_mean_square, std, amax, median_amax,
):
    for metric_id, name in ((200, "act/residual/moments"), (201, "act/residual/amax")):
        sql_observations.execute(
            "INSERT INTO metric_names (exp,run,attempt,metric_id,name) VALUES ('e','r','new',?,?)", (metric_id, name)
        )
        sql_observations.execute("INSERT INTO metrics VALUES (?,?)", (metric_id, name))
    for observed_rank, observed_microbatch, value in ((0, 0, 2), (0, 1, 8), (1, 0, 4), (1, 1, 6)):
        sql_observations.execute(
            "INSERT INTO router_vectors (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
            "VALUES ('e','r','new',1,0,'node',?,?,200,?)",
            (observed_rank, observed_microbatch, json.dumps([float(value), float(value * value), 0, 1])),
        )
        sql_observations.execute(
            "INSERT INTO scalars (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
            "VALUES ('e','r','new',1,0,'node',?,?,201,?)",
            (observed_rank, observed_microbatch, json.dumps([value])),
        )
    rows = sql_observations.execute(fixture_view_sql("activation_selected"), {
        "exp_filter": "e", "run_filter": "r", "rank_filter": rank, "microbatch_filter": microbatch,
    }).fetchall()

    count = (2 if rank == -2 else 1) * (2 if microbatch == -2 else 1)
    assert len(rows) == 1
    assert rows[0][:5] == ("e", "r", 1, 0, "residual")
    assert rows[0][5:14] == pytest.approx((count, mean, root_mean_square, std, 0, amax, median_amax,
                                        127 - math.log2(amax), math.log2(448) - math.log2(amax)))
    assert rows[0][14:] == (0, None, None, None, None)


def _skewness_and_kurtosis(population):
    mean = sum(population) / len(population)
    variance = sum((v - mean) ** 2 for v in population) / len(population)
    return (sum((v - mean) ** 3 for v in population) / len(population) / variance ** 1.5,
            sum((v - mean) ** 4 for v in population) / len(population) / variance ** 2)


@pytest.mark.parametrize(
    ("observations", "power_sum_ranks", "expected"),
    [({0: [1.0, 2.0], 1: [3.0, 10.0]}, (0, 1), _skewness_and_kurtosis([1.0, 2.0, 3.0, 10.0])),
     ({0: [1.0, 2.0], 1: [3.0, 10.0]}, (0,), (None, None)),
     ({0: [2.0, 2.0], 1: [2.0]}, (0, 1), (None, None))],
    ids=["every_observation", "power_sums_of_some_observations", "zero_variance"],
)
def test_activation_skewness_and_kurtosis_are_the_central_moments_of_every_selected_observation(
    sql_observations, observations, power_sum_ranks, expected,
):
    names = {200: "act/residual/moments", 201: "act/residual/amax", 202: "act/residual/power_sums"}
    for metric_id, name in names.items():
        sql_observations.execute(
            "INSERT INTO metric_names (exp,run,attempt,metric_id,name) VALUES ('e','r','new',?,?)", (metric_id, name)
        )
        sql_observations.execute("INSERT INTO metrics VALUES (?,?)", (metric_id, name))
    for rank, values in observations.items():
        rows = [(200, [sum(values), sum(v ** 2 for v in values), 0, len(values)])]
        if rank in power_sum_ranks:
            rows.append((202, [sum(v ** 3 for v in values), sum(v ** 4 for v in values)]))
        for metric_id, words in rows:
            sql_observations.execute(
                "INSERT INTO router_vectors (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
                "VALUES ('e','r','new',1,0,'node',?,0,?,?)",
                (rank, metric_id, json.dumps(words)),
            )
        sql_observations.execute(
            "INSERT INTO scalars (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
            "VALUES ('e','r','new',1,0,'node',?,0,201,?)",
            (rank, json.dumps([max(values)])),
        )
    rows = sql_observations.execute(fixture_view_sql("activation_selected"), {
        "exp_filter": "e", "run_filter": "r", "rank_filter": -2, "microbatch_filter": -2,
    }).fetchall()
    assert len(rows) == 1
    assert rows[0][17:] == pytest.approx(expected)


@pytest.mark.parametrize("rank,microbatch", [(-2, -2), (0, -2), (-2, 0), (0, 0), (3, 0)])
def test_exact_activation_histograms_replace_only_the_same_observation_and_preserve_full_values(
    sql_observations, rank, microbatch,
):
    connection = sql_observations
    for point, base in (("residual", 48), ("attn_out", 52)):
        for metric_id, suffix in ((base + 1, "exp_hist"), (base + 2, "token_rms_hist"),
                                  (102 if point == "residual" else 103, "counts")):
            name = f"act/{point}/{suffix}"
            connection.execute("INSERT INTO metrics VALUES (?,?)", (metric_id, name))
            for attempt in ("new", "old"):
                connection.execute("INSERT INTO metric_names (exp,run,attempt,metric_id,name) VALUES ('e','r',?,?,?)",
                                   (attempt, metric_id, name))
    observations = [
        ("node", 0, 0, 0, "residual", 2**24 + 1, True),
        ("node", 1, 0, 0, "residual", 5, False),
        ("node", 0, 1, 0, "residual", 7, False),
        ("node", 0, 0, 1, "residual", 11, False),
        ("node", 0, 0, 0, "attn_out", 13, False),
        ("other-node", 0, 0, 0, "residual", 17, False),
    ]
    expected = {}
    for node, observed_rank, observed_microbatch, layer, point, count, exact in observations:
        for histogram, suffix, bin_index in (("exp", 1, 0), ("token_rms", 2, 63)):
            values = [0] * 64
            values[bin_index] = count if not exact else 100
            metric_id = (48 if point == "residual" else 52) + suffix
            connection.execute(
                "INSERT INTO router_vectors (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
                "VALUES ('e','r','new',1,?,?,?,?,?,?)",
                (layer, node, observed_rank, observed_microbatch, metric_id, json.dumps(values)),
            )
            values[bin_index] = count
            if rank in (-2, observed_rank) and microbatch in (-2, observed_microbatch):
                expected[(node, observed_rank, observed_microbatch, layer, point, histogram)] = (
                    values if exact else [], [] if exact else values, int(exact),
                )
        if exact:
            values = [0, count] + [count] + [0] * 63 + [0] * 63 + [count]
            connection.execute(
                "INSERT INTO expert_counts (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
                "VALUES ('e','r','new',1,?,?,?,?,102,?)",
                (layer, node, observed_rank, observed_microbatch, json.dumps(values)),
            )
    for attempt, source in (("old", 0), ("new", 19)):
        connection.execute(
            "INSERT INTO expert_counts (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals,"
            "source_metric_id) "
            "VALUES ('e','r',?,1,0,'node',1,0,102,?,?)",
            (attempt, json.dumps([0, 999] + [999] * 128), source),
        )
    actual = connection.execute(fixture_view_sql("activation_histograms_selected"), {
        "exp_filter": "e", "run_filter": "r", "rank_filter": rank, "microbatch_filter": microbatch,
    }).fetchall()
    assert {row[4:10]: (json.loads(row[13]), json.loads(row[14]), row[15]) for row in actual} == expected
    assert all(row[:4] == ("e", "r", "new", 1) and row[10:13] == (-40, 1, 64) for row in actual)
    assert connection.execute(fixture_view_sql("activation_histograms_selected"), {
        "exp_filter": "other", "run_filter": "r", "rank_filter": rank, "microbatch_filter": microbatch,
    }).fetchall() == []


@pytest.mark.parametrize("rank", [-2, 0, 1])
def test_activation_moments_use_exact_counts_and_label_partial_historical_precision(sql_observations, rank):
    connection = sql_observations
    for metric_id, name in ((48, "act/residual/moments"), (51, "act/residual/amax"), (102, "act/residual/counts")):
        connection.execute("INSERT INTO metrics VALUES (?,?)", (metric_id, name))
        connection.execute("INSERT INTO metric_names (exp,run,attempt,metric_id,name) VALUES ('e','r','new',?,?)",
                           (metric_id, name))
    finite = 2**24 + 1
    connection.execute("INSERT INTO expert_counts (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
                       "VALUES ('e','r','new',1,0,'node',0,0,102,?)",
                       (json.dumps([3, finite] + [finite] + [0] * 63 + [finite] + [0] * 63),))
    for observed_rank, moments, amax in ((0, [2.0 * 2**24, 4.0 * 2**24, 2, 2**24], 2), (1, [15.0, 45.0, 1, 5], 3)):
        connection.execute("INSERT INTO router_vectors (exp,run,attempt,step,layer,node,rank,microbatch,"
                           "metric_id,vals) "
                           "VALUES ('e','r','new',1,0,'node',?,0,48,?)", (observed_rank, json.dumps(moments)))
        connection.execute("INSERT INTO scalars (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
                           "VALUES ('e','r','new',1,0,'node',?,0,51,?)", (observed_rank, json.dumps([amax])))
    rows = connection.execute(fixture_view_sql("activation_selected"), {
        "exp_filter": "e", "run_filter": "r", "rank_filter": rank, "microbatch_filter": 0,
    }).fetchall()
    assert len(rows) == 1
    row = rows[0]
    expected_elements = finite + 5 if rank == -2 else finite if rank == 0 else 5
    expected_sum = 2.0 * 2**24 + 15 if rank == -2 else 2.0 * 2**24 if rank == 0 else 15
    assert row[5] == expected_elements
    assert row[6] == pytest.approx(expected_sum / expected_elements)
    assert row[9] == (4 if rank == -2 else 3 if rank == 0 else 1)
    assert row[14:17] == ((1, finite, 3) if rank == 0 else (0, None, None))
    assert row[17:] == (None, None)


@pytest.mark.parametrize("dashboard_name", ["activations", "attention", "experts", "node-rank"])
def test_raw_dashboards_offer_only_explicit_all_and_recorded_rank_microbatch_selections(dashboard_name):
    dashboard = load_dashboard(f"{dashboard_name}.json")
    variables = {variable["name"]: variable for variable in dashboard["templating"]["list"]}
    for name in ("rank", "microbatch"):
        assert variables[name]["current"]["value"] == "-2"
        assert "Legacy aggregate (not separable)" in variables[name]["query"]
        assert f"SELECT DISTINCT {name} AS v FROM metric_sampler.scalars" in variables[name]["query"]
    for panel in [nested for group in dashboard["panels"] for nested in group.get("panels", [group])]:
        assert RANK in panel["targets"][0]["rawSql"]


def test_trends_break_at_missing_steps_without_expanding_the_step_range():
    for path in DASHBOARDS_DIR.glob("*.json"):
        dashboard = json.loads(path.read_text(encoding="utf-8"))
        for panel in dashboard["panels"]:
            if panel["type"] != "trend":
                continue
            sql = panel["targets"][0]["rawSql"]
            # A thinned trend breaks at the end of a step bucket, which is the step itself up to the row limit; a
            # run that captures every Nth step breaks only where more than N steps lie between captured ones.
            assert ("marked.next_step > marked.step + ${capture_interval}" in sql
                    or "marked.next_step > marked.last_step + ${capture_interval}" in sql)
            assert "CAST(NULL AS Nullable(Float64))" in sql
            assert "numbers(" not in sql and "WITH FILL" not in sql
            assert panel["fieldConfig"]["defaults"]["custom"]["spanNulls"] is False
            variable = next(item for item in dashboard["templating"]["list"] if item["name"] == "capture_interval")
            assert variable["hide"] == 2
            assert variable["query"].startswith(
                "SELECT greatest(max(capture_interval), 1) FROM metric_sampler.node_manifests WHERE ")


@pytest.mark.parametrize(
    ("filename", "panel_ids"),
    [("experts.json", [1, 2, 11]), ("loss-distributions.json", [2, 9, 10, 11, 13]), ("activations.json", [1, 2, 3, 8])],
)
def test_long_trends_keep_every_point_up_to_the_row_limit_then_each_bucket_extremes(filename, panel_ids):
    dashboard = load_dashboard(filename)
    panels = {panel["id"]: panel for panel in dashboard["panels"]}
    for panel_id in panel_ids:
        sql = panels[panel_id]["targets"][0]["rawSql"]

        assert (
            "(SELECT count() FROM shown) AS shown_points, "
            f"toUInt32(ceil(shown_points / {_ROW_LIMIT // 2})) AS width, "
        ) in sql
        assert (
            f"step AS first_step, step AS last_step FROM shown WHERE shown_points <= {_ROW_LIMIT} UNION ALL "
        ) in sql
        assert "(argMax(step, " in sql and "arrayDistinct([(argMin(step, " in sql
        assert (
            f"WHERE shown_points > {_ROW_LIMIT} GROUP BY " in sql
            and ", intDiv(step, width)) ARRAY JOIN points AS point" in sql
        )
        assert "leadInFrame(first_step, 1, first_step)" in sql and "SELECT last_step + 1 AS step, " in sql


def test_long_heatmaps_keep_every_step_up_to_the_row_limit_then_bucket_the_steps():
    experts = load_dashboard("experts.json")
    load = next(panel for panel in experts["panels"] if panel["id"] == 3)["targets"][0]["rawSql"]
    losses = load_dashboard("loss-distributions.json")
    position = next(panel for panel in losses["panels"] if panel["id"] == 14)["targets"][0]["rawSql"]

    assert f"(SELECT (count(), min(step)) FROM cells) AS meta, toUInt32(ceil(meta.1 / {_ROW_LIMIT})) AS width" in load
    assert f"FROM cells WHERE meta.1 <= {_ROW_LIMIT} UNION ALL" in load
    assert "argMax(v, dev) AS v" in load and f"FROM cells WHERE meta.1 > {_ROW_LIMIT} GROUP BY bucket, expert" in load
    assert (
        f"(SELECT count() FROM cells) AS cell_count, toUInt32(ceil(cell_count / {_ROW_LIMIT})) AS width" in position
    )
    assert f"FROM cells WHERE cell_count <= {_ROW_LIMIT} UNION ALL" in position
    assert f"avg(mean) AS mean FROM cells WHERE cell_count > {_ROW_LIMIT} GROUP BY step, bucket" in position


def test_loss_per_rank_switches_to_the_band_its_overrides_style_past_the_row_limit():
    dashboard = load_dashboard("loss-per-rank.json")
    panel = next(panel for panel in dashboard["panels"] if panel["id"] == 1)
    sql = panel["targets"][0]["rawSql"]
    overrides = panel["fieldConfig"]["overrides"]
    styled = {override["matcher"]["options"] for override in overrides}
    filled = {item["value"] for override in overrides for item in override["properties"]
              if item["id"] == "custom.fillBelowTo"}

    assert styled | filled == {"max over ranks", "median over ranks", "min over ranks"}
    assert (
        "['max over ranks', 'median over ranks', 'min over ranks'] AS band_name, [high, mid, low] AS band_value" in sql
    )
    assert f"FROM ranks WHERE (SELECT count() FROM ranks) <= {_ROW_LIMIT} UNION ALL" in sql
    assert sql.count(f"(SELECT count() FROM ranks) > {_ROW_LIMIT}") == 2
    assert "toString(rank) IN (${show:sqlstring})" in sql
    show = next(variable for variable in dashboard["templating"]["list"] if variable["name"] == "show")
    assert show["multi"] is True and show["current"]["value"] == ["none"]
    assert show["query"].startswith("SELECT 'none' AS __value, 'none' AS __text UNION ALL")


@pytest.mark.parametrize(
    ("capture_interval", "expected"),
    [(1, [(3, 0, None), (2, 1, None), (29, 2, None)]), (2, [(2, 1, None), (29, 2, None)])],
    ids=["every step", "every second step"],
)
def test_gap_marker_preserves_series_and_marks_only_the_first_missing_step(capture_interval, expected):
    dashboard = load_dashboard("activations.json")
    sql = dashboard["panels"][0]["targets"][0]["rawSql"]
    missing = "SELECT " + sql.split(" FROM marked UNION ALL SELECT ", 1)[1].split(" SETTINGS ", 1)[0]
    missing = missing.replace("Nullable(Float64)", "REAL").replace("${capture_interval}", str(capture_interval))
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(
            "CREATE TABLE marked (step INTEGER, next_step INTEGER, last_step INTEGER, layer INTEGER, rms REAL)")
        # Every step is kept up to the row limit (last_step = step); past it layer 2 keeps two extremes of each
        # 10-step bucket, which has data through step 19, 28 and 47: the line breaks only after the bucket before
        # the missing one.
        connection.executemany("INSERT INTO marked VALUES (?,?,?,?,?)", [
            (1, 2, 1, 0, 2.0), (2, 4, 2, 0, 3.0), (4, 4, 4, 0, 4.0), (1, 1000000, 1, 1, 5.0),
            (12, 10, 19, 2, 1.0), (17, 20, 19, 2, 9.0), (25, 41, 28, 2, 3.0), (44, 41, 47, 2, 4.0),
        ])

        assert connection.execute(missing + " ORDER BY layer, step").fetchall() == expected


@pytest.mark.parametrize(
    "dashboard_name",
    ["activations", "attention", "experts", "loss-distributions", "run-overview", "run-compare",
     "training-diagnostics"],
)
def test_partitioned_trends_keep_empty_frame_fields_without_fabricating_values(dashboard_name):
    dashboard = load_dashboard(f"{dashboard_name}.json")
    for panel in dashboard["panels"]:
        partition = [item for item in panel.get("transformations", []) if item["id"] == "partitionByValues"]
        if not partition:
            continue
        group = partition[0]["options"]["fields"][0]
        projection = panel["targets"][0]["rawSql"].split(" FROM (", 1)[0]
        value = projection.rsplit(" AS ", 1)[1]
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.create_function("toString", 1, lambda value: None if value is None else str(value))
            connection.execute(f"CREATE TABLE measured (step INTEGER, {group} TEXT, {value} REAL)")
            connection.executemany("INSERT INTO measured VALUES (?,?,?)", [
                (None, None, None), (12, "0", 2.5), (13, "0", None), (14, "0", 0.0),
            ])

            result = connection.execute(projection + " FROM measured")
            assert [column[0] for column in result.description] == ["step", group, value]
            assert result.fetchall() == [
                (None, "No data", None), (12, "0", 2.5), (13, "0", None), (14, "0", 0.0),
            ]


def test_logged_source_step_is_nullable_for_historical_records():
    schema = SCHEMA_SQL.read_text()
    assert "source_step     Nullable(UInt32) DEFAULT NULL" in schema
    assert (
        "ALTER TABLE metric_sampler.logged ADD COLUMN IF NOT EXISTS source_step Nullable(UInt32) DEFAULT NULL" in schema
    )
    assert "l.source_step AS source_step" in view_body(schema, "logged_per_step")


def test_logged_per_step_joins_attempt_starts_per_distinct_attempt_and_step_not_per_logged_row():
    view = view_body(SCHEMA_SQL.read_text(), "logged_per_step")

    assert "FROM (SELECT DISTINCT exp, run, attempt, step FROM metric_sampler.logged) AS p" in view
    assert "FROM metric_sampler.logged AS ll" not in view


@pytest.mark.parametrize("has_microbatch_routes", [False, True])
def test_router_counts_prefer_current_attempt_microbatch_routes_without_double_counting(
    sql_observations, has_microbatch_routes,
):
    sql_observations.execute(
        "INSERT INTO expert_counts (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
        "VALUES ('e','r','new',1,0,'node',0,-1,2,'[99,99]')"
    )
    sql_observations.execute(
        "INSERT INTO expert_counts (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
        "VALUES ('e','r','old',1,0,'node',0,0,1,'[999,999]')"
    )
    if has_microbatch_routes:
        sql_observations.execute(
            "INSERT INTO expert_counts (exp,run,attempt,step,layer,node,rank,microbatch,metric_id,vals) "
            "VALUES ('e','r','new',1,0,'node',0,0,1,'[2,0]')"
        )
    rows = sql_observations.execute(
        fixture_view_sql("router_counts_selected"), {"exp_filter": "e", "run_filter": "r"}
    ).fetchall()

    assert rows == [("e", "r", "new", 1, 0, "node", 0,
                     0 if has_microbatch_routes else -1,
                     1 if has_microbatch_routes else 2,
                     "[2,0]" if has_microbatch_routes else "[99,99]")]

class _HealthArgMax:
    def __init__(self):
        self.value = None
        self.version = None

    def step(self, value, version):
        parsed = json.loads(version) if isinstance(version, str) and version.startswith("[") else version
        if self.version is None or parsed > self.version:
            self.value = value
            self.version = parsed

    def finalize(self):
        return self.value


class _HealthSumIf:
    def __init__(self):
        self.total = 0

    def step(self, value, condition):
        if condition:
            self.total += value

    def finalize(self):
        return self.total


def _health_sqlite_view(name: str) -> str:
    sql = view_body(SCHEMA_SQL.read_text(), name).split(";", 1)[0]
    sql = sql.replace("metric_sampler.", "").replace(" FINAL", "")
    for family in _FAMILY_TABLES:
        sql = sql.replace(f"rows_per_family['{family}']", f"coalesce(json_extract(rows_per_family, '$.{family}'), 0)")
    for index in range(1, 6):
        sql = sql.replace(f"health_values.{index}", f"json_extract(health_values, '$[{index - 1}]')")
    return sql


def test_health_ready_batches_require_exact_counts_and_ignore_duplicate_markers() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.create_aggregate("argMax", 2, _HealthArgMax)
        connection.create_aggregate("sumIf", 2, _HealthSumIf)
        connection.create_function("extract", 2, lambda value, pattern: re.search(pattern, value).group(1))
        connection.execute(
            "CREATE TABLE batches (exp, run, attempt, batch, first_step, last_step, "
            "rows_per_family, collected_at, ingested_at)"
        )
        connection.execute("CREATE TABLE processed_objects (kind, object, rows)")
        connection.executemany("INSERT INTO batches VALUES (?,?,?,?,?,?,?,?,?)", [
            ("e", "r", "a", "ready", 10, 12, json.dumps({"scalars": 4, "expert_counts": 2}), 1, 1),
            ("e", "r", "a", "ready", 10, 12, json.dumps({"scalars": 4, "expert_counts": 2}), 1, 2),
            ("e", "r", "a", "partial", 13, 15, json.dumps({"scalars": 4, "expert_counts": 2}), 2, 2),
            ("e", "r", "a", "wrong", 16, 18, json.dumps({"scalars": 4}), 3, 3),
            ("e", "r", "a", "logged", 19, 20, json.dumps({"logged": 9}), 4, 4),
            ("e", "r", "b", "ready", 10, 12, json.dumps({"scalars": 4, "expert_counts": 2}), 1, 1),
        ])
        connection.executemany("INSERT INTO processed_objects VALUES (?,?,?)", [
            ("scalars", "family=scalars/exp=e/run=r/attempt=a/part-10-12-ready.parquet", 4),
            ("scalars", "family=scalars/exp=e/run=r/attempt=a/part-10-12-ready.parquet", 4),
            ("expert_counts", "family=expert_counts/exp=e/run=r/attempt=a/part-10-12-ready.parquet", 2),
            ("scalars", "family=scalars/exp=e/run=r/attempt=a/part-13-15-partial.parquet", 4),
            ("scalars", "family=scalars/exp=e/run=r/attempt=a/part-16-18-wrong.parquet", 5),
            ("logged", "family=logged/exp=e/run=r/attempt=a/part-19-20-logged.parquet", 9),
        ])
        assert connection.execute(_health_sqlite_view("health_ready_batches")).fetchall() == [
            ("e", "r", "a", "ready", 10, 12, 1),
        ]


def test_health_publication_uses_complete_generation_and_latest_attempt_without_adding_snapshots() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.create_function("tuple", -1, lambda *values: json.dumps(values))
        connection.create_aggregate("argMax", 2, _HealthArgMax)
        connection.execute(
            "CREATE TABLE health_snapshots (exp,run,attempt,batch,window_first,generation,step,nodes_present,"
            "nodes_expected,contributors_total,expected_contributors_total,identical_violations)"
        )
        connection.execute("CREATE TABLE health_refreshes (exp,run,attempt,batch,window_first,generation,completed_at)")
        connection.execute("CREATE TABLE latest_attempt_step (exp,run,step,attempt)")
        connection.executemany("INSERT INTO latest_attempt_step VALUES (?,?,?,?)", [("e","r",10,"a"), ("e","r",11,"b")])
        connection.executemany("INSERT INTO health_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            ("e","r","a","first",10,"g1",10,1,2,5,6,1),
            ("e","r","a","first",10,"g1",10,1,2,5,6,1),
            ("e","r","a","late",10,"g2",10,2,2,9,10,2),
            ("e","r","a","partial",10,"g3",10,99,99,999,999,999),
            ("e","r","a","first",10,"g1",11,1,2,5,6,1),
            ("e","r","b","new",11,"g4",11,1,1,3,3,0),
        ])
        connection.executemany("INSERT INTO health_refreshes VALUES (?,?,?,?,?,?,?)", [
            ("e","r","a","first",10,"g1",100),
            ("e","r","a","first",10,"g1",100),
            ("e","r","a","late",10,"g2",200),
        ])
        sql = _health_sqlite_view("health_serving_per_step") + " ORDER BY step"
        assert connection.execute(sql).fetchall() == [("e","r","a",10,2,2,9,10,2,200)]
        connection.execute("INSERT INTO health_refreshes VALUES (?,?,?,?,?,?,?)", ("e","r","b","new",11,"g4",300))
        assert connection.execute(sql).fetchall() == [
            ("e","r","a",10,2,2,9,10,2,200), ("e","r","b",11,1,1,3,3,0,300),
        ]


@pytest.mark.parametrize(("first", "last", "expected"), [
    (0, 0, [(0, 0)]), (0, 9, [(0, 9)]), (1, 11, [(1, 10), (11, 11)]),
    (184851, 184870, [(184851, 184860), (184861, 184870)]),
    (4294967290, 4294967295, [(4294967290, 4294967295)]),
])
def test_health_refresh_windows_are_bounded_and_completed_windows_are_skipped(first, last, expected) -> None:
    sql = _health_sqlite_view("health_pending_windows")
    sql = sql.replace(
        "ARRAY JOIN range(toUInt64(first_step), toUInt64(last_step) + 1, toUInt64(10)) AS window_start",
        "CROSS JOIN json_each(step_range(first_step, last_step + 1, 10)) AS window",
    ).replace("window_start", "window.value")
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.create_function("toUInt32", 1, int)
        connection.create_function("toUInt64", 1, int)
        connection.create_function("least", 2, min)
        connection.create_function(
            "step_range", 3, lambda start, stop, size: json.dumps(list(range(start, stop, size)))
        )
        connection.execute(
            "CREATE TABLE health_ready_batches (exp,run,attempt,batch,first_step,last_step,collected_at)"
        )
        connection.execute("CREATE TABLE health_refreshes (exp,run,attempt,batch,window_first)")
        connection.execute("INSERT INTO health_ready_batches VALUES ('e','r','a','b',?,?,1)", (first,last))
        result = connection.execute(sql + " ORDER BY window_first").fetchall()
        assert result == [("e","r","a","b",start,stop,1) for start,stop in expected]
        connection.execute("INSERT INTO health_refreshes VALUES ('e','r','a','b',?)", (first,))
        assert connection.execute(sql + " ORDER BY window_first").fetchall() == result[1:]


def test_health_refresh_publishes_only_after_snapshot_and_keeps_old_generations() -> None:
    publish = health_loop.REFRESH_STATEMENT
    statements = (
        health_loop.PENDING_STATEMENT + health_loop.SNAPSHOT_STATEMENT + publish + health_loop.GENERATION_STATEMENT
    )
    assert "ORDER BY collected_at DESC, exp, run, attempt, batch, window_first DESC" in health_loop.PENDING_STATEMENT
    assert "now64(6)" in publish
    assert "collected_at" not in publish
    assert "max(completed_at) + toIntervalMicrosecond(1)" in publish
    schema = SCHEMA_SQL.read_text()
    for table in ("health_snapshots", "health_refreshes"):
        definition = schema.split(f"CREATE TABLE IF NOT EXISTS metric_sampler.{table}", 1)[1].split(";", 1)[0]
        assert "ENGINE = ReplacingMergeTree" in definition
        assert "window_first, generation" in definition
    assert "DELETE" not in statements


class _MaxByVersion:
    """sqlite stand-in for ClickHouse argMax over tuple() versions encoded as JSON lists."""

    def __init__(self):
        self.value = None
        self.version = None

    def step(self, value, version):
        parsed = json.loads(version)
        if self.version is None or parsed > self.version:
            self.value, self.version = value, parsed

    def finalize(self):
        return self.value


@pytest.fixture
def attempts_db():
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.create_function("tuple", -1, lambda *values: json.dumps(values))
        connection.create_function("toDateTime64", 2, lambda seconds, precision: "1970-01-01 00:00:00.000")
        connection.create_aggregate("argMax", 2, _MaxByVersion)
        connection.execute("CREATE TABLE attempt_start_index (exp, run, attempt, started_at)")
        connection.execute(f"CREATE VIEW attempt_starts AS {fixture_view_sql('attempt_starts')}")
        connection.execute("CREATE TABLE step_attempts (exp, run, step, attempt)")
        connection.execute("CREATE TABLE streams (exp, run, attempt, stream_id, name)")
        connection.execute("CREATE TABLE logged (exp, run, attempt, step, name, value, source_step)")
        yield connection


# (attempt, its window start times, its steps): the name of the attempt that replaces the other sorts before it:
# past restart 9, past a job-id digit rollover and for a later job with a smaller id.
_RESTARTS = [
    [("100-9", ["2026-10-01 10:00:00.000", "2026-10-01 11:00:00.000"], [1, 2, 3]),
     ("100-10", ["2026-10-01 12:00:00.000"], [3, 4])],
    [("999999-0", ["2026-10-01 10:00:00.000"], [1, 2, 3]),
     ("1000000-0", ["2026-10-02 09:00:00.000"], [3, 4])],
    [("774990-0", ["2026-10-01 10:00:00.000"], [1, 2, 3]),
     ("655300-0", ["2026-10-01 12:00:00.000"], [3, 4])],
]


@pytest.mark.parametrize("attempts", _RESTARTS, ids=["restart_10", "job_id_rollover", "smaller_job_id"])
def test_latest_attempt_of_a_step_is_the_latest_started(attempts_db, attempts):
    for attempt, starts, steps in attempts:
        attempts_db.executemany(
            "INSERT INTO attempt_start_index VALUES ('e','r',?,?)", [(attempt, at) for at in starts]
        )
        attempts_db.executemany("INSERT INTO step_attempts VALUES ('e','r',?,?)", [(step, attempt) for step in steps])
    (old, _, _), (new, _, _) = attempts

    rows = attempts_db.execute(fixture_view_sql("latest_attempt_step") + " ORDER BY step").fetchall()

    assert rows == [("e", "r", 1, old), ("e", "r", 2, old), ("e", "r", 3, new), ("e", "r", 4, new)]


def test_attempt_without_node_windows_ranks_before_every_started_attempt(attempts_db):
    attempts_db.execute("INSERT INTO attempt_start_index VALUES ('e','r','100-1','2026-10-01 10:00:00.000')")
    attempts_db.executemany("INSERT INTO step_attempts VALUES ('e','r',1,?)", [("100-1",), ("100-2",)])

    assert attempts_db.execute(fixture_view_sql("latest_attempt_step")).fetchall() == [("e", "r", 1, "100-1")]


def test_stream_names_and_logged_values_come_from_the_latest_started_attempt(attempts_db):
    attempts_db.executemany("INSERT INTO attempt_start_index VALUES ('e','r',?,?)", [
        ("100-9", "2026-10-01 10:00:00.000"), ("100-10", "2026-10-01 12:00:00.000"),
    ])
    attempts_db.executemany("INSERT INTO streams VALUES ('e','r',?,0,?)", [("100-9", "old"), ("100-10", "new")])
    attempts_db.executemany("INSERT INTO logged VALUES ('e','r',?,5,'lr',?,5)", [("100-9", 1.0), ("100-10", 2.0)])

    assert attempts_db.execute(fixture_view_sql("stream_names")).fetchall() == [("e", "r", 0, "new")]
    assert attempts_db.execute(fixture_view_sql("logged_per_step")).fetchall() == [("e", "r", 5, "lr", 2.0, 5)]


def test_capture_intervals_reach_the_manifests_table_and_space_the_windows_of_complete_steps():
    schema = SCHEMA_SQL.read_text(encoding="utf-8")
    ingest = INGEST_SQL.read_text(encoding="utf-8")

    assert ("ALTER TABLE metric_sampler.node_manifests ADD COLUMN IF NOT EXISTS capture_interval UInt32 DEFAULT 1;"
            in schema)
    assert "nodes_in_job, capture_interval, written_at, raw, source_object, ingested_at)" in ingest
    assert "greatest(JSONExtractUInt(json, 'capture_interval'), 1) AS capture_interval," in ingest
    complete_steps = serving_view("complete_steps")
    assert "argMax(capture_interval, written_at) AS capture_interval" in complete_steps
    assert "if(i = 1, start, ((ws[i - 1]).2) + capture_interval)" in complete_steps


def test_run_steps_refresh_answers_the_summary_loops_with_each_runs_newest_step_arrival_and_ranks():

    refresh = SERVING_SQL.split("CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.run_steps_refresh\n", 1)[1]
    refresh = refresh.split(";\n", 1)[0]
    assert refresh.startswith("REFRESH EVERY 1 MINUTE TO metric_sampler.run_steps_latest\n")
    assert "SELECT exp, run, max(step) AS max_step, max(max_ingested_at) AS ingested\n" \
           "    FROM metric_sampler.step_index\n    GROUP BY exp, run" in refresh
    assert "SELECT exp, run, uniqExact(rank) AS rank_count\n    FROM metric_sampler.node_manifests\n" \
           "    ARRAY JOIN ranks AS rank\n    GROUP BY exp, run" in refresh
    assert "LEFT JOIN" in refresh and refresh.endswith("SETTINGS join_use_nulls = 0")


@pytest.mark.parametrize("family", ["scalars", "expert_counts", "router_vectors", "logged"])
def test_family_inserts_filter_objects_once_over_their_listing_and_rows_by_the_listed_objects(family):
    ingest = INGEST_SQL.read_text(encoding="utf-8")
    insert = ingest.split(f"INSERT INTO metric_sampler.{family}\n", 1)[1].split(";\n", 1)[0]

    listing = insert.split("WHERE _path IN (\n", 1)[1]
    assert listing.startswith(
        f"    SELECT _path FROM s3(obs_ms, filename = 'family={family}/**/*.parquet', format = 'One')\n"
    )
    assert (
        f"WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = '{family}')" in listing
    )
    assert "IN (SELECT exp, run, attempt, batch FROM metric_sampler.batches))" in listing
    assert insert.count("_path NOT IN") == 1


def test_health_window_view_is_health_per_step_of_one_attempts_steps():
    schema = SCHEMA_SQL.read_text(encoding="utf-8")
    general = schema.split("CREATE OR REPLACE VIEW metric_sampler.health_per_step AS\n", 1)[1].split(";\n", 1)[0]
    window = schema.split("CREATE OR REPLACE VIEW metric_sampler.health_per_step_window AS\n", 1)[1].split(";\n", 1)[0]

    attempt = "exp = {exp_filter:String} AND run = {run_filter:String} AND attempt = {attempt_filter:String}"
    restored = window.replace(
        "{exp_filter:String} AS exp, {run_filter:String} AS run, {attempt_filter:String} AS attempt,",
        "exp, run, attempt,",
    ).replace(f"\n        WHERE {attempt}\n    )", "\n    )")
    for table in _FAMILY_TABLES:
        restored = restored.replace(
            f"    FROM metric_sampler.{table} AS f FINAL\n"
            f"    WHERE f.{attempt.replace(' AND ', ' AND f.')}\n"
            "      AND f.step >= {step_from:UInt32} AND f.step <= {step_to:UInt32}\n"
            "    GROUP BY step, node, metric_id, source_metric_id, layer, microbatch",
            f"    FROM metric_sampler.{table} FINAL\n"
            "    GROUP BY exp, run, attempt, step, node, metric_id, source_metric_id, layer, microbatch",
        )
    assert restored == general
    assert window.count(f"WHERE {attempt}\n    )") == 2
