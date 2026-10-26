CREATE DATABASE IF NOT EXISTS metric_sampler;

-- Objects of the retired v1 schema, dropped on upgrade.
DROP VIEW IF EXISTS metric_sampler.expert_load;
DROP VIEW IF EXISTS metric_sampler.loss_per_step;
DROP VIEW IF EXISTS metric_sampler.runs;
DROP TABLE IF EXISTS metric_sampler.raw;

-- Family tables hold one row per (node, step, rank, microbatch, layer, metric_id). ReplacingMergeTree(ingested_at)
-- keeps a re-ingested object from duplicating rows; queries fold duplicates away with FINAL.

CREATE TABLE IF NOT EXISTS metric_sampler.scalars
(
    exp                     LowCardinality(String),
    run                     LowCardinality(String),
    attempt                 LowCardinality(String),
    node                    LowCardinality(String),
    step                    UInt32,
    rank                    Int16,
    microbatch              Int16,
    layer                   Int16,
    metric_id               UInt16,
    source_metric_id        UInt16,
    contributors            UInt8,
    expected_contributors   UInt8,
    identical               Bool,
    verbose                 Bool,
    values                  Array(Float32) CODEC(ZSTD(3)),
    source_object           LowCardinality(String),
    ingested_at             DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, metric_id, source_metric_id, node, step, rank, microbatch, layer);

ALTER TABLE metric_sampler.scalars MODIFY COLUMN values Array(Float32) CODEC(ZSTD(3));
-- The sort key puts `node` before `step`: a step filter alone would read every node's granules.
-- Parts written before the index existed need ALTER TABLE ... MATERIALIZE INDEX.
ALTER TABLE metric_sampler.scalars ADD INDEX IF NOT EXISTS step_minmax step TYPE minmax GRANULARITY 1;

CREATE TABLE IF NOT EXISTS metric_sampler.expert_counts
(
    exp                     LowCardinality(String),
    run                     LowCardinality(String),
    attempt                 LowCardinality(String),
    node                    LowCardinality(String),
    step                    UInt32,
    rank                    Int16,
    microbatch              Int16,
    layer                   Int16,
    metric_id               UInt16,
    source_metric_id        UInt16,
    contributors            UInt8,
    expected_contributors   UInt8,
    identical               Bool,
    verbose                 Bool,
    -- Non-zero: the row has no values and repeats this metric's row of the same rank, layer and
    -- microbatch word for word (MetricSpec.alias_of).
    alias_metric_id         UInt16 DEFAULT 0,
    values                  Array(UInt32) CODEC(T64, LZ4),
    source_object           LowCardinality(String),
    ingested_at             DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (exp, run)
-- Panels and summaries read one metric over a step range, and monitor metrics keep a parameter index
-- of up to ~14k in `layer`: metric_id and step lead the key so such a read is one contiguous range.
ORDER BY (exp, run, attempt, metric_id, source_metric_id, step, layer, node, rank, microbatch);

ALTER TABLE metric_sampler.expert_counts ADD COLUMN IF NOT EXISTS alias_metric_id UInt16 DEFAULT 0 AFTER verbose;

CREATE TABLE IF NOT EXISTS metric_sampler.router_vectors
(
    exp                     LowCardinality(String),
    run                     LowCardinality(String),
    attempt                 LowCardinality(String),
    node                    LowCardinality(String),
    step                    UInt32,
    rank                    Int16,
    microbatch              Int16,
    layer                   Int16,
    metric_id               UInt16,
    source_metric_id        UInt16,
    contributors            UInt8,
    expected_contributors   UInt8,
    identical               Bool,
    verbose                 Bool,
    values                  Array(Float32) CODEC(LZ4),
    source_object           LowCardinality(String),
    ingested_at             DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (exp, run)
-- Same sort key as expert_counts, for the same reason.
ORDER BY (exp, run, attempt, metric_id, source_metric_id, step, layer, node, rank, microbatch);

-- The raw values use LZ4 instead of ZSTD(1); parts written earlier keep ZSTD(1) until they are merged.
ALTER TABLE metric_sampler.expert_counts MODIFY COLUMN values Array(UInt32) CODEC(T64, LZ4);
ALTER TABLE metric_sampler.router_vectors MODIFY COLUMN values Array(Float32) CODEC(LZ4);

-- A collector catching up a backlog lands faster than merges keep up: inserts are delayed only after 3000 parts per
-- partition (default 1000) and rejected after 6000. Merges stop at 4 GiB parts: rewriting larger parts saturates the
-- data disk together with ingest, and a finished run re-sorted into parts of a few GB each (disjoint metric and step
-- ranges) needs no merging at all.
ALTER TABLE metric_sampler.expert_counts MODIFY SETTING parts_to_delay_insert = 3000, parts_to_throw_insert = 6000, max_bytes_to_merge_at_max_space_in_pool = 4294967296;
ALTER TABLE metric_sampler.router_vectors MODIFY SETTING parts_to_delay_insert = 3000, parts_to_throw_insert = 6000, max_bytes_to_merge_at_max_space_in_pool = 4294967296;
-- Reset so that merges of large parts are not restricted while many merge slots are busy.
ALTER TABLE metric_sampler.expert_counts RESET SETTING number_of_free_entries_in_pool_to_lower_max_size_of_merge;
ALTER TABLE metric_sampler.router_vectors RESET SETTING number_of_free_entries_in_pool_to_lower_max_size_of_merge;

-- One row per (step, name) logged by training monitors; it has no owning node.
CREATE TABLE IF NOT EXISTS metric_sampler.logged
(
    exp             LowCardinality(String),
    run             LowCardinality(String),
    attempt         LowCardinality(String),
    step            UInt32,
    source_step     Nullable(UInt32) DEFAULT NULL,
    name            LowCardinality(String),
    value           Float64,
    source_object   LowCardinality(String),
    ingested_at     DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, name, step);
ALTER TABLE metric_sampler.logged ADD COLUMN IF NOT EXISTS source_step Nullable(UInt32) DEFAULT NULL;

CREATE TABLE IF NOT EXISTS metric_sampler.metrics
(
    exp             LowCardinality(String),
    run             LowCardinality(String),
    attempt         LowCardinality(String),
    metric_id       UInt16,
    name            LowCardinality(String),
    family          LowCardinality(String),
    value_type      LowCardinality(String),
    node_merge      LowCardinality(String),
    unit            LowCardinality(String),
    description     String,
    source_object   LowCardinality(String),
    ingested_at     DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, metric_id);

-- Bins are 0 for a metric without them. ADD COLUMN IF NOT EXISTS keeps this idempotent for an already-deployed
-- table.
ALTER TABLE metric_sampler.metrics ADD COLUMN IF NOT EXISTS bins_start Float64 DEFAULT 0;
ALTER TABLE metric_sampler.metrics ADD COLUMN IF NOT EXISTS bins_width Float64 DEFAULT 0;
ALTER TABLE metric_sampler.metrics ADD COLUMN IF NOT EXISTS bins_count UInt32 DEFAULT 0;
ALTER TABLE metric_sampler.metrics ADD COLUMN IF NOT EXISTS summarize_experts Bool DEFAULT false;
ALTER TABLE metric_sampler.metrics ADD COLUMN IF NOT EXISTS detailed Bool DEFAULT false;

ALTER TABLE metric_sampler.metrics ADD COLUMN IF NOT EXISTS job_identical Bool DEFAULT false;

-- One row per data stream of an attempt, from its streams.json.
CREATE TABLE IF NOT EXISTS metric_sampler.streams
(
    exp             LowCardinality(String),
    run             LowCardinality(String),
    attempt         LowCardinality(String),
    stream_id       UInt16,
    name            String,
    source_object   LowCardinality(String),
    ingested_at     DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, stream_id);

CREATE TABLE IF NOT EXISTS metric_sampler.monitor_parameters
(
    exp             LowCardinality(String),
    run             LowCardinality(String),
    attempt         LowCardinality(String),
    kind            LowCardinality(String),
    rank            Int16,
    `index`         UInt16,
    name            String,
    source_object   LowCardinality(String),
    ingested_at     DateTime64(3) DEFAULT now64(3),
    CONSTRAINT valid_kind CHECK kind IN ('parameter', 'activation', 'optimizer_group', 'learning_rate_group'),
    CONSTRAINT valid_rank CHECK rank >= -1,
    CONSTRAINT nonempty_name CHECK name != ''
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, kind, rank, index);

CREATE TABLE IF NOT EXISTS metric_sampler.monitor_expert_ownership
(
    exp                 LowCardinality(String),
    run                 LowCardinality(String),
    attempt             LowCardinality(String),
    rank                Int16,
    layer               Int16,
    global_expert_ids   Array(UInt16),
    source_object       LowCardinality(String),
    ingested_at         DateTime64(3) DEFAULT now64(3),
    CONSTRAINT valid_rank CHECK rank >= 0,
    CONSTRAINT unique_experts CHECK length(arrayDistinct(global_expert_ids)) = length(global_expert_ids)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, rank, layer);

-- One row per node-window manifest. Idempotency is handled by processed_objects, so a plain MergeTree is enough here.
CREATE TABLE IF NOT EXISTS metric_sampler.node_manifests
(
    exp                         LowCardinality(String),
    run                         LowCardinality(String),
    attempt                     LowCardinality(String),
    node                        LowCardinality(String),
    schema_version              UInt32,
    first_step                  UInt32,
    last_step                   UInt32,
    ranks                       Array(UInt16),
    steps                       Array(UInt32),
    rows_per_family             Map(String, UInt64),
    missing_ranks               Map(String, Array(UInt16)),
    dropped_steps_per_rank      Map(String, UInt32),
    torn_slots_per_rank         Map(String, UInt32),
    late_slots_per_rank         Map(String, UInt32),
    identical_violations        UInt32,
    written_at                  DateTime64(3),
    raw                         String,
    source_object               LowCardinality(String),
    ingested_at                 DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, node, first_step);

-- Job size the node process saw. DEFAULT 0 so a manifest without it stops contributing to the greatest() in
-- health_per_step rather than faking a job of size zero.
ALTER TABLE metric_sampler.node_manifests ADD COLUMN IF NOT EXISTS nodes_in_job UInt32 DEFAULT 0;

-- Ranks whose ring never attached before the deadline. DEFAULT [] so a manifest without it reads as "nothing
-- absent" rather than unknown.
ALTER TABLE metric_sampler.node_manifests ADD COLUMN IF NOT EXISTS absent_ranks Array(UInt16) DEFAULT [];

-- The training captures every capture_interval-th step, so consecutive windows are that many steps apart (see
-- complete_steps). DEFAULT 1 for manifests from before capture intervals, which captured every step.
ALTER TABLE metric_sampler.node_manifests ADD COLUMN IF NOT EXISTS capture_interval UInt32 DEFAULT 1;

-- One row per batch manifest written by the collector.
CREATE TABLE IF NOT EXISTS metric_sampler.batches
(
    exp             LowCardinality(String),
    run             LowCardinality(String),
    attempt         LowCardinality(String),
    schema_version  UInt32,
    batch           String,
    first_step      UInt32,
    last_step       UInt32,
    source_windows  Array(String),
    nodes           Array(String),
    rows_per_family Map(String, UInt64),
    collected_at    DateTime64(3),
    raw             String,
    source_object   LowCardinality(String),
    ingested_at     DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, batch);

ALTER TABLE metric_sampler.batches ADD COLUMN IF NOT EXISTS logged_dropped_rows UInt64 DEFAULT 0;

-- One row per uploaded activation dump: a small index over the heavy parquet file, which is not ingested and is
-- read directly with s3() or read_dump.
CREATE TABLE IF NOT EXISTS metric_sampler.dumps
(
    exp             LowCardinality(String),
    run             LowCardinality(String),
    attempt         LowCardinality(String),
    step            UInt32,
    rank            Int16,
    rows            UInt64,
    bytes           UInt64,
    points          Array(String),
    layers          Array(Int32),
    tokens          UInt32,
    s3_key          String,
    raw             String,
    source_object   LowCardinality(String),
    ingested_at     DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, step, rank);

CREATE TABLE IF NOT EXISTS metric_sampler.processed_objects
(
    kind            LowCardinality(String),
    object          String,
    rows            UInt64,
    processed_at    DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (kind, object);

-- The rows each ingest statement wrote per family object, kept by the insert triggers below, so
-- ingest.sql marks an object processed without reading it from S3 a second time. A trigger fires per
-- insert block, and every row of one statement has the statement's start as ingested_at: summing per
-- (object, ingested_at) gives the rows one statement wrote, and a retry after a failed statement
-- has a later ingested_at.
CREATE TABLE IF NOT EXISTS metric_sampler.ingested_objects
(
    kind            LowCardinality(String),
    object          String,
    ingested_at     DateTime64(3),
    rows            UInt64
)
ENGINE = SummingMergeTree
ORDER BY (kind, object, ingested_at);

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.scalars_ingested_objects
TO metric_sampler.ingested_objects AS
SELECT 'scalars' AS kind, source_object AS object, ingested_at, count() AS rows
FROM metric_sampler.scalars
GROUP BY source_object, ingested_at;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.expert_counts_ingested_objects
TO metric_sampler.ingested_objects AS
SELECT 'expert_counts' AS kind, source_object AS object, ingested_at, count() AS rows
FROM metric_sampler.expert_counts
GROUP BY source_object, ingested_at;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.router_vectors_ingested_objects
TO metric_sampler.ingested_objects AS
SELECT 'router_vectors' AS kind, source_object AS object, ingested_at, count() AS rows
FROM metric_sampler.router_vectors
GROUP BY source_object, ingested_at;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.logged_ingested_objects
TO metric_sampler.ingested_objects AS
SELECT 'logged' AS kind, source_object AS object, ingested_at, count() AS rows
FROM metric_sampler.logged
GROUP BY source_object, ingested_at;

-- A generation is visible only after its completion marker. An interrupted refresh cannot
-- replace the last complete result, and overlapping batches replace full values rather than sums.
CREATE TABLE IF NOT EXISTS metric_sampler.health_snapshots
(
    exp                         LowCardinality(String),
    run                         LowCardinality(String),
    attempt                     LowCardinality(String),
    batch                       String,
    window_first                UInt32,
    generation                  UUID,
    step                        UInt32,
    nodes_present               UInt64,
    nodes_expected              UInt64,
    contributors_total          UInt64,
    expected_contributors_total UInt64,
    identical_violations        UInt64
)
ENGINE = ReplacingMergeTree
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, batch, window_first, generation, step);

CREATE TABLE IF NOT EXISTS metric_sampler.health_refreshes
(
    exp             LowCardinality(String),
    run             LowCardinality(String),
    attempt         LowCardinality(String),
    batch           String,
    window_first    UInt32,
    window_last     UInt32,
    generation      UUID,
    steps           UInt64,
    completed_at    DateTime64(6)
)
ENGINE = ReplacingMergeTree(completed_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt, batch, window_first, generation);

-- Step metadata is a max-only rollup, so retries and backfills cannot double-count it.
CREATE TABLE IF NOT EXISTS metric_sampler.step_index
(
    exp                 LowCardinality(String),
    run                 LowCardinality(String),
    step                UInt32,
    attempt             SimpleAggregateFunction(max, String),
    max_ingested_at      SimpleAggregateFunction(max, DateTime64(3))
)
ENGINE = AggregatingMergeTree
PARTITION BY (exp, run)
ORDER BY (exp, run, step);

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.scalars_step_index
TO metric_sampler.step_index AS
SELECT exp, run, step, max(attempt) AS attempt, max(ingested_at) AS max_ingested_at
FROM metric_sampler.scalars
GROUP BY exp, run, step;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.expert_counts_step_index
TO metric_sampler.step_index AS
SELECT exp, run, step, max(attempt) AS attempt, max(ingested_at) AS max_ingested_at
FROM metric_sampler.expert_counts
GROUP BY exp, run, step;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.router_vectors_step_index
TO metric_sampler.step_index AS
SELECT exp, run, step, max(attempt) AS attempt, max(ingested_at) AS max_ingested_at
FROM metric_sampler.router_vectors
GROUP BY exp, run, step;

-- Backfill after the materialized views exist: overlap with live ingestion or a repeated schema application is
-- harmless because both aggregates are max. It spans every run's partition, past the default limit for one insert.
INSERT INTO metric_sampler.step_index
SELECT exp, run, step, max(attempt), max(ingested_at)
FROM metric_sampler.scalars
GROUP BY exp, run, step
SETTINGS max_partitions_per_insert_block = 0;

INSERT INTO metric_sampler.step_index
SELECT exp, run, step, max(attempt), max(ingested_at)
FROM metric_sampler.expert_counts
GROUP BY exp, run, step
SETTINGS max_partitions_per_insert_block = 0;

INSERT INTO metric_sampler.step_index
SELECT exp, run, step, max(attempt), max(ingested_at)
FROM metric_sampler.router_vectors
GROUP BY exp, run, step
SETTINGS max_partitions_per_insert_block = 0;

-- Attempts are named "<job id>-<restart count>", which does not order them as strings: "1-9" sorts after "1-10".
-- They are ordered by when each started: the earliest UTC time one of its node windows was written.
CREATE TABLE IF NOT EXISTS metric_sampler.attempt_start_index
(
    exp                 LowCardinality(String),
    run                 LowCardinality(String),
    attempt             LowCardinality(String),
    started_at          SimpleAggregateFunction(min, DateTime64(3))
)
ENGINE = AggregatingMergeTree
PARTITION BY (exp, run)
ORDER BY (exp, run, attempt);

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.node_manifests_attempt_start_index
TO metric_sampler.attempt_start_index AS
SELECT exp, run, attempt, min(written_at) AS started_at
FROM metric_sampler.node_manifests
GROUP BY exp, run, attempt;

INSERT INTO metric_sampler.attempt_start_index
SELECT exp, run, attempt, min(written_at)
FROM metric_sampler.node_manifests
GROUP BY exp, run, attempt
SETTINGS max_partitions_per_insert_block = 0;

CREATE OR REPLACE VIEW metric_sampler.attempt_starts AS
SELECT exp, run, attempt, min(started_at) AS started_at
FROM metric_sampler.attempt_start_index
GROUP BY exp, run, attempt;

-- Every attempt with data for a step, for latest_attempt_step.
CREATE TABLE IF NOT EXISTS metric_sampler.step_attempts
(
    exp                 LowCardinality(String),
    run                 LowCardinality(String),
    step                UInt32,
    attempt             LowCardinality(String)
)
ENGINE = ReplacingMergeTree
PARTITION BY (exp, run)
ORDER BY (exp, run, step, attempt);

-- GROUP BY rather than DISTINCT: the same rows for less CPU per inserted row.
-- To change this view on a live server, drop and create it while ingest is stopped.
CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.scalars_step_attempts
TO metric_sampler.step_attempts AS
SELECT exp, run, step, attempt
FROM metric_sampler.scalars
GROUP BY exp, run, step, attempt;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.expert_counts_step_attempts
TO metric_sampler.step_attempts AS
SELECT exp, run, step, attempt
FROM metric_sampler.expert_counts
GROUP BY exp, run, step, attempt;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.router_vectors_step_attempts
TO metric_sampler.step_attempts AS
SELECT exp, run, step, attempt
FROM metric_sampler.router_vectors
GROUP BY exp, run, step, attempt;

INSERT INTO metric_sampler.step_attempts
SELECT DISTINCT exp, run, step, attempt
FROM metric_sampler.scalars
SETTINGS max_partitions_per_insert_block = 0;

INSERT INTO metric_sampler.step_attempts
SELECT DISTINCT exp, run, step, attempt
FROM metric_sampler.expert_counts
SETTINGS max_partitions_per_insert_block = 0;

INSERT INTO metric_sampler.step_attempts
SELECT DISTINCT exp, run, step, attempt
FROM metric_sampler.router_vectors
SETTINGS max_partitions_per_insert_block = 0;

CREATE OR REPLACE VIEW metric_sampler.metric_names AS
SELECT exp, run, attempt, metric_id, name, bins_start, bins_width, bins_count, summarize_experts, detailed
FROM metric_sampler.metrics FINAL;

-- One row per (exp, run, stream_id), taken from the latest attempt that has written streams.json.
CREATE OR REPLACE VIEW metric_sampler.stream_names AS
SELECT s.exp AS exp, s.run AS run, s.stream_id AS stream_id, s.name AS name
FROM metric_sampler.streams AS s FINAL
INNER JOIN
(
    SELECT ls.exp AS exp, ls.run AS run, argMax(ls.attempt, tuple(coalesce(st.started_at, toDateTime64(0, 3)), ls.attempt)) AS attempt
    FROM metric_sampler.streams AS ls
    LEFT JOIN metric_sampler.attempt_starts AS st
        ON st.exp = ls.exp AND st.run = ls.run AND st.attempt = ls.attempt
    GROUP BY ls.exp, ls.run
) AS latest
    ON s.exp = latest.exp AND s.run = latest.run AND s.attempt = latest.attempt;

-- Latest-started attempt with data for each (exp, run, step): after a restart, an older attempt's
-- steps stay in the table but are hidden from the per-step views below, which all join on this.
-- A plain hash join: the default parallel one is slower for this view.
CREATE OR REPLACE VIEW metric_sampler.latest_attempt_step AS
SELECT sa.exp AS exp, sa.run AS run, sa.step AS step, argMax(sa.attempt, tuple(coalesce(st.started_at, toDateTime64(0, 3)), sa.attempt)) AS attempt
FROM metric_sampler.step_attempts AS sa
LEFT JOIN metric_sampler.attempt_starts AS st
    ON st.exp = sa.exp AND st.run = sa.run AND st.attempt = sa.attempt
GROUP BY sa.exp, sa.run, sa.step
SETTINGS join_algorithm = 'hash';

-- The metric_id sets below let MergeTree prune granules before FINAL and the joins. They
-- are only prefilters: the scoped registry joins and name predicates remain authoritative.
CREATE OR REPLACE VIEW metric_sampler.loss_per_step AS
SELECT
    s.exp AS exp,
    s.run AS run,
    s.step AS step,
    avg(s.values[1]) AS loss_mean,
    min(s.values[1]) AS loss_min,
    max(s.values[1]) AS loss_max
FROM metric_sampler.scalars AS s FINAL
INNER JOIN metric_sampler.latest_attempt_step AS las
    ON s.exp = las.exp AND s.run = las.run AND s.step = las.step AND s.attempt = las.attempt
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = s.exp AND mn.run = s.run AND mn.attempt = s.attempt AND mn.metric_id = s.metric_id
WHERE s.layer <= 255
  AND mn.name = 'loss/total'
  AND s.metric_id IN
      (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/total')
GROUP BY s.exp, s.run, s.step;

-- Prefer original per-microbatch routes when present; never add the bias-update step total again.
CREATE OR REPLACE VIEW metric_sampler.router_counts_selected AS
WITH preferred AS
(
    SELECT exp, run, attempt, step, layer, min(metric_id) AS selected_metric_id
    FROM metric_sampler.expert_counts
    WHERE expert_counts.layer <= 255
      AND ({exp_filter:String} = '' OR exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR run = {run_filter:String})
      AND metric_id IN (1, 2)
    GROUP BY exp, run, attempt, step, layer
)
SELECT
    ec.exp AS exp,
    ec.run AS run,
    ec.attempt AS attempt,
    ec.step AS step,
    ec.layer AS layer,
    ec.node AS node,
    ec.rank AS rank,
    ec.microbatch AS microbatch,
    ec.metric_id AS metric_id,
    ec.values AS `values`
FROM metric_sampler.expert_counts AS ec FINAL
INNER JOIN preferred AS source
    ON ec.exp = source.exp AND ec.run = source.run AND ec.attempt = source.attempt
   AND ec.step = source.step AND ec.layer = source.layer AND ec.metric_id = source.selected_metric_id
INNER JOIN metric_sampler.latest_attempt_step AS latest
    ON ec.exp = latest.exp AND ec.run = latest.run AND ec.step = latest.step AND ec.attempt = latest.attempt
WHERE ec.layer <= 255
  AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
  AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
  AND ec.metric_id IN (1, 2);

CREATE OR REPLACE VIEW metric_sampler.router_counts AS
SELECT * FROM metric_sampler.router_counts_selected(exp_filter = '', run_filter = '');

-- Per layer and step: router and dispatcher input assignments before capacity or EP exchange.
-- Sum over the selected ranks/microbatches; legacy node/step aggregates remain available.
-- Selected views use -2 for all observations and -1 for legacy aggregates.
CREATE OR REPLACE VIEW metric_sampler.layer_load_selected AS
WITH router_totals AS
(
    SELECT
        ec.exp AS exp, ec.run AS run, ec.step AS step, ec.layer AS layer,
        sumForEach(ec.values) AS router_tokens
    FROM metric_sampler.router_counts_selected(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}) AS ec
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND mn.name IN ('router/tokens_per_expert', 'router/local_tokens_before_bias_update')
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name IN ('router/tokens_per_expert', 'router/local_tokens_before_bias_update'))
    GROUP BY ec.exp, ec.run, ec.step, ec.layer
),
dispatcher_rows AS
(
    -- An aliased assignments row has no values: it repeats router/tokens_per_expert of its rank.
    SELECT
        ec.exp AS exp, ec.run AS run, ec.step AS step, ec.layer AS layer,
        countIf(mn.name = 'dispatcher/assignments_before_capacity') AS assignment_rows,
        if(maxIf(ec.alias_metric_id, mn.name = 'dispatcher/assignments_before_capacity') != 0,
            anyIf(ec.values, mn.name = 'router/tokens_per_expert'),
            anyIf(ec.values, mn.name = 'dispatcher/assignments_before_capacity')) AS assignments,
        anyIf(ec.values, mn.name = 'dispatcher/tokens_per_local_expert') AS legacy_tokens
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND mn.name IN ('dispatcher/assignments_before_capacity', 'dispatcher/tokens_per_local_expert', 'router/tokens_per_expert')
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name IN
              ('dispatcher/assignments_before_capacity', 'dispatcher/tokens_per_local_expert', 'router/tokens_per_expert'))
    GROUP BY ec.exp, ec.run, ec.step, ec.layer, ec.node, ec.rank, ec.microbatch
),
dispatcher_totals AS
(
    SELECT exp, run, step, layer,
        if(sum(assignment_rows) > 0,
            sumForEachIf(assignments, assignment_rows > 0),
            sumForEachIf(legacy_tokens, length(legacy_tokens) > 0)) AS dispatcher_tokens
    FROM dispatcher_rows
    GROUP BY exp, run, step, layer
)
SELECT
    router_totals.exp AS exp,
    router_totals.run AS run,
    router_totals.step AS step,
    router_totals.layer AS layer,
    router_totals.router_tokens AS router_tokens,
    dispatcher_totals.dispatcher_tokens AS dispatcher_tokens,
    arrayMax(router_totals.router_tokens) / (arraySum(router_totals.router_tokens) / length(router_totals.router_tokens)) AS max_load,
    arrayMin(router_totals.router_tokens) / (arraySum(router_totals.router_tokens) / length(router_totals.router_tokens)) AS min_load,
    arrayCount(x -> x = 0, router_totals.router_tokens) AS dead_experts
FROM router_totals
LEFT JOIN dispatcher_totals
    ON router_totals.exp = dispatcher_totals.exp AND router_totals.run = dispatcher_totals.run
   AND router_totals.step = dispatcher_totals.step AND router_totals.layer = dispatcher_totals.layer;

CREATE OR REPLACE VIEW metric_sampler.layer_load_per_step AS
SELECT * FROM metric_sampler.layer_load_selected(exp_filter = '', run_filter = '', rank_filter = -2, microbatch_filter = -2);

-- Capacity is measured with the same valid-token mask before and after filtering.
-- Legacy dispatcher metrics count input padding before capacity; they cannot recover drops.
CREATE OR REPLACE VIEW metric_sampler.capacity_selected AS
WITH observations AS
(
    SELECT
        ec.exp AS exp, ec.run AS run, ec.step AS step, ec.layer AS layer,
        ec.node AS node, ec.rank AS rank, ec.microbatch AS microbatch,
        -- An aliased row has no values: it repeats the row named by MetricSpec.alias_of of its rank.
        sumIf(arraySum(ec.values), mn.name = 'dispatcher/assignments_before_capacity')
            + countIf(mn.name = 'dispatcher/assignments_before_capacity' AND ec.alias_metric_id != 0)
            * sumIf(arraySum(ec.values), mn.name = 'router/tokens_per_expert') AS before_capacity,
        sumIf(arraySum(ec.values), mn.name = 'dispatcher/valid_assignments_before_capacity') AS valid_before_capacity,
        sumIf(arraySum(ec.values), mn.name = 'dispatcher/assignments_after_capacity')
            + countIf(mn.name = 'dispatcher/assignments_after_capacity' AND ec.alias_metric_id != 0)
            * sumIf(arraySum(ec.values), mn.name = 'router/tokens_per_expert') AS after_capacity,
        sumIf(arraySum(ec.values), mn.name = 'dispatcher/valid_assignments_after_capacity')
            + countIf(mn.name = 'dispatcher/valid_assignments_after_capacity' AND ec.alias_metric_id != 0)
            * sumIf(arraySum(ec.values), mn.name = 'dispatcher/valid_assignments_before_capacity') AS valid_after_capacity,
        sumIf(arraySum(ec.values), mn.name = 'dispatcher/tokens_per_local_expert') AS legacy_input,
        sumIf(arraySum(ec.values), mn.name = 'dispatcher/tokens_per_local_expert_no_pad')
            + countIf(mn.name = 'dispatcher/tokens_per_local_expert_no_pad' AND ec.alias_metric_id != 0)
            * sumIf(arraySum(ec.values), mn.name = 'dispatcher/tokens_per_local_expert') AS legacy_valid_input,
        max(mn.name = 'dispatcher/assignments_before_capacity')
            AND max(mn.name = 'dispatcher/valid_assignments_before_capacity')
            AND max(mn.name = 'dispatcher/assignments_after_capacity')
            AND max(mn.name = 'dispatcher/valid_assignments_after_capacity') AS capacity_observed,
        max(mn.name = 'dispatcher/tokens_per_local_expert')
            AND max(mn.name = 'dispatcher/tokens_per_local_expert_no_pad') AS legacy_observed
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND mn.name IN ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity',
        'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity',
        'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad', 'router/tokens_per_expert')
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name IN
              ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity',
               'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity',
               'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad',
               'router/tokens_per_expert'))
    GROUP BY ec.exp, ec.run, ec.step, ec.layer, ec.node, ec.rank, ec.microbatch
), totals AS
(
    SELECT exp, run, step, layer,
        sum(before_capacity) AS before_capacity, sum(valid_before_capacity) AS valid_before_capacity,
        sum(after_capacity) AS after_capacity, sum(valid_after_capacity) AS valid_after_capacity,
        sum(legacy_input) AS legacy_input, sum(legacy_valid_input) AS legacy_valid_input,
        min(capacity_observed) AS capacity_observed, min(legacy_observed) AS legacy_observed
    FROM observations
    GROUP BY exp, run, step, layer
)
SELECT exp, run, step, layer,
    CASE WHEN capacity_observed THEN before_capacity ELSE NULL END AS router_assignments,
    CASE WHEN capacity_observed THEN valid_after_capacity ELSE NULL END AS dispatched_no_pad,
    CASE WHEN capacity_observed THEN after_capacity ELSE NULL END AS dispatched,
    CASE WHEN capacity_observed THEN 1.0 - valid_after_capacity / nullIf(valid_before_capacity, 0)
        ELSE NULL END AS dropped_share,
    CASE WHEN capacity_observed THEN 1.0 - valid_before_capacity / nullIf(before_capacity, 0)
        WHEN legacy_observed THEN 1.0 - legacy_valid_input / nullIf(legacy_input, 0)
        ELSE NULL END AS padding_share,
    capacity_observed
FROM totals;

CREATE OR REPLACE VIEW metric_sampler.capacity_per_step AS
SELECT * FROM metric_sampler.capacity_selected(exp_filter = '', run_filter = '', rank_filter = -2, microbatch_filter = -2);

-- Per layer, step and expert index: router token total over nodes, for heatmaps.
CREATE OR REPLACE VIEW metric_sampler.expert_totals_selected AS
SELECT
    ec.exp AS exp,
    ec.run AS run,
    ec.step AS step,
    ec.layer AS layer,
    expert_index,
    sum(token_count) AS total_tokens
FROM metric_sampler.router_counts_selected(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}) AS ec
INNER JOIN metric_sampler.latest_attempt_step AS las
    ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
ARRAY JOIN ec.values AS token_count, arrayEnumerate(ec.values) AS expert_index
WHERE ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
  AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
  AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
  AND mn.name IN ('router/tokens_per_expert', 'router/local_tokens_before_bias_update')
  AND ec.metric_id IN
      (SELECT metric_id FROM metric_sampler.metrics WHERE name IN ('router/tokens_per_expert', 'router/local_tokens_before_bias_update'))
GROUP BY ec.exp, ec.run, ec.step, ec.layer, expert_index;

CREATE OR REPLACE VIEW metric_sampler.expert_totals_per_step AS
SELECT * FROM metric_sampler.expert_totals_selected(exp_filter = '', run_filter = '', rank_filter = -2, microbatch_filter = -2);

-- nodes_expected counts distinct nodes across every batch covering the step (arrayJoin, not
-- ARRAY JOIN, since that can't cross-join two differently-shaped arrays from one row), then takes
-- the greatest with node_manifests.nodes_in_job so a node whose aggregator never starts (absent
-- from every batch's `nodes`) still counts, via manifests other nodes reported.
--
-- identical_violations counts every IDENTICAL-metric row with identical = false, not only
-- verbose ones: the non-verbose path computes the same flag every step.
CREATE OR REPLACE VIEW metric_sampler.health_per_step AS
WITH identical_metrics AS
(
    SELECT DISTINCT exp, run, attempt, metric_id
    FROM metric_sampler.metrics FINAL
    WHERE node_merge = 'identical'
),
per_rank_metrics AS
(
    SELECT DISTINCT exp, run, attempt, metric_id
    FROM metric_sampler.metrics FINAL
    WHERE node_merge = 'per_rank'
),
-- With optimize_aggregation_in_order on the outer query, each family releases groups in storage
-- order. UNION ALL before aggregation would instead retain every slot's group in memory.
-- metric_id is constant within a group: the metric lookups run once per group, not once per row.
node_metric_health AS
(
    SELECT
        exp, run, attempt, step, node, metric_id, source_metric_id, layer, microbatch,
        sum(contributors) AS contributors_seen,
        -- PER_RANK rows repeat the node's expected rank count; summaries already store 1 per row.
        if(source_metric_id = 0 AND
           (exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM per_rank_metrics),
           max(expected_contributors), sum(expected_contributors)) AS contributors_expected,
        if((exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM identical_metrics),
           countIf(NOT identical), 0) AS identical_violations
    FROM metric_sampler.scalars FINAL
    GROUP BY exp, run, attempt, step, node, metric_id, source_metric_id, layer, microbatch
    UNION ALL
    SELECT
        exp, run, attempt, step, node, metric_id, source_metric_id, layer, microbatch,
        sum(contributors) AS contributors_seen,
        if(source_metric_id = 0 AND
           (exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM per_rank_metrics),
           max(expected_contributors), sum(expected_contributors)) AS contributors_expected,
        if((exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM identical_metrics),
           countIf(NOT identical), 0) AS identical_violations
    FROM metric_sampler.expert_counts FINAL
    GROUP BY exp, run, attempt, step, node, metric_id, source_metric_id, layer, microbatch
    UNION ALL
    SELECT
        exp, run, attempt, step, node, metric_id, source_metric_id, layer, microbatch,
        sum(contributors) AS contributors_seen,
        if(source_metric_id = 0 AND
           (exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM per_rank_metrics),
           max(expected_contributors), sum(expected_contributors)) AS contributors_expected,
        if((exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM identical_metrics),
           countIf(NOT identical), 0) AS identical_violations
    FROM metric_sampler.router_vectors FINAL
    GROUP BY exp, run, attempt, step, node, metric_id, source_metric_id, layer, microbatch
),
health AS
(
    SELECT
        exp, run, attempt, step,
        uniqExact(node) AS nodes_present,
        sum(contributors_seen) AS contributors_total,
        sum(contributors_expected) AS expected_contributors_total,
        sum(identical_violations) AS identical_violations
    FROM node_metric_health
    GROUP BY exp, run, attempt, step
),
expected_nodes AS
(
    SELECT exp, run, attempt, step, uniqExact(node) AS nodes_expected
    FROM
    (
        SELECT
            exp, run, attempt,
            arrayJoin(range(toUInt64(first_step), toUInt64(last_step) + 1)) AS step,
            arrayJoin(nodes) AS node
        FROM metric_sampler.batches
    )
    GROUP BY exp, run, attempt, step
),
manifest_nodes_in_job AS
(
    SELECT exp, run, attempt, step, max(nodes_in_job) AS nodes_in_job
    FROM
    (
        SELECT
            exp, run, attempt,
            arrayJoin(range(toUInt64(first_step), toUInt64(last_step) + 1)) AS step,
            nodes_in_job
        FROM metric_sampler.node_manifests
    )
    GROUP BY exp, run, attempt, step
)
SELECT
    health.exp AS exp,
    health.run AS run,
    health.attempt AS attempt,
    health.step AS step,
    health.nodes_present AS nodes_present,
    greatest(coalesce(expected_nodes.nodes_expected, 0), coalesce(manifest_nodes_in_job.nodes_in_job, 0)) AS nodes_expected,
    health.contributors_total AS contributors_total,
    health.expected_contributors_total AS expected_contributors_total,
    health.identical_violations AS identical_violations
FROM health
LEFT JOIN expected_nodes
    ON health.exp = expected_nodes.exp AND health.run = expected_nodes.run
   AND health.attempt = expected_nodes.attempt AND health.step = expected_nodes.step
LEFT JOIN manifest_nodes_in_job
    ON health.exp = manifest_nodes_in_job.exp AND health.run = manifest_nodes_in_job.run
   AND health.attempt = manifest_nodes_in_job.attempt AND health.step = manifest_nodes_in_job.step;

-- health_per_step of one attempt's steps, for the health refresh: the families are grouped without the exp, run and
-- attempt string keys, which is much cheaper.
CREATE OR REPLACE VIEW metric_sampler.health_per_step_window AS
WITH identical_metrics AS
(
    SELECT DISTINCT exp, run, attempt, metric_id
    FROM metric_sampler.metrics FINAL
    WHERE node_merge = 'identical'
),
per_rank_metrics AS
(
    SELECT DISTINCT exp, run, attempt, metric_id
    FROM metric_sampler.metrics FINAL
    WHERE node_merge = 'per_rank'
),
-- With optimize_aggregation_in_order on the outer query, each family releases groups in storage
-- order. UNION ALL before aggregation would instead retain every slot's group in memory.
-- metric_id is constant within a group: the metric lookups run once per group, not once per row.
node_metric_health AS
(
    SELECT
        {exp_filter:String} AS exp, {run_filter:String} AS run, {attempt_filter:String} AS attempt, step, node, metric_id, source_metric_id, layer, microbatch,
        sum(contributors) AS contributors_seen,
        -- PER_RANK rows repeat the node's expected rank count; summaries already store 1 per row.
        if(source_metric_id = 0 AND
           (exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM per_rank_metrics),
           max(expected_contributors), sum(expected_contributors)) AS contributors_expected,
        if((exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM identical_metrics),
           countIf(NOT identical), 0) AS identical_violations
    FROM metric_sampler.scalars AS f FINAL
    WHERE f.exp = {exp_filter:String} AND f.run = {run_filter:String} AND f.attempt = {attempt_filter:String}
      AND f.step >= {step_from:UInt32} AND f.step <= {step_to:UInt32}
    GROUP BY step, node, metric_id, source_metric_id, layer, microbatch
    UNION ALL
    SELECT
        {exp_filter:String} AS exp, {run_filter:String} AS run, {attempt_filter:String} AS attempt, step, node, metric_id, source_metric_id, layer, microbatch,
        sum(contributors) AS contributors_seen,
        if(source_metric_id = 0 AND
           (exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM per_rank_metrics),
           max(expected_contributors), sum(expected_contributors)) AS contributors_expected,
        if((exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM identical_metrics),
           countIf(NOT identical), 0) AS identical_violations
    FROM metric_sampler.expert_counts AS f FINAL
    WHERE f.exp = {exp_filter:String} AND f.run = {run_filter:String} AND f.attempt = {attempt_filter:String}
      AND f.step >= {step_from:UInt32} AND f.step <= {step_to:UInt32}
    GROUP BY step, node, metric_id, source_metric_id, layer, microbatch
    UNION ALL
    SELECT
        {exp_filter:String} AS exp, {run_filter:String} AS run, {attempt_filter:String} AS attempt, step, node, metric_id, source_metric_id, layer, microbatch,
        sum(contributors) AS contributors_seen,
        if(source_metric_id = 0 AND
           (exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM per_rank_metrics),
           max(expected_contributors), sum(expected_contributors)) AS contributors_expected,
        if((exp, run, attempt, metric_id) IN (SELECT exp, run, attempt, metric_id FROM identical_metrics),
           countIf(NOT identical), 0) AS identical_violations
    FROM metric_sampler.router_vectors AS f FINAL
    WHERE f.exp = {exp_filter:String} AND f.run = {run_filter:String} AND f.attempt = {attempt_filter:String}
      AND f.step >= {step_from:UInt32} AND f.step <= {step_to:UInt32}
    GROUP BY step, node, metric_id, source_metric_id, layer, microbatch
),
health AS
(
    SELECT
        exp, run, attempt, step,
        uniqExact(node) AS nodes_present,
        sum(contributors_seen) AS contributors_total,
        sum(contributors_expected) AS expected_contributors_total,
        sum(identical_violations) AS identical_violations
    FROM node_metric_health
    GROUP BY exp, run, attempt, step
),
expected_nodes AS
(
    SELECT exp, run, attempt, step, uniqExact(node) AS nodes_expected
    FROM
    (
        SELECT
            exp, run, attempt,
            arrayJoin(range(toUInt64(first_step), toUInt64(last_step) + 1)) AS step,
            arrayJoin(nodes) AS node
        FROM metric_sampler.batches
        WHERE exp = {exp_filter:String} AND run = {run_filter:String} AND attempt = {attempt_filter:String}
    )
    GROUP BY exp, run, attempt, step
),
manifest_nodes_in_job AS
(
    SELECT exp, run, attempt, step, max(nodes_in_job) AS nodes_in_job
    FROM
    (
        SELECT
            exp, run, attempt,
            arrayJoin(range(toUInt64(first_step), toUInt64(last_step) + 1)) AS step,
            nodes_in_job
        FROM metric_sampler.node_manifests
        WHERE exp = {exp_filter:String} AND run = {run_filter:String} AND attempt = {attempt_filter:String}
    )
    GROUP BY exp, run, attempt, step
)
SELECT
    health.exp AS exp,
    health.run AS run,
    health.attempt AS attempt,
    health.step AS step,
    health.nodes_present AS nodes_present,
    greatest(coalesce(expected_nodes.nodes_expected, 0), coalesce(manifest_nodes_in_job.nodes_in_job, 0)) AS nodes_expected,
    health.contributors_total AS contributors_total,
    health.expected_contributors_total AS expected_contributors_total,
    health.identical_violations AS identical_violations
FROM health
LEFT JOIN expected_nodes
    ON health.exp = expected_nodes.exp AND health.run = expected_nodes.run
   AND health.attempt = expected_nodes.attempt AND health.step = expected_nodes.step
LEFT JOIN manifest_nodes_in_job
    ON health.exp = manifest_nodes_in_job.exp AND health.run = manifest_nodes_in_job.run
   AND health.attempt = manifest_nodes_in_job.attempt AND health.step = manifest_nodes_in_job.step;

CREATE OR REPLACE VIEW metric_sampler.health_ready_batches AS
WITH manifests AS
(
    SELECT exp, run, attempt, batch,
        argMax(first_step, ingested_at) AS first_step,
        argMax(last_step, ingested_at) AS last_step,
        argMax(rows_per_family, ingested_at) AS rows_per_family,
        max(collected_at) AS collected_at
    FROM metric_sampler.batches
    GROUP BY exp, run, attempt, batch
),
objects AS
(
    SELECT kind, object, max(rows) AS rows
    FROM metric_sampler.processed_objects
    WHERE kind IN ('scalars', 'expert_counts', 'router_vectors')
    GROUP BY kind, object
),
counts AS
(
    SELECT extract(object, 'exp=([^/]+)') AS exp,
        extract(object, 'run=([^/]+)') AS run,
        extract(object, 'attempt=([^/]+)') AS attempt,
        extract(object, 'part-\d+-\d+-([^/.]+)\.parquet$') AS batch,
        sumIf(rows, kind = 'scalars') AS scalars,
        sumIf(rows, kind = 'expert_counts') AS expert_counts,
        sumIf(rows, kind = 'router_vectors') AS router_vectors
    FROM objects
    GROUP BY exp, run, attempt, batch
)
SELECT manifests.exp AS exp, manifests.run AS run, manifests.attempt AS attempt,
    manifests.batch AS batch, first_step, last_step, collected_at
FROM manifests
INNER JOIN counts USING (exp, run, attempt, batch)
WHERE scalars = rows_per_family['scalars']
  AND expert_counts = rows_per_family['expert_counts']
  AND router_vectors = rows_per_family['router_vectors']
  AND scalars + expert_counts + router_vectors > 0;

CREATE OR REPLACE VIEW metric_sampler.health_pending_windows AS
SELECT exp, run, attempt, batch,
    toUInt32(window_start) AS window_first,
    toUInt32(least(window_start + 9, toUInt64(last_step))) AS window_last,
    collected_at
FROM metric_sampler.health_ready_batches
ARRAY JOIN range(toUInt64(first_step), toUInt64(last_step) + 1, toUInt64(10)) AS window_start
WHERE (exp, run, attempt, batch, toUInt32(window_start)) NOT IN
    (SELECT exp, run, attempt, batch, window_first FROM metric_sampler.health_refreshes);

CREATE OR REPLACE VIEW metric_sampler.health_serving_per_step AS
WITH latest AS
(
    SELECT s.exp AS exp, s.run AS run, s.attempt AS attempt, s.step AS step,
        argMax(tuple(s.nodes_present, s.nodes_expected, s.contributors_total,
                     s.expected_contributors_total, s.identical_violations),
               tuple(r.completed_at, r.generation)) AS health_values,
        max(r.completed_at) AS refreshed_at
    FROM metric_sampler.health_snapshots AS s FINAL
    INNER JOIN metric_sampler.health_refreshes AS r FINAL
        ON s.exp = r.exp AND s.run = r.run AND s.attempt = r.attempt
       AND s.batch = r.batch AND s.window_first = r.window_first AND s.generation = r.generation
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON s.exp = las.exp AND s.run = las.run AND s.step = las.step AND s.attempt = las.attempt
    GROUP BY s.exp, s.run, s.attempt, s.step
)
SELECT exp, run, attempt, step,
    health_values.1 AS nodes_present, health_values.2 AS nodes_expected,
    health_values.3 AS contributors_total, health_values.4 AS expected_contributors_total,
    health_values.5 AS identical_violations, refreshed_at
FROM latest;

CREATE OR REPLACE VIEW metric_sampler.pipeline_freshness AS
SELECT
    ingested.exp AS exp,
    ingested.run AS run,
    ingested.max_step AS max_step,
    ingested.max_ingested_at AS max_ingested_at,
    batches_latest.last_collected_at AS last_collected_at,
    health_latest.health_max_step AS health_max_step,
    health_latest.health_refreshed_at AS health_refreshed_at
FROM
(
    SELECT exp, run, max(step) AS max_step, max(max_ingested_at) AS max_ingested_at
    FROM metric_sampler.step_index
    GROUP BY exp, run
) AS ingested
LEFT JOIN
(
    SELECT exp, run, max(collected_at) AS last_collected_at
    FROM metric_sampler.batches
    GROUP BY exp, run
) AS batches_latest
    ON ingested.exp = batches_latest.exp AND ingested.run = batches_latest.run
LEFT JOIN
(
    SELECT exp, run, toNullable(max(step)) AS health_max_step,
        toNullable(max(refreshed_at)) AS health_refreshed_at
    FROM metric_sampler.health_serving_per_step
    GROUP BY exp, run
) AS health_latest
    ON ingested.exp = health_latest.exp AND ingested.run = health_latest.run;

-- Loss tails per segment (main, or an MTP head by layer) from fixed-bin histograms: quantiles
-- via arrayCumSum + arrayFirstIndex use the bin center, except the last (open) bin's lower bound.
CREATE OR REPLACE VIEW metric_sampler.loss_quantiles_per_step AS
WITH main_summed AS
(
    SELECT
        ec.exp AS exp, ec.run AS run, ec.step AS step,
        any(mn.bins_start) AS bins_start,
        any(mn.bins_width) AS bins_width,
        any(mn.bins_count) AS bins_count,
        sumForEach(ec.values) AS by_stream
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'loss/token_hist_by_stream'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/token_hist_by_stream')
    GROUP BY ec.exp, ec.run, ec.step
),
main_hist AS
(
    -- by_stream is [stream][bin]; summing the per-stream slices folds it to one histogram.
    SELECT
        exp, run, step, 'main' AS segment, bins_start, bins_width, bins_count,
        arrayMap(x -> toFloat64(x), arrayReduce(
            'sumForEach',
            arrayMap(s -> arraySlice(by_stream, s * bins_count + 1, bins_count), range(intDiv(length(by_stream), bins_count)))
        )) AS hist
    FROM main_summed
),
mtp_hist AS
(
    SELECT
        ec.exp AS exp, ec.run AS run, ec.step AS step, toString(ec.layer) AS segment,
        any(mn.bins_start) AS bins_start, any(mn.bins_width) AS bins_width, any(mn.bins_count) AS bins_count,
        arrayMap(x -> toFloat64(x), sumForEach(ec.values)) AS hist
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'loss/mtp_token_hist'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/mtp_token_hist')
    GROUP BY ec.exp, ec.run, ec.step, ec.layer
),
with_totals AS
(
    SELECT
        exp, run, step, segment, bins_start, bins_width, bins_count, hist,
        arraySum(hist) AS tokens,
        arrayCumSum(hist) AS cum
    FROM main_hist
    UNION ALL
    SELECT
        exp, run, step, segment, bins_start, bins_width, bins_count, hist,
        arraySum(hist) AS tokens,
        arrayCumSum(hist) AS cum
    FROM mtp_hist
)
SELECT
    exp, run, step, segment, tokens,
    arraySum(arrayMap((c, i) -> c * (bins_start + bins_width * (i - 0.5)), hist, arrayEnumerate(hist))) / tokens AS mean,
    multiIf(
        tokens = 0, NULL,
        arrayFirstIndex(x -> x >= 0.5 * tokens, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.5 * tokens, cum) - 0.5)
    ) AS p50,
    multiIf(
        tokens = 0, NULL,
        arrayFirstIndex(x -> x >= 0.9 * tokens, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.9 * tokens, cum) - 0.5)
    ) AS p90,
    multiIf(
        tokens = 0, NULL,
        arrayFirstIndex(x -> x >= 0.99 * tokens, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.99 * tokens, cum) - 0.5)
    ) AS p99,
    multiIf(
        tokens = 0, NULL,
        arrayFirstIndex(x -> x >= 0.999 * tokens, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.999 * tokens, cum) - 0.5)
    ) AS p999
FROM with_totals;

-- Per (exp, run, step, bucket): main-segment loss binned by token position in the packed sequence
-- (bucket = position * 64 // seq_len), summed over nodes.
CREATE OR REPLACE VIEW metric_sampler.loss_position_per_step AS
WITH position_sum AS
(
    SELECT rv.exp AS exp, rv.run AS run, rv.step AS step, sumForEach(rv.values) AS values
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE rv.layer <= 255
      AND mn.name = 'loss/position_sum'
      AND rv.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/position_sum')
    GROUP BY rv.exp, rv.run, rv.step
),
position_count AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.step AS step, sumForEach(ec.values) AS values
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'loss/position_count'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/position_count')
    GROUP BY ec.exp, ec.run, ec.step
)
SELECT
    position_count.exp AS exp,
    position_count.run AS run,
    position_count.step AS step,
    bucket_index - 1 AS bucket,
    position_count.values[bucket_index] AS tokens,
    position_sum.values[bucket_index] / nullIf(position_count.values[bucket_index], 0) AS mean
FROM position_count
ARRAY JOIN arrayEnumerate(position_count.values) AS bucket_index
INNER JOIN position_sum
    ON position_sum.exp = position_count.exp AND position_sum.run = position_count.run AND position_sum.step = position_count.step;

-- Per (exp, run, step, stream): exact mean/std from the additive moments, plus sample count so
-- a genuinely bad stream can be told apart from one that just carries short documents.
CREATE OR REPLACE VIEW metric_sampler.stream_loss_per_step AS
WITH stream_sum AS
(
    SELECT rv.exp AS exp, rv.run AS run, rv.step AS step, sumForEach(rv.values) AS values
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE rv.layer <= 255
      AND mn.name = 'loss/stream_sum'
      AND rv.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/stream_sum')
    GROUP BY rv.exp, rv.run, rv.step
),
stream_sumsq AS
(
    SELECT rv.exp AS exp, rv.run AS run, rv.step AS step, sumForEach(rv.values) AS values
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE rv.layer <= 255
      AND mn.name = 'loss/stream_sumsq'
      AND rv.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/stream_sumsq')
    GROUP BY rv.exp, rv.run, rv.step
),
stream_count AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.step AS step, sumForEach(ec.values) AS values
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'loss/stream_count'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/stream_count')
    GROUP BY ec.exp, ec.run, ec.step
),
stream_samples AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.step AS step, sumForEach(ec.values) AS values
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'loss/stream_samples'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/stream_samples')
    GROUP BY ec.exp, ec.run, ec.step
)
SELECT
    stream_count.exp AS exp,
    stream_count.run AS run,
    stream_count.step AS step,
    stream_index - 1 AS stream_id,
    coalesce(stream_names.name, concat('stream', toString(stream_index - 1))) AS stream_name,
    stream_count.values[stream_index] AS tokens,
    stream_samples.values[stream_index] AS samples,
    stream_sum.values[stream_index] / stream_count.values[stream_index] AS mean,
    sqrt(greatest(
        stream_sumsq.values[stream_index] / stream_count.values[stream_index]
        - pow(stream_sum.values[stream_index] / stream_count.values[stream_index], 2),
        0
    )) AS std
FROM stream_count
ARRAY JOIN arrayEnumerate(stream_count.values) AS stream_index
INNER JOIN stream_sum
    ON stream_sum.exp = stream_count.exp AND stream_sum.run = stream_count.run AND stream_sum.step = stream_count.step
INNER JOIN stream_sumsq
    ON stream_sumsq.exp = stream_count.exp AND stream_sumsq.run = stream_count.run AND stream_sumsq.step = stream_count.step
INNER JOIN stream_samples
    ON stream_samples.exp = stream_count.exp AND stream_samples.run = stream_count.run AND stream_samples.step = stream_count.step
LEFT JOIN metric_sampler.stream_names AS stream_names
    ON stream_names.exp = stream_count.exp AND stream_names.run = stream_count.run
   AND stream_names.stream_id = stream_index - 1
WHERE stream_count.values[stream_index] > 0;

-- Per (exp, run, step, stream): built directly from loss/stream_* metrics rather than on top of
-- stream_loss_per_step, whose `tokens > 0` filter would drop a stream that has samples but no
-- valid tokens this batch.
CREATE OR REPLACE VIEW metric_sampler.stream_share_per_step AS
WITH stream_samples AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.step AS step, sumForEach(ec.values) AS values
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'loss/stream_samples'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/stream_samples')
    GROUP BY ec.exp, ec.run, ec.step
),
stream_count AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.step AS step, sumForEach(ec.values) AS values
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'loss/stream_count'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/stream_count')
    GROUP BY ec.exp, ec.run, ec.step
),
stream_sum AS
(
    SELECT rv.exp AS exp, rv.run AS run, rv.step AS step, sumForEach(rv.values) AS values
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE rv.layer <= 255
      AND mn.name = 'loss/stream_sum'
      AND rv.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/stream_sum')
    GROUP BY rv.exp, rv.run, rv.step
)
SELECT
    stream_samples.exp AS exp,
    stream_samples.run AS run,
    stream_samples.step AS step,
    stream_index - 1 AS stream_id,
    coalesce(stream_names.name, concat('stream', toString(stream_index - 1))) AS stream_name,
    stream_samples.values[stream_index] AS samples,
    stream_count.values[stream_index] AS tokens,
    stream_samples.values[stream_index] / arraySum(stream_samples.values) AS sample_share,
    stream_count.values[stream_index] / arraySum(stream_count.values) AS token_share,
    stream_sum.values[stream_index] / arraySum(stream_sum.values) AS loss_share,
    stream_sum.values[stream_index] / nullIf(stream_count.values[stream_index], 0) AS mean,
    stream_count.values[stream_index] / stream_samples.values[stream_index] AS tokens_per_sample
FROM stream_samples
ARRAY JOIN arrayEnumerate(stream_samples.values) AS stream_index
INNER JOIN stream_count
    ON stream_count.exp = stream_samples.exp AND stream_count.run = stream_samples.run AND stream_count.step = stream_samples.step
INNER JOIN stream_sum
    ON stream_sum.exp = stream_samples.exp AND stream_sum.run = stream_samples.run AND stream_sum.step = stream_samples.step
LEFT JOIN metric_sampler.stream_names AS stream_names
    ON stream_names.exp = stream_samples.exp AND stream_names.run = stream_samples.run
   AND stream_names.stream_id = stream_index - 1
WHERE stream_samples.values[stream_index] > 0 OR stream_count.values[stream_index] > 0;

-- Per (exp, run, step, stream): quantiles of the mean sample loss histogram (detailed steps
-- only); same bin-center convention as loss_quantiles_per_step, but by_stream is kept as separate
-- per-stream histograms instead of folded into one.
CREATE OR REPLACE VIEW metric_sampler.sample_loss_quantiles_per_step AS
WITH hist_summed AS
(
    SELECT
        ec.exp AS exp, ec.run AS run, ec.step AS step,
        any(mn.bins_start) AS bins_start,
        any(mn.bins_width) AS bins_width,
        any(mn.bins_count) AS bins_count,
        sumForEach(ec.values) AS by_stream
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'loss/sample_hist_by_stream'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/sample_hist_by_stream')
    GROUP BY ec.exp, ec.run, ec.step
),
per_stream AS
(
    SELECT
        exp, run, step, bins_start, bins_width, bins_count,
        arrayMap(
            s -> arraySlice(by_stream, s * bins_count + 1, bins_count),
            range(intDiv(length(by_stream), bins_count))
        ) AS hists
    FROM hist_summed
),
with_totals AS
(
    SELECT
        exp, run, step, stream_index - 1 AS stream_id, bins_start, bins_width, bins_count,
        hist, arraySum(hist) AS samples, arrayCumSum(hist) AS cum
    FROM per_stream
    ARRAY JOIN hists AS hist, arrayEnumerate(hists) AS stream_index
)
SELECT
    exp, run, step, stream_id,
    coalesce(stream_names.name, concat('stream', toString(stream_id))) AS stream_name,
    samples,
    multiIf(
        samples = 0, NULL,
        arrayFirstIndex(x -> x >= 0.5 * samples, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.5 * samples, cum) - 0.5)
    ) AS p50,
    multiIf(
        samples = 0, NULL,
        arrayFirstIndex(x -> x >= 0.9 * samples, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.9 * samples, cum) - 0.5)
    ) AS p90,
    multiIf(
        samples = 0, NULL,
        arrayFirstIndex(x -> x >= 0.99 * samples, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.99 * samples, cum) - 0.5)
    ) AS p99
FROM with_totals
LEFT JOIN metric_sampler.stream_names AS stream_names
    ON stream_names.exp = with_totals.exp AND stream_names.run = with_totals.run
   AND stream_names.stream_id = with_totals.stream_id;

-- Per (exp, run, step, layer, expert): exact mean/std, plus "hardness" (expert mean minus the
-- layer's token-weighted mean), for spotting a hard-token specialist versus a broken expert.
CREATE OR REPLACE VIEW metric_sampler.expert_loss_per_step AS
WITH expert_sum AS
(
    SELECT rv.exp AS exp, rv.run AS run, rv.step AS step, rv.layer AS layer, sumForEach(rv.values) AS values
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE rv.layer <= 255
      AND mn.name = 'expert_loss/sum'
      AND rv.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'expert_loss/sum')
    GROUP BY rv.exp, rv.run, rv.step, rv.layer
),
expert_sumsq AS
(
    SELECT rv.exp AS exp, rv.run AS run, rv.step AS step, rv.layer AS layer, sumForEach(rv.values) AS values
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE rv.layer <= 255
      AND mn.name = 'expert_loss/sumsq'
      AND rv.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'expert_loss/sumsq')
    GROUP BY rv.exp, rv.run, rv.step, rv.layer
),
expert_count AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.step AS step, ec.layer AS layer, sumForEach(ec.values) AS values
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'expert_loss/count'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'expert_loss/count')
    GROUP BY ec.exp, ec.run, ec.step, ec.layer
),
per_expert AS
(
    SELECT
        expert_sum.exp AS exp, expert_sum.run AS run, expert_sum.step AS step, expert_sum.layer AS layer,
        expert_index AS expert,
        expert_count.values[expert_index] AS tokens,
        expert_sum.values[expert_index] / expert_count.values[expert_index] AS mean,
        sqrt(greatest(
            expert_sumsq.values[expert_index] / expert_count.values[expert_index]
            - pow(expert_sum.values[expert_index] / expert_count.values[expert_index], 2),
            0
        )) AS std
    FROM expert_sum
    ARRAY JOIN arrayEnumerate(expert_sum.values) AS expert_index
    INNER JOIN expert_sumsq
        ON expert_sumsq.exp = expert_sum.exp AND expert_sumsq.run = expert_sum.run
       AND expert_sumsq.step = expert_sum.step AND expert_sumsq.layer = expert_sum.layer
    INNER JOIN expert_count
        ON expert_count.exp = expert_sum.exp AND expert_count.run = expert_sum.run
       AND expert_count.step = expert_sum.step AND expert_count.layer = expert_sum.layer
    WHERE expert_count.values[expert_index] > 0
),
layer_mean AS
(
    SELECT exp, run, step, layer, sum(tokens * mean) / sum(tokens) AS layer_mean
    FROM per_expert
    GROUP BY exp, run, step, layer
)
SELECT
    per_expert.exp AS exp,
    per_expert.run AS run,
    per_expert.step AS step,
    per_expert.layer AS layer,
    per_expert.expert AS expert,
    per_expert.tokens AS tokens,
    per_expert.mean AS mean,
    per_expert.std AS std,
    per_expert.mean - layer_mean.layer_mean AS hardness
FROM per_expert
INNER JOIN layer_mean
    ON layer_mean.exp = per_expert.exp AND layer_mean.run = per_expert.run
   AND layer_mean.step = per_expert.step AND layer_mean.layer = per_expert.layer;

-- Per (exp, run, step, layer, expert, stream), detailed steps only. [expert][stream] arrays are
-- reshaped using the actual stream count from stream_names, not a hardcoded stride.
CREATE OR REPLACE VIEW metric_sampler.expert_stream_map AS
WITH stream_totals AS
(
    SELECT exp, run, count() AS num_streams
    FROM metric_sampler.stream_names
    GROUP BY exp, run
),
stream_count AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.step AS step, ec.layer AS layer, sumForEach(ec.values) AS values
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND mn.name = 'expert_loss/stream_count'
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'expert_loss/stream_count')
    GROUP BY ec.exp, ec.run, ec.step, ec.layer
),
stream_sum AS
(
    SELECT rv.exp AS exp, rv.run AS run, rv.step AS step, rv.layer AS layer, sumForEach(rv.values) AS values
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE rv.layer <= 255
      AND mn.name = 'expert_loss/stream_sum'
      AND rv.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'expert_loss/stream_sum')
    GROUP BY rv.exp, rv.run, rv.step, rv.layer
)
SELECT
    stream_count.exp AS exp,
    stream_count.run AS run,
    stream_count.step AS step,
    stream_count.layer AS layer,
    intDiv(idx - 1, stream_totals.num_streams) + 1 AS expert,
    (idx - 1) % stream_totals.num_streams AS stream_id,
    token_count AS tokens,
    loss_sum / token_count AS mean
FROM stream_count
INNER JOIN stream_totals
    ON stream_totals.exp = stream_count.exp AND stream_totals.run = stream_count.run
INNER JOIN stream_sum
    ON stream_sum.exp = stream_count.exp AND stream_sum.run = stream_count.run
   AND stream_sum.step = stream_count.step AND stream_sum.layer = stream_count.layer
-- Expand the values together so the full arrays are not repeated for every expert/stream.
-- arrayResize preserves arrayElement's zero for a missing sum entry in an older window.
ARRAY JOIN
    arrayEnumerate(stream_count.values) AS idx,
    stream_count.values AS token_count,
    arrayResize(stream_sum.values, length(stream_count.values)) AS loss_sum
WHERE token_count > 0;

-- Per (exp, run, step, node, rank): gradient norms, plus a robust z-score of the dense norm
-- (log-ratio to the step median, scaled by the run-pooled MAD since per-step MAD is unstable
-- with few ranks). Only the dense norm is comparable across ranks; expert parallelism gives
-- each rank different experts.
CREATE OR REPLACE VIEW metric_sampler.grad_per_rank AS
-- grad_metrics reads scalars FINAL once and splits the three grad/* metrics out with conditional aggregation.
WITH grad_metrics AS
(
    SELECT
        s.exp AS exp, s.run AS run, s.step AS step, s.node AS node, s.rank AS rank,
        sumIf(s.values[1], mn.name = 'grad/pre_reduce_sq_dense') AS dense_sq,
        countIf(mn.name = 'grad/pre_reduce_sq_dense') AS has_dense,
        sumIf(s.values[1], mn.name = 'grad/pre_reduce_sq_expert') AS expert_sq,
        countIf(mn.name = 'grad/pre_reduce_sq_expert') AS has_expert,
        anyIf(s.values[1], mn.name = 'grad/clip_total_norm') AS clip_total_norm,
        countIf(mn.name = 'grad/clip_total_norm') AS has_clip
    FROM metric_sampler.scalars AS s FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON s.exp = las.exp AND s.run = las.run AND s.step = las.step AND s.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = s.exp AND mn.run = s.run AND mn.attempt = s.attempt AND mn.metric_id = s.metric_id
    WHERE s.layer <= 255
      AND mn.name IN ('grad/pre_reduce_sq_dense', 'grad/pre_reduce_sq_expert', 'grad/clip_total_norm')
      AND s.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name IN ('grad/pre_reduce_sq_dense', 'grad/pre_reduce_sq_expert', 'grad/clip_total_norm'))
    GROUP BY s.exp, s.run, s.step, s.node, s.rank
),
norms AS
(
    -- A rank missing either metric for a step is dropped, not treated as zero.
    SELECT
        exp, run, step, node, rank,
        sqrt(dense_sq) AS dense_norm,
        sqrt(expert_sq) AS expert_norm,
        sqrt(dense_sq + expert_sq) AS total_norm
    FROM grad_metrics
    WHERE has_dense > 0 AND has_expert > 0
),
step_median AS
(
    SELECT exp, run, step, medianExact(dense_norm) AS median_norm
    FROM norms
    GROUP BY exp, run, step
),
deviations AS
(
    SELECT
        norms.exp AS exp, norms.run AS run, norms.step AS step, norms.node AS node, norms.rank AS rank,
        log(norms.dense_norm / step_median.median_norm) AS log_ratio
    FROM norms
    INNER JOIN step_median
        ON step_median.exp = norms.exp AND step_median.run = norms.run AND step_median.step = norms.step
),
run_scale AS
(
    SELECT exp, run, 1.4826 * medianExact(abs(log_ratio)) AS scale
    FROM deviations
    GROUP BY exp, run
),
clip_norm AS
(
    SELECT exp, run, step, any(clip_total_norm) AS clip_total_norm
    FROM grad_metrics
    WHERE has_clip > 0
    GROUP BY exp, run, step
)
SELECT
    norms.exp AS exp,
    norms.run AS run,
    norms.step AS step,
    norms.node AS node,
    norms.rank AS rank,
    norms.dense_norm AS dense_norm,
    norms.expert_norm AS expert_norm,
    norms.total_norm AS total_norm,
    deviations.log_ratio / nullIf(run_scale.scale, 0) AS robust_z,
    clip_norm.clip_total_norm AS clip_total_norm
FROM norms
INNER JOIN deviations
    ON deviations.exp = norms.exp AND deviations.run = norms.run AND deviations.step = norms.step
   AND deviations.node = norms.node AND deviations.rank = norms.rank
INNER JOIN run_scale
    ON run_scale.exp = norms.exp AND run_scale.run = norms.run
LEFT JOIN clip_norm
    ON clip_norm.exp = norms.exp AND clip_norm.run = norms.run AND clip_norm.step = norms.step;

-- Per (exp, run, step, node, rank, layer, point): same per-card outlier construction as
-- grad_per_rank, but median/scale are grouped by (layer, point) since magnitudes differ there.
CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_selected AS
WITH ranks AS
(
    SELECT
        s.exp AS exp, s.run AS run, s.step AS step, s.node AS node, s.rank AS rank, s.layer AS layer,
        extract(mn.name, '^act/([^/]+)/amax$') AS point,
        max(s.values[1]) AS amax
    FROM metric_sampler.scalars AS s FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON s.exp = las.exp AND s.run = las.run AND s.step = las.step AND s.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = s.exp AND mn.run = s.run AND mn.attempt = s.attempt AND mn.metric_id = s.metric_id
    WHERE s.layer <= 255
      AND ({exp_filter:String} = '' OR s.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR s.run = {run_filter:String})
      AND ({microbatch_filter:Int32} = -2 OR s.microbatch = {microbatch_filter:Int32})
      AND match(mn.name, '^act/[^/]+/amax$')
      AND s.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE match(name, '^act/[^/]+/amax$'))
    GROUP BY s.exp, s.run, s.step, s.node, s.rank, s.layer, point
),
step_median AS
(
    SELECT exp, run, step, layer, point, medianExact(amax) AS median_amax
    FROM ranks
    GROUP BY exp, run, step, layer, point
),
deviations AS
(
    SELECT
        ranks.exp AS exp, ranks.run AS run, ranks.step AS step, ranks.node AS node, ranks.rank AS rank,
        ranks.layer AS layer, ranks.point AS point, ranks.amax AS amax,
        log(ranks.amax / step_median.median_amax) AS log_ratio
    FROM ranks
    INNER JOIN step_median
        ON step_median.exp = ranks.exp AND step_median.run = ranks.run AND step_median.step = ranks.step
       AND step_median.layer = ranks.layer AND step_median.point = ranks.point
),
run_scale AS
(
    SELECT exp, run, layer, point, 1.4826 * medianExact(abs(log_ratio)) AS scale
    FROM deviations
    GROUP BY exp, run, layer, point
)
SELECT
    deviations.exp AS exp,
    deviations.run AS run,
    deviations.step AS step,
    deviations.node AS node,
    deviations.rank AS rank,
    deviations.layer AS layer,
    deviations.point AS point,
    deviations.amax AS amax,
    deviations.log_ratio / nullIf(run_scale.scale, 0) AS robust_z
FROM deviations
INNER JOIN run_scale
    ON run_scale.exp = deviations.exp AND run_scale.run = deviations.run
   AND run_scale.layer = deviations.layer AND run_scale.point = deviations.point
WHERE ({rank_filter:Int32} = -2 OR deviations.rank = {rank_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank AS
SELECT * FROM metric_sampler.activation_per_rank_selected(exp_filter = '', run_filter = '', rank_filter = -2, microbatch_filter = -2);

-- Per (exp, run, step, node, rank, metric): same outlier construction as grad_per_rank, grouped
-- by metric name. forward/backward_gpu_s are summed per step (recorded per microbatch); the sum
-- is a no-op for step_gpu_s, which is already one value per step.
CREATE OR REPLACE VIEW metric_sampler.timing_per_rank AS
WITH ranks AS
(
    SELECT
        s.exp AS exp, s.run AS run, s.step AS step, s.node AS node, s.rank AS rank,
        mn.name AS metric,
        sum(s.values[1]) AS value
    FROM metric_sampler.scalars AS s FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON s.exp = las.exp AND s.run = las.run AND s.step = las.step AND s.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = s.exp AND mn.run = s.run AND mn.attempt = s.attempt AND mn.metric_id = s.metric_id
    WHERE s.layer <= 255
      AND mn.name IN ('time/step_gpu_s', 'time/forward_gpu_s', 'time/backward_gpu_s', 'time/optimizer_gpu_s')
      AND s.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE name IN ('time/step_gpu_s', 'time/forward_gpu_s', 'time/backward_gpu_s', 'time/optimizer_gpu_s'))
    GROUP BY s.exp, s.run, s.step, s.node, s.rank, mn.name
),
step_median AS
(
    SELECT exp, run, step, metric, medianExact(value) AS median_value
    FROM ranks
    GROUP BY exp, run, step, metric
),
deviations AS
(
    SELECT
        ranks.exp AS exp, ranks.run AS run, ranks.step AS step, ranks.node AS node, ranks.rank AS rank,
        ranks.metric AS metric, ranks.value AS value,
        log(ranks.value / step_median.median_value) AS log_ratio
    FROM ranks
    INNER JOIN step_median
        ON step_median.exp = ranks.exp AND step_median.run = ranks.run AND step_median.step = ranks.step
       AND step_median.metric = ranks.metric
),
run_scale AS
(
    SELECT exp, run, metric, 1.4826 * medianExact(abs(log_ratio)) AS scale
    FROM deviations
    GROUP BY exp, run, metric
)
SELECT
    deviations.exp AS exp,
    deviations.run AS run,
    deviations.step AS step,
    deviations.node AS node,
    deviations.rank AS rank,
    deviations.metric AS metric,
    deviations.value AS value,
    deviations.log_ratio / nullIf(run_scale.scale, 0) AS robust_z
FROM deviations
INNER JOIN run_scale
    ON run_scale.exp = deviations.exp AND run_scale.run = deviations.run AND run_scale.metric = deviations.metric;

-- Four candidates per rank/microbatch on detailed steps, paired by their original capture key.
-- Zero token counts mark padding; an all-nonfinite sample has no defined finite mean/max.
CREATE OR REPLACE VIEW metric_sampler.worst_samples AS
WITH candidates AS
(
    SELECT info.exp AS exp, info.run AS run, info.attempt AS attempt,
        info.step AS step, info.node AS node, info.rank AS rank, info.microbatch AS microbatch,
        sample.1 AS stream_id, sample.2 AS shard_id, sample.3 AS index_in_shard,
        sample.4 AS loss_position, sample.5 AS finite_tokens, sample.6 AS nonfinite_tokens,
        if(sample.5 > 0, sample.7, NULL) AS mean_loss,
        if(sample.5 > 0, sample.8, NULL) AS max_loss
    FROM
    (
        SELECT * FROM metric_sampler.expert_counts FINAL
        WHERE metric_id IN (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/worst_sample_info')
            AND source_metric_id = 0 AND layer = -1 AND length(values) = 24
    ) AS info
    INNER JOIN
    (
        SELECT * FROM metric_sampler.router_vectors FINAL
        WHERE metric_id IN (SELECT metric_id FROM metric_sampler.metrics WHERE name = 'loss/worst_sample_stats')
            AND source_metric_id = 0 AND layer = -1 AND length(values) = 8
    ) AS stats
        ON info.exp = stats.exp AND info.run = stats.run AND info.attempt = stats.attempt
        AND info.node = stats.node AND info.step = stats.step AND info.rank = stats.rank
        AND info.microbatch = stats.microbatch
    INNER JOIN metric_sampler.latest_attempt_step AS latest
        ON info.exp = latest.exp AND info.run = latest.run AND info.step = latest.step
        AND info.attempt = latest.attempt
    ARRAY JOIN arrayMap(i -> tuple(
        info.values[6 * i + 1], info.values[6 * i + 2], info.values[6 * i + 3],
        info.values[6 * i + 4], info.values[6 * i + 5], info.values[6 * i + 6],
        stats.values[2 * i + 1], stats.values[2 * i + 2]), range(4)) AS sample
    WHERE sample.5 > 0 OR sample.6 > 0
)
SELECT c.*, if(s.name = '', concat('stream_', toString(c.stream_id)), s.name) AS stream
FROM candidates AS c
LEFT JOIN metric_sampler.streams AS s FINAL
    ON c.exp = s.exp AND c.run = s.run AND c.attempt = s.attempt AND c.stream_id = s.stream_id;

-- Convenience view over `logged`, mirroring latest_attempt_step so a restart's stale attempt
-- does not shadow the current one.
-- The latest attempt is chosen among the distinct (attempt, step) pairs: joining attempt_starts to every logged row
-- is much slower for a panel that reads one name.
CREATE OR REPLACE VIEW metric_sampler.logged_per_step AS
SELECT l.exp AS exp, l.run AS run, l.step AS step, l.name AS name, l.value AS value, l.source_step AS source_step
FROM metric_sampler.logged AS l FINAL
INNER JOIN
(
    SELECT p.exp AS exp, p.run AS run, p.step AS step, argMax(p.attempt, tuple(coalesce(st.started_at, toDateTime64(0, 3)), p.attempt)) AS attempt
    FROM (SELECT DISTINCT exp, run, attempt, step FROM metric_sampler.logged) AS p
    LEFT JOIN metric_sampler.attempt_starts AS st
        ON st.exp = p.exp AND st.run = p.run AND st.attempt = p.attempt
    GROUP BY p.exp, p.run, p.step
) AS latest
    ON l.exp = latest.exp AND l.run = latest.run AND l.step = latest.step AND l.attempt = latest.attempt;

-- Metric names: `act/<point>/moments|exp_hist|token_rms_hist` and `act/<point>/amax`; `point` is
-- pulled from the name via regex rather than a hardcoded list.
--
-- Historical moments contain rounded Float32 counts. New counts are UInt32, keyed to the
-- same observation; retain exact UInt64 totals separately from the compatible display fields.
CREATE OR REPLACE VIEW metric_sampler.activation_selected AS
WITH exact_counts AS
(
    SELECT
        ec.exp AS exp, ec.run AS run, ec.attempt AS attempt, ec.step AS step, ec.node AS node,
        ec.rank AS rank, ec.microbatch AS microbatch, ec.layer AS layer,
        extract(mn.name, '^act/([^/]+)/counts$') AS point,
        toUInt64(ec.values[1]) AS nonfinite, toUInt64(ec.values[2]) AS elements,
        1 AS counts_exact
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND match(mn.name, '^act/[^/]+/counts$')
      AND ec.source_metric_id = 0
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE match(name, '^act/[^/]+/counts$'))
), moments AS
(
    SELECT
        rv.exp AS exp, rv.run AS run, rv.step AS step, rv.layer AS layer,
        extract(mn.name, '^act/([^/]+)/moments$') AS point,
        sum(rv.values[1]) AS sum_x,
        sum(rv.values[2]) AS sumsq_x,
        sum(CASE WHEN ec.counts_exact = 1 THEN toFloat64(ec.nonfinite) ELSE rv.values[3] END) AS nonfinite,
        sum(CASE WHEN ec.counts_exact = 1 THEN toFloat64(ec.elements) ELSE rv.values[4] END) AS elements,
        sum(coalesce(ec.nonfinite, toUInt64(0))) AS exact_nonfinite,
        sum(coalesce(ec.elements, toUInt64(0))) AS exact_elements,
        min(coalesce(ec.counts_exact, 0)) AS all_counts_exact,
        count() AS observations
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    LEFT JOIN exact_counts AS ec
        ON ec.exp = rv.exp AND ec.run = rv.run AND ec.attempt = rv.attempt AND ec.step = rv.step
       AND ec.node = rv.node AND ec.rank = rv.rank AND ec.microbatch = rv.microbatch AND ec.layer = rv.layer
       AND ec.point = extract(mn.name, '^act/([^/]+)/moments$')
    WHERE rv.layer <= 255
      AND ({exp_filter:String} = '' OR rv.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR rv.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR rv.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR rv.microbatch = {microbatch_filter:Int32})
      AND match(mn.name, '^act/[^/]+/moments$')
      AND rv.source_metric_id = 0
      AND rv.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE match(name, '^act/[^/]+/moments$'))
    GROUP BY rv.exp, rv.run, rv.step, rv.layer, extract(mn.name, '^act/([^/]+)/moments$')
),
power_sums AS
(
    -- Absent for observations without power sums, which leaves skewness and kurtosis NULL.
    SELECT
        rv.exp AS exp, rv.run AS run, rv.step AS step, rv.layer AS layer,
        extract(mn.name, '^act/([^/]+)/power_sums$') AS point,
        sum(rv.values[1]) AS sum_cube, sum(rv.values[2]) AS sum_fourth, count() AS observations
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE rv.layer <= 255
      AND ({exp_filter:String} = '' OR rv.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR rv.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR rv.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR rv.microbatch = {microbatch_filter:Int32})
      AND match(mn.name, '^act/[^/]+/power_sums$')
      AND rv.source_metric_id = 0
      AND rv.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE match(name, '^act/[^/]+/power_sums$'))
    GROUP BY rv.exp, rv.run, rv.step, rv.layer, extract(mn.name, '^act/([^/]+)/power_sums$')
),
rank_amax AS
(
    SELECT
        s.exp AS exp, s.run AS run, s.step AS step, s.layer AS layer, s.node AS node, s.rank AS rank,
        extract(mn.name, '^act/([^/]+)/amax$') AS point,
        max(s.values[1]) AS amax
    FROM metric_sampler.scalars AS s FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON s.exp = las.exp AND s.run = las.run AND s.step = las.step AND s.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = s.exp AND mn.run = s.run AND mn.attempt = s.attempt AND mn.metric_id = s.metric_id
    WHERE s.layer <= 255
      AND ({exp_filter:String} = '' OR s.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR s.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR s.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR s.microbatch = {microbatch_filter:Int32})
      AND match(mn.name, '^act/[^/]+/amax$')
      AND s.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE match(name, '^act/[^/]+/amax$'))
    GROUP BY s.exp, s.run, s.step, s.layer, s.node, s.rank, point
), amax AS
(
    SELECT exp, run, step, layer, point, max(amax) AS amax_max, medianExact(amax) AS amax_median
    FROM rank_amax
    GROUP BY exp, run, step, layer, point
)
SELECT
    moments.exp AS exp,
    moments.run AS run,
    moments.step AS step,
    moments.layer AS layer,
    moments.point AS point,
    moments.elements AS elements,
    moments.sum_x / nullIf(moments.elements, 0) AS mean,
    sqrt(moments.sumsq_x / nullIf(moments.elements, 0)) AS rms,
    sqrt(greatest(moments.sumsq_x / nullIf(moments.elements, 0) - pow(moments.sum_x / nullIf(moments.elements, 0), 2), 0)) AS std,
    moments.nonfinite AS nonfinite,
    amax.amax_max AS amax_max,
    amax.amax_median AS amax_median,
    127 - log2(amax.amax_max) AS bf16_headroom_log2,
    log2(448) - log2(amax.amax_max) AS fp8_e4m3_headroom_log2,
    moments.all_counts_exact AS all_counts_exact,
    CASE WHEN moments.all_counts_exact = 1 THEN moments.exact_elements ELSE NULL END AS exact_elements,
    CASE WHEN moments.all_counts_exact = 1 THEN moments.exact_nonfinite ELSE NULL END AS exact_nonfinite,
    -- Central moments from the raw power sums of every selected observation: a normal distribution
    -- has skewness 0 and kurtosis 3.
    CASE WHEN power_sums.observations = moments.observations THEN
        (power_sums.sum_cube / nullIf(moments.elements, 0)
            - 3 * (moments.sum_x / nullIf(moments.elements, 0)) * (moments.sumsq_x / nullIf(moments.elements, 0))
            + 2 * pow(moments.sum_x / nullIf(moments.elements, 0), 3))
            / nullIf(pow(greatest(moments.sumsq_x / nullIf(moments.elements, 0)
                                  - pow(moments.sum_x / nullIf(moments.elements, 0), 2), 0), 1.5), 0)
    ELSE NULL END AS skewness,
    CASE WHEN power_sums.observations = moments.observations THEN
        (power_sums.sum_fourth / nullIf(moments.elements, 0)
            - 4 * (moments.sum_x / nullIf(moments.elements, 0)) * (power_sums.sum_cube / nullIf(moments.elements, 0))
            + 6 * pow(moments.sum_x / nullIf(moments.elements, 0), 2) * (moments.sumsq_x / nullIf(moments.elements, 0))
            - 3 * pow(moments.sum_x / nullIf(moments.elements, 0), 4))
            / nullIf(pow(greatest(moments.sumsq_x / nullIf(moments.elements, 0)
                                  - pow(moments.sum_x / nullIf(moments.elements, 0), 2), 0), 2), 0)
    ELSE NULL END AS kurtosis
FROM moments
INNER JOIN amax
    ON amax.exp = moments.exp AND amax.run = moments.run AND amax.step = moments.step
   AND amax.layer = moments.layer AND amax.point = moments.point
LEFT JOIN power_sums
    ON power_sums.exp = moments.exp AND power_sums.run = moments.run AND power_sums.step = moments.step
   AND power_sums.layer = moments.layer AND power_sums.point = moments.point;

CREATE OR REPLACE VIEW metric_sampler.activation_per_step AS
SELECT * FROM metric_sampler.activation_selected(exp_filter = '', run_filter = '', rank_filter = -2, microbatch_filter = -2);

-- Keep the UInt32/UInt64 histogram path separate from historical Float32 values. Exact rows
-- supersede a legacy histogram only for the same complete observation key.
CREATE OR REPLACE VIEW metric_sampler.activation_histograms_selected AS
WITH exact_counts AS
(
    SELECT
        ec.exp AS exp, ec.run AS run, ec.attempt AS attempt, ec.step AS step, ec.node AS node,
        ec.rank AS rank, ec.microbatch AS microbatch, ec.layer AS layer,
        extract(mn.name, '^act/([^/]+)/counts$') AS point, ec.values AS counts
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS las
        ON ec.exp = las.exp AND ec.run = las.run AND ec.step = las.step AND ec.attempt = las.attempt
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND match(mn.name, '^act/[^/]+/counts$')
      AND ec.source_metric_id = 0
      AND ec.metric_id IN
          (SELECT metric_id FROM metric_sampler.metrics WHERE match(name, '^act/[^/]+/counts$'))
)
SELECT
    exp, run, attempt, step, node, rank, microbatch, layer, point, 'exp' AS histogram,
    -40.0 AS bins_start, 1.0 AS bins_width, toUInt32(64) AS bins_count,
    arraySlice(counts, 3, 64) AS exact_hist, emptyArrayFloat64() AS legacy_hist, 1 AS counts_exact
FROM exact_counts
UNION ALL
SELECT
    exp, run, attempt, step, node, rank, microbatch, layer, point, 'token_rms' AS histogram,
    -40.0 AS bins_start, 1.0 AS bins_width, toUInt32(64) AS bins_count,
    arraySlice(counts, 67, 64) AS exact_hist, emptyArrayFloat64() AS legacy_hist, 1 AS counts_exact
FROM exact_counts
UNION ALL
SELECT
    rv.exp AS exp, rv.run AS run, rv.attempt AS attempt, rv.step AS step, rv.node AS node,
    rv.rank AS rank, rv.microbatch AS microbatch, rv.layer AS layer,
    extract(mn.name, '^act/([^/]+)/') AS point,
    extract(mn.name, '^act/[^/]+/(exp|token_rms)_hist$') AS histogram,
    mn.bins_start AS bins_start, mn.bins_width AS bins_width, mn.bins_count AS bins_count,
    emptyArrayUInt64() AS exact_hist, rv.values AS legacy_hist, 0 AS counts_exact
FROM metric_sampler.router_vectors AS rv FINAL
INNER JOIN metric_sampler.latest_attempt_step AS las
    ON rv.exp = las.exp AND rv.run = las.run AND rv.step = las.step AND rv.attempt = las.attempt
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
WHERE rv.layer <= 255
  AND ({exp_filter:String} = '' OR rv.exp = {exp_filter:String})
  AND ({run_filter:String} = '' OR rv.run = {run_filter:String})
  AND ({rank_filter:Int32} = -2 OR rv.rank = {rank_filter:Int32})
  AND ({microbatch_filter:Int32} = -2 OR rv.microbatch = {microbatch_filter:Int32})
  AND match(mn.name, '^act/[^/]+/(exp|token_rms)_hist$')
  AND rv.source_metric_id = 0
  AND rv.metric_id IN
      (SELECT metric_id FROM metric_sampler.metrics WHERE match(name, '^act/[^/]+/(exp|token_rms)_hist$'))
  AND (rv.exp, rv.run, rv.attempt, rv.step, rv.node, rv.rank, rv.microbatch, rv.layer,
       extract(mn.name, '^act/([^/]+)/')) NOT IN
      (SELECT exp, run, attempt, step, node, rank, microbatch, layer, point FROM exact_counts);

-- Per (exp, run, step, layer, point): |x| quantiles from the exp_hist grid (same bin-center
-- convention as loss_quantiles_per_step), plus max_log2 (exact to within a factor of 2).
CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_selected AS
WITH hist_sum AS
(
    SELECT
        exp, run, step, layer, point,
        any(bins_start) AS bins_start, any(bins_width) AS bins_width, any(bins_count) AS bins_count,
        sumForEach(exact_hist) AS exact_hist, sumForEach(legacy_hist) AS legacy_hist,
        min(counts_exact) AS all_counts_exact
    FROM metric_sampler.activation_histograms_selected(
        exp_filter = {exp_filter:String}, run_filter = {run_filter:String},
        rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32})
    WHERE histogram = 'exp'
    GROUP BY exp, run, step, layer, point
),
histograms AS
(
    SELECT *, arrayMap(i -> exact_hist[i] + legacy_hist[i], range(1, bins_count + 1)) AS hist
    FROM hist_sum
),
with_totals AS
(
    SELECT
        exp, run, step, layer, point, bins_start, bins_width, bins_count, hist,
        arraySum(hist) AS elements,
        arrayCumSum(hist) AS cum,
        all_counts_exact, arraySum(exact_hist) AS exact_elements
    FROM histograms
)
SELECT
    exp, run, step, layer, point,
    multiIf(
        elements = 0, NULL,
        arrayFirstIndex(x -> x >= 0.5 * elements, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.5 * elements, cum) - 0.5)
    ) AS p50_log2,
    multiIf(
        elements = 0, NULL,
        arrayFirstIndex(x -> x >= 0.99 * elements, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.99 * elements, cum) - 0.5)
    ) AS p99_log2,
    multiIf(
        elements = 0, NULL,
        arrayFirstIndex(x -> x >= 0.999 * elements, cum) IN (0, bins_count),
            bins_start + bins_width * (bins_count - 1),
        bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.999 * elements, cum) - 0.5)
    ) AS p999_log2,
    multiIf(
        elements = 0, NULL,
        bins_start + bins_width * (arrayLastIndex(x -> x > 0, hist) - 1)
    ) AS max_log2,
    all_counts_exact,
    CASE WHEN all_counts_exact = 1 THEN exact_elements ELSE NULL END AS exact_elements
FROM with_totals;

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles AS
SELECT * FROM metric_sampler.activation_exp_quantiles_selected(exp_filter = '', run_filter = '', rank_filter = -2, microbatch_filter = -2);

-- From the token_rms_hist grid: median_log2_rms (bin center), massive_share (tokens with log2
-- RMS >= median + 7, i.e. >=128x median) and max_log2_rms (like activation_exp_quantiles.max_log2).
CREATE OR REPLACE VIEW metric_sampler.activation_massive_tokens_selected AS
WITH hist_sum AS
(
    SELECT
        exp, run, step, layer, point,
        any(bins_start) AS bins_start, any(bins_width) AS bins_width, any(bins_count) AS bins_count,
        sumForEach(exact_hist) AS exact_hist, sumForEach(legacy_hist) AS legacy_hist,
        min(counts_exact) AS all_counts_exact
    FROM metric_sampler.activation_histograms_selected(
        exp_filter = {exp_filter:String}, run_filter = {run_filter:String},
        rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32})
    WHERE histogram = 'token_rms'
    GROUP BY exp, run, step, layer, point
),
histograms AS
(
    SELECT *, arrayMap(i -> exact_hist[i] + legacy_hist[i], range(1, bins_count + 1)) AS hist
    FROM hist_sum
),
with_totals AS
(
    SELECT
        exp, run, step, layer, point, bins_start, bins_width, bins_count, hist,
        arraySum(hist) AS tokens,
        arrayCumSum(hist) AS cum,
        all_counts_exact, arraySum(exact_hist) AS exact_tokens
    FROM histograms
),
with_median AS
(
    SELECT
        exp, run, step, layer, point, bins_start, bins_width, bins_count, hist, tokens,
        all_counts_exact, exact_tokens,
        multiIf(
            tokens = 0, NULL,
            arrayFirstIndex(x -> x >= 0.5 * tokens, cum) IN (0, bins_count),
                bins_start + bins_width * (bins_count - 1),
            bins_start + bins_width * (arrayFirstIndex(x -> x >= 0.5 * tokens, cum) - 0.5)
        ) AS median_log2_rms
    FROM with_totals
)
SELECT
    exp, run, step, layer, point, tokens, median_log2_rms,
    if(
        tokens = 0, NULL,
        arraySum(arrayMap(
            (h, i) -> if(bins_start + bins_width * (i - 0.5) >= median_log2_rms + 7, h, 0),
            hist, arrayEnumerate(hist)
        )) / tokens
    ) AS massive_share,
    multiIf(
        tokens = 0, NULL,
        bins_start + bins_width * (arrayLastIndex(x -> x > 0, hist) - 1)
    ) AS max_log2_rms,
    all_counts_exact,
    CASE WHEN all_counts_exact = 1 THEN exact_tokens ELSE NULL END AS exact_tokens
FROM with_median;

CREATE OR REPLACE VIEW metric_sampler.activation_massive_tokens AS
SELECT * FROM metric_sampler.activation_massive_tokens_selected(exp_filter = '', run_filter = '', rank_filter = -2, microbatch_filter = -2);

-- These views retain observations; selecting all ranks/microbatches does not aggregate them.
CREATE OR REPLACE VIEW metric_sampler.scalar_observations_per_rank_selected AS
SELECT
    s.exp AS exp, s.run AS run, s.attempt AS attempt, s.step AS step,
    s.node AS node, s.rank AS rank, s.microbatch AS microbatch, s.layer AS layer,
    s.metric_id AS metric_id, s.source_metric_id AS source_metric_id,
    mn.name AS metric, s.values AS values
FROM metric_sampler.scalars AS s FINAL
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = s.exp AND mn.run = s.run AND mn.attempt = s.attempt AND mn.metric_id = s.metric_id
WHERE ({exp_filter:String} = '' OR s.exp = {exp_filter:String})
  AND ({run_filter:String} = '' OR s.run = {run_filter:String})
  AND ({rank_filter:Int32} = -2 OR s.rank = {rank_filter:Int32})
  AND ({microbatch_filter:Int32} = -2 OR s.microbatch = {microbatch_filter:Int32})
  AND s.rank >= 0;

CREATE OR REPLACE VIEW metric_sampler.expert_load_per_rank_selected AS
WITH rank_rows AS
(
    SELECT
        ec.exp AS exp, ec.run AS run, ec.attempt AS attempt, ec.step AS step,
        ec.node AS node, ec.rank AS rank, ec.microbatch AS microbatch, ec.layer AS layer,
        groupArray(ec.metric_id) AS metric_ids, groupArray(ec.alias_metric_id) AS alias_ids,
        groupArray(ec.values) AS rows_values
    FROM metric_sampler.expert_counts AS ec FINAL
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND ec.rank >= 0 AND ec.microbatch >= 0 AND ec.source_metric_id = 0
      AND ec.metric_id IN (1, 95, 96, 97, 98)
    GROUP BY ec.exp, ec.run, ec.attempt, ec.step, ec.node, ec.rank, ec.microbatch, ec.layer
),
resolved AS
(
    -- An aliased row has no values: it repeats the row of the metric it names.
    SELECT exp, run, attempt, step, node, rank, microbatch, layer, metric_id,
        if(alias_id = 0, row_values, rows_values[indexOf(metric_ids, alias_id)]) AS assignments
    FROM rank_rows
    ARRAY JOIN metric_ids AS metric_id, alias_ids AS alias_id, rows_values AS row_values
)
SELECT
    r.exp AS exp, r.run AS run, r.attempt AS attempt, r.step AS step,
    r.node AS node, r.rank AS rank, r.microbatch AS microbatch, r.layer AS layer,
    mn.name AS population, expert_index - 1 AS expert, tokens,
    arraySum(r.assignments) AS total_assignments,
    length(r.assignments) AS experts,
    tokens / nullIf(arraySum(r.assignments) / length(r.assignments), 0) AS relative_load
FROM resolved AS r
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = r.exp AND mn.run = r.run AND mn.attempt = r.attempt AND mn.metric_id = r.metric_id
ARRAY JOIN r.assignments AS tokens, arrayEnumerate(r.assignments) AS expert_index
WHERE mn.name IN ('router/tokens_per_expert', 'dispatcher/assignments_before_capacity',
                  'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity',
                  'dispatcher/valid_assignments_after_capacity');

-- Inverse empirical CDF over each original finite population; never combine per-population quantiles.
CREATE OR REPLACE VIEW metric_sampler.exact_quantiles_per_rank_selected AS
WITH quantiles AS
(
    SELECT rv.exp AS exp, rv.run AS run, rv.attempt AS attempt, rv.step AS step,
        rv.node AS node, rv.rank AS rank, rv.microbatch AS microbatch, rv.layer AS layer,
        mn.name AS metric, arrayMap(x -> toFloat64(x), rv.values) AS quantiles, 1 AS present,
        length(rv.values) = 9 AS width_valid
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE rv.layer <= 255
      AND ({exp_filter:String} = '' OR rv.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR rv.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR rv.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR rv.microbatch = {microbatch_filter:Int32})
      AND rv.rank >= 0 AND rv.source_metric_id = 0
      AND (rv.metric_id BETWEEN 114 AND 137 OR rv.metric_id = 177) AND match(mn.name, '(/quantiles|_quantiles)$')
    UNION ALL
    SELECT ec.exp, ec.run, ec.attempt, ec.step, ec.node, ec.rank, ec.microbatch, ec.layer,
        mn.name AS metric,
        arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.values[2 * i + 1]),
            bitShiftLeft(toUInt64(ec.values[2 * i + 2]), 32))), range(9)) AS quantiles, 1 AS present,
        length(ec.values) = 18 AS width_valid
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND ec.rank >= 0 AND ec.source_metric_id = 0
      AND ec.metric_id = 138 AND mn.name = 'router/entropy_quantiles'
), counts AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.attempt AS attempt, ec.step AS step,
        ec.node AS node, ec.rank AS rank, ec.microbatch AS microbatch, ec.layer AS layer,
        replaceRegexpOne(mn.name, '_counts$', 's') AS metric, ec.values AS counts, 1 AS present
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND ec.rank >= 0 AND ec.source_metric_id = 0
      AND (ec.metric_id BETWEEN 115 AND 139 OR ec.metric_id = 178) AND match(mn.name, '(/quantile_counts|_quantile_counts)$')
)
SELECT
    if(q.present = 1, q.exp, c.exp) AS exp, if(q.present = 1, q.run, c.run) AS run,
    if(q.present = 1, q.attempt, c.attempt) AS attempt, if(q.present = 1, q.step, c.step) AS step,
    if(q.present = 1, q.node, c.node) AS node, if(q.present = 1, q.rank, c.rank) AS rank,
    if(q.present = 1, q.microbatch, c.microbatch) AS microbatch,
    if(q.present = 1, q.layer, c.layer) AS layer, if(q.present = 1, q.metric, c.metric) AS metric,
    [0., 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.] AS levels,
    q.quantiles AS quantiles, c.counts AS population_counts,
    coalesce(q.present, 0) = 1 AS quantiles_present, coalesce(c.present, 0) = 1 AS counts_present,
    coalesce(q.present, 0) = 1 AND coalesce(c.present, 0) = 1
        AND q.width_valid AND length(c.counts) = 5 AS complete_pair
FROM quantiles AS q
FULL OUTER JOIN counts AS c
    ON q.exp = c.exp AND q.run = c.run AND q.attempt = c.attempt AND q.step = c.step
   AND q.node = c.node AND q.rank = c.rank AND q.microbatch = c.microbatch
   AND q.layer = c.layer AND q.metric = c.metric;

CREATE OR REPLACE VIEW metric_sampler.monitor_local_per_rank_selected AS
WITH observations AS
(
    SELECT rv.exp AS exp, rv.run AS run, rv.attempt AS attempt, rv.step AS step,
        rv.node AS node, rv.rank AS rank, rv.microbatch AS microbatch, rv.layer AS layer,
        rv.metric_id AS metric_id, mn.name AS metric,
        arrayMap(x -> toFloat64(x), rv.values) AS stats,
        multiIf(rv.metric_id = 140, '', rv.metric_id = 144, 'activation', 'parameter') AS catalog_kind,
        toUInt16(rv.layer) AS catalog_index,
        length(rv.values) = multiIf(rv.metric_id = 140, 8, rv.metric_id = 149, 18, rv.metric_id = 160, 3, 2) AS width_valid
    FROM metric_sampler.router_vectors AS rv FINAL
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
    WHERE ({exp_filter:String} = '' OR rv.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR rv.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR rv.rank = {rank_filter:Int32}
           OR (rv.rank = -1 AND rv.identical AND rv.node IN (SELECT node FROM metric_sampler.node_manifests
               WHERE exp = {exp_filter:String} AND run = {run_filter:String} AND has(ranks, toUInt16(greatest({rank_filter:Int32}, 0))))))
      AND ({microbatch_filter:Int32} = -2 OR rv.microbatch = {microbatch_filter:Int32})
      AND (rv.rank >= 0 OR rv.identical) AND rv.source_metric_id = 0
      AND rv.metric_id IN (140, 144, 145, 149, 160)
      AND mn.name IN ('monitor/router_margin_local', 'monitor/activation_local', 'monitor/weight_local', 'monitor/optimizer_local',
                      'monitor/materialized_weight_local')
    UNION ALL
    SELECT ec.exp, ec.run, ec.attempt, ec.step, ec.node, ec.rank, ec.microbatch, ec.layer,
        ec.metric_id, mn.name,
        multiIf(ec.metric_id = 159,
            arrayConcat(arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.values[2 * i + 1]),
                bitShiftLeft(toUInt64(ec.values[2 * i + 2]), 32))), range(6)),
                arrayMap(x -> toFloat64(x), arraySlice(ec.values, 13, 6))),
            ec.metric_id IN (142, 150, 156),
            arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.values[2 * i + 1]),
                bitShiftLeft(toUInt64(ec.values[2 * i + 2]), 32))), range(multiIf(ec.metric_id = 150, 7, ec.metric_id = 156, 1, 4))),
            arrayMap(x -> toFloat64(x), ec.values)) AS stats,
        multiIf(ec.metric_id = 142, 'parameter', ec.metric_id = 146, 'activation', ec.metric_id = 156, 'learning_rate_group', '') AS catalog_kind,
        toUInt16(ec.layer) AS catalog_index,
        length(ec.values) = multiIf(ec.metric_id = 142, 8, ec.metric_id = 143, 4,
                                   ec.metric_id = 150, 14, ec.metric_id = 154, 3, ec.metric_id = 159, 18, 2) AS width_valid
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    WHERE ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32}
           OR (ec.rank = -1 AND ec.identical AND ec.node IN (SELECT node FROM metric_sampler.node_manifests
               WHERE exp = {exp_filter:String} AND run = {run_filter:String} AND has(ranks, toUInt16(greatest({rank_filter:Int32}, 0))))))
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND (ec.rank >= 0 OR ec.identical) AND ec.source_metric_id = 0
      AND ec.metric_id IN (142, 143, 146, 150, 154, 156, 159)
      AND mn.name IN ('monitor/update_weight_local', 'monitor/topk_local_counts', 'monitor/activation_local_observations',
                      'monitor/router_balance_local', 'monitor/router_balance_counts_local',
                      'monitor/lr_group_local', 'monitor/stage_time_local')
    UNION ALL
    SELECT ec.exp, ec.run, ec.attempt, ec.step, ec.node, ec.rank, ec.microbatch, ec.layer,
        ec.metric_id, mn.name,
        arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.values[12 * group_index + 2 * i + 1]),
            bitShiftLeft(toUInt64(ec.values[12 * group_index + 2 * i + 2]), 32))), range(6)) AS stats,
        'optimizer_group' AS catalog_kind, toUInt16(group_index) AS catalog_index,
        length(ec.values) = 132 AS width_valid
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    ARRAY JOIN range(11) AS group_index
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32}
           OR (ec.rank = -1 AND ec.identical AND ec.node IN (SELECT node FROM metric_sampler.node_manifests
               WHERE exp = {exp_filter:String} AND run = {run_filter:String} AND has(ranks, toUInt16(greatest({rank_filter:Int32}, 0))))))
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND (ec.rank >= 0 OR ec.identical) AND ec.source_metric_id = 0
      AND ec.metric_id = 141 AND mn.name = 'monitor/optimizer_health_local'
), catalog AS
(
    SELECT exp, run, attempt, kind, rank, index, name, 1 AS present
    FROM metric_sampler.monitor_parameters FINAL
)
SELECT o.exp AS exp, o.run AS run, o.attempt AS attempt, o.step AS step,
    o.node AS node, o.rank AS rank, o.microbatch AS microbatch, o.layer AS layer,
    o.metric AS metric, o.stats AS stats, o.width_valid AS width_valid,
    o.catalog_kind AS catalog_kind, o.catalog_index AS catalog_index,
    if(local.present = 1, local.name, global.name) AS source_name,
    o.catalog_kind = '' OR coalesce(local.present, 0) = 1 OR coalesce(global.present, 0) = 1 AS metadata_present,
    multiIf(o.metric_id = 140, ['score_median_gap', 'score_median_near', 'score_p25_unselected', 'score_p10_unselected',
                              'logit_median_gap', 'logit_median_near', 'logit_p25_unselected', 'logit_p10_unselected'],
        o.metric_id = 141, ['numel', 'frac_eps_count', 'frac_10eps_count', 'update_abs_sum', 'grad_contrib_sq_sum', 'wd_contrib_sq_sum'],
        o.metric_id = 142, ['numel', 'update_abs_sum', 'w_sum', 'w_sq_sum'],
        o.metric_id = 143, ['validtokens', 'correct1', 'correct5', 'correct20'],
        o.metric_id = 144, ['amax', 'mean_row_l2'],
        o.metric_id = 145, ['amax', 'whole_shard_l2'],
        o.metric_id = 150, ['normalized_topk_sum', 'raw_topk_sum', 'biased_topk_sum', 'ordinary_biased_score_entropy_sum',
                           'router_z_loss_sum', 'seq_aux_loss_sum', 'alpha_sum'],
        o.metric_id = 154, ['all_tokens', 'valid_tokens', 'forward_calls'],
        o.metric_id = 156, ['learning_rate'],
        o.metric_id = 159, ['batch_total', 'forward', 'loss_calculation', 'backward', 'optimizer_step', 'dataloader_fetch',
                           'batch_total_present', 'forward_present', 'loss_calculation_present', 'backward_present',
                           'optimizer_step_present', 'dataloader_fetch_present'],
        o.metric_id = 160, ['amax', 'whole_tensor_l2', 'present'],
        o.metric_id = 149, ['grad_l2', 'moment_l2', 'moment_sq_l2', 'param_l2', 'update_l2', 'moment_bc_l2',
                           'moment_sq_bc_l2', 'moment_amax', 'moment_sq_amax', 'grad_l2_present', 'moment_l2_present',
                           'moment_sq_l2_present', 'param_l2_present', 'update_l2_present', 'moment_bc_l2_present',
                           'moment_sq_bc_l2_present', 'moment_amax_present', 'moment_sq_amax_present'],
        ['observation_count', 'row_count']) AS stat_names
FROM observations AS o
LEFT JOIN catalog AS local
    ON local.exp = o.exp AND local.run = o.run AND local.attempt = o.attempt
   AND local.kind = o.catalog_kind AND local.rank = o.rank AND local.index = o.catalog_index
LEFT JOIN catalog AS global
    ON global.exp = o.exp AND global.run = o.run AND global.attempt = o.attempt
   AND global.kind = o.catalog_kind AND global.rank = -1 AND global.index = o.catalog_index;

CREATE OR REPLACE VIEW metric_sampler.training_counts_per_rank_selected AS
SELECT ec.exp AS exp, ec.run AS run, ec.attempt AS attempt, ec.step AS step,
    ec.node AS node, ec.rank AS rank, ec.microbatch AS microbatch, ec.layer AS layer,
    arrayMap(i -> bitOr(toUInt64(ec.values[2 * i + 1]),
        bitShiftLeft(toUInt64(ec.values[2 * i + 2]), 32)), range(4)) AS counts,
    arraySlice(ec.values, 9, 4) AS present,
    length(ec.values) = 12 AS width_valid
FROM metric_sampler.expert_counts AS ec FINAL
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
WHERE ec.layer <= 255
  AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
  AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
  AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
  AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
  AND ec.rank >= 0 AND ec.source_metric_id = 0
  AND ec.metric_id = 147 AND mn.name = 'monitor/training_local_counts';

CREATE OR REPLACE VIEW metric_sampler.memory_local_per_rank_selected AS
SELECT ec.exp AS exp, ec.run AS run, ec.attempt AS attempt, ec.step AS step,
    ec.node AS node, ec.rank AS rank, ec.microbatch AS microbatch, ec.layer AS layer,
    arrayMap(i -> bitOr(toUInt64(ec.values[2 * i + 1]),
        bitShiftLeft(toUInt64(ec.values[2 * i + 2]), 32)), range(11)) AS counters,
    arraySlice(ec.values, 23, 11) AS present, length(ec.values) = 33 AS width_valid,
    ['allocated_bytes.all.current', 'active_bytes.all.current', 'inactive_split_bytes.all.current', 'reserved_bytes.all.current',
     'allocated_bytes.all.peak', 'active_bytes.all.peak', 'inactive_split_bytes.all.peak', 'reserved_bytes.all.peak',
     'num_alloc_retries', 'device_free', 'device_total'] AS counter_names
FROM metric_sampler.expert_counts AS ec FINAL
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
WHERE ec.layer <= 255
  AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
  AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
  AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
  AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
  AND ec.rank >= 0 AND ec.source_metric_id = 0
  AND ec.metric_id = 157 AND mn.name = 'monitor/memory_local';

CREATE OR REPLACE VIEW metric_sampler.training_local_per_rank_selected AS
SELECT ec.exp AS exp, ec.run AS run, ec.attempt AS attempt, ec.step AS step,
    ec.node AS node, ec.rank AS rank, ec.microbatch AS microbatch, ec.layer AS layer,
    bitOr(toUInt64(ec.values[1]), bitShiftLeft(toUInt64(ec.values[2]), 32)) AS samples,
    bitOr(toUInt64(ec.values[3]), bitShiftLeft(toUInt64(ec.values[4]), 32)) AS tokens,
    reinterpretAsFloat64(bitOr(toUInt64(ec.values[5]), bitShiftLeft(toUInt64(ec.values[6]), 32))) AS batch_seconds,
    reinterpretAsFloat64(bitOr(toUInt64(ec.values[7]), bitShiftLeft(toUInt64(ec.values[8]), 32))) AS device_flops,
    arraySlice(ec.values, 9, 4) AS present, length(ec.values) = 12 AS width_valid
FROM metric_sampler.expert_counts AS ec FINAL
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
WHERE ec.layer <= 255
  AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
  AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
  AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
  AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
  AND ec.rank >= 0 AND ec.source_metric_id = 0
  AND ec.metric_id = 158 AND mn.name = 'monitor/training_local';

CREATE OR REPLACE VIEW metric_sampler.expert_activation_per_rank_selected AS
WITH sums AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.attempt AS attempt, ec.step AS step,
        ec.node AS node, ec.rank AS rank, ec.microbatch AS microbatch, ec.layer AS layer,
        expert, reinterpretAsFloat64(bitOr(toUInt64(ec.values[2 * expert + 1]),
            bitShiftLeft(toUInt64(ec.values[2 * expert + 2]), 32))) AS sum_squares,
        1 AS present, length(ec.values) % 2 = 0 AS width_valid
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    ARRAY JOIN range(toUInt32(intDiv(length(ec.values) + 1, 2))) AS expert
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND ec.rank >= 0 AND ec.source_metric_id = 0
      AND ec.metric_id = 151 AND mn.name = 'monitor/expert_activation_sumsq_local'
), counts AS
(
    SELECT ec.exp AS exp, ec.run AS run, ec.attempt AS attempt, ec.step AS step,
        ec.node AS node, ec.rank AS rank, ec.microbatch AS microbatch, ec.layer AS layer,
        toUInt32(expert_index - 1) AS expert, token_rows, 1 AS present
    FROM metric_sampler.expert_counts AS ec FINAL
    INNER JOIN metric_sampler.metric_names AS mn
        ON mn.exp = ec.exp AND mn.run = ec.run AND mn.attempt = ec.attempt AND mn.metric_id = ec.metric_id
    ARRAY JOIN ec.values AS token_rows, arrayEnumerate(ec.values) AS expert_index
    WHERE ec.layer <= 255
      AND ({exp_filter:String} = '' OR ec.exp = {exp_filter:String})
      AND ({run_filter:String} = '' OR ec.run = {run_filter:String})
      AND ({rank_filter:Int32} = -2 OR ec.rank = {rank_filter:Int32})
      AND ({microbatch_filter:Int32} = -2 OR ec.microbatch = {microbatch_filter:Int32})
      AND ec.rank >= 0 AND ec.source_metric_id = 0
      AND ec.metric_id = 152 AND mn.name = 'monitor/expert_activation_count_local'
), paired AS
(
    SELECT if(s.present = 1, s.exp, c.exp) AS exp, if(s.present = 1, s.run, c.run) AS run,
        if(s.present = 1, s.attempt, c.attempt) AS attempt, if(s.present = 1, s.step, c.step) AS step,
        if(s.present = 1, s.node, c.node) AS node, if(s.present = 1, s.rank, c.rank) AS rank,
        if(s.present = 1, s.microbatch, c.microbatch) AS microbatch,
        if(s.present = 1, s.layer, c.layer) AS layer, if(s.present = 1, s.expert, c.expert) AS expert,
        s.sum_squares AS sum_squares, c.token_rows AS token_rows,
        coalesce(s.present, 0) = 1 AND coalesce(c.present, 0) = 1 AND s.width_valid AS complete_pair
    FROM sums AS s FULL OUTER JOIN counts AS c
        ON s.exp = c.exp AND s.run = c.run AND s.attempt = c.attempt AND s.step = c.step
       AND s.node = c.node AND s.rank = c.rank AND s.microbatch = c.microbatch AND s.layer = c.layer AND s.expert = c.expert
)
SELECT p.exp AS exp, p.run AS run, p.attempt AS attempt, p.step AS step,
    p.node AS node, p.rank AS rank, p.microbatch AS microbatch, p.layer AS layer, p.expert AS expert,
    p.sum_squares AS sum_squares, p.token_rows AS token_rows, p.complete_pair AS complete_pair,
    coalesce(o.present, 0) = 1 AS metadata_present, has(o.global_expert_ids, p.expert) AS owned,
    if(p.complete_pair AND o.present = 1 AND has(o.global_expert_ids, p.expert) AND p.token_rows > 0,
        sqrt(p.sum_squares / p.token_rows), NULL) AS per_token_l2_rms
FROM paired AS p
LEFT JOIN (SELECT exp, run, attempt, rank, layer, global_expert_ids, 1 AS present FROM metric_sampler.monitor_expert_ownership FINAL) AS o
    ON o.exp = p.exp AND o.run = p.run AND o.attempt = p.attempt AND o.rank = p.rank AND o.layer = p.layer;

CREATE OR REPLACE VIEW metric_sampler.expert_vectors_per_rank_selected AS
SELECT rv.exp AS exp, rv.run AS run, rv.attempt AS attempt, rv.step AS step,
    rv.node AS node, rv.rank AS rank, rv.microbatch AS microbatch, rv.layer AS layer,
    mn.name AS metric, expert_index - 1 AS expert, value
FROM metric_sampler.router_vectors AS rv FINAL
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
ARRAY JOIN rv.values AS value, arrayEnumerate(rv.values) AS expert_index
WHERE rv.layer <= 255
  AND ({exp_filter:String} = '' OR rv.exp = {exp_filter:String})
  AND ({run_filter:String} = '' OR rv.run = {run_filter:String})
  AND ({rank_filter:Int32} = -2 OR rv.rank = {rank_filter:Int32}
           OR (rv.rank = -1 AND rv.identical AND rv.node IN (SELECT node FROM metric_sampler.node_manifests
               WHERE exp = {exp_filter:String} AND run = {run_filter:String} AND has(ranks, toUInt16(greatest({rank_filter:Int32}, 0))))))
  AND ({microbatch_filter:Int32} = -2 OR rv.microbatch = {microbatch_filter:Int32})
  AND (rv.rank >= 0 OR rv.identical) AND rv.source_metric_id = 0
  AND rv.metric_id IN (5, 153) AND mn.name IN ('router/scores_mean', 'monitor/router_bias_local');

-- One row per head of every attention statistics vector. A vector is its row count followed by one block of
-- per-head values for each field, so field f (from 0) of head h is at [2 + f * heads + h]; stats lists the head's
-- fields in the order of stat_names, and the *_sum fields add up over rows (mean = sum / rows).
CREATE OR REPLACE VIEW metric_sampler.attention_heads_per_rank_selected AS
SELECT rv.exp AS exp, rv.run AS run, rv.attempt AS attempt, rv.step AS step,
    rv.node AS node, rv.rank AS rank, rv.microbatch AS microbatch, rv.layer AS layer,
    mn.name AS metric, head, rv.values[1] AS rows,
    multiIf(
        mn.name = 'attn/softmax/lse_stats', ['lse_sum', 'lse_min', 'lse_max'],
        mn.name = 'attn/softmax/qkv_stats', ['q_nope_norm_sum', 'q_rope_norm_sum', 'k_nope_norm_sum', 'k_rope_norm_sum',
                                            'v_norm_sum', 'q_amax', 'k_amax', 'v_amax'],
        mn.name = 'attn/softmax/gate_stats', ['gate_sum', 'gate_min', 'gate_max'],
        mn.name = 'attn/softmax/row_stats', ['ent_sum', 'ent_sumsq', 'ent_min', 'ent_max', 'entn_sum', 'entn_min', 'entn_max',
                                            'pmax_sum', 'pmax_min', 'pmax_max', 'pmin_sum', 'lpmin_sum', 'lpmin_min', 'lpmin_max',
                                            'mlogit_sum', 'mlogit_max', 'coll_sum', 'sink_sum', 'sink_max', 'dist_sum', 'local_sum'],
        mn.name = 'attn/linear/gate_stats', ['beta_sum', 'beta_min', 'beta_max', 'alpha_sum', 'alpha_min', 'alpha_max', 'g_sum'],
        mn.name = 'attn/linear/norm_stats', ['v_norm_sum', 'o_norm_sum', 'o_amax'],
        ['state_norm_sum', 'state_norm_max']) AS stat_names,
    intDiv(length(rv.values) - 1, length(stat_names)) AS heads,
    arrayMap(field -> toFloat64(rv.values[2 + field * heads + head]), range(length(stat_names))) AS stats,
    (length(rv.values) - 1) % length(stat_names) = 0 AS width_valid
FROM metric_sampler.router_vectors AS rv FINAL
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = rv.exp AND mn.run = rv.run AND mn.attempt = rv.attempt AND mn.metric_id = rv.metric_id
ARRAY JOIN range(heads) AS head
WHERE rv.layer <= 255
  AND ({exp_filter:String} = '' OR rv.exp = {exp_filter:String})
  AND ({run_filter:String} = '' OR rv.run = {run_filter:String})
  AND ({rank_filter:Int32} = -2 OR rv.rank = {rank_filter:Int32})
  AND ({microbatch_filter:Int32} = -2 OR rv.microbatch = {microbatch_filter:Int32})
  AND rv.rank >= 0 AND rv.source_metric_id = 0
  AND rv.metric_id IN (SELECT metric_id FROM metric_sampler.metrics WHERE name IN (
      'attn/softmax/lse_stats', 'attn/softmax/qkv_stats', 'attn/softmax/gate_stats', 'attn/softmax/row_stats',
      'attn/linear/gate_stats', 'attn/linear/norm_stats', 'attn/linear/state_stats'))
  AND mn.name IN ('attn/softmax/lse_stats', 'attn/softmax/qkv_stats', 'attn/softmax/gate_stats', 'attn/softmax/row_stats',
                  'attn/linear/gate_stats', 'attn/linear/norm_stats', 'attn/linear/state_stats');
