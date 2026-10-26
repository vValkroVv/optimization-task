-- summary_*: per-step results of the matching *_range view, kept current by one summaries loop per table. Every
-- table here can be rebuilt from the raw tables.
-- summary_progress holds, per summary and run, the last step that is final; a step is final only up to
-- complete_steps, through which every node's windows and every family of their batches are ingested.
-- The dashboards choose between routes with hidden variables: *_live views read the raw tables and the
-- other routes the summaries. A summary route is chosen only when summary_progress covers the run, so no
-- route hides data.
-- Summary tables compress with ZSTD(3): ZSTD(9) costs about twice the CPU of every insert and merge for ~1% less
-- space.

CREATE TABLE IF NOT EXISTS metric_sampler.summary_progress
(
    `summary` LowCardinality(String),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `complete_through` UInt32,
    `refreshed_at` DateTime64(3)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (summary, exp, run)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.complete_steps
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `complete_through` Int64,
    `last_ingested` DateTime64(3)
)
AS WITH
    latest AS
    (
        SELECT
            exp,
            run,
            argMax(attempt, written_at) AS attempt,
            max(nodes_in_job) AS nodes_in_job,
            argMax(capture_interval, written_at) AS capture_interval
        FROM metric_sampler.node_manifests
        GROUP BY
            exp,
            run
    ),
    processed AS
    (
        SELECT
            extract(object, 'exp=([^/]+)') AS exp,
            extract(object, 'run=([^/]+)') AS run,
            extract(object, 'attempt=([^/]+)') AS attempt,
            extract(object, 'part-\\d+-\\d+-([^/.]+)\\.parquet$') AS batch,
            groupUniqArray(kind) AS kinds
        FROM metric_sampler.processed_objects
        WHERE kind IN ('scalars', 'expert_counts', 'router_vectors')
        GROUP BY
            exp,
            run,
            attempt,
            batch
    ),
    windows AS
    (
        SELECT
            exp,
            run,
            node,
            first,
            last,
            max(complete) AS complete,
            max(ingested_at) AS ingested_at
        FROM
        (
            SELECT
                b.exp AS exp,
                b.run AS run,
                extract(w, '^node=([^/]+)/') AS node,
                toUInt32(extract(w, '/w(\\d+)-')) AS first,
                toUInt32(extract(w, '-(\\d+)$')) AS last,
                arrayAll(f -> (((b.rows_per_family[f]) = 0) OR has(p.kinds, f)), ['scalars', 'expert_counts', 'router_vectors']) AS complete,
                b.ingested_at AS ingested_at
            FROM metric_sampler.batches AS b
            ARRAY JOIN b.source_windows AS w
            LEFT JOIN processed AS p ON (p.exp = b.exp) AND (p.run = b.run) AND (p.attempt = b.attempt) AND (p.batch = b.batch)
            WHERE ((b.exp, b.run, b.attempt) IN (
                SELECT
                    exp,
                    run,
                    attempt
                FROM latest
            )) AND startsWith(w, 'node=')
        )
        GROUP BY
            exp,
            run,
            node,
            first,
            last
    ),
    nodes AS
    (
        SELECT
            w.exp AS exp,
            w.run AS run,
            w.node AS node,
            any(s.start) AS start,
            any(s.capture_interval) AS capture_interval,
            arraySort(x -> (x.1), groupArray((w.first, w.last, w.complete))) AS ws,
            countIf(NOT w.complete) AS pending,
            max(w.last) AS node_last,
            max(w.ingested_at) AS ingested_at
        FROM windows AS w
        INNER JOIN
        (
            SELECT
                windows.exp AS exp,
                windows.run AS run,
                min(windows.first) AS start,
                any(latest.capture_interval) AS capture_interval
            FROM windows
            INNER JOIN latest ON (latest.exp = windows.exp) AND (latest.run = windows.run)
            GROUP BY
                windows.exp,
                windows.run
        ) AS s ON (s.exp = w.exp) AND (s.run = w.run)
        GROUP BY
            w.exp,
            w.run,
            w.node
    ),
    node_frontier AS
    (
        SELECT
            exp,
            run,
            pending,
            node_last,
            ingested_at,
            -- Consecutive windows are capture_interval steps apart.
            arrayFirstIndex(i -> ((NOT ((ws[i]).3)) OR (((ws[i]).1) != if(i = 1, start, ((ws[i - 1]).2) + capture_interval))), arrayEnumerate(ws)) AS gap,
            if(gap = 0, toInt64((ws[-1]).2), if(gap = 1, toInt64(start) - 1, toInt64((ws[gap - 1]).2))) AS frontier
        FROM nodes
    )
SELECT
    f.exp AS exp,
    f.run AS run,
    multiIf((sum(f.pending) = 0) AND (max(f.ingested_at) < (now() - toIntervalHour(1))), toInt64(max(f.node_last)), (count() >= any(l.nodes_in_job)) OR (any(l.nodes_in_job) = 0), min(f.frontier), toInt64(-1)) AS complete_through,
    max(f.ingested_at) AS last_ingested
FROM node_frontier AS f
INNER JOIN latest AS l ON (l.exp = f.exp) AND (l.run = f.run)
GROUP BY
    f.exp,
    f.run
SETTINGS join_use_nulls = 0;

-- complete_steps aggregates every node manifest, processed object and batch to answer which steps are complete,
-- which is too costly to ask from every summary loop and dashboard load: a refresh every minute into a small
-- table answers them all.
CREATE TABLE IF NOT EXISTS metric_sampler.complete_steps_latest
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `complete_through` Int64,
    `last_ingested` DateTime64(3)
)
ENGINE = MergeTree
ORDER BY (exp, run);

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.complete_steps_refresh
REFRESH EVERY 1 MINUTE TO metric_sampler.complete_steps_latest
AS SELECT exp, run, complete_through, last_ingested
FROM metric_sampler.complete_steps;

-- Every summary loop needs each run's newest ingested step, when it arrived and how many ranks its nodes have;
-- reading step_index and node_manifests for that is costly, so a refresh every minute into a small table answers
-- them all.
CREATE TABLE IF NOT EXISTS metric_sampler.run_steps_latest
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `max_step` UInt32,
    `ingested` DateTime64(3),
    `rank_count` UInt64
)
ENGINE = MergeTree
ORDER BY (exp, run);

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.run_steps_refresh
REFRESH EVERY 1 MINUTE TO metric_sampler.run_steps_latest
AS SELECT si.exp AS exp, si.run AS run, si.max_step AS max_step, si.ingested AS ingested, manifests.rank_count AS rank_count
FROM
(
    SELECT exp, run, max(step) AS max_step, max(max_ingested_at) AS ingested
    FROM metric_sampler.step_index
    GROUP BY exp, run
) AS si
LEFT JOIN
(
    SELECT exp, run, uniqExact(rank) AS rank_count
    FROM metric_sampler.node_manifests
    ARRAY JOIN ranks AS rank
    GROUP BY exp, run
) AS manifests ON (manifests.exp = si.exp) AND (manifests.run = si.run)
SETTINGS join_use_nulls = 0;

CREATE OR REPLACE VIEW metric_sampler.latest_step_limit
AS SELECT if((count() = 0) OR (max(complete_through) <= 0), toUInt32(4294967295), toUInt32(max(complete_through))) AS step_limit
FROM metric_sampler.complete_steps_latest
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});
