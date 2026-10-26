CREATE TABLE IF NOT EXISTS metric_sampler.summary_weight_stats
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `param_index` UInt16,
    `amax` Float32,
    `l2` Float32
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, param_index, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.weight_stats_range
AS SELECT
    rv.exp AS exp,
    rv.run AS run,
    rv.step AS step,
    toUInt16(rv.layer) AS param_index,
    max(rv.`values`[1]) AS amax,
    max(rv.`values`[2]) AS l2
FROM metric_sampler.router_vectors AS rv
FINAL
INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
WHERE (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND (mn.name = 'monitor/materialized_weight_local') AND (length(rv.`values`) = 3) AND (rv.`values`[3] = 1) AND (rv.metric_id IN (
    SELECT metric_id
    FROM metric_sampler.metrics
    WHERE name = 'monitor/materialized_weight_local'
)) AND (rv.layer IN (
    SELECT index
    FROM metric_sampler.monitor_parameters
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (kind = 'parameter') AND (name NOT LIKE '%__expert_idx_%')
))
GROUP BY
    rv.exp,
    rv.run,
    rv.step,
    rv.layer;

CREATE OR REPLACE VIEW metric_sampler.weight_stats_all
AS SELECT
    s.exp AS exp,
    s.run AS run,
    s.step AS step,
    p.name AS param,
    toInt16OrNull(extract(p.name, 'layers\\.(\\d+)\\.')) AS model_layer,
    replaceRegexpOne(p.name, '^(?:model\\.|_module\\.)*(?:layers\\.\\d+\\.)?', '') AS projection,
    s.amax AS amax,
    s.l2 AS l2
FROM metric_sampler.summary_weight_stats AS s
FINAL
INNER JOIN
(
    SELECT
        index,
        any(name) AS name
    FROM metric_sampler.monitor_parameters
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (kind = 'parameter')
    GROUP BY index
) AS p ON p.index = s.param_index
WHERE (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String});

CREATE TABLE IF NOT EXISTS metric_sampler.summary_module_activation
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `entry_index` UInt16,
    `amax` Float32,
    `mean_row_l2` Nullable(Float64),
    `rows` UInt64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, entry_index, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.module_activation_range
AS WITH
    stats AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.attempt AS attempt,
            rv.step AS step,
            rv.node AS node,
            rv.rank AS rank,
            rv.microbatch AS microbatch,
            rv.layer AS layer,
            rv.`values`[1] AS amax,
            rv.`values`[2] AS mean_l2
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        WHERE (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND (length(rv.`values`) = 2) AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metric_names
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (name = 'monitor/activation_local')
        )) AND (rv.layer IN (
            SELECT index
            FROM metric_sampler.monitor_parameters
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (kind = 'activation') AND match(name, '(_proj|\\._gate\\.gate):(input|grad_output)\\.0$')
        ))
    ),
    counts AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.attempt AS attempt,
            ec.step AS step,
            ec.node AS node,
            arrayJoin(if(ec.rank = -1, arrayMap(r -> toInt16(r), nr.ranks), [ec.rank])) AS rank,
            ec.microbatch AS microbatch,
            ec.layer AS layer,
            ec.`values`[1] AS observations,
            ec.`values`[2] AS rows
        FROM metric_sampler.expert_counts AS ec
        FINAL
        LEFT JOIN
        (
            SELECT
                attempt,
                node,
                any(ranks) AS ranks
            FROM metric_sampler.node_manifests
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
            GROUP BY
                attempt,
                node
        ) AS nr ON (nr.attempt = ec.attempt) AND (nr.node = ec.node)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND ((ec.step, ec.attempt) IN (
            SELECT
                step,
                attempt
            FROM metric_sampler.latest_attempt_step
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
        )) AND (({rank_filter:Int32} = -2) OR (ec.rank IN ({rank_filter:Int32}, -1))) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (length(ec.`values`) = 2) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metric_names
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (name = 'monitor/activation_local_observations')
        )) AND (ec.layer IN (
            SELECT index
            FROM metric_sampler.monitor_parameters
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (kind = 'activation') AND match(name, '(_proj|\\._gate\\.gate):(input|grad_output)\\.0$')
        ))
    )
SELECT
    stats.exp AS exp,
    stats.run AS run,
    stats.step AS step,
    toUInt16(stats.layer) AS entry_index,
    max(stats.amax) AS amax,
    sum(stats.mean_l2 * counts.rows) / nullIf(sum(counts.rows), 0) AS mean_row_l2,
    sum(counts.rows) AS rows
FROM stats
INNER JOIN counts ON (counts.step = stats.step) AND (counts.layer = stats.layer) AND (counts.rank = stats.rank) AND (counts.microbatch = stats.microbatch) AND (counts.node = stats.node)
WHERE counts.observations > 0
GROUP BY
    stats.exp,
    stats.run,
    stats.step,
    stats.layer;

CREATE OR REPLACE VIEW metric_sampler.module_activation_all
AS SELECT
    s.exp AS exp,
    s.run AS run,
    s.step AS step,
    p.name AS entry,
    toInt16OrNull(extract(p.name, 'layers\\.(\\d+)\\.')) AS model_layer,
    replaceRegexpAll(replaceRegexpOne(p.name, '^.*?layers\\.\\d+\\.', ''), '(root_module|_checkpoint_wrapped_module)\\.', '') AS module,
    s.amax AS amax,
    s.mean_row_l2 AS mean_row_l2,
    s.rows AS rows
FROM metric_sampler.summary_module_activation AS s
FINAL
INNER JOIN
(
    SELECT
        index,
        any(name) AS name
    FROM metric_sampler.monitor_parameters
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (kind = 'activation')
    GROUP BY index
) AS p ON p.index = s.entry_index
WHERE (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String});
