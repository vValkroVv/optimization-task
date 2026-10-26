CREATE TABLE IF NOT EXISTS metric_sampler.summary_loss_components
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `metric` LowCardinality(String),
    `value` Float64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, metric, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_loss_position
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `bucket` Int64,
    `tokens` UInt64,
    `mean` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, bucket)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_loss_quantiles
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `segment` String,
    `tokens` Float64,
    `mean` Float64,
    `p50` Nullable(Float64),
    `p90` Nullable(Float64),
    `p99` Nullable(Float64),
    `p999` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, segment)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_loss_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `metric` LowCardinality(String),
    `step` UInt32,
    `rank` Int16,
    `node` String,
    `value` Float64,
    `observations` UInt64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, metric, rank, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_loss_spread
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `metric` LowCardinality(String),
    `step` UInt32,
    `minimum` Float32,
    `median` Float32,
    `maximum` Float32
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, metric, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_loss_step
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `loss_mean` Float64,
    `loss_min` Float32,
    `loss_max` Float32
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_sample_loss_quantiles
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `stream_id` Int64,
    `stream_name` String,
    `samples` UInt64,
    `p50` Nullable(Float64),
    `p90` Nullable(Float64),
    `p99` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, stream_id)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_step_time
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` Int64,
    `step_time_max` Nullable(Float32),
    `step_time_median` Nullable(Float64),
    `loop_time_max` Nullable(Float32),
    `loop_time_median` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_stream_loss
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `stream_id` Int64,
    `stream_name` String,
    `tokens` UInt64,
    `samples` UInt64,
    `mean` Float64,
    `std` Float64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, stream_id)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_stream_share
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `stream_id` Int64,
    `stream_name` String,
    `samples` UInt64,
    `tokens` UInt64,
    `sample_share` Float64,
    `token_share` Float64,
    `loss_share` Float64,
    `mean` Nullable(Float64),
    `tokens_per_sample` Float64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, stream_id)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.loss_components_range
AS SELECT
    s.exp AS exp,
    s.run AS run,
    s.step AS step,
    mn.name AS metric,
    avg(s.`values`[1]) AS value
FROM metric_sampler.scalars AS s
FINAL
INNER JOIN metric_sampler.latest_attempt_step AS latest ON (s.exp = latest.exp) AND (s.run = latest.run) AND (s.step = latest.step) AND (s.attempt = latest.attempt)
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = s.exp) AND (mn.run = s.run) AND (mn.attempt = s.attempt) AND (mn.metric_id = s.metric_id)
WHERE (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND (s.step <= {step_to:UInt32}) AND (s.metric_id IN (
    SELECT metric_id
    FROM metric_sampler.metrics
    WHERE (name LIKE 'loss/%') AND (name != 'loss/total')
)) AND (mn.name LIKE 'loss/%') AND (mn.name != 'loss/total')
GROUP BY
    s.exp,
    s.run,
    s.step,
    mn.name;

CREATE OR REPLACE VIEW metric_sampler.loss_components_live
AS SELECT
    step,
    metric,
    value
FROM `metric_sampler.loss_components_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, step_from = 0, step_to = 4294967295);

CREATE OR REPLACE VIEW metric_sampler.loss_components_sum
AS SELECT
    step,
    metric,
    value
FROM metric_sampler.summary_loss_components
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.loss_per_rank
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `node` LowCardinality(String),
    `rank` Int16,
    `microbatch` Int16,
    `metric` LowCardinality(String),
    `value` Float32
)
AS SELECT
    s.exp AS exp,
    s.run AS run,
    s.step AS step,
    s.node AS node,
    s.rank AS rank,
    s.microbatch AS microbatch,
    mn.name AS metric,
    s.`values`[1] AS value
FROM metric_sampler.scalars AS s
FINAL
INNER JOIN metric_sampler.latest_attempt_step AS las ON (s.exp = las.exp) AND (s.run = las.run) AND (s.step = las.step) AND (s.attempt = las.attempt)
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = s.exp) AND (mn.run = s.run) AND (mn.attempt = s.attempt) AND (mn.metric_id = s.metric_id)
WHERE (s.layer <= 255) AND startsWith(mn.name, 'loss/') AND (s.rank >= 0) AND (s.metric_id IN (
    SELECT metric_id
    FROM metric_sampler.metrics
    WHERE startsWith(name, 'loss/')
));

CREATE OR REPLACE VIEW metric_sampler.loss_per_rank_selected
AS SELECT
    s.exp AS exp,
    s.run AS run,
    s.step AS step,
    s.node AS node,
    s.rank AS rank,
    s.microbatch AS microbatch,
    mn.name AS metric,
    s.`values`[1] AS value
FROM metric_sampler.scalars AS s
FINAL
INNER JOIN
(
    SELECT
        exp,
        run,
        step,
        attempt
    FROM metric_sampler.latest_attempt_step
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
) AS las ON (s.exp = las.exp) AND (s.run = las.run) AND (s.step = las.step) AND (s.attempt = las.attempt)
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = s.exp) AND (mn.run = s.run) AND (mn.attempt = s.attempt) AND (mn.metric_id = s.metric_id)
WHERE (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.layer <= 255) AND (mn.name = {metric_filter:String}) AND startsWith(mn.name, 'loss/') AND (s.rank >= 0) AND (s.metric_id IN (
    SELECT metric_id
    FROM metric_sampler.metrics
    WHERE name = {metric_filter:String}
));

CREATE OR REPLACE VIEW metric_sampler.loss_rank_values_range
AS SELECT
    s.exp AS exp,
    s.run AS run,
    mn.name AS metric,
    s.step AS step,
    s.node AS node,
    s.rank AS rank,
    s.microbatch AS microbatch,
    s.`values`[1] AS value
FROM metric_sampler.scalars AS s
FINAL
INNER JOIN
(
    SELECT
        exp,
        run,
        step,
        attempt
    FROM metric_sampler.latest_attempt_step
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
) AS las ON (s.exp = las.exp) AND (s.run = las.run) AND (s.step = las.step) AND (s.attempt = las.attempt)
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = s.exp) AND (mn.run = s.run) AND (mn.attempt = s.attempt) AND (mn.metric_id = s.metric_id)
WHERE (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND (s.step <= {step_to:UInt32}) AND (s.layer <= 255) AND startsWith(mn.name, 'loss/') AND (s.rank >= 0) AND (s.metric_id IN (
    SELECT metric_id
    FROM metric_sampler.metrics
    WHERE startsWith(name, 'loss/')
));

CREATE OR REPLACE VIEW metric_sampler.loss_per_rank_step
AS SELECT
    step,
    node,
    rank,
    microbatch,
    value
FROM `metric_sampler.loss_rank_values_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, step_from = {step_filter:UInt32}, step_to = {step_filter:UInt32})
WHERE metric = {metric_filter:String};

CREATE OR REPLACE VIEW metric_sampler.loss_per_step_range
AS SELECT
    s.exp AS exp,
    s.run AS run,
    s.step AS step,
    avg(s.`values`[1]) AS loss_mean,
    min(s.`values`[1]) AS loss_min,
    max(s.`values`[1]) AS loss_max
FROM metric_sampler.scalars AS s
FINAL
INNER JOIN metric_sampler.latest_attempt_step AS las ON (s.exp = las.exp) AND (s.run = las.run) AND (s.step = las.step) AND (s.attempt = las.attempt)
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = s.exp) AND (mn.run = s.run) AND (mn.attempt = s.attempt) AND (mn.metric_id = s.metric_id)
WHERE (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND (s.step <= {step_to:UInt32}) AND (s.layer <= 255) AND (mn.name = 'loss/total') AND (s.metric_id IN (
    SELECT metric_id
    FROM metric_sampler.metrics
    WHERE name = 'loss/total'
))
GROUP BY
    s.exp,
    s.run,
    s.step;

CREATE OR REPLACE VIEW metric_sampler.loss_per_step_live
AS SELECT
    step,
    loss_mean,
    loss_min,
    loss_max
FROM `metric_sampler.loss_per_step_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, step_from = 0, step_to = 4294967295);

CREATE OR REPLACE VIEW metric_sampler.loss_per_step_sum
AS SELECT
    step,
    loss_mean,
    loss_min,
    loss_max
FROM metric_sampler.summary_loss_step
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.loss_position_per_step_fast
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `bucket` Int64,
    `tokens` UInt64,
    `mean` Nullable(Float64)
)
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_loss_position
FINAL;

CREATE OR REPLACE VIEW metric_sampler.loss_position_per_step_live
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `bucket` Int64,
    `tokens` UInt64,
    `mean` Nullable(Float64)
)
AS SELECT *
FROM metric_sampler.loss_position_per_step;

CREATE OR REPLACE VIEW metric_sampler.loss_position_per_step_range
AS WITH
    position_sum AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            sumForEach(rv.`values`) AS `values`
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (mn.name = 'loss/position_sum') AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/position_sum'
        ))
        GROUP BY
            rv.exp,
            rv.run,
            rv.step
    ),
    position_count AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            sumForEach(ec.`values`) AS `values`
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (mn.name = 'loss/position_count') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/position_count'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step
    )
SELECT
    position_count.exp AS exp,
    position_count.run AS run,
    position_count.step AS step,
    bucket_index - 1 AS bucket,
    position_count.`values`[bucket_index] AS tokens,
    (position_sum.`values`[bucket_index]) / nullIf(position_count.`values`[bucket_index], 0) AS mean
FROM position_count
ARRAY JOIN arrayEnumerate(position_count.`values`) AS bucket_index
INNER JOIN position_sum ON (position_sum.exp = position_count.exp) AND (position_sum.run = position_count.run) AND (position_sum.step = position_count.step);

CREATE OR REPLACE VIEW metric_sampler.loss_quantiles_fast
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `segment` String,
    `tokens` Float64,
    `mean` Float64,
    `p50` Nullable(Float64),
    `p90` Nullable(Float64),
    `p99` Nullable(Float64),
    `p999` Nullable(Float64)
)
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_loss_quantiles
FINAL;

CREATE OR REPLACE VIEW metric_sampler.loss_quantiles_live
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `segment` String,
    `tokens` Float64,
    `mean` Float64,
    `p50` Nullable(Float64),
    `p90` Nullable(Float64),
    `p99` Nullable(Float64),
    `p999` Nullable(Float64)
)
AS SELECT *
FROM metric_sampler.loss_quantiles_per_step;

CREATE OR REPLACE VIEW metric_sampler.loss_quantiles_per_step_range
AS WITH
    main_summed AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            any(mn.bins_start) AS bins_start,
            any(mn.bins_width) AS bins_width,
            any(mn.bins_count) AS bins_count,
            sumForEach(ec.`values`) AS by_stream
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (mn.name = 'loss/token_hist_by_stream') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/token_hist_by_stream'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step
    ),
    main_hist AS
    (
        SELECT
            exp,
            run,
            step,
            'main' AS segment,
            bins_start,
            bins_width,
            bins_count,
            arrayMap(x -> toFloat64(x), arrayReduce('sumForEach', arrayMap(s -> arraySlice(by_stream, (s * bins_count) + 1, bins_count), range(intDiv(length(by_stream), bins_count))))) AS hist
        FROM main_summed
    ),
    mtp_hist AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            toString(ec.layer) AS segment,
            any(mn.bins_start) AS bins_start,
            any(mn.bins_width) AS bins_width,
            any(mn.bins_count) AS bins_count,
            arrayMap(x -> toFloat64(x), sumForEach(ec.`values`)) AS hist
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (mn.name = 'loss/mtp_token_hist') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/mtp_token_hist'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            segment,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS tokens,
            arrayCumSum(hist) AS cum
        FROM main_hist
        UNION ALL
        SELECT
            exp,
            run,
            step,
            segment,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS tokens,
            arrayCumSum(hist) AS cum
        FROM mtp_hist
    )
SELECT
    exp,
    run,
    step,
    segment,
    tokens,
    arraySum(arrayMap((c, i) -> (c * (bins_start + (bins_width * (i - 0.5)))), hist, arrayEnumerate(hist))) / tokens AS mean,
    multiIf(tokens = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) - 0.5))) AS p50,
    multiIf(tokens = 0, NULL, (arrayFirstIndex(x -> (x >= (0.9 * tokens)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.9 * tokens)), cum) - 0.5))) AS p90,
    multiIf(tokens = 0, NULL, (arrayFirstIndex(x -> (x >= (0.99 * tokens)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.99 * tokens)), cum) - 0.5))) AS p99,
    multiIf(tokens = 0, NULL, (arrayFirstIndex(x -> (x >= (0.999 * tokens)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.999 * tokens)), cum) - 0.5))) AS p999
FROM with_totals;

CREATE OR REPLACE VIEW metric_sampler.loss_rank_live
AS SELECT
    step,
    rank,
    any(node) AS node,
    avg(value) AS value
FROM `metric_sampler.loss_per_rank_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, metric_filter = {metric_filter:String})
WHERE ({microbatch_filter:Int32} = -2) OR (microbatch = {microbatch_filter:Int32})
GROUP BY
    step,
    rank;

CREATE OR REPLACE VIEW metric_sampler.loss_rank_range
AS SELECT
    exp,
    run,
    metric,
    step,
    rank,
    any(node) AS node,
    avg(value) AS value,
    count() AS observations
FROM `metric_sampler.loss_rank_values_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
GROUP BY
    exp,
    run,
    metric,
    step,
    rank;

CREATE OR REPLACE VIEW metric_sampler.loss_rank_sum
AS SELECT
    step,
    rank,
    node,
    value
FROM metric_sampler.summary_loss_rank
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (metric = {metric_filter:String});

CREATE OR REPLACE VIEW metric_sampler.loss_spread_live
AS SELECT
    step,
    min(value) AS min,
    quantileExact(0.5)(value) AS median,
    max(value) AS max
FROM `metric_sampler.loss_per_rank_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, metric_filter = {metric_filter:String})
WHERE ({microbatch_filter:Int32} = -2) OR (microbatch = {microbatch_filter:Int32})
GROUP BY step;

CREATE OR REPLACE VIEW metric_sampler.loss_spread_range
AS SELECT
    exp,
    run,
    metric,
    step,
    min(value) AS minimum,
    quantileExact(0.5)(value) AS median,
    max(value) AS maximum
FROM `metric_sampler.loss_rank_values_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
GROUP BY
    exp,
    run,
    metric,
    step;

CREATE OR REPLACE VIEW metric_sampler.loss_spread_sum
AS SELECT
    step,
    minimum AS min,
    median,
    maximum AS max
FROM metric_sampler.summary_loss_spread
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (metric = {metric_filter:String});

CREATE OR REPLACE VIEW metric_sampler.sample_loss_quantiles_per_step_fast
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `stream_id` Int64,
    `stream_name` String,
    `samples` UInt64,
    `p50` Nullable(Float64),
    `p90` Nullable(Float64),
    `p99` Nullable(Float64)
)
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_sample_loss_quantiles
FINAL;

CREATE OR REPLACE VIEW metric_sampler.sample_loss_quantiles_per_step_live
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `stream_id` Int64,
    `stream_name` String,
    `samples` UInt64,
    `p50` Nullable(Float64),
    `p90` Nullable(Float64),
    `p99` Nullable(Float64)
)
AS SELECT *
FROM metric_sampler.sample_loss_quantiles_per_step;

CREATE OR REPLACE VIEW metric_sampler.sample_loss_quantiles_per_step_range
AS WITH
    hist_summed AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            any(mn.bins_start) AS bins_start,
            any(mn.bins_width) AS bins_width,
            any(mn.bins_count) AS bins_count,
            sumForEach(ec.`values`) AS by_stream
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (mn.name = 'loss/sample_hist_by_stream') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/sample_hist_by_stream'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step
    ),
    per_stream AS
    (
        SELECT
            exp,
            run,
            step,
            bins_start,
            bins_width,
            bins_count,
            arrayMap(s -> arraySlice(by_stream, (s * bins_count) + 1, bins_count), range(intDiv(length(by_stream), bins_count))) AS hists
        FROM hist_summed
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            stream_index - 1 AS stream_id,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS samples,
            arrayCumSum(hist) AS cum
        FROM per_stream
        ARRAY JOIN
            hists AS hist,
            arrayEnumerate(hists) AS stream_index
    )
SELECT
    exp,
    run,
    step,
    stream_id,
    coalesce(stream_names.name, concat('stream', toString(stream_id))) AS stream_name,
    samples,
    multiIf(samples = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * samples)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * samples)), cum) - 0.5))) AS p50,
    multiIf(samples = 0, NULL, (arrayFirstIndex(x -> (x >= (0.9 * samples)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.9 * samples)), cum) - 0.5))) AS p90,
    multiIf(samples = 0, NULL, (arrayFirstIndex(x -> (x >= (0.99 * samples)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.99 * samples)), cum) - 0.5))) AS p99
FROM with_totals
LEFT JOIN metric_sampler.stream_names AS stream_names ON (stream_names.exp = with_totals.exp) AND (stream_names.run = with_totals.run) AND (stream_names.stream_id = with_totals.stream_id);

CREATE OR REPLACE VIEW metric_sampler.step_time_range
AS SELECT
    exp,
    run,
    step,
    step_time_max,
    step_time_median,
    loop_time_max,
    loop_time_median
FROM
(
    SELECT
        s.exp AS exp,
        s.run AS run,
        if(mn.name = 'time/loop_s', s.step - 1, s.step) AS step,
        maxOrNullIf(s.`values`[1], mn.name = 'time/step_s') AS step_time_max,
        medianOrNullIf(s.`values`[1], mn.name = 'time/step_s') AS step_time_median,
        maxOrNullIf(s.`values`[1], mn.name = 'time/loop_s') AS loop_time_max,
        medianOrNullIf(s.`values`[1], mn.name = 'time/loop_s') AS loop_time_median
    FROM metric_sampler.scalars AS s
    FINAL
    INNER JOIN metric_sampler.latest_attempt_step AS latest ON (s.exp = latest.exp) AND (s.run = latest.run) AND (s.step = latest.step) AND (s.attempt = latest.attempt)
    INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = s.exp) AND (mn.run = s.run) AND (mn.attempt = s.attempt) AND (mn.metric_id = s.metric_id)
    WHERE (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND (s.step <= ({step_to:UInt32} + 1)) AND (s.layer <= 255) AND (s.metric_id IN (
        SELECT metric_id
        FROM metric_sampler.metrics
        WHERE name IN ('time/step_s', 'time/loop_s')
    )) AND (mn.name IN ('time/step_s', 'time/loop_s'))
    GROUP BY
        s.exp,
        s.run,
        step
)
WHERE (step > 0) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32});

CREATE OR REPLACE VIEW metric_sampler.step_time_live
AS SELECT
    step,
    step_time_max,
    step_time_median,
    loop_time_max,
    loop_time_median
FROM `metric_sampler.step_time_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, step_from = 0, step_to = 4294967294);

CREATE OR REPLACE VIEW metric_sampler.step_time_sum
AS SELECT
    step,
    step_time_max,
    step_time_median,
    loop_time_max,
    loop_time_median
FROM metric_sampler.summary_step_time
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.stream_loss_fast
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `stream_id` Int64,
    `stream_name` String,
    `tokens` UInt64,
    `samples` UInt64,
    `mean` Float64,
    `std` Float64
)
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_stream_loss
FINAL;

CREATE OR REPLACE VIEW metric_sampler.stream_loss_live
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `stream_id` Int64,
    `stream_name` String,
    `tokens` UInt64,
    `samples` UInt64,
    `mean` Float64,
    `std` Float64
)
AS SELECT *
FROM metric_sampler.stream_loss_per_step;

CREATE OR REPLACE VIEW metric_sampler.stream_loss_per_step_range
AS WITH
    stream_sum AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            sumForEach(rv.`values`) AS `values`
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (mn.name = 'loss/stream_sum') AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/stream_sum'
        ))
        GROUP BY
            rv.exp,
            rv.run,
            rv.step
    ),
    stream_sumsq AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            sumForEach(rv.`values`) AS `values`
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (mn.name = 'loss/stream_sumsq') AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/stream_sumsq'
        ))
        GROUP BY
            rv.exp,
            rv.run,
            rv.step
    ),
    stream_count AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            sumForEach(ec.`values`) AS `values`
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (mn.name = 'loss/stream_count') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/stream_count'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step
    ),
    stream_samples AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            sumForEach(ec.`values`) AS `values`
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (mn.name = 'loss/stream_samples') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/stream_samples'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step
    )
SELECT
    stream_count.exp AS exp,
    stream_count.run AS run,
    stream_count.step AS step,
    stream_index - 1 AS stream_id,
    coalesce(stream_names.name, concat('stream', toString(stream_index - 1))) AS stream_name,
    stream_count.`values`[stream_index] AS tokens,
    stream_samples.`values`[stream_index] AS samples,
    (stream_sum.`values`[stream_index]) / (stream_count.`values`[stream_index]) AS mean,
    sqrt(greatest(((stream_sumsq.`values`[stream_index]) / (stream_count.`values`[stream_index])) - pow((stream_sum.`values`[stream_index]) / (stream_count.`values`[stream_index]), 2), 0)) AS std
FROM stream_count
ARRAY JOIN arrayEnumerate(stream_count.`values`) AS stream_index
INNER JOIN stream_sum ON (stream_sum.exp = stream_count.exp) AND (stream_sum.run = stream_count.run) AND (stream_sum.step = stream_count.step)
INNER JOIN stream_sumsq ON (stream_sumsq.exp = stream_count.exp) AND (stream_sumsq.run = stream_count.run) AND (stream_sumsq.step = stream_count.step)
INNER JOIN stream_samples ON (stream_samples.exp = stream_count.exp) AND (stream_samples.run = stream_count.run) AND (stream_samples.step = stream_count.step)
LEFT JOIN metric_sampler.stream_names AS stream_names ON (stream_names.exp = stream_count.exp) AND (stream_names.run = stream_count.run) AND (stream_names.stream_id = (stream_index - 1))
WHERE (stream_count.`values`[stream_index]) > 0;

CREATE OR REPLACE VIEW metric_sampler.stream_share_fast
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `stream_id` Int64,
    `stream_name` String,
    `samples` UInt64,
    `tokens` UInt64,
    `sample_share` Float64,
    `token_share` Float64,
    `loss_share` Float64,
    `mean` Nullable(Float64),
    `tokens_per_sample` Float64
)
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_stream_share
FINAL;

CREATE OR REPLACE VIEW metric_sampler.stream_share_live
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `stream_id` Int64,
    `stream_name` String,
    `samples` UInt64,
    `tokens` UInt64,
    `sample_share` Float64,
    `token_share` Float64,
    `loss_share` Float64,
    `mean` Nullable(Float64),
    `tokens_per_sample` Float64
)
AS SELECT *
FROM metric_sampler.stream_share_per_step;

CREATE OR REPLACE VIEW metric_sampler.stream_share_per_step_range
AS WITH
    stream_samples AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            sumForEach(ec.`values`) AS `values`
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (mn.name = 'loss/stream_samples') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/stream_samples'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step
    ),
    stream_count AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            sumForEach(ec.`values`) AS `values`
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (mn.name = 'loss/stream_count') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/stream_count'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step
    ),
    stream_sum AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            sumForEach(rv.`values`) AS `values`
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (mn.name = 'loss/stream_sum') AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'loss/stream_sum'
        ))
        GROUP BY
            rv.exp,
            rv.run,
            rv.step
    )
SELECT
    stream_samples.exp AS exp,
    stream_samples.run AS run,
    stream_samples.step AS step,
    stream_index - 1 AS stream_id,
    coalesce(stream_names.name, concat('stream', toString(stream_index - 1))) AS stream_name,
    stream_samples.`values`[stream_index] AS samples,
    stream_count.`values`[stream_index] AS tokens,
    (stream_samples.`values`[stream_index]) / arraySum(stream_samples.`values`) AS sample_share,
    (stream_count.`values`[stream_index]) / arraySum(stream_count.`values`) AS token_share,
    (stream_sum.`values`[stream_index]) / arraySum(stream_sum.`values`) AS loss_share,
    (stream_sum.`values`[stream_index]) / nullIf(stream_count.`values`[stream_index], 0) AS mean,
    (stream_count.`values`[stream_index]) / (stream_samples.`values`[stream_index]) AS tokens_per_sample
FROM stream_samples
ARRAY JOIN arrayEnumerate(stream_samples.`values`) AS stream_index
INNER JOIN stream_count ON (stream_count.exp = stream_samples.exp) AND (stream_count.run = stream_samples.run) AND (stream_count.step = stream_samples.step)
INNER JOIN stream_sum ON (stream_sum.exp = stream_samples.exp) AND (stream_sum.run = stream_samples.run) AND (stream_sum.step = stream_samples.step)
LEFT JOIN metric_sampler.stream_names AS stream_names ON (stream_names.exp = stream_samples.exp) AND (stream_names.run = stream_samples.run) AND (stream_names.stream_id = (stream_index - 1))
WHERE ((stream_samples.`values`[stream_index]) > 0) OR ((stream_count.`values`[stream_index]) > 0);

CREATE TABLE IF NOT EXISTS metric_sampler.summary_loss_rank_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `metric` LowCardinality(String),
    `microbatch` Int16,
    `step` UInt32,
    `rank` Int16,
    `node` String,
    `value` Float64,
    `observations` UInt64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, metric, microbatch, rank, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.loss_rank_mb_range
AS SELECT
    exp,
    run,
    metric,
    microbatch,
    step,
    rank,
    any(node) AS node,
    avg(value) AS value,
    count() AS observations
FROM `metric_sampler.loss_rank_values_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
GROUP BY
    exp,
    run,
    metric,
    microbatch,
    step,
    rank;

CREATE OR REPLACE VIEW metric_sampler.loss_rank_mbsum
AS SELECT
    step,
    rank,
    node,
    value
FROM metric_sampler.summary_loss_rank_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (metric = {metric_filter:String}) AND (microbatch = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.loss_spread_mbsum
AS SELECT
    step,
    min(value) AS min,
    quantileExact(0.5)(value) AS median,
    max(value) AS max
FROM metric_sampler.summary_loss_rank_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (metric = {metric_filter:String}) AND (microbatch = {microbatch_filter:Int32})
GROUP BY step;

CREATE OR REPLACE VIEW metric_sampler.worst_samples_run
AS WITH candidates AS
    (
        SELECT
            info.exp AS exp,
            info.run AS run,
            info.attempt AS attempt,
            info.step AS step,
            info.node AS node,
            info.rank AS rank,
            info.microbatch AS microbatch,
            sample.1 AS stream_id,
            sample.2 AS shard_id,
            sample.3 AS index_in_shard,
            sample.4 AS loss_position,
            sample.5 AS finite_tokens,
            sample.6 AS nonfinite_tokens,
            if((sample.5) > 0, sample.7, NULL) AS mean_loss,
            if((sample.5) > 0, sample.8, NULL) AS max_loss
        FROM
        (
            SELECT *
            FROM metric_sampler.expert_counts
            FINAL
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (metric_id IN (
                SELECT metric_id
                FROM metric_sampler.metrics
                WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (name = 'loss/worst_sample_info')
            )) AND (source_metric_id = 0) AND (layer = -1) AND (length(`values`) = 24)
        ) AS info
        INNER JOIN
        (
            SELECT *
            FROM metric_sampler.router_vectors
            FINAL
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (metric_id IN (
                SELECT metric_id
                FROM metric_sampler.metrics
                WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (name = 'loss/worst_sample_stats')
            )) AND (source_metric_id = 0) AND (layer = -1) AND (length(`values`) = 8)
        ) AS stats ON (info.exp = stats.exp) AND (info.run = stats.run) AND (info.attempt = stats.attempt) AND (info.node = stats.node) AND (info.step = stats.step) AND (info.rank = stats.rank) AND (info.microbatch = stats.microbatch)
        INNER JOIN
        (
            SELECT *
            FROM metric_sampler.latest_attempt_step
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
        ) AS latest ON (info.exp = latest.exp) AND (info.run = latest.run) AND (info.step = latest.step) AND (info.attempt = latest.attempt)
        ARRAY JOIN arrayMap(i -> tuple(info.`values`[(6 * i) + 1], info.`values`[(6 * i) + 2], info.`values`[(6 * i) + 3], info.`values`[(6 * i) + 4], info.`values`[(6 * i) + 5], info.`values`[(6 * i) + 6], stats.`values`[(2 * i) + 1], stats.`values`[(2 * i) + 2]), range(4)) AS sample
        WHERE ((sample.5) > 0) OR ((sample.6) > 0)
    )
SELECT
    c.*,
    if(s.name = '', concat('stream_', toString(c.stream_id)), s.name) AS stream
FROM candidates AS c
LEFT JOIN
(
    SELECT *
    FROM metric_sampler.streams
    FINAL
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
) AS s ON (c.exp = s.exp) AND (c.run = s.run) AND (c.attempt = s.attempt) AND (c.stream_id = s.stream_id);
