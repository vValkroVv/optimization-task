CREATE TABLE IF NOT EXISTS metric_sampler.summary_router_balance
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `top_k_normalized` Nullable(Float64),
    `top_k_raw` Nullable(Float64),
    `top_k_biased` Nullable(Float64),
    `entropy` Nullable(Float64),
    `z_loss` Nullable(Float64),
    `seq_aux_loss` Nullable(Float64),
    `alpha` Nullable(Float64),
    `all_tokens` UInt64,
    `valid_tokens` UInt64,
    `forward_calls` UInt64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, layer, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.router_balance_range
AS WITH
    totals AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            sumForEachIf(arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[(2 * i) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * i) + 2]), 32))), range(7)), (mn.name = 'monitor/router_balance_local') AND (length(ec.`values`) = 14)) AS sums,
            sumForEachIf(arrayMap(x -> toUInt64(x), ec.`values`), (mn.name = 'monitor/router_balance_counts_local') AND (length(ec.`values`) = 3)) AS counts
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('monitor/router_balance_local', 'monitor/router_balance_counts_local')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('monitor/router_balance_local', 'monitor/router_balance_counts_local')
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer
    )
SELECT
    exp,
    run,
    step,
    layer,
    if((length(sums) = 7) AND (length(counts) = 3), sums[1] / nullIf(counts[2], 0), NULL) AS top_k_normalized,
    if((length(sums) = 7) AND (length(counts) = 3), sums[2] / nullIf(counts[2], 0), NULL) AS top_k_raw,
    if((length(sums) = 7) AND (length(counts) = 3), sums[3] / nullIf(counts[2], 0), NULL) AS top_k_biased,
    if((length(sums) = 7) AND (length(counts) = 3), sums[4] / nullIf(counts[1], 0), NULL) AS entropy,
    if((length(sums) = 7) AND (length(counts) = 3), sums[5] / nullIf(counts[3], 0), NULL) AS z_loss,
    if((length(sums) = 7) AND (length(counts) = 3), sums[6] / nullIf(counts[3], 0), NULL) AS seq_aux_loss,
    if((length(sums) = 7) AND (length(counts) = 3), sums[7] / nullIf(counts[3], 0), NULL) AS alpha,
    if(length(counts) = 3, counts[1], 0) AS all_tokens,
    if(length(counts) = 3, counts[2], 0) AS valid_tokens,
    if(length(counts) = 3, counts[3], 0) AS forward_calls
FROM totals;

CREATE OR REPLACE VIEW metric_sampler.router_balance_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_router_balance
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.router_balance_live
AS SELECT *
FROM `metric_sampler.router_balance_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = 0, step_to = 4294967295);

CREATE TABLE IF NOT EXISTS metric_sampler.summary_router_balance_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `rank` Int16,
    `top_k_normalized` Nullable(Float64),
    `top_k_raw` Nullable(Float64),
    `top_k_biased` Nullable(Float64),
    `entropy` Nullable(Float64),
    `z_loss` Nullable(Float64),
    `seq_aux_loss` Nullable(Float64),
    `alpha` Nullable(Float64),
    `all_tokens` UInt64,
    `valid_tokens` UInt64,
    `forward_calls` UInt64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, rank, layer, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.router_balance_by_rank_range
AS WITH
    totals AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            ec.rank AS rank,
            sumForEachIf(arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[(2 * i) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * i) + 2]), 32))), range(7)), (mn.name = 'monitor/router_balance_local') AND (length(ec.`values`) = 14)) AS sums,
            sumForEachIf(arrayMap(x -> toUInt64(x), ec.`values`), (mn.name = 'monitor/router_balance_counts_local') AND (length(ec.`values`) = 3)) AS counts
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('monitor/router_balance_local', 'monitor/router_balance_counts_local')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('monitor/router_balance_local', 'monitor/router_balance_counts_local')
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer,
            ec.rank
    )
SELECT
    exp,
    run,
    step,
    layer,
    rank,
    if((length(sums) = 7) AND (length(counts) = 3), sums[1] / nullIf(counts[2], 0), NULL) AS top_k_normalized,
    if((length(sums) = 7) AND (length(counts) = 3), sums[2] / nullIf(counts[2], 0), NULL) AS top_k_raw,
    if((length(sums) = 7) AND (length(counts) = 3), sums[3] / nullIf(counts[2], 0), NULL) AS top_k_biased,
    if((length(sums) = 7) AND (length(counts) = 3), sums[4] / nullIf(counts[1], 0), NULL) AS entropy,
    if((length(sums) = 7) AND (length(counts) = 3), sums[5] / nullIf(counts[3], 0), NULL) AS z_loss,
    if((length(sums) = 7) AND (length(counts) = 3), sums[6] / nullIf(counts[3], 0), NULL) AS seq_aux_loss,
    if((length(sums) = 7) AND (length(counts) = 3), sums[7] / nullIf(counts[3], 0), NULL) AS alpha,
    if(length(counts) = 3, counts[1], 0) AS all_tokens,
    if(length(counts) = 3, counts[2], 0) AS valid_tokens,
    if(length(counts) = 3, counts[3], 0) AS forward_calls
FROM totals;

CREATE OR REPLACE VIEW metric_sampler.router_balance_ranksum
AS SELECT * EXCEPT (refreshed_at, rank)
FROM metric_sampler.summary_router_balance_rank
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32});

CREATE TABLE IF NOT EXISTS metric_sampler.summary_router_balance_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `microbatch` Int16,
    `top_k_normalized` Nullable(Float64),
    `top_k_raw` Nullable(Float64),
    `top_k_biased` Nullable(Float64),
    `entropy` Nullable(Float64),
    `z_loss` Nullable(Float64),
    `seq_aux_loss` Nullable(Float64),
    `alpha` Nullable(Float64),
    `all_tokens` UInt64,
    `valid_tokens` UInt64,
    `forward_calls` UInt64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, microbatch, layer, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.router_balance_by_mb_range
AS WITH
    totals AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            ec.microbatch AS microbatch,
            sumForEachIf(arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[(2 * i) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * i) + 2]), 32))), range(7)), (mn.name = 'monitor/router_balance_local') AND (length(ec.`values`) = 14)) AS sums,
            sumForEachIf(arrayMap(x -> toUInt64(x), ec.`values`), (mn.name = 'monitor/router_balance_counts_local') AND (length(ec.`values`) = 3)) AS counts
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('monitor/router_balance_local', 'monitor/router_balance_counts_local')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('monitor/router_balance_local', 'monitor/router_balance_counts_local')
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer,
            ec.microbatch
    )
SELECT
    exp,
    run,
    step,
    layer,
    microbatch,
    if((length(sums) = 7) AND (length(counts) = 3), sums[1] / nullIf(counts[2], 0), NULL) AS top_k_normalized,
    if((length(sums) = 7) AND (length(counts) = 3), sums[2] / nullIf(counts[2], 0), NULL) AS top_k_raw,
    if((length(sums) = 7) AND (length(counts) = 3), sums[3] / nullIf(counts[2], 0), NULL) AS top_k_biased,
    if((length(sums) = 7) AND (length(counts) = 3), sums[4] / nullIf(counts[1], 0), NULL) AS entropy,
    if((length(sums) = 7) AND (length(counts) = 3), sums[5] / nullIf(counts[3], 0), NULL) AS z_loss,
    if((length(sums) = 7) AND (length(counts) = 3), sums[6] / nullIf(counts[3], 0), NULL) AS seq_aux_loss,
    if((length(sums) = 7) AND (length(counts) = 3), sums[7] / nullIf(counts[3], 0), NULL) AS alpha,
    if(length(counts) = 3, counts[1], 0) AS all_tokens,
    if(length(counts) = 3, counts[2], 0) AS valid_tokens,
    if(length(counts) = 3, counts[3], 0) AS forward_calls
FROM totals;

CREATE OR REPLACE VIEW metric_sampler.router_balance_mbsum
AS SELECT * EXCEPT (refreshed_at, microbatch)
FROM metric_sampler.summary_router_balance_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32});

-- Partitioned by run like the raw tables, so an idle run's rows move to the cold tier and merges stay within a run.
CREATE TABLE IF NOT EXISTS metric_sampler.summary_router_balance_rank_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `rank` Int16,
    `microbatch` Int16,
    `top_k_normalized` Nullable(Float64),
    `top_k_raw` Nullable(Float64),
    `top_k_biased` Nullable(Float64),
    `entropy` Nullable(Float64),
    `z_loss` Nullable(Float64),
    `seq_aux_loss` Nullable(Float64),
    `alpha` Nullable(Float64),
    `all_tokens` UInt64,
    `valid_tokens` UInt64,
    `forward_calls` UInt64
)
ENGINE = ReplacingMergeTree(refreshed_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, rank, microbatch, layer, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)', max_bytes_to_merge_at_max_space_in_pool = 4294967296;

CREATE OR REPLACE VIEW metric_sampler.router_balance_by_rank_mb_range
AS WITH
    totals AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            sumForEachIf(arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[(2 * i) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * i) + 2]), 32))), range(7)), (mn.name = 'monitor/router_balance_local') AND (length(ec.`values`) = 14)) AS sums,
            sumForEachIf(arrayMap(x -> toUInt64(x), ec.`values`), (mn.name = 'monitor/router_balance_counts_local') AND (length(ec.`values`) = 3)) AS counts
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('monitor/router_balance_local', 'monitor/router_balance_counts_local')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('monitor/router_balance_local', 'monitor/router_balance_counts_local')
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer,
            ec.rank,
            ec.microbatch
    )
SELECT
    exp,
    run,
    step,
    layer,
    rank,
    microbatch,
    if((length(sums) = 7) AND (length(counts) = 3), sums[1] / nullIf(counts[2], 0), NULL) AS top_k_normalized,
    if((length(sums) = 7) AND (length(counts) = 3), sums[2] / nullIf(counts[2], 0), NULL) AS top_k_raw,
    if((length(sums) = 7) AND (length(counts) = 3), sums[3] / nullIf(counts[2], 0), NULL) AS top_k_biased,
    if((length(sums) = 7) AND (length(counts) = 3), sums[4] / nullIf(counts[1], 0), NULL) AS entropy,
    if((length(sums) = 7) AND (length(counts) = 3), sums[5] / nullIf(counts[3], 0), NULL) AS z_loss,
    if((length(sums) = 7) AND (length(counts) = 3), sums[6] / nullIf(counts[3], 0), NULL) AS seq_aux_loss,
    if((length(sums) = 7) AND (length(counts) = 3), sums[7] / nullIf(counts[3], 0), NULL) AS alpha,
    if(length(counts) = 3, counts[1], 0) AS all_tokens,
    if(length(counts) = 3, counts[2], 0) AS valid_tokens,
    if(length(counts) = 3, counts[3], 0) AS forward_calls
FROM totals;

CREATE OR REPLACE VIEW metric_sampler.router_balance_rankmbsum
AS SELECT * EXCEPT (refreshed_at, rank, microbatch)
FROM metric_sampler.summary_router_balance_rank_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32}) AND (microbatch = {microbatch_filter:Int32});
