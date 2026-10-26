-- Partitioned by run like the raw tables, so an idle run's rows move to the cold tier and merges stay within a run.
CREATE TABLE IF NOT EXISTS metric_sampler.summary_router_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `attempt` LowCardinality(String),
    `node` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `rank` Int16,
    `microbatch` Int16,
    `metric_id` UInt16,
    `values` Array(UInt32) CODEC(T64, ZSTD(1))
)
ENGINE = ReplacingMergeTree(refreshed_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, rank, microbatch, layer, metric_id, step, attempt, node)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)', max_bytes_to_merge_at_max_space_in_pool = 4294967296;

CREATE OR REPLACE VIEW metric_sampler.router_counts_selected_range
AS WITH preferred AS
    (
        SELECT
            exp,
            run,
            attempt,
            step,
            layer,
            min(metric_id) AS selected_metric_id
        FROM metric_sampler.expert_counts
        WHERE (expert_counts.layer <= 255) AND (expert_counts.exp = {exp_filter:String}) AND (expert_counts.run = {run_filter:String}) AND (expert_counts.step >= {step_from:UInt32}) AND (expert_counts.step <= {step_to:UInt32}) AND (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String})) AND (metric_id IN (1, 2))
        GROUP BY
            exp,
            run,
            attempt,
            step,
            layer
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
    ec.`values` AS `values`
FROM metric_sampler.expert_counts AS ec
FINAL
INNER JOIN preferred AS source ON (ec.exp = source.exp) AND (ec.run = source.run) AND (ec.attempt = source.attempt) AND (ec.step = source.step) AND (ec.layer = source.layer) AND (ec.metric_id = source.selected_metric_id)
INNER JOIN metric_sampler.latest_attempt_step AS latest ON (ec.exp = latest.exp) AND (ec.run = latest.run) AND (ec.step = latest.step) AND (ec.attempt = latest.attempt)
WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (ec.metric_id IN (1, 2));

CREATE OR REPLACE VIEW metric_sampler.router_counts_selected_live
AS SELECT *
FROM `metric_sampler.router_counts_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String});
