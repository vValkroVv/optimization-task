CREATE TABLE IF NOT EXISTS metric_sampler.summary_expert_totals
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `expert_index` UInt32,
    `total_tokens` UInt64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, layer, step, expert_index)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_expert_totals_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `microbatch` Int16,
    `expert_index` UInt32,
    `total_tokens` UInt64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, microbatch, layer, step, expert_index)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.expert_totals_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_expert_totals
FINAL
WHERE (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}));

CREATE OR REPLACE VIEW metric_sampler.expert_totals_by_mb_range
AS SELECT
    exp,
    run,
    step,
    layer,
    microbatch,
    expert_index,
    total_tokens
FROM
(
    SELECT
        ec.exp AS exp,
        ec.run AS run,
        ec.step AS step,
        ec.layer AS layer,
        ec.microbatch AS microbatch,
        sumForEach(ec.`values`) AS token_counts
    FROM `metric_sampler.router_counts_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32}) AS ec
    INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
    INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
    WHERE (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('router/tokens_per_expert', 'router/local_tokens_before_bias_update')) AND (ec.metric_id IN (
        SELECT metric_id
        FROM metric_sampler.metrics
        WHERE name IN ('router/tokens_per_expert', 'router/local_tokens_before_bias_update')
    ))
    GROUP BY
        ec.exp,
        ec.run,
        ec.step,
        ec.layer,
        ec.microbatch
)
ARRAY JOIN
    token_counts AS total_tokens,
    arrayEnumerate(token_counts) AS expert_index;

CREATE OR REPLACE VIEW metric_sampler.expert_totals_fast
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_expert_totals
FINAL
WHERE ({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2) AND (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}))
UNION ALL
SELECT
    exp,
    run,
    step,
    layer,
    toUInt32(expert_index) AS expert_index,
    sum(token_count) AS total_tokens
FROM
(
    SELECT
        r.exp AS exp,
        r.run AS run,
        r.step AS step,
        r.layer AS layer,
        r.metric_id AS metric_id,
        r.`values` AS `values`,
        min(r.metric_id) OVER (PARTITION BY r.step, r.layer) AS preferred
    FROM metric_sampler.summary_router_rank AS r
    FINAL
    INNER JOIN
    (
        SELECT
            exp,
            run,
            step,
            attempt
        FROM metric_sampler.latest_attempt_step
        WHERE ({rank_filter:Int32} != -2) AND (exp = {exp_filter:String}) AND (run = {run_filter:String})
    ) AS las ON (r.exp = las.exp) AND (r.run = las.run) AND (r.step = las.step) AND (r.attempt = las.attempt)
    WHERE ({rank_filter:Int32} != -2) AND (r.rank = {rank_filter:Int32}) AND (({microbatch_filter:Int32} = -2) OR (r.microbatch = {microbatch_filter:Int32})) AND (({exp_filter:String} = '') OR (r.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (r.run = {run_filter:String}))
)
ARRAY JOIN
    `values` AS token_count,
    arrayEnumerate(`values`) AS expert_index
WHERE metric_id = preferred
GROUP BY
    exp,
    run,
    step,
    layer,
    expert_index
UNION ALL
SELECT *
FROM `metric_sampler.expert_totals_selected`(exp_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} != -2), {exp_filter:String}, '-'), run_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} != -2), {run_filter:String}, '-'), rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32})
WHERE ({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} != -2);

CREATE OR REPLACE VIEW metric_sampler.expert_totals_mb
AS SELECT *
FROM `metric_sampler.expert_totals_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.expert_totals_mbsum
AS SELECT * EXCEPT (refreshed_at, microbatch)
FROM metric_sampler.summary_expert_totals_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32});

-- Sums the vectors of a layer over microbatches before expanding them to experts, and keeps the sums of the
-- lowest metric id: a layer filter outside the view then reaches the summary read, and the expansion to
-- experts sees one vector per step instead of one per microbatch.
CREATE OR REPLACE VIEW metric_sampler.expert_totals_rank
AS SELECT
    exp,
    run,
    step,
    layer,
    toUInt32(expert_index) AS expert_index,
    total_tokens
FROM
(
    SELECT
        exp,
        run,
        step,
        layer,
        argMin(metric_tokens, metric_id) AS router_tokens
    FROM
    (
        SELECT
            r.exp AS exp,
            r.run AS run,
            r.step AS step,
            r.layer AS layer,
            r.metric_id AS metric_id,
            sumForEach(r.`values`) AS metric_tokens
        FROM metric_sampler.summary_router_rank AS r
        FINAL
        WHERE (r.rank = {rank_filter:Int32}) AND (({microbatch_filter:Int32} = -2) OR (r.microbatch = {microbatch_filter:Int32})) AND (r.exp = {exp_filter:String}) AND (r.run = {run_filter:String}) AND ((r.step, r.attempt) IN (
            SELECT
                step,
                attempt
            FROM metric_sampler.latest_attempt_step
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
        ))
        GROUP BY
            r.exp,
            r.run,
            r.step,
            r.layer,
            r.metric_id
    )
    GROUP BY
        exp,
        run,
        step,
        layer
)
ARRAY JOIN
    router_tokens AS total_tokens,
    arrayEnumerate(router_tokens) AS expert_index;

CREATE OR REPLACE VIEW metric_sampler.expert_totals_selected_live
AS SELECT *
FROM `metric_sampler.expert_totals_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.expert_totals_selected_range
AS SELECT
    exp,
    run,
    step,
    layer,
    expert_index,
    total_tokens
FROM
(
    SELECT
        ec.exp AS exp,
        ec.run AS run,
        ec.step AS step,
        ec.layer AS layer,
        sumForEach(ec.`values`) AS token_counts
    FROM `metric_sampler.router_counts_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32}) AS ec
    INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
    INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
    WHERE (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('router/tokens_per_expert', 'router/local_tokens_before_bias_update')) AND (ec.metric_id IN (
        SELECT metric_id
        FROM metric_sampler.metrics
        WHERE name IN ('router/tokens_per_expert', 'router/local_tokens_before_bias_update')
    ))
    GROUP BY
        ec.exp,
        ec.run,
        ec.step,
        ec.layer
)
ARRAY JOIN
    token_counts AS total_tokens,
    arrayEnumerate(token_counts) AS expert_index;
