CREATE TABLE IF NOT EXISTS metric_sampler.summary_layer_load
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `router_tokens` Array(UInt64),
    `dispatcher_tokens` Array(UInt64),
    `max_load` Float64,
    `min_load` Float64,
    `dead_experts` UInt32
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_layer_load_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `microbatch` Int16,
    `router_tokens` Array(UInt64),
    `dispatcher_tokens` Array(UInt64),
    `max_load` Float64,
    `min_load` Float64,
    `dead_experts` UInt32
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, microbatch, step, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_layer_load_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `rank` Int16,
    `layer` Int16,
    `max_load` Float64,
    `dead_experts` UInt32
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, rank, step, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.layer_load_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_layer_load
FINAL
WHERE (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}));

CREATE OR REPLACE VIEW metric_sampler.layer_load_by_mb_range
AS WITH
    router_totals AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            ec.microbatch AS microbatch,
            sumForEach(ec.`values`) AS router_tokens
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
    ),
    dispatcher_rows AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            ec.microbatch AS microbatch,
            countIf(mn.name = 'dispatcher/assignments_before_capacity') AS assignment_rows,
            if(maxIf(ec.alias_metric_id, mn.name = 'dispatcher/assignments_before_capacity') != 0, anyIf(ec.`values`, mn.name = 'router/tokens_per_expert'), anyIf(ec.`values`, mn.name = 'dispatcher/assignments_before_capacity')) AS assignments,
            anyIf(ec.`values`, mn.name = 'dispatcher/tokens_per_local_expert') AS legacy_tokens
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('dispatcher/assignments_before_capacity', 'dispatcher/tokens_per_local_expert', 'router/tokens_per_expert')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('dispatcher/assignments_before_capacity', 'dispatcher/tokens_per_local_expert', 'router/tokens_per_expert')
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer,
            ec.node,
            ec.rank,
            ec.microbatch
    ),
    dispatcher_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            microbatch,
            if(sum(assignment_rows) > 0, sumForEachIf(assignments, assignment_rows > 0), sumForEachIf(legacy_tokens, length(legacy_tokens) > 0)) AS dispatcher_tokens
        FROM dispatcher_rows
        GROUP BY
            exp,
            run,
            step,
            layer,
            microbatch
    )
SELECT
    router_totals.exp AS exp,
    router_totals.run AS run,
    router_totals.step AS step,
    router_totals.layer AS layer,
    router_totals.microbatch AS microbatch,
    router_totals.router_tokens AS router_tokens,
    dispatcher_totals.dispatcher_tokens AS dispatcher_tokens,
    arrayMax(router_totals.router_tokens) / (arraySum(router_totals.router_tokens) / length(router_totals.router_tokens)) AS max_load,
    arrayMin(router_totals.router_tokens) / (arraySum(router_totals.router_tokens) / length(router_totals.router_tokens)) AS min_load,
    arrayCount(x -> (x = 0), router_totals.router_tokens) AS dead_experts
FROM router_totals
LEFT JOIN dispatcher_totals ON (router_totals.exp = dispatcher_totals.exp) AND (router_totals.run = dispatcher_totals.run) AND (router_totals.step = dispatcher_totals.step) AND (router_totals.layer = dispatcher_totals.layer) AND (router_totals.microbatch = dispatcher_totals.microbatch);

CREATE OR REPLACE VIEW metric_sampler.layer_load_fast
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_layer_load
FINAL
WHERE ({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2) AND (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}))
UNION ALL
SELECT
    exp,
    run,
    step,
    layer,
    router_tokens,
    CAST([], 'Array(UInt64)') AS dispatcher_tokens,
    arrayMax(router_tokens) / (arraySum(router_tokens) / length(router_tokens)) AS max_load,
    arrayMin(router_tokens) / (arraySum(router_tokens) / length(router_tokens)) AS min_load,
    toUInt32(arrayCount(x -> (x = 0), router_tokens)) AS dead_experts
FROM
(
    SELECT
        exp,
        run,
        step,
        layer,
        sumForEach(`values`) AS router_tokens
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
    WHERE metric_id = preferred
    GROUP BY
        exp,
        run,
        step,
        layer
)
UNION ALL
SELECT *
FROM `metric_sampler.layer_load_selected`(exp_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} != -2), {exp_filter:String}, '-'), run_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} != -2), {run_filter:String}, '-'), rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32})
WHERE ({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} != -2);

CREATE OR REPLACE VIEW metric_sampler.layer_load_mb
AS SELECT *
FROM `metric_sampler.layer_load_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.layer_load_mbsum
AS SELECT * EXCEPT (refreshed_at, microbatch)
FROM metric_sampler.summary_layer_load_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.layer_load_rank
AS SELECT
    exp,
    run,
    step,
    layer,
    router_tokens,
    CAST([], 'Array(UInt64)') AS dispatcher_tokens,
    arrayMax(router_tokens) / (arraySum(router_tokens) / length(router_tokens)) AS max_load,
    arrayMin(router_tokens) / (arraySum(router_tokens) / length(router_tokens)) AS min_load,
    toUInt32(arrayCount(x -> (x = 0), router_tokens)) AS dead_experts
FROM
(
    SELECT
        exp,
        run,
        step,
        layer,
        sumForEach(`values`) AS router_tokens
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
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
        ) AS las ON (r.exp = las.exp) AND (r.run = las.run) AND (r.step = las.step) AND (r.attempt = las.attempt)
        WHERE (r.rank = {rank_filter:Int32}) AND (({microbatch_filter:Int32} = -2) OR (r.microbatch = {microbatch_filter:Int32})) AND (({exp_filter:String} = '') OR (r.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (r.run = {run_filter:String}))
    )
    WHERE metric_id = preferred
    GROUP BY
        exp,
        run,
        step,
        layer
);

-- The loads of every rank over a range of steps, as layer_load_rank computes them for one rank. It reads the raw
-- counts that summary_router_rank copies: that table is sorted by rank first, so a range of steps reads all of it.
CREATE OR REPLACE VIEW metric_sampler.layer_load_rank_range
AS SELECT
    exp,
    run,
    step,
    rank,
    layer,
    arrayMax(router_tokens) / (arraySum(router_tokens) / length(router_tokens)) AS max_load,
    toUInt32(arrayCount(x -> (x = 0), router_tokens)) AS dead_experts
FROM
(
    SELECT
        exp,
        run,
        step,
        rank,
        layer,
        sumForEach(`values`) AS router_tokens
    FROM
    (
        SELECT
            r.exp AS exp,
            r.run AS run,
            r.step AS step,
            r.rank AS rank,
            r.layer AS layer,
            r.metric_id AS metric_id,
            r.`values` AS `values`,
            min(r.metric_id) OVER (PARTITION BY r.rank, r.step, r.layer) AS preferred
        FROM metric_sampler.expert_counts AS r
        FINAL
        INNER JOIN
        (
            SELECT
                exp,
                run,
                step,
                attempt
            FROM metric_sampler.latest_attempt_step
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32})
        ) AS las ON (r.exp = las.exp) AND (r.run = las.run) AND (r.step = las.step) AND (r.attempt = las.attempt)
        WHERE (r.layer <= 255) AND (r.exp = {exp_filter:String}) AND (r.run = {run_filter:String}) AND (r.step >= {step_from:UInt32}) AND (r.step <= {step_to:UInt32}) AND (r.metric_id IN (1, 2))
    )
    WHERE metric_id = preferred
    GROUP BY
        exp,
        run,
        step,
        rank,
        layer
);

CREATE OR REPLACE VIEW metric_sampler.layer_load_ranksum
AS SELECT * EXCEPT (refreshed_at, rank)
FROM metric_sampler.summary_layer_load_rank
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.layer_load_selected_live
AS SELECT *
FROM `metric_sampler.layer_load_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.layer_load_selected_range
AS WITH
    router_totals AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            sumForEach(ec.`values`) AS router_tokens
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
    ),
    dispatcher_rows AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            countIf(mn.name = 'dispatcher/assignments_before_capacity') AS assignment_rows,
            if(maxIf(ec.alias_metric_id, mn.name = 'dispatcher/assignments_before_capacity') != 0, anyIf(ec.`values`, mn.name = 'router/tokens_per_expert'), anyIf(ec.`values`, mn.name = 'dispatcher/assignments_before_capacity')) AS assignments,
            anyIf(ec.`values`, mn.name = 'dispatcher/tokens_per_local_expert') AS legacy_tokens
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('dispatcher/assignments_before_capacity', 'dispatcher/tokens_per_local_expert', 'router/tokens_per_expert')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('dispatcher/assignments_before_capacity', 'dispatcher/tokens_per_local_expert', 'router/tokens_per_expert')
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer,
            ec.node,
            ec.rank,
            ec.microbatch
    ),
    dispatcher_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            if(sum(assignment_rows) > 0, sumForEachIf(assignments, assignment_rows > 0), sumForEachIf(legacy_tokens, length(legacy_tokens) > 0)) AS dispatcher_tokens
        FROM dispatcher_rows
        GROUP BY
            exp,
            run,
            step,
            layer
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
    arrayCount(x -> (x = 0), router_totals.router_tokens) AS dead_experts
FROM router_totals
LEFT JOIN dispatcher_totals ON (router_totals.exp = dispatcher_totals.exp) AND (router_totals.run = dispatcher_totals.run) AND (router_totals.step = dispatcher_totals.step) AND (router_totals.layer = dispatcher_totals.layer);
