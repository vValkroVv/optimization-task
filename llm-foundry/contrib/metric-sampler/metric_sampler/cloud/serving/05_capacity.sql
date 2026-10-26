CREATE TABLE IF NOT EXISTS metric_sampler.summary_capacity
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `router_assignments` Nullable(UInt64),
    `dispatched_no_pad` Nullable(UInt64),
    `dispatched` Nullable(UInt64),
    `dropped_share` Nullable(Float64),
    `padding_share` Nullable(Float64),
    `capacity_observed` UInt8
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_capacity_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `microbatch` Int16,
    `router_assignments` Nullable(UInt64),
    `dispatched_no_pad` Nullable(UInt64),
    `dispatched` Nullable(UInt64),
    `dropped_share` Nullable(Float64),
    `padding_share` Nullable(Float64),
    `capacity_observed` UInt8
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, microbatch, step, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_capacity_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `rank` Int16,
    `router_assignments` Nullable(UInt64),
    `dispatched_no_pad` Nullable(UInt64),
    `dispatched` Nullable(UInt64),
    `dropped_share` Nullable(Float64),
    `padding_share` Nullable(Float64),
    `capacity_observed` UInt8
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, rank, step, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_capacity_rank_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `rank` Int16,
    `microbatch` Int16,
    `router_assignments` Nullable(UInt64),
    `dispatched_no_pad` Nullable(UInt64),
    `dispatched` Nullable(UInt64),
    `dropped_share` Nullable(Float64),
    `padding_share` Nullable(Float64),
    `capacity_observed` UInt8
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, rank, microbatch, step, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.capacity_by_mb_range
AS WITH
    observations AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            ec.node AS node,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/assignments_before_capacity') + (countIf((mn.name = 'dispatcher/assignments_before_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'router/tokens_per_expert')) AS before_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_before_capacity') AS valid_before_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/assignments_after_capacity') + (countIf((mn.name = 'dispatcher/assignments_after_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'router/tokens_per_expert')) AS after_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_after_capacity') + (countIf((mn.name = 'dispatcher/valid_assignments_after_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_before_capacity')) AS valid_after_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert') AS legacy_input,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert_no_pad') + (countIf((mn.name = 'dispatcher/tokens_per_local_expert_no_pad') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert')) AS legacy_valid_input,
            max(mn.name = 'dispatcher/assignments_before_capacity') AND max(mn.name = 'dispatcher/valid_assignments_before_capacity') AND max(mn.name = 'dispatcher/assignments_after_capacity') AND max(mn.name = 'dispatcher/valid_assignments_after_capacity') AS capacity_observed,
            max(mn.name = 'dispatcher/tokens_per_local_expert') AND max(mn.name = 'dispatcher/tokens_per_local_expert_no_pad') AS legacy_observed
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (ec.layer <= 255) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity', 'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad', 'router/tokens_per_expert')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity', 'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad', 'router/tokens_per_expert')
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
    totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            microbatch,
            sum(before_capacity) AS before_capacity,
            sum(valid_before_capacity) AS valid_before_capacity,
            sum(after_capacity) AS after_capacity,
            sum(valid_after_capacity) AS valid_after_capacity,
            sum(legacy_input) AS legacy_input,
            sum(legacy_valid_input) AS legacy_valid_input,
            min(capacity_observed) AS capacity_observed,
            min(legacy_observed) AS legacy_observed
        FROM observations
        GROUP BY
            exp,
            run,
            step,
            layer,
            microbatch
    )
SELECT
    exp,
    run,
    step,
    layer,
    microbatch,
    multiIf(capacity_observed, before_capacity, NULL) AS router_assignments,
    multiIf(capacity_observed, valid_after_capacity, NULL) AS dispatched_no_pad,
    multiIf(capacity_observed, after_capacity, NULL) AS dispatched,
    multiIf(capacity_observed, 1. - (valid_after_capacity / nullIf(valid_before_capacity, 0)), NULL) AS dropped_share,
    multiIf(capacity_observed, 1. - (valid_before_capacity / nullIf(before_capacity, 0)), legacy_observed, 1. - (legacy_valid_input / nullIf(legacy_input, 0)), NULL) AS padding_share,
    capacity_observed
FROM totals;

CREATE OR REPLACE VIEW metric_sampler.capacity_by_rank_mb_range
AS WITH
    observations AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            ec.node AS node,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/assignments_before_capacity') + (countIf((mn.name = 'dispatcher/assignments_before_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'router/tokens_per_expert')) AS before_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_before_capacity') AS valid_before_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/assignments_after_capacity') + (countIf((mn.name = 'dispatcher/assignments_after_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'router/tokens_per_expert')) AS after_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_after_capacity') + (countIf((mn.name = 'dispatcher/valid_assignments_after_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_before_capacity')) AS valid_after_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert') AS legacy_input,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert_no_pad') + (countIf((mn.name = 'dispatcher/tokens_per_local_expert_no_pad') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert')) AS legacy_valid_input,
            max(mn.name = 'dispatcher/assignments_before_capacity') AND max(mn.name = 'dispatcher/valid_assignments_before_capacity') AND max(mn.name = 'dispatcher/assignments_after_capacity') AND max(mn.name = 'dispatcher/valid_assignments_after_capacity') AS capacity_observed,
            max(mn.name = 'dispatcher/tokens_per_local_expert') AND max(mn.name = 'dispatcher/tokens_per_local_expert_no_pad') AS legacy_observed
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (ec.layer <= 255) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity', 'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad', 'router/tokens_per_expert')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity', 'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad', 'router/tokens_per_expert')
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
    totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            rank,
            microbatch,
            sum(before_capacity) AS before_capacity,
            sum(valid_before_capacity) AS valid_before_capacity,
            sum(after_capacity) AS after_capacity,
            sum(valid_after_capacity) AS valid_after_capacity,
            sum(legacy_input) AS legacy_input,
            sum(legacy_valid_input) AS legacy_valid_input,
            min(capacity_observed) AS capacity_observed,
            min(legacy_observed) AS legacy_observed
        FROM observations
        GROUP BY
            exp,
            run,
            step,
            layer,
            rank,
            microbatch
    )
SELECT
    exp,
    run,
    step,
    layer,
    rank,
    microbatch,
    multiIf(capacity_observed, before_capacity, NULL) AS router_assignments,
    multiIf(capacity_observed, valid_after_capacity, NULL) AS dispatched_no_pad,
    multiIf(capacity_observed, after_capacity, NULL) AS dispatched,
    multiIf(capacity_observed, 1. - (valid_after_capacity / nullIf(valid_before_capacity, 0)), NULL) AS dropped_share,
    multiIf(capacity_observed, 1. - (valid_before_capacity / nullIf(before_capacity, 0)), legacy_observed, 1. - (legacy_valid_input / nullIf(legacy_input, 0)), NULL) AS padding_share,
    capacity_observed
FROM totals;

CREATE OR REPLACE VIEW metric_sampler.capacity_by_rank_range
AS WITH
    observations AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            ec.node AS node,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/assignments_before_capacity') + (countIf((mn.name = 'dispatcher/assignments_before_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'router/tokens_per_expert')) AS before_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_before_capacity') AS valid_before_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/assignments_after_capacity') + (countIf((mn.name = 'dispatcher/assignments_after_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'router/tokens_per_expert')) AS after_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_after_capacity') + (countIf((mn.name = 'dispatcher/valid_assignments_after_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_before_capacity')) AS valid_after_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert') AS legacy_input,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert_no_pad') + (countIf((mn.name = 'dispatcher/tokens_per_local_expert_no_pad') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert')) AS legacy_valid_input,
            max(mn.name = 'dispatcher/assignments_before_capacity') AND max(mn.name = 'dispatcher/valid_assignments_before_capacity') AND max(mn.name = 'dispatcher/assignments_after_capacity') AND max(mn.name = 'dispatcher/valid_assignments_after_capacity') AS capacity_observed,
            max(mn.name = 'dispatcher/tokens_per_local_expert') AND max(mn.name = 'dispatcher/tokens_per_local_expert_no_pad') AS legacy_observed
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (ec.layer <= 255) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity', 'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad', 'router/tokens_per_expert')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity', 'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad', 'router/tokens_per_expert')
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
    totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            rank,
            sum(before_capacity) AS before_capacity,
            sum(valid_before_capacity) AS valid_before_capacity,
            sum(after_capacity) AS after_capacity,
            sum(valid_after_capacity) AS valid_after_capacity,
            sum(legacy_input) AS legacy_input,
            sum(legacy_valid_input) AS legacy_valid_input,
            min(capacity_observed) AS capacity_observed,
            min(legacy_observed) AS legacy_observed
        FROM observations
        GROUP BY
            exp,
            run,
            step,
            layer,
            rank
    )
SELECT
    exp,
    run,
    step,
    layer,
    rank,
    multiIf(capacity_observed, before_capacity, NULL) AS router_assignments,
    multiIf(capacity_observed, valid_after_capacity, NULL) AS dispatched_no_pad,
    multiIf(capacity_observed, after_capacity, NULL) AS dispatched,
    multiIf(capacity_observed, 1. - (valid_after_capacity / nullIf(valid_before_capacity, 0)), NULL) AS dropped_share,
    multiIf(capacity_observed, 1. - (valid_before_capacity / nullIf(before_capacity, 0)), legacy_observed, 1. - (legacy_valid_input / nullIf(legacy_input, 0)), NULL) AS padding_share,
    capacity_observed
FROM totals;

CREATE OR REPLACE VIEW metric_sampler.capacity_selected_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_capacity
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.capacity_selected_live
AS SELECT *
FROM `metric_sampler.capacity_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.capacity_selected_mbsum
AS SELECT * EXCEPT (refreshed_at, microbatch)
FROM metric_sampler.summary_capacity_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.capacity_selected_range
AS WITH
    observations AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            ec.node AS node,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/assignments_before_capacity') + (countIf((mn.name = 'dispatcher/assignments_before_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'router/tokens_per_expert')) AS before_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_before_capacity') AS valid_before_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/assignments_after_capacity') + (countIf((mn.name = 'dispatcher/assignments_after_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'router/tokens_per_expert')) AS after_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_after_capacity') + (countIf((mn.name = 'dispatcher/valid_assignments_after_capacity') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/valid_assignments_before_capacity')) AS valid_after_capacity,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert') AS legacy_input,
            sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert_no_pad') + (countIf((mn.name = 'dispatcher/tokens_per_local_expert_no_pad') AND (ec.alias_metric_id != 0)) * sumIf(arraySum(ec.`values`), mn.name = 'dispatcher/tokens_per_local_expert')) AS legacy_valid_input,
            max(mn.name = 'dispatcher/assignments_before_capacity') AND max(mn.name = 'dispatcher/valid_assignments_before_capacity') AND max(mn.name = 'dispatcher/assignments_after_capacity') AND max(mn.name = 'dispatcher/valid_assignments_after_capacity') AS capacity_observed,
            max(mn.name = 'dispatcher/tokens_per_local_expert') AND max(mn.name = 'dispatcher/tokens_per_local_expert_no_pad') AS legacy_observed
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (ec.layer <= 255) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (mn.name IN ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity', 'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad', 'router/tokens_per_expert')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity', 'dispatcher/tokens_per_local_expert', 'dispatcher/tokens_per_local_expert_no_pad', 'router/tokens_per_expert')
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
    totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            sum(before_capacity) AS before_capacity,
            sum(valid_before_capacity) AS valid_before_capacity,
            sum(after_capacity) AS after_capacity,
            sum(valid_after_capacity) AS valid_after_capacity,
            sum(legacy_input) AS legacy_input,
            sum(legacy_valid_input) AS legacy_valid_input,
            min(capacity_observed) AS capacity_observed,
            min(legacy_observed) AS legacy_observed
        FROM observations
        GROUP BY
            exp,
            run,
            step,
            layer
    )
SELECT
    exp,
    run,
    step,
    layer,
    multiIf(capacity_observed, before_capacity, NULL) AS router_assignments,
    multiIf(capacity_observed, valid_after_capacity, NULL) AS dispatched_no_pad,
    multiIf(capacity_observed, after_capacity, NULL) AS dispatched,
    multiIf(capacity_observed, 1. - (valid_after_capacity / nullIf(valid_before_capacity, 0)), NULL) AS dropped_share,
    multiIf(capacity_observed, 1. - (valid_before_capacity / nullIf(before_capacity, 0)), legacy_observed, 1. - (legacy_valid_input / nullIf(legacy_input, 0)), NULL) AS padding_share,
    capacity_observed
FROM totals;

CREATE OR REPLACE VIEW metric_sampler.capacity_selected_rankmbsum
AS SELECT * EXCEPT (refreshed_at, rank, microbatch)
FROM metric_sampler.summary_capacity_rank_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32}) AND (microbatch = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.capacity_selected_ranksum
AS SELECT * EXCEPT (refreshed_at, rank)
FROM metric_sampler.summary_capacity_rank
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32});
