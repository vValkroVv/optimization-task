CREATE TABLE IF NOT EXISTS metric_sampler.summary_expert_loss
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `expert` UInt32,
    `tokens` UInt64,
    `mean` Float64,
    `std` Float64,
    `hardness` Float64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, layer, expert)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.expert_loss_fast
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `expert` UInt32,
    `tokens` UInt64,
    `mean` Float64,
    `std` Float64,
    `hardness` Float64
)
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_expert_loss
FINAL;

CREATE OR REPLACE VIEW metric_sampler.expert_loss_live
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `expert` UInt32,
    `tokens` UInt64,
    `mean` Float64,
    `std` Float64,
    `hardness` Float64
)
AS SELECT *
FROM metric_sampler.expert_loss_per_step;

CREATE OR REPLACE VIEW metric_sampler.expert_loss_per_step_range
AS WITH
    expert_sum AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            rv.layer AS layer,
            sumForEach(rv.`values`) AS `values`
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (mn.name = 'expert_loss/sum') AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'expert_loss/sum'
        ))
        GROUP BY
            rv.exp,
            rv.run,
            rv.step,
            rv.layer
    ),
    expert_sumsq AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            rv.layer AS layer,
            sumForEach(rv.`values`) AS `values`
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (mn.name = 'expert_loss/sumsq') AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'expert_loss/sumsq'
        ))
        GROUP BY
            rv.exp,
            rv.run,
            rv.step,
            rv.layer
    ),
    expert_count AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            sumForEach(ec.`values`) AS `values`
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (mn.name = 'expert_loss/count') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'expert_loss/count'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer
    ),
    per_expert AS
    (
        SELECT
            expert_sum.exp AS exp,
            expert_sum.run AS run,
            expert_sum.step AS step,
            expert_sum.layer AS layer,
            expert_index AS expert,
            expert_count.`values`[expert_index] AS tokens,
            (expert_sum.`values`[expert_index]) / (expert_count.`values`[expert_index]) AS mean,
            sqrt(greatest(((expert_sumsq.`values`[expert_index]) / (expert_count.`values`[expert_index])) - pow((expert_sum.`values`[expert_index]) / (expert_count.`values`[expert_index]), 2), 0)) AS std
        FROM expert_sum
        ARRAY JOIN arrayEnumerate(expert_sum.`values`) AS expert_index
        INNER JOIN expert_sumsq ON (expert_sumsq.exp = expert_sum.exp) AND (expert_sumsq.run = expert_sum.run) AND (expert_sumsq.step = expert_sum.step) AND (expert_sumsq.layer = expert_sum.layer)
        INNER JOIN expert_count ON (expert_count.exp = expert_sum.exp) AND (expert_count.run = expert_sum.run) AND (expert_count.step = expert_sum.step) AND (expert_count.layer = expert_sum.layer)
        WHERE (expert_count.`values`[expert_index]) > 0
    ),
    layer_mean AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            sum(tokens * mean) / sum(tokens) AS layer_mean
        FROM per_expert
        GROUP BY
            exp,
            run,
            step,
            layer
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
INNER JOIN layer_mean ON (layer_mean.exp = per_expert.exp) AND (layer_mean.run = per_expert.run) AND (layer_mean.step = per_expert.step) AND (layer_mean.layer = per_expert.layer);

CREATE TABLE IF NOT EXISTS metric_sampler.summary_expert_stream
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `expert` Int64,
    `stream_id` Int64,
    `tokens` UInt64,
    `mean` Float64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, layer, expert, stream_id)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.expert_stream_range
AS WITH
    stream_totals AS
    (
        SELECT
            exp,
            run,
            count() AS num_streams
        FROM metric_sampler.stream_names
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
        GROUP BY
            exp,
            run
    ),
    stream_count AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            sumForEach(ec.`values`) AS `values`
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (ec.layer <= 255) AND (mn.name = 'expert_loss/stream_count') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'expert_loss/stream_count'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer
    ),
    stream_sum AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            rv.layer AS layer,
            sumForEach(rv.`values`) AS `values`
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND (rv.layer <= 255) AND (mn.name = 'expert_loss/stream_sum') AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'expert_loss/stream_sum'
        ))
        GROUP BY
            rv.exp,
            rv.run,
            rv.step,
            rv.layer
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
INNER JOIN stream_totals ON (stream_totals.exp = stream_count.exp) AND (stream_totals.run = stream_count.run)
INNER JOIN stream_sum ON (stream_sum.exp = stream_count.exp) AND (stream_sum.run = stream_count.run) AND (stream_sum.step = stream_count.step) AND (stream_sum.layer = stream_count.layer)
ARRAY JOIN
    arrayEnumerate(stream_count.`values`) AS idx,
    stream_count.`values` AS token_count,
    arrayResize(stream_sum.`values`, length(stream_count.`values`)) AS loss_sum
WHERE token_count > 0;

CREATE OR REPLACE VIEW metric_sampler.expert_stream_latest_fast
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_expert_stream
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (layer = {layer_filter:Int32}) AND (step = (
    SELECT step
    FROM metric_sampler.summary_expert_stream
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (layer = {layer_filter:Int32}) AND (step <= (SELECT step_limit FROM metric_sampler.latest_step_limit(exp_filter = {exp_filter:String}, run_filter = {run_filter:String})))
    ORDER BY step DESC
    LIMIT 1
));

CREATE OR REPLACE VIEW metric_sampler.expert_stream_latest_live
AS SELECT *
FROM metric_sampler.expert_stream_map
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (layer = {layer_filter:Int32}) AND (step = (
    SELECT max(ec.step)
    FROM metric_sampler.expert_counts AS ec
    INNER JOIN metric_sampler.latest_attempt_step AS latest ON (ec.exp = latest.exp) AND (ec.run = latest.run) AND (ec.step = latest.step) AND (ec.attempt = latest.attempt)
    INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
    WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.layer = {layer_filter:Int32}) AND (ec.step <= (SELECT step_limit FROM metric_sampler.latest_step_limit(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}))) AND (ec.step >= ((SELECT step_limit FROM metric_sampler.latest_step_limit(exp_filter = {exp_filter:String}, run_filter = {run_filter:String})) - 2000)) AND (mn.name = 'expert_loss/stream_count') AND (ec.metric_id IN (
        SELECT metric_id
        FROM metric_sampler.metrics
        WHERE name = 'expert_loss/stream_count'
    ))
));

CREATE OR REPLACE VIEW metric_sampler.expert_loss_latest_fast
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_expert_loss
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (layer = {layer_filter:Int32}) AND (step = (
    SELECT step
    FROM metric_sampler.summary_expert_loss
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (layer = {layer_filter:Int32}) AND (step <= (SELECT step_limit FROM metric_sampler.latest_step_limit(exp_filter = {exp_filter:String}, run_filter = {run_filter:String})))
    ORDER BY step DESC
    LIMIT 1
));

CREATE OR REPLACE VIEW metric_sampler.expert_loss_latest_live
AS SELECT *
FROM metric_sampler.expert_loss_live
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (layer = {layer_filter:Int32}) AND (step = (
    SELECT max(ec.step)
    FROM metric_sampler.expert_counts AS ec
    INNER JOIN metric_sampler.latest_attempt_step AS latest ON (ec.exp = latest.exp) AND (ec.run = latest.run) AND (ec.step = latest.step) AND (ec.attempt = latest.attempt)
    INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
    WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.layer = {layer_filter:Int32}) AND (ec.step <= (SELECT step_limit FROM metric_sampler.latest_step_limit(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}))) AND (ec.step >= ((SELECT step_limit FROM metric_sampler.latest_step_limit(exp_filter = {exp_filter:String}, run_filter = {run_filter:String})) - 2000)) AND (mn.name = 'expert_loss/count') AND (ec.metric_id IN (
        SELECT metric_id
        FROM metric_sampler.metrics
        WHERE name = 'expert_loss/count'
    ))
));
