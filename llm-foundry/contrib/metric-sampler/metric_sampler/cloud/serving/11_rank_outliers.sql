CREATE TABLE IF NOT EXISTS metric_sampler.summary_grad_per_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `node` LowCardinality(String),
    `rank` Int16,
    `dense_norm` Float64,
    `expert_norm` Float64,
    `total_norm` Float64,
    `robust_z` Nullable(Float64),
    `clip_total_norm` Float32
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, rank)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_timing_per_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `node` LowCardinality(String),
    `rank` Int16,
    `metric` LowCardinality(String),
    `value` Float64,
    `robust_z` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, rank, metric)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

-- The robust z of a card scales by the median over the whole run, so these views (and the timing ones) take the
-- run as parameters: a WHERE outside a view does not reach below its window function, and the summary would
-- be read for every run.
CREATE OR REPLACE VIEW metric_sampler.grad_per_rank_fast
AS WITH deviations AS
    (
        SELECT
            exp,
            run,
            step,
            node,
            rank,
            dense_norm,
            expert_norm,
            total_norm,
            clip_total_norm,
            log(dense_norm / medianExact(dense_norm) OVER (PARTITION BY step)) AS log_ratio
        FROM metric_sampler.summary_grad_per_rank
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
    )
SELECT
    exp,
    run,
    step,
    node,
    rank,
    dense_norm,
    expert_norm,
    total_norm,
    log_ratio / nullIf((
        SELECT 1.4826 * medianExact(abs(log_ratio))
        FROM deviations
    ), 0) AS robust_z,
    clip_total_norm
FROM deviations;

CREATE OR REPLACE VIEW metric_sampler.grad_per_rank_live
AS SELECT *
FROM metric_sampler.grad_per_rank
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

-- The summary loop's source for the run's per-rank outliers. The robust z scales by the median over the whole run,
-- so the raw scalars are read once and the run's rows are packed into an array next to that median: a CTE referenced
-- twice reads the scalars twice, and a window median over a whole run is quadratic.
-- The run's scale takes the norms of the steps before step_from from summary_grad_per_rank, where they are final and
-- equal to the ones computed here, so a chunk of a long run reads the raw scalars of its own and later steps only.
CREATE OR REPLACE VIEW metric_sampler.grad_per_rank_run
AS WITH
    names AS
    (
        SELECT
            attempt,
            metric_id,
            name
        FROM metric_sampler.metric_names
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (name IN ('grad/pre_reduce_sq_dense', 'grad/pre_reduce_sq_expert', 'grad/clip_total_norm'))
    ),
    latest AS
    (
        SELECT
            step,
            attempt
        FROM metric_sampler.latest_attempt_step
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
    ),
    grad_metrics AS
    (
        SELECT
            s.exp AS exp,
            s.run AS run,
            s.step AS step,
            s.node AS node,
            s.rank AS rank,
            sumIf(s.`values`[1], names.name = 'grad/pre_reduce_sq_dense') AS dense_sq,
            countIf(names.name = 'grad/pre_reduce_sq_dense') AS has_dense,
            sumIf(s.`values`[1], names.name = 'grad/pre_reduce_sq_expert') AS expert_sq,
            countIf(names.name = 'grad/pre_reduce_sq_expert') AS has_expert
        FROM metric_sampler.scalars AS s
        FINAL
        INNER JOIN names ON (names.attempt = s.attempt) AND (names.metric_id = s.metric_id)
        WHERE (s.layer <= 255) AND (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND ((s.step, s.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (s.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('grad/pre_reduce_sq_dense', 'grad/pre_reduce_sq_expert')
        ))
        GROUP BY
            s.exp,
            s.run,
            s.step,
            s.node,
            s.rank
    ),
    norms AS
    (
        SELECT
            exp,
            run,
            step,
            node,
            rank,
            dense_norm,
            expert_norm,
            total_norm,
            log(dense_norm / medianExact(dense_norm) OVER (PARTITION BY exp, run, step)) AS log_ratio
        FROM
        (
            SELECT
                exp,
                run,
                step,
                node,
                rank,
                sqrt(dense_sq) AS dense_norm,
                sqrt(expert_sq) AS expert_norm,
                sqrt(dense_sq + expert_sq) AS total_norm
            FROM grad_metrics
            WHERE (has_dense > 0) AND (has_expert > 0)
            UNION ALL
            SELECT
                exp,
                run,
                step,
                node,
                rank,
                dense_norm,
                expert_norm,
                total_norm
            FROM metric_sampler.summary_grad_per_rank
            FINAL
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step < {step_from:UInt32})
        )
    ),
    run_norms AS
    (
        SELECT
            exp,
            run,
            groupArrayIf((step, node, rank, dense_norm, expert_norm, total_norm, log_ratio), step >= {step_from:UInt32}) AS observations,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM norms
        GROUP BY
            exp,
            run
    ),
    clip_norm AS
    (
        SELECT
            s.exp AS exp,
            s.run AS run,
            s.step AS step,
            any(s.`values`[1]) AS clip_total_norm
        FROM metric_sampler.scalars AS s
        FINAL
        INNER JOIN names ON (names.attempt = s.attempt) AND (names.metric_id = s.metric_id)
        WHERE (names.name = 'grad/clip_total_norm') AND (s.layer <= 255) AND (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND ((s.step, s.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (s.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'grad/clip_total_norm'
        ))
        GROUP BY
            s.exp,
            s.run,
            s.step
    )
SELECT
    run_norms.exp AS exp,
    run_norms.run AS run,
    observation.1 AS step,
    observation.2 AS node,
    observation.3 AS rank,
    observation.4 AS dense_norm,
    observation.5 AS expert_norm,
    observation.6 AS total_norm,
    (observation.7) / nullIf(run_norms.scale, 0) AS robust_z,
    clip_norm.clip_total_norm AS clip_total_norm
FROM run_norms
ARRAY JOIN run_norms.observations AS observation
LEFT JOIN clip_norm ON (clip_norm.exp = run_norms.exp) AND (clip_norm.run = run_norms.run) AND (clip_norm.step = (observation.1));

CREATE OR REPLACE VIEW metric_sampler.timing_per_rank_fast
AS WITH
    deviations AS
    (
        SELECT
            exp,
            run,
            step,
            node,
            rank,
            metric,
            value,
            log(value / medianExact(value) OVER (PARTITION BY step, metric)) AS log_ratio
        FROM metric_sampler.summary_timing_per_rank
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
    ),
    run_scale AS
    (
        SELECT
            metric,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM deviations
        GROUP BY metric
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
INNER JOIN run_scale ON run_scale.metric = deviations.metric;

CREATE OR REPLACE VIEW metric_sampler.timing_per_rank_live
AS SELECT *
FROM metric_sampler.timing_per_rank
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

-- The summary loop's source for the run's per-rank timing outliers: like grad_per_rank_run, it reads the raw scalars
-- once and takes the run's scale of the steps before step_from from summary_timing_per_rank.
CREATE OR REPLACE VIEW metric_sampler.timing_per_rank_run
AS WITH
    names AS
    (
        SELECT
            attempt,
            metric_id,
            name
        FROM metric_sampler.metric_names
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (name IN ('time/step_gpu_s', 'time/forward_gpu_s', 'time/backward_gpu_s', 'time/optimizer_gpu_s'))
    ),
    latest AS
    (
        SELECT
            step,
            attempt
        FROM metric_sampler.latest_attempt_step
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
    ),
    ranks AS
    (
        SELECT
            s.exp AS exp,
            s.run AS run,
            s.step AS step,
            s.node AS node,
            s.rank AS rank,
            names.name AS metric,
            sum(s.`values`[1]) AS value
        FROM metric_sampler.scalars AS s
        FINAL
        INNER JOIN names ON (names.attempt = s.attempt) AND (names.metric_id = s.metric_id)
        WHERE (s.layer <= 255) AND (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND ((s.step, s.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (s.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('time/step_gpu_s', 'time/forward_gpu_s', 'time/backward_gpu_s', 'time/optimizer_gpu_s')
        ))
        GROUP BY
            s.exp,
            s.run,
            s.step,
            s.node,
            s.rank,
            names.name
    ),
    deviations AS
    (
        SELECT
            exp,
            run,
            step,
            node,
            rank,
            metric,
            value,
            log(value / medianExact(value) OVER (PARTITION BY exp, run, step, metric)) AS log_ratio
        FROM
        (
            SELECT
                exp,
                run,
                step,
                node,
                rank,
                metric,
                value
            FROM ranks
            UNION ALL
            SELECT
                exp,
                run,
                step,
                node,
                rank,
                metric,
                value
            FROM metric_sampler.summary_timing_per_rank
            FINAL
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step < {step_from:UInt32})
        )
    ),
    run_values AS
    (
        SELECT
            exp,
            run,
            metric,
            groupArrayIf((step, node, rank, value, log_ratio), step >= {step_from:UInt32}) AS observations,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM deviations
        GROUP BY
            exp,
            run,
            metric
    )
SELECT
    exp,
    run,
    observation.1 AS step,
    observation.2 AS node,
    observation.3 AS rank,
    metric,
    observation.4 AS value,
    (observation.5) / nullIf(scale, 0) AS robust_z
FROM run_values
ARRAY JOIN observations AS observation;

CREATE TABLE IF NOT EXISTS metric_sampler.summary_top_cards
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `microbatch` Int16,
    `kind` LowCardinality(String),
    `node` LowCardinality(String),
    `rank` Int16,
    `max_abs_z` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, microbatch, kind, node, rank)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.grad_per_rank_cards
AS SELECT
    exp,
    run,
    node,
    rank,
    max_abs_z AS robust_z
FROM metric_sampler.summary_top_cards
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (kind = 'grad') AND (microbatch = -2);

CREATE OR REPLACE VIEW metric_sampler.timing_per_rank_cards
AS SELECT
    exp,
    run,
    node,
    rank,
    max_abs_z AS robust_z
FROM metric_sampler.summary_top_cards
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (kind = 'timing') AND (microbatch = -2);

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_cards
AS SELECT
    exp,
    run,
    node,
    rank,
    max_abs_z AS robust_z
FROM metric_sampler.summary_top_cards
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (kind = 'activation') AND (microbatch = {microbatch_filter:Int32}) AND (({rank_filter:Int32} = -2) OR (rank = {rank_filter:Int32}));

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_peak
AS WITH
    run_scale AS
    (
        SELECT
            layer,
            point,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM metric_sampler.summary_activation_amax
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
        GROUP BY
            layer,
            point
    ),
    peaks AS
    (
        SELECT
            node,
            rank,
            layer,
            point,
            max(abs(log_ratio)) AS peak
        FROM metric_sampler.summary_activation_amax
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
        GROUP BY
            node,
            rank,
            layer,
            point
    )
SELECT
    toLowCardinality({exp_filter:String}) AS exp,
    toLowCardinality({run_filter:String}) AS run,
    peaks.node AS node,
    peaks.rank AS rank,
    max(peaks.peak / nullIf(run_scale.scale, 0)) AS max_abs_z
FROM peaks
INNER JOIN run_scale ON (run_scale.layer = peaks.layer) AND (run_scale.point = peaks.point)
GROUP BY
    peaks.node,
    peaks.rank;

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_mbpeak
AS WITH
    run_scale AS
    (
        SELECT
            layer,
            point,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM metric_sampler.summary_activation_amax_mb
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32})
        GROUP BY
            layer,
            point
    ),
    peaks AS
    (
        SELECT
            node,
            rank,
            layer,
            point,
            max(abs(log_ratio)) AS peak
        FROM metric_sampler.summary_activation_amax_mb
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32})
        GROUP BY
            node,
            rank,
            layer,
            point
    )
SELECT
    toLowCardinality({exp_filter:String}) AS exp,
    toLowCardinality({run_filter:String}) AS run,
    peaks.node AS node,
    peaks.rank AS rank,
    max(peaks.peak / nullIf(run_scale.scale, 0)) AS max_abs_z
FROM peaks
INNER JOIN run_scale ON (run_scale.layer = peaks.layer) AND (run_scale.point = peaks.point)
GROUP BY
    peaks.node,
    peaks.rank;
