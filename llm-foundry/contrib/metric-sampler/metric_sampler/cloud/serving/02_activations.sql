CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `elements` Float64,
    `mean` Nullable(Float64),
    `rms` Nullable(Float64),
    `std` Nullable(Float64),
    `nonfinite` Float64,
    `amax_max` Float32,
    `amax_median` Float32,
    `bf16_headroom_log2` Float64,
    `fp8_e4m3_headroom_log2` Float64,
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64),
    `exact_nonfinite` Nullable(UInt64),
    `skewness` Nullable(Float64),
    `kurtosis` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_amax
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `node` LowCardinality(String),
    `rank` Int16,
    `layer` Int16,
    `point` LowCardinality(String),
    `amax` Float32,
    `log_ratio` Float64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, layer, point, step, node, rank)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

-- Partitioned by run like the raw tables, so an idle run's rows move to the cold tier and merges stay within a run.
CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_amax_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `node` LowCardinality(String),
    `rank` Int16,
    `microbatch` Int16,
    `layer` Int16,
    `point` LowCardinality(String),
    `amax` Float32,
    `log_ratio` Float64
)
ENGINE = ReplacingMergeTree(refreshed_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, microbatch, layer, point, step, node, rank)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)', max_bytes_to_merge_at_max_space_in_pool = 4294967296;

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_hist_latest
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `histogram` String,
    `bins_start` Float64,
    `bins_width` Float64,
    `bins_count` UInt32,
    `exact_hist` Array(UInt64),
    `legacy_hist` Array(Float64),
    `counts_exact` UInt8
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, point, histogram, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_hist_latest_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `microbatch` Int16,
    `layer` Int16,
    `point` LowCardinality(String),
    `histogram` String,
    `bins_start` Float64,
    `bins_width` Float64,
    `bins_count` UInt32,
    `exact_hist` Array(UInt64),
    `legacy_hist` Array(Float64),
    `counts_exact` UInt8
)
ENGINE = ReplacingMergeTree(step)
ORDER BY (exp, run, microbatch, point, histogram, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_hist_latest_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `rank` Int16,
    `layer` Int16,
    `point` LowCardinality(String),
    `histogram` String,
    `bins_start` Float64,
    `bins_width` Float64,
    `bins_count` UInt32,
    `exact_hist` Array(UInt64),
    `legacy_hist` Array(Float64),
    `counts_exact` UInt8
)
ENGINE = ReplacingMergeTree(step)
ORDER BY (exp, run, rank, point, histogram, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_hist_latest_rank_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `rank` Int16,
    `microbatch` Int16,
    `layer` Int16,
    `point` LowCardinality(String),
    `histogram` String,
    `bins_start` Float64,
    `bins_width` Float64,
    `bins_count` UInt32,
    `exact_hist` Array(UInt64),
    `legacy_hist` Array(Float64),
    `counts_exact` UInt8
)
ENGINE = ReplacingMergeTree(step)
ORDER BY (exp, run, rank, microbatch, point, histogram, layer)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `microbatch` Int16,
    `elements` Float64,
    `mean` Nullable(Float64),
    `rms` Nullable(Float64),
    `std` Nullable(Float64),
    `nonfinite` Float64,
    `amax_max` Float32,
    `amax_median` Float32,
    `bf16_headroom_log2` Float64,
    `fp8_e4m3_headroom_log2` Float64,
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64),
    `exact_nonfinite` Nullable(UInt64),
    `skewness` Nullable(Float64),
    `kurtosis` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, microbatch, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_per_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `node` LowCardinality(String),
    `rank` Int16,
    `layer` Int16,
    `point` LowCardinality(String),
    `amax` Float32,
    `robust_z` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, layer, point, rank)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `elements` Float64,
    `mean` Nullable(Float64),
    `rms` Nullable(Float64),
    `std` Nullable(Float64),
    `nonfinite` Float64,
    `amax_max` Float32,
    `amax_median` Float32,
    `bf16_headroom_log2` Float64,
    `fp8_e4m3_headroom_log2` Float64,
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64),
    `exact_nonfinite` Nullable(UInt64),
    `skewness` Nullable(Float64),
    `kurtosis` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, rank, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)', max_bytes_to_merge_at_max_space_in_pool = 4294967296;

CREATE OR REPLACE VIEW metric_sampler.activation_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_activation
FINAL
WHERE (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}));

CREATE OR REPLACE VIEW metric_sampler.activation_amax_mb_range
AS WITH
    ranks AS
    (
        SELECT
            s.exp AS exp,
            s.run AS run,
            s.step AS step,
            s.node AS node,
            s.rank AS rank,
            s.microbatch AS microbatch,
            s.layer AS layer,
            extract(mn.name, '^act/([^/]+)/amax$') AS point,
            max(s.`values`[1]) AS amax
        FROM metric_sampler.scalars AS s
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (s.exp = las.exp) AND (s.run = las.run) AND (s.step = las.step) AND (s.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = s.exp) AND (mn.run = s.run) AND (mn.attempt = s.attempt) AND (mn.metric_id = s.metric_id)
        WHERE (s.layer <= 255) AND (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND (s.step <= {step_to:UInt32}) AND (({exp_filter:String} = '') OR (s.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (s.run = {run_filter:String})) AND (({microbatch_filter:Int32} = -2) OR (s.microbatch = {microbatch_filter:Int32})) AND match(mn.name, '^act/[^/]+/amax$') AND (s.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/amax$')
        ))
        GROUP BY
            s.exp,
            s.run,
            s.step,
            s.node,
            s.rank,
            s.microbatch,
            s.layer,
            point
    ),
    step_median AS
    (
        SELECT
            exp,
            run,
            step,
            microbatch,
            layer,
            point,
            medianExact(amax) AS median_amax
        FROM ranks
        GROUP BY
            exp,
            run,
            step,
            microbatch,
            layer,
            point
    ),
    deviations AS
    (
        SELECT
            ranks.exp AS exp,
            ranks.run AS run,
            ranks.step AS step,
            ranks.node AS node,
            ranks.rank AS rank,
            ranks.microbatch AS microbatch,
            ranks.layer AS layer,
            ranks.point AS point,
            ranks.amax AS amax,
            log(ranks.amax / step_median.median_amax) AS log_ratio
        FROM ranks
        INNER JOIN step_median ON (step_median.exp = ranks.exp) AND (step_median.run = ranks.run) AND (step_median.step = ranks.step) AND (step_median.layer = ranks.layer) AND (step_median.point = ranks.point) AND (step_median.microbatch = ranks.microbatch)
    )
SELECT
    exp,
    run,
    step,
    node,
    rank,
    microbatch,
    layer,
    point,
    amax,
    log_ratio
FROM deviations
WHERE (({rank_filter:Int32} = -2) OR (rank = {rank_filter:Int32}));

-- The raw amax rows are aggregated by metric and the point is named afterwards: naming each row by the regex over
-- its metric's name is much more expensive.
CREATE OR REPLACE VIEW metric_sampler.activation_amax_range
AS WITH
    ranks_by_metric AS
    (
        SELECT
            s.exp AS exp,
            s.run AS run,
            s.step AS step,
            s.node AS node,
            s.rank AS rank,
            s.layer AS layer,
            s.attempt AS attempt,
            s.metric_id AS metric_id,
            max(s.`values`[1]) AS amax
        FROM metric_sampler.scalars AS s
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (s.exp = las.exp) AND (s.run = las.run) AND (s.step = las.step) AND (s.attempt = las.attempt)
        WHERE (s.layer <= 255) AND (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND (s.step <= {step_to:UInt32}) AND (({exp_filter:String} = '') OR (s.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (s.run = {run_filter:String})) AND (({microbatch_filter:Int32} = -2) OR (s.microbatch = {microbatch_filter:Int32})) AND (s.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/amax$')
        ))
        GROUP BY
            s.exp,
            s.run,
            s.step,
            s.node,
            s.rank,
            s.layer,
            s.attempt,
            s.metric_id
    ),
    ranks AS
    (
        SELECT
            r.exp AS exp,
            r.run AS run,
            r.step AS step,
            r.node AS node,
            r.rank AS rank,
            r.layer AS layer,
            extract(mn.name, '^act/([^/]+)/amax$') AS point,
            r.amax AS amax
        FROM ranks_by_metric AS r
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = r.exp) AND (mn.run = r.run) AND (mn.attempt = r.attempt) AND (mn.metric_id = r.metric_id)
        WHERE match(mn.name, '^act/[^/]+/amax$')
    ),
    step_median AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            medianExact(amax) AS median_amax
        FROM ranks
        GROUP BY
            exp,
            run,
            step,
            layer,
            point
    ),
    deviations AS
    (
        SELECT
            ranks.exp AS exp,
            ranks.run AS run,
            ranks.step AS step,
            ranks.node AS node,
            ranks.rank AS rank,
            ranks.layer AS layer,
            ranks.point AS point,
            ranks.amax AS amax,
            log(ranks.amax / step_median.median_amax) AS log_ratio
        FROM ranks
        INNER JOIN step_median ON (step_median.exp = ranks.exp) AND (step_median.run = ranks.run) AND (step_median.step = ranks.step) AND (step_median.layer = ranks.layer) AND (step_median.point = ranks.point)
    )
SELECT
    exp,
    run,
    step,
    node,
    rank,
    layer,
    point,
    amax,
    log_ratio
FROM deviations
WHERE (({rank_filter:Int32} = -2) OR (rank = {rank_filter:Int32}));

CREATE OR REPLACE VIEW metric_sampler.activation_histograms_selected_range
AS WITH
    counts_names AS
    (
        SELECT
            attempt,
            metric_id,
            extract(name, '^act/([^/]+)/counts$') AS point
        FROM metric_sampler.metric_names
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND match(name, '^act/[^/]+/counts$')
    ),
    latest AS
    (
        SELECT
            step,
            attempt
        FROM metric_sampler.latest_attempt_step
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32})
    ),
    exact_counts AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.attempt AS attempt,
            ec.step AS step,
            ec.node AS node,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            ec.layer AS layer,
            counts_names.point AS point,
            ec.`values` AS counts
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN counts_names ON (counts_names.attempt = ec.attempt) AND (counts_names.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (ec.source_metric_id = 0) AND ((ec.step, ec.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/counts$')
        ))
    )
SELECT *
FROM
(
    SELECT
        exp,
        run,
        attempt,
        step,
        node,
        rank,
        microbatch,
        layer,
        point,
        'exp' AS histogram,
        -40. AS bins_start,
        1. AS bins_width,
        toUInt32(64) AS bins_count,
        arraySlice(counts, 3, 64) AS exact_hist,
        emptyArrayFloat64() AS legacy_hist,
        1 AS counts_exact
    FROM exact_counts
    UNION ALL
    SELECT
        exp,
        run,
        attempt,
        step,
        node,
        rank,
        microbatch,
        layer,
        point,
        'token_rms' AS histogram,
        -40. AS bins_start,
        1. AS bins_width,
        toUInt32(64) AS bins_count,
        arraySlice(counts, 67, 64) AS exact_hist,
        emptyArrayFloat64() AS legacy_hist,
        1 AS counts_exact
    FROM exact_counts
    UNION ALL
    SELECT
        rv.exp AS exp,
        rv.run AS run,
        rv.attempt AS attempt,
        rv.step AS step,
        rv.node AS node,
        rv.rank AS rank,
        rv.microbatch AS microbatch,
        rv.layer AS layer,
        extract(mn.name, '^act/([^/]+)/') AS point,
        extract(mn.name, '^act/[^/]+/(exp|token_rms)_hist$') AS histogram,
        mn.bins_start AS bins_start,
        mn.bins_width AS bins_width,
        mn.bins_count AS bins_count,
        emptyArrayUInt64() AS exact_hist,
        rv.`values` AS legacy_hist,
        0 AS counts_exact
    FROM metric_sampler.router_vectors AS rv
    FINAL
    INNER JOIN latest ON (latest.step = rv.step) AND (latest.attempt = rv.attempt)
    INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
    WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND match(mn.name, '^act/[^/]+/(exp|token_rms)_hist$') AND (rv.source_metric_id = 0) AND (rv.metric_id IN (
        SELECT metric_id
        FROM metric_sampler.metrics
        WHERE match(name, '^act/[^/]+/(exp|token_rms)_hist$')
    )) AND ((rv.exp, rv.run, rv.attempt, rv.step, rv.node, rv.rank, rv.microbatch, rv.layer, extract(mn.name, '^act/([^/]+)/')) NOT IN (
        SELECT
            exp,
            run,
            attempt,
            step,
            node,
            rank,
            microbatch,
            layer,
            point
        FROM exact_counts
        WHERE step IN (
            SELECT step
            FROM metric_sampler.router_vectors
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32}) AND (metric_id IN (
                SELECT metric_id
                FROM metric_sampler.metrics
                WHERE match(name, '^act/[^/]+/(exp|token_rms)_hist$')
            ))
        )
    ))
);

CREATE OR REPLACE VIEW metric_sampler.activation_fast
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_activation
FINAL
WHERE ({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2) AND (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}))
UNION ALL
SELECT *
FROM `metric_sampler.activation_selected`(exp_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2), '-', {exp_filter:String}), run_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2), '-', {run_filter:String}), rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32})
WHERE NOT (({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2));

CREATE OR REPLACE VIEW metric_sampler.activation_histograms_all
AS SELECT
    h.exp AS exp,
    h.run AS run,
    '' AS attempt,
    h.step AS step,
    '' AS node,
    toInt16(-2) AS rank,
    toInt16(-2) AS microbatch,
    h.layer AS layer,
    h.point AS point,
    h.histogram AS histogram,
    h.bins_start AS bins_start,
    h.bins_width AS bins_width,
    h.bins_count AS bins_count,
    h.exact_hist AS exact_hist,
    h.legacy_hist AS legacy_hist,
    h.counts_exact AS counts_exact
FROM metric_sampler.summary_activation_hist_latest AS h
FINAL
WHERE (h.exp = {exp_filter:String}) AND (h.run = {run_filter:String}) AND (h.step = (
    SELECT max(step)
    FROM metric_sampler.summary_activation_hist_latest
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
));

CREATE OR REPLACE VIEW metric_sampler.activation_histograms_live
AS SELECT *
FROM `metric_sampler.activation_histograms_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_histograms_mblatest
AS SELECT
    h.exp AS exp,
    h.run AS run,
    '' AS attempt,
    h.step AS step,
    '' AS node,
    toInt16(-2) AS rank,
    h.microbatch AS microbatch,
    h.layer AS layer,
    h.point AS point,
    h.histogram AS histogram,
    h.bins_start AS bins_start,
    h.bins_width AS bins_width,
    h.bins_count AS bins_count,
    h.exact_hist AS exact_hist,
    h.legacy_hist AS legacy_hist,
    h.counts_exact AS counts_exact
FROM metric_sampler.summary_activation_hist_latest_mb AS h
FINAL
WHERE (h.exp = {exp_filter:String}) AND (h.run = {run_filter:String}) AND (h.microbatch = {microbatch_filter:Int32}) AND (h.step = (
    SELECT max(step)
    FROM metric_sampler.summary_activation_hist_latest_mb
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32})
));

CREATE OR REPLACE VIEW metric_sampler.activation_histograms_ranklatest
AS SELECT
    h.exp AS exp,
    h.run AS run,
    '' AS attempt,
    h.step AS step,
    '' AS node,
    h.rank AS rank,
    toInt16(-2) AS microbatch,
    h.layer AS layer,
    h.point AS point,
    h.histogram AS histogram,
    h.bins_start AS bins_start,
    h.bins_width AS bins_width,
    h.bins_count AS bins_count,
    h.exact_hist AS exact_hist,
    h.legacy_hist AS legacy_hist,
    h.counts_exact AS counts_exact
FROM metric_sampler.summary_activation_hist_latest_rank AS h
FINAL
WHERE (h.exp = {exp_filter:String}) AND (h.run = {run_filter:String}) AND (h.rank = {rank_filter:Int32}) AND (h.step = (
    SELECT max(step)
    FROM metric_sampler.summary_activation_hist_latest_rank
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32})
));

CREATE OR REPLACE VIEW metric_sampler.activation_histograms_rankmblatest
AS SELECT
    h.exp AS exp,
    h.run AS run,
    '' AS attempt,
    h.step AS step,
    '' AS node,
    h.rank AS rank,
    h.microbatch AS microbatch,
    h.layer AS layer,
    h.point AS point,
    h.histogram AS histogram,
    h.bins_start AS bins_start,
    h.bins_width AS bins_width,
    h.bins_count AS bins_count,
    h.exact_hist AS exact_hist,
    h.legacy_hist AS legacy_hist,
    h.counts_exact AS counts_exact
FROM metric_sampler.summary_activation_hist_latest_rank_mb AS h
FINAL
WHERE (h.exp = {exp_filter:String}) AND (h.run = {run_filter:String}) AND (h.rank = {rank_filter:Int32}) AND (h.microbatch = {microbatch_filter:Int32}) AND (h.step = (
    SELECT max(step)
    FROM metric_sampler.summary_activation_hist_latest_rank_mb
    WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32}) AND (microbatch = {microbatch_filter:Int32})
));

CREATE OR REPLACE VIEW metric_sampler.activation_histograms_selected_live
AS SELECT *
FROM `metric_sampler.activation_histograms_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.activation_histograms_step
AS WITH exact_counts AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.attempt AS attempt,
            ec.step AS step,
            ec.node AS node,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            ec.layer AS layer,
            extract(mn.name, '^act/([^/]+)/counts$') AS point,
            ec.`values` AS counts
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND match(mn.name, '^act/[^/]+/counts$') AND (ec.source_metric_id = 0) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/counts$')
        ))
    )
SELECT
    exp,
    run,
    attempt,
    step,
    node,
    rank,
    microbatch,
    layer,
    point,
    'exp' AS histogram,
    -40. AS bins_start,
    1. AS bins_width,
    toUInt32(64) AS bins_count,
    arraySlice(counts, 3, 64) AS exact_hist,
    emptyArrayFloat64() AS legacy_hist,
    1 AS counts_exact
FROM exact_counts
UNION ALL
WITH exact_counts AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.attempt AS attempt,
            ec.step AS step,
            ec.node AS node,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            ec.layer AS layer,
            extract(mn.name, '^act/([^/]+)/counts$') AS point,
            ec.`values` AS counts
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND match(mn.name, '^act/[^/]+/counts$') AND (ec.source_metric_id = 0) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/counts$')
        ))
    )
SELECT
    exp,
    run,
    attempt,
    step,
    node,
    rank,
    microbatch,
    layer,
    point,
    'token_rms' AS histogram,
    -40. AS bins_start,
    1. AS bins_width,
    toUInt32(64) AS bins_count,
    arraySlice(counts, 67, 64) AS exact_hist,
    emptyArrayFloat64() AS legacy_hist,
    1 AS counts_exact
FROM exact_counts
UNION ALL
WITH exact_counts AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.attempt AS attempt,
            ec.step AS step,
            ec.node AS node,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            ec.layer AS layer,
            extract(mn.name, '^act/([^/]+)/counts$') AS point,
            ec.`values` AS counts
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND match(mn.name, '^act/[^/]+/counts$') AND (ec.source_metric_id = 0) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/counts$')
        ))
    )
SELECT
    rv.exp AS exp,
    rv.run AS run,
    rv.attempt AS attempt,
    rv.step AS step,
    rv.node AS node,
    rv.rank AS rank,
    rv.microbatch AS microbatch,
    rv.layer AS layer,
    extract(mn.name, '^act/([^/]+)/') AS point,
    extract(mn.name, '^act/[^/]+/(exp|token_rms)_hist$') AS histogram,
    mn.bins_start AS bins_start,
    mn.bins_width AS bins_width,
    mn.bins_count AS bins_count,
    emptyArrayUInt64() AS exact_hist,
    rv.`values` AS legacy_hist,
    0 AS counts_exact
FROM metric_sampler.router_vectors AS rv
FINAL
INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
WHERE (rv.layer <= 255) AND (rv.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (rv.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (rv.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND match(mn.name, '^act/[^/]+/(exp|token_rms)_hist$') AND (rv.source_metric_id = 0) AND (rv.metric_id IN (
    SELECT metric_id
    FROM metric_sampler.metrics
    WHERE match(name, '^act/[^/]+/(exp|token_rms)_hist$')
)) AND ((rv.exp, rv.run, rv.attempt, rv.step, rv.node, rv.rank, rv.microbatch, rv.layer, extract(mn.name, '^act/([^/]+)/')) NOT IN (
    SELECT
        exp,
        run,
        attempt,
        step,
        node,
        rank,
        microbatch,
        layer,
        point
    FROM exact_counts
));

CREATE OR REPLACE VIEW metric_sampler.activation_histograms_step_live
AS SELECT *
FROM `metric_sampler.activation_histograms_step`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_filter = {step_filter:UInt32});

CREATE OR REPLACE VIEW metric_sampler.activation_live
AS SELECT *
FROM `metric_sampler.activation_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_mb
AS SELECT *
FROM `metric_sampler.activation_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_mbsum
AS SELECT * EXCEPT (refreshed_at, microbatch)
FROM metric_sampler.summary_activation_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_fast
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `node` LowCardinality(String),
    `rank` Int16,
    `layer` Int16,
    `point` LowCardinality(String),
    `amax` Float32,
    `robust_z` Nullable(Float64)
)
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_activation_per_rank
FINAL;

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_live
(
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `node` LowCardinality(String),
    `rank` Int16,
    `layer` Int16,
    `point` LowCardinality(String),
    `amax` Float32,
    `robust_z` Nullable(Float64)
)
AS SELECT *
FROM metric_sampler.activation_per_rank;

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_mbsum
AS WITH
    observed AS
    (
        SELECT
            exp,
            run,
            step,
            node,
            rank,
            layer,
            point,
            amax,
            log_ratio
        FROM metric_sampler.summary_activation_amax_mb
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32})
    ),
    run_scale AS
    (
        SELECT
            layer,
            point,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM observed
        GROUP BY
            layer,
            point
    )
SELECT
    observed.exp AS exp,
    observed.run AS run,
    observed.step AS step,
    observed.node AS node,
    observed.rank AS rank,
    observed.layer AS layer,
    observed.point AS point,
    observed.amax AS amax,
    observed.log_ratio / nullIf(run_scale.scale, 0) AS robust_z
FROM observed
INNER JOIN run_scale ON (run_scale.layer = observed.layer) AND (run_scale.point = observed.point)
WHERE (({rank_filter:Int32} = -2) OR (observed.rank = {rank_filter:Int32}));

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_mbsum_point
AS WITH
    observed AS
    (
        SELECT
            exp,
            run,
            step,
            node,
            rank,
            layer,
            point,
            amax,
            log_ratio
        FROM metric_sampler.summary_activation_amax_mb
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32}) AND (layer = {layer_filter:Int32}) AND (point = {point_filter:String})
    ),
    run_scale AS
    (
        SELECT
            layer,
            point,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM observed
        GROUP BY
            layer,
            point
    )
SELECT
    observed.exp AS exp,
    observed.run AS run,
    observed.step AS step,
    observed.node AS node,
    observed.rank AS rank,
    observed.layer AS layer,
    observed.point AS point,
    observed.amax AS amax,
    observed.log_ratio / nullIf(run_scale.scale, 0) AS robust_z
FROM observed
INNER JOIN run_scale ON (run_scale.layer = observed.layer) AND (run_scale.point = observed.point)
WHERE (({rank_filter:Int32} = -2) OR (observed.rank = {rank_filter:Int32}));

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_point
AS WITH
    ranks AS
    (
        SELECT
            s.exp AS exp,
            s.run AS run,
            s.step AS step,
            s.node AS node,
            s.rank AS rank,
            s.layer AS layer,
            extract(mn.name, '^act/([^/]+)/amax$') AS point,
            max(s.`values`[1]) AS amax
        FROM metric_sampler.scalars AS s
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (s.exp = las.exp) AND (s.run = las.run) AND (s.step = las.step) AND (s.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = s.exp) AND (mn.run = s.run) AND (mn.attempt = s.attempt) AND (mn.metric_id = s.metric_id)
        WHERE (s.layer <= 255) AND (({exp_filter:String} = '') OR (s.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (s.run = {run_filter:String})) AND (({microbatch_filter:Int32} = -2) OR (s.microbatch = {microbatch_filter:Int32})) AND (s.layer = {layer_filter:Int32}) AND (mn.name = concat('act/', {point_filter:String}, '/amax')) AND (s.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = concat('act/', {point_filter:String}, '/amax')
        ))
        GROUP BY
            s.exp,
            s.run,
            s.step,
            s.node,
            s.rank,
            s.layer,
            point
    ),
    step_median AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            medianExact(amax) AS median_amax
        FROM ranks
        GROUP BY
            exp,
            run,
            step,
            layer,
            point
    ),
    deviations AS
    (
        SELECT
            ranks.exp AS exp,
            ranks.run AS run,
            ranks.step AS step,
            ranks.node AS node,
            ranks.rank AS rank,
            ranks.layer AS layer,
            ranks.point AS point,
            ranks.amax AS amax,
            log(ranks.amax / step_median.median_amax) AS log_ratio
        FROM ranks
        INNER JOIN step_median ON (step_median.exp = ranks.exp) AND (step_median.run = ranks.run) AND (step_median.step = ranks.step) AND (step_median.layer = ranks.layer) AND (step_median.point = ranks.point)
    ),
    run_scale AS
    (
        SELECT
            exp,
            run,
            layer,
            point,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM deviations
        GROUP BY
            exp,
            run,
            layer,
            point
    )
SELECT
    deviations.exp AS exp,
    deviations.run AS run,
    deviations.step AS step,
    deviations.node AS node,
    deviations.rank AS rank,
    deviations.layer AS layer,
    deviations.point AS point,
    deviations.amax AS amax,
    deviations.log_ratio / nullIf(run_scale.scale, 0) AS robust_z
FROM deviations
INNER JOIN run_scale ON (run_scale.exp = deviations.exp) AND (run_scale.run = deviations.run) AND (run_scale.layer = deviations.layer) AND (run_scale.point = deviations.point)
WHERE (({rank_filter:Int32} = -2) OR (deviations.rank = {rank_filter:Int32}));

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.activation_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_summary
AS WITH
    observed AS
    (
        SELECT
            exp,
            run,
            step,
            node,
            rank,
            layer,
            point,
            amax,
            log_ratio
        FROM metric_sampler.summary_activation_amax
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String})
    ),
    run_scale AS
    (
        SELECT
            layer,
            point,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM observed
        GROUP BY
            layer,
            point
    )
SELECT
    observed.exp AS exp,
    observed.run AS run,
    observed.step AS step,
    observed.node AS node,
    observed.rank AS rank,
    observed.layer AS layer,
    observed.point AS point,
    observed.amax AS amax,
    observed.log_ratio / nullIf(run_scale.scale, 0) AS robust_z
FROM observed
INNER JOIN run_scale ON (run_scale.layer = observed.layer) AND (run_scale.point = observed.point)
WHERE (({rank_filter:Int32} = -2) OR (observed.rank = {rank_filter:Int32}));

CREATE OR REPLACE VIEW metric_sampler.activation_per_rank_summary_point
AS WITH
    observed AS
    (
        SELECT
            exp,
            run,
            step,
            node,
            rank,
            layer,
            point,
            amax,
            log_ratio
        FROM metric_sampler.summary_activation_amax
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (layer = {layer_filter:Int32}) AND (point = {point_filter:String})
    ),
    run_scale AS
    (
        SELECT
            layer,
            point,
            1.4826 * medianExact(abs(log_ratio)) AS scale
        FROM observed
        GROUP BY
            layer,
            point
    )
SELECT
    observed.exp AS exp,
    observed.run AS run,
    observed.step AS step,
    observed.node AS node,
    observed.rank AS rank,
    observed.layer AS layer,
    observed.point AS point,
    observed.amax AS amax,
    observed.log_ratio / nullIf(run_scale.scale, 0) AS robust_z
FROM observed
INNER JOIN run_scale ON (run_scale.layer = observed.layer) AND (run_scale.point = observed.point)
WHERE (({rank_filter:Int32} = -2) OR (observed.rank = {rank_filter:Int32}));

CREATE OR REPLACE VIEW metric_sampler.activation_ranksum
AS SELECT * EXCEPT (refreshed_at, rank)
FROM metric_sampler.summary_activation_rank
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_selected_live
AS SELECT *
FROM `metric_sampler.activation_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

-- The observations are grouped by a numeric id of the point (its dense rank among the run's names) and the point
-- joined back afterwards: grouping by the point string costs more CPU.
CREATE OR REPLACE VIEW metric_sampler.activation_by_rank_mb_range
AS WITH
    names AS
    (
        SELECT
            attempt,
            metric_id,
            extract(name, '^act/([^/]+)/') AS point,
            multiIf(endsWith(name, '/moments'), 1, endsWith(name, '/power_sums'), 2, endsWith(name, '/counts'), 3, 4) AS source,
            toUInt16(denseRank() OVER (ORDER BY extract(name, '^act/([^/]+)/') ASC)) AS point_id
        FROM metric_sampler.metric_names
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND match(name, '^act/[^/]+/(moments|power_sums|counts|amax)$')
    ),
    latest AS
    (
        SELECT
            step,
            attempt
        FROM metric_sampler.latest_attempt_step
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32})
    ),
    observed AS
    (
        SELECT
            names.source AS source,
            rv.step AS step,
            rv.layer AS layer,
            rv.rank AS rank,
            rv.microbatch AS microbatch,
            names.point_id AS point_id,
            toFloat64(rv.`values`[1]) AS first,
            toFloat64(rv.`values`[2]) AS second,
            toFloat64(rv.`values`[3]) AS third,
            toFloat64(rv.`values`[4]) AS fourth
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN names ON (names.attempt = rv.attempt) AND (names.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND (rv.source_metric_id = 0) AND ((rv.step, rv.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/(moments|power_sums)$')
        ))
        UNION ALL
        SELECT
            names.source AS source,
            ec.step AS step,
            ec.layer AS layer,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            names.point_id AS point_id,
            toFloat64(ec.`values`[1]) AS first,
            toFloat64(ec.`values`[2]) AS second,
            0. AS third,
            0. AS fourth
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN names ON (names.attempt = ec.attempt) AND (names.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (ec.source_metric_id = 0) AND ((ec.step, ec.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/counts$')
        ))
        UNION ALL
        SELECT
            names.source AS source,
            s.step AS step,
            s.layer AS layer,
            s.rank AS rank,
            s.microbatch AS microbatch,
            names.point_id AS point_id,
            toFloat64(s.`values`[1]) AS first,
            0. AS second,
            0. AS third,
            0. AS fourth
        FROM metric_sampler.scalars AS s
        FINAL
        INNER JOIN names ON (names.attempt = s.attempt) AND (names.metric_id = s.metric_id)
        WHERE (s.layer <= 255) AND (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String}) AND (s.step >= {step_from:UInt32}) AND (s.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (s.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (s.microbatch = {microbatch_filter:Int32})) AND ((s.step, s.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (s.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/amax$')
        ))
    ),
    observations AS
    (
        SELECT
            step,
            layer,
            rank,
            microbatch,
            point_id,
            countIf(source = 1) AS moment_rows,
            sumIf(first, source = 1) AS sum_x,
            sumIf(second, source = 1) AS sumsq_x,
            sumIf(third, source = 1) AS legacy_nonfinite,
            sumIf(fourth, source = 1) AS legacy_elements,
            countIf(source = 2) AS power_rows,
            sumIf(first, source = 2) AS sum_cube,
            sumIf(second, source = 2) AS sum_fourth,
            countIf(source = 3) AS count_rows,
            sumIf(first, source = 3) AS counted_nonfinite,
            sumIf(second, source = 3) AS counted_elements,
            countIf(source = 4) AS amax_rows,
            maxIf(first, source = 4) AS amax
        FROM observed
        GROUP BY
            step,
            layer,
            rank,
            microbatch,
            point_id
        HAVING (moment_rows > 0) AND (amax_rows > 0)
    ),
    point_names AS
    (
        SELECT DISTINCT
            point_id,
            point
        FROM names
    )
SELECT
    toLowCardinality({exp_filter:String}) AS exp,
    toLowCardinality({run_filter:String}) AS run,
    step,
    layer,
    point,
    rank,
    microbatch,
    if(count_rows > 0, counted_elements, legacy_elements) AS elements,
    sum_x / nullIf(elements, 0) AS mean,
    sqrt(sumsq_x / nullIf(elements, 0)) AS rms,
    sqrt(greatest((sumsq_x / nullIf(elements, 0)) - pow(sum_x / nullIf(elements, 0), 2), 0)) AS std,
    if(count_rows > 0, counted_nonfinite, legacy_nonfinite) AS nonfinite,
    toFloat32(amax) AS amax_max,
    toFloat32(amax) AS amax_median,
    127 - log2(amax_max) AS bf16_headroom_log2,
    log2(448) - log2(amax_max) AS fp8_e4m3_headroom_log2,
    toUInt8(count_rows > 0) AS all_counts_exact,
    multiIf(all_counts_exact = 1, toUInt64(counted_elements), NULL) AS exact_elements,
    multiIf(all_counts_exact = 1, toUInt64(counted_nonfinite), NULL) AS exact_nonfinite,
    multiIf(power_rows = moment_rows, (((sum_cube / nullIf(elements, 0)) - ((3 * (sum_x / nullIf(elements, 0))) * (sumsq_x / nullIf(elements, 0)))) + (2 * pow(sum_x / nullIf(elements, 0), 3))) / nullIf(pow(greatest((sumsq_x / nullIf(elements, 0)) - pow(sum_x / nullIf(elements, 0), 2), 0), 1.5), 0), NULL) AS skewness,
    multiIf(power_rows = moment_rows, ((((sum_fourth / nullIf(elements, 0)) - ((4 * (sum_x / nullIf(elements, 0))) * (sum_cube / nullIf(elements, 0)))) + ((6 * pow(sum_x / nullIf(elements, 0), 2)) * (sumsq_x / nullIf(elements, 0)))) - (3 * pow(sum_x / nullIf(elements, 0), 4))) / nullIf(pow(greatest((sumsq_x / nullIf(elements, 0)) - pow(sum_x / nullIf(elements, 0), 2), 0), 2), 0), NULL) AS kurtosis
FROM observations
INNER JOIN point_names USING (point_id);

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_rank_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `microbatch` Int16,
    `elements` Float64,
    `mean` Nullable(Float64),
    `rms` Nullable(Float64),
    `std` Nullable(Float64),
    `nonfinite` Float64,
    `amax_max` Float32,
    `amax_median` Float32,
    `bf16_headroom_log2` Float64,
    `fp8_e4m3_headroom_log2` Float64,
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64),
    `exact_nonfinite` Nullable(UInt64),
    `skewness` Nullable(Float64),
    `kurtosis` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, rank, microbatch, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)', max_bytes_to_merge_at_max_space_in_pool = 4294967296;

CREATE OR REPLACE VIEW metric_sampler.activation_rankmbsum
AS SELECT * EXCEPT (refreshed_at, rank, microbatch)
FROM metric_sampler.summary_activation_rank_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32}) AND (microbatch = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_by_rank_range
AS WITH
    names AS
    (
        SELECT
            attempt,
            metric_id,
            extract(name, '^act/([^/]+)/') AS point,
            endsWith(name, '/power_sums') AS power
        FROM metric_sampler.metric_names
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND match(name, '^act/[^/]+/(moments|power_sums)$')
    ),
    latest AS
    (
        SELECT
            step,
            attempt
        FROM metric_sampler.latest_attempt_step
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32})
    ),
    sums AS
    (
        SELECT
            rv.step AS step,
            rv.layer AS layer,
            rv.rank AS rank,
            names.point AS point,
            sumIf(rv.`values`[1], NOT names.power) AS sum_x,
            sumIf(rv.`values`[2], NOT names.power) AS sumsq_x,
            countIf(NOT names.power) AS moment_rows,
            sumIf(rv.`values`[1], names.power) AS sum_cube,
            sumIf(rv.`values`[2], names.power) AS sum_fourth,
            countIf(names.power) AS power_rows
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN names ON (names.attempt = rv.attempt) AND (names.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND (rv.source_metric_id = 0) AND ((rv.step, rv.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/(moments|power_sums)$')
        ))
        GROUP BY
            rv.step,
            rv.layer,
            rv.rank,
            names.point
    ),
    fine AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            sum(elements) AS total_elements,
            sum(nonfinite) AS total_nonfinite,
            max(amax_max) AS amax_top,
            max(amax_max) AS amax_mid,
            min(all_counts_exact) AS exact_counts,
            sum(exact_elements) AS total_exact_elements,
            sum(exact_nonfinite) AS total_exact_nonfinite
        FROM metric_sampler.summary_activation_rank_mb
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (microbatch = {microbatch_filter:Int32}))
        GROUP BY
            exp,
            run,
            step,
            layer,
            rank,
            point
    )
SELECT
    fine.exp AS exp,
    fine.run AS run,
    sums.step AS step,
    sums.layer AS layer,
    sums.point AS point,
    sums.rank AS rank,
    fine.total_elements AS elements,
    sums.sum_x / nullIf(fine.total_elements, 0) AS mean,
    sqrt(sums.sumsq_x / nullIf(fine.total_elements, 0)) AS rms,
    sqrt(greatest((sums.sumsq_x / nullIf(fine.total_elements, 0)) - pow(sums.sum_x / nullIf(fine.total_elements, 0), 2), 0)) AS std,
    fine.total_nonfinite AS nonfinite,
    fine.amax_top AS amax_max,
    fine.amax_mid AS amax_median,
    127 - log2(fine.amax_top) AS bf16_headroom_log2,
    log2(448) - log2(fine.amax_top) AS fp8_e4m3_headroom_log2,
    fine.exact_counts AS all_counts_exact,
    multiIf(fine.exact_counts = 1, fine.total_exact_elements, NULL) AS exact_elements,
    multiIf(fine.exact_counts = 1, fine.total_exact_nonfinite, NULL) AS exact_nonfinite,
    multiIf(sums.power_rows = sums.moment_rows, (((sums.sum_cube / nullIf(fine.total_elements, 0)) - ((3 * (sums.sum_x / nullIf(fine.total_elements, 0))) * (sums.sumsq_x / nullIf(fine.total_elements, 0)))) + (2 * pow(sums.sum_x / nullIf(fine.total_elements, 0), 3))) / nullIf(pow(greatest((sums.sumsq_x / nullIf(fine.total_elements, 0)) - pow(sums.sum_x / nullIf(fine.total_elements, 0), 2), 0), 1.5), 0), NULL) AS skewness,
    multiIf(sums.power_rows = sums.moment_rows, ((((sums.sum_fourth / nullIf(fine.total_elements, 0)) - ((4 * (sums.sum_x / nullIf(fine.total_elements, 0))) * (sums.sum_cube / nullIf(fine.total_elements, 0)))) + ((6 * pow(sums.sum_x / nullIf(fine.total_elements, 0), 2)) * (sums.sumsq_x / nullIf(fine.total_elements, 0)))) - (3 * pow(sums.sum_x / nullIf(fine.total_elements, 0), 4))) / nullIf(pow(greatest((sums.sumsq_x / nullIf(fine.total_elements, 0)) - pow(sums.sum_x / nullIf(fine.total_elements, 0), 2), 0), 2), 0), NULL) AS kurtosis
FROM sums
INNER JOIN fine ON (fine.step = sums.step) AND (fine.layer = sums.layer) AND (fine.point = sums.point) AND (fine.rank = sums.rank);

CREATE OR REPLACE VIEW metric_sampler.activation_by_mb_range
AS WITH
    names AS
    (
        SELECT
            attempt,
            metric_id,
            extract(name, '^act/([^/]+)/') AS point,
            endsWith(name, '/power_sums') AS power
        FROM metric_sampler.metric_names
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND match(name, '^act/[^/]+/(moments|power_sums)$')
    ),
    latest AS
    (
        SELECT
            step,
            attempt
        FROM metric_sampler.latest_attempt_step
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32})
    ),
    sums AS
    (
        SELECT
            rv.step AS step,
            rv.layer AS layer,
            rv.microbatch AS microbatch,
            names.point AS point,
            sumIf(rv.`values`[1], NOT names.power) AS sum_x,
            sumIf(rv.`values`[2], NOT names.power) AS sumsq_x,
            countIf(NOT names.power) AS moment_rows,
            sumIf(rv.`values`[1], names.power) AS sum_cube,
            sumIf(rv.`values`[2], names.power) AS sum_fourth,
            countIf(names.power) AS power_rows
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN names ON (names.attempt = rv.attempt) AND (names.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND (rv.source_metric_id = 0) AND ((rv.step, rv.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/(moments|power_sums)$')
        ))
        GROUP BY
            rv.step,
            rv.layer,
            rv.microbatch,
            names.point
    ),
    fine AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            microbatch,
            sum(elements) AS total_elements,
            sum(nonfinite) AS total_nonfinite,
            max(amax_max) AS amax_top,
            medianExact(amax_max) AS amax_mid,
            min(all_counts_exact) AS exact_counts,
            sum(exact_elements) AS total_exact_elements,
            sum(exact_nonfinite) AS total_exact_nonfinite
        FROM metric_sampler.summary_activation_rank_mb
        FINAL
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (microbatch = {microbatch_filter:Int32}))
        GROUP BY
            exp,
            run,
            step,
            layer,
            microbatch,
            point
    )
SELECT
    fine.exp AS exp,
    fine.run AS run,
    sums.step AS step,
    sums.layer AS layer,
    sums.point AS point,
    sums.microbatch AS microbatch,
    fine.total_elements AS elements,
    sums.sum_x / nullIf(fine.total_elements, 0) AS mean,
    sqrt(sums.sumsq_x / nullIf(fine.total_elements, 0)) AS rms,
    sqrt(greatest((sums.sumsq_x / nullIf(fine.total_elements, 0)) - pow(sums.sum_x / nullIf(fine.total_elements, 0), 2), 0)) AS std,
    fine.total_nonfinite AS nonfinite,
    fine.amax_top AS amax_max,
    fine.amax_mid AS amax_median,
    127 - log2(fine.amax_top) AS bf16_headroom_log2,
    log2(448) - log2(fine.amax_top) AS fp8_e4m3_headroom_log2,
    fine.exact_counts AS all_counts_exact,
    multiIf(fine.exact_counts = 1, fine.total_exact_elements, NULL) AS exact_elements,
    multiIf(fine.exact_counts = 1, fine.total_exact_nonfinite, NULL) AS exact_nonfinite,
    multiIf(sums.power_rows = sums.moment_rows, (((sums.sum_cube / nullIf(fine.total_elements, 0)) - ((3 * (sums.sum_x / nullIf(fine.total_elements, 0))) * (sums.sumsq_x / nullIf(fine.total_elements, 0)))) + (2 * pow(sums.sum_x / nullIf(fine.total_elements, 0), 3))) / nullIf(pow(greatest((sums.sumsq_x / nullIf(fine.total_elements, 0)) - pow(sums.sum_x / nullIf(fine.total_elements, 0), 2), 0), 1.5), 0), NULL) AS skewness,
    multiIf(sums.power_rows = sums.moment_rows, ((((sums.sum_fourth / nullIf(fine.total_elements, 0)) - ((4 * (sums.sum_x / nullIf(fine.total_elements, 0))) * (sums.sum_cube / nullIf(fine.total_elements, 0)))) + ((6 * pow(sums.sum_x / nullIf(fine.total_elements, 0), 2)) * (sums.sumsq_x / nullIf(fine.total_elements, 0)))) - (3 * pow(sums.sum_x / nullIf(fine.total_elements, 0), 4))) / nullIf(pow(greatest((sums.sumsq_x / nullIf(fine.total_elements, 0)) - pow(sums.sum_x / nullIf(fine.total_elements, 0), 2), 0), 2), 0), NULL) AS kurtosis
FROM sums
INNER JOIN fine ON (fine.step = sums.step) AND (fine.layer = sums.layer) AND (fine.point = sums.point) AND (fine.microbatch = sums.microbatch);

CREATE OR REPLACE VIEW metric_sampler.activation_selected_range
AS WITH
    names AS
    (
        SELECT
            attempt,
            metric_id,
            extract(name, '^act/([^/]+)/') AS point,
            endsWith(name, '/power_sums') AS power
        FROM metric_sampler.metric_names
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND match(name, '^act/[^/]+/(moments|power_sums)$')
    ),
    latest AS
    (
        SELECT
            step,
            attempt
        FROM metric_sampler.latest_attempt_step
        WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32})
    ),
    sums AS
    (
        SELECT
            rv.step AS step,
            rv.layer AS layer,
            names.point AS point,
            sumIf(rv.`values`[1], NOT names.power) AS sum_x,
            sumIf(rv.`values`[2], NOT names.power) AS sumsq_x,
            countIf(NOT names.power) AS moment_rows,
            sumIf(rv.`values`[1], names.power) AS sum_cube,
            sumIf(rv.`values`[2], names.power) AS sum_fourth,
            countIf(names.power) AS power_rows
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN names ON (names.attempt = rv.attempt) AND (names.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND (rv.source_metric_id = 0) AND ((rv.step, rv.attempt) IN (
            SELECT
                step,
                attempt
            FROM latest
        )) AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE match(name, '^act/[^/]+/(moments|power_sums)$')
        ))
        GROUP BY
            rv.step,
            rv.layer,
            names.point
    ),
    fine AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            sum(rank_elements) AS total_elements,
            sum(rank_nonfinite) AS total_nonfinite,
            max(rank_amax) AS amax_top,
            medianExact(rank_amax) AS amax_mid,
            min(rank_exact) AS exact_counts,
            sum(rank_exact_elements) AS total_exact_elements,
            sum(rank_exact_nonfinite) AS total_exact_nonfinite
        FROM
        (
            SELECT
                exp,
                run,
                step,
                layer,
                point,
                rank,
                sum(elements) AS rank_elements,
                sum(nonfinite) AS rank_nonfinite,
                max(amax_max) AS rank_amax,
                min(all_counts_exact) AS rank_exact,
                sum(exact_elements) AS rank_exact_elements,
                sum(exact_nonfinite) AS rank_exact_nonfinite
            FROM metric_sampler.summary_activation_rank_mb
            FINAL
            WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (step >= {step_from:UInt32}) AND (step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (microbatch = {microbatch_filter:Int32}))
            GROUP BY
                exp,
                run,
                step,
                layer,
                point,
                rank
        )
        GROUP BY
            exp,
            run,
            step,
            layer,
            point
    )
SELECT
    fine.exp AS exp,
    fine.run AS run,
    sums.step AS step,
    sums.layer AS layer,
    sums.point AS point,
    fine.total_elements AS elements,
    sums.sum_x / nullIf(fine.total_elements, 0) AS mean,
    sqrt(sums.sumsq_x / nullIf(fine.total_elements, 0)) AS rms,
    sqrt(greatest((sums.sumsq_x / nullIf(fine.total_elements, 0)) - pow(sums.sum_x / nullIf(fine.total_elements, 0), 2), 0)) AS std,
    fine.total_nonfinite AS nonfinite,
    fine.amax_top AS amax_max,
    fine.amax_mid AS amax_median,
    127 - log2(fine.amax_top) AS bf16_headroom_log2,
    log2(448) - log2(fine.amax_top) AS fp8_e4m3_headroom_log2,
    fine.exact_counts AS all_counts_exact,
    multiIf(fine.exact_counts = 1, fine.total_exact_elements, NULL) AS exact_elements,
    multiIf(fine.exact_counts = 1, fine.total_exact_nonfinite, NULL) AS exact_nonfinite,
    multiIf(sums.power_rows = sums.moment_rows, (((sums.sum_cube / nullIf(fine.total_elements, 0)) - ((3 * (sums.sum_x / nullIf(fine.total_elements, 0))) * (sums.sumsq_x / nullIf(fine.total_elements, 0)))) + (2 * pow(sums.sum_x / nullIf(fine.total_elements, 0), 3))) / nullIf(pow(greatest((sums.sumsq_x / nullIf(fine.total_elements, 0)) - pow(sums.sum_x / nullIf(fine.total_elements, 0), 2), 0), 1.5), 0), NULL) AS skewness,
    multiIf(sums.power_rows = sums.moment_rows, ((((sums.sum_fourth / nullIf(fine.total_elements, 0)) - ((4 * (sums.sum_x / nullIf(fine.total_elements, 0))) * (sums.sum_cube / nullIf(fine.total_elements, 0)))) + ((6 * pow(sums.sum_x / nullIf(fine.total_elements, 0), 2)) * (sums.sumsq_x / nullIf(fine.total_elements, 0)))) - (3 * pow(sums.sum_x / nullIf(fine.total_elements, 0), 4))) / nullIf(pow(greatest((sums.sumsq_x / nullIf(fine.total_elements, 0)) - pow(sums.sum_x / nullIf(fine.total_elements, 0), 2), 0), 2), 0), NULL) AS kurtosis
FROM sums
INNER JOIN fine ON (fine.step = sums.step) AND (fine.layer = sums.layer) AND (fine.point = sums.point);

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_rank_by_layer
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `elements` Float64,
    `mean` Nullable(Float64),
    `rms` Nullable(Float64),
    `std` Nullable(Float64),
    `nonfinite` Float64,
    `amax_max` Float32,
    `amax_median` Float32,
    `bf16_headroom_log2` Float64,
    `fp8_e4m3_headroom_log2` Float64,
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64),
    `exact_nonfinite` Nullable(UInt64),
    `skewness` Nullable(Float64),
    `kurtosis` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, layer, point, step, rank)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)', max_bytes_to_merge_at_max_space_in_pool = 4294967296;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.summary_activation_rank_by_layer_mv TO metric_sampler.summary_activation_rank_by_layer
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `elements` Float64,
    `mean` Nullable(Float64),
    `rms` Nullable(Float64),
    `std` Nullable(Float64),
    `nonfinite` Float64,
    `amax_max` Float32,
    `amax_median` Float32,
    `bf16_headroom_log2` Float64,
    `fp8_e4m3_headroom_log2` Float64,
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64),
    `exact_nonfinite` Nullable(UInt64),
    `skewness` Nullable(Float64),
    `kurtosis` Nullable(Float64)
)
AS SELECT *
FROM metric_sampler.summary_activation_rank;
