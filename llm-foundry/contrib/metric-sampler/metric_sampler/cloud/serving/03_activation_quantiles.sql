CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_exp_quantiles
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `p50_log2` Nullable(Float64),
    `p99_log2` Nullable(Float64),
    `p999_log2` Nullable(Float64),
    `max_log2` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_exp_quantiles_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `microbatch` Int16,
    `p50_log2` Nullable(Float64),
    `p99_log2` Nullable(Float64),
    `p999_log2` Nullable(Float64),
    `max_log2` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, microbatch, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_exp_quantiles_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `p50_log2` Nullable(Float64),
    `p99_log2` Nullable(Float64),
    `p999_log2` Nullable(Float64),
    `max_log2` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, rank, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_massive
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `tokens` Float64,
    `median_log2_rms` Nullable(Float64),
    `massive_share` Nullable(Float64),
    `max_log2_rms` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_tokens` Nullable(UInt64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_massive_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `microbatch` Int16,
    `tokens` Float64,
    `median_log2_rms` Nullable(Float64),
    `massive_share` Nullable(Float64),
    `max_log2_rms` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_tokens` Nullable(UInt64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, microbatch, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_massive_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `tokens` Float64,
    `median_log2_rms` Nullable(Float64),
    `massive_share` Nullable(Float64),
    `max_log2_rms` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_tokens` Nullable(UInt64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, rank, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_activation_exp_quantiles
FINAL
WHERE (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}));

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_by_mb_range
AS WITH
    hist_sum AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            microbatch,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'exp'
        GROUP BY
            exp,
            run,
            step,
            layer,
            point,
            microbatch
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            microbatch,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS elements,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_elements
        FROM histograms
    )
SELECT
    exp,
    run,
    step,
    layer,
    point,
    microbatch,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) - 0.5))) AS p50_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) - 0.5))) AS p99_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) - 0.5))) AS p999_log2,
    multiIf(elements = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_elements, NULL) AS exact_elements
FROM with_totals;

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_by_rank_range
AS WITH
    hist_sum AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'exp'
        GROUP BY
            exp,
            run,
            step,
            layer,
            point,
            rank
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS elements,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_elements
        FROM histograms
    )
SELECT
    exp,
    run,
    step,
    layer,
    point,
    rank,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) - 0.5))) AS p50_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) - 0.5))) AS p99_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) - 0.5))) AS p999_log2,
    multiIf(elements = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_elements, NULL) AS exact_elements
FROM with_totals;

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_fast
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_activation_exp_quantiles
FINAL
WHERE ({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2) AND (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}))
UNION ALL
SELECT *
FROM `metric_sampler.activation_exp_quantiles_selected`(exp_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2), '-', {exp_filter:String}), run_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2), '-', {run_filter:String}), rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32})
WHERE NOT (({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2));

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_live
AS SELECT *
FROM `metric_sampler.activation_exp_quantiles_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_mb
AS SELECT *
FROM `metric_sampler.activation_exp_quantiles_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_mbsum
AS SELECT * EXCEPT (refreshed_at, microbatch)
FROM metric_sampler.summary_activation_exp_quantiles_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_ranksum
AS SELECT * EXCEPT (refreshed_at, rank)
FROM metric_sampler.summary_activation_exp_quantiles_rank
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_selected_live
AS SELECT *
FROM `metric_sampler.activation_exp_quantiles_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_selected_range
AS WITH
    hist_sum AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'exp'
        GROUP BY
            exp,
            run,
            step,
            layer,
            point
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS elements,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_elements
        FROM histograms
    )
SELECT
    exp,
    run,
    step,
    layer,
    point,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) - 0.5))) AS p50_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) - 0.5))) AS p99_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) - 0.5))) AS p999_log2,
    multiIf(elements = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_elements, NULL) AS exact_elements
FROM with_totals;

CREATE OR REPLACE VIEW metric_sampler.activation_massive_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_activation_massive
FINAL
WHERE (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}));

CREATE OR REPLACE VIEW metric_sampler.activation_massive_by_mb_range
AS WITH
    hist_sum AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            microbatch,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'token_rms'
        GROUP BY
            exp,
            run,
            step,
            layer,
            point,
            microbatch
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            microbatch,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS tokens,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_tokens
        FROM histograms
    ),
    with_median AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            microbatch,
            bins_start,
            bins_width,
            bins_count,
            hist,
            tokens,
            all_counts_exact,
            exact_tokens,
            multiIf(tokens = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) - 0.5))) AS median_log2_rms
        FROM with_totals
    )
SELECT
    exp,
    run,
    step,
    layer,
    point,
    microbatch,
    tokens,
    median_log2_rms,
    if(tokens = 0, NULL, arraySum(arrayMap((h, i) -> if((bins_start + (bins_width * (i - 0.5))) >= (median_log2_rms + 7), h, 0), hist, arrayEnumerate(hist))) / tokens) AS massive_share,
    multiIf(tokens = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2_rms,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_tokens, NULL) AS exact_tokens
FROM with_median;

CREATE OR REPLACE VIEW metric_sampler.activation_massive_by_rank_range
AS WITH
    hist_sum AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'token_rms'
        GROUP BY
            exp,
            run,
            step,
            layer,
            point,
            rank
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS tokens,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_tokens
        FROM histograms
    ),
    with_median AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            bins_start,
            bins_width,
            bins_count,
            hist,
            tokens,
            all_counts_exact,
            exact_tokens,
            multiIf(tokens = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) - 0.5))) AS median_log2_rms
        FROM with_totals
    )
SELECT
    exp,
    run,
    step,
    layer,
    point,
    rank,
    tokens,
    median_log2_rms,
    if(tokens = 0, NULL, arraySum(arrayMap((h, i) -> if((bins_start + (bins_width * (i - 0.5))) >= (median_log2_rms + 7), h, 0), hist, arrayEnumerate(hist))) / tokens) AS massive_share,
    multiIf(tokens = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2_rms,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_tokens, NULL) AS exact_tokens
FROM with_median;

CREATE OR REPLACE VIEW metric_sampler.activation_massive_fast
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_activation_massive
FINAL
WHERE ({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2) AND (({exp_filter:String} = '') OR (exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (run = {run_filter:String}))
UNION ALL
SELECT *
FROM `metric_sampler.activation_massive_tokens_selected`(exp_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2), '-', {exp_filter:String}), run_filter = if(({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2), '-', {run_filter:String}), rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32})
WHERE NOT (({rank_filter:Int32} = -2) AND ({microbatch_filter:Int32} = -2));

CREATE OR REPLACE VIEW metric_sampler.activation_massive_live
AS SELECT *
FROM `metric_sampler.activation_massive_tokens_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_massive_mb
AS SELECT *
FROM `metric_sampler.activation_massive_tokens_selected`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_massive_mbsum
AS SELECT * EXCEPT (refreshed_at, microbatch)
FROM metric_sampler.summary_activation_massive_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (microbatch = {microbatch_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_massive_ranksum
AS SELECT * EXCEPT (refreshed_at, rank)
FROM metric_sampler.summary_activation_massive_rank
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32});

CREATE OR REPLACE VIEW metric_sampler.activation_massive_tokens_selected_live
AS SELECT *
FROM `metric_sampler.activation_massive_tokens_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.activation_massive_tokens_selected_range
AS WITH
    hist_sum AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'token_rms'
        GROUP BY
            exp,
            run,
            step,
            layer,
            point
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS tokens,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_tokens
        FROM histograms
    ),
    with_median AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            bins_start,
            bins_width,
            bins_count,
            hist,
            tokens,
            all_counts_exact,
            exact_tokens,
            multiIf(tokens = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) - 0.5))) AS median_log2_rms
        FROM with_totals
    )
SELECT
    exp,
    run,
    step,
    layer,
    point,
    tokens,
    median_log2_rms,
    if(tokens = 0, NULL, arraySum(arrayMap((h, i) -> if((bins_start + (bins_width * (i - 0.5))) >= (median_log2_rms + 7), h, 0), hist, arrayEnumerate(hist))) / tokens) AS massive_share,
    multiIf(tokens = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2_rms,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_tokens, NULL) AS exact_tokens
FROM with_median;

CREATE OR REPLACE VIEW metric_sampler.activation_massive_by_rank_mb_range
AS WITH
    hist_sum AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'token_rms'
        GROUP BY
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS tokens,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_tokens
        FROM histograms
    ),
    with_median AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            bins_start,
            bins_width,
            bins_count,
            hist,
            tokens,
            all_counts_exact,
            exact_tokens,
            multiIf(tokens = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) - 0.5))) AS median_log2_rms
        FROM with_totals
    )
SELECT
    exp,
    run,
    step,
    layer,
    point,
    rank,
    microbatch,
    tokens,
    median_log2_rms,
    if(tokens = 0, NULL, arraySum(arrayMap((h, i) -> if((bins_start + (bins_width * (i - 0.5))) >= (median_log2_rms + 7), h, 0), hist, arrayEnumerate(hist))) / tokens) AS massive_share,
    multiIf(tokens = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2_rms,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_tokens, NULL) AS exact_tokens
FROM with_median;

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_by_rank_mb_range
AS WITH
    hist_sum AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'exp'
        GROUP BY
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS elements,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_elements
        FROM histograms
    )
SELECT
    exp,
    run,
    step,
    layer,
    point,
    rank,
    microbatch,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) - 0.5))) AS p50_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) - 0.5))) AS p99_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) - 0.5))) AS p999_log2,
    multiIf(elements = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_elements, NULL) AS exact_elements
FROM with_totals;

-- Partitioned by run like the raw tables, so an idle run's rows move to the cold tier and merges stay within a run.
CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_massive_rank_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `microbatch` Int16,
    `tokens` Float64,
    `median_log2_rms` Nullable(Float64),
    `massive_share` Nullable(Float64),
    `max_log2_rms` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_tokens` Nullable(UInt64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, rank, microbatch, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)', max_bytes_to_merge_at_max_space_in_pool = 4294967296;

CREATE OR REPLACE VIEW metric_sampler.activation_massive_rankmbsum
AS SELECT * EXCEPT (refreshed_at, rank, microbatch)
FROM metric_sampler.summary_activation_massive_rank_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32}) AND (microbatch = {microbatch_filter:Int32});

-- The four levels of the massive-token statistics (per rank and microbatch, per rank, per microbatch, of all ranks)
-- come from one read of the histograms, grouped by GROUPING SETS; level = grouping(rank, microbatch): 0 per rank and
-- microbatch, 1 per rank, 2 per microbatch, 3 all.
CREATE OR REPLACE VIEW metric_sampler.activation_massive_levels_range
AS WITH
    hist_sum AS
    (
        SELECT
            grouping(rank, microbatch) AS level,
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'token_rms'
        GROUP BY
            GROUPING SETS (
                (exp, run, step, layer, point, rank, microbatch),
                (exp, run, step, layer, point, rank),
                (exp, run, step, layer, point, microbatch),
                (exp, run, step, layer, point))
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            level,
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS tokens,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_tokens
        FROM histograms
    ),
    with_median AS
    (
        SELECT
            level,
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            bins_start,
            bins_width,
            bins_count,
            hist,
            tokens,
            all_counts_exact,
            exact_tokens,
            multiIf(tokens = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * tokens)), cum) - 0.5))) AS median_log2_rms
        FROM with_totals
    )
SELECT
    level,
    exp,
    run,
    step,
    layer,
    point,
    rank,
    microbatch,
    tokens,
    median_log2_rms,
    if(tokens = 0, NULL, arraySum(arrayMap((h, i) -> if((bins_start + (bins_width * (i - 0.5))) >= (median_log2_rms + 7), h, 0), hist, arrayEnumerate(hist))) / tokens) AS massive_share,
    multiIf(tokens = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2_rms,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_tokens, NULL) AS exact_tokens
FROM with_median;

-- The levels loop inserts here; the table keeps no rows, its views route each level to its own summary table.
CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_massive_levels
(
    `refreshed_at` DateTime64(3),
    `level` UInt8,
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `microbatch` Int16,
    `tokens` Float64,
    `median_log2_rms` Nullable(Float64),
    `massive_share` Nullable(Float64),
    `max_log2_rms` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_tokens` Nullable(UInt64)
)
ENGINE = Null;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.summary_activation_massive_rank_mb_mv TO metric_sampler.summary_activation_massive_rank_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `microbatch` Int16,
    `tokens` Float64,
    `median_log2_rms` Nullable(Float64),
    `massive_share` Nullable(Float64),
    `max_log2_rms` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_tokens` Nullable(UInt64)
)
AS SELECT * EXCEPT (level)
FROM metric_sampler.summary_activation_massive_levels
WHERE level = 0;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.summary_activation_massive_rank_mv TO metric_sampler.summary_activation_massive_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `tokens` Float64,
    `median_log2_rms` Nullable(Float64),
    `massive_share` Nullable(Float64),
    `max_log2_rms` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_tokens` Nullable(UInt64)
)
AS SELECT * EXCEPT (level, microbatch)
FROM metric_sampler.summary_activation_massive_levels
WHERE level = 1;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.summary_activation_massive_mb_mv TO metric_sampler.summary_activation_massive_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `microbatch` Int16,
    `tokens` Float64,
    `median_log2_rms` Nullable(Float64),
    `massive_share` Nullable(Float64),
    `max_log2_rms` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_tokens` Nullable(UInt64)
)
AS SELECT * EXCEPT (level, rank)
FROM metric_sampler.summary_activation_massive_levels
WHERE level = 2;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.summary_activation_massive_mv TO metric_sampler.summary_activation_massive
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `tokens` Float64,
    `median_log2_rms` Nullable(Float64),
    `massive_share` Nullable(Float64),
    `max_log2_rms` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_tokens` Nullable(UInt64)
)
AS SELECT * EXCEPT (level, rank, microbatch)
FROM metric_sampler.summary_activation_massive_levels
WHERE level = 3;

CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_exp_quantiles_rank_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `microbatch` Int16,
    `p50_log2` Nullable(Float64),
    `p99_log2` Nullable(Float64),
    `p999_log2` Nullable(Float64),
    `max_log2` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
PARTITION BY (exp, run)
ORDER BY (exp, run, rank, microbatch, step, layer, point)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)', max_bytes_to_merge_at_max_space_in_pool = 4294967296;

CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_rankmbsum
AS SELECT * EXCEPT (refreshed_at, rank, microbatch)
FROM metric_sampler.summary_activation_exp_quantiles_rank_mb
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND (rank = {rank_filter:Int32}) AND (microbatch = {microbatch_filter:Int32});

-- The four levels of the exact activation quantiles (per rank and microbatch, per rank, per microbatch, of all ranks)
-- come from one read of the histograms, grouped by GROUPING SETS; level = grouping(rank, microbatch): 0 per rank and
-- microbatch, 1 per rank, 2 per microbatch, 3 all.
CREATE OR REPLACE VIEW metric_sampler.activation_exp_quantiles_levels_range
AS WITH
    hist_sum AS
    (
        SELECT
            grouping(rank, microbatch) AS level,
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            any(bins_start) AS bins_start,
            any(bins_width) AS bins_width,
            any(bins_count) AS bins_count,
            sumForEach(exact_hist) AS exact_hist,
            sumForEach(legacy_hist) AS legacy_hist,
            min(counts_exact) AS all_counts_exact
        FROM `metric_sampler.activation_histograms_selected_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32})
        WHERE histogram = 'exp'
        GROUP BY
            GROUPING SETS (
                (exp, run, step, layer, point, rank, microbatch),
                (exp, run, step, layer, point, rank),
                (exp, run, step, layer, point, microbatch),
                (exp, run, step, layer, point))
    ),
    histograms AS
    (
        SELECT
            *,
            arrayMap((exact, legacy) -> (exact + legacy), arrayResize(exact_hist, bins_count), arrayResize(legacy_hist, bins_count)) AS hist
        FROM hist_sum
    ),
    with_totals AS
    (
        SELECT
            level,
            exp,
            run,
            step,
            layer,
            point,
            rank,
            microbatch,
            bins_start,
            bins_width,
            bins_count,
            hist,
            arraySum(hist) AS elements,
            arrayCumSum(hist) AS cum,
            all_counts_exact,
            arraySum(exact_hist) AS exact_elements
        FROM histograms
    )
SELECT
    level,
    exp,
    run,
    step,
    layer,
    point,
    rank,
    microbatch,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.5 * elements)), cum) - 0.5))) AS p50_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.99 * elements)), cum) - 0.5))) AS p99_log2,
    multiIf(elements = 0, NULL, (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) IN (0, bins_count)), bins_start + (bins_width * (bins_count - 1)), bins_start + (bins_width * (arrayFirstIndex(x -> (x >= (0.999 * elements)), cum) - 0.5))) AS p999_log2,
    multiIf(elements = 0, NULL, bins_start + (bins_width * (arrayLastIndex(x -> (x > 0), hist) - 1))) AS max_log2,
    all_counts_exact,
    multiIf(all_counts_exact = 1, exact_elements, NULL) AS exact_elements
FROM with_totals;

-- The levels loop inserts here; the table keeps no rows, its views route each level to its own summary table.
CREATE TABLE IF NOT EXISTS metric_sampler.summary_activation_exp_quantiles_levels
(
    `refreshed_at` DateTime64(3),
    `level` UInt8,
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `microbatch` Int16,
    `p50_log2` Nullable(Float64),
    `p99_log2` Nullable(Float64),
    `p999_log2` Nullable(Float64),
    `max_log2` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64)
)
ENGINE = Null;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.summary_activation_exp_quantiles_rank_mb_mv TO metric_sampler.summary_activation_exp_quantiles_rank_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `microbatch` Int16,
    `p50_log2` Nullable(Float64),
    `p99_log2` Nullable(Float64),
    `p999_log2` Nullable(Float64),
    `max_log2` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64)
)
AS SELECT * EXCEPT (level)
FROM metric_sampler.summary_activation_exp_quantiles_levels
WHERE level = 0;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.summary_activation_exp_quantiles_rank_mv TO metric_sampler.summary_activation_exp_quantiles_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `rank` Int16,
    `p50_log2` Nullable(Float64),
    `p99_log2` Nullable(Float64),
    `p999_log2` Nullable(Float64),
    `max_log2` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64)
)
AS SELECT * EXCEPT (level, microbatch)
FROM metric_sampler.summary_activation_exp_quantiles_levels
WHERE level = 1;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.summary_activation_exp_quantiles_mb_mv TO metric_sampler.summary_activation_exp_quantiles_mb
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `microbatch` Int16,
    `p50_log2` Nullable(Float64),
    `p99_log2` Nullable(Float64),
    `p999_log2` Nullable(Float64),
    `max_log2` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64)
)
AS SELECT * EXCEPT (level, rank)
FROM metric_sampler.summary_activation_exp_quantiles_levels
WHERE level = 2;

CREATE MATERIALIZED VIEW IF NOT EXISTS metric_sampler.summary_activation_exp_quantiles_mv TO metric_sampler.summary_activation_exp_quantiles
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `point` LowCardinality(String),
    `p50_log2` Nullable(Float64),
    `p99_log2` Nullable(Float64),
    `p999_log2` Nullable(Float64),
    `max_log2` Nullable(Float64),
    `all_counts_exact` UInt8,
    `exact_elements` Nullable(UInt64)
)
AS SELECT * EXCEPT (level, rank, microbatch)
FROM metric_sampler.summary_activation_exp_quantiles_levels
WHERE level = 3;
