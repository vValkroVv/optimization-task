-- Attention summaries (loops attention_softmax, attention_linear, attention_rank and attention_entropy_hist) hold every
-- rank and microbatch together, per layer and head. Head -1 is the layer as a whole: the mean of the head means, the
-- minimum of the minima, the maximum of the maxima and the pooled histogram of all heads. rows counts the rows of the
-- LSE (gate) statistics, stat_rows those of the row statistics, which skip rows with fewer than 32 keys, and documents
-- those of the state statistics. The *_live views take the panel's step range and layer because the range views cannot
-- push those filters into their scan (layer -1: all layers).
CREATE TABLE IF NOT EXISTS metric_sampler.summary_attention_softmax
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `head` Int16,
    `rows` UInt64,
    `stat_rows` UInt64,
    `lse_mean` Nullable(Float64),
    `lse_min` Nullable(Float32),
    `lse_max` Nullable(Float32),
    `q_nope_norm_mean` Nullable(Float64),
    `q_rope_norm_mean` Nullable(Float64),
    `k_nope_norm_mean` Nullable(Float64),
    `k_rope_norm_mean` Nullable(Float64),
    `v_norm_mean` Nullable(Float64),
    `q_amax` Nullable(Float32),
    `k_amax` Nullable(Float32),
    `v_amax` Nullable(Float32),
    `gate_mean` Nullable(Float64),
    `gate_min` Nullable(Float32),
    `gate_max` Nullable(Float32),
    `ent_mean` Nullable(Float64),
    `ent_std` Nullable(Float64),
    `ent_min` Nullable(Float32),
    `ent_max` Nullable(Float32),
    `entn_mean` Nullable(Float64),
    `entn_min` Nullable(Float32),
    `entn_max` Nullable(Float32),
    `pmax_mean` Nullable(Float64),
    `pmax_min` Nullable(Float32),
    `pmax_max` Nullable(Float32),
    `pmin_mean` Nullable(Float64),
    `lpmin_mean` Nullable(Float64),
    `lpmin_min` Nullable(Float32),
    `lpmin_max` Nullable(Float32),
    `mlogit_mean` Nullable(Float64),
    `mlogit_max` Nullable(Float32),
    `coll_mean` Nullable(Float64),
    `sink_mean` Nullable(Float64),
    `sink_max` Nullable(Float32),
    `dist_mean` Nullable(Float64),
    `local_mean` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, head, layer, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.attention_softmax_range
AS WITH
    merged AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            rv.layer AS layer,
            sumForEachIf(rv.`values`, mn.name = 'attn/softmax/lse_stats') AS lse_sums,
            minForEachIf(rv.`values`, mn.name = 'attn/softmax/lse_stats') AS lse_mins,
            maxForEachIf(rv.`values`, mn.name = 'attn/softmax/lse_stats') AS lse_maxs,
            sumForEachIf(rv.`values`, mn.name = 'attn/softmax/qkv_stats') AS qkv_sums,
            maxForEachIf(rv.`values`, mn.name = 'attn/softmax/qkv_stats') AS qkv_maxs,
            sumForEachIf(rv.`values`, mn.name = 'attn/softmax/gate_stats') AS gate_sums,
            minForEachIf(rv.`values`, mn.name = 'attn/softmax/gate_stats') AS gate_mins,
            maxForEachIf(rv.`values`, mn.name = 'attn/softmax/gate_stats') AS gate_maxs,
            sumForEachIf(rv.`values`, mn.name = 'attn/softmax/row_stats') AS row_sums,
            minForEachIf(rv.`values`, mn.name = 'attn/softmax/row_stats') AS row_mins,
            maxForEachIf(rv.`values`, mn.name = 'attn/softmax/row_stats') AS row_maxs,
            greatest(intDiv(length(lse_sums) - 1, 3), intDiv(length(qkv_sums) - 1, 8), intDiv(length(gate_sums) - 1, 3), intDiv(length(row_sums) - 1, 21)) AS heads
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND (rv.layer <= 255) AND (({layer_filter:Int32} = -1) OR (rv.layer = {layer_filter:Int32})) AND ((rv.`values`[1]) > 0) AND (((length(rv.`values`) - 1) % multiIf(mn.name = 'attn/softmax/lse_stats', 3, mn.name = 'attn/softmax/qkv_stats', 8, mn.name = 'attn/softmax/gate_stats', 3, 21)) = 0) AND (mn.name IN ('attn/softmax/lse_stats', 'attn/softmax/qkv_stats', 'attn/softmax/gate_stats', 'attn/softmax/row_stats')) AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('attn/softmax/lse_stats', 'attn/softmax/qkv_stats', 'attn/softmax/gate_stats', 'attn/softmax/row_stats')
        ))
        GROUP BY
            rv.exp,
            rv.run,
            rv.step,
            rv.layer
    ),
    per_head AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            h,
            toUInt64(lse_sums[1]) AS rows,
            toUInt64(row_sums[1]) AS stat_rows,
            arrayElementOrNull(lse_sums, 2 + h) / nullIf(lse_sums[1], 0) AS lse_mean,
            arrayElementOrNull(lse_mins, (2 + (1 * heads)) + h) AS lse_min,
            arrayElementOrNull(lse_maxs, (2 + (2 * heads)) + h) AS lse_max,
            arrayElementOrNull(qkv_sums, 2 + h) / nullIf(qkv_sums[1], 0) AS q_nope_norm_mean,
            arrayElementOrNull(qkv_sums, (2 + (1 * heads)) + h) / nullIf(qkv_sums[1], 0) AS q_rope_norm_mean,
            arrayElementOrNull(qkv_sums, (2 + (2 * heads)) + h) / nullIf(qkv_sums[1], 0) AS k_nope_norm_mean,
            arrayElementOrNull(qkv_sums, (2 + (3 * heads)) + h) / nullIf(qkv_sums[1], 0) AS k_rope_norm_mean,
            arrayElementOrNull(qkv_sums, (2 + (4 * heads)) + h) / nullIf(qkv_sums[1], 0) AS v_norm_mean,
            arrayElementOrNull(qkv_maxs, (2 + (5 * heads)) + h) AS q_amax,
            arrayElementOrNull(qkv_maxs, (2 + (6 * heads)) + h) AS k_amax,
            arrayElementOrNull(qkv_maxs, (2 + (7 * heads)) + h) AS v_amax,
            arrayElementOrNull(gate_sums, 2 + h) / nullIf(gate_sums[1], 0) AS gate_mean,
            arrayElementOrNull(gate_mins, (2 + (1 * heads)) + h) AS gate_min,
            arrayElementOrNull(gate_maxs, (2 + (2 * heads)) + h) AS gate_max,
            arrayElementOrNull(row_sums, 2 + h) / nullIf(row_sums[1], 0) AS ent_mean,
            arrayElementOrNull(row_sums, (2 + (1 * heads)) + h) / nullIf(row_sums[1], 0) AS ent_sq_mean,
            arrayElementOrNull(row_mins, (2 + (2 * heads)) + h) AS ent_min,
            arrayElementOrNull(row_maxs, (2 + (3 * heads)) + h) AS ent_max,
            arrayElementOrNull(row_sums, (2 + (4 * heads)) + h) / nullIf(row_sums[1], 0) AS entn_mean,
            arrayElementOrNull(row_mins, (2 + (5 * heads)) + h) AS entn_min,
            arrayElementOrNull(row_maxs, (2 + (6 * heads)) + h) AS entn_max,
            arrayElementOrNull(row_sums, (2 + (7 * heads)) + h) / nullIf(row_sums[1], 0) AS pmax_mean,
            arrayElementOrNull(row_mins, (2 + (8 * heads)) + h) AS pmax_min,
            arrayElementOrNull(row_maxs, (2 + (9 * heads)) + h) AS pmax_max,
            arrayElementOrNull(row_sums, (2 + (10 * heads)) + h) / nullIf(row_sums[1], 0) AS pmin_mean,
            arrayElementOrNull(row_sums, (2 + (11 * heads)) + h) / nullIf(row_sums[1], 0) AS lpmin_mean,
            arrayElementOrNull(row_mins, (2 + (12 * heads)) + h) AS lpmin_min,
            arrayElementOrNull(row_maxs, (2 + (13 * heads)) + h) AS lpmin_max,
            arrayElementOrNull(row_sums, (2 + (14 * heads)) + h) / nullIf(row_sums[1], 0) AS mlogit_mean,
            arrayElementOrNull(row_maxs, (2 + (15 * heads)) + h) AS mlogit_max,
            arrayElementOrNull(row_sums, (2 + (16 * heads)) + h) / nullIf(row_sums[1], 0) AS coll_mean,
            arrayElementOrNull(row_sums, (2 + (17 * heads)) + h) / nullIf(row_sums[1], 0) AS sink_mean,
            arrayElementOrNull(row_maxs, (2 + (18 * heads)) + h) AS sink_max,
            arrayElementOrNull(row_sums, (2 + (19 * heads)) + h) / nullIf(row_sums[1], 0) AS dist_mean,
            arrayElementOrNull(row_sums, (2 + (20 * heads)) + h) / nullIf(row_sums[1], 0) AS local_mean
        FROM merged
        ARRAY JOIN range(toUInt32(heads)) AS h
    )
SELECT
    exp,
    run,
    step,
    layer,
    if(grouping(h) = 1, toInt16(-1), toInt16(h)) AS head,
    any(rows) AS rows,
    any(stat_rows) AS stat_rows,
    avg(lse_mean) AS lse_mean,
    min(lse_min) AS lse_min,
    max(lse_max) AS lse_max,
    avg(q_nope_norm_mean) AS q_nope_norm_mean,
    avg(q_rope_norm_mean) AS q_rope_norm_mean,
    avg(k_nope_norm_mean) AS k_nope_norm_mean,
    avg(k_rope_norm_mean) AS k_rope_norm_mean,
    avg(v_norm_mean) AS v_norm_mean,
    max(q_amax) AS q_amax,
    max(k_amax) AS k_amax,
    max(v_amax) AS v_amax,
    avg(gate_mean) AS gate_mean,
    min(gate_min) AS gate_min,
    max(gate_max) AS gate_max,
    avg(ent_mean) AS ent_mean,
    if(count(per_head.ent_mean) = 0, NULL, sqrt(greatest(avg(per_head.ent_sq_mean) - (avg(per_head.ent_mean) * avg(per_head.ent_mean)), 0))) AS ent_std,
    min(ent_min) AS ent_min,
    max(ent_max) AS ent_max,
    avg(entn_mean) AS entn_mean,
    min(entn_min) AS entn_min,
    max(entn_max) AS entn_max,
    avg(pmax_mean) AS pmax_mean,
    min(pmax_min) AS pmax_min,
    max(pmax_max) AS pmax_max,
    avg(pmin_mean) AS pmin_mean,
    avg(lpmin_mean) AS lpmin_mean,
    min(lpmin_min) AS lpmin_min,
    max(lpmin_max) AS lpmin_max,
    avg(mlogit_mean) AS mlogit_mean,
    max(mlogit_max) AS mlogit_max,
    avg(coll_mean) AS coll_mean,
    avg(sink_mean) AS sink_mean,
    max(sink_max) AS sink_max,
    avg(dist_mean) AS dist_mean,
    avg(local_mean) AS local_mean
FROM per_head
GROUP BY
    GROUPING SETS (
        (exp, run, step, layer, h),
        (exp, run, step, layer));

CREATE OR REPLACE VIEW metric_sampler.attention_softmax_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_attention_softmax
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.attention_softmax_live
AS SELECT *
FROM `metric_sampler.attention_softmax_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, layer_filter = {layer_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32});

CREATE TABLE IF NOT EXISTS metric_sampler.summary_attention_linear
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `head` Int16,
    `rows` UInt64,
    `documents` UInt64,
    `beta_mean` Nullable(Float64),
    `beta_min` Nullable(Float32),
    `beta_max` Nullable(Float32),
    `alpha_mean` Nullable(Float64),
    `alpha_min` Nullable(Float32),
    `alpha_max` Nullable(Float32),
    `g_mean` Nullable(Float64),
    `v_norm_mean` Nullable(Float64),
    `o_norm_mean` Nullable(Float64),
    `o_amax` Nullable(Float32),
    `state_norm_mean` Nullable(Float64),
    `state_norm_max` Nullable(Float32),
    `decay_a` Nullable(Float64),
    `dt_bias` Nullable(Float64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, head, layer, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.attention_linear_range
AS WITH
    merged AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            rv.layer AS layer,
            sumForEachIf(rv.`values`, mn.name = 'attn/linear/gate_stats') AS gate_sums,
            minForEachIf(rv.`values`, mn.name = 'attn/linear/gate_stats') AS gate_mins,
            maxForEachIf(rv.`values`, mn.name = 'attn/linear/gate_stats') AS gate_maxs,
            sumForEachIf(rv.`values`, mn.name = 'attn/linear/norm_stats') AS norm_sums,
            maxForEachIf(rv.`values`, mn.name = 'attn/linear/norm_stats') AS norm_maxs,
            sumForEachIf(rv.`values`, mn.name = 'attn/linear/state_stats') AS state_sums,
            maxForEachIf(rv.`values`, mn.name = 'attn/linear/state_stats') AS state_maxs,
            anyIf(rv.`values`, mn.name = 'attn/linear/decay_params') AS decay,
            greatest(intDiv(length(gate_sums) - 1, 7), intDiv(length(norm_sums) - 1, 3), intDiv(length(state_sums) - 1, 2), intDiv(toInt64(length(decay)), 2)) AS heads
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (rv.layer <= 255) AND (({layer_filter:Int32} = -1) OR (rv.layer = {layer_filter:Int32})) AND (mn.name IN ('attn/linear/gate_stats', 'attn/linear/norm_stats', 'attn/linear/state_stats', 'attn/linear/decay_params')) AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('attn/linear/gate_stats', 'attn/linear/norm_stats', 'attn/linear/state_stats', 'attn/linear/decay_params')
        )) AND ((mn.name = 'attn/linear/decay_params') OR ((({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND ((rv.`values`[1]) > 0) AND (((length(rv.`values`) - 1) % multiIf(mn.name = 'attn/linear/gate_stats', 7, mn.name = 'attn/linear/norm_stats', 3, 2)) = 0)))
        GROUP BY
            rv.exp,
            rv.run,
            rv.step,
            rv.layer
    ),
    per_head AS
    (
        SELECT
            exp,
            run,
            step,
            layer,
            h,
            toUInt64(gate_sums[1]) AS rows,
            toUInt64(state_sums[1]) AS documents,
            arrayElementOrNull(gate_sums, 2 + h) / nullIf(gate_sums[1], 0) AS beta_mean,
            arrayElementOrNull(gate_mins, (2 + (1 * heads)) + h) AS beta_min,
            arrayElementOrNull(gate_maxs, (2 + (2 * heads)) + h) AS beta_max,
            arrayElementOrNull(gate_sums, (2 + (3 * heads)) + h) / nullIf(gate_sums[1], 0) AS alpha_mean,
            arrayElementOrNull(gate_mins, (2 + (4 * heads)) + h) AS alpha_min,
            arrayElementOrNull(gate_maxs, (2 + (5 * heads)) + h) AS alpha_max,
            arrayElementOrNull(gate_sums, (2 + (6 * heads)) + h) / nullIf(gate_sums[1], 0) AS g_mean,
            arrayElementOrNull(norm_sums, 2 + h) / nullIf(norm_sums[1], 0) AS v_norm_mean,
            arrayElementOrNull(norm_sums, (2 + (1 * heads)) + h) / nullIf(norm_sums[1], 0) AS o_norm_mean,
            arrayElementOrNull(norm_maxs, (2 + (2 * heads)) + h) AS o_amax,
            arrayElementOrNull(state_sums, 2 + h) / nullIf(state_sums[1], 0) AS state_norm_mean,
            arrayElementOrNull(state_maxs, (2 + (1 * heads)) + h) AS state_norm_max,
            arrayElementOrNull(decay, 1 + h) AS decay_a,
            arrayElementOrNull(decay, (1 + heads) + h) AS dt_bias
        FROM merged
        ARRAY JOIN range(toUInt32(heads)) AS h
    )
SELECT
    exp,
    run,
    step,
    layer,
    if(grouping(h) = 1, toInt16(-1), toInt16(h)) AS head,
    any(rows) AS rows,
    any(documents) AS documents,
    avg(beta_mean) AS beta_mean,
    min(beta_min) AS beta_min,
    max(beta_max) AS beta_max,
    avg(alpha_mean) AS alpha_mean,
    min(alpha_min) AS alpha_min,
    max(alpha_max) AS alpha_max,
    avg(g_mean) AS g_mean,
    avg(v_norm_mean) AS v_norm_mean,
    avg(o_norm_mean) AS o_norm_mean,
    max(o_amax) AS o_amax,
    avg(state_norm_mean) AS state_norm_mean,
    max(state_norm_max) AS state_norm_max,
    avg(decay_a) AS decay_a,
    avg(dt_bias) AS dt_bias
FROM per_head
GROUP BY
    GROUPING SETS (
        (exp, run, step, layer, h),
        (exp, run, step, layer));

CREATE OR REPLACE VIEW metric_sampler.attention_linear_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_attention_linear
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.attention_linear_live
AS SELECT *
FROM `metric_sampler.attention_linear_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, layer_filter = {layer_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32});

CREATE TABLE IF NOT EXISTS metric_sampler.summary_attention_rank
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `rank` Int16,
    `stat_rows` UInt64,
    `ent_mean` Float64,
    `entn_mean` Float64
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, layer, step, rank)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.attention_rank_range
AS WITH per_row AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.step AS step,
            rv.layer AS layer,
            rv.rank AS rank,
            rv.`values`[1] AS rows,
            intDiv(length(rv.`values`) - 1, 21) AS heads,
            arraySum(arraySlice(rv.`values`, 2, heads)) AS ent_total,
            arraySum(arraySlice(rv.`values`, 2 + (4 * heads), heads)) AS entn_total
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (rv.exp = las.exp) AND (rv.run = las.run) AND (rv.step = las.step) AND (rv.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.exp = {exp_filter:String}) AND (rv.run = {run_filter:String}) AND (rv.step >= {step_from:UInt32}) AND (rv.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32})) AND (rv.layer <= 255) AND (({layer_filter:Int32} = -1) OR (rv.layer = {layer_filter:Int32})) AND ((rv.`values`[1]) > 0) AND (((length(rv.`values`) - 1) % 21) = 0) AND (mn.name = 'attn/softmax/row_stats') AND (rv.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'attn/softmax/row_stats'
        ))
    )
SELECT
    exp,
    run,
    step,
    layer,
    rank,
    toUInt64(sum(rows)) AS stat_rows,
    sum(ent_total) / (sum(rows) * any(heads)) AS ent_mean,
    sum(entn_total) / (sum(rows) * any(heads)) AS entn_mean
FROM per_row
GROUP BY
    exp,
    run,
    step,
    layer,
    rank;

CREATE OR REPLACE VIEW metric_sampler.attention_rank_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_attention_rank
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.attention_rank_live
AS SELECT *
FROM `metric_sampler.attention_rank_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, layer_filter = {layer_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32});

CREATE TABLE IF NOT EXISTS metric_sampler.summary_attention_entropy_hist
(
    `refreshed_at` DateTime64(3),
    `exp` LowCardinality(String),
    `run` LowCardinality(String),
    `step` UInt32,
    `layer` Int16,
    `head` Int16,
    `counts` Array(UInt64)
)
ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (exp, run, head, layer, step)
SETTINGS index_granularity = 8192, default_compression_codec = 'ZSTD(3)';

CREATE OR REPLACE VIEW metric_sampler.attention_entropy_hist_range
AS WITH merged AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.step AS step,
            ec.layer AS layer,
            mn.name AS metric,
            sumForEach(ec.`values`) AS summed
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (ec.layer <= 255) AND (({layer_filter:Int32} = -1) OR (ec.layer = {layer_filter:Int32})) AND (mn.name IN ('attn/softmax/entropy_hist', 'attn/softmax/head_entropy_hist')) AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name IN ('attn/softmax/entropy_hist', 'attn/softmax/head_entropy_hist')
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.step,
            ec.layer,
            mn.name
    )
SELECT
    exp,
    run,
    step,
    layer,
    toInt16(-1) AS head,
    summed AS counts
FROM merged
WHERE metric = 'attn/softmax/entropy_hist'
UNION ALL
SELECT
    exp,
    run,
    step,
    layer,
    toInt16(h) AS head,
    arraySlice(summed, (32 * h) + 1, 32) AS counts
FROM merged
ARRAY JOIN range(toUInt32(intDiv(length(summed), 32))) AS h
WHERE metric = 'attn/softmax/head_entropy_hist';

CREATE OR REPLACE VIEW metric_sampler.attention_entropy_hist_all
AS SELECT * EXCEPT refreshed_at
FROM metric_sampler.summary_attention_entropy_hist
FINAL
WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.attention_entropy_hist_live
AS SELECT *
FROM `metric_sampler.attention_entropy_hist_range`(exp_filter = {exp_filter:String}, run_filter = {run_filter:String}, rank_filter = {rank_filter:Int32}, microbatch_filter = {microbatch_filter:Int32}, layer_filter = {layer_filter:Int32}, step_from = {step_from:UInt32}, step_to = {step_to:UInt32});
