-- Stream-attributed router assignments. The source metric is captured before validity masking,
-- dispatch and capacity handling, and its dense [expert][stream] layout includes zero-count pairs.
CREATE OR REPLACE VIEW metric_sampler.router_stream_load_range
AS WITH
    stream_totals AS
    (
        SELECT
            s.exp AS exp,
            s.run AS run,
            s.attempt AS attempt,
            count() AS num_streams
        FROM metric_sampler.streams AS s FINAL
        WHERE (s.exp = {exp_filter:String}) AND (s.run = {run_filter:String})
        GROUP BY
            s.exp,
            s.run,
            s.attempt
    ),
    counts AS
    (
        SELECT
            ec.exp AS exp,
            ec.run AS run,
            ec.attempt AS attempt,
            ec.step AS step,
            ec.layer AS layer,
            ec.rank AS rank,
            ec.microbatch AS microbatch,
            sumForEach(ec.`values`) AS `values`
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.latest_attempt_step AS las ON (ec.exp = las.exp) AND (ec.run = las.run) AND (ec.step = las.step) AND (ec.attempt = las.attempt)
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.exp = {exp_filter:String}) AND (ec.run = {run_filter:String}) AND (ec.step >= {step_from:UInt32}) AND (ec.step <= {step_to:UInt32}) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32})) AND (ec.layer <= 255) AND (ec.rank >= 0) AND (ec.microbatch >= 0) AND (ec.source_metric_id = 0) AND (mn.name = 'router/stream_tokens_per_expert') AND (ec.metric_id IN (
            SELECT metric_id
            FROM metric_sampler.metrics
            WHERE name = 'router/stream_tokens_per_expert'
        ))
        GROUP BY
            ec.exp,
            ec.run,
            ec.attempt,
            ec.step,
            ec.layer,
            ec.rank,
            ec.microbatch
    ),
    expanded AS
    (
        SELECT
            counts.exp AS exp,
            counts.run AS run,
            counts.attempt AS attempt,
            counts.step AS step,
            counts.layer AS layer,
            counts.rank AS rank,
            counts.microbatch AS microbatch,
            intDiv(idx - 1, stream_totals.num_streams) + 1 AS expert,
            (idx - 1) % stream_totals.num_streams AS stream_id,
            token_count AS tokens
        FROM counts
        INNER JOIN stream_totals ON (stream_totals.exp = counts.exp) AND (stream_totals.run = counts.run) AND (stream_totals.attempt = counts.attempt)
        ARRAY JOIN
            arrayEnumerate(counts.`values`) AS idx,
            counts.`values` AS token_count
    ),
    totals AS
    (
        SELECT
            *,
            sum(tokens) OVER (PARTITION BY exp, run, step, layer, rank, microbatch, stream_id) AS stream_tokens,
            sum(tokens) OVER (PARTITION BY exp, run, step, layer, rank, microbatch, expert) AS expert_tokens,
            sum(tokens) OVER (PARTITION BY exp, run, step, layer, rank, microbatch) AS layer_tokens,
            avg(tokens) OVER (PARTITION BY exp, run, step, layer, rank, microbatch, stream_id) AS stream_mean_tokens,
            max(tokens) OVER (PARTITION BY exp, run, step, layer, rank, microbatch, stream_id) AS stream_max_tokens
        FROM expanded
    )
SELECT
    totals.exp AS exp,
    totals.run AS run,
    totals.attempt AS attempt,
    totals.step AS step,
    totals.layer AS layer,
    totals.rank AS rank,
    totals.microbatch AS microbatch,
    totals.expert AS expert,
    totals.stream_id AS stream_id,
    coalesce(stream_names.name, concat('stream', toString(totals.stream_id))) AS stream_name,
    totals.tokens AS tokens,
    totals.stream_tokens AS stream_tokens,
    totals.expert_tokens AS expert_tokens,
    totals.layer_tokens AS layer_tokens,
    totals.tokens / nullIf(totals.stream_tokens, 0) AS expert_share_within_stream,
    totals.tokens / nullIf(totals.expert_tokens, 0) AS stream_mix_within_expert,
    totals.stream_tokens / nullIf(totals.layer_tokens, 0) AS stream_share,
    (totals.stream_max_tokens - totals.stream_mean_tokens) / nullIf(totals.stream_mean_tokens, 0) AS stream_max_vio
FROM totals
LEFT JOIN metric_sampler.streams AS stream_names FINAL
    ON (stream_names.exp = totals.exp) AND (stream_names.run = totals.run) AND (stream_names.attempt = totals.attempt) AND (stream_names.stream_id = totals.stream_id);
