from __future__ import annotations

LOSS_PER_STEP = """
SELECT exp, run, step, loss_mean, loss_min, loss_max
FROM metric_sampler.loss_per_step
WHERE exp = '{exp}' AND run = '{run}'
ORDER BY step
""".strip()

LAYER_LOAD_HEATMAP = """
SELECT exp, run, step, layer, expert_index, total_tokens
FROM metric_sampler.expert_totals_per_step
WHERE exp = '{exp}' AND run = '{run}' AND layer = {layer} AND step BETWEEN {step_from} AND {step_to}
ORDER BY step, expert_index
""".strip()

STEP_DRILLDOWN = """
SELECT mn.name AS metric_name, s.node, s.rank, s.microbatch, s.layer, s.attempt, s.values
FROM metric_sampler.scalars AS s FINAL
INNER JOIN metric_sampler.latest_attempt_step AS las
    ON s.exp = las.exp AND s.run = las.run AND s.step = las.step AND s.attempt = las.attempt
INNER JOIN metric_sampler.metric_names AS mn
    ON mn.exp = s.exp AND mn.run = s.run AND mn.attempt = s.attempt AND mn.metric_id = s.metric_id
WHERE s.exp = '{exp}' AND s.run = '{run}' AND s.step = {step}
ORDER BY metric_name, s.node, s.rank, s.microbatch
""".strip()

HEALTH = """
SELECT exp, run, attempt, step, nodes_present, nodes_expected,
       contributors_total, expected_contributors_total, identical_violations
FROM metric_sampler.health_per_step
WHERE exp = '{exp}' AND run = '{run}'
ORDER BY step
""".strip()

LOSS_QUANTILES = """
SELECT exp, run, step, segment, tokens, mean, p50, p90, p99, p999
FROM metric_sampler.loss_quantiles_per_step
WHERE exp = '{exp}' AND run = '{run}'
ORDER BY segment, step
""".strip()

STREAM_LOSS = """
SELECT exp, run, step, stream_id, stream_name, tokens, samples, mean, std
FROM metric_sampler.stream_loss_per_step
WHERE exp = '{exp}' AND run = '{run}'
ORDER BY stream_id, step
""".strip()

EXPERT_LOSS = """
SELECT exp, run, step, layer, expert, tokens, mean, std, hardness
FROM metric_sampler.expert_loss_per_step
WHERE exp = '{exp}' AND run = '{run}' AND layer = {layer}
ORDER BY step, expert
""".strip()

GRAD_PER_RANK = """
SELECT exp, run, step, node, rank, dense_norm, expert_norm, total_norm, robust_z, clip_total_norm
FROM metric_sampler.grad_per_rank
WHERE exp = '{exp}' AND run = '{run}'
ORDER BY step, node, rank
""".strip()

ACTIVATION_PER_STEP = """
SELECT exp, run, step, layer, point, elements, mean, rms, std, nonfinite,
       amax_max, amax_median, bf16_headroom_log2, fp8_e4m3_headroom_log2
FROM metric_sampler.activation_per_step
WHERE exp = '{exp}' AND run = '{run}'
ORDER BY point, layer, step
""".strip()

ACTIVATION_EXP_QUANTILES = """
SELECT exp, run, step, layer, point, p50_log2, p99_log2, p999_log2, max_log2
FROM metric_sampler.activation_exp_quantiles
WHERE exp = '{exp}' AND run = '{run}' AND layer = {layer} AND point = '{point}'
ORDER BY step
""".strip()

ACTIVATION_MASSIVE_TOKENS = """
SELECT exp, run, step, layer, point, tokens, median_log2_rms, massive_share, max_log2_rms
FROM metric_sampler.activation_massive_tokens
WHERE exp = '{exp}' AND run = '{run}'
ORDER BY point, layer, step
""".strip()
