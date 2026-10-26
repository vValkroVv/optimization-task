CREATE OR REPLACE VIEW metric_sampler.exact_quantiles_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.exact_quantiles_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.exact_quantiles_per_rank_step
AS WITH
    quantiles AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.attempt AS attempt,
            rv.step AS step,
            rv.node AS node,
            rv.rank AS rank,
            rv.microbatch AS microbatch,
            rv.layer AS layer,
            mn.name AS metric,
            arrayMap(x -> toFloat64(x), rv.`values`) AS quantiles,
            1 AS present,
            length(rv.`values`) = 9 AS width_valid
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.layer <= 255) AND (rv.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (rv.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (rv.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32}) OR (rv.microbatch = -1)) AND (rv.rank >= 0) AND (rv.source_metric_id = 0) AND (((rv.metric_id >= 114) AND (rv.metric_id <= 137)) OR (rv.metric_id = 177)) AND match(mn.name, '(/quantiles|_quantiles)$')
        UNION ALL
        SELECT
            ec.exp,
            ec.run,
            ec.attempt,
            ec.step,
            ec.node,
            ec.rank,
            ec.microbatch,
            ec.layer,
            mn.name AS metric,
            arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[(2 * i) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * i) + 2]), 32))), range(9)) AS quantiles,
            1 AS present,
            length(ec.`values`) = 18 AS width_valid
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND (ec.rank >= 0) AND (ec.source_metric_id = 0) AND (ec.metric_id = 138) AND (mn.name = 'router/entropy_quantiles')
    ),
    counts AS
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
            replaceRegexpOne(mn.name, '_counts$', 's') AS metric,
            ec.`values` AS counts,
            1 AS present
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND (ec.rank >= 0) AND (ec.source_metric_id = 0) AND (((ec.metric_id >= 115) AND (ec.metric_id <= 139)) OR (ec.metric_id = 178)) AND match(mn.name, '(/quantile_counts|_quantile_counts)$')
    )
SELECT
    if(q.present = 1, q.exp, c.exp) AS exp,
    if(q.present = 1, q.run, c.run) AS run,
    if(q.present = 1, q.attempt, c.attempt) AS attempt,
    if(q.present = 1, q.step, c.step) AS step,
    if(q.present = 1, q.node, c.node) AS node,
    if(q.present = 1, q.rank, c.rank) AS rank,
    if(q.present = 1, q.microbatch, c.microbatch) AS microbatch,
    if(q.present = 1, q.layer, c.layer) AS layer,
    if(q.present = 1, q.metric, c.metric) AS metric,
    [0., 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.] AS levels,
    q.quantiles AS quantiles,
    c.counts AS population_counts,
    coalesce(q.present, 0) = 1 AS quantiles_present,
    coalesce(c.present, 0) = 1 AS counts_present,
    (coalesce(q.present, 0) = 1) AND (coalesce(c.present, 0) = 1) AND q.width_valid AND (length(c.counts) = 5) AS complete_pair
FROM quantiles AS q
FULL OUTER JOIN counts AS c ON (q.exp = c.exp) AND (q.run = c.run) AND (q.attempt = c.attempt) AND (q.step = c.step) AND (q.node = c.node) AND (q.rank = c.rank) AND (q.microbatch = c.microbatch) AND (q.layer = c.layer) AND (q.metric = c.metric);

CREATE OR REPLACE VIEW metric_sampler.exact_quantiles_per_rank_step_live
AS SELECT *
FROM `metric_sampler.exact_quantiles_per_rank_step`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String}, step_filter = {step_filter:UInt32});

CREATE OR REPLACE VIEW metric_sampler.expert_activation_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.expert_activation_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.expert_activation_per_rank_step
AS WITH
    sums AS
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
            expert,
            reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[(2 * expert) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * expert) + 2]), 32))) AS sum_squares,
            1 AS present,
            (length(ec.`values`) % 2) = 0 AS width_valid
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        ARRAY JOIN range(toUInt32(intDiv(length(ec.`values`) + 1, 2))) AS expert
        WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND (ec.rank >= 0) AND (ec.source_metric_id = 0) AND (ec.metric_id = 151) AND (mn.name = 'monitor/expert_activation_sumsq_local')
    ),
    counts AS
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
            toUInt32(expert_index - 1) AS expert,
            token_rows,
            1 AS present
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        ARRAY JOIN
            ec.`values` AS token_rows,
            arrayEnumerate(ec.`values`) AS expert_index
        WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND (ec.rank >= 0) AND (ec.source_metric_id = 0) AND (ec.metric_id = 152) AND (mn.name = 'monitor/expert_activation_count_local')
    ),
    paired AS
    (
        SELECT
            if(s.present = 1, s.exp, c.exp) AS exp,
            if(s.present = 1, s.run, c.run) AS run,
            if(s.present = 1, s.attempt, c.attempt) AS attempt,
            if(s.present = 1, s.step, c.step) AS step,
            if(s.present = 1, s.node, c.node) AS node,
            if(s.present = 1, s.rank, c.rank) AS rank,
            if(s.present = 1, s.microbatch, c.microbatch) AS microbatch,
            if(s.present = 1, s.layer, c.layer) AS layer,
            if(s.present = 1, s.expert, c.expert) AS expert,
            s.sum_squares AS sum_squares,
            c.token_rows AS token_rows,
            (coalesce(s.present, 0) = 1) AND (coalesce(c.present, 0) = 1) AND s.width_valid AS complete_pair
        FROM sums AS s
        FULL OUTER JOIN counts AS c ON (s.exp = c.exp) AND (s.run = c.run) AND (s.attempt = c.attempt) AND (s.step = c.step) AND (s.node = c.node) AND (s.rank = c.rank) AND (s.microbatch = c.microbatch) AND (s.layer = c.layer) AND (s.expert = c.expert)
    )
SELECT
    p.exp AS exp,
    p.run AS run,
    p.attempt AS attempt,
    p.step AS step,
    p.node AS node,
    p.rank AS rank,
    p.microbatch AS microbatch,
    p.layer AS layer,
    p.expert AS expert,
    p.sum_squares AS sum_squares,
    p.token_rows AS token_rows,
    p.complete_pair AS complete_pair,
    coalesce(o.present, 0) = 1 AS metadata_present,
    has(o.global_expert_ids, p.expert) AS owned,
    if(p.complete_pair AND (o.present = 1) AND has(o.global_expert_ids, p.expert) AND (p.token_rows > 0), sqrt(p.sum_squares / p.token_rows), NULL) AS per_token_l2_rms
FROM paired AS p
LEFT JOIN
(
    SELECT
        exp,
        run,
        attempt,
        rank,
        layer,
        global_expert_ids,
        1 AS present
    FROM metric_sampler.monitor_expert_ownership
    FINAL
) AS o ON (o.exp = p.exp) AND (o.run = p.run) AND (o.attempt = p.attempt) AND (o.rank = p.rank) AND (o.layer = p.layer);

CREATE OR REPLACE VIEW metric_sampler.expert_activation_per_rank_step_live
AS SELECT *
FROM `metric_sampler.expert_activation_per_rank_step`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String}, step_filter = {step_filter:UInt32});

CREATE OR REPLACE VIEW metric_sampler.expert_load_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.expert_load_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.expert_load_per_rank_step
AS WITH
    rank_rows AS
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
            groupArray(ec.metric_id) AS metric_ids,
            groupArray(ec.alias_metric_id) AS alias_ids,
            groupArray(ec.`values`) AS rows_values
        FROM metric_sampler.expert_counts AS ec
        FINAL
        WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND (ec.rank >= 0) AND (ec.microbatch >= 0) AND (ec.source_metric_id = 0) AND (ec.metric_id IN (1, 95, 96, 97, 98))
        GROUP BY
            ec.exp,
            ec.run,
            ec.attempt,
            ec.step,
            ec.node,
            ec.rank,
            ec.microbatch,
            ec.layer
    ),
    resolved AS
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
            metric_id,
            if(alias_id = 0, row_values, rows_values[indexOf(metric_ids, alias_id)]) AS assignments
        FROM rank_rows
        ARRAY JOIN
            metric_ids AS metric_id,
            alias_ids AS alias_id,
            rows_values AS row_values
    )
SELECT
    r.exp AS exp,
    r.run AS run,
    r.attempt AS attempt,
    r.step AS step,
    r.node AS node,
    r.rank AS rank,
    r.microbatch AS microbatch,
    r.layer AS layer,
    mn.name AS population,
    expert_index - 1 AS expert,
    tokens,
    arraySum(r.assignments) AS total_assignments,
    length(r.assignments) AS experts,
    tokens / nullIf(arraySum(r.assignments) / length(r.assignments), 0) AS relative_load
FROM resolved AS r
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = r.exp) AND (mn.run = r.run) AND (mn.attempt = r.attempt) AND (mn.metric_id = r.metric_id)
ARRAY JOIN
    r.assignments AS tokens,
    arrayEnumerate(r.assignments) AS expert_index
WHERE mn.name IN ('router/tokens_per_expert', 'dispatcher/assignments_before_capacity', 'dispatcher/valid_assignments_before_capacity', 'dispatcher/assignments_after_capacity', 'dispatcher/valid_assignments_after_capacity');

CREATE OR REPLACE VIEW metric_sampler.expert_load_per_rank_step_live
AS SELECT *
FROM `metric_sampler.expert_load_per_rank_step`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String}, step_filter = {step_filter:UInt32});

CREATE OR REPLACE VIEW metric_sampler.expert_vectors_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.expert_vectors_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.expert_vectors_per_rank_step
AS SELECT
    rv.exp AS exp,
    rv.run AS run,
    rv.attempt AS attempt,
    rv.step AS step,
    rv.node AS node,
    rv.rank AS rank,
    rv.microbatch AS microbatch,
    rv.layer AS layer,
    mn.name AS metric,
    expert_index - 1 AS expert,
    value
FROM metric_sampler.router_vectors AS rv
FINAL
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
ARRAY JOIN
    rv.`values` AS value,
    arrayEnumerate(rv.`values`) AS expert_index
WHERE (rv.layer <= 255) AND (rv.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (rv.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (rv.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32}) OR ((rv.rank = -1) AND rv.identical AND (rv.node IN (SELECT node FROM metric_sampler.node_manifests WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND has(ranks, toUInt16(greatest({rank_filter:Int32}, 0))))))) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32}) OR (rv.microbatch = -1)) AND ((rv.rank >= 0) OR rv.identical) AND (rv.source_metric_id = 0) AND (rv.metric_id IN (5, 153)) AND (mn.name IN ('router/scores_mean', 'monitor/router_bias_local'));

CREATE OR REPLACE VIEW metric_sampler.expert_vectors_per_rank_step_live
AS SELECT *
FROM `metric_sampler.expert_vectors_per_rank_step`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String}, step_filter = {step_filter:UInt32});

CREATE OR REPLACE VIEW metric_sampler.memory_local_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.memory_local_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.memory_local_per_rank_step
AS SELECT
    ec.exp AS exp,
    ec.run AS run,
    ec.attempt AS attempt,
    ec.step AS step,
    ec.node AS node,
    ec.rank AS rank,
    ec.microbatch AS microbatch,
    ec.layer AS layer,
    arrayMap(i -> bitOr(toUInt64(ec.`values`[(2 * i) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * i) + 2]), 32)), range(11)) AS counters,
    arraySlice(ec.`values`, 23, 11) AS present,
    length(ec.`values`) = 33 AS width_valid,
    ['allocated_bytes.all.current', 'active_bytes.all.current', 'inactive_split_bytes.all.current', 'reserved_bytes.all.current', 'allocated_bytes.all.peak', 'active_bytes.all.peak', 'inactive_split_bytes.all.peak', 'reserved_bytes.all.peak', 'num_alloc_retries', 'device_free', 'device_total'] AS counter_names
FROM metric_sampler.expert_counts AS ec
FINAL
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND (ec.rank >= 0) AND (ec.source_metric_id = 0) AND (ec.metric_id = 157) AND (mn.name = 'monitor/memory_local');

CREATE OR REPLACE VIEW metric_sampler.memory_local_per_rank_step_live
AS SELECT *
FROM `metric_sampler.memory_local_per_rank_step`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String}, step_filter = {step_filter:UInt32});

CREATE OR REPLACE VIEW metric_sampler.monitor_local_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.monitor_local_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.monitor_local_per_rank_step
AS WITH
    observations AS
    (
        SELECT
            rv.exp AS exp,
            rv.run AS run,
            rv.attempt AS attempt,
            rv.step AS step,
            rv.node AS node,
            rv.rank AS rank,
            rv.microbatch AS microbatch,
            rv.layer AS layer,
            rv.metric_id AS metric_id,
            mn.name AS metric,
            arrayMap(x -> toFloat64(x), rv.`values`) AS stats,
            multiIf(rv.metric_id = 140, '', rv.metric_id = 144, 'activation', 'parameter') AS catalog_kind,
            toUInt16(rv.layer) AS catalog_index,
            length(rv.`values`) = multiIf(rv.metric_id = 140, 8, rv.metric_id = 149, 18, rv.metric_id = 160, 3, 2) AS width_valid
        FROM metric_sampler.router_vectors AS rv
        FINAL
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = rv.exp) AND (mn.run = rv.run) AND (mn.attempt = rv.attempt) AND (mn.metric_id = rv.metric_id)
        WHERE (rv.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (rv.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (rv.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (rv.rank = {rank_filter:Int32}) OR ((rv.rank = -1) AND rv.identical AND (rv.node IN (SELECT node FROM metric_sampler.node_manifests WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND has(ranks, toUInt16(greatest({rank_filter:Int32}, 0))))))) AND (({microbatch_filter:Int32} = -2) OR (rv.microbatch = {microbatch_filter:Int32}) OR (rv.microbatch = -1)) AND ((rv.rank >= 0) OR rv.identical) AND (rv.source_metric_id = 0) AND (rv.metric_id IN (140, 144, 145, 149, 160)) AND (mn.name IN ('monitor/router_margin_local', 'monitor/activation_local', 'monitor/weight_local', 'monitor/optimizer_local', 'monitor/materialized_weight_local'))
        UNION ALL
        SELECT
            ec.exp,
            ec.run,
            ec.attempt,
            ec.step,
            ec.node,
            ec.rank,
            ec.microbatch,
            ec.layer,
            ec.metric_id,
            mn.name,
            multiIf(ec.metric_id = 159, arrayConcat(arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[(2 * i) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * i) + 2]), 32))), range(6)), arrayMap(x -> toFloat64(x), arraySlice(ec.`values`, 13, 6))), (ec.metric_id IN (142, 150, 156)), arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[(2 * i) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * i) + 2]), 32))), range(multiIf(ec.metric_id = 150, 7, ec.metric_id = 156, 1, 4))), arrayMap(x -> toFloat64(x), ec.`values`)) AS stats,
            multiIf(ec.metric_id = 142, 'parameter', ec.metric_id = 146, 'activation', ec.metric_id = 156, 'learning_rate_group', '') AS catalog_kind,
            toUInt16(ec.layer) AS catalog_index,
            length(ec.`values`) = multiIf(ec.metric_id = 142, 8, ec.metric_id = 143, 4, ec.metric_id = 150, 14, ec.metric_id = 154, 3, ec.metric_id = 159, 18, 2) AS width_valid
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        WHERE (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32}) OR ((ec.rank = -1) AND ec.identical AND (ec.node IN (SELECT node FROM metric_sampler.node_manifests WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND has(ranks, toUInt16(greatest({rank_filter:Int32}, 0))))))) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND ((ec.rank >= 0) OR ec.identical) AND (ec.source_metric_id = 0) AND (ec.metric_id IN (142, 143, 146, 150, 154, 156, 159)) AND (mn.name IN ('monitor/update_weight_local', 'monitor/topk_local_counts', 'monitor/activation_local_observations', 'monitor/router_balance_local', 'monitor/router_balance_counts_local', 'monitor/lr_group_local', 'monitor/stage_time_local'))
        UNION ALL
        SELECT
            ec.exp,
            ec.run,
            ec.attempt,
            ec.step,
            ec.node,
            ec.rank,
            ec.microbatch,
            ec.layer,
            ec.metric_id,
            mn.name,
            arrayMap(i -> reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[((12 * group_index) + (2 * i)) + 1]), bitShiftLeft(toUInt64(ec.`values`[((12 * group_index) + (2 * i)) + 2]), 32))), range(6)) AS stats,
            'optimizer_group' AS catalog_kind,
            toUInt16(group_index) AS catalog_index,
            length(ec.`values`) = 132 AS width_valid
        FROM metric_sampler.expert_counts AS ec
        FINAL
        INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
        ARRAY JOIN range(11) AS group_index
        WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32}) OR ((ec.rank = -1) AND ec.identical AND (ec.node IN (SELECT node FROM metric_sampler.node_manifests WHERE (exp = {exp_filter:String}) AND (run = {run_filter:String}) AND has(ranks, toUInt16(greatest({rank_filter:Int32}, 0))))))) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND ((ec.rank >= 0) OR ec.identical) AND (ec.source_metric_id = 0) AND (ec.metric_id = 141) AND (mn.name = 'monitor/optimizer_health_local')
    ),
    catalog AS
    (
        SELECT
            exp,
            run,
            attempt,
            kind,
            rank,
            index,
            name,
            1 AS present
        FROM metric_sampler.monitor_parameters
        FINAL
    )
SELECT
    o.exp AS exp,
    o.run AS run,
    o.attempt AS attempt,
    o.step AS step,
    o.node AS node,
    o.rank AS rank,
    o.microbatch AS microbatch,
    o.layer AS layer,
    o.metric AS metric,
    o.stats AS stats,
    o.width_valid AS width_valid,
    o.catalog_kind AS catalog_kind,
    o.catalog_index AS catalog_index,
    if(local.present = 1, local.name, global.name) AS source_name,
    (o.catalog_kind = '') OR (coalesce(local.present, 0) = 1) OR (coalesce(global.present, 0) = 1) AS metadata_present,
    multiIf(o.metric_id = 140, ['score_median_gap', 'score_median_near', 'score_p25_unselected', 'score_p10_unselected', 'logit_median_gap', 'logit_median_near', 'logit_p25_unselected', 'logit_p10_unselected'], o.metric_id = 141, ['numel', 'frac_eps_count', 'frac_10eps_count', 'update_abs_sum', 'grad_contrib_sq_sum', 'wd_contrib_sq_sum'], o.metric_id = 142, ['numel', 'update_abs_sum', 'w_sum', 'w_sq_sum'], o.metric_id = 143, ['validtokens', 'correct1', 'correct5', 'correct20'], o.metric_id = 144, ['amax', 'mean_row_l2'], o.metric_id = 145, ['amax', 'whole_shard_l2'], o.metric_id = 150, ['normalized_topk_sum', 'raw_topk_sum', 'biased_topk_sum', 'ordinary_biased_score_entropy_sum', 'router_z_loss_sum', 'seq_aux_loss_sum', 'alpha_sum'], o.metric_id = 154, ['all_tokens', 'valid_tokens', 'forward_calls'], o.metric_id = 156, ['learning_rate'], o.metric_id = 159, ['batch_total', 'forward', 'loss_calculation', 'backward', 'optimizer_step', 'dataloader_fetch', 'batch_total_present', 'forward_present', 'loss_calculation_present', 'backward_present', 'optimizer_step_present', 'dataloader_fetch_present'], o.metric_id = 160, ['amax', 'whole_tensor_l2', 'present'], o.metric_id = 149, ['grad_l2', 'moment_l2', 'moment_sq_l2', 'param_l2', 'update_l2', 'moment_bc_l2', 'moment_sq_bc_l2', 'moment_amax', 'moment_sq_amax', 'grad_l2_present', 'moment_l2_present', 'moment_sq_l2_present', 'param_l2_present', 'update_l2_present', 'moment_bc_l2_present', 'moment_sq_bc_l2_present', 'moment_amax_present', 'moment_sq_amax_present'], ['observation_count', 'row_count']) AS stat_names
FROM observations AS o
LEFT JOIN catalog AS local ON (local.exp = o.exp) AND (local.run = o.run) AND (local.attempt = o.attempt) AND (local.kind = o.catalog_kind) AND (local.rank = o.rank) AND (local.index = o.catalog_index)
LEFT JOIN catalog AS global ON (global.exp = o.exp) AND (global.run = o.run) AND (global.attempt = o.attempt) AND (global.kind = o.catalog_kind) AND (global.rank = -1) AND (global.index = o.catalog_index);

CREATE OR REPLACE VIEW metric_sampler.monitor_local_per_rank_step_live
AS SELECT *
FROM `metric_sampler.monitor_local_per_rank_step`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String}, step_filter = {step_filter:UInt32});

CREATE OR REPLACE VIEW metric_sampler.scalar_observations_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.scalar_observations_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.scalar_observations_per_rank_step
AS SELECT
    s.exp AS exp,
    s.run AS run,
    s.attempt AS attempt,
    s.step AS step,
    s.node AS node,
    s.rank AS rank,
    s.microbatch AS microbatch,
    s.layer AS layer,
    s.metric_id AS metric_id,
    s.source_metric_id AS source_metric_id,
    mn.name AS metric,
    s.`values` AS `values`
FROM metric_sampler.scalars AS s
FINAL
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = s.exp) AND (mn.run = s.run) AND (mn.attempt = s.attempt) AND (mn.metric_id = s.metric_id)
WHERE (s.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (s.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (s.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (s.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (s.microbatch = {microbatch_filter:Int32}) OR (s.microbatch = -1)) AND (s.rank >= 0);

CREATE OR REPLACE VIEW metric_sampler.scalar_observations_per_rank_step_live
AS SELECT *
FROM `metric_sampler.scalar_observations_per_rank_step`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String}, step_filter = {step_filter:UInt32});

CREATE OR REPLACE VIEW metric_sampler.training_counts_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.training_counts_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.training_counts_per_rank_step
AS SELECT
    ec.exp AS exp,
    ec.run AS run,
    ec.attempt AS attempt,
    ec.step AS step,
    ec.node AS node,
    ec.rank AS rank,
    ec.microbatch AS microbatch,
    ec.layer AS layer,
    arrayMap(i -> bitOr(toUInt64(ec.`values`[(2 * i) + 1]), bitShiftLeft(toUInt64(ec.`values`[(2 * i) + 2]), 32)), range(4)) AS counts,
    arraySlice(ec.`values`, 9, 4) AS present,
    length(ec.`values`) = 12 AS width_valid
FROM metric_sampler.expert_counts AS ec
FINAL
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND (ec.rank >= 0) AND (ec.source_metric_id = 0) AND (ec.metric_id = 147) AND (mn.name = 'monitor/training_local_counts');

CREATE OR REPLACE VIEW metric_sampler.training_counts_per_rank_step_live
AS SELECT *
FROM `metric_sampler.training_counts_per_rank_step`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String}, step_filter = {step_filter:UInt32});

CREATE OR REPLACE VIEW metric_sampler.training_local_per_rank_selected_live
AS SELECT *
FROM `metric_sampler.training_local_per_rank_selected`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String});

CREATE OR REPLACE VIEW metric_sampler.training_local_per_rank_step
AS SELECT
    ec.exp AS exp,
    ec.run AS run,
    ec.attempt AS attempt,
    ec.step AS step,
    ec.node AS node,
    ec.rank AS rank,
    ec.microbatch AS microbatch,
    ec.layer AS layer,
    bitOr(toUInt64(ec.`values`[1]), bitShiftLeft(toUInt64(ec.`values`[2]), 32)) AS samples,
    bitOr(toUInt64(ec.`values`[3]), bitShiftLeft(toUInt64(ec.`values`[4]), 32)) AS tokens,
    reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[5]), bitShiftLeft(toUInt64(ec.`values`[6]), 32))) AS batch_seconds,
    reinterpretAsFloat64(bitOr(toUInt64(ec.`values`[7]), bitShiftLeft(toUInt64(ec.`values`[8]), 32))) AS device_flops,
    arraySlice(ec.`values`, 9, 4) AS present,
    length(ec.`values`) = 12 AS width_valid
FROM metric_sampler.expert_counts AS ec
FINAL
INNER JOIN metric_sampler.metric_names AS mn ON (mn.exp = ec.exp) AND (mn.run = ec.run) AND (mn.attempt = ec.attempt) AND (mn.metric_id = ec.metric_id)
WHERE (ec.layer <= 255) AND (ec.step = {step_filter:UInt32}) AND (({exp_filter:String} = '') OR (ec.exp = {exp_filter:String})) AND (({run_filter:String} = '') OR (ec.run = {run_filter:String})) AND (({rank_filter:Int32} = -2) OR (ec.rank = {rank_filter:Int32})) AND (({microbatch_filter:Int32} = -2) OR (ec.microbatch = {microbatch_filter:Int32}) OR (ec.microbatch = -1)) AND (ec.rank >= 0) AND (ec.source_metric_id = 0) AND (ec.metric_id = 158) AND (mn.name = 'monitor/training_local');

CREATE OR REPLACE VIEW metric_sampler.training_local_per_rank_step_live
AS SELECT *
FROM `metric_sampler.training_local_per_rank_step`(exp_filter = {exp_filter:String}, microbatch_filter = {microbatch_filter:Int32}, rank_filter = {rank_filter:Int32}, run_filter = {run_filter:String}, step_filter = {step_filter:UInt32});
