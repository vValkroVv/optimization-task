-- exp, run and attempt come from the object key, not the JSON body, so every table's partitioning key is derived
-- the same way regardless of what a manifest's own fields say.

-- JSONAsString already splits the array into one row per element, so `json` is one metric
-- object here, not the array: JSONExtractArrayRaw on an object would silently drop every row.
INSERT INTO metric_sampler.metrics
    (exp, run, attempt, metric_id, name, family, value_type, node_merge, unit, description,
     bins_start, bins_width, bins_count, summarize_experts, detailed, job_identical, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    JSONExtractUInt(json, 'metric_id') AS metric_id,
    JSONExtractString(json, 'name') AS name,
    JSONExtractString(json, 'family') AS family,
    JSONExtractString(json, 'value_type') AS value_type,
    JSONExtractString(json, 'node_merge') AS node_merge,
    JSONExtractString(json, 'unit') AS unit,
    JSONExtractString(json, 'description') AS description,
    JSONExtractFloat(json, 'bins', 'start') AS bins_start,
    JSONExtractFloat(json, 'bins', 'width') AS bins_width,
    JSONExtractUInt(json, 'bins', 'count') AS bins_count,
    JSONExtractBool(json, 'summarize_experts') AS summarize_experts,
    JSONExtractBool(json, 'detailed') AS detailed,
    JSONExtractBool(json, 'job_identical') AS job_identical,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'metrics/**/metrics.json', format = 'JSONAsString', structure = 'json String')
WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'metrics')
SETTINGS s3_throw_on_zero_files_match = 0;

-- A JSON object is marked processed from the rows it produced, not from a second S3 listing: one uploaded
-- between the two statements would otherwise be marked without being ingested.
INSERT INTO metric_sampler.processed_objects
SELECT 'metrics' AS kind, source_object AS object, count() AS rows, now() AS processed_at
FROM metric_sampler.metrics
WHERE source_object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'metrics')
GROUP BY source_object;

-- streams.json is written only once a run reports stream ids; a run without them never matches this glob.
INSERT INTO metric_sampler.streams
    (exp, run, attempt, stream_id, name, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    JSONExtractUInt(json, 'stream_id') AS stream_id,
    JSONExtractString(json, 'name') AS name,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'metrics/**/streams.json', format = 'JSONAsString', structure = 'json String')
WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'streams')
SETTINGS s3_throw_on_zero_files_match = 0;

INSERT INTO metric_sampler.processed_objects
SELECT 'streams' AS kind, source_object AS object, count() AS rows, now() AS processed_at
FROM metric_sampler.streams
WHERE source_object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'streams')
GROUP BY source_object;

INSERT INTO metric_sampler.monitor_parameters
    (exp, run, attempt, kind, rank, index, name, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    JSONExtractString(json, 'kind') AS kind,
    if(JSONHas(json, 'rank'), JSONExtractInt(json, 'rank'), -1) AS rank,
    JSONExtractUInt(json, 'index') AS index,
    JSONExtractString(json, 'name') AS name,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'metrics/**/monitor_parameters.json', format = 'JSONAsString', structure = 'json String')
WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'monitor_parameters')
SETTINGS s3_throw_on_zero_files_match = 0;

INSERT INTO metric_sampler.processed_objects
SELECT 'monitor_parameters' AS kind, source_object AS object, count() AS rows, now() AS processed_at
FROM metric_sampler.monitor_parameters
WHERE source_object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'monitor_parameters')
GROUP BY source_object;

INSERT INTO metric_sampler.monitor_expert_ownership
    (exp, run, attempt, rank, layer, global_expert_ids, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    JSONExtractInt(json, 'rank') AS rank,
    JSONExtractInt(json, 'layer') AS layer,
    JSONExtract(json, 'global_expert_ids', 'Array(UInt16)') AS global_expert_ids,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'metrics/**/monitor_expert_ownership.json', format = 'JSONAsString', structure = 'json String')
WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'monitor_expert_ownership')
SETTINGS s3_throw_on_zero_files_match = 0;

INSERT INTO metric_sampler.processed_objects
SELECT 'monitor_expert_ownership' AS kind, source_object AS object, count() AS rows, now() AS processed_at
FROM metric_sampler.monitor_expert_ownership
WHERE source_object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'monitor_expert_ownership')
GROUP BY source_object;

-- A batch's node manifests are uploaded as one object whose documents are the rows; objects of older collectors
-- hold a single document.
INSERT INTO metric_sampler.node_manifests
    (exp, run, attempt, node, schema_version, first_step, last_step, ranks, absent_ranks, steps, rows_per_family,
     missing_ranks, dropped_steps_per_rank, torn_slots_per_rank, late_slots_per_rank, identical_violations,
     nodes_in_job, capture_interval, written_at, raw, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    JSONExtractString(json, 'node') AS node,
    JSONExtractUInt(json, 'schema_version') AS schema_version,
    JSONExtractUInt(json, 'first_step') AS first_step,
    JSONExtractUInt(json, 'last_step') AS last_step,
    JSONExtract(json, 'ranks', 'Array(UInt16)') AS ranks,
    JSONExtract(json, 'absent_ranks', 'Array(UInt16)') AS absent_ranks,
    JSONExtract(json, 'steps', 'Array(UInt32)') AS steps,
    JSONExtract(json, 'rows_per_family', 'Map(String, UInt64)') AS rows_per_family,
    JSONExtract(json, 'missing_ranks', 'Map(String, Array(UInt16))') AS missing_ranks,
    JSONExtract(json, 'dropped_steps_per_rank', 'Map(String, UInt32)') AS dropped_steps_per_rank,
    JSONExtract(json, 'torn_slots_per_rank', 'Map(String, UInt32)') AS torn_slots_per_rank,
    JSONExtract(json, 'late_slots_per_rank', 'Map(String, UInt32)') AS late_slots_per_rank,
    JSONExtractUInt(json, 'identical_violations') AS identical_violations,
    JSONExtractUInt(json, 'nodes_in_job') AS nodes_in_job,
    -- A manifest without capture_interval means every step was captured.
    greatest(JSONExtractUInt(json, 'capture_interval'), 1) AS capture_interval,
    parseDateTime64BestEffort(JSONExtractString(json, 'written_at')) AS written_at,
    json AS raw,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'node_manifests/**/*.json', format = 'JSONAsString', structure = 'json String')
WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'node_manifests')
-- Per-window objects (.../node=<node>/<window>.json) and per-batch ones (.../part-....json) share listing
-- pages: hive partition columns derived from key=value path segments then differ in size between keys, and
-- the _path filter fails on them ("Size of filter doesn't match size of column"). exp, run and attempt are
-- read from _path below.
SETTINGS s3_throw_on_zero_files_match = 0, use_hive_partitioning = 0;

INSERT INTO metric_sampler.processed_objects
SELECT 'node_manifests' AS kind, source_object AS object, 1 AS rows, now() AS processed_at
FROM metric_sampler.node_manifests
WHERE source_object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'node_manifests')
GROUP BY source_object;

-- The batch manifest is written last, so its presence is what makes a batch ingestible.
INSERT INTO metric_sampler.batches
    (exp, run, attempt, schema_version, batch, first_step, last_step, source_windows, nodes,
     rows_per_family, logged_dropped_rows, collected_at, raw, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    JSONExtractUInt(json, 'schema_version') AS schema_version,
    JSONExtractString(json, 'batch') AS batch,
    JSONExtractUInt(json, 'first_step') AS first_step,
    JSONExtractUInt(json, 'last_step') AS last_step,
    JSONExtract(json, 'source_windows', 'Array(String)') AS source_windows,
    JSONExtract(json, 'nodes', 'Array(String)') AS nodes,
    JSONExtract(json, 'rows_per_family', 'Map(String, UInt64)') AS rows_per_family,
    JSONExtractUInt(json, 'logged_dropped_rows') AS logged_dropped_rows,
    parseDateTime64BestEffort(JSONExtractString(json, 'collected_at')) AS collected_at,
    json AS raw,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'batches/**/*.json', format = 'JSONAsString', structure = 'json String')
WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'batches')
SETTINGS s3_throw_on_zero_files_match = 0;

INSERT INTO metric_sampler.processed_objects
SELECT 'batches' AS kind, source_object AS object, 1 AS rows, now() AS processed_at
FROM metric_sampler.batches
WHERE source_object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'batches');

-- Only the index of the activation dumps is ingested; the dump parquet files are not under `obs_ms`.
INSERT INTO metric_sampler.dumps
    (exp, run, attempt, step, rank, rows, bytes, points, layers, tokens, s3_key, raw, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    JSONExtractUInt(json, 'step') AS step,
    JSONExtractInt(json, 'rank') AS rank,
    JSONExtractUInt(json, 'rows') AS rows,
    JSONExtractUInt(json, 'bytes') AS bytes,
    JSONExtract(json, 'points', 'Array(String)') AS points,
    JSONExtract(json, 'layers', 'Array(Int32)') AS layers,
    JSONExtractUInt(json, 'tokens') AS tokens,
    JSONExtractString(json, 's3_key') AS s3_key,
    json AS raw,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'dumps/**/*.json', format = 'JSONAsString', structure = 'json String')
WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'dumps')
SETTINGS s3_throw_on_zero_files_match = 0;

INSERT INTO metric_sampler.processed_objects
SELECT 'dumps' AS kind, source_object AS object, 1 AS rows, now() AS processed_at
FROM metric_sampler.dumps
WHERE source_object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'dumps');

-- Family data is ingested after the manifests, gated on the owning batch manifest: the <batch> token is the same
-- source-windows hash in the parquet name and in batches.batch.
-- Rows with equal sort keys are left to merges and FINAL to replace, since replacing them within each inserted
-- block is expensive.
-- The filters on _path decide per object, so they run once over the listing of the objects (format One reads none
-- of their data) instead of on every decoded row.
INSERT INTO metric_sampler.scalars
    (exp, run, attempt, node, step, rank, microbatch, layer, metric_id, source_metric_id,
     contributors, expected_contributors, identical, verbose, values, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    node, step, rank, microbatch, layer, metric_id, source_metric_id, contributors, expected_contributors, identical, verbose, values,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'family=scalars/**/*.parquet', format = 'Parquet',
        structure = 'node String, step UInt32, rank Int16, microbatch Int16, layer Int16, metric_id UInt16, source_metric_id UInt16, contributors UInt8, expected_contributors UInt8, identical Bool, verbose Bool, values Array(Float32)')
WHERE _path IN (
    SELECT _path FROM s3(obs_ms, filename = 'family=scalars/**/*.parquet', format = 'One')
    WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'scalars')
      AND (extract(_path, 'exp=([^/]+)'), extract(_path, 'run=([^/]+)'), extract(_path, 'attempt=([^/]+)'),
           extract(_path, 'part-\d+-\d+-([^/.]+)\.parquet$'))
          IN (SELECT exp, run, attempt, batch FROM metric_sampler.batches))
SETTINGS s3_throw_on_zero_files_match = 0, optimize_on_insert = 0;

-- A family object is too large to count by reading it again: the marker reads ingested_objects, which the materialized
-- views of schema.sql fill with the rows each insert wrote. A statement that failed midway leaves a partial count under
-- an older ingested_at; the marker only runs after its insert succeeded, which wrote the object again, so it takes the
-- latest count of each object.
INSERT INTO metric_sampler.processed_objects
SELECT 'scalars' AS kind, object, argMax(generation_rows, ingested_at) AS rows, now() AS processed_at
FROM
(
    SELECT object, ingested_at, sum(rows) AS generation_rows
    FROM metric_sampler.ingested_objects
    WHERE kind = 'scalars'
      AND object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'scalars')
    GROUP BY object, ingested_at
)
GROUP BY object;

INSERT INTO metric_sampler.expert_counts
    (exp, run, attempt, node, step, rank, microbatch, layer, metric_id, source_metric_id,
     contributors, expected_contributors, identical, verbose, alias_metric_id, values, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    node, step, rank, microbatch, layer, metric_id, source_metric_id, contributors, expected_contributors, identical, verbose,
    alias_metric_id, values,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'family=expert_counts/**/*.parquet', format = 'Parquet',
        structure = 'node String, step UInt32, rank Int16, microbatch Int16, layer Int16, metric_id UInt16, source_metric_id UInt16, contributors UInt8, expected_contributors UInt8, identical Bool, verbose Bool, alias_metric_id UInt16, values Array(UInt32)')
WHERE _path IN (
    SELECT _path FROM s3(obs_ms, filename = 'family=expert_counts/**/*.parquet', format = 'One')
    WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'expert_counts')
      AND (extract(_path, 'exp=([^/]+)'), extract(_path, 'run=([^/]+)'), extract(_path, 'attempt=([^/]+)'),
           extract(_path, 'part-\d+-\d+-([^/.]+)\.parquet$'))
          IN (SELECT exp, run, attempt, batch FROM metric_sampler.batches))
SETTINGS s3_throw_on_zero_files_match = 0, optimize_on_insert = 0;

INSERT INTO metric_sampler.processed_objects
SELECT 'expert_counts' AS kind, object, argMax(generation_rows, ingested_at) AS rows, now() AS processed_at
FROM
(
    SELECT object, ingested_at, sum(rows) AS generation_rows
    FROM metric_sampler.ingested_objects
    WHERE kind = 'expert_counts'
      AND object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'expert_counts')
    GROUP BY object, ingested_at
)
GROUP BY object;

INSERT INTO metric_sampler.router_vectors
    (exp, run, attempt, node, step, rank, microbatch, layer, metric_id, source_metric_id,
     contributors, expected_contributors, identical, verbose, values, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    node, step, rank, microbatch, layer, metric_id, source_metric_id, contributors, expected_contributors, identical, verbose, values,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'family=router_vectors/**/*.parquet', format = 'Parquet',
        structure = 'node String, step UInt32, rank Int16, microbatch Int16, layer Int16, metric_id UInt16, source_metric_id UInt16, contributors UInt8, expected_contributors UInt8, identical Bool, verbose Bool, values Array(Float32)')
WHERE _path IN (
    SELECT _path FROM s3(obs_ms, filename = 'family=router_vectors/**/*.parquet', format = 'One')
    WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'router_vectors')
      AND (extract(_path, 'exp=([^/]+)'), extract(_path, 'run=([^/]+)'), extract(_path, 'attempt=([^/]+)'),
           extract(_path, 'part-\d+-\d+-([^/.]+)\.parquet$'))
          IN (SELECT exp, run, attempt, batch FROM metric_sampler.batches))
SETTINGS s3_throw_on_zero_files_match = 0, optimize_on_insert = 0;

INSERT INTO metric_sampler.processed_objects
SELECT 'router_vectors' AS kind, object, argMax(generation_rows, ingested_at) AS rows, now() AS processed_at
FROM
(
    SELECT object, ingested_at, sum(rows) AS generation_rows
    FROM metric_sampler.ingested_objects
    WHERE kind = 'router_vectors'
      AND object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'router_vectors')
    GROUP BY object, ingested_at
)
GROUP BY object;

-- `logged` has no owning node: source_step preserves the monitor timestamp, and older files leave it NULL.
INSERT INTO metric_sampler.logged
    (exp, run, attempt, step, name, value, source_step, source_object, ingested_at)
SELECT
    extract(_path, 'exp=([^/]+)') AS exp,
    extract(_path, 'run=([^/]+)') AS run,
    extract(_path, 'attempt=([^/]+)') AS attempt,
    step, name, value, source_step,
    _path AS source_object,
    now64(3) AS ingested_at
FROM s3(obs_ms, filename = 'family=logged/**/*.parquet', format = 'Parquet',
        structure = 'step UInt32, name String, value Float64, source_step Nullable(UInt32)')
WHERE _path IN (
    SELECT _path FROM s3(obs_ms, filename = 'family=logged/**/*.parquet', format = 'One')
    WHERE _path NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'logged')
      AND (extract(_path, 'exp=([^/]+)'), extract(_path, 'run=([^/]+)'), extract(_path, 'attempt=([^/]+)'),
           extract(_path, 'part-\d+-\d+-([^/.]+)\.parquet$'))
          IN (SELECT exp, run, attempt, batch FROM metric_sampler.batches))
SETTINGS s3_throw_on_zero_files_match = 0, input_format_parquet_allow_missing_columns = 1;

INSERT INTO metric_sampler.processed_objects
SELECT 'logged' AS kind, object, argMax(generation_rows, ingested_at) AS rows, now() AS processed_at
FROM
(
    SELECT object, ingested_at, sum(rows) AS generation_rows
    FROM metric_sampler.ingested_objects
    WHERE kind = 'logged'
      AND object NOT IN (SELECT object FROM metric_sampler.processed_objects WHERE kind = 'logged')
    GROUP BY object, ingested_at
)
GROUP BY object;
