-- Whole-capture predicate evidence includes SAC calls without a fresh all-reduce.
WITH RECURSIVE
copies AS MATERIALIZED (
  SELECT g.id AS copy_id, g.ts, g.dur, h.id AS runtime_id, h.parent_id, h.track_id,
         h.ts + h.dur AS submit_end, EXTRACT_ARG(h.arg_set_id, 'args.External id') AS ext
  FROM slice g
  JOIN flow f ON f.slice_in = g.id
  JOIN slice h ON h.id = f.slice_out
  WHERE g.category = 'gpu_memcpy' AND g.name GLOB 'Memcpy DtoH*'
    AND EXTRACT_ARG(g.arg_set_id, 'args.bytes') = 1
    AND h.category = 'cuda_runtime' AND h.name = 'cudaMemcpyAsync'
),
ancestors(copy_id, ancestor_id) AS (
  SELECT copy_id, parent_id FROM copies WHERE parent_id IS NOT NULL
  UNION ALL
  SELECT a.copy_id, p.parent_id
  FROM ancestors a JOIN slice p ON p.id = a.ancestor_id
  WHERE p.parent_id IS NOT NULL
),
predicate AS (
  SELECT a.copy_id, p.id AS predicate_id
  FROM ancestors a JOIN slice p ON p.id = a.ancestor_id
  WHERE p.name = 'torchtitan::conditional_capacity_mask'
)
SELECT c.copy_id, p.predicate_id, c.runtime_id, c.dur AS copy_ns,
       h.id AS sync_id, h.dur AS sync_ns,
       h.ts + h.dur - (c.ts + c.dur) AS sync_return_after_copy_ns
FROM copies c LEFT JOIN predicate p USING (copy_id)
LEFT JOIN slice h ON h.parent_id = c.parent_id AND h.track_id = c.track_id
  AND h.name = 'cudaStreamSynchronize' AND h.category = 'cuda_runtime'
  AND EXTRACT_ARG(h.arg_set_id, 'args.External id') = c.ext
  AND h.ts >= c.submit_end AND h.ts + h.dur >= c.ts + c.dur
ORDER BY c.ts;
