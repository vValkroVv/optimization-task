-- Native S rank traces. Endpoints are GPU timestamps. Host timestamps come from CUPTI flows.
WITH
capacity AS MATERIALIZED (
  SELECT id, ts, dur, ts + dur AS start_ns,
         EXTRACT_ARG(arg_set_id, 'args.Seq') AS collective_seq,
         EXTRACT_ARG(arg_set_id, 'args.Process Group Ranks') AS ranks,
         LEAD(ts) OVER (ORDER BY ts, id) AS next_capacity_ns
  FROM slice
  WHERE category = 'kernel'
    AND name GLOB 'ncclDevKernel_AllReduce_Sum_u64*'
    AND EXTRACT_ARG(arg_set_id, 'args.In msg nelems') = 128
    AND EXTRACT_ARG(arg_set_id, 'args.Group size') = 8
),
windows AS MATERIALIZED (
  SELECT c.*, (
    SELECT q.id FROM slice q
    WHERE q.category = 'kernel' AND q.name = '_rowwise_1d_quantize_kernel'
      AND q.ts >= c.start_ns
      AND (c.next_capacity_ns IS NULL OR q.ts < c.next_capacity_ns)
    ORDER BY q.ts, q.id LIMIT 1
  ) AS quant_id
  FROM capacity c
),
clipped AS MATERIALIZED (
  SELECT w.id, s.id AS event_id, MAX(s.ts, w.start_ns) AS a,
         MIN(s.ts + s.dur, q.ts) AS b
  FROM windows w JOIN slice q ON q.id = w.quant_id
  JOIN slice s ON s.category IN ('kernel', 'gpu_memcpy', 'gpu_memset')
    AND s.dur > 0 AND s.ts < q.ts AND s.ts + s.dur > w.start_ns
),
ordered AS (
  SELECT *, MAX(b) OVER (
    PARTITION BY id ORDER BY a, b, event_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  ) AS previous_end
  FROM clipped
),
activity AS (
  SELECT id, SUM(MAX(0, b - MAX(a, COALESCE(previous_end, a)))) AS active_ns
  FROM ordered GROUP BY id
),
launch AS MATERIALIZED (
  SELECT f.slice_in AS device_id, MIN(h.ts) AS submit,
         MAX(h.ts + h.dur) AS submit_end
  FROM flow f JOIN slice h ON h.id = f.slice_out
  WHERE h.category IN ('cuda_driver', 'cuda_runtime')
  GROUP BY f.slice_in
),
detail AS MATERIALIZED (
  SELECT w.id, s.id AS event_id, s.name, s.ts, s.dur, l.submit,
         CASE WHEN s.category = 'gpu_memcpy' AND s.name GLOB 'Memcpy DtoH*'
              THEN EXTRACT_ARG(s.arg_set_id, 'args.bytes') END AS dtoh_bytes
  FROM windows w JOIN slice q ON q.id = w.quant_id
  JOIN slice s ON s.category IN ('kernel', 'gpu_memcpy')
    AND s.ts >= w.start_ns AND s.ts < q.ts
  LEFT JOIN launch l ON l.device_id = s.id
),
prep AS (
  SELECT id,
    SUM(CASE WHEN dtoh_bytes = 1 THEN 1 ELSE 0 END) AS one_byte_dtoh,
    SUM(CASE WHEN name GLOB '*_capacity_mask_kernel*' THEN dur ELSE 0 END)
      AS selector_ns,
    SUM(CASE WHEN name GLOB '*_capacity_mask_kernel*' THEN 1 ELSE 0 END)
      AS selector_launches,
    MIN(CASE WHEN name GLOB '*_capacity_mask_kernel*' THEN submit END)
      AS selector_submit,
    MIN(CASE WHEN name = '_routing_metadata_kernel' THEN ts END) AS metadata_ts,
    MIN(CASE WHEN name = '_routing_metadata_kernel' THEN submit END) AS metadata_submit
  FROM detail GROUP BY id
)
SELECT w.id AS capacity_slice_id, w.collective_seq, w.ranks,
       w.dur AS allreduce_ns, q.id AS quant_slice_id, q.dur AS quant_ns,
       q.ts - w.start_ns AS elapsed_ns, COALESCE(a.active_ns, 0) AS active_ns,
       q.ts - w.start_ns - COALESCE(a.active_ns, 0) AS idle_ns,
       COALESCE(p.one_byte_dtoh, 0) AS one_byte_dtoh_in_window,
       COALESCE(p.selector_launches, 0) AS selector_launches_in_window,
       COALESCE(p.selector_ns, 0) AS selector_ns_in_window,
       p.selector_submit - w.start_ns AS selector_submit_vs_allreduce_end_ns,
       p.metadata_submit - w.start_ns AS metadata_submit_vs_allreduce_end_ns,
       p.metadata_ts - p.metadata_submit AS metadata_start_after_submit_ns,
       l.submit - w.start_ns AS quant_submit_vs_allreduce_end_ns,
       q.ts - l.submit_end AS quant_start_after_submit_end_ns
FROM windows w LEFT JOIN slice q ON q.id = w.quant_id
LEFT JOIN activity a ON a.id = w.id
LEFT JOIN prep p ON p.id = w.id
LEFT JOIN launch l ON l.device_id = q.id
ORDER BY w.ts, w.id;
