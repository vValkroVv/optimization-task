-- G3 baseline update111: executed through actual Perfetto MCP on BOTH handles.
-- r0 trace_d9e7ae24dd735c55; r1 trace_40a5092f8f0c7a38.
-- Schema: list_tables both handles; list_table_structure(slice,track,args,flow) r0.
-- Stdlib discovery: list_stdlib_modules(domain='generic'); slices.with_context used below.
-- Each statement below was executed once per handle; JSON query_results retains exact outputs.
-- Units: ns base; /1e6 ms. Kernel sums are explicitly not elapsed time.

-- Corrected strict NCCL-exclusive union: executed successfully on both handles after
-- one load_trace reload of the same immutable pair. Unlike original nccl_only_ms,
-- this excludes overlapping gpu_memset. Full trace equals the proven CPU frame.
WITH e AS (SELECT ts,ts+dur AS en,CASE WHEN category='kernel' AND name GLOB 'ncclDevKernel*' THEN 'nccl' WHEN category='gpu_memset' THEN 'memset' ELSE 'other' END AS fam FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0),
p AS (SELECT ts AS t,fam,1 AS d FROM e UNION ALL SELECT en AS t,fam,-1 AS d FROM e),
a AS (SELECT t,sum(CASE WHEN fam='nccl' THEN d ELSE 0 END) AS nd,sum(CASE WHEN fam='memset' THEN d ELSE 0 END) AS md,sum(CASE WHEN fam='other' THEN d ELSE 0 END) AS od FROM p GROUP BY t),
z AS (SELECT t,lead(t)OVER(ORDER BY t)-t AS dt,sum(nd)OVER(ORDER BY t) AS n,sum(md)OVER(ORDER BY t) AS m,sum(od)OVER(ORDER BY t) AS o FROM a)
SELECT sum(CASE WHEN n>0 AND m=0 AND o=0 THEN dt ELSE 0 END)/1e6 AS strict_nccl_exclusive_ms,sum(CASE WHEN m>0 THEN dt ELSE 0 END)/1e6 AS memset_union_ms,sum(CASE WHEN n>0 AND o=0 THEN dt ELSE 0 END)/1e6 AS original_nccl_excluding_other_but_allowing_memset_ms,sum(CASE WHEN n>0 AND o=0 AND m>0 THEN dt ELSE 0 END)/1e6 AS removed_memset_overlap_ms FROM z;

-- Follow-up containment query attempts: actual MCP RPC failed on both cached processors.
-- First pair: response-body decode failure; revised pair: request send failure on ports9002/9003.
-- No new measurement resulted. Existing categories min/max/count rows prove containment.
-- Failed attempt1 rank0
WITH f AS (SELECT ts AS a, ts+dur AS b FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*'),
d AS (SELECT * FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset'))
SELECT f.a AS cpu_frame_start_ns,f.b AS cpu_frame_end_ns,min(d.ts) AS all_device_min_start_ns,max(d.ts+d.dur) AS all_device_max_end_ns,count(*) AS full_trace_device_event_count,sum(CASE WHEN d.ts>=f.a AND d.ts+d.dur<=f.b THEN 1 ELSE 0 END) AS fully_contained_device_event_count,sum(CASE WHEN d.ts<f.b AND d.ts+d.dur>f.a THEN 1 ELSE 0 END) AS overlapping_frame_device_event_count,sum(CASE WHEN d.ts<f.a THEN 1 ELSE 0 END) AS starting_before_frame_count,sum(CASE WHEN d.ts+d.dur>f.b THEN 1 ELSE 0 END) AS ending_after_frame_count FROM d CROSS JOIN f;

-- Failed attempt2 rank1
WITH f AS (SELECT ts AS a, ts+dur AS b FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*'),
d AS (SELECT * FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset'))
SELECT f.a AS cpu_frame_start_ns,f.b AS cpu_frame_end_ns,min(d.ts) AS all_device_min_start_ns,max(d.ts+d.dur) AS all_device_max_end_ns,count(*) AS full_trace_device_event_count,sum(CASE WHEN d.ts>=f.a AND d.ts+d.dur<=f.b THEN 1 ELSE 0 END) AS fully_contained_device_event_count,sum(CASE WHEN d.ts<f.b AND d.ts+d.dur>f.a THEN 1 ELSE 0 END) AS overlapping_frame_device_event_count,sum(CASE WHEN d.ts<f.a THEN 1 ELSE 0 END) AS starting_before_frame_count,sum(CASE WHEN d.ts+d.dur>f.b THEN 1 ELSE 0 END) AS ending_after_frame_count FROM d CROSS JOIN f;

-- Failed attempt3 rank0
SELECT (SELECT ts FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*') AS cpu_frame_start_ns,(SELECT ts+dur FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*') AS cpu_frame_end_ns,min(ts) AS all_device_min_start_ns,max(ts+dur) AS all_device_max_end_ns,count(*) AS full_trace_device_event_count,sum(CASE WHEN ts>=3693419130056914 AND ts+dur<=3693433259854297 THEN 1 ELSE 0 END) AS fully_contained_device_event_count,sum(CASE WHEN ts<3693433259854297 AND ts+dur>3693419130056914 THEN 1 ELSE 0 END) AS overlapping_frame_device_event_count,sum(CASE WHEN ts<3693419130056914 THEN 1 ELSE 0 END) AS starting_before_frame_count,sum(CASE WHEN ts+dur>3693433259854297 THEN 1 ELSE 0 END) AS ending_after_frame_count FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset');

-- Failed attempt4 rank1
SELECT (SELECT ts FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*') AS cpu_frame_start_ns,(SELECT ts+dur FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*') AS cpu_frame_end_ns,min(ts) AS all_device_min_start_ns,max(ts+dur) AS all_device_max_end_ns,count(*) AS full_trace_device_event_count,sum(CASE WHEN ts>=3693419129364432 AND ts+dur<=3693433259196156 THEN 1 ELSE 0 END) AS fully_contained_device_event_count,sum(CASE WHEN ts<3693433259196156 AND ts+dur>3693419129364432 THEN 1 ELSE 0 END) AS overlapping_frame_device_event_count,sum(CASE WHEN ts<3693419129364432 THEN 1 ELSE 0 END) AS starting_before_frame_count,sum(CASE WHEN ts+dur>3693433259196156 THEN 1 ELSE 0 END) AS ending_after_frame_count FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset');


-- marker_multiplicity
SELECT s.category,s.name,t.name AS track_name,count(*) AS n,min(s.ts) AS start_ns,max(s.ts+s.dur) AS end_ns,sum(s.dur)/1e6 AS sum_ms FROM slice s JOIN track t ON t.id=s.track_id WHERE s.name GLOB 'task.profile_update.*' GROUP BY 1,2,3 ORDER BY 2,1;

-- categories
SELECT category,count(*) n,sum(dur)/1e6 sum_ms,min(ts) start,max(ts+dur) end FROM slice GROUP BY category ORDER BY n DESC;

-- argskeys
SELECT key,count(*) n FROM args GROUP BY key ORDER BY n DESC LIMIT 70;

-- topkernels
SELECT name,count(*) n,sum(dur)/1e6 sum_ms,avg(dur)/1e3 avg_us FROM slice WHERE category='kernel' GROUP BY name ORDER BY sum(dur) DESC LIMIT 35;

-- flows
SELECT count(*) n FROM flow;

-- clock
SELECT * FROM clock_snapshot LIMIT 10;

-- markers
SELECT category,name,count(*) n,sum(dur)/1e6 sum_ms FROM slice WHERE category IN ('cpu_op','user_annotation') AND (name GLOB '*fetch*' OR name GLOB '*ompil*' OR name GLOB '*ecomput*' OR name GLOB '*checkpoint*' OR name GLOB '*nductor*') GROUP BY 1,2 ORDER BY sum(dur) DESC LIMIT 50;

-- interval_scopes
WITH frame AS (SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*'),
e AS (SELECT s.*,CASE WHEN category='kernel' AND name GLOB 'ncclDevKernel*' THEN 'nccl' WHEN category='kernel' AND instr(name,'hybrid_ep::')>0 THEN 'ep_mixed' WHEN category='kernel' THEN 'compute_or_other' ELSE category END fam FROM slice s,frame WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0 AND ts<frame.b AND ts+dur>frame.a),
p AS (SELECT ts t,fam,1 d FROM e UNION ALL SELECT ts+dur,fam,-1 FROM e),
agg AS (SELECT t,sum(d) all_d,sum(CASE WHEN fam='nccl' THEN d ELSE 0 END) n,sum(CASE WHEN fam='ep_mixed' THEN d ELSE 0 END) ep,sum(CASE WHEN fam='compute_or_other' THEN d ELSE 0 END)c,sum(CASE WHEN fam='gpu_memcpy' THEN d ELSE 0 END)m FROM p GROUP BY t),
z AS (SELECT t,lead(t) OVER(ORDER BY t)-t dt,sum(all_d) OVER(ORDER BY t) busy,sum(n)OVER(ORDER BY t)nccl,sum(ep)OVER(ORDER BY t)ep,sum(c)OVER(ORDER BY t)compute,sum(m)OVER(ORDER BY t)copy FROM agg)
SELECT (SELECT (b-a)/1e6 FROM frame) frame_ms,sum(CASE WHEN busy>0 THEN dt ELSE 0 END)/1e6 busy_union_ms,sum(CASE WHEN nccl>0 THEN dt ELSE 0 END)/1e6 nccl_union_ms,sum(CASE WHEN ep>0 THEN dt ELSE 0 END)/1e6 ep_mixed_union_ms,sum(CASE WHEN compute>0 THEN dt ELSE 0 END)/1e6 compute_other_union_ms,sum(CASE WHEN copy>0 THEN dt ELSE 0 END)/1e6 copy_union_ms,sum(CASE WHEN nccl>0 AND compute=0 AND ep=0 AND copy=0 THEN dt ELSE 0 END)/1e6 nccl_only_ms,sum(CASE WHEN (nccl>0 OR ep>0) AND compute=0 THEN dt ELSE 0 END)/1e6 comm_or_ep_without_compute_ms,(SELECT max(ts+dur) FROM e)-(SELECT b FROM frame) gpu_tail_beyond_cpu_ns FROM z;

-- copies
SELECT name,count(*) n,sum(dur)/1e6 summed_ms,sum(CAST(EXTRACT_ARG(arg_set_id,'args.bytes') AS INT)) bytes FROM slice WHERE category='gpu_memcpy' GROUP BY name ORDER BY summed_ms DESC;

-- hostwaits
SELECT category,name,count(*) n,sum(dur)/1e6 sum_ms,max(dur)/1e6 max_ms FROM slice WHERE category IN ('cuda_runtime','cuda_driver','cpu_op') AND (name GLOB '*Synchronize*' OR name GLOB '*Wait*' OR name GLOB '*item*' OR name GLOB '*local_scalar*' OR name GLOB '*fetching*') GROUP BY 1,2 ORDER BY sum(dur) DESC LIMIT 30;

-- flowtypes
SELECT a.category source_cat,b.category dest_cat,count(*) n FROM flow f JOIN slice a ON a.id=f.slice_out JOIN slice b ON b.id=f.slice_in GROUP BY 1,2;

-- smallkernels
SELECT count(*) n,sum(CASE WHEN dur<15000 THEN 1 ELSE 0 END) under15us,sum(CASE WHEN dur<15000 THEN dur ELSE 0 END)/1e6 under15us_sum_ms FROM slice WHERE category='kernel';

-- compile_census
SELECT name,count(*) n FROM slice WHERE (name GLOB '*compile_inner*' OR name GLOB '*_compile.compile*' OR name GLOB '*GraphLowering*' OR name GLOB '*compile_fx*' OR name GLOB '*recompile*' OR name GLOB '*Cache*' OR name GLOB '*checkpoint*' OR name GLOB '*fetching_batch*') GROUP BY name ORDER BY n DESC LIMIT 50;

-- context_module
INCLUDE PERFETTO MODULE slices.with_context; SELECT name,thread_name,process_name,count(*) n,sum(dur)/1e6 sum_ms FROM thread_slice WHERE name GLOB '*DataLoader*' OR name GLOB '*next*' OR name GLOB '*fetch*' GROUP BY 1,2,3 ORDER BY sum(dur) DESC LIMIT 20;

-- stream_queue
WITH g AS (SELECT *,lag(ts+dur) OVER(PARTITION BY track_id ORDER BY ts,id) prev_end FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset')), j AS (SELECT g.*,h.ts launch_ts,h.ts+h.dur launch_end FROM g JOIN flow f ON f.slice_in=g.id JOIN slice h ON h.id=f.slice_out WHERE h.category IN ('cuda_runtime','cuda_driver')) SELECT category,count(*) linked_n,sum(CASE WHEN launch_end<=prev_end THEN 1 ELSE 0 END) queued_before_prior_same_track_end,sum(CASE WHEN launch_ts>prev_end THEN 1 ELSE 0 END) launch_after_prior_same_track_end,avg((ts-launch_end)/1e3) avg_launch_end_to_gpu_us,max((ts-launch_end)/1e3) max_launch_end_to_gpu_us FROM j GROUP BY category;

-- gpu_idle_gaps
WITH frame AS (SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*'),r AS (SELECT id,ts,ts+dur en,max(ts+dur)OVER(ORDER BY ts,id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prevmax FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0), g AS (SELECT ts,en,prevmax,id FROM r WHERE ts>prevmax) SELECT g.prevmax gap_start_ns,g.ts gap_end_ns,(g.ts-g.prevmax)/1e6 gap_ms,s.name next_device_name,h.name host_launch,(h.ts-g.prevmax)/1e6 launch_start_after_gap_ms,(h.ts+h.dur-g.prevmax)/1e6 launch_end_after_gap_ms,p.name launch_parent FROM g JOIN slice s ON s.id=g.id LEFT JOIN flow f ON f.slice_in=s.id LEFT JOIN slice h ON h.id=f.slice_out LEFT JOIN slice p ON p.id=h.parent_id ORDER BY gap_ms DESC LIMIT 20;

-- item_context
SELECT p.name parent,count(*) n,sum(s.dur)/1e6 sum_ms,max(s.dur)/1e6 max_ms FROM slice s LEFT JOIN slice p ON p.id=s.parent_id WHERE s.name='aten::item' GROUP BY p.name ORDER BY sum(s.dur) DESC LIMIT 20;

-- prefix_host
SELECT s.id,s.name,(s.ts-f.a)/1e6 rel_ms,s.dur/1e6 dur_ms FROM slice s,(SELECT ts a FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*') f WHERE s.ts>=f.a AND s.ts<f.a+670000000 AND s.category IN ('cpu_op','user_annotation') AND s.dur>1000000 ORDER BY s.ts LIMIT 50;

-- ce
SELECT name,count(*) n,sum(dur)/1e6 sum_ms FROM slice WHERE category='kernel' AND (name GLOB '*log_softmax*' OR name GLOB '*cross_entropy*' OR name GLOB '*nll_loss*') GROUP BY name ORDER BY sum(dur) DESC;

-- gfsdp
SELECT name,count(*) n,sum(dur)/1e6 sum_ms FROM slice WHERE category='user_annotation' AND name GLOB 'GFSDP*' GROUP BY name ORDER BY sum(dur) DESC LIMIT 50;

-- loader_frame
WITH f AS(SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*') SELECT count(*) n,sum(s.dur)/1e6 sum_ms,min(s.ts-f.a)/1e6 first_rel_ms,max(s.ts+s.dur-f.a)/1e6 last_end_rel_ms,max(s.dur)/1e6 max_ms FROM slice s,f WHERE s.name GLOB 'enumerate(DataLoader)*' AND s.ts>=f.a AND s.ts<f.b;

-- scalar_ancestors
SELECT p.name parent,g.name grandparent,gg.name ancestor,count(*) n,sum(s.dur)/1e6 sum_ms FROM slice s LEFT JOIN slice p ON p.id=s.parent_id LEFT JOIN slice g ON g.id=p.parent_id LEFT JOIN slice gg ON gg.id=g.parent_id WHERE s.name='aten::is_nonzero' GROUP BY 1,2,3 ORDER BY sum(s.dur) DESC LIMIT 20;

-- early_item
WITH f AS (SELECT ts a FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*') SELECT s.name,(s.ts-f.a)/1e6 rel_ms,s.dur/1e6 dur_ms,p.name parent FROM slice s,f LEFT JOIN slice p ON p.id=s.parent_id WHERE s.name IN ('aten::item','aten::_local_scalar_dense') AND s.ts>=f.a AND s.ts<f.a+700000000 ORDER BY s.ts;

-- distinct_queue
WITH g AS(SELECT *,lag(ts+dur)OVER(PARTITION BY track_id ORDER BY ts,id) prev_end FROM slice WHERE category IN('kernel','gpu_memcpy','gpu_memset')),l AS(SELECT f.slice_in id,min(h.ts) launch_start,max(h.ts+h.dur) launch_end,count(*) links FROM flow f JOIN slice h ON h.id=f.slice_out WHERE h.category IN('cuda_runtime','cuda_driver') GROUP BY f.slice_in) SELECT g.category,count(*) linked_device_events,sum(CASE WHEN l.launch_end<=g.prev_end THEN 1 ELSE 0 END) queued_before_prior_end,sum(CASE WHEN l.launch_start>g.prev_end THEN 1 ELSE 0 END) late_after_prior_end,sum(CASE WHEN l.links>1 THEN 1 ELSE 0 END) multiple_launch_flows FROM g JOIN l USING(id) GROUP BY category;

-- families
SELECT CASE WHEN name GLOB 'ncclDevKernel*AllGather*' THEN 'NCCL AllGather' WHEN name GLOB 'ncclDevKernel*ReduceScatter*' THEN 'NCCL ReduceScatter' WHEN name GLOB 'ncclDevKernel*' THEN 'NCCL other' WHEN instr(name,'hybrid_ep::dispatch')>0 THEN 'HybridEP dispatch mixed' WHEN instr(name,'hybrid_ep::combine')>0 THEN 'HybridEP combine mixed' WHEN instr(name,'hybrid_ep::')>0 THEN 'HybridEP other mixed' WHEN instr(name,'deep_gemm::')>0 THEN 'DeepGEMM' WHEN name GLOB 'nvjet*' THEN 'nvjet GEMM' WHEN name GLOB '*log_softmax*' THEN 'Inductor CE' WHEN name GLOB '*quant*' THEN 'quantization other' ELSE 'other compute/unknown' END family,count(*) n,sum(dur)/1e6 sum_ms FROM slice WHERE category='kernel' GROUP BY 1 ORDER BY sum(dur) DESC;

-- integrity
SELECT category,sum(CASE WHEN dur<0 THEN 1 ELSE 0 END) incomplete,count(*) n FROM slice GROUP BY category;

-- optimizer_launches
WITH o AS(SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.optimizer_step.*') SELECT g.name,count(*) n,sum(g.dur)/1e6 device_sum_ms,min(g.ts-o.a)/1e6 first_device_start_after_cpu_start_ms,max(g.ts+g.dur-o.b)/1e6 last_device_end_after_cpu_end_ms FROM slice h,o JOIN flow f ON f.slice_out=h.id JOIN slice g ON g.id=f.slice_in WHERE h.ts>=o.a AND h.ts<o.b AND g.category='kernel' GROUP BY g.name ORDER BY device_sum_ms DESC LIMIT 10;

-- kernel_sources
WITH l AS(SELECT slice_in,min(slice_out) host_id FROM flow GROUP BY slice_in) SELECT CASE WHEN g.name GLOB 'nvjet*' THEN 'nvjet' WHEN g.name GLOB '*deep_gemm*' THEN 'DeepGEMM' WHEN g.name GLOB 'ncclDevKernel*' THEN 'NCCL' WHEN g.name GLOB '*hybrid_ep*' THEN 'HybridEP' WHEN g.name GLOB '*log_softmax*' THEN 'CE' ELSE 'other' END family,p.name launch_parent,q.name launch_grandparent,count(*) n,sum(g.dur)/1e6 kernel_sum_ms FROM slice g JOIN l ON l.slice_in=g.id JOIN slice h ON h.id=l.host_id LEFT JOIN slice p ON p.id=h.parent_id LEFT JOIN slice q ON q.id=p.parent_id WHERE g.category='kernel' GROUP BY 1,2,3 ORDER BY kernel_sum_ms DESC LIMIT 30;

-- head_gather
SELECT name,count(*) n,sum(dur)/1e6 host_sum_ms FROM slice WHERE category='user_annotation' AND (name GLOB 'GFSDP::all_gather*' OR name GLOB 'GFSDP::reshard*') GROUP BY name ORDER BY name;

-- gpu_gap_summary
WITH r AS(SELECT ts,ts+dur en,max(ts+dur)OVER(ORDER BY ts,id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prevmax FROM slice WHERE category IN('kernel','gpu_memcpy','gpu_memset') AND dur>0),g AS(SELECT ts-prevmax gap FROM r WHERE ts>prevmax) SELECT count(*) n,sum(gap)/1e6 interior_idle_ms,sum(CASE WHEN gap>1000000 THEN 1 ELSE 0 END) gaps_over1ms,sum(CASE WHEN gap>1000000 THEN gap ELSE 0 END)/1e6 over1ms_total_ms,max(gap)/1e6 max_ms FROM g;

-- mb_integrity
WITH f AS(SELECT * FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*.train_step.*') SELECT count(*) n,sum(CASE WHEN m.ts<f.ts OR m.ts+m.dur>f.ts+f.dur THEN 1 ELSE 0 END) outside_train,min(m.dur)/1e6 min_ms,max(m.dur)/1e6 max_ms FROM slice m,f WHERE m.category='user_annotation' AND m.name GLOB 'task.profile_update.*.microbatch.*';
