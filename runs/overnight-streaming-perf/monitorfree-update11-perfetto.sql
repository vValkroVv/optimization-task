-- Queries executed with actual Perfetto MCP; handles and results are in the sibling JSON.
-- Original imports drop overlapping CuTe complete events; use the device-lanes handle for GPU totals.
-- Failed and superseded queries are retained; do not interpret their names as acceptance status.
-- markers [trace_6d2ec67f96a83162]
SELECT category,name,count(*) n,min(ts) start_ns,max(ts+dur) end_ns,sum(dur)/1e6 sum_ms FROM slice WHERE name GLOB 'task.profile_update.*' OR name GLOB 'ProfilerStep*' GROUP BY 1,2;

-- argskeys [trace_6d2ec67f96a83162]
SELECT key,count(*) n FROM args GROUP BY key ORDER BY n DESC LIMIT 35;

-- stream_counts [trace_6d2ec67f96a83162]
SELECT s.track_id,t.name,s.category,count(*) n,sum(s.dur)/1e6 sum_ms FROM slice s JOIN track t ON t.id=s.track_id WHERE s.category IN ('kernel','gpu_memcpy','gpu_memset') GROUP BY 1,2,3 ORDER BY sum(s.dur) DESC;

-- stats [trace_6d2ec67f96a83162]
SELECT name,idx,severity,value FROM stats WHERE value!=0 AND severity IN ('error','data_loss') ORDER BY name;

-- waits [trace_6d2ec67f96a83162]
SELECT category,name,count(*) n,sum(dur)/1e6 sum_ms,max(dur)/1e6 max_ms FROM slice WHERE category IN ('cuda_runtime','cuda_driver','cpu_op','user_annotation') AND (name GLOB '*Synchronize*' OR name GLOB '*Wait*' OR name GLOB '*item*' OR name GLOB '*local_scalar*' OR name GLOB '*fetch*' OR name GLOB '*DataLoader*') GROUP BY 1,2 ORDER BY sum(dur) DESC LIMIT 25;

-- topkernels [trace_6d2ec67f96a83162]
SELECT name,count(*) n,sum(dur)/1e6 sum_ms,avg(dur)/1e3 avg_us FROM slice WHERE category='kernel' GROUP BY name ORDER BY sum(dur) DESC LIMIT 45;

-- item_ancestors [trace_6d2ec67f96a83162]
SELECT p.name parent,g.name grandparent,gg.name ancestor,count(*) n,sum(s.dur)/1e6 sum_ms,max(s.dur)/1e6 max_ms FROM slice s LEFT JOIN slice p ON p.id=s.parent_id LEFT JOIN slice g ON g.id=p.parent_id LEFT JOIN slice gg ON gg.id=g.parent_id WHERE s.name='aten::item' GROUP BY 1,2,3 ORDER BY sum(s.dur) DESC LIMIT 18;

-- top_gaps [trace_6d2ec67f96a83162]
WITH r AS(SELECT id,ts,ts+dur en,max(ts+dur)OVER(ORDER BY ts,id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prevmax FROM slice WHERE category IN('kernel','gpu_memcpy','gpu_memset')AND dur>0),g AS(SELECT id,prevmax a,ts b FROM r WHERE ts>prevmax) SELECT g.a start_ns,g.b end_ns,(g.b-g.a)/1e6 gap_ms,s.name next_device,h.name launch,p.name parent,(h.ts-g.a)/1e6 launch_start_ms,(h.ts+h.dur-g.a)/1e6 launch_end_ms FROM g JOIN slice s USING(id)LEFT JOIN flow f ON f.slice_in=s.id LEFT JOIN slice h ON h.id=f.slice_out LEFT JOIN slice p ON p.id=h.parent_id ORDER BY gap_ms DESC LIMIT 12;

-- kernel_hosts [trace_6d2ec67f96a83162]
WITH l AS(SELECT slice_in,min(slice_out) h FROM flow GROUP BY slice_in) SELECT g.name,p.name parent,q.name grandparent,count(*) n,sum(g.dur)/1e6 sum_ms FROM slice g JOIN l ON l.slice_in=g.id JOIN slice h ON h.id=l.h LEFT JOIN slice p ON p.id=h.parent_id LEFT JOIN slice q ON q.id=p.parent_id WHERE g.category='kernel' AND(g.name GLOB 'nvjet_sm90_tst*' OR g.name GLOB '*hybrid_ep*notify*')GROUP BY 1,2,3 ORDER BY sum(g.dur)DESC LIMIT 20;

-- scope_union [trace_6d2ec67f96a83162]
WITH frame AS (SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation' AND name='task.profile_update.11.train_step.rank0'),
e AS (SELECT max(s.ts,a) ts,min(s.ts+s.dur,b) en,CASE WHEN category='kernel' AND name GLOB 'ncclDevKernel*' THEN 'nccl' WHEN category='kernel' AND instr(name,'hybrid_ep::')>0 THEN 'ep' WHEN category='kernel' THEN 'compute_other' ELSE category END fam FROM slice s,frame WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0 AND ts<b AND ts+dur>a),
p AS (SELECT ts t,fam,1 d FROM e UNION ALL SELECT en,fam,-1 FROM e UNION ALL SELECT a,'boundary',0 FROM frame UNION ALL SELECT b,'boundary',0 FROM frame),
a AS (SELECT t,sum(d) bd,sum(CASE WHEN fam='nccl' THEN d ELSE 0 END) nd,sum(CASE WHEN fam='ep' THEN d ELSE 0 END) ed,sum(CASE WHEN fam='compute_other' THEN d ELSE 0 END) cd,sum(CASE WHEN fam='gpu_memcpy' THEN d ELSE 0 END) md,sum(CASE WHEN fam='gpu_memset' THEN d ELSE 0 END) zd FROM p GROUP BY t),
z AS (SELECT t,lead(t)OVER(ORDER BY t)-t dt,sum(bd)OVER(ORDER BY t) busy,sum(nd)OVER(ORDER BY t)n,sum(ed)OVER(ORDER BY t)ep,sum(cd)OVER(ORDER BY t)c,sum(md)OVER(ORDER BY t)m,sum(zd)OVER(ORDER BY t)memset FROM a)
SELECT (SELECT(b-a)/1e6 FROM frame) frame_ms,sum(CASE WHEN busy>0 THEN dt ELSE 0 END)/1e6 busy_ms,sum(CASE WHEN busy=0 THEN dt ELSE 0 END)/1e6 idle_ms,sum(CASE WHEN c>0 THEN dt ELSE 0 END)/1e6 compute_other_ms,sum(CASE WHEN n>0 THEN dt ELSE 0 END)/1e6 nccl_ms,sum(CASE WHEN ep>0 THEN dt ELSE 0 END)/1e6 hybrid_ep_mixed_ms,sum(CASE WHEN m>0 THEN dt ELSE 0 END)/1e6 memcpy_ms,sum(CASE WHEN n>0 AND c=0 AND ep=0 AND m=0 AND memset=0 THEN dt ELSE 0 END)/1e6 nccl_exclusive_ms,sum(CASE WHEN(n>0 OR ep>0)AND c=0 THEN dt ELSE 0 END)/1e6 comm_without_compute_ms FROM z;

-- ce_and_attention [trace_6d2ec67f96a83162]
SELECT name,count(*) n,sum(dur)/1e6 sum_ms FROM slice WHERE category='kernel' AND(name GLOB '*log_softmax*' OR name GLOB '*nll_loss*' OR name GLOB '*cudnn*sdpa*' OR name GLOB '*delta*' OR name GLOB '*recompute_w*')GROUP BY name ORDER BY sum(dur)DESC LIMIT 25;

-- copies [trace_6d2ec67f96a83162]
SELECT name,count(*)n,sum(dur)/1e6 sum_ms,sum(CAST(EXTRACT_ARG(arg_set_id,'args.bytes') AS INT)) bytes FROM slice WHERE category='gpu_memcpy' GROUP BY name ORDER BY sum(dur)DESC;

-- loader_in_frame [trace_6d2ec67f96a83162]
WITH f AS(SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation' AND name='task.profile_update.11.train_step.rank0') SELECT count(*) n,sum(s.dur)/1e6 host_sum_ms,min(s.ts-f.a)/1e6 first_ms,max(s.ts+s.dur-f.a)/1e6 last_ms,max(s.dur)/1e6 max_ms FROM slice s,f WHERE s.name GLOB 'enumerate(DataLoader)*' AND s.ts>=f.a AND s.ts+s.dur<=f.b;

-- head_ancestry [trace_6d2ec67f96a83162]
SELECT g.id,g.name,p.name p,q.name q,r.name r,u.name u,v.name v,w.name w FROM slice g JOIN flow f ON f.slice_in=g.id JOIN slice h ON h.id=f.slice_out LEFT JOIN slice p ON p.id=h.parent_id LEFT JOIN slice q ON q.id=p.parent_id LEFT JOIN slice r ON r.id=q.parent_id LEFT JOIN slice u ON u.id=r.parent_id LEFT JOIN slice v ON v.id=u.parent_id LEFT JOIN slice w ON w.id=v.parent_id WHERE g.name='nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN' AND q.name='aten::matmul' LIMIT 1;

-- capacity_api [trace_6d2ec67f96a83162]
SELECT s.name,count(*) n,sum(s.dur)/1e6 host_sum_ms,sum(CASE WHEN s.name='cudaStreamSynchronize' THEN s.dur ELSE 0 END)/1e6 sync_ms FROM slice s LEFT JOIN slice p ON p.id=s.parent_id WHERE s.name='torchtitan::conditional_capacity_mask' OR(s.name='cudaStreamSynchronize' AND p.name='aten::_local_scalar_dense') GROUP BY s.name;

-- scope_union_complete [trace_cee2983a842c428a]
WITH frame AS (SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation' AND name='task.profile_update.11.train_step.rank0'),
e AS (SELECT max(s.ts,a) ts,min(s.ts+s.dur,b) en,CASE WHEN category='kernel' AND name GLOB 'ncclDevKernel*' THEN 'nccl' WHEN category='kernel' AND instr(name,'hybrid_ep::')>0 THEN 'ep' WHEN category='kernel' THEN 'compute_other' ELSE category END fam FROM slice s,frame WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0 AND ts<b AND ts+dur>a),
p AS (SELECT ts t,fam,1 d FROM e UNION ALL SELECT en,fam,-1 FROM e UNION ALL SELECT a,'boundary',0 FROM frame UNION ALL SELECT b,'boundary',0 FROM frame),
a AS (SELECT t,sum(d) bd,sum(CASE WHEN fam='nccl' THEN d ELSE 0 END) nd,sum(CASE WHEN fam='ep' THEN d ELSE 0 END) ed,sum(CASE WHEN fam='compute_other' THEN d ELSE 0 END) cd,sum(CASE WHEN fam='gpu_memcpy' THEN d ELSE 0 END) md,sum(CASE WHEN fam='gpu_memset' THEN d ELSE 0 END) zd FROM p GROUP BY t),
z AS (SELECT t,lead(t)OVER(ORDER BY t)-t dt,sum(bd)OVER(ORDER BY t) busy,sum(nd)OVER(ORDER BY t)n,sum(ed)OVER(ORDER BY t)ep,sum(cd)OVER(ORDER BY t)c,sum(md)OVER(ORDER BY t)m,sum(zd)OVER(ORDER BY t)memset FROM a)
SELECT (SELECT(b-a)/1e6 FROM frame) frame_ms,sum(CASE WHEN busy>0 THEN dt ELSE 0 END)/1e6 busy_ms,sum(CASE WHEN busy=0 THEN dt ELSE 0 END)/1e6 idle_ms,sum(CASE WHEN c>0 THEN dt ELSE 0 END)/1e6 compute_other_ms,sum(CASE WHEN n>0 THEN dt ELSE 0 END)/1e6 nccl_ms,sum(CASE WHEN ep>0 THEN dt ELSE 0 END)/1e6 hybrid_ep_mixed_ms,sum(CASE WHEN m>0 THEN dt ELSE 0 END)/1e6 memcpy_ms,sum(CASE WHEN n>0 AND c=0 AND ep=0 AND m=0 AND memset=0 THEN dt ELSE 0 END)/1e6 nccl_exclusive_ms,sum(CASE WHEN(n>0 OR ep>0)AND c=0 THEN dt ELSE 0 END)/1e6 comm_without_compute_ms FROM z;

-- integrity_complete [trace_cee2983a842c428a]
SELECT category,count(*) n,sum(dur)/1e6 sum_ms,sum(CASE WHEN dur<0 THEN 1 ELSE 0 END) incomplete,min(ts) first_ns,max(ts+dur) last_ns FROM slice GROUP BY category ORDER BY n DESC;

-- capacity_sync_context_complete [trace_cee2983a842c428a]
WITH c AS(SELECT id,ts,ts+dur en,dur FROM slice WHERE name='torchtitan::conditional_capacity_mask')
SELECT count(DISTINCT c.id) predicate_n,sum(s.dur)/1e6 nested_sync_ms,count(s.id) sync_n FROM c JOIN slice s ON s.ts>=c.ts AND s.ts+s.dur<=c.en WHERE s.name='cudaStreamSynchronize';

-- kernel_families_complete [trace_cee2983a842c428a]
SELECT CASE WHEN name GLOB 'ncclDevKernel*AllGather*' THEN 'NCCL AllGather' WHEN name GLOB 'ncclDevKernel*ReduceScatter*' THEN 'NCCL ReduceScatter' WHEN name GLOB 'ncclDevKernel*' THEN 'NCCL other' WHEN instr(name,'hybrid_ep::dispatch_kernel')>0 THEN 'HybridEP dispatch mixed' WHEN instr(name,'hybrid_ep::combine_kernel')>0 THEN 'HybridEP combine mixed' WHEN instr(name,'hybrid_ep::')>0 THEN 'HybridEP other mixed' WHEN instr(name,'deep_gemm::')>0 THEN 'DeepGEMM' WHEN name GLOB 'nvjet*' THEN 'nvjet GEMM' WHEN name GLOB '*log_softmax*' THEN 'CE' WHEN name GLOB '*quant*' THEN 'quantization' WHEN name GLOB '*cudnn*sdpa*' THEN 'cuDNN SDPA' WHEN name GLOB '*cutlass*norm*' THEN 'CuTe norms' ELSE 'other compute/unknown' END family,count(*) n,sum(dur)/1e6 sum_ms FROM slice WHERE category='kernel' GROUP BY 1 ORDER BY sum(dur)DESC;

-- copies_complete [trace_cee2983a842c428a]
SELECT name,count(*)n,sum(dur)/1e6 sum_ms,sum(CAST(EXTRACT_ARG(arg_set_id,'args.bytes') AS INT)) bytes FROM slice WHERE category='gpu_memcpy' GROUP BY name ORDER BY sum(dur)DESC;

-- ce_and_attention_complete [trace_cee2983a842c428a]
SELECT name,count(*) n,sum(dur)/1e6 sum_ms FROM slice WHERE category='kernel' AND(name GLOB '*log_softmax*' OR name GLOB '*nll_loss*' OR name GLOB '*cudnn*sdpa*' OR name GLOB '*delta*' OR name GLOB '*recompute_w*')GROUP BY name ORDER BY sum(dur)DESC LIMIT 25;

-- item_ancestors_complete [trace_cee2983a842c428a]
SELECT p.name parent,g.name grandparent,gg.name ancestor,count(*) n,sum(s.dur)/1e6 sum_ms,max(s.dur)/1e6 max_ms FROM slice s LEFT JOIN slice p ON p.id=s.parent_id LEFT JOIN slice g ON g.id=p.parent_id LEFT JOIN slice gg ON gg.id=g.parent_id WHERE s.name='aten::item' GROUP BY 1,2,3 ORDER BY sum(s.dur) DESC LIMIT 18;

-- loader_in_frame_complete [trace_cee2983a842c428a]
WITH f AS(SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation' AND name='task.profile_update.11.train_step.rank0') SELECT count(*) n,sum(s.dur)/1e6 host_sum_ms,min(s.ts-f.a)/1e6 first_ms,max(s.ts+s.dur-f.a)/1e6 last_ms,max(s.dur)/1e6 max_ms FROM slice s,f WHERE s.name GLOB 'enumerate(DataLoader)*' AND s.ts>=f.a AND s.ts+s.dur<=f.b;

-- markers_complete [trace_cee2983a842c428a]
SELECT category,name,count(*) n,min(ts) start_ns,max(ts+dur) end_ns,sum(dur)/1e6 sum_ms FROM slice WHERE name GLOB 'task.profile_update.*' OR name GLOB 'ProfilerStep*' GROUP BY 1,2;

-- stats_complete [trace_cee2983a842c428a]
SELECT name,severity,value FROM stats WHERE value!=0 AND severity IN('error','data_loss');

-- r1_capacity [trace_d64072c4a30ae977]
SELECT count(*) n,sum(s.dur)/1e6 host_sum_ms FROM slice s WHERE name='torchtitan::conditional_capacity_mask';

-- r1_ce [trace_d64072c4a30ae977]
SELECT name,count(*) n,sum(dur)/1e6 sum_ms FROM slice WHERE category='kernel'AND name GLOB '*log_softmax*'GROUP BY name;

-- r1_frame [trace_d64072c4a30ae977]
SELECT name,count(*) n,min(ts)start_ns,max(ts+dur)end_ns,sum(dur)/1e6 sum_ms FROM slice WHERE category='user_annotation' AND name GLOB 'task.profile_update.*'GROUP BY name;

-- r1_integrity [trace_d64072c4a30ae977]
SELECT name,severity,value FROM stats WHERE value!=0 AND severity IN('error','data_loss');

-- r1_loader [trace_d64072c4a30ae977]
SELECT count(*) n,sum(dur)/1e6 sum_ms,max(dur)/1e6 max_ms FROM slice WHERE category='user_annotation' AND name GLOB 'enumerate(DataLoader)*';

-- loader_calls [trace_6d2ec67f96a83162]
SELECT row_number()OVER(ORDER BY ts) call,(ts-(SELECT ts FROM slice WHERE category='user_annotation'AND name='task.profile_update.11.train_step.rank0'))/1e6 rel_ms,dur/1e6 dur_ms FROM slice WHERE category='user_annotation'AND name GLOB 'enumerate(DataLoader)*'ORDER BY ts;

-- gfsdp_gathers [trace_6d2ec67f96a83162]
SELECT name,count(*) n,sum(dur)/1e6 host_ms FROM slice WHERE category='user_annotation'AND name GLOB 'GFSDP::all_gather*'GROUP BY name ORDER BY name;

-- capacity_idle_overlap [trace_6d2ec67f96a83162]
WITH d AS(SELECT ts,ts+dur en,max(ts+dur)OVER(ORDER BY ts,id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prevmax FROM slice WHERE category IN('kernel','gpu_memcpy','gpu_memset')AND dur>0),g AS(SELECT prevmax a,ts b FROM d WHERE ts>prevmax),c AS(SELECT ts a,ts+dur b FROM slice WHERE name='torchtitan::conditional_capacity_mask') SELECT count(*) intersecting_gap_predicate_pairs,sum(min(c.b,g.b)-max(c.a,g.a))/1e6 gpu_idle_inside_capacity_ms FROM c JOIN g ON g.a<c.b AND g.b>c.a;

-- stats_lanes [trace_fb8d93262d864d34]
SELECT name,severity,value FROM stats WHERE value!=0 AND severity IN('error','data_loss');

-- integrity_lanes [trace_fb8d93262d864d34]
SELECT category,count(*) n,sum(dur)/1e6 sum_ms,sum(CASE WHEN dur<0 THEN 1 ELSE 0 END) incomplete,min(ts) first_ns,max(ts+dur) last_ns FROM slice GROUP BY category ORDER BY n DESC;

-- kernel_families_lanes [trace_fb8d93262d864d34]
SELECT CASE WHEN name GLOB 'ncclDevKernel*AllGather*' THEN 'NCCL AllGather' WHEN name GLOB 'ncclDevKernel*ReduceScatter*' THEN 'NCCL ReduceScatter' WHEN name GLOB 'ncclDevKernel*' THEN 'NCCL other' WHEN instr(name,'hybrid_ep::dispatch_kernel')>0 THEN 'HybridEP dispatch mixed' WHEN instr(name,'hybrid_ep::combine_kernel')>0 THEN 'HybridEP combine mixed' WHEN instr(name,'hybrid_ep::')>0 THEN 'HybridEP other mixed' WHEN instr(name,'deep_gemm::')>0 THEN 'DeepGEMM' WHEN name GLOB 'nvjet*' THEN 'nvjet GEMM' WHEN name GLOB '*log_softmax*' THEN 'CE' WHEN name GLOB '*quant*' THEN 'quantization' WHEN name GLOB '*cudnn*sdpa*' THEN 'cuDNN SDPA' WHEN name GLOB '*cutlass*norm*' THEN 'CuTe norms' ELSE 'other compute/unknown' END family,count(*) n,sum(dur)/1e6 sum_ms FROM slice WHERE category='kernel' GROUP BY 1 ORDER BY sum(dur)DESC;

-- scope_union_lanes [trace_fb8d93262d864d34]
WITH frame AS (SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation' AND name='task.profile_update.11.train_step.rank0'),
e AS (SELECT max(s.ts,a) ts,min(s.ts+s.dur,b) en,CASE WHEN category='kernel' AND name GLOB 'ncclDevKernel*' THEN 'nccl' WHEN category='kernel' AND instr(name,'hybrid_ep::')>0 THEN 'ep' WHEN category='kernel' THEN 'compute_other' ELSE category END fam FROM slice s,frame WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0 AND ts<b AND ts+dur>a),
p AS (SELECT ts t,fam,1 d FROM e UNION ALL SELECT en,fam,-1 FROM e UNION ALL SELECT a,'boundary',0 FROM frame UNION ALL SELECT b,'boundary',0 FROM frame),
a AS (SELECT t,sum(d) bd,sum(CASE WHEN fam='nccl' THEN d ELSE 0 END) nd,sum(CASE WHEN fam='ep' THEN d ELSE 0 END) ed,sum(CASE WHEN fam='compute_other' THEN d ELSE 0 END) cd,sum(CASE WHEN fam='gpu_memcpy' THEN d ELSE 0 END) md,sum(CASE WHEN fam='gpu_memset' THEN d ELSE 0 END) zd FROM p GROUP BY t),
z AS (SELECT t,lead(t)OVER(ORDER BY t)-t dt,sum(bd)OVER(ORDER BY t) busy,sum(nd)OVER(ORDER BY t)n,sum(ed)OVER(ORDER BY t)ep,sum(cd)OVER(ORDER BY t)c,sum(md)OVER(ORDER BY t)m,sum(zd)OVER(ORDER BY t)memset FROM a)
SELECT (SELECT(b-a)/1e6 FROM frame) frame_ms,sum(CASE WHEN busy>0 THEN dt ELSE 0 END)/1e6 busy_ms,sum(CASE WHEN busy=0 THEN dt ELSE 0 END)/1e6 idle_ms,sum(CASE WHEN c>0 THEN dt ELSE 0 END)/1e6 compute_other_ms,sum(CASE WHEN n>0 THEN dt ELSE 0 END)/1e6 nccl_ms,sum(CASE WHEN ep>0 THEN dt ELSE 0 END)/1e6 hybrid_ep_mixed_ms,sum(CASE WHEN m>0 THEN dt ELSE 0 END)/1e6 memcpy_ms,sum(CASE WHEN n>0 AND c=0 AND ep=0 AND m=0 AND memset=0 THEN dt ELSE 0 END)/1e6 nccl_exclusive_ms,sum(CASE WHEN(n>0 OR ep>0)AND c=0 THEN dt ELSE 0 END)/1e6 comm_without_compute_ms FROM z;

-- flow_coverage_lanes [trace_fb8d93262d864d34]
SELECT g.category,count(DISTINCT g.id)n FROM slice g JOIN flow f ON f.slice_in=g.id JOIN slice h ON h.id=f.slice_out WHERE g.category IN('kernel','gpu_memcpy','gpu_memset')AND h.category IN('cuda_runtime','cuda_driver')GROUP BY g.category;

-- queue_by_stream [trace_fb8d93262d864d34]
WITH g AS(SELECT id,category,ts,dur,CAST(EXTRACT_ARG(arg_set_id,'args.stream')AS INT)stream FROM slice WHERE category IN('kernel','gpu_memcpy','gpu_memset')),p AS(SELECT *,max(ts+dur)OVER(PARTITION BY stream ORDER BY ts,id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)prev_end FROM g),l AS(SELECT f.slice_in id,min(h.ts)launch_start,max(h.ts+h.dur)launch_end,count(*)links FROM flow f JOIN slice h ON h.id=f.slice_out WHERE h.category IN('cuda_runtime','cuda_driver')GROUP BY f.slice_in) SELECT stream,category,count(*)n,sum(CASE WHEN launch_end<=prev_end THEN 1 ELSE 0 END)already_queued_n,sum(CASE WHEN launch_start>prev_end THEN 1 ELSE 0 END)late_launch_n,sum(CASE WHEN links>1 THEN 1 ELSE 0 END)multiple_flows_n FROM p JOIN l USING(id)GROUP BY stream,category ORDER BY stream,category;

-- exposed_communication [trace_fb8d93262d864d34]
WITH f AS(SELECT ts a,ts+dur b FROM slice WHERE category='user_annotation'AND name='task.profile_update.11.train_step.rank0'),
e AS(SELECT ts,ts+dur en,CASE WHEN name GLOB 'ncclDevKernel*AllGather*'THEN'ag'WHEN name GLOB 'ncclDevKernel*ReduceScatter*'THEN'rs'WHEN name GLOB 'ncclDevKernel*'THEN'ar'WHEN name GLOB '*hybrid_ep::*'THEN'ep'ELSE'other'END fam FROM slice,f WHERE category IN('kernel','gpu_memcpy','gpu_memset')AND dur>0 AND ts>=a AND ts+dur<=b),
p AS(SELECT ts t,fam,1 d FROM e UNION ALL SELECT en,fam,-1 FROM e),
a AS(SELECT t,sum(CASE WHEN fam='ag'THEN d ELSE 0 END)da,sum(CASE WHEN fam='rs'THEN d ELSE 0 END)dr,sum(CASE WHEN fam='ar'THEN d ELSE 0 END)dn,sum(CASE WHEN fam='ep'THEN d ELSE 0 END)de,sum(CASE WHEN fam='other'THEN d ELSE 0 END)do FROM p GROUP BY t),
z AS(SELECT t,lead(t)OVER(ORDER BY t)-t dt,sum(da)OVER(ORDER BY t)ag,sum(dr)OVER(ORDER BY t)rs,sum(dn)OVER(ORDER BY t)ar,sum(de)OVER(ORDER BY t)ep,sum(do)OVER(ORDER BY t)o FROM a)
SELECT sum(CASE WHEN ag>0 AND o=0 AND ep=0 THEN dt ELSE 0 END)/1e6 ag_no_other_device_ms,sum(CASE WHEN rs>0 AND o=0 AND ep=0 THEN dt ELSE 0 END)/1e6 rs_no_other_device_ms,sum(CASE WHEN ar>0 AND o=0 AND ep=0 THEN dt ELSE 0 END)/1e6 ar_no_other_device_ms,sum(CASE WHEN ag>0 AND rs>0 THEN dt ELSE 0 END)/1e6 ag_rs_overlap_ms FROM z;

-- gdn_call_context [trace_fb8d93262d864d34]
SELECT p.name parent,q.name grandparent,r.name greatgrandparent,g.name,count(*)n,sum(g.dur)/1e6 sum_ms FROM slice g JOIN flow f ON f.slice_in=g.id JOIN slice h ON h.id=f.slice_out LEFT JOIN slice p ON p.id=h.parent_id LEFT JOIN slice q ON q.id=p.parent_id LEFT JOIN slice r ON r.id=q.parent_id WHERE g.category='kernel' AND(g.name='recompute_w_u_fwd_kernel'OR g.name='chunk_gated_delta_rule_fwd_kernel_h_blockdim64'OR g.name='chunk_gated_delta_rule_fwd_kkt_solve_kernel')GROUP BY 1,2,3,4 ORDER BY sum(g.dur)DESC LIMIT 22;

-- head_device_cost [trace_fb8d93262d864d34]
WITH l AS(SELECT slice_in,min(slice_out)host_id FROM flow GROUP BY slice_in),g AS(SELECT s.*,p.name parent,q.name grandparent,r.name greatgrandparent FROM slice s JOIN l ON l.slice_in=s.id JOIN slice h ON h.id=l.host_id LEFT JOIN slice p ON p.id=h.parent_id LEFT JOIN slice q ON q.id=p.parent_id LEFT JOIN slice r ON r.id=q.parent_id WHERE s.category='kernel')
SELECT name,parent,grandparent,count(*)n,sum(dur)/1e6 sum_ms FROM g WHERE (name='nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN'AND grandparent='aten::matmul')OR(name IN('nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_NTT','nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT')AND grandparent='MmBackward0')GROUP BY 1,2,3;

-- dense_swiglu_active [trace_fb8d93262d864d34]
SELECT name,count(*)n,sum(dur)/1e6 sum_ms FROM slice WHERE category='kernel'AND(name GLOB '*swiglu*'OR name GLOB '*z_loss*')GROUP BY name ORDER BY sum(dur)DESC LIMIT 15;

-- microbatch_stats [trace_fb8d93262d864d34]
SELECT count(*)n,min(dur)/1e6 min_ms,max(dur)/1e6 max_ms,avg(dur)/1e6 mean_ms FROM slice WHERE category='user_annotation'AND name='task.profile_update.11.microbatch.rank0';

-- metadata_other_sync [trace_fb8d93262d864d34]
SELECT p.name parent,q.name grandparent,count(*)n,sum(s.dur)/1e6 sum_ms,max(s.dur)/1e6 max_ms FROM slice s LEFT JOIN slice p ON p.id=s.parent_id LEFT JOIN slice q ON q.id=p.parent_id WHERE s.name='cudaStreamSynchronize'GROUP BY 1,2 ORDER BY sum(s.dur)DESC LIMIT 15;

-- gdn_replay_summary [trace_fb8d93262d864d34]
WITH l AS(SELECT slice_in,min(slice_out)h FROM flow GROUP BY slice_in),g AS(SELECT s.*,p.name parent,q.name grandparent FROM slice s JOIN l ON l.slice_in=s.id JOIN slice h ON h.id=l.h LEFT JOIN slice p ON p.id=h.parent_id LEFT JOIN slice q ON q.id=p.parent_id WHERE s.category='kernel')
SELECT name,parent,grandparent,count(*)n,sum(dur)/1e6 sum_ms FROM g WHERE name IN('recompute_w_u_fwd_kernel','chunk_gated_delta_rule_fwd_kernel_h_blockdim64','chunk_gated_delta_rule_fwd_kkt_solve_kernel')GROUP BY name,parent,grandparent ORDER BY name,parent,grandparent;

-- capacity_predicate_details [trace_fb8d93262d864d34]
SELECT s.name,count(*)n,sum(s.dur)/1e6 ms FROM slice s LEFT JOIN slice p ON p.id=s.parent_id LEFT JOIN slice q ON q.id=p.parent_id LEFT JOIN slice r ON r.id=q.parent_id WHERE s.name='torchtitan::conditional_capacity_mask' OR(s.name='aten::item'AND q.name='torchtitan::conditional_capacity_mask')OR(s.name='cudaStreamSynchronize'AND r.name='aten::is_nonzero'AND q.name='aten::item')GROUP BY s.name;

-- capacity_idle_overlap_lanes [trace_fb8d93262d864d34]
WITH d AS(SELECT ts,ts+dur en,max(ts+dur)OVER(ORDER BY ts,id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prevmax FROM slice WHERE category IN('kernel','gpu_memcpy','gpu_memset')AND dur>0),g AS(SELECT prevmax a,ts b FROM d WHERE ts>prevmax),c AS(SELECT ts a,ts+dur b FROM slice WHERE name='torchtitan::conditional_capacity_mask') SELECT count(*) intersecting_gap_predicate_pairs,sum(min(c.b,g.b)-max(c.a,g.a))/1e6 gpu_idle_inside_capacity_ms FROM c JOIN g ON g.a<c.b AND g.b>c.a;

-- attention_conversion [trace_fb8d93262d864d34]
SELECT p.name parent,q.name grandparent,count(*)n,sum(g.dur)/1e6 sum_ms FROM slice g JOIN flow f ON f.slice_in=g.id JOIN slice h ON h.id=f.slice_out LEFT JOIN slice p ON p.id=h.parent_id LEFT JOIN slice q ON q.id=p.parent_id WHERE g.category='kernel'AND(g.name GLOB '*CatArrayBatchedCopy*'OR g.name GLOB '*bfloat16_copy*')GROUP BY 1,2 ORDER BY sum(g.dur)DESC LIMIT 12;

-- kernel_gaps_lanes [trace_fb8d93262d864d34]
WITH r AS(SELECT id,ts,ts+dur en,max(ts+dur)OVER(ORDER BY ts,id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prevmax FROM slice WHERE category IN('kernel','gpu_memcpy','gpu_memset')AND dur>0),g AS(SELECT id,prevmax a,ts b FROM r WHERE ts>prevmax) SELECT g.a start_ns,g.b end_ns,(g.b-g.a)/1e6 gap_ms,s.name next_device,h.name launch,p.name parent,(h.ts-g.a)/1e6 launch_start_ms,(h.ts+h.dur-g.a)/1e6 launch_end_ms FROM g JOIN slice s USING(id)LEFT JOIN flow f ON f.slice_in=s.id LEFT JOIN slice h ON h.id=f.slice_out LEFT JOIN slice p ON p.id=h.parent_id ORDER BY gap_ms DESC LIMIT 12;

-- input_prefix [trace_fb8d93262d864d34]
WITH f AS(SELECT ts a FROM slice WHERE category='user_annotation'AND name='task.profile_update.11.train_step.rank0')SELECT s.name,(s.ts-a)/1e6 rel_ms,s.dur/1e6 ms FROM slice s,f WHERE s.ts>=a AND s.ts<a+1020000000 AND s.name IN('torchtitan::conditional_capacity_mask','aten::item','c10d::allreduce_','_c10d_functional::all_reduce','c10d::allreduce')ORDER BY s.ts;

-- gdn_total_device [trace_fb8d93262d864d34] FAILED
WITH launches AS(SELECT h.id,h.ts,h.dur,h.track_id,f.slice_in FROM slice h JOIN flow f ON f.slice_out=h.id WHERE h.category IN('cuda_runtime','cuda_driver')),roots AS(SELECT id,ts,dur,track_id,name FROM slice WHERE name IN('torchtitan::gdn_chunk','torchtitan::fla_vjp'))
SELECT r.name,count(DISTINCT r.id)cpu_calls,count(*)gpu_kernels,sum(g.dur)/1e6 kernel_sum_ms FROM roots r JOIN launches l ON l.track_id=r.track_id AND l.ts>=r.ts AND l.ts+l.dur<=r.ts+r.dur JOIN slice g ON g.id=l.slice_in WHERE g.category='kernel'GROUP BY r.name;

-- gdn_vjp_forward_kernels [trace_fb8d93262d864d34] FAILED
WITH l AS(SELECT slice_in,min(slice_out)h FROM flow GROUP BY slice_in),x AS(SELECT g.*,p.name parent,q.name grandparent FROM slice g JOIN l ON l.slice_in=g.id JOIN slice h ON h.id=l.h LEFT JOIN slice p ON p.id=h.parent_id LEFT JOIN slice q ON q.id=p.parent_id WHERE g.category='kernel') SELECT parent,grandparent,count(*)n,sum(dur)/1e6 kernel_sum_ms FROM x WHERE grandparent='torchtitan::fla_vjp' AND parent='ChunkGatedDeltaRuleFunction'GROUP BY parent,grandparent;
