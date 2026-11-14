"""Exclusive V2 kernel attribution; CPU launch ancestry, never GPU clock phase guesses."""
import gzip,json,collections,bisect,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
TRACE=ROOT/'artifacts/raw/v2-long-epoch-candidate-trace-001/native/profiling/traces/iteration_21/rank0_trace.json.gz'
es=json.load(gzip.open(TRACE))['traceEvents']
hosts=collections.defaultdict(list)
for e in es:
 if e.get('ph')=='X' and e.get('cat') in ('cpu_op','user_annotation'): hosts[(e['pid'],e['tid'])].append(e)
# Sweep each CPU thread to save enclosing ranges for CUDA launches.
launches={}
for key,h in hosts.items():
 items=[(e['ts'],0,e) for e in h]+[(e['ts'],1,e) for e in es if (e.get('pid'),e.get('tid'))==key and e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})]
 active=[]
 for ts,kind,e in sorted(items,key=lambda x:(x[0],x[1])):
  active=[a for a in active if a['ts']+a.get('dur',0)>=ts]
  if kind==0: active.append(e)
  else: launches[e['args']['correlation']]={'launch':e,'ancestors':sorted(active,key=lambda a:-a['dur'])}
records=[]
for i,e in enumerate(es):
 if e.get('cat')!='kernel':continue
 h=launches.get(e['args'].get('correlation'),{})
 aa=h.get('ancestors',[])
 records.append({'event_index':i,'name':e['name'],'ts':e['ts'],'dur_us':e['dur'],'stream':e['args'].get('stream'),'correlation':e['args'].get('correlation'),'launch_ts':h.get('launch',{}).get('ts'),'launch_tid':h.get('launch',{}).get('tid'),'ancestors':[{'name':a['name'],'ts':a['ts'],'dur':a['dur'],'args':a.get('args',{})} for a in aa]})
(ROOT/'v2-window-component-launches.json').write_text(json.dumps(records))
# Show first forward compiled graphs with ordered launches.
for g in [e for e in es if e.get('name')=='CompiledFunction' and e['tid']==661176][:12]:
 rr=[r for r in records if any(a['ts']==g['ts'] and a['name']==g['name'] for a in r['ancestors'])]
 print('\nGRAPH',g['args'].get('Sequence number'),len(rr),sum(r['dur_us'] for r in rr)/1000)
 for r in rr:print(round(r['dur_us']/1000,4),r['name'][:145], ' | '.join(a['name'][:45] for a in r['ancestors'][-3:]))

def names(r):return [a['name'] for a in r['ancestors']]
def has(r,s):return any(s in n for n in names(r))
def gemm(r):return any(s in r['name'] for s in ['nvjet','gemm','Kernel2<cutlass','splitKreduce']) and 'deep_gemm::transpose' not in r['name']
# Group matching forward and backward compiled graph calls by autograd sequence number.
groups=collections.defaultdict(list)
for r in records:
 g=next((a for a in r['ancestors'] if a['name'] in ('CompiledFunction','CompiledFunctionBackward')),None)
 if g:groups[(g['name'],g['ts'])].append(r)
for key,rr in groups.items():
 rr.sort(key=lambda r:r['launch_ts'])
 is_bwd=key[0]=='CompiledFunctionBackward'
 is_gdn=any('qkv_conv1d_unpack' in r['name'] for r in rr)
 is_moe=any(has(r,'hybridep::') for r in rr)
 family=('gdn' if is_gdn else 'mla') if any(gemm(r) for r in rr) else 'loss'
 # AOT backward graphs begin by replaying forward; first actual norm VJP marks gradient onset.
 boundary=next((r['launch_ts'] for r in rr if has(r,'torchtitan::cute_gated_norm_bwd')),float('inf')) if is_bwd and family!='loss' else -float('inf')
 fp=[r for r in rr if has(r,'torchtitan::fp8_dense_gemm')]
 prefix_n=6 if family=='gdn' else 4
 expected_n=prefix_n*(3 if is_bwd else 1)
 if family!='loss':assert len(fp)==expected_n,(key,family,len(fp),expected_n)
 for j,r in enumerate(fp):
  if j<prefix_n:r['gemm_component']=(family.upper()+' projections') if j<(3 if family=='gdn' else 1) else ('Shared expert GEMMs' if is_moe else 'Dense FFN GEMMs')
  elif not is_moe:r['gemm_component']='Dense FFN GEMMs' if j<prefix_n+6 else family.upper()+' projections'
  else:
   k=j-prefix_n
   r['gemm_component']='Shared expert GEMMs' if k<3 or (6<=k<9 if family=='gdn' else 4<=k<7) else family.upper()+' projections'
 for r in rr:
  r['graph_family']=family;r['graph_moe']=is_moe
  r['phase']='forward' if not is_bwd else ('recompute' if r['launch_ts']<boundary else 'backward')
  if is_bwd and (has(r,'torchtitan::cute_gated_norm_fwd') or has(r,'ChunkGatedDeltaRuleFunction') and not has(r,'ChunkGatedDeltaRuleFunctionBackward') or has(r,'LayerNormGatedFunction') and not has(r,'LayerNormGatedFunctionBackward') or r['name'].startswith(('recompute_w_u_fwd_kernel','chunk_gated_delta_rule_fwd_kernel_h','causal_conv1d_fwd_kernel','_qkv_conv1d_unpack_kernel'))):r['phase']='recompute'
for r in records:
 n=r['name'];ns=names(r)
 if 'phase' not in r:r['phase']='backward' if any(a.startswith('autograd::engine::evaluate_function:') for a in ns) or r['launch_tid']==662652 else 'forward'
 if n.startswith('nccl'):
  r['component']='GFSDP all-gather/reduce-scatter' if any(t in n for t in ['AllGather','ReduceScatter']) else 'Count/control all-reduce'
 elif any(t in n for t in ['hybrid_ep::','ag_nvl_kernel','rs_nvl_kernel']):r['component']='EP communication/synchronization'
 elif 'sm90_fp8_gemm' in n and 'deep_gemm::' in n:r['component']='Routed expert GEMMs'
 elif 'gemm_component' in r:r['component']=r['gemm_component']
 elif gemm(r):
  if r.get('graph_family')=='gdn':r['component']='Router GEMMs' if any(t in n for t in ['f32f32','simt_sgemm']) or ('float, float, float, float, false' in n) else 'GDN projections'
  elif r.get('graph_family')=='mla':r['component']='Router GEMMs' if r.get('graph_moe') and (any(t in n for t in ['f32f32','simt_sgemm']) or ('float, float, float, float, false' in n)) else 'MLA projections / MTP fusion'
  else:r['component']='Output head projection'
 elif any(t in n for t in ['log_softmax','nll_loss','prepare_softmax']):r['component']='Cross entropy / softmax'
 elif has(r,'torchtitan::gdn_chunk') or has(r,'ChunkGatedDeltaRuleFunction'):r['component']='GDN scan and scan backward'
 elif has(r,'_FusedQKVConv1dUnpackFn') or any(t in n for t in ['qkv_bwd_pack','causal_conv1d']):r['component']='GDN unpack / short convolution'
 elif 'cudnn' in n and ('flash_' in n or 'cudnn::fusion::' in n):r['component']='MLA attention kernel'
 elif any(t in n for t in ['quantiz','transpose','swiglu','row2col']) or any('quantiz' in a for a in ns):r['component']='Quantization / layout / SwiGLU'
 elif any(t in n for t in ['router_topk','gatherTopK','bitwise','scatter','capacity','tokens_per_expert','_build_m_indices','any_gt','any_bitwise']) or has(r,'hybridep::'):r['component']='Router / dispatch control helpers'
 elif has(r,'cute_gated_norm') or 'layer_norm_gated' in n:r['component']='Normalization'
 else:r['component']='Other pointwise / reduction / copies / embedding'
# Keep launch clock phase distinct from semantic forward: chunked CE interleaves both.
mbs=sorted(e['ts'] for e in es if e.get('name')=='task.profile_update.21.microbatch.rank0' and e.get('cat')=='user_annotation')
bwd_starts=[min(e['ts'] for e in es if e.get('name','').startswith('autograd::engine::evaluate_function') and start<=e['ts']<(mbs[i+1] if i+1<len(mbs) else float('inf'))) for i,start in enumerate(mbs)]
for r in records:
 i=bisect.bisect_right(mbs,r['launch_ts'])-1
 r['clock_phase']='before_microbatch' if i<0 else ('early_forward' if r['launch_ts']<bwd_starts[i] else 'backward_time')
 if has(r,'GFSDP::') or has(r,'FSDP::'):
  if not r['name'].startswith('nccl'):r['component']='GFSDP staging / casting / gradient accumulation'
 if has(r,'LayerNormGatedFunction') and not r['component'].startswith('GFSDP'):r['component']='Normalization'
 if r['name'].startswith('void cudnn::fusion::'):r['component']='MLA attention backward helpers'
 if has(r,'SplitBackward0') and has(r,'aten::cat'):r['component']='Autograd split-gradient concatenation'
summary=collections.defaultdict(lambda:collections.defaultdict(lambda:{'count':0,'sum_ms':0.0}))
for r in records:
 for stream in ['all',str(r['stream'])]:
  q=summary[stream][r['component']+'|'+r['phase']];q['count']+=1;q['sum_ms']+=r['dur_us']/1000
out={'trace':str(TRACE),'kernel_count':len(records),'kernel_sum_ms':sum(r['dur_us'] for r in records)/1000,'unmatched_host_launch_count':sum(r['launch_ts'] is None for r in records),'summary':summary,'limitations':['Kernel duration sums, not elapsed time; streams overlap. Full captured window including two kernels preceding first microbatch.','No Python stacks, module ranges, GEMM shapes or generated FX source present. Projection/shared/router GEMM splits are source-order inferences from repeated compiled graph calls, not direct module tags.','AOT prefix before first cute_gated_norm_bwd is recompute. Additional explicit forward calls and FLA internal forward kernels under backward are recompute. Fused kernels with mixed forward/backward prep after this point are kept in backward; exact semantic split is unavailable.','Quantization/layout/SwiGLU is separate from GEMMs; EP includes transport and synchronization kernels only. Router/dispatch helpers and quantization are separated from EP.','MLA projections category includes MTP fusion because module-free trace cannot directly separate every BF16 GEMM.']}
assert len(records)==8795
(ROOT/'v2-window-component-summary.json').write_text(json.dumps(out,indent=2))
(ROOT/'v2-window-component-kernels.json').write_text(json.dumps(records))
print('\nEXCLUSIVE SUMMARY')
for stream in ['all','7','24','23']:
 print('STREAM',stream)
 for comp in sorted({k.split('|')[0] for k in summary[stream]}):print(comp, *[round(summary[stream].get(comp+'|'+ph,{}).get('sum_ms',0),6) for ph in ['forward','recompute','backward']])
print(out['kernel_count'],out['kernel_sum_ms'],out['unmatched_host_launch_count'])
# Evidence metadata and arithmetic checks are part of the generated receipt.
import hashlib
out['trace_gzip_sha256']=hashlib.sha256(TRACE.read_bytes()).hexdigest()
source_root=ROOT.parent.parent/'gift'
source_files=['torchtitan/models/common/attention/linear/gdn.py','torchtitan/models/common/attention/linear/fla_custom_ops.py','torchtitan/models/common/attention/softmax/mla.py','torchtitan/models/common/moe/__init__.py','torchtitan/models/common/decoder_blocks/mtp.py','torchtitan/models/common/decoder_blocks/decoder.py']
out['source_sha256']={f:hashlib.sha256((source_root/f).read_bytes()).hexdigest() for f in source_files}
out['method']={'host_launch':'GPU correlation -> CUDA runtime/driver launch -> same-thread enclosing CPU ranges; all launches matched','gemm_mapping':'For each compiled graph: GDN FP8 forward order qkv/z/o then three FFN matrices; MLA FP8 forward gate then three FFN matrices. Backward ordered first shared/dense input gradients then attention input gradients, late FFN/attention weight gradients. Group sizes asserted for every compiled graph instance. BF16 GDN small gating GEMMs and FP32 router GEMMs separated by kernel family plus graph family; remaining MLA BF16 includes MTP fusion.','phase_caution':'Operational replay/VJP split is exact for explicitly tagged custom ops, source-order inference for compiled GEMM ownership, and approximate for fused mixed operators. It must not be advertised as an exact semantic forward/recompute/backward decomposition of every instruction.'}
out['stream_totals']={stream:{'count':sum(v['count'] for v in dd.values()),'sum_ms':sum(v['sum_ms'] for v in dd.values())} for stream,dd in summary.items()}
out['phase_totals']={ph:sum(v['sum_ms'] for k,v in summary['all'].items() if k.endswith('|'+ph)) for ph in ['forward','recompute','backward']}
first_mb_ts=min(e['ts'] for e in es if e.get('name')=='task.profile_update.21.microbatch.rank0')
out['before_first_microbatch']=[{'event_index':r['event_index'],'name':r['name'],'sum_ms':r['dur_us']/1000,'stream':r['stream']} for r in records if r['ts']<first_mb_ts]
assert abs(sum(out['phase_totals'].values())-out['kernel_sum_ms'])<1e-8
assert abs(out['stream_totals']['7']['sum_ms']-1146.581352)<1e-7
(ROOT/'v2-window-component-summary.json').write_text(json.dumps(out,indent=2))

out['clock_phase_cross_semantic_phase']={stream:{clock:{ph:sum(r['dur_us']/1000 for r in records if (stream=='all' or str(r['stream'])==stream) and r['clock_phase']==clock and r['phase']==ph) for ph in ['forward','recompute','backward']} for clock in ['before_microbatch','early_forward','backward_time']} for stream in ['all','7','24','23']}
out['source_sha256']['torchtitan/components/loss.py']=hashlib.sha256((source_root/'torchtitan/components/loss.py').read_bytes()).hexdigest()
out['limitations'].append('Original semantic forward includes first-time chunked lm_head/CE work after first autograd start; this is not backward recomputation. See loss.py:516-527,728-759. clock_phase_cross_semantic_phase reconciles the distinct definitions.')
(ROOT/'v2-window-component-summary.json').write_text(json.dumps(out,indent=2))
out['source_revision_assurance']={'base_revision':'3b6b8928ec4b15a529d07ee99e672f2fc75ef73a','manifest':'source-manifest.json','inspected_model_paths_vs_base':'git -C optimization-task diff BASE -- gift/torchtitan/models/common/attention/linear/gdn.py gift/torchtitan/models/common/attention/linear/fla_custom_ops.py gift/torchtitan/models/common/attention/softmax/mla.py gift/torchtitan/models/common/moe/__init__.py gift/torchtitan/models/common/decoder_blocks/mtp.py produced empty diff on 2026-11-14','loss_assurance':'git show BASE:gift/torchtitan/components/loss.py lines 513 onward independently confirms first-time chunk forward/backward interleaving. Current GFSDP loss implementation has later changes; no assertion that current entire loss.py equals executed source.'}
out['head_kernel_families']={n:{'count':sum(r['name']==n and r['component']=='Output head projection' for r in records),'sum_ms':sum(r['dur_us']/1000 for r in records if r['name']==n and r['component']=='Output head projection')} for n in sorted({r['name'] for r in records if r['component']=='Output head projection'})}
(ROOT/'v2-window-component-summary.json').write_text(json.dumps(out,indent=2))
