import json,pathlib,collections,gc
OUT=pathlib.Path('runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A/analysis/007');a=json.loads((OUT/'replay-counts.json').read_text());out=[]
for r in a:
 es=json.load(open(r['trace']))['traceEvents'];steps=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#')],key=lambda e:e['ts']);groups=[e for e in es if 'Call CompiledFxGraph' in e.get('name','')];apis=[e for e in es if e.get('name') in ['cudaGraphLaunch','cudaStreamBeginCapture','cudaGraphInstantiateWithFlags','cuLaunchKernel','cuLaunchKernelEx','cudaLaunchKernel','cudaLaunchKernelExC']];rows={};kernelids=collections.defaultdict(set)
 for e in es:
  if e.get('cat')=='kernel' and e.get('args',{}).get('graph id'):kernelids[e['args'].get('correlation')].add(e['args']['graph id'])
 for e in apis:
  step=next((s for s in steps if s['ts']<=e['ts']<s['ts']+s['dur']),None)
  if step is None:continue
  row=rows.setdefault(step['name'],{'span_us':step['dur'],'apis':{},'phase':{}});n=e['name'];api=row['apis'].setdefault(n,{'count':0,'sum_us':0});api['count']+=1;api['sum_us']+=e['dur']
  if n in ['cudaGraphLaunch','cudaStreamBeginCapture']:
   parent=next((g for g in groups if g['tid']==e['tid'] and g['ts']<=e['ts']<g['ts']+g['dur']),None)
   key=parent['name'] if parent else 'unknown';v=row['phase'].setdefault(key,{'launches':0,'captures':0,'graph_ids':set()});v['launches' if n=='cudaGraphLaunch' else 'captures']+=1
   if n=='cudaGraphLaunch':v['graph_ids'].update(kernelids[e['args']['correlation']])
 for row in rows.values():
  for v in row['phase'].values():v['graph_ids']=sorted(v['graph_ids'])
 out.append({'rank':r['rank'],'steps':rows});(OUT/'phase-counts.json').write_text(json.dumps(out,indent=2)+'\n');print(r['rank'],[(s,[(v['launches'],v['captures'],len(v['graph_ids'])) for v in row['phase'].values()]) for s,row in rows.items()],flush=True);del es;gc.collect()
