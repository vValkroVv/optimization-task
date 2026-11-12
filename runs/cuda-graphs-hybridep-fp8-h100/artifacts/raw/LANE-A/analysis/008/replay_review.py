"""Bounded offline replay attribution; stdlib, one native rank trace in memory."""
import collections,json,pathlib,re,gc
ROOT=pathlib.Path('runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A')
OUT=ROOT/'analysis/008'
results=json.loads((OUT/'replay-counts.json').read_text()) if (OUT/'replay-counts.json').exists() else []
for rank in range(len(results),8):
 path=ROOT/f'attempts/008/native/profiling/traces/iteration_24/rank{rank}_trace.json'
 events=json.load(path.open())['traceEvents']
 launches=[e for e in events if 'GraphLaunch' in e.get('name','')]
 parents=[e for e in events if e.get('ph')=='X' and ('Call CompiledFxGraph' in e.get('name','') or e.get('name','').startswith('ProfilerStep#') or e.get('name','').startswith('Torch-Compiled Region') or e.get('name')=='CompiledFunctionBackward')]
 kernels=collections.defaultdict(list)
 for e in events:
  if e.get('cat')=='kernel':kernels[e['args'].get('correlation')].append(e)
 groups={}
 capture_apis=[e for e in events if e.get('name') in ['cudaStreamBeginCapture','cudaGraphInstantiateWithFlags']]
 for e in launches:
  enc=sorted([p for p in parents if p['tid']==e['tid'] and p['ts']<=e['ts'] and p['ts']+p['dur']>=e['ts']+e['dur']],key=lambda p:p['dur'])
  comp=next((p['name'] for p in enc if 'CompiledFxGraph' in p['name']),'unknown')
  region=next((p['name'] for p in enc if 'Torch-Compiled Region' in p['name']), 'backward' if any(p['name']=='CompiledFunctionBackward' for p in enc) else 'unknown')
  step=next((p['name'] for p in enc if 'ProfilerStep' in p['name']),'unknown')
  key=comp+' | '+region
  g=groups.setdefault(key,{'launches':0,'by_step':collections.Counter(),'kernels':collections.Counter(),'graph_ids':set(),'kernel_count':0,'kernel_sum_us':0,'launch_api_sum_us':0,'capture_count':0,'instantiate_count':0})
  g['launches']+=1;g['by_step'][step]+=1;g['launch_api_sum_us']+=e['dur']
  for k in kernels[e['args']['correlation']]:
   g['kernels'][k['name']]+=1;g['graph_ids'].add(k['args'].get('graph id'));g['kernel_count']+=1;g['kernel_sum_us']+=k['dur']
  # Captures in this compiled call, counted only on the first launch of a given parent.
  if comp!='unknown':
   parent=next(p for p in enc if 'CompiledFxGraph' in p['name'])
   if not parent.get('_counted'):
    parent['_counted']=True
    for api in capture_apis:
     if api.get('name') in ['cudaStreamBeginCapture','cudaGraphInstantiateWithFlags'] and api['tid']==parent['tid'] and parent['ts']<=api['ts']<parent['ts']+parent['dur']:
      g['capture_count' if api['name']=='cudaStreamBeginCapture' else 'instantiate_count']+=1
 for g in groups.values():g['graph_ids']=sorted(g['graph_ids']);g['kernels']=dict(g['kernels']);g['by_step']=dict(g['by_step'])
 api={}
 for e in events:
  if e.get('cat') in ['cuda_runtime','cuda_driver'] and any(w in e.get('name','') for w in ['Graph','Capture','LaunchKernel']):
   a=api.setdefault(e['name'],{'count':0,'sum_us':0,'max_us':0});a['count']+=1;a['sum_us']+=e.get('dur',0);a['max_us']=max(a['max_us'],e.get('dur',0))
 logs=next((ROOT/'attempts/008/torchrun').glob(f'*/attempt_0/{rank}/stderr.log'))
 counts=collections.Counter();last=[];first_records={};partitions=[]
 with logs.open() as f:
  for ln,line in enumerate(f,1):
   for token in ['Re-recording function=','Recording function=','Running warmup function=','skipping cudagraphs']:
    if token in line:
     m=re.search(r'\[([0-9]+/[0-9]+)\].*?'+re.escape(token)+r'(partition_\d+), mode=(\w+)',line)
     key=token+((' '.join(m.groups())) if m else '')
     counts[key]+=1
     if token=='Recording function=':first_records.setdefault(key,{'line':ln,'text':line.strip()[:600]})
     if token=='Re-recording function=':last.append({'line':ln,'text':line.strip()[:300]});last=last[-2:]
   if 'Created ' in line and 'graph partitions' in line:partitions.append({'line':ln,'text':line.strip()})
 results.append({'rank':rank,'trace':str(path),'groups':groups,'apis':api,'log':str(logs),'log_counts':dict(counts),'last_rerecords':last,'first_records':first_records,'partition_summaries':partitions})
 (OUT/'replay-counts.json').write_text(json.dumps(results,indent=2)+'\n')
 print(rank,'launches',len(launches),'captures',api.get('cudaStreamBeginCapture'), 'groups',[(k[:90],v['launches'],v['capture_count']) for k,v in groups.items()],flush=True)
 del events,parents,kernels,launches;gc.collect()
