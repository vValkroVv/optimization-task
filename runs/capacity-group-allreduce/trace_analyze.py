"""Build the scoped capacity mechanism result from saved actual MCP queries."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAM = Path(__file__).resolve().parent
OUT = CAM / 'artifacts/raw/trace-analysis'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(result):
    assert not result.get('isError'), result
    text = json.loads(result['content'][0]['text'])
    assert not text.get('truncated') and not text.get('string_truncated'), text
    return [dict(zip(text['columns'], row)) for row in text['rows']]

historical = json.loads((OUT / 'historical-originals.mcp.json').read_text())
identities = json.loads((ROOT / 'runs/production-data-abcd/artifacts/raw/trace-analysis/identities.json').read_text())
result = {'status': 'complete', 'method': 'Actual MCP Perfetto original import, then annotation-preserving display-lane derivative; bounded saved SQL.', 'scope': 'Update 21 after 20 warmup optimizer updates, ranks 0/1, three complete F+B microbatches each. All-eight-rank accumulation32/update completion is owned by the native receipt, outside this trace.', 'limitations': ['No full optimizer update or optimizer is captured.', 'Historical D is compatible mechanism context, not a matched timing comparison.', 'Kernel names and host/device flows support placement; names alone do not identify every fused arithmetic operation.', 'Host API spans, kernel duration sums, and communication payload sizes do not establish critical-path latency or speedup.', 'Only ranks0/1 are traced; complete inter-stream wait dependency causality is unavailable.'], 'ranks': []}
for variant in ('historical', 'candidate'):
    for rank in (0, 1):
        evidence = json.loads((OUT / f'{variant}-rank{rank}.mcp.json').read_text())
        if variant == 'historical':
            identity = next(x for x in identities if x['variant'] == 'D' and x['rank'] == rank)
            original_import = historical['originals'][str(rank)]
            neighbors = rows(json.loads((OUT / f'{variant}-rank{rank}.neighbors.mcp.json').read_text()))
        else:
            identity = json.loads((OUT / f'candidate-rank{rank}.lanes.identity.json').read_text())
            original_import = json.loads((OUT / f'candidate-rank{rank}.original.mcp.json').read_text())
            neighbors = rows(evidence['capacity_neighbors'])
        assert sha(identity['original']) == identity['original_sha256']
        assert sha(identity['derivative']) == identity['derivative_sha256']
        original = json.loads(Path(identity['original']).read_bytes())
        derivative = json.loads(Path(identity['derivative']).read_bytes())
        oe, de = original['traceEvents'], derivative['traceEvents']
        assert len(oe) == len(de)
        assert {k:v for k,v in original.items() if k != 'traceEvents'} == {k:v for k,v in derivative.items() if k != 'traceEvents'}
        for a, b in zip(oe, de):
            assert {k:v for k,v in a.items() if k != 'tid'} == {k:v for k,v in b.items() if k != 'tid'}
            if a.get('tid') != b.get('tid'):
                assert a.get('ph') == 'f' or (a.get('ph') == 'X' and a.get('cat') in ('kernel','gpu_memcpy','gpu_memset','gpu_user_annotation'))
        census = collections.Counter(e.get('cat') for e in oe if e.get('ph') == 'X')
        imported = {x['category']:x['n'] for x in rows(evidence['census'])}
        assert all(imported.get(k) == n for k,n in census.items())
        assert not rows(evidence['errors'])
        flows = rows(evidence['flows'])[0]
        device_count = sum(census[k] for k in ('kernel','gpu_memcpy','gpu_memset'))
        assert flows['endpoints'] == flows['distinct_device'] == flows['matching_correlation'] == device_count
        scopes = rows(evidence['scopes'])
        micro = [s for s in scopes if s['category']=='user_annotation' and '.microbatch.' in s['name']]
        assert len(micro) == 3
        collectives = rows(evidence['collectives'])
        n = 8 if variant == 'candidate' else 128
        cap = [c for c in collectives if 'AllReduce' in c['name'] and c['dtype']=='Long' and c['input_elements']==n]
        assert len(cap)==39
        assert all(c['output_elements']==n and c['group_size']==8 for c in cap)
        assert not [c for c in collectives if 'AllReduce' in c['name'] and c['dtype']=='Long' and c['input_elements']==(128 if n==8 else 8)]
        record = {'variant':variant,'rank':rank,'identity':{k:identity[k] for k in ('original','original_sha256','derivative','derivative_sha256')}, 'reconciliation':{'original_errors': rows(original_import['errors']), 'original_census':rows(original_import['census']), 'raw_complete_event_census':dict(census), 'derivative_census':rows(evidence['census']), 'derivative_errors':[], 'all_event_fields_except_permitted_tid_preserved':True, 'device_flows':flows, 'gpu_annotations':census['gpu_user_annotation'], 'gpu_post_backward_annotations':sum(e.get('cat')=='gpu_user_annotation' and 'post_backward' in e.get('name','') for e in oe)}, 'microbatches':[], 'capacity_collectives':[], 'capacity_total_input_bytes':len(cap)*n*8, 'capacity_kernel_sum_ms':sum(c['dur'] for c in cap)/1e6, 'host_apis_entire_capture':rows(evidence['host']), 'compiled_forward_scopes':sum(s['name']=='CompiledFunction' for s in scopes),'compiled_backward_scopes':sum(s['name']=='CompiledFunctionBackward' for s in scopes),'collective_order_signature':[(c['name'],c['dtype'],c['input_elements'],c['output_elements'],c['group_size'],c['sequence'])for c in collectives]}
        for index,m in enumerate(micro):
            calls=[c for c in cap if m['ts']<=c['launch_ts']<m['ts']+m['dur']]
            backward=sum('CompiledFunctionBackward' in c['ancestors'] for c in calls)
            assert len(calls)==13 and backward==6
            record['microbatches'].append({'index':index,'scope':m,'capacity_collectives':len(calls),'forward':len(calls)-backward,'backward_recompute':backward})
        for c in cap:
            near=[k for k in neighbors if k['capacity_id']==c['id'] and k['id']!=c['id']]
            before=sorted((k for k in near if k['launch_ts']<c['launch_ts']),key=lambda k:k['launch_ts'])
            after=sorted((k for k in near if k['launch_ts']>c['launch_ts']),key=lambda k:k['launch_ts'])
            record['capacity_collectives'].append({**c,'input_bytes':n*8,'backward_recompute':'CompiledFunctionBackward' in c['ancestors'],'previous_launches':before[-3:],'next_launches':after[:3]})
        result['ranks'].append(record)
for variant in ('historical','candidate'):
    a,b=[r for r in result['ranks'] if r['variant']==variant]
    rank_order = lambda sig: [(name, dtype, group, seq) for name, dtype, _, _, group, seq in sig]
    assert rank_order(a['collective_order_signature']) == rank_order(b['collective_order_signature']), variant
for rank in (0,1):
    a,b=[r for r in result['ranks'] if r['rank']==rank]
    normalize=lambda sig:[(name,dtype,group)for name,dtype,n,out,group,_ in sig]
    assert normalize(a['collective_order_signature'])==normalize(b['collective_order_signature'])
result['collective_order_verified']='Every imported NCCL kernel has matching ordered name/dtype/group-size/sequence across ranks0/1 within each run. Historical/candidate ordered collective kinds/dtypes/groups match. Capacity payloads are checked separately at128 versus8; variable owner SendRecv payloads are not required equal. Numeric sequence IDs are run-local.'
result['artifacts']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for p in sorted(OUT.glob('*.json')) if not p.name.endswith('.lanes.json')]
(CAM/'trace-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: four originals/derivatives, event invariants/category census/flows; 39 capacity collectives per rank, 7F+6B per microbatch; rank and historical/candidate collective order.')
