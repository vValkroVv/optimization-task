"""Reconcile eight fresh MCP captures and retain bounded semantic evidence."""
import collections
import hashlib
import json
from pathlib import Path

from analyze import HERE, RAW, ROOT, brief, family, iv, merged, ms, overlap, owned, rows

TAGS = ('d0', 's', 'se', 'se-fused')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}


def counter_diff(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    return [{'identity': list(k) if isinstance(k, tuple) else k,
             'before': ca[k], 'after': cb[k]} for k in sorted(ca.keys() | cb.keys()) if ca[k] != cb[k]]


def process(tag, rank):
    path = RAW / f'{tag}-004-rank{rank}.mcp.json'
    q = json.loads(path.read_text())['queries']
    extra_path = RAW / f'{tag}-004-rank{rank}.extra.mcp.json'
    extra = json.loads(extra_path.read_text())['queries']
    result = json.loads((RAW / f'{tag}-004-rank{rank}.detail.json').read_text())
    es = sum([rows(c) for c in q['device']], [])
    launches = sum([rows(c) for c in extra['tail']], [])
    scopes = rows(q['scopes'])
    micros = [s for s in scopes if s['category'] == 'user_annotation' and 'microbatch' in s['name']]
    assert len(micros) == 3
    origin = round(result['trace_origin_us'] * 1000)
    rel = lambda t: (t-origin)/1e6
    original_path = ROOT / f'runs/schedule-embedding/artifacts/raw/schedule-embedding-{tag}-diagnostic-140-004/native/profiling/traces/iteration_21/rank{rank}_trace.json'
    original = json.loads(original_path.read_bytes())['traceEvents']
    expected = collections.Counter(e['cat'] for e in original if e.get('ph') == 'X')
    assert all(result['census'].get(cat) == n for cat,n in expected.items())
    result['all_original_complete_category_counts_match'] = dict(expected)
    original_kernels = {round(e['ts']*1000):e for e in original if e.get('cat')=='kernel'}
    result['owner_bf16_fill_launches'] = [{'name':e['name'], 'grid':original_kernels[e['ts']]['args']['grid'], 'block':original_kernels[e['ts']]['args']['block']} for e in es if '_OwnerLookupBackward' in (e['ancestors'] or '') and 'FillFunctor<c10::BFloat16>' in e['name']]
    # Every actual launch/copy/memset has one captured flow target. Synchronize
    # calls enqueue no device work and are excluded from the inverse census.
    work = [l for l in launches if 'Synchronize' not in l['name']]
    assert all(l['device_links'] == 1 for l in work)
    assert len(work) == len(es)
    drain = next(l for l in launches if l['name'] == 'cudaDeviceSynchronize' and l['ts'] >= micros[-1]['ts'] + micros[-1]['dur'])
    end = rows(extra['bounds'])[0]['end_ts']
    last = max(e['ts']+e['dur'] for e in es)
    assert last <= drain['ts']+drain['dur'] <= end
    assert all(e['dur'] >= 0 and e['launch_ts'] <= e['ts']+e['dur'] for e in es)
    tail = []
    for i, micro in enumerate(micros):
        selected = [e for e in es if micro['ts'] <= e['launch_ts'] < micro['ts']+micro['dur']]
        assert selected and any(e['name'].startswith('nccl') for e in selected)
        beyond = [e for e in selected if e['ts']+e['dur'] > micro['ts']+micro['dur']]
        assert beyond
        tail.append({'microbatch': i, 'device_count': len(selected),
                     'categories': dict(collections.Counter(e['category'] for e in selected)),
                     'host_end_ms': rel(micro['ts']+micro['dur']),
                     'last_device_end_ms': rel(max(e['ts']+e['dur'] for e in selected)),
                     'asynchronous_tail_events': len(beyond),
                     'tail_ms': (max(e['ts']+e['dur'] for e in selected)-micro['ts']-micro['dur'])/1e6})
    result['tail_coverage'] = {'status': 'PASS', 'inverse_launch_flow_count': len(work),
        'unmatched_work_launches': 0, 'microbatches': tail,
        'last_device_end_ms': rel(last), 'drain_start_ms': rel(drain['ts']),
        'drain_end_ms': rel(drain['ts']+drain['dur']), 'capture_end_ms': rel(end),
        'device_to_drain_end_margin_ms': (drain['ts']+drain['dur']-last)/1e6,
        'drain_to_capture_end_ms': (end-drain['ts']-drain['dur'])/1e6}
    result['sources'] = [ref(path), ref(extra_path)]
    for i, (micro, record) in enumerate(zip(micros, result['microbatches'])):
        dev = [e for e in es if micro['ts'] <= e['launch_ts'] < micro['ts']+micro['dur']]
        def compiled(module):
            pre = next(s for s in scopes if s['name'] == f'GFSDP::pre_backward ({module})' and micro['ts'] <= s['ts'] < micro['ts']+micro['dur'])
            cb = min((s for s in scopes if s['name'] == 'CompiledFunctionBackward' and s['ts'] > pre['ts']), key=lambda s:s['ts'])
            kernels = [e for e in dev if owned(e, cb) and family(e) == 'local_compute']
            assert kernels
            return cb, kernels
        cb, l7 = compiled('layers.7[fsdp]')
        _, ffn = compiled('mtp.layers.0.block.feed_forward')
        col = record['collectives']
        get = lambda mod, op: next(e for e in col[f'{op}:{mod}'] if op != 'all_gather' or 'forward_prefetch' not in e['ancestors'])
        head = get('lm_head', 'post_backward_reduce')
        ffng = get('mtp.layers.0.block.feed_forward', 'all_gather')
        parentg = get('mtp.layers.0', 'all_gather')
        ffnr = get('mtp.layers.0.block.feed_forward', 'post_backward_reduce')
        parentr = get('mtp.layers.0', 'post_backward_reduce')
        finalg = get('layers.7[fsdp]', 'all_gather')
        rs = [(origin+parentr['start_ms']*1e6, origin+parentr['end_ms']*1e6)]
        record['semantic_checks'] = {
            'both_mtp_ag_complete_before_head_rs': max(parentg['end_ms'],ffng['end_ms']) < head['start_ms'],
            'ffn_compiled_first_ms': rel(min(e['ts'] for e in ffn)),
            'ffn_compiled_first_minus_ag_end_ms': rel(min(e['ts'] for e in ffn))-ffng['end_ms'],
            'ffn_rs_overlap_local_compute_ms': overlap([(origin+ffnr['start_ms']*1e6,origin+ffnr['end_ms']*1e6)],iv([e for e in dev if family(e)=='local_compute'])),
            'l7_ag_overlap_non_l7_local_compute_ms': overlap([(origin+finalg['start_ms']*1e6,origin+finalg['end_ms']*1e6)],iv([e for e in dev if family(e)=='local_compute' and 'layers.7' not in (e['ancestors'] or '')])),
            'l7_ag_overlap_owned_mtp_prefix_status': 'UNESTABLISHED',
            'ffn_rs_overlap_owned_mtp_prefix_status': 'UNESTABLISHED',
            'norm_pack_before_parent_rs_end': all(e['start_ms']<parentr['end_ms'] for e in record['norm_packing']),
            'l7_compiled_scope_id_local_only': cb['id'],
            'l7_compiled_host_start_ms': rel(cb['ts']),
            'l7_compute_parent_rs_union_overlap_ms': overlap(rs,iv(l7)),
            'l7_compute_start_minus_parent_rs_end_ms': rel(min(e['ts'] for e in l7))-parentr['end_ms']}
        record['l7_first_flow_matched_kernels'] = [brief(e, origin) for e in sorted(l7,key=lambda e:e['ts'])[:3]]
        record['later_backward'] = {'local_compute_union_ms': ms(iv([e for e in dev if family(e)=='local_compute' and e['ts']>=min(x['ts'] for x in l7)])),
            'all_device_union_ms': ms(iv(dev)), 'host_microbatch_ms': micro['dur']/1e6}
        later = [e for e in dev if family(e)=='local_compute' and e['ts']>=min(x['ts'] for x in l7)]
        unions = merged(iv(later))
        gaps=[]
        for (_,end_local),(start_local,_) in zip(unions,unions[1:]):
            nxt=next(e for e in later if e['ts']==start_local)
            gaps.append({'start_ms':rel(end_local),'end_ms':rel(start_local),
                         'duration_ms':(start_local-end_local)/1e6,
                         'before_next_host_launch_ms':max(0,min(start_local,nxt['launch_ts'])-end_local)/1e6,
                         'nccl_activity_union_ms':overlap([(end_local,start_local)],iv([e for e in dev if family(e)=='nccl']))})
        record['later_backward']['no_local_compute_gap_sum_ms']=sum(g['duration_ms'] for g in gaps)
        record['later_backward']['largest_no_local_compute_gaps']=sorted(gaps,key=lambda g:g['duration_ms'],reverse=True)[:5]
    for emb in result['embedding']:
        emb['host_operations'] = [r for r in rows(extra['owner_host']) if r['owner_name']==emb['phase'] and next(s for s in scopes if s['id']==r['owner_scope'])['ts']>=micros[emb['microbatch']]['ts'] and next(s for s in scopes if s['id']==r['owner_scope'])['ts']<micros[emb['microbatch']]['ts']+micros[emb['microbatch']]['dur']]
        emb['gap_union_ms'] = emb['device_span_ms']-emb['device_busy_union_ms']
    return result


def main():
    results = {tag: {'ranks':[process(tag,r) for r in (0,1)]} for tag in TAGS}
    comparison = {}
    for before,after in zip(TAGS,TAGS[1:]):
        items=[]
        for rank in (0,1):
            a,b = results[before]['ranks'][rank], results[after]['ranks'][rank]
            for ea,eb in zip(a['embedding'],b['embedding']):
                device_id=lambda e:(e['category'],e['name'],e['input_elements'],e['output_elements'],e['dtype'],e['group_size'])
                host_id=lambda e:(e['category'],e['name'])
                items.append({'rank':rank,'microbatch':ea['microbatch'],'phase':ea['phase'],
                    'device_differences':counter_diff(map(device_id,ea['operations']),map(device_id,eb['operations'])),
                    'host_differences':counter_diff([host_id(e) for e in ea['host_operations'] for _ in range(e['n'])],[host_id(e) for e in eb['host_operations'] for _ in range(e['n'])]),
                    'before_span_ms':ea['device_span_ms'],'after_span_ms':eb['device_span_ms'],
                    'before_gap_ms':ea['gap_union_ms'],'after_gap_ms':eb['gap_union_ms']})
        comparison[before+'→'+after]=items
    payloads = {}
    for tag,result in results.items():
        payloads[tag] = []
        for rank in result['ranks']:
            payloads[tag].append([sorted((key,e['name'],e['input_elements'],e['output_elements'],e['dtype'],e['group_size']) for key,events in m['collectives'].items() for e in events) for m in rank['microbatches']])
    assert all(value == payloads['d0'] for value in payloads.values())
    out = {'scope':'Eight fresh update21 rank0/1 captures, three complete F+B microbatches; actual Perfetto MCP only. No native throughput or full-update stall claim.', 'comparisons':comparison,'variants':results}
    out['gfsdp_collective_counts_payloads_dtype_groups_equal'] = True
    detail=RAW/'comparison-004-detail.json';detail.write_text(json.dumps(out,indent=2)+'\n')
    compact=json.loads(json.dumps(out));compact['detail']=ref(detail)
    for variant in compact['variants'].values():
        for rank in variant['ranks']:
            for micro in rank['microbatches']:
                micro.pop('parent_rs_overlapped_operations')
                for events in list(micro['collectives'].values())+[micro['norm_packing']]:
                    for event in events:event.pop('ancestors')
            for emb in rank['embedding']:
                emb.pop('operations');emb.pop('host_operations')
    (HERE/'comparison-004.json').write_text(json.dumps(compact,indent=2)+'\n')
    for tag,result in results.items():
        (RAW/f'{tag}-004-detail.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({t:[r['tail_coverage'] for r in v['ranks']] for t,v in results.items()},indent=2))


if __name__=='__main__':
    main()
