"""Summarize actual Perfetto MCP flow rows for D0 and future matched captures."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'artifacts/raw/trace-analysis'


def rows(result):
    assert not result.get('isError'), result
    result = json.loads(result['content'][0]['text'])
    assert not result.get('truncated'), result.keys()
    return [dict(zip(result['columns'], row)) for row in result['rows']]


def merged(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def iv(events):
    return [(e['ts'], e['ts'] + e['dur']) for e in events]


def ms(intervals):
    return sum(b-a for a, b in merged(intervals)) / 1e6


def overlap(a, b):
    return ms([(max(x, u), min(y, v)) for x, y in merged(a) for u, v in merged(b) if max(x, u) < min(y, v)])


def family(e):
    if e['category'] != 'kernel':
        return e['category']
    return 'nccl' if e['name'].startswith('nccl') else 'local_compute'


def owned(e, scope):
    # Scope IDs only join within one trace; cross-trace identity uses semantic names.
    return re.search(r'(?:^| > )' + str(scope['id']) + ':', e.get('ancestors') or '') is not None


def brief(e, origin):
    return {**{k: e[k] for k in ('name', 'stream', 'category', 'input_elements', 'output_elements', 'dtype', 'group_size', 'sequence')},
            'start_ms': (e['ts']-origin)/1e6, 'end_ms': (e['ts']+e['dur']-origin)/1e6,
            'duration_ms': e['dur']/1e6, 'host_launch_ms': (e['launch_ts']-origin)/1e6,
            'input_payload_bytes': e['input_elements'] * {'BFloat16': 2, 'Float': 4, 'Long': 8}.get(e['dtype'], 0) if e['input_elements'] is not None else None,
            'output_payload_bytes': e['output_elements'] * {'BFloat16': 2, 'Float': 4, 'Long': 8}.get(e['dtype'], 0) if e['output_elements'] is not None else None,
            'ancestors': e['ancestors']}


def summarize(rank, tag="d0", raw_path=None):
    q = json.load(open(RAW / f'{tag}-rank{rank}.mcp.json'))['queries']
    device = sum([rows(chunk) for chunk in q['device']], [])
    scopes = rows(q['scopes'])
    micros = [s for s in scopes if s['category'] == 'user_annotation' and 'microbatch' in s['name']]
    raw_path = raw_path or ROOT / f'runs/production-data-abcd/artifacts/raw/proddata-d-diagnostic-001/native/profiling/traces/iteration_21/rank{rank}_trace.json'
    raw = json.load(open(raw_path))['traceEvents']
    origin = round(min(e['ts'] for e in raw if 'ts' in e)*1000)
    census = {r['category']: r['n'] for r in rows(q['census'])}
    for cat in ('kernel', 'gpu_memcpy', 'gpu_memset', 'gpu_user_annotation'):
        assert census[cat] == sum(e.get('ph') == 'X' and e.get('cat') == cat for e in raw)
    assert not rows(q['errors'])
    flow = rows(q['flows'])[0]
    assert flow['endpoints'] == flow['distinct_device'] == flow['matching_correlation'] == len(device) == sum(census[c] for c in ('kernel', 'gpu_memcpy', 'gpu_memset'))
    result = {'rank': rank, 'trace_origin_us': origin/1000, 'census': census, 'errors': [], 'flows': flow, 'microbatches': [], 'embedding': []}
    cb = [s for s in scopes if s['name'] == 'CompiledFunctionBackward']
    for i, micro in enumerate(micros):
        es = [e for e in device if micro['ts'] <= e['launch_ts'] < micro['ts'] + micro['dur']]
        comp = [e for e in es if family(e) == 'local_compute']
        def select(module, operation):
            return [e for e in es if e['name'].startswith('nccl') and f'GFSDP::{operation} ({module})' in (e['ancestors'] or '')]
        def one(module, operation):
            events = select(module, operation)
            if operation == 'all_gather':
                events = [e for e in events if 'forward_prefetch' not in e['ancestors']]
            assert len(events) == 1, (module, operation, len(events))
            return events[0]
        def block(module):
            pre = [s for s in scopes if s['category'] == 'user_annotation' and s['name'] == f'GFSDP::pre_backward ({module})' and micro['ts'] <= s['ts'] < micro['ts']+micro['dur']]
            assert len(pre) == 1
            scope = min((s for s in cb if s['ts'] > pre[0]['ts']), key=lambda s:s['ts'])
            return [e for e in es if owned(e, scope) and family(e) == 'local_compute']
        one('lm_head', 'post_backward_reduce')
        ffng = one('mtp.layers.0.block.feed_forward', 'all_gather')
        ffnr = one('mtp.layers.0.block.feed_forward', 'post_backward_reduce')
        parentr = one('mtp.layers.0', 'post_backward_reduce')
        finalg = one('layers.7[fsdp]', 'all_gather')
        finalr = one('layers.7[fsdp]', 'post_backward_reduce')
        one('layers.6[fsdp]', 'all_gather')
        final = block('layers.7[fsdp]')
        layer6 = block('layers.6[fsdp]')
        # Separate GPU annotation occupancy, all stream activity and local compute.
        packing = [e for e in es if 'FSDP::post_backward_accumulate (norm)' in (e['ancestors'] or '')]
        previous = max((e for e in es if e['stream'] == 7 and e['ts']+e['dur'] <= ffng['ts']+ffng['dur']), key=lambda e:e['ts']+e['dur'])
        resume = min((e for e in es if e['stream'] == 7 and e['ts'] >= ffng['ts']+ffng['dur']), key=lambda e:e['ts'])
        parent_iv = iv([parentr])
        stream7 = [e for e in es if e['stream'] == 7]
        annotations = [s for s in rows(q['gpu_annotations']) if s['stream'] == 7]
        collectives = {f'{op}:{module}': [brief(e, origin) for e in select(module, op)] for module in ('lm_head', 'mtp.layers.0', 'mtp.layers.0.block.feed_forward', 'norm', 'layers.7[fsdp]', 'layers.7[efsdp]', 'layers.6[fsdp]', 'layers.5[fsdp]') for op in ('all_gather', 'post_backward_reduce')}
        result['microbatches'].append({'index': i, 'collectives': collectives,
            'head_to_ffn_gap_ms': (resume['ts']-previous['ts']-previous['dur'])/1e6,
            'available_suffix_end_ms': (previous['ts']+previous['dur']-origin)/1e6,
            'ffn_resume_ms': (resume['ts']-origin)/1e6,
            'ffn_rs_end_to_parent_rs_start_ms': (parentr['ts']-ffnr['ts']-ffnr['dur'])/1e6,
            'ffn_rs_end_to_final_dense_ag_start_ms': (finalg['ts']-ffnr['ts']-ffnr['dur'])/1e6,
            'parent_rs_overlapped_operations': [brief(e, origin) for e in es if e['ts'] < parentr['ts']+parentr['dur'] and e['ts']+e['dur'] > parentr['ts'] and not e['name'].startswith('nccl')],
            'parent_rs_overlap_local_compute_ms': overlap(parent_iv,iv(comp)),
            'parent_rs_overlap_all_stream7_device_events_ms': overlap(parent_iv,iv(stream7)),
            'parent_rs_overlap_stream7_gpu_annotations_ms': overlap(parent_iv,iv(annotations)),
            'parent_rs_overlap_final_layer_compiled_compute_ms': overlap(parent_iv,iv(final)),
            'norm_packing': [brief(e, origin) for e in packing],
            'final_layer_compiled_compute': {'first_ms': (min(e['ts'] for e in final)-origin)/1e6, 'last_ms': (max(e['ts']+e['dur'] for e in final)-origin)/1e6, 'count':len(final),'union_ms':ms(iv(final))},
            'layer6_compiled_compute': {'first_ms': (min(e['ts'] for e in layer6)-origin)/1e6,'union_ms':ms(iv(layer6))},
            'final_dense_rs_end_to_layer6_compute_ms': (min(e['ts'] for e in layer6)-finalr['ts']-finalr['dur'])/1e6,
            'device_first_ms': (min(e['ts'] for e in es)-origin)/1e6, 'device_last_ms': (max(e['ts']+e['dur'] for e in es)-origin)/1e6})
    for scope in [s for s in scopes if s['name'] in ('_OwnerLookup', '_OwnerLookupBackward')]:
        es = [e for e in device if owned(e, scope)]
        start = min(e['ts'] for e in es)
        end = max(e['ts']+e['dur'] for e in es)
        gaps = []
        unions = merged(iv(es))
        for (_, prev), (nxt, _) in zip(unions, unions[1:]):
            next_events = [e for e in es if e['ts'] == nxt]
            launch = min(e['launch_ts'] for e in next_events)
            gaps.append({'start_ms':(prev-origin)/1e6,'end_ms':(nxt-origin)/1e6,'gap_ms':(nxt-prev)/1e6,'before_next_host_launch_ms':max(0,min(nxt,launch)-prev)/1e6})
        result['embedding'].append({'phase':scope['name'],'microbatch':sum(s['ts']<scope['ts'] for s in micros)-1,
            'host_scope_ms':scope['dur']/1e6, 'device_start_ms':(start-origin)/1e6,'device_end_ms':(end-origin)/1e6,'device_span_ms':(end-start)/1e6,'device_busy_union_ms':ms(iv(es)),
            'families':{f:{'count':len([e for e in es if family(e)==f]),'sum_ms':sum(e['dur'] for e in es if family(e)==f)/1e6,'union_ms':ms(iv([e for e in es if family(e)==f]))} for f in sorted(set(map(family,es)))},
            'gaps':gaps,'host_waits':[x for x in rows(q['owner_host_waits']) if x['owner_scope']==scope['id']],
            'operations':[brief(e,origin) for e in es]})
    for a,b in zip(result['microbatches'],result['microbatches'][1:]):
        a['device_tail_to_next_microbatch_first_ms']=b['device_first_ms']-a['device_last_ms']
    result['kernel_window_ms']=(max(e['ts']+e['dur'] for e in device if e['category']=='kernel')-min(e['ts'] for e in device if e['category']=='kernel'))/1e6
    return result


if __name__ == '__main__':
    result={'scope':'Existing current D control; first three F+B microbatches of update 21 after 20 warmup; ranks 0 and 1 only. No speed or full-update critical-path claim.', 'timing':'Device times relative to earliest original trace timestamp; interval unions exclude GPU annotations. Host waits are host API spans, not device latency. Synthetic lane tids are not streams.', 'ranks':[summarize(r) for r in (0,1)]}
    detail = RAW / 'control-detail.json'
    detail.write_text(json.dumps(result, indent=2) + '\n')
    compact = json.loads(json.dumps(result))
    compact['detail'] = {'path': str(detail.relative_to(ROOT)), 'sha256': hashlib.sha256(detail.read_bytes()).hexdigest()}
    for rank in compact['ranks']:
        for micro in rank['microbatches']:
            micro.pop('parent_rs_overlapped_operations')
            for events in list(micro['collectives'].values()) + [micro['norm_packing']]:
                for event in events:
                    event.pop('ancestors')
        for embedding in rank['embedding']:
            embedding.pop('operations')
    (HERE/'control.json').write_text(json.dumps(compact,indent=2)+'\n')
    print(json.dumps([{'rank':r['rank'],'kernel_window_ms':r['kernel_window_ms'],'head_gaps_ms':[m['head_to_ffn_gap_ms'] for m in r['microbatches']],'parent_rs_compute_overlap_ms':[m['parent_rs_overlap_local_compute_ms'] for m in r['microbatches']],'embedding_span_ms':sum(e['device_span_ms'] for e in r['embedding'])} for r in result['ranks']],indent=2))
