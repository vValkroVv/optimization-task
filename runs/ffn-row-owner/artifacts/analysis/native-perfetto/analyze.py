import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / 'runs/ffn-row-owner/artifacts/analysis/native-perfetto'
RAW = ROOT / 'runs/ffn-row-owner/artifacts/raw/ffn-row-owner-diagnostic-001'

def union(intervals):
    total = 0
    end = -1
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, end))
        end = max(end, hi)
    return total / 1e6

def rows(table):
    return [dict(zip(table['columns'], r)) for r in table['rows']]

def owned(events, scope):
    return [e for e in events if re.search(r'(?:^| > )' + str(scope['id']) + ':', e['ancestors'] or '')]

def brief(e):
    return {k:e[k] for k in ['id','ts','dur','name','stream','launch_id','launch_ts','ancestors'] if k in e}

def end(e):
    return e['ts'] + e['dur']

result = {'ranks': []}
for rank in [0, 1]:
    data = json.loads((OUT / f'rank{rank}.mcp.json').read_text())
    events = [e for part in data['device'] for e in rows(part)]
    scopes = rows(data['scopes'])
    comp = [s for s in scopes if s['name'] == 'CompiledFunctionBackward']
    fwd = [s for s in scopes if s['name'] == '_OwnerLookup']
    bwd = [s for s in scopes if s['name'] == '_OwnerLookupBackward']
    micro = [s for s in scopes if '.microbatch.' in s['name']]
    assert len(fwd) == len(bwd) == len(micro) == 3
    rr = {'rank':rank, 'census':data['census'], 'flows':data['flows'], 'errors':data['errors'], 'microbatches':micro, 'ffn':[], 'owner':[], 'reuse':[]}
    assert len(events) == data['flows']['rows'][0][0] == data['flows']['rows'][0][1] == data['flows']['rows'][0][2]
    for group, parent in [('layers.0.feed_forward','layers.0'),('mtp.layers.0.block.feed_forward','mtp.layers.0')]:
        reduced = [e for e in events if f'GFSDP::post_backward_reduce ({group})' in (e['ancestors'] or '') and 'ncclDevKernel_ReduceScatter' in e['name']]
        assert len(reduced)==3
        for mb, rs in enumerate(reduced):
            prior = max((s for s in comp if s['ts'] < rs['launch_ts']),key=lambda s:s['ts'])
            # Parent may already be submitted before NCCL launch. Select the FFN
            # by its three TNT gradients, rather than host timestamp adjacency.
            candidates = [s for s in comp if s['ts'] < rs['launch_ts']]
            ffns = [(s,owned(events,s)) for s in candidates]
            ffns = [(s,es) for s,es in ffns if len([e for e in es if 'Avec128_Bvec128_TNT' in e['name']])==3]
            fs, fe = max(ffns,key=lambda pair:pair[0]['ts'])
            ps = min((s for s in comp if s['ts'] > fs['ts']),key=lambda s:s['ts'])
            pe = [e for e in owned(events,ps) if e['category']=='kernel' and not e['name'].startswith('nccl')]
            wg = [e for e in fe if 'Avec128_Bvec128_TNT' in e['name']]
            di = [e for e in fe if e['name']=='triton_poi_fused_add_3']
            pack = [e for e in events if f'FSDP::post_backward_accumulate ({group})' in (e['ancestors']or'') and 'aten::_foreach_copy_' in e['ancestors'] and fs['ts']<=e['launch_ts']<rs['launch_ts']]
            assert len(pack)==1 and len(di)==1 and len(wg)==3
            prs = min((e for e in events if f'GFSDP::post_backward_reduce ({parent})' in (e['ancestors']or'') and e['launch_ts']>rs['launch_ts'] and 'ncclDevKernel_ReduceScatter' in e['name']),key=lambda e:e['launch_ts'])
            overlap = union([(max(rs['ts'],e['ts']),min(end(rs),end(e))) for e in pe if max(rs['ts'],e['ts'])<min(end(rs),end(e))])
            ready = max(end(e) for e in wg+di)
            assert ready<=pack[0]['ts'] and end(pack[0])<=rs['ts'] and rs['ts']<prs['ts'] and overlap>0
            assert rs['input_elements']==134479872 and rs['dtype']=='Float' and rs['group_size']==8
            rr['ffn'].append({'group':group,'microbatch':mb,'compiled_ffn_scope':fs,'compiled_parent_scope':ps,'dInput':brief(di[0]),'wgrad_w2_w3_w1':[brief(e) for e in wg],'pack':brief(pack[0]),'rs':brief(rs),'parent_rs':brief(prs),'payload':{k:rs[k] for k in ['input_elements','output_elements','dtype','group_size','sequence']},'parent_payload':{k:prs[k] for k in ['input_elements','output_elements','dtype','group_size','sequence']},'parent_compute_end_ns':max(end(e) for e in pe),'rs_ms':rs['dur']/1e6,'parent_compute_overlap_ms':overlap,'coverage_pct':overlap/(rs['dur']/1e6)*100,'readiness_order_pass':True})
    heads = [e for e in events if 'GFSDP::post_backward_reduce (lm_head)' in (e['ancestors']or'') and 'ncclDevKernel_ReduceScatter' in e['name']]
    assert len(heads)==3
    for mb,(fw,bw) in enumerate(zip(fwd,bwd)):
        fs,bs=owned(events,fw),owned(events,bw)
        count=[e for e in fs if e['input_elements']==9]
        send=[e for e in fs+bs if 'ncclDevKernel_SendRecv' in e['name']]
        dense=[e for e in bs if 'compute_grad_weight<c10::BFloat16, long>' in e['name'] or 'sum_and_scatter<c10::BFloat16, long>' in e['name']]
        scratch=[e for e in fs if 'aten::index_select' in (e['ancestors']or'')]
        # Rank 1 owns no requested rows here: its first scratch write is the
        # value exchange receive, not reconstruction into the owned output.
        first_scratch = scratch[0] if send[0]['output_elements'] else send[1]
        assert len(count)==1 and len(send)==3 and len(dense)==2
        rr['owner'].append({'microbatch':mb,'forward_scope':fw,'backward_scope':bw,'forward_device': [brief(e) for e in fs],'backward_device':[brief(e) for e in bs],'counts_payload':{k:count[0][k] for k in ['input_elements','output_elements','dtype','group_size']},'routed_transports':[dict(brief(e),payload={k:e[k] for k in ['input_elements','output_elements','dtype','group_size','sequence']}) for e in send],'native_bf16_coalescing':[brief(e) for e in dense],'forward_first_scratch_write':brief(first_scratch),'owner_last_device_ns':max(end(e) for e in bs),'head_reuse_pack':[brief(e) for e in events if 'FSDP::post_backward_accumulate (lm_head)' in (e['ancestors']or'') and 'aten::_foreach_copy_' in (e['ancestors']or'') and (mb+1==3 or e['launch_ts']<fwd[mb+1]['ts']) and e['launch_ts']>fw['ts']]})
        if mb:
            prior_backward=owned(events,bwd[mb-1])
            next_head_pack=min((e for e in events if 'FSDP::post_backward_accumulate (lm_head)' in (e['ancestors']or'') and 'aten::_foreach_copy_' in (e['ancestors']or'') and e['launch_ts']>fw['ts']),key=lambda e:e['ts'])
            rr['reuse'].append({'boundary':f'{mb-1}->{mb}','prior_head_rs':brief(heads[mb-1]),'next_owner_first_scratch_write':brief(first_scratch),'head_rs_to_next_owner_write_ms':(first_scratch['ts']-end(heads[mb-1]))/1e6,'prior_owner_last_consumer':brief(max(prior_backward,key=end)),'next_head_first_pack':brief(next_head_pack),'owner_last_consumer_to_head_reuse_ms':(next_head_pack['ts']-max(end(e) for e in prior_backward))/1e6})
    rr['fulltable_collectives']=[dict(brief(e),input_elements=e['input_elements'],output_elements=e['output_elements'],dtype=e['dtype']) for e in events if e['input_elements'] in [65667072,525336576] and e['name'].startswith('nccl')]
    assert not any('tok_embeddings' in (e['ancestors']or'') for e in events)
    result['ranks'].append(rr)
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for rr in result['ranks']:
    print('RANK',rr['rank'])
    for f in rr['ffn']:print(f['group'],f['microbatch'],round(f['rs_ms'],6),round(f['parent_compute_overlap_ms'],6),round(f['coverage_pct'],3))
    print('reuse',[(r['boundary'],r['head_rs_to_next_owner_write_ms']) for r in rr['reuse']])
    print('fulltable',len(rr['fulltable_collectives']),collections.Counter(e['dtype'] for e in rr['fulltable_collectives']))
