"""Render all three matched microbatches of both ranks without stream relabeling."""
import json
from analyze import HERE, RAW, family, owned, rows
from timeline import panel


def main():
    data=json.loads((RAW/'comparison-004-detail.json').read_text())['variants']
    for rank in (0,1):
        out=['<!doctype html><meta charset="utf-8"><title>Four-way update21 rank '+str(rank)+'</title>',
             '<style>body{font:14px system-ui;margin:24px;color:#20252d}svg{width:100%;min-width:1000px;border:1px solid #ddd}svg text{font:11px system-ui}.scroll{overflow:auto}h2{margin-top:42px}p{max-width:1100px}</style>',
             '<h1>D0 / S / SE / SE-fused: rank '+str(rank)+'</h1>',
             '<p>All three captured F+B microbatches. Each panel has its own zero; hover for trace-relative time, exact kernel and original CUDA stream. Blue local kernels; red NCCL (may include peer waiting); purple copy; ochre memset. Device intervals follow actual launch flows. CPU spans and synthetic display lane IDs are excluded. Overlapping durations are not additive. No throughput claim.</p>']
        for mb in range(3):
            for mode in ('head','parent','embedding'):
                out.append('<h2>Microbatch '+str(mb)+' — '+mode+'</h2>')
                for tag,variant in data.items():
                    r=variant['ranks'][rank];micro=r['microbatches'][mb];origin=round(r['trace_origin_us']*1000)
                    q=json.loads((RAW/f'{tag}-004-rank{rank}.mcp.json').read_text())['queries']
                    es=sum([rows(c) for c in q['device']],[]);scopes=rows(q['scopes'])
                    micros=[s for s in scopes if s['category']=='user_annotation' and 'microbatch' in s['name']];scope=micros[mb]
                    es=[e for e in es if scope['ts']<=e['launch_ts']<scope['ts']+scope['dur']]
                    conv=lambda e:{'name':e['name'],'start_ms':(e['ts']-origin)/1e6,'end_ms':(e['ts']+e['dur']-origin)/1e6,'stream':e['stream'],'family':family(e)}
                    lanes=[]
                    if mode=='embedding':
                        for emb in r['embedding']:
                            if emb['microbatch']==mb:
                                for kind in ('local_compute','nccl','gpu_memcpy','gpu_memset'):
                                    events=[{**e,'family':kind} for e in emb['operations'] if ('nccl' if e['name'].startswith('nccl') else e['category'] if e['category']!='kernel' else 'local_compute')==kind]
                                    if events:lanes.append((emb['phase']+': '+kind,events))
                    else:
                        mods=['lm_head','mtp.layers.0','mtp.layers.0.block.feed_forward'] if mode=='head' else ['mtp.layers.0.block.feed_forward','mtp.layers.0','norm','layers.7[fsdp]']
                        for mod in mods:
                            for op in ('all_gather','post_backward_reduce'):
                                if mode=='head' and ((mod=='lm_head')!=(op=='post_backward_reduce')):continue
                                if mode=='parent' and op=='all_gather' and mod!='layers.7[fsdp]':continue
                                events=[{**e,'family':'nccl'} for e in micro['collectives'][f'{op}:{mod}'] if op!='all_gather' or 'forward_prefetch' not in e['ancestors']]
                                if events:lanes.append((mod.replace('mtp.layers.0.block.','MTP ')+' '+op,events))
                        anchors=[e for _,ev in lanes for e in ev];lo=min(e['start_ms'] for e in anchors);hi=max(e['end_ms'] for e in anchors)+1
                        lanes.append(('Original stream 7 local kernels',[conv(e) for e in es if e['stream']==7 and family(e)=='local_compute' and lo<=(e['ts']-origin)/1e6<=hi]))
                        if mode=='parent':
                            cb=next(s for s in scopes if s['id']==micro['semantic_checks']['l7_compiled_scope_id_local_only'])
                            lanes.append(('layers.7 compiled compute',[conv(e) for e in es if owned(e,cb) and family(e)=='local_compute' and (e['ts']-origin)/1e6<=hi]))
                            lanes.append(('Norm gradient packing',[{**e,'family':'local_compute'} for e in micro['norm_packing']]))
                    out.append('<div class="scroll">'+panel(tag.upper(),lanes)+'</div>')
        out.append('<p>Evidence: comparison-004.json and its pinned ignored detail; identities-004.json; saved actual MCP SQL responses. Slice IDs join only within their own trace.</p>')
        (HERE/f'four-way-004-rank{rank}.html').write_text('\n'.join(out)+'\n')


if __name__=='__main__':main()
