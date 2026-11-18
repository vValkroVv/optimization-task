"""Render bounded actual-event schedule and owner-embedding comparison panels."""
import html
import json
from pathlib import Path

from analyze import RAW, HERE, rows, owned, family


COLORS = {'local_compute': '#4879b9', 'nccl': '#c65b45',
          'gpu_memcpy': '#8b61a8', 'gpu_memset': '#9a8638'}


def panel(title, lanes):
    events = [e for _, es in lanes for e in es]
    if not events:
        return ''
    lo = min(e['start_ms'] for e in events)
    hi = max(e['end_ms'] for e in events)
    scale = 960 / max(hi - lo, .001)
    height = 45 + 28 * len(lanes)
    out = [f'<h3>{html.escape(title)}</h3>',
           f'<svg viewBox="0 0 1220 {height}" role="img" aria-label="{html.escape(title)}">']
    for i in range(7):
        t = lo + (hi-lo)*i/6
        x = 240 + (t-lo)*scale
        out += [f'<path d="M{x} 22V{height}" stroke="#ddd"/>',
                f'<text x="{x}" y="14" text-anchor="middle">{t-lo:.3f}</text>']
    for i, (label, es) in enumerate(lanes):
        y = 28+i*28
        out.append(f'<text x="4" y="{y+15}">{html.escape(label)}</text>')
        for e in es:
            x = 240+(e['start_ms']-lo)*scale
            width = max(.4,(e['end_ms']-e['start_ms'])*scale)
            kind = e.get('family', 'local_compute')
            tip = f"{e['name']} | {e['start_ms']:.6f}–{e['end_ms']:.6f} ms trace-relative | {e['end_ms']-e['start_ms']:.6f} ms | stream {e.get('stream')}"
            out.append(f'<rect x="{x}" y="{y}" width="{width}" height="20" fill="{COLORS[kind]}"><title>{html.escape(tip)}</title></rect>')
    out.append('</svg>')
    return '\n'.join(out)


def raw_events(tag, rank, origin):
    q = json.loads((RAW/f'{tag}-rank{rank}.mcp.json').read_text())['queries']
    events = sum([rows(chunk) for chunk in q['device']], [])
    scopes = rows(q['scopes'])
    converted = {}
    for e in events:
        converted[e['id']] = {'name': e['name'], 'start_ms': (e['ts']-origin)/1e6,
                              'end_ms': (e['ts']+e['dur']-origin)/1e6,
                              'stream': e['stream'], 'family': family(e)}
    return events, scopes, converted


def render(tags, output):
    details = {tag: json.loads((RAW/f'{"control" if tag == "d0" else tag}-detail.json').read_text()) for tag in tags}
    output = Path(output)
    out = ['<!doctype html><meta charset="utf-8"><title>Schedule and embedding actual-event comparison</title>',
           '<style>body{font:14px system-ui;margin:24px;color:#20252d}svg{width:100%;min-width:900px;border:1px solid #ddd}svg text{font:11px system-ui}h2{margin-top:36px}h3{font-size:15px}.scroll{overflow:auto}p{max-width:1000px}</style>',
           '<h1>Schedule and embedding: actual device events</h1>',
           '<p>Rank 0, microbatch 0 of update 21 after 20 warmups. Each panel uses its own zero; hover for original trace-relative times and exact kernel names. Blue: local kernels; red: NCCL; purple: copy; ochre: memset. Rectangles show measured intervals, including NCCL peer waiting. Overlapping intervals are not additive. CPU annotation spans are excluded. These profiled panels establish mechanisms, not throughput.</p>']
    for mode in ('head', 'parent', 'embedding'):
        out.append(f'<h2>{ {"head":"Head → MTP FFN transition", "parent":"MTP FFN release → final-layer transition", "embedding":"Owner embedding forward / backward"}[mode]}</h2>')
        for tag in tags:
            result = details[tag]['ranks'][0]
            micro = result['microbatches'][0]
            es, scopes, converted = raw_events(tag, 0, round(result['trace_origin_us']*1000))
            col = micro['collectives']
            def coll(module, op):
                return [{**e, 'family':'nccl'} for e in col[f'{op}:{module}']]
            if mode == 'embedding':
                lanes = []
                for emb in result['embedding']:
                    if emb['microbatch'] == 0:
                        label = 'Forward' if emb['phase'] == '_OwnerLookup' else 'Backward'
                        for kind in COLORS:
                            chosen = [converted[e['id']] for e in es
                                      if emb['device_start_ms'] <= converted[e['id']]['start_ms']
                                      and converted[e['id']]['end_ms'] <= emb['device_end_ms']
                                      and family(e) == kind
                                      and ('_OwnerLookup' in (e['ancestors'] or ''))]
                            if chosen:
                                lanes.append((f'{label}: {kind}', chosen))
            else:
                modules = (['lm_head', 'mtp.layers.0', 'mtp.layers.0.block.feed_forward'] if mode == 'head'
                           else ['mtp.layers.0.block.feed_forward', 'mtp.layers.0', 'norm', 'layers.7[fsdp]'])
                lanes = []
                for module in modules:
                    for op in ('all_gather', 'post_backward_reduce'):
                        if mode == 'head' and ((module == 'lm_head') != (op == 'post_backward_reduce')):
                            continue
                        events = coll(module, op)
                        if mode == 'head' and op == 'all_gather':
                            events = [e for e in events if 'forward_prefetch' not in e.get('ancestors', '')]
                        if mode == 'parent' and op == 'all_gather' and module != 'layers.7[fsdp]':
                            continue
                        if op == 'all_gather' and events:
                            events = [max(events, key=lambda e:e['start_ms'])]
                        if events:
                            lanes.append((module.replace('mtp.layers.0.block.', 'MTP ') + (' AG' if op == 'all_gather' else ' RS'), events))
                anchors = [e for _, events in lanes for e in events]
                start = min(e['start_ms'] for e in anchors)
                end = max(e['end_ms'] for e in anchors)
                end = max(end, micro['final_layer_compiled_compute']['first_ms']+1) if mode == 'parent' else end+1
                local = [converted[e['id']] for e in es if e['stream'] == 7 and family(e) == 'local_compute'
                         and start <= converted[e['id']]['start_ms'] <= end]
                lanes.append(('Stream 7 local kernels', local))
                if mode == 'parent':
                    pre = next(s for s in scopes if s['name'] == 'GFSDP::pre_backward (layers.7[fsdp])')
                    cb = min((s for s in scopes if s['name'] == 'CompiledFunctionBackward' and s['ts'] > pre['ts']), key=lambda s:s['ts'])
                    lanes.append(('layers.7 compiled kernels', [converted[e['id']] for e in es if owned(e, cb) and family(e) == 'local_compute']))
                    lanes.append(('Norm gradient packing', [{**e, 'family':'local_compute'} for e in micro['norm_packing']]))
            out.append('<div class="scroll">'+panel(tag.upper(), lanes)+'</div>')
    out.append('<p>Sources: saved actual Perfetto MCP device/flow/ancestor queries and reconciled detail JSON beside this artifact. Cross-trace matching uses module/operation/payload and microbatch, never slice IDs.</p>')
    output.write_text('\n'.join(out)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags', nargs='+')
    parser.add_argument('--output', type=Path, default=HERE/'before-after.html')
    args = parser.parse_args()
    render(args.tags, args.output)
