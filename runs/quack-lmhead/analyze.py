"""Summarize already-produced timing and single-stream profiler evidence."""
import argparse
import json
from pathlib import Path
import statistics
p=argparse.ArgumentParser(); p.add_argument('raw',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
out={'timing':{},'profiles':{}}
for path in sorted(a.raw.glob('*/result.json')):
    data=json.loads(path.read_text())
    if data['mode']=='timing' and 'pipeline' not in str(path):
        wall=data['wall_ms']
        start,peak=data['memory_start'],data['memory_peak']
        out['timing'][path.parent.name]={'variant':data['variant'],'wall_ms':wall,'median_wall_ms':statistics.median(wall),'mean_wall_ms':statistics.mean(wall),'p95_wall_ms_nearest_rank':sorted(wall)[int(.95*len(wall)+.999999)-1],'min_wall_ms':min(wall),'max_wall_ms':max(wall),'p25_wall_ms':statistics.quantiles(wall,n=4)[0],'p75_wall_ms':statistics.quantiles(wall,n=4)[2], 'median_cuda_event_host_submission_inclusive_ms':data['median_cuda_event_ms'],'memory_start':start,'memory_peak':peak,'memory_increment':{k:peak[k]-start[k] for k in peak}}
    if data['mode']=='profile':
        events=json.loads((path.parent/'trace.json').read_text())['traceEvents']
        kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
        intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
        merged=[]
        for st,en in intervals:
            if merged and st<=merged[-1][1]: merged[-1][1]=max(en,merged[-1][1])
            else: merged.append([st,en])
        span=max(en for st,en in intervals)-min(st for st,en in intervals)
        busy=sum(en-st for st,en in merged)
        names={}
        for e in kernels:
            n=e['name']; names[n]=names.get(n,0)+e['dur']/1000
        sync=[e for e in events if e.get('ph')=='X' and e.get('cat')=='cuda_runtime' and any(s in e['name'].lower() for s in ['synchronize','memcpy'])]
        copies=[e for e in events if e.get('ph')=='X' and 'gpu_memcpy' in e.get('cat','')]
        comm=[e['name'] for e in kernels if any(s in e['name'].lower() for s in ['nccl','allreduce','reducescatter'])]
        out['profiles'][path.parent.name]={'activation':data.get('activation'),'native_stable_ce_kernel_counts':{name:sum(e['name']==name for e in kernels) for name in ['forward_kernel','backward_kernel']},'kernel_count':len(kernels),'kernel_sum_ms':sum(e['dur'] for e in kernels)/1000,'kernel_interval_union_ms':busy/1000,'first_to_last_kernel_span_ms':span/1000,'gaps_within_kernel_span_ms':(span-busy)/1000,'host_runtime_sync_or_copy_calls':[{'name':e['name'],'duration_ms':e['dur']/1000} for e in sync],'copy_count':len(copies),'copy_duration_sum_ms':sum(e['dur'] for e in copies)/1000,'communication_kernel_names':comm,'top_kernel_totals_ms':sorted(names.items(),key=lambda kv:kv[1],reverse=True)[:12]}
a.output.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
