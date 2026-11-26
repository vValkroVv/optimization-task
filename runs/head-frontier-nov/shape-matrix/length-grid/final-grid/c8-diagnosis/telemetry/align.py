"""Match observed GPU samples to each rank's synchronized head calls (UTC)."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import json
from statistics import median

def samples(csv_path):
    devices = {}
    with csv_path.open() as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if not row or row[0]=='timestamp' or len(row)!=11:
                continue
            t = datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
            devices.setdefault(int(row[1]), []).append((int(t.timestamp()*1e9), row))
    return devices

def summary(rows):
    if not rows:
        return {'samples':0}
    return dict(samples=len(rows), median_sm_mhz=median(int(r[2]) for r in rows),
                median_watts=median(float(r[5]) for r in rows),
                median_gpu_util=median(float(r[8]) for r in rows),
                max_temp_c=max(int(r[7]) for r in rows),
                sw_power_cap_fraction=sum(r[9]=='Active' for r in rows)/len(rows),
                hardware_slowdown_fraction=sum(r[10]=='Active' for r in rows)/len(rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file',type=Path)
    parser.add_argument('timings_root',type=Path)
    parser.add_argument('--reasons', type=Path)
    parser.add_argument('--thermal', type=Path)
    args=parser.parse_args()
    by_rank=samples(args.csv_file)
    reasons={}
    if args.reasons:
        with args.reasons.open() as stream:
            for row in csv.reader(stream, skipinitialspace=True):
                if len(row) != 10 or row[0] == 'timestamp':
                    continue
                stamp=datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
                reasons.setdefault(int(row[1]), []).append((int(stamp.timestamp()*1e9), row))
    thermal={}
    if args.thermal:
        with args.thermal.open() as stream:
            for obs in csv.DictReader(stream):
                gpu=int(obs['physical_gpu'])
                thermal.setdefault(gpu, []).append((1000*int(obs['nvml_utc_us']), int(obs['thermal_violation_ns'])))
    result={}
    for path in sorted(args.timings_root.rglob('timing-0.json')):
        inv=json.loads(path.with_name('invocation.json').read_text())
        uuids={g['uuid']:g['index'] for g in inv['gpu_preflight']}
        order=[uuids[u] for u in inv['settings']['CUDA_VISIBLE_DEVICES'].split(',')]
        sections={'unprofiled':[], 'profiled':[]}
        per_gpu={}

        for rank in range(8):
            timing=path.with_name(f'timing-{rank}.json')
            if not timing.exists():
                continue
            physical=order[rank]
            records={}
            for s in json.loads(timing.read_text()):
                subset=[row for t,row in by_rank[physical] if s['start_utc_ns']<=t<=s['stop_utc_ns']]
                key='profiled' if s['profiled'] else 'unprofiled'
                sections[key].extend(subset)
                record=records.setdefault(key, {'samples':[], 'reason_samples':0, 'hw_thermal':0, 'sw_thermal':0, 'hw_brake':0, 'temp_delta_lower_bound_ns':0,'counter_intervals':0})
                record['samples'].extend(subset)
                hits=[row for t,row in reasons.get(physical,[]) if s['start_utc_ns']<=t<=s['stop_utc_ns']]
                record['reason_samples']+=len(hits)
                for name,index in [('hw_thermal',6),('hw_brake',7),('sw_thermal',8)]:
                    record[name]+=sum(row[index]=='Active' for row in hits)
                counted=[value for t,value in thermal.get(physical,[]) if s['start_utc_ns']<=t<=s['stop_utc_ns']]
                if len(counted)>1:
                    record['temp_delta_lower_bound_ns']+=counted[-1]-counted[0]
                    record['counter_intervals']+=len(counted)-1
            for record in records.values():
                metrics=record.pop('samples')
                record.update(summary(metrics))
                if metrics:
                    record['min_sm_mhz']=min(int(m[2]) for m in metrics)
            per_gpu[physical]={'rank':rank, 'detail':records}

        result[str(path.parent.relative_to(args.timings_root))]={'all':{name:summary(rows) for name,rows in sections.items()},'physical_gpus':per_gpu}
    print(json.dumps(result, indent=2))
