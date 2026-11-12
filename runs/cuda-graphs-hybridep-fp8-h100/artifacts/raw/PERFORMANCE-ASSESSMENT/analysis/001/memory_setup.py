"""Join existing external NVML samples to accepted per-rank UTC boundaries."""
import csv,datetime,json,pathlib,statistics,sys
HERE=pathlib.Path(__file__).resolve().parent;PERF=HERE.parents[1]
def read(p):return json.loads(p.read_text())
def utc(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def extract(n):
 p=PERF/'attempts'/n
 available=read(p/'root-availability.json')['gpu_rows'];uuid_index={row[1]:int(row[0]) for row in available}
 samples={i:[] for i in range(8)}
 with (p/'device-memory.csv').open() as f:
  for row in csv.DictReader(f):
   t=datetime.datetime.strptime(row['timestamp_utc'].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
   samples[int(row['index'])].append((t,int(row['memory_used_mib'])*1048576))
 memories=read(HERE/f'{n}-memory.json');details=[]
 launch=utc(read(p/'root-launch.json')['at_utc']);series=[]
 for m in memories:
  rank=m['rank'];idx=uuid_index[m['device']['device_uuid']];anchors=m['timing_observer']['anchors']
  opening=anchors['window_open']['utc_ns_bracket'];closing=anchors['completed_close']['utc_ns_bracket']
  low=opening[1]/1e9;high=closing[0]/1e9
  selected=[(t,v) for t,v in samples[idx] if low<=t<=high];assert selected
  previous=[t for t,v in samples[idx] if t<low];following=[t for t,v in samples[idx] if t>high]
  gaps=[b[0]-a[0] for a,b in zip(selected,selected[1:])]
  whole=samples[idx]
  log=next((p/'native/structured_logs').glob(f'*global_rank_{rank}.*'))
  events=[json.loads(l) for l in log.read_text().splitlines()]
  native_start=next(e['time_us']/1e6 for e in events if e['log_type_name']=='training_start')
  first_update=next(e['time_us']/1e6 for e in events if e['log_type_name']=='step_end')
  record={'rank':rank,'physical_index':idx,'uuid':m['device']['device_uuid'],'window_utc_ns_brackets':{'opening':opening,'closing':closing},'sample_count':len(selected),'first_last_sample_unix_seconds':[selected[0][0],selected[-1][0]],'maximum_sample_gap_ms':max(gaps)*1000,'opening_to_first_sample_ms':(selected[0][0]-low)*1000,'last_sample_to_close_ms':(high-selected[-1][0])*1000,'preceding_sample_gap_to_open_ms':(low-previous[-1])*1000 if previous else None,'following_sample_gap_from_close_ms':(following[0]-high)*1000 if following else None,'sampled_window_peak_bytes':max(v for t,v in selected),'sampled_window_minimum_free_bytes':m['device_capacity_bytes']-max(v for t,v in selected),'sampled_whole_process_peak_bytes':max(v for t,v in whole),'launch_to_native_training_start_seconds':native_start-launch,'launch_to_first_update_completed_seconds':first_update-launch,'launch_to_window_open_seconds':low-launch,'native_start_to_window_open_seconds':low-native_start,'allocator_open_close':m['timing_observer']['existing_memory_samples']}
  details.append(record)
 result={'attempt':n,'sample_scope':'500ms device-level samples; peaks are lower bounds, minimum free is an upper bound on true minimum; windows exclude boundary-straddling samples','ranks':details,'sampled_window_peak_bytes_max_rank':max(d['sampled_window_peak_bytes'] for d in details),'sampled_whole_process_peak_bytes_max_rank':max(d['sampled_whole_process_peak_bytes'] for d in details),'maximum_sample_gap_ms':max(d['maximum_sample_gap_ms'] for d in details),'setup_seconds_max_rank':{key:max(d[key] for d in details) for key in ['launch_to_native_training_start_seconds','launch_to_first_update_completed_seconds','launch_to_window_open_seconds','native_start_to_window_open_seconds']}}
 (HERE/f'{n}-nvml-setup.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 r=extract(sys.argv[1]);print(json.dumps({k:v for k,v in r.items() if k!='ranks'},indent=2))
