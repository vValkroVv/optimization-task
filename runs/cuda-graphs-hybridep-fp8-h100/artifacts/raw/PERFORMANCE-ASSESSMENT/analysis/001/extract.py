"""Offline, bounded extraction of the accepted native short-run records."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,math,re,statistics,sys
import yaml
HERE=Path(__file__).resolve().parent
PERF=HERE.parents[1]
TASK=PERF.parents[4]

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    prep=PERF/'preparation/002'
    assert sha(prep/'artifact-hashes.json')=='e440255d61acb668e000e532b94432a80952959acc9adcba0b07bdffbdc10c05'
    for name,expected in read(prep/'artifact-hashes.json').items():
        assert sha(PERF/name)==expected,name
    source=PERF.parent/'SOURCE-SNAPSHOT/010/manifest.json'
    assert sha(source)=='b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20'
    for name,record in read(source)['files'].items():
        assert sha(TASK/name)==record['sha256'],name
    return {'frozen_artifacts':True,'source_files':len(read(source)['files'])}

def summary(values):
    return {'min':min(values),'median':statistics.median(values),'max':max(values)}

def extract(n):
    attempt=PERF/'attempts'/n
    result={'attempt':n,'variant':{'006':'off-AA1','007':'off-AA2','008':'on-B1','009':'off-A2','010':'on-B2'}[n]}
    assert read(attempt/'controller-complete.json')['exit_code']==0
    owner=read(attempt/'owner-status.json')
    assert owner['exit_code']==0 and owner['all_recorded_identities_absent'] and not owner['remaining_identities']
    assert read(attempt/'exit-status.json')['exit_code']==0
    config=yaml.safe_load((attempt/'native/config/resolved_config.yaml').read_text())
    baseline=yaml.safe_load((PERF/'attempts/006/native/config/resolved_config.yaml').read_text())
    config.pop('dump_folder');baseline.pop('dump_folder')
    assert config==baseline,'resolved configuration differs beyond attempt dump path'
    assert config['trainer']['type']=='default' and set(config['callbacks'])=={'campaign_timing','campaign_step_boundary'}
    assert config['metrics']['save_for_all_ranks'] is False
    result['resolved_configuration_equal_to_006_except_dump_folder']=True
    assert not (attempt/'nvidia-smi.stderr').read_text().strip()
    for p in (attempt/'torchrun').glob('*/attempt_0/*/stderr.log'):
        text=p.read_text()
        assert not re.search(r'Traceback|CUDA error|device-side assert|illegal memory|OutOfMemory|Error:',text),str(p)
    logs=sorted((attempt/'native/structured_logs').glob('*.jsonl'))
    assert len(logs)==8
    ranks=[]; series=[]; timings=[]; memories=[]; configs=[]; metrics=[]
    for log in logs:
        events=[json.loads(line) for line in log.read_text().splitlines()]
        rank=events[0]['global_rank'];ranks.append(rank)
        assert all(x['global_rank']==rank for x in events)
        seq=[x['seq_id'] for x in events]
        assert seq==sorted(set(seq))
        counts=collections.Counter(x['log_type_name'] for x in events)
        assert counts['optim_end']==41 and counts['fwd_bwd_end']==164 and counts['step_end']==41,(rank,counts)
        assert [x['step'] for x in events if x['log_type_name']=='optim_end']==list(range(1,42))
        for x in events:
            if x['log_type_name']=='step_end':series.append({'attempt':n,'rank':rank,'step':x['step'],'native_step_host_span_ms':x['value'],'time_us':x['time_us']})
        cp=read(attempt/f'compiler-options/rank-{rank:05d}.json')
        assert cp['triton.cudagraphs']==(n in ['008','010']) and cp['triton.mix_order_reduction'] is False and cp['graph_partition'] is True
        env=read(attempt/f'runtime/rank-{rank:05d}-environment.json')
        assert 'CUDA_LAUNCH_BLOCKING' in env['unset_or_unrecorded_allowlist_names']
        timing_path=list(attempt.glob(f'**/rank-{rank:05d}.jsonl'))
        assert len(timing_path)==1,timing_path
        activation,window=[json.loads(l) for l in timing_path[0].read_text().splitlines()]
        assert window['valid'] and window['status']=='complete'
        assert window['start_step']==21 and window['end_boundary_step']==41 and window['observed_end_boundary_step']==41 and window['update_count']==20
        assert window['global_valid_tokens_per_update_this_rank']==[1048576]*20 and window['global_valid_tokens_total_this_rank']==20971520
        assert len(window['host_update_submission_spans_seconds'])==len(window['cuda_current_stream_boundary_spans_seconds'])==20
        assert activation['world_size']==8
        markers=activation['constructed_markers']
        assert markers['activation_checkpoint_wrappers']['count']==18 and markers['compiled_model_callables_installed']['count']==18
        assert markers['dense_fp8_linear_modules']['count']==72 and markers['sparse_fp8_grouped_expert_modules']['count']==17 and markers['hybridep_dispatchers']['count']==17
        contract=activation['training_contract']
        assert [contract[k] for k in ['local_batch_size','configured_global_batch_size','gradient_accumulation_steps','sequence_length']]==[8,256,4,4096]
        assert activation['parallelism']['degrees']['ep']==8
        timings.append(window);configs.append(activation)
        memory=read(attempt/f'phase_memory/rank-{rank:05d}.json')
        assert memory['status']=='complete' and memory['phase_boundaries']=={'startup_end_step':20,'expected_end_step':40}
        assert memory['training_log_frequency']==10 and memory['coverage']['last_training_metrics_step']==40
        assert all(e['status']=='complete' for e in memory['events'])
        snapshots=[e for e in memory['events'] if e['event_kind']=='native_snapshot' and e['step'] in [30,40]]
        assert len(snapshots)==2
        assert [e['actual_window']['label'] for e in snapshots]==['21_to_30','31_to_40']
        assert all(e['allocator_counters_bytes']['num_alloc_retries']==0 and e['allocator_counters_bytes']['num_ooms']==0 for e in snapshots)
        stdout=next((attempt/'torchrun').glob(f'*/attempt_0/{rank}/stdout.log')).read_text()
        stdout=re.sub(r'\x1b\[[0-9;]*m','',stdout)
        found=re.findall(r'step:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)',stdout)
        assert [int(a) for a,b,c in found]==([1,10,20,30,40] if rank==0 else []),(rank,found)
        assert all(math.isfinite(float(b)) and math.isfinite(float(c)) for a,b,c in found)
        metrics.append({'rank':rank,'loss_and_norm':[{'step':int(a),'loss':float(b),'grad_norm':float(c)} for a,b,c in found]})
        memories.append(memory)
    assert sorted(ranks)==list(range(8))
    wall=max(t['local_completed_window_elapsed_seconds'] for t in timings)
    assert all(abs(t['slowest_rank_completed_window_elapsed_seconds']-wall)<1e-8 for t in timings)
    result.update(valid=True,window_seconds=wall,global_useful_tokens=20971520,global_useful_tps=20971520/wall,completed_mean_update_seconds=wall/20,ranks=ranks)
    result['host_submission_span_seconds']=summary([v for t in timings for v in t['host_update_submission_spans_seconds']])
    result['current_stream_boundary_span_seconds']=summary([v for t in timings for v in t['cuda_current_stream_boundary_spans_seconds']])
    result['allocator_measured_envelope_bytes']={key:max(e['allocator_counters_bytes'][counter] for m in memories for e in m['events'] if e['event_kind']=='native_snapshot' and e['step'] in [30,40]) for key,counter in [('allocated','allocated_bytes.all.peak'),('reserved','reserved_bytes.all.peak')]}
    result['native_metrics']=metrics
    (HERE/f'{n}-activation.json').write_text(json.dumps(configs,indent=2)+'\n')
    (HERE/f'{n}-timing.json').write_text(json.dumps(timings,indent=2)+'\n')
    (HERE/f'{n}-memory.json').write_text(json.dumps(memories,indent=2)+'\n')
    with (HERE/f'{n}-step-series.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=series[0]);writer.writeheader();writer.writerows(series)
    (HERE/f'{n}-extraction.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':
    print(json.dumps(freeze()))
    if len(sys.argv)>1:print(json.dumps(extract(sys.argv[1]),indent=2))
