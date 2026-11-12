"""Extract the two repaired native200 runs' existing completion evidence."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

base = Path(__file__).resolve().parents[2]
attempt_id = sys.argv[1]
assert attempt_id in {'003', '004'}
a = base / 'attempts' / attempt_id
out = base / 'analysis' / attempt_id
out.mkdir(exist_ok=True)
assert (a / 'controller-complete.json').is_file(), 'attempt is not terminal'
rank_records = []
for f in sorted((a / 'native/structured_logs').glob('*.jsonl')):
    raw = f.read_bytes()
    assert raw.endswith(b'\n'), f
    rows = [json.loads(line) for line in raw.splitlines()]
    ids = [r['seq_id'] for r in rows]
    ranks = {r['rank'] for r in rows}
    assert len(ranks) == 1, ranks
    rank = ranks.pop()
    errors = [r for r in rows if r.get('log_type_name', '').endswith('_error') or r.get('log_level') in {'ERROR', 'CRITICAL'}]
    counts = defaultdict(Counter)
    pending = Counter()
    ordering_errors = []
    for r in rows:
        name = r.get('log_type_name', '')
        if name in {f'{n}_{suffix}' for n in ('step','optim','fwd_bwd') for suffix in ('start','end')}:
            name_base, suffix = name.rsplit('_', 1)
            step = r['step']
            counts[step][name] += 1
            key = (step, name_base)
            pending[key] += 1 if suffix == 'start' else -1
            if pending[key] not in {0, 1}:
                ordering_errors.append({'step': step, 'event': name})
    expected = Counter(step_start=1, step_end=1, optim_start=1, optim_end=1, fwd_bwd_start=4, fwd_bwd_end=4)
    completed = [s for s,c in counts.items() if c == expected and not any(e.get('step') == s for e in errors)]
    token_metrics = [r for r in rows if r.get('event_name') in {'local_valid_tokens','global_valid_tokens'}]
    stdout_path = next((a / 'torchrun').glob(f'*/attempt_0/{rank}/stdout.log'))
    stdout = stdout_path.read_text()
    activation = {label: text in stdout for label,text in {
        'fp8':'FP8 projections in full model structure: 72 dense, 34 grouped',
        'loss_compile':'Compiling the loss function with torch.compile',
        'model_compile':'Compiling each block with torch.compile (18 in layers)',
        'sac':'selective activation checkpointing',
        'fsdp2':'Applied fully_shard to the model',
        'accumulation4':'gradient accumulation steps 4, sequence length 4096, total steps 200',
        'hybridep':'Initializing HybridEP buffer: hidden_dim=1536, max_tokens_per_rank=32768, num_local_experts=4, ep_size=8',
        'teardown':'Process group destroyed',
    }.items()}
    metrics = [{'step':int(m[0]),'loss':float(m[1]),'grad_norm':float(m[2])} for m in re.findall(r'step:\s*(\d+).*?loss:\s*([-+\w.]+).*?grad_norm:\s*([-+\w.]+)',stdout)]
    rank_records.append({'rank':rank,'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(rows),
        'sequence_ids_strict':all(x<y for x,y in zip(ids,ids[1:])), 'errors':errors,'ordering_errors':ordering_errors,
        'unclosed_spans':[str(k) for k,v in pending.items() if v], 'completed_steps':completed,
        'exact200':sorted(completed)==list(range(1,201)) and set(counts)==set(range(1,201)),
        'successful_microbatches':sum(counts[s]['fwd_bwd_end'] for s in completed),'activation':activation,
        'native_metrics':metrics,'native_metrics_finite':all(math.isfinite(x[k]) for x in metrics for k in ('loss','grad_norm')),
        'token_metric_count':len(token_metrics),'token_metrics_finite':bool(token_metrics) and all(math.isfinite(r['value']) for r in token_metrics),
        'token_counts_exact':all(r['value']==({'local_valid_tokens':131072,'global_valid_tokens':1048576}[r['event_name']]) for r in token_metrics)})
controller=json.loads((a/'controller-complete.json').read_text())
owner=json.loads((a/'owner-status.json').read_text())
receipt=json.loads((out/'live-dependency-activation.json').read_text())
pids_absent=all(not (Path('/proc')/str(r['pid'])).exists() or int((Path('/proc')/str(r['pid'])/'stat').read_text().rsplit(')',1)[1].split()[19])!=r['start_ticks'] for r in receipt['records'])
result={'attempt':attempt_id,'ranks':rank_records,'controller':controller,'owner':owner,'recorded_worker_identities_absent':pids_absent,
    'all8_exact200_800':sorted(r['rank'] for r in rank_records)==list(range(8)) and all(r['exact200'] and r['successful_microbatches']==800 and not r['errors'] and not r['ordering_errors'] and not r['unclosed_spans'] and r['sequence_ids_strict'] for r in rank_records),
    'clean_terminal':controller['exit_code']==0 and owner['exit_code']==0 and owner['reason']=='natural_exit' and owner['all_recorded_identities_absent'] and pids_absent}
(out/'completion-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'attempt':attempt_id,'all8_exact200_800':result['all8_exact200_800'],'clean_terminal':result['clean_terminal'],'activation':{r['rank']:r['activation'] for r in rank_records},'metrics':{r['rank']:{'count':len(r['native_metrics']),'finite':r['native_metrics_finite'],'token_count':r['token_metric_count'],'tokens_finite':r['token_metrics_finite']} for r in rank_records}},indent=2))
