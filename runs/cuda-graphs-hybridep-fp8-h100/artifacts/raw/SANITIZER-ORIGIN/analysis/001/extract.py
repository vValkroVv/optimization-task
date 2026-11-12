"""Extract only completed attempt files; do not query processes or CUDA."""
import collections
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ATTEMPT = OUT.parents[1] / 'attempts' / '001'
inputs = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(ATTEMPT))] = {
        'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)
    }
    return data.decode()


def load(name):
    return json.loads(read(ATTEMPT / name))


ranks = []
for path in sorted((ATTEMPT / 'native/structured_logs').glob('*.jsonl')):
    rows = [json.loads(line) for line in read(path).splitlines()]
    counts = collections.Counter(row['log_type_name'] for row in rows)
    ends = [row['step'] for row in rows if row['log_type_name'] == 'optim_end']
    ranks.append({
        'rank': rows[0]['global_rank'], 'pid': rows[0]['pid'],
        'path': str(path.relative_to(ATTEMPT)), 'rows': len(rows),
        'seq_contiguous': [row['seq_id'] for row in rows] == list(range(len(rows))),
        'identity_consistent': len({(r['global_rank'], r['pid']) for r in rows}) == 1,
        'event_counts': dict(counts), 'optim_end_steps': ends,
        'fwd_bwd_end_by_step': dict(collections.Counter(
            row['step'] for row in rows if row['log_type_name'] == 'fwd_bwd_end')),
        'last': {k: rows[-1].get(k) for k in ['seq_id', 'step', 'log_type_name', 'time_us']},
        'error_records': [row for row in rows if row.get('log_level') in ['ERROR', 'CRITICAL']],
    })
ranks.sort(key=lambda x: x['rank'])
assert [r['rank'] for r in ranks] == list(range(8))
assert all(r['seq_contiguous'] and r['identity_consistent'] for r in ranks)
assert all(r['optim_end_steps'] == list(range(1, 17)) for r in ranks)
assert all(r['event_counts']['fwd_bwd_start'] == 68 and r['event_counts']['fwd_bwd_end'] == 67 for r in ranks)
compiler = [load(str(p.relative_to(ATTEMPT))) for p in sorted((ATTEMPT / 'compiler-options').glob('*.json'))]
assert len(compiler) == 8 and all(not r['triton.cudagraphs'] and not r['triton.mix_order_reduction'] and r['graph_partition'] for r in compiler)
resolved = read(ATTEMPT / 'native/config/resolved_config.yaml')
provenance = load('binary-provenance-receipt.json')
binaries = [{k: entry[k] for k in ['selected_path', 'sha256', 'size_bytes', 'identity', 'status']} | {'ranks': sorted({ref['rank'] for ref in entry['mapping_refs']})} for entry in provenance['unique_mapped_family_binaries']]
read(ATTEMPT / 'launch.sh')
read(ATTEMPT / 'lifecycle.log')
read(ATTEMPT / 'config/sanitizer_origin.yaml')
read(ATTEMPT / 'pins.json')
read(ATTEMPT / 'preflight.json')
logs = []
for path in sorted((ATTEMPT / 'torchrun').rglob('*.log')):
    content = read(path)
    pattern = r'Traceback|CUDA error|cudaError|RuntimeError|illegal (?:memory|instruction)|unspecified launch failure|\b715\b'
    logs.append({'path': str(path.relative_to(ATTEMPT)), 'fatal_candidates': [{'line': n, 'text': s} for n, s in enumerate(content.splitlines(), 1) if re.search(pattern, s, re.I)]})
supervisor = read(ATTEMPT / 'supervisor.log')
reports = []
for path in sorted(ATTEMPT.glob('sanitizer.*.log')):
    text = read(path)
    blocks = re.split(r'(?m)(?=^========= CUDA API Error:)', text)[1:]
    reports.append({'path': path.name, 'report_count': len(blocks), 'report_types': dict(collections.Counter(b.splitlines()[0] for b in blocks)), 'all_report_blocks_contain_nccl_init': all('ncclInitKernelsForDevice' in b for b in blocks), 'summary_present': 'ERROR SUMMARY' in text, 'suppression_or_limit_lines': [s for s in text.splitlines() if re.search('suppress|print.limit|maximum.*error', s, re.I)], 'kernel_exception_lines': [s for s in text.splitlines() if re.search('hardware exception|Invalid __|Program hit|Device Frame', s)], 'first_report_lines': blocks[0].splitlines()[:8]})
result = {'attempt': str(ATTEMPT), 'ranks': ranks, 'compiler': compiler, 'native_rank_logs': logs, 'runner_status': load('runner-status.json'), 'watcher_failure': load('watcher-failure.json'), 'task_stop': load('task-stop.json'), 'helper_completion': load('helper-completion.json'), 'process_discovery': load('process-discovery.json'), 'sanitizer': reports, 'provenance': {'capture_status': provenance['capture_status'], 'coverage': provenance['coverage'], 'timestamps_utc': provenance['timestamps_utc'], 'limitations': provenance['limitations'], 'binaries': binaries}, 'supervisor_fatal_candidates': [{'line': n, 'text': s} for n, s in enumerate(supervisor.splitlines(), 1) if re.search(r'Traceback|SignalException|CUDA error|RuntimeError|715', s)], 'activation_lines': [s for s in supervisor.splitlines() if any(x in s for x in ['Building device mesh', 'FP8 projections', 'selective', 'Compil', 'compil', 'gradient accumulation'])], 'input_files': inputs}
(OUT / 'extraction.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'ranks': len(ranks), 'rows_each': [r['rows'] for r in ranks], 'optim_end_each': [len(r['optim_end_steps']) for r in ranks], 'native_fatal_candidates': sum(len(x['fatal_candidates']) for x in logs), 'sanitizer_reports': [x['report_count'] for x in reports], 'hashed_inputs': len(inputs)}, indent=2))
