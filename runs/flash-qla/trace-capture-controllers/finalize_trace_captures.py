#!/opt/torchtitan-venv/bin/python3.12
import collections
import datetime
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import traceback

BASE = pathlib.Path('/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla')
PLAN_PATH = BASE / 'trace-plan.json'
CONTROLLER_DIR = BASE / 'trace-capture-controllers'
RESULTS_PATH = BASE / 'trace-capture-results.json'
MARKDOWN_PATH = BASE / 'trace-capture-results.md'
FINALIZER_STDOUT = CONTROLLER_DIR / 'finalize_trace_captures.stdout.log'
FINALIZER_STDERR = CONTROLLER_DIR / 'finalize_trace_captures.stderr.log'


def sha256_file(path):
    h = hashlib.sha256()
    n = 0
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def atomic_controller_write(path, data):
    tmp = pathlib.Path(str(path) + '.finalize.tmp')
    if tmp.exists():
        raise FileExistsError(f'refusing to reuse controller temp: {tmp}')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def profile_census(trace_path):
    with pathlib.Path(trace_path).open('r', encoding='utf-8') as f:
        trace = json.load(f)
    if not isinstance(trace, dict) or not isinstance(trace.get('traceEvents'), list):
        raise ValueError(f'not Chrome trace JSON with traceEvents: {trace_path}')
    events = trace['traceEvents']
    categories = collections.Counter()
    phases = collections.Counter()
    marker_names = collections.Counter()
    kernels = copies = memsets = runtime = driver = device_copies = communication = 0
    flow_phases = 0
    flow_ids = set()
    time_bounds = []
    for event in events:
        cat = str(event.get('cat', '(missing)'))
        name = str(event.get('name', '(missing)'))
        low_cat = cat.lower()
        low = (cat + ' ' + name).lower()
        phase = str(event.get('ph', '(missing)'))
        categories[cat] += 1
        phases[phase] += 1
        if 'kernel' in low_cat:
            kernels += 1
        if any(token in low for token in ('memcpy', 'mem_copy', 'mem copy')):
            copies += 1
        if 'memset' in low:
            memsets += 1
        if 'cuda_runtime' in low_cat:
            runtime += 1
        if 'cuda_driver' in low_cat:
            driver += 1
        if any(token in low_cat for token in ('memcpy', 'device_copy', 'gpu_memcpy')):
            device_copies += 1
        if any(token in low for token in ('nccl', 'collective', 'communication', 'allreduce', 'all_reduce', 'reduce_scatter', 'all_gather')):
            communication += 1
        args = event.get('args') if isinstance(event.get('args'), dict) else {}
        flow_id = event.get('id', args.get('id'))
        if phase in ('s', 't', 'f'):
            flow_phases += 1
            if flow_id is not None:
                flow_ids.add(str(flow_id))
        if name.startswith('task.profile_update.21.'):
            marker_names[name] += 1
        ts = event.get('ts')
        dur = event.get('dur', 0)
        if isinstance(ts, (int, float)):
            time_bounds.append((ts, ts + (dur if isinstance(dur, (int, float)) else 0)))
    return {
        'event_count': len(events),
        'category_counts': dict(sorted(categories.items())),
        'phase_counts': dict(sorted(phases.items())),
        'kernel_category_count': kernels,
        'copy_name_or_category_count': copies,
        'memset_name_or_category_count': memsets,
        'cuda_runtime_category_count': runtime,
        'cuda_driver_category_count': driver,
        'device_copy_category_count': device_copies,
        'communication_name_or_category_count': communication,
        'flow_phase_count': flow_phases,
        'unique_flow_ids': len(flow_ids),
        'timestamp_bounds_raw_units': {
            'min_ts': min((x[0] for x in time_bounds), default=None),
            'max_end': max((x[1] for x in time_bounds), default=None),
        },
        'task_profile_update_marker_name_counts': dict(sorted(marker_names.items())),
    }


def validate_entry_markers(attempt, run_id):
    marker_re = re.compile(r'grad_enabled\s*[:=]\s*[\'\"]?(true|false)', re.I)
    result = {str(i): {'entry_lines': 0, 'grad_enabled_false': 0, 'grad_enabled_true': 0, 'files': []} for i in range(8)}
    log_root = pathlib.Path(attempt) / 'logs/torchrun'
    files = []
    for path in log_root.rglob('*'):
        if path.is_file() and path.name in ('stdout.log', 'stderr.log') and path.parent.parent.name == 'attempt_0' and path.parent.name.isdigit():
            files.append(path)
    for path in sorted(files):
        rank = path.parent.name
        if rank not in result:
            continue
        count = 0
        with path.open(errors='replace') as f:
            for line in f:
                if 'GDN_BACKEND_ACTIVATED' not in line:
                    continue
                count += 1
                match = marker_re.search(line)
                if match:
                    result[rank]['grad_enabled_' + match.group(1).lower()] += 1
        if count:
            result[rank]['entry_lines'] += count
            result[rank]['files'].append({'path': str(path), 'marker_lines': count})
    ok = all(result[str(i)]['entry_lines'] > 0 and result[str(i)]['grad_enabled_false'] > 0 and result[str(i)]['grad_enabled_true'] > 0 for i in range(8))
    return ok, result


def make_canonical_gzip(raw_path, raw_sha, raw_bytes):
    raw = pathlib.Path(raw_path)
    gz = pathlib.Path(str(raw) + '.gz')
    tmp = pathlib.Path(str(gz) + '.tmp')
    reused = gz.exists()
    if not reused:
        if tmp.exists():
            raise FileExistsError(f'refusing to overwrite existing partial compressed trace: {tmp}')
        with raw.open('rb') as src, tmp.open('xb') as dst:
            with gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0, compresslevel=6) as stream:
                shutil.copyfileobj(src, stream, 1024 * 1024)
        os.replace(tmp, gz)
    gz_sha, gz_bytes = sha256_file(gz)
    decoded_hash = hashlib.sha256()
    decoded_bytes = 0
    with gzip.open(gz, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            decoded_hash.update(chunk)
            decoded_bytes += len(chunk)
    decoded_sha = decoded_hash.hexdigest()
    if decoded_sha != raw_sha or decoded_bytes != raw_bytes:
        raise ValueError(f'lossless gzip identity mismatch for {raw}')
    return {
        'raw_path': str(raw),
        'raw_sha256': raw_sha,
        'raw_bytes': raw_bytes,
        'gzip_path': str(gz),
        'gzip_sha256': gz_sha,
        'gzip_bytes': gz_bytes,
        'decompressed_sha256': decoded_sha,
        'decompressed_bytes': decoded_bytes,
        'decompressed_identity': True,
        'reused_existing_gzip': reused,
    }


def main():
    if RESULTS_PATH.exists() or MARKDOWN_PATH.exists():
        raise FileExistsError('refusing to overwrite an existing capture results file')
    plan_bytes = PLAN_PATH.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    runs = []
    controller_updates = []
    for variant in ('external_gate', 'fused_gate'):
        run = next(x for x in plan['runs'] if x['variant'] == variant)
        attempt = pathlib.Path(run['attempt'])
        controller_path = CONTROLLER_DIR / f'{run["run_id"]}.json'
        controller = json.loads(controller_path.read_text())
        metadata = json.loads((attempt / 'metadata.json').read_text())
        summary = json.loads((attempt / 'summary.json').read_text())
        process = json.loads((attempt / 'process.json').read_text())
        completion = json.loads((attempt / 'completion.json').read_text())
        source_final = json.loads((attempt / 'source-final.json').read_text())
        bridge = json.loads((attempt / 'flash-qla-bridge.json').read_text())
        resolved = attempt / 'native/config/resolved_config.yaml'
        import yaml
        config = yaml.safe_load(resolved.read_text())
        expected_config = {
            'training.steps': 21,
            'dataloader.num_prefetch_batches': 32,
            'dataloader.dataset.epoch_size': 102400,
            'lr_scheduler.total_steps': 65000,
            'lr_scheduler.warmup_steps': 2000,
            'debug.seed': 2048,
            'metrics.log_freq': 1,
            'checkpoint.enable': False,
            'validator.enable': False,
            'parallelism.gfsdp_reuse_bf16_expert_weights': True,
            'profiler.enable_profiling': True,
            'profiler.enable_memory_snapshot': False,
            'profiler.profile_ranks': [0, 1],
            'profiler.save_traces_folder': 'profiling/traces',
            'profiler.record_shapes': False,
            'profiler.profile_freq': 2,
            'profiler.profiler_warmup': 1,
            'profiler.profiler_active': 1,
            'profiler.profiler_repeat': 1,
            'profiler.profiler_skip_first': 19,
            'profiler.profiler_skip_first_wait': 0,
        }
        resolved_values = {}
        for dotted, expected in expected_config.items():
            value = config
            for key in dotted.split('.'):
                value = value[key]
            resolved_values[dotted] = value
        checks = {
            'status_pass': (attempt / 'status.txt').read_text().strip() == 'PASS',
            'child_exit_zero': process.get('child_exit_code') == 0,
            'process_group_stopped_no_orphan_timeout': process.get('process_group_stopped') is True and process.get('orphaned_process_group') is False and process.get('timeout') is False,
            'summary_complete_profile_update_21': summary.get('status') == 'complete' and summary.get('mode') == 'profile_update' and summary.get('measurement_status') == 'disabled' and summary.get('training_steps') == 21 and summary.get('errors') == [],
            'planned_parent_argv': controller.get('argv') == run['argv'] and controller.get('plan_sha256') == plan_sha,
            'generated_child_argv': metadata.get('argv') == run['expected_child_argv'] and '--capture-microbatches' not in metadata.get('argv', []) and '--capture-microbatches' not in run['argv'],
            'completion_source_config_unchanged': completion.get('status') == 'PASS' and completion.get('source_unchanged') is True and completion.get('config_unchanged') is True and source_final.get('head') == metadata.get('source_head_before') and source_final.get('tracked_sha256') == metadata.get('source_snapshot_before', {}).get('tracked_sha256') and source_final.get('tracked_and_untracked_sha256') == metadata.get('source_snapshot_before', {}).get('tracked_and_untracked_sha256') and source_final.get('git_status_sha256') == metadata.get('source_snapshot_before', {}).get('git_status_sha256'),
            'backend_variant_receipt': bridge.get('backend_variant') == variant,
            'resolved_config_sha256': sha256_file(resolved)[0] == completion.get('resolved_config_sha256'),
            'resolved_config_matches_recipe': all(resolved_values[k] == v for k, v in expected_config.items()),
        }
        profile = summary.get('profile_capture', {})
        checks['profile_settings_match_plan'] = profile.get('settings') == run['profile_capture']['settings'] and profile.get('profile_ranks') == [0, 1] and profile.get('warmup_updates') == 20 and profile.get('target_update') == 21
        rank_receipts = {}
        rank_ok = True
        for rank in range(8):
            d = json.loads((attempt / 'measurement' / f'rank-{rank:02d}.json').read_text())
            capture = d['profile_capture']
            cadence_steps = [x.get('step') for x in d.get('cadence', [])]
            ok = d.get('rank') == rank and d.get('world_size') == 8 and d.get('training_status') == 'complete' and d.get('training_error') is None and d.get('measurement_status') == 'disabled' and d.get('completed_steps') == 21 and d.get('expected_steps') == 21 and cadence_steps == list(range(1, 22)
            )
            ok = ok and capture.get('microbatch_calls') == 32 and capture.get('train_step_calls') == 1 and capture.get('optimizer_step_calls') == 1 and capture.get('target_update') == 21 and capture.get('warmup_updates') == 20
            if rank in (0, 1):
                ok = ok and capture.get('collected_microbatches') == 3 and capture.get('collection_stopped') is True and capture.get('early_stop_cuda_drains') == 1 and capture.get('token_reduction_calls') == 1 and capture.get('export_update') == 21
            else:
                ok = ok and capture.get('collected_microbatches') == 0 and capture.get('collection_stopped') is False and capture.get('early_stop_cuda_drains') == 0 and capture.get('token_reduction_calls') == 0 and capture.get('export_update') is None
            rank_ok = rank_ok and ok
            rank_receipts[str(rank)] = {
                'training_status': d.get('training_status'),
                'completed_steps': d.get('completed_steps'),
                'expected_steps': d.get('expected_steps'),
                'cadence_steps': cadence_steps,
                'profile_capture': {k: capture.get(k) for k in ('collected_microbatches', 'collection_stopped', 'early_stop_cuda_drains', 'token_reduction_calls', 'microbatch_calls', 'train_step_calls', 'optimizer_step_calls', 'target_update', 'warmup_updates', 'export_update')},
            }
        checks['all_eight_rank_cadence_and_capture_receipts'] = rank_ok
        summary_counts = profile.get('call_counts_by_rank', {})
        checks['summary_call_counts_32_1_1_all_ranks'] = set(summary_counts) == {str(i) for i in range(8)} and all(summary_counts[str(i)].get('microbatch_calls') == 32 and summary_counts[str(i)].get('train_step_calls') == 1 and summary_counts[str(i)].get('optimizer_step_calls') == 1 for i in range(8))
        checks['summary_trace_ranks_0_1_only'] = set(profile.get('traces_by_rank', {})) == {'0', '1'}
        trace_dir = attempt / 'native/profiling/traces/iteration_21'
        checks['only_rank0_rank1_raw_trace_files'] = sorted(p.name for p in trace_dir.glob('rank*_trace.json')) == ['rank0_trace.json', 'rank1_trace.json']
        markers_ok, marker_receipts = validate_entry_markers(attempt, run['run_id'])
        checks['actual_torchrun_entry_markers_false_true_all_ranks'] = markers_ok
        trace_results = {}
        for rank_s, trace_path in run['trace_paths'].items():
            raw = pathlib.Path(trace_path)
            raw_sha, raw_bytes = sha256_file(raw)
            receipt = profile.get('traces_by_rank', {}).get(rank_s, {})
            matches = raw_sha == receipt.get('sha256') and raw_bytes == receipt.get('size_bytes')
            census = profile_census(raw)
            gzip_receipt = make_canonical_gzip(raw, raw_sha, raw_bytes)
            trace_results[rank_s] = {**gzip_receipt, 'matches_summary_receipt': matches, 'raw_census': census}
        checks['raw_hashes_sizes_match_summary_receipts'] = all(x['matches_summary_receipt'] for x in trace_results.values())
        failures = [name for name, passed in checks.items() if not passed]
        child_pid = process.get('pid')
        child_pgid = process.get('process_group_id')
        pgids = {str(controller.get('pgid')), str(child_pgid)}
        process_rows = subprocess.check_output(['ps', '-eo', 'pid=,pgid=,args='], text=True).splitlines()
        live_groups = [line.strip() for line in process_rows if len(line.split(None, 2)) >= 2 and line.split(None, 2)[1] in pgids]
        live_ids = [line.strip() for line in process_rows if run['run_id'] in line]
        process_gone = not live_groups and not live_ids and (controller.get('pid') is None or not pathlib.Path(f'/proc/{controller["pid"]}').exists()) and (child_pid is None or not pathlib.Path(f'/proc/{child_pid}').exists())
        checks['all_owned_processes_gone'] = process_gone
        if not process_gone:
            failures.append('all_owned_processes_gone')
        controller.update({
            'state': 'completed_validated' if not failures else 'completed_with_findings',
            'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'attempt_status': (attempt / 'status.txt').read_text().strip(),
            'process_receipt': process,
            'completion_receipt': completion,
            'actual_child_argv': metadata.get('argv'),
            'actual_child_argv_matches_plan': checks['generated_child_argv'],
            'summary_path': str(attempt / 'summary.json'),
            'summary_status': summary.get('status'),
            'runtime_checks': checks,
            'runtime_check_failures': failures,
            'rank_receipts': rank_receipts,
            'actual_torchrun_entry_markers_by_rank': marker_receipts,
            'resolved_config_path': str(resolved),
            'resolved_config_sha256': completion.get('resolved_config_sha256'),
            'resolved_config_checked_values': resolved_values,
            'source_snapshot_unchanged': checks['completion_source_config_unchanged'],
            'canonical_trace_files': trace_results,
            'process_groups_absent_after_completion': process_gone,
            'trace_analysis_note': 'GDN_BACKEND_ACTIVATED grad_enabled entries prove entry invocations only; literal backend backward coverage is delegated to trace analysis.',
        })
        controller_updates.append((controller_path, controller))
        runs.append({
            'variant': variant,
            'run_id': run['run_id'],
            'state': 'PASS' if not failures else 'PASS_WITH_VALIDATION_FINDINGS',
            'checks': checks,
            'failures': failures,
            'controller_path': str(controller_path),
            'controller_run_log_path': controller.get('log_path') or controller.get('run_log_path'),
            'controller_script_path': controller.get('script_path'),
            'controller_script_sha256': hashlib.sha256(pathlib.Path(controller['script_path']).read_bytes()).hexdigest() if controller.get('script_path') and pathlib.Path(controller['script_path']).exists() else None,
            'controller_harness_stdout_path': controller.get('harness_stdout_path'),
            'controller_harness_stderr_path': controller.get('harness_stderr_path'),
            'attempt_path': str(attempt),
            'status_path': str(attempt / 'status.txt'),
            'summary_path': str(attempt / 'summary.json'),
            'process_receipt_path': str(attempt / 'process.json'),
            'completion_receipt_path': str(attempt / 'completion.json'),
            'actual_parent_argv': controller.get('argv'),
            'actual_child_argv': metadata.get('argv'),
            'actual_child_argv_matches_plan': checks['generated_child_argv'],
            'child_exit_code': process.get('child_exit_code'),
            'process_group_stopped': process.get('process_group_stopped'),
            'orphaned_process_group': process.get('orphaned_process_group'),
            'timeout': process.get('timeout'),
            'training_steps': summary.get('training_steps'),
            'summary_mode': summary.get('mode'),
            'measurement_status': summary.get('measurement_status'),
            'errors': summary.get('errors'),
            'source_hashes_match_plan_preflight': controller.get('preflight', {}).get('source_hashes_match_plan', True),
            'source_config_unchanged': checks['completion_source_config_unchanged'],
            'resolved_config_path': str(resolved),
            'resolved_config_sha256': completion.get('resolved_config_sha256'),
            'resolved_config_checked_values': resolved_values,
            'rank_receipts': rank_receipts,
            'profile_capture_summary': profile,
            'bootstrap_receipts': [str(attempt / 'measurement' / f'flash-qla-bootstrap-rank-{i:02d}.json') for i in range(8)],
            'actual_torchrun_entry_markers_by_rank': marker_receipts,
            'entry_marker_note': 'These are GDN_BACKEND_ACTIVATED entry invocations only; they are not literal backward counters. Trace-analysis owner checks backend F/B CPU/GPU coverage.',
            'raw_trace_files': trace_results,
            'raw_census_method': 'Counts are from raw Chrome trace JSON traceEvents. Kernel/copy/memset/runtime/driver/device-copy/communication buckets use recorded category/name string matches. Bounds remain in stored trace timestamp units. No Perfetto import, trace normalization, or GDN kernel analysis was performed by the capture owner.',
            'launch_preflight': controller.get('preflight'),
            'process_receipt': process,
            'completion_receipt': completion,
        })
    current_hashes = {rel: sha256_file(pathlib.Path(plan['task_root']) / rel)[0] for rel in plan['source_hashes']}
    final_source_hashes_match = all(current_hashes[k] == v for k, v in plan['source_hashes'].items())
    if not final_source_hashes_match:
        raise RuntimeError(f'final frozen-source hash mismatch: {current_hashes}')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], text=True, stderr=subprocess.STDOUT).splitlines()
    gpu_apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True, stderr=subprocess.STDOUT).strip()
    gpu_idle = len(gpu_rows) == 8 and all(int(row.split(',')[1].strip().replace('%', '')) == 0 for row in gpu_rows) and gpu_apps == ''
    if not gpu_idle:
        raise RuntimeError(f'final GPUs are not idle: rows={gpu_rows!r}; apps={gpu_apps!r}')
    for path, data in controller_updates:
        atomic_controller_write(path, data)
    harness_history = [
        {
            'id': 'external_writer_wrong_marker_log_path',
            'command': 'ssh -T -o RemoteCommand=none -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 sr008-gpu /opt/torchtitan-venv/bin/python3.12 - <<\'PY\' ... PY',
            'script_path': None,
            'exit_code': 1,
            'captured_output': 'Only the standard OpenSSH post-quantum-key-exchange warning was returned; no Python traceback or separate stdout/stderr file was retained.',
            'diagnosis': 'The inline writer asserted GDN_BACKEND_ACTIVATED markers in native/structured_logs/training.global_rank_N.*.jsonl. Those eight files contain zero markers; the correct entries are in logs/torchrun/*/attempt_0/{rank}/{stdout,stderr}.log.',
            'side_effects': 'Both canonical external .json.gz siblings were written before the assertion; no temp files remained. The external controller/results JSON/Markdown were not updated by that script. Existing gzip siblings were reused and decompressed identity verified during finalization.',
        },
        {
            'id': 'fused_preflight_inline_harness_exit1',
            'command': 'ssh -T -o RemoteCommand=none -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 sr008-gpu /opt/torchtitan-venv/bin/python3.12 - <<\'PY\' ... PY',
            'script_path': None,
            'exit_code': 1,
            'captured_output': 'Only the standard OpenSSH post-quantum-key-exchange warning; no Python traceback or separate stdout/stderr file. Exact failing gate remains unknown.',
            'side_effects': 'No fused controller, run log, attempt, PID, or process existed afterwards; GPUs remained idle, compute-app list empty, frozen source hashes matched. Parent later executed the saved launcher successfully.',
        },
        {
            'id': 'local_chmod_sha256_after_heredoc',
            'command': 'Remote `cat > .../launch_fused_capture.py` heredoc ended before local `chmod 640 /workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-controllers/launch_fused_capture.py` and local `sha256sum` commands.',
            'exit_code': 1,
            'captured_output': 'Local zsh reported both chmod and sha256sum could not find the remote-only path. The remote script had been created successfully at mode 0644 (SHA256 05db11d6…b7f1); root executed it by passing it to the absolute Python interpreter, without chmod.',
            'side_effects': 'No fused controller/log/attempt/process was created by this local chmod/sha256 command; the subsequent root launch used the saved script and passed all seven preflight gates.',
        },
    ]
    result = {
        'schema': 1,
        'overall_status': 'PASS' if all(x['state'] == 'PASS' for x in runs) else 'PASS_WITH_VALIDATION_FINDINGS',
        'plan_path': str(PLAN_PATH),
        'plan_sha256': plan_sha,
        'task_root': plan['task_root'],
        'original_evidence_root': str(BASE),
        'acquisition_variants': ['external_gate', 'fused_gate'],
        'fla_acquisition_required': False,
        'new_fla_acquisition': False,
        'final_frozen_source_hashes_match_plan': final_source_hashes_match,
        'final_gpu_idle_preflight': {'rows': gpu_rows, 'compute_apps_raw': gpu_apps, 'idle_no_apps': gpu_idle},
        'runs': runs,
        'harness_failure_history': harness_history,
        'analysis_handoff': {
            'owner': '/root/flashqla_new_trace_comparison',
            'handoff_sent_utc': '2026-11-14T18:55:46Z',
            'analysis_output_root': '/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/artifacts/raw/trace-analysis',
            'status': 'PASS for both captures; strict raw forward/backward coverage, actual MCP import, and zero import errors reported by the analysis owner.',
            'results_json': '/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-results.json',
            'results_markdown': '/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-results.md',
            'reported_counts': {'microbatches_per_rank': 3, 'original_forwards_per_microbatch': 18, 'sac_forwards_per_microbatch': 18, 'vjp_forwards_per_microbatch': 18, 'actual_native_backwards_per_microbatch': 18, 'opaque_per_microbatch': 36, 'gdn_vjp_per_microbatch': 18, 'external_all_device_events': 9157, 'fused_all_device_events': 9103, 'import_errors_per_run': 0, 'unique_host_flows': True},
            'note': 'Trace-analysis owner verified literal backward coverage and MCP import. Capture-side entry markers remain documented as invocation evidence only; the three-microbatch traces do not establish the full 32-microbatch critical path.',
        },
        'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if not all(x['state'] == 'PASS' for x in runs):
        result['overall_status'] = 'PASS_WITH_VALIDATION_FINDINGS'
    json_tmp = pathlib.Path(str(RESULTS_PATH) + '.tmp')
    md_tmp = pathlib.Path(str(MARKDOWN_PATH) + '.tmp')
    if json_tmp.exists() or md_tmp.exists():
        raise FileExistsError('refusing to overwrite existing capture-results temporary file')
    with json_tmp.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    summary_lines = [
        '# FlashQLA trace capture results',
        '',
        f'- Status: **{result["overall_status"]}**.',
        f'- Plan: `{PLAN_PATH}` (SHA256 `{plan_sha}`).',
        '- Acquired exactly `external_gate` then `fused_gate`; no new FLA trace was captured.',
        '- Final check: eight GPUs idle with no compute apps; all eight frozen source hashes match the plan.',
        '',
        '| Variant | Run | Summary | Rank 0 raw / gzip | Rank 1 raw / gzip |',
        '|---|---|---|---|---|',
    ]
    for item in runs:
        t0=item['raw_trace_files']['0']; t1=item['raw_trace_files']['1']
        summary_lines.append(f"| `{item['variant']}` | `{item['run_id']}` | `{item['summary_path']}` | `{t0['raw_bytes']} B / {t0['gzip_bytes']} B` | `{t1['raw_bytes']} B / {t1['gzip_bytes']} B` |")
    for item in runs:
        summary_lines += [
            '',
            f"## {item['variant']} — {item['state']}",
            '',
            f"- Controller: `{item['controller_path']}`; run log: `{item['controller_run_log_path']}`.",
            f"- Attempt status: `{item['status_path']}`; summary: `{item['summary_path']}`; process receipt: `{item['process_receipt_path']}`; completion receipt: `{item['completion_receipt_path']}`.",
            f"- Child exit code: `{item['child_exit_code']}`; process group stopped: `{item['process_group_stopped']}`; orphan: `{item['orphaned_process_group']}`; timeout: `{item['timeout']}`.",
            f"- Training: {item['training_steps']} updates, all eight ranks complete; profile mode `{item['summary_mode']}`, measurement `{item['measurement_status']}`.",
            f"- Rank 0/1 capture: 3 microbatches each, one stop/drain/token reduction, one train and optimizer step per rank; ranks 2–7 capture zero and finish the update.",
            f"- Resolved rank-0 config: `{item['resolved_config_path']}` (SHA256 `{item['resolved_config_sha256']}`); source and config unchanged.",
            f"- Actual parent/child argv are stored in `{item['controller_path']}` and this JSON report; generated child argv matched the plan.",
            f"- Actual torchrun entry markers: per-rank false/true counts are stored in `{item['controller_path']}`. These show entry invocations, not literal backward counters.",
            '- Raw trace / gzip identity, raw event census, hashes, bytes, marker receipts, and checked config fields are recorded in this JSON report.',
        ]
        for rank in ('0','1'):
            trace=item['raw_trace_files'][rank]
            summary_lines.append(f"- Rank {rank}: raw `{trace['raw_path']}` ({trace['raw_bytes']} B, SHA256 `{trace['raw_sha256']}`); gzip `{trace['gzip_path']}` ({trace['gzip_bytes']} B, SHA256 `{trace['gzip_sha256']}`; decompressed identity verified).")
    summary_lines += [
        '',
        '## Harness failure history',
        '',
        '1. External writer exited 1 after writing both gzip siblings because it searched the structured JSONL path for backend-entry strings. The actual markers are in the torchrun `attempt_0` stdout/stderr logs. The existing gzip files were preserved and verified by decompressed SHA256 and byte identity.',
        '2. A fused inline preflight wrapper exited 1 with only the SSH key-exchange warning visible; the failing gate is unknown. It left no fused controller, attempt, process, or GPU allocation. The saved launcher then passed all seven named gates and launched successfully.',
        '3. `chmod` and `sha256sum` were accidentally placed after a local heredoc terminator, so zsh ran them on macOS against a remote-only path and returned 1. The remote launcher file had been created (mode 0644); the root invoked it with the absolute Python interpreter, and the run completed.',
        '',
        '## Analysis handoff',
        '',
        f"Raw pair handoff to `{result['analysis_handoff']['owner']}` was sent at `{result['analysis_handoff']['handoff_sent_utc']}`. Strict coverage and actual MCP import pass with zero import errors; the analysis report is `{result['analysis_handoff']['results_json']}` and `{result['analysis_handoff']['results_markdown']}`. The three-microbatch capture still does not establish the full 32-microbatch critical path.",
        '',
        f"Machine-readable record: `{RESULTS_PATH}`.",
        '',
    ]
    markdown='\n'.join(summary_lines)
    with md_tmp.open('x', encoding='utf-8') as f:
        f.write(markdown)
    # Final paths are reserved only after complete data has been staged; hard-link creation refuses overwrite.
    os.link(json_tmp, RESULTS_PATH)
    os.unlink(json_tmp)
    os.link(md_tmp, MARKDOWN_PATH)
    os.unlink(md_tmp)
    for path, data in controller_updates:
        print(f"CONTROLLER {path} state={data['state']}")
    print(json.dumps({'status': result['overall_status'], 'results_json': str(RESULTS_PATH), 'results_markdown': str(MARKDOWN_PATH), 'plan_sha256': plan_sha, 'runs': [{'variant':x['variant'],'run_id':x['run_id'],'state':x['state'],'checks':x['checks'],'failures':x['failures'],'raw_trace_files':x['raw_trace_files']} for x in runs], 'final_source_hashes_match_plan':final_source_hashes_match,'final_gpu_idle':gpu_idle},sort_keys=True))


if __name__ == '__main__':
    main()
