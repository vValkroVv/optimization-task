#!/opt/torchtitan-venv/bin/python3.12
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import traceback

BASE = pathlib.Path('/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla')
PLAN_PATH = BASE / 'trace-plan.json'
CONTROLLER_DIR = BASE / 'trace-capture-controllers'
RUN_ID = 'fqla-trace-fused-001'
CONTROLLER_PATH = CONTROLLER_DIR / f'{RUN_ID}.json'
RUN_LOG = CONTROLLER_DIR / f'{RUN_ID}.run.log'
HARNESS_OUT = CONTROLLER_DIR / f'{RUN_ID}.harness.stdout.log'
HARNESS_ERR = CONTROLLER_DIR / f'{RUN_ID}.harness.stderr.log'
TMP_CONTROLLER = CONTROLLER_PATH.with_suffix('.json.tmp')

# Open persistent, exclusive harness streams before preflight so failures retain tracebacks.
out_fd = os.open(HARNESS_OUT, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
try:
    err_fd = os.open(HARNESS_ERR, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
except BaseException:
    os.write(out_fd, b'failed to exclusively open harness stderr log\n')
    os.close(out_fd)
    raise
os.dup2(out_fd, 1)
os.dup2(err_fd, 2)
os.close(out_fd)
os.close(err_fd)

record = {
    'schema': 1,
    'run_id': RUN_ID,
    'variant': 'fused_gate',
    'state': 'checking_preflight',
    'script_path': str(pathlib.Path(__file__).resolve()),
    'plan_path': str(PLAN_PATH),
    'controller_path': str(CONTROLLER_PATH),
    'run_log_path': str(RUN_LOG),
    'harness_stdout_path': str(HARNESS_OUT),
    'harness_stderr_path': str(HARNESS_ERR),
    'gates': [],
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
controller_created = False

def persist():
    with TMP_CONTROLLER.open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(TMP_CONTROLLER, CONTROLLER_PATH)

def gate(name, passed, evidence):
    item = {'name': name, 'passed': bool(passed), 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'evidence': evidence}
    record['gates'].append(item)
    persist()
    print(json.dumps({'gate': name, 'passed': bool(passed), 'evidence': evidence}, sort_keys=True), flush=True)
    if not passed:
        record['state'] = 'preflight_failed'
        record['failed_gate'] = name
        persist()
        raise RuntimeError(f'preflight gate failed: {name}')

def main():
    global controller_created
    if CONTROLLER_PATH.exists() or RUN_LOG.exists() or TMP_CONTROLLER.exists():
        raise FileExistsError('exclusive fresh controller/run-log gate failed')
    with CONTROLLER_PATH.open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
    controller_created = True
    plan_bytes = PLAN_PATH.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    run = next(x for x in plan['runs'] if x['variant'] == 'fused_gate')
    if run['run_id'] != RUN_ID:
        raise RuntimeError(f'planned run_id mismatch: {run["run_id"]}')
    checkout = pathlib.Path(plan['task_root'])
    record.update({'plan_sha256': plan_sha, 'checkout': str(checkout), 'cwd': str(checkout), 'argv': run['argv'], 'expected_child_argv': run['expected_child_argv'], 'attempt_path': run['attempt']})
    gate('native_external_complete',
         (BASE.parent.parent / 'unused').exists() is False and
         pathlib.Path(next(x for x in plan['runs'] if x['variant'] == 'external_gate')['attempt'], 'status.txt').read_text().strip() == 'PASS',
         {'external_status_path': str(pathlib.Path(next(x for x in plan['runs'] if x['variant'] == 'external_gate')['attempt'], 'status.txt')),
          'external_status': pathlib.Path(next(x for x in plan['runs'] if x['variant'] == 'external_gate')['attempt'], 'status.txt').read_text().strip()})
    ext = next(x for x in plan['runs'] if x['variant'] == 'external_gate')
    ext_attempt = pathlib.Path(ext['attempt'])
    ext_process = json.loads((ext_attempt / 'process.json').read_text())
    ext_completion = json.loads((ext_attempt / 'completion.json').read_text())
    ext_controller_path = CONTROLLER_DIR / f'{ext["run_id"]}.json'
    ext_controller = json.loads(ext_controller_path.read_text())
    ext_state_ok = (ext_process.get('child_exit_code') == 0 and
                    ext_process.get('process_group_stopped') is True and
                    ext_process.get('orphaned_process_group') is False and
                    ext_process.get('timeout') is False and
                    ext_completion.get('status') == 'PASS' and
                    ext_completion.get('source_unchanged') is True and
                    ext_completion.get('config_unchanged') is True)
    gate('external_group_stopped_and_clean', ext_state_ok,
         {'process_receipt': str(ext_attempt / 'process.json'), 'process': ext_process,
          'completion_receipt': str(ext_attempt / 'completion.json'), 'completion': ext_completion,
          'external_controller_pgid': ext_controller.get('pgid')})
    source_hashes = {}
    for rel, expected in plan['source_hashes'].items():
        actual = hashlib.sha256((checkout / rel).read_bytes()).hexdigest()
        source_hashes[rel] = actual
    source_ok = all(source_hashes[k] == v for k, v in plan['source_hashes'].items())
    gate('eight_frozen_source_hashes_match_plan', source_ok,
         {'expected': plan['source_hashes'], 'actual': source_hashes})
    attempt = pathlib.Path(run['attempt'])
    gate('fused_attempt_path_fresh', not attempt.exists(), {'attempt_path': str(attempt), 'exists': attempt.exists()})
    gpu_cmd = ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total', '--format=csv,noheader']
    gpu_raw = subprocess.check_output(gpu_cmd, text=True, stderr=subprocess.STDOUT)
    gpu_rows = [line.strip() for line in gpu_raw.splitlines() if line.strip()]
    apps_raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True, stderr=subprocess.STDOUT).strip()
    parsed = {}
    for row in gpu_rows:
        f = [x.strip() for x in row.split(',')]
        parsed[int(f[0])] = {'uuid': f[1], 'utilization_percent': int(f[2].replace('%', '').strip()), 'memory_used': f[3], 'memory_total': f[4]}
    gpu_ok = len(gpu_rows) == 8 and sorted(parsed) == list(range(8)) and all(v['utilization_percent'] == 0 for v in parsed.values()) and apps_raw == ''
    gpu_preflight = {'query_argv': gpu_cmd, 'raw_rows': gpu_rows, 'parsed': {str(k): v for k, v in parsed.items()}, 'compute_apps_raw': apps_raw}
    record['preflight'] = {'gpu': gpu_preflight}
    gate('all_eight_gpus_idle_no_compute_apps', gpu_ok, gpu_preflight)
    proc_rows = subprocess.check_output(['ps', '-eo', 'pid=,pgid=,args='], text=True).splitlines()
    prior_pgids = {str(ext_controller['pgid']), str(ext_process['process_group_id'])}
    prior_group_rows = [line.strip() for line in proc_rows if len(line.split(None, 2)) >= 2 and line.split(None, 2)[1] in prior_pgids]
    prior_run_matches = [line.strip() for line in proc_rows if any(rid in line for rid in (ext['run_id'], run['run_id']))]
    process_ok = not prior_group_rows and not prior_run_matches
    gate('prior_exact_owned_process_groups_gone', process_ok,
         {'prior_pgids': sorted(prior_pgids), 'matching_process_rows': prior_group_rows, 'matching_run_id_rows': prior_run_matches})
    outputs_fresh = not RUN_LOG.exists() and not TMP_CONTROLLER.exists()
    gate('exclusive_fresh_run_log_and_controller', outputs_fresh,
         {'controller_path': str(CONTROLLER_PATH), 'run_log_path': str(RUN_LOG), 'controller_exists': CONTROLLER_PATH.exists(), 'run_log_exists': RUN_LOG.exists(), 'tmp_controller_exists': TMP_CONTROLLER.exists()})
    if '--capture-microbatches' in run['argv'] or '--capture-microbatches' in run['expected_child_argv']:
        raise RuntimeError('unexpected generic --capture-microbatches flag in plan argv')
    record['source_hashes_before'] = source_hashes
    record['preflight_passed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record['state'] = 'preflight_passed_launching'
    persist()
    run_fd = os.open(RUN_LOG, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
    try:
        proc = subprocess.Popen(run['argv'], cwd=str(checkout), stdin=subprocess.DEVNULL,
                                stdout=run_fd, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    finally:
        os.close(run_fd)
    record.update({'state': 'running', 'pid': proc.pid, 'pgid': os.getpgid(proc.pid),
                   'launched_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'detached_popen': {'stdin': 'DEVNULL', 'stdout': str(RUN_LOG), 'stderr': 'STDOUT', 'start_new_session': True, 'close_fds': True}})
    persist()
    print(json.dumps({'state': record['state'], 'run_id': RUN_ID, 'pid': proc.pid, 'pgid': record['pgid'], 'controller_path': str(CONTROLLER_PATH), 'run_log_path': str(RUN_LOG), 'plan_sha256': plan_sha}, sort_keys=True), flush=True)

try:
    main()
except BaseException as exc:
    record['state'] = 'harness_failed'
    record['failure_type'] = type(exc).__name__
    record['failure_message'] = str(exc)
    record['failure_traceback'] = traceback.format_exc()
    if controller_created:
        try:
            persist()
        except BaseException:
            traceback.print_exc(file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
