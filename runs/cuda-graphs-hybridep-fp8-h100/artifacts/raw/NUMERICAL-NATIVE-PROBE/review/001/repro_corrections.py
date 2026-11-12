"""Bounded CPU reproductions of native-preparation correction failures."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path.cwd()
PREP = ROOT / 'runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/001'
OUT = Path(__file__).resolve().parent
source = (PREP / 'preflight.sh').read_text()
block = source[source.index('topology_ready = ('):source.index('\ngpu_ready = (')]
namespace = {'header': list(range(8)), 'gpu_labels': [f'GPU{i}' for i in range(8)],
             'matrix': {f'GPU{i}': ['X' if i == j else 'NV18' for j in range(8)] for i in range(8)}}
try:
    exec(block, namespace)
    print('topology_fixture_result', namespace['topology_ready'], flush=True)
except Exception as error:
    print('topology_fixture_error', type(error).__name__, str(error), flush=True)

case = OUT / f'cpu-fast-adoption-{time.time_ns()}'
case.mkdir()
entry = case / 'fast_parent.py'
child_pid_file = case / 'child.pid'
child_code = 'import time; time.sleep(2)'
entry.write_text('import subprocess,sys,time,os\nfrom pathlib import Path\ntime.sleep(0.08)\np=subprocess.Popen([sys.executable,"-c",'+repr(child_code)+'],start_new_session=True)\nPath('+repr(str(child_pid_file))+').write_text(str(p.pid))\nos._exit(0)\n')
raw = Path(f'/proc/{os.getpid()}/stat').read_text()
ticks = raw[raw.rfind(')')+2:].split()[19]
cmd = [sys.executable, str(PREP/'phase_process_owner.py'), '--phase','cpu-fast-adoption',
       '--parent-pid',str(os.getpid()),'--parent-start-ticks',ticks,'--entrypoint',str(entry),
       '--cwd',str(ROOT),'--ledger',str(case/'ledger.jsonl'),'--status',str(case/'status.json'),
       '--owner-record',str(case/'owner.json'),'--log',str(case/'phase.log'),
       '--deadline-seconds','3','--term-grace','0.2','--kill-grace','0.2','--',sys.executable,str(entry)]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
print('owner_returncode', p.returncode, flush=True)
print('owner_status', json.loads((case/'status.json').read_text()), flush=True)
child_pid = int(child_pid_file.read_text())
print('unrecorded_child_alive_after_owner_exit', Path(f'/proc/{child_pid}').exists(), flush=True)
print('ledger', (case/'ledger.jsonl').read_text(), flush=True)
# The bounded CPU sleeper exits naturally; no signals or cleanup are used.
time.sleep(2.1)
print('cpu_repro_finished_without_signalling', flush=True)
