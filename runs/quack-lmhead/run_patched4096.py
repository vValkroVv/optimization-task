"""Bounded serial one-GPU campaign; no cleanup, optimizer, or full training."""
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
AREA=ROOT/'runs/quack-lmhead'
RAW=AREA/'artifacts/raw'
PYTHON=AREA/'runtime/venv/bin/python'
env=dict(os.environ)
env.update(PYTHONPATH=str(AREA/'runtime/patched/quack-bb0e6b2-colvec-v'), CUDA_VISIBLE_DEVICES='0', TORCHINDUCTOR_CUDAGRAPHS='0', TORCHINDUCTOR_CACHE_DIR=str(AREA/'artifacts/cache/inductor'), TRITON_CACHE_DIR=str(AREA/'artifacts/cache/triton'))
plan=[('q4096','timing','pipeline-patched-q4096'),('native','timing','A3'),('q4096','timing','B4096-0'),('native','timing','A4'),('q4096','timing','B4096-1'),('q4096','profile','profile-q4096'),('q4096','numeric','numeric-q4096'),('oracle','numeric','numeric-oracle-patched4096')]
results=[]
deadline=time.monotonic()+10800
for variant,mode,name in plan:
    if time.monotonic()>deadline:
        raise RuntimeError('campaign deadline')
    command=[str(PYTHON),str(AREA/'benchmark.py'),'--variant',variant,'--mode',mode,'--output',str(RAW/name)]
    if name.startswith('pipeline'):
        command += ['--warmup','1','--samples','1']
    with (RAW/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(1800,deadline-time.monotonic()))
    results.append({'name':name,'command':command,'returncode':result.returncode})
    (RAW/'patched4096-state.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results[-1]),flush=True)
    if result.returncode:
        raise RuntimeError('failed '+name)
