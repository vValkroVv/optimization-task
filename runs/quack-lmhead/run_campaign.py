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
env.update(PYTHONPATH=str(AREA/'runtime/source/quack-bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8'), CUDA_VISIBLE_DEVICES='0', TORCHINDUCTOR_CUDAGRAPHS='0', TORCHINDUCTOR_CACHE_DIR=str(AREA/'artifacts/cache/inductor'), TRITON_CACHE_DIR=str(AREA/'artifacts/cache/triton'))
plan=[('native','timing','A0'),('native','timing','AA0'),('q8192','timing','B8192-0'),('native','timing','A2'),('q8192','timing','B8192-1'),('native','profile','profile-native'),('q8192','profile','profile-q8192'),('native','numeric','numeric-native'),('q8192','numeric','numeric-q8192'),('oracle','numeric','numeric-oracle')]
results=[]
deadline=time.monotonic()+10800
for variant,mode,name in plan:
    if time.monotonic()>deadline:
        raise RuntimeError('campaign deadline')
    command=[str(PYTHON),str(AREA/'benchmark.py'),'--variant',variant,'--mode',mode,'--output',str(RAW/name)]
    with (RAW/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(1800,deadline-time.monotonic()))
    results.append({'name':name,'command':command,'returncode':result.returncode})
    (RAW/'campaign-state.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results[-1]),flush=True)
    if result.returncode:
        raise RuntimeError('failed '+name)
