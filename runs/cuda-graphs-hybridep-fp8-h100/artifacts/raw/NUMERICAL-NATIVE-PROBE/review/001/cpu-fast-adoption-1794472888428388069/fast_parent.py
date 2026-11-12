import subprocess,sys,time,os
from pathlib import Path
time.sleep(0.08)
p=subprocess.Popen([sys.executable,"-c",'import time; time.sleep(2)'],start_new_session=True)
Path('/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/review/001/cpu-fast-adoption-1794472888428388069/child.pid').write_text(str(p.pid))
os._exit(0)
