"""One fresh-process execution after exact allocation and capacity checks."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
assert not (HERE / "launch.json").exists(), "No automatic retry"
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"], text=True)
apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
expected = [line.split(",")[1].strip() for line in (HERE.parent.parent / "reconciliation/002/gpu.csv").read_text().splitlines()]
assert [line.split(",")[1].strip() for line in gpu.splitlines()] == expected
assert not apps.strip(), apps
assert all("H100" in line and int(line.split(",")[4].strip().split()[0]) > 75000 for line in gpu.splitlines())
assert os.statvfs(HERE).f_bavail * os.statvfs(HERE).f_frsize > 20 * 1024**3
env = os.environ.copy()
env.update({
    "PYTHONPATH": str(ROOT / "gift"),
    "CUDA_VISIBLE_DEVICES": expected[0],
    "DG_JIT_SHARED_CACHE_DIR": str(ROOT / "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/cache/deepgemm_shared"),
    "DG_JIT_DEBUG": "1",
    "DG_JIT_CACHE_DIR": str(HERE.parent / "001/cache"),
    "DG_CACHE_FORCE_SKIP": "1",
    "TORCHINDUCTOR_CUDAGRAPHS": "0",
    "TORCHINDUCTOR_MIX_ORDER_REDUCTION": "0",
    "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
    "CUDA_COREDUMP_GENERATION_FLAGS": "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory,skip_abort",
    "CUDA_COREDUMP_SHOW_PROGRESS": "1",
    "CUDA_COREDUMP_FILE": str(HERE / "core.%p.%t.nvcudmp"),
})
for key in ("CUDA_LAUNCH_BLOCKING", "TORCH_LOGS", "TORCH_COMPILE_DEBUG", "TORCH_TRACE"):
    env.pop(key, None)
command = [sys.executable, str(HERE / "fixture.py"), "--gpu"]
receipt = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": command, "environment": {key: env[key] for key in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "DG_JIT_SHARED_CACHE_DIR", "DG_JIT_CACHE_DIR", "DG_CACHE_FORCE_SKIP", "DG_JIT_DEBUG", "TORCHINDUCTOR_CUDAGRAPHS", "TORCHINDUCTOR_MIX_ORDER_REDUCTION", "CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "CUDA_COREDUMP_GENERATION_FLAGS", "CUDA_COREDUMP_SHOW_PROGRESS", "CUDA_COREDUMP_FILE")}, "gpu": gpu, "apps": apps, "fixture_sha256": hashlib.sha256((HERE / "fixture.py").read_bytes()).hexdigest()}
(HERE / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
with (HERE / "stdout.log").open("w") as out, (HERE / "stderr.log").open("w") as err:
    child = subprocess.Popen(command, env=env, stdout=out, stderr=err)
    (HERE / "owner.json").write_text(json.dumps({"pid": child.pid, "command": command}) + "\n")
    try:
        code = child.wait(timeout=180)
    except subprocess.TimeoutExpired:
        print("Child still active; preserve any in-progress core and inspect exact task PID", flush=True)
        raise
(HERE / "completion.json").write_text(json.dumps({"returncode": code, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
print(json.dumps({"returncode": code}), flush=True)
sys.exit(code)
