"""Create separate task-local patched reference; never modify pinned source."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parent
source=ROOT/'runtime/source/quack-bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8'
patched=ROOT/'runtime/patched/quack-bb0e6b2-colvec-v'
if patched.exists():
    raise RuntimeError('Refusing to replace an existing patched reference')
shutil.copytree(source,patched)
patch=ROOT/'vendor-patches/0001-pin-scaled-exp-v-axis.patch'
subprocess.run(['patch','-p1','-i',str(patch)],cwd=patched,check=True)
name='quack/linear_cross_entropy.py'
receipt={'upstream_revision':'bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8','original_sha256':hashlib.sha256((source/name).read_bytes()).hexdigest(),'patched_sha256':hashlib.sha256((patched/name).read_bytes()).hexdigest(),'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'reason':'v has shape (1,M), scales each dx row; explicit ColVecLoad disambiguates M == D without changing the arithmetic'}
(ROOT/'vendor-patches/receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
