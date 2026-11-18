"""Package compact receipts; retain large GPU artifacts only on authorized NFS."""
import hashlib
import json
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parent
files=sorted((ROOT/'artifacts/raw').glob('*/result.json'))
files+=sorted((ROOT/'artifacts/raw').glob('*state.json'))
files+=sorted((ROOT/'artifacts/raw').glob('*campaign.log'))
files+=sorted((ROOT/'artifacts/raw').glob('pipeline*.log'))
files += [ROOT/'artifacts/input/receipt.json', ROOT/'artifacts/setup/setupmanifest.json', ROOT/'vendor-patches/receipt.json', ROOT/'artifacts/summary.json']
trace_receipts=[]
for path in sorted((ROOT/'artifacts/raw').glob('profile-*/trace.json')):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    trace_receipts.append({'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
trace_path=ROOT/'artifacts/trace-receipts.json'
trace_path.write_text(json.dumps(trace_receipts,indent=2))
files.append(trace_path)
with tarfile.open(ROOT/'artifacts/compact-evidence.tar.gz','w:gz') as archive:
    for path in files:
        if path.exists(): archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
print(json.dumps({'files':len(files),'archive_bytes':(ROOT/'artifacts/compact-evidence.tar.gz').stat().st_size}),flush=True)
