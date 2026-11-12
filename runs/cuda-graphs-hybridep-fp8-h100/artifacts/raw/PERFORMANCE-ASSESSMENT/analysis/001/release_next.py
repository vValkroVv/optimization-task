"""Release one root-authorized successor after verified predecessor completion."""
import csv,datetime,importlib.util,json,os,pathlib,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('extract',HERE/'extract.py');ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
n=sys.argv[1];assert n in ['007','008','009','010']
previous=f'{int(n)-1:03d}'
valid=ex.extract(previous);assert valid['valid'];frozen=ex.freeze()
p=ex.PERF/'attempts'/n
expected=json.loads((ex.PERF/'attempts/006/root-availability.json').read_text())['gpu_rows']
command=['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits']
rows=[[v.strip() for v in row] for row in csv.reader(subprocess.check_output(command,text=True).splitlines())]
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
assert rows==expected and not apps,(rows,apps)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
availability={'at_utc':now,'gpu_rows':rows,'compute_apps':[],'ready':True,'previous_valid_attempt':previous,'freeze':frozen}
with (p/'root-availability.json').open('x') as f:json.dump(availability,f,indent=2);f.write('\n')
release=json.loads((ex.PERF/'attempts/006/release.json').read_text())
release.update(scope=f'PERFORMANCE-ASSESSMENT/attempts/{n}',at_utc=now,variant={'007':'off-AA2','008':'on-B1','009':'off-A2','010':'on-B2'}[n])
with (p/'release.json').open('x') as f:json.dump(release,f,indent=2);f.write('\n')
env=os.environ.copy();env['EXPECTED_SOURCE_MANIFEST_SHA256']=release['source_manifest_sha256']
command=['/opt/torchtitan-venv/bin/python3.12',str(ex.PERF/'preparation/002/controller.py'),n]
with (p/'controller.log').open('xb') as log:
 process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=ex.TASK)
 ticks=int(pathlib.Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
 receipt={'at_utc':now,'pid':process.pid,'start_ticks':ticks,'command':command,'previous_valid_attempt':previous}
 with (p/'root-launch.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
