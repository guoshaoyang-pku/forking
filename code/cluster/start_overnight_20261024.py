#!/usr/bin/env python3
"""Idempotently start the registered queue on one allowlisted host; never retries runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
QUEUE='/home/cfs/guoshaoyang/nglab_queue_overnight_20261024'
CUTOFF=1792800000
REMOTE=r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import shlex
p=json.loads(sys.argv[1]);q=Path(p['queue']);root=Path(p['root']);host=p['host']
assert time.time()<p['cutoff'],'Claim deadline passed; do not extend automatically'
for rel,expected in p['hashes'].items():
    path=q/'jobs.txt' if rel=='jobs.txt' else root/rel
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Fingerprint mismatch: '+rel
assert not (q/f'STOP_{host}').exists(),'New queue is explicitly drained; inspect before resuming'
old=[];current=[]
for f in Path('/proc').glob('[0-9]*/cmdline'):
    try:argv=[x.decode(errors='replace') for x in f.read_bytes().split(b'\0') if x]
    except OSError:continue
    positions=[i for i,x in enumerate(argv) if Path(x).name=='fleet_queue.py']
    if len(positions)!=1 or len(argv)<=positions[0]+1:continue
    queue_arg=argv[positions[0]+1]
    if queue_arg==str(q):current.append(int(f.parent.name))
    elif Path(queue_arg).name in ('nglab_queue_v27','student_queue'):old.append(int(f.parent.name))
assert not old,'Old dispatcher remains; wait for registered STOP drain'
if current:
    print(json.dumps({'host':host,'state':'agent_already_running','pids':current}));sys.exit()
python=str(root/'.venv/bin/python')
args=[python,str(root/'code/cluster/fleet_queue.py'),str(q),str(root),'code/cluster/run_v27.sh','--gpus','0,1,2,3,4,5,6,7','--deadline',str(p['cutoff']),'--python',python]
quote=shlex.quote
loop=f"while [ $(date +%s) -lt {p['cutoff']} ] && [ ! -e {quote(str(q/f'STOP_{host}'))} ]; do {shlex.join(args)}; sleep 15; done"
lock=f'/tmp/nglab_overnight_20261024_{os.getuid()}_{host}.lock'
env=dict(os.environ);env['FLEET_HOST']=host
with (q/f'supervisor_{host}.log').open('a') as log:
    child=subprocess.Popen(['flock','-n',lock,'bash','-c',loop],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={'host':host,'supervisor_pid':child.pid,'started_epoch':time.time(),'cutoff':p['cutoff'],'gpu_allowlist':list(range(8))}
(q/f'supervisor_{host}.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--host',required=True,choices=['a100_perm','a100_t1','a100_t1_2','a100_t1_4']);a=ap.parse_args()
    if time.time()>=CUTOFF:ap.error('08:00 claim cutoff passed; new permission and plan required')
    m=json.loads((ROOT/'docs/plans/overnight-20261024.json').read_text());host='perm' if a.host=='a100_perm' else a.host
    files=['code/train.py','code/ngram_freq.py','code/cluster/run_v27.sh','code/cluster/fleet_queue.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    hashes['jobs.txt']=hashlib.sha256((ROOT/'docs/plans/overnight-20261024.jobs').read_bytes()).hexdigest()
    payload={'host':host,'root':m['host_roots'][host],'queue':QUEUE,'cutoff':CUTOFF,'hashes':hashes}
    command=f"{payload['root']}/.venv/bin/python - "+shlex.quote(json.dumps(payload))
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,command],input=REMOTE,text=True,capture_output=True,timeout=45)
    if r.returncode:
        print('START_REFUSED',a.host,r.stderr[-700:]);raise SystemExit(r.returncode)
    print(r.stdout.strip())
