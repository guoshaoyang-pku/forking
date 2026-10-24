#!/usr/bin/env python3
"""Bounded control for the separately registered 24-arm E3 capacity queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import overnight_status as health
import collect_overnight_20261024 as collector

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/daytime_20261024'
MANIFEST = ROOT / 'docs/plans/daytime-20261024.json'
JOBS = ROOT / 'docs/plans/daytime-20261024.jobs'

START_REMOTE = r'''
import hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
p=json.loads(sys.argv[1]);q=Path(p['queue']);root=Path(p['root']);host=p['host']
assert time.time()<p['cutoff'],'Claim deadline passed; do not extend'
for rel,expected in p['code_hashes'].items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected,'Source changed: '+rel
assert all(shutil.disk_usage(x).free>20*1024**3 for x in (root,q.parent)),'Disk below reserve'
assert not (q/f'STOP_{host}').exists(),'Queue explicitly drained'
current=[];other=[]
for f in Path('/proc').glob('[0-9]*/cmdline'):
    try:argv=[x.decode(errors='replace') for x in f.read_bytes().split(b'\0') if x]
    except OSError:continue
    pos=[i for i,x in enumerate(argv) if Path(x).name=='fleet_queue.py']
    if len(pos)!=1 or len(argv)<=pos[0]+1:continue
    (current if argv[pos[0]+1]==str(q) else other).append(int(f.parent.name))
assert not other,'Another queue dispatcher is present: '+str(other)
q.mkdir(exist_ok=True)
for name,data in [('jobs.txt',p['jobs_text']),('manifest.json',p['manifest_text'])]:
    f=q/name
    try:
        with f.open('x') as stream:stream.write(data)
    except FileExistsError:assert f.read_text()==data,'Queue contents changed: '+name
assert hashlib.sha256((q/'jobs.txt').read_bytes()).hexdigest()==p['jobs_sha256']
if current:
    print(json.dumps({'host':host,'state':'already_running','pids':current}));sys.exit()
pending=[]
for job in p['host_jobs']:
    key=job['claim_key'];claim=q/'claims'/key
    if claim.exists():continue
    assert not (root/'data/runs_fixed'/job['run_id']).exists(),'Unclaimed output already exists: '+job['run_id']
    pending.append(job['run_id'])
if not pending:
    print(json.dumps({'host':host,'state':'no_pending_tasks'}));sys.exit()
if len(pending)==len(p['host_jobs']):
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    assert all(int(line.split(',')[1])==0 and int(line.split(',')[2])==0 for line in smi.splitlines()),'Initial launch requires all assigned GPUs idle'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU processes appeared'
python=str(root/'.venv/bin/python')
args=[python,str(root/'code/cluster/fleet_queue.py'),str(q),str(root),'code/cluster/run_v27.sh','--gpus','0,1,2,3,4,5,6,7','--deadline',str(p['cutoff']),'--python',python]
env={k:v for k,v in os.environ.items() if not k.startswith('NGLAB_')};env['FLEET_HOST']=host
with (q/f'dispatcher_{host}.log').open('a') as log:
    child=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={'host':host,'dispatcher_pid':child.pid,'started_epoch':time.time(),'cutoff':p['cutoff'],'pending':pending,'gpu_allowlist':list(range(8))}
(q/f'dispatcher_{host}.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def configure():
    m = json.loads(MANIFEST.read_text())
    assert set(m['allowed_hosts']) == {'perm', 'a100_t1_2', 'a100_t1_4'}
    assert len(m['jobs']) == len({j['run_id'] for j in m['jobs']}) == 24
    assert all(j['epochs'] == 3 for j in m['jobs'])
    assert hashlib.sha256(JOBS.read_bytes()).hexdigest() == m['jobs_sha256']
    health.QUEUE = collector.QUEUE = m['queue_dir']
    health.CUTOFF = int(datetime.fromisoformat(m['claim_cutoff']).timestamp())
    collector.HOSTS = {h: ('a100_perm' if h == 'perm' else h, r)
                       for h, r in m['host_roots'].items()}
    health.HOSTS = {a: (h, r) for h, (a, r) in collector.HOSTS.items()}
    return m


def status():
    BASE.mkdir(parents=True, exist_ok=True)
    target = BASE / 'status.json'
    try:previous = json.loads(target.read_text())
    except (OSError, ValueError):previous = None
    with ThreadPoolExecutor(max_workers=3) as pool:
        snapshot = {'observed_epoch': time.time(), 'hosts': list(pool.map(health.inspect, health.HOSTS.items()))}
    health.annotate_progress(snapshot, previous)
    if previous is not None:health.save_snapshot(BASE / 'status.previous.json', previous)
    health.save_snapshot(target, snapshot)
    for h in snapshot['hosts']:
        jobs = h.get('jobs', [])
        counts = {s:sum(j['state'] == s for j in jobs) for s in sorted({j['state'] for j in jobs})}
        print(h['host'], counts, 'occupied=', sum(g['memory_mib'] >= 1500 for g in h.get('gpus', [])),
              'progress=', str(h['progressed'])+'/'+str(h['progress_comparable']), 'alerts=', h['alerts'])
    return snapshot


def collect(m, snapshot):
    selected, edges = collector.select_requests(m, snapshot)
    dest = BASE / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest.mkdir(parents=True)
    records = []; errors = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {h:pool.submit(collector.fetch_host, h, [r for (host,_),r in selected.items() if host == h and not r.get('blocked')], dest)
                   for h in collector.HOSTS if any(host == h and not r.get('blocked') for (host,_),r in selected.items())}
        for host,future in futures.items():
            try:records.extend(future.result())
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as e:errors.append({'host':host,'error':str(e)})
    jobs = {(j['host'],j['run_id']):j for j in m['jobs']}
    runs = {(r['host'],r['run_id']):collector.validate_run(r,dest,jobs.get((r['host'],r['run_id']))) for r in records}
    pairs = [collector.compare_runs(runs[a],runs[b]) if a in runs and b in runs else
             {'left':a,'right':b,'state':'manual_review','issues':['evidence_not_collected']} for a,b in edges]
    report = {'observed_epoch':snapshot['observed_epoch'],'errors':errors,
              'runs':[{k:v for k,v in r.items() if not k.startswith('_')} for r in runs.values()], 'comparisons':pairs}
    (dest/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Collected:', dest/'review.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['status','start','collect'])
    ap.add_argument('--host', choices=['a100_perm','a100_t1_2','a100_t1_4'])
    a = ap.parse_args(); m = configure()
    if a.action != 'start':
        snapshot = status()
        if a.action == 'collect':collect(m, snapshot)
        return
    if not a.host:ap.error('start requires --host')
    if time.time() >= health.CUTOFF:ap.error('22:00 CST cutoff passed; no new claims')
    host = 'perm' if a.host == 'a100_perm' else a.host
    payload = {'host':host,'root':m['host_roots'][host],'queue':m['queue_dir'], 'cutoff':health.CUTOFF,
               'code_hashes':m['code_sha256'],'jobs_sha256':m['jobs_sha256'],
               'jobs_text':JOBS.read_text(),'manifest_text':MANIFEST.read_text(),
               'host_jobs':[j for j in m['jobs'] if j['host'] == host]}
    command = shlex.join([payload['root']+'/.venv/bin/python','-B','-',json.dumps(payload)])
    r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,command],
                       input=START_REMOTE,text=True,capture_output=True,timeout=60)
    if r.returncode:raise SystemExit('START_REFUSED '+a.host+' '+r.stderr[-1500:])
    receipt = json.loads(r.stdout)
    BASE.mkdir(parents=True,exist_ok=True)
    (BASE/f'launch_{host}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':main()
