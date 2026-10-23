#!/usr/bin/env python3
"""Read-only, bounded health check for the registered 2026-10-24 A100 queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
QUEUE = '/home/cfs/guoshaoyang/nglab_queue_overnight_20261024'
CUTOFF = 1792800000
HOSTS = {
    'a100_perm': ('perm', '/data2/guoshaoyang/ngram-gap-lab-v27'),
    'a100_t1_2': ('a100_t1_2', '/home/tione/guoshaoyang/ngram-gap-lab'),
    'a100_t1_4': ('a100_t1_4', '/home/tione/guoshaoyang/ngram-gap-lab'),
    'a100_t1': ('a100_t1', '/home/tione/guoshaoyang/ngram-gap-lab'),
}
REMOTE = r'''
import hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
host,root,q,cutoff = sys.argv[1:]
cutoff=float(cutoff)
q=Path(q); root=Path(root)
result={'host':host,'jobs':[],'agents':[],'alerts':[]}
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try: argv=[x.decode(errors='replace') for x in p.read_bytes().split(b'\0') if x]
    except OSError: continue
    positions=[i for i,x in enumerate(argv) if Path(x).name=='fleet_queue.py']
    if len(positions)==1 and len(argv)>positions[0]+1 and argv[positions[0]+1]==str(q):
        result['agents'].append(int(p.parent.name))
try:
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=12)
    result['gpus']=[dict(zip(['index','memory_mib','util'],map(int,x.split(',')))) for x in smi.strip().splitlines()]
except Exception as e: result['alerts'].append('gpu_query_failed:'+type(e).__name__)
result['disk_free_gib']=round(__import__('shutil').disk_usage(root).free/1024**3,1)
if result['disk_free_gib']<20: result['alerts'].append('disk_below_20GiB')
if not (q/'jobs.txt').exists():
    result['alerts'].append('queue_not_deployed');print(json.dumps(result));sys.exit()
for line in (q/'jobs.txt').read_text().splitlines():
    line=line.strip()
    if not line or line.startswith('#'):continue
    envs,args=line.split('|',1)
    env=dict(x.split('=',1) for x in envs.split())
    if env.get('FLEET_TARGET')!=host:continue
    rid=env['FLEET_RUN_ID'];key=hashlib.sha1(line.encode()).hexdigest()[:12]
    claim=q/'claims'/key; d=root/'data/runs_fixed'/rid
    j={'run_id':rid,'key':key,'state':'queued'}
    if claim.exists():
        meta=json.loads((claim/'claim.json').read_text()) if (claim/'claim.json').exists() else {}
        j['gpu']=meta.get('gpu')
        rc=(claim/'rc').read_text().strip() if (claim/'rc').exists() else None
        j['rc']=rc
        if rc is not None:j['state']='finished_unreviewed' if rc=='0' else 'failed'
        else:
            pid=meta.get('pid') or meta.get('wrapper_pid')
            if not pid and (claim/'pid').exists():
                try:pid=int((claim/'pid').read_text().strip())
                except ValueError:pass
            live=False
            if pid:
                try:live=str(claim) in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                except OSError:pass
            j['wrapper_live']=live
            age=time.time()-claim.stat().st_mtime
            j['state']='active' if live else 'starting' if age<600 else 'needs_process_check'
        log=d/'train_log.jsonl'
        if log.exists():
            rows=[]
            for line in log.read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if 'step' in r:rows.append(r)
            if rows:
                j['step']=rows[-1]['step'];j['last_train']=rows[-1].get('train_loss');j['last_val']=rows[-1].get('val_loss')
                j['log_age_s']=int(time.time()-log.stat().st_mtime)
                if any(not math.isfinite(v) for r in rows[-60:] for k,v in r.items() if k in ('train_loss','val_loss') and isinstance(v,(float,int))):
                    result['alerts'].append('nonfinite:'+rid)
                if j['state']=='active' and j['log_age_s']>900:result['alerts'].append('no_progress_15min:'+rid)
        summary=d/'summary.json'
        if summary.exists():
            s=json.loads(summary.read_text())
            j['summary_steps']=s.get('steps');j['target_steps']=s.get('config',{}).get('max_steps')
            j['completion_valid']=(s.get('run_id')==rid and j.get('step')==s.get('steps')==j['target_steps'] and all(isinstance(s.get(k),(int,float)) and math.isfinite(s[k]) for k in ('final_train_loss','final_val_loss')))
            if rc=='0' and not j['completion_valid']:result['alerts'].append('completion_mismatch:'+rid)
        elif rc=='0':result['alerts'].append('missing_summary:'+rid)
    result['jobs'].append(j)
if time.time()<cutoff and any(j['state']=='queued' for j in result['jobs']):
    if not result['agents']:result['alerts'].append('pending_without_agent')
    free=[g['index'] for g in result.get('gpus',[]) if g['memory_mib']<1500]
    if free:result['free_with_pending']=free
print(json.dumps(result))
'''

def inspect(item):
    alias,(name,root)=item
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',alias,
         f'{root}/.venv/bin/python - {name} {root} {QUEUE} {CUTOFF}']
    try:
        r=subprocess.run(cmd,input=REMOTE,text=True,capture_output=True,timeout=45)
        if r.returncode:return {'host':alias,'alerts':['ssh_or_probe_failed'],'rc':r.returncode}
        out=json.loads(r.stdout);out['alias']=alias;return out
    except (subprocess.TimeoutExpired,json.JSONDecodeError):return {'host':alias,'alerts':['timeout_or_invalid_probe']}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json',action='store_true');args=ap.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(inspect,HOSTS.items()))
    snapshot={'observed_epoch':time.time(),'hosts':results}
    target=ROOT/'outputs/takeover_20261024/overnight_status.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    if args.json:print(json.dumps(snapshot,ensure_ascii=False,indent=2))
    else:
        for h in results:
            counts={s:sum(j['state']==s for j in h.get('jobs',[])) for s in sorted({j['state'] for j in h.get('jobs',[])})}
            occupied=sum(g['memory_mib']>=1500 for g in h.get('gpus',[]))
            print(h.get('alias',h['host']),f'occupied={occupied}/{len(h.get("gpus",[]))}',counts,'alerts='+str(h['alerts']),'free_pending='+str(h.get('free_with_pending',[])))
        print('Saved:',target)
