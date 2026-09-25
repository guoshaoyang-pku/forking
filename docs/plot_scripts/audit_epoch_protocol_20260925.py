#!/usr/bin/env python3
"""Read-only remote audit of pool length versus replay count; no training."""
import concurrent.futures
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/appendices/epoch_protocol_audit_20260925'

REMOTE = r'''
from pathlib import Path
import datetime, hashlib, json, math, sys
host = sys.argv[1]
root = Path('/data4/guoshaoyang/ngram-gap-lab' if host == 'ophis-gpu' else '/data/home/guoshaoyang/ngram-gap-lab')
def sha(b): return hashlib.sha256(b).hexdigest()
def finite(x):
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict): return {k:finite(v) for k,v in x.items()}
    if isinstance(x, list): return [finite(v) for v in x]
    return x
patterns = ['s1v5_128_ep*', 'sftbase_ep20_tri_ckpt_fixed', 'ls20ep*', 'fig1v5_spc*']
runs=[]
for sub in ['runs_scaling','runs_fixed']:
    dirs=set()
    for pat in patterns: dirs.update((root/'data'/sub).glob(pat))
    for p in sorted(dirs):
        f=p/'summary.json'
        if not f.exists():
            runs.append(dict(run_id=p.name,path=str(p),status='no_summary'))
            continue
        s=json.loads(f.read_text()); c=s['config']
        log=p/'train_log.jsonl'
        rows=[json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        shards={}
        for sid in sorted(set(c['train_shards']+c['val_shards'])):
            q=Path(c['data_dir'])/('shard_%05d.bin'%sid)
            if not q.exists(): raise FileNotFoundError(q)
            size=q.stat().st_size
            shards[str(sid)]={'bytes':size,'chunks':size//2//(c['sequence_len']+1)}
        capacity=sum(shards[str(s)]['chunks'] for s in c['train_shards'])//c['device_batch_size']
        ep=c.get('epoch_batches',0) or capacity
        boundaries=[x for x in rows if x['step']%ep==0]
        runs.append(dict(run_id=p.name,path=str(p),status='summary_present',summary=finite(s),
            summary_sha256=sha(f.read_bytes()),log_sha256=sha(log.read_bytes()),
            logged_steps=[x['step'] for x in rows],boundary_rows=finite(boundaries),
            max_gap_identity_error=max(abs(x['gap']-(x['val_loss']-x['train_loss'])) for x in rows),
            shards=shards,pool_capacity_batches=capacity,declared_epoch_batches=ep,
            total_steps=s['steps'],declared_passes=s['steps']/ep,
            actual_prefix_batches=min(ep,capacity),
            pool_smaller_than_declared_epoch=capacity<ep,
            checkpoints=[x.name for x in p.glob('*model.pt')]))
result=dict(host=host,root=str(root),captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=runs)
if host=='ophis-gpu':
    data=root/'data/tokenized_epoch20'
    hashes={}; metadata={}; locations={}
    for sid in list(range(24))+[6542]:
        p=data/('shard_%05d.bin'%sid); buf=p.read_bytes(); chunk_bytes=2049*2
        assert len(buf)%chunk_bytes==0
        rows=[sha(buf[i:i+chunk_bytes]) for i in range(0,len(buf),chunk_bytes)]
        hashes[sid]=rows
        metadata[str(sid)]={'bytes':len(buf),'chunks':len(rows),'sha256':sha(buf)}
        for i,h in enumerate(rows): locations.setdefault(h,[]).append([sid,i])
    checks={}
    for name,train,val,budget in [
        ('epoch20_actual_split',list(range(21)),[21,22,23],6740),
        ('clean_singlepass',[1,11,12,13,14,15,16],list(range(2,11))+[6542],2000)]:
        pool=[h for sid in train for h in hashes[sid]]
        used=pool[:budget*72]; v=[h for sid in val for h in hashes[sid]]
        overlap=set(used)&set(v)
        counts={}
        for h in used:counts[h]=counts.get(h,0)+1
        checks[name]=dict(train_shards=train,val_shards=val,used_batches=budget,
            available_batches=len(pool)//72,used_chunks=len(used),
            within_used_duplicate_excess=len(used)-len(set(used)),
            used_train_val_overlap_distinct_chunks=len(overlap),
            duplicate_locations=[locations[h] for h,n in counts.items() if n>1],
            overlap_locations=[locations[h] for h in sorted(overlap)])
    result['fresh_chunk_audit']=dict(method='SHA256 of every full packed 2049-token row, including targets; exact duplicates only, no near-duplicate/document audit',shards=metadata,checks=checks)
    result['legacy_audit_json']=json.loads((data/'audit.json').read_text())
print(json.dumps(finite(result),allow_nan=False))
'''


def collect(host):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
                      'python3','-',host],input=REMOTE,text=True,capture_output=True,timeout=90)
    if p.returncode:
        return {'host':host,'error':p.stderr.strip(),'exit_code':p.returncode}
    return json.loads(p.stdout)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(collect,['ophis-gpu','360-1','360-2']))
    for result in results:
        (OUT/(result['host']+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(result['host'],len(result.get('runs',[])),result.get('error','collected'))
    historical=json.loads((ROOT/'docs/appendices/epoch20_20260914/epoch2_snapshot.json').read_text())
    ophis=results[0]
    live={x['run_id']:x for x in ophis['runs']}
    checks=[]
    for old in historical['runs']:
        x=live[old['run_id']];s=x['summary'];c=s['config'];ep=x['declared_epoch_batches']
        expected=set(range(10,3*ep+1,10))|{ep,2*ep,3*ep}
        check=dict(run_id=x['run_id'],epoch_batches=ep,steps=x['total_steps'],
            pool_batches=x['pool_capacity_batches'],passes=x['declared_passes'],
            summary_hash_unchanged=x['summary_sha256']==old['files']['summary.json']['sha256'],
            log_hash_unchanged=x['log_sha256']==old['files']['train_log.jsonl']['sha256'],
            exact_logging_grid=expected==set(x['logged_steps']),
            complete_three_passes=x['total_steps']==3*ep,
            no_within_epoch_capacity_wrap=x['pool_capacity_batches']>=ep,
            correct_split=c['train_shards']==list(range(21)) and c['val_shards']==[21,22,23],
            seed42=s['seed']==42,gap_identity=x['max_gap_identity_error']<1e-7,
            boundary_gaps=[r['gap'] for r in x['boundary_rows']])
        checks.append(check)
    assert len(checks)==20
    fields=['summary_hash_unchanged','log_hash_unchanged','exact_logging_grid','complete_three_passes','no_within_epoch_capacity_wrap','correct_split','seed42','gap_identity']
    assert all(all(x[k] for k in fields) for x in checks),checks
    (OUT/'epoch20_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    with (OUT/'run_inventory.csv').open('w') as f:
        keys=['host','run_id','status','total_steps','pool_capacity_batches','declared_epoch_batches','declared_passes','pool_smaller_than_declared_epoch']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for result in results:
            for run in result.get('runs',[]):
                writer.writerow({k:result['host'] if k=='host' else run.get(k) for k in keys})
    print('All 20 historical summaries/logs unchanged; grid, split, capacity, 3-pass completion and metric checks pass.')
    print(json.dumps(ophis['fresh_chunk_audit']['checks'],indent=2))


if __name__=='__main__':main()
