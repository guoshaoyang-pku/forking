#!/usr/bin/env python3
"""Recover immutable completed LR-sweep logs with SHA256 provenance."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / 'docs/appendices/posttraining_lr_sweep_20260925'
REMOTE = '/data/shared/guoshaoyang/ngram-gap-lab/posttraining_lr_20260925/data/runs_fixed'
SCRIPT = '''import hashlib,json
from pathlib import Path
r=Path(REMOTE_ROOT)
cache=CACHE
rows=[]
for p in sorted(r.glob('ptlr_*_s42_fixed')):
 if not (p/'status.json').exists():continue
 files={}
 for name in ['summary.json','train_log.jsonl','fixed_train_loss.jsonl','table_norm.jsonl','status.json','launch.json','train.log','final_model.pt','probe_tokens.npz']:
  f=p/name
  if f.exists():
   previous=cache.get(p.name,{}).get(name,{})
   if previous.get('bytes')==f.stat().st_size and previous.get('mtime_ns')==f.stat().st_mtime_ns:
    files[name]=previous
    continue
   h=hashlib.sha256()
   with f.open('rb') as src:
    for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b)
   files[name]={'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,'sha256':h.hexdigest()}
 rows.append({'run_id':p.name,'host':'360-2','path':str(p),'files':files})
print(json.dumps(rows))
'''.replace('REMOTE_ROOT', repr(REMOTE))


def main():
    previous=PACKET/'artifact_manifest.json'
    cache={r['run_id']:r['files'] for r in json.loads(previous.read_text())} if previous.exists() else {}
    rows=json.loads(subprocess.check_output(['ssh','360-2','python3 -'],input=SCRIPT.replace('CACHE',repr(cache)),text=True))
    wanted=[]
    for row in rows:
        for name in row['files']:
            if name not in ['final_model.pt','probe_tokens.npz']:
                wanted.append(row['run_id']+'/'+name)
    if not wanted:
        print('No completed runs yet');return
    cmd='tar -cf - -C '+shlex.quote(REMOTE)+' -T -'
    blob=subprocess.check_output(['ssh','360-2',cmd],input=(chr(10).join(wanted)+chr(10)).encode())
    raw=PACKET/'raw';raw.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for member in tar:
            assert member.name in wanted and member.isfile()
            data=tar.extractfile(member).read()
            rid,name=member.name.split('/')
            meta=next(x for x in rows if x['run_id']==rid)['files'][name]
            assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
            dst=raw/member.name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(data)
    (PACKET/'artifact_manifest.json').write_text(json.dumps(rows,indent=2)+chr(10))
    print('Recovered',len(rows),'completed attempts;',len(wanted),'small artifacts verified')


if __name__=='__main__':main()
