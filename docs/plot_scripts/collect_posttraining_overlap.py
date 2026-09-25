#!/usr/bin/env python3
"""Fetch small registered artifacts, never model weights, with remote hashes."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'docs/appendices/posttraining_overlap_20260925'
REMOTE='/data/guoshaoyang/ngram-gap-lab/provisioning/posttraining_overlap_20260925'

def main():
    script='''from pathlib import Path
import hashlib,json
root=Path(REMOTE)
records=[]
for p in sorted((root/'data/runs_fixed').glob('ptov_*_fixed')):
 files={}
 for name in ['launch.json','initial_probe.json','train_log.jsonl','fixed_train_loss.jsonl','table_norm.jsonl','summary.json','status.json']:
  f=p/name
  if f.exists():
   raw=f.read_bytes();files[name]={'text':raw.decode(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 records.append({'run_id':p.name,'files':files,'checkpoint_bytes':(p/'final_model.pt').stat().st_size if (p/'final_model.pt').exists() else None})
print(json.dumps(records))
'''.replace('REMOTE',repr(REMOTE))
    result=subprocess.run(['ssh','a100_perm','/data/guoshaoyang/ngram-gap-lab/.venv/bin/python','-'],input=script,text=True,capture_output=True,check=True)
    runs=json.loads(result.stdout);manifest=[]
    for r in runs:
        target=PACK/'runs'/r['run_id'];target.mkdir(parents=True,exist_ok=True)
        canonical=ROOT/'data/runs_fixed'/r['run_id'];canonical.mkdir(exist_ok=True)
        for name,obj in r['files'].items():
            raw=obj['text'].encode();assert hashlib.sha256(raw).hexdigest()==obj['sha256']
            (target/name).write_bytes(raw);(canonical/name).write_bytes(raw)
            manifest.append({'run_id':r['run_id'],'file':name,'remote':REMOTE+'/data/runs_fixed/'+r['run_id']+'/'+name,'sha256':obj['sha256'],'bytes':len(raw)})
        summary=json.loads(r['files']['summary.json']['text']) if 'summary.json' in r['files'] else None
        train=[json.loads(x) for x in r['files'].get('train_log.jsonl',{}).get('text','').splitlines()]
        print(r['run_id'], 'done' if summary else 'running', 'step',train[-1]['step'] if train else 0,
              'gap',round(train[-1]['gap'],4) if train else None,'checkpoint_bytes',r['checkpoint_bytes'])
    (PACK/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+chr(10))
    (PACK/'remote_checkpoint_sizes.json').write_text(json.dumps({r['run_id']:r['checkpoint_bytes'] for r in runs},indent=2)+chr(10))

if __name__=='__main__':main()
