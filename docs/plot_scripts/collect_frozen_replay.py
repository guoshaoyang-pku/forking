#!/usr/bin/env python3
"""Fetch only small hashed logs from the registered frozen-table replay grid."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'docs/appendices/frozen_replay_20260925'
REMOTE='/data/guoshaoyang/ngram-gap-lab/provisioning/frozen_replay_20260925'
NAMES=['launch.json','initial_probe.json','train_log.jsonl','fixed_train_loss.jsonl',
       'table_norm.jsonl','summary.json','status.json','freeze_audit.json']

def main():
    script='''from pathlib import Path
import hashlib,json
root=Path(REMOTE)
records=[];sizes={}
for sub in ['data/runs_fixed','smoke']:
 for p in sorted((root/sub).glob('ptfr_*_fixed')):
  for name in NAMES:
   f=p/name
   if f.exists():
    raw=f.read_bytes();assert len(raw)<2000000
    records.append(dict(path=str(f.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode()))
  sizes[p.name]={name:(p/name).stat().st_size if (p/name).exists() else None for name in ['final_model.pt','probe_tokens.npz']}
print(json.dumps(dict(records=records,sizes=sizes)))
'''.replace('REMOTE',repr(REMOTE)).replace('NAMES',repr(NAMES))
    response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','a100_perm','python3','-'],
                            input=script,text=True,capture_output=True,check=True)
    payload=json.loads(response.stdout);manifest=[]
    for r in payload['records']:
        rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
        raw=r.pop('text').encode();assert hashlib.sha256(raw).hexdigest()==r['sha256']
        path=PACK/'evidence'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        if rel.parts[:2]==('data','runs_fixed') and (ROOT/'data').is_dir():
            canonical=ROOT/rel;canonical.parent.mkdir(parents=True,exist_ok=True);canonical.write_bytes(raw)
        manifest.append(dict(r,remote=REMOTE+'/'+r['path']))
    (PACK/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+chr(10))
    (PACK/'remote_artifact_sizes.json').write_text(json.dumps(payload['sizes'],indent=2)+chr(10))
    for p in sorted((PACK/'evidence/data/runs_fixed').glob('ptfr_*_fixed')):
        log=p/'train_log.jsonl'
        rows=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        status=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else dict(qc='running')
        print(p.name,status['qc'],rows[-1] if rows else 'initializing')

if __name__=='__main__':main()
