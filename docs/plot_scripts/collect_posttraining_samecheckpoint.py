#!/usr/bin/env python3
"""Archive registered small logs and QC evidence; never transfer model weights."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / 'docs/appendices/posttraining_samecheckpoint_20260925'
REMOTE = '/data/guoshaoyang/ngram-gap-lab/provisioning/posttraining_samecheckpoint_20260925'
NAMES = ['launch.json', 'initial_probe.json', 'train_log.jsonl',
         'fixed_train_loss.jsonl', 'table_norm.jsonl', 'summary.json', 'status.json']


def main():
    script = '''from pathlib import Path
import hashlib,json
root=Path(REMOTE)
records=[]
def add(f):
 raw=f.read_bytes()
 assert len(raw)<2000000
 records.append(dict(path=str(f.relative_to(root)),text=raw.decode(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
sizes={}
for sub in ['data/runs_fixed','smoke']:
 for p in sorted((root/sub).glob('ptsc_*_fixed')):
  for name in NAMES:
   if (p/name).exists():add(p/name)
  sizes[p.name]={name:(p/name).stat().st_size if (p/name).exists() else None for name in ['final_model.pt','probe_tokens.npz']}
add(root/'equivalence.json')
references=[]
refroot=root.parent/'posttraining_overlap_20260925/data/runs_fixed'
for p in sorted(refroot.glob('ptov_*_fixed')):
 for name in NAMES:
  f=p/name
  references.append(dict(run_id=p.name,file=name,remote=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
print(json.dumps(dict(records=records,sizes=sizes,references=references)))
'''.replace('REMOTE', repr(REMOTE)).replace('NAMES', repr(NAMES))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'a100_perm', '/data/guoshaoyang/ngram-gap-lab/.venv/bin/python', '-'],
                            input=script, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    canonical_available = (ROOT/'data').is_dir()
    if not canonical_available:
        print('Local data target unavailable; archiving verified remote runs_fixed logs in packet only.')
    refs = {(r['run_id'],r['file']):r for r in payload['references']}
    for r in json.loads((PACK/'reference_manifest.json').read_text()):
        assert refs[r['run_id'],r['file']]['sha256'] == r['sha256']
    manifest = []
    for item in payload['records']:
        rel = Path(item['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        raw = item.pop('text').encode()
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        target = PACK / 'evidence' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        if canonical_available and rel.parts[:2] == ('data', 'runs_fixed'):
            canonical = ROOT / rel
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_bytes(raw)
        manifest.append(dict(item, remote=REMOTE + '/' + item['path']))
    (PACK / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + chr(10))
    (PACK / 'remote_artifact_sizes.json').write_text(json.dumps(payload['sizes'], indent=2) + chr(10))
    (PACK / 'reference_remote_verification.json').write_text(json.dumps(payload['references'],indent=2)+chr(10))
    (PACK / 'local_archive_status.json').write_text(json.dumps(dict(canonical_available=canonical_available,
        data_target=str((ROOT/'data').readlink()) if (ROOT/'data').is_symlink() else str(ROOT/'data'),
        packet_is_verified_remote_copy=True),indent=2)+chr(10))
    for p in sorted((PACK / 'evidence/data/runs_fixed').glob('ptsc_*_fixed')):
        rows = [json.loads(line) for line in (p/'train_log.jsonl').read_text().splitlines()]
        status = json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {'qc': 'running'}
        print(p.name, 'step', rows[-1]['step'], status)


if __name__ == '__main__':
    main()
