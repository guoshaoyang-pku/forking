#!/usr/bin/env python3
"""Collect small, read-only epoch20 artifacts and running-process metadata.

The registered source is ophis-gpu:data/runs_scaling/*_fixed (log section 55).
No model weights or tokenized shards are transferred.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'docs/appendices/epoch20_20260914/epoch2_snapshot.json'
REMOTE = r'''
import datetime, hashlib, json
from pathlib import Path
root = Path('/data4/guoshaoyang/ngram-gap-lab')
processes = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        argv = (p/'cmdline').read_bytes().decode().split('\0')
        if str(root/'code/train.py') not in argv or '--run_id' not in argv:
            continue
        rid = argv[argv.index('--run_id') + 1]
        if not rid.startswith('s1v5_128_ep20_tri_'):
            continue
        env = (p/'environ').read_bytes().decode().split('\0')
        gpu = next((s.split('=', 1)[1] for s in env if s.startswith('CUDA_VISIBLE_DEVICES=')), None)
        config = {argv[i][2:]: argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}
        processes.append(dict(pid=int(p.name), run_id=rid, gpu=gpu, config=config))
    except (OSError, UnicodeError, ValueError):
        pass
runs = []
for p in sorted((root/'data/runs_scaling').glob('s1v5_128_ep20_tri_*xL4_3ep_v2_fixed')):
    files = {}
    for name in ('summary.json', 'train_log.jsonl'):
        f = p/name
        if f.exists():
            files[name] = dict(text=f.read_text(), mtime=f.stat().st_mtime)
    runs.append(dict(run_id=p.name, path=str(p), files=files))
shards = {}
for p in sorted((root/'data/tokenized_epoch20').glob('shard_*.bin')):
    shards[p.name] = dict(bytes=p.stat().st_size, batches=p.stat().st_size//2//2049//72)
code_sha256 = {str(p): hashlib.sha256((root/p).read_bytes()).hexdigest()
               for p in (Path('code/train.py'), Path('code/ngram_freq.py'))}
print(json.dumps(dict(host='ophis-gpu', root=str(root),
    captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    runs=runs, processes=processes, shards=shards, code_sha256=code_sha256,
    queue_log=(root/'data/runs_scaling/epoch20_queue.nohup.log').read_text())))
'''


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def collect(output):
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
         'ophis-gpu', 'python3', '-'], input=REMOTE, text=True,
        capture_output=True, check=True, timeout=60)
    snapshot = json.loads(result.stdout)
    mirror = Path.home()/'.cache/ngram-gap-lab/epoch20/data/runs_fixed'
    for run in snapshot['runs']:
        dest = mirror/run['run_id']
        dest.mkdir(parents=True, exist_ok=True)
        run['summary'] = None
        run['rows'] = []
        for name, meta in run['files'].items():
            contents = meta.pop('text')
            (dest/name).write_text(contents)
            meta['sha256'] = hashlib.sha256(contents.encode()).hexdigest()
            meta['mirror_path'] = str(dest/name)
            if name == 'summary.json':
                run['summary'] = json.loads(contents)
            else:
                lines = contents.splitlines()
                for i, line in enumerate(lines):
                    try:
                        run['rows'].append(json.loads(line))
                    except json.JSONDecodeError:
                        if i != len(lines)-1 or contents.endswith('\n'):
                            raise
                        run['partial_last_line_omitted'] = True
    snapshot['local_code_sha256'] = {
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in snapshot['code_sha256']}
    snapshot = finite_json(snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, allow_nan=False)+'\n')
    print(f"Collected {len(snapshot['runs'])} runs at {snapshot['captured_at_utc']} → {output}")
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    collect(parser.parse_args().output)
