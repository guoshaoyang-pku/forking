#!/usr/bin/env python3
"""Nested exact pools for a registered frozen-table vs no-gram replay test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    a=p.parse_args()
    source=a.data_root/'posttraining_overlap_20260925/unseen'
    train=(source/'shard_00000.bin').read_bytes()
    val=(source/'shard_00001.bin').read_bytes()
    assert sha(train)=='3df7458d09f7e412a8e42904c1bc5b031048badb3539651d094ee594036cccb6'
    assert sha(val)=='53f483abca9bd5d805a16c183550a1bdc5c4db9af5c38f34d9d3818734a610dd'
    root=a.data_root/'frozen_replay_20260925'
    root.mkdir(exist_ok=False)
    valhash={sha(r.tobytes()) for r in np.frombuffer(val,dtype=np.uint16).reshape(-1,2049)}
    manifests={}
    for updates in [42,84]:
        raw=train[:updates*72*2049*2]
        rows=np.frombuffer(raw,dtype=np.uint16).reshape(-1,2049)
        hashes={sha(r.tobytes()) for r in rows}
        assert len(hashes)==len(rows) and not hashes&valhash and rows.max()<8192
        dest=root/f'u{updates}';dest.mkdir()
        (dest/'shard_00000.bin').write_bytes(raw)
        (dest/'shard_00001.bin').write_bytes(val)
        manifests[str(updates)]=dict(train_rows=len(rows),val_rows=288,
            train_sha256=sha(raw),val_sha256=sha(val),source_train_sha256=sha(train),
            source=str(source),updates_per_pass=updates,target_tokens_per_pass=updates*147456,
            optimizer_updates=1680,replay_passes=1680//updates,
            train_val_duplicate_rows=0,duplicate_train_rows=0,
            relation='Nested prefix of same-domain pretraining-validation shard21; never used for base optimization')
    (root/'manifest.json').write_text(json.dumps(manifests,indent=2)+chr(10))
    print(json.dumps(manifests,indent=2))

if __name__=='__main__':main()
