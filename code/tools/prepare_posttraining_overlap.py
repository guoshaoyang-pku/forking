#!/usr/bin/env python3
"""Create exact, nonoverlapping 333-update downstream pools from verified bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args()
    specs={'seen':('tokenized/shard_00001.bin','e8dfa753beed211885b60b4a71e79a40c9a7249c465dcf405d4a4aefc9764911'),
           'unseen':('tokenized_epoch20/shard_00021.bin','3df7458d09f7e412a8e42904c1bc5b031048badb3539651d094ee594036cccb6')}
    valpath=a.data_root/'tokenized_epoch20/shard_00022.bin'
    valbytes=valpath.read_bytes();assert sha(valbytes)=='c27c45bfe146a090a78f5f20394c5d3fb61d192ce2fe1fb2f10837048d1caad9'
    val=valbytes[:288*2049*2];valrows=np.frombuffer(val,dtype=np.uint16).reshape(-1,2049)
    valhash={sha(r.tobytes()) for r in valrows}
    root=a.data_root/'posttraining_overlap_20260925';root.mkdir(exist_ok=False)
    result={}
    for pool,(source,h) in specs.items():
        raw=(a.data_root/source).read_bytes();assert sha(raw)==h
        data=raw[:333*72*2049*2];assert len(data)==98253648
        rows=np.frombuffer(data,dtype=np.uint16).reshape(-1,2049)
        rowhash={sha(r.tobytes()) for r in rows}
        assert not rowhash&valhash and rows.max()<8192
        dest=root/pool;dest.mkdir()
        (dest/'shard_00000.bin').write_bytes(data);(dest/'shard_00001.bin').write_bytes(val)
        result[pool]={'source':str(a.data_root/source),'source_sha256':h,
            'train_rows':len(rows),'duplicate_train_rows':len(rows)-len(rowhash),
            'val_rows':len(valrows),'train_val_duplicate_rows':0,'train_sha256':sha(data),'val_sha256':sha(val),
            'target_tokens_per_pass':333*72*2048,'updates_per_pass':333,'passes':5,
            'pretraining_relation':'exact prefix within pretraining shards 0-20' if pool=='seen' else 'same-domain shard 21; no pretraining optimization; prior pretraining validation shard'}
    (root/'manifest.json').write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
