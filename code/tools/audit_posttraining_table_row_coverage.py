#!/usr/bin/env python3
"""Address coverage of the actual clean trigram hash, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists()
    seen=np.zeros(1048576,dtype=bool);remaining=6740*72;count=0
    for sid in range(21):
        buf=np.memmap(a.data_dir/f'shard_{sid:05d}.bin',dtype=np.uint16,mode='r').reshape(-1,2049);take=min(remaining,len(buf))
        for i in range(0,take,512):
            x=np.asarray(buf[i:min(i+512,take),:2048],dtype=np.uint64)
            h=((x[:,:-2]*16777619)^(x[:,1:-1]*2166136261)^(x[:,2:]*3432918353))%1048576
            seen[h.ravel()]=True;count+=len(x)
            if seen.all():break
        remaining-=take
        if seen.all() or not remaining:break
    result={'pretrain_data':str(a.data_dir),'pretrain_shards':list(range(21)),'pretrain_row_budget':6740*72,
            'hash_primes':[16777619,2166136261,3432918353],
            'row_formula':'((prev2*p0) XOR (prev1*p1) XOR current*p2) modulo1048576',
            'positions':'Within-row trigram contexts in input positions2..2047; boundary padding omitted',
            'method':'Scan consecutive pretraining rows in512-row CPU chunks; stop when all rows visited',
            'rows_scanned':count,'unique_table_rows_visited':int(seen.sum()),'table_rows':len(seen),'fraction':float(seen.mean()),
            'ceiling':'Parameter-row address coverage only, not exact context identity, information quality, or proof of nonzero net gradient on every row',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result))


if __name__=='__main__':main()
