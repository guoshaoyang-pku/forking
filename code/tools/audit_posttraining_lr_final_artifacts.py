#!/usr/bin/env python3
"""Validate saved fixed probes and exact frozen table contents, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();torch.set_num_threads(2)
    base=torch.load(a.data_root/'runs_scaling/sftbase_ep20_tri_ckpt_fixed/final_model.pt',map_location='cpu',weights_only=True,mmap=True)
    table_keys=[k for k in base if 'trigram_ves' in k]
    assert len(table_keys)==1
    rows=[];probe_hashes={}
    for source in ['id','magicoder','openr1','rlvr']:
        for arm in ['trainable','frozen','nogram']:
            for lr in ['q4','q10']:
                rid=f'ptlr_{source}_{arm}_{lr}_s42_fixed';run=a.runs/rid
                s=json.loads((run/'summary.json').read_text());status=json.loads((run/'status.json').read_text())
                assert status['qc']=='pass' and s['steps']==1665
                with np.load(run/'probe_tokens.npz') as z:
                    chunk=z['fixed_train_probe_chunk'];assert chunk.shape==(4,72,2049)
                    sha=hashlib.sha256(np.ascontiguousarray(chunk[:,:,:-1]).tobytes()).hexdigest()[:16]
                    assert sha==str(z['fixed_train_probe_sha256'])==s['fixed_train_probe_sha256']
                    probe_hashes.setdefault(source,set()).add(sha)
                row={'run_id':rid,'probe_sha256':sha,'probe_shape':list(chunk.shape),'checkpoint_bytes':(run/'final_model.pt').stat().st_size}
                if arm=='frozen':
                    state=torch.load(run/'final_model.pt',map_location='cpu',weights_only=True,mmap=True)
                    assert set(state)==set(base)
                    assert all(torch.equal(state[k],base[k]) for k in table_keys),rid
                    changed=any(not torch.equal(state[k],base[k]) for k in base if k not in table_keys)
                    assert changed
                    row.update(table_bitwise_equal_to_initial=True,backbone_changed=True,table_keys=table_keys)
                    del state
                rows.append(row);print(rid,'PASS',flush=True)
    assert all(len(x)==1 for x in probe_hashes.values())
    result={'complete':True,'runs':rows,'source_probe_hashes':{k:list(v)[0] for k,v in probe_hashes.items()},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+chr(10))


if __name__=='__main__':main()
