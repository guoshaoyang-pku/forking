#!/usr/bin/env python3
"""Evaluate original-LR final/base checkpoints on the new uniform4 probe."""
import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import Config,NanoGPT,TokenizedShardDataset,_autocast_ctx


def loss(model,batches):
    out=[]
    with torch.inference_mode():
        for x,y in batches:
            with _autocast_ctx('bf16'):v=model(x.to('cuda'),targets=y.to('cuda'))
            out.append(float(v.item()))
    return float(np.mean(out))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise RuntimeError('Refusing overwrite')
    result={'protocol':'uniform288 chunks rebatch into4 batches; identical to train.py uniform4','runs':[]}
    for source in ['id','magicoder','openr1_10shards','rlvr_math']:
        for arm in ['trainable','frozen','nogram']:
            if source=='id':
                parent='runs_sft' if arm=='nogram' else 'runs_sft_freq10'
                rid='sft_id_e5_nogram_s42' if arm=='nogram' else f'sft_id_tri_e5_freq10_{arm}_s42'
            else:parent='runs_sft_fixtok';rid=f'sft_{source}_fixtok_e5_{arm}_s42'
            run=a.root/parent/rid;s=json.loads((run/'summary.json').read_text())
            cfg=Config(**{k:v for k,v in s['config'].items() if k in {f.name for f in dataclasses.fields(Config)}})
            tr=TokenizedShardDataset(cfg.data_dir,cfg.train_shards,2048,72,42,epoch_batches=333)
            va=TokenizedShardDataset(cfg.data_dir,cfg.val_shards,2048,72,42)
            assert tr.total_chunks()==333*72
            indices=[round(i*(tr.total_chunks()-1)/287) for i in range(288)]
            pairs=[tr.get_chunk(i) for i in indices]
            train=[(torch.stack([x for x,y in pairs[i:i+72]]),torch.stack([y for x,y in pairs[i:i+72]])) for i in range(0,288,72)]
            vp=[va.get_chunk(i) for i in range(288)]
            val=[(torch.stack([x for x,y in vp[i:i+72]]),torch.stack([y for x,y in vp[i:i+72]])) for i in range(0,288,72)]
            probe_hash=hashlib.sha256(b''.join(x.numpy().tobytes() for x,y in train)).hexdigest()[:16]
            model=NanoGPT(cfg);state=torch.load(run/'final_model.pt',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);del state;model.to('cuda').eval()
            ft=loss(model,train);fv=loss(model,val)
            row={'source':source,'arm':arm,'run_id':rid,'seed':42,'step':1665,'probe_sha256':probe_hash,'fixed_train':ft,'fixed_val':fv,'fixed_gap':fv-ft,'online_gap':s['final_gap'],'val_reproduction_error':fv-s['final_val_loss']}
            state=torch.load(s['init_model'],map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);del state
            bt=loss(model,train);bv=loss(model,val);row.update({'base_fixed_train':bt,'base_fixed_val':bv,'base_fixed_gap':bv-bt})
            result['runs'].append(row);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(row),flush=True)
            del model,tr,va,train,val;gc.collect();torch.cuda.empty_cache()
    result['complete']=True;a.out.write_text(json.dumps(result,indent=2)+chr(10))


if __name__=='__main__':main()
