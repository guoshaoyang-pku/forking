#!/usr/bin/env python3
"""Read-only checkpoint audit of online versus sampled fixed-train loss.

No optimization or checkpoint mutation. Runs on the host holding the checkpoints.
"""
import argparse
import dataclasses
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import Config, NanoGPT, TokenizedShardDataset, _autocast_ctx


def evaluate(model, dataset, batch_ids, device):
    result = []
    with torch.inference_mode():
        for bi in batch_ids:
            pairs = [dataset.get_chunk(int(bi) * 72 + j) for j in range(72)]
            x = torch.stack([p[0] for p in pairs]).to(device)
            y = torch.stack([p[1] for p in pairs]).to(device)
            with _autocast_ctx('bf16'):
                loss = model(x, targets=y)
            result.append({'batch_index_0based': int(bi), 'loss': float(loss.item())})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('Refusing to overwrite existing diagnostic output')
    specs = []
    for source in ['id', 'magicoder', 'openr1_10shards', 'rlvr_math']:
        for arm in ['trainable', 'frozen', 'nogram']:
            if source == 'id':
                parent = 'runs_sft' if arm == 'nogram' else 'runs_sft_freq10'
                rid = 'sft_id_e5_nogram_s42' if arm == 'nogram' else f'sft_id_tri_e5_freq10_{arm}_s42'
            else:
                parent = 'runs_sft_fixtok'
                rid = f'sft_{source}_fixtok_e5_{arm}_s42'
            specs.append((source, arm, args.data_root / parent / rid))
    specs.append(('rlvr_math', 'nogram_s43', args.data_root / 'runs_sft_fixtok/sft_rlvr_math_fixtok_e5_nogram_s43'))
    ids = sorted(set(np.linspace(0, 332, 33).round().astype(int).tolist()))
    result = {'protocol': 'Same-checkpoint BF16 CE. 33 deterministic equally spaced training batches spanning the entire 333-batch pool; head and tail separately. Four original fixed-validation batches. Sampled diagnostic, not a replacement for online gap.',
              'train_batch_indices': ids, 'code_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'train.py').read_bytes()).hexdigest(),
              'torch_version': torch.__version__, 'runs': []}
    device = torch.device('cuda:0')
    for source, arm, run in specs:
        summary = json.loads((run / 'summary.json').read_text())
        fields = {f.name for f in dataclasses.fields(Config)}
        cfg = Config(**{k:v for k,v in summary['config'].items() if k in fields})
        train = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, 2048, 72, seed=42, epoch_batches=333)
        val = TokenizedShardDataset(cfg.data_dir, cfg.val_shards, 2048, 72, seed=42)
        model = NanoGPT(cfg)
        ckpt = run / 'final_model.pt'
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.to(device).eval()
        train_rows = evaluate(model, train, ids, device)
        val_rows = evaluate(model, val, range(4), device)
        tm = float(np.mean([r['loss'] for r in train_rows]))
        vm = float(np.mean([r['loss'] for r in val_rows]))
        row = {'source':source,'arm':arm,'run_id':run.name,'seed':summary['seed'],'checkpoint_step':summary['steps'],
               'checkpoint_bytes':ckpt.stat().st_size,'train_batches':train_rows,'val_batches':val_rows,
               'sampled_train_mean':tm,'fixed_val_mean':vm,'sampled_fixed_gap':vm-tm,
               'logged_online_train':summary['final_train_loss'],'logged_fixed_val':summary['final_val_loss'],
               'logged_online_gap':summary['final_gap'],'val_reproduction_error':vm-summary['final_val_loss']}
        if arm == 'nogram':
            base = torch.load(summary['init_model'], map_location='cpu', weights_only=True)
            model.load_state_dict(base, strict=True); del base
            bt = evaluate(model, train, ids, device); bv = evaluate(model, val, range(4), device)
            row['base_train_batches'] = bt; row['base_val_batches'] = bv
            row['base_sampled_gap'] = float(np.mean([r['loss'] for r in bv])-np.mean([r['loss'] for r in bt]))
        result['runs'].append(row)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:row[k] for k in ['source','arm','sampled_train_mean','fixed_val_mean','sampled_fixed_gap','logged_online_gap','val_reproduction_error']}),flush=True)
        del model,train,val
        gc.collect();torch.cuda.empty_cache()
    result['complete'] = True
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
