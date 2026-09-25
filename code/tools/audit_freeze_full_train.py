#!/usr/bin/env python3
"""Read-only full-pool CE of final freeze-table checkpoints; no optimization."""
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


def evaluate(model, dataset, n_batches, batch_size, device):
    result = []
    with torch.inference_mode():
        for bi in range(n_batches):
            pairs = [dataset.get_chunk(bi * batch_size + j) for j in range(batch_size)]
            x = torch.stack([p[0] for p in pairs]).to(device)
            y = torch.stack([p[1] for p in pairs]).to(device)
            with _autocast_ctx('bf16'):
                loss = model(x, targets=y)
            result.append(float(loss.item()))
            if (bi + 1) % 50 == 0:
                print(f'evaluated {bi + 1}/{n_batches} batches', flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise RuntimeError('Refusing to overwrite an existing diagnostic')
    result = {'metric': 'same-final-checkpoint full 337-batch train pool vs original four fixed val batches',
              'train_code_sha256': hashlib.sha256((Path(__file__).resolve().parents[1]/'train.py').read_bytes()).hexdigest(),
              'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runs': []}
    for rid in args.runs:
        run = args.root / rid
        summary = json.loads((run/'summary.json').read_text())
        cfg = Config(**{k:v for k,v in summary['config'].items() if k in {f.name for f in dataclasses.fields(Config)}})
        assert cfg.data_mode == 'fixed' and cfg.epoch_batches in (0, 337)
        assert cfg.device_batch_size == 72 and cfg.sequence_len == 2048
        assert cfg.dropout == 0 and summary['steps'] == 1000
        assert summary['intervention']['kind'] == 'freeze_table'
        train = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, cfg.sequence_len, cfg.device_batch_size,
                                     cfg.data_seed, epoch_batches=cfg.epoch_batches)
        val = TokenizedShardDataset(cfg.data_dir, cfg.val_shards, cfg.sequence_len, cfg.device_batch_size, cfg.data_seed)
        assert train.total_chunks() == 337 * 72
        model = NanoGPT(cfg)
        state = torch.load(run/'final_model.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.to('cuda:0').eval()
        tr = evaluate(model, train, 337, 72, 'cuda:0')
        va = evaluate(model, val, 4, 72, 'cuda:0')
        item = {'run_id':rid,'seed':summary['seed'],'step':1000,'train_batch_losses':tr,'val_batch_losses':va,
                'train_full_mean':float(np.mean(tr)),'val_mean':float(np.mean(va)),
                'fixed_full_gap':float(np.mean(va)-np.mean(tr)),
                'train_head4_mean':float(np.mean(tr[:4])),
                'train_tail4_mean':float(np.mean(tr[-4:])),
                'fixed_head_gap':float(np.mean(va)-np.mean(tr[:4])),
                'online_gap':summary['final_gap'],
                'val_reproduction_error':float(np.mean(va)-summary['final_val_loss']),
                'checkpoint_bytes':(run/'final_model.pt').stat().st_size}
        result['runs'].append(item)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in item.items() if not isinstance(v,list)}),flush=True)
        del model,train,val
        gc.collect();torch.cuda.empty_cache()
    result['complete'] = True
    args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
