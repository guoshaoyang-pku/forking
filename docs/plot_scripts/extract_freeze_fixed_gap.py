#!/usr/bin/env python3
"""Read-only reduction of archived exact-frequency diagnostics to scalar CE.

Run on the source host. The train diagnostic is the fixed HEAD of the stream,
not a uniform sample. Both branch partitions must reconstruct the same loss.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--runs', nargs='+', required=True)
    args = p.parse_args()
    result = []
    for rid in args.runs:
        d = args.root / rid
        if not (d / 'summary.json').exists():
            result.append({'run_id': rid, 'missing': str(d)})
            continue
        summary = json.loads((d / 'summary.json').read_text())
        online = {x['step']: x for x in map(json.loads, (d / 'train_log.jsonl').read_text().splitlines())}
        item = {'run_id': rid, 'source_path': str(d), 'summary': summary,
                'sha256': {f: sha(d/f) for f in ['summary.json', 'train_log.jsonl', 'exact_freq_loss.jsonl']},
                'available_checkpoints': [str(x.relative_to(d)) for x in d.rglob('*.pt')], 'points': []}
        count_reference = {}
        branch_disagreement = 0.0
        val_disagreement = 0.0
        with (d / 'exact_freq_loss.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                point = {'step': row['step'], 'epoch': row['epoch']}
                for split in ['train', 'val']:
                    losses = []
                    for branch in ['bigram', 'trigram']:
                        bins = row[split][branch]
                        counts = {k: b['token_count'] for k, b in bins.items()}
                        key = (split, branch)
                        if key not in count_reference:
                            count_reference[key] = counts
                        assert counts == count_reference[key], (rid, row['step'], key)
                        n = sum(counts.values())
                        assert n == 4 * 72 * 2048, (rid, n)
                        losses.append(sum(b['loss_sum'] for b in bins.values()) / n)
                    branch_disagreement = max(branch_disagreement, abs(losses[0]-losses[1]))
                    point[split + '_fixed_head' if split == 'train' else 'val_fixed_diag'] = losses[0]
                point['fixed_head_gap'] = point['val_fixed_diag'] - point['train_fixed_head']
                point.update({k: online[row['step']][k] for k in ['train_loss', 'val_loss', 'gap']})
                val_disagreement = max(val_disagreement, abs(point['val_fixed_diag']-point['val_loss']))
                item['points'].append(point)
        assert branch_disagreement < 1e-10
        item['checks'] = {'constant_frequency_counts': True, 'tokens_each_split': 589824,
                          'max_branch_loss_disagreement': branch_disagreement,
                          'max_diag_vs_logged_val_difference': val_disagreement}
        result.append(item)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
