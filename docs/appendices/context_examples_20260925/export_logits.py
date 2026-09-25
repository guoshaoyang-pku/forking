"""Read-only CPU export of full logits for existing sec66 checkpoint examples.

No training, checkpoint changes, or GPU use. Named tokens must be single tokens.
others = arithmetic mean of the 20 largest probabilities after excluding all
individually displayed token IDs, selected separately in each distribution.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--tokens', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.root / 'code/tools'))
    from probe_context_subspace import load

    tok = json.loads(args.tokens.read_text())
    targets = [' dog', ' cat', ' fish', ' horse', ' bird']
    assert all(len(tok['targets'][word]) == 1 for word in targets)
    target_ids = [tok['targets'][word][0] for word in targets]
    chosen = ['petting your', 'introducing your']
    contexts = [next(c for c in tok['dogcat_candidates'] if c['ctx'].strip() == key)
                for key in chosen]
    contexts += [{'ctx': 'there is a', 'ctx_ids': tok['contexts']['there is a']}]
    seqs = torch.tensor([[tok['bos']] + c['ctx_ids'] for c in contexts])
    result = {
        'metric': 'full-vocabulary softmax probability, not renormalized over displayed bars',
        'others_definition': 'mean of top 20 probabilities excluding dog/cat/fish/horse/bird; reselected per distribution',
        'device': 'cpu', 'dtype': 'float32', 'input': 'BOS + exactly three context tokens',
        'training_seed': 42, 'named_tokens': targets, 'named_token_ids': target_ids,
        'contexts': contexts, 'rows': [],
        'source_sha256': {str(p.relative_to(args.root)): sha(p) for p in [
            args.root / 'code/train.py', args.root / 'code/ngram_freq.py',
            args.root / 'code/tools/probe_context_subspace.py',
            args.root / 'code/tools/eval_logits_rank.py', args.tokens]},
        'exporter_sha256': sha(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for arm in ['nogram', 'both']:
        run = args.root / 'data/runs_fixed' / f'ep5ckpt_v5_{arm}_s42_fixed'
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['seed'] == 42 and summary['steps'] == 1685
        for ck, step in [('epoch1_model.pt', 337), ('final_model.pt', 1685)]:
            model, _ = load(str(run), ck, torch.device('cpu'))
            with torch.inference_mode():
                logits = model(seqs)[:, -1].float()
                logp = torch.log_softmax(logits, dim=-1)
                probs = logp.exp()
            ck_hash = sha(run / ck)
            for i, context in enumerate(contexts):
                mask = probs[i].clone()
                mask[target_ids] = -1
                values, ids = mask.topk(20)
                result['rows'].append({
                    'run_id': summary['run_id'], 'arm': arm, 'checkpoint': ck,
                    'checkpoint_sha256': ck_hash, 'summary_sha256': sha(run / 'summary.json'),
                    'step': step, 'pass': step // 337, 'context': context['ctx'].strip(),
                    'ctx_ids': context['ctx_ids'], 'logits': logits[i].tolist(),
                    'named_probability': {word.strip(): float(probs[i, t])
                                          for word, t in zip(targets, target_ids)},
                    'named_logp': {word.strip(): float(logp[i, t])
                                   for word, t in zip(targets, target_ids)},
                    'others_mean_probability': float(values.mean()),
                    'others_top20_ids': ids.tolist(),
                    'others_top20_tokens': [tok['vocab'][t] for t in ids.tolist()],
                    'others_top20_probabilities': values.tolist(),
                })
            del model, logits, logp, probs
            gc.collect()
            args.out.write_text(json.dumps(result, indent=2) + '\n')
            print(f'{arm} {ck}: exported {len(contexts)} full-vocabulary logit vectors', flush=True)


if __name__ == '__main__':
    main()
