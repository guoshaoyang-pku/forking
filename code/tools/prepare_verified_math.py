#!/usr/bin/env python3
"""Build an example-disjoint, verifier-selected OpenR1 algebra replay pool.

Selection is independent of model losses. Every problem contributes at most one
complete Math-Verify-correct generation. No internal cycling during packing.
This remains all-token continued LM on static RFT data, not on-policy RLVR.
"""
import argparse
import collections
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REVISION = 'e4e141ec9dea9f8326f4d347be56105859b2bd68'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rlvr-raw', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    with a.tokenizer.open('rb') as f:
        enc = pickle.load(f)
    bos = enc.encode_single_token('<|reserved_0|>')
    assert enc.n_vocab == 8192
    # Fixed token budget matching the preceding 333-update post-training line.
    targets = {'train': 333 * 72, 'val': 4 * 72}
    counts = dict.fromkeys(targets, 0)
    buffers = {k: [] for k in targets}
    files = {k: (a.out / f'shard_{i:05d}.bin').open('wb') for i,k in enumerate(targets)}
    used = {k: [] for k in targets}
    seen_problem, seen_uuid = set(), set()
    scanned = collections.Counter()
    prefixes = collections.Counter()
    token_total = collections.Counter()
    source_hashes = {}
    row_provenance = (a.out / 'examples.jsonl').open('w')
    # Scan the complete pinned default files, then order both splits by a
    # second independent hash. A short validation budget must not select only
    # the first source file while training spans several source blocks.
    candidates_by_problem = {}
    for path in sorted(a.raw_dir.glob('default-*.parquet')):
        source_hashes[str(path)] = sha(path)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for r in batch.to_pylist():
                scanned['source_records'] += 1
                if r['problem_type'] != 'Algebra':
                    continue
                scanned['algebra_records'] += 1
                problem = re.sub(r'\s+', ' ', r['problem']).strip()
                key = hashlib.sha256(problem.encode()).hexdigest()
                uid = r['uuid']
                ok = r.get('correctness_math_verify') or []
                complete = r.get('is_reasoning_complete') or []
                indices = [i for i,g in enumerate(r['generations'])
                           if i < len(ok) and ok[i] is True
                           and i < len(complete) and complete[i] is True
                           and isinstance(g,str) and g.strip()]
                if not indices:
                    scanned['no_complete_math_verified_trace'] += 1
                    continue
                if key in seen_problem or uid in seen_uuid:
                    scanned['duplicate_problem_or_uuid'] += 1
                    continue
                seen_problem.add(key); seen_uuid.add(uid)
                index = indices[0]
                split = 'val' if int(key[:16],16)%10==0 else 'train'
                candidates_by_problem[key] = dict(split=split,uuid=uid,
                    problem_sha256=key,generation_index=index,source_file=path.name,
                    source=r['source'],problem_type=r['problem_type'],math_verify=True,
                    reasoning_complete=True,problem=r['problem'],response=r['generations'][index])
    ordered = sorted(candidates_by_problem.values(),
                     key=lambda r: hashlib.sha256(('order42:'+r['problem_sha256']).encode()).hexdigest())
    try:
        for r in ordered:
            split=r['split']
            if counts[split] == targets[split]:
                continue
            response=r.pop('response');problem=r.pop('problem')
            tokens=[bos]+enc.encode_ordinary(problem+'\n\n'+response)
            assert max(tokens)<8192
            offset=counts[split]*2049+len(buffers[split])
            meta=dict(r,token_offset=offset,record_tokens=len(tokens),response_chars=len(response))
            used[split].append(meta)
            row_provenance.write(json.dumps(meta)+'\n')
            prefixes[' '.join(response.split()[:8])] += 1
            token_total[split] += len(tokens)
            buffers[split].extend(tokens)
            n=min(len(buffers[split])//2049,targets[split]-counts[split])
            if n:
                np.asarray(buffers[split][:n*2049],dtype=np.uint16).tofile(files[split])
                del buffers[split][:n*2049]
                counts[split]+=n
            if all(counts[k]==targets[k] for k in targets):
                break
    finally:
        for f in files.values():f.close()
        row_provenance.close()
    assert counts == targets, (counts, targets)
    assert not ({r['uuid'] for r in used['train']} & {r['uuid'] for r in used['val']})
    assert not ({r['problem_sha256'] for r in used['train']} & {r['problem_sha256'] for r in used['val']})
    # Quantify the old RLVR prompt template, with the SAME tokenizer.
    old = pq.read_table(a.rlvr_raw).to_pylist()
    roles = collections.Counter(m['role'] for r in old for m in r['messages'])
    templates = collections.Counter()
    ntotal = ntemplate = 0
    for r in old:
        text = '\n'.join(m['content'] for m in r['messages'])
        prefix, question = text.rsplit('Question:', 1)
        templates[prefix] += 1
        ntemplate += len(enc.encode_ordinary(prefix))
        ntotal += len([bos] + enc.encode_ordinary(text + '\n\nAnswer:' + r['ground_truth']))
    audit = dict(source='allenai/RLVR-MATH', revision='bd2a93551b503a395fadd1a740d957559cfe6f3c',
                 sha256=sha(a.rlvr_raw), records=len(old), message_roles=dict(roles),
                 unique_fewshot_prefixes=len(templates),
                 common_prefix_characters=len(templates.most_common(1)[0][0]),
                 prefix_token_fraction=ntemplate/ntotal,
                 record_token_total=ntotal,
                 note='user prompts include four fixed worked examples; no sampled assistant rollout; token fraction counts prefix independently')
    manifest = dict(kind='static-verifier-selected-R1-algebra-all-token-LM',
                    dataset='open-r1/OpenR1-Math-220k',revision=REVISION,
                    problem_type='Algebra', selection='first complete Math-Verify-correct generation per unique normalized problem/uuid',
                    split='sha256(whitespace-normalized problem) first64bits modulo10: 0=val, else train; before packing',
                    order='all pinned default shards scanned; independent sha256(order42:problem_hash) ordering before token-budget selection',
                    internal_replay=False, examples={k:len(v) for k,v in used.items()},
                    train_val_problem_overlap=0,train_val_uuid_overlap=0,
                    rows=counts,row_len=2049,sequence_len=2048,
                    target_tokens_per_pass=counts['train']*2048,
                    source_token_totals=dict(token_total),scanned=dict(scanned),
                    tokenizer=str(a.tokenizer),tokenizer_sha256=sha(a.tokenizer),
                    raw_sha256=source_hashes,
                    output_sha256={p.name:sha(p) for p in a.out.glob('*.bin')},
                    split_source_counts={k:dict(collections.Counter(r['source'] for r in v)) for k,v in used.items()},
                    split_file_counts={k:dict(collections.Counter(r['source_file'] for r in v)) for k,v in used.items()},
                    top_response_8word_prefixes=prefixes.most_common(10),
                    old_rlvr_audit=audit,
                    ceiling='static all-token continued LM; not response-only SFT or on-policy RLVR; exact normalized-text disjoint, no semantic-duplicate guarantee')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
