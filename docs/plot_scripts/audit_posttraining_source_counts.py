#!/usr/bin/env python3
"""Read-only source-record accounting, validated against corrected token shards."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import pickle
import numpy as np
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    args = p.parse_args()
    root = args.data_root
    with args.tokenizer.open('rb') as f:
        enc = pickle.load(f)
    bos = enc.encode_single_token('<|reserved_0|>')
    train = 333 * 72 * 2049
    total = (333 + 4) * 72 * 2049

    def audit(source):
        if source == 'magicoder':
            files = [root/'ood_raw/magicoder_oss_instruct.jsonl']
            count = sum(1 for line in files[0].open() if line.strip())
            def texts():
                for line in files[0].open():
                    row = json.loads(line)
                    yield str(row.get('problem', '')) + chr(10)*2 + str(row.get('solution', ''))
        else:
            files = sorted((root/'ood_raw/openr1_default10').glob('*.parquet'))
            count = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
            def texts():
                for f in files:
                    for batch in pq.ParquetFile(f).iter_batches(batch_size=128, columns=['problem','solution']):
                        for q, a in zip(batch.column('problem').to_pylist(), batch.column('solution').to_pylist()):
                            yield str(q) + chr(10)*2 + str(a)
        sha_tr = hashlib.sha256(); sha_val = hashlib.sha256()
        offset = 0; ntr = 0; nval = 0; cross = []
        for i, text in enumerate(texts()):
            ids = [bos] + enc.encode_ordinary(text)
            end = offset + len(ids)
            if offset < train and end > 0:
                ntr += 1
            if offset < total and end > train:
                nval += 1
            if offset < train < end:
                cross.append(i)
            a = np.asarray(ids, dtype=np.uint16)
            if offset < train:
                sha_tr.update(a[:min(len(a), train-offset)].tobytes())
            if end > train and offset < total:
                sha_val.update(a[max(0,train-offset):min(len(a),total-offset)].tobytes())
            offset = end
            if offset >= total:
                break
        directory = root / ('ood_tokenized_magicoder_fixtok' if source == 'magicoder' else 'ood_tokenized_openr1_10shards_fixtok')
        actual = [hashlib.sha256((directory/f'shard_{j:05}.bin').read_bytes()).hexdigest() for j in [0,1]]
        generated = [sha_tr.hexdigest(), sha_val.hexdigest()]
        assert generated == actual, source
        return {'source':source, 'raw_files':[str(f) for f in files],
                'raw_available_records':count, 'records_consumed':i+1,
                'train_contributing_records':ntr, 'val_contributing_records':nval,
                'train_val_boundary_spanning_records_0based':cross,
                'train_stored_tokens':train, 'val_stored_tokens':total-train,
                'reencoded_sha256':generated, 'actual_sha256':actual, 'byte_exact_match':True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        result = list(ex.map(audit, ['magicoder','openr1']))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
