"""ngram-gap-lab · code/prepare_data.py

Pre-tokenize parquet shards into packed .bin files using the same packing
logic as the configured upstream lib.py (BOS-aligned, best-fit packing).

Output: data/tokenized/shard_<id>.bin  (uint16 token stream, packed rows of
        sequence_len+1 each, BOS at start of each row)

Usage:
  python code/prepare_data.py \
    --lib_dir /path/to/upstream \
    --cache_dir /path/to/cache \
    --out_dir data/tokenized \
    --split_path /path/to/upstream/data_split.json \
    --shards 1,2,3,4,5,6,7,8,9,10,6542 \
    --seq_len 2048
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib_dir", required=True,
                        help="Directory containing the compatible lib.py")
    parser.add_argument("--cache_dir", required=True,
                        help="AUTORESEARCH_CACHE_DIR (parquet + tokenizer)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--split_path",
        default=None,
        help="Path to data_split.json; defaults to <lib_dir>/data_split.json",
    )
    parser.add_argument("--shards", required=True,
                        help="comma-separated shard ids to tokenize")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=72,
                        help="device batch size for packing (must match training)")
    parser.add_argument("--token_dtype", default="u16", choices=["u16", "u32"],
                        help="shard dtype; u32 required for tokenizer vocab > 65535 "
                             "(train.py --token_dtype must match)")
    parser.add_argument("--lib_data_dir", default="",
                        help="override lib.DATA_DIR (parquet pool) after import, e.g. "
                             "the full-163 shared pool on /data2. Pair with "
                             "FIXED_TOKENIZER_DIR env to point at a readable tokenizer.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Set the cache location before loading the compatible upstream library.
    os.environ["AUTORESEARCH_CACHE_DIR"] = args.cache_dir
    sys.path.insert(0, args.lib_dir)

    from lib import Tokenizer, TOKENIZER_DIR, make_dataloader
    import lib as _lib

    if args.lib_data_dir:
        _lib.DATA_DIR = args.lib_data_dir
        print(f"[prepare] lib.DATA_DIR overridden -> {args.lib_data_dir}")
    tok = Tokenizer.from_directory(TOKENIZER_DIR)
    print(f"[prepare] vocab_size={tok.get_vocab_size()} BOS={tok.bos_token_id}")

    shard_ids = [int(x) for x in args.shards.split(",") if x.strip()]
    data_split_path = os.path.abspath(
        args.split_path or os.path.join(args.lib_dir, "data_split.json")
    )
    with open(data_split_path) as f:
        original_split = json.load(f)

    T = args.seq_len
    B = args.batch_size
    row_capacity = T + 1
    out_dtype = np.uint32 if args.token_dtype == "u32" else np.uint16

    for sid in shard_ids:
        # Write a temporary data_split.json that puts this single shard as "train"
        # and a dummy different shard as "test" (required non-empty by lib.py).
        # lib.py rejects train/test overlap, so the dummy must differ from sid.
        dummy_test = [s for s in shard_ids if s != sid][:1]
        if not dummy_test:
            dummy_test = [6542] if sid != 6542 else [0]
        tmp_split = {"train": [sid], "test": dummy_test}
        with open(data_split_path, "w") as f:
            json.dump(tmp_split, f)
        try:
            dl = make_dataloader(tok, B=B, T=T, split="train",
                                 data_mode="fixed", data_seed=42)
            # Collect exactly one epoch of packed batches
            all_rows = []
            epoch = 1
            while True:
                inp, tgt, ep = next(dl)
                if ep != epoch:
                    break
                # reconstruct full rows (T+1): input + last target token
                # inp: (B, T), tgt: (B, T) where tgt = inp shifted by 1
                # full row = [inp[b, 0], ..., inp[b, T-1], tgt[b, T-1]]
                last_tgt = tgt[:, -1:].squeeze(-1)  # (B,)
                full = torch.cat([inp, last_tgt.unsqueeze(-1)], dim=-1).cpu()  # (B, T+1)
                for b in range(B):
                    all_rows.append(full[b].numpy().astype(out_dtype))
            arr = np.stack(all_rows)  # (n_rows, T+1)
            out_path = os.path.join(args.out_dir, f"shard_{sid:05d}.bin")
            arr.tofile(out_path)
            print(f"[prepare] shard {sid}: {len(all_rows)} rows -> {out_path} "
                  f"({os.path.getsize(out_path)/1e6:.1f} MB, {args.token_dtype})")
        finally:
            # restore
            with open(data_split_path, "w") as f:
                json.dump(original_split, f)


if __name__ == "__main__":
    main()
