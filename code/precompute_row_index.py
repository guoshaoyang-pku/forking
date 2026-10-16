#!/usr/bin/env python3
"""ngram-gap-lab · code/precompute_row_index.py

Offline precomputation of n-gram table row indices for the training fast path.

The clean single-table mainline (input injection, K=1, first hash family)
derives each position's table row deterministically from the token stream:

    bigram row  = ((prev * p1) ^ (cur * p2)) % R_bigram
    trigram row = ((prev2 * q1) ^ (prev * q2) ^ (cur * q3)) % R_trigram

with chunk-local boundary semantics identical to train.py's forward
(position 0 repeats itself; position 1's prev2 is the current token).
Because the replay data stream is fixed, these indices can be computed once
offline and stored next to the tokens, so the training forward becomes a pure
embedding gather (no Python-dict hash primes, no dynamic arithmetic) — which
is what made torch.compile slow (experiment-log §18: 3.5x slowdown).

Outputs (per shard, written next to the token shards in --data-dir):
    shard_<id>.brows.u32.bin   uint32 bigram row per token position (1:1 aligned)
    shard_<id>.trows.u32.bin   uint32 trigram row per token position
    shard_<id>.bmask.u8.bin    uint8 validity bit per position (1 = apply OE)
    shard_<id>.tmask.u8.bin    uint8 validity bit (trigram branch)
    rows_meta.json             contract (primes, R, vocab, dtype, shards, mask)

Static low-frequency mask (--freq_index + --min_count): enabled = (f >= t),
plus novel (f == 0) contexts kept when --keep_novel 1.  This is the
PREVENTION form of the mask (disabled from step 0), the dictionary/top-K
semantics of the index-first setting.  It deliberately differs from the
`mask_low_freq` intervention (which always masks novel): plan-8 G3 records
why — §35.8 `causalv5m_mask_low_le0_e1` showed masking ONLY novel contexts
RAISES gap (2.945 vs control 2.724), i.e. novel table readout is a net
positive for val, so the prevention dictionary keeps novel contexts hashed.

Rows are computed for EVERY packed-stream position with the model's exact
chunk-local boundary semantics; the loader slices the first
sequence_len positions of each chunk (the input part). Train shards only:
val evals keep the dynamic hash path, so val shards need no row files.

Verification contract: train.py --row_index re-derives rows for the first
batch from the model's own hash and refuses to start on any mismatch, so a
stale or mismatched row file fails loudly instead of silently forking.

Usage:
  python code/precompute_row_index.py \
    --data_dir data/tokenized --shards 1 \
    --vocab_size 8192 --bigram_clean_table 1048576 --trigram_clean_table 1048576
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import _BASE_BIGRAM_PRIMES, _BASE_TRIGRAM_PRIMES  # noqa: E402

BLOCK_ROWS = 4096  # packed rows processed per numpy block (RAM bound)


def compute_rows(tokens_2d: np.ndarray, primes: tuple, R: int) -> np.ndarray:
    """Row index per position for one branch.

    tokens_2d: (n, chunk) int64 packed rows.  Returns (n, chunk) uint32.
    Boundary semantics replicate train.py forward:
      prev  = cat([tok[:, :1],  tok[:, :-1]], axis=1)
      prev2 = cat([tok[:, :2],  tok[:, :-2]], axis=1)
    """
    if len(primes) == 2:      # bigram: (p1, p2)
        prev = np.concatenate([tokens_2d[:, :1], tokens_2d[:, :-1]], axis=1)
        rows = ((prev * primes[0]) ^ (tokens_2d * primes[1])) % R
    else:                     # trigram: (q1, q2, q3) on (prev2, prev, cur)
        prev2 = np.concatenate([tokens_2d[:, :2], tokens_2d[:, :-2]], axis=1)
        prev = np.concatenate([tokens_2d[:, :1], tokens_2d[:, :-1]], axis=1)
        rows = ((prev2 * primes[0]) ^ (prev * primes[1]) ^ (tokens_2d * primes[2])) % R
    return rows.astype(np.uint32)


def compute_context_keys_2d(tokens_2d: np.ndarray, vocab_size: int, branch: str) -> np.ndarray:
    """Context keys per position, matching ngram_freq semantics.

    bigram:  prev * V + cur
    trigram: prev2 * V^2 + prev * V + cur
    """
    V = vocab_size
    prev = np.concatenate([tokens_2d[:, :1], tokens_2d[:, :-1]], axis=1)
    if branch == "bigram":
        return prev * V + tokens_2d
    prev2 = np.concatenate([tokens_2d[:, :2], tokens_2d[:, :-2]], axis=1)
    return prev2 * (V * V) + prev * V + tokens_2d


def compute_enabled_bits(keys: np.ndarray, freq_index, branch: str,
                         min_count: int, keep_novel: bool) -> np.ndarray:
    """uint8 validity per position: 1 = apply OE.

    enabled = (f >= min_count) | (novel & keep_novel), where novel = f == 0
    (context absent from the train-scan frequency index).
    """
    f = freq_index.hit_count_numpy(branch, keys)
    enabled = f >= min_count
    if keep_novel:
        enabled |= f == 0
    return enabled.astype(np.uint8)



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--shards", required=True, help="comma-separated shard ids (train only)")
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--bigram_clean_table", type=int, default=0)
    ap.add_argument("--trigram_clean_table", type=int, default=0)
    ap.add_argument("--sequence_len", type=int, default=2048)
    ap.add_argument("--token_dtype", default="u16", choices=["u16", "u32"])
    ap.add_argument("--out_dir", default="",
                    help="where to write rows/mask streams + rows_meta.json "
                         "(default: data_dir). Use a WRITABLE local volume — on "
                         "ophis-gpu data/tokenized symlinks into the quota-dead "
                         "home; pass --out_dir data/rows there")
    ap.add_argument("--freq_index", default="",
                    help="freq_index.npz (GlobalFrequencyIndex); enables the static "
                         "low-frequency validity bits when --min_count > 0")
    ap.add_argument("--min_count", type=int, default=0,
                    help="static mask threshold t: contexts with f < t are disabled "
                         "(validity 0); 0 = write no mask streams")
    ap.add_argument("--keep_novel", type=int, default=1,
                    help="1: novel (f==0) contexts stay enabled in the static mask "
                         "(plan-8 G3; le0 evidence: novel readout is net positive)")
    args = ap.parse_args()

    if args.min_count > 0 and not args.freq_index:
        raise SystemExit("--min_count > 0 requires --freq_index")

    if args.bigram_clean_table <= 0 and args.trigram_clean_table <= 0:
        raise SystemExit("enable at least one branch: --bigram_clean_table / --trigram_clean_table")
    for name in ("bigram_clean_table", "trigram_clean_table"):
        R = getattr(args, name)
        if R > 2 ** 32 - 1:
            raise SystemExit(f"{name}={R} exceeds uint32 row storage")
    if args.vocab_size > 65535 and args.token_dtype != "u32":
        raise SystemExit("vocab_size > 65535 requires --token_dtype u32 shards")

    bi_primes = _BASE_BIGRAM_PRIMES[0][0]        # clean K=1 first family
    tri_primes = tuple(_BASE_TRIGRAM_PRIMES[0][:3])
    chunk = args.sequence_len + 1
    np_dtype = np.uint32 if args.token_dtype == "u32" else np.uint16
    shard_ids = [int(x) for x in args.shards.split(",") if x.strip()]
    out_dir = args.out_dir or args.data_dir
    if out_dir != args.data_dir:
        os.makedirs(out_dir, exist_ok=True)

    freq_index = None
    if args.min_count > 0:
        from ngram_freq import GlobalFrequencyIndex
        freq_index = GlobalFrequencyIndex.load(args.freq_index)

    for sid in shard_ids:
        tok_path = os.path.join(args.data_dir, f"shard_{sid:05d}.bin")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"Missing shard: {tok_path}")
        tok = np.memmap(tok_path, dtype=np_dtype, mode="r")
        n_rows = len(tok) // chunk
        b_path = os.path.join(out_dir, f"shard_{sid:05d}.brows.u32.bin")
        t_path = os.path.join(out_dir, f"shard_{sid:05d}.trows.u32.bin")
        bm_path = os.path.join(out_dir, f"shard_{sid:05d}.bmask.u8.bin")
        tm_path = os.path.join(out_dir, f"shard_{sid:05d}.tmask.u8.bin")
        written = 0
        want_masks = freq_index is not None
        with open(b_path, "wb") as fb, open(t_path, "wb") as ft:
            fbm = open(bm_path, "wb") if want_masks else None
            ftm = open(tm_path, "wb") if want_masks else None
            try:
                for r0 in range(0, n_rows, BLOCK_ROWS):
                    r1 = min(n_rows, r0 + BLOCK_ROWS)
                    toks = np.array(tok[r0 * chunk: r1 * chunk],
                                    dtype=np.int64).reshape(r1 - r0, chunk)
                    if args.bigram_clean_table > 0:
                        compute_rows(toks, bi_primes, args.bigram_clean_table).tofile(fb)
                        if want_masks:
                            bkeys = compute_context_keys_2d(toks, args.vocab_size, "bigram")
                            compute_enabled_bits(bkeys, freq_index, "bigram",
                                                 args.min_count, bool(args.keep_novel)).tofile(fbm)
                    if args.trigram_clean_table > 0:
                        compute_rows(toks, tri_primes, args.trigram_clean_table).tofile(ft)
                        if want_masks:
                            tkeys = compute_context_keys_2d(toks, args.vocab_size, "trigram")
                            compute_enabled_bits(tkeys, freq_index, "trigram",
                                                 args.min_count, bool(args.keep_novel)).tofile(ftm)
                    written += r1 - r0
            finally:
                if fbm is not None:
                    fbm.close()
                if ftm is not None:
                    ftm.close()
        size_info = []
        if args.bigram_clean_table > 0:
            size_info.append(f"{os.path.getsize(b_path)/1e6:.1f}MB (b)")
            if want_masks:
                size_info.append(f"{os.path.getsize(bm_path)/1e6:.1f}MB (bm)")
        if args.trigram_clean_table > 0:
            size_info.append(f"{os.path.getsize(t_path)/1e6:.1f}MB (t)")
            if want_masks:
                size_info.append(f"{os.path.getsize(tm_path)/1e6:.1f}MB (tm)")
        print(f"[precomp] shard {sid}: {written:,} rows x {chunk} positions -> "
              + " / ".join(size_info))

    meta = {
        "schema_version": 1,
        "vocab_size": args.vocab_size,
        "bigram_clean_table": args.bigram_clean_table,
        "trigram_clean_table": args.trigram_clean_table,
        "enable_bigram": args.bigram_clean_table > 0,
        "enable_trigram": args.trigram_clean_table > 0,
        "sequence_len": args.sequence_len,
        "token_dtype": args.token_dtype,
        "bigram_primes": list(bi_primes),
        "trigram_primes": list(tri_primes),
        "shards": shard_ids,
        "static_low_mask": {
            "min_count": args.min_count,
            "keep_novel": bool(args.keep_novel),
            "freq_index": os.path.abspath(args.freq_index) if args.freq_index else None,
            "enabled": args.min_count > 0,
        },
        "alignment": "1:1 with the packed token stream; chunk-local boundary "
                     "semantics identical to train.py forward (pos0 repeats, "
                     "pos1 prev2 = current token)",
        "source": "code/precompute_row_index.py",
    }
    meta_path = os.path.join(out_dir, "rows_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"[precomp] wrote {meta_path}")


if __name__ == "__main__":
    main()
