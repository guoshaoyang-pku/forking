"""ngram-gap-lab · code/table_occupancy.py

Table occupancy / collision diagnostics for the table-size scaling line.

For each branch (bigram / trigram), each active layer, and each hash table
(the two decorrelated hashes), we hash every context key that actually
appears in one training epoch (the first `epoch_batches * device_batch_size`
chunks of the train prefix, using the model's exact chunk-boundary semantics)
into the physical hash-table rows, then measure:

  - physical rows R and logical addresses (R for clean tables; 2R for legacy two-hash tables)
  - exact distinct contexts K
  - occupied rows / occupancy = occupied / R
  - singleton context fraction (fraction of occupied rows holding exactly 1 key)
  - collision rate = (K - occupied) / K
  - mean / p95 co-occupants (distinct keys per occupied row)
  - frequency-weighted row load (sum of train hit counts per row / R)

The hash used here MUST be identical to the model's (code/train.py
`NanoGPT._compute_input_ngram_residual`): bigram keys
((prev*p1) ^ (cur*p2)) % table_size, trigram keys
((prev2*p1) ^ (prev*p2) ^ (cur*p3)) % table_size, with position-0/1
repetition semantics and per-layer prime families.  We import the primes
from train.py so there is exactly one source of truth.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from train import (  # noqa: F401  (re-exported for tests)
    _BASE_BIGRAM_PRIMES,
    _BASE_TRIGRAM_PRIMES,
    expand_bigram_hash_primes,
)


def _load_chunk_matrix(data_dir: str, shard_ids: list, chunk_size: int,
                       n_chunks: Optional[int] = None) -> np.ndarray:
    """Return (M, chunk_size) int64 chunk matrix from the train prefix.

    n_chunks=None reads all chunks; otherwise the first n_chunks.
    """
    out = []
    seen = 0
    for sid in shard_ids:
        if n_chunks is not None and seen >= n_chunks:
            break
        path = os.path.join(data_dir, f"shard_{sid:05d}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing shard: {path}")
        buf = np.memmap(path, dtype=np.uint16, mode="r")
        n = len(buf) // chunk_size
        take = min(n, (n_chunks - seen) if n_chunks is not None else n)
        out.append(np.array(buf[:take * chunk_size], dtype=np.int64).reshape(take, chunk_size))
        seen += take
    if not out:
        raise ValueError("no chunks loaded (empty shard list?)")
    return np.concatenate(out, axis=0)


def _context_keys(tokens: np.ndarray, vocab_size: int, branch: str) -> dict:
    """Return per-layer context-key arrays (the exact token tuples).

    Returns {layer_index: np.ndarray (M,T)} where each entry is the encoded
    context key (bigram: prev*vocab+cur; trigram: prev2*vocab^2+prev*vocab+cur).
    Layer index is only used as a dict key (keys are identical across layers;
    only the hash primes differ per layer).
    """
    T = tokens.shape[1] - 1
    cur = tokens[:, :T]
    prev = np.concatenate([tokens[:, :1], tokens[:, :T - 1]], axis=1)
    if branch == "bigram":
        return {0: prev * vocab_size + cur}
    prev2 = np.concatenate([tokens[:, :2], tokens[:, :T - 2]], axis=1)
    return {0: prev2 * (vocab_size * vocab_size) + prev * vocab_size + cur}


def hash_rows_for_branch(tokens: np.ndarray, vocab_size: int, table_size: int,
                         branch: str, primes_list: list) -> dict:
    """Hash one chunk-matrix's contexts into physical rows, per layer / per hash.

    tokens: (M, chunk_size) int64.  chunk_size = sequence_len + 1.
    branch: "bigram" | "trigram".
    primes_list: one prime family per layer (len == number of active layers).

    Returns {layer_index: [np.ndarray of row ids (M,T) for each hash, ...]}.
    """
    T = tokens.shape[1] - 1
    cur = tokens[:, :T]
    prev = np.concatenate([tokens[:, :1], tokens[:, :T - 1]], axis=1)
    rows = {}
    if branch == "bigram":
        for li, family in enumerate(primes_list):
            h_rows = []
            for (p1, p2) in family:
                h_rows.append(((prev * p1) ^ (cur * p2)) % table_size)
            rows[li] = h_rows
    else:  # trigram
        prev2 = np.concatenate([tokens[:, :2], tokens[:, :T - 2]], axis=1)
        for li, fam in enumerate(primes_list):
            h_rows = []
            for k in range(0, len(fam), 3):
                p1, p2, p3 = fam[k], fam[k + 1], fam[k + 2]
                h_rows.append(((prev2 * p1) ^ (prev * p2) ^ (cur * p3)) % table_size)
            rows[li] = h_rows
    return rows


def _row_occupancy_stats(row_ids: np.ndarray, context_keys: np.ndarray,
                         table_size: int, context_counts: Optional[dict] = None,
                         clean_single_table: bool = False) -> dict:
    """Compute occupancy metrics for one (branch, layer, hash) table.

    row_ids: (M,T) int64 row assignments for this hash.
    context_keys: (M,T) int64 exact context-key encoding (same shape as row_ids).
    table_size: physical row count R.
    context_counts: optional dict {context_key: train hit count} for
      frequency-weighted load; keys absent are treated as hit count 1.
    """
    flat_rows = row_ids.ravel()
    flat_keys = context_keys.ravel()
    distinct_contexts = len(np.unique(flat_keys))
    pairs = np.unique(np.column_stack((flat_rows, flat_keys)), axis=0)
    occupied_rows, co_counts = np.unique(pairs[:, 0], return_counts=True)
    occupied = len(occupied_rows)
    singleton = int(np.count_nonzero(co_counts == 1))
    if context_counts is not None:
        get = context_counts.get
        freq_load = float(sum(get(int(key), 1) for key in pairs[:, 1]) / table_size)
    else:
        freq_load = float(flat_rows.size / table_size) if table_size else 0.0
    return {
        "physical_rows_R": int(table_size),
        "logical_addresses": int(table_size if clean_single_table else 2 * table_size),
        "distinct_contexts_K": int(distinct_contexts),
        "occupied_rows": int(occupied),
        "occupancy": float(occupied / table_size) if table_size else 0.0,
        "singleton_context_fraction": float(singleton / occupied) if occupied else 0.0,
        "collision_rate": float(1.0 - occupied / distinct_contexts) if distinct_contexts else 0.0,
        "mean_co_occupants": float(co_counts.mean()) if len(co_counts) else 0.0,
        "p95_co_occupants": float(np.percentile(co_counts, 95)) if len(co_counts) else 0.0,
        "freq_weighted_row_load": freq_load,
    }


def compute_occupancy(data_dir: str, shard_ids: list, vocab_size: int,
                      sequence_len: int, device_batch_size: int,
                      epoch_batches: int, table_mult: int,
                      branch_primes: Optional[dict] = None,
                      context_counts: Optional[dict] = None,
                      bigram_clean_table: int = 0,
                      trigram_clean_table: int = 0) -> dict:
    """Compute per-branch / per-layer / per-hash occupancy for one training epoch.

    The epoch is the first `epoch_batches * device_batch_size` chunks of the
    train prefix (nested-prefix control).  If epoch_batches == 0, uses all
    chunks (full epoch).

    *_clean_table > 0: clean 单表 mode (SSOT clean-table-rework.md) --
    branch = single layer, single hash, R = its clean-table row count.
    """
    chunk_size = sequence_len + 1
    n_chunks = epoch_batches * device_batch_size if epoch_batches > 0 else None
    tokens = _load_chunk_matrix(data_dir, shard_ids, chunk_size, n_chunks)
    table_size = vocab_size * table_mult
    result = {
        "chunk_size": chunk_size,
        "n_chunks": tokens.shape[0],
        "physical_rows_per_hash": int(table_size),
        "logical_addresses": int(2 * table_size),
        "vocab_size": vocab_size,
        "table_mult": table_mult,
    }
    bigram_primes = branch_primes.get("bigram") if branch_primes else None
    trigram_primes = branch_primes.get("trigram") if branch_primes else None
    if bigram_primes is None:
        # 4 active layers in the default 8L config (has_ve alternates)
        bigram_primes = expand_bigram_hash_primes(_BASE_BIGRAM_PRIMES, 4)
    if trigram_primes is None:
        trigram_primes = _BASE_TRIGRAM_PRIMES[:3]
    out = {"bigram": {}, "trigram": {}}
    for branch, primes_list in (("bigram", bigram_primes), ("trigram", trigram_primes)):
        keys = _context_keys(tokens, vocab_size, branch)
        clean_table = bigram_clean_table if branch == "bigram" else trigram_clean_table
        if clean_table > 0:
            first_hash = (
                primes_list[0][:1]
                if branch == "bigram"
                else primes_list[0][:3]
            )
            clean_primes = [first_hash]
            rows = hash_rows_for_branch(tokens, vocab_size, clean_table,
                                        branch, clean_primes)
            for li, h_rows in rows.items():
                per_hash = [_row_occupancy_stats(h, keys[0], clean_table,
                                                 context_counts, clean_single_table=True)
                            for h in h_rows]
                out[branch][str(li)] = per_hash
            result[f"{branch}_clean_table"] = int(clean_table)
            continue
        rows = hash_rows_for_branch(tokens, vocab_size, table_size, branch, primes_list)
        for li, h_rows in rows.items():
            per_hash = [_row_occupancy_stats(h, keys[0], table_size, context_counts)
                        for h in h_rows]
            out[branch][str(li)] = per_hash
    result["branches"] = out
    return result


def save_occupancy_json(path: str, occupancy: dict):
    with open(path, "w") as f:
        json.dump(occupancy, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_shards", required=True)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--sequence_len", type=int, default=2048)
    parser.add_argument("--device_batch_size", type=int, default=72)
    parser.add_argument("--epoch_batches", type=int, default=0)
    parser.add_argument("--table_mult", type=int, default=64)
    parser.add_argument("--bigram_clean_table", type=int, default=0,
                        help=">0: clean 单表 occupancy for the bigram branch "
                             "(single layer, single hash, R rows)")
    parser.add_argument("--trigram_clean_table", type=int, default=0,
                        help=">0: clean 单表 occupancy for the trigram branch "
                        "(single layer, single hash, R rows)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    occ = compute_occupancy(
        args.data_dir,
        [int(x) for x in args.train_shards.split(",") if x.strip()],
        args.vocab_size, args.sequence_len, args.device_batch_size,
        args.epoch_batches, args.table_mult,
        bigram_clean_table=args.bigram_clean_table,
        trigram_clean_table=args.trigram_clean_table)
    save_occupancy_json(args.out, occ)
    print(f"[table_occupancy] saved {args.out}")
    print(f"[table_occupancy] default legacy logical addresses 2R = {occ['logical_addresses']}, "
          f"n_chunks = {occ['n_chunks']}")
