#!/usr/bin/env python3
"""Independent Good--Turing missing-mass statistics from a token shard.

This script deliberately does *not* derive continuation counts from the
historical ``freq_index.npz`` or from a raw-contiguous n-gram scan.  It builds
the exact context -> target stream used by ``TokenizedShardDataset`` and
``GlobalFrequencyIndex.build_from_chunks``:

* the shard is split into non-overlapping ``chunk_size`` (= sequence_len+1)
  chunks;
* at input position t, the bigram/trigram context uses repeated first-token
  padding at t=0/1;
* the target is the next token in the same chunk; no pair crosses a chunk
  boundary.

For a context frequency layer f, the output uses

    M(f) = N1(f) / (f * n(f)),

where n(f) is the number of contexts with total count f and N1(f) is the
number of (context, continuation) types occurring exactly once among those
contexts.  This is the aggregate Good--Turing missing-mass estimator.

The pair counts are computed independently from the raw token stream in
bounded batches, then globally merged with a numpy sort.  An optional old
index is used only for a diagnostic comparison of context frequencies; it is
never used to construct M(f).

Example (on the training host)::

    python3 compute_good_turing_independent.py \
      --shard /data4/.../tokenized/shard_00001.bin \
      --old-index /data4/.../data/freq_index.npz \
      --out gt_independent_20260920
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def context_keys(mat: np.ndarray, vocab: int, branch: str) -> tuple[np.ndarray, np.ndarray]:
    """Return context keys and next-token targets for a chunk batch."""
    # mat shape: (number_of_chunks, chunk_size), int64.  The first T tokens
    # are inputs and the final T tokens are their targets.
    t_len = mat.shape[1] - 1
    cur = mat[:, :t_len]
    prev = np.concatenate((mat[:, :1], mat[:, : t_len - 1]), axis=1)
    targets = mat[:, 1:]
    if branch == "bigram":
        return prev * vocab + cur, targets
    if branch != "trigram":
        raise ValueError(branch)
    prev2 = np.concatenate((mat[:, :2], mat[:, : t_len - 2]), axis=1)
    v2 = vocab * vocab
    return prev2 * v2 + prev * vocab + cur, targets


def iter_pair_parts(
    buf: np.ndarray,
    chunk_size: int,
    vocab: int,
    branch: str,
    batch_chunks: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield sorted unique (context,target) keys and counts per batch."""
    n_chunks = len(buf) // chunk_size
    used = n_chunks * chunk_size
    for first in range(0, n_chunks, batch_chunks):
        last = min(first + batch_chunks, n_chunks)
        # Copy just this batch.  Keeping the memmap out of the arithmetic
        # avoids surprising uint16 overflow and bounds peak memory.
        mat = np.asarray(buf[first * chunk_size : last * chunk_size], dtype=np.int64)
        mat = mat.reshape(last - first, chunk_size)
        ctx, target = context_keys(mat, vocab, branch)
        pair = (ctx * vocab + target).ravel()
        keys, counts = np.unique(pair, return_counts=True)
        yield keys.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)
        del mat, ctx, target, pair
    del used


def merge_pair_parts(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Globally merge per-batch pair counts."""
    if not parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    keys = np.concatenate([p[0] for p in parts])
    counts = np.concatenate([p[1] for p in parts]).astype(np.int64, copy=False)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    counts = counts[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    merged_keys = keys[starts]
    merged_counts = np.add.reduceat(counts, starts)
    return merged_keys, merged_counts


def per_context_from_pairs(
    pair_keys: np.ndarray, pair_counts: np.ndarray, vocab: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return context key, total f, support size, and singleton count."""
    contexts = pair_keys // vocab
    starts = np.r_[0, np.flatnonzero(contexts[1:] != contexts[:-1]) + 1]
    ctx = contexts[starts]
    f = np.add.reduceat(pair_counts, starts).astype(np.int64, copy=False)
    support = np.diff(np.r_[starts, len(pair_keys)]).astype(np.int64, copy=False)
    singleton = np.add.reduceat((pair_counts == 1).astype(np.int64), starts)
    return ctx, f, support, singleton


def aggregate_frequency(
    f: np.ndarray, support: np.ndarray, singleton: np.ndarray
) -> list[dict]:
    """Aggregate per-context statistics into exact frequency layers."""
    fmax = int(f.max()) if len(f) else 0
    n_ctx = np.bincount(f, minlength=fmax + 1).astype(np.int64)
    n_types = np.bincount(f, weights=support, minlength=fmax + 1)
    n1 = np.bincount(f, weights=singleton, minlength=fmax + 1)
    rows = []
    for freq in range(1, fmax + 1):
        if n_ctx[freq] == 0:
            continue
        denom = float(freq * n_ctx[freq])
        rows.append(
            {
                "f": freq,
                "n_contexts": int(n_ctx[freq]),
                "N1_types": int(round(n1[freq])),
                "cont_types": int(round(n_types[freq])),
                "context_tokens": int(freq * n_ctx[freq]),
                "M": float(n1[freq] / denom),
                "S_eff": float(n_types[freq] / n_ctx[freq]),
            }
        )
    return rows


def compare_old_index(ctx: np.ndarray, f: np.ndarray, old_index: str, branch: str) -> dict:
    """Compare chunk-semantic context frequencies with a legacy index."""
    if not old_index or not os.path.exists(old_index):
        return {"available": False}
    z = np.load(old_index)
    key_name = f"{branch}_keys"
    count_name = f"{branch}_counts"
    ref_k = np.asarray(z[key_name], dtype=np.int64)
    ref_f = np.asarray(z[count_name], dtype=np.int64)
    order = np.argsort(ref_k, kind="stable")
    ref_k, ref_f = ref_k[order], ref_f[order]
    pos = np.searchsorted(ref_k, ctx)
    ok = pos < len(ref_k)
    safe = np.minimum(pos, max(len(ref_k) - 1, 0))
    ok &= len(ref_k) > 0
    ok &= ref_k[safe] == ctx
    diff = f[ok] - ref_f[safe[ok]]
    return {
        "available": True,
        "old_entries": int(len(ref_k)),
        "chunk_contexts": int(len(ctx)),
        "matched_contexts": int(ok.sum()),
        "contexts_with_diff": int(np.count_nonzero(diff)),
        "max_abs_diff": int(np.max(np.abs(diff))) if len(diff) else 0,
        "sum_chunk_f": int(f.sum()),
        "sum_old_on_matched": int(ref_f[safe[ok]].sum()) if ok.any() else 0,
        "note": "old index is diagnostic only; M(f) is built from raw chunk-semantic pairs",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["f", "n_contexts", "N1_types", "cont_types",
              "context_tokens", "M", "S_eff"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-index", default="")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--chunk-size", type=int, default=2049)
    ap.add_argument("--batch-chunks", type=int, default=1024)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buf = np.memmap(args.shard, dtype=np.uint16, mode="r")
    n_chunks, tail = divmod(len(buf), args.chunk_size)
    print(f"tokens={len(buf)} chunks={n_chunks} tail={tail}")
    metadata = {
        "shard": os.path.abspath(args.shard),
        "shard_sha256": sha256(args.shard),
        "old_index": os.path.abspath(args.old_index) if args.old_index else None,
        "old_index_sha256": sha256(args.old_index) if args.old_index else None,
        "vocab_size": args.vocab_size,
        "chunk_size": args.chunk_size,
        "n_chunks": int(n_chunks),
        "tail_tokens_dropped": int(tail),
        "positions": int(n_chunks * (args.chunk_size - 1)),
        "definition": {
            "f": "chunk-semantic context occurrence count",
            "n_contexts": "number of contexts with exact f",
            "N1_types": "continuation types occurring exactly once within f layer",
            "M": "N1_types / (f * n_contexts)",
        },
    }

    for branch in ("bigram", "trigram"):
        print(f"[{branch}] counting pair types...")
        parts = []
        for i, part in enumerate(iter_pair_parts(
            buf, args.chunk_size, args.vocab_size, branch, args.batch_chunks
        ), 1):
            parts.append(part)
            if i % 4 == 0:
                print(f"[{branch}] batches={i} partial_pair_types={sum(len(p[0]) for p in parts)}", flush=True)
        pair_keys, pair_counts = merge_pair_parts(parts)
        print(f"[{branch}] unique pair types={len(pair_keys)}")
        ctx, f, support, singleton = per_context_from_pairs(
            pair_keys, pair_counts, args.vocab_size
        )
        rows = aggregate_frequency(f, support, singleton)
        write_csv(out / f"good_turing_{branch}.csv", rows)
        np.savez_compressed(
            out / f"per_context_{branch}.npz",
            context_keys=ctx, f=f, cont_types=support, N1_types=singleton
        )
        metadata[branch] = {
            "contexts": int(len(ctx)),
            "pair_types": int(len(pair_keys)),
            "positions": int(f.sum()),
            "f_max": int(f.max()),
            "old_index_crosscheck": compare_old_index(
                ctx, f, args.old_index, branch
            ),
        }
        print(
            f"[{branch}] contexts={len(ctx)} pair_types={len(pair_keys)} "
            f"f_max={int(f.max())}"
        )
        del parts, pair_keys, pair_counts, ctx, f, support, singleton

    with (out / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
