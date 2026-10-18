#!/usr/bin/env python3
"""W0 stats pass (teacher-free OE algorithm search, plan v2 2026-10-18).

One pass over the fixed 1B-token train pool (22 shards, 2049-token chunks,
contexts never cross chunks -- bit-identical to MultiOrderFrequencyIndex build
and to the model's in-chunk shift rule).

For every order n in {2,3,4,5} and every position whose context belongs to the
selected f>=256 set (key-sorted, exactly the v25 f256 dict membership), record:
  sel_idx   int32  index into the selected key list of this order
  y         uint16 continuation token (next token in chunk)
  fold      uint8  0=fit-A 1=fit-B 2=routing-dev (shard-level document folds)
  first_tok uint16 first token of the context (n=5 keys are splitmix64-hashed;
                    n<=4 keys decode exactly, this is for uniform decode)
  suffix    int64  key of the order-(n-1) suffix context ending at the same
                    position (n=2: the unigram token id itself)

Also records per-fold unigram target counts and per-order/fold hit totals.

Output: $ROOT/data/w0_stats/hits_n{n}.npz + unigram.npy + manifest.json
"""
import os, sys, json, time
import numpy as np

ROOT = os.environ.get("W0_ROOT", "/data/shared/guoshaoyang/forking-v25")
sys.path.insert(0, os.path.join(ROOT, "code"))
from ngram_freq import context_keys_np  # bit-identical keys

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "w0_stats")
SMOKE = os.environ.get("W0_SMOKE", "") == "1"
if SMOKE:
    OUT += "_smoke"
os.makedirs(OUT, exist_ok=True)

ORDERS = (2, 3, 4, 5)
VOCAB = 8192
CHUNK = 2049
FMIN = 256
FOLDS = {0: [1, 11, 12, 13, 14, 15, 16, 17, 18, 19],
         1: [20, 21, 22, 23, 25, 26, 27, 28, 29, 30],
         2: [31, 32]}
shard_fold = {s: f for f, ss in FOLDS.items() for s in ss}

SHARDS = [int(x) for x in open(os.path.join(DATA, "train_shards_1b.txt")).read().strip().split(",")]
assert SMOKE or set(SHARDS) == set(shard_fold), f"shard list mismatch: {SHARDS}"
if SMOKE:
    SHARDS = SHARDS[:1]

z = np.load(os.path.join(DATA, "freq_index_o2to5_1b_f30.npz"))
sel_keys, sel_counts_idx = {}, {}
for n in ORDERS:
    c = z[f"counts_{n}"].astype(np.int64)
    m = c >= FMIN
    sel_keys[n] = np.ascontiguousarray(z[f"keys_{n}"][m])
    sel_counts_idx[n] = c[m]  # index-count sanity reference
    print(f"[w0] n={n}: selected {sel_keys[n].size:,} contexts "
          f"(coverage ref {c[m].sum()/c.sum():.4f})", flush=True)

hits = {n: {"idx": [], "y": [], "fold": [], "first": [], "suffix": []} for n in ORDERS}
uni = np.zeros((3, VOCAB), dtype=np.int64)
hit_tot = {(n, f): 0 for n in ORDERS for f in (0, 1, 2)}

t0 = time.time()
for si, sid in enumerate(SHARDS):
    path = os.path.join(DATA, "tokenized", f"shard_{sid:05d}.bin")
    toks = np.memmap(path, dtype=np.uint16, mode="r")
    nchunks = toks.size // CHUNK
    fold = shard_fold[sid]
    buf = {n: {"idx": [], "y": [], "first": [], "suffix": []} for n in ORDERS}
    for ci in range(nchunks):
        chunk = np.asarray(toks[ci * CHUNK:(ci + 1) * CHUNK], dtype=np.int64)
        x = chunk[:-1][None, :]          # (1, 2048) inputs
        y = chunk[1:]                    # (2048,) targets
        uni[fold] += np.bincount(y, minlength=VOCAB)
        keys_nm1 = x.ravel()             # order-1 "keys" = tokens themselves
        for n in ORDERS:
            keys_n = context_keys_np(x, n, VOCAB).ravel()
            pos = np.searchsorted(sel_keys[n], keys_n)
            pos_c = np.clip(pos, 0, sel_keys[n].size - 1)
            hit = sel_keys[n][pos_c] == keys_n
            if not hit.any():
                keys_nm1 = keys_n
                continue
            h = np.flatnonzero(hit)
            buf[n]["idx"].append(pos_c[h].astype(np.int32))
            buf[n]["y"].append(y[h].astype(np.uint16))
            buf[n]["first"].append(x.ravel()[np.clip(h - (n - 1), 0, None)].astype(np.uint16))
            # hmm: first token of context ending at h is x[h-(n-1)] with clamp
            buf[n]["suffix"].append(keys_nm1[h].astype(np.int64) if n > 2
                                    else x.ravel()[h].astype(np.int64))
            hit_tot[(n, fold)] += h.size
            keys_nm1 = keys_n
    for n in ORDERS:
        for k in ("idx", "y", "first", "suffix"):
            hits[n][k].append(np.concatenate(buf[n][k]) if buf[n][k] else
                              np.empty(0, dtype={"idx": np.int32, "y": np.uint16,
                                                 "first": np.uint16, "suffix": np.int64}[k]))
        hits[n]["fold"].append(np.full(sum(a.size for a in buf[n]["idx"]), fold, dtype=np.uint8))
    print(f"[w0] shard {sid} ({si+1}/{len(SHARDS)}) fold={fold} chunks={nchunks} "
          f"elapsed {time.time()-t0:.0f}s", flush=True)

manifest = {"root": ROOT, "shards": SHARDS, "folds": {str(k): v for k, v in FOLDS.items()},
            "fmin": FMIN, "chunk": CHUNK, "vocab": VOCAB, "orders": ORDERS,
            "selected": {str(n): int(sel_keys[n].size) for n in ORDERS},
            "hit_totals": {f"{n},{f}": int(hit_tot[(n, f)]) for n in ORDERS for f in (0, 1, 2)},
            "elapsed_s": round(time.time() - t0, 1)}
for n in ORDERS:
    out = {k: np.concatenate(v) for k, v in
           [("sel_idx", hits[n]["idx"]), ("y", hits[n]["y"]), ("fold", hits[n]["fold"]),
            ("first_tok", hits[n]["first"]), ("suffix", hits[n]["suffix"])]}
    # sanity: bincount(sel_idx) must equal the index counts of selected contexts
    bc = np.bincount(out["sel_idx"], minlength=sel_keys[n].size).astype(np.int64)
    if not SMOKE:
        diff = int(np.abs(bc - sel_counts_idx[n]).max())
        assert diff == 0, f"n={n}: hit-count vs freq-index mismatch {diff}"
    out["keys"] = sel_keys[n]
    np.savez_compressed(os.path.join(OUT, f"hits_n{n}.npz"), **out)
    print(f"[w0] n={n}: saved {out['sel_idx'].size:,} hits, count sanity OK", flush=True)
np.save(os.path.join(OUT, "unigram.npy"), uni)
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("[w0] DONE", json.dumps(manifest["hit_totals"]), flush=True)
