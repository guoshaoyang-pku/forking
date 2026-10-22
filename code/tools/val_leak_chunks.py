"""Leak audit of an explicit val chunk set: every K-token window of the chosen
val chunks vs every train shard. Prints per-shard hit fraction and, per val
chunk, how many of its windows occur verbatim anywhere in the pool.

usage: val_leak_chunks.py <tokenized_dir> <val_sid> <first_chunk> <n_chunks> <K> <out.npz> [exclude_sids]
"""
import os, sys
import numpy as np
from multiprocessing import Pool

D, VSID, C0, NC, K, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
EXCL = {int(x) for x in (sys.argv[7].split(",") if len(sys.argv) > 7 else [])}
CH = 2049
P = np.uint64(1000003)


def hashes(t):
    t = t.astype(np.uint64)
    n = len(t) - K + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(K):
            h = h * P + t[k:k + n]
    return h


def load(sid):
    return np.fromfile(os.path.join(D, f"shard_{sid:05d}.bin"), dtype=np.uint16)


v = load(VSID)[C0 * CH:(C0 + NC) * CH].reshape(NC, CH)
vh = np.stack([hashes(c) for c in v])          # (NC, CH-K+1)
flat = vh.ravel()
uniq = np.unique(flat)


def check(sid):
    th = np.unique(hashes(load(sid)))
    return sid, np.isin(uniq, th, assume_unique=True)


if __name__ == "__main__":
    sids = sorted(int(f[6:11]) for f in os.listdir(D) if f.startswith("shard_") and f.endswith(".bin"))
    sids = [s for s in sids if s != VSID and s not in EXCL]
    any_hit = np.zeros(uniq.size, dtype=bool)
    per = {}
    with Pool(24) as p:
        for sid, hit in p.imap(check, sids):
            per[sid] = float(hit.mean()); any_hit |= hit
            if per[sid] > 0.002:
                print("shard", sid, f"{per[sid]:.5f}", flush=True)
    hit_flat = any_hit[np.searchsorted(uniq, flat)].reshape(vh.shape)
    per_chunk = hit_flat.sum(1)
    print(f"val chunks {NC}: windows hit anywhere {hit_flat.mean():.5f}; chunks with >=1 hit "
          f"{(per_chunk > 0).sum()}; with >=10% windows hit {(per_chunk > 0.1 * vh.shape[1]).sum()}", flush=True)
    np.savez(OUT, per_chunk=per_chunk, sids=np.array(list(per)), frac=np.array(list(per.values())))
