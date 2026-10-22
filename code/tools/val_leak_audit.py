"""Val-leak audit: fraction of val 32-token windows that occur verbatim in each train shard.
Run on a host with all shards. Prints one line per train shard: sid frac_of_val_sample."""
import numpy as np, os, sys
from multiprocessing import Pool

D = sys.argv[1]
K = 32
VAL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 6542]
P = np.uint64(1000003)

def hashes(tok, stride=1):
    t = tok.astype(np.uint64)
    n = len(t) - K + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(K):
            h = h * P + t[k:k + n]
    return h[::stride]

def load(sid):
    return np.fromfile(os.path.join(D, f"shard_{sid:05d}.bin"), dtype=np.uint16)

vs = np.unique(np.concatenate([hashes(load(s), 61) for s in VAL]))

def check(sid):
    th = np.unique(hashes(load(sid)))
    return sid, float(np.isin(vs, th, assume_unique=True).mean())

if __name__ == "__main__":
    sids = sorted(int(f[6:11]) for f in os.listdir(D) if f.startswith("shard_") and f.endswith(".bin"))
    sids = [s for s in sids if s not in VAL]
    print("val_sample", len(vs), flush=True)
    with Pool(24) as p:
        for sid, fr in p.imap(check, sids):
            print(sid, f"{fr:.6f}", flush=True)
