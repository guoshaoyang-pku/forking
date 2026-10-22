#!/usr/bin/env python3
"""v27 clean pool: the val set is fixed (the chunks train.py actually evaluates);
the TRAIN side is filtered so that no train chunk shares any K-token span with
the val set. Guarantee holds for any pool size (10B plan): new shards go through
the same filter before entering the manifest.

audit mode (needs all shards): writes <out>/blocklist.json
  {"val": {...}, "K": 32, "drop_shards": [...], "blocked": {sid: [chunk ids]}}
  A shard is dropped whole if > DROP_FRAC of its chunks are blocked or if it
  contains > 1% of the val windows (a val copy).
apply mode: reads blocklist.json, writes <out>/shard_<sid>.bin with blocked
  chunks removed (val shards copied verbatim) and <out>/manifest.json
  (train order, per-shard chunk counts, md5 of every written file).

usage: v27_clean_pool.py audit <src_dir> <out_dir>
       v27_clean_pool.py apply <src_dir> <out_dir> [sids...]   (default: all in blocklist)
"""
import hashlib, json, os, sys
import numpy as np
from multiprocessing import Pool

K = 32
CH = 2049
VAL_SHARDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 6542]
VAL_EVAL = (2, 0, 288)          # shard, first chunk, n chunks: 4 val_batches x 72 rows
VAL_EXTRA = (2, 288, 224)       # spare clean chunks (val up to 512 chunks ~1.05M tok)
ORDER_HEAD = [1] + list(range(11, 24)) + list(range(25, 61)) + list(range(61, 100000))
DROP_FRAC = 0.05
# §137 whole-val audit: 61/64/65/66 hold 2.5-7.6% of windows of the other val shards
# (3-10, 6542) verbatim; drop them even where the evaluated chunks are not hit.
FORCE_DROP = {61, 64, 65, 66}
P = np.uint64(1000003)
MODE, SRC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def hashes(t):
    t = t.astype(np.uint64)
    n = len(t) - K + 1
    h = np.zeros(max(n, 0), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(K):
            h = h * P + t[k:k + n]
    return h


def load(sid):
    return np.fromfile(os.path.join(SRC, f"shard_{sid:05d}.bin"), dtype=np.uint16)


def val_set():
    sid, c0, n = VAL_EVAL
    _, _, n2 = VAL_EXTRA
    v = load(sid)[c0 * CH:(c0 + n + n2) * CH].reshape(n + n2, CH)
    return np.unique(np.concatenate([hashes(c) for c in v]))


VS = val_set() if MODE == "audit" else None


def audit_one(sid):
    tok = load(sid)
    nch = len(tok) // CH
    h = hashes(tok[:nch * CH])
    hit = np.isin(h, VS)
    pos = np.nonzero(hit)[0]
    # a window starting at p touches chunks p//CH .. (p+K-1)//CH
    blocked = np.unique(np.concatenate([pos // CH, (pos + K - 1) // CH])) if pos.size else np.array([], int)
    blocked = blocked[blocked < nch]
    val_frac = float(np.isin(VS, np.unique(h[hit]), assume_unique=True).mean()) if pos.size else 0.0
    return sid, nch, blocked.tolist(), val_frac


def md5(path):
    m = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            m.update(b)
    return m.hexdigest()


def apply_one(args):
    sid, blocked = args
    tok = load(sid)
    nch = len(tok) // CH
    keep = np.ones(nch, bool)
    keep[np.array(blocked, int)] = False
    out = tok[:nch * CH].reshape(nch, CH)[keep].ravel()
    path = os.path.join(OUT, f"shard_{sid:05d}.bin")
    out.tofile(path)
    return sid, int(keep.sum()), md5(path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if MODE == "audit":
        sids = sorted(int(f[6:11]) for f in os.listdir(SRC) if f.startswith("shard_") and f.endswith(".bin"))
        train = [s for s in sids if s not in VAL_SHARDS]
        res = {}
        with Pool(24) as p:
            for sid, nch, bl, vf in p.imap(audit_one, train):
                res[sid] = (nch, bl, vf)
                print(sid, nch, len(bl), f"{vf:.5f}", flush=True)
        drop = sorted(s for s, (nch, bl, vf) in res.items()
                      if vf > 0.01 or len(bl) > DROP_FRAC * nch or s in FORCE_DROP)
        bl = {str(s): res[s][1] for s in res if s not in drop}
        tot = sum(res[s][0] for s in res if s not in drop)
        nb = sum(len(v) for v in bl.values())
        meta = {"K": K, "val_eval": VAL_EVAL, "val_extra": VAL_EXTRA, "val_windows": int(VS.size),
                "drop_shards": drop, "drop_reason": {str(s): {"val_frac": res[s][2], "blocked": len(res[s][1]),
                                                               "chunks": res[s][0]} for s in drop},
                "blocked_chunks": nb, "kept_chunks": tot - nb, "blocked": bl}
        json.dump(meta, open(os.path.join(OUT, "blocklist.json"), "w"))
        print(f"drop {drop}; blocked {nb}/{tot} chunks ({nb / tot:.4%}); kept tokens {(tot - nb) * 2048 / 1e9:.3f}B")
    else:
        meta = json.load(open(os.path.join(OUT, "blocklist.json")))
        want = [int(x) for x in sys.argv[4:]] or sorted(int(s) for s in meta["blocked"])
        jobs = [(s, meta["blocked"][str(s)]) for s in want if str(s) in meta["blocked"]]
        vals = [s for s in VAL_SHARDS if os.path.exists(os.path.join(SRC, f"shard_{s:05d}.bin"))]
        with Pool(16) as p:
            done = dict((s, (n, m)) for s, n, m in p.imap(apply_one, jobs))
        for s in vals:
            dst = os.path.join(OUT, f"shard_{s:05d}.bin")
            if not os.path.exists(dst):
                os.link(os.path.join(SRC, f"shard_{s:05d}.bin"), dst) if os.stat(SRC).st_dev == os.stat(OUT).st_dev \
                    else load(s).tofile(dst)
        order = [s for s in ORDER_HEAD if s in done]
        man = {"K": K, "val_shards": VAL_SHARDS, "val_eval": meta["val_eval"],
               "train_order": order, "chunks": {str(s): done[s][0] for s in order},
               "md5": {str(s): done[s][1] for s in order},
               "tokens_total": int(sum(done[s][0] for s in order) * 2048)}
        mp = os.path.join(OUT, "manifest.json")
        if os.path.exists(mp):
            old = json.load(open(mp))
            for k in ("chunks", "md5"):
                old[k].update(man[k])
            old["train_order"] = [s for s in ORDER_HEAD if str(s) in old["chunks"]]
            old["tokens_total"] = int(sum(old["chunks"].values()) * 2048)
            man = old
        json.dump(man, open(mp, "w"), indent=0)
        print(f"applied {len(done)} shards; manifest tokens {man['tokens_total'] / 1e9:.3f}B")
