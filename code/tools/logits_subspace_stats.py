"""Rank-resolved "where does the probability mass go" statistics.

Core-logic figure for Section 4.1: at a context that was seen in training,
the n-gram module concentrates output mass on the continuations that were
seen with that context in the train shard (the "subspace"). On val positions
whose true continuation is novel for that context, the true token lies
outside this set, so sharpening the head necessarily crowds it out.

For every eval position (per-position npz from eval_logits_rank.py
--save_topk, top-32) with context (a,b,c) and true token y:
- key "bigram":  ctx = (b,c), F = count(b,c), cont seen iff count(b,c,y)>0,
                 candidate p at rank r is "seen" iff count(b,c,p)>0
- key "trigram": ctx = (a,b,c), F = count(a,b,c), seen iff count(a,b,c,y)>0

Groups: ctx seen/novel x cont seen/novel (4 groups) and, for seen ctx, the
5 frequency bins used elsewhere. Per group:
  n, true_prob_mean, rank0_frac,
  rank_prob[r]       mean p at rank r (r < 10)
  rank_seen_prob[r]  mean p at rank r restricted to train-seen candidates
  seen_mass          mean top-32 mass on train-seen candidates
  top32_mass         mean total top-32 mass
  true_rank_hist     counts in [0,1,2,3,4,5-9,10-31,32+]

Usage:
  python code/tools/logits_subspace_stats.py \
    --shard data/tokenized/shard_00001.bin \
    --npz_dir data/runs_scaling/logits_rank --stem marm_epochs \
    --out_json data/runs_scaling/marm_subspace_mass.json
"""

import argparse
import json
import os

import numpy as np

ARMS = ["nogram", "bigram", "trigram", "both"]
EPOCHS = ["e1", "e2", "e3"]
V = 8192
F_BINS = [(1, 1), (2, 3), (4, 31), (32, 511), (512, 1 << 30)]
BIN_NAMES = ["1", "2-3", "4-31", "32-511", "512+"]
RANK_EDGES = [0, 1, 2, 3, 4, 5, 10, 32, 1 << 30]
RANK_NAMES = ["0", "1", "2", "3", "4", "5-9", "10-31", "32+"]
NR = 10


def load_tokens(path):
    return np.fromfile(path, dtype=np.uint16).astype(np.int64)


def ngram_counts(tokens, n):
    key = np.zeros(len(tokens) - n + 1, dtype=np.int64)
    for i in range(n):
        key = key * V + tokens[i:len(tokens) - n + 1 + i]
    return np.unique(key, return_counts=True)


def count_at(keys, counts, q):
    idx = np.searchsorted(keys, q)
    idx_c = np.clip(idx, 0, len(keys) - 1)
    hit = keys[idx_c] == q
    return np.where(hit, counts[idx_c], 0).astype(np.int32)


def bin_of(f):
    out = np.zeros(len(f), dtype=np.int8)
    for bi, (lo, hi) in enumerate(F_BINS):
        out[(f >= lo) & (f <= hi)] = bi + 1
    return out


def group_stats(m, probs, seen_top, true_prob, trank):
    n = int(m.sum())
    if n == 0:
        return None
    p = probs[m]
    s = seen_top[m]
    tr = trank[m]
    hist = np.histogram(tr, bins=RANK_EDGES)[0]
    ce = -np.log(np.maximum(true_prob[m], 1e-9))
    return {
        "n": n,
        "true_prob_mean": float(true_prob[m].mean()),
        "ce_mean": float(ce.mean()),
        "ppl": float(np.exp(ce.mean())),
        "rank0_frac": float((tr == 0).mean()),
        "rank_prob": [float(v) for v in p[:, :NR].mean(axis=0)],
        "rank_seen_prob": [float(v) for v in (p[:, :NR] * s[:, :NR]).mean(axis=0)],
        "seen_mass": float((p * s).sum(axis=1).mean()),
        "top32_mass": float(p.sum(axis=1).mean()),
        "true_rank_hist": [int(v) for v in hist],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--stem", default="marm_epochs")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    tok = load_tokens(args.shard)
    print(f"[mass] {len(tok):,} tokens; counting 2/3/4-grams ...")
    k2, c2 = ngram_counts(tok, 2)
    k3, c3 = ngram_counts(tok, 3)
    k4, c4 = ngram_counts(tok, 4)
    print(f"[mass] distinct 2/3/4-grams: {len(k2):,} {len(k3):,} {len(k4):,}")

    out = {"meta": {
        "shard": os.path.basename(args.shard), "n_tokens": int(len(tok)),
        "f_bins": BIN_NAMES, "rank_hist_bins": RANK_NAMES, "n_rank": NR,
        "topk": 32,
        "note": "seen = candidate/true token co-occurred with the context in "
                "the train shard; masses are top-32 only",
    }, "arms": {}}

    for arm in ARMS:
        for ep in EPOCHS:
            for side in ("val", "train"):
                path = os.path.join(args.npz_dir,
                                    f"{args.stem}_{arm}_{arm}_{ep}_{side}.npz")
                if not os.path.exists(path):
                    print(f"[skip] {path}")
                    continue
                with np.load(path) as z:
                    ctx = z["ctx3"].astype(np.int64)
                    y = z["true"].astype(np.int64)
                    probs = z["topk_probs"].astype(np.float32)
                    ids = z["topk_ids"].astype(np.int64)
                    true_prob = z["true_prob"].astype(np.float32)
                    trank = z["true_rank"].astype(np.int64)
                a, b, c = ctx[:, 0], ctx[:, 1], ctx[:, 2]
                res = {}
                for key in ("bigram", "trigram"):
                    if key == "bigram":
                        f = count_at(k2, c2, b * V + c)
                        cont = count_at(k3, c3, (b * V + c) * V + y) > 0
                        q = ((b[:, None] * V + c[:, None]) * V + ids)
                        seen_top = count_at(k3, c3, q.ravel()).reshape(ids.shape) > 0
                    else:
                        f = count_at(k3, c3, a * V * V + b * V + c)
                        cont = count_at(k4, c4, ((a * V + b) * V + c) * V + y) > 0
                        q = (((a[:, None] * V + b[:, None]) * V + c[:, None]) * V + ids)
                        seen_top = count_at(k4, c4, q.ravel()).reshape(ids.shape) > 0
                    fb = bin_of(f)
                    groups = {}
                    for cname, cm in (("ctxseen", f > 0), ("ctxnovel", f == 0)):
                        for sname, sm in (("seen", cont), ("novel", ~cont)):
                            g = group_stats(cm & sm, probs, seen_top, true_prob, trank)
                            if g is not None:
                                groups[f"{cname}|{sname}"] = g
                    for bi, bn in enumerate(BIN_NAMES):
                        for sname, sm in (("seen", cont), ("novel", ~cont)):
                            g = group_stats((fb == bi + 1) & sm, probs, seen_top,
                                            true_prob, trank)
                            if g is not None:
                                groups[f"ctx{bn}|{sname}"] = g
                    groups["all"] = group_stats(np.ones(len(y), bool), probs,
                                                seen_top, true_prob, trank)
                    res[key] = groups
                out["arms"].setdefault(arm, {}).setdefault(ep, {})[side] = res
                g = res["bigram"]
                print(f"[mass] {arm} {ep} {side}: "
                      f"seen-cont p(true)={g.get('ctxseen|seen', {}).get('true_prob_mean', 0):.3f} "
                      f"novel-cont p(true)={g.get('ctxseen|novel', {}).get('true_prob_mean', 0):.3f} "
                      f"novel-cont seen_mass={g.get('ctxseen|novel', {}).get('seen_mass', 0):.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh)
    print(f"[mass] wrote {args.out_json}")


if __name__ == "__main__":
    main()
