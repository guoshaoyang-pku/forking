"""Seen-vs-novel continuation split of the module-arm logits eval npz files.

Tests the core mechanism claim: continuations that were SEEN in the training
data at a given n-gram context get their probability sharpened over epochs,
while other (novel) continuations at seen contexts are suppressed or left
behind — the "seen continuation crowds out other continuations" effect.

For every position of the fixed eval batches (stored per-position in
<stem>_<arm>_<arm>_<e>_<side>.npz by eval_logits_rank.py --save_topk):
- context = (a, b, c) = (t-2, t-1, t), continuation y = true token at t+1;
- trigram context count  F3 = count of (a,b,c) in the train shard,
- bigram context count   F2 = count of (b,c),
- seen-continuation counts: c4(a,b,c,y) and c3(b,c,y) (n-gram with the
  continuation appended), unigram count of y;
- top-k predicted tokens p: c4(a,b,c,p) seen mask -> seen probability mass.

Aggregates per (arm, epoch, side) into context-frequency bins x seen/novel:
n, mean true_prob, rank-0 fraction, mean top-1 prob; plus overall
seen/novel mass within the top-k. N-gram counts come from a single scan of
the train shard (uint16 token ids).

Usage:
  python code/tools/context_freq_split.py \
    --shard /path/data/tokenized/shard_00001.bin \
    --npz_dir /path/logits_rank --stem marm_epochs \
    --out_json /path/marm_seen_novel.json
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


def load_tokens(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16).astype(np.int64)


def ngram_counts(tokens: np.ndarray, n: int):
    key = np.zeros(len(tokens) - n + 1, dtype=np.int64)
    for i in range(n):
        key = key * V + tokens[i:len(tokens) - n + 1 + i]
    return np.unique(key, return_counts=True)


def count_at(keys_sorted: np.ndarray, counts_sorted: np.ndarray,
             q: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(keys_sorted, q)
    idx_c = np.clip(idx, 0, len(keys_sorted) - 1)
    hit = keys_sorted[idx_c] == q
    return np.where(hit, counts_sorted[idx_c], 0).astype(np.int32)


def bin_of(f: np.ndarray) -> np.ndarray:
    out = np.zeros(len(f), dtype=np.int8)  # 0 = novel context
    for bi, (lo, hi) in enumerate(F_BINS):
        out[(f >= lo) & (f <= hi)] = bi + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--stem", default="marm_epochs")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    print("[split] scanning train shard for n-gram counts ...")
    tok = load_tokens(args.shard)
    print(f"[split] {len(tok):,} tokens")
    k2, c2 = ngram_counts(tok, 2)
    k3, c3 = ngram_counts(tok, 3)
    k4, c4 = ngram_counts(tok, 4)
    uni = np.bincount(tok, minlength=V).astype(np.int32)
    print(f"[split] distinct 2/3/4-grams: {len(k2):,} {len(k3):,} {len(k4):,}")

    out = {"meta": {
        "shard": os.path.basename(args.shard),
        "n_tokens": int(len(tok)),
        "f_bins": {str(bi + 1): BIN_NAMES[bi] for bi in range(len(BIN_NAMES))},
        "group_note": "bin 0 = context not in train (novel); seen4 = the "
                      "exact (a,b,c,y) 4-gram occurred in train; seen_mass = "
                      "top-k probability mass on tokens p with (a,b,c,p) "
                      "seen in train",
    }, "arms": {}}

    for arm in ARMS:
        for ep in EPOCHS:
            for side in ("val", "train"):
                path = os.path.join(
                    args.npz_dir,
                    f"{args.stem}_{arm}_{arm}_{ep}_{side}.npz")
                if not os.path.exists(path):
                    print(f"[skip] {path}")
                    continue
                with np.load(path) as z:
                    ctx = z["ctx3"].astype(np.int64)      # (N,3)
                    y = z["true"].astype(np.int64)        # (N,)
                    probs = z["topk_probs"].astype(np.float32)
                    ids = z["topk_ids"].astype(np.int64)
                    true_prob = z["true_prob"].astype(np.float32)
                    trank = z["true_rank"]
                    ent = z["entropy"].astype(np.float32)
                a, b, c = ctx[:, 0], ctx[:, 1], ctx[:, 2]
                f3 = count_at(k3, c3, a * V * V + b * V + c)
                f2 = count_at(k2, c2, b * V + c)
                c4_y = count_at(k4, c4, ((a * V + b) * V + c) * V + y)
                c3_y = count_at(k3, c3, (b * V + c) * V + y)
                # seen mask for every top-k prediction
                q4 = (((a[:, None] * V + b[:, None]) * V + c[:, None]) * V
                      + ids)
                seen_top = count_at(k4, c4, q4.ravel()).reshape(ids.shape) > 0
                ptop = probs / probs.sum(axis=1, keepdims=True).clip(1e-9)

                res = {
                    "n": int(len(y)),
                    "y_unigram_mean": float(uni[y].mean()),
                    "seen_mass_topk": float(ptop[seen_top].sum() / len(y)),
                    "top1_seen4_frac": float((seen_top[:, 0]).mean()),
                    "seen4_frac": float((c4_y > 0).mean()),
                    "groups": {},
                }
                bin3 = bin_of(f3)
                seen = c4_y > 0
                for bi in range(len(BIN_NAMES) + 1):
                    for sv in (False, True):
                        m = (bin3 == bi) & (seen == sv)
                        if m.sum() == 0:
                            continue
                        key = (f"ctx{BIN_NAMES[bi-1]}" if bi > 0 else "ctxnovel")
                        key += "|seen" if sv else "|novel"
                        res["groups"][key] = {
                            "n": int(m.sum()),
                            "true_prob_mean": float(true_prob[m].mean()),
                            "rank0_frac": float((trank[m] == 0).mean()),
                            "top1_prob_mean": float(probs[m, 0].mean()),
                            "entropy_mean": float(ent[m].mean()),
                        }
                # bigram-context variant of the same split (uses c3 as seen)
                bin2 = bin_of(f2)
                seen3 = c3_y > 0
                res["groups_bigramctx"] = {}
                for bi in range(len(BIN_NAMES) + 1):
                    for sv in (False, True):
                        m = (bin2 == bi) & (seen3 == sv)
                        if m.sum() == 0:
                            continue
                        key = (f"ctx{BIN_NAMES[bi-1]}" if bi > 0 else "ctxnovel")
                        key += "|seen" if sv else "|novel"
                        res["groups_bigramctx"][key] = {
                            "n": int(m.sum()),
                            "true_prob_mean": float(true_prob[m].mean()),
                            "rank0_frac": float((trank[m] == 0).mean()),
                            "top1_prob_mean": float(probs[m, 0].mean()),
                        }
                out["arms"].setdefault(arm, {}).setdefault(ep, {})[side] = res
                print(f"[split] {arm} {ep} {side}: seen4_frac="
                      f"{res['seen4_frac']:.4f} seen_mass={res['seen_mass_topk']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"[split] wrote {args.out_json}")


if __name__ == "__main__":
    main()
