"""Per-position normalized top-k share stats from eval_logits_rank.py npz files.

For each (arm, epoch, side) npz (written by eval_logits_rank.py --save_topk),
compute the mean over positions of the probability share within the top-K
head:

    share[r] = mean_pos( p(rank r | top-K) ) = mean_pos( probs[r] / sum_j<K probs[j] )

This removes each position's overall confidence (how much mass sits in the
head at all) and shows the pure SHAPE of the head distribution, which raw
mean-probability curves mix together. Also reports mean within-head entropy.

Usage: python code/tools/topk_share_stats.py --npz_dir <dir> --stem marm_epochs --out_json <json>
"""
import argparse
import json
import os

import numpy as np

ARMS = ["nogram", "bigram", "trigram", "both"]
EPOCHS = ["e1", "e2", "e3"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--stem", default="marm_epochs")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    K = args.topk
    out = {"meta": {"topk": K,
                    "definition": "share[r] = mean over positions of "
                                  "probs[r]/sum_{j<K} probs[j]; head "
                                  "entropy = mean entropy of the top-K "
                                  "conditional distribution"},
           "arms": {}}
    for arm in ARMS:
        for ep in EPOCHS:
            for side in ("train", "val"):
                path = os.path.join(
                    args.npz_dir, f"{args.stem}_{arm}_{arm}_{ep}_{side}.npz")
                with np.load(path) as z:
                    probs = z["topk_probs"][:, :K].astype(np.float64)
                s = probs.sum(axis=1, keepdims=True)
                share = probs / s
                ent = -(share * np.log(share + 1e-12)).sum(axis=1)
                key = f"{arm}_{ep}"
                out["arms"].setdefault(key, {})[side] = {
                    "n_positions": int(probs.shape[0]),
                    "share_at_rank": [float(v) for v in share.mean(axis=0)],
                    "head_entropy_mean": float(ent.mean()),
                }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    for arm in ARMS:
        row = " ".join(
            f"{ep}:H={out['arms'][f'{arm}_{ep}']['val']['head_entropy_mean']:.2f}"
            for ep in EPOCHS)
        print(f"val {arm:8s} {row}")


if __name__ == "__main__":
    main()
