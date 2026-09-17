"""Per-position paired excess-NLL vs native, by trigram-context frequency bin.

Direct experimental test of the hypothesis "the n-gram-module val damage is
concentrated at novel / low-frequency contexts": for every eval position i
(fixed batches -> identical across arms), compute

    d_i = NLL_arm(i) - NLL_nogram(i),   NLL_i = -log p_i(true token)

from the per-position npz files of eval_logits_rank.py --save_topk, bucket
positions by train-stream frequency of the trigram context (a,b,c), and
report per-bin paired mean difference with a normal-approx 95% CI
(mean +/- 1.96*SE, SE = sd/sqrt(n)). Pairing removes position-level
difficulty; bins and arm/epoch labels make the monotonicity claim testable.

Usage (cluster):
  python code/tools/nll_excess_by_ctx.py \
    --shard data/tokenized/shard_00001.bin \
    --npz_dir data/runs_scaling/logits_rank --stem marm_epochs \
    --out_json data/runs_scaling/logits_rank/marm_nll_excess.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context_freq_split import (V, load_tokens, ngram_counts, count_at,  # noqa
                                bin_of)

ARMS = ["bigram", "trigram", "both"]
EPOCHS = ["e1", "e2", "e3"]
BIN_NAMES = ["novel", "1", "2-3", "4-31", "32-511", "512+"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--stem", default="marm_epochs")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    print("[excess] scanning train shard ...")
    tok = load_tokens(args.shard)
    k3, c3 = ngram_counts(tok, 3)
    print(f"[excess] distinct trigrams: {len(k3):,}")

    def nll(path):
        with np.load(path) as z:
            ctx = z["ctx3"].astype(np.int64)
            tp = z["true_prob"].astype(np.float64)
        f3 = count_at(k3, c3, ctx[:, 0] * V * V + ctx[:, 1] * V + ctx[:, 2])
        return -np.log(np.clip(tp, 1e-6, 1.0)), bin_of(f3)

    base = {}
    for ep in EPOCHS:
        base[ep] = {}
        nll_v, bin_v = nll(os.path.join(
            args.npz_dir, f"{args.stem}_nogram_nogram_{ep}_val.npz"))
        base[ep] = (nll_v, bin_v)

    out = {"meta": {
        "shard": os.path.basename(args.shard),
        "definition": "per-position paired excess NLL vs the native run at "
                      "the same epoch, bucketed by train-stream trigram "
                      "context count; CI = mean +/- 1.96*sd/sqrt(n)",
        "bins": BIN_NAMES,
    }, "arms": {}}
    for arm in ARMS:
        for ep in EPOCHS:
            nll_a, _ = nll(os.path.join(
                args.npz_dir, f"{args.stem}_{arm}_{arm}_{ep}_val.npz"))
            nll_n, bins = base[ep]
            d = nll_a - nll_n
            total = float(d.mean())
            rec = {"total_excess_nll": total, "n": int(len(d)), "bins": {}}
            for bi, name in enumerate(BIN_NAMES):
                m = bins == bi
                n = int(m.sum())
                dv = d[m]
                se = dv.std() / np.sqrt(n)
                rec["bins"][name] = {
                    "n": n,
                    "frac_val": n / len(d),
                    "excess_mean": float(dv.mean()),
                    "ci95_lo": float(dv.mean() - 1.96 * se),
                    "ci95_hi": float(dv.mean() + 1.96 * se),
                    "contrib_share": float(dv.mean() * n / len(d) / total)
                    if abs(total) > 1e-9 else None,
                }
            out["arms"].setdefault(arm, {})[ep] = rec
            row = " ".join(f"{name}:{rec['bins'][name]['excess_mean']:+.2f}"
                           for name in BIN_NAMES)
            print(f"[excess] {arm} {ep} total={total:+.3f} | {row}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"[excess] wrote {args.out_json}")


if __name__ == "__main__":
    main()
