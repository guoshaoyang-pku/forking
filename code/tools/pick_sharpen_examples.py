"""Pick concrete sharpening examples from eval_logits_rank.py per-position npz.

Reads the <stem>_<label>_<side>.npz files for the three epoch boundaries of
one module arm (e1 / e2 / e3; identical position order because the eval
batches are fixed), scores every position by the top-1 probability gain from
epoch 1 to epoch 3, and dumps the top-N positions as JSON with:

- context token ids (t-2, t-1, t) and true next token,
- per-epoch top-k predictions (ids + probabilities), true-token rank/prob.

Token ids are stored raw (vocab 8192); no decoder ships with the repo, so
decoding to text happens wherever the tokenizer lives.

Usage:
  python code/tools/pick_sharpen_examples.py \
    --npz_e1 .../s1v5_128_marm_logits_rank_epochs_both_e1_val.npz \
    --npz_e2 .../s1v5_128_marm_logits_rank_epochs_both_e2_val.npz \
    --npz_e3 .../s1v5_128_marm_logits_rank_epochs_both_e3_val.npz \
    --out_json sharpen_examples_both_val.json --top 8
"""

import argparse
import json
import os

import numpy as np


def load(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz_e1", required=True)
    ap.add_argument("--npz_e2", required=True)
    ap.add_argument("--npz_e3", required=True)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    d1, d2, d3 = load(args.npz_e1), load(args.npz_e2), load(args.npz_e3)
    for name, d in (("e1", d1), ("e2", d2), ("e3", d3)):
        assert d["true"].shape == d1["true"].shape, f"{name} shape mismatch"

    p1 = d1["topk_probs"][:, 0].astype(np.float64)
    p3 = d3["topk_probs"][:, 0].astype(np.float64)
    gain = p3 - p1
    # require the epoch-3 prediction to actually be the true token: these are
    # the cleanest "n-gram memorisation sharpens the prediction" cases
    hit3 = d3["true_rank"] == 0
    order = np.where(hit3)[0][np.argsort(-gain[hit3])][:args.top]

    out = {"meta": {
        "npz_e1": os.path.basename(args.npz_e1),
        "npz_e2": os.path.basename(args.npz_e2),
        "npz_e3": os.path.basename(args.npz_e3),
        "n_positions": int(d1["true"].shape[0]),
        "top1_gain_threshold_note": "top-N by p3 - p1 among positions where "
                                    "the true token is rank-0 at epoch 3",
    }, "examples": []}
    for i in order:
        ex = {"position": int(i), "gain_top1": float(gain[i]),
              "ctx3": d1["ctx3"][i].tolist(), "true_token": int(d1["true"][i])}
        for name, d in (("e1", d1), ("e2", d2), ("e3", d3)):
            k = int(d["topk_probs"].shape[1])
            ex[name] = {
                "topk_ids": d["topk_ids"][i].tolist(),
                "topk_probs": [round(float(x), 6) for x in d["topk_probs"][i]],
                "true_rank": int(d["true_rank"][i]),
                "true_prob": float(d["true_prob"][i]),
                "entropy": float(d["entropy"][i]),
            }
        out["examples"].append(ex)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[examples] wrote {args.out_json} ({len(out['examples'])} cases)")
    for ex in out["examples"][:3]:
        print(f"  pos {ex['position']}: ctx={ex['ctx3']} true={ex['true_token']} "
              f"p_top1 {ex['e1']['topk_probs'][0]:.3f} -> {ex['e3']['topk_probs'][0]:.3f}")


if __name__ == "__main__":
    main()
