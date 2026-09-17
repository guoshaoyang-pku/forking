"""Pick concrete suppression examples from eval_logits_rank.py per-position npz.

Mirror of pick_sharpen_examples.py, for the OPPOSITE effect: a val position
whose bigram context was seen in train and whose true continuation is
reasonable (native predicts it at rank 0 at epoch 1) but whose true token is
progressively crowded out by +both (true rank worsens / true prob collapses
by epoch 3, while the +both top predictions are train-seen candidates).

For every val position (per-position npz, top-32) with context (a,b,c) and
true token y:
- bigram ctx = (b, c), F = count(b, c) in the train shard,
- cont novel iff count(b, c, y) == 0 (3-gram),
- candidate token p is train-seen iff count(b, c, p) > 0.

Selection: nogram e1 true_rank == 0 AND both e3 true_rank >= --min_rank_e3
AND both e3 true_prob < nogram e1 true_prob / --prob_drop. Output JSON has,
per example and per (arm, epoch): top-k token ids + probs + train-seen
flags, true rank/prob, and decoded strings (tokenizer.pkl must be provided).

Usage (ophis-gpu, repo venv):
  .venv/bin/python code/tools/pick_suppression_examples.py \
    --shard data/tokenized/shard_00001.bin \
    --npz_dir data/runs_scaling/logits_rank --stem marm_epochs \
    --tokenizer /data4/shared/ph/.cache/autoresearch/tokenizer/tokenizer.pkl \
    --out_json data/runs_scaling/logits_rank/marm_suppression_examples.json \
    --top 8
"""

import argparse
import json
import os
import pickle

import numpy as np

V = 8192
ARMS = ("nogram", "both")
EPOCHS = ("e1", "e2", "e3")


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


def make_decoder(pkl_path):
    with open(pkl_path, "rb") as f:
        enc = pickle.load(f)

    def one(tok):
        try:
            return enc.decode([int(tok)])
        except Exception:
            try:
                return enc.decode_bytes([int(tok)]).decode("utf-8", "replace")
            except Exception:
                return f"<{int(tok)}>"

    return one, lambda ids: enc.decode([int(i) for i in ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--stem", default="marm_epochs")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min_rank_e3", type=int, default=10,
                    help="+both e3 true_rank must be >= this")
    ap.add_argument("--prob_drop", type=float, default=3.0,
                    help="both e3 true_prob must be < nogram e1 true_prob / this")
    ap.add_argument("--gradual", action="store_true",
                    help="also require +both e1 true_rank <= --max_rank_e1 "
                         "and rank the collapse by relative prob drop "
                         "e1 -> e3 (gradual suppression), instead of by the "
                         "e3 rank depth (already-dead cases)")
    ap.add_argument("--max_rank_e1", type=int, default=10,
                    help="+both e1 true_rank bound for --gradual")
    ap.add_argument("--gradual_drop", type=float, default=5.0,
                    help="both e3 true_prob < both e1 true_prob / this")
    ap.add_argument("--topk_show", type=int, default=8)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    tok = load_tokens(args.shard)
    print(f"[suppress] {len(tok):,} tokens; counting 2/3-grams ...")
    k2, c2 = ngram_counts(tok, 2)
    k3, c3 = ngram_counts(tok, 3)
    print(f"[suppress] distinct 2/3-grams: {len(k2):,} {len(k3):,}")

    decode_one, decode_seq = make_decoder(args.tokenizer)

    data = {}
    for arm in ARMS:
        for ep in EPOCHS:
            path = os.path.join(args.npz_dir,
                                f"{args.stem}_{arm}_{arm}_{ep}_val.npz")
            with np.load(path) as z:
                data[(arm, ep)] = {k: z[k] for k in z.files}
    n = data[("nogram", "e1")]["true"].shape[0]
    for ep in EPOCHS:
        assert data[("both", ep)]["true"].shape[0] == n

    a1 = data[("nogram", "e1")]
    b3 = data[("both", "e3")]
    b, c = a1["ctx3"][:, 1].astype(np.int64), a1["ctx3"][:, 2].astype(np.int64)
    y = a1["true"].astype(np.int64)
    f_ctx = count_at(k2, c2, b * V + c)
    cont_novel = count_at(k3, c3, (b * V + c) * V + y) == 0

    r1n = a1["true_rank"].astype(np.int64)
    r3b = b3["true_rank"].astype(np.int64)
    p1n = a1["true_prob"].astype(np.float64)
    p3b = b3["true_prob"].astype(np.float64)
    b1 = data[("both", "e1")]
    r1b = b1["true_rank"].astype(np.int64)
    p1b = b1["true_prob"].astype(np.float64)

    if args.gradual:
        mask = (f_ctx > 0) & cont_novel & (r1n == 0) \
            & (r1b <= args.max_rank_e1) & (r3b >= args.min_rank_e3) \
            & (p3b < p1b / args.gradual_drop)
    else:
        mask = (f_ctx > 0) & cont_novel & (r1n == 0) \
            & (r3b >= args.min_rank_e3) & (p3b < p1n / args.prob_drop)
    cand = np.where(mask)[0]
    print(f"[suppress] candidate positions: {len(cand):,}")
    if len(cand) == 0:
        raise SystemExit("no candidate; relax --min_rank_e3/--prob_drop")
    if args.gradual:
        ratio = p3b[cand] / np.maximum(p1b[cand], 1e-12)
        order = cand[np.argsort(ratio)][:args.top]
    else:
        rank_fall = r3b[cand].astype(np.float64)
        ratio = p3b[cand] / np.maximum(p1n[cand], 1e-9)
        order = cand[np.lexsort((ratio, -rank_fall))][:args.top]

    out = {"meta": {
        "shard": os.path.basename(args.shard),
        "npz_stem": args.stem,
        "tokenizer": os.path.basename(args.tokenizer.rstrip('/')),
        "n_positions": int(n),
        "selection": "val, bigram ctx seen (count>0), true continuation "
                     "novel (3-gram absent); nogram e1 true_rank==0; "
                     f"both e3 true_rank>={args.min_rank_e3}; "
                     f"both e3 true_prob < nogram e1 true_prob/{args.prob_drop:g}; "
                     "sorted by e3 true rank desc then prob ratio asc",
        "topk": int(data[("nogram", "e1")]["topk_probs"].shape[1]),
    }, "examples": []}

    for i in order:
        i = int(i)
        ex = {"position": i, "ctx_ids": a1["ctx3"][i].tolist(),
              "true_token": int(y[i]),
              "ctx_str": decode_seq(a1["ctx3"][i]),
              "ctx_true_str": decode_seq(list(a1["ctx3"][i]) + [int(y[i])]),
              "true_str": decode_one(y[i]),
              "f_bigram": int(f_ctx[i])}
        for arm in ARMS:
            for ep in EPOCHS:
                d = data[(arm, ep)]
                k = args.topk_show
                ids = d["topk_ids"][i][:k].astype(np.int64)
                probs = d["topk_probs"][i][:k].astype(np.float64)
                seen = count_at(k3, c3,
                                (b[i] * V + c[i]) * V + ids) > 0
                ex[f"{arm}_{ep}"] = {
                    "topk_ids": ids.tolist(),
                    "topk_strs": [decode_one(t) for t in ids],
                    "topk_probs": [round(float(v), 6) for v in probs],
                    "topk_seen": seen.tolist(),
                    "true_rank": int(d["true_rank"][i]),
                    "true_prob": float(d["true_prob"][i]),
                }
        out["examples"].append(ex)

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[suppress] wrote {args.out_json} ({len(out['examples'])} cases)")
    for ex in out["examples"][:3]:
        print(f"  pos {ex['position']} ctx={ex['ctx_str']!r} "
              f"true={ex['true_str']!r} f={ex['f_bigram']}: "
              f"true_prob nogram_e1 {ex['nogram_e1']['true_prob']:.4f} -> "
              f"both_e3 {ex['both_e3']['true_prob']:.6f} "
              f"(rank 0 -> {ex['both_e3']['true_rank']})")


if __name__ == "__main__":
    main()
