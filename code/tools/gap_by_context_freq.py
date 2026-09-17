"""Gap split by n-gram context frequency bin, from freq_bin_loss.jsonl.

Reads the freq_bin_loss.jsonl of module-arm runs at the three epoch
boundaries (e1ckpt last step=337, e2ckpt=674, ckpt=1000) and reports, per
branch (bigram/trigram) and context-frequency bin:

- val mean loss, train mean loss, per-token gap (val - train) at that bin,
- frac of tokens in the bin (val side),
- gap contribution of the bin (val frac x per-token gap).

The bins come from the training-time accumulator (exact context hit count f,
plus "novel" = context never seen in the training stream). Usage:

  python code/tools/gap_by_context_freq.py \
    --run_tpl '/path/s1v5_128_marm_{arm}_{e}ckpt_fixed' \
    --out_json /path/marm_gap_by_ctxfreq.json
"""

import argparse
import json
import os

ARMS = ["nogram", "bigram", "trigram", "both"]
EPOCH_STEP = {"e1": 337, "e2": 674, "e3": 1000}
BRANCHES = ["bigram", "trigram"]


def last_record(path: str) -> dict:
    rec = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_tpl", required=True,
                    help="template with {arm} and {e} placeholders, e.g. "
                         "'/path/s1v5_128_marm_{arm}_{e}ckpt_fixed'")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    out = {"meta": {"note": "per-token gap = val mean_loss - train mean_loss "
                            "within the same context-frequency bin; bins are "
                            "context hit-count ranges in the training stream "
                            "(novel = context never seen in train)"},
           "arms": {}}
    for arm in ARMS:
        for e, step in EPOCH_STEP.items():
            cands = [args.run_tpl.format(arm=arm, e=e),
                     args.run_tpl.format(arm=arm, e="").replace("__", "_")]
            path = next((c + "/freq_bin_loss.jsonl" for c in cands
                         if os.path.exists(c + "/freq_bin_loss.jsonl")), None)
            if path is None:
                print(f"[skip] {arm} {e}: no freq_bin_loss.jsonl")
                continue
            rec = last_record(path)
            assert rec["step"] == step, (path, rec["step"], step)
            for branch in BRANCHES:
                if branch not in rec.get("val", {}) or branch not in rec.get("train", {}):
                    continue
                vb, tb = rec["val"][branch], rec["train"][branch]
                total = sum(v["token_count"] for v in vb.values()) or 1
                bins = {}
                for key in vb:  # recorded bin order (novel, 1, 2, ..., 10k+)
                    if key not in tb:
                        continue
                    tv, tt = vb[key]["token_count"], tb[key]["token_count"]
                    if tv == 0 or tt == 0:
                        continue
                    mv = vb[key]["mean_loss"]
                    mt = tb[key]["mean_loss"]
                    bins[key] = {
                        "n_val": int(tv), "frac_val": tv / total,
                        "val_mean_loss": mv, "train_mean_loss": mt,
                        "gap_mean": mv - mt,
                        "gap_contrib": (mv - mt) * tv / total,
                    }
                out["arms"].setdefault(arm, {}).setdefault(
                    branch, {})[f"{e}_step{step}"] = bins
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[gap-by-freq] wrote {args.out_json}")
    for arm in ("bigram", "trigram", "both"):
        for branch in BRANCHES:
            d = out["arms"].get(arm, {}).get(branch, {}).get("e3_step1000")
            if not d:
                continue
            row = " ".join(f"{g}:{d[g]['gap_mean']:+.2f}(n={d[g]['frac_val']:.2f})"
                           for g in d)
            print(f"{arm}/{branch} e3 gap: {row}")


if __name__ == "__main__":
    main()
