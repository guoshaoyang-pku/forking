"""Per-order gap-vs-frequency table from freq_bin_loss.jsonl (v2 order sweep).

gap(bucket) = val mean loss - online train mean loss, both over the same hit-count
bucket at the same logged step (train = current batch, pre-update; val = fixed val).
Novel (f=0) has no train side and is skipped.  Reads all runs under RUNS_DIR.
"""
import glob, json, os, sys, csv, math
import numpy as np

RUNS = sys.argv[1]
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
OUT = sys.argv[3]
EDGES = [(1,1,"1"),(2,2,"2"),(3,3,"3"),(4,4,"4"),(5,5,"5"),(6,8,"6-8"),(9,12,"9-12"),
         (13,20,"13-20"),(21,30,"21-30"),(31,50,"31-50"),(51,75,"51-75"),(76,100,"76-100"),
         (101,150,"101-150"),(151,200,"151-200"),(201,300,"201-300"),(301,500,"301-500"),
         (501,750,"501-750"),(751,1000,"751-1k"),(1001,2000,"1k-2k"),(2001,5000,"2k-5k"),
         (5001,10000,"5k-10k"),(10001,10**9,"10k+")]
rows = []
for d in sorted(glob.glob(os.path.join(RUNS, "v2ord_*_fixed"))):
    name = os.path.basename(d)
    if not os.path.exists(os.path.join(d, "summary.json")):
        continue
    arm, seed = name.split("_")[1], int(name.split("_")[2][1:])
    fb = [json.loads(l) for l in open(os.path.join(d, "freq_bin_loss.jsonl"))]
    rec = [r for r in fb if r["step"] == STEP]
    if not rec:
        print("no step", STEP, name, file=sys.stderr); continue
    r = rec[0]
    for branch in r["val"]:
        for lo, hi, label in EDGES:
            v = r["val"][branch].get(label, {}); t = r["train"].get(branch, {}).get(label, {})
            if not v.get("token_count") or not t.get("token_count"):
                continue
            rows.append(dict(arm=arm, seed=seed, branch=branch, bucket=label, F_lo=lo, F_hi=hi,
                             n_train=t["token_count"], n_val=v["token_count"],
                             train_mean=t["mean_loss"], val_mean=v["mean_loss"],
                             gap=v["mean_loss"] - t["mean_loss"]))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("rows", len(rows), "->", OUT)
