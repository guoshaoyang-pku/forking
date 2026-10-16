"""Diagonal gap-frequency power-law fits for the v2 order sweep (step STEP).

Per run: gap(bucket) = val - online train mean loss, buckets with >=MIN_TOK tokens
on BOTH sides, F_mid = geometric centre of the bucket.  Fit log gap = log A - alpha log F
(OLS, gap>0 only) per seed; report mean +- SD over seeds.  Also the val novel share
(f=0 positions) and the final gap from summary.json.
"""
import glob, json, os, sys
import numpy as np

RUNS, STEP, OUT = sys.argv[1], 2000, sys.argv[2]
MIN_TOK = 300
EDGES = [(1,1,"1"),(2,2,"2"),(3,3,"3"),(4,4,"4"),(5,5,"5"),(6,8,"6-8"),(9,12,"9-12"),
         (13,20,"13-20"),(21,30,"21-30"),(31,50,"31-50"),(51,75,"51-75"),(76,100,"76-100"),
         (101,150,"101-150"),(151,200,"151-200"),(201,300,"201-300"),(301,500,"301-500"),
         (501,750,"501-750"),(751,1000,"751-1k"),(1001,2000,"1k-2k"),(2001,5000,"2k-5k"),
         (5001,10000,"5k-10k"),(10001,10**9,"10k+")]
DIAG = {"2": "bigram", "3": "trigram", "4": "4gram", "5": "5gram", "6": "6gram", "7": "7gram", "8": "8gram"}
res = {}
for d in sorted(glob.glob(os.path.join(RUNS, "v2ord_*_fixed"))):
    if not os.path.exists(os.path.join(d, "summary.json")):
        continue
    name = os.path.basename(d)
    arm = name.split("_")[1]; seed = int(name.split("_")[2][1:])
    summ = json.load(open(os.path.join(d, "summary.json")))
    rec = [json.loads(l) for l in open(os.path.join(d, "freq_bin_loss.jsonl"))]
    r = [x for x in rec if x["step"] == STEP][0]
    targets = ["bigram", "trigram", "4gram", "5gram", "6gram", "7gram", "8gram"] if arm == "n2-3-4-5" else \
              ([DIAG[arm]] if arm in DIAG else ["bigram", "trigram", "4gram", "5gram", "6gram", "7gram", "8gram"])
    for br in targets:
        if br not in r["val"]:
            continue
        val = r["val"][br]
        tot = sum(v["token_count"] for v in val.values())
        novel_share = val["novel"]["token_count"] / tot
        F, G = [], []
        for lo, hi, lab in EDGES:
            v = val.get(lab, {}); t = r["train"].get(br, {}).get(lab, {})
            if v.get("token_count", 0) >= MIN_TOK and t.get("token_count", 0) >= MIN_TOK:
                gap = v["mean_loss"] - t["mean_loss"]
                F.append(np.sqrt(lo * hi)); G.append(gap)
        F, G = np.array(F), np.array(G)
        pos = G > 0
        alpha = r2 = A = np.nan
        if pos.sum() >= 3:
            X = np.log(F[pos]); Y = np.log(G[pos])
            b, a = np.polyfit(X, Y, 1)
            alpha, A = -b, np.exp(a)
            pred = a + b * X
            r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
        res.setdefault((arm, br), []).append(dict(seed=seed, alpha=alpha, A=A, r2=r2,
            n_bins=int(pos.sum()), novel=novel_share, final_gap=summ["final_gap"],
            F_range=(int(F.min()) if F.size else 0, int(F.max()) if F.size else 0),
            g_F1=float(G[0]) if F.size and F[0] < 2 else np.nan))
lines = ["arm,branch,n_seeds,alpha_mean,alpha_sd,A_mean,R2_mean,bins,val_novel_share,final_gap_mean,final_gap_sd,gap_F1_mean"]
for (arm, br), L in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    al = np.array([x["alpha"] for x in L]); A = np.array([x["A"] for x in L])
    r2 = np.array([x["r2"] for x in L]); nov = np.array([x["novel"] for x in L])
    fg = np.array([x["final_gap"] for x in L]); g1 = np.array([x["g_F1"] for x in L])
    sd = lambda a: float(np.nanstd(a, ddof=1)) if np.sum(~np.isnan(a)) > 1 else float("nan")
    lines.append(f"{arm},{br},{len(L)},{np.nanmean(al):.3f},{sd(al):.3f},{np.nanmean(A):.3g},{np.nanmean(r2):.3f},"
                 f"{int(np.mean([x['n_bins'] for x in L]))},{nov.mean():.4f},{fg.mean():.4f},{sd(fg):.4f},{np.nanmean(g1):.3f}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
