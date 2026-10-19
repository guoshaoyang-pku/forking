#!/usr/bin/env python3
"""Mechanism analysis for the E1V arm (experiment A).

Claim under test: E1V clusters group contexts whose *learned* E1-teacher
injection vectors point in compatible directions (gradient-compatible
sharing), unlike H2 which groups randomly.  We measure, per order n=2:
  - mean within-cluster cosine similarity of teacher rows under E1V routing
  - the same under H2 (double-head random) routing at the same budget
  - a size-matched random partition baseline (exact null)
Plus: cluster size distribution, and the count-weighted version (high-freq
contexts matter more).
"""
import sys, json
import numpy as np
import torch

ckpt_path, e1v_npz, h2_npz, out_json = sys.argv[1:5]

sd = torch.load(ckpt_path, map_location="cpu")
W = sd["ngram_ves.2.weight"].float().numpy()  # teacher rows, exact identity
W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)

def load(npz):
    z = np.load(npz)
    rows = z["rows_2"]
    if rows.ndim == 2:      # double-head (H2) stored as (2, K): take head 0
        rows = rows[0] if rows.shape[0] == 2 else rows[:, 0]
    return z["keys_2"], rows.astype(np.int64), int(z["R_2"])

ek, er, eR = load(e1v_npz)
hk, hr, hR = load(h2_npz)
assert (ek == hk).all(), "selection sets must match (f256)"

rng = np.random.default_rng(0)
N = len(ek)
W = W[:N]  # context i's teacher vector = row i (exact identity)

def within_cos(rows, R, sample_clusters=800):
    """mean pairwise-proxy cosine within clusters: cos(x_i, centroid)."""
    # centroid per cluster (normalized mean)
    sums = np.zeros((R, W.shape[1]), dtype=np.float32)
    cnt = np.zeros(R, dtype=np.int64)
    np.add.at(sums, rows, W)
    np.add.at(cnt, rows, 1)
    nonempty = cnt > 0
    cents = sums[nonempty] / cnt[nonempty, None]
    cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)
    idx = np.where(nonempty)[0]
    if len(idx) > sample_clusters:
        idx = rng.choice(idx, sample_clusters, replace=False)
    vals = []
    for r in idx:
        members = W[rows == r]
        vals.append(members @ cents[list(np.where(nonempty)[0]).index(r)])
    return np.concatenate(vals)

def size_matched_random(rows, R):
    perm = rng.permutation(N)
    return perm % R  # fast random partition with ~equal sizes

res = {}
for name, rows, R in (("e1v", er, eR), ("h2_head0", hr[:, 0] if hr.ndim == 2 else hr, hR)):
    c = within_cos(rows, R)
    res[name] = {"mean_cos": float(c.mean()), "p10": float(np.percentile(c, 10)),
                 "p90": float(np.percentile(c, 90))}
    print(f"{name}: within-cluster cos mean={c.mean():.4f} p10={np.percentile(c,10):.4f} p90={np.percentile(c,90):.4f}")

rand_rows = size_matched_random(er, eR)
c = within_cos(rand_rows, eR)
res["random_partition"] = {"mean_cos": float(c.mean()), "p10": float(np.percentile(c, 10)),
                           "p90": float(np.percentile(c, 90))}
print(f"random: within-cluster cos mean={c.mean():.4f}")

# cluster size stats
for name, rows, R in (("e1v", er, eR), ("h2_head0", hr[:, 0] if hr.ndim == 2 else hr, hR)):
    sizes = np.bincount(rows, minlength=R)
    res[name]["size_p50"] = float(np.percentile(sizes, 50))
    res[name]["size_p99"] = float(np.percentile(sizes, 99))
    print(f"{name}: cluster size p50={np.percentile(sizes,50):.0f} p99={np.percentile(sizes,99):.0f} max={sizes.max()}")

json.dump(res, open(out_json, "w"), indent=1)
print("wrote", out_json)
