#!/usr/bin/env python3
"""W1 E1-vector clustering arm (E1V), experiment A of the shared-memory plan.

Teacher: an exact-f256 (974k-row) table trained for exactly one epoch
(run v26w1_h1_tls4_p1b_s42_e1_fd, --save_final_model).  Row i of
ngram_ves.<n>.weight is the *actually injected* vector of selected context
keys_n[i] after epoch 1 (exact974k rows are identity, verified).

Clustering: weighted spherical k-means (cosine geometry, weights = context
token counts) into R_n clusters per order, R from the largest-remainder
128k budget in w1_common.  Contexts in one cluster share one physical row.
Output is a standard single-head assignment npz (algo=e1v).

Determinism: fixed seed; assignment step is argmax over cosine sims with
ties broken by lowest cluster index (torch.max returns first max).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w1_common import (ORDERS, ROUTING_SEED, load_selected, allocate_R,
                       write_assignment)


def weighted_spherical_kmeans(X, w, K, iters, seed, chunk=8192, device="cuda"):
    import torch
    torch.manual_seed(seed)
    N, D = X.shape
    x = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    wt = torch.from_numpy(w).to(device=device, dtype=torch.float32)
    # init: weight-proportional random picks without replacement
    idx = torch.multinomial(wt, K, replacement=False)
    C = x[idx].clone()
    C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
    assign = torch.empty(N, dtype=torch.long, device=device)
    for it in range(iters):
        # E step (chunked argmax over cosine similarities)
        for s in range(0, N, chunk):
            sim = x[s:s + chunk] @ C.T
            assign[s:s + chunk] = sim.argmax(dim=1)
        # M step: weighted mean per cluster, renormalize
        Cnew = torch.zeros_like(C)
        Cnew.index_add_(0, assign, x * wt.unsqueeze(1))
        nrm = Cnew.norm(dim=1, keepdim=True)
        # empty clusters keep their previous center
        empty = (nrm.squeeze(1) <= 1e-12)
        Cnew = torch.where(empty.unsqueeze(1), C, Cnew / nrm.clamp_min(1e-12))
        shift = (Cnew - C).abs().max().item()
        C = Cnew
        print(f"[e1v] iter {it}: max center shift {shift:.5f}, empty {int(empty.sum())}", flush=True)
        if shift < 1e-5:
            break
    # distortion: weighted 1 - cos(x, center)
    tot = 0.0
    for s in range(0, N, chunk):
        sim = (x[s:s + chunk] * C[assign[s:s + chunk]]).sum(dim=1)
        tot += (wt[s:s + chunk] * (1.0 - sim)).sum().item()
    return assign.cpu().numpy(), float(tot / w.sum())


def build(ckpt_path, index_path, out_path, iters=30, seed=ROUTING_SEED, device="cuda",
          centroids_path=None):
    import torch
    sel, total = load_selected(index_path)
    R = allocate_R(sel)
    sd = torch.load(ckpt_path, map_location="cpu")
    rows, dist, cents = {}, {}, {}
    for n in ORDERS:
        key = f"ngram_ves.{n}.weight"
        W = sd[key].float().numpy()
        keys, counts = sel[n]
        K_n = keys.shape[0]
        assert W.shape[0] >= K_n, f"{key}: {W.shape} < K={K_n}"
        X = W[:K_n]  # exact974k identity: row i = context keys_n[i]
        a, d = weighted_spherical_kmeans(X, counts.astype(np.float64), R[n],
                                         iters, seed ^ (0x9E3779B9 * n), device=device)
        rows[n] = a.astype(np.int32)
        dist[n] = d
        print(f"[e1v] n={n}: K={K_n} -> R={R[n]}, weighted distortion={d:.5f}", flush=True)
        if centroids_path is not None:
            # table row content for experiment B: count-weighted mean of the
            # cluster's teacher vectors (not renormalized - norm carries the
            # teacher's learned magnitude)
            C = np.zeros((R[n], X.shape[1]), dtype=np.float32)
            np.add.at(C, a, X * counts[:, None].astype(np.float32))
            wsum = np.bincount(a, weights=counts, minlength=R[n]).astype(np.float32)
            C[wsum > 0] /= wsum[wsum > 0, None]
            cents[n] = C
    write_assignment(out_path, sel, R, rows, total,
                     {"algo": "e1v", "seed": seed, "fmin": 256,
                      "m_total": sum(R.values()), "teacher": ckpt_path,
                      "teacher_epochs": 1, "iters": iters,
                      "distortion": dist})
    print(f"[e1v] wrote {out_path}")
    if centroids_path is not None:
        payload = {f"table_{n}": cents[n] for n in ORDERS}
        payload["meta_json"] = json.dumps({"teacher": ckpt_path, "assign": out_path,
                                           "content": "count-weighted teacher centroid"})
        np.savez_compressed(centroids_path, **payload)
        print(f"[e1v] wrote centroids {centroids_path}")


if __name__ == "__main__":
    ckpt, index_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    centroids = sys.argv[5] if len(sys.argv) > 5 else None
    build(ckpt, index_path, out_path, iters=iters, centroids_path=centroids)
