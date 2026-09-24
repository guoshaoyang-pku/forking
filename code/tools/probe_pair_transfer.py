"""Many-pair context-subspace probes on epoch checkpoints (experiment-log sec66, part 2).

A. Cross-context transfer. Pairs of train trigram contexts c1=(a1,b,c), c2=(a2,b,c) that share the
   bigram suffix and a continuation y seen >=2 times after each; control c3 has a different suffix
   and also sees y >=2 times. Sequences are real train occurrences with up to 63 tokens of left
   context. An update u is built from grad of mean log p(y | c1 occurrences) with the training LR
   ratio (backbone lr, tables lr*table_lr_scale), in "sign" (fresh Adam/RMSProp-like) and "grad"
   form. T_pair = <grad f(c2), u> / <grad f(c1), u>, T_ctrl likewise for c3.
B. Sample-ratio copying. Low-frequency train contexts whose top continuation (count a) beats the
   runner-up (count b>=1). At up to 4 held-out occurrences (with left context) we read the model's
   log-odds log p(top1)/p(top2) and compare it with the train sample log(a/b) and with the held-out
   empirical log-odds (+0.5 smoothing).
Read-only on checkpoints and data.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_context_subspace import load, group_of, _autocast_ctx  # noqa: E402

V = 8192
CHUNK = 2049
LEFT = 63


def tri_keys(x):
    x = x.astype(np.int64)
    return x[:-3] * V * V + x[1:-2] * V + x[2:-1], x[3:]


def occ_windows(x, pos_last, maxn, rng):
    """pos_last = index of the 3rd context token; returns list of token windows ending there."""
    pos_last = pos_last[(pos_last % CHUNK) <= CHUNK - 2]
    if len(pos_last) > maxn:
        pos_last = rng.choice(pos_last, size=maxn, replace=False)
    out = []
    for i in pos_last:
        start = max((i // CHUNK) * CHUNK, i - LEFT)
        out.append(np.asarray(x[start:i + 1], dtype=np.int64))
    return out


def batch_logp(model, seqs, target, device):
    """Right-padded batch; read the log-softmax at each sequence's last real position (causal: pads unseen)."""
    L = max(len(s) for s in seqs)
    X = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for r, s in enumerate(seqs):
        X[r, :len(s)] = torch.from_numpy(s)
    logits = model(X)
    last = torch.tensor([len(s) - 1 for s in seqs], device=device)
    lp = F.log_softmax(logits[torch.arange(len(seqs), device=device), last].float(), dim=-1)
    return lp[:, target] if target is not None else lp


def build_pairs(x, n_pairs, rng):
    k, nx = tri_keys(x)
    last = np.arange(len(k)) + 2
    ok = (last % CHUNK) <= CHUNK - 2
    k, nx, last = k[ok], nx[ok], last[ok]
    uctx, uF = np.unique(k, return_counts=True)
    uk, cnt = np.unique(k * V + nx, return_counts=True)
    keep = cnt >= 2
    ctx, y = uk[keep] // V, uk[keep] % V
    f = uF[np.searchsorted(uctx, ctx)]
    m = (f >= 8) & (f <= 400)
    ctx, y = ctx[m], y[m]
    sf = ctx % (V * V)
    g2 = sf * V + y
    ug, gc = np.unique(g2, return_counts=True)
    multi = ug[gc >= 2]
    order_y = np.argsort(y, kind="stable")
    y_sorted = y[order_y]
    rng.shuffle(multi)
    pairs = []
    for key in multi:
        members = ctx[g2 == key]
        yy = int(key % V); sfx = int(key // V)
        lo, hi = np.searchsorted(y_sorted, yy), np.searchsorted(y_sorted, yy, side="right")
        same_y = order_y[lo:hi]
        ctrl = ctx[same_y][sf[same_y] != sfx]
        if len(ctrl) == 0:
            continue
        c1, c2 = rng.choice(members, size=2, replace=False).tolist()
        c3 = int(rng.choice(ctrl))
        pos = {int(cx): last[(k == cx) & (nx == yy)] for cx in (c1, c2, c3)}
        pairs.append({"c1": int(c1), "c2": int(c2), "c3": int(c3), "y": yy, "pos": pos})
        if len(pairs) >= n_pairs:
            break
    return pairs


def transfer_for_pair(model, groups, params, eta, seqs, y, device):
    def grads(sq):
        with torch.enable_grad():
            f = batch_logp(model, sq, y, device).mean()
            gs = torch.autograd.grad(f, params, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g for g, p in zip(gs, params)]
    g = {key: grads(seqs[key]) for key in ("c1", "c2", "c3")}
    res = {}
    for variant in ("sign", "grad"):
        u = [eta[gr] * (torch.sign(x) if variant == "sign" else x) for gr, x in zip(groups, g["c1"])]
        out = {}
        for key in ("c1", "c2", "c3"):
            per = {}
            for gr, a, b in zip(groups, g[key], u):
                per[gr] = per.get(gr, 0.0) + float((a.double() * b.double()).sum())
            out[key] = per
        self_gain = sum(out["c1"].values())
        res[variant] = {"self": self_gain,
                        "c2": sum(out["c2"].values()) / self_gain, "c3": sum(out["c3"].values()) / self_gain,
                        "c2_by_group": {gr: v / self_gain for gr, v in out["c2"].items()},
                        "self_by_group": {gr: v / self_gain for gr, v in out["c1"].items()}}
        del u
    del g
    return res


def ratio_contexts(x_train, heldouts, n_ctx, rng):
    k, nx = tri_keys(x_train)
    last = np.arange(len(k)) + 2
    ok = (last % CHUNK) <= CHUNK - 2
    k, nx = k[ok], nx[ok]
    u, c = np.unique(k * V + nx, return_counts=True)
    kk, yy = u // V, u % V
    order = np.lexsort((-c, kk))
    kk, yy, c = kk[order], yy[order], c[order]
    start = np.r_[0, np.nonzero(np.diff(kk))[0] + 1]
    end = np.r_[start[1:], len(kk)]
    Fs = np.add.reduceat(c, start)
    s2 = np.minimum(start + 1, len(c) - 1)
    m = ((end - start) >= 2) & (Fs >= 4) & (Fs <= 64) & (c[start] >= 2) & (c[start] > c[s2])
    idx = np.nonzero(m)[0]
    idx = rng.choice(idx, size=min(n_ctx * 3, len(idx)), replace=False)
    cand = {int(kk[start[i]]): (int(yy[start[i]]), int(yy[s2[i]]), int(c[start[i]]), int(c[s2[i]]), int(Fs[i])) for i in idx}
    keys = np.array(sorted(cand), dtype=np.int64)
    ho_pos = {kx: [] for kx in cand}
    ho_cnt = {kx: [0, 0, 0] for kx in cand}
    for hname, xh in heldouts.items():
        kh, nh = tri_keys(np.asarray(xh))
        lh = np.arange(len(kh)) + 2
        mm = np.isin(kh, keys)
        for kx, nn_, ll_ in zip(kh[mm].tolist(), nh[mm].tolist(), lh[mm].tolist()):
            t = cand[kx]
            ho_cnt[kx][0] += 1
            ho_cnt[kx][1] += int(nn_ == t[0])
            ho_cnt[kx][2] += int(nn_ == t[1])
            if (ll_ % CHUNK) <= CHUNK - 2 and len(ho_pos[kx]) < 4:
                ho_pos[kx].append((hname, ll_))
    out = []
    for kx, t in cand.items():
        if ho_cnt[kx][0] >= 4 and ho_pos[kx]:
            out.append({"ctx": kx, "top1": t[0], "top2": t[1], "a": t[2], "b": t[3], "F": t[4],
                        "ho_F": ho_cnt[kx][0], "ho_a": ho_cnt[kx][1], "ho_b": ho_cnt[kx][2], "ho_pos": ho_pos[kx]})
        if len(out) >= n_ctx:
            break
    return out


def ratio_eval(model, ctxs, heldouts, device):
    rows = []
    with torch.no_grad(), _autocast_ctx("bf16"):
        for c in ctxs:
            seqs = []
            for hname, i in c["ho_pos"]:
                xh = heldouts[hname]
                start = max((i // CHUNK) * CHUNK, i - LEFT)
                seqs.append(np.asarray(xh[start:i + 1], dtype=np.int64))
            lp = batch_logp(model, seqs, None, device)
            lo = (lp[:, c["top1"]] - lp[:, c["top2"]]).mean().item()
            rows.append(lo)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpts", default="epoch1_model.pt,epoch2_model.pt,epoch3_model.pt,epoch4_model.pt,final_model.pt")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_pairs", type=int, default=48)
    ap.add_argument("--n_ratio", type=int, default=400)
    ap.add_argument("--max_occ", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    x_train = np.asarray(np.memmap(os.path.join(args.data_dir, "shard_00001.bin"), dtype=np.uint16, mode="r"))
    heldouts = {f"heldout{s}": np.memmap(os.path.join(args.data_dir, f"shard_{s:05d}.bin"), dtype=np.uint16, mode="r")
                for s in (2, 3)}
    t0 = time.time()
    pairs = build_pairs(x_train, args.n_pairs, rng)
    for p in pairs:
        p["seqs"] = {key: occ_windows(x_train, p["pos"][p[key]], args.max_occ, rng) for key in ("c1", "c2", "c3")}
    ctxs = ratio_contexts(x_train, heldouts, args.n_ratio, rng)
    print(f"[pairs] {len(pairs)} pairs, {len(ctxs)} ratio contexts, setup {time.time() - t0:.0f}s", flush=True)
    result = {"run_dir": args.run_dir, "seed": args.seed, "n_pairs": len(pairs),
              "pairs": [{k: p[k] for k in ("c1", "c2", "c3", "y")} | {"n_occ": {kk: len(v) for kk, v in p["seqs"].items()}}
                        for p in pairs],
              "ratio_ctx": [{k: c[k] for k in ("ctx", "top1", "top2", "a", "b", "F", "ho_F", "ho_a", "ho_b")} for c in ctxs],
              "ckpts": {}}
    for ck in args.ckpts.split(","):
        t1 = time.time()
        model, sc = load(args.run_dir, ck, device)
        params = [p for _, p in model.named_parameters()]
        groups = [group_of(n) for n, _ in model.named_parameters()]
        lr = sc.get("nanogpt_adam_lr", 6e-4)
        eta = {g: (lr if g == "backbone" else lr * sc.get("table_lr_scale", 128.0)) for g in set(groups)}
        tr = [transfer_for_pair(model, groups, params, eta, p["seqs"], p["y"], device) for p in pairs]
        lo = ratio_eval(model, ctxs, heldouts, device)
        result["ckpts"][ck] = {"transfer": tr, "ratio_logodds": lo}
        def med(v): return float(np.median(v))
        s2 = [t["sign"]["c2"] for t in tr]; s3 = [t["sign"]["c3"] for t in tr]
        la = np.array([math.log(c["a"] / c["b"]) for c in ctxs]); lm = np.array(lo)
        slope = float(np.polyfit(la, lm, 1)[0])
        print(f"[probe] {os.path.basename(args.run_dir)} {ck}: T_pair med {med(s2):.3f} T_ctrl med {med(s3):.3f} "
              f"| ratio slope {slope:.3f} | {time.time() - t1:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
        json.dump(result, open(args.out, "w"))


if __name__ == "__main__":
    main()
