"""Context-subspace probes on epoch checkpoints (experiment-log sec66).

Three measurements per model and per replay-epoch checkpoint:

1. Minimal-pair influence. Source s_k = [BOS] P_k " that is a" -> " dog", probe
   q_k = [BOS] P_k " there is a" -> " dog" (same prefixes P_k). An update
   u that raises mean log p(dog | s) is built from the source gradient, with
   the training learning-rate ratio (backbone 6e-4, tables 128x). Reported:
   the first-order gain on every probe scalar, split by parameter group, the
   transfer ratio T = gain(q) / gain(s), and the exact change after a finite
   step scaled so that the first-order self gain is 0.5 nats. Two update
   shapes bracket the real optimizer: "grad" (plain gradient x group LR) and
   "sign" (fresh-state Adam/RMSProp-like step, group LR x sign(grad)).
2. In-situ statistics for the trigram context " there is a" at its natural
   train (shard 1) and held-out (shard 2) occurrences, with the real prefix.
3. Dog/cat contexts: low-frequency trigram contexts where " dog" follows in
   train, " cat" never does, and " cat" does in held-out data. In-situ
   log p(dog), log p(cat) at held-out occurrences, plus context-only probes.

fp32 forward for the influence probe (small finite differences), bf16 autocast
for the in-situ sweeps (as in training). Read-only on checkpoints and data.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from train import NanoGPT, _autocast_ctx  # noqa: E402
from eval_logits_rank import cfg_from_summary  # noqa: E402

V = 8192
CHUNK = 2049
GROUP_MARKERS = {"bigram": ("bigram_ves", "bigram_gate"),
                 "trigram": ("trigram_ves", "trigram_gate"),
                 "unigram": ("value_embeds", "ve_gate")}


def group_of(name):
    for g, ms in GROUP_MARKERS.items():
        if any(m in name for m in ms):
            return g
    return "backbone"


def load(run_dir, ckpt, device):
    sc = json.load(open(os.path.join(run_dir, "summary.json")))["config"]
    model = NanoGPT(cfg_from_summary(sc))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), sc


def last_logprobs(model, seqs, bos, device):
    """log-softmax at the last real position of each (right-padded) sequence."""
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), bos, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    logits = model(x)
    last = torch.tensor([len(s) - 1 for s in seqs], device=device)
    return F.log_softmax(logits[torch.arange(len(seqs), device=device), last].float(), dim=-1)


def mean_target_logp(model, ctx_seqs, target, bos, device):
    """mean_k log p(target | ctx_seq_k); multi-token targets are chained."""
    total = 0.0
    seqs = [list(s) for s in ctx_seqs]
    for t in target:
        lp = last_logprobs(model, seqs, bos, device)
        total = total + lp[:, t]
        seqs = [s + [t] for s in seqs]
    return total.mean()


def influence(model, tok, device, lr_backbone=6e-4, table_scale=128.0):
    bos = tok["bos"]
    pre = tok["prefix_ids"]
    c_that, c_there = tok["contexts"]["that is a"], tok["contexts"]["there is a"]
    src = [[bos] + p + c_that for p in pre]
    prb = [[bos] + p + c_there for p in pre]
    tg = tok["targets"]
    scalars = {
        "s_dog": (src, tg[" dog"]), "q_dog": (prb, tg[" dog"]),
        "s_cat": (src, tg[" cat"]), "q_cat": (prb, tg[" cat"]),
        "s_rabbit": (src, tg[" rabbit"]), "q_rabbit": (prb, tg[" rabbit"]),
        "s_lot": (src, tg[" lot"]), "q_lot": (prb, tg[" lot"]),
    }
    params = [(n, p) for n, p in model.named_parameters()]
    groups = [group_of(n) for n, _ in params]
    eta = {g: (lr_backbone if g == "backbone" else lr_backbone * table_scale) for g in set(groups)}

    def grads(fn_key):
        seqs, t = scalars[fn_key]
        with torch.enable_grad():
            f = mean_target_logp(model, seqs, t, bos, device)
            gs = torch.autograd.grad(f, [p for _, p in params], allow_unused=True)
        return float(f.detach()), [torch.zeros_like(p) if g is None else g for g, (_, p) in zip(gs, params)]

    base = {}
    with torch.no_grad():
        for k, (seqs, t) in scalars.items():
            base[k] = float(mean_target_logp(model, seqs, t, bos, device))
    _, g_src = grads("s_dog")
    out = {"base_logp": base, "eta": eta, "variants": {}}
    for variant in ("grad", "sign"):
        if variant == "grad":
            u = [eta[g] * gr for g, gr in zip(groups, g_src)]
        else:
            u = [eta[g] * torch.sign(gr) for g, gr in zip(groups, g_src)]
        first = {}
        for k in scalars:
            _, gk = grads(k)
            per = {}
            for g, a, b in zip(groups, gk, u):
                per[g] = per.get(g, 0.0) + float((a.double() * b.double()).sum())
            first[k] = {"total": sum(per.values()), "by_group": per}
            del gk
        self_gain = first["s_dog"]["total"]
        alpha = 0.5 / self_gain
        with torch.no_grad():
            for (_, p), du in zip(params, u):
                p.add_(du, alpha=alpha)
            after = {k: float(mean_target_logp(model, seqs, t, bos, device)) for k, (seqs, t) in scalars.items()}
            for (_, p), du in zip(params, u):
                p.sub_(du, alpha=alpha)
        out["variants"][variant] = {
            "first_order": first,
            "transfer_q_dog": first["q_dog"]["total"] / self_gain,
            "transfer_by_group": {g: v / self_gain for g, v in first["q_dog"]["by_group"].items()},
            "finite_step_alpha": alpha,
            "finite_step_delta": {k: after[k] - base[k] for k in scalars},
        }
        del u
    rows = {}
    if hasattr(model, "trigram_ve_layers") and model.config.enable_trigram_ve:
        li = sorted(model.trigram_ve_layers)[0]
        a = torch.tensor([c_that[-3], c_there[-3]], device=device)
        b = torch.tensor([c_that[-2], c_there[-2]], device=device)
        c = torch.tensor([c_that[-1], c_there[-1]], device=device)
        rows["trigram_row_at_a"] = [int(r) for r in model._trigram_row_indices(li, a, b, c)[0].tolist()]
    if hasattr(model, "bigram_ve_layers") and model.config.enable_bigram_ve:
        li = sorted(model.bigram_ve_layers)[0]
        b = torch.tensor([c_that[-2], c_there[-2]], device=device)
        c = torch.tensor([c_that[-1], c_there[-1]], device=device)
        rows["bigram_row_at_a"] = [int(r) for r in model._bigram_row_indices(li, b, c)[0].tolist()]
    out["rows_that_vs_there"] = rows
    return out


def tri_keys(x):
    x = x.astype(np.int64)
    return x[:-3] * V * V + x[1:-2] * V + x[2:-1], x[3:]


def positions_for(x_u16, keyset):
    """Global index i of the third context token (the model predicts x[i+1])."""
    k, _ = tri_keys(x_u16)
    hit = np.nonzero(np.isin(k, keyset))[0]
    i = hit + 2
    j = i % CHUNK
    ok = (j >= 2) & (j <= CHUNK - 2)
    return i[ok], k[hit[ok]]


def insitu(model, x_u16, pos, device, row_chunk=24):
    """Per-position log-softmax rows (bf16 autocast forward over whole chunks)."""
    g = pos // CHUNK
    uniq = np.unique(g)
    res = {}
    ctx = _autocast_ctx("bf16")
    with torch.no_grad():
        for s in range(0, len(uniq), row_chunk):
            cs = uniq[s:s + row_chunk]
            batch = np.stack([np.asarray(x_u16[c * CHUNK:c * CHUNK + CHUNK - 1], dtype=np.int64) for c in cs])
            with ctx:
                logits = model(torch.from_numpy(batch).to(device))
            sel = np.isin(g, cs)
            for p_ in pos[sel]:
                c = p_ // CHUNK
                r = int(np.nonzero(cs == c)[0][0])
                res[int(p_)] = F.log_softmax(logits[r, p_ % CHUNK].float(), dim=-1).cpu()
    return res


def context_sets(tok, x_train):
    kt, nt = tri_keys(x_train)
    there = tok["contexts"]["there is a"]
    k_there = there[0] * V * V + there[1] * V + there[2]
    cand = tok["dogcat_candidates"]
    k_dc = np.array([c["ctx_ids"][0] * V * V + c["ctx_ids"][1] * V + c["ctx_ids"][2] for c in cand], dtype=np.int64)
    seen = {}
    for kk in [k_there] + k_dc.tolist():
        seen[int(kk)] = np.unique(nt[kt == kk])
    return k_there, k_dc, seen


def summarize(lp_rows, pos, keys, x, seen, tg, novel_split=True):
    dog, cat, pet = tg[" dog"][0], tg[" cat"][0], tg[" pet"][0]
    acc = {"n": 0, "logp_dog": 0.0, "logp_cat": 0.0, "logp_pet": 0.0, "mass_seen": 0.0, "entropy": 0.0,
           "ce": 0.0, "n_seen_next": 0, "ce_seen_next": 0.0, "n_novel_next": 0, "ce_novel_next": 0.0,
           "n_next_dog": 0, "ce_next_dog": 0.0, "n_next_cat": 0, "ce_next_cat": 0.0}
    for p_, kk in zip(pos, keys):
        lp = lp_rows[int(p_)]
        s = seen[int(kk)]
        nxt = int(x[p_ + 1])
        pr = lp.exp()
        acc["n"] += 1
        acc["logp_dog"] += float(lp[dog]); acc["logp_cat"] += float(lp[cat]); acc["logp_pet"] += float(lp[pet])
        acc["mass_seen"] += float(pr[torch.from_numpy(s.astype(np.int64))].sum()) if len(s) else 0.0
        acc["entropy"] += float(-(pr * lp).sum())
        ce = -float(lp[nxt]); acc["ce"] += ce
        if novel_split:
            if np.isin(nxt, s):
                acc["n_seen_next"] += 1; acc["ce_seen_next"] += ce
            else:
                acc["n_novel_next"] += 1; acc["ce_novel_next"] += ce
        if nxt == dog:
            acc["n_next_dog"] += 1; acc["ce_next_dog"] += ce
        if nxt == cat:
            acc["n_next_cat"] += 1; acc["ce_next_cat"] += ce
    n = max(1, acc["n"])
    out = {"n": acc["n"]}
    for k in ("logp_dog", "logp_cat", "logp_pet", "mass_seen", "entropy", "ce"):
        out[k] = acc[k] / n
    for a, b in (("seen_next", "ce_seen_next"), ("novel_next", "ce_novel_next"),
                 ("next_dog", "ce_next_dog"), ("next_cat", "ce_next_cat")):
        out["n_" + a] = acc["n_" + a]
        out[b] = acc[b] / acc["n_" + a] if acc["n_" + a] else None
    return out


def context_only(model, tok, device):
    bos = tok["bos"]
    tg = tok["targets"]
    rows = []
    seqs = [[bos] + tok["contexts"]["there is a"], [bos] + tok["contexts"]["that is a"]]
    seqs += [[bos] + c["ctx_ids"] for c in tok["dogcat_candidates"]]
    with torch.no_grad():
        lp = last_logprobs(model, seqs, bos, device)
    for i, s in enumerate(seqs):
        rows.append({"ctx_ids": s[1:], "logp_dog": float(lp[i, tg[" dog"][0]]), "logp_cat": float(lp[i, tg[" cat"][0]]),
                     "logp_pet": float(lp[i, tg[" pet"][0]]), "logp_lot": float(lp[i, tg[" lot"][0]])})
    return rows


def fixed_val_loss(model, data_dir, device, n_chunks=288, row_chunk=24):
    """Loss on the training-time fixed validation window (first 4x72 chunks of shard 2)."""
    x = np.memmap(os.path.join(data_dir, "shard_00002.bin"), dtype=np.uint16, mode="r")
    tot, n = 0.0, 0
    with torch.no_grad(), _autocast_ctx("bf16"):
        for s in range(0, n_chunks, row_chunk):
            b = np.stack([np.asarray(x[c * CHUNK:(c + 1) * CHUNK], dtype=np.int64) for c in range(s, s + row_chunk)])
            b = torch.from_numpy(b).to(device)
            loss = model(b[:, :-1].contiguous(), b[:, 1:].contiguous())
            tot += float(loss) * row_chunk; n += row_chunk
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpts", default="epoch1_model.pt,epoch2_model.pt,epoch3_model.pt,epoch4_model.pt,final_model.pt")
    ap.add_argument("--tokens", required=True, help="plan7_probe_tokens.json")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--heldout_shards", default="2,3,4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    tok = json.load(open(args.tokens))
    for k in ("contexts", "targets"):
        tok[k] = {kk: list(vv) for kk, vv in tok[k].items()}
    x_train = np.memmap(os.path.join(args.data_dir, "shard_00001.bin"), dtype=np.uint16, mode="r")
    k_there, k_dc, seen = context_sets(tok, np.asarray(x_train))
    keyset = np.array([k_there] + k_dc.tolist(), dtype=np.int64)
    shards = {"train1": x_train}
    for s in args.heldout_shards.split(","):
        shards[f"heldout{s}"] = np.memmap(os.path.join(args.data_dir, f"shard_{int(s):05d}.bin"), dtype=np.uint16, mode="r")
    pos = {name: positions_for(np.asarray(x), keyset) for name, x in shards.items()}
    result = {"run_dir": args.run_dir, "heldout_shards": args.heldout_shards, "ckpts": {},
              "n_positions": {n: int(len(p[0])) for n, p in pos.items()},
              "train_seen_sizes": {"there is a": int(len(seen[int(k_there)]))}}
    for ck in args.ckpts.split(","):
        t0 = time.time()
        model, sc = load(args.run_dir, ck, device)
        entry = {"sanity_fixed_val_loss": fixed_val_loss(model, args.data_dir, device),
                 "influence": influence(model, tok, device, lr_backbone=sc.get("nanogpt_adam_lr", 6e-4),
                                        table_scale=sc.get("table_lr_scale", 128.0)),
                 "context_only": context_only(model, tok, device), "insitu": {}}
        for name, (p_, kk) in pos.items():
            x = shards[name]
            rows = insitu(model, x, p_, device)
            m_there = kk == k_there
            entry["insitu"][name] = {
                "there_is_a": summarize(rows, p_[m_there], kk[m_there], x, seen, tok["targets"]),
                "dogcat_pooled": summarize(rows, p_[~m_there], kk[~m_there], x, seen, tok["targets"]),
            }
            del rows
        result["ckpts"][ck] = entry
        del model
        torch.cuda.empty_cache()
        inf = entry["influence"]["variants"]
        print(f"[probe] {os.path.basename(args.run_dir)} {ck}: val4={entry['sanity_fixed_val_loss']:.4f} "
              f"T_grad={inf['grad']['transfer_q_dog']:.3f} "
              f"T_sign={inf['sign']['transfer_q_dog']:.3f} | there-is-a heldout mass_seen="
              f"{entry['insitu'][list(pos)[1]]['there_is_a']['mass_seen']:.3f} | {time.time() - t0:.0f}s", flush=True)
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
