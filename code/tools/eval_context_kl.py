#!/usr/bin/env python3
"""Per-context KL drift between two checkpoints, split by random context sets.

Fig7* measurement: for a context set S (A = the 2% of distinct trigram
context types that the random_context_loss_mask run masks, B = unmasked
peers, same selection as train.py's setup_context_eval_sets), compute

    KL(p_ref || p_new) and KL(p_new || p_ref)

per validation position over the fixed val batches, and aggregate
(mean/median/p90) over positions whose trigram context is in S. Full-vocab
exact KL (no top-k truncation): both checkpoints are scored in the same
process.

Typical pairs (all runs under data/runs_scaling/, shard-1 marm, seed 42):
  preceding normal epoch:  ref = <arm>_e1ckpt_fixed/final_model.pt
                           new = <arm>_e2ckpt_fixed/final_model.pt
  masked epoch:            ref = <arm>_ctxmask_e2_fixed/epoch2_model.pt
                           new = <arm>_ctxmask_e2_fixed/final_model.pt
  normal control epoch:    ref = <arm>_e2ckpt_fixed/final_model.pt
                           new = <arm>_ckpt_fixed/final_model.pt

Example:
  python code/tools/eval_context_kl.py \
    --data_dir data/tokenized \
    --freq_index data/freq_index.npz \
    --pair prev_normal=both_e1ckpt@final_model.pt,both_e2ckpt@final_model.pt \
    --out_json docs/appendices/context_subspace_mask_20260921/context_kl.json

Restricted-KL mode (--restrict_seen_novel): per val position with trigram
context ctx, restrict both distributions to

    S(ctx) = {y': (ctx, y') is a train-shard 4-gram}     (seen continuations)
             U {true tokens at val positions with ctx}   (val continuations)

renormalize p_ref and p_new on S, and report KL_S(p_ref || p_new) =
sum_{i in S} p~_i log(p~_i / q~_i). Also reports the exact decomposition

    KL_full = Z_ref * KL_S + Z_ref * log(Z_ref / Z_new) + [outside-S part]

which separates the within-subspace rearrangement from the mass exchange
with the "other" background. Supports/gather indices are built once and
shared across pairs (val batches are fixed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # code/
sys.path.insert(0, HERE)                   # code/tools/
from train import _autocast_ctx, TokenizedShardDataset  # noqa: E402
from eval_logits_rank import load_model  # noqa: E402
from ngram_freq import GlobalFrequencyIndex  # noqa: E402


def resolve_model(spec: str, runs_root: str) -> str:
    """'run@file' -> <runs_root>/run_fixed/file; absolute paths pass through."""
    if os.path.isfile(spec):
        return spec
    run, fname = spec.split("@", 1)
    return os.path.join(runs_root, f"{run}_fixed", fname)


def load_model_from_file(path: str, ref_run_dir: str, device):
    """Load a model whose weights live in `path`, taking the architecture
    config from the run directory (defaults: the run containing `path`)."""
    run_dir = os.path.dirname(path) or ref_run_dir
    model, summary = load_model(run_dir, device)
    if os.path.basename(path) != "final_model.pt":
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
    return model, summary


def _ngram_keys(tokens: np.ndarray, n: int, V: int) -> np.ndarray:
    out = np.zeros(len(tokens) - n + 1, dtype=np.int64)
    for i in range(n):
        out = out * V + tokens[i:len(tokens) - n + 1 + i]
    return out


def _csr_from_sorted_unique(keys: np.ndarray, V: int):
    """Sorted unique (ctx*V + y) keys -> CSR: context dctx[i] has
    continuations ys[off[i]:off[i+1]] (sorted, unique)."""
    ctx = keys // V
    ys = (keys - ctx * V).astype(np.int32)
    first = np.flatnonzero(np.concatenate(([1], ctx[1:] != ctx[:-1])))
    off = np.concatenate([first, [keys.size]]).astype(np.int64)
    return ctx[first], off, ys


def _ctx_slice(k: int, dctx, off, ys):
    i = int(np.searchsorted(dctx, k))
    if i < dctx.size and dctx[i] == k:
        return ys[off[i]:off[i + 1]]
    return None


def build_restricted_support(val_batches, train_shard: str, V: int, device):
    """Per-position restricted supports S(ctx), built once for all pairs.

    S(ctx) = train-shard 4-gram continuations of ctx UNION the true tokens
    at val positions with ctx (from these fixed val batches). Context keys
    are computed exactly like train.py's context_set_membership (padded
    prev2/prev). The two components (train-seen continuations, val-only
    novel continuations) are kept separately as well. Returns a dict with
    per-batch gather groups for "S", "seen" and "valn".
    """
    tok = np.fromfile(train_shard, dtype=np.uint16).astype(np.int64)
    print(f"[rkl] train shard {train_shard}: {len(tok):,} tokens; counting 4-grams ...")
    k4 = np.unique(_ngram_keys(tok, 4, V))
    dctx_t, off_t, ys_t = _csr_from_sorted_unique(k4, V)
    del k4, tok
    print(f"[rkl] train contexts with >=1 continuation: {dctx_t.size:,}")

    per_batch, vkeys_all, keys_all = [], [], []
    for inp, tgt in val_batches:
        idx = inp.cpu().numpy().astype(np.int64)
        y = tgt.cpu().numpy().astype(np.int64)
        prev = np.concatenate([idx[:, :1], idx[:, :-1]], axis=1)
        prev2 = np.concatenate([idx[:, :2], idx[:, :-2]], axis=1)
        key = prev2 * (V * V) + prev * V + idx
        valid = tgt.cpu().numpy().reshape(-1) != -1
        vf = np.flatnonzero(valid)
        kv = key.reshape(-1)[vf]
        per_batch.append((vf, kv))
        vkeys_all.append(kv * V + y.reshape(-1)[vf])
        keys_all.append(kv)
    dctx_v, off_v, ys_v = _csr_from_sorted_unique(
        np.unique(np.concatenate(vkeys_all)), V)
    uk = np.unique(np.concatenate(keys_all))
    print(f"[rkl] distinct val contexts: {uk.size:,}; building supports ...")

    sup_off = np.zeros(uk.size + 1, dtype=np.int64)
    seen_off = np.zeros(uk.size + 1, dtype=np.int64)
    valn_off = np.zeros(uk.size + 1, dtype=np.int64)
    parts, seen_parts, valn_parts = [], [], []
    empty = np.empty(0, dtype=np.int64)
    for i, k in enumerate(uk):
        a = _ctx_slice(int(k), dctx_t, off_t, ys_t)
        b = _ctx_slice(int(k), dctx_v, off_v, ys_v)
        seen = empty if a is None else a.astype(np.int64)
        valn = empty if b is None else (
            b.astype(np.int64) if a is None else np.setdiff1d(b, a))
        parts.append(np.union1d(seen, valn))
        seen_parts.append(seen)
        valn_parts.append(valn)
        sup_off[i + 1] = sup_off[i] + parts[-1].size
        seen_off[i + 1] = seen_off[i] + seen.size
        valn_off[i + 1] = valn_off[i] + valn.size
    sup_arr = np.concatenate(parts)
    seen_arr = np.concatenate(seen_parts) if seen_off[-1] else empty
    valn_arr = np.concatenate(valn_parts) if valn_off[-1] else empty
    sup_size = np.diff(sup_off)
    print(f"[rkl] support |S|: mean={sup_size.mean():.2f} "
          f"median={np.median(sup_size):.0f} "
          f"p99={np.quantile(sup_size, 0.99):.0f} max={sup_size.max()}; "
          f"|seen| mean={np.diff(seen_off).mean():.2f}, "
          f"|val-novel| mean={np.diff(valn_off).mean():.2f}")

    return {
        comp: _bucket_groups(per_batch, uk, off, arr, device)
        for comp, off, arr in (("S", sup_off, sup_arr),
                               ("seen", seen_off, seen_arr),
                               ("valn", valn_off, valn_arr))
    }


def _bucket_groups(per_batch, uk, off, arr, device):
    """Per batch: valid rows + gather groups [(rows (m,), ids (m, s))]
    bucketed by the support size s of each position's context."""
    out = []
    for vf, kv in per_batch:
        inv = np.searchsorted(uk, kv)
        sizes = np.diff(off)[inv]
        order = np.argsort(sizes, kind="stable")
        ss = sizes[order]
        bounds = np.flatnonzero(np.concatenate(([1], ss[1:] != ss[:-1])))
        bounds = np.concatenate([bounds, [order.size]])
        groups = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            s = int(ss[lo])
            pos = order[lo:hi].astype(np.int64)
            idx2 = off[inv[pos]][:, None] + np.arange(s, dtype=np.int64)[None, :]
            groups.append((torch.from_numpy(pos).to(device),
                           torch.from_numpy(arr[idx2]).to(device)))
        out.append({"vf": torch.from_numpy(vf).to(device), "groups": groups,
                    "n_valid": int(vf.size)})
    return out


def support_mass_batch(p_ref, p_new, comp):
    """Per valid position: total probability mass of p_ref and p_new on the
    component support (seen or val-novel continuations) of one batch."""
    V = p_ref.shape[-1]
    pr = p_ref.reshape(-1, V)
    pn = p_new.reshape(-1, V)
    n = comp["n_valid"]
    mr = np.zeros(n, dtype=np.float64)
    mn = np.zeros(n, dtype=np.float64)
    vf = comp["vf"]
    for pos, sexp in comp["groups"]:
        rows = vf[pos]
        p = pos.cpu().numpy()
        mr[p] = pr[rows[:, None], sexp].sum(dim=1).double().cpu().numpy()
        mn[p] = pn[rows[:, None], sexp].sum(dim=1).double().cpu().numpy()
    return mr, mn


def restricted_kl_batch(logp_ref, logp_new, support):
    """Per valid position: KL on S(ctx) with both distributions renormalized
    on S, plus the log normalizers. Returns (rkl, logZ_ref, logZ_new)."""
    V = logp_ref.shape[-1]
    lp_r = logp_ref.reshape(-1, V)
    lp_n = logp_new.reshape(-1, V)
    n = support["n_valid"]
    rkl = np.zeros(n, dtype=np.float64)
    lzr = np.zeros(n, dtype=np.float64)
    lzn = np.zeros(n, dtype=np.float64)
    vf = support["vf"]
    for pos, sexp in support["groups"]:
        rows = vf[pos]
        lr = lp_r[rows[:, None], sexp]
        ln = lp_n[rows[:, None], sexp]
        zr = torch.logsumexp(lr, dim=1)
        zn = torch.logsumexp(ln, dim=1)
        nlr = lr - zr[:, None]
        nln = ln - zn[:, None]
        kl = (nlr.exp() * (nlr - nln)).sum(dim=1)
        pos_np = pos.cpu().numpy()
        rkl[pos_np] = kl.double().cpu().numpy()
        lzr[pos_np] = zr.double().cpu().numpy()
        lzn[pos_np] = zn.double().cpu().numpy()
    return rkl, lzr, lzn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--freq_index", required=True)
    ap.add_argument("--runs_root", default="data/runs_scaling")
    ap.add_argument("--pair", action="append", required=True,
                    help="label=ref_spec,new_spec with spec = run@file or absolute path")
    ap.add_argument("--val_shards", default="2,3,4,5,6,7,8,9,10,6542")
    ap.add_argument("--n_val_batches", type=int, default=4)
    ap.add_argument("--device_batch_size", type=int, default=72)
    ap.add_argument("--sequence_len", type=int, default=2048)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context_set_mode", choices=["type", "row"], default="type",
                    help="A/B selection space: distinct trigram types (needs "
                         "--freq_index) or trigram table rows (clean-table hash)")
    ap.add_argument("--restrict_seen_novel", action="store_true",
                    help="also compute KL restricted to S(ctx) = seen (train "
                         "4-gram) UNION val continuations, renormalized on S")
    ap.add_argument("--train_shard", default="data/tokenized/shard_00001.bin",
                    help="train shard for seen-continuation 4-gram counts")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_shards = [int(x) for x in args.val_shards.split(",") if x.strip()]
    val_ds = TokenizedShardDataset(args.data_dir, val_shards,
                                   args.sequence_len, args.device_batch_size, 42)
    it = val_ds.iter_batches(device)
    val_batches = [next(it) for _ in range(args.n_val_batches)]
    print(f"[kl] val batches={len(val_batches)} shape={tuple(val_batches[0][0].shape)} device={device}")

    restricted = None
    if args.restrict_seen_novel:
        t0 = time.time()
        restricted = build_restricted_support(val_batches, args.train_shard,
                                              args.vocab_size, device)
        print(f"[rkl] support build took {time.time() - t0:.1f}s")

    freq_index = GlobalFrequencyIndex.load(args.freq_index)
    ctx = _autocast_ctx(args.dtype)
    results = {}
    for spec in args.pair:
        label, rest = spec.split("=", 1)
        ref_spec, new_spec = rest.split(",", 1)
        ref_path = resolve_model(ref_spec, args.runs_root)
        new_path = resolve_model(new_spec, args.runs_root)
        print(f"[kl] {label}: ref={ref_path} new={new_path}")
        model_ref, s_ref = load_model_from_file(ref_path, None, device)
        model_new, s_new = load_model_from_file(new_path, None, device)
        model_new.config.context_set_mode = args.context_set_mode
        info = model_new.setup_context_eval_sets(freq_index, args.frac, args.seed)

        kl_rn, kl_nr, memb_all = [], [], []
        rkl_all, lzr_all, lzn_all = [], [], []
        msr_all, msn_all, mvr_all, mvn_all = [], [], [], []
        with torch.no_grad():
            for bi, (inp, tgt) in enumerate(val_batches):
                with ctx:
                    logits_ref = model_ref(inp)
                    logits_new = model_new(inp)
                logp_ref = F.log_softmax(logits_ref.float(), dim=-1)
                logp_new = F.log_softmax(logits_new.float(), dim=-1)
                p_ref = logp_ref.exp()
                p_new = logp_new.exp()
                valid = tgt.ne(-1)
                kl1 = (p_ref * (logp_ref - logp_new)).sum(dim=-1)
                kl2 = (p_new * (logp_new - logp_ref)).sum(dim=-1)
                kl_rn.append(kl1[valid].cpu().numpy())
                kl_nr.append(kl2[valid].cpu().numpy())
                memb_all.append(model_new.context_set_membership(inp)[valid].cpu().numpy())
                if restricted is not None:
                    r, zr, zn = restricted_kl_batch(logp_ref, logp_new,
                                                    restricted["S"][bi])
                    rkl_all.append(r)
                    lzr_all.append(zr)
                    lzn_all.append(zn)
                    sr, sn = support_mass_batch(p_ref, p_new,
                                                restricted["seen"][bi])
                    vr, vn = support_mass_batch(p_ref, p_new,
                                                restricted["valn"][bi])
                    msr_all.append(sr)
                    msn_all.append(sn)
                    mvr_all.append(vr)
                    mvn_all.append(vn)
        kl_rn = np.concatenate(kl_rn)
        kl_nr = np.concatenate(kl_nr)
        memb = np.concatenate(memb_all)
        if restricted is not None:
            rkl = np.concatenate(rkl_all)
            lz_r = np.concatenate(lzr_all)
            lz_n = np.concatenate(lzn_all)
            z_r = np.exp(lz_r)
            z_n = np.exp(lz_n)
            seen_r = np.concatenate(msr_all)
            seen_n = np.concatenate(msn_all)
            valn_r = np.concatenate(mvr_all)
            valn_n = np.concatenate(mvn_all)
            dec_weighted = z_r * rkl                 # Z_ref * KL_S
            dec_mass = z_r * (lz_r - lz_n)           # S mass-exchange term
            dec_tail = kl_rn - dec_weighted - dec_mass  # outside-S part

        sets = {"A": 1, "B": 2, "other": 0}
        out = {"ref": os.path.basename(os.path.dirname(ref_path)) + "/" + os.path.basename(ref_path),
               "new": os.path.basename(os.path.dirname(new_path)) + "/" + os.path.basename(new_path),
               "selection": info, "n_positions": int(memb.size), "sets": {}}
        for name, code in sets.items():
            m = memb == code
            out["sets"][name] = {
                "n": int(m.sum()),
                "kl_ref_new_mean": float(kl_rn[m].mean()),
                "kl_ref_new_median": float(np.median(kl_rn[m])),
                "kl_ref_new_p90": float(np.quantile(kl_rn[m], 0.9)),
                "kl_new_ref_mean": float(kl_nr[m].mean()),
            }
            if restricted is not None:
                out["sets"][name].update({
                    "rkl_ref_new_mean": float(rkl[m].mean()),
                    "rkl_ref_new_median": float(np.median(rkl[m])),
                    "rkl_ref_new_p90": float(np.quantile(rkl[m], 0.9)),
                    "z_ref_mean": float(z_r[m].mean()),
                    "z_new_mean": float(z_n[m].mean()),
                    "z_seen_ref_mean": float(seen_r[m].mean()),
                    "z_seen_new_mean": float(seen_n[m].mean()),
                    "z_valn_ref_mean": float(valn_r[m].mean()),
                    "z_valn_new_mean": float(valn_n[m].mean()),
                    "kl_full_in_S_weighted_mean": float(dec_weighted[m].mean()),
                    "kl_full_S_mass_term_mean": float(dec_mass[m].mean()),
                    "kl_full_outside_S_mean": float(dec_tail[m].mean()),
                })
        results[label] = out
        a = out["sets"]["A"]
        print(f"[kl] {label}: A n={a['n']} KL(ref||new) mean={a['kl_ref_new_mean']:.4f} "
              f"median={a['kl_ref_new_median']:.4f}")
        if restricted is not None:
            b = out["sets"]["B"]
            print(f"[rkl] {label}: A rKL={a['rkl_ref_new_mean']:.4f} | "
                  f"seen {a['z_seen_ref_mean']:.3f}->{a['z_seen_new_mean']:.3f} "
                  f"valn {a['z_valn_ref_mean']:.3f}->{a['z_valn_new_mean']:.3f} || "
                  f"B seen {b['z_seen_ref_mean']:.3f}->{b['z_seen_new_mean']:.3f} "
                  f"valn {b['z_valn_ref_mean']:.3f}->{b['z_valn_new_mean']:.3f}")
        del model_ref, model_new
        torch.cuda.empty_cache()

    payload = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "data_dir": args.data_dir,
            "val_shards": val_shards,
            "n_val_batches": args.n_val_batches,
            "frac": args.frac,
            "seed": args.seed,
            "context_set_mode": args.context_set_mode,
            "dtype": args.dtype,
            "note": "Full-vocab exact KL per validation position over the fixed "
                    "val batches; A/B from train.py setup_context_eval_sets "
                    "(2% of distinct trigram types or of trigram table rows, "
                    "seed 42).",
        },
        "pairs": results,
    }
    if restricted is not None:
        payload["meta"]["restricted"] = {
            "train_shard": args.train_shard,
            "definition": "S(ctx) = train-shard 4-gram continuations of ctx "
                          "UNION true tokens at val positions with ctx; rkl = "
                          "KL(p_ref||p_new) with both renormalized on S; "
                          "decomposition: kl_full = Z_ref*rkl (in_S_weighted) "
                          "+ Z_ref*log(Z_ref/Z_new) (S_mass_term) + outside_S",
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[kl] wrote {args.out_json}")


if __name__ == "__main__":
    main()
