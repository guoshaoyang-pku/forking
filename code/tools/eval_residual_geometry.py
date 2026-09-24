"""Offline residual-geometry eval: ||wte|| vs ||OE|| vs ||TF|| and pairwise cos^2.

Loads final_model.pt from completed run dirs (same loader as
eval_logits_rank.py), runs the SAME fixed train-head / val batches used
during training, and measures per-position geometry of the three vectors in
the user's over-encoding question:

- wte : token embedding row               (input component)
- oe  : over-encoding residual, i.e. _compute_input_ngram_residual(idx)
        (bigram + trigram clean-table contribution added at the input)
- tf  : transformer output hidden state; reported both pre-ln_f (residual
        stream) and post-ln_f (the vector lm_head actually sees)

Metrics per side: mean/std of L2 norm per vector, and per-pair mean cos and
mean cos^2 over positions. Positions t<2 of each sequence are excluded
(degenerate bigram/trigram context there). wpe norm is recorded as auxiliary
context only.

Read-only diagnostic; training code untouched.

Usage (on the training cluster):
  python code/tools/eval_residual_geometry.py \
    --run_dir both=/path/s1v5_128_marm_both_ckpt_fixed \
    --run_dir nogram=/path/s1v5_128_marm_nogram_ckpt_fixed \
    --data_dir /path/data/tokenized \
    --out_json /path/residual_geometry.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import TokenizedShardDataset, _autocast_ctx, set_seed  # noqa: E402
from eval_logits_rank import iter_fixed_batches, load_model  # noqa: E402

PAIRS = (("wte", "oe"), ("wte", "tf"), ("oe", "tf"))


def _new_acc() -> dict:
    acc = {"count": 0}
    for name in ("wte", "oe", "tf", "tf_resid", "wpe"):
        acc[f"{name}_norm_sum"] = 0.0
        acc[f"{name}_norm_sq_sum"] = 0.0
    for a, b in PAIRS:
        acc[f"{a}__{b}_cos_sum"] = 0.0
        acc[f"{a}__{b}_cos2_sum"] = 0.0
    return acc


def _acc_update(acc: dict, vecs: dict, do_oe: bool) -> None:
    """vecs: name -> (N, C) float32, already flattened over valid positions."""
    n_pos = vecs["wte"].size(0)
    norms = {}
    for name, v in vecs.items():
        n = v.norm(dim=-1).double()
        norms[name] = n
        acc[f"{name}_norm_sum"] += n.sum().item()
        acc[f"{name}_norm_sq_sum"] += (n * n).sum().item()
    for a, b in PAIRS:
        if not do_oe and (a == "oe" or b == "oe"):
            continue
        dot = (vecs[a] * vecs[b]).sum(dim=-1).double()
        denom = (norms[a] * norms[b]).clamp_min(1e-12)
        cos = dot / denom
        acc[f"{a}__{b}_cos_sum"] += cos.sum().item()
        acc[f"{a}__{b}_cos2_sum"] += (cos * cos).sum().item()
    acc["count"] += n_pos


def _acc_finalize(acc: dict, do_oe: bool) -> dict:
    n = max(acc["count"], 1)
    out = {"n_positions": int(acc["count"])}
    for name in ("wte", "oe", "tf", "tf_resid", "wpe"):
        mean = acc[f"{name}_norm_sum"] / n
        var = max(acc[f"{name}_norm_sq_sum"] / n - mean * mean, 0.0)
        out[f"{name}_norm_mean"] = mean
        out[f"{name}_norm_std"] = var ** 0.5
    out["norm_ratio_oe_over_wte"] = out["oe_norm_mean"] / out["wte_norm_mean"]
    out["norm_ratio_oe_over_tf"] = out["oe_norm_mean"] / out["tf_norm_mean"]
    out["norm_ratio_tf_over_wte"] = out["tf_norm_mean"] / out["wte_norm_mean"]
    for a, b in PAIRS:
        if not do_oe and (a == "oe" or b == "oe"):
            continue
        out[f"cos_{a}_{b}_mean"] = acc[f"{a}__{b}_cos_sum"] / n
        out[f"cos2_{a}_{b}_mean"] = acc[f"{a}__{b}_cos2_sum"] / n
    return out


def _branch_residuals(model, idx) -> dict:
    """Per-branch split of _compute_input_ngram_residual (mirrors the
    return_injection_stats path in NanoGPT.forward; no frequency masks)."""
    prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
    prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
    out = {}
    if model.config.enable_bigram_ve:
        parts = []
        for li in sorted(model.bigram_ve_layers):
            lvs = model.bigram_ves[str(li)]
            bi = model._bigram_row_indices(li, prev_idx, idx)
            parts.append(model._pad_to_embd(
                torch.cat([lvs[k](bi[k]) for k in range(model.bigram_K)], dim=-1),
                model.bigram_table_dim))
        out["bigram"] = torch.stack(parts).sum(dim=0).float()
    if model.config.enable_trigram_ve:
        parts = []
        for li in sorted(model.trigram_ve_layers):
            lvs = model.trigram_ves[str(li)]
            ti = model._trigram_row_indices(li, prev2_idx, prev_idx, idx)
            parts.append(model._pad_to_embd(
                torch.cat([lvs[k](ti[k]) for k in range(model.trigram_K)], dim=-1),
                model.trigram_table_dim))
        out["trigram"] = torch.stack(parts).sum(dim=0).float()
    return out


def _decomp_update(acc: dict, emb, oe, delta, s, branches: dict) -> None:
    """Exact per-position triangle s = emb + oe + delta (s = ln_f input).

    emb = wte + wpe; delta = s - (emb + oe) is the net Transformer-block
    contribution. share_x = (x . s_hat) / ||s||, so emb/oe/delta shares sum
    to 1 and per-branch (bigram/trigram) shares sum to share_oe.
    """
    emb, oe, delta, s = (t.double() for t in (emb, oe, delta, s))
    sn = s.norm(dim=-1).clamp_min(1e-12)
    on = oe.norm(dim=-1).clamp_min(1e-12)
    dn = delta.norm(dim=-1).clamp_min(1e-12)
    shat = s / sn.unsqueeze(-1)
    p_emb, p_oe, p_d = ((t * shat).sum(-1) for t in (emb, oe, delta))
    c_ds = p_d / dn
    vals = {
        "delta_norm": dn, "share_emb": p_emb / sn, "share_oe": p_oe / sn,
        "share_delta": p_d / sn, "ratio_oe": on / sn, "ratio_delta": dn / sn,
        "cos_oe_delta": (oe * delta).sum(-1) / (on * dn), "cos_oe_s": p_oe / on,
        "cos_delta_s": c_ds, "cos2_delta_s": c_ds * c_ds,
    }
    for name, v in branches.items():
        v = v.double()
        vn = v.norm(dim=-1).clamp_min(1e-12)
        pv = (v * shat).sum(-1)
        vals[f"{name}_norm"] = vn
        vals[f"share_{name}"] = pv / sn
        vals[f"ratio_{name}"] = vn / sn
        vals[f"cos_{name}_s"] = pv / vn
    for k, v in vals.items():
        acc[k] = acc.get(k, 0.0) + v.sum().item()
    acc["count"] = acc.get("count", 0) + s.size(0)


def _decomp_finalize(acc: dict, do_oe: bool) -> dict:
    n = max(acc.get("count", 0), 1)
    out = {f"{k}_mean": v / n for k, v in acc.items() if k != "count"}
    if not do_oe:
        out.pop("cos_oe_delta_mean")
        out.pop("cos_oe_s_mean")
    return out


def eval_side(model, batches, device, dtype: str, do_oe: bool,
              tf_variant: str, row_chunk: int = 12) -> dict:
    """tf_variant: 'post' -> tf = ln_f output; 'pre' -> tf = ln_f input.

    The 'pre' pass additionally returns the exact triangle decomposition
    under key 'decomp'.
    """
    acc = _new_acc()
    dacc = {} if tf_variant == "pre" else None
    ctx = _autocast_ctx(dtype)
    captured = {}
    hook = model.transformer.ln_f.register_forward_hook(
        lambda m, inp, out: captured.update(inp=inp[0], out=out))
    try:
        with torch.no_grad():
            for inp, _tgt in batches:
                inp = inp.to(device)
                pos = torch.arange(0, inp.size(1), dtype=torch.long,
                                   device=inp.device).unsqueeze(0)
                for r0 in range(0, inp.size(0), row_chunk):
                    x = inp[r0:r0 + row_chunk]
                    p = pos[:, :x.size(1)]
                    wte = model.transformer.wte(x).float()
                    wpe = model.transformer.wpe(p).float().expand_as(wte)
                    if do_oe:
                        oe = model._compute_input_ngram_residual(x).float()
                    else:
                        oe = torch.zeros_like(wte)
                    with ctx:
                        _ = model(x)
                    tf_pre = captured["inp"].float()
                    tf_post = captured["out"].float()
                    tf = tf_post if tf_variant == "post" else tf_pre
                    vecs = {
                        "wte": wte[:, 2:, :].reshape(-1, wte.size(-1)),
                        "oe": oe[:, 2:, :].reshape(-1, oe.size(-1)),
                        "tf": tf[:, 2:, :].reshape(-1, tf.size(-1)),
                        "tf_resid": tf_pre[:, 2:, :].reshape(-1, tf_pre.size(-1)),
                        "wpe": wpe[:, 2:, :].reshape(-1, wpe.size(-1)),
                    }
                    _acc_update(acc, vecs, do_oe)
                    if dacc is not None:
                        emb = wte + wpe
                        delta = tf_pre - (emb + oe)
                        branches = _branch_residuals(model, x) if do_oe else {}
                        if branches:
                            bsum = torch.stack(list(branches.values())).sum(0)
                            assert torch.allclose(bsum, oe, rtol=1e-3, atol=1e-3), \
                                "branch split does not reproduce oe"
                        flat = lambda t: t[:, 2:, :].reshape(-1, t.size(-1))  # noqa: E731
                        _decomp_update(
                            dacc, flat(emb), flat(oe), flat(delta), flat(tf_pre),
                            {k: flat(v) for k, v in branches.items()})
    finally:
        hook.remove()
    out = _acc_finalize(acc, do_oe)
    if dacc is not None:
        out["decomp"] = _decomp_finalize(dacc, do_oe)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", action="append", required=True,
                    help="label=/path/to/run_dir (repeatable)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_shards", default="1")
    ap.add_argument("--val_shards", default="2,3,4,5,6,7,8,9,10,6542")
    ap.add_argument("--epoch_batches", type=int, default=337)
    ap.add_argument("--device_batch_size", type=int, default=72)
    ap.add_argument("--sequence_len", type=int, default=2048)
    ap.add_argument("--n_train_batches", type=int, default=4)
    ap.add_argument("--n_val_batches", type=int, default=4)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    train_shards = [int(x) for x in args.train_shards.split(",") if x.strip()]
    val_shards = [int(x) for x in args.val_shards.split(",") if x.strip()]

    train_ds = TokenizedShardDataset(args.data_dir, train_shards,
                                     args.sequence_len, args.device_batch_size,
                                     42, epoch_batches=args.epoch_batches)
    val_ds = TokenizedShardDataset(args.data_dir, val_shards,
                                   args.sequence_len, args.device_batch_size, 42)
    train_batches = list(iter_fixed_batches(train_ds, args.n_train_batches, device))
    val_batches = list(iter_fixed_batches(val_ds, args.n_val_batches, device))
    print(f"[geom] train batches={len(train_batches)} val batches={len(val_batches)} "
          f"shape={tuple(train_batches[0][0].shape)} device={device}")

    arms = {}
    for spec in args.run_dir:
        label, run_dir = spec.split("=", 1)
        print(f"[geom] arm={label} run_dir={run_dir}")
        model, summary = load_model(run_dir, device)
        do_oe = bool(model.config.enable_nanogpt_ngram_ve
                     and (model.config.enable_bigram_ve
                          or model.config.enable_trigram_ve)
                     and model.config.nanogpt_ngram_injection_position == "input")
        res = {
            "run_id": summary.get("run_id", os.path.basename(run_dir)),
            "run_dir": run_dir,
            "final_train_loss": summary.get("final_train_loss"),
            "final_val_loss": summary.get("final_val_loss"),
            "final_gap": summary.get("final_gap"),
            "oe_active": do_oe,
        }
        for side, batches in (("train", train_batches), ("val", val_batches)):
            for variant in ("post", "pre"):
                res[f"{side}_tf_{variant}"] = eval_side(
                    model, batches, device, args.dtype, do_oe, variant)
        r = res["val_tf_post"]
        print(f"[geom] {label} val: ||wte||={r['wte_norm_mean']:.3f} "
              f"||oe||={r['oe_norm_mean']:.3f} ||tf||={r['tf_norm_mean']:.3f} "
              f"cos2(wte,oe)={r.get('cos2_wte_oe_mean', float('nan')):.4f} "
              f"cos2(wte,tf)={r.get('cos2_wte_tf_mean', float('nan')):.4f} "
              f"cos2(oe,tf)={r.get('cos2_oe_tf_mean', float('nan')):.4f}")
        dc = res["val_tf_pre"]["decomp"]
        print(f"[geom] {label} val decomp: share emb/oe/delta="
              f"{dc['share_emb_mean']:.3f}/{dc['share_oe_mean']:.3f}/"
              f"{dc['share_delta_mean']:.3f} ratio oe/delta="
              f"{dc['ratio_oe_mean']:.3f}/{dc['ratio_delta_mean']:.3f} "
              f"||delta||={dc['delta_norm_mean']:.2f} "
              f"cos(oe,delta)={dc.get('cos_oe_delta_mean', float('nan')):.4f} "
              f"share tri/bi={dc.get('share_trigram_mean', float('nan')):.3f}/"
              f"{dc.get('share_bigram_mean', float('nan')):.3f}")
        arms[label] = res
        del model
        torch.cuda.empty_cache()

    out = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "data_dir": args.data_dir,
            "train_shards": train_shards,
            "val_shards": val_shards,
            "epoch_batches": args.epoch_batches,
            "device_batch_size": args.device_batch_size,
            "sequence_len": args.sequence_len,
            "n_train_batches": args.n_train_batches,
            "n_val_batches": args.n_val_batches,
            "dtype": args.dtype,
            "note": "Positions t>=2 only. wte = token embedding row; oe = "
                    "_compute_input_ngram_residual (bigram+trigram clean-table "
                    "input injection); tf = final hidden state, reported both "
                    "post-ln_f (tf) and pre-ln_f residual (tf_resid). "
                    "*_tf_pre.decomp = exact per-position triangle "
                    "s = (wte+wpe) + oe + delta with s = ln_f input and "
                    "delta = net Transformer-block contribution; share_x = "
                    "(x . s_hat)/||s|| (shares sum to 1). "
                    "train = head of training stream (seen ~3x by step 1000); "
                    "val = the fixed val batches used during training.",
        },
        "arms": arms,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[geom] wrote {args.out_json}")


if __name__ == "__main__":
    main()
