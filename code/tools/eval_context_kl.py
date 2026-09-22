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
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
        info = model_new.setup_context_eval_sets(freq_index, args.frac, args.seed)

        kl_rn, kl_nr, memb_all = [], [], []
        with torch.no_grad():
            for inp, tgt in val_batches:
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
        kl_rn = np.concatenate(kl_rn)
        kl_nr = np.concatenate(kl_nr)
        memb = np.concatenate(memb_all)

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
        results[label] = out
        a = out["sets"]["A"]
        print(f"[kl] {label}: A n={a['n']} KL(ref||new) mean={a['kl_ref_new_mean']:.4f} "
              f"median={a['kl_ref_new_median']:.4f}")
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
            "dtype": args.dtype,
            "note": "Full-vocab exact KL per validation position over the fixed "
                    "val batches; A/B from train.py setup_context_eval_sets "
                    "(2% of distinct trigram types, seed 42).",
        },
        "pairs": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[kl] wrote {args.out_json}")


if __name__ == "__main__":
    main()
