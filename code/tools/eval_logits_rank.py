"""Offline logits rank-distribution eval for trained ngram-gap-lab runs.

Loads final_model.pt from completed run dirs, rebuilds the exact training
model from summary.json, runs the model on the SAME fixed train/val batches
used during training, and records:

- mean predictive probability as a function of logits rank (rank-probability
  curve; shows how concentrated the output distribution is),
- full histogram of the true next token's logit rank,
- entropy, NLL / perplexity, and top-k probability mass.

Training is untouched; this is a read-only diagnostic on saved checkpoints.
Output is one JSON consumed by docs/plot_scripts/plot_marm_logits_rank.py.

Usage (on the training cluster):
  python code/tools/eval_logits_rank.py \
    --run_dir nogram=/path/s1v5_128_marm_nogram_fixed \
    --run_dir bigram=/path/s1v5_128_marm_bigram_fixed \
    --run_dir trigram=/path/s1v5_128_marm_trigram_fixed \
    --run_dir both=/path/s1v5_128_marm_both_fixed \
    --data_dir /path/data/tokenized \
    --out_json /path/logits_rank_stats.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import Config, NanoGPT, TokenizedShardDataset, _autocast_ctx, set_seed  # noqa: E402


def cfg_from_summary(sc: dict) -> Config:
    """Rebuild the training Config from a run's summary.json config section."""
    return Config(
        vocab_size=sc.get("vocab_size", 8192),
        n_layer=sc.get("n_layer", 8),
        n_head=sc.get("n_head", 6),
        n_embd=sc.get("n_embd", 768),
        sequence_len=sc.get("sequence_len", 2048),
        dropout=sc.get("dropout", 0.0),
        bias=sc.get("bias", True),
        enable_nanogpt_ngram_ve=sc.get("enable_nanogpt_ngram_ve", True),
        enable_unigram_ve=sc.get("enable_unigram_ve", False),
        enable_bigram_ve=sc.get("enable_bigram_ve", True),
        enable_trigram_ve=sc.get("enable_trigram_ve", False),
        enable_fourgram_ve=sc.get("enable_fourgram_ve", False),
        nanogpt_ngram_injection_position=sc.get(
            "nanogpt_ngram_injection_position",
            sc.get("injection_position", "input")),
        nanogpt_inject_layers=sc.get("nanogpt_inject_layers", ""),
        bigram_clean_table=sc.get("bigram_clean_table", 0),
        trigram_clean_table=sc.get("trigram_clean_table", 0),
        bigram_table_dim=sc.get("bigram_table_dim", 0),
        trigram_table_dim=sc.get("trigram_table_dim", 0),
        bigram_perfect_map=sc.get("bigram_perfect_map", "") or "",
        bigram_single_layer=sc.get("bigram_single_layer", False),
        table_mult=sc.get("table_mult", 64),
        seed=sc.get("seed", 42),
    )


def load_model(run_dir: str, device: torch.device) -> NanoGPT:
    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    cfg = cfg_from_summary(summary["config"])
    model = NanoGPT(cfg)
    sd_path = os.path.join(run_dir, "final_model.pt")
    state = torch.load(sd_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, summary


def iter_fixed_batches(ds: TokenizedShardDataset, n_batches: int, device):
    it = ds.iter_batches(device)
    for _ in range(n_batches):
        yield next(it)


def eval_side(model, batches, device, dtype: str, row_chunk: int = 12,
              topk: int = 0) -> dict:
    """Accumulate rank statistics over the given (inp, tgt) batches.

    With topk > 0, additionally returns per-position arrays (under
    "per_position") for offline analysis: context token ids (t-2,t-1,t),
    true next token, top-k ids/probabilities, true-token rank/logit/prob,
    entropy and margin. Positions are flattened (batch, seq) in batch order,
    which is identical across arms/epochs because the batches are fixed.
    """
    vocab = model.config.vocab_size
    prob_sum = np.zeros(vocab, dtype=np.float64)      # summed over positions
    true_rank_hist = np.zeros(vocab, dtype=np.int64)  # rank of true token
    ent_sum = nll_sum = top1_sum = top10_sum = top100_sum = margin_sum = 0.0
    n_pos = 0
    nll_per_batch = []
    store = topk > 0
    if store:
        pp = {k: [] for k in ("ctx3", "true", "topk_ids", "topk_probs",
                              "true_rank", "true_prob", "entropy", "margin")}

    ctx = _autocast_ctx(dtype)
    with torch.no_grad():
        for bi, (inp, tgt) in enumerate(batches):
            inp, tgt = inp.to(device), tgt.to(device)
            batch_nll = 0.0
            batch_pos = 0
            for r0 in range(0, inp.size(0), row_chunk):
                x = inp[r0:r0 + row_chunk]
                y = tgt[r0:r0 + row_chunk]
                with ctx:
                    logits = model(x)  # (b, T, V) float32
                logp = F.log_softmax(logits, dim=-1)
                probs = logp.exp()
                b, t, _ = logits.shape

                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                prob_sum += sorted_probs.sum(dim=(0, 1)).double().cpu().numpy()

                true_logit = logits.gather(-1, y.unsqueeze(-1))
                true_rank = (logits > true_logit).sum(dim=-1)
                true_rank_hist += torch.bincount(
                    true_rank.flatten().cpu(), minlength=vocab).numpy()

                ent = -(probs * logp).sum(dim=-1)
                nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                top1 = probs.max(dim=-1).values
                top2 = sorted_probs[..., 1]
                margin = top1 - top2
                top10 = sorted_probs[..., :10].sum(dim=-1)
                top100 = sorted_probs[..., :100].sum(dim=-1)

                if store:
                    # Per-position records for label position t (y[t] = x[t+1]):
                    # context = (x[t-2], x[t-1], x[t]); valid for t >= 2, so
                    # the first two positions of each sequence are dropped and
                    # arrays are indexed by context-start j = t-2.
                    ctx_full = torch.stack(
                        (x[:, :-2], x[:, 1:-1], x[:, 2:]), dim=-1)  # (b,T-2,3)
                    pp["ctx3"].append(ctx_full.reshape(-1, 3).cpu().numpy())
                    pp["true"].append(y[:, 2:].reshape(-1).cpu().numpy())
                    pp["topk_ids"].append(
                        sorted_idx[:, 2:, :topk].reshape(-1, topk).to(torch.int32).cpu().numpy())
                    pp["topk_probs"].append(
                        sorted_probs[:, 2:, :topk].reshape(-1, topk).to(torch.float16).cpu().numpy())
                    pp["true_rank"].append(
                        true_rank[:, 2:].reshape(-1).to(torch.int32).cpu().numpy())
                    pp["true_prob"].append(
                        probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)[:, 2:].reshape(-1).to(torch.float16).cpu().numpy())
                    pp["entropy"].append(
                        ent[:, 2:].reshape(-1).to(torch.float32).cpu().numpy())
                    pp["margin"].append(
                        margin[:, 2:].reshape(-1).to(torch.float32).cpu().numpy())

                ent_sum += ent.sum().item()
                nll_sum += nll.sum().item()
                top1_sum += top1.sum().item()
                margin_sum += margin.sum().item()
                top10_sum += top10.sum().item()
                top100_sum += top100.sum().item()
                batch_nll += nll.sum().item()
                batch_pos += b * t
            n_pos += batch_pos
            nll_per_batch.append(batch_nll / batch_pos)

    p_at_rank = (prob_sum / n_pos).tolist()
    out = {
        "n_positions": int(n_pos),
        "p_at_rank": [float(v) for v in p_at_rank],
        "true_rank_hist": [int(v) for v in true_rank_hist],
        "entropy_mean": ent_sum / n_pos,
        "nll_mean": nll_sum / n_pos,
        "ppl": float(np.exp(nll_sum / n_pos)),
        "top1_mean": top1_sum / n_pos,
        "margin_mean": margin_sum / n_pos,
        "top10_mass": top10_sum / n_pos,
        "top100_mass": top100_sum / n_pos,
        "nll_per_batch": nll_per_batch,
    }
    if store:
        # NOTE: aggregate stats above cover ALL positions (b*T); the stored
        # per-position arrays drop the first two positions of each sequence
        # (no (t-2,t-1) context there), so their counts differ by 2*b per
        # batch. Analysis code should use the per-position arrays self-consistently.
        out["per_position"] = {
            "topk": int(topk),
            **{k: np.concatenate(v) for k, v in pp.items()},
        }
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
    ap.add_argument("--save_topk", type=int, default=0,
                    help="if >0, also store per-position top-k logits data "
                         "as <out_json_stem>_<label>_<side>.npz next to the "
                         "output JSON (for offline sharpening/example analysis)")
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
    print(f"[eval] train batches={len(train_batches)} val batches={len(val_batches)} "
          f"shape={tuple(train_batches[0][0].shape)} device={device}")

    arms = {}
    for spec in args.run_dir:
        label, run_dir = spec.split("=", 1)
        print(f"[eval] arm={label} run_dir={run_dir}")
        model, summary = load_model(run_dir, device)
        res = {
            "run_id": summary.get("run_id", os.path.basename(run_dir)),
            "run_dir": run_dir,
            "final_train_loss": summary.get("final_train_loss"),
            "final_val_loss": summary.get("final_val_loss"),
            "final_gap": summary.get("final_gap"),
            "n_params": summary.get("n_params"),
        }
        res["train"] = eval_side(model, train_batches, device, args.dtype,
                                 topk=args.save_topk)
        res["val"] = eval_side(model, val_batches, device, args.dtype,
                               topk=args.save_topk)
        if args.save_topk > 0:
            stem = os.path.splitext(os.path.abspath(args.out_json))[0]
            for side in ("train", "val"):
                npz_path = f"{stem}_{label}_{side}.npz"
                np.savez_compressed(npz_path, **res[side]["per_position"])
                res[side]["per_position_npz"] = os.path.basename(npz_path)
                res[side].pop("per_position")
                print(f"[eval] saved per-position arrays -> {npz_path}")
        arms[label] = res
        print(f"[eval] {label}: train ppl={res['train']['ppl']:.3f} "
              f"val ppl={res['val']['ppl']:.3f} "
              f"(summary val_loss={res['final_val_loss']})")
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
            "note": "train batches = head of the training stream (seen ~3x by "
                    "step 1000); val batches = the fixed val batches used "
                    "during training (shards " + args.val_shards + ")",
        },
        "arms": arms,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"[eval] wrote {args.out_json}")


if __name__ == "__main__":
    main()
