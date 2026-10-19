#!/usr/bin/env python3
"""W2 residual-dump (user directive 2026-10-19, route (b)): per-context mean
residual statistics at candidate OE injection layers, from a trained nogram
teacher backbone.

For each injection layer l in --layers (0 = after wte+wpe, l>=1 = after block
l-1's output, i.e. the point where a layer-l OE table would be added), and
each order n in 2..5, accumulate over ONE full pass of the train pool:
    sum_n[key]  += h_l(t)   (fp32, residual at the injection point)
    cnt_n[key]  += 1
restricted to the f>=256 selected context set (same as w1_common.load_selected,
bit-identical key packing via NanoGPT._ngram_context_keys_torch).

Output: <out_dir>/resid_L<l>.npz with keys_<n> (int64, sorted), sum_<n>
(K_n, n_embd) fp32, cnt_<n> int64, plus meta_json (ckpt/layers/coverage).
The cluster tool (w2_cluster_residuals.py) divides sum/cnt.

CLI:
  python3 w2_dump_residuals.py <ckpt.pt> <freq_index.npz> <out_dir> \
      --layers 0,2,5 [--batch 72] [--max_batches 0]
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import Config, NanoGPT, TokenizedShardDataset  # noqa: E402

ORDERS = (2, 3, 4, 5)
FMIN = 256


def load_selected(index_path):
    z = np.load(index_path)
    sel = {}
    for n in ORDERS:
        c = z[f"counts_{n}"].astype(np.int64)
        m = c >= FMIN
        k = np.ascontiguousarray(z[f"keys_{n}"][m])
        o = np.argsort(k)
        sel[n] = (k[o], c[m][o])
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("index_path")
    ap.add_argument("out_dir")
    ap.add_argument("--layers", default="0,2,5")
    ap.add_argument("--data_dir", default="data/tokenized")
    ap.add_argument("--train_shards", default="")
    ap.add_argument("--shards_file", default="data/train_shards_1b.txt")
    ap.add_argument("--batch", type=int, default=72)
    ap.add_argument("--max_batches", type=int, default=0,
                    help="0 = one full epoch")
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--sequence_len", type=int, default=2048)
    args = ap.parse_args()

    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    assert all(0 <= l <= args.n_layer for l in layers), layers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(
        data_dir=args.data_dir, out_dir=args.out_dir,
        vocab_size=args.vocab_size, sequence_len=args.sequence_len,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        device_batch_size=args.batch,
        enable_unigram_ve=False, enable_bigram_ve=False, enable_trigram_ve=False,
    )
    model = NanoGPT(cfg)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:4]}"
    assert not missing, f"missing keys: {missing[:4]}"
    model.to(device).eval()

    sel = load_selected(args.index_path)
    sel_gpu = {n: torch.from_numpy(k).to(device) for n, (k, _) in sel.items()}
    sums = {l: {n: torch.zeros(k.size, cfg.n_embd, dtype=torch.float32,
                               device=device)
                for n, (k, _) in sel.items()} for l in layers}
    cnts = {l: {n: torch.zeros(k.size, dtype=torch.int64, device=device)
                for n, (k, _) in sel.items()} for l in layers}

    shards = args.train_shards or open(args.shards_file).read().strip()
    shard_ids = [int(s) for s in shards.replace(";", ",").split(",") if s.strip()]
    ds = TokenizedShardDataset(args.data_dir, shard_ids, args.sequence_len,
                               args.batch, 42)

    def capture(l, x):
        xf = x.reshape(-1, cfg.n_embd).float()
        for n in ORDERS:
            keys = model._ngram_context_keys_torch(idx, n).reshape(-1)
            table = sel_gpu[n]
            pos = torch.searchsorted(table, keys)
            pos_c = pos.clamp(max=table.numel() - 1)
            valid = table[pos_c] == keys
            p = pos_c[valid]
            sums[l][n].index_add_(0, p, xf[valid])
            cnts[l][n].index_add_(0, p, torch.ones_like(p))

    t0 = time.time()
    nb = 0
    it = ds.iter_batches(device)
    while True:
        idx, _ = next(it)
        if args.max_batches and nb >= args.max_batches:
            break
        if not args.max_batches and ds._epoch >= 1:
            break
        nb += 1
        with torch.no_grad(), torch.autocast(device_type="cuda",
                                             dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            B, T = idx.size()
            pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
            x = model.transformer.wte(idx)
            x = x + model.transformer.wpe(pos) if cfg.pos_type == "learned" else x
            if 0 in layers:
                capture(0, x)
            x = model.transformer.drop(x)
            for i, block in enumerate(model.transformer.h):
                x = block(x)
                if (i + 1) in layers:
                    capture(i + 1, x)
        if nb % 500 == 0:
            print(f"[dump] {nb} batches, {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for l in layers:
        out = {}
        for n in ORDERS:
            k, _ = sel[n]
            out[f"keys_{n}"] = k
            out[f"sum_{n}"] = sums[l][n].cpu().numpy()
            out[f"cnt_{n}"] = cnts[l][n].cpu().numpy()
        cov = {n: float((cnts[l][n] > 0).float().mean().item()) for n in ORDERS}
        out["meta_json"] = np.array(json.dumps(dict(
            ckpt=args.ckpt, layer=l, layers=layers, batches=nb,
            shards=shard_ids, fmin=FMIN, coverage=cov,
            elapsed_s=round(time.time() - t0, 1))))
        path = os.path.join(args.out_dir, f"resid_L{l}.npz")
        np.savez(path, **out)
        print(f"[dump] wrote {path} coverage={cov}", flush=True)


if __name__ == "__main__":
    main()
