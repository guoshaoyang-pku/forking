#!/usr/bin/env python3
"""OE (n-gram table) parameter/memory audit — plan-9 P1, handoff section 5.2.

Accounts the REAL implementation (not estimates):
  1. table params instantiated by code/train.py for the v5 clean setting
     (input injection, bigram+trigram clean tables R=2^20) and for the
     dense/nogram arm (zero tables);
  2. optimizer state: RMSProp(no momentum) = 1 fp32 buffer (exp_avg_sq) per
     table param; AdamW = 2 fp32 buffers; grads transient;
  3. device residency: tables live on GPU as fp32 nn.Embedding;
  4. touched rows: unique hashed rows actually hit by the training stream,
     from offline row-index files (data/rows/shard_*.brows.u32.bin) or from
     on-the-fly hashing of a .bin shard when the row file is absent.

Usage (CPU is fine; GPU numbers optional):
  python3 code/tools/oe_param_audit.py [--data_dir data/tokenized] \
      [--row_dir data/rows] [--shards 1] [--gpu]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

R = 2 ** 20          # clean-table rows per order (v5 SSOT)
D = 768              # n_embd
N_LAYER, VOCAB, SEQ = 8, 8192, 2048


def real_table_params(injection: str = "input") -> dict:
    """Instantiate the actual model table modules (no backbone) and count."""
    from train import Config, NanoGPT
    cfg = Config(n_layer=N_LAYER, vocab_size=VOCAB, sequence_len=SEQ,
                 nanogpt_ngram_injection_position=injection,
                 enable_bigram_ve=True, enable_trigram_ve=True,
                 bigram_clean_table=R, trigram_clean_table=R)
    model = NanoGPT(cfg)
    per_attr = {}
    for attr in ("ngram_ves", "bigram_ves", "trigram_ves", "value_embeds"):
        mod = getattr(model, attr, None)
        if isinstance(mod, torch.nn.ModuleDict) and len(mod) > 0:
            per_attr[attr] = sum(p.numel() for m in mod.values()
                                 for p in m.parameters())
    total = sum(per_attr.values())
    return {"per_attr": per_attr, "total": total}


def memory_accounting(table_params: int, dense_params: int) -> dict:
    """fp32 master weights everywhere; RMSProp: +1 buffer; AdamW: +2 buffers.

    bf16 autocast does NOT create bf16 master copies of table weights (the
    embedding lookup runs the table in fp32 and casts the gathered rows).
    """
    gb = 1024 ** 3
    return {
        "table_param_gb_fp32": table_params * 4 / gb,
        "table_rmsprop_state_gb": table_params * 4 / gb,     # exp_avg_sq
        "table_grad_gb_transient": table_params * 4 / gb,    # dense grad copy
        "table_total_gb": 3 * table_params * 4 / gb,
        "dense_param_gb_fp32": dense_params * 4 / gb,
        "dense_adamw_state_gb": 2 * dense_params * 4 / gb,
        "dense_total_gb": 4 * dense_params * 4 / gb,
    }


def touched_rows(data_dir: str, row_dir: str, shards: list, seq_len: int = 2048):
    """Unique hashed rows hit by the given shards (per order), + hit fraction."""
    out = {}
    for order, prefix in ((2, "b"), (3, "t")):
        uniq = set()
        tokens_seen = 0
        for sid in shards:
            row_path = os.path.join(row_dir, f"shard_{sid:05d}.{prefix}rows.u32.bin")
            bin_path = os.path.join(data_dir, f"shard_{sid:05d}.bin")
            if os.path.exists(row_path):
                rows = np.fromfile(row_path, dtype=np.uint32).astype(np.int64)
                uniq.update(rows.tolist())
                tokens_seen += len(rows)
            elif os.path.exists(bin_path):
                # fall back to on-the-fly hashing; formula and primes copied
                # from code/precompute_row_index.py (clean-table family 0)
                from train import _BASE_BIGRAM_PRIMES, _BASE_TRIGRAM_PRIMES
                if order == 2:
                    primes = _BASE_BIGRAM_PRIMES[0][0]
                else:
                    primes = tuple(_BASE_TRIGRAM_PRIMES[0][:3])
                toks = np.fromfile(bin_path, dtype=np.uint16).astype(np.uint64)
                chunk = seq_len + 1
                n_chunks = len(toks) // chunk
                toks = toks[: n_chunks * chunk].reshape(n_chunks, chunk)
                prev1 = toks[:, :-1]
                if order == 2:
                    h = ((prev1 * np.uint64(primes[0]))
                         ^ (toks[:, 1:] * np.uint64(primes[1]))) % np.uint64(R)
                else:
                    prev2 = toks[:, :-2]
                    h = ((prev2 * np.uint64(primes[0]))
                         ^ (prev1[:, 1:] * np.uint64(primes[1]))
                         ^ (toks[:, 2:] * np.uint64(primes[2]))) % np.uint64(R)
                uniq.update(h.reshape(-1).astype(np.int64).tolist())
                tokens_seen += h.size
            else:
                print(f"[audit] missing both row and bin for shard {sid}, skipped")
        out[f"{order}gram"] = {
            "tokens_seen": tokens_seen,
            "unique_rows": len(uniq),
            "table_rows": R,
            "occupancy": len(uniq) / R,
            "rows_per_token": len(uniq) / max(1, tokens_seen),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/tokenized")
    ap.add_argument("--row_dir", default="data/rows")
    ap.add_argument("--shards", default="1",
                    help="comma list of shard ids for touched-rows stats")
    ap.add_argument("--gpu", action="store_true",
                    help="report torch.cuda peak after one embedding gather")
    args = ap.parse_args()

    print("== OE audit (real implementation) ==")
    tp = real_table_params()
    print(f"table params (modules instantiated from Config): {tp}")
    assert tp["total"] == 2 * R * D, \
        f"expected 2 x 2^20 x 768 = {2*R*D:,}, got {tp['total']:,}"

    # dense-side params for the same backbone (from the ledger formula)
    d = D
    ln = 2 * d
    per_layer_attn = 4 * d * d + 4 * d + ln
    per_layer_ffn = 2 * d * (4 * d) + d + 4 * d
    non_ffn = N_LAYER * per_layer_attn + 2 * ln
    embeddings = VOCAB * d + SEQ * d
    dense_params = non_ffn + N_LAYER * per_layer_ffn + embeddings

    mem = memory_accounting(tp["total"], dense_params)
    print("\n== memory (fp32 master + optimizer state) ==")
    for k, v in mem.items():
        print(f"  {k:32s} {v:8.2f} GB")

    print("\n== touched rows (training stream occupancy) ==")
    shards = [int(x) for x in args.shards.split(",") if x.strip()]
    tr = touched_rows(args.data_dir, args.row_dir, shards)
    for order, st in tr.items():
        print(f"  {order}: tokens {st['tokens_seen']:,} | unique rows "
              f"{st['unique_rows']:,} / {st['table_rows']:,} "
              f"({100*st['occupancy']:.1f}% occupancy) | "
              f"{st['rows_per_token']:.3f} unique-rows/token")

    if args.gpu and torch.cuda.is_available():
        tab = torch.nn.Embedding(R, D).cuda()
        rows = torch.randint(0, R, (72, 2048), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        _ = tab(rows)
        print(f"\n== gpu: one 72x2048 gather peak "
              f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB "
              f"(table itself {R*D*4/1024**3:.2f} GB) ==")

    print("\naudit done.")


if __name__ == "__main__":
    main()
