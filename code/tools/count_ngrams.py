#!/usr/bin/env python3
"""Count n-gram occurrences (orders 2..5) over the ClimbMix corpus.

Reproduces the production tokenization/packing convention EXACTLY:
  * tokenizer  : /data/tokenizer.pkl (tiktoken, vocab 8192, bos 8188)
  * packing    : Packer copied VERBATIM from /data/distill_prod.py (class Packer,
                 lines 106-160) -- BOS-prefixed document units packed losslessly
                 into rows of exactly --seq tokens; a shard ends with a flush()
                 partial row.
  * keys       : ngram_freq.context_keys_np (imported read-only) with the same
                 in-chunk shift rule, orders 2..5, vocab_size=8192.

Per shard we build a dict {int64 key -> int64 count} per order, then dump a
per-shard npz (keys_<n>, counts_<n>) so the pass is resumable.  A final merge
step concatenates per-shard arrays per order and computes the histogram.

Memory: only one shard's dicts are alive per worker; the merge is done per order
(one order at a time) so we never hold all orders of all shards at once.

Usage:
  python3 count_ngrams.py --mode shard --shard 0 --out /data/counts/shards
  python3 count_ngrams.py --mode run   --workers 96 --out /data/counts/shards
  python3 count_ngrams.py --mode merge --out /data/counts/shards --merged /data/counts/merged
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/data")
from ngram_freq import context_keys_np  # noqa: E402  (read-only import)

SEQ_DEFAULT = 2048
BOS_DEFAULT = 8188
VOCAB_DEFAULT = 8192
ORDERS = [2, 3, 4, 5]
THRESHOLDS = [2, 4, 8, 16, 32, 64, 128, 256, 1024]


# --------------------------------------------------------------------------- #
# Packer: copied VERBATIM from /data/distill_prod.py (class Packer).
# --------------------------------------------------------------------------- #
class Packer:
    """BOS-prefixed document units packed losslessly into --seq-token rows."""

    def __init__(self, lab_enc, lab_tb, seq, bos):
        self.enc = lab_enc
        self.tb = lab_tb
        self.seq = seq
        self.bos = bos
        self.reset()

    def reset(self):
        self.ids = []      # current row: lab token ids
        self.parts = []    # raw byte chunks; concatenation == row text
        self.rows = 0

    def feed(self, text):
        """Add one document; yield completed rows as (ids, text, byte_ends)."""
        ids = self.enc.encode(text)
        if not ids:
            return
        if len(ids) > self.seq - 1:              # a single doc longer than a row
            ids = ids[: self.seq - 1]
        raw = text.encode("utf-8")
        nb_total = int(np.sum(self.tb[np.asarray(ids, dtype=np.int64)]))
        raw = raw[:nb_total] if nb_total < len(raw) else raw
        pos = 0                                   # byte cursor into this doc
        while True:
            if not self.ids:                      # start a new unit
                self.ids.append(self.bos)
            space = self.seq - len(self.ids)
            if space <= 0:
                yield self._emit()
                continue
            take = min(len(ids), space)
            if take:
                chunk = ids[:take]
                ids = ids[take:]
                nb = int(np.sum(self.tb[np.asarray(chunk, dtype=np.int64)]))
                self.ids.extend(chunk)
                if nb:
                    self.parts.append(raw[pos: pos + nb])
                    pos += nb
            if not ids:
                return
            if len(self.ids) >= self.seq:         # row exactly full
                yield self._emit()

    def flush(self):
        """Emit a final partial row (used when a shard ends mid-row)."""
        if len(self.ids) > 1:
            return self._emit()
        return None

    def _emit(self):
        row_ids = np.asarray(self.ids, dtype=np.int64)
        be = np.cumsum(self.tb[row_ids])
        text = b"".join(self.parts).decode("utf-8", errors="ignore")
        self.rows += 1
        self.reset()
        return row_ids, text, be


def lab_token_byte_table(tok_bytes_npz, vocab):
    """Per-lab-token byte length from tok_bytes.npz (specials -> 0)."""
    tb = np.zeros(vocab + 1, dtype=np.int32)
    z = np.load(tok_bytes_npz)
    off = z["off"]
    for i in range(vocab):
        tb[i] = int(off[i + 1] - off[i])
    return tb


def count_shard(shard, args):
    """Count orders 2..5 for one shard; write per-shard npz. Returns stats."""
    import pyarrow.parquet as pq

    t0 = time.time()
    with open(args.tokenizer, "rb") as fh:
        enc = pickle.load(fh)
    tb = lab_token_byte_table(args.tok_bytes, args.vocab)

    path = os.path.join(args.raw_dir, f"shard_{shard:05d}.parquet")
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    packer = Packer(enc, tb, args.seq, args.bos)

    # per-order dict: int64 key -> int64 count
    dicts = {n: {} for n in ORDERS}
    n_rows = 0
    n_pos = 0
    for t in texts:
        for row_ids, _text, _be in packer.feed(t):
            n_rows += 1
            n_pos += row_ids.size
            for n in ORDERS:
                keys = context_keys_np(row_ids, n, args.vocab)[0]
                d = dicts[n]
                for k in keys.tolist():
                    d[k] = d.get(k, 0) + 1
    last = packer.flush()
    if last is not None:
        row_ids = last[0]
        n_rows += 1
        n_pos += row_ids.size
        for n in ORDERS:
            keys = context_keys_np(row_ids, n, args.vocab)[0]
            d = dicts[n]
            for k in keys.tolist():
                d[k] = d.get(k, 0) + 1

    os.makedirs(args.out, exist_ok=True)
    out = {"shard": np.int64(shard), "n_rows": np.int64(n_rows),
           "n_pos": np.int64(n_pos)}
    stats = {"shard": shard, "n_rows": n_rows, "n_pos": n_pos,
             "orders": {}, "secs": round(time.time() - t0, 1)}
    for n in ORDERS:
        d = dicts[n]
        keys = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
        counts = np.fromiter(d.values(), dtype=np.int64, count=len(d))
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        counts = counts[order]
        out[f"keys_{n}"] = keys
        out[f"counts_{n}"] = counts
        stats["orders"][str(n)] = {"total_occ": int(counts.sum()),
                                   "distinct": int(keys.size)}
    np.savez(os.path.join(args.out, f"shard_{shard:05d}.npz"), **out)
    with open(os.path.join(args.out, f"shard_{shard:05d}.json"), "w") as fh:
        json.dump(stats, fh)
    print(f"[shard {shard}] rows={n_rows} pos={n_pos} "
          + " ".join(f"n{n}:occ={stats['orders'][str(n)]['total_occ']}"
                     f"/dist={stats['orders'][str(n)]['distinct']}" for n in ORDERS)
          + f" secs={stats['secs']}", flush=True)
    return stats


def _shard_worker(payload):
    shard, args = payload
    try:
        return count_shard(shard, args)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"shard": shard, "error": f"{e}\n{traceback.format_exc()}"}


def run_all(args):
    from multiprocessing import Pool

    shards = list(range(args.shard_start, args.shard_end))
    todo = [s for s in shards
            if not os.path.exists(os.path.join(args.out, f"shard_{s:05d}.npz"))]
    print(f"[run] {len(shards)} shards, {len(todo)} to do, workers={args.workers}",
          flush=True)
    t0 = time.time()
    done = 0
    with Pool(processes=args.workers) as pool:
        for st in pool.imap_unordered(_shard_worker, [(s, args) for s in todo]):
            done += 1
            if "error" in st:
                print(f"[run] ERROR shard {st['shard']}: {st['error']}", flush=True)
            el = time.time() - t0
            print(f"[run] {done}/{len(todo)} done elapsed={el/60:.1f}min "
                  f"eta={el/max(done,1)*(len(todo)-done)/60:.1f}min", flush=True)
    print(f"[run] complete in {(time.time()-t0)/60:.1f}min", flush=True)


def merge(args):
    """Merge per-shard npz per order; compute histogram + distinct counts."""
    os.makedirs(args.merged, exist_ok=True)
    shard_files = sorted(f for f in os.listdir(args.out) if f.endswith(".npz"))
    print(f"[merge] {len(shard_files)} shard files", flush=True)
    summary = {"n_shards": len(shard_files), "orders": {}}
    for n in ORDERS:
        t0 = time.time()
        parts_k, parts_c = [], []
        for f in shard_files:
            z = np.load(os.path.join(args.out, f))
            parts_k.append(z[f"keys_{n}"])
            parts_c.append(z[f"counts_{n}"])
        keys = np.concatenate(parts_k)
        counts = np.concatenate(parts_c)
        del parts_k, parts_c
        # merge duplicate keys across shards
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        counts = counts[order]
        del order
        uniq, start = np.unique(keys, return_index=True)
        # sum counts per unique key
        merged = np.add.reduceat(counts, start)
        del keys, counts
        total_occ = int(merged.sum())
        distinct = int(uniq.size)
        hist = {}
        for th in THRESHOLDS:
            hist[str(th)] = int((merged >= th).sum())
        np.savez(os.path.join(args.merged, f"counts_{n}.npz"),
                 keys=uniq, counts=merged)
        summary["orders"][str(n)] = {
            "total_occ": total_occ, "distinct": distinct, "hist": hist}
        print(f"[merge] n={n} distinct={distinct} total_occ={total_occ} "
              f"hist={hist} secs={time.time()-t0:.1f}", flush=True)
        del uniq, merged
    with open(os.path.join(args.merged, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[merge] wrote", os.path.join(args.merged, "summary.json"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shard", "run", "merge"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shard_start", type=int, default=0)
    ap.add_argument("--shard_end", type=int, default=210)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", default="/data/counts/shards")
    ap.add_argument("--merged", default="/data/counts/merged")
    ap.add_argument("--raw_dir", default="/data/climbmix_raw")
    ap.add_argument("--tokenizer", default="/data/tokenizer.pkl")
    ap.add_argument("--tok_bytes", default="/data/tok_bytes.npz")
    ap.add_argument("--seq", type=int, default=SEQ_DEFAULT)
    ap.add_argument("--bos", type=int, default=BOS_DEFAULT)
    ap.add_argument("--vocab", type=int, default=VOCAB_DEFAULT)
    args = ap.parse_args()
    if args.mode == "shard":
        count_shard(args.shard, args)
    elif args.mode == "run":
        run_all(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
