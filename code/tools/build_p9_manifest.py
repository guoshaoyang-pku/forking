#!/usr/bin/env python3
"""Build the plan-9 immutable 1B data manifest (row-level permutation).

Contract (docs/plans/plan-9-moe-oe-baseline.md section 5, handoff section 6):
  - pool = all packed 2049-token rows of the 162 full-163 train shards
    (val shard 6542 is EXCLUDED from the pool and copied for val use);
  - one fixed row permutation (seed 20260917, independent of model seeds)
    is materialized as mshard_*.bin in row-permuted order, so the existing
    TokenizedShardDataset reproduces the manifest order with NO train.py
    changes and exact epoch replay;
  - nested subsets are prefixes of the same permutation, consumed with
    --train_max_tokens (token-precise);
  - manifest.json records source ids + sha256, permutation npy + sha256,
    materialized mshard sha256, tokenizer hash, exact token counts.

Usage:
  python3 code/tools/build_p9_manifest.py --src_dir data/tokenized_full163 \
      --out_dir data/manifest_full163 [--subset_tokens 1000000000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

SEQ_LEN = 2049          # packed row size (BOS-aligned chunks)
ROW_BYTES = SEQ_LEN * 2  # uint16
MSHARD_ROWS = 24000     # rows per materialized shard (mirrors source granularity)
PERM_SEED = 20260917    # fixed, independent of model seeds (documented)
VAL_SHARD = 6542


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/tokenized_full163")
    ap.add_argument("--out_dir", default="data/manifest_full163")
    ap.add_argument("--subset_tokens", type=int, default=1_000_000_000,
                    help="primary unique-token budget (main 2x2); nested "
                         "prefix stats for 100M/1B/2B are always reported")
    ap.add_argument("--tokenizer_dir",
                    default="/data2/ncpl-pathA/work/vbird_autoresearch/cache/tokenizer")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. inventory source train shards (exclude val 6542) --------------
    src_ids = []
    for fn in sorted(os.listdir(args.src_dir)):
        if not fn.startswith("shard_") or not fn.endswith(".bin"):
            continue
        sid = int(fn[len("shard_"):-len(".bin")])
        if sid == VAL_SHARD:
            continue
        src_ids.append(sid)
    src_ids.sort()
    assert src_ids, f"no source shards under {args.src_dir}"

    row_counts = {}
    src_sha = {}
    total_rows = 0
    for sid in src_ids:
        p = os.path.join(args.src_dir, f"shard_{sid:05d}.bin")
        n_rows = os.path.getsize(p) // ROW_BYTES
        assert os.path.getsize(p) % ROW_BYTES == 0, f"{p}: not row-aligned"
        row_counts[sid] = n_rows
        total_rows += n_rows
    total_tokens = total_rows * (SEQ_LEN - 1)
    print(f"[manifest] {len(src_ids)} source train shards, {total_rows:,} rows, "
          f"{total_tokens:,} train tokens")

    # ---- 2. fixed permutation over pool rows ------------------------------
    rng = np.random.default_rng(np.random.PCG64(PERM_SEED))
    perm = rng.permutation(total_rows)
    perm_path = os.path.join(args.out_dir, "permutation.npy")
    np.save(perm_path, perm)
    perm_sha = sha256_file(perm_path)
    print(f"[manifest] permutation seed {PERM_SEED} -> {perm_path} "
          f"({perm_sha[:12]}...)")

    # global row index -> (shard array offset, local row)
    offsets = {}
    off = 0
    for sid in src_ids:
        offsets[sid] = off
        off += row_counts[sid]
    off_arr = np.fromiter(offsets.values(), dtype=np.int64, count=len(offsets))

    # ---- 3. materialize mshards in permuted order -------------------------
    mms = {}
    for sid in src_ids:
        mms[sid] = np.memmap(os.path.join(args.src_dir, f"shard_{sid:05d}.bin"),
                             dtype=np.uint16, mode="r")
    n_shards_out = (total_rows + MSHARD_ROWS - 1) // MSHARD_ROWS
    mshard_meta = []
    buf = np.empty((MSHARD_ROWS, SEQ_LEN), dtype=np.uint16)
    buf_n = 0
    mi = 0
    def flush():
        nonlocal buf_n, mi
        if buf_n == 0:
            return
        name = f"shard_{mi:05d}.bin"  # dataset loader expects shard_%05d.bin
        path = os.path.join(args.out_dir, name)
        buf[:buf_n].tofile(path)
        mshard_meta.append({
            "shard": mi, "file": name, "rows": int(buf_n),
            "sha256": sha256_file(path),
            "tokens": int(buf_n * (SEQ_LEN - 1)),
        })
        print(f"[manifest] {name}: {buf_n:,} rows "
              f"({mshard_meta[-1]['sha256'][:12]}...)")
        mi += 1
        buf_n = 0

    for g in perm:
        # binary search the owning shard
        sid_i = int(np.searchsorted(off_arr, g, side="right") - 1)
        sid = src_ids[sid_i]
        local = g - offsets[sid]
        buf[buf_n] = mms[sid][local * SEQ_LEN:(local + 1) * SEQ_LEN]
        buf_n += 1
        if buf_n == MSHARD_ROWS:
            flush()
    flush()

    # ---- 4. nested subset stats (prefix property) --------------------------
    subsets = {}
    for label, tok in (("s0_pilot_100m", 100_000_000),
                       ("main_1b", 1_000_000_000),
                       ("scale_2b", 2_000_000_000)):
        rows = tok // (SEQ_LEN - 1)
        subsets[label] = {
            "tokens_budget": tok,
            "rows": int(min(rows, total_rows)),
            "exact_tokens": int(min(rows, total_rows) * (SEQ_LEN - 1)),
        }

    # ---- 5. val shard copy (6542; panel = its first 8 rows) ----------------
    val_src = os.path.join(args.src_dir, f"shard_{VAL_SHARD:05d}.bin")
    val_dst = os.path.join(args.out_dir, f"shard_{VAL_SHARD:05d}.bin")
    if not os.path.exists(val_dst):
        import shutil
        shutil.copyfile(val_src, val_dst)
    val_sha = sha256_file(val_dst)
    val_rows = os.path.getsize(val_dst) // ROW_BYTES

    # ---- 6. tokenizer hash --------------------------------------------------
    tok_sha = {}
    for fn in sorted(os.listdir(args.tokenizer_dir)):
        p = os.path.join(args.tokenizer_dir, fn)
        if os.path.isfile(p):
            tok_sha[fn] = sha256_file(p)

    # ---- 7. manifest.json ----------------------------------------------------
    manifest = {
        "name": "p9-full163-rowperm-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "row_level_note": (
            "packed .bin rows carry no document ids (upstream best-fit "
            "packing); permutation granularity is the 2049-token packed row. "
            "Cross-shard row permutation removes document order/source "
            "locality; nested subsets are exact row prefixes."),
        "perm_seed": PERM_SEED,
        "perm_file": "permutation.npy",
        "perm_sha256": perm_sha,
        "seq_len": SEQ_LEN - 1,
        "row_tokens": SEQ_LEN - 1,
        "source": {
            "train_shard_ids": src_ids,
            "train_row_counts": {str(k): v for k, v in row_counts.items()},
            "train_rows_total": total_rows,
            "train_tokens_total": total_tokens,
        },
        "materialized": {
            "mshard_rows": MSHARD_ROWS,
            "shards": mshard_meta,
        },
        "val": {
            "shard": VAL_SHARD,
            "rows": val_rows,
            "sha256": val_sha,
            "panel_rows": 8,
            "panel_tokens": 8 * (SEQ_LEN - 1),
        },
        "tokenizer_sha256": tok_sha,
        "subsets": subsets,
        "replay": ("--train_epochs N replays the identical mshard row order; "
                   "--train_max_tokens selects the nested prefix (token-exact)"),
    }
    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {mpath}")

    # ---- 8. verification -----------------------------------------------------
    # total tokens preserved
    m_tokens = sum(m["tokens"] for m in mshard_meta)
    assert m_tokens == total_tokens, (m_tokens, total_tokens)
    # spot-check: mshard 0 row 0 == source row perm[0]
    g = int(perm[0])
    sid_i = int(np.searchsorted(off_arr, g, side="right") - 1)
    sid = src_ids[sid_i]
    local = g - offsets[sid]
    src_row = np.array(
        mms[sid][local * SEQ_LEN:(local + 1) * SEQ_LEN], dtype=np.uint16)
    m0 = np.fromfile(os.path.join(args.out_dir, "shard_00000.bin"),
                     dtype=np.uint16, count=SEQ_LEN)
    assert (src_row == m0).all(), "row 0 mismatch"
    print(f"[manifest] verified: token total {m_tokens:,}, row-0 spot check OK "
          f"(source shard {sid} row {local})")
    print(f"[manifest] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
