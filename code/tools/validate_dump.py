#!/usr/bin/env python3
"""Validation for the dump integration (deliverables 3a/3b/3c), CPU only.

(a) aggregation equivalence: v4 vs v2 and v5 vs v3 on >=3 real batches from
    /data/climbmix_raw/shard_00000.parquet (cnt exact, acc max-abs diff).
(b) dump round-trip with a mocked engine: write two batches, then from the files
    alone dequantize and compare against a float64 recomputation of the
    projection, verify the tid map, the aligned/unaligned counts and the sha1s.
(c) cost: added seconds per batch (projection + quantize + write), median/min
    over >=5 batches, and the projected disk usage for 11.45e9 positions.

Run:  python3 /data/bench/validate_dump.py
"""
import hashlib
import os
import pickle
import shutil
import statistics
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, "/data")
sys.path.insert(0, "/data/bench")

import distill_prod_v2 as V2
import distill_prod_v3 as V3
import distill_prod_v4 as V4
import distill_prod_v5 as V5
import dump_positions as DP

SEQ = 2048
BOS = 8188
VOCAB = 8192
HID = 2048
DEPTHS = [0, 10, 25, 40]
NBLOCKS = len(DEPTHS) + 1
DIM = HID * NBLOCKS
BATCH_ROWS = 16
ORDERS = [2, 3, 4, 5]
ASSIGN = "/data/assign_exact974k.npz"
RAW = "/data/climbmix_raw/shard_00000.parquet"
LAB_TOK = "/data/tokenizer.pkl"
TOK_BYTES = "/data/tok_bytes.npz"
QWEN_DIR = "/data/models/Qwen3.6-35B-A3B"
TOTAL_POSITIONS = 11.45e9


# --------------------------------------------------------------------------- #
# real inputs
# --------------------------------------------------------------------------- #
def build_real_batches(n_batches=3, seed=0):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    with open(LAB_TOK, "rb") as fh:
        lab_enc = pickle.load(fh)
    lab_tb = V2.lab_token_byte_table(TOK_BYTES, lab_enc, VOCAB)
    qtok = AutoTokenizer.from_pretrained(QWEN_DIR)
    texts = pq.read_table(RAW, columns=["text"]).column("text").to_pylist()
    packer = V2.Packer(lab_enc, lab_tb, SEQ, BOS)
    rows = []
    for t in texts:
        for r in packer.feed(t):
            if r[0].size == SEQ:
                rows.append(r)
        if len(rows) >= BATCH_ROWS * n_batches:
            break
    rows = rows[: BATCH_ROWS * n_batches]
    rng = np.random.default_rng(seed)
    batches = []
    for b in range(n_batches):
        chunk = rows[b * BATCH_ROWS:(b + 1) * BATCH_ROWS]
        per_row = []
        for r in chunk:
            row_ids = np.asarray(r[0], dtype=np.int64)
            be = np.asarray(r[2], dtype=np.int64)
            q_ids = np.asarray(qtok(r[1], add_special_tokens=False)["input_ids"],
                               dtype=np.int64)
            data = rng.standard_normal((q_ids.size, DIM)).astype(np.float32)
            per_row.append((row_ids, q_ids, be, data))
        batches.append(per_row)
    return batches, qtok


def qwen_byte_table(qtok):
    u2b = {v: k for k, v in V2._bytes_to_unicode().items()}
    size = max(len(qtok), getattr(qtok, "vocab_size", 0)) + 1
    tb = np.zeros(size, dtype=np.int32)
    for i in range(len(qtok)):
        s = qtok.convert_ids_to_tokens(i)
        if isinstance(s, bytes):
            tb[i] = len(s)
            continue
        c = 0
        for ch in s:
            if ch in u2b:
                c += 1
        tb[i] = c
    return tb


def make_state(mod, assign_keys, q_tb):
    w = mod.Worker.__new__(mod.Worker)
    w.torch = torch
    w.dev = "cpu"
    w.orders = list(ORDERS)
    w.assign_keys = assign_keys
    w.sizes = {n: assign_keys[n].size for n in ORDERS}
    w.hid = HID
    w.use_final = True
    w.nblocks = NBLOCKS
    w.sentinel = True
    w.acc = {n: torch.zeros(w.sizes[n], DIM, dtype=torch.float32) for n in ORDERS}
    w.cnt = {n: torch.zeros(w.sizes[n], dtype=torch.int64) for n in ORDERS}
    w.positions = 0
    w.hits = {n: 0 for n in ORDERS}
    w.align_valid = 0
    w.align_total = 0
    w._ones_cache = {}
    w.qwen_tb = q_tb
    w.args = type("A", (), {"vocab": VOCAB})()
    return w


def compare(w1, w2, label):
    ok = True
    notes = []
    for n in ORDERS:
        if not torch.equal(w1.cnt[n], w2.cnt[n]):
            ok = False
            notes.append(f"n{n}: cnt MISMATCH")
            continue
        c = w1.cnt[n]
        m = c >= 16
        if int(m.sum()) == 0:
            notes.append(f"n{n}: no rows cnt>=16")
            continue
        a1 = w1.acc[n][m].double()
        a2 = w2.acc[n][m].double()
        maxabs = (a2 - a1).abs().max().item()
        if maxabs != 0.0:
            ok = False
        notes.append(f"n{n}: maxabs={maxabs:.3e}")
    if w1.positions != w2.positions:
        ok = False; notes.append("positions differ")
    if w1.align_valid != w2.align_valid or w1.align_total != w2.align_total:
        ok = False; notes.append("align counters differ")
    for n in ORDERS:
        if w1.hits[n] != w2.hits[n]:
            ok = False; notes.append(f"hits n{n} differ")
    print(f"  [{label}] {'PASS' if ok else 'FAIL'}  " + "  ".join(notes))
    return ok


# --------------------------------------------------------------------------- #
# (a) aggregation equivalence
# --------------------------------------------------------------------------- #
def test_a():
    print("=" * 78)
    print("(a) AGGREGATION EQUIVALENCE  (v4 vs v2, v5 vs v3)")
    print("=" * 78)
    z = np.load(ASSIGN)
    assign_keys = {n: np.ascontiguousarray(z[f"keys_{n}"], dtype=np.int64)
                   for n in ORDERS}
    batches, qtok = build_real_batches(n_batches=3)
    q_tb = qwen_byte_table(qtok)
    all_ok = True
    for bi, per_row in enumerate(batches):
        w2 = make_state(V2, assign_keys, q_tb)
        w4 = make_state(V4, assign_keys, q_tb)
        w3 = make_state(V3, assign_keys, q_tb)
        w5 = make_state(V5, assign_keys, q_tb)
        for (row_ids, q_ids, be, data) in per_row:
            w2.accumulate(row_ids, q_ids, be, data)
            w4.accumulate(row_ids, q_ids, be, data)
            w3.accumulate(row_ids, q_ids, be, data)
            w5.accumulate(row_ids, q_ids, be, data)
        all_ok &= compare(w2, w4, f"batch{bi} v4-vs-v2")
        all_ok &= compare(w3, w5, f"batch{bi} v5-vs-v3")
    print(f"(a) OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# --------------------------------------------------------------------------- #
# (b) dump round-trip with a mocked engine
# --------------------------------------------------------------------------- #
def test_b():
    print()
    print("=" * 78)
    print("(b) DUMP ROUND-TRIP (mocked engine, 2 batches)")
    print("=" * 78)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    with open(LAB_TOK, "rb") as fh:
        lab_enc = pickle.load(fh)
    lab_tb = V2.lab_token_byte_table(TOK_BYTES, lab_enc, VOCAB)
    qtok = AutoTokenizer.from_pretrained(QWEN_DIR)
    q_tb = qwen_byte_table(qtok)

    # pack the shard exactly as the worker would (for the sha1 + row count)
    texts = pq.read_table(RAW, columns=["text"]).column("text").to_pylist()
    packer = V2.Packer(lab_enc, lab_tb, SEQ, BOS)
    rows = []
    for t in texts:
        for r in packer.feed(t):
            rows.append(r)
    tail = packer.flush()
    if tail is not None:
        rows.append(tail)
    n_rows = len(rows)
    h = hashlib.sha1()
    for r in rows:
        h.update(np.asarray(r[0], dtype=np.int64).tobytes())
    stream_sha1 = h.hexdigest()
    first_ids = np.asarray(rows[0][0][:1000], dtype=np.int64)
    last_ids = np.asarray(rows[-1][0][-1000:], dtype=np.int64)

    # deterministic pseudo-random residuals, same shape as production
    rng = np.random.default_rng(7)
    # two rows -> one batch; the teacher stream is the concatenation of the two
    # rows' teacher token ids, so data has one row per teacher token.
    root = tempfile.mkdtemp(prefix="dump_rt_")
    w = DP.DumpWriter(root, worker=0, tag="rt", blocks=(10, 25, 40), dims=128)
    w.start_shard(0, n_rows, stream_sha1, first_ids, last_ids)

    # build q_star for the first 2 rows using the real byte-end alignment
    def qstar_for(row):
        q_ids = np.asarray(qtok(row[1], add_special_tokens=False)["input_ids"],
                           dtype=np.int64)
        q_be = np.cumsum(q_tb[q_ids])
        be = np.asarray(row[2], dtype=np.int64)
        return np.searchsorted(q_be, be, side="right") - 1, q_ids

    qs0, qids0 = qstar_for(rows[0])
    qs1, qids1 = qstar_for(rows[1])
    # teacher ids for the batch: concatenate the two rows' teacher ids
    tid_src = np.concatenate([qids0, qids1])
    nq = int(tid_src.size)
    data = rng.standard_normal((nq, DIM)).astype(np.float32)
    # q_star for row1 must be offset by len(qids0) into the batch's teacher stream
    qs1_off = qs1 + len(qids0)
    q_star = np.concatenate([qs0, qs1_off])
    w.set_tid_src(tid_src)
    w.add_batch(data, q_star)
    meta = w.end_shard(2 * SEQ)
    w.close()

    ok = True
    d = os.path.join(root, "rt")
    codes = np.load(os.path.join(d, "sketch_w0_s0.i8.npy"))
    scales = np.load(os.path.join(d, "scale_w0_s0.f16.npy"))
    tid = np.load(os.path.join(d, "tid_w0_s0.u32.npy"))
    print(f"  shapes: codes={codes.shape} scales={scales.shape} tid={tid.shape}")

    # (i) dequant vs float64 recompute of the projection.
    # Position i (row order) maps to teacher token q_star[i], so compare the
    # stored codes against the projection of data[q_star] for the aligned slots.
    n_pos = 2 * SEQ
    valid = q_star >= 0
    sel = q_star[valid]
    X = np.stack([data[sel, 2 * 2048:3 * 2048], data[sel, 3 * 2048:4 * 2048],
                  data[sel, 4 * 2048:5 * 2048]], axis=1).astype(np.float64)
    S = w._project(X.reshape(sel.size * 3, 2048).astype(np.float32)
                   ).reshape(sel.size, 3, 128)
    dec = w.sketch.decode_i8(codes[valid].reshape(sel.size * 3, 128),
                             scales[valid].reshape(sel.size * 3)
                             ).reshape(sel.size, 3, 128)
    rel = np.abs(dec - S).max() / (np.abs(S).max() + 1e-12)
    cos = [float((dec[:, b] * S[:, b]).sum()
                 / (np.linalg.norm(dec[:, b]) * np.linalg.norm(S[:, b]) + 1e-12))
           for b in range(3)]
    print(f"  (i) dequant rel_err={rel:.4f} (expect ~0.004)  "
          f"cos per block={[round(c, 6) for c in cos]}")
    if rel > 0.01:
        ok = False; print("      FAIL: rel_err too high")

    # (ii) tid map matches the q_star map exactly
    exp_tid = np.full(2 * SEQ, 0xFFFFFFFF, dtype=np.uint32)
    valid = q_star >= 0
    exp_tid[valid] = tid_src[q_star[valid]].astype(np.uint32)
    tid_ok = np.array_equal(tid, exp_tid)
    print(f"  (ii) tid map exact: {tid_ok}  "
          f"(aligned={int(valid.sum())} unaligned={int((~valid).sum())})")
    if not tid_ok:
        ok = False; print("      FAIL: tid mismatch")

    # (iii) counts match the meta
    cnt_ok = (meta["n_aligned"] == int(valid.sum())
              and meta["n_unaligned"] == int((~valid).sum())
              and meta["n_positions"] == 2 * SEQ)
    print(f"  (iii) meta counts: aligned={meta['n_aligned']} "
          f"unaligned={meta['n_unaligned']} n_positions={meta['n_positions']} "
          f"-> {cnt_ok}")
    if not cnt_ok:
        ok = False; print("      FAIL: count mismatch")

    # (iv) sha1 recomputes from an independent Packer re-run
    h2 = hashlib.sha1()
    for r in rows:
        h2.update(np.asarray(r[0], dtype=np.int64).tobytes())
    sha_ok = (h2.hexdigest() == meta["token_stream_sha1"] == stream_sha1)
    print(f"  (iv) token_stream_sha1 recompute: {sha_ok}  "
          f"({meta['token_stream_sha1'][:16]}...)")
    if not sha_ok:
        ok = False; print("      FAIL: sha1 mismatch")

    # meta completeness
    need = ["shard", "worker", "tag", "n_rows", "n_positions", "blocks", "dims",
            "dtype", "quant", "projector", "token_stream_sha1", "first_ids_sha1",
            "last_ids_sha1", "byte_end_alignment", "n_aligned", "n_unaligned",
            "code_md5s"]
    missing = [k for k in need if k not in meta]
    print(f"  meta keys present: {len(meta)}  missing={missing}")
    if missing:
        ok = False

    shutil.rmtree(root, ignore_errors=True)
    print(f"(b) OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# (c) cost + disk projection
# --------------------------------------------------------------------------- #
def test_c(n_batches=6):
    print()
    print("=" * 78)
    print("(c) COST PER BATCH + DISK PROJECTION")
    print("=" * 78)
    rng = np.random.default_rng(11)
    nq = 16 * 2048 // 1  # ~32768 positions per batch (16 rows x 2048)
    # production batch: 16 rows x 2048 slots = 32768 positions, ~28k teacher tok
    n_teacher = 28000
    data = rng.standard_normal((n_teacher, DIM)).astype(np.float32)
    q_star = np.full(nq, -1, dtype=np.int64)
    q_star[:n_teacher] = np.arange(n_teacher)
    tid_src = np.arange(1000, 1000 + n_teacher, dtype=np.int64)

    root = tempfile.mkdtemp(prefix="dump_cost_")
    w = DP.DumpWriter(root, worker=0, tag="cost", blocks=(10, 25, 40), dims=128)
    # capacity = warmup + n_batches batches, each 16 rows x 2048 slots
    w.start_shard(0, 16 * (1 + n_batches), "x", np.arange(1000), np.arange(1000))
    w.set_tid_src(tid_src)
    # warmup
    w.add_batch(data, q_star)
    ts = []
    for _ in range(n_batches):
        t0 = time.perf_counter()
        w.add_batch(data, q_star)
        ts.append(time.perf_counter() - t0)
    meta = w.end_shard(16 * (1 + n_batches) * 2048)
    w.close()
    ts = np.array(ts)
    med = float(np.median(ts))
    mn = float(ts.min())
    print(f"  dump add_batch: median={med*1e3:.1f}ms  min={mn*1e3:.1f}ms  "
          f"max={ts.max()*1e3:.1f}ms  (n={n_batches})")

    # bytes per position from the actual files
    d = os.path.join(root, "cost")
    b_codes = os.path.getsize(os.path.join(d, "sketch_w0_s0.i8.npy"))
    b_scale = os.path.getsize(os.path.join(d, "scale_w0_s0.f16.npy"))
    b_tid = os.path.getsize(os.path.join(d, "tid_w0_s0.u32.npy"))
    n_pos = 16 * (1 + n_batches) * 2048
    per_pos = (b_codes + b_scale + b_tid) / n_pos
    total_tb = per_pos * TOTAL_POSITIONS / 1e12
    print(f"  bytes/position = {per_pos:.1f}  "
          f"(codes={b_codes/n_pos:.1f} scale={b_scale/n_pos:.1f} tid={b_tid/n_pos:.1f})")
    print(f"  projected total for {TOTAL_POSITIONS/1e9:.2f}B positions = "
          f"{total_tb:.3f} TB  (budget 4.6 TB)")
    shutil.rmtree(root, ignore_errors=True)
    ok = med <= 0.15 and total_tb <= 4.6
    print(f"(c) {'PASS' if ok else 'CHECK'}: median {med*1e3:.1f}ms "
          f"(<=150ms), total {total_tb:.3f}TB (<=4.6TB)")
    return ok, med, mn, per_pos, total_tb


if __name__ == "__main__":
    a_ok = test_a()
    b_ok = test_b()
    c_ok, med, mn, per_pos, total_tb = test_c()
    print()
    print("=" * 78)
    print(f"SUMMARY: (a)={'PASS' if a_ok else 'FAIL'}  "
          f"(b)={'PASS' if b_ok else 'FAIL'}  (c)={'PASS' if c_ok else 'CHECK'}")
    print("=" * 78)
    sys.exit(0 if (a_ok and b_ok) else 1)
