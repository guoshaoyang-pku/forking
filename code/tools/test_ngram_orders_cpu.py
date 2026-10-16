#!/usr/bin/env python3
"""CPU acceptance tests for the v2 n-gram order sweep (n = 2..8).

Run:  python3 code/tools/test_ngram_orders_cpu.py

  T1  context_keys_np(n=2,3) == compute_context_keys (bitwise)
  T2  MultiOrderFrequencyIndex counts == brute-force tuple counting (n=2..8,
      chunk semantics; exact-pack and hashed key paths)
  T3  generic order-n rows == independent numpy hash; rows in [0, R)
  T4  legacy model (no --ngram_orders) bitwise equal to the HEAD~ train.py model;
      adding orders >= 4 leaves backbone/bigram/trigram init untouched
  T5  generic tables are in the RMSProp group, receive updates, no weight decay
  T6  process_job: token payloads == legacy key payloads for bigram/trigram
  T7  end-to-end tiny train.py run with --ngram_orders 2,3,4,5 + multi index +
      2 diag workers: every order logged in freq_bin/exact_freq
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

import diag_worker  # noqa: E402
from ngram_freq import (MultiOrderFrequencyIndex, compute_context_keys,  # noqa: E402
                        context_keys_np, load_frequency_index)
from train import Config, MixedOptimizer, NanoGPT, ngram_hash_constants  # noqa: E402

SEQ = 64
VOCAB = 8192
R = 997


def repetitive_tokens(rng, n_chunks):
    """Few distinct ids spread over the 8192 vocab so high orders repeat."""
    alphabet = rng.choice(VOCAB, size=12, replace=False)
    return alphabet[rng.integers(0, len(alphabet), size=(n_chunks, SEQ + 1))].astype(np.uint16)


def brute_counts(chunks, n):
    c = collections.Counter()
    T = SEQ
    for row in chunks:
        x = [int(v) for v in row[:T]]
        for t in range(T):
            c[tuple(x[t - k] if t >= k else x[t] for k in range(n))] += 1
    return c


def make_model(orders=(), R_extra=R, seed=42):
    cfg = Config(vocab_size=VOCAB, n_layer=2, n_head=2, n_embd=32, sequence_len=SEQ,
                 enable_bigram_ve=True, enable_trigram_ve=True,
                 bigram_clean_table=R, trigram_clean_table=R,
                 ngram_extra_orders=tuple(orders),
                 ngram_clean_table=R_extra if orders else 0)
    torch.manual_seed(seed)
    m = NanoGPT(cfg)
    m.init_weights()
    return m


def t1():
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.integers(0, VOCAB, size=(3, SEQ)))
    b, t = compute_context_keys(x, VOCAB)
    assert np.array_equal(context_keys_np(x.numpy(), 2, VOCAB), b.numpy())
    assert np.array_equal(context_keys_np(x.numpy(), 3, VOCAB), t.numpy())
    print("T1 ok")


def t2(tmp):
    rng = np.random.default_rng(2)
    chunks = repetitive_tokens(rng, 30)
    chunks.tofile(os.path.join(tmp, "shard_00001.bin"))
    idx = MultiOrderFrequencyIndex.build_from_chunks(tmp, [1], VOCAB, range(2, 9),
                                                     chunk_size=SEQ + 1, log=lambda *_: None)
    path = os.path.join(tmp, "multi.npz")
    idx.save(path)
    idx = load_frequency_index(path)
    assert idx.is_multi_order and idx.branches[0] == "bigram" and idx.branches[-1] == "8gram"
    x = chunks[:, :SEQ].astype(np.int64)
    for n in range(2, 9):
        ref = brute_counts(chunks, n)
        br = idx.branches[n - 2]
        assert idx.n_distinct(br) == len(ref), (n, idx.n_distinct(br), len(ref))
        keys = idx.keys_for(br, x)
        got = idx.hit_count_numpy(br, keys)
        T = SEQ
        exp = np.array([ref[tuple(int(row[t - k]) if t >= k else int(row[t]) for k in range(n))]
                        for row in x for t in range(T)])
        assert np.array_equal(got, exp), n
        novel = idx.hit_count_numpy(br, keys + 1)  # perturbed keys: mostly novel
        assert (novel == 0).mean() > 0.5
    print("T2 ok (exact-pack n<=4, hashed n>=5)")
    return path


def t3():
    m = make_model(orders=(4, 6, 8))
    rng = np.random.default_rng(3)
    x = torch.from_numpy(rng.integers(0, VOCAB, size=(2, SEQ)))
    xn = x.numpy()
    for n in (4, 6, 8):
        c = ngram_hash_constants(n)
        ref = np.zeros_like(xn)
        for k in range(n):
            sh = xn if k == 0 else np.concatenate([xn[:, :k], xn[:, :-k]], axis=1)
            ref ^= sh * c[k]
        ref %= R
        got = m._ngram_row_indices(n, x).numpy()
        assert np.array_equal(got, ref) and got.min() >= 0 and got.max() < R, n
    assert len({c for n in (4, 6, 8) for c in ngram_hash_constants(n)}) > 3
    print("T3 ok")


def t4():
    # HEAD~ model (before the v2 change) vs current legacy path
    import subprocess as sp
    src = sp.run(["git", "-C", CODE, "show", "49a543d:code/train.py"],
                 capture_output=True, text=True, check=True).stdout
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "train_old.py")
        open(p, "w").write(src)
        spec = importlib.util.spec_from_file_location("train_old", p)
        old = importlib.util.module_from_spec(spec)
        sys.modules["train_old"] = old
        spec.loader.exec_module(old)
    cfg_old = old.Config(vocab_size=VOCAB, n_layer=2, n_head=2, n_embd=32, sequence_len=SEQ,
                         bigram_clean_table=R, trigram_clean_table=R)
    torch.manual_seed(42)
    mo = old.NanoGPT(cfg_old)
    mo.init_weights()
    mn = make_model()
    so, sn = mo.state_dict(), mn.state_dict()
    assert so.keys() == sn.keys() and all(torch.equal(so[k], sn[k]) for k in so)
    rng = np.random.default_rng(4)
    x = torch.from_numpy(rng.integers(0, VOCAB, size=(2, SEQ)))
    y = torch.from_numpy(rng.integers(0, VOCAB, size=(2, SEQ)))
    assert torch.equal(mo(x, targets=y), mn(x, targets=y))
    me = make_model(orders=(4, 5))
    se = me.state_dict()
    assert all(torch.equal(sn[k], se[k]) for k in sn)
    assert {k for k in se if k not in sn} == {"ngram_ves.4.weight", "ngram_ves.5.weight"}
    print("T4 ok (legacy bitwise; extra orders leave other init untouched)")


def t5():
    m = make_model(orders=(4, 5))
    opt = MixedOptimizer(m, lr=1e-3, ngram_betas=(0.0, 0.99), adam_betas=(0.8, 0.95),
                         weight_decay=0.1, table_lr_scale=128.0, table_betas=(0.0, 0.99))
    names = {n for n, _ in opt.ngram_params}
    assert {"ngram_ves.4.weight", "ngram_ves.5.weight"} <= names
    assert not any("ngram_ves" in n for n, _ in opt.adam_params)
    rng = np.random.default_rng(5)
    x = torch.from_numpy(rng.integers(0, VOCAB, size=(2, SEQ)))
    y = torch.from_numpy(rng.integers(0, VOCAB, size=(2, SEQ)))
    before = m.ngram_ves["5"].weight.detach().clone()
    m(x, targets=y).backward()
    opt.step()
    rows = m._ngram_row_indices(5, x).unique()
    changed = (m.ngram_ves["5"].weight != before).any(dim=1)
    assert changed[rows].all() and changed.sum() == rows.numel()
    print("T5 ok")


def t6(path):
    idx = load_frequency_index(path)
    rng = np.random.default_rng(6)
    toks = torch.from_numpy(repetitive_tokens(rng, 4)[:, :SEQ].astype(np.int64))
    ptl = rng.random(toks.numel())
    b, t = compute_context_keys(toks, VOCAB)
    legacy = [(b.numpy().ravel(), t.numpy().ravel(), ptl)]
    tokp = [(toks.numpy(), ptl)]
    ex_l, fb_l = diag_worker.process_job(idx, VOCAB, legacy, legacy, legacy, legacy[0])
    ex_t, fb_t = diag_worker.process_job(idx, VOCAB, tokp, tokp, tokp, tokp[0])
    for br in ("bigram", "trigram"):
        for side in ("train", "val"):
            assert json.dumps(ex_l[side][br], sort_keys=True) == json.dumps(ex_t[side][br], sort_keys=True)
            assert fb_l[side][br] == fb_t[side][br]
    assert set(fb_t["val"]) == set(idx.branches)
    print("T6 ok")


def t7(tmp):
    rng = np.random.default_rng(7)
    data = os.path.join(tmp, "e2e")
    os.makedirs(data)
    repetitive_tokens(rng, 64).tofile(os.path.join(data, "shard_00001.bin"))
    rng.integers(0, VOCAB, size=(64, SEQ + 1)).astype(np.uint16).tofile(
        os.path.join(data, "shard_00002.bin"))
    idx_path = os.path.join(data, "multi.npz")
    MultiOrderFrequencyIndex.build_from_chunks(data, [1], VOCAB, range(2, 9), SEQ + 1,
                                               log=lambda *_: None).save(idx_path)
    out = os.path.join(tmp, "runs")
    cmd = [sys.executable, os.path.join(CODE, "train.py"), "--run_id", "e2e",
           "--out_dir", out, "--data_dir", data, "--train_shards", "1", "--val_shards", "2",
           "--steps", "6", "--dtype", "fp32", "--ngram_orders", "2,3,4,5",
           "--bigram_clean_table", str(R), "--trigram_clean_table", str(R),
           "--ngram_clean_table", str(R), "--n_layer", "2", "--n_head", "2", "--n_embd", "32",
           "--sequence_len", str(SEQ), "--device_batch_size", "4",
           "--total_batch_size", str(4 * SEQ), "--lr", "0.0006",
           "--table_lr_scale", "128", "--table_betas", "0.0,0.99", "--val_interval", "2",
           "--val_batches", "2", "--freq_eval_batches", "2", "--freq_index", idx_path,
           "--diag_workers", "2", "--table_norm_interval", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=CODE)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rd = os.path.join(out, "e2e")
    fb = [json.loads(line) for line in open(os.path.join(rd, "freq_bin_loss.jsonl"))]
    ex = [json.loads(line) for line in open(os.path.join(rd, "exact_freq_loss.jsonl"))]
    want = {"bigram", "trigram"} | {f"{n}gram" for n in range(4, 9)}
    assert sorted(r["step"] for r in fb) == [2, 4, 6], [r["step"] for r in fb]
    assert all(set(r["val"]) == want and set(r["train"]) == want for r in fb)
    assert all(set(r["val"]) == want for r in ex)
    summ = json.load(open(os.path.join(rd, "summary.json")))
    assert summ["ngram_orders"] == [2, 3, 4, 5] and summ["config"]["ngram_extra_orders"] == [4, 5]
    tn = [json.loads(line) for line in open(os.path.join(rd, "table_norm.jsonl"))]
    assert "5gram.table_0.rms" in tn[-1] and "bigram.layer_01.table_0.rms" in tn[-1]
    print("T7 ok (e2e: 7 orders logged, 2 diag workers)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        t1()
        path = t2(tmp)
        t3()
        t4()
        t5()
        t6(path)
        t7(tmp)
    print("ALL PASS")
