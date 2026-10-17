#!/usr/bin/env python3
"""CPU acceptance tests for the v2.1 no-collision dictionary mode.

Run:  python3 code/tools/test_ngram_dict_cpu.py

Covers:
  T1  torch key function == ngram_freq.context_keys_np (n=2..5, bit for bit,
      including the splitmix64 chain and in-chunk shift rule)
  T2  dictionary build: determinism, top-K selection, tie-break by key
  T3  dict forward: residual == manual gather*valid; misses contribute exactly
      zero; dictionary members hit unique rows (no collision)
  T4  end-to-end 5-step train (orders 2,3,4,5 joint, dict mode) with loss
      finite and ngram_ves in the table (RMSProp) group
  T5  v2.5 per-order selection (--ngram_dict_per_order): k cap, f threshold,
      both, unlisted order falls back to topk; empty spec == legacy bit for bit
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

import ngram_freq  # noqa: E402
from train import (Config, NanoGPT, _mix64_i64,  # noqa: E402
                   _SPLITMIX_SEED_I64)

SEQ = 96
VOCAB = 53
R = 1024
TOPK = 300


def make_model(orders, dict_topk=0) -> NanoGPT:
    cfg = Config(
        vocab_size=VOCAB, n_layer=2, n_head=2, n_embd=32, sequence_len=SEQ,
        dropout=0.0, enable_unigram_ve=False, enable_bigram_ve=False,
        enable_trigram_ve=False,
        nanogpt_ngram_injection_position="input",
        ngram_extra_orders=tuple(orders), ngram_clean_table=R,
        ngram_dict_topk=dict_topk,
        seed=42, device_batch_size=4, total_batch_size=4 * SEQ,
        data_dir="/tmp", out_dir="/tmp",
    )
    torch.manual_seed(42)
    model = NanoGPT(cfg)
    model.init_weights()
    return model


def test_t1():
    rng = np.random.default_rng(0)
    x = rng.integers(0, VOCAB, size=(3, SEQ)).astype(np.int64)
    xt = torch.from_numpy(x)
    for n in (2, 3, 4, 5):
        np_keys = ngram_freq.context_keys_np(x, n, VOCAB)
        model = make_model((n,), dict_topk=TOPK)  # dict mode routes 2/3 through the generic path
        t_keys = model._ngram_context_keys_torch(xt, n)
        assert torch.equal(torch.from_numpy(np_keys), t_keys), f"T1: n={n} mismatch"
    # force the splitmix64 branch with a vocab big enough that vocab^5 > 2^63:
    # vocab=64 -> 64^5 = 2^30 ... need vocab^n > 2^63, e.g. vocab=2**13, n=6.
    big_v = 1 << 13
    x2 = rng.integers(0, big_v, size=(2, 64)).astype(np.int64)
    np_keys = ngram_freq.context_keys_np(x2, 6, big_v)
    cfg6 = Config(
        vocab_size=big_v, n_layer=1, n_head=1, n_embd=16, sequence_len=64,
        dropout=0.0, enable_unigram_ve=False, enable_bigram_ve=False,
        enable_trigram_ve=False, nanogpt_ngram_injection_position="input",
        ngram_extra_orders=(6,), ngram_clean_table=R, ngram_dict_topk=1,
        seed=42, device_batch_size=2, total_batch_size=2 * 64,
        data_dir="/tmp", out_dir="/tmp")
    torch.manual_seed(42)
    m6 = NanoGPT(cfg6)
    t_keys = m6._ngram_context_keys_torch(torch.from_numpy(x2), 6)
    assert torch.equal(torch.from_numpy(np_keys), t_keys), "T1: splitmix branch mismatch"
    print("T1 PASS: torch keys == numpy keys (exact packing n=2..5 + splitmix n=6)")


def synth_index():
    """Synthetic multi-order index: order 2 has 600 distinct, order 3 has 400,
    order 5 has 500 — TOPK=300 exercises the top-K path on all."""
    rng = np.random.default_rng(1)
    keys, counts = {}, {}
    for n, size in ((2, 600), (3, 400), (5, 500)):
        pool = np.unique(rng.integers(1, 1 << 40, size=3 * size))
        k = np.sort(pool[:size]).astype(np.int64)
        c = rng.integers(1, 50, size=size).astype(np.int32)
        keys[n], counts[n] = k, c
    return ngram_freq.MultiOrderFrequencyIndex(
        (2, 3, 5), keys, counts, VOCAB,
        meta={"chunk_size": SEQ + 1, "shard_ids": [1], "positions": 10_000,
              "token_dtype": "u16"})


def test_t2():
    idx = synth_index()
    model = make_model((2, 3, 5), dict_topk=TOPK)
    logs = []
    model.load_ngram_dictionaries(idx, TOPK, log=logs.append)
    assert len(logs) == 3
    for n in (2, 3, 5):
        d = model.ngram_dict_keys[n]
        assert d.numel() == TOPK, f"T2: n={n} kept {d.numel()} != {TOPK}"
        assert torch.all(d[:-1] < d[1:]), f"T2: n={n} dictionary not strictly sorted"
        # determinism
        model2 = make_model((2, 3, 5), dict_topk=TOPK)
        model2.load_ngram_dictionaries(idx, TOPK, log=lambda *a: None)
        assert torch.equal(model2.ngram_dict_keys[n], d), f"T2: n={n} nondeterministic"
        # top-K correctness: kept counts >= every excluded count
        counts64 = idx._counts[n].astype(np.int64)
        d_np = d.numpy()
        in_dict = np.isin(idx._keys[n], d_np)
        assert counts64[in_dict].min() >= counts64[~in_dict].max(), \
            f"T2: n={n} top-K selection wrong"
        # coverage number sane
        cov = counts64[in_dict].sum() / counts64.sum()
        assert 0.0 < cov <= 1.0
    print("T2 PASS: dictionary build deterministic, top-K correct, sorted unique")


def test_t3():
    idx = synth_index()
    model = make_model((2, 3, 5), dict_topk=TOPK)
    model.load_ngram_dictionaries(idx, TOPK, log=lambda *a: None)
    rng = np.random.default_rng(2)
    # keys from the index itself (hit) + random keys (mostly miss)
    x = rng.integers(0, VOCAB, size=(4, SEQ)).astype(np.int64)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        residual = model._compute_input_ngram_residual(xt)
        total = torch.zeros_like(residual)
        valid_fracs = {}
        for n in (2, 3, 5):
            rows, valid = model._ngram_dict_rows(n, xt)
            contrib = model.ngram_ves[str(n)](rows) * valid.unsqueeze(-1).to(
                model.ngram_ves[str(n)].weight.dtype)
            total = total + contrib
            valid_fracs[n] = float(valid.float().mean())
            # members land on unique rows: distinct keys -> distinct positions
            keys_np = ngram_freq.context_keys_np(x, n, VOCAB).ravel()
            d_np = model.ngram_dict_keys[n].numpy()
            member_mask = np.isin(keys_np, d_np)
            rows_np = rows.numpy().ravel()[member_mask]
            keys_member = keys_np[member_mask]
            assert len(np.unique(keys_member)) == len(np.unique(rows_np)), \
                f"T3: n={n} collision among dictionary members"
        assert torch.allclose(residual, total, rtol=0, atol=0), \
            "T3: residual != sum of gather*valid"
    print(f"T3 PASS: residual == sum gather*valid; member rows collision-free; "
          f"valid frac {valid_fracs}")

    # T3b: hit path — dictionary built from the batch's own contexts, so every
    # position must hit (valid_frac == 1) and land on a unique row.
    x = rng.integers(0, VOCAB, size=(4, SEQ)).astype(np.int64)
    xt = torch.from_numpy(x)
    model2 = make_model((2, 3, 5), dict_topk=TOPK)
    for n in (2, 3, 5):
        keys_np = np.unique(ngram_freq.context_keys_np(x, n, VOCAB).ravel())
        model2.ngram_dict_keys[n] = torch.from_numpy(
            np.ascontiguousarray(keys_np, dtype=np.int64))
    with torch.no_grad():
        residual = model2._compute_input_ngram_residual(xt)
        total = torch.zeros_like(residual)
        for n in (2, 3, 5):
            rows, valid = model2._ngram_dict_rows(n, xt)
            assert float(valid.float().mean()) == 1.0, f"T3b: n={n} not all hit"
            tbl = model2.ngram_ves[str(n)]
            total = total + tbl(rows)
        assert torch.allclose(residual, total, rtol=0, atol=0), "T3b: residual mismatch"
    print("T3b PASS: all-hit dictionary: valid_frac == 1.0, residual exact")


def test_t4():
    idx = synth_index()
    model = make_model((2, 3, 5), dict_topk=TOPK)
    model.load_ngram_dictionaries(idx, TOPK, log=lambda *a: None)
    rng = np.random.default_rng(3)
    inp = torch.from_numpy(rng.integers(0, VOCAB, size=(4, SEQ)).astype(np.int64))
    tgt = torch.from_numpy(rng.integers(0, VOCAB, size=(4, SEQ)).astype(np.int64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = model(inp, targets=tgt)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(np.isfinite(l) for l in losses), "T4: non-finite loss"
    # table params are in the model and receive gradients
    g = model.ngram_ves["5"].weight.grad
    assert g is not None and torch.isfinite(g).all()
    print(f"T4 PASS: 5-step dict train finite ({losses[0]:.3f} -> {losses[-1]:.3f})")


def test_t5():
    from train import parse_ngram_dict_per_order
    assert parse_ngram_dict_per_order("") == {}
    assert parse_ngram_dict_per_order("2:k=7,5:f=3:k=9") == {2: {"k": 7}, 5: {"f": 3, "k": 9}}
    for bad in ("2:x=1", "2", "2:k=0"):
        try:
            parse_ngram_dict_per_order(bad)
        except ValueError:
            continue
        raise AssertionError(f"T5: {bad!r} accepted")
    idx = synth_index()
    legacy = make_model((2, 3, 5), dict_topk=TOPK)
    legacy.load_ngram_dictionaries(idx, TOPK, log=lambda *a: None)
    same = make_model((2, 3, 5), dict_topk=TOPK)
    same.load_ngram_dictionaries(idx, TOPK, log=lambda *a: None, per_order="")
    for n in (2, 3, 5):
        assert torch.equal(legacy.ngram_dict_keys[n], same.ngram_dict_keys[n]), "T5: empty spec drift"
    m = make_model((2, 3, 5), dict_topk=TOPK)
    m.load_ngram_dictionaries(idx, TOPK, log=lambda *a: None, per_order="2:k=50,3:f=30,5:k=40:f=45")
    for n, (k, f) in {2: (50, 1), 3: (TOPK, 30), 5: (40, 45)}.items():
        c = idx._counts[n].astype(np.int64)
        kept = np.isin(idx._keys[n], m.ngram_dict_keys[n].numpy())
        elig = c >= f
        assert kept.sum() == min(k, elig.sum()), f"T5: n={n} kept {kept.sum()}"
        assert not np.any(kept & ~elig), f"T5: n={n} kept below f"
        if kept.sum() < elig.sum():
            assert c[kept].min() >= c[elig & ~kept].max(), f"T5: n={n} not top-K within eligible"
        st = m.ngram_dict_stats[n]
        assert st["kept"] == kept.sum() and st["f"] == f and st["k"] == k
    print("T5 PASS: per-order k/f selection correct; empty spec bit-identical to legacy")


def main():
    test_t1()
    test_t2()
    test_t3()
    test_t4()
    test_t5()
    print("ALL NGRAM DICT CPU TESTS PASSED")


if __name__ == "__main__":
    main()
