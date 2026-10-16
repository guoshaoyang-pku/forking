#!/usr/bin/env python3
"""CPU acceptance tests for the plan-8 fast path (G1/G2/G3).

Run:  python3 code/tools/test_fastpath_cpu.py

Covers:
  T1  offline row index == model dynamic hash (bitwise, incl. chunk boundaries)
  T2  forward(pre_rows) loss/paths bitwise identical to dynamic-hash forward
  T3  chunked checkpoint CE == full-logit CE (per-token allclose, scalar loss)
  T4  static low-mask validity bits == freq-index semantics (keep_novel both)
  T5  uint32 token loader roundtrip + large-vocab (>65535) row math
  T6  _check_pre_rows_supported refusals (v-injection, perfect map)
  T7  stream length mismatch fails loudly
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

from train import (Config, NanoGPT, TokenizedShardDataset,  # noqa: E402
                   _BASE_BIGRAM_PRIMES, _BASE_TRIGRAM_PRIMES)

DEVICE = torch.device("cpu")
SEQ = 64
VOCAB = 97
R_BI, R_TRI = 211, 503


def write_shard(path: str, rng, n_rows: int, vocab: int, dtype):
    """Random packed rows; keep token diversity high for context variety."""
    toks = rng.integers(0, vocab, size=(n_rows, SEQ + 1)).astype(dtype)
    toks.tofile(path)
    return toks


def make_model(tmp: str, vocab: int, token_dtype: str, ce_chunk: int = 0,
               injection: str = "input", perfect_map: str = "") -> NanoGPT:
    cfg = Config(
        vocab_size=vocab, n_layer=2, n_head=2, n_embd=32, sequence_len=SEQ,
        dropout=0.0, enable_unigram_ve=False, enable_bigram_ve=True,
        enable_trigram_ve=True, nanogpt_ngram_injection_position=injection,
        bigram_clean_table=R_BI, trigram_clean_table=R_TRI,
        bigram_perfect_map=perfect_map,
        seed=42, device_batch_size=4, total_batch_size=4 * SEQ,
        ce_chunk_positions=ce_chunk, token_dtype=token_dtype,
        data_dir=tmp, out_dir=os.path.join(tmp, "out"),
    )
    torch.manual_seed(42)
    model = NanoGPT(cfg)
    model.init_weights()
    return model


def run_precompute(tmp: str, vocab: int, token_dtype: str, extra=()):
    cmd = [sys.executable, os.path.join(CODE, "precompute_row_index.py"),
           "--data_dir", tmp, "--shards", "0",
           "--vocab_size", str(vocab),
           "--bigram_clean_table", str(R_BI),
           "--trigram_clean_table", str(R_TRI),
           "--sequence_len", str(SEQ),
           "--token_dtype", token_dtype, *extra]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def first_rows_batch(tmp: str, vocab: int, token_dtype: str, batch: int = 4):
    ds = TokenizedShardDataset(tmp, [0], SEQ, batch, 42,
                               token_dtype=token_dtype,
                               row_index_dir=tmp, static_low_mask=0,
                               rows_requested=True)
    out = next(ds.iter_batches(DEVICE, with_rows=True))
    return out  # inp, tgt, bi, ti, bv, tv


def test_t1_t2(tmp: str):
    rng = np.random.default_rng(0)
    write_shard(os.path.join(tmp, "shard_00000.bin"), rng, 9, VOCAB, np.uint16)
    run_precompute(tmp, VOCAB, "u16")
    model = make_model(tmp, VOCAB, "u16")

    inp, _, bi, ti, _, _ = first_rows_batch(tmp, VOCAB, "u16")
    prev = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
    prev2 = torch.cat([inp[:, :2], inp[:, :-2]], dim=1)
    dyn_b = model._bigram_row_indices(sorted(model.bigram_ve_layers)[0], prev, inp)[0]
    dyn_t = model._trigram_row_indices(
        sorted(model.trigram_ve_layers)[0], prev2, prev, inp)[0]
    assert torch.equal(dyn_b, bi), "T1: bigram rows differ from dynamic hash"
    assert torch.equal(dyn_t, ti), "T1: trigram rows differ from dynamic hash"
    print("T1 PASS: offline rows == dynamic hash (bitwise, boundaries included)")

    # T2a: loss bitwise identical (same GEMM shapes on both paths)
    tgt = torch.roll(inp, -1, dims=1)
    torch.manual_seed(7)
    loss_dyn = model(inp, targets=tgt)
    loss_pre = model(inp, targets=tgt, pre_rows=(bi, ti, None, None))
    assert torch.equal(loss_dyn, loss_pre), \
        f"T2: loss mismatch {float(loss_dyn)} vs {float(loss_pre)}"
    # T2b: token losses + injection stats paths
    _, ptl_dyn = model(inp, targets=tgt, return_token_losses=True)
    _, ptl_pre = model(inp, targets=tgt, return_token_losses=True,
                       pre_rows=(bi, ti, None, None))
    assert torch.equal(ptl_dyn, ptl_pre), "T2: token losses differ"
    loss_s, stats = model(inp, targets=tgt, return_injection_stats=True,
                          pre_rows=(bi, ti, None, None))
    assert torch.equal(loss_s, loss_dyn), "T2: stats-path loss differs"
    assert float(stats["total_rms"]["batch_rms"]) > 0.0
    print("T2 PASS: pre_rows forward bitwise identical (loss/token-losses/stats)")


def test_t3(tmp: str):
    rng = np.random.default_rng(1)
    write_shard(os.path.join(tmp, "shard_00000.bin"), rng, 9, VOCAB, np.uint16)
    model = make_model(tmp, VOCAB, "u16", ce_chunk=0)
    model_chunk = make_model(tmp, VOCAB, "u16", ce_chunk=37)  # odd, non-dividing
    inp, _, bi, ti, _, _ = first_rows_batch(tmp, VOCAB, "u16")
    tgt = torch.roll(inp, -1, dims=1)
    with torch.no_grad():
        _, ptl_full = model(inp, targets=tgt, return_token_losses=True)
        _, ptl_chunk = model_chunk(inp, targets=tgt, return_token_losses=True)
        assert torch.allclose(ptl_full, ptl_chunk, rtol=0, atol=1e-6), \
            "T3: chunked per-token CE deviates"
        loss_full = model(inp, targets=tgt)
        loss_chunk = model_chunk(inp, targets=tgt)
        assert torch.allclose(loss_full, loss_chunk, rtol=0, atol=1e-6)
        # ignore_index path: -1 targets excluded identically
        tgt_m = tgt.clone()
        tgt_m[:, 0] = -1
        lf = model(inp, targets=tgt_m)
        lc = model_chunk(inp, targets=tgt_m)
        assert torch.allclose(lf, lc, rtol=0, atol=1e-6), "T3: ignore_index mismatch"
    print("T3 PASS: chunked checkpoint CE == full CE (atol 1e-6, ignore_index OK)")


def test_t4(tmp: str):
    from train import Config as C  # noqa: F401
    from ngram_freq import GlobalFrequencyIndex, compute_context_keys
    rng = np.random.default_rng(2)
    toks = write_shard(os.path.join(tmp, "shard_00000.bin"), rng, 9, VOCAB, np.uint16)
    # index over the SAME chunks with model boundary semantics
    idx = GlobalFrequencyIndex.build_from_chunks(tmp, [0], VOCAB, 9, SEQ + 1)
    idx_path = os.path.join(tmp, "freq_index.npz")
    idx.save(idx_path)
    for keep_novel in (1, 0):
        min_count = 3
        run_precompute(tmp, VOCAB, "u16",
                       extra=("--freq_index", idx_path,
                              "--min_count", str(min_count),
                              "--keep_novel", str(keep_novel)))
        ds = TokenizedShardDataset(tmp, [0], SEQ, 4, 42, token_dtype="u16",
                                   row_index_dir=tmp, static_low_mask=min_count)
        inp, _, _, _, bv, tv = next(ds.iter_batches(DEVICE, with_rows=True))
        bk, tk = compute_context_keys(inp, VOCAB)
        f_b = idx.hit_count_numpy("bigram", bk.numpy().ravel())
        f_t = idx.hit_count_numpy("trigram", tk.numpy().ravel())
        exp_b = (f_b >= min_count) | (keep_novel & (f_b == 0))
        exp_t = (f_t >= min_count) | (keep_novel & (f_t == 0))
        assert np.array_equal(bv.numpy().ravel(), exp_b), "T4: bigram bits mismatch"
        assert np.array_equal(tv.numpy().ravel(), exp_t), "T4: trigram bits mismatch"
    print("T4 PASS: static low-mask bits == freq-index semantics (keep_novel 0/1)")

    # static-mask forward zeroes disabled contexts (gather * validity)
    model = make_model(tmp, VOCAB, "u16")
    inp, _, bi, ti, bv, tv = next(TokenizedShardDataset(
        tmp, [0], SEQ, 4, 42, token_dtype="u16", row_index_dir=tmp,
        static_low_mask=3).iter_batches(DEVICE, with_rows=True))
    tgt = torch.roll(inp, -1, dims=1)
    with torch.no_grad():
        h_pre, res_pre = model._trunk(inp, pre_rows=(bi, ti, bv, tv))
        # reference: manual residual with validity zeroing
        lvs_b = model.bigram_ves[str(sorted(model.bigram_ve_layers)[0])]
        lvs_t = model.trigram_ves[str(sorted(model.trigram_ve_layers)[0])]
        ref = (lvs_b[0](bi) * bv.unsqueeze(-1).to(lvs_b[0](bi).dtype)
               + lvs_t[0](ti) * tv.unsqueeze(-1).to(lvs_t[0](ti).dtype))
        assert torch.equal(res_pre, ref), "T4: masked residual != gather*validity"
        model.config.static_low_mask = 3
        model.config.static_low_mask_keep_novel = True
    print("T4 PASS: masked residual == gather*validity (zero OE on disabled)")


def test_t5(tmp: str):
    big_vocab = 70000
    rng = np.random.default_rng(3)
    write_shard(os.path.join(tmp, "shard_00000.bin"), rng, 9, big_vocab, np.uint32)
    run_precompute(tmp, big_vocab, "u32")
    model = make_model(tmp, big_vocab, "u32")
    inp, tgt, bi, ti, _, _ = first_rows_batch(tmp, big_vocab, "u32")
    prev = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
    prev2 = torch.cat([inp[:, :2], inp[:, :-2]], dim=1)
    dyn_b = model._bigram_row_indices(sorted(model.bigram_ve_layers)[0], prev, inp)[0]
    dyn_t = model._trigram_row_indices(
        sorted(model.trigram_ve_layers)[0], prev2, prev, inp)[0]
    assert torch.equal(dyn_b, bi) and torch.equal(dyn_t, ti), \
        "T5: large-vocab rows differ (int64 hash math)"
    loss = model(inp, targets=tgt, pre_rows=(bi, ti, None, None))
    assert torch.isfinite(loss), "T5: large-vocab forward not finite"
    print("T5 PASS: u32 loader roundtrip + >65535 vocab row math bitwise OK")


def test_t6(tmp: str):
    try:
        make_model(tmp, VOCAB, "u16", injection="v")._check_pre_rows_supported()
        raise AssertionError("T6: v-injection not refused")
    except SystemExit:
        pass
    fake_map = os.path.join(tmp, "perfect.npz")
    n_bigram_distinct = VOCAB * VOCAB  # upper bound rows; only shape is checked here
    np.savez(fake_map,
             map=np.zeros(min(n_bigram_distinct, 1 << 20), dtype=np.int32),
             n_distinct=np.array([R_BI - 1], dtype=np.int64))
    try:
        make_model(tmp, VOCAB, "u16", perfect_map=fake_map)._check_pre_rows_supported()
        raise AssertionError("T6: perfect-map not refused")
    except SystemExit:
        pass
    print("T6 PASS: pre_rows refusals (v-injection, perfect map)")


def test_t7(tmp: str):
    rng = np.random.default_rng(4)
    write_shard(os.path.join(tmp, "shard_00000.bin"), rng, 9, VOCAB, np.uint16)
    run_precompute(tmp, VOCAB, "u16")
    # truncate one rows stream -> length check must fire
    b_path = os.path.join(tmp, "shard_00000.brows.u32.bin")
    raw = np.fromfile(b_path, dtype=np.uint32)
    raw[:-100].tofile(b_path)
    ds = TokenizedShardDataset(tmp, [0], SEQ, 4, 42, token_dtype="u16",
                               row_index_dir=tmp, static_low_mask=0,
                               rows_requested=True)
    try:
        ds._load_rows(0)
        raise AssertionError("T7: length mismatch not detected")
    except SystemExit:
        pass
    # missing meta contract file -> train.py refuses (meta check lives in main;
    # here we only assert the dataset surfaces missing-stream errors)
    os.remove(b_path)
    try:
        ds = TokenizedShardDataset(tmp, [0], SEQ, 4, 42, token_dtype="u16",
                                   row_index_dir=tmp, static_low_mask=3,
                                   rows_requested=True)
        ds._load_rows(0)
        raise AssertionError("T7: missing mask streams not detected")
    except SystemExit:
        pass
    print("T7 PASS: stream length mismatch / missing mask streams fail loudly")


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        test_t1_t2(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_t3(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_t4(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_t5(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_t6(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_t7(tmp)
    print("ALL FASTPATH CPU TESTS PASSED")


if __name__ == "__main__":
    main()
