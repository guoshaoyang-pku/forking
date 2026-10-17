#!/usr/bin/env python3
"""CPU acceptance tests for the coverage-matched control (--ngram_keep_rows hash).

Run:  python3 code/tools/test_ngram_keep_cpu.py

Dictionary mode (--ngram_dict_topk K) keeps the top-K train contexts of each
order as unique rows.  The control keeps the SAME membership mask but reads
the hash row, so collisions among kept contexts remain.  Dict vs keephash
therefore isolates the collision factor from the coverage factor.

Covers:
  T1  membership mask identical between dict and hash-keep models
  T2  hash-keep forward == independent numpy oracle: sum_n W_n[hash row] * valid
  T3  dict forward == independent numpy oracle (regression for the dict path)
  T4  valid.mean over the index's own positions == coverage printed at load
  T5  RMSProp row-sliced table update == unsliced update, bitwise (patched slice
      threshold so the sliced branch really runs)
  T6  CLI: hash-keep run completes, summary records ngram_keep_rows, dict
      artifact written; flag validation refuses hash without --ngram_dict_topk
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

import ngram_freq  # noqa: E402
import train  # noqa: E402
from train import Config, MixedOptimizer, NanoGPT, ngram_hash_constants  # noqa: E402

SEQ = 96
VOCAB = 53
R = 1024
TOPK = 300
ORDERS = (2, 3, 5)


def make_model(keep_rows: str, dict_topk: int = TOPK, seed: int = 42) -> NanoGPT:
    cfg = Config(
        vocab_size=VOCAB, n_layer=2, n_head=2, n_embd=32, sequence_len=SEQ,
        dropout=0.0, enable_unigram_ve=False, enable_bigram_ve=False,
        enable_trigram_ve=False, nanogpt_ngram_injection_position="input",
        ngram_extra_orders=ORDERS, ngram_clean_table=R,
        ngram_dict_topk=dict_topk, ngram_keep_rows=keep_rows,
        seed=seed, device_batch_size=4, total_batch_size=4 * SEQ,
        data_dir="/tmp", out_dir="/tmp",
    )
    torch.manual_seed(seed)
    model = NanoGPT(cfg)
    model.init_weights()
    return model


def make_chunks(seed: int, n_chunks: int) -> np.ndarray:
    """Small alphabet so high-order contexts repeat and the top-K cut is real."""
    rng = np.random.default_rng(seed)
    alphabet = rng.choice(VOCAB, size=30, replace=False)
    return alphabet[rng.integers(0, len(alphabet), size=(n_chunks, SEQ + 1))].astype(np.uint16)


def build_index(data_dir: str, chunks: np.ndarray):
    chunks.tofile(os.path.join(data_dir, "shard_00001.bin"))
    return ngram_freq.MultiOrderFrequencyIndex.build_from_chunks(
        data_dir, [1], VOCAB, ORDERS, chunk_size=SEQ + 1, log=lambda *_: None)


def loaded(model: NanoGPT, idx) -> list:
    logs: list = []
    model.load_ngram_dictionaries(idx, TOPK, log=logs.append)
    return logs


def numpy_oracle(model: NanoGPT, x: np.ndarray, dict_keys: dict, mode: str) -> np.ndarray:
    out = np.zeros((x.shape[0], x.shape[1], model.config.n_embd), dtype=np.float64)
    for n in ORDERS:
        keys = ngram_freq.context_keys_np(x, n, VOCAB)
        dk = dict_keys[n]
        pos = np.searchsorted(dk, keys)
        pos_c = np.clip(pos, 0, dk.size - 1)
        valid = (dk[pos_c] == keys).astype(np.float64)
        if mode == "hash":
            consts = ngram_hash_constants(n)
            rows = np.zeros_like(x, dtype=np.int64)
            for k, c in enumerate(consts):
                sh = x if k == 0 else np.concatenate([x[:, :k], x[:, :x.shape[1] - k]], axis=1)
                rows ^= sh.astype(np.int64) * c
            rows = rows % R
        else:
            rows = pos_c
        W = model.ngram_ves[str(n)].weight.detach().double().numpy()
        out += W[rows] * valid[..., None]
    return out


def main():
    with tempfile.TemporaryDirectory() as tmp:
        chunks = make_chunks(7, 24)
        idx = build_index(tmp, chunks)
        x = chunks[:, :SEQ].astype(np.int64)
        xt = torch.from_numpy(x)

        dict_m = make_model("dict")
        hash_m = make_model("hash")
        logs_d = loaded(dict_m, idx)
        logs_h = loaded(hash_m, idx)
        hash_m.load_state_dict(dict_m.state_dict())  # identical weights

        # T1 membership identical
        for n in ORDERS:
            _, vd = dict_m._ngram_dict_rows(n, xt)
            _, vh = hash_m._ngram_dict_rows(n, xt)
            assert torch.equal(vd, vh), f"T1: membership differs for n={n}"
            assert 0 < int(vd.sum()) < vd.numel(), f"T1: degenerate mask for n={n}"
        print("T1 PASS: membership mask identical (dict vs hash-keep)")

        # T2 hash-keep forward == oracle
        dict_keys = {n: dict_m.ngram_dict_keys[n].numpy() for n in ORDERS}
        with torch.no_grad():
            got_h = hash_m._compute_input_ngram_residual(xt).double().numpy()
        exp_h = numpy_oracle(hash_m, x, dict_keys, "hash")
        assert np.allclose(got_h, exp_h, atol=1e-6), \
            f"T2: max abs err {np.abs(got_h - exp_h).max():.3e}"
        print("T2 PASS: hash-keep residual == numpy oracle (hash rows x membership)")

        # T3 dict forward == oracle (regression)
        with torch.no_grad():
            got_d = dict_m._compute_input_ngram_residual(xt).double().numpy()
        exp_d = numpy_oracle(dict_m, x, dict_keys, "dict")
        assert np.allclose(got_d, exp_d, atol=1e-6), \
            f"T3: max abs err {np.abs(got_d - exp_d).max():.3e}"
        assert not np.allclose(got_d, got_h), "T3: dict and hash-keep residuals identical"
        print("T3 PASS: dict residual == numpy oracle; the two modes differ")

        # T4 coverage at load == valid.mean on the index's own positions
        for n in ORDERS:
            printed = [float(line.split("position coverage ")[1])
                       for line in logs_d if f"dict n={n}:" in line]
            _, v = dict_m._ngram_dict_rows(n, xt)
            measured = float(v.double().mean())
            # the loader prints :.4f, so compare at that precision
            assert len(printed) == 1 and abs(printed[0] - measured) < 5e-5, \
                f"T4: n={n} printed {printed} measured {measured:.6f}"
            printed_h = float([l for l in logs_h if f"dict n={n}:" in l][0].split("position coverage ")[1])
            assert abs(printed_h - measured) < 5e-5
        print("T4 PASS: valid.mean == printed coverage for every order")

        # T5 sliced RMSProp update == unsliced, bitwise
        m_a = make_model("hash", seed=3)
        m_b = make_model("hash", seed=3)
        opt_a = MixedOptimizer(m_a, lr=6e-4, ngram_betas=(0.0, 0.99), adam_betas=(0.8, 0.95),
                               weight_decay=0.1, table_optimizer="rmsprop",
                               table_lr_scale=128.0, table_betas=(0.0, 0.99))
        opt_b = MixedOptimizer(m_b, lr=6e-4, ngram_betas=(0.0, 0.99), adam_betas=(0.8, 0.95),
                               weight_decay=0.1, table_optimizer="rmsprop",
                               table_lr_scale=128.0, table_betas=(0.0, 0.99))
        rng = np.random.default_rng(11)
        saved = (train._TABLE_SLICE_BYTES, train._TABLE_SLICE_ROWS)
        try:
            for step in range(3):
                grads = {name: torch.from_numpy(rng.standard_normal(p.shape).astype(np.float32))
                         for name, p in opt_a.ngram_params}
                for opt, mdl, mode in ((opt_a, m_a, "unsliced"), (opt_b, m_b, "sliced")):
                    if mode == "unsliced":
                        train._TABLE_SLICE_BYTES, train._TABLE_SLICE_ROWS = 1 << 60, 1
                    else:
                        train._TABLE_SLICE_BYTES, train._TABLE_SLICE_ROWS = 1, 100
                    for name, p in opt.ngram_params:
                        p.grad = grads[name].clone()
                    with torch.no_grad():
                        for name, p in opt.ngram_params:
                            opt._table_rmsprop_step(name, p, 6e-4 * 128.0)
        finally:
            train._TABLE_SLICE_BYTES, train._TABLE_SLICE_ROWS = saved
        for (name, pa), (_, pb) in zip(m_a.named_parameters(), m_b.named_parameters()):
            if any(k in name for k in ("ngram_ves", "bigram_ves", "trigram_ves", "value_embeds")):
                assert torch.equal(pa, pb), f"T5: sliced != unsliced for {name}"
        print("T5 PASS: sliced table update == unsliced (bitwise, 3 steps)")

        # T6 CLI end to end + validation
        data = os.path.join(tmp, "cli")
        os.makedirs(data)
        chunks2 = make_chunks(9, 40)
        chunks2.tofile(os.path.join(data, "shard_00001.bin"))
        chunks2[:8].tofile(os.path.join(data, "shard_00002.bin"))
        idx_path = os.path.join(data, "multi.npz")
        ngram_freq.MultiOrderFrequencyIndex.build_from_chunks(
            data, [1], VOCAB, ORDERS, chunk_size=SEQ + 1, log=lambda *_: None).save(idx_path)
        out = os.path.join(tmp, "runs")
        common = [sys.executable, os.path.join(CODE, "train.py"), "--out_dir", out,
                  "--data_dir", data, "--train_shards", "1", "--val_shards", "2",
                  "--steps", "4", "--dtype", "fp32", "--ngram_orders", "2,3,5",
                  "--ngram_dict_topk", str(TOPK), "--ngram_clean_table", str(R),
                  "--n_layer", "2", "--n_head", "2", "--n_embd", "32",
                  "--vocab_size", str(VOCAB), "--sequence_len", str(SEQ),
                  "--device_batch_size", "2", "--total_batch_size", str(2 * SEQ),
                  "--lr", "0.0006", "--table_lr_scale", "128", "--table_betas", "0.0,0.99",
                  "--val_interval", "2", "--val_batches", "1", "--freq_eval_batches", "1",
                  "--freq_eval_interval", "2", "--exact_freq_eval_interval", "2",
                  "--freq_index", idx_path, "--diag_workers", "1"]
        r = subprocess.run(common + ["--run_id", "hash", "--ngram_keep_rows", "hash"],
                           capture_output=True, text=True, cwd=CODE)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        summ = json.load(open(os.path.join(out, "hash", "summary.json")))
        assert summ["ngram_keep_rows"] == "hash" and summ["config"]["ngram_keep_rows"] == "hash"
        assert os.path.exists(os.path.join(out, "hash", f"ngram_dicts_top{TOPK}.npz"))
        bad = subprocess.run(common + ["--run_id", "bad", "--ngram_dict_topk", "0",
                                       "--ngram_keep_rows", "hash"],
                             capture_output=True, text=True, cwd=CODE)
        assert bad.returncode != 0 and "requires --ngram_dict_topk > 0" in bad.stderr, bad.stderr[-500:]
        print("T6 PASS: CLI hash-keep run completes; hash without dict topk refused")

    print("ALL KEEP TESTS PASSED")


if __name__ == "__main__":
    main()
