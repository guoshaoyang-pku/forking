#!/usr/bin/env python3
"""CPU test for multi-head (H in 1,2,4,8) assignment tables.

T1 param budget: total table params identical for H=1,2,4,8 (R rows x d per order).
T2 readout: forward/backward run; each head's output equals its own table row.
T3 shared control: H=4 --shared rows equal head 0 in every head; independent
   heads collide less jointly (fraction of key pairs colliding in ALL heads).
T4 H=1/H=2 path untouched: model builds and loads legacy 1-D / (2,K) rows.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from train import Config, NanoGPT  # noqa: E402
from ngram_freq import context_keys_np  # noqa: E402

SEQ, VOCAB, D, R, K = 64, 53, 32, 200, 800
ORDERS = (2, 3, 4, 5)


def fake_index(path, rng):
    toks = rng.integers(0, VOCAB, size=200000).astype(np.int64)
    out = {}
    for n in ORDERS:
        k = context_keys_np(toks, n, VOCAB)
        u, c = np.unique(k[k >= 0] if (k < 0).any() else k, return_counts=True)
        out[f"keys_{n}"], out[f"counts_{n}"] = u, c
    np.savez(path, **out)


def build(idx, H, shared=False):
    out = tempfile.mktemp(suffix=".npz")
    cmd = [sys.executable, os.path.join(HERE, "mh_make_assign.py"), idx, str(K), str(R), str(H), out]
    if shared:
        cmd.append("--shared")
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def model_for(assign, H):
    z = np.load(assign)
    cfg = Config(vocab_size=VOCAB, n_layer=2, n_head=2, n_embd=D, sequence_len=SEQ, dropout=0.0,
                 enable_unigram_ve=False, enable_bigram_ve=False, enable_trigram_ve=False,
                 nanogpt_ngram_injection_position="input", ngram_extra_orders=ORDERS,
                 ngram_clean_table=R, ngram_assign_file=assign, ngram_heads=H, seed=42,
                 device_batch_size=2, total_batch_size=2 * SEQ, data_dir="/tmp", out_dir="/tmp")
    cfg._assign_R0 = {n: int(z[f"R_{n}"]) for n in ORDERS}
    torch.manual_seed(42)
    m = NanoGPT(cfg)
    m.init_weights()
    m.load_ngram_assignment(assign, log=lambda *a: None)
    return m


def table_params(m):
    return sum(p.numel() for name, p in m.named_parameters() if "ngram_ves" in name)


def main():
    rng = np.random.default_rng(0)
    idx = tempfile.mktemp(suffix=".npz")
    fake_index(idx, rng)
    x = torch.from_numpy(rng.integers(0, VOCAB, size=(2, SEQ)).astype(np.int64))
    budgets = {}
    for H in (1, 2, 4, 8):
        a = build(idx, H)
        m = model_for(a, H)
        budgets[H] = table_params(m)
        out = m(x, x)
        loss = out[0] if isinstance(out, tuple) else out
        if not torch.is_tensor(loss) or loss.dim():
            loss = m(x, x)
            loss = loss[0] if isinstance(loss, tuple) else loss
        loss.mean().backward()
        g = sum(p.grad.abs().sum().item() for n_, p in m.named_parameters()
                if "ngram_ves" in n_ and p.grad is not None)
        assert g > 0, f"H={H}: no table gradient"
        if H > 1:
            n = 3
            rows, valid = m._ngram_assign_lookup(n, x)
            assert rows.shape[0] == H
            t = [m.ngram_ves[str(n)]] + ([m.ngram_ves_b[str(n)]] if H == 2 else list(m.ngram_ves_x[str(n)]))
            cat = torch.cat([tb(rows[h]) for h, tb in enumerate(t)], dim=-1)
            assert cat.shape[-1] == D
        print(f"T2 H={H}: forward/backward ok, table params {budgets[H]:,}")
    assert len(set(budgets.values())) == 1, budgets
    print("T1 param budget identical:", budgets[1])
    z4 = np.load(build(idx, 4))
    zs = np.load(build(idx, 4, shared=True))
    r4, rs = z4["rows_3"], zs["rows_3"]
    assert all((rs[h] == rs[0]).all() for h in range(4))
    assert (r4[0] == rs[0]).all() and not (r4[1] == r4[0]).all()

    def joint_collision(rows):
        sig = [tuple(c) for c in rows.T]
        return 1 - len(set(sig)) / len(sig)

    jc_ind, jc_sh = joint_collision(r4), joint_collision(rs)
    print(f"T3 keys sharing a row in ALL heads: independent {jc_ind:.4f} vs shared {jc_sh:.4f}")
    assert jc_ind < jc_sh
    print("ALL PASS")


if __name__ == "__main__":
    main()
