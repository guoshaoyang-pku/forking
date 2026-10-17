"""CPU test for count_init_clean_tables (§80): synthetic stub model + freq npz.

Checks: (1) row content == count-weighted mean of wte[cur] over the bucket,
(2) unseen rows are zero, (3) determinism across calls, (4) hash alignment
with the model's own _bigram_row_indices/_trigram_row_indices formulas.
Run: python3 code/tools/check_count_init_tables.py  (repo root, no GPU needed)
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import count_init_clean_tables  # noqa: E402


class _Stub:
    def __init__(self, V, R, D, seed=0):
        self.config = type("C", (), {"vocab_size": V})()
        self.bigram_table_size = R
        self.trigram_table_size = R
        self.bigram_perfect = False
        self.bigram_K = 1
        self.trigram_K = 1
        from train import _BASE_BIGRAM_PRIMES, _BASE_TRIGRAM_PRIMES
        self.bigram_hash_primes_per_layer = {0: _BASE_BIGRAM_PRIMES[0][0]}
        self.trigram_hash_primes_per_layer = {0: _BASE_TRIGRAM_PRIMES[0]}
        g = torch.Generator().manual_seed(seed)
        self.transformer = type("T", (), {})()
        self.transformer.wte = nn.Embedding(V, D)
        self.transformer.wte.weight.data = torch.randn(V, D, generator=g) * 0.02
        self.bigram_ves = nn.ModuleDict({"0": nn.ModuleList([nn.Embedding(R, D)])})
        self.trigram_ves = nn.ModuleDict({"0": nn.ModuleList([nn.Embedding(R, D)])})


def main():
    torch.manual_seed(0)
    V, R, D = 64, 256, 8
    m = _Stub(V, R, D)
    wte = m.transformer.wte.weight.detach().clone()

    rng = np.random.default_rng(7)
    # 2000 distinct bigram keys, 1500 distinct trigram keys; counts Zipf-ish
    # (capped at 1e6: raw zipf(1.3) tail can exceed int32 and wrap negative)
    cap = lambda a: np.minimum(a, 10**6)
    b_keys = rng.choice(V * V, size=2000, replace=False).astype(np.int64)
    b_counts = (cap(rng.zipf(1.3, size=2000)) + 1).astype(np.int32)
    t_keys = rng.choice(V ** 3, size=1500, replace=False).astype(np.int64)
    t_counts = (cap(rng.zipf(1.3, size=1500)) + 1).astype(np.int32)

    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "freq_index.npz")
        np.savez(npz, bigram_keys=b_keys, bigram_counts=b_counts,
                 trigram_keys=t_keys, trigram_counts=t_counts,
                 vocab_size=np.array([V]))
        s1 = count_init_clean_tables(m, npz)
        table_b = m.bigram_ves["0"][0].weight.detach().clone()
        table_t = m.trigram_ves["0"][0].weight.detach().clone()

    # (3) determinism
    m2 = _Stub(V, R, D)
    m2.transformer.wte.weight.data = wte.clone()
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "freq_index.npz")
        np.savez(npz, bigram_keys=b_keys, bigram_counts=b_counts,
                 trigram_keys=t_keys, trigram_counts=t_counts,
                 vocab_size=np.array([V]))
        count_init_clean_tables(m2, npz)
    assert torch.equal(table_b, m2.bigram_ves["0"][0].weight.detach()), "bigram not deterministic"
    assert torch.equal(table_t, m2.trigram_ves["0"][0].weight.detach()), "trigram not deterministic"

    # (1) expected rows via the model's own hash formula
    from train import _BASE_BIGRAM_PRIMES, _BASE_TRIGRAM_PRIMES
    p1, p2 = _BASE_BIGRAM_PRIMES[0][0][:2]
    q0, q1, q2 = _BASE_TRIGRAM_PRIMES[0][:3]
    prev = b_keys // V
    cur = b_keys % V
    rows = ((prev * p1) ^ (cur * p2)) % R
    exp = torch.zeros(R, D)
    cnt = torch.zeros(R)
    for r, c, cc in zip(rows, b_counts.astype(np.float64), cur):
        exp[r] += c * wte[cc]
        cnt[r] += c
    exp = exp / cnt.clamp(min=1).unsqueeze(1)
    exp[cnt == 0] = 0
    assert torch.allclose(table_b, exp, atol=1e-5), \
        f"bigram mismatch, max diff {(table_b - exp).abs().max()}"

    prev2 = t_keys // (V * V)
    prevt = (t_keys // V) % V
    curt = t_keys % V
    rows_t = ((prev2 * q0) ^ (prevt * q1) ^ (curt * q2)) % R
    exp_t = torch.zeros(R, D)
    cnt_t = torch.zeros(R)
    for r, c, cc in zip(rows_t, t_counts.astype(np.float64), curt):
        exp_t[r] += c * wte[cc]
        cnt_t[r] += c
    exp_t = exp_t / cnt_t.clamp(min=1).unsqueeze(1)
    exp_t[cnt_t == 0] = 0
    assert torch.allclose(table_t, exp_t, atol=1e-5), \
        f"trigram mismatch, max diff {(table_t - exp_t).abs().max()}"

    # (2) unseen rows zero (with 2000 keys over 64 rows, all rows likely hit;
    # force an unseen row by using a tiny key set on a fresh stub)
    m3 = _Stub(V, R, D)
    m3.transformer.wte.weight.data = wte.clone()
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "freq_index.npz")
        k = np.array([3 * V + 5], dtype=np.int64)
        np.savez(npz, bigram_keys=k, bigram_counts=np.array([10], dtype=np.int32),
                 trigram_keys=np.array([2 * V * V + 3 * V + 4], dtype=np.int64),
                 trigram_counts=np.array([7], dtype=np.int32),
                 vocab_size=np.array([V]))
        count_init_clean_tables(m3, npz)
    tb = m3.bigram_ves["0"][0].weight.detach()
    tt = m3.trigram_ves["0"][0].weight.detach()
    r_exp = ((3 * p1) ^ (5 * p2)) % R
    rt_exp = ((2 * q0) ^ (3 * q1) ^ (4 * q2)) % R
    assert torch.allclose(tb[r_exp], wte[5], atol=1e-6), "single-key row != wte[cur]"
    assert torch.allclose(tt[rt_exp], wte[4], atol=1e-6), "trigram single-key row wrong"
    mask = torch.ones(R, dtype=torch.bool)
    mask[r_exp] = False
    assert torch.count_nonzero(tb[mask]) == 0, "unseen bigram rows not zero"
    mask_t = torch.ones(R, dtype=torch.bool)
    mask_t[rt_exp] = False
    assert torch.count_nonzero(tt[mask_t]) == 0, "unseen trigram rows not zero"

    print(f"PASS: rows_nonzero b/t = {s1['bigram']['rows_nonzero']}/{R}, "
          f"{s1['trigram']['rows_nonzero']}/{R}; weighted-mean, zero-unseen, "
          "determinism, hash alignment all OK")


if __name__ == "__main__":
    main()
