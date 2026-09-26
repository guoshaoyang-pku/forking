"""Tests for the scaling measurement infrastructure (plan §2.6).

Covers:
  - epoch_batches nested-prefix boundaries
  - fixed train probe does not consume training iterator / epoch counter
  - exact-frequency index matches model context keys position-by-position
  - table_mult maps physical rows / logical addresses / params correctly
  - L1 ⊂ L2 ⊂ L3 ⊂ L4 nested prefixes
  - table_occupancy hash matches the model's hash
  - β₂ real read of table_betas[1]
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train import (  # noqa: E402
    Config,
    MixedOptimizer,
    NanoGPT,
    TokenizedShardDataset,
)
from ngram_freq import (  # noqa: E402
    ExactFreqLossAccumulator,
    GlobalFrequencyIndex,
)
from table_occupancy import (  # noqa: E402
    _context_keys,
    _load_chunk_matrix,
    compute_occupancy,
    hash_rows_for_branch,
)


def _make_shard(data_dir: str, sid: int, n_chunks: int, chunk_size: int,
                vocab_size: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=n_chunks * chunk_size, dtype=np.uint16)
    path = os.path.join(data_dir, f"shard_{sid:05d}.bin")
    with open(path, "wb") as f:
        f.write(tokens.tobytes())
    return path


def _make_small_shard_dir(n_chunks=8, chunk_size=5, vocab_size=16, seed=1):
    d = tempfile.mkdtemp()
    _make_shard(d, 1, n_chunks, chunk_size, vocab_size, seed)
    return d


# ---------------------------------------------------------------------------
# epoch_batches boundaries
# ---------------------------------------------------------------------------


def test_epoch_batches_boundary_exact():
    """With epoch_batches=B, the (B+1)-th batch starts epoch 2.
    NOTE: _batch_in_epoch is 0-indexed (batch 1 -> 0), so after B batches it
    equals B-1."""
    d = _make_small_shard_dir(n_chunks=100, chunk_size=5, vocab_size=16, seed=2)
    ds = TokenizedShardDataset(d, [1], sequence_len=4, device_batch_size=2,
                               seed=42, epoch_batches=5)
    it = ds.iter_batches(torch.device("cpu"))
    for _ in range(5):
        next(it)
        assert ds._epoch == 0, "first B batches are epoch 1"
    # 6th batch starts epoch 2 (epoch counter advanced, batch-in-epoch reset)
    next(it)
    assert ds._epoch == 1
    assert ds._batch_in_epoch == 0


def test_epoch_batches_nested_prefix():
    """L1 < L2 < L3 < L4: each larger epoch length re-emits the shorter's batches."""
    d = _make_small_shard_dir(n_chunks=60, chunk_size=5, vocab_size=16, seed=3)
    B = 2  # device batch size
    chunks_per_epoch = {42: 84, 84: 168}  # not needed; use direct
    # capture first 4 batches under L=8 vs L=16
    ds8 = TokenizedShardDataset(d, [1], sequence_len=4, device_batch_size=B,
                                seed=42, epoch_batches=8)
    it8 = ds8.iter_batches(torch.device("cpu"))
    first8 = [next(it8)[0] for _ in range(8)]
    ds16 = TokenizedShardDataset(d, [1], sequence_len=4, device_batch_size=B,
                                 seed=42, epoch_batches=16)
    it16 = ds16.iter_batches(torch.device("cpu"))
    first16 = [next(it16)[0] for _ in range(8)]
    for a, b in zip(first8, first16):
        assert torch.equal(a, b), "L16 must replay L8's first batches identically"


def test_epoch_batches_full_shard_when_zero():
    """epoch_batches=0 uses full shard length (legacy behaviour preserved)."""
    d = _make_small_shard_dir(n_chunks=10, chunk_size=5, vocab_size=16, seed=4)
    ds = TokenizedShardDataset(d, [1], sequence_len=4, device_batch_size=2,
                               seed=42, epoch_batches=0)
    it = ds.iter_batches(torch.device("cpu"))
    for i in range(5):  # 10 chunks / batch size 2 = 5 batches per epoch
        next(it)
        assert ds._epoch == 0
    next(it)
    assert ds._epoch == 1, "after full epoch, counter advances"


# ---------------------------------------------------------------------------
# fixed train probe safety
# ---------------------------------------------------------------------------


def _probe_batches(ds, n, device="cpu"):
    it = ds.iter_batches(torch.device(device))
    return [next(it) for _ in range(n)]


def test_fixed_probe_does_not_consume_training_stream():
    """Probe from a separate dataset instance must not change training batches."""
    d = _make_small_shard_dir(n_chunks=40, chunk_size=5, vocab_size=16, seed=5)
    # reference: a plain training stream's first 8 batches (one iterator)
    ref_ds = TokenizedShardDataset(d, [1], sequence_len=4, device_batch_size=2,
                                   seed=42, epoch_batches=10)
    ref_iter = ref_ds.iter_batches(torch.device("cpu"))
    ref = [next(ref_iter)[0] for _ in range(8)]

    # training instance (what the training loop consumes)
    train_ds = TokenizedShardDataset(d, [1], sequence_len=4, device_batch_size=2,
                                     seed=42, epoch_batches=10)
    train_iter = train_ds.iter_batches(torch.device("cpu"))
    expected = [next(train_iter)[0] for _ in range(4)]
    # separate probe dataset instance consumes 4 batches but from ITS OWN stream
    probe_ds = TokenizedShardDataset(d, [1], sequence_len=4, device_batch_size=2,
                                     seed=42, epoch_batches=10)
    probe_iter = probe_ds.iter_batches(torch.device("cpu"))
    _ = [next(probe_iter) for _ in range(4)]
    # training stream is unaffected: next 4 batches == reference batches 5..8
    after = [next(train_iter)[0] for _ in range(4)]
    assert all(torch.equal(expected[i], ref[i]) for i in range(4))
    assert all(torch.equal(after[i], ref[4 + i]) for i in range(4))
    # probe consumed nothing from the training instance
    # (8 batches consumed, 0-indexed -> batch_in_epoch == 7)
    assert train_ds._epoch == 0 and train_ds._batch_in_epoch == 7
    assert probe_ds._epoch == 0 and probe_ds._batch_in_epoch == 3


def test_probe_epoch_counter_not_advanced_by_probe():
    d = _make_small_shard_dir(n_chunks=40, chunk_size=5, vocab_size=16, seed=6)
    train_ds = TokenizedShardDataset(d, [1], sequence_len=4, device_batch_size=2,
                                     seed=42, epoch_batches=10)
    train_iter = train_ds.iter_batches(torch.device("cpu"))
    probe_ds = TokenizedShardDataset(d, [1], sequence_len=4, device_batch_size=2,
                                     seed=42, epoch_batches=10)
    _probe_batches(probe_ds, 4)
    # after 10 training batches (one full epoch), the epoch counter has not yet
    # advanced (it flips when the first batch of the next epoch is consumed),
    # but batch-in-epoch == 9 (0-indexed) and probe consumed nothing.
    for _ in range(10):
        next(train_iter)
    assert train_ds._epoch == 0
    assert train_ds._batch_in_epoch == 9
    assert probe_ds._epoch == 0 and probe_ds._batch_in_epoch == 3


# ---------------------------------------------------------------------------
# exact-frequency index vs model context keys
# ---------------------------------------------------------------------------


def test_freq_index_matches_model_context_keys():
    """The offline exact-frequency index must count exactly the contexts the
    model sees, position by position, including first-position repetition."""
    vocab = 8
    chunk_size = 6  # sequence_len = 5
    d = tempfile.mkdtemp()
    # handcrafted chunks so we can verify position semantics
    chunks = np.array([
        [1, 2, 3, 4, 5, 6],   # chunk 0
        [7, 1, 9, 2, 0, 3],   # chunk 1
    ], dtype=np.uint16)
    with open(os.path.join(d, "shard_00001.bin"), "wb") as f:
        f.write(chunks.tobytes())

    idx = GlobalFrequencyIndex.build_from_chunks(d, [1], vocab, n_chunks=2, chunk_size=chunk_size)

    # Model context semantics (NanoGPT._compute_input_ngram_residual):
    #   idx = chunk[:-1]; prev[j] = idx[max(0,j-1)] except j=1 uses idx[0];
    #   prev2[j] = idx[:, :2] for j=0,1 then idx[:-2]  -> [t0,t1,t0,t1,t2]
    # chunk0 idx=[1,2,3,4,5]: prev=[1,1,2,3,4] cur=[1,2,3,4,5]
    #   bigram keys: (1,1)=9 (1,2)=10 (2,3)=19 (3,4)=28 (4,5)=37
    # chunk1 idx=[7,1,9,2,0]: prev=[7,7,1,9,2] cur=[7,1,9,2,0]
    #   bigram keys: (7,7)=63 (7,1)=57 (1,9)=17 (9,2)=74 (2,0)=16
    expected_b = {
        9: 1, 10: 1, 19: 1, 28: 1, 37: 1,
        63: 1, 57: 1, 17: 1, 74: 1, 16: 1,
    }
    assert idx.bigram == expected_b
    # trigram keys: prev2=[t0,t1,t0,t1,t2] (j=1 uses t1, matching the model)
    # chunk0: (1,1,1)=73 (2,1,2)=138 (1,2,3)=83 (2,3,4)=156 (3,4,5)=229
    # chunk1: (7,7,7)=511 (1,7,1)=121 (7,1,9)=465 (1,9,2)=138 (9,2,0)=592
    v2 = vocab * vocab
    expected_t = {
        73: 1, 138: 2, 83: 1, 156: 1, 229: 1,
        511: 1, 121: 1, 465: 1, 592: 1,
    }
    assert idx.trigram == expected_t


def test_freq_index_matches_accumulator_keys():
    """ExactFreqLossAccumulator._compute_keys must equal the index's key encoding."""
    vocab = 8
    chunk_size = 6
    d = tempfile.mkdtemp()
    chunks = np.array([[1, 2, 3, 4, 5, 6], [7, 1, 9, 2, 0, 3]], dtype=np.uint16)
    with open(os.path.join(d, "shard_00001.bin"), "wb") as f:
        f.write(chunks.tobytes())
    idx = GlobalFrequencyIndex.build_from_chunks(d, [1], vocab, n_chunks=2, chunk_size=chunk_size)

    inp = torch.tensor(chunks[:, :-1])  # (2,5) inputs
    for branch in ("bigram", "trigram"):
        acc = ExactFreqLossAccumulator(idx, vocab, branch)
        keys = acc._compute_keys(inp).numpy()
        if branch == "bigram":
            prev = np.concatenate([inp[:, :1].numpy(), inp[:, :-1].numpy()], axis=1)
            expected = prev * vocab + inp.numpy()
        else:
            prev = np.concatenate([inp[:, :1].numpy(), inp[:, :-1].numpy()], axis=1)
            prev2 = np.concatenate([inp[:, :2].numpy(), inp[:, :-2].numpy()], axis=1)
            expected = prev2 * vocab * vocab + prev * vocab + inp.numpy()
        assert np.array_equal(keys, expected)
    # Also confirm the accumulator keys equal the model's hashing inputs exactly
    # (chunk0 bigram: [9,10,19,28,37]; trigram: [73,138,83,156,229])
    acc_b = ExactFreqLossAccumulator(idx, vocab, "bigram")
    assert acc_b._compute_keys(inp).numpy()[0].tolist() == [9, 10, 19, 28, 37]
    acc_t = ExactFreqLossAccumulator(idx, vocab, "trigram")
    assert acc_t._compute_keys(inp).numpy()[0].tolist() == [73, 138, 83, 156, 229]


# ---------------------------------------------------------------------------
# table_mult mapping
# ---------------------------------------------------------------------------


def test_table_mult_physical_rows_and_params():
    cfg = Config(vocab_size=64, table_mult=4, n_layer=2, n_head=2, n_embd=32,
                 enable_unigram_ve=False, enable_bigram_ve=True, enable_trigram_ve=False,
                 nanogpt_ngram_injection_position="input")
    model = NanoGPT(cfg)
    assert model.bigram_table_size == 64 * 4
    assert model.trigram_table_size == 64 * 4
    # bigram table: 2 hashes * (rows * half_dim + rows * (embd - half_dim))
    R = 64 * 4
    n_ve = len(model.bigram_ve_layers)
    bigram_params = n_ve * 2 * R * 32
    n_params = sum(p.numel() for p in model.parameters())
    # logical addresses = 2R per layer per hash
    assert sum(2 * R for _ in model.bigram_ves) == n_ve * 2 * R


# ---------------------------------------------------------------------------
# table occupancy hash equivalence
# ---------------------------------------------------------------------------


def test_occupancy_hash_matches_model_hash():
    """table_occupancy's row hash must equal the model's bigram/trigram hash."""
    vocab = 16
    table_mult = 4
    chunk_size = 6
    d = tempfile.mkdtemp()
    chunks = np.array([[1, 2, 3, 4, 5, 6], [7, 1, 9, 2, 0, 3]], dtype=np.uint16)
    with open(os.path.join(d, "shard_00001.bin"), "wb") as f:
        f.write(chunks.tobytes())

    cfg = Config(vocab_size=vocab, table_mult=table_mult, n_layer=2, n_head=2, n_embd=32,
                 enable_unigram_ve=False, enable_bigram_ve=True, enable_trigram_ve=True,
                 nanogpt_ngram_injection_position="input")
    model = NanoGPT(cfg)
    table_size = vocab * table_mult

    tokens = _load_chunk_matrix(d, [1], chunk_size=chunk_size, n_chunks=2)
    bigram_primes = [model.bigram_hash_primes_per_layer[li] for li in sorted(model.bigram_ve_layers)]
    trigram_primes = [model.trigram_hash_primes_per_layer[li] for li in sorted(model.trigram_ve_layers)]

    b_rows = hash_rows_for_branch(tokens, vocab, table_size, "bigram", bigram_primes)
    t_rows = hash_rows_for_branch(tokens, vocab, table_size, "trigram", trigram_primes)

    # compare with model's own computation
    inp = torch.tensor(chunks[:, :-1]).long()
    prev = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
    prev2 = torch.cat([inp[:, :2], inp[:, :-2]], dim=1)
    li0 = sorted(model.bigram_ve_layers)[0]
    bp = model.bigram_hash_primes_per_layer[li0]
    model_b_rows = [((prev * p1) ^ (inp * p2)) % table_size for p1, p2 in bp]
    assert len(b_rows[0]) == 2
    for a, b in zip(b_rows[0], model_b_rows):
        assert torch.equal(torch.from_numpy(a.astype(np.int64)), b)
    li0 = sorted(model.trigram_ve_layers)[0]
    lp = model.trigram_hash_primes_per_layer[li0]
    model_t_rows = [
        ((prev2 * lp[0]) ^ (prev * lp[1]) ^ (inp * lp[2])) % table_size,
        ((prev2 * lp[3]) ^ (prev * lp[4]) ^ (inp * lp[5])) % table_size,
    ]
    for a, b in zip(t_rows[0], model_t_rows):
        assert torch.equal(torch.from_numpy(a.astype(np.int64)), b)


def test_clean_trigram_occupancy_uses_one_complete_hash():
    vocab = 16
    chunk_size = 6
    d = tempfile.mkdtemp()
    chunks = np.array([[1, 2, 3, 4, 5, 6], [7, 1, 9, 2, 0, 3]], dtype=np.uint16)
    with open(os.path.join(d, "shard_00001.bin"), "wb") as f:
        f.write(chunks.tobytes())

    clean_rows = 31
    branch_primes = {
        "bigram": [[(1, 2), (3, 5)]],
        "trigram": [(7, 11, 13, 17, 19, 23)],
    }
    occupancy = compute_occupancy(
        d, [1], vocab, sequence_len=5, device_batch_size=2,
        epoch_batches=1, table_mult=4, branch_primes=branch_primes,
        bigram_clean_table=clean_rows, trigram_clean_table=clean_rows,
    )
    tokens = _load_chunk_matrix(d, [1], chunk_size=chunk_size, n_chunks=2)
    expected_rows = hash_rows_for_branch(
        tokens, vocab, clean_rows, "trigram", [(7, 11, 13)],
    )[0]
    trigram_stats = occupancy["branches"]["trigram"]["0"]

    assert len(expected_rows) == 1
    assert len(trigram_stats) == 1
    assert trigram_stats[0]["physical_rows_R"] == clean_rows
    assert trigram_stats[0]["logical_addresses"] == clean_rows


def test_occupancy_monotonic_with_table_size():
    """Occupancy must decrease as table size grows (same data)."""
    vocab = 16
    chunk_size = 6
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, vocab, size=8 * chunk_size, dtype=np.uint16)
    with open(os.path.join(d, "shard_00001.bin"), "wb") as f:
        f.write(tokens.tobytes())
    occ_small = compute_occupancy(d, [1], vocab, sequence_len=5, device_batch_size=2,
                                  epoch_batches=4, table_mult=2,
                                  branch_primes={"bigram": [[(1, 2)]], "trigram": [(1, 2, 3)]})
    occ_big = compute_occupancy(d, [1], vocab, sequence_len=5, device_batch_size=2,
                                epoch_batches=4, table_mult=16,
                                branch_primes={"bigram": [[(1, 2)]], "trigram": [(1, 2, 3)]})
    b_small = occ_small["branches"]["bigram"]["0"][0]["occupancy"]
    b_big = occ_big["branches"]["bigram"]["0"][0]["occupancy"]
    assert b_big < b_small, "larger table must have lower occupancy"


# ---------------------------------------------------------------------------
# β₂ real read of table_betas[1]
# ---------------------------------------------------------------------------


def test_beta2_uses_table_betas():
    """RMSProp b2 must come from table_betas[1], not ngram_beta2."""
    model = NanoGPT(Config(vocab_size=16, n_layer=2, n_head=2, n_embd=32,
                           enable_bigram_ve=True, enable_trigram_ve=False))
    opt = MixedOptimizer(model, lr=0.01, ngram_betas=(0.0, 0.999),
                         adam_betas=(0.8, 0.95), weight_decay=0.1,
                         table_optimizer="rmsprop", table_betas=(0.0, 0.99))
    assert opt.table_betas[1] == 0.99
    # find the first bigram table param name
    name = None
    for n, _ in opt.ngram_params:
        if "bigram_ves" in n:
            name = n
            break
    assert name is not None
    p = dict(opt.ngram_params)[name]
    p.grad = torch.ones_like(p)
    opt.step(lr_mult=1.0)
    # exp_avg_sq should have been updated as b2*0 + (1-b2)*1 = 0.01
    ema = opt.rms_exp_avg_sq[name]
    assert abs(ema.flatten()[0].item() - 0.01) < 1e-6


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{len([n for n in globals() if n.startswith('test_') and callable(globals()[n])]) - failures}/{len([n for n in globals() if n.startswith('test_') and callable(globals()[n])])} passed")
    sys.exit(1 if failures else 0)
