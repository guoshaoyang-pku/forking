"""ngram-gap-lab · code/ngram_freq.py

Clean per-frequency-bin loss statistics for n-gram gap analysis.

This module implements a from-scratch frequency index and per-bin loss
accumulator, avoiding the legacy NgramHitTracker format incompatibility
(scalar-hash `keys` vs tuple `ctx_keys`) that blocked the old codebase.

Two pieces:
  1. GlobalFrequencyIndex  : offline scan of train tokens -> per-context hit count.
  2. FreqBinLossAccumulator: online accumulation of per-bin loss + token count,
     during train/val evaluation, using the same hash as the model's n-gram
     table (so bucket assignment matches the table row the model reads).

Buckets: novel (hit=0 in train), 1, 2, 3, 4, 5, 6-10, 11-20, 21-50,
         51-100, 101-200, 201-500, 501-1k, 1k-5k, 5k+.

Usage:
  # 1. build index once (offline)
  idx = GlobalFrequencyIndex.build(train_tokens, vocab_size)
  idx.save("data/runs/myrun/freq_index.npz")

  # 2. during eval, accumulate per-bin losses
  acc = FreqBinLossAccumulator(idx, n_head=6, head_dim=128)
  for inp, tgt in val_batches:
      per_token_loss = compute_per_token_loss(model, inp, tgt)  # (B, T)
      acc.update(inp, per_token_loss, branch="trigram")
  stats = acc.summary()   # dict of bucket -> {frac, mean_loss, total_contrib}
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Same hash prime families as train.py (must match for bucket assignment to
# match the table row the model reads).
_BASE_BIGRAM_PRIMES = [
    [(2654435761, 2246822519), (1013904223, 6291469)],
    [(374761393, 668265263), (3266489917, 104729)],
    [(1640531527, 97531), (48271, 40503)],
    [(16777619, 2166136261), (3432918353, 461845907)],
]
_BASE_TRIGRAM_PRIMES = [
    (16777619, 2166136261, 3432918353, 461845907, 2654435769, 1540483477),
    (3405403843, 2654435761, 2246822519, 1013904223, 6291469, 374761393),
    (668265263, 3266489917, 104729, 1640531527, 97531, 48271),
]

# Standard bucket edges (hit-count ranges). Novel = hit 0.
BUCKET_EDGES = [
    (0, 0, "novel"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 4, "4"),
    (5, 5, "5"),
    (6, 8, "6-8"),
    (9, 12, "9-12"),
    (13, 20, "13-20"),
    (21, 30, "21-30"),
    (31, 50, "31-50"),
    (51, 75, "51-75"),
    (76, 100, "76-100"),
    (101, 150, "101-150"),
    (151, 200, "151-200"),
    (201, 300, "201-300"),
    (301, 500, "301-500"),
    (501, 750, "501-750"),
    (751, 1000, "751-1k"),
    (1001, 2000, "1k-2k"),
    (2001, 5000, "2k-5k"),
    (5001, 10000, "5k-10k"),
    (10001, 10**9, "10k+"),
]


def _bucket_label(hit_count: int) -> str:
    for lo, hi, label in BUCKET_EDGES:
        if lo <= hit_count <= hi:
            return label
    return "5k+"


def _all_bucket_labels() -> list:
    return [label for _, _, label in BUCKET_EDGES]


# ---------------------------------------------------------------------------
# 1. Global frequency index (offline scan of train tokens)
# ---------------------------------------------------------------------------


class GlobalFrequencyIndex:
    """Per-context (bigram/trigram) hit count from one train epoch.

    Uses the actual token tuple (not the hash table row) as the key, so that
    novel contexts (never seen in train) are correctly identified. This is
    critical: the hash table has only vocab*64 rows, but the true context
    space is vocab^2 (bigram) or vocab^3 (trigram), so hash collisions would
    mask novel contexts if we used row ids as keys.

    Key encoding:
      bigram:  prev * vocab_size + cur           (range: vocab^2)
      trigram: prev2 * vocab^2 + prev * vocab + cur  (range: vocab^3)

    Stored format: npz with keys
      bigram_keys (int64), bigram_counts (int32),
      trigram_keys (int64), trigram_counts (int32),
      vocab_size (int64)
    """

    def __init__(self, bigram: dict, trigram: dict, vocab_size: int):
        self.bigram = bigram        # context_key -> count
        self.trigram = trigram
        self.vocab_size = vocab_size
        self._bigram_keys_sorted = None
        self._bigram_counts_sorted = None
        self._trigram_keys_sorted = None
        self._trigram_counts_sorted = None

    def _ensure_sorted(self):
        """Lazily build sorted key/count arrays if not set (e.g. built via
        `build`/`build_from_*` rather than `load`)."""
        if self._bigram_keys_sorted is None:
            b_keys = np.array(list(self.bigram.keys()), dtype=np.int64)
            b_counts = np.array(list(self.bigram.values()), dtype=np.int32)
            o = np.argsort(b_keys, kind="stable")
            self._bigram_keys_sorted = b_keys[o]
            self._bigram_counts_sorted = b_counts[o]
        if self._trigram_keys_sorted is None:
            t_keys = np.array(list(self.trigram.keys()), dtype=np.int64)
            t_counts = np.array(list(self.trigram.values()), dtype=np.int32)
            o = np.argsort(t_keys, kind="stable")
            self._trigram_keys_sorted = t_keys[o]
            self._trigram_counts_sorted = t_counts[o]

    @classmethod
    def build(cls, train_tokens: np.ndarray, vocab_size: int) -> "GlobalFrequencyIndex":
        """Scan train tokens (1D uint16/uint32 array) once, count all bigram/trigram contexts."""
        tokens = train_tokens.astype(np.int64)
        N = len(tokens)
        # bigram: (prev, cur) -> prev * vocab + cur
        prev = tokens[:-1]
        cur = tokens[1:]
        b_keys = prev * vocab_size + cur
        # trigram: (prev2, prev, cur) -> prev2 * vocab^2 + prev * vocab + cur
        prev2 = tokens[:-2]
        prev1 = tokens[1:-1]
        cur_t = tokens[2:]
        t_keys = prev2 * (vocab_size * vocab_size) + prev1 * vocab_size + cur_t
        # count unique keys
        b_unique, b_counts = np.unique(b_keys, return_counts=True)
        t_unique, t_counts = np.unique(t_keys, return_counts=True)
        bigram = {int(k): int(c) for k, c in zip(b_unique, b_counts)}
        trigram = {int(k): int(c) for k, c in zip(t_unique, t_counts)}
        return cls(bigram, trigram, vocab_size)

    @classmethod
    def build_from_shards(cls, data_dir: str, shard_ids: list, vocab_size: int) -> "GlobalFrequencyIndex":
        """Concatenate all train shards and build the index."""
        chunks = []
        for sid in shard_ids:
            path = os.path.join(data_dir, f"shard_{sid:05d}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing shard: {path}")
            buf = np.memmap(path, dtype=np.uint16, mode="r")
            chunks.append(np.array(buf))
        all_tokens = np.concatenate(chunks)
        return cls.build(all_tokens, vocab_size)

    @classmethod
    def build_from_chunks(cls, data_dir: str, shard_ids: list, vocab_size: int,
                          n_chunks: int, chunk_size: int) -> "GlobalFrequencyIndex":
        """Build the index from the FIRST `n_chunks` training chunks.

        Each chunk is processed independently with the model's exact context
        semantics (first / first-two positions repeated, never crossing a
        chunk boundary), so the index counts exactly the contexts the model
        actually sees in one epoch.  This fixes the old build_from_shards
        behaviour, which concatenated raw tokens and therefore counted
        cross-chunk pairs the model never sees.

        Chunk layout matches TokenizedShardDataset: each chunk is
        `chunk_size = sequence_len + 1` packed tokens; position t predicts
        token t+1, and the n-gram context at position t is built from
        chunk[max(0, t-1)] (bigram) / chunk[max(0, t-2)], chunk[max(0, t-1)]
        (trigram) plus the current token chunk[t].
        """
        bigram: dict = {}
        trigram: dict = {}
        v2 = vocab_size * vocab_size
        seen = 0
        T = chunk_size - 1
        for sid in shard_ids:
            if seen >= n_chunks:
                break
            path = os.path.join(data_dir, f"shard_{sid:05d}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing shard: {path}")
            buf = np.memmap(path, dtype=np.uint16, mode="r")
            n = len(buf) // chunk_size
            take = min(n, n_chunks - seen)
            tokens = np.array(buf[:take * chunk_size], dtype=np.int64).reshape(take, chunk_size)
            cur = tokens[:, :T]                                        # (M, T)
            prev = np.concatenate([tokens[:, :1], tokens[:, :T - 1]], axis=1)   # (M, T)
            prev2 = np.concatenate([tokens[:, :2], tokens[:, :T - 2]], axis=1)  # (M, T)
            b_keys = prev * vocab_size + cur
            t_keys = prev2 * v2 + prev * vocab_size + cur
            for b_u, b_c in zip(*np.unique(b_keys, return_counts=True)):
                key = int(b_u)
                bigram[key] = bigram.get(key, 0) + int(b_c)
            for t_u, t_c in zip(*np.unique(t_keys, return_counts=True)):
                key = int(t_u)
                trigram[key] = trigram.get(key, 0) + int(t_c)
            seen += take
        return cls(bigram, trigram, vocab_size)

    def save(self, path: str):
        b_keys = np.array(list(self.bigram.keys()), dtype=np.int64)
        b_counts = np.array(list(self.bigram.values()), dtype=np.int32)
        t_keys = np.array(list(self.trigram.keys()), dtype=np.int64)
        t_counts = np.array(list(self.trigram.values()), dtype=np.int32)
        np.savez(path, bigram_keys=b_keys, bigram_counts=b_counts,
                 trigram_keys=t_keys, trigram_counts=t_counts,
                 vocab_size=np.array([self.vocab_size]))

    @classmethod
    def load(cls, path: str) -> "GlobalFrequencyIndex":
        d = np.load(path)
        bigram = {int(k): int(c) for k, c in zip(d["bigram_keys"], d["bigram_counts"])}
        trigram = {int(k): int(c) for k, c in zip(d["trigram_keys"], d["trigram_counts"])}
        vocab_size = int(d["vocab_size"][0])
        obj = cls(bigram, trigram, vocab_size)
        # pre-sort key arrays for vectorized searchsorted lookup (see hit_count_tensor)
        b_keys = np.asarray(d["bigram_keys"], dtype=np.int64)
        t_keys = np.asarray(d["trigram_keys"], dtype=np.int64)
        b_counts = np.asarray(d["bigram_counts"], dtype=np.int32)
        t_counts = np.asarray(d["trigram_counts"], dtype=np.int32)
        b_order = np.argsort(b_keys, kind="stable")
        t_order = np.argsort(t_keys, kind="stable")
        obj._bigram_keys_sorted = b_keys[b_order]
        obj._bigram_counts_sorted = b_counts[b_order]
        obj._trigram_keys_sorted = t_keys[t_order]
        obj._trigram_counts_sorted = t_counts[t_order]
        return obj

    def hit_count(self, branch: str, row: int) -> int:
        """Lookup hit count for a given (branch, row). Returns 0 if novel."""
        tbl = self.bigram if branch == "bigram" else self.trigram
        return tbl.get(int(row), 0)

    def hit_count_numpy(self, branch: str, key_np: np.ndarray) -> np.ndarray:
        """Vectorized hit count lookup on numpy context keys (worker/CPU path).

        Exact-match semantics identical to hit_count_tensor: novel keys -> 0.
        Uses np.searchsorted over the pre-sorted key arrays.
        """
        self._ensure_sorted()
        if branch == "bigram":
            keys_sorted = self._bigram_keys_sorted
            counts_sorted = self._bigram_counts_sorted
        else:
            keys_sorted = self._trigram_keys_sorted
            counts_sorted = self._trigram_counts_sorted
        k = np.ascontiguousarray(key_np, dtype=np.int64).ravel()
        pos = np.searchsorted(keys_sorted, k)
        np.clip(pos, 0, len(keys_sorted) - 1, out=pos)
        hit = keys_sorted[pos] == k
        return np.where(hit, counts_sorted[pos], 0).astype(np.int64)

    def hit_count_tensor(self, branch: str, rows: torch.Tensor) -> torch.Tensor:
        """Vectorized hit count lookup. rows: (B,T) long tensor -> (B,T) int32.

        Uses np.searchsorted over pre-sorted key arrays (built in `load`), which
        is a fully vectorized numpy operation — much faster than the previous
        np.vectorize + Python dict.get per-element loop (which dominated
        freq-bin eval wall time, ~20s per eval on 28.8B runs)."""
        self._ensure_sorted()
        if branch == "bigram":
            keys_sorted = self._bigram_keys_sorted
            counts_sorted = self._bigram_counts_sorted
        else:
            keys_sorted = self._trigram_keys_sorted
            counts_sorted = self._trigram_counts_sorted
        rows_np = rows.detach().cpu().numpy().reshape(-1).astype(np.int64)
        pos = np.searchsorted(keys_sorted, rows_np)
        pos = np.clip(pos, 0, len(keys_sorted) - 1)
        hit = keys_sorted[pos] == rows_np
        counts = np.where(hit, counts_sorted[pos], 0).astype(np.int32)
        return torch.from_numpy(counts.reshape(rows.shape)).to(rows.device)


# ---------------------------------------------------------------------------
# 2. Per-frequency-bin loss accumulator (online, during eval)
# ---------------------------------------------------------------------------


class FreqBinLossAccumulator:
    """Accumulates per-token loss grouped by n-gram context hit-count bucket.

    For each batch (inp, tgt, per_token_loss):
      1. Compute the n-gram context row for each position (using layer-0 hash).
      2. Lookup hit count from GlobalFrequencyIndex.
      3. Map hit count -> bucket label.
      4. Accumulate loss_sum and token_count per bucket.

    Final summary gives per-bucket:
      frac          = token_count / total_tokens
      mean_loss     = loss_sum / token_count
      total_contrib = frac * mean_loss  (contribution to overall loss)
    """

    def __init__(self, freq_index: GlobalFrequencyIndex, vocab_size: int,
                 branch: str = "trigram"):
        self.freq_index = freq_index
        self.vocab_size = vocab_size
        self.branch = branch  # "bigram" or "trigram"
        self._loss_sum = {label: 0.0 for label in _all_bucket_labels()}
        self._token_count = {label: 0 for label in _all_bucket_labels()}
        self._total_tokens = 0

    def _compute_keys(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute context keys for each position. Returns (B,T) long.

        Key encoding (must match GlobalFrequencyIndex.build):
          bigram:  prev * vocab + cur
          trigram: prev2 * vocab^2 + prev * vocab + cur
        """
        B, T = inp.size()
        if self.branch == "bigram":
            prev = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
            keys = prev.long() * self.vocab_size + inp.long()
        else:  # trigram
            prev2 = torch.cat([inp[:, :2], inp[:, :-2]], dim=1)
            prev1 = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
            keys = (prev2.long() * (self.vocab_size * self.vocab_size)
                    + prev1.long() * self.vocab_size + inp.long())
        return keys

    def update(self, inp: torch.Tensor, per_token_loss: torch.Tensor):
        """Accumulate one batch. inp: (B,T), per_token_loss: (B,T) float."""
        keys = self._compute_keys(inp)  # (B,T) long
        hits = self.freq_index.hit_count_tensor(self.branch, keys)  # (B,T) int32
        self.update_numpy(hits.cpu().numpy(), per_token_loss)

    def update_numpy(self, hits_np: np.ndarray, loss_np: np.ndarray):
        """Accumulate from precomputed hit counts (torch-free, worker path)."""
        hits_flat = np.asarray(hits_np).astype(np.int64).ravel()
        loss_flat = np.asarray(loss_np, dtype=np.float64).ravel()
        for lo, hi, label in BUCKET_EDGES:
            mask = (hits_flat >= lo) & (hits_flat <= hi)
            n = int(mask.sum())
            if n > 0:
                self._loss_sum[label] += float(loss_flat[mask].sum())
                self._token_count[label] += n
        self._total_tokens += loss_flat.size

    def summary(self) -> dict:
        out = {}
        for label in _all_bucket_labels():
            n = self._token_count[label]
            if n > 0:
                mean_loss = self._loss_sum[label] / n
                frac = n / max(1, self._total_tokens)
                out[label] = {
                    "token_count": n,
                    "frac": frac,
                    "mean_loss": mean_loss,
                    "total_contrib": frac * mean_loss,
                }
            else:
                out[label] = {"token_count": 0, "frac": 0.0, "mean_loss": 0.0, "total_contrib": 0.0}
        return out

    def save_json(self, path: str, meta: Optional[dict] = None):
        d = {"branch": self.branch, "total_tokens": self._total_tokens}
        if meta:
            d["meta"] = meta
        d["buckets"] = self.summary()
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


class ExactFreqLossAccumulator:
    """Accumulate per-exact-frequency loss statistics (no wide buckets).

    For each position we compute the exact n-gram context key (identical to
    the model's hashing input), look up its train hit count f in the
    GlobalFrequencyIndex, and accumulate token count / loss sum / loss^2 sum
    keyed by that exact f.  f=0 (novel context) is reported separately with
    only its mean loss (no gap is defined for novel contexts).

    Provides both marginal summaries (token-weighted mean loss per f) and the
    raw sufficient statistics needed for context-matched gap analysis.
    """

    def __init__(self, freq_index: GlobalFrequencyIndex, vocab_size: int,
                 branch: str = "trigram"):
        self.freq_index = freq_index
        self.vocab_size = vocab_size
        self.branch = branch
        # exact f -> dict(token_count, loss_sum, loss_sq_sum, distinct_keys)
        self._stats: dict = {}
        self._total_tokens = 0

    def _compute_keys(self, inp: torch.Tensor) -> torch.Tensor:
        """Context key per position, matching FreqBinLossAccumulator semantics."""
        return FreqBinLossAccumulator._compute_keys(self, inp)

    def update(self, inp: torch.Tensor, per_token_loss: torch.Tensor):
        """Accumulate one batch. inp: (B,T) token ids, per_token_loss: (B,T)."""
        keys = self._compute_keys(inp)                    # (B,T) long
        key_np = keys.cpu().numpy().ravel()
        loss_np = per_token_loss.detach().cpu().numpy().astype(np.float64).ravel()
        self.update_numpy(key_np, loss_np)

    def update_numpy(self, key_np: np.ndarray, loss_np: np.ndarray):
        """Accumulate from precomputed context keys (torch-free, worker path).

        Vectorized accumulation: hit counts via searchsorted, loss sums via
        np.bincount per exact f; distinct-context counts via a single lexsort
        over (f, key) — the whole batch is handled in numpy.
        """
        key_np = np.ascontiguousarray(key_np).ravel()
        loss_np = np.asarray(loss_np, dtype=np.float64).ravel()
        freq_np = self.freq_index.hit_count_numpy(self.branch, key_np)
        self._total_tokens += len(loss_np)

        # distinct (f, key) pairs -> distinct contexts per f
        order = np.lexsort((key_np, freq_np))
        sf = freq_np[order]
        sk = key_np[order]
        # mark start of each new (f, key) group: length len(sf)+1 sentinel
        is_start = np.ones(len(sf) + 1, dtype=bool)
        is_start[1:-1] = (sf[1:] != sf[:-1]) | (sk[1:] != sk[:-1])
        is_start[-1] = False
        # per-f token counts and distinct-context counts
        f_groups, starts = np.unique(sf, return_index=True)
        token_counts = np.add.reduceat(np.ones(len(sf), dtype=np.int64),
                                       starts) if len(sf) else np.array([], dtype=np.int64)
        # distinct contexts per f = count of starts within [starts[i], starts[i+1])
        dc_counts = np.add.reduceat(is_start.astype(np.int64)[:-1],
                                    starts) if len(sf) else np.array([], dtype=np.int64)
        # per-f loss sums / sq sums via bincount (f may be large; use sparse dict path)
        max_f = int(freq_np.max()) if len(freq_np) else 0
        loss_sums = np.bincount(freq_np, weights=loss_np, minlength=max_f + 1)
        loss_sq_sums = np.bincount(freq_np, weights=loss_np * loss_np, minlength=max_f + 1)
        for f, tc, dc in zip(f_groups.tolist(), token_counts.tolist(), dc_counts.tolist()):
            st = self._stats.setdefault(int(f), {
                "token_count": 0, "distinct_contexts": 0,
                "loss_sum": 0.0, "loss_sq_sum": 0.0,
            })
            st["token_count"] += int(tc)
            st["distinct_contexts"] += int(dc)
            st["loss_sum"] += float(loss_sums[int(f)])
            st["loss_sq_sum"] += float(loss_sq_sums[int(f)])

    def summary(self, min_contexts: int = 0) -> dict:
        """Return per-exact-f summary dict (f -> stats with mean loss)."""
        out = {}
        for f, st in sorted(self._stats.items()):
            n = st["token_count"]
            if n <= 0:
                continue
            entry = {
                "f": int(f),
                "token_count": n,
                "distinct_contexts": st["distinct_contexts"],
                "mean_loss": st["loss_sum"] / n,
                "loss_sum": st["loss_sum"],
                "loss_sq_sum": st["loss_sq_sum"],
            }
            if min_contexts and st["distinct_contexts"] < min_contexts:
                entry["excluded"] = f"distinct_contexts < {min_contexts}"
            out[int(f)] = entry
        return out

    def clean(self) -> dict:
        """Return JSON-serializable per-f dict."""
        return {int(f): {k: v for k, v in st.items()} for f, st in self._stats.items()}


# ---------------------------------------------------------------------------
# 3. Convenience: compute per-token loss from model
# ---------------------------------------------------------------------------


def compute_context_keys(inp: torch.Tensor, vocab_size: int):
    """Module-level context keys for one batch: returns (bigram_keys, trigram_keys),
    each (B,T) long, on the same device as inp. Encoding matches
    GlobalFrequencyIndex.build / FreqBinLossAccumulator._compute_keys."""
    prev = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
    b_keys = prev.long() * vocab_size + inp.long()
    prev2 = torch.cat([inp[:, :2], inp[:, :-2]], dim=1)
    prev1 = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
    t_keys = (prev2.long() * (vocab_size * vocab_size)
              + prev1.long() * vocab_size + inp.long())
    return b_keys, t_keys


def compute_shared_from_keys(train_key_np, train_loss_np, val_key_np, val_loss_np,
                             freq_index: GlobalFrequencyIndex, branch: str) -> dict:
    """Context-matched gap statistics from precomputed keys/losses (torch-free).

    Semantics identical to train.py's compute_shared_contexts: a context is
    represented by its LAST occurrence within each probe set; shared contexts
    are those present in both sets; per-exact-f train/val mean and gap.
    """
    train_key_np = np.ascontiguousarray(train_key_np).ravel()
    train_loss_np = np.asarray(train_loss_np, dtype=np.float64).ravel()
    val_key_np = np.ascontiguousarray(val_key_np).ravel()
    val_loss_np = np.asarray(val_loss_np, dtype=np.float64).ravel()
    # last occurrence per context: unique on the reversed array yields the
    # index of the first occurrence in reversed order == last in original.
    tu, t_last = np.unique(train_key_np[::-1], return_index=True)
    train_last_loss = train_loss_np[::-1][t_last]
    vu, v_last = np.unique(val_key_np[::-1], return_index=True)
    val_last_loss = val_loss_np[::-1][v_last]
    shared, t_idx, v_idx = np.intersect1d(tu, vu, assume_unique=True, return_indices=True)
    per_f = {}
    if len(shared):
        f_arr = freq_index.hit_count_numpy(branch, shared)
        train_shared = train_last_loss[t_idx]
        val_shared = val_last_loss[v_idx]
        for f in np.unique(f_arr):
            m = f_arr == f
            n = int(m.sum())
            t_sum = float(train_shared[m].sum())
            v_sum = float(val_shared[m].sum())
            per_f[int(f)] = {
                "f": int(f),
                "shared_contexts": n,
                "train_mean": t_sum / n,
                "val_mean": v_sum / n,
                "gap": (v_sum - t_sum) / n,
            }
    return {"shared_total": int(len(shared)), "per_f": per_f, "branch": branch}


@torch.no_grad()
def compute_per_token_loss(model, inp: torch.Tensor, tgt: torch.Tensor,
                           amp_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Returns (B, T) per-token cross-entropy loss (no reduction).

    amp_dtype: optional autocast dtype (e.g. torch.bfloat16) for the model
    forward — matches evaluate_val's numerics. The cross-entropy itself always
    runs in fp32 on upcast logits, so the returned loss is fp32 either way.
    """
    if amp_dtype is not None and inp.is_cuda:
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(inp)
    else:
        logits = model(inp)
    loss = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        tgt.view(-1), ignore_index=-1, reduction="none").view(tgt.size())
    return loss


# ---------------------------------------------------------------------------
# 4. CLI: build index from shards
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Logit-stats f-buckets (confidence polarization probe, §5.2 supplement)
# ---------------------------------------------------------------------------

# Coarser buckets than BUCKET_EDGES, tuned for confidence-polarization analysis:
# low-frequency contexts are the predicted locus of strongest polarization.
LOGIT_F_BUCKETS = [
    (0, 0, "f0"),
    (1, 1, "f1"),
    (2, 3, "f2-3"),
    (4, 7, "f4-7"),
    (8, 15, "f8-15"),
    (16, 63, "f16-63"),
    (64, 255, "f64-255"),
    (256, 10**9, "f256+"),
]


def logit_f_bucket_label(hit_count: int) -> str:
    for lo, hi, label in LOGIT_F_BUCKETS:
        if lo <= hit_count <= hi:
            return label
    return "f256+"


def all_logit_f_bucket_labels() -> list:
    return [label for _, _, label in LOGIT_F_BUCKETS]


class LogitStatsAccumulator:
    """Aggregate per-token confidence statistics by (branch, f_bucket, seen/novel).

    Inputs (all numpy, torch-free):
      - key_np:     context keys (int64), shape (N,)
      - hits_np:    train hit counts for each key (int64), shape (N,)
      - entropy_np: per-token softmax entropy H(p) in nats (float64), shape (N,)
      - margin_np:  top1_prob - top2_prob (float64), shape (N,)
      - ptrue_np:   p(y_true) under the model softmax (float64), shape (N,)

    Accumulates sum/count for each (bucket, seen/novel) cell so the caller can
    compute means after the batch. seen = (hit > 0); novel = (hit == 0). This
    matches the exact-freq pipeline's definition of novel contexts.
    """

    def __init__(self, branch: str):
        self.branch = branch
        labels = all_logit_f_bucket_labels()
        self._sum_h = {sn: {lb: 0.0 for lb in labels} for sn in ("seen", "novel")}
        self._sum_m = {sn: {lb: 0.0 for lb in labels} for sn in ("seen", "novel")}
        self._sum_p = {sn: {lb: 0.0 for lb in labels} for sn in ("seen", "novel")}
        self._count = {sn: {lb: 0 for lb in labels} for sn in ("seen", "novel")}

    def update_numpy(self, key_np, hits_np, entropy_np, margin_np, ptrue_np):
        hits = np.asarray(hits_np, dtype=np.int64).ravel()
        ent = np.asarray(entropy_np, dtype=np.float64).ravel()
        mar = np.asarray(margin_np, dtype=np.float64).ravel()
        ptr = np.asarray(ptrue_np, dtype=np.float64).ravel()
        # bucket assignment
        bucket_idx = np.zeros(len(hits), dtype=np.int32)
        for i, (lo, hi, _) in enumerate(LOGIT_F_BUCKETS):
            mask = (hits >= lo) & (hits <= hi)
            bucket_idx[mask] = i
        # clamp any overflow to last bucket
        bucket_idx = np.clip(bucket_idx, 0, len(LOGIT_F_BUCKETS) - 1)
        seen_mask = hits > 0
        for sn, m in (("seen", seen_mask), ("novel", ~seen_mask)):
            if not np.any(m):
                continue
            bi = bucket_idx[m]
            e = ent[m]; mg = mar[m]; p = ptr[m]
            for j, (_, _, lb) in enumerate(LOGIT_F_BUCKETS):
                sel = bi == j
                n = int(sel.sum())
                if n == 0:
                    continue
                self._count[sn][lb] += n
                self._sum_h[sn][lb] += float(e[sel].sum())
                self._sum_m[sn][lb] += float(mg[sel].sum())
                self._sum_p[sn][lb] += float(p[sel].sum())

    def summary(self) -> dict:
        out = {}
        for sn in ("seen", "novel"):
            d = {}
            for _, _, lb in LOGIT_F_BUCKETS:
                n = self._count[sn][lb]
                if n > 0:
                    d[lb] = {
                        "count": n,
                        "mean_entropy": self._sum_h[sn][lb] / n,
                        "mean_margin": self._sum_m[sn][lb] / n,
                        "mean_ptrue": self._sum_p[sn][lb] / n,
                    }
                else:
                    d[lb] = {"count": 0, "mean_entropy": 0.0,
                             "mean_margin": 0.0, "mean_ptrue": 0.0}
            out[sn] = d
        return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_shards", required=True, help="comma-separated shard ids")
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--out", required=True, help="output npz path")
    args = parser.parse_args()
    shard_ids = [int(x) for x in args.train_shards.split(",") if x.strip()]
    idx = GlobalFrequencyIndex.build_from_shards(args.data_dir, shard_ids, args.vocab_size)
    idx.save(args.out)
    print(f"[ngram_freq] saved index to {args.out}")
    print(f"[ngram_freq] bigram unique rows: {len(idx.bigram)}")
    print(f"[ngram_freq] trigram unique rows: {len(idx.trigram)}")
