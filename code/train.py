"""ngram-gap-lab · code/train.py

Minimal clean reproduction of n-gram-value-memory-induced replay-specific
train/val gap on vanilla nanoGPT.

Only three things are kept from the larger predecessor codebase:
  1. vanilla nanoGPT (NanoGPTOriginal) — Karpathy-style transformer.
  2. n-gram value table (bigram + trigram, hash + embedding + gate).
  3. Three injection points (v / y / input) and RMSProp(table)/AdamW(backbone)
     optimizer grouping.

Removed (proven unnecessary for gap):
  current shell, Muon, RoPE, RMSNorm, untied embedding, split QKV,
  x0 residual, layer pool, head gate, softcap, full theory-obs system.

The optimizer/evaluation defaults follow the standard contract, but a new
main-line run must explicitly select clean-table capacity with
--bigram_clean_table / --trigram_clean_table (or a perfect-map condition).
The legacy --table_mult path is retained only to read or rerun historical
experiments; it is not a new-baseline default.

Outputs JSONL logs under data/runs_fixed/<run_id>/:
  - train_log.jsonl   : one line per step {step, train_loss, val_loss, epoch, ...}
  - summary.json      : final gap + config snapshot
  - table_norm.jsonl  : periodic table param RMS (every TABLE_NORM_INTERVAL steps)

Usage:
  python train.py --run_id myrun --injection_position input --steps 1000 \
      --bigram_clean_table <R_bigram> --trigram_clean_table <R_trigram>
  (for the main-line command, see cluster/run_baseline.sh)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import contextlib
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # model
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 768
    sequence_len: int = 2048
    dropout: float = 0.0
    bias: bool = True
    # n-gram value table
    enable_nanogpt_ngram_ve: bool = True
    enable_unigram_ve: bool = False
    enable_bigram_ve: bool = True
    enable_trigram_ve: bool = True
    enable_fourgram_ve: bool = False
    nanogpt_ngram_injection_position: str = "input"  # v | y | input
    nanogpt_inject_layers: str = ""   # v/y only: comma layer ids for multi-layer
                                      # injection (one independent table per layer,
                                      # per-layer hash primes). "" = default
                                      # (single layer ngram_layers[0]).
    # optimizer
    nanogpt_adam_lr: float = 0.004
    ngram_table_betas: tuple = (0.0, 0.99)
    table_optimizer: str = "rmsprop"  # rmsprop | adamw | sgd
    table_lr_scale: float = 2.0       # multiplier on the n-gram table LR
    table_betas: tuple = (0.0, 0.99)  # (beta1, beta2) for adamw; beta1 = momentum for sgd
    table_mult: int = 64              # legacy multi-layer/two-hash path only
    bigram_clean_table: int = 0       # clean single-table rows R (0=off); SSOT: clean-table-rework.md
    trigram_clean_table: int = 0      # clean single-table rows R for trigram (0=off); same SSOT
    bigram_table_dim: int = 0         # clean-table row width d (0=n_embd); frozen zero-pad projection back to n_embd
    trigram_table_dim: int = 0        # same rule for the trigram branch
    bigram_perfect_map: str = ""      # npz with packed-bigram->row map: collision-free table (rows = n_distinct+1)
    bigram_single_layer: bool = False # restrict bigram table to the first ngram layer only
    ngram_extra_orders: tuple = ()    # v2: generic clean single tables for orders n >= 4 (input only)
    ngram_clean_table: int = 0        # rows R of every generic order-n table
    ngram_dict_topk: int = 0          # v2.1: >0 = no-collision dictionary mode (see load_ngram_dictionaries)
    ngram_keep_rows: str = "dict"     # with ngram_dict_topk: dict = unique rows | hash = hash rows under the same top-K membership (coverage control)
    intervention: str = "none"        # none | freeze_table | freeze_backbone | freeze_both | hash_reseed | mask_low_freq | mask_high_freq | random_context_loss_mask
    intervention_epoch: int = 1       # fire when 0-indexed epoch reaches this value (1 = start of epoch 2)
    intervention_epochs: tuple = ()   # extra 0-indexed due epochs for multi-fire interventions (e.g. (1, 2))
    intervention_freq_threshold: int = 200
    intervention_low_inclusive: bool = False  # mask_low masks f <= threshold (default f < threshold)
    intervention_mask_frac: float = 0.02  # random_context_loss_mask: fraction of distinct trigram context types to exclude from the loss
    context_eval_ab: bool = False  # log per-eval A/B context gap contributions to context_gap.jsonl
    context_set_mode: str = "type"  # A/B selection space: "type" (distinct trigram types) | "row" (trigram table rows)
    adam_betas: tuple = (0.8, 0.95)
    weight_decay: float = 0.1
    # training
    seed: int = 42
    max_steps: int = 1000
    device_batch_size: int = 72
    total_batch_size: int = 147456
    val_interval_steps: int = 10
    val_steps: list = field(default_factory=list)  # exact steps to eval val (overrides interval if set)
    val_batches: int = 4
    table_norm_interval_steps: int = 10
    warmup_steps: int = 100       # step 1..100: warmup_start_lr_mult -> 1.0x
    warmup_start_lr_mult: float = 0.25
    cosine_min_lr_mult: float = 0.05  # terminal multiplier for warmup_cosine
    warmdown_ratio: float = 0.65  # historical warmdown parameter
    lr_schedule_epochs: int = 0    # >0: anchor LR schedule to epoch count (ignores max_steps)
    lr_schedule: str = "warmup_constant"  # new standard | constant/warmdown historical controls
    # data (paths)
    data_dir: str = ""          # directory with train.bin / val.bin
    train_shards: list = field(default_factory=list)  # list of shard indices for train
    val_shards: list = field(default_factory=list)    # list of shard indices for val
    data_mode: str = "fixed"    # fixed = deterministic epoch replay
    data_seed: int = 42
    epoch_batches: int = 0      # >0: fix one epoch to exactly this many device batches
                                # (nested-prefix length control); 0 = use full shard length
    train_max_tokens: int = 0   # >0: train pool = first P content tokens of the train
                                # shard stream (token-granular nested-prefix truncation,
                                # rounded down to whole device batches); val unaffected
    replay_mix_passes: int = 0  # >0: one globally chunk-shuffled stream of N passes
                                # (shuffle experiment mixed arm); overrides epoch replay
    # infra: precomputed n-gram row index / large vocab / precision (plan-8)
    row_index: bool = False     # consume offline row indices (pure-gather fast path)
    row_index_dir: str = ""     # where shard_<id>.{b,t}rows.u32.bin live (default data_dir)
    token_dtype: str = "u16"    # u16 | u32 (u32 required for vocab > 65535)
    ce_chunk_positions: int = 0  # chunked CE: 0=auto, -1=off, >0=positions per chunk
    static_low_mask: int = 0    # >0: apply precomputed validity bits (f >= t) from the
                                # offline index (prevention arm; plan-8 G3; needs --row_index)
    static_low_mask_keep_novel: bool = True  # novel (f==0) stays enabled in the static mask
    fp8_torchao: bool = False   # torchao float8 on backbone Linears only (table untouched)
    # output
    out_dir: str = ""


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Alternating VE layers."""
    return layer_idx % 2 == (n_layer - 1) % 2


def _parse_inject_layers(config) -> "set[int] | None":
    """--inject_layers for v/y/engram multi-layer injection; None = default layout."""
    s = getattr(config, "nanogpt_inject_layers", "") or ""
    if not s.strip():
        return None
    if config.nanogpt_ngram_injection_position not in {"v", "y", "engram"}:
        raise ValueError("--inject_layers applies to v/y/engram injection")
    layers = {int(t) for t in s.split(",") if t.strip()}
    if not layers or min(layers) < 0 or max(layers) >= config.n_layer:
        raise ValueError(f"--inject_layers out of range [0,{config.n_layer}): {s!r}")
    return layers


# ---------------------------------------------------------------------------
# 1. Model: vanilla nanoGPT + n-gram value table
# ---------------------------------------------------------------------------

# Hash prime families (decorrelated across layers). First 4 are historical;
# deeper models deterministically extend via expand_bigram_hash_primes.
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


def _next_odd_hash_constant(seed, used):
    value = (seed & 0xFFFFFFFF) | 1
    while value in used:
        value = ((value + 0x9E3779B2) & 0xFFFFFFFF) | 1
    used.add(value)
    return value


def expand_bigram_hash_primes(base, count):
    if count <= len(base):
        return base[:count]
    expanded = list(base)
    used = {c for fam in expanded for pair in fam for c in pair}
    salts = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F)
    while len(expanded) < count:
        fi = len(expanded)
        b = (fi + 1) * 0xA24BAED5
        expanded.append([
            (_next_odd_hash_constant(b ^ salts[0], used),
             _next_odd_hash_constant((b + 0x632BE59B) ^ salts[1], used)),
            (_next_odd_hash_constant((b + 0x85157AF5) ^ salts[2], used),
             _next_odd_hash_constant((b + 0xC2B2AE35) ^ salts[3], used)),
        ])
    return expanded


def ngram_hash_constants(n: int) -> tuple:
    """Per-position odd 32-bit multipliers of the generic order-n table hash
    rows = XOR_k (shift_k(idx) * c_k) % R (v2, n >= 4)."""
    used = set()
    return tuple(
        _next_odd_hash_constant((n * 0x9E3779B1) ^ ((k + 1) * 0x85EBCA77), used)
        for k in range(n)
    )


# row-slice threshold for the RMSProp table update (see _table_rmsprop_step);
# module-level so test_ngram_keep_cpu.py can exercise the sliced path on CPU
_TABLE_SLICE_BYTES = 2 << 30
_TABLE_SLICE_ROWS = 1 << 18

# splitmix64 finalizer constants as int64 bit patterns (== ngram_freq.py's
# numpy uint64 version; used by the dictionary-mode key function, whose
# bit-identity with context_keys_np is covered by test_ngram_dict_cpu.py)
_MIX_C1_I64 = 0xBF58476D1CE4E5B9 - (1 << 64)
_MIX_C2_I64 = 0x94D049BB133111EB - (1 << 64)
_SPLITMIX_SEED_I64 = 0x9E3779B97F4A7C15 - (1 << 64)


def _mix64_i64(z: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer on int64 bit patterns.

    torch int64 >> is arithmetic, numpy uint64 >> is logical — mask off the
    sign-extended top bits to recover the logical shift; multiplies wrap
    mod 2^64 in both, so results are bit-identical to ngram_freq._mix64."""
    z = z ^ ((z >> 30) & ((1 << 34) - 1))
    z = z * _MIX_C1_I64
    z = z ^ ((z >> 27) & ((1 << 37) - 1))
    z = z * _MIX_C2_I64
    return z ^ ((z >> 31) & ((1 << 33) - 1))


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """nanoGPT fused-c_attn attention with optional n-gram value injection.

    Injection positions:
      v     : add gated n-gram value to V before attention  (ResFormer-style)
      y     : add gated n-gram value to attention output y  (post-attn residual)
      engram: add gated n-gram residual to the normalized hidden state before QKV
      input : handled in GPT.forward (over-encoding to wte); gates not used here
    """

    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.ngram_injection_position = config.nanogpt_ngram_injection_position
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels = 32
        use_ngram = config.enable_nanogpt_ngram_ve
        use_layer_ve = has_ve(layer_idx, config.n_layer)
        inject = _parse_inject_layers(config)
        if inject is not None:
            bigram_gate_layers = trigram_gate_layers = inject
        else:
            ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
            bigram_gate_layers = set(ve_layers)
            trigram_gate_layers = (
                {ve_layers[0], ve_layers[-2], ve_layers[-1]}
                if len(ve_layers) >= 2 else {ve_layers[-1]}
            )
        # gates are used by value, output-residual, and Engram-like routes.
        # Input over-encoding remains the deliberately ungated baseline.
        need_gate = use_ngram and config.nanogpt_ngram_injection_position in {"v", "y", "engram"}
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
            if need_gate and config.enable_unigram_ve and use_layer_ve else None
        )
        self.bigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
            if need_gate and config.enable_bigram_ve and layer_idx in bigram_gate_layers else None
        )
        self.trigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
            if need_gate and config.enable_trigram_ve and layer_idx in trigram_gate_layers else None
        )
        self.fourgram_gate = None  # fourgram not supported in minimal version

    def _add_value_residual(self, branch, v_heads, residual, gate, x, ch_start, heads):
        residual = residual.view(x.size(0), x.size(1), heads, self.head_dim)
        gate_input = x[..., ch_start: ch_start + self.ve_gate_channels]
        value_gate = 2 * torch.sigmoid(gate(gate_input)).to(residual.dtype)
        gated = value_gate.unsqueeze(-1) * residual
        return v_heads + gated

    def _add_ngram_residuals(self, v_heads, x, ve, bigram_ve, trigram_ve, heads):
        if ve is not None:
            v_heads = self._add_value_residual("unigram", v_heads, ve, self.ve_gate, x, 0, heads)
        if bigram_ve is not None:
            v_heads = self._add_value_residual(
                "bigram", v_heads, bigram_ve, self.bigram_gate, x, self.ve_gate_channels, heads)
        if trigram_ve is not None:
            v_heads = self._add_value_residual(
                "trigram", v_heads, trigram_ve, self.trigram_gate, x, 2 * self.ve_gate_channels, heads)
        return v_heads

    def _compute_ngram_residual_flat(self, x, ve, bigram_ve, trigram_ve, heads):
        """Post-attention (y) injection: sum gated residuals as (B,T,C)."""
        r = torch.zeros_like(x)
        if ve is not None:
            v = ve.view(x.size(0), x.size(1), heads, self.head_dim)
            g = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])).to(v.dtype)
            r = r + (g.unsqueeze(-1) * v).view_as(x)
        if bigram_ve is not None:
            v = bigram_ve.view(x.size(0), x.size(1), heads, self.head_dim)
            g = 2 * torch.sigmoid(self.bigram_gate(
                x[..., self.ve_gate_channels:2 * self.ve_gate_channels])).to(v.dtype)
            r = r + (g.unsqueeze(-1) * v).view_as(x)
        if trigram_ve is not None:
            v = trigram_ve.view(x.size(0), x.size(1), heads, self.head_dim)
            g = 2 * torch.sigmoid(self.trigram_gate(
                x[..., 2 * self.ve_gate_channels:3 * self.ve_gate_channels])).to(v.dtype)
            r = r + (g.unsqueeze(-1) * v).view_as(x)
        return r

    def forward(self, x, ve=None, bigram_ve=None, trigram_ve=None):
        B, T, C = x.size()
        if self.ngram_injection_position == "engram":
            # Minimal Engram-like route: lookup tables are owned by the
            # selected intermediate blocks, gated by the current hidden state,
            # and added before the block's QKV projections.
            x = x + self._compute_ngram_residual_flat(
                x, ve, bigram_ve, trigram_ve, self.n_head
            )
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        v_heads = v.view(B, T, self.n_head, self.head_dim)
        if self.ngram_injection_position == "v":
            v_heads = self._add_ngram_residuals(v_heads, x, ve, bigram_ve, trigram_ve, self.n_head)
        v = v_heads.reshape(B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if self.ngram_injection_position == "y":
            ngram_res = self._compute_ngram_residual_flat(x, ve, bigram_ve, trigram_ve, self.n_head)
            y = y + ngram_res
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, ve=None, bigram_ve=None, trigram_ve=None):
        x = x + self.attn(self.ln_1(x), ve=ve, bigram_ve=bigram_ve, trigram_ve=trigram_ve)
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """vanilla nanoGPT + n-gram value table (bigram/trigram)."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.transformer["wpe"] = nn.Embedding(config.sequence_len, config.n_embd)
        self.transformer["ln_f"] = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tied embeddings (matches baseline_input setting)
        self.lm_head.weight = self.transformer.wte.weight

        ngram_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, config.n_embd)
            for i in ngram_layers
            if config.enable_nanogpt_ngram_ve and config.enable_unigram_ve
        })
        self.bigram_perfect = bool(
            config.enable_nanogpt_ngram_ve and config.enable_bigram_ve
            and config.bigram_perfect_map
        )
        if self.bigram_perfect:
            _pack = np.load(config.bigram_perfect_map)
            # int64 for safe embedding indexing; OOV contexts are pre-filled
            # with n_distinct (the shared UNK row) by the map builder.
            self.register_buffer(
                "bigram_perfect_index",
                torch.from_numpy(np.ascontiguousarray(_pack["map"], dtype=np.int64)),
                persistent=False,
            )
            # +1 row: OOV/UNK bucket for contexts absent from the train scan.
            self.bigram_table_size = int(_pack["n_distinct"]) + 1
            if config.bigram_clean_table:
                # clean 单表 perfect anchor (SSOT): single layer, K=1 full-width.
                self.bigram_ve_layers = {ngram_layers[0]}
                self.bigram_K = 1
            else:
                # legacy 4-layer framework: single layer keeps the fp32 table
                # within one H200 (4 layers x N x 768 x 12B would not fit).
                self.bigram_ve_layers = {ngram_layers[0]}
                self.bigram_K = 2
        elif config.enable_nanogpt_ngram_ve and config.enable_bigram_ve:
            if config.bigram_clean_table:
                # clean 单表 (SSOT): one nn.Embedding(R, n_embd), single layer,
                # single hash, R freely set (no vocab*mult lock-in).
                self.bigram_table_size = config.bigram_clean_table
                self.bigram_ve_layers = {ngram_layers[0]}
                self.bigram_K = 1
            else:
                self.bigram_table_size = config.vocab_size * config.table_mult
                self.bigram_ve_layers = (
                    {ngram_layers[0]} if config.bigram_single_layer else set(ngram_layers)
                )
                self.bigram_K = 2
        else:
            self.bigram_table_size = config.vocab_size * config.table_mult
            self.bigram_ve_layers = set()
            self.bigram_K = 2
        _inject = _parse_inject_layers(config)
        if _inject is not None and self.bigram_ve_layers:
            # multi-layer v/y: one independent clean table per inject layer
            self.bigram_ve_layers = set(_inject)
        half_dim = config.n_embd // 2
        _bp = expand_bigram_hash_primes(_BASE_BIGRAM_PRIMES, len(ngram_layers))
        self.bigram_hash_primes_per_layer = {}
        self.bigram_ves = nn.ModuleDict()
        # X2: clean-table row width d (0 = full n_embd). Narrow rows are read
        # through a frozen zero-pad projection (functional pad: no parameter,
        # no RNG, no optimizer group), so d=0 is bit-identical to the baseline.
        self.bigram_table_dim = (
            config.bigram_table_dim if config.bigram_table_dim > 0 else config.n_embd
        )
        for j, li in enumerate(sorted(self.bigram_ve_layers)):
            if self.bigram_K == 1:
                # clean 单表: one embedding per layer with row width d
                self.bigram_ves[str(li)] = nn.ModuleList([
                    nn.Embedding(self.bigram_table_size, self.bigram_table_dim),
                ])
            else:
                self.bigram_ves[str(li)] = nn.ModuleList([
                    nn.Embedding(self.bigram_table_size, half_dim),
                    nn.Embedding(self.bigram_table_size, config.n_embd - half_dim),
                ])
            self.bigram_hash_primes_per_layer[li] = _bp[j]
        if config.enable_nanogpt_ngram_ve and config.enable_trigram_ve and config.trigram_clean_table:
            # clean 单表 trigram (SSOT): single layer, K=1 full-width, R freely set
            self.trigram_ve_layers = {ngram_layers[0]}
            self.trigram_table_size = config.trigram_clean_table
            self.trigram_K = 1
        else:
            self.trigram_ve_layers = (
                ({ngram_layers[0], ngram_layers[-2], ngram_layers[-1]}
                 if len(ngram_layers) >= 2 else {ngram_layers[-1]})
                if config.enable_nanogpt_ngram_ve and config.enable_trigram_ve and ngram_layers
                else set()
            )
            self.trigram_table_size = config.vocab_size * config.table_mult
            self.trigram_K = 2
        if _inject is not None and self.trigram_ve_layers:
            self.trigram_ve_layers = set(_inject)
        _tp = _BASE_TRIGRAM_PRIMES[:max(1, len(self.trigram_ve_layers))]
        while len(_tp) < len(self.trigram_ve_layers):
            _tp.append(_tp[len(_tp) % len(_BASE_TRIGRAM_PRIMES)])
        self.trigram_hash_primes_per_layer = {}
        self.trigram_ves = nn.ModuleDict()
        self.trigram_table_dim = (
            config.trigram_table_dim if config.trigram_table_dim > 0 else config.n_embd
        )
        for j, li in enumerate(sorted(self.trigram_ve_layers)):
            if self.trigram_K == 1:
                self.trigram_ves[str(li)] = nn.ModuleList([
                    nn.Embedding(self.trigram_table_size, self.trigram_table_dim),
                ])
            else:
                self.trigram_ves[str(li)] = nn.ModuleList([
                    nn.Embedding(self.trigram_table_size, half_dim),
                    nn.Embedding(self.trigram_table_size, config.n_embd - half_dim),
                ])
            self.trigram_hash_primes_per_layer[li] = _tp[j]
        # v2: generic higher-order tables (n >= 4), one clean single table per
        # order, single hash, input over-encoding only.  Registered after the
        # bigram/trigram tables so legacy init order is untouched.
        self.ngram_extra_orders = tuple(int(n) for n in config.ngram_extra_orders)
        if self.ngram_extra_orders:
            if config.nanogpt_ngram_injection_position != "input":
                raise ValueError("--ngram_orders supports input injection only")
            if config.ngram_clean_table <= 0:
                raise ValueError("generic n-gram tables need ngram_clean_table > 0")
            if min(self.ngram_extra_orders) < 4 and config.ngram_dict_topk <= 0:
                raise ValueError("generic n-gram tables need orders >= 4 (orders 2/3 use the "
                                 "legacy branches unless --ngram_dict_topk is set)")
        # _weight=empty: no default init, so constructing these tables does not
        # advance the global RNG (init_weights fills them uniformly last)
        if config.ngram_dict_topk > 0 and (config.enable_unigram_ve
                                           or config.enable_bigram_ve
                                           or config.enable_trigram_ve):
            raise ValueError("dictionary mode (--ngram_dict_topk) routes every order "
                             "through the generic path; legacy branches must be off")
        self.ngram_ves = nn.ModuleDict({
            str(n): nn.Embedding(config.ngram_clean_table, config.n_embd,
                                 _weight=torch.empty(config.ngram_clean_table, config.n_embd))
            for n in self.ngram_extra_orders
        })
        self.ngram_hash_constants = {n: ngram_hash_constants(n) for n in self.ngram_extra_orders}
        # v2.1 dictionary (no-collision) mode: per order, a sorted int64 tensor of
        # the top-ngram_dict_topk train contexts; rows come from searchsorted and
        # misses contribute exactly zero (filled by load_ngram_dictionaries)
        self.ngram_dict_keys: dict = {}
        self._freq_mask_index = None
        self._freq_mask_mode = "none"
        self._freq_mask_threshold = 0
        self._context_set_keys = None   # sorted trigram context keys (device)
        self._context_set_memb = None   # int8 per key: 0=other, 1=A, 2=B
        self._context_eval_sets = None  # selection info dict once configured
        self._context_loss_mask_active = False
        self._hash_reseed_count = 0
        self._intervention_due = frozenset(
            {config.intervention_epoch} | set(config.intervention_epochs)
        )

    @torch.no_grad()
    def init_weights(self):
        # generic order tables are re-initialized uniformly below; skipping
        # their normal_ keeps the RNG stream (and thus bigram/trigram init)
        # identical to a run without them
        generic = {id(m) for m in self.ngram_ves.values()}
        for m in self.modules():
            if id(m) in generic:
                continue
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        self._initialize_ngram_tables()

    @torch.no_grad()
    def _initialize_ngram_tables(self):
        s = 3 ** 0.5 * self.config.n_embd ** -0.5
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for lvs in self.bigram_ves.values():
            for bve in lvs:
                torch.nn.init.uniform_(bve.weight, -s, s)
        for lvs in self.trigram_ves.values():
            for tve in lvs:
                torch.nn.init.uniform_(tve.weight, -s, s)
        for nve in self.ngram_ves.values():
            torch.nn.init.uniform_(nve.weight, -s, s)
        for blk in self.transformer.h:
            for gname in ("ve_gate", "bigram_gate", "trigram_gate"):
                g = getattr(blk.attn, gname, None)
                if g is not None:
                    torch.nn.init.zeros_(g.weight)

    @staticmethod
    def _reseed_constants(constants, salt):
        used = set()
        reseeded = []
        for index, value in enumerate(constants):
            seed = (int(value) ^ (salt + 0x9E3779B9 * (index + 1))) & 0xFFFFFFFF
            reseeded.append(_next_odd_hash_constant(seed, used))
            used.add(reseeded[-1])
        return tuple(reseeded)

    def hash_identity(self):
        return {
            "bigram_primes_per_layer": {
                str(layer): [list(pair) for pair in primes]
                for layer, primes in sorted(self.bigram_hash_primes_per_layer.items())
            },
            "trigram_primes_per_layer": {
                str(layer): list(primes)
                for layer, primes in sorted(self.trigram_hash_primes_per_layer.items())
            },
            "reseed_count": self._hash_reseed_count,
        }

    @torch.no_grad()
    def reseed_ngram_hashes(self):
        if self.bigram_perfect:
            raise RuntimeError("hash_reseed is undefined for --bigram_perfect_map")
        salt = 0xA5A5A5A5 + self._hash_reseed_count * 0x6D2B79F5
        self.bigram_hash_primes_per_layer = {
            layer: [self._reseed_constants(pair, salt + layer * 0x10001)
                    for pair in primes]
            for layer, primes in self.bigram_hash_primes_per_layer.items()
        }
        self.trigram_hash_primes_per_layer = {
            layer: self._reseed_constants(primes, salt + layer * 0x20003)
            for layer, primes in self.trigram_hash_primes_per_layer.items()
        }
        self._hash_reseed_count += 1

    def set_frequency_mask(self, freq_index, mode: str, threshold: int,
                           low_inclusive: bool = False):
        if mode not in {"low", "high"}:
            raise ValueError(f"unsupported frequency mask mode: {mode}")
        if threshold < 0:
            raise ValueError("frequency mask threshold must be non-negative")
        freq_index._ensure_sorted()
        device = next(self.parameters()).device
        self._freq_mask_index = {}
        high_context_counts = {}
        for branch, keys, counts in (
            ("bigram", freq_index._bigram_keys_sorted, freq_index._bigram_counts_sorted),
            ("trigram", freq_index._trigram_keys_sorted, freq_index._trigram_counts_sorted),
        ):
            if mode == "high" and threshold == 0:
                high_keys = np.empty(0, dtype=np.int64)
                high_context_counts[branch] = int(keys.size)
            else:
                # mask_high masks f >= t (inclusive boundary). mask_low is the
                # complement: f < t plus novel f=0 contexts (f <= t when
                # low_inclusive). Novel contexts are never in the index, so
                # low mode always masks them at forward time (train and eval).
                if mode == "low" and low_inclusive:
                    high_keys = keys[counts > threshold]
                else:
                    high_keys = keys[counts >= threshold]
                high_context_counts[branch] = int(high_keys.size)
            self._freq_mask_index[branch] = torch.from_numpy(
                np.ascontiguousarray(high_keys, dtype=np.int64)
            ).to(device)
        self._freq_mask_mode = mode
        self._freq_mask_threshold = int(threshold)
        return {
            "mode": mode,
            "threshold": int(threshold),
            "low_inclusive": bool(low_inclusive),
            "high_context_counts": high_context_counts,
        }

    def _frequency_mask(self, branch, context_keys):
        if self._freq_mask_index is None:
            return None
        if self._freq_mask_mode == "high" and self._freq_mask_threshold == 0:
            return torch.ones_like(context_keys, dtype=torch.bool)
        high_keys = self._freq_mask_index[branch]
        flat_keys = context_keys.reshape(-1)
        if high_keys.numel() == 0:
            high_match = torch.zeros_like(flat_keys, dtype=torch.bool)
        else:
            positions = torch.searchsorted(high_keys, flat_keys)
            in_range = positions < high_keys.numel()
            high_match = torch.zeros_like(flat_keys, dtype=torch.bool)
            high_match[in_range] = high_keys[positions[in_range]] == flat_keys[in_range]
        high_match = high_match.view_as(context_keys)
        if self._freq_mask_mode == "low":
            return ~high_match
        return high_match

    def setup_context_eval_sets(self, freq_index, frac: float, seed: int):
        """Select disjoint random context sets A (to be loss-masked) and B
        (unmasked peers): each a uniform `frac` of the distinct TRIGRAM
        context types in the training frequency index, drawn by type (not by
        occurrence) so the empirical context distribution is preserved.
        Stores the sorted keys and an int8 membership array (0 = other,
        1 = A, 2 = B) on the model device. Deterministic given (frac, seed),
        so control and masked runs share the same A/B."""
        if not (0.0 < frac < 0.5):
            raise ValueError("context set fraction must be in (0, 0.5)")
        device = next(self.parameters()).device
        mode = self.config.context_set_mode
        if mode == "row":
            # A/B over trigram table rows: masking by row makes the masked
            # subspaces private by construction (colliding contexts of a masked
            # row are masked together). Keys array unused; membership via
            # _trigram_row_indices.
            n_rows = int(self.config.trigram_clean_table)
            n_sel = int(round(frac * n_rows))
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_rows)
            memb = np.zeros(n_rows, dtype=np.int8)
            memb[perm[:n_sel]] = 1
            memb[perm[n_sel:2 * n_sel]] = 2
            self._context_set_keys = None
            self._context_set_memb = torch.from_numpy(memb).to(device)
            self._context_eval_sets = {
                "branch": "trigram", "mode": "row", "frac": float(frac),
                "seed": int(seed), "n_types": n_rows, "n_set_types": n_sel,
            }
            return self._context_eval_sets
        freq_index._ensure_sorted()
        keys = np.ascontiguousarray(freq_index._trigram_keys_sorted, dtype=np.int64)
        n_types = int(keys.size)
        n_sel = int(round(frac * n_types))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_types)
        memb = np.zeros(n_types, dtype=np.int8)
        memb[perm[:n_sel]] = 1           # A: loss-masked after the intervention
        memb[perm[n_sel:2 * n_sel]] = 2  # B: unmasked peers
        self._context_set_keys = torch.from_numpy(keys).to(device)
        self._context_set_memb = torch.from_numpy(memb).to(device)
        self._context_eval_sets = {
            "branch": "trigram", "mode": "type",
            "frac": float(frac),
            "seed": int(seed),
            "n_types": n_types,
            "n_set_types": n_sel,
        }
        return self._context_eval_sets

    def context_set_membership(self, idx):
        """int8 membership (0 = other, 1 = A, 2 = B) of each position's trigram
        context, keyed exactly like GlobalFrequencyIndex (prev2*V^2+prev*V+cur
        with boundary padding). Positions whose context is absent from the
        index (novel) map to 0. None if sets are not configured."""
        if self._context_set_memb is None:
            return None
        if self._context_set_keys is None:  # row mode
            prev = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
            prev2 = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
            # Clean-table trigram row hash at the single injection layer
            # (K=1): (prev2*p0 ^ prev*p1 ^ idx*p2) % R with the base primes.
            # Computed directly (not via _trigram_row_indices) so arms without
            # a trigram table (nogram) select the identical rows.
            p0, p1, p2 = _BASE_TRIGRAM_PRIMES[0][:3]
            rows = (
                (prev2.long() * p0) ^ (prev.long() * p1) ^ (idx.long() * p2)
            ) % int(self.config.trigram_clean_table)
            return self._context_set_memb[rows]
        prev = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2 = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        V = self.config.vocab_size
        keys = (prev2.long() * (V * V) + prev.long() * V + idx.long()).reshape(-1)
        pos = torch.searchsorted(self._context_set_keys, keys)
        pos = pos.clamp(max=self._context_set_keys.numel() - 1)
        hit = self._context_set_keys[pos] == keys
        out = torch.zeros_like(keys, dtype=torch.int8)
        out[hit] = self._context_set_memb[pos[hit]]
        return out.view(idx.shape)

    def _context_loss_keep_mask(self, idx):
        """Per-position keep mask (True = keep in the loss) once the random
        context loss mask has fired; None while inactive. Positions whose
        trigram context is in set A are dropped, everything else is kept."""
        if not self._context_loss_mask_active:
            return None
        return self.context_set_membership(idx) != 1

    def apply_intervention(self, epoch0: int, freq_index=None):
        """Fire intervention when 0-indexed epoch reaches a due epoch.

        Due epochs = {intervention_epoch} | set(intervention_epochs); the
        train loop guarantees each due epoch fires at most once."""
        if self.config.intervention == "none":
            return {"kind": "none"}
        if epoch0 not in self._intervention_due:
            return {"kind": "not_due"}
        event = {
            "epoch0": epoch0,
            "intervention": self.config.intervention,
            "table_parameters_preserved": True,
            "optimizer_state_preserved": True,
            "hash_identity_before": self.hash_identity(),
        }
        if self.config.intervention == "freeze_table":
            for p in self._ngram_params():
                p.requires_grad_(False)
        elif self.config.intervention == "freeze_backbone":
            for p in self._backbone_params():
                p.requires_grad_(False)
        elif self.config.intervention == "freeze_both":
            for p in self._ngram_params():
                p.requires_grad_(False)
            for p in self._backbone_params():
                p.requires_grad_(False)
        elif self.config.intervention == "hash_reseed":
            self.reseed_ngram_hashes()
        elif self.config.intervention in {"mask_low_freq", "mask_high_freq"}:
            if freq_index is None:
                raise RuntimeError(
                    f"{self.config.intervention} requires --freq_index at the intervention boundary"
                )
            mode = "low" if self.config.intervention == "mask_low_freq" else "high"
            event["frequency_mask"] = self.set_frequency_mask(
                freq_index,
                mode,
                self.config.intervention_freq_threshold,
                low_inclusive=self.config.intervention_low_inclusive,
            )
            event["frequency_mask"]["novel_contexts_masked"] = (
                mode == "low"
                or (mode == "high" and self.config.intervention_freq_threshold == 0)
            )
        elif self.config.intervention == "random_context_loss_mask":
            if self._context_eval_sets is None:
                raise RuntimeError(
                    "random_context_loss_mask requires context eval sets: run "
                    "setup_context_eval_sets first (needs --freq_index)"
                )
            self._context_loss_mask_active = True
            event["context_loss_mask"] = dict(self._context_eval_sets)
        elif self.config.intervention == "readout_last_epoch":
            # Table used ONLY in the final epoch: fully masked from step 0
            # (rows receive no gradient, backbone sees no table signal), then
            # unmasked at the start of epoch intervention_epoch. Due epochs
            # must include 0 and intervention_epoch.
            if epoch0 == 0:
                if freq_index is None:
                    raise RuntimeError(
                        "readout_last_epoch requires --freq_index at step 0")
                event["frequency_mask"] = self.set_frequency_mask(
                    freq_index, "high", 0)
                event["frequency_mask"]["novel_contexts_masked"] = True
                event["phase"] = "mask_all_from_start"
            elif epoch0 == self.config.intervention_epoch:
                self._freq_mask_index = None
                self._freq_mask_mode = "none"
                self._freq_mask_threshold = 0
                event["phase"] = "unmask_last_epoch"
        event["hash_identity_after"] = self.hash_identity()
        return event

    def _ngram_params(self):
        markers = ("value_embeds", "bigram_ves", "trigram_ves", "ngram_ves",
                   "ve_gate", "bigram_gate", "trigram_gate")
        for name, p in self.named_parameters():
            if any(m in name for m in markers):
                yield p

    def _backbone_params(self):
        markers = ("value_embeds", "bigram_ves", "trigram_ves", "ngram_ves",
                   "ve_gate", "bigram_gate", "trigram_gate")
        for name, p in self.named_parameters():
            if not any(m in name for m in markers):
                yield p

    def _pad_to_embd(self, vec, dim: int):
        """X2: frozen zero-pad projection from table row width d to n_embd."""
        if vec.size(-1) == self.config.n_embd:
            return vec
        return F.pad(vec, (0, self.config.n_embd - dim))

    def _bigram_row_indices(self, li, prev_idx, idx):
        """Row indices into the bigram hash embedding(s) for layer li.

        Perfect mode: groups share the collision-free packed-context map
        (clean K=1: a single index list; legacy K=2: identical rows twice,
        equivalent to one table split across half-dims).
        """
        if self.bigram_perfect:
            rows = self.bigram_perfect_index[prev_idx * self.config.vocab_size + idx]
            return [rows] * self.bigram_K
        primes = self.bigram_hash_primes_per_layer[li]
        return [((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size
                for p1, p2 in primes[:self.bigram_K]]

    def _trigram_row_indices(self, li, prev2_idx, prev_idx, idx):
        """Row indices into the trigram hash embedding(s) for layer li.
        clean 单表 (K=1): first hash family only."""
        lp = self.trigram_hash_primes_per_layer[li]
        return [
            ((prev2_idx * lp[3 * k]) ^ (prev_idx * lp[3 * k + 1]) ^ (idx * lp[3 * k + 2]))
            % self.trigram_table_size
            for k in range(self.trigram_K)
        ]

    def _ngram_row_indices(self, n, idx):
        """Rows of the generic order-n table: XOR_k (shift_k(idx) * c_k) % R,
        shift_k as in the bigram/trigram branches (in-chunk, no wrap)."""
        rows = None
        for k, c in enumerate(self.ngram_hash_constants[n]):
            tok = idx if k == 0 else torch.cat([idx[:, :k], idx[:, :-k]], dim=1)
            term = tok.long() * c
            rows = term if rows is None else rows ^ term
        return rows % self.config.ngram_clean_table

    def _ngram_context_keys_torch(self, idx, n):
        """Torch twin of ngram_freq.context_keys_np: exact packing while
        vocab^n <= 2^63, chained splitmix64 above.  Bit-identical to the numpy
        uint64 implementation, including logical right shifts (emulated on
        signed int64 via masks) and wrapping multiplies (int64 wrap = uint64
        wrap).  The dictionary build and the model MUST agree bit for bit or
        the dictionary silently misses — checked by test_ngram_dict_cpu.py."""
        x = idx.long()
        T = x.shape[1]

        def shift(k):
            if k == 0:
                return x
            if k >= T:
                return x.clone()
            return torch.cat([x[:, :k], x[:, :T - k]], dim=1)

        if self.config.vocab_size ** n <= 2 ** 63:
            key = torch.zeros_like(x)
            for k in range(n):
                key = key + shift(k) * (self.config.vocab_size ** k)
            return key
        h = torch.full(x.shape, _SPLITMIX_SEED_I64 ^ n, dtype=torch.int64, device=x.device)
        for k in range(n - 1, -1, -1):
            h = _mix64_i64(h ^ shift(k))
        return h

    def _ngram_dict_rows(self, n, idx):
        """(rows, valid) for the order-n no-collision dictionary: rows =
        searchsorted position of the context key (clamped), valid = exact
        match against the sorted dictionary.  Misses are multiplied out in the
        caller, so the clamped row is inert; members never share a row."""
        keys = self._ngram_context_keys_torch(idx, n)
        table = self.ngram_dict_keys[n]
        pos = torch.searchsorted(table, keys.reshape(-1))
        pos_c = pos.clamp(max=table.numel() - 1)
        valid = table[pos_c] == keys.reshape(-1)
        return pos_c.view_as(keys), valid.view_as(keys)

    def load_ngram_dictionaries(self, freq_index, topk: int, log=print):
        """Build per-order no-collision dictionaries from a multi-order index.

        Selection: the top-`topk` train contexts by count, ties broken by key
        ascending (the index stores keys sorted ascending, so a stable argsort
        on -counts is deterministic).  Row of a member = its position in the
        key-sorted dictionary; a miss contributes exactly zero via the validity
        mask.  If an order's distinct count is within topk, EVERY context gets
        a row (perfect table, zero collision by construction).  For n >= 5 the
        key space itself is splitmix64-hashed, so two distinct contexts share a
        key with probability ~pairs/2^64 (none at corpus scale).
        """
        for n in self.ngram_extra_orders:
            keys = freq_index._keys.get(n)
            counts = freq_index._counts.get(n)
            if keys is None or counts is None:
                raise SystemExit(f"multi-order freq index lacks order {n} "
                                 f"(has {getattr(freq_index, 'orders', '?')}); "
                                 "rebuild with ngram_freq.py --orders")
            counts64 = counts.astype(np.int64)
            if keys.size <= topk:
                sel = np.arange(keys.size)
            else:
                order = np.argsort(-counts64, kind="stable")[:topk]
                sel = np.sort(order)
            dkeys = np.ascontiguousarray(keys[sel], dtype=np.int64)
            device = next(self.parameters()).device
            self.ngram_dict_keys[n] = torch.from_numpy(dkeys).to(device)
            coverage = float(counts64[sel].sum()) / max(1, int(counts64.sum()))
            log(f"[nglab] dict n={n}: {keys.size:,} distinct -> kept {dkeys.size:,} "
                f"(topk={topk}), position coverage {coverage:.4f}")

    def _check_pre_rows_supported(self):
        """Precomputed rows are only valid for the clean single-table mainline.

        Conditions: input injection, K=1, exactly one table layer per enabled
        branch, first hash family, no hash reseed, no perfect map.  Anything
        else must refuse loudly (the offline rows would silently fork).
        """
        if self.config.nanogpt_ngram_injection_position != "input":
            raise SystemExit("--row_index requires --injection_position input")
        if self.bigram_perfect:
            raise SystemExit("--row_index is incompatible with --bigram_perfect_map")
        if self._hash_reseed_count != 0:
            raise SystemExit("--row_index is incompatible with hash reseed")
        if self.ngram_extra_orders:
            raise SystemExit("--row_index does not cover generic orders >= 4")
        if self.config.enable_bigram_ve:
            if self.bigram_K != 1 or len(self.bigram_ve_layers) != 1:
                raise SystemExit(
                    "--row_index requires the clean single-table bigram branch "
                    "(K=1, one layer); legacy multi-hash paths are unsupported")
        if self.config.enable_trigram_ve:
            if self.trigram_K != 1 or len(self.trigram_ve_layers) != 1:
                raise SystemExit(
                    "--row_index requires the clean single-table trigram branch "
                    "(K=1, one layer); legacy multi-hash paths are unsupported")

    def _compute_input_ngram_residual(self, idx, pre_rows=None):
        """Over-encoding: sum all enabled layers' n-gram values, no gate.

        pre_rows (plan-8 fast path): (b_rows, t_rows, b_valid, t_valid) with
        (B,T) int64 row indices and (B,T) bool validity (None where the branch
        has no offline stream).  When given, the hash arithmetic is skipped and
        the caller-supplied rows are gathered directly; validity False zeroes
        the branch contribution (static low-frequency mask semantics, identical
        to masked_fill).  Bit-identity with the dynamic path is enforced at
        startup by the first-batch verification in main().
        """
        _B, T = idx.size()
        if pre_rows is not None:
            self._check_pre_rows_supported()
            b_rows, t_rows, b_valid, t_valid = pre_rows
            residual = None
            if self.config.enable_bigram_ve:
                li = sorted(self.bigram_ve_layers)[0]
                lvs = self.bigram_ves[str(li)]
                bgve = self._pad_to_embd(lvs[0](b_rows), self.bigram_table_dim)
                if b_valid is not None:
                    bgve = bgve * b_valid.unsqueeze(-1).to(bgve.dtype)
                residual = bgve
            if self.config.enable_trigram_ve:
                li = sorted(self.trigram_ve_layers)[0]
                lvs = self.trigram_ves[str(li)]
                tgve = self._pad_to_embd(lvs[0](t_rows), self.trigram_table_dim)
                if t_valid is not None:
                    tgve = tgve * t_valid.unsqueeze(-1).to(tgve.dtype)
                residual = tgve if residual is None else residual + tgve
            if residual is None:
                residual = torch.zeros(idx.size(0), T, self.config.n_embd,
                                       device=idx.device, dtype=self.transformer.wte.weight.dtype)
            return residual
        residual = None
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        if self.config.enable_unigram_ve:
            for li in sorted(self.value_embeds.keys()):
                ve = self.value_embeds[li](idx)
                residual = ve if residual is None else residual + ve
        if self.config.enable_bigram_ve:
            for li in sorted(self.bigram_ve_layers):
                lvs = self.bigram_ves[str(li)]
                idxs = self._bigram_row_indices(li, prev_idx, idx)
                bgve = torch.cat([lvs[k](idxs[k]) for k in range(self.bigram_K)], dim=-1)
                bgve = self._pad_to_embd(bgve, self.bigram_table_dim)
                mask = self._frequency_mask(
                    "bigram",
                    prev_idx.long() * self.config.vocab_size + idx.long(),
                )
                if mask is not None:
                    bgve = bgve.masked_fill(mask.unsqueeze(-1), 0)
                residual = bgve if residual is None else residual + bgve
        if self.config.enable_trigram_ve:
            for li in sorted(self.trigram_ve_layers):
                ti = self._trigram_row_indices(li, prev2_idx, prev_idx, idx)
                lvs = self.trigram_ves[str(li)]
                tgve = torch.cat([lvs[k](ti[k]) for k in range(self.trigram_K)], dim=-1)
                tgve = self._pad_to_embd(tgve, self.trigram_table_dim)
                mask = self._frequency_mask(
                    "trigram",
                    prev2_idx.long() * (self.config.vocab_size * self.config.vocab_size)
                    + prev_idx.long() * self.config.vocab_size + idx.long(),
                )
                if mask is not None:
                    tgve = tgve.masked_fill(mask.unsqueeze(-1), 0)
                residual = tgve if residual is None else residual + tgve
        for n in self.ngram_extra_orders:
            nve_tbl = self.ngram_ves[str(n)]
            if n in self.ngram_dict_keys and self.config.ngram_keep_rows == "hash":
                # coverage-matched control: identical top-K membership mask, but the
                # row is the hash row, so collisions among kept contexts remain
                _, valid = self._ngram_dict_rows(n, idx)
                nve = nve_tbl(self._ngram_row_indices(n, idx)) * valid.unsqueeze(-1).to(nve_tbl.weight.dtype)
            elif n in self.ngram_dict_keys:
                rows, valid = self._ngram_dict_rows(n, idx)
                nve = nve_tbl(rows) * valid.unsqueeze(-1).to(nve_tbl.weight.dtype)
            else:
                nve = nve_tbl(self._ngram_row_indices(n, idx))
            residual = nve if residual is None else residual + nve
        if residual is None:
            residual = torch.zeros(idx.size(0), T, self.config.n_embd,
                                   device=idx.device, dtype=self.transformer.wte.weight.dtype)
        return residual

    def _trunk(self, idx, pre_rows=None):
        """Everything from embeddings through ln_f. Returns (h, ngram_residual).

        ngram_residual is the input-injection over-encoding sum (None unless
        injection_position == "input").
        """
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.transformer.wte(idx)
        x = x + self.transformer.wpe(pos)
        ngram_residual = None
        input_pos = self.config.nanogpt_ngram_injection_position == "input"
        if input_pos:
            ngram_residual = self._compute_input_ngram_residual(idx, pre_rows=pre_rows)
            x = x + ngram_residual
        x = self.transformer.drop(x)
        # Per-block VE (unigram/bigram/trigram gathers + freq-mask searchsorted)
        # is consumed only by the v / y / engram injection routes.  For input
        # injection it is dead compute: CausalSelfAttention ignores these
        # residuals.  Skipping it is numerically identical, removes ~4 gather +
        # 2 searchsorted passes per microbatch, and is a prerequisite for a
        # graph-break-free torch.compile (plan-8 G1; experiment-log §18).
        compute_block_ve = not input_pos
        bigram_indices = {}
        trigram_indices = {}
        prev_idx = prev2_idx = None
        if compute_block_ve and self.config.enable_bigram_ve:
            prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
            prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
            for li in self.bigram_ve_layers:
                bigram_indices[li] = self._bigram_row_indices(li, prev_idx, idx)
        if compute_block_ve and self.config.enable_trigram_ve:
            if prev_idx is None:
                prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
                prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
            for li in self.trigram_ve_layers:
                trigram_indices[li] = self._trigram_row_indices(li, prev2_idx, prev_idx, idx)
        for i, block in enumerate(self.transformer.h):
            ve = bgve = tgve = None
            if compute_block_ve:
                ve = (self.value_embeds[str(i)](idx)
                      if self.config.enable_unigram_ve and str(i) in self.value_embeds else None)
                if self.config.enable_bigram_ve and i in self.bigram_ve_layers:
                    lvs = self.bigram_ves[str(i)]
                    bi = bigram_indices[i]
                    bgve = self._pad_to_embd(
                        torch.cat([lvs[k](bi[k]) for k in range(self.bigram_K)], dim=-1),
                        self.bigram_table_dim,
                    )
                    bmask = self._frequency_mask(
                        "bigram",
                        prev_idx.long() * self.config.vocab_size + idx.long(),
                    )
                    if bmask is not None:
                        bgve = bgve.masked_fill(bmask.unsqueeze(-1), 0)
                if self.config.enable_trigram_ve and i in self.trigram_ve_layers:
                    ti = trigram_indices[i]
                    lvs = self.trigram_ves[str(i)]
                    tgve = self._pad_to_embd(
                        torch.cat([lvs[k](ti[k]) for k in range(self.trigram_K)], dim=-1),
                        self.trigram_table_dim,
                    )
                    tmask = self._frequency_mask(
                        "trigram",
                        prev2_idx.long() * (self.config.vocab_size * self.config.vocab_size)
                        + prev_idx.long() * self.config.vocab_size + idx.long(),
                    )
                    if tmask is not None:
                        tgve = tgve.masked_fill(tmask.unsqueeze(-1), 0)
            x = block(x, ve=ve, bigram_ve=bgve, trigram_ve=tgve)
        x = self.transformer.ln_f(x)
        return x, ngram_residual

    def _input_injection_stats(self, idx, ngram_residual):
        """Branch RMS stats for the input route (eval-only; recomputes gathers)."""
        injection_stats = {
            "bigram_rms": {"sequence_rms": [], "batch_rms": 0.0},
            "trigram_rms": {"sequence_rms": [], "batch_rms": 0.0},
            "total_rms": {"sequence_rms": [], "batch_rms": 0.0},
            "fourgram_rms": {"sequence_rms": [], "batch_rms": 0.0},
        }
        if ngram_residual is None:
            return injection_stats
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        branch_residuals: dict = {}
        branch_residuals["bigram"] = [
            self._pad_to_embd(
                torch.cat([self.bigram_ves[str(li)][k](bi[k])
                           for k in range(self.bigram_K)], dim=-1),
                self.bigram_table_dim,
            )
            for li in sorted(self.bigram_ve_layers)
            for bi in [self._bigram_row_indices(li, prev_idx, idx)]
        ]
        branch_residuals["trigram"] = [
            self._pad_to_embd(
                torch.cat([self.trigram_ves[str(li)][k](ti[k])
                           for k in range(self.trigram_K)], dim=-1),
                self.trigram_table_dim,
            )
            for li in sorted(self.trigram_ve_layers)
            for ti in [self._trigram_row_indices(li, prev2_idx, prev_idx, idx)]
        ]
        for branch, residuals in branch_residuals.items():
            if residuals:
                branch_residual = torch.stack(residuals).sum(dim=0)
                branch_rms = torch.sqrt(torch.mean(branch_residual.float() ** 2, dim=-1))
                injection_stats[f"{branch}_rms"] = {
                    "sequence_rms": branch_rms.mean(dim=0).detach().cpu().tolist(),
                    "batch_rms": float(branch_rms.mean()),
                }
        total_rms = torch.sqrt(torch.mean(ngram_residual.float() ** 2, dim=-1))
        injection_stats["total_rms"] = {
            "sequence_rms": total_rms.mean(dim=0).detach().cpu().tolist(),
            "batch_rms": float(total_rms.mean()),
        }
        return injection_stats

    def _resolve_ce_chunk(self, n_positions: int) -> int:
        """Chunk size for the checkpointed CE; 0 = full-logit path.

        auto (0): enable only when full bf16 logits would exceed 4 GiB, then
        size chunks to ~2 GiB of bf16 logits.  -1 forces off; >0 forces the
        given chunk size.  8k-vocab mainline runs never trigger auto.
        """
        c = self.config.ce_chunk_positions
        V = self.config.vocab_size
        if c == -1:
            return 0
        if c > 0:
            return min(c, n_positions)
        if n_positions * V * 2 <= (4 << 30):
            return 0
        return max(1, min(n_positions, (2 << 30) // max(1, 2 * V)))

    def _ce_token_losses(self, h, targets):
        """Per-token CE from final hidden states h (B,T,C).

        Memory-safe at large vocab: with chunking active, lm_head GEMM + CE
        run under torch.utils.checkpoint per chunk so full logits are never
        materialized (129k-vocab logits would be ~38GB bf16 / ~76GB fp32).
        Per-token CE is row-local, so chunking does not change per-token
        values; the masked mean runs on the concatenated vector exactly like
        the unchunked path.
        """
        chunk = self._resolve_ce_chunk(h.shape[0] * h.shape[1])
        if chunk == 0:
            logits = self.lm_head(h)
            token_losses = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.reshape(-1), ignore_index=-1, reduction="none"
            ).view_as(targets)
            return token_losses
        from torch.utils.checkpoint import checkpoint

        flat_h = h.reshape(-1, h.size(-1))
        flat_t = targets.reshape(-1)
        lm_w = self.lm_head.weight
        outs = []
        for s in range(0, flat_h.size(0), chunk):
            h_c = flat_h[s:s + chunk]
            t_c = flat_t[s:s + chunk]
            outs.append(checkpoint(
                lambda hc, tc, w=lm_w: F.cross_entropy(
                    F.linear(hc, w).float(), tc,
                    ignore_index=-1, reduction="none"),
                h_c, t_c, use_reentrant=False))
        token_losses = torch.cat(outs) if len(outs) > 1 else outs[0]
        return token_losses.view_as(targets)

    def per_token_losses(self, idx, targets):
        """(B, T) fp32 per-token CE without materializing full logits.

        Memory-safe entry point for diagnostics at large vocab
        (ngram_freq.compute_per_token_loss routes here when present).
        """
        h, _ = self._trunk(idx)
        return self._ce_token_losses(h, targets).float()

    @torch.no_grad()
    def projection_stats(self, idx, targets, topk: int = 20):
        """Chunked softmax stats for the logit probe (entropy/margin/ptrue/top-k).

        Never materializes full logits: per chunk of positions, lm_head GEMM ->
        fp32 softmax stats -> discard.  Values are row-local and identical to
        the full-logit computation.  Returns CPU tensors flattened to (N,).
        """
        h, _ = self._trunk(idx)
        flat_h = h.reshape(-1, h.size(-1))
        flat_t = targets.reshape(-1).clamp(min=0)
        chunk = self._resolve_ce_chunk(flat_h.size(0)) or flat_h.size(0)
        ents, margins, ptrues, tk_ids, tk_probs = [], [], [], [], []
        for s in range(0, flat_h.size(0), chunk):
            logits = self.lm_head(flat_h[s:s + chunk]).float()
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            ents.append(-(probs * log_probs).sum(dim=-1))
            top2 = torch.topk(probs, k=2, dim=-1).values
            margins.append(top2[:, 0] - top2[:, 1])
            ptrues.append(probs.gather(-1, flat_t[s:s + chunk, None]).squeeze(-1))
            topk_out = torch.topk(probs, k=min(topk, probs.size(-1)), dim=-1)
            tk_ids.append(topk_out.indices)
            tk_probs.append(topk_out.values)
        cat = torch.cat
        return (cat(ents).cpu(), cat(margins).cpu(), cat(ptrues).cpu(),
                cat(tk_ids).cpu(), cat(tk_probs).cpu())

    def forward(self, idx, targets=None, return_token_losses=False,
                return_injection_stats=False, pre_rows=None):
        h, ngram_residual = self._trunk(idx, pre_rows=pre_rows)
        if targets is None:
            if return_token_losses:
                raise ValueError("return_token_losses requires targets")
            logits = self.lm_head(h)
            if return_injection_stats:
                return logits.float(), self._input_injection_stats(idx, ngram_residual)
            return logits.float()
        token_losses = self._ce_token_losses(h, targets)
        loss = token_losses[targets.ne(-1)].mean()
        if return_token_losses and return_injection_stats:
            return loss, token_losses, self._input_injection_stats(idx, ngram_residual)
        if return_token_losses:
            return loss, token_losses
        if return_injection_stats:
            return loss, self._input_injection_stats(idx, ngram_residual)
        return loss


# ---------------------------------------------------------------------------
# 2. Optimizer: AdamW (backbone) + RMSProp (n-gram table), mixed grouping
# ---------------------------------------------------------------------------


class MixedOptimizer:
    """Minimal AdamW + RMSProp optimizer (no Muon).

    n-gram table params (value_embeds / bigram_ves / trigram_ves / *_gate)
    -> table_optimizer: RMSProp betas=(0.0, 0.99) bias-corrected (default),
       AdamW with table_betas, or SGD with momentum=table_betas[0].
    Everything else -> AdamW with betas=(0.8, 0.95), weight_decay=0.1 (matrices only).
    """

    def __init__(self, model: nn.Module, lr: float, ngram_betas, adam_betas,
                 weight_decay: float, table_optimizer: str = "rmsprop",
                 table_lr_scale: float = 1.0, table_betas=None):
        self.model = model
        self.lr = lr
        self.ngram_beta2 = ngram_betas[1]
        self.table_optimizer = table_optimizer
        self.table_lr_scale = table_lr_scale
        self.table_betas = tuple(table_betas) if table_betas is not None else tuple(ngram_betas)
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay
        self.adam_steps = {}
        self.adam_exp_avg = {}
        self.adam_exp_avg_sq = {}
        self.rms_steps = {}
        self.rms_exp_avg_sq = {}
        self.table_exp_avg = {}  # adamw first moment / sgd momentum buffer
        self._diag = None        # per-step table-update diagnostics (opt-in)
        ngram_markers = ("value_embeds", "bigram_ves", "trigram_ves", "ngram_ves",
                         "ve_gate", "bigram_gate", "trigram_gate")
        self.ngram_params = []
        self.adam_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(m in name for m in ngram_markers):
                self.ngram_params.append((name, p))
            else:
                self.adam_params.append((name, p))

    def zero_grad(self, set_to_none=True):
        for _, p in self.adam_params + self.ngram_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    # --- per-step table-update diagnostics (opt-in via enable_table_diag) ---
    def enable_table_diag(self):
        self._diag = self._new_table_diag()

    @staticmethod
    def _new_table_diag():
        return {"n_hit": 0, "g_sq_sum": 0.0, "g_sq_n": 0,
                "denom_sq_sum": 0.0, "denom_sq_n": 0,
                "amp_sum": 0.0, "amp_max": 0.0, "amp_n": 0}

    def _record_table_diag(self, p, g, denom):
        """Accumulate hit-row gradient/denom/update-amplification stats.

        Shock fingerprint: per-row |g|/denom. A fresh row (its RMSProp
        second-moment EMA still tracks the current gradient) gives ~1.
        A stale row (EMA decayed while unvisited, then revisited at an epoch
        boundary) gives ~1/sqrt(beta2^k) >> 1 — the replay-shock amplifier.
        """
        if self._diag is None:
            return
        row_g2 = g.pow(2).sum(-1)          # (n_rows,)
        hit = row_g2 > 0
        n_hit = int(hit.sum().item())
        if n_hit == 0:
            return
        row_d2 = denom.pow(2).sum(-1)
        row_g = row_g2.sqrt()
        row_d = row_d2.sqrt()
        amp = row_g[hit] / row_d[hit].clamp_min(1e-10)
        d = self._diag
        d["n_hit"] += n_hit
        d["g_sq_sum"] += float(row_g2[hit].sum().item())
        d["g_sq_n"] += n_hit
        d["denom_sq_sum"] += float(row_d2[hit].sum().item())
        d["denom_sq_n"] += n_hit
        d["amp_sum"] += float(amp.sum().item())
        d["amp_max"] = max(d["amp_max"], float(amp.max().item()))
        d["amp_n"] += n_hit

    def pop_table_diag(self):
        """Return the accumulated diagnostics dict and reset for the next step."""
        d = self._diag
        if d is None:
            return None
        out = {"n_hit": d["n_hit"],
               "g_rms": (d["g_sq_sum"] / d["g_sq_n"]) ** 0.5 if d["g_sq_n"] else 0.0,
               "denom_rms": (d["denom_sq_sum"] / d["denom_sq_n"]) ** 0.5 if d["denom_sq_n"] else 0.0,
               "shock_mean": d["amp_sum"] / d["amp_n"] if d["amp_n"] else 0.0,
               "shock_max": d["amp_max"]}
        self._diag = self._new_table_diag()
        return out

    def _adamw_step(self, name, p, lr_t):
        g = p.grad
        if g is None:
            return
        if name not in self.adam_steps:
            self.adam_steps[name] = 0
            self.adam_exp_avg[name] = torch.zeros_like(p)
            self.adam_exp_avg_sq[name] = torch.zeros_like(p)
        self.adam_steps[name] += 1
        step = self.adam_steps[name]
        b1, b2 = self.adam_betas
        exp_avg = self.adam_exp_avg[name]
        exp_avg_sq = self.adam_exp_avg_sq[name]
        # weight decay on 2D matrix params only
        is_matrix = p.dim() >= 2 and not name.endswith("wte.weight") \
            and not name.endswith("wpe.weight") and not name.endswith("lm_head.weight")
        wd = self.weight_decay if is_matrix else 0.0
        p.mul_(1 - lr_t * wd)
        exp_avg.lerp_(g, 1 - b1)
        exp_avg_sq.lerp_(g.square(), 1 - b2)
        bias1 = 1 - b1 ** step
        bias2 = 1 - b2 ** step
        denom = (exp_avg_sq / bias2).sqrt() + 1e-8
        step_size = lr_t / bias1
        p.add_(exp_avg / denom, alpha=-step_size)

    def _table_rmsprop_step(self, name, p, lr_t):
        g = p.grad
        if g is None:
            return
        if name not in self.rms_steps:
            self.rms_steps[name] = 0
            self.rms_exp_avg_sq[name] = torch.zeros_like(p)
        self.rms_steps[name] += 1
        step = self.rms_steps[name]
        # beta2 for RMSProp EMA comes from table_betas.
        b2 = self.table_betas[1]
        exp_avg_sq = self.rms_exp_avg_sq[name]
        bias2 = 1 - b2 ** step
        # Large tables (>= 2 GiB fp32, e.g. 5+ orders x R=3M on H200) are
        # updated in row-slices: elementwise math is identical, but the
        # square/denom temporaries are bounded to one slice instead of three
        # full-table copies (~28 GiB peak at 3M x 768).
        if p.numel() * p.element_size() > _TABLE_SLICE_BYTES:
            for i in range(0, p.shape[0], _TABLE_SLICE_ROWS):
                sl = slice(i, min(i + _TABLE_SLICE_ROWS, p.shape[0]))
                es = exp_avg_sq[sl]
                gs = g[sl]
                es.lerp_(gs.square(), 1 - b2)
                denom = (es / bias2).sqrt_().add_(1e-10)
                p[sl].add_(gs / denom, alpha=-lr_t)
                self._record_table_diag(p[sl], gs, denom)
            return
        # no weight decay on n-gram tables
        exp_avg_sq.lerp_(g.square(), 1 - b2)
        denom = (exp_avg_sq / bias2).sqrt() + 1e-10
        p.add_(g / denom, alpha=-lr_t)
        self._record_table_diag(p, g, denom)

    def _table_adamw_step(self, name, p, lr_t):
        g = p.grad
        if g is None:
            return
        if name not in self.rms_steps:
            self.rms_steps[name] = 0
            self.rms_exp_avg_sq[name] = torch.zeros_like(p)
            self.table_exp_avg[name] = torch.zeros_like(p)
        self.rms_steps[name] += 1
        step = self.rms_steps[name]
        b1, b2 = self.table_betas
        exp_avg = self.table_exp_avg[name]
        exp_avg_sq = self.rms_exp_avg_sq[name]
        exp_avg.lerp_(g, 1 - b1)
        exp_avg_sq.lerp_(g.square(), 1 - b2)
        bias1 = 1 - b1 ** step
        bias2 = 1 - b2 ** step
        denom = (exp_avg_sq / bias2).sqrt() + 1e-8
        p.add_(exp_avg / denom, alpha=-lr_t / bias1)

    def _table_sgd_step(self, name, p, lr_t):
        g = p.grad
        if g is None:
            return
        momentum = self.table_betas[0]
        buf = self.table_exp_avg.get(name)
        if buf is None:
            buf = torch.zeros_like(p)
            self.table_exp_avg[name] = buf
        buf.mul_(momentum).add_(g)
        p.add_(buf, alpha=-lr_t)

    def step(self, lr_mult: float = 1.0):
        lr_t = self.lr * lr_mult
        table_lr_t = lr_t * self.table_lr_scale
        with torch.no_grad():
            for name, p in self.adam_params:
                if p.requires_grad:
                    self._adamw_step(name, p, lr_t)
            for name, p in self.ngram_params:
                if not p.requires_grad:
                    continue
                if self.table_optimizer == "adamw":
                    self._table_adamw_step(name, p, table_lr_t)
                elif self.table_optimizer == "sgd":
                    self._table_sgd_step(name, p, table_lr_t)
                else:
                    self._table_rmsprop_step(name, p, table_lr_t)

    def state_dict(self):
        return {
            "adam_steps": dict(self.adam_steps),
            "rms_steps": dict(self.rms_steps),
        }


def get_lr_multiplier(progress: float, warmdown_ratio: float = 0.65,
                      schedule: str = "warmup_constant",
                      warmup_steps: int = 100, step: int = 1,
                      max_steps: int = 1000,
                      warmup_start_lr_mult: float = 0.25,
                      cosine_min_lr_mult: float = 0.05) -> float:
    """LR multiplier by schedule.

    - "warmup_constant": linearly warm from warmup_start_lr_mult to 1.0x
      over warmup_steps optimizer updates, then hold the base LR fixed.
    - "warmup_cosine": the same step-anchored linear warmup, then cosine
      decay to cosine_min_lr_mult at max_steps. It is independent of epoch
      boundaries and is a schedule-search candidate, not the baseline.
    - "constant": zero-warmup fixed LR; retained as a diagnostic control.
    - "warmdown": historical Karpathy-style linear warmup (0->1 over first
      1-warmdown), then linear decay to 0.05.
    """
    if schedule == "constant":
        return 1.0
    if schedule in {"warmup_constant", "warmup_cosine"}:
        if warmup_steps <= 1:
            warmup_mult = 1.0
        else:
            warmup_progress = min(1.0, max(0, step - 1) / (warmup_steps - 1))
            warmup_mult = warmup_start_lr_mult + (1.0 - warmup_start_lr_mult) * warmup_progress
        if step <= warmup_steps or schedule == "warmup_constant":
            return warmup_mult
        decay_progress = min(
            1.0,
            max(0.0, (step - warmup_steps) / max(1, max_steps - warmup_steps)),
        )
        return cosine_min_lr_mult + 0.5 * (1.0 - cosine_min_lr_mult) * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    if schedule != "warmdown":
        raise ValueError(f"unknown lr schedule: {schedule}")
    if progress < 1.0 - warmdown_ratio:
        # warmup phase: linear from 0.1 to 1.0
        w = progress / max(1e-6, 1.0 - warmdown_ratio)
        return 0.1 + 0.9 * w
    else:
        # warmdown: linear from 1.0 to 0.05
        w = (progress - (1.0 - warmdown_ratio)) / max(1e-6, warmdown_ratio)
        return 1.0 - 0.95 * w


# ---------------------------------------------------------------------------
# 3. Data: tokenized shard loader (fixed-order epoch replay)
# ---------------------------------------------------------------------------


class TokenizedShardDataset:
    """Loads tokenized shards as packed uint16 sequences.

    Expects data_dir to contain shard_<id>.bin files (raw uint16 token ids,
    packed sequentially, BOS-aligned chunks of sequence_len+1 each).
    Fixed mode: iterates shards in order, yielding non-overlapping chunks;
    each epoch restarts from shard 0. This gives deterministic replay.
    """

    def __init__(self, data_dir: str, shard_ids: list, sequence_len: int,
                 device_batch_size: int, seed: int = 42, epoch_batches: int = 0,
                 replay_mix_passes: int = 0, token_dtype: str = "u16",
                 row_index_dir: str = "", static_low_mask: int = 0,
                 rows_requested: bool = False, max_tokens: int = 0):
        self.data_dir = data_dir
        self.shard_ids = list(shard_ids)
        self.sequence_len = sequence_len
        self.chunk_size = sequence_len + 1
        self.device_batch_size = device_batch_size
        self.seed = seed
        self.epoch_batches = epoch_batches
        self.replay_mix_passes = replay_mix_passes
        self._mix_order = None
        self._buffers = {}
        self._epoch = 0
        self._batch_in_epoch = 0
        # plan-8 infra: token dtype + offline row/validity streams (train fast path)
        self.token_np_dtype = np.uint32 if token_dtype == "u32" else np.uint16
        self.row_index_dir = row_index_dir
        self.static_low_mask = static_low_mask
        self.rows_requested = rows_requested
        self._row_buffers = {}
        # total tokens per shard (lazy)
        self._shard_lens = {}
        # yielded batches per full epoch (drop_last per shard), for epoch-anchored LR
        if max_tokens > 0:
            if epoch_batches > 0 or replay_mix_passes > 0:
                raise SystemExit("--train_max_tokens is mutually exclusive with "
                                 "--epoch_batches / --replay_mix_passes")
            # Token-granular nested-prefix pool: first floor(P/seq_len) chunks of
            # the concatenated shard stream, rounded down to whole device batches
            # so every epoch replays the identical chunk set (no cross-epoch tail).
            self._chunk_budget = (max_tokens // self.sequence_len) \
                // self.device_batch_size * self.device_batch_size
            if self._chunk_budget <= 0:
                raise SystemExit(
                    f"--train_max_tokens {max_tokens} is below one device batch "
                    f"({self.device_batch_size} chunks x {self.sequence_len} tokens)")
            self._batches_per_epoch = self._chunk_budget // self.device_batch_size
        elif epoch_batches > 0:
            # Nested-prefix epoch length: the first `epoch_batches` device
            # batches from the shard prefix form one epoch; all epoch lengths
            # are strict prefixes of the same data stream.
            self._chunk_budget = epoch_batches * self.device_batch_size
            self._batches_per_epoch = epoch_batches
        else:
            self._chunk_budget = None
            self._batches_per_epoch = sum(
                self.shard_chunks(sid) // self.device_batch_size for sid in self.shard_ids)
        if self._batches_per_epoch == 0:
            raise SystemExit(
                f"shards {self.shard_ids} contain fewer than one device batch of "
                f"complete chunks (chunk_size={self.chunk_size}); the fixed replay "
                "iterator would spin forever — check --sequence_len and shard sizes")

    def _load_shard(self, sid):
        if sid not in self._buffers:
            path = os.path.join(self.data_dir, f"shard_{sid:05d}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing shard: {path}")
            buf = np.memmap(path, dtype=self.token_np_dtype, mode="r")
            self._buffers[sid] = buf
        return self._buffers[sid]

    def _load_rows(self, sid: int):
        """(brows_memmap|None, trows_memmap|None, bmask|None, tmask|None)."""
        if not self.rows_requested and self.static_low_mask == 0:
            return (None, None, None, None)
        n_tok = len(self._load_shard(sid))
        streams = []
        for suffix, dt in (("brows.u32.bin", np.uint32), ("trows.u32.bin", np.uint32),
                           ("bmask.u8.bin", np.uint8), ("tmask.u8.bin", np.uint8)):
            path = os.path.join(self.row_index_dir or self.data_dir,
                                f"shard_{sid:05d}.{suffix}")
            if not os.path.exists(path):
                streams.append(None)
                continue
            stream = np.memmap(path, dtype=dt, mode="r")
            if len(stream) != n_tok:
                raise SystemExit(
                    f"row-index stream length mismatch for shard {sid}: "
                    f"{suffix} has {len(stream):,} entries, token stream has {n_tok:,}")
            self._row_buffers[(sid, suffix)] = stream
            streams.append(stream)
        brows, trows, bmask, tmask = streams
        if self.static_low_mask > 0 and (bmask is None or tmask is None):
            raise SystemExit(
                f"--static_low_mask requires bmask/tmask streams for shard {sid} "
                "(re-run precompute_row_index.py with --freq_index --min_count)")
        return (brows, trows, bmask, tmask)

    def _row_stream_tensor(self, sid: int, suffix: str):
        """Cached zero-copy torch view (uint32/uint8) of an offline stream.

        from_numpy on the memmap shares pages (no copy); chunk slicing yields
        views, so per-step host work is just the batch stack + one H2D
        transfer; dtype conversion happens on the GPU.
        """
        key = (sid, suffix, "torch")
        if key not in self._row_buffers:
            stream = self._row_buffers.get((sid, suffix))
            if stream is None:
                self._load_rows(sid)
                stream = self._row_buffers.get((sid, suffix))
            self._row_buffers[key] = torch.from_numpy(stream)
        return self._row_buffers[key]

    def shard_chunks(self, sid):
        buf = self._load_shard(sid)
        return len(buf) // self.chunk_size

    def total_chunks(self):
        total = sum(self.shard_chunks(s) for s in self.shard_ids)
        if self._chunk_budget is not None:
            return min(total, self._chunk_budget)
        return total

    def _iter_chunks(self, with_rows: bool = False):
        """Yield (input, target) tensors for each chunk in fixed order.

        With with_rows=True, yield (input, target, b_rows, t_rows, b_valid,
        t_valid) where the row/validity tensors are (B,)-shaped flat int64 /
        bool tensors from the offline index (plan-8 fast path); branches
        without an offline stream yield None in that slot.

        With epoch_batches > 0, only the first `epoch_batches * device_batch_size`
        chunks of the train prefix are yielded each epoch (nested-prefix control).

        With replay_mix_passes > 0, yield ONE globally chunk-shuffled stream of
        replay_mix_passes passes instead of ordered replay (shuffle experiment
        mixed arm). Chunk-level permutation preserves n-gram continuity within
        each chunk while removing epoch structure; the stream has no epoch
        boundaries, so `_epoch` stays 0 (no boundary interventions allowed on
        this arm).
        """
        if self.replay_mix_passes > 0:
            per_pass = (self.epoch_batches * self.device_batch_size
                        if self.epoch_batches > 0 else self.total_chunks())
            n_stream = self.replay_mix_passes * per_pass
            if self._mix_order is None:
                rng = np.random.default_rng(self.seed)
                self._mix_order = rng.permutation(n_stream)
            for g in self._mix_order:
                _, local = divmod(int(g), per_pass)
                yield self.get_chunk(local, with_rows=with_rows)
            return
        budget = self._chunk_budget
        emitted = 0
        for sid in self.shard_ids:
            buf = self._load_shard(sid)
            n = self.shard_chunks(sid)
            have_rows = with_rows and (self.rows_requested or self.static_low_mask > 0)
            if have_rows:
                self._load_rows(sid)
            for i in range(n):
                if budget is not None and emitted >= budget:
                    return
                start = i * self.chunk_size
                chunk = np.array(buf[start:start + self.chunk_size], dtype=np.int64)
                inp = torch.from_numpy(chunk[:-1])
                tgt = torch.from_numpy(chunk[1:])
                if have_rows:
                    T = self.chunk_size - 1

                    def _stream_view(suffix, dt):
                        # Canonical per-chunk dtypes for the offline streams:
                        # rows int64, validity bool (see get_chunk/_slice).
                        # Raw streams are uint32/uint8 memmaps; converting here
                        # keeps every consumer (model, verification, tests) on
                        # one dtype contract.
                        key = (sid, suffix, dt)
                        t = self._row_buffers.get(key)
                        if t is None:
                            stream = self._row_buffers.get((sid, suffix))
                            if stream is None:
                                return None
                            t = torch.from_numpy(np.ascontiguousarray(stream))
                            t = t.to(dt)
                            self._row_buffers[key] = t
                        return t[start:start + T]

                    yield inp, tgt, _stream_view("brows.u32.bin", torch.int64), \
                        _stream_view("trows.u32.bin", torch.int64), \
                        _stream_view("bmask.u8.bin", torch.bool), \
                        _stream_view("tmask.u8.bin", torch.bool)
                else:
                    yield inp, tgt
                emitted += 1

    def chunk_offsets(self) -> list:
        """Global chunk offset of each shard (cumulative over shard_chunks).

        Returns [(sid, offset_start), ...]; global chunk index g belongs to
        shard sid at local index g - offset_start.  Uniform train probes use
        this to sample the ENTIRE train stream, not just its head.
        """
        offs = []
        acc = 0
        for sid in self.shard_ids:
            offs.append((sid, acc))
            acc += self.shard_chunks(sid)
        return offs

    def get_chunk(self, global_idx: int, with_rows: bool = False):
        """Return (input, target) tensors for the g-th chunk of the full stream.

        Uniform train probe uses this to grab batches spread across the whole
        train set (independent of the training iterator, which is what the
        head-only `first` probe cannot do).

        with_rows=True appends the offline (b_rows, t_rows, b_valid, t_valid)
        tensors the same way _iter_chunks does (plan-8 fast path).
        """
        for sid, off in self.chunk_offsets():
            n = self.shard_chunks(sid)
            if off <= global_idx < off + n:
                buf = self._load_shard(sid)
                local = global_idx - off
                start = local * self.chunk_size
                chunk = np.array(buf[start:start + self.chunk_size], dtype=np.int64)
                inp = torch.from_numpy(chunk[:-1])
                tgt = torch.from_numpy(chunk[1:])
                if not with_rows:
                    return inp, tgt
                brows, trows, bmask, tmask = self._load_rows(sid)
                T = self.chunk_size - 1

                def _slice(stream, dt):
                    if stream is None:
                        return None
                    return torch.from_numpy(np.array(stream[start:start + T], dtype=dt))

                bi = _slice(brows, np.int64)
                ti = _slice(trows, np.int64)
                bv = _slice(bmask, np.int64).to(torch.bool) if bmask is not None else None
                tv = _slice(tmask, np.int64).to(torch.bool) if tmask is not None else None
                return inp, tgt, bi, ti, bv, tv
        raise IndexError(f"chunk {global_idx} out of range")

    def pool_content_tokens(self) -> "int | None":
        """Exact content-token size of the effective train pool (None = full
        shards, modulo per-shard drop_last in the legacy epoch-length path)."""
        if self._chunk_budget is not None:
            return self._chunk_budget * self.sequence_len
        return None

    def epoch_progress(self) -> float:
        """Fraction of the current epoch completed (0..1 within an epoch)."""
        return self._batch_in_epoch / max(1, self._batches_per_epoch)

    def iter_batches(self, device: torch.device, with_rows: bool = False):
        """Yield (inputs, targets) batches of shape (B, T) in fixed order, looping forever.

        with_rows=True yields (inputs, targets, bi, ti, bv, tv) device batches
        for the offline-index fast path (plan-8); bi/ti are (B,T) int64 row
        indices, bv/tv are (B,T) bool validity (None where the offline stream
        is absent, e.g. a disabled branch). Epoch/batch counters advance
        identically in both modes.
        """
        batch_inp, batch_tgt = [], []
        batch_rows: list = []
        while True:
            for chunk in self._iter_chunks(with_rows=with_rows):
                if with_rows:
                    inp, tgt, bi, ti, bv, tv = chunk
                    batch_rows.append((bi, ti, bv, tv))
                else:
                    inp, tgt = chunk
                batch_inp.append(inp)
                batch_tgt.append(tgt)
                if len(batch_inp) == self.device_batch_size:
                    out = (torch.stack(batch_inp).to(device),
                           torch.stack(batch_tgt).to(device))
                    if with_rows:
                        rows = []
                        for k in range(4):
                            parts = [r[k] for r in batch_rows]
                            if parts[0] is None:
                                rows.append(None)
                                continue
                            t = torch.stack(parts).to(device)
                            rows.append(t.to(torch.int64) if k < 2 else t.to(torch.bool))
                        out = out + tuple(rows)
                    yield out
                    batch_inp, batch_tgt, batch_rows = [], [], []
                    self._batch_in_epoch += 1
            self._epoch += 1
            self._batch_in_epoch = 0
            # fixed mode: no shuffle, deterministic replay


# ---------------------------------------------------------------------------
# 4. Table-norm observable
# ---------------------------------------------------------------------------


def table_param_rms(model: NanoGPT) -> dict:
    """Compute RMS of n-gram table params (bigram/trigram layer_1 table_0)."""
    out = {}
    for li in sorted(model.bigram_ve_layers):
        lvs = model.bigram_ves[str(li)]
        rms = lvs[0].weight.detach().float().pow(2).mean().sqrt().item()
        out[f"bigram.layer_{li:02d}.table_0.rms"] = rms
    for li in sorted(model.trigram_ve_layers):
        lvs = model.trigram_ves[str(li)]
        rms = lvs[0].weight.detach().float().pow(2).mean().sqrt().item()
        out[f"trigram.layer_{li:02d}.table_0.rms"] = rms
    for n in model.ngram_extra_orders:
        rms = model.ngram_ves[str(n)].weight.detach().float().pow(2).mean().sqrt().item()
        out[f"{n}gram.table_0.rms"] = rms
    return out


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _autocast_ctx(dtype: str):
    """Return the autocast context for a compute dtype spec.

    fp32  -> nullcontext (no autocast)
    bf16  -> torch.autocast bfloat16
    fp8   -> torch.autocast float8_e4m3fn (H200/TensorRT capable)
    """
    if dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype == "fp8":
        return torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn)
    return contextlib.nullcontext()


def evaluate_val(model: NanoGPT, fixed_val_batches, dtype: str = "fp32") -> float:
    """Val loss on a FIXED batch set captured once at startup and reused for
    every evaluation, so the val-loss curve always measures the same val data."""
    model.eval()
    losses = []
    ctx = _autocast_ctx(dtype)
    with torch.no_grad():
        for inp, tgt in fixed_val_batches:
            with ctx:
                loss = model(inp, targets=tgt)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_fixed_probe(model: NanoGPT, fixed_train_probe, dtype: str = "fp32") -> float:
    """Loss on the FIXED train probe (same train batches every eval).

    Returns the mean loss over the probe.  Used with evaluate_val to form
    the optional fixed-probe diagnostic gap = fixed_val − fixed_train_probe.
    The standard scaling gap is the online batch-averaged train loss."""
    model.eval()
    losses = []
    ctx = _autocast_ctx(dtype)
    with torch.no_grad():
        for inp, tgt in fixed_train_probe:
            with ctx:
                loss = model(inp, targets=tgt)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def _per_position_context_stats(model, batches, dtype):
    """Per-position (context_key, loss, set_membership) over the given fixed
    batches, moved to CPU numpy. Used by evaluate_context_gap."""
    keys_all, loss_all, memb_all = [], [], []
    ctx = _autocast_ctx(dtype)
    with torch.no_grad():
        for inp, tgt in batches:
            with ctx:
                _, ptl = model(inp, targets=tgt, return_token_losses=True)
            memb = model.context_set_membership(inp)
            prev = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
            prev2 = torch.cat([inp[:, :2], inp[:, :-2]], dim=1)
            V = model.config.vocab_size
            keys = prev2.long() * (V * V) + prev.long() * V + inp.long()
            valid = tgt.ne(-1)
            keys_all.append(keys[valid].cpu().numpy())
            loss_all.append(ptl[valid].double().cpu().numpy())
            memb_all.append(memb[valid].cpu().numpy())
    return (np.concatenate(keys_all), np.concatenate(loss_all),
            np.concatenate(memb_all))


def evaluate_context_gap(model, fixed_train_probe, fixed_val_batches,
                         dtype: str = "fp32") -> dict:
    """Gap contribution of context sets A (masked) and B (peers), matching the
    paper's C_S(t) = sum_{i in S} q_i [l_val,i(t) - l_tr,i(t)]: q_i is the
    context's share of FIXED-VAL positions, l_val,i / l_tr,i are per-context
    mean losses on the fixed val batches and the fixed train probe,
    restricted to contexts present in both. Also logs plain position-level
    means and shares for diagnostics."""
    model.eval()
    tr_keys, tr_loss, _ = _per_position_context_stats(model, fixed_train_probe, dtype)
    va_keys, va_loss, va_memb = _per_position_context_stats(model, fixed_val_batches, dtype)
    model.train()

    tu, t_inv = np.unique(tr_keys, return_inverse=True)
    tr_sum = np.bincount(t_inv, weights=tr_loss)
    tr_cnt = np.bincount(t_inv).astype(np.float64)
    vu, v_inv = np.unique(va_keys, return_inverse=True)
    va_sum = np.bincount(v_inv, weights=va_loss)
    va_cnt = np.bincount(v_inv).astype(np.float64)
    # membership of each unique val context (first occurrence)
    v_first = np.unique(va_keys, return_index=True)[1]
    va_memb_u = va_memb[v_first]
    shared, t_idx, v_idx = np.intersect1d(tu, vu, assume_unique=True, return_indices=True)

    out = {}
    n_val_pos = float(va_cnt.sum())
    for tag, code in (("A", 1), ("B", 2)):
        in_set = va_memb_u[v_idx] == code
        q = va_cnt[v_idx] / n_val_pos
        gap_i = (va_sum[v_idx] / va_cnt[v_idx]) - (tr_sum[t_idx] / tr_cnt[t_idx])
        out[f"C_{tag}"] = float((q[in_set] * gap_i[in_set]).sum())
        out[f"n_join_ctx_{tag}"] = int(in_set.sum())
        pos = va_memb == code
        out[f"val_mean_{tag}"] = float(va_loss[pos].mean()) if pos.any() else float("nan")
        out[f"val_share_{tag}"] = float(pos.mean())
        # train-side mean over the same context keys (joined subset)
        tj = t_idx[in_set]
        out[f"tr_mean_{tag}"] = float((tr_sum[tj] / tr_cnt[tj]).mean()) if tj.size else float("nan")
    return out


def dump_token_stats_npz(out_path, model, batches, orders, vocab_size,
                         dtype_spec: str) -> dict:
    """Per-token loss + per-order hashed context keys for post-hoc gap-freq
    statistics (user directive 2026-10-17: every sweep run must allow exact
    per-token analysis).  Keys use ngram_freq.context_keys_np — the same hash
    as the frequency index — so pool-relative f lookups join bit for bit.

    Forward follows the diag convention: autocast per `dtype_spec` (bf16 for
    the standard bf16/fp8 runs), matching exact_freq_loss rows rather than the
    fp32 scalar val.  Rows are flat over (batch, position) with valid
    (target != -1) positions kept in batch-major order; `n_per_batch` records
    the split points.
    """
    from ngram_freq import context_keys_np
    ctx = _autocast_ctx(dtype_spec)
    loss_all, n_per_batch = [], []
    key_all = {int(n): [] for n in orders}
    model.eval()
    with torch.no_grad():
        for inp, tgt in batches:
            with ctx:
                _, ptl = model(inp, targets=tgt, return_token_losses=True)
            valid = tgt.ne(-1)
            loss_all.append(ptl[valid].float().cpu().numpy())
            n_per_batch.append(int(valid.sum()))
            toks = inp.cpu().numpy()
            vmask = valid.cpu().numpy()
            for n in key_all:
                k = context_keys_np(toks, n, vocab_size)
                key_all[n].append(k[vmask])
    model.train()
    arrays = {
        "loss": np.concatenate(loss_all).astype(np.float32),
        "n_per_batch": np.array(n_per_batch, dtype=np.int64),
        "orders": np.array(sorted(key_all), dtype=np.int64),
    }
    for n in sorted(key_all):
        arrays[f"key_{n}"] = np.concatenate(key_all[n]).astype(np.int64)
    np.savez(out_path, **arrays)
    return {"path": out_path, "tokens": int(sum(n_per_batch))}


def accumulate_freq_bins(freq_index, inp, tgt, per_token_loss, vocab_size) -> dict:
    """Accumulate per-frequency-bin stats from ONE already-forwarded batch.

    Reuses the per-token loss of the current training batch (pre-update, the
    same quantity as the online train loss), so freq-bin train-side stats are
    a pure online measurement with zero extra compute.
    """
    from ngram_freq import FreqBinLossAccumulator
    accs = {
        "bigram": FreqBinLossAccumulator(freq_index, vocab_size, "bigram"),
        "trigram": FreqBinLossAccumulator(freq_index, vocab_size, "trigram"),
    }
    for branch in accs:
        accs[branch].update(inp, per_token_loss)
    return {branch: acc.summary() for branch, acc in accs.items()}


def evaluate_freq_bins(model: NanoGPT, loader, freq_index, n_batches: int,
                       vocab_size: int) -> dict:
    """Run per-frequency-bin loss accumulation on n_batches from loader.

    Returns dict with 'bigram' and 'trigram' keys, each mapping to
    {bucket_label: {frac, mean_loss, total_contrib, token_count}}.
    """
    from ngram_freq import FreqBinLossAccumulator, compute_per_token_loss
    model.eval()
    accs = {
        "bigram": FreqBinLossAccumulator(freq_index, vocab_size, "bigram"),
        "trigram": FreqBinLossAccumulator(freq_index, vocab_size, "trigram"),
    }
    with torch.no_grad():
        loader_iter = iter(loader)
        for _ in range(n_batches):
            try:
                inp, tgt = next(loader_iter)
            except StopIteration:
                break
            ptl = compute_per_token_loss(model, inp, tgt)
            for branch in accs:
                accs[branch].update(inp, ptl)
    model.train()
    return {branch: acc.summary() for branch, acc in accs.items()}


def evaluate_exact_freq(model: NanoGPT, loader, freq_index, n_batches: int,
                        vocab_size: int, branch: str) -> dict:
    """Run exact-frequency loss accumulation on n_batches from loader.

    Returns {exact_f: {token_count, distinct_contexts, mean_loss, loss_sum,
    loss_sq_sum}} for the given branch ("bigram"|"trigram").  Uses an
    independent iterator over `loader` (a list of batches or an iterable),
    so it never consumes the training stream.
    """
    from ngram_freq import ExactFreqLossAccumulator, compute_per_token_loss
    model.eval()
    acc = ExactFreqLossAccumulator(freq_index, vocab_size, branch)
    with torch.no_grad():
        loader_iter = iter(loader)
        for _ in range(n_batches):
            try:
                inp, tgt = next(loader_iter)
            except StopIteration:
                break
            ptl = compute_per_token_loss(model, inp, tgt)
            acc.update(inp, ptl)
    model.train()
    return acc.summary()


def compute_shared_contexts(model: NanoGPT, train_probe, val_probe,
                            freq_index, vocab_size, branch: str) -> dict:
    """Context-matched gap statistics.

    For contexts that appear in BOTH the fixed train probe and the fixed val
    probe, compute per-exact-f aggregated (train_loss, val_loss) differences.
    Returns {f: {shared_contexts, train_mean, val_mean, gap}} plus a count of
    shared contexts overall.
    """
    from ngram_freq import ExactFreqLossAccumulator, compute_per_token_loss
    model.eval()
    train_keys: dict = {}
    val_keys: dict = {}
    with torch.no_grad():
        for inp, tgt in train_probe:
            ptl = compute_per_token_loss(model, inp, tgt)
            acc = ExactFreqLossAccumulator(freq_index, vocab_size, branch)
            keys = acc._compute_keys(inp).cpu().numpy().ravel()
            losses = ptl.cpu().numpy().ravel()
            for k, l in zip(keys, losses):
                train_keys[int(k)] = float(l)
        for inp, tgt in val_probe:
            ptl = compute_per_token_loss(model, inp, tgt)
            acc = ExactFreqLossAccumulator(freq_index, vocab_size, branch)
            keys = acc._compute_keys(inp).cpu().numpy().ravel()
            losses = ptl.cpu().numpy().ravel()
            for k, l in zip(keys, losses):
                val_keys[int(k)] = float(l)
    # contexts present in both
    shared = set(train_keys) & set(val_keys)
    per_f = {}
    for k in shared:
        f = freq_index.bigram.get(k, 0) if branch == "bigram" else freq_index.trigram.get(k, 0)
        st = per_f.setdefault(int(f), {"shared_contexts": 0, "train_sum": 0.0, "val_sum": 0.0})
        st["shared_contexts"] += 1
        st["train_sum"] += train_keys[k]
        st["val_sum"] += val_keys[k]
    out = {}
    for f, st in per_f.items():
        n = st["shared_contexts"]
        out[f] = {
            "f": f,
            "shared_contexts": n,
            "train_mean": st["train_sum"] / n,
            "val_mean": st["val_sum"] / n,
            "gap": (st["val_sum"] - st["train_sum"]) / n,
        }
    model.train()
    return {"shared_total": len(shared), "per_f": out, "branch": branch}


def count_init_clean_tables(model, npz_path: str, log=print) -> dict:
    """Overwrite clean single-table embeddings with deterministic count-based
    content (§80 golf-equivalent arm): each row becomes the count-weighted
    empirical mean of the init wte rows of its bucket's continuation tokens,
        row_r = (1/C_r) * sum_{keys k in bucket r} c_k * wte[cur(k)],
    with zero rows where C_r = 0. Content is pure frequency statistics of the
    training text (no learning); pair with --freeze_table to keep it fixed.
    Bucketing uses the model's own clean hash (K=1 families), so rows line up
    exactly with the contexts the forward pass addresses. Bigram keys decode
    as prev*V+cur, trigram keys as prev2*V^2+prev*V+cur (boundary padding in
    the index matches the model's chunk-start padding)."""
    from ngram_freq import GlobalFrequencyIndex

    if getattr(model, "bigram_perfect", False):
        raise SystemExit("--table_init_count does not support perfect-map tables")
    if model.bigram_K != 1 or model.trigram_K != 1:
        raise SystemExit("--table_init_count requires clean single-table branches (K=1)")
    if len(model.bigram_ves) != 1 or len(model.trigram_ves) != 1:
        raise SystemExit("--table_init_count supports exactly one table layer per branch "
                         "(standard input injection); multi-layer v/y is out of scope")
    fi = GlobalFrequencyIndex.load(npz_path)
    fi._ensure_sorted()
    V = model.config.vocab_size
    wte = model.transformer.wte.weight.detach().to(torch.float32)
    D = wte.shape[1]
    for branch, counts in (("bigram", fi._bigram_counts_sorted),
                           ("trigram", fi._trigram_counts_sorted)):
        if (np.asarray(counts, dtype=np.int64) <= 0).any():
            raise SystemExit(f"--table_init_count: {branch} counts contain non-positive "
                             "entries (corrupt freq index); refusing silent wrap")
    out = {}
    for branch, keys, counts, table_attr, table_size in (
        ("bigram", fi._bigram_keys_sorted, fi._bigram_counts_sorted,
         model.bigram_ves, model.bigram_table_size),
        ("trigram", fi._trigram_keys_sorted, fi._trigram_counts_sorted,
         model.trigram_ves, model.trigram_table_size),
    ):
        keys64 = np.ascontiguousarray(keys, dtype=np.int64)
        counts64 = np.ascontiguousarray(counts, dtype=np.int64)
        cur = keys64 % V
        if branch == "bigram":
            prev = keys64 // V
            # per-layer value = list of K hash families; K=1 clean uses [0]
            p1, p2 = next(iter(model.bigram_hash_primes_per_layer.values()))[0][:2]
            rows = ((prev * p1) ^ (cur * p2)) % table_size
        else:
            prev = (keys64 // V) % V
            prev2 = keys64 // (V * V)
            q0, q1, q2 = next(iter(model.trigram_hash_primes_per_layer.values()))[:3]
            rows = ((prev2 * q0) ^ (prev * q1) ^ (cur * q2)) % table_size
        acc = torch.zeros(table_size, D, dtype=torch.float32, device=wte.device)
        cnt = torch.zeros(table_size, dtype=torch.float64, device=wte.device)
        rows_t = torch.from_numpy(rows).to(wte.device)
        cur_t = torch.from_numpy(cur).to(wte.device)
        c_t = torch.from_numpy(counts64.astype(np.float64)).to(wte.device)
        CH = 1_000_000
        for s in range(0, rows_t.numel(), CH):
            e = min(s + CH, rows_t.numel())
            wc = wte[cur_t[s:e]]
            acc.index_add_(0, rows_t[s:e], (c_t[s:e].unsqueeze(1) * wc).float())
            cnt.index_add_(0, rows_t[s:e], c_t[s:e])
        row = acc / cnt.clamp(min=1.0).unsqueeze(1).float()
        row[cnt == 0] = 0.0
        written = 0
        for li, mlist in table_attr.items():
            emb = mlist[0]
            if emb.weight.shape != row.shape:
                raise SystemExit(
                    f"--table_init_count shape mismatch for {branch} layer {li}: "
                    f"table {tuple(emb.weight.shape)} vs init {tuple(row.shape)}")
            emb.weight.data.copy_(row.to(emb.weight.dtype))
            written += row.numel()
        nz = int((cnt > 0).sum())
        norms = row.norm(dim=1)
        stats = {
            "branch": branch,
            "keys": int(keys64.size),
            "rows_nonzero": nz,
            "rows_total": int(table_size),
            "mean_row_norm": float(norms[norms > 0].mean()) if nz else 0.0,
            "max_row_norm": float(norms.max()) if nz else 0.0,
            "params_written": written,
        }
        out[branch] = stats
        log(f"[nglab] table_init_count[{branch}]: {stats['keys']:,} keys -> "
            f"{nz:,}/{table_size:,} nonzero rows; mean|row| {stats['mean_row_norm']:.4f}, "
            f"max|row| {stats['max_row_norm']:.4f}")
    return out


def steps_for_epochs(n_epochs: int, batches_per_epoch: int, device_batch_size: int,
                     sequence_len: int, total_batch_size: int) -> int:
    """Optimizer steps needed to see exactly n_epochs of the effective pool.

    The pool size in tokens is batches_per_epoch * device_batch_size *
    sequence_len (drop_last already applied at the batch level); steps round UP
    so the last, partial step never shortchanges the requested epoch count.
    """
    tokens_per_epoch = batches_per_epoch * device_batch_size * sequence_len
    return -(-n_epochs * tokens_per_epoch // total_batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="run")
    parser.add_argument("--injection_position", default="input",
                        choices=["v", "y", "engram", "input"])
    parser.add_argument("--inject_layers", default="",
                        help="v/y only: comma layer ids for multi-layer injection, "
                             "one independent clean table per layer (e.g. '1,3,5,7'); "
                             "empty = default single-layer layout")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train_epochs", type=int, default=0,
                        help=">0: run exactly this many epochs over the effective "
                             "train pool; --steps is derived as ceil(E * pool / "
                             "total_batch) and must be left at its default.  The "
                             "epoch-count-matched protocol for data-size envelopes "
                             "(2026-10-17 user directive)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", type=int, default=0,
                        help="1 = strict run-to-run determinism: "
                             "torch.use_deterministic_algorithms(True) + math-only SDPA "
                             "(flash/mem-efficient backward are non-deterministic) + "
                             "CUBLAS_WORKSPACE_CONFIG. Slower; bitwise only within the "
                             "same GPU/driver/torch env.")
    parser.add_argument("--data_dir", default=os.environ.get("NGLAB_DATA_DIR", ""))
    parser.add_argument("--out_dir", default=os.environ.get("NGLAB_OUT_DIR", "data/runs_fixed"))
    parser.add_argument("--train_shards", default=os.environ.get("NGLAB_TRAIN_SHARDS", "1"),
                        help="comma-separated shard ids for train")
    parser.add_argument("--val_shards", default=os.environ.get("NGLAB_VAL_SHARDS", "2,3,4,5,6,7,8,9,10,6542"),
                        help="comma-separated shard ids for val")
    parser.add_argument("--device_batch_size", type=int, default=72)
    parser.add_argument("--total_batch_size", type=int, default=147456)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--val_steps", default="",
                        help="comma-separated exact steps at which to run val eval, "
                             "e.g. '1000' to measure gap only at the end; "
                             "overrides --val_interval when set")
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument("--table_norm_interval", type=int, default=10)
    parser.add_argument("--table_diag", type=int, default=0,
                        help="1: log per-step table-update diagnostics to table_diag.jsonl "
                             "(replay-shock fingerprint: hit rows, |g|, denom, |g|/denom amplification)")
    parser.add_argument("--save_final_model", action="store_true",
                        help="save the final model.state_dict() to final_model.pt "
                             "and the fixed train/val probe tokens to probe_tokens.npz "
                             "for offline per-token / row-level loss analysis")
    parser.add_argument("--save_epoch_checkpoints", action="store_true",
                        help="with --epoch_batches, save epoch-boundary weights as "
                             "epoch{n}_model.pt in addition to final_model.pt")
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="AdamW weight decay for backbone matrices (default 0.1; "
                             "tables are never decayed)")
    parser.add_argument("--table_optimizer", default="rmsprop",
                        choices=["rmsprop", "adamw", "sgd"],
                        help="optimizer for n-gram table params (default rmsprop)")
    parser.add_argument("--table_lr_scale", type=float, default=2.0,
                        help="multiplier on the n-gram table LR (default 2.0; the SSOT standard)")
    parser.add_argument("--table_betas", default=None,
                        help="beta1,beta2 for table (adamw: both; sgd: beta1=momentum); default 0.0,0.99")
    parser.add_argument("--table_mult", type=int, default=64,
                        help="[legacy only] multi-layer/two-hash table multiplier (default 64)")
    parser.add_argument("--bigram_clean_table", type=int, default=0,
                        help="clean single-table rows R (0=off): one nn.Embedding(R, n_embd), "
                             "single layer, single hash, R freely set; required for a new main-line "
                             "bigram run. SSOT: "
                             "docs/notes/method/clean-table-rework.md")
    parser.add_argument("--trigram_clean_table", type=int, default=0,
                        help="clean single-table rows R for the trigram branch (0=off); required "
                             "for a new main-line trigram run; same SSOT as --bigram_clean_table")
    parser.add_argument("--bigram_table_dim", type=int, default=0,
                        help="X2: clean-table row width d (0=n_embd); rows are read through a "
                             "frozen zero-pad projection back to n_embd; only valid with "
                             "--bigram_clean_table")
    parser.add_argument("--trigram_table_dim", type=int, default=0,
                        help="X2: trigram clean-table row width d (0=n_embd); same rule as "
                             "--bigram_table_dim")
    parser.add_argument("--bigram_perfect_map", default="",
                        help="path to npz with int32 packed-bigram->row map; enables "
                             "collision-free bigram table (first ngram layer only, "
                             "rows = n_distinct+1, OOV -> shared UNK row)")
    parser.add_argument("--bigram_single_layer", action="store_true",
                        help="restrict the bigram table to the first ngram layer "
                             "(control arm for the collision-free perfect-map run)")
    parser.add_argument("--intervention", default="none",
                        choices=["none", "freeze_table", "freeze_backbone", "freeze_both", "hash_reseed",
                                 "mask_low_freq", "mask_high_freq", "readout_last_epoch",
                                 "random_context_loss_mask"],
                        help="causal intervention fired at intervention_epoch boundary")
    parser.add_argument("--intervention_epoch", type=int, default=1,
                        help="0-indexed epoch at which the intervention fires (1 = start of epoch 2)")
    parser.add_argument("--intervention_epochs", default="",
                        help="comma-separated extra 0-indexed due epochs for multi-fire "
                             "interventions, e.g. \"1,2\" = fire at the start of epochs 2 and 3")
    parser.add_argument("--intervention_freq_threshold", type=int, default=200,
                        help="mask threshold: mask_high masks f >= threshold; mask_low masks "
                             "f < threshold (f <= threshold with --intervention_low_inclusive 1); "
                             "novel f=0 contexts are always masked in low mode, at train and eval")
    parser.add_argument("--intervention_low_inclusive", type=int, default=0,
                        help="1: mask_low masks f <= threshold instead of f < threshold")
    parser.add_argument("--intervention_mask_frac", type=float, default=0.02,
                        help="random_context_loss_mask: fraction of distinct trigram context "
                             "types to exclude from the loss (default 0.02)")
    parser.add_argument("--context_eval_ab", type=int, default=0,
                        help="1: log per-eval A/B context gap contributions to "
                             "context_gap.jsonl (auto-enabled for random_context_loss_mask)")
    parser.add_argument("--context_set_mode", choices=["type", "row"], default="type",
                        help="A/B selection space for random_context_loss_mask: "
                             "'type' = distinct trigram types (default), "
                             "'row' = trigram table rows (collision-private subspaces)")
    parser.add_argument("--ngram_orders", default="",
                        help="v2: comma n-gram orders to over-encode, e.g. '5' or '2,3,4,5'. "
                             "Overrides --enable_bigram/--enable_trigram (2/3 keep their legacy "
                             "branches and --*_clean_table); orders >= 4 get generic clean single "
                             "tables of --ngram_clean_table rows. Empty = legacy flags")
    parser.add_argument("--ngram_clean_table", type=int, default=0,
                        help="rows R of each generic order >= 4 table (required with such orders)")
    parser.add_argument("--ngram_dict_topk", type=int, default=0,
                        help="v2.1 no-collision dictionary mode: per order in --ngram_orders "
                             "(2/3 included, routed through the generic path), keep the top-K "
                             "train contexts by count as unique rows (miss contributes zero). "
                             "Requires --freq_index with those orders; K <= --ngram_clean_table")
    parser.add_argument("--ngram_keep_rows", default="dict", choices=["dict", "hash"],
                        help="with --ngram_dict_topk: dict = unique rows (no-collision dictionary); "
                             "hash = hash rows masked to the same top-K membership (coverage-matched "
                             "control: separates collisions from coverage)")
    parser.add_argument("--diag_workers", type=int, default=1,
                        help="background diag worker processes (v2 multi-order index: >1 keeps "
                             "up with 7 orders; rows may land out of step order)")
    parser.add_argument("--enable_unigram", type=int, default=0)
    parser.add_argument("--enable_bigram", type=int, default=1)
    parser.add_argument("--enable_trigram", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--sequence_len", type=int, default=2048)
    parser.add_argument("--freq_index", default="",
                        help="path to freq_index.npz; if set, enables freq-bin eval")
    parser.add_argument("--freq_eval_interval", type=int, default=10)
    parser.add_argument("--freq_eval_batches", type=int, default=4)
    parser.add_argument("--exact_freq_eval_interval", type=int, default=10,
                        help="exact-frequency interval when --val_steps is unset; "
                             "otherwise follows the specified validation steps")
    parser.add_argument("--logit_stats", action="store_true",
                        help="enable confidence-polarization probe (§5.2 supplement): "
                             "per-token entropy/margin/p_true + top-20 softmax, aggregated "
                             "by (branch × f_bucket × seen/novel) in logit_stats.jsonl. "
                             "Reuses the existing bf16 diagnostic forward; no extra forward.")
    parser.add_argument("--logit_stats_exemplars", type=int, default=64,
                        help="number of fixed exemplar contexts to dump full top-20 each eval "
                             "(default 64; 16 per quadrant of freq×seen/novel)")
    parser.add_argument("--lr_schedule_epochs", type=int, default=0,
                        help=">0: anchor LR schedule to this many epochs (epoch-based progress)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="optimizer updates for warmup_constant/warmup_cosine (default 100; "
                             "LR rises linearly from 0.25x (1e-3) at step 1 to 1.0x (4e-3) "
                             "at this step)")
    parser.add_argument("--warmup_start_lr_mult", type=float, default=0.25,
                        help="initial LR multiplier for warmup_constant/warmup_cosine "
                             "(default 0.25)")
    parser.add_argument("--cosine_min_lr_mult", type=float, default=0.05,
                        help="terminal LR multiplier for warmup_cosine (default 0.05)")
    parser.add_argument("--lr_schedule", default="warmup_constant",
                        choices=["warmup_constant", "warmup_cosine", "constant", "warmdown"],
                        help="LR schedule: warmup_constant (current default) | warmup_cosine "
                             "(schedule-search candidate) | constant (zero-warmup diagnostic) | "
                             "warmdown (historical control)")
    parser.add_argument("--epoch_batches", type=int, default=0,
                        help=">0: fix one epoch to exactly this many device batches "
                             "(nested-prefix epoch length); 0 = full shard length")
    parser.add_argument("--train_max_tokens", type=int, default=0,
                        help=">0: train pool = first P content tokens of the train "
                             "shard stream (token-granular nested-prefix truncation, "
                             "rounded down to whole device batches; val stream "
                             "unaffected). Mutually exclusive with --epoch_batches / "
                             "--replay_mix_passes")
    parser.add_argument("--dump_token_stats_steps", default="",
                        help="comma-separated 1-based steps at which to dump per-token "
                             "loss + per-order context keys (token_stats_step*_*.npz, "
                             "post-hoc gap-freq analysis). Empty = final step only; the "
                             "final step is always dumped")
    parser.add_argument("--replay_mix_passes", type=int, default=0,
                        help=">0: shuffle experiment mixed arm — consume ONE globally "
                             "chunk-shuffled stream of N passes (chunk-level permutation "
                             "preserves n-gram continuity) instead of ordered epoch replay")
    parser.add_argument("--fixed_train_probe", type=int, default=0,
                        help="number of fixed train batches to hold for the "
                             "fixed-train probe (0 disables)")
    parser.add_argument("--train_probe_mode", default="first",
                        choices=["first", "uniform"],
                        help="how the fixed train probe is sampled from the train "
                             "set: 'first' = head of the stream (default, backward "
                             "compatible); 'uniform' = evenly-spaced chunks across "
                             "the ENTIRE train stream (parallel to the fixed val "
                             "probe; avoids head-of-epoch memorization bias)")
    parser.add_argument("--probe_eval_interval", type=int, default=10,
                        help="steps between fixed-train/val probe evals (also fired "
                             "at every epoch boundary)")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp8"],
                        help="compute dtype for the forward pass (autocast; "
                             "weights/optimizer stay fp32): fp32 | bf16 | fp8")
    parser.add_argument("--compile", action="store_true",
                        help="wrap the model with torch.compile (opt-in; not part "
                             "of the standard bf16 configuration)")
    parser.add_argument("--init_model", default="",
                        help="initialize model weights from a compatible final_model.pt state_dict")
    parser.add_argument("--table_init_count", default="",
                        help="overwrite clean single-table embeddings with deterministic "
                             "count-based content (count-weighted mean of init-wte rows of "
                             "each hash bucket, from a freq_index npz of the TRAIN text; "
                             "§80 golf-equivalent arm). Pair with --freeze_table. "
                             "Standard input injection, K=1, single layer only")
    parser.add_argument("--freeze_table", action="store_true",
                        help="freeze n-gram tables and gates after initialization; optimizer state starts empty")
    parser.add_argument("--freeze_embeddings_only", action="store_true",
                        help="freeze lookup embeddings after initialization; gates remain trainable")
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="after checkpoint load, freeze backbone weights and train rank-r LoRA on block linear layers")
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--disable_ngram_tables", action="store_true",
                        help="zero and freeze n-gram table/gate parameters after loading; matched no-gram control")
    parser.add_argument("--row_index", action="store_true",
                        help="plan-8 fast path: consume offline precomputed n-gram row "
                             "indices (pure-gather forward, compile-friendly). Requires "
                             "code/precompute_row_index.py output next to the shards; "
                             "input injection + clean single-table branches only; "
                             "verified bitwise against the dynamic hash at startup")
    parser.add_argument("--row_index_dir", default="",
                        help="directory holding shard_<id>.{b,t}rows.u32.bin (default: data_dir)")
    parser.add_argument("--token_dtype", default="u16", choices=["u16", "u32"],
                        help="token shard dtype; u32 required for vocab > 65535")
    parser.add_argument("--ce_chunk_positions", type=int, default=0,
                        help="chunked checkpoint CE over the flattened (B*T) positions: "
                             "0=auto (on only when full bf16 logits would exceed 4GiB), "
                             "-1=off, >0=positions per chunk")
    parser.add_argument("--static_low_mask", type=int, default=0,
                        help="plan-8 G3 prevention arm: apply offline validity bits "
                             "(context f >= t stays enabled) from step 0; requires "
                             "--row_index and precomputed mask streams. Do NOT combine "
                             "with the mask_low_freq intervention")
    parser.add_argument("--static_low_mask_keep_novel", type=int, default=1,
                        help="1: novel (f==0) contexts stay enabled in the static mask "
                             "(le0 evidence: novel table readout is net positive for val)")
    parser.add_argument("--fp8_torchao", action="store_true",
                        help="torchao float8 training on backbone Linear modules only "
                             "(lm_head and all n-gram tables/gates stay bf16/fp32); "
                             "requires torchao; run the bf16 pairing before adopting")
    args = parser.parse_args()
    if args.freeze_embeddings_only and (args.freeze_table or args.disable_ngram_tables):
        parser.error("--freeze_embeddings_only is exclusive with --freeze_table/--disable_ngram_tables")
    if args.lora_rank < 0 or (args.lora_rank and (not args.init_model or args.lora_alpha <= 0)):
        parser.error("LoRA requires --init_model, --lora_rank > 0 and --lora_alpha > 0")
    if args.disable_ngram_tables and (not args.init_model or args.injection_position != "input"):
        parser.error("--disable_ngram_tables requires --init_model and input injection")
    if args.warmup_steps < 1:
        parser.error("--warmup_steps must be >= 1")
    if not 0.0 <= args.warmup_start_lr_mult <= 1.0:
        parser.error("--warmup_start_lr_mult must be in [0, 1]")
    if not 0.0 <= args.cosine_min_lr_mult <= 1.0:
        parser.error("--cosine_min_lr_mult must be in [0, 1]")
    if args.intervention_freq_threshold < 0:
        parser.error("--intervention_freq_threshold must be non-negative")
    if args.intervention in {"mask_low_freq", "mask_high_freq"} and (
        not args.freq_index or not os.path.isfile(args.freq_index)
    ):
        parser.error(f"--intervention {args.intervention} requires an existing --freq_index")
    if (args.intervention == "random_context_loss_mask"
            and args.context_set_mode == "type"
            and (not args.freq_index or not os.path.isfile(args.freq_index))):
        parser.error("--intervention random_context_loss_mask (type mode) requires "
                     "an existing --freq_index")
    if args.intervention == "none" and args.intervention_epochs.strip():
        parser.error("--intervention_epochs requires --intervention hash_reseed (or another intervention)")
    if args.vocab_size > 65535 and args.token_dtype != "u32":
        parser.error("--vocab_size > 65535 requires --token_dtype u32")
    if args.static_low_mask > 0 and not args.row_index:
        parser.error("--static_low_mask requires --row_index (validity bits ride on the offline index)")
    if args.static_low_mask > 0 and args.intervention in {"mask_low_freq", "mask_high_freq"}:
        parser.error("--static_low_mask must not be combined with the mask_low/high_freq "
                     "intervention (prevention vs intervention arms are separate)")
    if args.row_index and args.intervention == "hash_reseed":
        parser.error("--row_index is incompatible with --intervention hash_reseed "
                     "(reseed changes the hash after the offline index was built)")
    ngram_orders = sorted({int(x) for x in args.ngram_orders.split(",") if x.strip()})
    if args.ngram_dict_topk > 0:
        if not ngram_orders:
            parser.error("--ngram_dict_topk requires --ngram_orders")
        if not args.freq_index:
            parser.error("--ngram_dict_topk requires --freq_index (dictionaries are built from it)")
        if args.row_index:
            parser.error("--ngram_dict_topk is incompatible with --row_index")
        if args.ngram_dict_topk > args.ngram_clean_table > 0 or \
                (args.ngram_dict_topk > 0 and args.ngram_clean_table <= 0):
            parser.error("--ngram_dict_topk must be <= --ngram_clean_table (and R > 0)")
    if args.ngram_keep_rows == "hash" and args.ngram_dict_topk <= 0:
        parser.error("--ngram_keep_rows hash requires --ngram_dict_topk > 0 (membership mask)")
    if ngram_orders:
        if min(ngram_orders) < 2 or max(ngram_orders) > 16:
            parser.error("--ngram_orders must lie in [2, 16]")
        if args.ngram_dict_topk > 0:
            extra_orders = tuple(ngram_orders)      # 2/3 also take the generic path
            args.enable_bigram = 0
            args.enable_trigram = 0
        else:
            extra_orders = tuple(n for n in ngram_orders if n >= 4)
            args.enable_bigram = int(2 in ngram_orders)
            args.enable_trigram = int(3 in ngram_orders)
    else:
        extra_orders = ()
    if extra_orders and args.ngram_clean_table <= 0:
        parser.error("--ngram_orders requires --ngram_clean_table > 0")
    if extra_orders and args.injection_position != "input":
        parser.error("--ngram_orders supports --injection_position input only")
    if extra_orders and args.intervention in {"mask_low_freq", "mask_high_freq",
                                              "readout_last_epoch", "hash_reseed"}:
        parser.error(f"--intervention {args.intervention} does not cover orders >= 4")

    cfg = Config(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        sequence_len=args.sequence_len,
        nanogpt_ngram_injection_position=args.injection_position,
        nanogpt_inject_layers=args.inject_layers,
        enable_unigram_ve=bool(args.enable_unigram),
        enable_bigram_ve=bool(args.enable_bigram),
        enable_trigram_ve=bool(args.enable_trigram),
        seed=args.seed,
        max_steps=args.steps,
        device_batch_size=args.device_batch_size,
        total_batch_size=args.total_batch_size,
        val_interval_steps=args.val_interval,
        val_steps=[int(x) for x in args.val_steps.split(",") if x.strip()],
        val_batches=args.val_batches,
        table_norm_interval_steps=args.table_norm_interval,
        warmup_steps=args.warmup_steps,
        warmup_start_lr_mult=args.warmup_start_lr_mult,
        cosine_min_lr_mult=args.cosine_min_lr_mult,
        lr_schedule_epochs=args.lr_schedule_epochs,
        lr_schedule=args.lr_schedule,
        nanogpt_adam_lr=args.lr,
        weight_decay=args.weight_decay,
        table_optimizer=args.table_optimizer,
        table_lr_scale=args.table_lr_scale,
        table_mult=args.table_mult,
        bigram_clean_table=args.bigram_clean_table,
        trigram_clean_table=args.trigram_clean_table,
        bigram_table_dim=args.bigram_table_dim,
        trigram_table_dim=args.trigram_table_dim,
        bigram_perfect_map=args.bigram_perfect_map,
        bigram_single_layer=bool(args.bigram_single_layer),
        ngram_extra_orders=extra_orders,
        ngram_clean_table=args.ngram_clean_table if extra_orders else 0,
        ngram_dict_topk=args.ngram_dict_topk if extra_orders else 0,
        ngram_keep_rows=args.ngram_keep_rows if args.ngram_dict_topk > 0 else "dict",
        intervention=args.intervention,
        intervention_epoch=args.intervention_epoch,
        intervention_epochs=tuple(
            int(x) for x in args.intervention_epochs.split(",") if x.strip()
        ),
        intervention_freq_threshold=args.intervention_freq_threshold,
        intervention_low_inclusive=bool(args.intervention_low_inclusive),
        intervention_mask_frac=args.intervention_mask_frac,
        context_eval_ab=bool(args.context_eval_ab),
        context_set_mode=args.context_set_mode,
        data_dir=args.data_dir,
        train_shards=[int(x) for x in args.train_shards.split(",") if x.strip()],
        val_shards=[int(x) for x in args.val_shards.split(",") if x.strip()],
        epoch_batches=args.epoch_batches,
        train_max_tokens=args.train_max_tokens,
        replay_mix_passes=args.replay_mix_passes,
        row_index=bool(args.row_index),
        row_index_dir=args.row_index_dir,
        token_dtype=args.token_dtype,
        ce_chunk_positions=args.ce_chunk_positions,
        static_low_mask=args.static_low_mask,
        static_low_mask_keep_novel=bool(args.static_low_mask_keep_novel),
        fp8_torchao=bool(args.fp8_torchao),
        out_dir=os.path.join(args.out_dir, args.run_id),
    )

    if args.table_betas:
        parts = [float(x) for x in args.table_betas.replace(";", ",").split(",") if x.strip()]
        assert len(parts) == 2 and all(0.0 <= b < 1.0 for b in parts), \
            "table_betas must be two values in [0, 1)"
        cfg.table_betas = tuple(parts)
    else:
        cfg.table_betas = (0.0, 0.99)  # β₂=0.99 is the scaling default (plan §1)
    # X2: narrow table rows are a clean-table-only variable
    if cfg.bigram_table_dim and not cfg.bigram_clean_table:
        raise SystemExit("--bigram_table_dim requires --bigram_clean_table > 0")
    if cfg.trigram_table_dim and not cfg.trigram_clean_table:
        raise SystemExit("--trigram_table_dim requires --trigram_clean_table > 0")
    if min(cfg.bigram_table_dim, cfg.trigram_table_dim) < 0:
        raise SystemExit("table_dim must be >= 0")
    if max(cfg.bigram_table_dim, cfg.trigram_table_dim) > cfg.n_embd:
        raise SystemExit("table_dim must be <= n_embd (rows are zero-padded up to n_embd)")

    set_seed(cfg.seed)
    if args.deterministic:
        # Must be set before the first cuBLAS handle is created (first GEMM).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        # flash / mem-efficient SDPA backward are non-deterministic (atomic
        # reductions); math backend is the deterministic path.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        print("[nglab] deterministic mode: use_deterministic_algorithms=True, "
              "SDPA=math-only, CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nglab] device={device} injection={cfg.nanogpt_ngram_injection_position} steps={cfg.max_steps}")

    # data
    train_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, cfg.sequence_len,
                                     cfg.device_batch_size, cfg.data_seed,
                                     epoch_batches=cfg.epoch_batches,
                                     replay_mix_passes=cfg.replay_mix_passes,
                                     token_dtype=cfg.token_dtype,
                                     row_index_dir=cfg.row_index_dir,
                                     static_low_mask=cfg.static_low_mask,
                                     rows_requested=cfg.row_index,
                                     max_tokens=cfg.train_max_tokens)
    if args.train_epochs > 0:
        if args.steps != 1000:
            raise SystemExit("--train_epochs and --steps are mutually exclusive "
                             "(leave --steps at its default)")
        if args.train_max_tokens <= 0 and cfg.epoch_batches <= 0 \
                and cfg.replay_mix_passes <= 0:
            raise SystemExit("--train_epochs requires an explicit pool size: pass "
                             "--train_max_tokens (or --epoch_batches)")
        _ep_steps = steps_for_epochs(args.train_epochs, train_ds._batches_per_epoch,
                                     cfg.device_batch_size, cfg.sequence_len,
                                     cfg.total_batch_size)
        cfg.max_steps = _ep_steps
        print(f"[nglab] epoch-anchored protocol: {args.train_epochs} epochs x "
              f"{train_ds._batches_per_epoch} batches/epoch x "
              f"{cfg.device_batch_size * cfg.sequence_len} tok = "
              f"{train_ds._batches_per_epoch * cfg.device_batch_size * cfg.sequence_len:,} "
              f"tok/epoch -> steps={_ep_steps}")
    val_ds = TokenizedShardDataset(cfg.data_dir, cfg.val_shards, cfg.sequence_len,
                                   cfg.device_batch_size, cfg.data_seed,
                                   token_dtype=cfg.token_dtype)
    train_iter = train_ds.iter_batches(device, with_rows=cfg.row_index)
    val_iter = val_ds.iter_batches(device)
    # Fixed validation batches: captured once and reused for every val eval, so
    # the val-loss curve always measures the same val data.  The val-side
    # freq-bin eval uses its own moving window on val_iter; train is the main
    # queue consumed by training + train-side freq eval.
    fixed_val_batches = [
        next(val_iter) for _ in range(max(cfg.val_batches, args.freq_eval_batches))
    ]
    validation_batches = fixed_val_batches[:cfg.val_batches]
    # Fixed freq-bin val batches: use the same prefix as the scalar validation
    # set whenever the requested batch counts differ.
    fixed_freq_val_batches = fixed_val_batches[:args.freq_eval_batches]
    print(f"[nglab] fixed val batches: {len(fixed_val_batches)} x shape {tuple(fixed_val_batches[0][0].shape)} "
          f"+ {len(fixed_freq_val_batches)} freq-val batches")
    # Fixed train probe: captured from a SEPARATE dataset instance that owns its
    # own iterator, so grabbing the probe never consumes the training stream and
    # never advances the training epoch counter.  The probe is a fixed set of
    # train batches reused for every probe eval; its gap is fixed_val − fixed_train.
    # --train_probe_mode first: head of the train stream (backward compatible).
    # --train_probe_mode uniform: evenly-spaced chunks across the ENTIRE train
    # stream, so the probe parallels the fixed val probe instead of being biased
    # toward the head-of-epoch batches that get replayed/memorized first.
    fixed_train_probe = []
    probe_hash = None
    if args.fixed_train_probe > 0:
        probe_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, cfg.sequence_len,
                                         cfg.device_batch_size, cfg.data_seed,
                                         epoch_batches=cfg.epoch_batches,
                                         max_tokens=cfg.train_max_tokens)
        n_chunks = probe_ds.total_chunks()
        n_need = args.fixed_train_probe * cfg.device_batch_size
        if args.train_probe_mode == "uniform" and n_need < n_chunks:
            # evenly-spaced chunk indices covering [0, n_chunks): use the probe
            # seed so the sample is deterministic but independent of training.
            idx = [int(round(i * (n_chunks - 1) / (n_need - 1)))
                   for i in range(n_need)] if n_need > 1 else [0]
            chunks = [probe_ds.get_chunk(g) for g in idx]
        else:
            # head-of-stream (default) or degenerate small train set: just walk
            # the iterator in order (identical to the historical behavior).
            probe_iter = probe_ds.iter_batches(device)
            fixed_train_probe = [next(probe_iter) for _ in range(args.fixed_train_probe)]
            chunks = None
        if chunks is not None:
            # re-batch the sampled chunks into device batches
            fixed_train_probe = []
            for i in range(0, len(chunks), cfg.device_batch_size):
                inp = torch.stack([c[0] for c in chunks[i:i + cfg.device_batch_size]]).to(device)
                tgt = torch.stack([c[1] for c in chunks[i:i + cfg.device_batch_size]]).to(device)
                fixed_train_probe.append((inp, tgt))
        probe_hash = hashlib.sha256(
            b"".join(batch[0].cpu().numpy().tobytes() for batch in fixed_train_probe)).hexdigest()[:16]
        print(f"[nglab] fixed train probe: {len(fixed_train_probe)} batches "
              f"mode={args.train_probe_mode} sha256={probe_hash}")
    grad_accum = max(1, cfg.total_batch_size // (cfg.device_batch_size * cfg.sequence_len))
    print(f"[nglab] grad_accum={grad_accum} train_chunks={train_ds.total_chunks()} "
          f"val_chunks={val_ds.total_chunks()}")

    # model
    model = NanoGPT(cfg).to(device)
    model.init_weights()
    if args.init_model:
        init_path = os.path.abspath(args.init_model)
        if not os.path.isfile(init_path):
            raise SystemExit(f"--init_model not found: {init_path}")
        payload = torch.load(init_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise SystemExit(f"--init_model must contain a state_dict mapping: {init_path}")
        missing, unexpected = model.load_state_dict(payload, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"--init_model incompatible: missing={missing[:8]} unexpected={unexpected[:8]}")
        print(f"[nglab] initialized model from {init_path}")
    if args.table_init_count:
        if not os.path.isfile(args.table_init_count):
            raise SystemExit(f"--table_init_count not found: {args.table_init_count}")
        if args.intervention == "hash_reseed":
            raise SystemExit("--table_init_count is incompatible with hash_reseed "
                             "(count content is tied to the base hash families)")
        count_init_clean_tables(model, os.path.abspath(args.table_init_count))
    if args.disable_ngram_tables:
        disabled = 0
        for name, p in model.named_parameters():
            if any(marker in name for marker in ("value_embeds", "bigram_ves", "trigram_ves",
                                                 "ngram_ves",
                                                 "ve_gate", "bigram_gate", "trigram_gate")):
                with torch.no_grad():
                    p.zero_()
                p.requires_grad_(False)
                disabled += p.numel()
        print(f"[nglab] disable_ngram_tables enabled: zeroed and froze {disabled:,} parameters")
    if args.freeze_table:
        frozen = 0
        for name, p in model.named_parameters():
            if any(marker in name for marker in ("value_embeds", "bigram_ves", "trigram_ves",
                                                 "ngram_ves",
                                                 "ve_gate", "bigram_gate", "trigram_gate")):
                p.requires_grad_(False)
                frozen += p.numel()
        print(f"[nglab] freeze_table enabled: {frozen:,} parameters frozen")
    if args.freeze_embeddings_only:
        frozen = 0
        for name, p in model.named_parameters():
            if any(marker in name for marker in ("value_embeds", "bigram_ves", "trigram_ves",
                                                 "ngram_ves")):
                p.requires_grad_(False)
                frozen += p.numel()
        print(f"[nglab] freeze_embeddings_only enabled: {frozen:,} parameters frozen; gates trainable")
    lora_targets = []
    if args.lora_rank:
        from lora_adapter import install_lora
        for p in model._backbone_params():
            p.requires_grad_(False)
        lora_targets = install_lora(model, args.lora_rank, args.lora_alpha)
        print(f"[nglab] LoRA rank={args.lora_rank} alpha={args.lora_alpha} targets={len(lora_targets)}")

    _rows_meta = None
    fp8_modules = 0
    if args.row_index:
        # --- rows_meta.json contract check ---
        meta_path = os.path.join(cfg.row_index_dir or cfg.data_dir, "rows_meta.json")
        if not os.path.isfile(meta_path):
            raise SystemExit(f"--row_index requires {meta_path} "
                             "(run code/precompute_row_index.py first)")
        with open(meta_path) as f:
            _rows_meta = json.load(f)
        for key, want in (("vocab_size", cfg.vocab_size),
                          ("sequence_len", cfg.sequence_len),
                          ("token_dtype", cfg.token_dtype)):
            if _rows_meta.get(key) != want:
                raise SystemExit(f"rows_meta.json {key}={_rows_meta.get(key)!r} "
                                 f"!= run setting {want!r}")
        for key, want in (("bigram_clean_table", cfg.bigram_clean_table),
                          ("trigram_clean_table", cfg.trigram_clean_table)):
            if want and _rows_meta.get(key) != want:
                raise SystemExit(f"rows_meta.json {key}={_rows_meta.get(key)!r} "
                                 f"!= run setting {want!r}")
        mask_meta = _rows_meta.get("static_low_mask", {})
        if cfg.static_low_mask > 0:
            if not mask_meta.get("enabled"):
                raise SystemExit("--static_low_mask set but rows_meta.json has no "
                                 "static mask streams (re-run precompute_row_index.py "
                                 "with --freq_index --min_count)")
            if mask_meta.get("min_count") != cfg.static_low_mask:
                raise SystemExit(f"rows_meta.json static mask min_count="
                                 f"{mask_meta.get('min_count')} != "
                                 f"--static_low_mask {cfg.static_low_mask}")
            if bool(mask_meta.get("keep_novel")) != cfg.static_low_mask_keep_novel:
                raise SystemExit("rows_meta.json keep_novel != "
                                 "--static_low_mask_keep_novel")

        # --- first-batch bitwise verification (separate dataset instance so
        # the training stream is untouched) ---
        model._check_pre_rows_supported()
        ver_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, cfg.sequence_len,
                                       cfg.device_batch_size, cfg.data_seed,
                                       epoch_batches=cfg.epoch_batches,
                                       token_dtype=cfg.token_dtype,
                                       row_index_dir=cfg.row_index_dir,
                                       static_low_mask=cfg.static_low_mask,
                                       rows_requested=True)
        ver_batch = next(ver_ds.iter_batches(device, with_rows=True))
        v_inp, _, v_bi, v_ti, v_bv, v_tv = ver_batch
        with torch.no_grad():
            v_prev = torch.cat([v_inp[:, :1], v_inp[:, :-1]], dim=1)
            v_prev2 = torch.cat([v_inp[:, :2], v_inp[:, :-2]], dim=1)
            if cfg.enable_bigram_ve:
                if v_bi is None:
                    raise SystemExit("--row_index: bigram branch enabled but no "
                                     "brows.u32.bin stream found for the train shards")
                dyn = model._bigram_row_indices(
                    sorted(model.bigram_ve_layers)[0], v_prev, v_inp)[0]
                if not torch.equal(dyn, v_bi):
                    dflat, oflat = dyn.flatten(), v_bi.flatten()
                    i = int((dflat != oflat).nonzero()[0])
                    raise SystemExit(f"--row_index VERIFICATION FAILED (bigram): flat "
                                     f"position {i}: dynamic={int(dflat[i])} "
                                     f"offline={int(oflat[i])} — rows_meta/primes/"
                                     f"R mismatch or stale index; refusing to start")
                print(f"[nglab] row_index verify: bigram rows bitwise OK ({dyn.numel():,} pos)")
            if cfg.enable_trigram_ve:
                if v_ti is None:
                    raise SystemExit("--row_index: trigram branch enabled but no "
                                     "trows.u32.bin stream found for the train shards")
                dyn = model._trigram_row_indices(
                    sorted(model.trigram_ve_layers)[0], v_prev2, v_prev, v_inp)[0]
                if not torch.equal(dyn, v_ti):
                    dflat, oflat = dyn.flatten(), v_ti.flatten()
                    i = int((dflat != oflat).nonzero()[0])
                    raise SystemExit(f"--row_index VERIFICATION FAILED (trigram): flat "
                                     f"position {i}: dynamic={int(dflat[i])} "
                                     f"offline={int(oflat[i])} — refusing to start")
                print(f"[nglab] row_index verify: trigram rows bitwise OK ({dyn.numel():,} pos)")
            if cfg.static_low_mask > 0:
                if v_bv is None or v_tv is None:
                    raise SystemExit("--static_low_mask: validity streams missing")
                if args.freq_index and os.path.isfile(args.freq_index):
                    from ngram_freq import GlobalFrequencyIndex as _GFI
                    _fx = _GFI.load(args.freq_index)
                    from ngram_freq import compute_context_keys as _cck
                    bk, tk = _cck(v_inp, cfg.vocab_size)
                    f_b = _fx.hit_count_numpy("bigram", bk.cpu().numpy().ravel())
                    f_t = _fx.hit_count_numpy("trigram", tk.cpu().numpy().ravel())
                    exp_b = (f_b >= cfg.static_low_mask) | (
                        cfg.static_low_mask_keep_novel & (f_b == 0))
                    exp_t = (f_t >= cfg.static_low_mask) | (
                        cfg.static_low_mask_keep_novel & (f_t == 0))
                    ok_b = torch.equal(v_bv.cpu().flatten(),
                                       torch.from_numpy(exp_b))
                    ok_t = torch.equal(v_tv.cpu().flatten(),
                                       torch.from_numpy(exp_t))
                    if not (ok_b and ok_t):
                        raise SystemExit("--static_low_mask VERIFICATION FAILED: "
                                         "validity bits do not match the freq index "
                                         "(threshold/keep_novel/index mismatch)")
                    print(f"[nglab] static_low_mask verify: bits bitwise OK "
                          f"(enabled {int(exp_b.mean()*100)}% bigram / "
                          f"{int(exp_t.mean()*100)}% trigram, t={cfg.static_low_mask}, "
                          f"keep_novel={cfg.static_low_mask_keep_novel})")
                else:
                    print(f"[nglab] static_low_mask: skipping bit verification "
                          f"(no --freq_index provided); enabled frac "
                          f"b={float(v_bv.float().mean()):.4f} "
                          f"t={float(v_tv.float().mean()):.4f}")

    if args.fp8_torchao:
        try:
            from torchao.float8 import convert_to_float8_training
        except ImportError as e:
            raise SystemExit(f"--fp8_torchao requires torchao (pip install torchao): {e!r}")
        import inspect as _inspect
        _ngram_markers = ("value_embeds", "bigram_ves", "trigram_ves", "ngram_ves",
                          "ve_gate", "bigram_gate", "trigram_gate")

        def _fp8_filter(module, name):
            if not isinstance(module, nn.Linear):
                return False
            if name.endswith("lm_head"):
                return False
            if any(m in name for m in _ngram_markers):
                return False
            return True

        _sig = _inspect.signature(convert_to_float8_training)
        if "module_filter_fn" in _sig.parameters:
            convert_to_float8_training(model, module_filter_fn=_fp8_filter)
        else:
            convert_to_float8_training(model, filter_fn=_fp8_filter)
        fp8_modules = sum(1 for m in model.modules() if type(m).__name__ == "Float8Linear")
        print(f"[nglab] fp8_torchao: {fp8_modules} backbone Linear modules in float8 "
              f"(lm_head + n-gram tables/gates stay bf16/fp32)")
    if args.compile:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[nglab] model params: {n_params/1e6:.1f}M"
          + (f" | compile={args.compile}" if args.compile else ""))
    optimizer = MixedOptimizer(model, lr=cfg.nanogpt_adam_lr,
                               ngram_betas=cfg.ngram_table_betas,
                               adam_betas=cfg.adam_betas,
                               weight_decay=cfg.weight_decay,
                               table_optimizer=cfg.table_optimizer,
                               table_lr_scale=cfg.table_lr_scale,
                               table_betas=cfg.table_betas)
    if args.table_diag:
        optimizer.enable_table_diag()

    # logs
    train_log = open(os.path.join(cfg.out_dir, "train_log.jsonl"), "w")
    table_log = open(os.path.join(cfg.out_dir, "table_norm.jsonl"), "w")
    table_diag_log = open(os.path.join(cfg.out_dir, "table_diag.jsonl"), "w") \
        if args.table_diag else None
    probe_log = None
    if args.fixed_train_probe > 0:
        probe_log = open(os.path.join(cfg.out_dir, "fixed_train_loss.jsonl"), "w")
    fixed_train_freq_log = None
    freq_bin_log = None
    exact_freq_log = None
    freq_index_obj = None
    diag_amp_dtype = None
    diag_pool = None
    multi_order_index = False
    exact_freq_train_batches = None
    # per-token dump cadence (post-hoc gap-freq analysis): listed steps plus the
    # final step (always).  Empty flag = final step only.
    dump_steps = {int(s) for s in args.dump_token_stats_steps.split(",") if s.strip()}
    dump_orders = sorted(ngram_orders) if ngram_orders else \
        ([2] if cfg.enable_bigram_ve else []) + ([3] if cfg.enable_trigram_ve else [])
    if args.freq_index and os.path.exists(args.freq_index):
        from ngram_freq import load_frequency_index
        freq_index_obj = load_frequency_index(args.freq_index)
        multi_order_index = bool(getattr(freq_index_obj, "is_multi_order", False))
        if multi_order_index:
            if args.logit_stats:
                raise SystemExit("--logit_stats is not supported with a multi-order freq index")
            if fixed_train_probe:
                raise SystemExit("fixed-train freq bins are not supported with a multi-order "
                                 "freq index (use --fixed_train_probe 0)")
            if freq_index_obj.vocab_size != cfg.vocab_size:
                raise SystemExit("multi-order freq index vocab_size != --vocab_size")
            print(f"[nglab] multi-order freq index: branches={freq_index_obj.branches}")
            if cfg.ngram_dict_topk > 0:
                if not multi_order_index:
                    raise SystemExit("--ngram_dict_topk needs a multi-order freq index "
                                     "(rebuild with ngram_freq.py --orders)")
                model.load_ngram_dictionaries(freq_index_obj, cfg.ngram_dict_topk)
                dict_npz = os.path.join(cfg.out_dir, f"ngram_dicts_top{cfg.ngram_dict_topk}.npz")
                np.savez_compressed(dict_npz, **{
                    f"keys_{n}": model.ngram_dict_keys[n].cpu().numpy()
                    for n in cfg.ngram_extra_orders})
                print(f"[nglab] dictionaries saved: {dict_npz}")
        freq_bin_log = open(os.path.join(cfg.out_dir, "freq_bin_loss.jsonl"), "w")
        if fixed_train_probe:
            fixed_train_freq_log = open(
                os.path.join(cfg.out_dir, "fixed_train_freq_bin_loss.jsonl"), "w"
            )
        exact_freq_log = open(
            os.path.join(cfg.out_dir, "exact_freq_loss.jsonl"), "w"
        )
        print(f"[nglab] freq-bin eval enabled (index: {args.freq_index})")
        # Independent train-side diagnostic iterator: reads the same shards in
        # the same fixed order but owns its own dataset state, so freq-bin
        # diagnostics never consume training batches and never advance the
        # training epoch counter.
        freq_train_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards,
                                              cfg.sequence_len,
                                              cfg.device_batch_size,
                                              cfg.data_seed,
                                              max_tokens=cfg.train_max_tokens)
        freq_train_iter = freq_train_ds.iter_batches(device)
        # exact-frequency diagnostics need a fixed train-side reference even when
        # --fixed_train_probe is 0: grab a small fixed set from the independent
        # diagnostic iterator (never touches the training stream or epoch counter).
        if not fixed_train_probe:
            exact_freq_train_batches = [next(freq_train_iter) for _ in range(4)]
        else:
            exact_freq_train_batches = fixed_train_probe
        # Async diagnostics: the main process does ONE bf16-autocast forward per
        # fixed batch per eval block; hit-count lookup + exact-f/bin aggregation
        # run in a background CPU worker (diag_worker.py) and never block the
        # training loop. NGLAB_ASYNC_DIAG=0 falls back to the sync path.
        diag_amp_dtype = {"bf16": torch.bfloat16, "fp8": torch.bfloat16}.get(args.dtype)
        if os.environ.get("NGLAB_ASYNC_DIAG", "1") != "0":
            try:
                import diag_worker as dw
                dw.GLOBAL_INDEX = freq_index_obj
                diag_pool = dw.DiagPool(args.freq_index, cfg.vocab_size,
                                        inherited_index=freq_index_obj,
                                        n_workers=args.diag_workers)
                print(f"[nglab] async diag worker(s) started: {max(1, args.diag_workers)} "
                      f"(diagnostic forwards: {args.dtype})")
            except Exception as e:
                print(f"[nglab] async diag unavailable ({e!r}); using sync path")
                diag_pool = None

    # Per-token dump train batches: 4 fixed batches from an independent
    # pool-truncated iterator (same convention as the exact-freq train rows;
    # reuses those batches when the freq pipeline already built them).
    dump_train_batches = exact_freq_train_batches
    if dump_train_batches is None:
        _dump_train_ds = TokenizedShardDataset(
            cfg.data_dir, cfg.train_shards, cfg.sequence_len,
            cfg.device_batch_size, cfg.data_seed, max_tokens=cfg.train_max_tokens)
        _dump_train_iter = _dump_train_ds.iter_batches(device)
        dump_train_batches = [next(_dump_train_iter) for _ in range(4)]
    dump_amp_dtype_spec = args.dtype if args.dtype in ("bf16", "fp8") else "fp32"

    context_gap_log = None
    if cfg.intervention == "random_context_loss_mask" or cfg.context_eval_ab:
        if freq_index_obj is None and cfg.context_set_mode == "type":
            raise SystemExit("context A/B eval (type mode) requires an existing --freq_index")
        if not fixed_train_probe:
            raise SystemExit("context A/B eval requires --fixed_train_probe > 0")
        info = model.setup_context_eval_sets(freq_index_obj, cfg.intervention_mask_frac, cfg.seed)
        context_gap_log = open(os.path.join(cfg.out_dir, "context_gap.jsonl"), "w")
        print(f"[nglab] context A/B sets: {info['n_set_types']}/{info['n_types']} trigram "
              f"types per set (frac={info['frac']}, seed={info['seed']})")

    def on_diag_rows(rstep, repoch, sha, exact, fb):
        """Write one finished diagnostics block (async worker or sync path)."""
        if exact_freq_log is not None:
            exact_freq_log.write(json.dumps({
                "step": rstep, "epoch": repoch,
                "train": exact["train"], "val": exact["val"], "shared": exact["shared"],
                "probe_batch_sha256": sha,
            }) + "\n")
        if freq_bin_log is not None and fb.get("train"):
            freq_bin_log.write(json.dumps({
                "step": rstep, "epoch": repoch,
                "train": fb["train"], "val": fb["val"],
            }) + "\n")

    # --- logit-stats probe (§5.2 supplement) ---
    logit_stats_log = None
    logit_exemplar_ids = []  # list of (exemplar_id, branch, hit_count, quadrant)
    if args.logit_stats and freq_index_obj is not None:
        from ngram_freq import compute_context_keys as _ls_compute_context_keys
        logit_stats_log = open(os.path.join(cfg.out_dir, "logit_stats.jsonl"), "w")
        # Deterministic exemplar selection: pick ~N/4 contexts per quadrant
        # (highF-seen, highF-novel, lowF-seen, lowF-novel) from val batches.
        # "highF" = hit >= median of sampled hits; "seen" = hit > 0.
        import diag_worker as _dw
        n_ex = max(4, args.logit_stats_exemplars)
        per_quad = n_ex // 4
        # Sample context keys from first 2 val batches (deterministic)
        sample_batches = validation_batches[:2]
        all_keys_b = []; all_keys_t = []
        with torch.no_grad():
            for inp, _tgt in sample_batches:
                bk, tk = _ls_compute_context_keys(inp, cfg.vocab_size)
                all_keys_b.append(bk.cpu().numpy().ravel())
                all_keys_t.append(tk.cpu().numpy().ravel())
        all_keys_b = np.concatenate(all_keys_b)
        all_keys_t = np.concatenate(all_keys_t)
        hits_b = freq_index_obj.hit_count_numpy("bigram", all_keys_b)
        hits_t = freq_index_obj.hit_count_numpy("trigram", all_keys_t)
        # Use bigram branch for exemplar selection (covers both branches via shared positions)
        uniq_keys, uniq_idx = np.unique(all_keys_b, return_index=True)
        uniq_hits = hits_b[uniq_idx]
        med_hit = int(np.median(uniq_hits[uniq_hits > 0])) if np.any(uniq_hits > 0) else 1
        quadrants = {
            "highF_seen": (uniq_hits >= med_hit) & (uniq_hits > 0),
            "highF_novel": (uniq_hits >= med_hit) & (uniq_hits == 0),  # impossible by def; fallback below
            "lowF_seen": (uniq_hits < med_hit) & (uniq_hits > 0),
            "lowF_novel": uniq_hits == 0,
        }
        # highF_novel is empty by definition; redistribute to lowF_novel
        selected = []
        rng = random.Random(cfg.seed + 7)
        for qname, mask in quadrants.items():
            if qname == "highF_novel":
                continue
            candidates = uniq_keys[mask]
            if len(candidates) == 0:
                continue
            take = min(per_quad, len(candidates))
            chosen = rng.sample(list(candidates), take)
            for k in chosen:
                h = int(freq_index_obj.bigram.get(int(k), 0))
                selected.append((int(k), "bigram", h, qname))
        # If we still need more, fill from trigram lowF_novel
        if len(selected) < n_ex:
            t_uniq, t_uidx = np.unique(all_keys_t, return_index=True)
            t_uhits = hits_t[t_uidx]
            novel_mask = t_uhits == 0
            cand_t = t_uniq[novel_mask]
            if len(cand_t) > 0:
                extra = min(n_ex - len(selected), len(cand_t))
                chosen_t = rng.sample(list(cand_t), extra)
                for k in chosen_t:
                    h = int(freq_index_obj.trigram.get(int(k), 0))
                    selected.append((int(k), "trigram", h, "lowF_novel"))
        logit_exemplar_ids = selected[:n_ex]
        print(f"[nglab] logit_stats probe enabled: {len(logit_exemplar_ids)} exemplars "
              f"(quadrants: "
              + ", ".join(f"{q}={sum(1 for s in logit_exemplar_ids if s[3]==q)}"
                          for q in ("highF_seen","lowF_seen","lowF_novel"))
              + ")")

    def on_logit_rows(lstep, lepoch, lbranch, lsummary, lexemplar_dump):
        """Write one logit-stats aggregation result."""
        if logit_stats_log is None:
            return
        # Aggregated row (small)
        logit_stats_log.write(json.dumps({
            "type": "agg",
            "step": lstep, "epoch": lepoch, "branch": lbranch,
            "buckets": lsummary,
        }) + "\n")
        # Exemplar rows (one per exemplar)
        for ex in lexemplar_dump:
            logit_stats_log.write(json.dumps({
                "type": "exemplar",
                "step": lstep, "epoch": lepoch, "branch": lbranch,
                **ex,
            }) + "\n")
        logit_stats_log.flush()

    last_val_loss = float("nan")
    last_train_loss = float("nan")
    last_fixed_train_loss = float("nan")
    last_fixed_val_loss = float("nan")

    model.train()
    t0 = time.time()
    intervention_due = (
        frozenset({cfg.intervention_epoch} | set(cfg.intervention_epochs))
        if cfg.intervention != "none" else frozenset()
    )
    intervention_fired_epochs = set()
    intervention_events = []
    amp_ctx = _autocast_ctx(args.dtype)
    cur_inp = cur_tgt = cur_ptl = None
    ctx_tr_accum = None
    if context_gap_log is not None:
        ctx_tr_accum = {k: torch.zeros((), device=device) for k in
                        ("A_sum", "A_cnt", "B_sum", "B_cnt")}
    for step in range(cfg.max_steps):
        # gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        for micro in range(grad_accum):
            try:
                if cfg.row_index:
                    inp, tgt, b_rows, t_rows, b_valid, t_valid = next(train_iter)
                    pre_rows = (b_rows, t_rows, b_valid, t_valid)
                else:
                    inp, tgt = next(train_iter)
                    pre_rows = None
            except StopIteration:
                train_iter = train_ds.iter_batches(device, with_rows=cfg.row_index)
                if cfg.row_index:
                    inp, tgt, b_rows, t_rows, b_valid, t_valid = next(train_iter)
                    pre_rows = (b_rows, t_rows, b_valid, t_valid)
                else:
                    inp, tgt = next(train_iter)
                    pre_rows = None
            if (train_ds._epoch in intervention_due
                    and train_ds._epoch not in intervention_fired_epochs):
                event = model.apply_intervention(train_ds._epoch, freq_index_obj)
                event["step"] = step + 1
                intervention_events.append(event)
                intervention_fired_epochs.add(train_ds._epoch)
                print(f"[nglab] intervention={cfg.intervention} at step={step + 1} "
                      f"epoch={train_ds._epoch + 1}")
            with amp_ctx:
                if model._context_set_memb is not None or model._context_loss_mask_active:
                    loss, ptl = model(inp, targets=tgt, return_token_losses=True,
                                      pre_rows=pre_rows)
                    if model._context_loss_mask_active:
                        keep = model._context_loss_keep_mask(inp) & tgt.ne(-1)
                        loss = ptl[keep].mean()
                    memb = model.context_set_membership(inp)
                    valid = tgt.ne(-1)
                    for _code, _k in ((1, "A"), (2, "B")):
                        _m = (memb == _code) & valid
                        ctx_tr_accum[_k + "_sum"] += ptl[_m].sum()
                        ctx_tr_accum[_k + "_cnt"] += _m.sum()
                    if freq_bin_log is not None and micro == grad_accum - 1:
                        cur_inp, cur_tgt, cur_ptl = inp, tgt, ptl.detach()
                elif freq_bin_log is not None and micro == grad_accum - 1:
                    loss, ptl = model(inp, targets=tgt, return_token_losses=True,
                                      pre_rows=pre_rows)
                    cur_inp, cur_tgt, cur_ptl = inp, tgt, ptl.detach()
                else:
                    loss = model(inp, targets=tgt, pre_rows=pre_rows)
                loss = loss / grad_accum
            # freeze_both freezes every parameter: the graph then has no
            # grad_fn, so backward is skipped entirely (optimizer no-ops).
            if loss.requires_grad:
                loss.backward()
            accum_loss += loss.item()
        train_loss = accum_loss
        if cfg.lr_schedule_epochs > 0:
            # epoch-anchored LR: all runs share the same per-epoch trajectory
            progress = min(1.0, (train_ds._epoch + train_ds.epoch_progress()) / cfg.lr_schedule_epochs)
        else:
            progress = (step + 1) / cfg.max_steps
        lr_mult = get_lr_multiplier(
            progress,
            warmdown_ratio=cfg.warmdown_ratio,
            schedule=cfg.lr_schedule,
            warmup_steps=cfg.warmup_steps,
            step=step + 1,
            max_steps=cfg.max_steps,
            warmup_start_lr_mult=cfg.warmup_start_lr_mult,
            cosine_min_lr_mult=cfg.cosine_min_lr_mult,
        )
        optimizer.step(lr_mult=lr_mult)

        # per-step table-update diagnostics (replay-shock fingerprint)
        if table_diag_log is not None:
            diag = optimizer.pop_table_diag()
            if diag is not None:
                diag["step"] = step + 1
                diag["epoch"] = train_ds._epoch + 1
                table_diag_log.write(json.dumps(diag) + "\n")
                if (step + 1) % cfg.val_interval_steps == 0:
                    table_diag_log.flush()

        # periodic val
        val_due = (
            ((cfg.val_steps and (step + 1) in cfg.val_steps)
             or (not cfg.val_steps
                 and ((step + 1) % cfg.val_interval_steps == 0
                      or step == cfg.max_steps - 1)))
        )
        if val_due:
            last_val_loss = evaluate_val(model, validation_batches, dtype=args.dtype)
            last_train_loss = train_loss
            entry = {
                "step": step + 1,
                "train_loss": train_loss,
                "val_loss": last_val_loss,
                "gap": last_val_loss - train_loss,
                "lr_mult": lr_mult,
                "epoch": train_ds._epoch + 1,
                "elapsed_s": time.time() - t0,
            }
            train_log.write(json.dumps(entry) + "\n")
            train_log.flush()
            print(f"[nglab] step {step+1:4d} | train {train_loss:.4f} | val {last_val_loss:.4f} "
                  f"| gap {last_val_loss-train_loss:+.4f} | epoch {train_ds._epoch+1} | "
                  f"lr_m {lr_mult:.2f} | {(time.time()-t0):.0f}s")

        epoch_boundary = (train_ds._batch_in_epoch == 0 and step > 0)

        # periodic fixed-train probe eval (interval + epoch boundary)
        # when --val_steps is set, probe follows the exact val steps (it needs
        # a val eval to form the fixed gap)
        if cfg.val_steps:
            probe_due = (step + 1) in cfg.val_steps
        else:
            probe_due = (
                (step + 1) % args.probe_eval_interval == 0
                or epoch_boundary
                or step == cfg.max_steps - 1
            )
        if probe_log is not None and fixed_train_probe and probe_due:
            fixed_train_loss = evaluate_fixed_probe(model, fixed_train_probe, dtype=args.dtype)
            fixed_val_loss = evaluate_val(model, validation_batches, dtype=args.dtype)
            last_fixed_train_loss = fixed_train_loss
            last_fixed_val_loss = fixed_val_loss
            probe_entry = {
                "step": step + 1,
                "epoch": train_ds._epoch + 1,
                "fixed_train_loss": fixed_train_loss,
                "fixed_val_loss": fixed_val_loss,
                "fixed_gap": fixed_val_loss - fixed_train_loss,
                "lr_mult": lr_mult,
            }
            probe_log.write(json.dumps(probe_entry) + "\n")
            probe_log.flush()
            print(f"[nglab] probe step {step+1:4d} | ftrain {fixed_train_loss:.4f} "
                  f"| fval {fixed_val_loss:.4f} | fgap {fixed_val_loss-fixed_train_loss:+.4f} "
                  f"| epoch {train_ds._epoch+1}")
        if context_gap_log is not None and fixed_train_probe and probe_due:
            cg = evaluate_context_gap(model, fixed_train_probe, validation_batches,
                                      dtype=args.dtype)
            cg["step"] = step + 1
            cg["epoch"] = train_ds._epoch + 1
            cg["epoch_pos"] = train_ds._epoch + train_ds.epoch_progress()
            # online train loss on A/B positions (window mean since last eval),
            # and the paper-style gap contribution share * (val - train)
            acc = {k: float(v.item()) for k, v in ctx_tr_accum.items()}
            for k in ctx_tr_accum:
                ctx_tr_accum[k].zero_()
            for tag in ("A", "B"):
                cnt = acc[tag + "_cnt"]
                tr_online = acc[tag + "_sum"] / cnt if cnt > 0 else float("nan")
                cg[f"tr_online_{tag}"] = tr_online
                if cnt > 0 and cg.get(f"val_mean_{tag}") == cg.get(f"val_mean_{tag}"):
                    cg[f"Cpos_{tag}"] = cg[f"val_share_{tag}"] * (
                        cg[f"val_mean_{tag}"] - tr_online)
                cg[f"tr_online_n_{tag}"] = int(cnt)
            context_gap_log.write(json.dumps(cg) + "\n")
            context_gap_log.flush()
        fixed_freq_due = (
            ((cfg.val_steps and (step + 1) in cfg.val_steps)
             or (not cfg.val_steps
                 and ((step + 1) % args.freq_eval_interval == 0
                      or step == cfg.max_steps - 1)))
        )
        if fixed_train_freq_log is not None and fixed_freq_due:
            fixed_train_freq = evaluate_freq_bins(
                model,
                fixed_train_probe,
                freq_index_obj,
                len(fixed_train_probe),
                cfg.vocab_size,
            )
            fixed_train_freq_log.write(json.dumps({
                "step": step + 1,
                "epoch": train_ds._epoch + 1,
                "fixed_train": fixed_train_freq,
                "probe_batch_count": len(fixed_train_probe),
                "probe_batch_sha256": probe_hash,
            }) + "\n")
            fixed_train_freq_log.flush()
        # Unified periodic diagnostics (exact-freq + freq-bin). ONE bf16
        # forward per fixed batch on the main process; hit-count lookup and
        # per-exact-f / per-bucket aggregation run in the background CPU
        # worker, overlapping training. Cadence follows val_interval_steps
        # (exact_freq_eval_interval / freq_eval_interval default to the same 10).
        diag_due = (
            (cfg.val_steps and (step + 1) in cfg.val_steps)
            or (not cfg.val_steps
                and ((step + 1) % cfg.val_interval_steps == 0
                     or step == cfg.max_steps - 1))
        )
        if diag_due and (exact_freq_log is not None or freq_bin_log is not None):
            import diag_worker as diag_worker_mod
            train_pairs = diag_worker_mod.make_pair_payloads(
                exact_freq_train_batches, model, diag_amp_dtype, cfg.vocab_size,
                token_payload=multi_order_index)
            val_pairs_all = diag_worker_mod.make_pair_payloads(
                fixed_val_batches, model, diag_amp_dtype, cfg.vocab_size,
                token_payload=multi_order_index)
            val_exact_pairs = val_pairs_all[:cfg.val_batches]
            val_fb_pairs = val_pairs_all[:args.freq_eval_batches]
            cur_pair = None
            if freq_bin_log is not None and cur_inp is not None and multi_order_index:
                cur_pair = (cur_inp.cpu().numpy(),
                            cur_ptl.detach().float().cpu().numpy().ravel())
            elif freq_bin_log is not None and cur_inp is not None:
                from ngram_freq import compute_context_keys
                cb_keys, ct_keys = compute_context_keys(cur_inp, cfg.vocab_size)
                cur_pair = (cb_keys.cpu().numpy().ravel(),
                            ct_keys.cpu().numpy().ravel(),
                            cur_ptl.detach().float().cpu().numpy().ravel())
            if diag_pool is not None and diag_pool.alive:
                diag_pool.submit(step + 1, train_ds._epoch + 1, probe_hash,
                                 train_pairs, val_exact_pairs, val_fb_pairs, cur_pair)
                diag_pool.drain(on_diag_rows, on_logit_rows=on_logit_rows)
            else:
                exact_payload, fb_payload = diag_worker_mod.process_job(
                    freq_index_obj, cfg.vocab_size, train_pairs,
                    val_exact_pairs, val_fb_pairs, cur_pair)
                on_diag_rows(step + 1, train_ds._epoch + 1, probe_hash,
                             exact_payload, fb_payload)
            if exact_freq_log is not None:
                exact_freq_log.flush()
            if freq_bin_log is not None:
                freq_bin_log.flush()

        # --- per-token stats dump (post-hoc gap-freq analysis) ---
        # Diagnostic-only: failures here must never kill a finished run.
        if (step + 1) in dump_steps or step == cfg.max_steps - 1:
            _dtag = f"step{step + 1:06d}"
            try:
                _nv = dump_token_stats_npz(
                    os.path.join(cfg.out_dir, f"token_stats_{_dtag}_val.npz"),
                    model, fixed_val_batches[:cfg.val_batches], dump_orders,
                    cfg.vocab_size, dump_amp_dtype_spec)
                _nt = dump_token_stats_npz(
                    os.path.join(cfg.out_dir, f"token_stats_{_dtag}_train.npz"),
                    model, dump_train_batches, dump_orders,
                    cfg.vocab_size, dump_amp_dtype_spec)
                print(f"[nglab] token stats dumped step {step + 1}: "
                      f"val {_nv['tokens']} tok, train {_nt['tokens']} tok")
            except Exception as _de:
                print(f"[nglab] token stats dump FAILED at step {step + 1}: {_de!r}")

        # --- logit-stats probe submission (§5.2 supplement) ---
        # Runs on the SAME cadence as diag_due. Uses separate bf16 forwards on
        # val batches to keep the standard exact-freq/freq-bin pipeline byte-
        # identical when --logit_stats is off. Submits one job per branch.
        if diag_due and logit_stats_log is not None:
            import diag_worker as diag_worker_mod
            for _ls_branch in ("bigram", "trigram"):
                ls_payloads = diag_worker_mod.make_logit_payloads(
                    fixed_val_batches, model, diag_amp_dtype, cfg.vocab_size,
                    freq_index_obj, _ls_branch)
                # Aggregate payloads: list of (keys, hits, entropy, margin, ptrue)
                agg_pairs = [(p[0], p[1], p[2], p[3], p[4]) for p in ls_payloads]
                # Exemplar payloads: for each exemplar, find matching positions in
                # the val batches and extract top20. We precomputed exemplar keys;
                # now scan ls_payloads for matches.
                ex_pairs = []
                for eid, ebranch, ehit, eq in logit_exemplar_ids:
                    if ebranch != _ls_branch:
                        continue
                    # Find first occurrence of this key in any batch
                    found = False
                    for pi, p in enumerate(ls_payloads):
                        keys_flat = p[0]
                        idx_match = np.where(keys_flat == eid)[0]
                        if len(idx_match) > 0:
                            pos = int(idx_match[0])
                            ex_pairs.append((
                                eid,
                                np.array([eid], dtype=np.int64),
                                np.array([ehit], dtype=np.int64),
                                p[5][pos:pos+1],   # top20_ids (1, 20)
                                p[6][pos:pos+1],   # top20_probs (1, 20)
                            ))
                            found = True
                            break
                    # If not found in val batches, skip this exemplar for this step
                if diag_pool is not None and diag_pool.alive:
                    diag_pool.submit_logit(step + 1, train_ds._epoch + 1,
                                           _ls_branch, agg_pairs, ex_pairs)
                    diag_pool.drain(on_diag_rows, on_logit_rows=on_logit_rows)
                else:
                    # Sync fallback
                    from ngram_freq import LogitStatsAccumulator
                    acc = LogitStatsAccumulator(_ls_branch)
                    for keys_np, hits_np, entropy_np, margin_np, ptrue_np in agg_pairs:
                        acc.update_numpy(keys_np, hits_np, entropy_np, margin_np, ptrue_np)
                    summary = acc.summary()
                    exemplar_dump = []
                    for eid, ek, eh, et_ids, et_probs in ex_pairs:
                        exemplar_dump.append({
                            "exemplar_id": int(eid),
                            "branch": _ls_branch,
                            "hit_count": int(np.asarray(eh).ravel()[0]),
                            "top20_ids": [int(x) for x in np.asarray(et_ids).ravel()[:20]],
                            "top20_probs": [float(x) for x in np.asarray(et_probs).ravel()[:20]],
                        })
                    on_logit_rows(step + 1, train_ds._epoch + 1, _ls_branch,
                                  summary, exemplar_dump)
            if logit_stats_log is not None:
                logit_stats_log.flush()

        # periodic table norm
        if (step + 1) % cfg.table_norm_interval_steps == 0:
            tn = table_param_rms(model)
            tn_entry = {"step": step + 1, **tn}
            table_log.write(json.dumps(tn_entry) + "\n")
            table_log.flush()

        # Optional weights-only snapshots at complete replay-epoch boundaries.
        # The training loop has no resume-state semantics here: these files are
        # analysis/checkpoint artifacts, not optimizer-resumable checkpoints.
        if (args.save_epoch_checkpoints and cfg.epoch_batches > 0
                and train_ds._batch_in_epoch == 0 and step > 0):
            raw_model = getattr(model, "_orig_mod", model)
            epoch_no = train_ds._epoch
            path = os.path.join(cfg.out_dir, f"epoch{epoch_no}_model.pt")
            torch.save(raw_model.state_dict(), path)
            print(f"[nglab] saved epoch checkpoint {path}")

    # flush any diagnostics still queued in the background worker
    if diag_pool is not None:
        diag_pool.close_and_drain(on_diag_rows, on_logit_rows=on_logit_rows)
        if exact_freq_log is not None:
            exact_freq_log.flush()
        if freq_bin_log is not None:
            freq_bin_log.flush()
    if logit_stats_log is not None:
        logit_stats_log.flush()
        logit_stats_log.close()

    train_log.close()
    table_log.close()
    if table_diag_log is not None:
        table_diag_log.close()
    if probe_log is not None:
        probe_log.close()
    if fixed_train_freq_log is not None:
        fixed_train_freq_log.close()
    if freq_bin_log is not None:
        freq_bin_log.close()
    if exact_freq_log is not None:
        exact_freq_log.close()

    # save final model + probe tokens for offline per-token / row-level analysis
    if args.save_final_model:
        raw_model = getattr(model, "_orig_mod", model)
        torch.save(raw_model.state_dict(), os.path.join(cfg.out_dir, "final_model.pt"))
        print(f"[nglab] saved final_model.pt ({sum(p.numel() for p in raw_model.parameters())/1e6:.1f}M params)")
        if fixed_train_probe:
            # Save the full chunk (input + the single next token as target tail)
            # so downstream row-level analysis can rebuild exact (inp, tgt)
            # pairs and hash contexts without off-by-one.
            probe_chunk = torch.stack([
                torch.cat([b[0].cpu(), b[1][:, -1:].cpu()], dim=1)
                for b in fixed_train_probe
            ]).numpy()  # (n_probe, B, T+1)
            np.savez(
                os.path.join(cfg.out_dir, "probe_tokens.npz"),
                fixed_train_probe_chunk=probe_chunk,
                fixed_train_probe_sha256=probe_hash,
            )
            print(f"[nglab] saved probe_tokens.npz (fixed train probe chunk {probe_chunk.shape}, sha={probe_hash})")

    # summary
    summary = {
        "run_id": args.run_id,
        "injection_position": cfg.nanogpt_ngram_injection_position,
        "inject_layers": cfg.nanogpt_inject_layers,
        "steps": cfg.max_steps,
        "seed": cfg.seed,
        "epoch_batches": cfg.epoch_batches,
        "train_epochs": args.train_epochs,
        "train_max_tokens": cfg.train_max_tokens,
        "train_pool_content_tokens": train_ds.pool_content_tokens(),
        "dump_token_stats_steps": sorted(dump_steps) if dump_steps else "final_only",
        "replay_mix_passes": cfg.replay_mix_passes,
        "fixed_train_probe_sha256": probe_hash if args.fixed_train_probe > 0 else None,
        "fixed_train_probe_batches": len(fixed_train_probe),
        "train_probe_mode": args.train_probe_mode if args.fixed_train_probe > 0 else None,
        "probe_eval_interval": args.probe_eval_interval,
        "val_steps": cfg.val_steps or None,
        "exact_freq_eval_interval": args.exact_freq_eval_interval,
        "fixed_train_loss_log": (
            "fixed_train_loss.jsonl" if probe_log is not None else None
        ),
        "fixed_train_freq_bin_log": (
            "fixed_train_freq_bin_loss.jsonl"
            if fixed_train_freq_log is not None else None
        ),
        "exact_freq_log": (
            "exact_freq_loss.jsonl" if exact_freq_log is not None else None
        ),
        "logit_stats_log": (
            "logit_stats.jsonl" if logit_stats_log is not None else None
        ),
        "logit_stats_exemplars": len(logit_exemplar_ids) if logit_stats_log is not None else 0,
        "freq_index": args.freq_index if args.freq_index else None,
        "freq_index_branches": (list(freq_index_obj.branches)
                                if multi_order_index else None),
        "ngram_orders": ngram_orders or None,
        "diag_workers": args.diag_workers,
        "freq_index_sha256": (
            file_sha256(args.freq_index)
            if args.freq_index and os.path.isfile(args.freq_index) else None
        ),
        "intervention": {
            "kind": cfg.intervention,
            "epoch0": cfg.intervention_epoch,
            "epochs_due": (
                sorted({cfg.intervention_epoch} | set(cfg.intervention_epochs))
                if cfg.intervention != "none" else None
            ),
            "low_inclusive": cfg.intervention_low_inclusive,
            "frequency_threshold": cfg.intervention_freq_threshold,
            "events": intervention_events,
        },
        "compute_dtype": args.dtype,
        "torch_compile": args.compile,
        "init_model": os.path.abspath(args.init_model) if args.init_model else None,
        "freeze_table": bool(args.freeze_table),
        "table_init_count": str(args.table_init_count),
        "freeze_embeddings_only": bool(args.freeze_embeddings_only),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha if args.lora_rank else None,
        "lora_targets": lora_targets,
        "disable_ngram_tables": bool(args.disable_ngram_tables),
        "row_index": bool(args.row_index),
        "ngram_dict_topk": int(args.ngram_dict_topk),
        "ngram_keep_rows": args.ngram_keep_rows,
        "ngram_orders": sorted(ngram_orders),
        "rows_meta": _rows_meta,
        "token_dtype": cfg.token_dtype,
        "ce_chunk_positions": cfg.ce_chunk_positions,
        "static_low_mask": cfg.static_low_mask,
        "static_low_mask_keep_novel": cfg.static_low_mask_keep_novel,
        "fp8_torchao": bool(args.fp8_torchao),
        "fp8_converted_modules": fp8_modules,
        "diag_forward_dtype": ("bf16" if diag_amp_dtype is not None else "fp32")
                              if (exact_freq_log is not None or freq_bin_log is not None) else None,
        "diag_async": diag_pool is not None,
        "final_train_loss": last_train_loss,
        "final_val_loss": last_val_loss,
        "final_gap": last_val_loss - last_train_loss,
        "final_fixed_train_loss": last_fixed_train_loss,
        "final_fixed_val_loss": last_fixed_val_loss,
        "final_fixed_gap": last_fixed_val_loss - last_fixed_train_loss,
        "n_params": n_params,
        "bigram_perfect_rows": (
            getattr(getattr(model, "_orig_mod", model), "bigram_table_size", None)
            if cfg.bigram_perfect_map else None
        ),
        "config": cfg.__dict__,
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[nglab] DONE. final gap = {last_val_loss - last_train_loss:+.4f}")
    print(f"[nglab] output: {cfg.out_dir}")


if __name__ == "__main__":
    main()
