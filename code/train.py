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
    intervention: str = "none"        # none | freeze_table | freeze_backbone | freeze_both | hash_reseed | mask_low_freq | mask_high_freq
    intervention_epoch: int = 1       # fire when 0-indexed epoch reaches this value (1 = start of epoch 2)
    intervention_epochs: tuple = ()   # extra 0-indexed due epochs for multi-fire interventions (e.g. (1, 2))
    intervention_freq_threshold: int = 200
    intervention_low_inclusive: bool = False  # mask_low masks f <= threshold (default f < threshold)
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
    replay_mix_passes: int = 0  # >0: one globally chunk-shuffled stream of N passes
                                # (shuffle experiment mixed arm); overrides epoch replay
    # output
    out_dir: str = ""


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Alternating VE layers."""
    return layer_idx % 2 == (n_layer - 1) % 2


def _parse_inject_layers(config) -> "set[int] | None":
    """--inject_layers for v/y multi-layer injection; None = default layout."""
    s = getattr(config, "nanogpt_inject_layers", "") or ""
    if not s.strip():
        return None
    if config.nanogpt_ngram_injection_position not in {"v", "y"}:
        raise ValueError("--inject_layers only applies to v/y injection")
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
        # gates only needed when injection is v or y (input injection has no gate)
        need_gate = use_ngram and config.nanogpt_ngram_injection_position in {"v", "y"}
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
        self._freq_mask_index = None
        self._freq_mask_mode = "none"
        self._freq_mask_threshold = 0
        self._hash_reseed_count = 0
        self._intervention_due = frozenset(
            {config.intervention_epoch} | set(config.intervention_epochs)
        )

    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
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
        markers = ("value_embeds", "bigram_ves", "trigram_ves",
                   "ve_gate", "bigram_gate", "trigram_gate")
        for name, p in self.named_parameters():
            if any(m in name for m in markers):
                yield p

    def _backbone_params(self):
        markers = ("value_embeds", "bigram_ves", "trigram_ves",
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

    def _compute_input_ngram_residual(self, idx):
        """Over-encoding: sum all enabled layers' n-gram values, no gate."""
        _B, T = idx.size()
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
        if residual is None:
            residual = torch.zeros(idx.size(0), T, self.config.n_embd,
                                   device=idx.device, dtype=self.transformer.wte.weight.dtype)
        return residual

    def forward(self, idx, targets=None, return_token_losses=False,
                return_injection_stats=False):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.transformer.wte(idx)
        x = x + self.transformer.wpe(pos)
        injection_stats = {
            "bigram_rms": {"sequence_rms": [], "batch_rms": 0.0},
            "trigram_rms": {"sequence_rms": [], "batch_rms": 0.0},
            "total_rms": {"sequence_rms": [], "batch_rms": 0.0},
            "fourgram_rms": {"sequence_rms": [], "batch_rms": 0.0},
        }
        branch_residuals = {
            "bigram": [],
            "trigram": [],
        }
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        if self.config.nanogpt_ngram_injection_position == "input":
            ngram_residual = self._compute_input_ngram_residual(idx)
            x = x + ngram_residual
            if return_injection_stats:
                branch_residuals["bigram"] = []
                branch_residuals["trigram"] = []
                for li in sorted(self.bigram_ve_layers):
                    lvs = self.bigram_ves[str(li)]
                    bi = self._bigram_row_indices(li, prev_idx, idx)
                    branch_residuals["bigram"].append(
                        self._pad_to_embd(
                            torch.cat([lvs[k](bi[k]) for k in range(self.bigram_K)], dim=-1),
                            self.bigram_table_dim,
                        )
                    )
                for li in sorted(self.trigram_ve_layers):
                    lvs = self.trigram_ves[str(li)]
                    ti = self._trigram_row_indices(li, prev2_idx, prev_idx, idx)
                    branch_residuals["trigram"].append(
                        self._pad_to_embd(
                            torch.cat([lvs[k](ti[k]) for k in range(self.trigram_K)], dim=-1),
                            self.trigram_table_dim,
                        )
                    )
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
        x = self.transformer.drop(x)
        bigram_indices = {}
        if self.config.enable_bigram_ve:
            for li in self.bigram_ve_layers:
                bigram_indices[li] = self._bigram_row_indices(li, prev_idx, idx)
        trigram_indices = {}
        if self.config.enable_trigram_ve:
            for li in self.trigram_ve_layers:
                trigram_indices[li] = self._trigram_row_indices(li, prev2_idx, prev_idx, idx)
        for i, block in enumerate(self.transformer.h):
            ve = (self.value_embeds[str(i)](idx)
                  if self.config.enable_unigram_ve and str(i) in self.value_embeds else None)
            bgve = None
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
            tgve = None
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
        logits = self.lm_head(x)
        if targets is not None:
            token_losses = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1, reduction="none"
            ).view_as(targets)
            loss = token_losses[targets.ne(-1)].mean()
            if return_token_losses and return_injection_stats:
                return loss, token_losses, injection_stats
            if return_token_losses:
                return loss, token_losses
            if return_injection_stats:
                return loss, injection_stats
            return loss
        if return_token_losses:
            raise ValueError("return_token_losses requires targets")
        if return_injection_stats:
            return logits.float(), injection_stats
        return logits.float()


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
        ngram_markers = ("value_embeds", "bigram_ves", "trigram_ves",
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
        # no weight decay on n-gram tables
        exp_avg_sq.lerp_(g.square(), 1 - b2)
        bias2 = 1 - b2 ** step
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
                 replay_mix_passes: int = 0):
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
        # total tokens per shard (lazy)
        self._shard_lens = {}
        # yielded batches per full epoch (drop_last per shard), for epoch-anchored LR
        if epoch_batches > 0:
            # Nested-prefix epoch length: the first `epoch_batches` device
            # batches from the shard prefix form one epoch; all epoch lengths
            # are strict prefixes of the same data stream.
            self._batches_per_epoch = epoch_batches
        else:
            self._batches_per_epoch = sum(
                self.shard_chunks(sid) // self.device_batch_size for sid in self.shard_ids)

    def _load_shard(self, sid):
        if sid not in self._buffers:
            path = os.path.join(self.data_dir, f"shard_{sid:05d}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing shard: {path}")
            buf = np.memmap(path, dtype=np.uint16, mode="r")
            self._buffers[sid] = buf
        return self._buffers[sid]

    def shard_chunks(self, sid):
        buf = self._load_shard(sid)
        return len(buf) // self.chunk_size

    def total_chunks(self):
        return sum(self.shard_chunks(s) for s in self.shard_ids)

    def _iter_chunks(self):
        """Yield (input, target) tensors for each chunk in fixed order.

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
                yield self.get_chunk(local)
            return
        budget = self.epoch_batches * self.device_batch_size if self.epoch_batches > 0 else None
        emitted = 0
        for sid in self.shard_ids:
            buf = self._load_shard(sid)
            n = self.shard_chunks(sid)
            for i in range(n):
                if budget is not None and emitted >= budget:
                    return
                start = i * self.chunk_size
                chunk = np.array(buf[start:start + self.chunk_size], dtype=np.int64)
                inp = torch.from_numpy(chunk[:-1])
                tgt = torch.from_numpy(chunk[1:])
                emitted += 1
                yield inp, tgt

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

    def get_chunk(self, global_idx: int):
        """Return (input, target) tensors for the g-th chunk of the full stream.

        Uniform train probe uses this to grab batches spread across the whole
        train set (independent of the training iterator, which is what the
        head-only `first` probe cannot do).
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
                return inp, tgt
        raise IndexError(f"chunk {global_idx} out of range")

    def epoch_progress(self) -> float:
        """Fraction of the current epoch completed (0..1 within an epoch)."""
        return self._batch_in_epoch / max(1, self._batches_per_epoch)

    def iter_batches(self, device: torch.device):
        """Yield (inputs, targets) batches of shape (B, T) in fixed order, looping forever."""
        batch_inp, batch_tgt = [], []
        while True:
            for inp, tgt in self._iter_chunks():
                batch_inp.append(inp)
                batch_tgt.append(tgt)
                if len(batch_inp) == self.device_batch_size:
                    yield (torch.stack(batch_inp).to(device),
                           torch.stack(batch_tgt).to(device))
                    batch_inp, batch_tgt = [], []
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="run")
    parser.add_argument("--injection_position", default="input",
                        choices=["v", "y", "input"])
    parser.add_argument("--inject_layers", default="",
                        help="v/y only: comma layer ids for multi-layer injection, "
                             "one independent clean table per layer (e.g. '1,3,5,7'); "
                             "empty = default single-layer layout")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
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
                                 "mask_low_freq", "mask_high_freq", "readout_last_epoch"],
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
    args = parser.parse_args()
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
    if args.intervention == "none" and args.intervention_epochs.strip():
        parser.error("--intervention_epochs requires --intervention hash_reseed (or another intervention)")

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
        intervention=args.intervention,
        intervention_epoch=args.intervention_epoch,
        intervention_epochs=tuple(
            int(x) for x in args.intervention_epochs.split(",") if x.strip()
        ),
        intervention_freq_threshold=args.intervention_freq_threshold,
        intervention_low_inclusive=bool(args.intervention_low_inclusive),
        data_dir=args.data_dir,
        train_shards=[int(x) for x in args.train_shards.split(",") if x.strip()],
        val_shards=[int(x) for x in args.val_shards.split(",") if x.strip()],
        epoch_batches=args.epoch_batches,
        replay_mix_passes=args.replay_mix_passes,
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
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nglab] device={device} injection={cfg.nanogpt_ngram_injection_position} steps={cfg.max_steps}")

    # data
    train_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, cfg.sequence_len,
                                     cfg.device_batch_size, cfg.data_seed,
                                     epoch_batches=cfg.epoch_batches,
                                     replay_mix_passes=cfg.replay_mix_passes)
    val_ds = TokenizedShardDataset(cfg.data_dir, cfg.val_shards, cfg.sequence_len,
                                   cfg.device_batch_size, cfg.data_seed)
    train_iter = train_ds.iter_batches(device)
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
                                         epoch_batches=cfg.epoch_batches)
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
    if args.freq_index and os.path.exists(args.freq_index):
        from ngram_freq import GlobalFrequencyIndex
        freq_index_obj = GlobalFrequencyIndex.load(args.freq_index)
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
                                              cfg.data_seed)
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
                                        inherited_index=freq_index_obj)
                print("[nglab] async diag worker started "
                      f"(diagnostic forwards: {args.dtype})")
            except Exception as e:
                print(f"[nglab] async diag unavailable ({e!r}); using sync path")
                diag_pool = None

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
    for step in range(cfg.max_steps):
        # gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        for micro in range(grad_accum):
            try:
                inp, tgt = next(train_iter)
            except StopIteration:
                train_iter = train_ds.iter_batches(device)
                inp, tgt = next(train_iter)
            if (train_ds._epoch in intervention_due
                    and train_ds._epoch not in intervention_fired_epochs):
                event = model.apply_intervention(train_ds._epoch, freq_index_obj)
                event["step"] = step + 1
                intervention_events.append(event)
                intervention_fired_epochs.add(train_ds._epoch)
                print(f"[nglab] intervention={cfg.intervention} at step={step + 1} "
                      f"epoch={train_ds._epoch + 1}")
            with amp_ctx:
                if freq_bin_log is not None and micro == grad_accum - 1:
                    loss, ptl = model(inp, targets=tgt, return_token_losses=True)
                    cur_inp, cur_tgt, cur_ptl = inp, tgt, ptl.detach()
                    loss = loss / grad_accum
                else:
                    loss = model(inp, targets=tgt) / grad_accum
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
                exact_freq_train_batches, model, diag_amp_dtype, cfg.vocab_size)
            val_pairs_all = diag_worker_mod.make_pair_payloads(
                fixed_val_batches, model, diag_amp_dtype, cfg.vocab_size)
            val_exact_pairs = val_pairs_all[:cfg.val_batches]
            val_fb_pairs = val_pairs_all[:args.freq_eval_batches]
            cur_pair = None
            if freq_bin_log is not None and cur_inp is not None:
                from ngram_freq import compute_context_keys
                cb_keys, ct_keys = compute_context_keys(cur_inp, cfg.vocab_size)
                cur_pair = (cb_keys.cpu().numpy().ravel(),
                            ct_keys.cpu().numpy().ravel(),
                            cur_ptl.detach().float().cpu().numpy().ravel())
            if diag_pool is not None and diag_pool.alive:
                diag_pool.submit(step + 1, train_ds._epoch + 1, probe_hash,
                                 train_pairs, val_exact_pairs, val_fb_pairs, cur_pair)
                diag_pool.drain(on_diag_rows)
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

        # periodic table norm
        if (step + 1) % cfg.table_norm_interval_steps == 0:
            tn = table_param_rms(model)
            tn_entry = {"step": step + 1, **tn}
            table_log.write(json.dumps(tn_entry) + "\n")
            table_log.flush()

    # flush any diagnostics still queued in the background worker
    if diag_pool is not None:
        diag_pool.close_and_drain(on_diag_rows)
        if exact_freq_log is not None:
            exact_freq_log.flush()
        if freq_bin_log is not None:
            freq_bin_log.flush()

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
        "freq_index": args.freq_index if args.freq_index else None,
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
