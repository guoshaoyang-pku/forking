"""Plan-9 MoE: moeify() config generator, matched MoE FFN, parameter/FLOPs ledger.

Design SSOT: docs/plans/plan-9-moe-oe-baseline.md (v2) +
docs/plans/plan-9-handoff-gptpro-20261017.txt (GPT-pro handoff, sections 3/7/8).

Coordinates (Fine-Grained MoE Scaling notation):
    G = d_ff / d_e          (granularity; larger = finer experts)
    X = total MoE FFN / dense FFN   (expansion; = expert units / active units)
    S = shared expert units (always-on)
    d_e = d_ff / G, K_routed = G, E_routed = G*X - S
    per-token active units = K_routed + S = G  (strictly matched)

Structural contract (fixed by unit tests, code/tools/test_moe_cpu.py):
    - router: nn.Linear(d, E_routed, bias=False), logits+softmax in fp32
      (autocast disabled around the router)
    - softmax over ALL experts, then top-K; OLMoE-faithful default does NOT
      renormalize top-k weights (--moe_norm_topk_prob flips it)
    - shared experts are always-on with output weight 1.0 (DeepSeekMoE style)
    - aux (Switch-style, fp32): L_aux = E * sum_i f_i * p_i, where
      f_i = fraction of (token, slot) assignments to expert i (sums to 1),
      p_i = mean router probability of expert i (sums to 1).
      Uniform router gives exactly 1.0 (scale-free w.r.t. K).
    - router z-loss (fp32): L_z = mean(logsumexp(logits)^2)
    - selected_topk_probability_mass = sum of top-k probs BEFORE renormalize
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def moe_diag_paused(model):
    """Pause MoE load-stat accumulation (val/probe evals are train-only stats)."""
    mods = model.moe_layer_modules() if hasattr(model, "moe_layer_modules") else []
    for m in mods:
        m.diag_enabled = False
    try:
        yield
    finally:
        for m in mods:
            m.diag_enabled = True


@torch.no_grad()
def capture_panel_routing(model, panel_x: torch.Tensor) -> dict:
    """Route a FIXED held-out panel (N, T) through every MoE layer.

    Returns per-layer dict {topk_idx (N*T, K) int16, topk_probs (N*T, K) fp32,
    top1_margin (N*T,) fp32}. Raw captures are stored per checkpoint; the
    ossification metrics (consecutive/epoch-1 top-1 agreement, top-k Jaccard,
    rank-weighted overlap, K/K+1 margin) are computed offline from two or
    more captures (handoff 8.3). Must be called in eval mode.
    """
    mods = model.moe_layer_modules()
    stores = [{} for _ in mods]
    hooks = []

    def _hook(store):
        def fn(mod, inputs):
            x = inputs[0]
            logits, probs, topk_probs, topk_idx = mod.route(x)
            n = x.shape[0] * x.shape[1]
            store["topk_idx"] = topk_idx.reshape(n, mod.arch.k_routed).to(torch.int16).cpu()
            store["topk_probs"] = topk_probs.reshape(n, mod.arch.k_routed).float().cpu()
            top2 = torch.topk(probs.reshape(n, mod.arch.e_routed), 2, dim=-1)
            store["top1_margin"] = (top2.values[:, 0] - top2.values[:, 1]).float().cpu()
        return fn

    for m, store in zip(mods, stores):
        hooks.append(m.register_forward_pre_hook(_hook(store)))
    try:
        model(panel_x)  # routing fires in the pre-hooks; output discarded
    finally:
        for h in hooks:
            h.remove()
    return {f"l{i}": s for i, s in enumerate(stores)}


@dataclass
class MoEArch:
    d_model: int
    d_ff: int
    granularity: int          # G
    expansion: int            # X
    shared_units: int         # S
    d_e: int
    e_routed: int
    k_routed: int
    aux_weight: float
    zloss_weight: float
    norm_topk_prob: bool

    def describe(self) -> str:
        s = (f"{self.e_routed}R/top-{self.k_routed}"
             + (f"+{self.shared_units}S" if self.shared_units else ""))
        return (f"G{self.granularity}/X{self.expansion}/S{self.shared_units} "
                f"= {s}/d_e{self.d_e}")


def moeify(d_model: int, d_ff: int, granularity: int, expansion: int,
           shared_units: int = 0, aux_weight: float = 0.01,
           zloss_weight: float = 0.001, norm_topk_prob: bool = False) -> MoEArch:
    """Validate a (G, X, S) point and derive the concrete MoE configuration.

    Raises ValueError on any illegal combination (handoff section 3).
    """
    if granularity < 1 or expansion < 1:
        raise ValueError(f"G and X must be >= 1 (got G={granularity}, X={expansion})")
    if d_ff % granularity != 0:
        raise ValueError(f"d_ff ({d_ff}) must be divisible by G ({granularity})")
    if not (0 <= shared_units < granularity):
        raise ValueError(f"S must satisfy 0 <= S < G (got S={shared_units}, "
                         f"G={granularity})")
    e_routed = granularity * expansion - shared_units
    k_routed = granularity - shared_units
    if e_routed < k_routed:
        raise ValueError(f"E_routed ({e_routed}) must be >= K_routed ({k_routed})")
    if k_routed < 1:
        raise ValueError("need at least one routed slot (K_routed >= 1)")
    return MoEArch(
        d_model=d_model, d_ff=d_ff, granularity=granularity, expansion=expansion,
        shared_units=shared_units, d_e=d_ff // granularity, e_routed=e_routed,
        k_routed=k_routed, aux_weight=aux_weight, zloss_weight=zloss_weight,
        norm_topk_prob=norm_topk_prob,
    )


class ExpertMLP(nn.Module):
    """Identical layout to the dense train.MLP (c_fc GELU c_proj dropout)."""

    def __init__(self, d_model: int, d_e: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_e, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_e, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class MoEFFN(nn.Module):
    """Dropless token-choice MoE FFN replacing the dense MLP of a Block.

    forward() returns the FFN output only; the router losses are stashed as
    differentiable fp32 scalars (last_aux_loss / last_z_loss) for the training
    loop to add to the LM loss (keeps evaluate_val() a pure CE path).
    Load diagnostics accumulate in-place and are drained by
    drain_load_stats() at each eval block.
    """

    def __init__(self, arch: MoEArch, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.arch = arch
        self.experts = nn.ModuleList(
            [ExpertMLP(arch.d_model, arch.d_e, bias=bias, dropout=dropout)
             for _ in range(arch.e_routed)])
        self.shared_experts = nn.ModuleList(
            [ExpertMLP(arch.d_model, arch.d_e, bias=bias, dropout=dropout)
             for _ in range(arch.shared_units)])
        self.router = nn.Linear(arch.d_model, arch.e_routed, bias=False)
        # rolling load diagnostics (drained per eval block); paused during
        # val/probe evals so stats cover training tokens only
        self.diag_enabled = True
        self._assign_counts = torch.zeros(arch.e_routed, dtype=torch.float64)
        self._n_tokens = 0
        self._topk_mass_sum = 0.0
        self._n_forwards = 0
        self.last_aux_loss: torch.Tensor | None = None
        self.last_z_loss: torch.Tensor | None = None
        self.last_topk_mass: torch.Tensor | None = None

    # --- routing (fp32, autocast-immune) -----------------------------------
    def route(self, x: torch.Tensor):
        """x: (..., d) any compute dtype -> (probs, topk_probs, topk_idx) fp32.

        softmax over ALL routed experts, then top-k. Shared experts are not
        part of the router.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.router(x.float())
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_idx = torch.topk(probs, self.arch.k_routed, dim=-1)
        return logits, probs, topk_probs, topk_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arch = self.arch
        B, T, d = x.shape
        logits, probs, topk_probs, topk_idx = self.route(x)
        weights = topk_probs
        if arch.norm_topk_prob:
            weights = topk_probs / topk_probs.sum(-1, keepdim=True)
        n = B * T
        topk_probs = topk_probs.reshape(n, arch.k_routed)
        topk_idx = topk_idx.reshape(n, arch.k_routed)
        probs_flat = probs.reshape(n, arch.e_routed)

        flat_x = x.reshape(-1, d)
        out = None   # created at the first expert output: accumulation dtype
        # must match what the dense MLP would produce under the same context
        # (bf16 under autocast, fp32 in eager), see test_moe_cpu T2.
        # a top-k result never repeats an expert within one token, so
        # (topk_idx == e) marks exactly one slot per (token, expert) pair
        for e, expert in enumerate(self.experts):
            sel = topk_idx == e                       # (N, K) bool
            if not sel.any():
                continue
            rows = sel.any(-1)                        # (N,) tokens using expert e
            w = (topk_probs * sel).sum(-1)            # per-token weight (one slot)
            y = expert(flat_x[rows])
            if out is None:
                out = torch.zeros(n, d, dtype=y.dtype, device=x.device)
            out[rows] += (y * w[rows].unsqueeze(-1)).to(out.dtype)
        if out is None:  # unreachable for K >= 1, kept for safety
            out = torch.zeros_like(flat_x)
        for s_expert in self.shared_experts:          # always-on, weight 1.0
            out = out + s_expert(flat_x)
        out = out.view(B, T, d)

        # ---- router losses (fp32, differentiable) ----
        with torch.autocast(device_type=x.device.type, enabled=False):
            assign = F.one_hot(topk_idx, arch.e_routed).sum(1)   # (N, E) counts
            f = assign.to(torch.float32).sum(0) / (n * arch.k_routed)
            p = probs_flat.to(torch.float32).mean(0)
            self.last_aux_loss = arch.e_routed * (f * p).sum()
            self.last_z_loss = (torch.logsumexp(
                logits.reshape(-1, arch.e_routed), dim=-1) ** 2).mean()
            self.last_topk_mass = topk_probs.sum(-1).mean()

        # ---- load diagnostics (no grad) ----
        with torch.no_grad():
            if self.diag_enabled:
                self._assign_counts += assign.to(torch.float64).sum(0).cpu()
                self._n_tokens += n
                self._topk_mass_sum += float(self.last_topk_mass)
                self._n_forwards += 1
        return out

    def drain_load_stats(self) -> dict:
        """Per-eval-block load metrics; resets the accumulators.

        load entropy is normalized by log(E) so it is comparable across E
        (raw entropy thresholds across different E are forbidden, handoff 8.2).
        """
        e = self.arch.e_routed
        if self._n_tokens == 0:
            return {}
        f = (self._assign_counts / (self._n_tokens * self.arch.k_routed)).tolist()
        counts = self._assign_counts
        tot = counts.sum()
        nz = counts[counts > 0]
        h = -(nz / tot * (nz / tot).log()).sum().item() if len(nz) else 0.0
        mean = (tot / e).item()
        var = ((counts - mean) ** 2).sum().item() / e
        stats = {
            "n_tokens": self._n_tokens,
            "assign_frac": f,
            "load_entropy_norm": h / math.log(e),
            "eff_experts": math.exp(h),
            "load_cv": math.sqrt(var) / mean if mean > 0 else float("inf"),
            "max_over_mean": (counts.max().item() / mean) if mean > 0 else float("inf"),
            "dead_experts": int((counts == 0).sum().item()),
            "topk_mass_mean": self._topk_mass_sum / max(1, self._n_forwards),
        }
        self._assign_counts.zero_()
        self._n_tokens = 0
        self._topk_mass_sum = 0.0
        self._n_forwards = 0
        return stats


def params_ledger(arch: MoEArch, n_layer: int, vocab_size: int, sequence_len: int,
                  n_head: int, oe_table_params: int = 0) -> dict:
    """Full parameter + theoretical-FLOPs accounting (handoff 1.4 / 7.6).

    All counts are exact from the structural constants (no instantiation).
    OE table params are passed in by the caller from the real model
    (audit: code/tools/oe_param_audit.py).
    """
    d, d_ff = arch.d_model, arch.d_ff
    ln = 2 * d                                    # LayerNorm weight+bias
    per_layer_attn = 4 * d * d + 4 * d + ln       # qkvo + biases + ln_1
    per_layer_dense_ffn = 2 * d * d_ff + d + d_ff  # c_fc/c_proj + biases
    non_ffn = n_layer * per_layer_attn + 2 * ln   # + ln_f
    embeddings = vocab_size * d + sequence_len * d  # wte (+tied lm_head) + wpe
    dense_ffn = n_layer * per_layer_dense_ffn
    expert = 2 * d * arch.d_e + d + arch.d_e
    flat_x_matrix = 2 * d * arch.d_e             # matrix params per expert
    routed = n_layer * arch.e_routed * (flat_x_matrix + d + arch.d_e)
    shared = n_layer * arch.shared_units * (flat_x_matrix + d + arch.d_e)
    router = n_layer * (d * arch.e_routed)        # no bias
    active_ffn_per_layer = arch.k_routed * (flat_x_matrix + d + arch.d_e)
    dense_total = non_ffn + dense_ffn + embeddings
    moe_total = non_ffn + routed + shared + router + embeddings
    moe_active = non_ffn + n_layer * (active_ffn_per_layer
                                      + arch.shared_units * (flat_x_matrix + d + arch.d_e)
                                      + d * arch.e_routed) + embeddings
    # expansion X per the handoff formula (matrix params only): P_MoE_FFN_total
    # / P_dense_FFN = (E_routed + E_shared) * 2*d*d_e / (2*d*d_ff)
    ffn_expansion_matrix = ((arch.e_routed + arch.shared_units) * flat_x_matrix
                            / (2 * d * d_ff))
    # theoretical active matmul MACs/token: active matmul params (embeddings are
    # lookups; lm_head is tied so counted via wte) + attention scores/av
    attn_score_macs = 2 * n_layer * sequence_len * d
    return {
        "arch": arch.describe(),
        "granularity_G": arch.granularity, "expansion_X": arch.expansion,
        "shared_units_S": arch.shared_units, "d_e": arch.d_e,
        "e_routed": arch.e_routed, "k_routed": arch.k_routed,
        "k_over_e": arch.k_routed / arch.e_routed,
        "params": {
            "non_ffn": non_ffn,
            "embeddings_tied": embeddings,
            "dense_ffn": dense_ffn,
            "active_ffn_per_layer": active_ffn_per_layer,
            "routed_experts": routed,
            "shared_experts": shared,
            "router": router,
            "oe_table": oe_table_params,
            "dense_total": dense_total,
            "moe_total": moe_total,
            "moe_active": moe_active,
            "ffn_expansion_vs_dense": ffn_expansion_matrix,
            "whole_model_total_vs_dense": moe_total / dense_total,
            "whole_model_active_vs_dense_active": (
                (moe_active + oe_table_params) / (dense_total + oe_table_params)),
        },
        "matmul_macs_per_token": {
            "active_param_macs": moe_active - embeddings + oe_table_params,
            "attn_score_macs": attn_score_macs,
        },
        "matched_active_ffn": (arch.k_routed + arch.shared_units) * arch.d_e == d_ff,
    }


def format_ledger(led: dict) -> str:
    p = led["params"]
    lines = [
        f"[moe] arch {led['arch']}  (K/E = {led['k_over_e']:.3f}, "
        f"matched_active_ffn = {led['matched_active_ffn']})",
        f"[moe] params: non-FFN {p['non_ffn']:,} | embeddings(tied) "
        f"{p['embeddings_tied']:,} | dense-FFN-equivalent {p['dense_ffn']:,} "
        f"| active FFN/layer {p['active_ffn_per_layer']:,}",
        f"[moe] params: routed {p['routed_experts']:,} | shared "
        f"{p['shared_experts']:,} | router {p['router']:,} | OE table "
        f"{p['oe_table']:,}",
        f"[moe] params: whole-model total {p['moe_total'] + p['oe_table']:,} "
        f"(dense {p['dense_total'] + p['oe_table']:,}, "
        f"x{p['whole_model_total_vs_dense']:.2f}) | active "
        f"{p['moe_active'] + p['oe_table']:,} "
        f"(x{p['whole_model_active_vs_dense_active']:.2f})",
        f"[moe] FFN expansion X = {p['ffn_expansion_vs_dense']:.2f} | "
        f"theoretical active matmul MACs/token "
        f"{led['matmul_macs_per_token']['active_param_macs']:,} + attn scores "
        f"{led['matmul_macs_per_token']['attn_score_macs']:,}",
    ]
    return "\n".join(lines)
