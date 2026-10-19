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
    granularity: int          # G (legacy path); explicit path: active units K+S
    expansion: float          # X (legacy); explicit path: (E+S)/(K+S)
    shared_units: int         # S
    d_e: int
    e_routed: int
    k_routed: int
    aux_weight: float
    zloss_weight: float
    norm_topk_prob: bool
    ffn_type: str = "gelu"    # gelu | swiglu (FLAME profile)

    def describe(self) -> str:
        s = (f"{self.e_routed}R/top-{self.k_routed}"
             + (f"+{self.shared_units}S" if self.shared_units else ""))
        return (f"G{self.granularity}/X{self.expansion}/S{self.shared_units} "
                f"= {s}/d_e{self.d_e}" + (f"/{self.ffn_type}" if self.ffn_type != "gelu" else ""))


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


def moeify_explicit(d_model: int, d_e: int, e_routed: int, k_routed: int,
                    shared_units: int, aux_weight: float = 0.01,
                    zloss_weight: float = 0.001, norm_topk_prob: bool = False,
                    ffn_type: str = "gelu") -> MoEArch:
    """FLAME direct spec (plan-9 FLAME pivot): experts are specified by their
    concrete widths/counts, not derived from a dense d_ff via (G, X).

    FLAME Small: d_e=352, E_routed=64, K_routed=6, S=2 (the always-on shared
    FFN of width 2*d_e is exactly two shared d_e units, block-diagonal concat;
    SwiGLU gating is elementwise so the equivalence is exact).

    The ledger reference width d_ff is the per-token ACTIVE FFN width
    (K_routed + S) * d_e, which is the width the dense twin must match.
    """
    if d_e < 1:
        raise ValueError(f"d_e must be >= 1 (got {d_e})")
    if e_routed < 1 or k_routed < 1:
        raise ValueError(f"need E_routed, K_routed >= 1 (got {e_routed}, {k_routed})")
    if k_routed > e_routed:
        raise ValueError(f"K_routed ({k_routed}) must be <= E_routed ({e_routed})")
    if shared_units < 0:
        raise ValueError(f"S must be >= 0 (got {shared_units})")
    if ffn_type not in ("gelu", "swiglu"):
        raise ValueError(f"unsupported ffn_type: {ffn_type}")
    active_units = k_routed + shared_units
    d_ff = active_units * d_e
    return MoEArch(
        d_model=d_model, d_ff=d_ff, granularity=active_units,
        expansion=(e_routed + shared_units) / active_units,
        shared_units=shared_units, d_e=d_e, e_routed=e_routed,
        k_routed=k_routed, aux_weight=aux_weight, zloss_weight=zloss_weight,
        norm_topk_prob=norm_topk_prob, ffn_type=ffn_type,
    )


class ExpertMLP(nn.Module):
    """Expert FFN. gelu: identical layout to the dense train.MLP
    (c_fc GELU c_proj dropout). swiglu (FLAME): w1/w3 up-projections gated by
    SiLU, w2 down-projection (Megatron SwiGLU semantics, 3*d*d_e params).
    """

    def __init__(self, d_model: int, d_e: int, bias: bool = True,
                 dropout: float = 0.0, ffn_type: str = "gelu"):
        super().__init__()
        self.ffn_type = ffn_type
        if ffn_type == "swiglu":
            self.w1 = nn.Linear(d_model, d_e, bias=bias)
            self.w3 = nn.Linear(d_model, d_e, bias=bias)
            self.w2 = nn.Linear(d_e, d_model, bias=bias)
        else:
            self.c_fc = nn.Linear(d_model, d_e, bias=bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(d_e, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.ffn_type == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
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
            [ExpertMLP(arch.d_model, arch.d_e, bias=bias, dropout=dropout,
                       ffn_type=arch.ffn_type)
             for _ in range(arch.e_routed)])
        self.shared_experts = nn.ModuleList(
            [ExpertMLP(arch.d_model, arch.d_e, bias=bias, dropout=dropout,
                       ffn_type=arch.ffn_type)
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


# ---------------------------------------------------------------------------
# FLAME profile (plan-9 FLAME pivot, 2026-10-17): preset + exact ledger.
# ---------------------------------------------------------------------------

# name -> (n_layer, n_embd, n_head, d_ff_dense_layer0, d_e)
FLAME_SCALES = {
    "tiny":  (9, 256, 16, 1368, 176),
    "small": (12, 512, 16, 2736, 352),
    "large": (9, 1024, 16, 5472, 704),
}
FLAME_E_ROUTED = 64
FLAME_K_ROUTED = 6
FLAME_SHARED_UNITS = 2        # always-on shared FFN of width 2*d_e
FLAME_AUX_WEIGHT = 0.01
FLAME_ZLOSS_WEIGHT = 0.001


def flame_active_ffn_width(scale: str) -> int:
    """Per-token active FFN width (K+S)*d_e = 8*d_e; the dense-twin width."""
    return (FLAME_K_ROUTED + FLAME_SHARED_UNITS) * FLAME_SCALES[scale][4]


def flame_param_counts(n_layer: int, n_embd: int, n_head: int,
                       d_ff_dense: int, d_e: int, vocab_size: int,
                       dense_twin: bool, oe_table_params: int = 0) -> dict:
    """Exact FLAME-profile parameter accounting (no instantiation).

    SwiGLU everywhere (3*d*w per FFN), RMSNorm weights only, RoPE (no wpe),
    no biases, untied wte/lm_head. Layer 0 is always a dense SwiGLU FFN of
    width d_ff_dense; layers 1..L-1 are MoE (64R/top-6 + 2 shared units), or
    dense SwiGLU of width 8*d_e when dense_twin=True.
    """
    d = n_embd
    attn = n_layer * 4 * d * d                      # fused qkvo, no bias
    norms = (2 * n_layer + 1) * d                   # RMSNorm weights (block x2 + final)
    embed = 2 * vocab_size * d                      # untied wte + lm_head
    router_layer = d * FLAME_E_ROUTED
    router = (n_layer - 1) * router_layer if not dense_twin else 0
    layer0_ffn = 3 * d * d_ff_dense
    if dense_twin:
        twin_ffn = (n_layer - 1) * 3 * d * flame_active_ffn_width_from(d_e)
        total = attn + norms + embed + layer0_ffn + twin_ffn + oe_table_params
        active = total
    else:
        per_expert = 3 * d * d_e
        routed = (n_layer - 1) * FLAME_E_ROUTED * per_expert
        shared = (n_layer - 1) * FLAME_SHARED_UNITS * per_expert
        total = attn + norms + embed + layer0_ffn + routed + shared + router \
            + oe_table_params
        active_ffn_layer = (FLAME_K_ROUTED + FLAME_SHARED_UNITS) * per_expert
        active = attn + norms + embed + layer0_ffn \
            + (n_layer - 1) * (active_ffn_layer + router_layer) + oe_table_params
        # note: router params are included in active (always used), dispatch
        # overhead is excluded by definition of the active-FFN-matched boundary
    twin_width = flame_active_ffn_width_from(d_e)
    dense_twin_total = attn + norms + embed + layer0_ffn \
        + (n_layer - 1) * 3 * d * twin_width + oe_table_params
    return {
        "total": total,
        "active": active,
        "dense_twin_total": dense_twin_total,
        "router": router,
        "attn": attn, "norms": norms, "embed": embed, "layer0_ffn": layer0_ffn,
        "oe_table": oe_table_params,
        "active_ffn_width": twin_width,
    }


def flame_active_ffn_width_from(d_e: int) -> int:
    return (FLAME_K_ROUTED + FLAME_SHARED_UNITS) * d_e


def flame_name(counts: dict) -> str:
    """flame_{active}A_{total}T naming (never total-only)."""
    def fmt(x):
        if x >= 1e9:
            return f"{x/1e9:.2f}b".replace(".00", "").replace("0b", "b")
        return f"{x/1e6:.0f}m"
    return f"flame_{fmt(counts['active'])}A_{fmt(counts['total'])}T"


def format_flame_ledger(scale: str, counts: dict, dense_twin: bool) -> str:
    arm = "dense-twin" if dense_twin else "moe"
    lines = [
        f"[flame] profile {scale} ({arm}) {flame_name(counts)}",
        f"[flame] params: total {counts['total']:,} | active "
        f"{counts['active']:,} | dense-twin-total {counts['dense_twin_total']:,}",
        f"[flame] breakdown: attn {counts['attn']:,} | embed(untied) "
        f"{counts['embed']:,} | layer0-FFN {counts['layer0_ffn']:,} | router "
        f"{counts['router']:,} | OE table {counts['oe_table']:,}",
        f"[flame] active-FFN width {counts['active_ffn_width']} "
        f"(= (K{FLAME_K_ROUTED}+S{FLAME_SHARED_UNITS})*d_e); matched boundary "
        f"is active-FFN-matched (router/dispatch overhead excluded)",
    ]
    return "\n".join(lines)
