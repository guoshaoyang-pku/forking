"""Table row cosine trajectory + seen/novel/margin decomposition (GPU, single run).

Replays an authoritative 128x seed-42 run step-for-step (same data order, same
optimizer, same warmup_constant schedule) so table snapshots are the real
trajectory, not a re-fit. At every pass boundary (337 device steps at 1x dose):

  * snapshot bigram+trigram clean-table weights (fp16 -> .npy, ~3.2GB each per
    branch; only boundaries are kept)
  * run a seen/novel/margin diagnostic on the fixed val pool:
      for each val token: hit count f of its context, whether the true next
      token was EVER seen with that context in the 1x train shard (offline,
      from freq_index trigram keys), per-token loss of the trained model and
      of the ngram-disabled variant, plus margin
          logit(train-dominant continuation) - logit(true novel token)
      aggregated per exact-f bin into:
        seen/novel token counts, seen-loss sum, novel-loss sum,
        margin mean/median for seen and novel tokens.

Runs: input (nglab1x_input_v5_128x_freq10_fd), nogram, freeze_backbone_e1
(causalv5m3). Output: data/runs_theory/<run>_snap/
  rows_{bigram,trigram}_step<step>.npy, decomp.jsonl
Post-hoc (CPU, this dir): cos/norm ratios per f-bin from boundary snapshots.

NOTE: cosine at adjacent boundaries is one optimizer step apart by construction
(cos ~ 1 is a tautology there); the meaningful statistic is cos over a WHOLE
pass and the per-f-bin write intensity (row delta norm per pass).
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import torch
from train import (NanoGPT, Config, TokenizedShardDataset, MixedOptimizer,
                   get_lr_multiplier, set_seed, _autocast_ctx)
import ngram_freq as NF

P = 337
RUNS = {
    "input":    dict(run_id="nglab1x_input_v5_128x_freq10_fd", extra={}),
    "nogram":   dict(run_id="nglab1x_nogram_v5_128x_freq10_fd",
                     extra=dict(enable_bigram_ve=False, enable_trigram_ve=False)),
    "freezebb": dict(run_id="causalv5m3_freeze_backbone_e1",
                     extra=dict(intervention="freeze_backbone", intervention_epoch=1)),
}


def build_cfg(spec):
    c = Config(
        n_layer=8, n_head=6, n_embd=768, vocab_size=8192, sequence_len=2048,
        nanogpt_ngram_injection_position="input",
        nanogpt_adam_lr=0.0006, adam_betas=(0.8, 0.95), weight_decay=0.1,
        table_optimizer="rmsprop", table_betas=(0.0, 0.99), table_lr_scale=128.0,
        bigram_clean_table=1 << 20, trigram_clean_table=1 << 20,
        enable_unigram_ve=False,
        enable_bigram_ve=True, enable_trigram_ve=True,
        seed=42, max_steps=2022, device_batch_size=72, total_batch_size=147456,
        warmup_steps=100, warmup_start_lr_mult=0.25, lr_schedule="warmup_constant",
        data_dir=str(ROOT / "data/tokenized"),
        train_shards=[1], val_shards=[2, 3, 4, 5, 6, 7, 8, 9, 10, 6542],
        out_dir="",
    )
    for k, v in spec["extra"].items():
        setattr(c, k, v)
    return c


class ZeroNgram(torch.nn.Module):
    """model.forward with n-gram residual zeroed (clean comparison model)."""
    def __init__(self, model):
        super().__init__()
        self.m = model
    def forward(self, idx, targets=None):
        orig = self.m._compute_input_ngram_residual
        def zero(idx_):
            return torch.zeros_like(orig(idx_))
        self.m._compute_input_ngram_residual = zero
        try:
            return self.m(idx, targets=targets)
        finally:
            self.m._compute_input_ngram_residual = orig


@torch.no_grad()
def decomp_diag(model, clean, val_batches, seen_ctx_tokens, cont_rows, index,
                vocab_size, device, amp_dtype, has_table):
    """Per-token val decomposition. Returns list of per-f dicts (bigram branch).

    seen_ctx_tokens: set-like via sorted np array of bigram keys (c1*V+c2) that
    appear in the train shard.  cont_rows: dict c3_id(int) -> np int64 array of
    trigram keys, used for seen/novel membership via bigram key + token key.
    Margin uses the train-dominant continuation = argmax count within the
    context's row (computed offline into DOM map keyed by bigram key).
    """
    from ngram_freq import compute_context_keys, compute_per_token_loss
    model.eval()
    V = vocab_size
    acc = {}  # f -> [n_seen, n_novel, loss_sum_seen_t, loss_sum_novel_t,
              #       loss_sum_seen_c, loss_sum_novel_c,
              #       margin_seen_sum, margin_seen_sq, margin_novel_sum, margin_novel_sq, n_margin_seen, n_margin_novel]
    for inp, tgt in val_batches:
        ptl_t = compute_per_token_loss(model, inp, tgt, amp_dtype=amp_dtype)
        ptl_c = (compute_per_token_loss(clean, inp, tgt, amp_dtype=amp_dtype)
                 if has_table else ptl_t.clone())
        b_keys, _ = compute_context_keys(inp, V)          # (B*T,) bigram keys
        logits = model(inp, targets=None)                  # (B,T,V) fp32
        lt = logits.reshape(-1, V)
        nxt = tgt.reshape(-1).long()
        bkn = (b_keys.cpu().numpy() if torch.is_tensor(b_keys) else np.asarray(b_keys)).reshape(-1)
        f_np = index.hit_count_numpy("bigram", bkn).reshape(-1)
        nxt_np = nxt.cpu().numpy()
        tri_key = bkn.astype(np.int64) * V + nxt_np
        allk = cont_rows["all_tri_keys"]
        pos = np.searchsorted(allk, tri_key)
        pos = np.clip(pos, 0, len(allk) - 1)
        seen = allk[pos] == tri_key
        dom = cont_rows["dom_map"].get
        ptl_t_np = ptl_t.cpu().numpy().reshape(-1); ptl_c_np = ptl_c.cpu().numpy().reshape(-1)
        lt_np = lt.float().cpu().numpy(); del lt, logits; torch.cuda.empty_cache()
        dom_ids = np.array([dom(int(k), -1) for k in bkn])
        margin = np.where(dom_ids >= 0,
                          lt_np[np.arange(len(lt_np)), np.clip(dom_ids, 0, V - 1)] -
                          lt_np[np.arange(len(lt_np)), nxt_np], np.nan)
        for fi in np.unique(f_np):
            m = f_np == fi
            ms, mn = m & seen, m & (~seen)
            a = acc.setdefault(int(fi), np.zeros(12))
            a[0] += ms.sum(); a[1] += mn.sum()
            a[2] += ptl_t_np[ms].sum(); a[3] += ptl_t_np[mn].sum()
            a[4] += ptl_c_np[ms].sum(); a[5] += ptl_c_np[mn].sum()
            mg_s = margin[ms]; mg_s = mg_s[~np.isnan(mg_s)]
            mg_n = margin[mn]; mg_n = mg_n[~np.isnan(mg_n)]
            a[6] += mg_s.sum(); a[7] += (mg_s ** 2).sum(); a[10] += len(mg_s)
            a[8] += mg_n.sum(); a[9] += (mg_n ** 2).sum(); a[11] += len(mg_n)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS), required=True)
    ap.add_argument("--out_root", default=str(ROOT / "data/runs_theory"))
    ap.add_argument("--val_batches", type=int, default=4)
    args = ap.parse_args()
    spec = RUNS[args.run]
    out = Path(args.out_root) / f"{args.run}_snap"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = build_cfg(spec)
    set_seed(cfg.seed)

    print(f"[snap] run={args.run} -> {out}", flush=True)
    train_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, cfg.sequence_len,
                                     cfg.device_batch_size, cfg.seed)
    val_ds = TokenizedShardDataset(cfg.data_dir, cfg.val_shards, cfg.sequence_len,
                                   cfg.device_batch_size, cfg.seed)
    val_iter = val_ds.iter_batches(device)
    fixed_val = [next(val_iter) for _ in range(args.val_batches)]

    # offline seen/novel + dominant-continuation maps from freq_index (CPU)
    print("[snap] building offline continuation maps ...", flush=True)
    z = np.load(ROOT / "data/freq_index.npz"); V = int(z["vocab_size"][0])
    tk = z["trigram_keys"]
    cont_rows = {"all_tri_keys": np.sort(tk)}
    # dominant continuation per bigram context: group trigram keys by context
    ctx = tk // V
    order = np.argsort(ctx, kind="stable")
    ctx_s, tri_s = ctx[order], tk[order]
    cnt_s = z["trigram_counts"][order]
    uctx, start = np.unique(ctx_s, return_index=True)
    start = np.append(start, len(ctx_s))
    dom_map = {}
    for i in range(len(uctx)):
        seg = slice(start[i], start[i + 1])
        j = np.argmax(cnt_s[seg])
        dom_map[int(uctx[i])] = int(tri_s[seg][j] % V)
    cont_rows["dom_map"] = dom_map

    index = NF.GlobalFrequencyIndex.load(str(ROOT / "data/freq_index.npz"))
    model = NanoGPT(cfg).to(device)
    model.init_weights()
    clean = ZeroNgram(model) if (cfg.enable_bigram_ve or cfg.enable_trigram_ve) else None
    optimizer = MixedOptimizer(model, lr=cfg.nanogpt_adam_lr,
                               ngram_betas=cfg.ngram_table_betas,
                               adam_betas=cfg.adam_betas, weight_decay=cfg.weight_decay,
                               table_optimizer=cfg.table_optimizer,
                               table_lr_scale=cfg.table_lr_scale, table_betas=cfg.table_betas)
    amp_ctx = _autocast_ctx("bf16")
    amp_dtype = torch.bfloat16
    has_table = cfg.enable_bigram_ve or cfg.enable_trigram_ve
    decomp_f = open(out / "decomp.jsonl", "w")
    grad_accum = 1
    intervention_due = frozenset({cfg.intervention_epoch} | set(cfg.intervention_epochs)) \
        if cfg.intervention != "none" else frozenset()
    fired = set()
    train_iter = train_ds.iter_batches(device)

    def snapshot_rows(step):
        for name, ves in (("bigram", model.bigram_ves), ("trigram", model.trigram_ves)):
            for li in sorted(ves.keys()):
                w = ves[li][0].weight.detach().half().cpu().numpy()
                np.save(out / f"rows_{name}_step{step}.npy", w)

    def run_decomp(step):
        if not has_table and args.run != "nogram":
            return
        acc = decomp_diag(model, clean, fixed_val, None, cont_rows, index,
                          cfg.vocab_size, device, amp_dtype, has_table and args.run != "nogram")
        for fi, a in acc.items():
            rec = dict(step=step, f=int(fi), n_seen=int(a[0]), n_novel=int(a[1]),
                       loss_seen_t=a[2], loss_novel_t=a[3],
                       loss_seen_c=a[4], loss_novel_c=a[5],
                       marg_seen_sum=a[6], marg_seen_sq=a[7],
                       marg_novel_sum=a[8], marg_novel_sq=a[9],
                       marg_seen_n=int(a[10]), marg_novel_n=int(a[11]))
            decomp_f.write(json.dumps(rec) + "\n")
        decomp_f.flush()
        print(f"[snap] step {step}: decomp for {len(acc)} f-values", flush=True)

    for step in range(cfg.max_steps):
        optimizer.zero_grad()
        accum = 0.0
        for micro in range(grad_accum):
            try:
                inp, tgt = next(train_iter)
            except StopIteration:
                train_iter = train_ds.iter_batches(device)
                inp, tgt = next(train_iter)
            if (train_ds._epoch in intervention_due and train_ds._epoch not in fired):
                model.apply_intervention(train_ds._epoch, index)
                fired.add(train_ds._epoch)
                print(f"[snap] intervention {cfg.intervention} fired at step {step+1}", flush=True)
            with amp_ctx:
                loss = model(inp, targets=tgt) / grad_accum
            if loss.requires_grad:
                loss.backward()
            accum += loss.item()
        lr_mult = get_lr_multiplier((step + 1) / cfg.max_steps, schedule=cfg.lr_schedule,
                                    warmup_steps=cfg.warmup_steps, step=step + 1,
                                    max_steps=cfg.max_steps,
                                    warmup_start_lr_mult=cfg.warmup_start_lr_mult)
        optimizer.step(lr_mult=lr_mult)
        s = step + 1
        if s % P == 0 or s == cfg.max_steps:
            if has_table:
                snapshot_rows(s)
            run_decomp(s)
            print(f"[snap] step {s}/{cfg.max_steps} train={accum:.4f}", flush=True)
        elif s % 50 == 0:
            print(f"[snap] step {s} train={accum:.4f}", flush=True)
    decomp_f.close()
    print("[snap] done", flush=True)


if __name__ == "__main__":
    main()
