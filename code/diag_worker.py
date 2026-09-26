"""diag_worker.py · background CPU worker for n-gram diagnostics (torch-free math).

The main training process only runs the model forward (GPU) and ships small
numpy arrays (context keys + per-token losses) through a multiprocessing
queue; this worker does the searchsorted hit-count lookup and the per-exact-f
/ per-bucket aggregation off the training critical path.

Protocol (all messages are tuples):
  ("init", vocab_size, freq_index_path)
  ("job", step, epoch, probe_sha,
       train_pairs,   # list of (b_keys_np, t_keys_np, ptl_np) — fixed train-diag batches
       val_pairs,     # same shape — fixed val batches
       cur_pair)      # (b_keys_np, t_keys_np, ptl_np) or None — current training batch
  ("logit_job", step, epoch, branch, pairs, exemplar_pairs)
       pairs: list of (keys_np, hits_np, entropy_np, margin_np, ptrue_np)
       exemplar_pairs: list of (exemplar_id, keys_np, hits_np, top20_ids_np, top20_probs_np)
  None                # sentinel: finish pending jobs, then exit

Replies on the result queue:
  ("rows", step, epoch, probe_sha, exact_freq_row_payload, freq_bin_row_payload)
  ("logit_rows", step, epoch, branch, summary_dict, exemplar_dump_list)

The worker never touches torch/CUDA; ngram_freq is imported lazily so the
module stays importable in spawn contexts.
"""

import numpy as np


def _resolve_index(freq_index_path, inherited):
    if inherited is not None:
        return inherited
    from ngram_freq import GlobalFrequencyIndex
    return GlobalFrequencyIndex.load(freq_index_path)


def process_job(index, vocab_size, train_pairs, val_exact_pairs, val_fb_pairs, cur_pair):
    """Aggregate one eval block. Returns (exact_freq_payload, freq_bin_payload).

    exact_freq_payload: {"train": {branch: summary}, "val": {branch: summary},
                         "shared": {branch: shared_dict}}
    freq_bin_payload:   {"train": {branch: bucket_summary} or {},
                         "val": {branch: bucket_summary}}

    val_exact_pairs feed the exact-f val marginals + shared contexts;
    val_fb_pairs feed the freq-bin val buckets (both are prefixes of the same
    forwarded batch list, so no extra forwards are needed).
    """
    from ngram_freq import ExactFreqLossAccumulator, FreqBinLossAccumulator, compute_shared_from_keys

    exact = {"train": {}, "val": {}, "shared": {}}
    fb = {"train": {}, "val": {}}
    for branch, ki in (("bigram", 0), ("trigram", 1)):
        acc_tr = ExactFreqLossAccumulator(index, vocab_size, branch)
        tr_concat_k, tr_concat_l = [], []
        for pair in train_pairs:
            k = np.asarray(pair[ki]); l = np.asarray(pair[2], dtype=np.float64)
            acc_tr.update_numpy(k, l)
            tr_concat_k.append(k); tr_concat_l.append(l)
        acc_va = ExactFreqLossAccumulator(index, vocab_size, branch)
        va_concat_k, va_concat_l = [], []
        for pair in val_exact_pairs:
            k = np.asarray(pair[ki]); l = np.asarray(pair[2], dtype=np.float64)
            acc_va.update_numpy(k, l)
            va_concat_k.append(k); va_concat_l.append(l)
        exact["train"][branch] = acc_tr.summary()
        exact["val"][branch] = acc_va.summary()
        if tr_concat_k and va_concat_k:
            exact["shared"][branch] = compute_shared_from_keys(
                np.concatenate(tr_concat_k), np.concatenate(tr_concat_l),
                np.concatenate(va_concat_k), np.concatenate(va_concat_l),
                index, branch)
        else:
            exact["shared"][branch] = {"shared_total": 0, "per_f": {}, "branch": branch}

        acc_fb_va = FreqBinLossAccumulator(index, vocab_size, branch)
        for pair in val_fb_pairs:
            hits = index.hit_count_numpy(branch, np.asarray(pair[ki]))
            acc_fb_va.update_numpy(hits, pair[2])
        fb["val"][branch] = acc_fb_va.summary()

        if cur_pair is not None:
            acc_fb_tr = FreqBinLossAccumulator(index, vocab_size, branch)
            hits = index.hit_count_numpy(branch, np.asarray(cur_pair[ki]))
            acc_fb_tr.update_numpy(hits, cur_pair[2])
            fb["train"][branch] = acc_fb_tr.summary()
    return exact, fb


def worker_loop(job_q, res_q, freq_index_path=None, inherited_index=None):
    index = None
    vocab_size = None
    try:
        while True:
            msg = job_q.get()
            if msg is None:
                break
            kind = msg[0]
            if kind == "init":
                _, vocab_size, path = msg
                index = _resolve_index(path or freq_index_path, inherited_index)
                res_q.put(("ready",))
                continue
            if kind != "job":
                if kind == "logit_job":
                    _, step, epoch, branch, pairs, exemplar_pairs = msg
                    if index is None:
                        raise RuntimeError("diag_worker received logit_job before init")
                    from ngram_freq import LogitStatsAccumulator
                    acc = LogitStatsAccumulator(branch)
                    for keys_np, hits_np, entropy_np, margin_np, ptrue_np in pairs:
                        acc.update_numpy(keys_np, hits_np, entropy_np, margin_np, ptrue_np)
                    summary = acc.summary()
                    # exemplar dump: each entry is a small dict suitable for JSONL
                    exemplar_dump = []
                    for eid, ek, eh, et_ids, et_probs in (exemplar_pairs or []):
                        exemplar_dump.append({
                            "exemplar_id": int(eid),
                            "branch": branch,
                            "hit_count": int(np.asarray(eh).ravel()[0]) if np.asarray(eh).size > 0 else 0,
                            "top20_ids": [int(x) for x in np.asarray(et_ids).ravel()[:20]],
                            "top20_probs": [float(x) for x in np.asarray(et_probs).ravel()[:20]],
                        })
                    res_q.put(("logit_rows", step, epoch, branch, summary, exemplar_dump))
                    continue
                continue
            _, step, epoch, probe_sha, train_pairs, val_exact_pairs, val_fb_pairs, cur_pair = msg
            if index is None:
                raise RuntimeError("diag_worker received job before init")
            exact, fb = process_job(index, vocab_size, train_pairs,
                                    val_exact_pairs, val_fb_pairs, cur_pair)
            res_q.put(("rows", step, epoch, probe_sha, exact, fb))
    except Exception as e:  # surface worker failures without killing training
        try:
            import traceback
            res_q.put(("error", repr(e), traceback.format_exc()))
        except Exception:
            pass


GLOBAL_INDEX = None  # set by train.py before fork() so the child inherits it


class DiagPool:
    """Main-process handle for the background diagnostics worker."""

    def __init__(self, freq_index_path: str, vocab_size: int, inherited_index=None):
        import multiprocessing as mp
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork") if "fork" in methods else mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=8)
        self.res_q = ctx.Queue()
        self.proc = ctx.Process(target=worker_loop,
                                args=(self.job_q, self.res_q, freq_index_path, inherited_index),
                                daemon=True)
        self.proc.start()
        self.job_q.put(("init", vocab_size, freq_index_path))
        self.alive = True

    def submit(self, step, epoch, probe_sha, train_pairs, val_exact_pairs, val_fb_pairs, cur_pair):
        if self.alive:
            self.job_q.put(("job", step, epoch, probe_sha,
                            train_pairs, val_exact_pairs, val_fb_pairs, cur_pair))

    def submit_logit(self, step, epoch, branch, pairs, exemplar_pairs):
        """Submit a logit-stats aggregation job.

        pairs: list of (keys_np, hits_np, entropy_np, margin_np, ptrue_np)
        exemplar_pairs: list of (exemplar_id, keys_np, hits_np, top20_ids_np, top20_probs_np)
        """
        if self.alive:
            self.job_q.put(("logit_job", step, epoch, branch, pairs, exemplar_pairs))

    def drain(self, on_rows, on_logit_rows=None):
        """Non-blocking drain of finished rows. on_rows(step, epoch, sha, exact, fb)."""
        if not self.alive:
            return
        while True:
            try:
                msg = self.res_q.get_nowait()
            except Exception:
                return
            kind = msg[0]
            if kind == "rows":
                _, step, epoch, sha, exact, fb = msg
                on_rows(step, epoch, sha, exact, fb)
            elif kind == "logit_rows":
                _, step, epoch, branch, summary, exemplar_dump = msg
                if on_logit_rows is not None:
                    on_logit_rows(step, epoch, branch, summary, exemplar_dump)
            elif kind == "ready":
                continue
            elif kind == "error":
                print(f"[nglab] diag worker FAILED: {msg[1]}\n{msg[2]}", flush=True)
                self.alive = False
                return

    def close_and_drain(self, on_rows, on_logit_rows=None):
        """Sentinel, join, then blocking drain of any remaining rows."""
        if self.alive:
            try:
                self.job_q.put(None)
                self.proc.join(timeout=300)
            except Exception:
                pass
        while True:
            try:
                msg = self.res_q.get(timeout=5)
            except Exception:
                return
            if msg[0] == "rows":
                _, step, epoch, sha, exact, fb = msg
                on_rows(step, epoch, sha, exact, fb)
            elif msg[0] == "logit_rows":
                _, step, epoch, branch, summary, exemplar_dump = msg
                if on_logit_rows is not None:
                    on_logit_rows(step, epoch, branch, summary, exemplar_dump)
            elif msg[0] == "error":
                print(f"[nglab] diag worker FAILED: {msg[1]}\n{msg[2]}", flush=True)
                return


def make_pair_payloads(batches, model, amp_dtype, vocab_size, device_dtype_ctx=None):
    """Main-process helper: one forward per batch (bf16 autocast), keys on GPU.

    Returns list of (b_keys_np, t_keys_np, ptl_np) moved to CPU numpy.
    Kept here so sync/async paths share one implementation. Needs torch —
    only called from the training process.
    """
    import torch
    from ngram_freq import compute_context_keys, compute_per_token_loss
    out = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for inp, tgt in batches:
            ptl = compute_per_token_loss(model, inp, tgt, amp_dtype=amp_dtype)
            b_keys, t_keys = compute_context_keys(inp, vocab_size)
            out.append((b_keys.cpu().numpy().ravel(),
                        t_keys.cpu().numpy().ravel(),
                        ptl.detach().cpu().numpy().ravel()))
    if was_training:
        model.train()
    return out


def make_logit_payloads(batches, model, amp_dtype, vocab_size, freq_index, branch):
    """One bf16 forward per batch; returns per-token confidence stats for logit probe.

    Returns list of tuples per batch:
      (keys_np, hits_np, entropy_np, margin_np, ptrue_np,
       top20_ids_np, top20_probs_np)
    All arrays are CPU numpy, flattened to 1-D (keys/hits/entropy/margin/ptrue)
    or 2-D (top20: shape (N, 20)).

    This is a SEPARATE forward from make_pair_payloads to keep the existing
    exact-freq / freq-bin pipeline byte-identical when --logit_stats is off.
    When --logit_stats is on, the caller invokes this on the same val batches
    used by the standard diag block; the extra cost is ~4 bf16 forwards per
    eval block (~0.3-0.5s on H200).
    """
    import torch
    import torch.nn.functional as F
    from ngram_freq import compute_context_keys
    out = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for inp, tgt in batches:
            if amp_dtype is not None and inp.is_cuda:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(inp)
            else:
                logits = model(inp)
            # logits: (B, T, V) in bf16/fp32 depending on autocast
            logits_f = logits.float()
            probs = F.softmax(logits_f, dim=-1)
            # entropy H(p) = -sum p log p (nats)
            log_probs = F.log_softmax(logits_f, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
            # top-2 probs + margin
            top2_probs, _ = torch.topk(probs, k=2, dim=-1)
            margin = top2_probs[..., 0] - top2_probs[..., 1]  # (B, T)
            # p(y_true)
            tgt_clamped = tgt.clamp(min=0)
            ptrue = probs.gather(-1, tgt_clamped.unsqueeze(-1)).squeeze(-1)  # (B, T)
            # top-20 ids + probs
            top20_probs, top20_ids = torch.topk(probs, k=20, dim=-1)  # (B, T, 20)
            # context keys + hit counts
            b_keys, t_keys = compute_context_keys(inp, vocab_size)
            keys = b_keys if branch == "bigram" else t_keys
            hits = freq_index.hit_count_tensor(branch, keys)
            out.append((
                keys.cpu().numpy().ravel(),
                hits.cpu().numpy().ravel(),
                entropy.cpu().numpy().ravel(),
                margin.cpu().numpy().ravel(),
                ptrue.cpu().numpy().ravel(),
                top20_ids.cpu().numpy().reshape(-1, 20),
                top20_probs.cpu().numpy().reshape(-1, 20),
            ))
    if was_training:
        model.train()
    return out
