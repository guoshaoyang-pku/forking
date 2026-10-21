#!/usr/bin/env python3
"""Teacher-residual distillation for n-gram table init / clustering (pro5k6).

Streams ClimbMix raw text through Qwen3.6-35B-A3B and accumulates, per n-gram
context present in an assignment table, the mean teacher residual stream at the
last token of every occurrence of that context (the state after the context has
been read, i.e. what position t feeds forward when the row is injected at t).

Lab stream (matches data/tokenized packing): documents are BOS-prefixed units
packed losslessly into rows of exactly --seq tokens, so a teacher forward always
sees a full-length row.  Every row is one teacher sequence of ~0.86*seq tokens;
packing is what makes the engine efficient (a forward is dominated by weight
streaming, so rows must be long).

Alignment: our token t takes the last teacher token whose cumulative byte end is
<= our cumulative byte end.  Byte units, so it is valid for arbitrary UTF-8.

Keys: code/ngram_freq.context_keys_np (in-chunk shift rule); only contexts present
in --assign (e.g. assign_exact974k.npz) are accumulated.

Output: one npz part per worker, mergeable with merge_distill_parts.py.
  keys_<n>        (R_n,)      int64  assignment keys of order n
  sum_L<d>_<n>    (R_n, hid)  fp32   residual sums at teacher depth d
  cnt_<n>         (R_n,)      int64  occurrence counts
  meta_json       scalar      str

Depths follow HF hidden_states indexing: 0 = embedding output, k>=1 = residual
stream after k decoder blocks; the optional final block is the pre-logits normed
output.  vLLM's token_embed pooler L2-normalizes rows, so oe_cap_patch appends a
sentinel column of ones that lets the driver undo it exactly (see denorm()).
"""
import argparse
import json
import os
import pickle
import signal
import sys
import time

import numpy as np

SEQ_DEFAULT = 2048
BOS_DEFAULT = 8188
VOCAB_DEFAULT = 8192


# --------------------------------------------------------------------------- #
# tokenizer helpers
# --------------------------------------------------------------------------- #
def _bytes_to_unicode():
    """GPT-2/Qwen byte-level BPE alphabet: byte value -> printable unicode char."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def qwen_token_byte_table(model_dir):
    """(tokenizer, per-teacher-token byte length) for the Qwen BPE."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    u2b = {v: k for k, v in _bytes_to_unicode().items()}
    size = max(len(tok), getattr(tok, "vocab_size", 0)) + 1
    tb = np.zeros(size, dtype=np.int32)
    for i in range(len(tok)):
        s = tok.convert_ids_to_tokens(i)
        if isinstance(s, bytes):
            tb[i] = len(s)
            continue
        c = 0
        for ch in s:
            if ch in u2b:
                c += 1
        tb[i] = c
    return tok, tb


def lab_token_byte_table(tok_bytes_npz, lab_enc, vocab):
    """Per-lab-token byte length from tok_bytes.npz (specials -> 0)."""
    tb = np.zeros(vocab + 1, dtype=np.int32)
    if tok_bytes_npz and os.path.exists(tok_bytes_npz):
        z = np.load(tok_bytes_npz)
        off = z["off"]
        for i in range(vocab):
            tb[i] = int(off[i + 1] - off[i])
    else:  # fallback: decode each token
        for i in range(vocab):
            try:
                tb[i] = len(lab_enc.decode_single_token_bytes(i))
            except Exception:
                tb[i] = 0
    return tb


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #
class Packer:
    """BOS-prefixed document units packed losslessly into --seq-token rows."""

    def __init__(self, lab_enc, lab_tb, seq, bos):
        self.enc = lab_enc
        self.tb = lab_tb
        self.seq = seq
        self.bos = bos
        self.reset()

    def reset(self):
        self.ids = []      # current row: lab token ids
        self.parts = []    # raw byte chunks; concatenation == row text
        self.rows = 0

    def feed(self, text):
        """Add one document; yield completed rows as (ids, text, byte_ends)."""
        ids = self.enc.encode(text)
        if not ids:
            return
        if len(ids) > self.seq - 1:              # a single doc longer than a row
            ids = ids[: self.seq - 1]
        raw = text.encode("utf-8")
        nb_total = int(np.sum(self.tb[np.asarray(ids, dtype=np.int64)]))
        raw = raw[:nb_total] if nb_total < len(raw) else raw
        pos = 0                                   # byte cursor into this doc
        while True:
            if not self.ids:                      # start a new unit
                self.ids.append(self.bos)
            space = self.seq - len(self.ids)
            if space <= 0:
                yield self._emit()
                continue
            take = min(len(ids), space)
            if take:
                chunk = ids[:take]
                ids = ids[take:]
                nb = int(np.sum(self.tb[np.asarray(chunk, dtype=np.int64)]))
                self.ids.extend(chunk)
                if nb:
                    self.parts.append(raw[pos: pos + nb])
                    pos += nb
            if not ids:
                return
            if len(self.ids) >= self.seq:         # row exactly full
                yield self._emit()

    def flush(self):
        """Emit a final partial row (used when a shard ends mid-row)."""
        if len(self.ids) > 1:
            return self._emit()
        return None

    def _emit(self):
        row_ids = np.asarray(self.ids, dtype=np.int64)
        be = np.cumsum(self.tb[row_ids])
        text = b"".join(self.parts).decode("utf-8", errors="ignore")
        self.rows += 1
        self.reset()
        return row_ids, text, be


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
class Worker:
    def __init__(self, args):
        self.args = args
        self.orders = [int(x) for x in args.orders.split(",") if x.strip()]
        self.depths = [int(x) for x in args.depths.split(",") if x.strip()]

        import torch

        self.torch = torch
        self.dev = args.device

        with open(args.our_tokenizer, "rb") as fh:
            self.lab_enc = pickle.load(fh)
        self.lab_tb = lab_token_byte_table(args.tok_bytes, self.lab_enc, args.vocab)
        self.qwen_tok, self.qwen_tb = qwen_token_byte_table(args.model)

        z = np.load(args.assign)
        self.assign_keys = {}
        self.sizes = {}
        for n in self.orders:
            k = np.ascontiguousarray(z[f"keys_{n}"], dtype=np.int64)
            self.assign_keys[n] = k
            self.sizes[n] = int(k.size)

        hid = args.hid
        self.hid = hid
        self.use_final = not args.no_final_block
        self.nblocks = len(self.depths) + (1 if self.use_final else 0)
        self.sentinel = not args.no_sentinel
        dim = hid * self.nblocks

        self.acc = {n: torch.zeros(self.sizes[n], dim, dtype=torch.float32, device=self.dev)
                    for n in self.orders}
        self.cnt = {n: torch.zeros(self.sizes[n], dtype=torch.int64, device=self.dev)
                    for n in self.orders}

        self.tokens_done = 0
        self.docs_done = 0
        self.rows_done = 0
        self.positions = 0
        self.hits = {n: 0 for n in self.orders}
        self.align_valid = 0
        self.align_total = 0
        self.shards_done = []
        self.t_first_batch = None
        self._diag_align_done = False
        self._diag_blocks_done = False
        self.t0 = time.time()

    # -- normalization / accumulation --------------------------------------- #
    def denorm(self, data):
        """Undo vLLM's per-row L2 normalization using the trailing ones sentinel."""
        want = self.hid * self.nblocks
        if data.shape[1] == want + 1:
            return data[:, :-1] * (1.0 / data[:, -1])[:, None]
        if self.sentinel:
            raise RuntimeError(
                f"expected sentinel column: got width {data.shape[1]}, want {want + 1} "
                f"(is site-packages/oe_cap_patch.py up to date / OE_CAP_LAYERS set?)")
        return data

    def accumulate(self, row_ids, q_ids, be, data):
        torch = self.torch
        if data.shape[0] != len(q_ids):
            raise RuntimeError(f"teacher tokens {data.shape[0]} != tokenizer {len(q_ids)}")
        q_be = np.cumsum(np.asarray(self.qwen_tb, dtype=np.int64)[np.asarray(q_ids)])
        # our position t takes the last teacher token ending at or before byte be[t]
        q_star = np.searchsorted(q_be, be, side="right") - 1
        valid = q_star >= 0
        self.align_valid += int(valid.sum())
        self.align_total += int(valid.size)

        from ngram_freq import context_keys_np

        row_map = {}
        for n in self.orders:
            keys = context_keys_np(row_ids, n, self.args.vocab)[0]
            akeys = self.assign_keys[n]
            pos = np.searchsorted(akeys, keys, side="left")
            pos = np.where((pos < akeys.size)
                           & (akeys[np.clip(pos, 0, akeys.size - 1)] == keys), pos, -1)
            pos[~valid] = -1
            row_map[n] = pos

        if not valid.any():
            return 0
        t_sel = np.flatnonzero(valid)
        vals = np.ascontiguousarray(data[q_star[t_sel]])
        self.positions += int(t_sel.size)
        for n in self.orders:
            sel = row_map[n][t_sel]
            hit = sel >= 0
            self.hits[n] += int(hit.sum())
            if not hit.any():
                continue
            idx = torch.from_numpy(sel[hit].astype(np.int64))
            self.acc[n].index_add_(0, idx, torch.from_numpy(vals[hit]))
            self.cnt[n].index_add_(
                0, idx, torch.ones(idx.numel(), dtype=torch.int64, device=self.dev))
        return int(t_sel.size)

    # -- diagnostics -------------------------------------------------------- #
    def diag_align(self, q_ids, be):
        if self._diag_align_done:
            return
        self._diag_align_done = True
        q_be = np.cumsum(np.asarray(self.qwen_tb, dtype=np.int64)[np.asarray(q_ids)])
        print(f"[diag] teacher_tokens={len(q_ids)} lab_tokens={be.size} "
              f"ratio={len(q_ids)/be.size:.3f} "
              f"byte_end_match={int(be[-1]) == int(q_be[-1])}", flush=True)

    def diag_blocks(self, data, q_ids):
        if self._diag_blocks_done:
            return
        self._diag_blocks_done = True
        hid, nb = self.hid, self.nblocks
        blk = data.reshape(data.shape[0], nb, hid).astype(np.float32)
        names = (["final_normed"] if self.use_final else []) + [f"d{d}" for d in self.depths]
        rms = [float(np.sqrt((blk[:, i] ** 2).mean())) for i in range(nb)]
        print("[diag] block RMS (HF ref d0=0.0105 d10=0.0516 d25=0.1079 final=2.494) "
              + "  ".join(f"{n}={r:.6g}" for n, r in zip(names, rms)), flush=True)
        if self.use_final and self.depths and self.depths[-1] > 0:
            fin, last = blk[:, 0], blk[:, -1]
            cos = (fin * last).sum(1) / (np.linalg.norm(fin, axis=1)
                                         * np.linalg.norm(last, axis=1) + 1e-9)
            print(f"[diag] cos(final_normed, d{self.depths[-1]})={cos.mean():.4f}", flush=True)

    # -- encode ------------------------------------------------------------- #
    def _encode(self, qwen_ids_list):
        prompts = [{"prompt_token_ids": list(map(int, ids))} for ids in qwen_ids_list]
        return self.llm.encode(prompts, pooling_task="token_embed")

    def _encode_fake(self, qwen_ids_list):
        """Preflight stand-in: right shapes, deterministic values, engine contract."""

        class _Out:
            def __init__(self, data):
                self.outputs = type("O", (), {"data": data})()

        rng = np.random.default_rng(1234)
        dim = self.hid * self.nblocks
        out = []
        for ids in qwen_ids_list:
            blk = rng.standard_normal((len(ids), dim)).astype(np.float32)
            w = np.concatenate([blk, np.ones((len(ids), 1), dtype=np.float32)], axis=1)
            out.append(_Out(w / np.linalg.norm(w, axis=1, keepdims=True)))
        return out

    # -- shard/row iteration ------------------------------------------------ #
    def rows_for_shard(self, path):
        """Yield (row_ids, teacher_text, byte_ends) for one parquet shard."""
        import pyarrow.parquet as pq

        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        packer = Packer(self.lab_enc, self.lab_tb, self.args.seq, self.args.bos)
        for t in texts:
            self.docs_done += 1
            for row in packer.feed(t):
                yield row
        last = packer.flush()
        if last is not None:
            yield last

    def teacher_ids(self, text):
        ids = self.qwen_tok(text, add_special_tokens=False)["input_ids"]
        lim = self.args.max_model_len - 8
        guard = 0
        while len(ids) > lim and guard < 4:      # extreme safety; rows ~0.86*seq
            b = text.encode("utf-8")
            text = b[: int(len(b) * 0.9)].decode("utf-8", errors="ignore")
            ids = self.qwen_tok(text, add_special_tokens=False)["input_ids"]
            guard += 1
        return ids

    # -- main loop ---------------------------------------------------------- #
    def shard_list(self):
        a = self.args
        return list(range(a.shard_start + a.worker, a.shard_end, a.num_workers))

    def run(self):
        a = self.args
        shards = self.shard_list()
        print(f"[worker {a.worker}] {len(shards)} shards budget={a.token_budget/1e9:.2f}B "
              f"seq={a.seq} batch_rows={a.batch_rows} depths={self.depths} "
              f"final_block={self.use_final} device={self.dev}", flush=True)
        stop = False
        for si, sh in enumerate(shards):
            if stop:
                break
            path = os.path.join(a.raw_dir, f"shard_{sh:05d}.parquet")
            if not os.path.exists(path):
                print(f"[worker {a.worker}] missing {path}", flush=True)
                continue
            pending = []
            for row in self.rows_for_shard(path):
                pending.append(row)
                if len(pending) < a.batch_rows:
                    continue
                self.process_rows(pending)
                pending = []
                if a.max_docs and self.docs_done >= a.max_docs:
                    stop = True
                    break
                if self.tokens_done >= a.token_budget:
                    stop = True
                    break
                if self.rows_done % (a.log_every * a.batch_rows) < a.batch_rows:
                    el = time.time() - self.t0
                    work = time.time() - (self.t_first_batch or self.t0)
                    rate = self.tokens_done / work if self.t_first_batch and work > 0 else 0.0
                    hr = " ".join(f"n{n}={self.hits[n]/max(self.positions,1)*100:.1f}%"
                                  for n in self.orders)
                    print(f"[worker {a.worker}] shard {sh} rows={self.rows_done} "
                          f"tok={self.tokens_done/1e6:.2f}M rate={rate:.0f} lab tok/s "
                          f"elapsed={el/60:.1f}min hit_rate[{hr}] "
                          f"eta={(a.token_budget-self.tokens_done)/max(rate,1)/3600:.1f}h",
                          flush=True)
            if pending and not stop:
                self.process_rows(pending)
            self.shards_done.append(sh)
            print(f"[worker {a.worker}] shard {sh} done ({si+1}/{len(shards)}) "
                  f"tok={self.tokens_done/1e6:.2f}M", flush=True)
            if a.save_every_shards and len(self.shards_done) % a.save_every_shards == 0:
                self.save(final=False)
        self.save()

    def process_rows(self, rows):
        q_ids_list = [self.teacher_ids(text) for _, text, _ in rows]
        outs = self._encode(q_ids_list)
        if self.t_first_batch is None:
            self.t_first_batch = time.time()
        for (row_ids, _text, be), q_ids, o in zip(rows, q_ids_list, outs):
            data = self.denorm(np.asarray(o.outputs.data, dtype=np.float32))
            if not self._diag_blocks_done:
                self.diag_blocks(data, q_ids)
            if not self._diag_align_done:
                self.diag_align(q_ids, be)
            self.accumulate(row_ids, q_ids, be, data)
            self.tokens_done += int(row_ids.size)
            self.rows_done += 1

    # -- preflight ---------------------------------------------------------- #
    def preflight(self, n_docs=64):
        """CPU-only end-to-end check of pack/align/accumulate (no engine)."""
        import pyarrow.parquet as pq

        ok = True
        print(f"[preflight] orders={self.orders} depths={self.depths} "
              f"blocks={self.nblocks} rows={ {n: self.sizes[n] for n in self.orders} }",
              flush=True)
        path = os.path.join(self.args.raw_dir,
                            f"shard_{self.args.shard_start:05d}.parquet")
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()[:n_docs]
        packer = Packer(self.lab_enc, self.lab_tb, self.args.seq, self.args.bos)
        rows = []
        for t in texts:
            rows.extend(packer.feed(t))
        tail = packer.flush()
        if tail is not None:
            rows.append(tail)
        lens = [r[0].size for r in rows]
        print(f"[preflight] packed {len(rows)} rows from {len(texts)} docs; row len "
              f"min={min(lens)} max={max(lens)} (target {self.args.seq})", flush=True)
        if len(rows) > 1 and min(lens[:-1]) < self.args.seq:
            print("[preflight][FAIL] a non-final row came out short", flush=True)
            ok = False

        self._encode = self._encode_fake
        self.args.batch_rows = min(self.args.batch_rows, len(rows))
        for i in range(0, len(rows), self.args.batch_rows):
            chunk = rows[i: i + self.args.batch_rows]
            q_ids_list = [self.teacher_ids(text) for _, text, _ in chunk]
            outs = self._encode(q_ids_list)
            for (row_ids, _t, be), q_ids, o in zip(chunk, q_ids_list, outs):
                data = self.denorm(np.asarray(o.outputs.data, dtype=np.float32))
                want = self.hid * self.nblocks
                if data.shape[0] != len(q_ids) or data.shape[1] != want:
                    print(f"[preflight][FAIL] shape {data.shape} vs tokens {len(q_ids)} "
                          f"blocks {self.nblocks}", flush=True)
                    ok = False
                self.diag_blocks(data, q_ids)
                self.diag_align(q_ids, be)
                self.accumulate(row_ids, q_ids, be, data)
                self.tokens_done += int(row_ids.size)
                self.rows_done += 1

        for n in self.orders:
            if int(self.cnt[n].sum()) != self.hits[n]:
                print(f"[preflight][FAIL] n={n} cnt.sum()={int(self.cnt[n].sum())} "
                      f"!= hits={self.hits[n]}", flush=True)
                ok = False
        tot = sum(int(self.cnt[n].sum()) for n in self.orders)
        hr = " ".join(f"n{n}={self.hits[n]/max(self.positions,1)*100:.2f}%"
                      for n in self.orders)
        print(f"[preflight] rows={self.rows_done} lab_positions={self.positions} "
              f"scatter_ops={tot} hit_rate[{hr}] "
              f"align_valid={self.align_valid/max(self.align_total,1)*100:.2f}%", flush=True)
        if tot == 0 or self.align_valid / max(self.align_total, 1) < 0.9:
            ok = False
        print(f"[preflight] {'PASS' if ok else 'FAIL'}", flush=True)
        return ok

    # -- persistence -------------------------------------------------------- #
    def save(self, final=True):
        a = self.args
        os.makedirs(a.out_dir, exist_ok=True)
        if a.dry_run:
            print(f"[worker {a.worker}] dry run: not saving", flush=True)
            return
        name = (f"part_w{a.worker}_{a.tag}.npz" if final
                else f".part_w{a.worker}_{a.tag}.ckpt.npz")
        out = os.path.join(a.out_dir, name)
        tmp = os.path.join(a.out_dir, f".part_w{a.worker}.tmp.npz")
        arrs = {}
        for n in self.orders:
            arrs[f"keys_{n}"] = self.assign_keys[n]
            arrs[f"cnt_{n}"] = self.cnt[n].cpu().numpy()
            for bi, d in enumerate(self.depths):
                off = (bi + (1 if self.use_final else 0)) * self.hid
                arrs[f"sum_L{d}_{n}"] = self.acc[n][:, off: off + self.hid].cpu().numpy()
            if self.use_final:
                arrs[f"sum_Lfinal_{n}"] = self.acc[n][:, : self.hid].cpu().numpy()
        arrs["meta_json"] = np.array(json.dumps({
            "worker": a.worker, "tag": a.tag, "shards": self.shards_done,
            "depths": self.depths, "final_block": self.use_final, "orders": self.orders,
            "tokens": int(self.tokens_done), "docs": int(self.docs_done),
            "rows": int(self.rows_done), "elapsed_s": round(time.time() - self.t0, 1),
            "assign": os.path.basename(a.assign), "seq": a.seq, "bos": a.bos,
            "align": "packed_rows_byte_end_last_teacher_token",
            "packing": "bos_prefixed_units_lossless", "batch_rows": a.batch_rows,
            "hit_rate": {str(n): self.hits[n] / max(self.positions, 1) for n in self.orders},
            "hid": self.hid, "vocab": a.vocab,
        }))
        np.savez(tmp, **arrs)
        os.replace(tmp, out)
        print(f"[worker {a.worker}] saved {out} ({os.path.getsize(out)/1e9:.2f}GB)",
              flush=True)
        with open(os.path.join(a.out_dir, f"state_w{a.worker}_{a.tag}.json"), "w") as fh:
            json.dump({"shards": self.shards_done, "tokens": int(self.tokens_done),
                       "docs": int(self.docs_done), "rows": int(self.rows_done)}, fh)


def selftest(nblocks, hid, sentinel=True, seed=0):
    """CPU check that the pooler-normalization round trip recovers the blocks."""
    rng = np.random.default_rng(seed)
    rows, want = 7, nblocks * hid
    raw = rng.standard_normal((rows, want)).astype(np.float32) * 2.0
    if not sentinel:
        return True, "sentinel disabled"
    w = np.concatenate([raw, np.ones((rows, 1), dtype=np.float32)], axis=1)
    recv = (w / np.linalg.norm(w, axis=1, keepdims=True)).astype(np.float32)
    got = recv[:, :-1] * (1.0 / recv[:, -1])[:, None]
    err = float(np.abs(got - raw).max() / (np.abs(raw).max() + 1e-12))
    return err < 1e-5, f"max rel err {err:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--shard_start", type=int, default=0)
    ap.add_argument("--shard_end", type=int, default=210)
    ap.add_argument("--raw_dir", default="/data/climbmix_raw")
    ap.add_argument("--out_dir", default="/data/distill_out")
    ap.add_argument("--assign", default="/data/assign_exact974k.npz")
    ap.add_argument("--our_tokenizer", default="/data/tokenizer.pkl")
    ap.add_argument("--tok_bytes", default="/data/tok_bytes.npz")
    ap.add_argument("--model", default="/data/models/Qwen3.6-35B-A3B")
    ap.add_argument("--depths", default="0,10,25,40")
    ap.add_argument("--no_final_block", action="store_true")
    ap.add_argument("--no_sentinel", action="store_true",
                    help="disable un-normalizing the pooler's L2 output (debug only)")
    ap.add_argument("--orders", default="2,3,4,5")
    ap.add_argument("--hid", type=int, default=2048)
    ap.add_argument("--vocab", type=int, default=VOCAB_DEFAULT)
    ap.add_argument("--bos", type=int, default=BOS_DEFAULT)
    ap.add_argument("--seq", type=int, default=SEQ_DEFAULT)
    ap.add_argument("--batch_rows", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=64, help="log every N batches")
    ap.add_argument("--token_budget", type=float, default=2.5e9)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=6144)
    ap.add_argument("--save_every_shards", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="w3r")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="CPU-only end-to-end check of pack/align/accumulate; no GPU")
    ap.add_argument("--preflight_docs", type=int, default=64)
    ap.add_argument("--enforce_eager", action="store_true",
                    help="disable CUDA graphs (slower; use if capture misbehaves)")
    ap.add_argument("--max_num_batched_tokens", type=int, default=32768)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, "/data")

    w = Worker(args)
    ok, msg = selftest(w.nblocks, args.hid, sentinel=w.sentinel)
    print(f"[worker {args.worker}] selftest {'PASS' if ok else 'FAIL'}: {msg}", flush=True)
    if not ok:
        sys.exit(1)
    if args.preflight:
        sys.exit(0 if w.preflight(args.preflight_docs) else 1)

    from vllm import LLM
    from vllm.config.pooler import PoolerConfig

    t0 = time.time()
    w.llm = LLM(
        model=args.model, runner="pooling", dtype="bfloat16",
        tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
        trust_remote_code=True, disable_log_stats=True,
        max_num_seqs=4 * args.batch_rows,
        max_num_batched_tokens=args.max_num_batched_tokens,
        pooler_config=PoolerConfig(task="token_embed"),
    )
    print(f"[worker {args.worker}] engine ready in {time.time()-t0:.0f}s", flush=True)

    def _term(signum, _frame):
        print(f"[worker {args.worker}] signal {signum}, saving", flush=True)
        try:
            w.save()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    w.run()


if __name__ == "__main__":
    main()
