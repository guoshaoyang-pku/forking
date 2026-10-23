#!/usr/bin/env python3
"""--stability_interval is logging-only: training must be bit-identical with/without it.

Run: python3 code/tools/test_stability_log_cpu.py
  S1 tiny 2-order input-injection run, 12 steps fp32, --stability_interval 0 vs 3:
     train_log.jsonl losses identical, stability.jsonl written with the expected keys.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_ngram_orders_cpu as T  # noqa: E402
from ngram_freq import MultiOrderFrequencyIndex  # noqa: E402


def run(tmp, data, idx_path, tag, extra):
    out = os.path.join(tmp, "runs")
    cmd = [sys.executable, os.path.join(T.CODE, "train.py"), "--run_id", tag,
           "--out_dir", out, "--data_dir", data, "--train_shards", "1", "--val_shards", "2",
           "--steps", "12", "--dtype", "fp32", "--ngram_orders", "2,3,4,5",
           "--bigram_clean_table", str(T.R), "--trigram_clean_table", str(T.R),
           "--ngram_clean_table", str(T.R), "--n_layer", "2", "--n_head", "2", "--n_embd", "32",
           "--sequence_len", str(T.SEQ), "--device_batch_size", "4",
           "--total_batch_size", str(4 * T.SEQ), "--lr", "0.0006",
           "--table_optimizer", "adamw", "--table_betas", "0.9,0.95", "--table_grad_clip", "1.0",
           "--val_interval", "2", "--val_batches", "2", "--freq_eval_batches", "2",
           "--freq_index", idx_path, "--diag_workers", "0"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=T.CODE)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return os.path.join(out, tag)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(11)
        data = os.path.join(tmp, "d")
        os.makedirs(data)
        T.repetitive_tokens(rng, 64).tofile(os.path.join(data, "shard_00001.bin"))
        rng.integers(0, T.VOCAB, size=(64, T.SEQ + 1)).astype(np.uint16).tofile(
            os.path.join(data, "shard_00002.bin"))
        idx_path = os.path.join(data, "multi.npz")
        MultiOrderFrequencyIndex.build_from_chunks(data, [1], T.VOCAB, range(2, 6), T.SEQ + 1,
                                                   log=lambda *_: None).save(idx_path)
        a = run(tmp, data, idx_path, "plain", [])
        b = run(tmp, data, idx_path, "stab", ["--stability_interval", "3"])
        la = [json.loads(x) for x in open(os.path.join(a, "train_log.jsonl"))]
        lb = [json.loads(x) for x in open(os.path.join(b, "train_log.jsonl"))]
        assert len(la) == len(lb) > 0
        for x, y in zip(la, lb):
            assert x["train_loss"] == y["train_loss"] and x.get("val_loss") == y.get("val_loss"), (x, y)
        assert not os.path.exists(os.path.join(a, "stability.jsonl"))
        st = [json.loads(x) for x in open(os.path.join(b, "stability.jsonl"))]
        steps = [r["step"] for r in st]
        assert steps == sorted(set(list(range(1, 11)) + [12])), steps
        need = {"g_backbone", "g_tables", "g_wte", "g_q0", "g_k1", "w_wte_rms", "w_q1",
                "x_tok_rms", "x_ngram_rms", "x_rms0", "x_rms1", "att_max0", "att_max1",
                "att_ent1", "att_entmin0", "logit_max", "lse_mean", "x_amax1", "x_tokmax1",
                "att_bos0", "att_prev1", "att_self0", "att_topkey_frac1", "att_topkey_tok0"}
        assert need <= set(st[-1]), need - set(st[-1])
        assert st[-1]["g_tables"] > 0 and st[-1]["x_ngram_rms"] > 0
        print(f"S1 ok ({len(la)} identical train/val rows; stability keys {len(st[-1])})")
        # S2: --adam_beta1 0.8 (explicit = contract default) is bit-identical; 0.9 differs
        c = run(tmp, data, idx_path, "b08", ["--adam_beta1", "0.8"])
        lc = [json.loads(x) for x in open(os.path.join(c, "train_log.jsonl"))]
        assert [x["train_loss"] for x in lc] == [x["train_loss"] for x in la]
        d = run(tmp, data, idx_path, "b09", ["--adam_beta1", "0.9"])
        ld = [json.loads(x) for x in open(os.path.join(d, "train_log.jsonl"))]
        assert [x["train_loss"] for x in ld][1:] != [x["train_loss"] for x in la][1:]
        print("S2 ok (beta1 0.8 explicit == default; 0.9 changes the run)")
        # S3: --qk_norm runs, changes the run, and the probe sees the normalised q/k
        e = run(tmp, data, idx_path, "qkn", ["--qk_norm", "--stability_interval", "3"])
        le = [json.loads(x) for x in open(os.path.join(e, "train_log.jsonl"))]
        assert [x["train_loss"] for x in le] != [x["train_loss"] for x in la]
        se = [json.loads(x) for x in open(os.path.join(e, "stability.jsonl"))]
        hd = 32 // 2
        assert se[-1]["att_max0"] <= hd ** 0.5 * 1.5 + 1e-3, se[-1]["att_max0"]
        print(f"S3 ok (qk_norm active; att_max0 {se[-1]['att_max0']:.2f} <= ~sqrt(head_dim))")
    print("ALL PASS")


if __name__ == "__main__":
    main()
