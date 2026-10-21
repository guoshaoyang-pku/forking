"""Engine throughput probe: batch size x CUDA-graphs x capture, on an idle GPU pair.

Reports teacher tokens/s so production can be re-planned before burning 10B tokens
at the wrong setting.

    CUDA_VISIBLE_DEVICES=4,5 OE_CAP_LAYERS=0,10,25,40 python3 bench_engine.py eager0
    CUDA_VISIBLE_DEVICES=6,7 OE_CAP_LAYERS= python3 bench_engine.py eager1_nocap
"""
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

MODE = sys.argv[1] if len(sys.argv) > 1 else "eager0"
EAGER = MODE.startswith("eager1")
CAP = [int(x) for x in os.environ.get("OE_CAP_LAYERS", "").split(",") if x.strip()]
NBLOCK = len(CAP) + 1
SEQ = 2048
BATCHES = [8, 32, 128]
REPS = 2
NDOC_POOL = 160

MODEL = "/data/models/Qwen3.6-35B-A3B"
sys.path.insert(0, "/data")
from distill_prod import Worker, prepare_doc, selftest, lab_token_byte_table, qwen_token_byte_table  # noqa: E402
import pickle  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def main():
    args = SimpleNamespace(
        worker=0, num_workers=4, shard_start=0, shard_end=210,
        raw_dir="/data/climbmix_raw", out_dir="/data/distill_out",
        assign="/data/assign_exact974k.npz", our_tokenizer="/data/tokenizer.pkl",
        tok_bytes="/data/tok_bytes.npz", model=MODEL, depths=",".join(map(str, CAP)) or "0",
        no_final_block=False, orders="2,3,4,5", hid=2048, vocab=8192, bos=8188,
        seq=SEQ, batch_docs=32, log_every=512, token_budget=1e18, diag_docs=0,
        max_docs=0, max_model_len=6144, device="cpu", tag="bench",
        dry_run=True, preflight=False, preflight_docs=0, gpu_mem_util=0.82, tp=2,
        no_sentinel=False,
    )
    if not CAP:
        args.no_final_block = True  # single final block only
    w = Worker(args)
    ok, msg = selftest(w.nblocks, args.hid, sentinel=w.sentinel)
    print(f"[bench] mode={MODE} blocks={w.nblocks} sentinel={w.sentinel} selftest={ok} {msg}",
          flush=True)

    from vllm import LLM
    from vllm.config.pooler import PoolerConfig

    t0 = time.time()
    llm = LLM(model=MODEL, runner="pooling", dtype="bfloat16", tensor_parallel_size=2,
              gpu_memory_utilization=0.85, max_model_len=args.max_model_len,
              enforce_eager=EAGER, trust_remote_code=True, disable_log_stats=True,
              pooler_config=PoolerConfig(task="token_embed"))
    print(f"[bench] engine ready in {time.time()-t0:.0f}s (enforce_eager={EAGER})", flush=True)

    texts = pq.read_table(f"{args.raw_dir}/shard_00000.parquet",
                          columns=["text"]).column("text").to_pylist()
    prep = []
    t0 = time.time()
    for t in texts[:NDOC_POOL]:
        p = prepare_doc(t, w.lab_enc, w.lab_tb, w.qwen_tok, args.seq, args.bos, 6000)
        if p is not None:
            prep.append(p)
        if len(prep) >= max(BATCHES):
            break
    print(f"[bench] prepared {len(prep)} docs in {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/max(len(prep),1)*1000:.1f} ms/doc)", flush=True)
    nlab = np.array([len(p[0]) for p in prep])
    nq = np.array([len(p[1]) for p in prep])
    print(f"[bench] lab tokens/doc mean={nlab.mean():.0f} p50={np.median(nlab):.0f} "
          f"max={nlab.max()}; teacher/doc mean={nq.mean():.0f}", flush=True)

    for bs in BATCHES:
        cur = prep[:bs]
        toks = int(nq[:bs].sum())
        for rep in range(REPS):
            if bs == BATCHES[0] and rep == 0:
                print("[bench] (first call may include compilation/graph capture)", flush=True)
            t0 = time.time()
            outs = llm.encode([{"prompt_token_ids": list(map(int, p[1]))} for p in cur],
                              pooling_task="token_embed")
            enc = time.time() - t0
            dim = np.asarray(outs[0].outputs.data).shape[1]
            got = sum(len(o.prompt_token_ids) for o in outs)
            if bs == BATCHES[0] and rep == 0:
                d0 = w.denorm(np.asarray(outs[0].outputs.data, dtype=np.float32))
                nb, hid = w.nblocks, w.hid
                rms = [float(np.sqrt((d0[:, i * hid:(i + 1) * hid] ** 2).mean()))
                       for i in range(nb)]
                print(f"[bench] block RMS (HF ref d0=0.0105 d10=0.0516 d25=0.1079 "
                      f"final=2.494): " + " ".join(f"{r:.6g}" for r in rms), flush=True)
            print(f"[bench] bs={bs:4d} rep={rep} teacher_tok={toks:7d} encode={enc:6.2f}s "
                  f"{toks/enc:8.0f} teacher tok/s ({toks/enc*(nlab[:bs].sum()/nq[:bs].sum()):.0f} lab)"
                  f"  dim={dim} rows_ok={got == toks}", flush=True)
        # driver CPU cost for this batch (once, using the last result)
        t0 = time.time()
        for p, o in zip(cur, outs):
            d = w.denorm(np.asarray(o.outputs.data, dtype=np.float32))
            w.accumulate(p[0], p[1], p[2], d)
        cpu = time.time() - t0
        print(f"[bench] bs={bs:4d} driver_cpu={cpu:6.2f}s ({cpu/enc*100:.1f}% of last encode) "
              f"-> {int(nlab[:bs].sum())/cpu:.0f} lab tok/s if CPU-bound", flush=True)


if __name__ == "__main__":
    main()
