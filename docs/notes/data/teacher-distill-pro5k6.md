# Teacher-residual distillation on pro5k6 (Qwen3.6-35B-A3B)

Purpose: initialise / cluster the n-gram tables (orders 2-5) from a strong
teacher's layer-wise residual streams instead of training the table from scratch.
Requested by the user on 2026-10-21 (10B-token target, 2B first for basic
training).

## Hardware / corpus

| item | value |
|---|---|
| host | `pro5kvrdma6` (8x RTX PRO 5000 72GB, 7TB RAID0 at `/data`) |
| teacher | `/data/models/Qwen3.6-35B-A3B` (40 layers, hidden 2048, TP=2 per worker) |
| corpus | `/data/climbmix_raw/shard_*.parquet` (210 shards, 18GB raw text) |
| lab tokenizer | `/data/tokenizer.pkl` (tiktoken, vocab 8192, BOS 8188) |
| byte tables | `/data/tok_bytes.npz` (lab), derived from the Qwen BPE (teacher) |
| targets | `/data/assign_exact974k.npz` (974,333 rows over orders 2-5) |

Measured corpus size: **59M project tokens per shard -> ~12.4B over 210 shards**,
so the 10B target needs roughly 170 shards (2B needs ~34 shards).

## Pipeline (`code/tools/distill_prod.py`)

1. **Pack** (matches `data/tokenized`): documents are BOS-prefixed units packed
   losslessly into rows of exactly `--seq` tokens.  Packing is what makes the
   engine efficient: vLLM's pooling runner executes roughly one sequence per
   engine step, so short rows leave the GPU idle between steps.
2. **Teacher tokenize** each row (Qwen BPE) -> ~0.86 teacher tokens per lab token.
3. **Align** by byte position: lab token *t* takes the teacher state at the last
   teacher token whose cumulative byte end is <= lab cumulative byte end — i.e.
   the teacher's residual stream *after* reading the context ending at *t*, which
   is exactly what position *t* feeds forward when the row is injected at *t*.
   Byte units keep this valid for arbitrary UTF-8 (a char-offset reconstruction
   of tiktoken tokens is not reliable).
4. **Keys** via `code/ngram_freq.context_keys_np` (in-chunk shift rule); only
   contexts present in the assignment table are accumulated.
5. **Accumulate** residual sums + counts per (order, table row) with GPU
   `index_add_`, then save one npz part per worker.

### Multi-layer capture

`oe_cap_patch.py` (installed at
`site-packages/oe_cap_patch.py`, hooked from the end of
`vllm/model_executor/models/qwen3_5.py`) registers forward hooks on the decoder
layers named by `OE_CAP_LAYERS` (HF `hidden_states` indices: `0` = embedding
output, `k` = residual stream after decoder layer `k`).  Each captured block is
`hidden_states + residual` of that layer's return tuple, i.e. the residual stream
the next block reads.

Two vLLM behaviours had to be handled:

* **The pooler L2-normalizes rows.** `token_embed` hardcodes `PoolerNormalize`,
  which normalizes *every* row of whatever the model returns, including appended
  capture blocks.  The patch therefore appends a sentinel column of ones, which
  arrives as `1/||row||`; the driver multiplies by its reciprocal to recover the
  raw values exactly (verified to 5e-8).
* **The capture must survive CUDA-graph capture / per-step calls.** The patch
  clears its store at the start of every outer forward and only emits capture
  blocks whose row count equals the output's; if anything does not line up it
  returns the plain output, and the driver raises (sentinel width check) instead
  of silently writing garbage.

### Validation performed

| check | result |
|---|---|
| HF transformers ground truth (CPU, real doc, 219 teacher tokens) | `hidden_states` RMS d0=0.0105174, d10=0.051604, d25=0.10792, d39=0.340081, normed=2.4941 |
| elementwise vs captured blocks | cos 1.000000 (d0), 0.999399 (d10), 0.998062 (d25), 0.998140 (normed) |
| production diag (same doc through vLLM) | d0=0.0105174, d10=0.051635, d25=0.108466, normed=2.49382 |
| byte-end alignment | exact (`byte_end_match=True`), 99.9% of positions aligned |
| key hit rate (top-974k table, 96-doc preflight) | n2 83.7%, n3 37.0%, n4 10.8%, n5 3.0% |

## Measured throughput (2026-10-21)

| config | lab tok/s per worker | note |
|---|---|---|
| model reloaded per shard | ~0 (11 min reload each) | first attempt, killed |
| resident engine, one doc per item (~560-token rows) | 575 | 50-day ETA for 10B |
| resident engine, packed 2048-token rows, `batch_rows=16` | 12,024-12,375 | 3 workers |
| same + `max_num_batched_tokens=32768` | 13,503 | 4th worker |

Four TP=2 workers give ~49.5k lab tok/s aggregate, but the forward is **not** the
bottleneck.  Per-batch accounting (16 packed rows = 32,768 lab positions, teacher
~28.2k tokens, `data` = (nq, 10,240) fp32): step **2.70 s = GPU forward ~0.67 s
(~24k teacher tok/s/GPU — the MoE forward is efficient) + CPU accumulation
~2.03 s**.  py-spy attributes 84% of the driver's CPU time to the
`torch.index_add_` calls into the (R_n, 10,240) fp32 accumulators, and each driver
process burns 2,500-2,800% CPU.  The root cause is the box default of **192 torch
threads**: isolated n2 `index_add_` takes 362 ms @192 threads vs 61 ms @32, and
under 4-worker contention the medians inflate ~7x.

Two validated successors of `distill_prod.py` (both bit-identical to production
on CPU replays of real shard data: `cnt` exact, `acc` max-abs diff 0, all
counters equal):

| variant | change | step time |
|---|---|---|
| `/data/distill_prod_v2.py` | `torch.set_num_threads(32)` + reusable `ones` | ~1.26 s measured in a loaded-box replay |
| `/data/distill_prod_v3.py` | v2 + producer/consumer overlap (encode in a producer thread, accumulate + all state in the consumer) | ~0.79 s |

ETAs: at 49.5k tok/s, 2B (checkpoint coverage) lands ~2026-10-21 22:45 and its
merge finishes ~23:30; 10B would need ~51h from the w3r start.  Restarting the
workers on v2/v3 at the first checkpoint boundary makes the remaining ~7B tokens
roughly 2-3x faster, so the 10B table is expected 2026-10-22 evening rather than
10-23 evening.  `run_distill_v2.sh` (tag `w3s`, `--shard_start 48`,
`--save_every_shards 1`, 1.8e9 tokens/worker) is prepared and validated but not
launched; the restart must happen right after a checkpoint so no in-flight shard
is lost, and the covered frontier has to be recomputed from
`state_w*_w3r.json` at that moment.


CUDA graphs are **not** usable with multi-layer capture: vLLM's graph runner
requires the model output to be `hidden_size` wide, while capture appends
`depths+1` blocks (the engine dies during graph init).  Either run eager with
capture, or use graphs and lose the intermediate depths.

## Milestones while the run is in flight

`code/tools/milestone_watch.py` (deployed at `/data/milestone_watch.py`, CPU only,
never touches the GPU) polls the four worker logs and checkpoint state files and
fires two bounded milestones, writing reports to
`/data/distill_out/milestone_report_*.txt`:

- **M1** as soon as all four workers have a periodic checkpoint: merge with
  `--include_ckpt --mean`, run `check_distill_aggregate.py` against the HF depth
  references, keep the report, delete the merged copy (the per-worker parts stay
  authoritative).  Coverage at M1 is ~0.95B tokens.
- **M2** once checkpoint coverage (summed `state_w*_w3r.json` tokens, which only
  counts data actually inside the ckpts) passes 2.5e9: merge again and **keep**
  the table as `teacher_table_o2to5_2b.npz` — this is the artifact for the first
  training experiments.  Expected coverage ~2.6B tokens.

The stale `part_w3_w3r.npz` from the earlier `--max_docs` test run was renamed to
`oldtest_w3_w3r.npz`: with `--include_ckpt` a final part shadows that worker's
ckpts, so leaving it in place would have silently dropped worker 3's real data
from the merge.

## Commands

```bash
# CPU-only preflight (no GPU): packing, alignment, accumulation invariants
python3 code/tools/distill_prod.py --preflight --preflight_docs 96

# production: four TP=2 workers over GPU pairs
bash code/cluster/run_w3_distill_prod.sh start     # 2.5B project tokens per worker
bash code/cluster/run_w3_distill_prod.sh status
bash code/cluster/run_w3_distill_prod.sh stop      # SIGTERM -> workers save parts

# merge the per-worker parts into one table
python3 code/tools/merge_distill_parts.py --in_dir /data/distill_out --tag w3r \
    --out /data/distill_out/teacher_table_o2to5.npz --mean
```

## Outputs

`/data/distill_out/part_w<k>_<tag>.npz`, each ~40GB with `keys_<n>`, `cnt_<n>`,
`sum_L<d>_<n>` for every captured depth and order, plus a `meta_json` block
carrying token/doc/row counts, hit rates and the exact alignment convention.
Parts resume independently; `merge_distill_parts.py` sums them and can emit
per-row means plus coverage counts.

## History (why the first attempts were slow)

1. First script reloaded the 67GB model **once per shard** (11 min each): ~10h of
   pure reload overhead.  Killed; nothing valid was produced.
2. Second script kept the model resident but processed one document per item with
   `batch_docs=32`; vLLM's pooling path still ran ~one sequence per step at
   ~560-token rows, wasting the GPU between steps (~575 lab tok/s/worker, i.e. a
   ~50-day ETA for 10B).  Fixed by packing to full 2048-token rows.
