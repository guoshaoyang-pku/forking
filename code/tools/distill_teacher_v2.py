"""Distill Qwen3.6-35B-A3B residual stream into our n-gram context keys.

Pipeline:
  1. Read parquet shard (ClimbMix raw text, column 'text')
  2. For each doc: tokenize with our 8192 BPE (tiktoken) AND Qwen tokenizer
  3. vLLM token_embed → per-Qwen-token residual (2048d, last layer)
  4. Align: for each our-token, weighted-mean of overlapping Qwen-token residuals
  5. Accumulate: for n in {2,3,4,5}, key = tuple(our_ids[t-n+1..t])
     sum[key] += residual, cnt[key] += 1
  6. Save per-shard npz

Output: /data/distill_out/shard_NNNNN.npz
  keys: L0_n{n}_keys (K,n) int32, L0_n{n}_sums (K,2048) fp16, L0_n{n}_cnts (K,) int64
  (L0 = last layer of Qwen = layer 40 output)

Run: 4 processes, each with TP=2 on 2 GPUs.
"""
import argparse, os, pickle, sys, time
import numpy as np
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor

ORDERS = (2, 3, 4, 5)
HIDDEN = 2048


def char_span_tokenize_ours(text, enc):
    """Tokenize text with our tiktoken tokenizer, return (ids, char_offsets).

    tiktoken doesn't give offset_mapping directly, so we decode each token
    to find its char span. For BPE this is exact.
    """
    ids = enc.encode(text)
    offsets = []
    pos = 0
    for tid in ids:
        tok_bytes = enc.decode_single_token_bytes(tid)
        tok_str = tok_bytes.decode('utf-8', errors='replace')
        # Find this token starting from pos
        idx = text.find(tok_str, pos)
        if idx < 0:
            # Fallback: approximate
            idx = pos
        offsets.append((idx, idx + len(tok_str)))
        pos = idx + len(tok_str)
    return ids, offsets


def char_span_tokenize_qwen(text, tokenizer):
    """Tokenize with Qwen HF tokenizer, return (ids, char_offsets)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc['input_ids'], enc['offset_mapping']


def align_residuals(our_offsets, qwen_offsets, qwen_residual):
    """Map our-token spans to weighted mean of overlapping Qwen-token residuals.

    our_offsets: list of (start, end) char spans, length M
    qwen_offsets: list of (start, end) char spans, length Q
    qwen_residual: (Q, H) tensor/numpy
    returns: (M, H) fp32 numpy
    """
    import torch
    if isinstance(qwen_residual, torch.Tensor):
        qwen_residual = qwen_residual.cpu().numpy()
    M = len(our_offsets)
    H = qwen_residual.shape[1]
    out = np.zeros((M, H), dtype=np.float32)
    wsum = np.zeros(M, dtype=np.float32)
    qi = 0
    Q = len(qwen_offsets)
    for mi in range(M):
        s, e = our_offsets[mi]
        while qi < Q and qwen_offsets[qi][1] <= s:
            qi += 1
        qj = qi
        while qj < Q and qwen_offsets[qj][0] < e:
            os_, oe = qwen_offsets[qj]
            ov = min(e, oe) - max(s, os_)
            if ov > 0:
                out[mi] += ov * qwen_residual[qj].astype(np.float32)
                wsum[mi] += ov
            qj += 1
    mask = wsum > 0
    out[mask] /= wsum[mask, None]
    return out


def process_shard(gpu_pair, shard_path, our_tokenizer_path, qwen_model_path,
                  out_dir, max_docs=None):
    """Process one parquet shard on a specific GPU pair."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_pair))

    name = os.path.basename(shard_path).replace('.parquet', '')
    out_path = os.path.join(out_dir, f'{name}.npz')
    if os.path.exists(out_path):
        print(f'[skip] {name}', flush=True)
        return name, 0

    t0 = time.time()

    # Load tokenizers
    import tiktoken
    with open(our_tokenizer_path, 'rb') as f:
        our_enc = pickle.load(f)
    from transformers import AutoTokenizer
    qwen_tok = AutoTokenizer.from_pretrained(qwen_model_path)

    # Read texts
    table = pq.read_table(shard_path, columns=['text'])
    texts = table.column('text').to_pylist()
    if max_docs:
        texts = texts[:max_docs]
    n_docs = len(texts)
    print(f'[load] GPU{gpu_pair} {name}: {n_docs} docs, {time.time()-t0:.0f}s', flush=True)

    # Tokenize all docs with both tokenizers
    t1 = time.time()
    our_data = []  # (ids, offsets) per doc
    for txt in texts:
        ids, offsets = char_span_tokenize_ours(txt, our_enc)
        our_data.append((ids, offsets))
    print(f'[tok-ours] GPU{gpu_pair} {time.time()-t1:.0f}s', flush=True)

    # vLLM inference
    from vllm import LLM, PoolingParams
    from vllm.config.pooler import PoolerConfig
    llm = LLM(model=qwen_model_path, runner='pooling', dtype='bfloat16',
              gpu_memory_utilization=0.90, max_model_len=8192, enforce_eager=True,
              trust_remote_code=True, tensor_parallel_size=2,
              pooler_config=PoolerConfig(task='token_embed'))

    t2 = time.time()
    outs = llm.encode(texts, pooling_task='token_embed')
    print(f'[vllm] GPU{gpu_pair} {n_docs} docs, {time.time()-t2:.0f}s', flush=True)

    # Align and accumulate
    t3 = time.time()
    acc = {}  # key (n, tuple) → [sum_vec, cnt]
    total_our_tokens = 0
    for di, (o, (our_ids, our_offsets)) in enumerate(zip(outs, our_data)):
        qwen_offsets = char_span_tokenize_qwen(texts[di], qwen_tok)[1]
        residual = o.outputs.data  # (Q, H) fp32
        agg = align_residuals(our_offsets, qwen_offsets, residual)
        M = len(our_ids)
        total_our_tokens += M
        for n in ORDERS:
            for t in range(n - 1, M):
                key = (n, tuple(our_ids[t - n + 1: t + 1]))
                if key in acc:
                    acc[key][0] += agg[t]
                    acc[key][1] += 1
                else:
                    acc[key] = [agg[t].copy(), 1]
    print(f'[accum] GPU{gpu_pair} {total_our_tokens} our-tokens, {len(acc)} keys, {time.time()-t3:.0f}s', flush=True)

    # Save
    save = {}
    for n in ORDERS:
        nkeys = [k for k in acc if k[0] == n]
        if not nkeys:
            continue
        kid = np.array([k[1] for k in nkeys], dtype=np.int32)
        sums = np.stack([acc[k][0] for k in nkeys])
        cnts = np.array([acc[k][1] for k in nkeys], dtype=np.int64)
        save[f'L0_n{n}_keys'] = kid
        save[f'L0_n{n}_sums'] = sums.astype(np.float16)
        save[f'L0_n{n}_cnts'] = cnts
    np.savez_compressed(out_path, **save)
    total_keys = len(acc)
    dt = time.time() - t0
    print(f'[done] GPU{gpu_pair} {name}: {total_keys} keys, {total_our_tokens} our-tokens, {dt:.0f}s → {out_path}', flush=True)
    return name, total_our_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard_dir', default='/data/climbmix_raw')
    ap.add_argument('--out_dir', default='/data/distill_out')
    ap.add_argument('--our_tokenizer', default='/data/tokenizer.pkl')
    ap.add_argument('--qwen_model', default='/data/models/Qwen3.6-35B-A3B')
    ap.add_argument('--max_docs', type=int, default=None)
    ap.add_argument('--shard_start', type=int, default=0)
    ap.add_argument('--shard_end', type=int, default=210)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    shards = sorted(
        os.path.join(args.shard_dir, f)
        for f in os.listdir(args.shard_dir)
        if f.endswith('.parquet')
    )[args.shard_start:args.shard_end]
    print(f'Processing {len(shards)} shards with {args.num_workers} workers', flush=True)

    gpu_pairs = [[i*2, i*2+1] for i in range(args.num_workers)]

    total_tokens = 0
    t0 = time.time()
    with ProcessPoolExecutor(args.num_workers) as ex:
        futs = []
        for si, sp in enumerate(shards):
            gp = gpu_pairs[si % args.num_workers]
            futs.append(ex.submit(process_shard, gp, sp, args.our_tokenizer,
                                  args.qwen_model, args.out_dir, args.max_docs))
        for f in futs:
            name, ntok = f.result()
            total_tokens += ntok

    wall = time.time() - t0
    print(f'\nALL DONE: {total_tokens} our-tokens, {wall:.0f}s, '
          f'{total_tokens/wall:.0f} tok/s', flush=True)


if __name__ == '__main__':
    main()
