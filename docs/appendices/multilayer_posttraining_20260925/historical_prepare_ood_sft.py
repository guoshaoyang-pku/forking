#!/usr/bin/env python3
"""Build fixed uint16 shards for a small OOD continued-LM stress test."""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

def iter_texts(paths, source: str):
    if source == "openr1":
        for path in paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=128, columns=["problem", "solution"]):
                for problem, solution in zip(batch.column("problem").to_pylist(), batch.column("solution").to_pylist()):
                    yield str(problem) + "\n\n" + str(solution)
    elif source == "magicoder":
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    yield str(row.get("problem", "")) + "\n\n" + str(row.get("solution", ""))
    else:
        raise ValueError(source)

def write_rows(texts, enc, bos_id: int, row_len: int, train_rows: int, val_rows: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out_dir / "shard_00000.bin", out_dir / "shard_00001.bin"
    target, written, buf = train_rows + val_rows, 0, []
    with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
        for text in texts:
            buf.extend([bos_id] + enc.encode_ordinary(text))
            while len(buf) >= row_len and written < target:
                row = np.asarray(buf[:row_len], dtype=np.uint16)
                del buf[:row_len]
                row.tofile(train_f if written < train_rows else val_f)
                written += 1
            if written >= target:
                break
    if written < target:
        raise RuntimeError(f"source exhausted after {written} rows, need {target}")
    meta = {"row_len": row_len, "train_rows": train_rows, "val_rows": val_rows,
            "train_batches": train_rows // 72, "val_batches": val_rows // 72,
            "tokenizer": "pretraining autoresearch tokenizer.pkl",
            "source": "streamed source; train rows precede held-out validation rows"}
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"written_rows": written, **meta}, sort_keys=True))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openr1", "magicoder"], required=True)
    ap.add_argument("--input", required=True, action="append"); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--sequence_len", type=int, default=2048)
    ap.add_argument("--train_batches", type=int, default=333); ap.add_argument("--val_batches", type=int, default=4)
    ap.add_argument("--device_batch_size", type=int, default=72); args = ap.parse_args()
    with open(args.tokenizer, "rb") as f: enc = pickle.load(f)
    bos_id = enc.encode_single_token("<|reserved_0|>")
    train_rows, val_rows = args.train_batches * args.device_batch_size, args.val_batches * args.device_batch_size
    write_rows(iter_texts([Path(p) for p in args.input], args.source), enc, bos_id, args.sequence_len + 1, train_rows, val_rows, Path(args.out_dir))

if __name__ == "__main__": main()
