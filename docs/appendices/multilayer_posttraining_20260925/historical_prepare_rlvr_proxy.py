#!/usr/bin/env python3
"""Prepare a small static verifier-selected continued-LM proxy."""
from __future__ import annotations
import argparse, hashlib, json, pickle
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

def rows(paths):
    for path in paths:
        table = pq.read_table(path, columns=["messages", "ground_truth", "dataset", "constraint_type"])
        for row in table.to_pylist():
            messages = row["messages"] or []
            text = "\n".join(str(m.get("content", "")) for m in messages if m)
            gt = row.get("ground_truth")
            if gt is not None and str(gt).strip():
                text += "\n\nAnswer:" + str(gt)
            yield text, row.get("dataset"), row.get("constraint_type")

def write(args):
    with open(args.tokenizer, "rb") as f:
        enc = pickle.load(f)
    bos = enc.encode_single_token("<|reserved_0|>")
    train_rows = args.train_batches * args.batch_size
    val_rows = args.val_batches * args.batch_size
    need = train_rows + val_rows
    records = list(rows([Path(x) for x in args.input]))
    if len(records) < 2:
        raise RuntimeError("need at least two source examples for a disjoint split")
    split = max(1, min(len(records) - 1, int(round(len(records) * args.train_fraction))))
    train_records, val_records = records[:split], records[split:]
    written = 0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out / "shard_00000.bin", out / "shard_00001.bin"
    stats = {"source_rows_seen": 0, "dataset_counts": {}, "constraint_counts": {}}
    def emit(stream_records, target_rows, fh):
        buf = []
        count = 0
        index = 0
        while count < target_rows:
            text, dataset, constraint = stream_records[index % len(stream_records)]
            index += 1
            stats["source_rows_seen"] += 1
            stats["dataset_counts"][str(dataset)] = stats["dataset_counts"].get(str(dataset), 0) + 1
            stats["constraint_counts"][str(constraint)] = stats["constraint_counts"].get(str(constraint), 0) + 1
            buf.extend([bos])
            buf.extend(enc.encode_ordinary(text))
            while len(buf) >= args.sequence_len + 1 and count < target_rows:
                np.asarray(buf[: args.sequence_len + 1], dtype=np.uint16).tofile(fh)
                del buf[: args.sequence_len + 1]
                count += 1
        return count
    with train_path.open("wb") as tf, val_path.open("wb") as vf:
        written += emit(train_records, train_rows, tf)
        written += emit(val_records, val_rows, vf)
    if written < need:
        raise RuntimeError(f"source produced {written} rows, need {need}")
    sha = hashlib.sha256()
    for p in args.input:
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                sha.update(block)
    manifest = {
        "kind": "verifier-selected-static-continued-pretraining-proxy",
        "source_files": [str(Path(x).resolve()) for x in args.input],
        "source_sha256_combined": sha.hexdigest(),
        "source_revision": args.revision,
        "text_rule": "messages content verbatim, then one Answer + ground_truth; no duplicate answer append",
        "sequence_len": args.sequence_len,
        "row_len": args.sequence_len + 1,
        "train_batches": args.train_batches,
        "val_batches": args.val_batches,
        "batch_size": args.batch_size,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "train_rows_precede_heldout_validation_rows": True,
        "train_fraction_by_source_example": args.train_fraction,
        "source_examples": len(records),
        "train_source_examples": len(train_records),
        "val_source_examples": len(val_records),
        "stats": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"written_rows": written, **manifest}, ensure_ascii=False, indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--sequence_len", type=int, default=2048)
    p.add_argument("--train_batches", type=int, default=333)
    p.add_argument("--val_batches", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=72)
    p.add_argument("--train_fraction", type=float, default=0.95)
    write(p.parse_args())

if __name__ == "__main__": main()
