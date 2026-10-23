#!/usr/bin/env python3
"""Recompute Appendix F five-epoch seen/novel validation CE from checkpoints.

This offline evaluation reads the original Plan 7 checkpoints and packed
shards. It does not train or modify checkpoints. Output is a plotted-value
manifest, not a paper figure. Use --counts-only before scheduling a GPU run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from train import NanoGPT, TokenizedShardDataset, _autocast_ctx  # noqa: E402
from tools.eval_logits_rank import cfg_from_summary  # noqa: E402

VOCAB = 8192
EPOCH_STEPS = (337, 674, 1011, 1348, 1685)
ARMS = ("both", "nogram")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def membership(keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    index = np.searchsorted(keys, queries)
    clipped = np.minimum(index, len(keys) - 1)
    return (index < len(keys)) & (keys[clipped] == queries)


def masks_for_batches(train_path: Path, batches: list[tuple[torch.Tensor, torch.Tensor]]):
    train = np.memmap(train_path, dtype=np.uint16, mode="r").astype(np.int64)
    bigram = np.unique(train[:-1] * VOCAB + train[1:])
    trigram = np.unique((train[:-2] * VOCAB + train[1:-1]) * VOCAB + train[2:])
    del train
    masks = []
    counts = {"seen": 0, "novel": 0, "unseen_context": 0}
    for x, y in batches:
        # Full-prefix prediction at t. The first two positions lack ctx3.
        x = x.numpy().astype(np.int64)
        y = y.numpy().astype(np.int64)
        context_key = x[:, 1:-1] * VOCAB + x[:, 2:]
        continuation_key = context_key * VOCAB + y[:, 2:]
        context_seen = membership(bigram, context_key)
        continuation_seen = membership(trigram, continuation_key)
        seen = context_seen & continuation_seen
        novel = context_seen & ~continuation_seen
        unseen = ~context_seen
        counts["seen"] += int(seen.sum())
        counts["novel"] += int(novel.sum())
        counts["unseen_context"] += int(unseen.sum())
        masks.append((seen, novel))
    assert sum(counts.values()) == sum(x.shape[0] * (x.shape[1] - 2) for x, _ in batches)
    return masks, counts


def eval_checkpoint(model: NanoGPT, batches, masks, device, row_chunk: int):
    totals = {"seen": 0.0, "novel": 0.0}
    counts = {"seen": 0, "novel": 0}
    full_sum = 0.0
    full_count = 0
    model.eval()
    with torch.inference_mode(), _autocast_ctx("bf16"):
        for (inp, target), (seen, novel) in zip(batches, masks):
            for start in range(0, len(inp), row_chunk):
                stop = min(start + row_chunk, len(inp))
                x = inp[start:stop].to(device)
                y = target[start:stop].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                                       reduction="none").reshape(y.shape).cpu().numpy()
                full_sum += float(loss.sum(dtype=np.float64))
                full_count += int(loss.size)
                classified = loss[:, 2:]
                for name, mask in (("seen", seen[start:stop]), ("novel", novel[start:stop])):
                    totals[name] += float(classified[mask].sum(dtype=np.float64))
                    counts[name] += int(mask.sum())
                del logits, loss, x, y
    return {"all_positions_ce": full_sum / full_count,
            **{name: {"n": counts[name], "ce_mean": totals[name] / counts[name]}
               for name in ("seen", "novel")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--row-chunk", type=int, default=6)
    parser.add_argument("--counts-only", action="store_true")
    args = parser.parse_args()
    assert args.row_chunk > 0

    runs = {arm: args.runs_root / f"ep5ckpt_v5_{arm}_s42_fixed" for arm in ARMS}
    summaries = {arm: json.loads((path / "summary.json").read_text())
                 for arm, path in runs.items()}
    reference = summaries["both"]["config"]
    for arm in ARMS:
        summary = summaries[arm]
        cfg = summary["config"]
        assert summary["run_id"] == runs[arm].name
        assert summary["seed"] == 42 and summary["steps"] == 1685
        assert summary["epoch_batches"] == 337 and summary["compute_dtype"] == "bf16"
        for field in ("train_shards", "val_shards", "val_batches", "device_batch_size",
                      "sequence_len"):
            assert cfg[field] == reference[field], (arm, field, cfg[field], reference[field])
        assert cfg["train_shards"] == [1] and cfg["val_shards"] == [2, 3, 4, 5, 6, 7, 8, 9, 10, 6542]
        assert cfg["val_batches"] == 4 and cfg["device_batch_size"] == 72
        assert cfg["sequence_len"] == 2048
    assert Path(reference["data_dir"]).resolve() == args.data_dir.resolve()

    val_ds = TokenizedShardDataset(str(args.data_dir), reference["val_shards"],
                                   reference["sequence_len"], reference["device_batch_size"], 42)
    iterator = val_ds.iter_batches(torch.device("cpu"))
    batches = [next(iterator) for _ in range(reference["val_batches"])]
    train_path = args.data_dir / "shard_00001.bin"
    masks, counts = masks_for_batches(train_path, batches)
    out = {"source": "ep5ckpt_v5_{both,nogram}_s42_fixed", "seed": 42,
           "steps": list(EPOCH_STEPS), "data_dir": str(args.data_dir.resolve()),
           "train_shard_sha256": sha256(train_path),
           "first_val_shard_sha256": sha256(args.data_dir / "shard_00002.bin"),
           "code_train_sha256": sha256(ROOT / "code/train.py"),
           "code_eval_logits_rank_sha256": sha256(ROOT / "code/tools/eval_logits_rank.py"),
           "metric": "position-weighted full-causal-prefix fixed-val CE, bf16 forward",
           "population": counts, "arms": {}}
    print(json.dumps({"population": counts}, indent=2), flush=True)
    if not args.counts_only:
        assert torch.cuda.is_available(), "GPU required for ten full checkpoint evaluations"
        device = torch.device("cuda")
        for arm in ARMS:
            model = NanoGPT(cfg_from_summary(summaries[arm]["config"]))
            out["arms"][arm] = []
            for epoch, step in enumerate(EPOCH_STEPS, start=1):
                checkpoint = runs[arm] / (f"epoch{epoch}_model.pt" if epoch < 5 else "final_model.pt")
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                del state
                model.to(device)
                values = eval_checkpoint(model, batches, masks, device, args.row_chunk)
                values.update({"epoch": epoch, "step": step,
                               "checkpoint": checkpoint.name,
                               "checkpoint_sha256": sha256(checkpoint)})
                out["arms"][arm].append(values)
                print(f"{arm} epoch={epoch} step={step} "
                      f"seen={values['seen']['ce_mean']:.6f} "
                      f"novel={values['novel']['ce_mean']:.6f} "
                      f"all={values['all_positions_ce']:.6f}", flush=True)
            del model
            torch.cuda.empty_cache()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(args.out_json)


if __name__ == "__main__":
    main()
