# Matched multi-layer memory rerun on 360-2

This packet contains lightweight evidence for the completed 20x, three-pass comparison. The remote final_model.pt files remain in /data/home/guoshaoyang/ngram-gap-lab/data/runs_scaling/; their SHA256 records are in final_manifest.txt.

Contract: vanilla nanoGPT 8L/6H/768D, seed 42, bf16, no compile, train shards 0--20, validation shards 21--23, 6740 batches per replay pass, 20220 total steps, clean 1,048,576-row bigram and trigram tables at 0-based injection blocks 1 and 4. The three routes differ only in injection position: v, y, or the current gated engram proxy. The proxy is not a claim of reproducing a full DeepSeek Engram implementation.

| route | step 6740 gap | step 13480 gap | step 20220 train | step 20220 val | step 20220 gap |
|---|---:|---:|---:|---:|---:|
| v | -0.011576 | +0.284653 | 2.266903 | 2.926786 | **+0.659884** |
| y | -0.007936 | +0.217367 | 2.314134 | 2.865290 | **+0.551156** |
| Engram-like gated proxy | -0.006052 | +0.100320 | 2.418342 | 2.797168 | **+0.378826** |

All three runs have three boundary records, 2022 table-norm records, a summary.json, and a roughly 13 GB final weight file. The final summaries report 3,285,795,072 parameters.
