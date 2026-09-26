# fig09_crowd_ce_seen_novel extraction

Coordinates recovered from the vector paths of `paper/figures/fig09_crowd_ce_seen_novel.pdf`.

## Source

- **PDF**: `paper/figures/fig09_crowd_ce_seen_novel.pdf`
- **SHA-256**: `d62d7544440a7fc07c2652f87353892d9870bf77bca32c5860e5d8f794cb17ba`
- **Page size**: 150.48 x 181.44 pt
- **Extraction method**: pypdf 6.12.2 content-stream parsing (moveto/lineto coordinates), calibrated against axis tick marks at epochs 1-5 (x) and CE 4/6/8/10 nats (y). Linear calibration verified to < 0.001 nats at all four y-ticks and < 0.001 epoch at all five x-ticks.

## Series identification

| Column       | Legend in original | Description                                                    |
|--------------|--------------------|----------------------------------------------------------------|
| ngram_seen   | ngram, seen        | Both-table model, train-seen bigram context + seen continuation |
| ngram_novel  | ngram, novel       | Both-table model, train-seen bigram context + novel continuation |
| nogram_seen  | no-gram, seen      | No-gram control, train-seen bigram context + seen continuation |
| nogram_novel | no-gram, novel     | No-gram control, train-seen bigram context + novel continuation |

Identification confirmed by matching endpoint values against `paper/sections/appendix_f_mechanism.tex`:
- no-gram seen: 4.3 -> 2.8 nats (paper) vs 4.29 -> 2.82 (extracted, epochs 1->3)
- no-gram novel: 6.9 -> 5.6 nats (paper) vs 6.90 -> 5.58 (extracted, epochs 1->3)
- ngram novel: 7.5 -> 9.0 nats (paper) vs 7.52 -> 9.00 (extracted, epochs 1->3)
- ngram seen: "stays near 3.4" (paper) vs 3.49 at epoch 1 (extracted)

## Position counts (from tex caption)

- Train-seen bigram context + seen continuation: 405,058 positions
- Train-seen bigram context + novel continuation: 158,790 positions
- Excluded (training-unseen contexts): 25,400 positions

## Checkpoints

Steps 337, 674, 1011, 1348, 1685 (epochs 1-5, seed 42).

## Limitations

The original generator script (`scripts/FIGURE_PROVENANCE.md` line ~50) has an unresolved hash mismatch with the stored PDF; the checkpoints are on cluster 360-2. These extracted coordinates are the best available artifact-traceable source for replotting at publication size.
