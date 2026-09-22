# context_subspace_mask_20260921

**→ 本实验（§62）的完整交接文档见 `HANDOVER.md`（setting、run 矩阵、口径、复现命令、已知坑）。论文 fig7 口径 = fixed（`fig07_context_mask_fixed_perstep`）。**

Curve coordinates for the paper's random-context loss-masking figure (Sec. 4.3).

- `fig_context_subspace_mask_extracted.csv`: `panel` (a = mask from epoch 1, b = mask from epoch 2), `series` (`A_masked`, `A_control`, `B_unmasked`, `B_control`), `epoch`, `gap_contribution` (nats).
- Source: vector paths of the collaborator's two-panel figure `overleaf-paper/figures/fig_context_subspace_mask.pdf` (sha256 `847f58bf46a40937960ef08c260eb139d14312f8126516a89282817be787f536`, Overleaf commit `ea97c5c`, 2026-09-21), calibrated against the axis tick marks (x 1..2 and 2..3, y 0..0.03). Recovery precision is about 1e-4 nats; 36 points per series.
- The underlying runs (both-table input model, seed 42, 2% of training trigram contexts masked after epoch 1 or 2, paired normal-training controls) are the collaborator's; run ids and raw logs are not in this repository. Replace this CSV with the raw logs once they are handed over.
- Plot: `docs/plot_scripts/plot_paper_fig7_context_mask.py` -> `docs/figs/paper/fig07_context_mask.pdf`.
