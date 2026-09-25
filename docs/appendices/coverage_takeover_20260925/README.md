# Coverage theory takeover — 2026-09-25

Offline audit and manuscript revision, not a new training run. The original
manuscript is archived under archive/paper_sections/ and archive/paper_figures/,
based on Overleaf revision d04f9ee8ae1b70641508c94b7d4a6b0307fd5ba7.
The pre-rewrite archive commit is 47f27d8. The user explicitly authorized direct
edits to paper/ and requested retention of valid handwritten text and the
mathematical style. No push was performed.

## Scientific result

The main claim is an approximate affine relation between the ONLINE GAP and
directly measured validation novelty, not equality of power-law exponents.
Setting: input injection, both clean tables R=2^20, RMSProp (0,.99), 128x table LR,
backbone LR 0.0006, warmup_constant(100), bf16 without compile, training shard 1
and disjoint validation shards. Sources: fig1v5_rp_{both,nogram}_s{42,43,44}_fixed,
steps 670, 1000, 1340 and 2000.

All 18 native bins in F=1..1000 are included, >=300 tokens per side. Gap uses
post-update fixed validation minus pre-update current-batch training CE.
Equal-bin fits to three-seed means have relative RMSE 3.0–8.2% (bigram),
3.2–10.7% (trigram), divided by mean absolute bin gap. Validation-token weighting
and alternating-bin interpolation holdout are exported separately.

Shifted GT uses training-only pooling: sum N1(F+1) / sum (F+1)n(F+1).
Its uncalibrated ratio to full-shard held-out novelty is 0.870–0.885 (bigram),
0.857–0.937 (trigram). The exact ratio-of-expectations identity requires Poisson
context counts, conditional independence and matched train/val distributions;
it does not prove finite-sample unbiasedness of the empirical ratio.

Exact F=2..8 strata provide a separate fixed-probe diagnostic, not causal
identification. Trigram strata are not strictly monotone. Stored probabilities
are floored at 5.96e-8, with up to 1094 zeros per array; loss values are censored.
The affine intercept is not generally equal to the measured seen component S.

## Reproduce and provenance

Run from repository root: .venv/bin/python docs/plot_scripts/plot_coverage_takeover.py

Compact inputs in inputs/ are sufficient. source_manifest.json records 21
remote/local SHA256 comparisons, coordinates and compact-input hashes.
Full recovered logs are in canonical data/runs_fixed/. Scalar endpoint losses
agree to <2e-7 nats; every plotted val-bin count matches the novelty arrays.
This task verified the supplied corpus summaries; it did not recount raw text.

- online_points.csv and online_fits.csv: online observations, weighting and seed sensitivity.
- corpus_bins.csv: uncalibrated train-only GT and direct counts.
- exactF_strata.csv and diagnostic_decomposition.csv: censored probe supplements.
- qc.json: checks and limitations.
- reviews/theory_review.md: completed independent mathematical review.
- reviews/evidence_review.md: second reviewer timeout, no findings claimed.

## Manuscript and figures

Edited in place: paper/sections/03_empirical.tex, 04_mechanism.tex and
appendix_e_theory.tex. Valid conditional-tail integral, crowding-out equations,
excess-validation-loss table and unrelated handwritten paragraphs are retained.
The old exponent comparison and GT/kernel figure references are removed.

Reader-facing figure: docs/figs/coverage_20260925/fig_coverage_linear.pdf,
three LINEAR comparisons: corpus estimate vs measured novelty, and online gap
vs novelty at bigram/trigram granularity. Complex residual and strata figures
remain auditable outputs but are not inserted into the paper. Figure 3a is
fig03a_frequency_coverage.pdf, using online means and all seed error bars.

The original archive/ is immutable. The initial overleaf_patch/ proposal in
commit 47f27d8 was not compiled; the working packet is superseded by the actual
edited manuscript. Use paper/ or revised/, not that initial proposal.

The current narrative revision is paper commit ed4205d. It supersedes the
readability-only structure: Chapter 4 again starts with the frequency-to-novel
clue (4.1), then replay/crowding-out (4.2), then a test of locality (4.3).
The full coverage decomposition, GT derivation and mathematical qualifications
remain in Appendix E. Appendix F records the measured early-pass increments
and an unresolved conflict between the original context-mask figures and
registered reruns. Independence is no longer stated as established.
Two further completed Fable calls drafted/reviewed the discovery narrative;
see narrative_revision/ for prompts, outputs and source checks.
The current isolated PDF is 31 pages (Chapter 4: 5–7; E: 23–27; F: 27–28).
The shared PDF is 34 pages and includes concurrent Appendix B/C work.
Both builds passed, without undefined references or overfull boxes; affected
pages were rendered and inspected. Current snapshots are in revised/.
