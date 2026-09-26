# Section 4.1 final wording check

Read-only check of the in-progress shared paper source on 2026-09-26. No manuscript file was changed by this reviewer.

The revised main-text mixture equation `gap ≈ a f + b(1-f)` is algebraically compatible with Appendix E's affine fit after its coefficients are renamed `A = a-b`, `B = b`. The text calls a and b effective loss differences; it does not claim that the fit proves constant measured component losses. Appendix E retains the exact F-dependent decomposition and expands the remainder.

The matched exponent comparison is reproducible over all 18 registered frequency bins spanning F=1..1000, with geometric-midpoint F and equal-bin log-log OLS. Direct novelty values from `coverage_takeover_20260925/corpus_bins.csv`, field `probe`, give bigram alpha=0.22922691964530045 (R2log=0.9812535138784072), trigram alpha=0.21217424980695673 (R2log=0.9845641698624683). The registered current online gap means give beta=0.2508722117071168 and 0.4660382448607564. Main-text 0.25 vs 0.23 and the Appendix table round these correctly; the trigram mismatch is explicitly reported.

The main text and figure caption now consistently describe the online three-seed means and log-log power-law display. The obsolete fixed-probe hardcoded bigram exponent 0.233 is not used.

`git -C paper diff --check` returned no errors at review time. Typesetting and remote integration are owned by the parent task.

One optional wording refinement: in Appendix E's correction paragraph, `This accounts for why coverage can track the gap` could become `These terms allow coverage to track the gap`; the formula establishes possibility rather than measuring each term's causal contribution.
