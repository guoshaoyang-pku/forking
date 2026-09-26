# Section 4 closure, 2026-09-26

Final choice: retain the current three-seed ONLINE data and restore log-log
power-law presentation. The setting is unchanged. Figure 5(a) previously
used log-x/linear-y axes with gap=A*f+B curves; this revision instead uses
log-log OLS of gap versus frequency. The older fixed-probe plot used other
runs, aggregation and hand-entered lines, so metric changes alone do not
explain its different exponents.

Main text: introduce novel continuation and gap approximately a*f+b*(1-f),
then show the matched bigram exponents 0.25 (gap) and 0.23 (direct novelty).
Appendix E: preserve the exact decomposition, distinguish affine A=a-b,
B=b, report both branches, and give the offset/frequency-dependent remainder.
Trigram's 0.466 gap exponent versus 0.212 novelty exponent is not presented
as a match. Existing affine coverage plots remain in the appendix.

Source runs: fig1v5_rp_both_s{42,43,44}_fixed, step 1000. All 18 native bins
F=1..1000 are retained, represented by geometric midpoints; equal-bin log-log
OLS follows arithmetic averaging of gap across seeds. Novel fractions use
the same fixed validation pool. Config and logged gaps are checked by the
reproduction script against archived summary/JSONL records.

Reproduce from repo root:

    .venv/bin/python docs/plot_scripts/plot_paper_fig345_row.py --frequency-only
    # copy the resulting PDF to paper/figures/ with the same filename
    .venv/bin/python docs/plot_scripts/audit_section4_powerlaw.py --update-paper-manifest

The audit updates only the frequency entry. build_appendix_c_manifest.py now
retains the current power-law protocol from the paper fit manifest when it
regenerates the crosswalk; all other panel definitions are preserved.
Final numbers, source SHA-256 hashes, per-bin values, Fable usage and the
independent review are in ../section4_close_20260926/.

Before-revision theoretical text is preserved in before/ from paper commit
22f0da3. paper_changes.patch and plot_changes.patch capture the final changes.
The PDF snapshot here is the final online panel. Fable drafts in this directory
record an abandoned fixed-probe drafting route; they are NOT evidence or final
paper text. The adjacent section4_close directory supplied the adopted draft,
which was subsequently fact-checked and aligned to the restored power-law plot.

Validation: all 18 bins positive in both branches; no log-fit deletions;
independent numerical and notation review passed; latexmk produced 40 pages;
no undefined citations/references or overfull boxes in final main.log.
