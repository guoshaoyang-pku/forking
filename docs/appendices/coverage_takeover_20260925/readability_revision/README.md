# Readability revision, 2026-09-25

The author requested a short, human-readable Chapter 4 that introduces novel
continuations and starts with the existing crowding-out and context-subspace
arguments. The power-law discussion belongs in Appendix E. This is an editorial
revision of the verified evidence package, not a new experiment.

## Result

- Paper parent: 9c549a8; new paper commit: 28efa42.
- Chapter 4 opens with three sentences. Former 4.2/4.3 are now 4.1/4.2.
- The context-subspace subsection is byte-for-byte identical to the original
  handwritten subsection in archive/paper_sections/04_mechanism.tex.
- The walk-their-dog and pooled-CE paragraph is preserved, with the added phrase
  "over the first three passes" delimiting its reported CE numbers. The figure
  extends to pass five, while the quoted 4.3→2.8, 6.9→5.6 and 7.5→9.0 values
  are verified at e1/e3 in marm_logits_20260916/marm_subspace_mass.json.
- Coverage decomposition and the linear triptych move to E.1. The GT expectation
  identity, conditional-tail integral and logit-odds calculation remain intact.
- E.5 keeps all numerical tests and limitations, including the late intercept.
- Chapter 3 changes only two appendix cross-references.
- No new citations, figures, numerical fits, training runs or push.

## Fable use

Two independent requests to claude-fable-5 completed via writer-engines:
polish:mechanism-opening and polish:coverage-equation-bridge. Briefs and raw
outputs are included here; usage.jsonl records success, sizes and duration.
Credentials were consumed by the dispatcher at runtime and not copied.
The native subagent tool returned unsupported call, so the independent drafting
work used parallel Fable dispatcher processes. Integration, source checks and
final editing were performed by the coordinating agent.

The final text shortens the model drafts, retains the project's gap notation,
adds F>=1 explicitly, and avoids universal proportionality or an exact exponent
claim. The original theory snapshot remains in archive/ (commit 47f27d8).

## Verification

Both shared and isolated builds passed latexmk: no undefined references,
overfull boxes, duplicate labels or LaTeX errors. Shared PDF: 32 pages; isolated
PDF: 29 pages. Chapter 4 is on pages 5–7; isolated Appendix E is on pages 22–26.
Affected pages were rendered and visually checked. The current standalone
artifact is ../revised/paper_coverage_revision.pdf; current TeX snapshots are
alongside it. paper_changes.patch records only this readability revision;
../paper_changes.patch records cumulative theory edits from d04f9ee.
