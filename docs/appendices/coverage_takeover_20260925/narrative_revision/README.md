# Discovery narrative revision, 2026-09-25

The author requested the discovery order: approximate frequency power law ->
novel-continuation coverage -> a fixed corpus but accumulating replay damage ->
suppression of alternatives -> context-locality test. The previous edit had
revealed the mechanism too early.

Paper parent: 28efa42. New paper commit: ed4205d. No push.

## Editorial result

4.1 returns as a short coverage clue with one affine-fit equation. Dashed curves
are identified as measured novel fractions rescaled and shifted, not pure
power-law fits or training-only predictions. Full sampling assumptions, GT
identities, decomposition and correction terms stay in E.

The author's original C/D finite-sample example moves from Chapter 3 to the
opening of 4.2. The temporal pivot is fixed coverage but growing gap, with
comparable early-pass increments, not a law of constant increments. The
crowding-out calculation and walk-their-dog illustration retain their prose.
The novel-CE sentence now refers to epoch-end observations, not unmeasured
dense novel-CE jumps.

4.3 retains the original figure and most descriptions but is titled "Testing
the locality of context subspaces". During source checking, experiment-log §63
and local row-mask JSONL revealed unresolved disagreement with the original
freeze-after-mask conclusion. The final paragraph and figure caption identify
that conflict; F.4 records the verified values and metric differences. An
asynchronous question asked whether the collaborator has since reconciled it;
no answer had arrived before this revision was committed. Original text remains
recoverable from 28efa42 and 04_before.tex. No claim of strict independence is
made in the revised subsection.

## Sources and scope

source_checks.json records source hashes, full long-replay boundary rows and
selected mask-rerun records. Early both-table online-gap increments are 1.261,
1.531 and 1.187 nats, declining to 0.241 at pass20. These are total gap, not novel
CE. Masked cohort contributions at step670 are .035263 vs control .024728;
at step790, .055126 vs .049196. The metric differs from the original fixed-probe
figure; neither equivalence nor matched refutation is claimed.

power_shape_check.json is a descriptive sanity check of the existing step1000
three-seed 18-bin gap. Bigram is close to a straight log-log slope; trigram is
less so. No exponent matching is promoted and no figure is regenerated.

Two parallel Fable calls completed: polish:discovery-narrative and
review:discovery-logic. Briefs, raw outputs and usage records are stored here.
The coordinator integrated and checked the text. Native subagents were probed
and unavailable (unsupported call); Fable calls supplied independent drafting
and critique. No credentials were printed or copied.

## Verification

Shared build: 34 pages. Isolated build: 31 pages, based on committed paper plus
these scoped edits. No undefined references, overfull boxes, duplicate labels
or LaTeX errors. Chapter 4 and new Appendix F pages were rendered and inspected.
Current standalone PDF: ../revised/paper_coverage_revision.pdf.
The current TeX snapshots match paper/; Appendix E content is unchanged in this
revision. No unrelated Appendix B/C edits were included.
