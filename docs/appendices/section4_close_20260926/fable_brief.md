Rewrite only a short ML-paper subsection (140–190 words) in polished human-readable English LaTeX. The user wants the simple original mathematical style and a discovery narrative: observed gap-frequency power law -> novel continuation coverage -> replay dynamics. Do not add a checklist, technical caveat dump, citations, or fabricated numbers.

Context: current Section 4 opens 'The approximate power law in Figure~\ref{fig:frequency} is our first clue: why should context frequency organize the gap so simply?' Section 4.1 is titled 'From the frequency law to novel continuations'. Section 4.2 discusses why fixed corpus coverage cannot alone specify continuing damage after each epoch.

Required facts and form:
- A validation target is novel if its context occurs in training but the context was never followed by this target there. The word itself may be familiar; novelty is a context-target pair property.
- f(F) is the fraction of validation positions in context-count bin F>=1 whose targets are novel.
- Exact accounting is gap_t(F)=f(F)D_t(F)+(1-f(F))S_t(F), where D and S are the mean novel/seen validation losses minus the same-bin training loss. The main text should display just the simple approximation gap_t(F) \approx a_t f(F)+b_t[1-f(F)] (eq label eq:coverage-main), verbally calling a_t and b_t typical novel and seen loss differences. The exact F-dependent decomposition is in Appendix E.
- Empirically f(F) declines approximately as F^{-alpha}; conditional heavy tails and Good–Turing offer an account of this corpus statistic. Do not claim GT alone mathematically implies a power law.
- The novel term can make gap inherit approximately the same decay when it dominates and the component differences vary slowly. Exponents need not universally match. The parent agent is checking numerical exponent comparisons; do not state numeric exponents in this draft.
- Current plot uses actual online gap three-seed means and an affine fit to measured f(F), drawn against F. Existing curves fit gap=A f+B, mathematically identical to a f+b(1-f) with a=A+B and b=B. It is empirical fit; fitted coefficients need not literally equal component means. Keep 'effective' or similarly concise wording if needed.
- Send exact decomposition, higher-order/correction terms, fitting details to Appendix~\ref{app:math}. End with the open question: why replay harms these novel continuations.

Writing principles: subject and verb near each other; familiar information before new; one paragraph per job; no vague intensifiers. Preserve the clean equation as the intuitive hinge. Return only ready-to-use LaTeX subsection.
