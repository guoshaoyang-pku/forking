Here's the draft (~220 words total, excluding the equation and bracketed notes):

---

**Section 4 opening**

Figure~\ref{fig:frequency} reveals an approximate power law: the forking gap falls off smoothly with fact frequency $F$. Why should frequency organize the gap? We propose a simple bookkeeping answer based on coverage. Call a validation target a \emph{novel continuation} if its exact context occurred in training but was never followed by that target. If $f(F)$ is the fraction of novel continuations at frequency $F$, then
\begin{equation}
\gap_t(F)\approx a_t f(F)+b_t(1-f(F)),
\label{eq:coverage-main}
\end{equation}
where $a_t$ is the typical penalty paid on novel continuations at step $t$ and $b_t$ the typical seen-continuation term. This is bookkeeping, not an exact identity: the coefficients are not literally constant across facts. Yet directly measured novelty shows the same slow decline with $F$, and Good--Turing-style heavy-tailed conditional continuation statistics make an approximate power law a natural hook (exact decomposition, estimators, and indexing are deferred to Appendix~\ref{app:math}). The dashed curves in Figure~\ref{fig:frequency} are measured $f$, rescaled and shifted per branch and step—not a training-only prediction. Coverage thus explains where the vulnerability lies. It does not explain why replay keeps widening the gap once the corpus—and hence $f(F)$—is fixed.

**4.2 Crowding-out under replay**

A finite sample can initially underrepresent a continuation. But after the first epoch the same data are replayed: $f(F)$ is unchanged, yet $\gap_t$ keeps growing. This suggests that each replay reinforces seen continuations and suppresses alternatives within the context-specific subspace.

*(Existing paragraph "This would make forking..." follows unchanged.)*

---

Notes on constraints: no exponent-equality or universality claims (only "the same slow decline," as specified); no numbers or citations introduced; the ending lands the suspense that sets up 4.2, and 4.2's opening hands off directly to your existing "This would make forking..." paragraph.