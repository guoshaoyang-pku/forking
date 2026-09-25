**(1) Chapter opener**

```latex
Our starting clue is a regularity in Figure~\ref{fig:frequency}: the gap declines with the training frequency of a context as an approximate power law, and any account of forking must explain why frequency protects a context.
```

**(2) Subsection 4.1**

```latex
\subsection{From the frequency law to novel continuations}
\label{sec:coverage}
Figure~\ref{fig:frequency} displays the regularity: binned by the training frequency $F$ of the context, the gap falls with $F$ in an approximate power law. The statistics of natural language suggest what else varies with $F$ at that rate. Call a validation target \emph{novel} when its context occurs in training ($F\ge 1$) but was never followed there by that target, and let $f(F)$ be the fraction of validation positions at frequency $F$ that are novel. This directly measured novelty fraction traces the gap: the dashed curves in Figure~\ref{fig:frequency} are $f$ itself, rescaled and shifted,
\begin{equation}
\mathrm{gap}_t(F)\;\approx\;a_t\, f(F)+b_t,
\label{eq:coverage-main}
\end{equation}
with one scale and one offset per training step $t$. Good--Turing estimation and the heavy tails of conditional next-token distributions connect the slow decay of $f$ to text statistics; the fit protocol, sampling assumptions and correction terms are in Appendix~\ref{app:math}. Coverage thus locates the vulnerability: the gap concentrates where continuations remain unseen. It does not yet say why replay worsens it.
```

**(3) Beginning of 4.2**

```latex
\subsection{Crowding-out under replay}
\label{sec:sharpening}
A static explanation looks sufficient at first: each context reveals only a finite sample of its continuations, a model fit to that sample must pay on the remainder, and $f$ says where. But the timing carries a further clue. After the first complete pass the corpus, and with it $f$, is fixed, yet the gap keeps growing at every replay, by comparable increments over the first few passes (Figure~\ref{fig:staircase}). Sampling alone does not explain this progressive damage. We hypothesize that training itself does the additional work: each repetition rewards the seen continuations and thereby progressively suppresses the alternatives.
```

Notes on fit: 4.1 is ~170 words plus the equation; `eq:coverage-main` is distinct from the appendix's equation, and the dashed curves are described only as the measured $f$ scaled and offset. The 4.2 opener is ~95 words, keeps the static mismatch alive as a locator rather than calling it disproved, refers only to the total gap (never novel-CE increments), and ends so that "Forking is ordinary overfitting confined to very small problems. For a context $c_i$, call..." can follow directly.