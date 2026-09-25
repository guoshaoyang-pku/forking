\subsection{From novel continuations to the frequency law}
\label{app:coverage}

Fix a training step $t$ and bin the held-out prediction positions by the training count $F$ of their context. Within a bin, let $f(F)$ denote the fraction of held-out positions whose true continuation never followed that particular context in training. Novelty is defined after conditioning: the token may be common and the context frequent; what is absent is the pair.

The bin's online train--validation gap splits exactly along this line:
\begin{equation}
G_t(F)=f(F)\,D_t(F)+\bigl(1-f(F)\bigr)\,S_t(F),
\label{eq:gap-decomp}
\end{equation}
where $D_t(F)$ is the mean held-out loss on novel continuations minus the bin's online training loss, and $S_t(F)$ is the mean held-out loss on seen continuations minus that same training loss. Equation~\eqref{eq:gap-decomp} is bookkeeping---it holds for any model---so the content lies in how the two components behave. Across bins, a single affine relation
\begin{equation}
G_t(F)\approx a_t\,f(F)+b_t
\label{eq:gap_j}
\end{equation}
describes the measured gaps well. Exact proportionality, $G_t\approx D_t\,f$, is the special case in which the seen term is negligible and the novel penalty is nearly independent of $F$; Equation~\eqref{eq:gap_j} is weaker, an empirical fit across bins that frequency-dependent $D_t(F)$ and $S_t(F)$ can jointly produce without either being constant.

Why should $f$ fall slowly with $F$? For a fixed context under independent draws, Good--Turing ties the expected novel mass after $F$ observations to singleton statistics (Appendix~\ref{app:gt-identity}), and if the conditional continuation probabilities have a heavy tail $p_{(r)}\sim A r^{-s}$ with $s>1$, that mass decays like $F^{-(1-1/s)}$ over an intermediate range (Equation~\eqref{eq:occupancy}). Pooling contexts with different tails and entropies can shift the effective slope, so we treat the power-law-like decay of $f$ as a motivated shape rather than a corpus-fixed exponent.

Figure~\ref{fig:coverage} shows the measured $f(F)$ alongside the linear relation~\eqref{eq:gap_j}; the detailed fits and their caveats appear in Appendix~\ref{app:coverage-tests}.